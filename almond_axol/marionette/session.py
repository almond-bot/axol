"""Marionette teleoperation: iPhones drive the arms, grippers, lift and base.

:class:`MarionetteSession` is a :class:`~almond_axol.jog.session.JogSession`
(same backends, control loop, hold arbitration, state feed and park on
stop) whose arms follow phones instead of jog buttons:

- A :class:`~almond_axol.marionette.gateway.GatewayClient` receives every
  phone's pose, presses and control values from the Marionette gateway.
- Each arm follows the phone that selected it while that phone holds its
  hold-to-move pad (see :mod:`almond_axol.marionette.teleop` for the
  clutch, mapping and gates). One step of the fast arm IK per control tick
  moves it toward the phone's target, within joint limits, a joint speed
  limit and the self-collision floors.
- Each arm's gripper follows the closure its phone sends.
- Each phone's drive pad, turn and lift buttons are a hold source, like
  the Jog page; the desktop Marionette page is another (keys W/A/S/D, Q/E,
  Z/X). The source that most recently started driving wins, and a source
  that goes quiet stops everything at once.
- Stop on any phone, or on the desktop page, holds the arms and stops the
  base and lift until every Stop is off; each pad must then be pressed
  again.
- A reset (a phone's More sheet, or the desktop page) sends one arm back
  to its rest pose along a collision-aware joint path, which undoes a
  wound-up or flipped joint configuration that IK steps alone would keep.
  It starts only while no arm is being moved, both arms ignore their pads
  until it ends, and Stop halts it where it is.
- Robot feedback goes back to each phone's screen (5 Hz), and the camera
  streams the phones are watching go to the gateway as JPEG frames.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..jog.session import ARM_DOF, SIDES, HoldCommand, JogError, JogSession
from .cameras import CameraHub, streams_for_view
from .gateway import GatewayClient
from .phones import Phone, PhoneTracker
from .teleop import ArmTeleop, TeleopLimits, gate_reason, owners

_logger = logging.getLogger(__name__)

# A live drive stick with no sample for this long (s) is a dead source. The
# app repeats the stick's latest sample every 50 ms while it is touched.
DRIVE_TIMEOUT_S = 0.25
# How fast a gripper command moves (fraction of full travel per second).
GRIP_RATE = 2.5
# How long a refusal or a stop reason stays on the phone (s).
NOTE_S = 5.0
FEEDBACK_PERIOD_S = 0.2
DESKTOP = "Desktop"


@dataclass
class _Reset:
    """One arm on its way back to rest."""

    side: str
    by: str  # who asked: a phone's name or the desktop
    path: list[np.ndarray] | None = None  # (14,) waypoints, once planned
    step: int = 0


class MarionetteSession(JogSession):
    """A jog session whose arms follow Marionette phones."""

    def __init__(
        self,
        backend: Any,
        planner_factory: Any,
        *,
        gateway: str,
        limits: TeleopLimits | None = None,
        cameras: CameraHub | None = None,
        rate: float = 125.0,
        drive_timeout: float = DRIVE_TIMEOUT_S,
    ) -> None:
        super().__init__(backend, planner_factory, rate=rate)
        self._limits = limits or TeleopLimits()
        self._drive_timeout = float(drive_timeout)
        sim = backend.mode == "sim"
        # On the robot a desktop mirror of a phone's screen can't move
        # anything, and a phone must be calibrated to drive an arm.
        self._tracker = PhoneTracker(allow_mirror=sim)
        self._link = GatewayClient(
            gateway,
            self._tracker.apply,
            self._tracker.lost_gateway,
            session_id=f"axol-{secrets.token_hex(4)}",
        )
        self.cameras = cameras
        self._allow_synthetic = sim
        self._require_calibration = not sim
        self._teleop = {side: ArmTeleop(side, self._limits, rate) for side in SIDES}
        # The gripper command per arm (0 closed .. 1 open), set from the
        # measured pose at start, and the grip request already applied:
        # (phone id, its grip_seq).
        self._grip_target: list[float] | None = None
        self._grip_seen: dict[str, tuple[str, int]] = {}
        self._owners: dict[str, str | None] = {"left": None, "right": None}
        self._phone_sources: set[str] = set()
        self._must_release: set[str] = set()
        self._desktop_stop = False
        self._stop_by: str | None = None
        self._notes: dict[str, tuple[str, float]] = {}
        self._view_lock = threading.Lock()
        # Resets asked for and not yet taken up by the loop: (arm, phone id,
        # or None for the desktop page). The reset running, its planning
        # task, and each phone's reset presses already seen.
        self._reset_requests: list[tuple[str, str | None]] = []
        self._reset: _Reset | None = None
        self._reset_task: asyncio.Task[None] | None = None
        self._resets_seen: dict[str, dict[str, int]] = {}

    # -- requests from the serve app ---------------------------------------

    def hold(
        self,
        vx: float,
        vy: float,
        wz: float,
        lift: int,
        source: str = "page",
        **_: Any,
    ) -> None:
        """Base and lift from the desktop page. The phones own the arms."""
        super().hold(vx, vy, wz, lift, source)

    def plan(self, *_: Any, **__: Any) -> dict[str, Any]:
        raise JogError("Planned moves are not available while Marionette is running.")

    def preview(self, *_: Any, **__: Any) -> dict[str, Any]:
        raise JogError("Planned moves are not available while Marionette is running.")

    def execute(self, *_: Any, **__: Any) -> dict[str, Any]:
        raise JogError("Planned moves are not available while Marionette is running.")

    def reset_arm(self, side: str) -> dict[str, Any]:
        """The desktop page's Reset arm: send ``side`` back to rest."""
        if side not in SIDES:
            raise JogError("Choose the left or right arm.")
        if not self._backend.has_arms:
            raise JogError("The arms are not connected.")
        # Refuse what can be refused now; the loop checks again as it starts.
        reason = self._reset_refusal(side, self._tracker.snapshot(), DESKTOP)
        if reason is not None:
            raise JogError(reason)
        with self._lock:
            self._reset_requests.append((side, None))
        return {"arm": side}

    def set_stop(self, on: bool) -> dict[str, Any]:
        """The desktop page's Stop."""
        self._desktop_stop = bool(on)
        if on:
            _logger.info("marionette: Stop turned on from the desktop page")
        else:
            # A phone that went offline with Stop on can't turn it off.
            self._tracker.clear_offline_stops()
        stop_by = self._current_stop(self._tracker.snapshot())
        # Takes effect now rather than at the next tick.
        self._stop_by = stop_by
        return {"on": stop_by is not None, "by": stop_by}

    # -- arbitration ---------------------------------------------------------

    def _effective_hold(self, now: float) -> HoldCommand | None:
        """Jog's arbitration, plus Stop (call with the lock held).

        Stop makes every source a dead one, so the base and lift stop at
        once. A source still driving when Stop came on is ignored until it
        lets go, so turning Stop off never restarts a drive by itself.
        """
        hold = super()._effective_hold(now)
        stopped = self._stop_by is not None
        for name, source in self._sources.items():
            if not source.command.engaged:
                self._must_release.discard(name)
            elif stopped:
                self._must_release.add(name)
        self._must_release &= set(self._sources)
        if stopped or hold is None or not self._must_release:
            return None if stopped else hold
        engaged = [
            s
            for n, s in self._sources.items()
            if s.command.engaged and n not in self._must_release
        ]
        if not engaged:
            return HoldCommand()
        return max(engaged, key=lambda s: s.engaged_at).command

    def _current_stop(self, phones: dict[str, Phone]) -> str | None:
        """Who has Stop on. A phone that drops off with Stop on keeps it on
        until it comes back and turns it off, or the desktop page clears it."""
        if self._desktop_stop:
            return DESKTOP
        on = sorted((p for p in phones.values() if p.stop), key=lambda p: p.number)
        if not on:
            return None
        phone = next((p for p in on if p.connected), on[0])
        return phone.name if phone.connected else f"{phone.name} (offline)"

    def _driver(self) -> str | None:
        """Who is driving the base or lift right now (lock held)."""
        engaged = [
            (n, s)
            for n, s in self._sources.items()
            if s.command.engaged and n not in self._must_release
        ]
        if not engaged or self._stop_by is not None:
            return None
        return max(engaged, key=lambda item: item[1].engaged_at)[0]

    def _driver_name(self, source: str | None, phones: dict[str, Phone]) -> str:
        if source is None:
            return ""
        if source.startswith("phone:"):
            phone = phones.get(source[len("phone:") :])
            return phone.name if phone else ""
        return DESKTOP

    def _drive_phones(self, phones: dict[str, Phone], now: float) -> None:
        """Latch each phone's base and lift command as its own hold source.

        The phone's drive stick (its own deadman), turn holds and lift holds
        drive while they are touched and their samples or repeats keep
        arriving. A connected phone touching none of them stays a live zero
        source, so Jelly ramps down on let-go and keeps hearing zeros
        instead of timing out. A control that is still touched but goes
        quiet, or a phone that disconnects, is a dead source: everything
        stops at once, as when a headset's link drops.
        """
        for phone in phones.values():
            source = f"phone:{phone.id}"
            command = (
                phone.base_command(
                    now, self._drive_timeout, self._limits.deadman_timeout
                )
                if phone.connected
                else None
            )
            if command is not None:
                vx, vy, wz, lift = command
                speed = phone.speed
                self._latch(
                    source, HoldCommand(vx * speed, vy * speed, wz * speed, lift)
                )
                self._phone_sources.add(source)
            elif source in self._phone_sources:
                self._phone_sources.discard(source)
                self.disconnect(source)

    # -- notes for the phones -------------------------------------------------

    def _note(self, phone: Phone, reason: str | None) -> None:
        if reason:
            self._notes[phone.id] = (reason, time.monotonic() + NOTE_S)

    def _note_for(self, phone_id: str, now: float) -> str:
        # Read from the serve thread too: never delete here.
        note = self._notes.get(phone_id)
        return "" if note is None or now > note[1] else note[0]

    # -- control loop ---------------------------------------------------------

    async def run(self, stop_event: threading.Event | None = None) -> None:
        if self.cameras is not None:
            self.cameras.start()
        tasks = [
            asyncio.create_task(self._link.run()),
            asyncio.create_task(self._feedback_loop()),
            asyncio.create_task(self._camera_loop()),
        ]
        try:
            await super().run(stop_event)
        finally:
            if self._reset_task is not None:
                tasks.append(self._reset_task)
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(BaseException):
                    await task
            if self.cameras is not None:
                await asyncio.to_thread(self.cameras.stop)

    async def _step_arms(self, hold: HoldCommand | None, dt: float) -> None:
        now = time.monotonic()
        phones = self._tracker.snapshot()
        stop_by = self._current_stop(phones)
        if stop_by is not None and self._stop_by is None:
            _logger.info("marionette: stopped by %s", stop_by)
        self._stop_by = stop_by
        self._drive_phones(phones, now)
        backend = self._backend
        if not backend.has_arms:
            return
        with self._lock:
            command = None if self._command is None else self._command.copy()
            grip = list(self._grip)
        if command is None:
            return
        if self._grip_target is None:
            self._grip_target = list(grip)
        planner = self._planner
        kin = None if planner is None else planner.kinematics
        if kin is not None:
            kin.set_lift(self._lift_raised())
        command = self._step_reset(phones, command, stop_by)
        arm_owners = owners(phones.values(), {s: self._teleop[s].owner for s in SIDES})
        for i, side in enumerate(SIDES):
            command = self._step_arm(side, phones, kin, command, stop_by, now, dt)
            owner = arm_owners[side]
            # A press can hand the arm to another phone this tick.
            if self._teleop[side].engaged:
                owner = phones.get(self._teleop[side].owner or "") or owner
            self._owners[side] = None if owner is None else owner.id
            grip[i] = self._step_grip(i, side, owner, stop_by, grip[i], dt)
        with self._lock:
            self._command = command.copy()
            self._grip = (grip[0], grip[1])
        left = np.append(command[:ARM_DOF], grip[0]).astype(np.float32)
        right = np.append(command[ARM_DOF:], grip[1]).astype(np.float32)
        await backend.send_arms(left, right)

    def _step_arm(
        self,
        side: str,
        phones: dict[str, Phone],
        kin: Any,
        command: np.ndarray,
        stop_by: str | None,
        now: float,
        dt: float,
    ) -> np.ndarray:
        arm = self._teleop[side]
        limits = self._limits
        contenders = sorted(
            (p for p in phones.values() if p.connected and p.arm == side),
            key=lambda p: p.number,
        )
        pressed = [p for p in contenders if arm.new_press(p)]
        for phone in phones.values():
            if phone not in contenders:
                # Presses made for the other arm never count here later.
                arm.forget_presses(phone)

        if arm.engaged:
            phone = phones.get(arm.owner or "")
            reason: str | None = None
            release = True
            if phone is None or not phone.connected:
                reason = None
            elif stop_by is not None:
                reason = (
                    "Stop is on." if stop_by == phone.name else f"Stopped by {stop_by}."
                )
            elif phone.arm != side:
                reason = None
            elif not phone.deadman_down:
                reason = None  # let go
            elif not phone.deadman_live(now, limits.deadman_timeout):
                reason = "The connection dropped. Press again to keep moving."
            elif not arm.anchor_valid(phone):
                reason = "Tracking restarted. Press again to keep moving."
            elif arm.jumped(phone):
                reason = "Tracking jumped. Press again to keep moving."
            else:
                reason = self._gate(phone, now)
                if reason is None:
                    release = False
                else:
                    reason += " Press again when it's back."
            if release:
                arm.release()
                if phone is not None:
                    self._note(phone, reason)
                _logger.info(
                    "marionette: %s arm released (%s)", side, reason or "let go"
                )

        for phone in pressed:
            if arm.engaged:
                owner = phones.get(arm.owner or "")
                name = owner.name if owner else "Another phone"
                self._note(phone, f"{name} is moving the {side} arm.")
                continue
            if stop_by is not None:
                reason = (
                    "Stop is on." if stop_by == phone.name else f"Stopped by {stop_by}."
                )
            elif kin is None:
                reason = (
                    self._planner_error
                    if self._planner_error
                    else "The arm controller is still starting. Try again in a moment."
                )
            elif self._reset is not None:
                reason = (
                    f"Wait for the {self._reset.side} arm to reach rest, "
                    "then press again."
                )
            else:
                reason = self._gate(phone, now)
            if reason is not None:
                self._note(phone, reason)
                continue
            if not phone.deadman_live(now, limits.deadman_timeout):
                continue  # pressed and let go between ticks
            arm.engage(phone, kin.tip(command, side))
            self._notes.pop(phone.id, None)
            _logger.info("marionette: %s arm engaged by %s", side, phone.name)

        if arm.engaged and kin is not None:
            phone = phones[arm.owner or ""]
            target = arm.target(phone, now, dt)
            result = kin.servo_step(command, side, target, limits.joint_speed * dt)
            command = result.q
            arm.note_tip(kin.tip(command, side), now)
        return command

    # -- resets ----------------------------------------------------------------

    def _reset_refusal(
        self, side: str, phones: dict[str, Phone], by: str
    ) -> str | None:
        """Why ``side`` can't go back to rest now, or None."""
        from ..jog.backends import rest_q
        from ..jog.session import PARK_TOLERANCE_RAD

        stop_by = self._stop_by
        if stop_by is not None:
            return "Stop is on." if stop_by == by else f"Stopped by {stop_by}."
        reset = self._reset
        if reset is not None:
            return f"The {reset.side} arm is returning to rest."
        planner = self._planner
        if planner is None:
            return (
                self._planner_error
                or "The arm controller is still starting. Try again in a moment."
            )
        for other in SIDES:
            arm = self._teleop[other]
            if arm.engaged:
                owner = phones.get(arm.owner or "")
                name = owner.name if owner is not None else "A phone"
                return f"{name} is moving the {other} arm. Wait for it to let go."
        with self._lock:
            command = None if self._command is None else self._command.copy()
        if command is None:
            return "The arms are still starting."
        own = planner.indices(side)
        if float(np.max(np.abs(command[own] - rest_q()[own]))) <= PARK_TOLERANCE_RAD:
            return f"The {side} arm is already at rest."
        return None

    def _reset_presses(self, phones: dict[str, Phone]) -> list[tuple[str, str]]:
        """New reset presses: ``(arm, phone id)``. Each counts once, and
        presses made before the session first saw the phone don't count."""
        pressed: list[tuple[str, str]] = []
        for phone in phones.values():
            seen = self._resets_seen.get(phone.id)
            self._resets_seen[phone.id] = dict(phone.resets)
            if seen is None or not phone.connected:
                continue
            for side in SIDES:
                if phone.resets[side] > seen.get(side, 0):
                    pressed.append((side, phone.id))
        return pressed

    def _step_reset(
        self, phones: dict[str, Phone], command: np.ndarray, stop_by: str | None
    ) -> np.ndarray:
        """Take up reset requests and move a reset on by one waypoint.

        Returns the arm command. Stop ends a reset where the arm is.
        """
        with self._lock:
            requests, self._reset_requests = self._reset_requests, []
        for side, phone_id in requests + self._reset_presses(phones):
            phone = phones.get(phone_id or "")
            if phone_id is not None and phone is None:
                continue
            by = DESKTOP if phone is None else phone.name
            reason = self._reset_refusal(side, phones, by)
            if reason is not None:
                if phone is not None:
                    self._link.send_status(phone.id, "axol.notice", {}, reason)
                _logger.info("marionette: %s arm reset refused (%s)", side, reason)
                continue
            self._reset = _Reset(side, by)
            self._reset_task = asyncio.create_task(
                self._plan_reset(self._reset, command.copy())
            )
            _logger.info("marionette: %s returning the %s arm to rest", by, side)
        reset, planner = self._reset, self._planner
        if reset is None or planner is None:
            return command
        if stop_by is not None:
            self._reset = None
            _logger.info(
                "marionette: %s arm reset halted by Stop (%s)", reset.side, stop_by
            )
            return command
        if reset.path is None:
            return command  # still planning: both arms hold
        if reset.step >= len(reset.path):
            self._reset = None
            _logger.info("marionette: %s arm is at rest", reset.side)
            return command
        own = planner.indices(reset.side)
        command = command.copy()
        command[own] = reset.path[reset.step][own]
        reset.step += 1
        return command

    async def _plan_reset(self, reset: _Reset, command: np.ndarray) -> None:
        """Plan ``reset``'s path off the control loop, as the park on stop
        does; the loop then plays it."""
        from ..jog.backends import rest_q
        from ..teleop.config import VRTeleopConfig

        planner = self._planner
        assert planner is not None
        own = planner.indices(reset.side)
        goal = command.copy()
        goal[own] = rest_q()[own]
        config = VRTeleopConfig()
        try:
            path = await asyncio.to_thread(
                planner.joint_trajectory,
                command,
                goal,
                speed=config.reset_speed,
                rate=self._rate,
                min_duration=config.reset_min_duration,
            )
        except Exception:  # noqa: BLE001 - the arm stays put; the log says why
            _logger.exception(
                "marionette: planning the %s arm reset failed", reset.side
            )
            if self._reset is reset:
                self._reset = None
            return
        if self._reset is reset:  # not halted meanwhile
            reset.path = path

    def _gate(self, phone: Phone, now: float) -> str | None:
        return gate_reason(
            phone,
            now,
            self._limits,
            allow_synthetic=self._allow_synthetic,
            require_calibration=self._require_calibration,
        )

    def _step_grip(
        self,
        index: int,
        side: str,
        owner: Phone | None,
        stop_by: str | None,
        current: float,
        dt: float,
    ) -> float:
        """Move the gripper toward its owner's latest request."""
        targets = self._grip_target
        assert targets is not None
        if stop_by is not None:
            # Stop freezes the gripper where it is.
            targets[index] = current
        if owner is None:
            # Whoever takes the arm next starts fresh, even the phone that
            # had it before: a grip it sent meanwhile was for another arm.
            self._grip_seen.pop(side, None)
        else:
            seen = self._grip_seen.get(side)
            if seen is None or seen[0] != owner.id:
                # A new owner: only grip changes made from now on count, so
                # taking over an arm never opens a closed gripper by itself.
                self._grip_seen[side] = (owner.id, owner.grip_seq)
            elif owner.grip_seq != seen[1] and owner.grip is not None:
                self._grip_seen[side] = (owner.id, owner.grip_seq)
                if stop_by is None:
                    targets[index] = 1.0 - owner.grip
        step = GRIP_RATE * dt
        return float(current + min(max(targets[index] - current, -step), step))

    # -- feedback and cameras ------------------------------------------------

    async def _feedback_loop(self) -> None:
        last: dict[str, tuple[dict[str, Any], float]] = {}
        while True:
            await asyncio.sleep(0.05)
            if not self._link.connected:
                last.clear()
                continue
            now = time.monotonic()
            phones = self._tracker.snapshot()
            for phone in phones.values():
                if not phone.connected:
                    last.pop(phone.id, None)
                    continue
                values = self._feedback(phone, phones, now)
                previous = last.get(phone.id)
                if (
                    previous is None
                    or previous[0] != values
                    or now - previous[1] >= FEEDBACK_PERIOD_S
                ):
                    self._link.send_status(phone.id, "axol.feedback", values)
                    last[phone.id] = (values, now)

    def _feedback(
        self, phone: Phone, phones: dict[str, Phone], now: float
    ) -> dict[str, Any]:
        side = phone.arm
        arm = self._teleop[side]
        owner_id = self._owners.get(side)
        engaged = arm.engaged and arm.owner == phone.id
        owner = phones.get(owner_id or "")
        other = owner.name if owner is not None and owner.id != phone.id else ""
        stop_by = self._stop_by
        reason = self._note_for(phone.id, now)
        if stop_by is not None:
            state = "stopped"
        elif engaged:
            state = "engaged"
        elif reason or other:
            state = "blocked"
        else:
            state = "idle"
        grip = self._grip[SIDES.index(side)] if self._grip else 1.0
        lift = self._backend.lift_state()
        base = self._backend.base_state()
        battery = self._backend.battery_state()
        with self._lock:
            driver = self._driver()
        cameras = self.cameras
        return {
            "mode": self._backend.mode,
            "ready": self._planner is not None,
            "arm": side,
            "state": state,
            "engaged": engaged,
            "limited": bool(engaged and arm.limited),
            "reason": "" if engaged else reason,
            "owner": other,
            "grip": round(1.0 - float(grip), 3),
            "hasLift": lift is not None,
            "lift": -1.0
            if lift is None or lift.get("raised") is None
            else round(float(lift["raised"]), 3),
            "liftMoving": bool(lift and lift.get("moving")),
            "battery": -1.0 if battery is None else round(float(battery["percent"])),
            "batteryCharging": bool(battery and battery["charging"]),
            "batteryUnderLoad": bool(battery and battery["underLoad"]),
            "hasBase": base is not None,
            "baseMoving": bool(base and base.get("moving")),
            "baseFault": bool(base and base.get("fault")),
            "driving": driver == f"phone:{phone.id}",
            "driver": self._driver_name(driver, phones),
            "stopBy": stop_by or "",
            "resetting": "" if self._reset is None else self._reset.side,
            "streams": ",".join(cameras.available) if cameras is not None else "",
        }

    async def _camera_loop(self) -> None:
        hub = self.cameras
        if hub is None:
            return
        sent: dict[str, int] = {}
        while True:
            await asyncio.sleep(0.5 / hub.fps)
            phones = self._tracker.snapshot()
            wanted: set[str] = set()
            for phone in phones.values():
                if phone.connected:
                    wanted.update(streams_for_view(phone.view))
            hub.want("phones", wanted)
            if not self._link.connected:
                sent.clear()
                continue
            for stream in wanted:
                frame = hub.latest(stream)
                if frame is not None and sent.get(stream) != frame.seq:
                    sent[stream] = frame.seq
                    self._link.send_frame(stream, frame.jpeg, frame.captured_ms)

    # -- state for the desktop page --------------------------------------------

    def sim_camera_state(self) -> dict[str, Any]:
        """Joint state for the sim camera renderer (any thread)."""
        from .sim_cameras import sim_state

        with self._lock:
            command = None if self._command is None else self._command.copy()
            grip = self._grip
        base = self._backend.base_state()
        lift = self._backend.lift_state()
        return sim_state(
            command,
            grip,
            None if lift is None else lift.get("raised"),
            None if base is None else (base["x"], base["y"], base["yaw"]),
        )

    def state(self) -> dict[str, Any]:
        state = super().state()
        now = time.monotonic()
        phones = self._tracker.snapshot()
        with self._lock:
            driver = self._driver()
        stop_by = self._stop_by
        reset = self._reset
        arms: dict[str, Any] = {}
        for i, side in enumerate(SIDES):
            arm = self._teleop[side]
            owner = phones.get(self._owners.get(side) or "")
            reason = self._note_for(owner.id, now) if owner is not None else ""
            arms[side] = {
                "owner": None if owner is None else owner.id,
                "ownerName": None if owner is None else owner.name,
                "engaged": arm.engaged,
                "limited": bool(arm.engaged and arm.limited),
                "grip": round(1.0 - float(self._grip[i]), 3),
                "reason": None if arm.engaged or not reason else reason,
                "resetting": reset is not None and reset.side == side,
            }
        state["marionette"] = {
            "gateway": {
                "address": self._link.url,
                "connected": self._link.connected,
                "error": None if self._link.connected else self._link.error,
            },
            "phones": [
                self._phone_state(phone, driver, now)
                for phone in sorted(phones.values(), key=lambda p: p.number)
            ],
            "arms": arms,
            "stop": {
                "on": stop_by is not None,
                "by": stop_by,
                "desktop": self._desktop_stop,
            },
            "cameras": self.cameras.describe()
            if self.cameras is not None
            else {"available": [], "source": None, "error": None},
        }
        return state

    def _phone_state(
        self, phone: Phone, driver: str | None, now: float
    ) -> dict[str, Any]:
        pose = phone.pose
        arm = self._teleop[phone.arm]
        age = phone.pose_age(now)
        return {
            "id": phone.id,
            "name": phone.name,
            "color": phone.color,
            "connected": phone.connected,
            "tracking": pose.tracking if pose is not None else "unavailable",
            "source": pose.source if pose is not None else None,
            "calibration": phone.calibration,
            "arm": phone.arm,
            "view": phone.view,
            "scale": phone.scale,
            "speed": phone.speed,
            "deadman": phone.deadman_live(now, self._limits.deadman_timeout),
            "stop": phone.stop,
            "engaged": arm.engaged and arm.owner == phone.id,
            "driving": driver == f"phone:{phone.id}",
            "poseAge": None if age is None or not math.isfinite(age) else round(age, 3),
        }
