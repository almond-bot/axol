"""The jog session: one owner for the arms, lift and base while jogging.

A :class:`JogSession` runs a fixed-rate control loop on the ``jog``
operation's thread. The control panel reaches it through the serve app,
from other threads, with three kinds of request:

- **Hold commands** (:meth:`JogSession.hold`): the base velocity and lift
  direction for the buttons currently held. The page sends them continuously
  while it is connected, zeros included, the way a headset streams teleop
  frames. A release ramps the base and lift down like the real controllers;
  a page that stops sending (closed, hidden, disconnected) is a dead command
  source and everything stops at once.
- **Plans** (:meth:`JogSession.plan`): IK to the closest reachable tip pose
  and a trajectory to it. Nothing moves.
- **Execution** (:meth:`JogSession.execute` / :meth:`JogSession.cancel`):
  stream a plan's trajectory to the arms. It is refused when the arms have
  moved since planning.

The arms are always streamed a command at the loop rate (the hold pose when
nothing is executing), which keeps the realtime core's watchdog fed on
hardware.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from .planner import Arm, JogPlan, JogPlanner, Pose

_logger = logging.getLogger(__name__)

# Seconds without a hold command before the page counts as gone. The page
# sends one every 50 ms; this rides out a dropped frame or two.
HOLD_TIMEOUT_S = 0.25
# A plan is refused at execute time if any arm joint has moved further than
# this from where the plan starts.
PLAN_START_TOLERANCE_RAD = 0.02
# Arms closer than this to the rest pose skip the return to rest on stop.
PARK_TOLERANCE_RAD = 0.01
# Number of joints per arm (the gripper is the 8th entry of a motion command).
ARM_DOF = 7

_session: JogSession | None = None
_session_lock = threading.Lock()


def active_session() -> JogSession | None:
    """The running jog session, or None when the ``jog`` operation is idle."""
    with _session_lock:
        return _session


def _set_active(session: JogSession | None) -> None:
    global _session
    with _session_lock:
        _session = session


class JogError(RuntimeError):
    """A jog request that cannot be carried out; the message is for the operator."""


class JogBackend(Protocol):
    """What the session drives: the simulator or the real robot."""

    mode: str
    has_arms: bool
    has_lift: bool
    has_base: bool

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def arm_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Measured ``(8,)`` pose per arm: 7 joints and the gripper."""
        ...

    async def send_arms(self, left: np.ndarray, right: np.ndarray) -> None: ...
    def drive(self, command: tuple[float, float, float, int] | None, dt: float) -> None:
        """Forward this tick's base and lift command; None is a dead source."""
        ...

    def lift_state(self) -> dict[str, Any] | None: ...
    def base_state(self) -> dict[str, Any] | None: ...


_IDLE = (0.0, 0.0, 0.0, 0)


def _engaged(command: tuple[float, float, float, int]) -> bool:
    return any(v != 0 for v in command)


@dataclass
class _Source:
    """One page's latest hold command."""

    command: tuple[float, float, float, int]
    at: float
    engaged_at: float


@dataclass
class _Execution:
    plan: JogPlan
    step: int = 0
    cancelled: bool = False


def _quaternion(rotation: np.ndarray) -> list[float]:
    """``[x, y, z, w]`` for a rotation matrix."""
    return [float(v) for v in Rotation.from_matrix(rotation).as_quat()]


def _pose_dict(pose: Pose) -> dict[str, list[float]]:
    return {
        "position": [float(v) for v in pose[0]],
        "quaternion": _quaternion(pose[1]),
    }


def _pose_from_request(position: list[float], quaternion: list[float]) -> Pose:
    if len(position) != 3 or len(quaternion) != 4:
        raise JogError("A target needs a 3D position and a quaternion.")
    values = [*position, *quaternion]
    if not all(math.isfinite(v) for v in values):
        raise JogError("The target pose has a non-finite value.")
    quat = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-6:
        raise JogError("The target orientation is not a valid rotation.")
    rotation = Rotation.from_quat(quat / norm).as_matrix().astype(np.float32)
    return np.asarray(position, dtype=np.float32), rotation


class JogSession:
    """Runs the jog control loop for one backend."""

    def __init__(
        self,
        backend: JogBackend,
        planner_factory: Any,
        *,
        rate: float = 125.0,
        arm_speed: float = 0.6,
        min_duration: float = 1.0,
    ) -> None:
        """
        Args:
            backend:         The simulator or the robot.
            planner_factory: Zero-argument callable building the
                             :class:`JogPlanner` (the IK warm-up takes
                             seconds, so it runs on its own thread).
            rate:            Control loop and trajectory rate in Hz.
            arm_speed:       Average speed (rad/s) of the furthest-moving
                             joint along a planned move.
            min_duration:    Shortest planned move in seconds.
        """
        self._backend = backend
        self._rate = float(rate)
        self._arm_speed = float(arm_speed)
        self._min_duration = float(min_duration)
        self._lock = threading.Lock()
        self._sources: dict[str, _Source] = {}
        self._dead_until = -math.inf
        self._command: np.ndarray | None = None  # (14,) joints being held
        self._grip = (0.0, 0.0)
        self._plans: dict[int, JogPlan] = {}
        self._execution: _Execution | None = None
        self._last_result: dict[str, Any] | None = None
        self._tips: dict[str, Any] | None = None
        self._planner: JogPlanner | None = None
        self._planner_error: str | None = None
        self._planner_ready = threading.Event()
        self._planner_factory = planner_factory
        self._stop = threading.Event()

    # -- planner -----------------------------------------------------------

    def _build_planner(self) -> None:
        try:
            self._planner = self._planner_factory()
        except Exception as exc:  # noqa: BLE001 - reported to the page
            _logger.exception("jog: IK solver failed to start")
            self._planner_error = f"The IK solver failed to start: {exc}"
        finally:
            self._planner_ready.set()

    def _require_planner(self, timeout: float = 120.0) -> JogPlanner:
        if not self._planner_ready.wait(timeout):
            raise JogError("The IK solver is still starting. Try again in a moment.")
        if self._planner is None:
            raise JogError(self._planner_error or "The IK solver is not available.")
        return self._planner

    # -- requests from the serve app (any thread) --------------------------

    def hold(
        self, vx: float, vy: float, wz: float, lift: int, source: str = "page"
    ) -> None:
        """Latch ``source``'s held base velocity (normalized) and lift direction.

        Each connected page is its own source. The base and lift follow the
        source that most recently started holding something, so a second
        page sitting idle never cancels the operator's command.
        """
        values = (vx, vy, wz)
        if not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in values
        ):
            self.release(source)
            raise JogError("Drive commands must be finite numbers.")
        if lift not in (-1, 0, 1) or isinstance(lift, bool):
            self.release(source)
            raise JogError("The lift command must be -1, 0 or 1.")

        def clamp(v: float) -> float:
            return max(-1.0, min(1.0, float(v)))

        # No stick deadzone here: the page's drive pad has its own dead
        # centre, and its speed limit must scale the command linearly.
        self._latch(source, (clamp(vx), clamp(vy), clamp(wz), int(lift)))

    def release(self, source: str = "page") -> None:
        """``source`` let go of every control."""
        self._latch(source, _IDLE)

    def _latch(self, source: str, command: tuple[float, float, float, int]) -> None:
        now = time.monotonic()
        with self._lock:
            previous = self._sources.get(source)
            if not _engaged(command):
                engaged_at = -math.inf
            elif previous is not None and _engaged(previous.command):
                engaged_at = previous.engaged_at
            else:
                engaged_at = now
            self._sources[source] = _Source(command, now, engaged_at)

    def disconnect(self, source: str = "page") -> None:
        """``source`` went away. If it was driving, everything stops at once."""
        with self._lock:
            gone = self._sources.pop(source, None)
            if gone is not None and _engaged(gone.command):
                self._dead_until = time.monotonic() + HOLD_TIMEOUT_S

    def _effective_hold(self, now: float) -> tuple[float, float, float, int] | None:
        """This tick's base and lift command; None is a dead command source.

        Call with the lock held. A source that stops sending is dropped; if it
        was holding a control, the base and lift stop at once as they do when
        a headset's link drops, instead of ramping down.
        """
        for name, source in list(self._sources.items()):
            if now - source.at > HOLD_TIMEOUT_S:
                del self._sources[name]
                if _engaged(source.command):
                    self._dead_until = now + HOLD_TIMEOUT_S
        if now < self._dead_until or not self._sources:
            return None
        engaged = [s for s in self._sources.values() if _engaged(s.command)]
        if not engaged:
            return _IDLE
        return max(engaged, key=lambda s: s.engaged_at).command

    def current_q(self) -> np.ndarray:
        with self._lock:
            if self._command is None:
                raise JogError("The arms are still starting.")
            return self._command.copy()

    def plan(
        self, arm: Arm, position: list[float], quaternion: list[float]
    ) -> dict[str, Any]:
        """Plan a move of ``arm``'s tip to a pose. Blocking: run off the event loop."""
        if arm not in ("left", "right"):
            raise JogError("Choose the left or right arm.")
        if not self._backend.has_arms:
            raise JogError("The arms are not connected.")
        target = _pose_from_request(position, quaternion)
        planner = self._require_planner()
        q_now = self.current_q()
        lift = self._backend.lift_state()
        plan = planner.plan(
            q_now,
            arm,
            target,
            speed=self._arm_speed,
            rate=self._rate,
            min_duration=self._min_duration,
            lift_raised=None if lift is None else lift.get("raised"),
        )
        with self._lock:
            # Only the latest plan can be executed.
            self._plans = {plan.id: plan}
        return self._plan_dict(plan, planner)

    def _plan_dict(self, plan: JogPlan, planner: JogPlanner) -> dict[str, Any]:
        solution = plan.solution
        own = planner.indices(plan.arm)
        return {
            "id": plan.id,
            "arm": plan.arm,
            "target": _pose_dict(plan.target),
            "reached": _pose_dict(solution.tip),
            "reachable": solution.reached,
            "positionError": float(solution.position_error),
            "orientationError": float(solution.orientation_error),
            "joints": [float(solution.q[i]) for i in own],
            "start": [float(plan.q_from[i]) for i in own],
            "path": [[float(v) for v in p] for p in plan.path],
            "duration": float(plan.duration),
        }

    def execute(self, plan_id: int) -> dict[str, Any]:
        """Start streaming a plan to the arms."""
        with self._lock:
            plan = self._plans.get(plan_id)
            if plan is None:
                raise JogError("That plan is out of date. Plan the move again.")
            if self._execution is not None:
                raise JogError("A move is already running. Stop it first.")
            if self._command is None:
                raise JogError("The arms are still starting.")
            drift = float(np.max(np.abs(self._command - plan.q_from)))
            if drift > PLAN_START_TOLERANCE_RAD:
                del self._plans[plan_id]
                raise JogError(
                    "The arm moved after this plan was made. Plan the move again."
                )
            del self._plans[plan_id]
            self._execution = _Execution(plan)
            self._last_result = None
        return {"id": plan.id, "duration": plan.duration}

    def cancel(self) -> bool:
        """Stop a running move where it is. False when nothing was running."""
        with self._lock:
            if self._execution is None:
                return False
            self._execution.cancelled = True
            return True

    def state(self) -> dict[str, Any]:
        """Snapshot for the page."""
        with self._lock:
            command = None if self._command is None else self._command.copy()
            execution = self._execution
            hold = self._effective_hold(time.monotonic())
            tips = self._tips
            last = self._last_result
        arms = None
        if command is not None:
            arms = {
                "left": [float(v) for v in command[:ARM_DOF]],
                "right": [float(v) for v in command[ARM_DOF:]],
            }
        return {
            "mode": self._backend.mode,
            "ready": self._planner_ready.is_set() and self._planner is not None,
            "solverError": self._planner_error,
            "hasArms": self._backend.has_arms,
            "arms": arms,
            "grip": list(self._grip),
            "tips": tips,
            "lift": self._backend.lift_state(),
            "base": self._backend.base_state(),
            "hold": {
                "active": hold is not None,
                "drive": list((hold or _IDLE)[:3]),
                "lift": (hold or _IDLE)[3],
            },
            "execution": None
            if execution is None
            else {
                "id": execution.plan.id,
                "arm": execution.plan.arm,
                "progress": execution.step / max(1, len(execution.plan.trajectory)),
            },
            "lastExecution": last,
        }

    # -- control loop (the operation's thread) -----------------------------

    def stop(self) -> None:
        self._stop.set()

    async def run(self, stop_event: threading.Event | None = None) -> None:
        """Run until :meth:`stop`, ``stop_event`` or cancellation."""
        backend = self._backend
        threading.Thread(
            target=self._build_planner, name="axol-jog-ik", daemon=True
        ).start()
        await backend.start()
        _set_active(self)
        try:
            if backend.has_arms:
                left, right = await backend.arm_positions()
                with self._lock:
                    self._command = np.concatenate(
                        [left[:ARM_DOF], right[:ARM_DOF]]
                    ).astype(np.float32)
                    self._grip = (float(left[ARM_DOF]), float(right[ARM_DOF]))
            dt = 1.0 / self._rate
            last = time.monotonic()
            tick = 0
            while not self._stop.is_set() and not (stop_event and stop_event.is_set()):
                started = time.monotonic()
                elapsed = min(0.1, started - last)
                last = started
                await self._step_arms()
                with self._lock:
                    hold = self._effective_hold(started)
                backend.drive(hold, elapsed)
                if tick % max(1, round(self._rate / 15)) == 0:
                    self._refresh_tips()
                tick += 1
                spent = time.monotonic() - started
                await asyncio.sleep(max(0.0, dt - spent))
        finally:
            _set_active(None)
            backend.drive(None, 0.0)
            try:
                await self._park()
            finally:
                await backend.stop()

    async def _park(self) -> None:
        """Drive the arms back to rest before the torque comes off.

        Stopping disables the arms, and a disabled arm falls, so the session
        ends the way teleop and waypoints do: a collision-aware move to the
        rest pose. Skipped when the arms never moved under this session (the
        IK solver never came up) or are already at rest.
        """
        from ..teleop.config import VRTeleopConfig
        from .backends import rest_q

        planner = self._planner
        with self._lock:
            self._execution = None
            command = None if self._command is None else self._command.copy()
            grip = self._grip
        if planner is None or command is None or not self._backend.has_arms:
            return
        rest = rest_q()
        if float(np.max(np.abs(command - rest))) <= PARK_TOLERANCE_RAD:
            return
        config = VRTeleopConfig()
        _logger.info("jog: returning the arms to rest before stopping")
        trajectory = await asyncio.to_thread(
            planner.joint_trajectory,
            command,
            rest,
            speed=config.reset_speed,
            rate=self._rate,
            min_duration=config.reset_min_duration,
        )
        dt = 1.0 / self._rate
        for q in trajectory:
            started = time.monotonic()
            with self._lock:
                self._command = q.copy()
            left = np.append(q[:ARM_DOF], grip[0]).astype(np.float32)
            right = np.append(q[ARM_DOF:], grip[1]).astype(np.float32)
            await self._backend.send_arms(left, right)
            await asyncio.sleep(max(0.0, dt - (time.monotonic() - started)))

    async def _step_arms(self) -> None:
        backend = self._backend
        if not backend.has_arms:
            return
        with self._lock:
            execution = self._execution
            if execution is not None:
                plan = execution.plan
                if execution.cancelled or execution.step >= len(plan.trajectory):
                    finished = not execution.cancelled
                    self._last_result = {
                        "id": plan.id,
                        "arm": plan.arm,
                        "status": "done" if finished else "stopped",
                    }
                    self._execution = None
                else:
                    self._command = plan.trajectory[execution.step].copy()
                    execution.step += 1
            command = None if self._command is None else self._command.copy()
            grip = self._grip
        if command is None:
            return
        left = np.append(command[:ARM_DOF], grip[0]).astype(np.float32)
        right = np.append(command[ARM_DOF:], grip[1]).astype(np.float32)
        await backend.send_arms(left, right)

    def _refresh_tips(self) -> None:
        planner = self._planner
        if planner is None:
            return
        with self._lock:
            command = None if self._command is None else self._command.copy()
        if command is None:
            return
        poses = planner.tips(command, blocking=False)
        if poses is None:
            return
        left, right = poses
        tips = {"left": _pose_dict(left), "right": _pose_dict(right)}
        with self._lock:
            self._tips = tips
