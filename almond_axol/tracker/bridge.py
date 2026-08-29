"""Tracker → VRServer bridge: streams tracker poses as VRFrame JSON.

The bridge is a WebSocket *client* of the existing VR server — exactly
what a Quest headset is — so teleop, IK, and collect-data run unchanged.
It composes a :class:`~almond_axol.vr.models.VRFrame` at the configured
rate from the latest tracker poses, stamps ``t`` (monotonic ms, for the
server's pose interpolator) and ``seq``, and ships it over WSS (the
server's self-signed certificate is not verified).

Grip comes from the Mantis trigger node's CAN messages when one is
configured per side (see :class:`~almond_axol.tracker.trigger.TriggerReader`):
the analog trigger position drives ``l_grip``/``r_grip`` proportionally
(fully squeezed = closed, released = open). Rapid full squeeze/release
sequences also control data collection: three presses starts a take when idle
or ends one successfully; four presses ends one as a failure. Managed plain
teleop/data-collection bridges engage automatically once both trackers are
live. Other flows use :class:`StdinControls` (the trigger frame carries no
buttons — session controls arrive with a later PCB revision): Enter toggles
tracking engage, ``r`` triggers a reset. A manual toggle is realised as a
short pulse of both lock bits — the shared teleop core enables on a rising
edge of both locks together and disables on a rising edge of either.

A side whose tracker stops reporting (occlusion, SLAM relocalising)
holds its last good pose rather than going quiet, so IK never chases a
glitch and the operator can recover by re-engaging. A stale trigger
node likewise holds its last grip command, never jumping on a dropout.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import ssl
import sys
import threading
import time
from typing import Any, Protocol

import numpy as np

from ..utils.ports import VR_PORT
from ..vr.models import VREpisodeOutcome, VRState
from .base import TrackerPose, TrackerSource
from .trigger import TriggerReader

_logger = logging.getLogger(__name__)

# A pose older than this is stale: hold the last streamed pose and warn.
_STALE_S = 0.5
# Lock/reset pulses span this many frames so the server-side edge
# detection can't miss them across interpolation or a dropped frame.
_PULSE_FRAMES = 10

# A gesture press is a deliberate near-full squeeze followed by a near-full
# release. Separate thresholds provide hysteresis around the analog endpoints.
_GESTURE_PRESS_GRIP = 0.2
_GESTURE_RELEASE_GRIP = 0.8
# Maximum quiet time between press edges. A three-press gesture resolves only
# after this expires, because a fourth press in the window changes the outcome.
_GESTURE_TIMEOUT_S = 0.6

# Placeholder pose streamed for a side with no tracker assigned (teleop can
# run one-armed, but VRFrame carries both sides).
_DEFAULT_POSE = {
    "left": (np.array([0.2, 1.0, -0.4]), np.array([0.0, 0.0, 0.0, 1.0])),
    "right": (np.array([-0.2, 1.0, -0.4]), np.array([0.0, 0.0, 0.0, 1.0])),
}


class TriggerGestureRecognizer:
    """Recognize triple/quadruple presses without disturbing analog grip.

    Each rig side owns one recognizer, so the free hand can issue a gesture
    while the other trigger remains squeezed around an object. Four presses
    resolve immediately; three resolve after the inter-press timeout so they
    cannot steal the prefix of a four-press failure gesture.
    """

    def __init__(self) -> None:
        self._pressed = False
        self._presses = 0
        self._last_press_at: float | None = None

    def update(self, grip: float, now: float) -> VREpisodeOutcome | None:
        """Consume one analog grip sample and return a completed gesture."""
        if self._pressed:
            if grip >= _GESTURE_RELEASE_GRIP:
                self._pressed = False
        elif grip <= _GESTURE_PRESS_GRIP:
            self._pressed = True
            if (
                self._last_press_at is None
                or now - self._last_press_at > _GESTURE_TIMEOUT_S
            ):
                self._presses = 0
            self._presses += 1
            self._last_press_at = now
            if self._presses == 4:
                self._clear_sequence()
                return VREpisodeOutcome.FAILURE

        if (
            self._last_press_at is not None
            and now - self._last_press_at > _GESTURE_TIMEOUT_S
        ):
            outcome = VREpisodeOutcome.SUCCESS if self._presses == 3 else None
            self._clear_sequence()
            return outcome
        return None

    def _clear_sequence(self) -> None:
        self._presses = 0
        self._last_press_at = None


class StdinControls:
    """Line-based stdin control surface (stopgap until the button PCB).

    Reads stdin on a daemon thread: an empty line (Enter) requests an
    engage toggle, ``r`` a reset, ``q`` a quit. The PCB input will replace
    this class with the same three request bits.
    """

    def __init__(self) -> None:
        self._toggle_requests = 0
        self._reset_requests = 0
        self.quit = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True, name="tracker-stdin"
        )

    def start(self) -> None:
        self._thread.start()

    def _read_loop(self) -> None:
        for line in sys.stdin:
            cmd = line.strip().lower()
            with self._lock:
                if cmd == "":
                    self._toggle_requests += 1
                elif cmd == "r":
                    self._reset_requests += 1
                elif cmd == "q":
                    self.quit.set()
                    return
        self.quit.set()  # stdin closed (e.g. running under a supervisor)

    def consume(self) -> tuple[bool, bool]:
        """Return and clear ``(toggle_requested, reset_requested)``."""
        with self._lock:
            toggle = self._toggle_requests > 0
            reset = self._reset_requests > 0
            self._toggle_requests = 0
            self._reset_requests = 0
        return toggle, reset


class BridgeControls(Protocol):
    """Control source consumed by :class:`TrackerBridge`."""

    quit: Any

    def start(self) -> None: ...

    def consume(self) -> tuple[bool, bool]: ...


class StopEventControls:
    """Headless controls for a bridge managed by another process.

    The owning process supplies an event that ends the bridge and an optional
    queue carrying control-panel toggle/reset requests.
    """

    def __init__(self, stop_event: Any, command_queue: Any = None) -> None:
        self.quit = stop_event
        self._commands = command_queue

    def start(self) -> None:
        pass

    def consume(self) -> tuple[bool, bool]:
        toggle = False
        reset = False
        if self._commands is None:
            return toggle, reset
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                break
            if command == "toggle":
                toggle = True
            elif command == "reset":
                reset = True
        return toggle, reset


class TrackerBridge:
    """Composes and streams VRFrames from a :class:`TrackerSource`.

    Args:
        source: Started tracker backend.
        left:   Device key bound to the left rig side, or ``None``.
        right:  Device key bound to the right rig side, or ``None``.
        host:   VR server host (the teleop machine; usually localhost).
        port:   VR server port.
        hz:     Frame streaming rate.
        controls: Engage/reset input; defaults to :class:`StdinControls`.
        left_trigger:  Started trigger-node reader for the left side, or
            ``None`` (grip streams as 1.0 = open).
        right_trigger: Trigger-node reader for the right side, or ``None``.
        allow_single_side: Permit exactly one bound side. Off by default
            because absolute-mode (Mantis) engagement fits the shared base
            transform from BOTH controller positions, so the placeholder
            pose streamed for an unbound side corrupts the fit.
        auto_engage: Engage once all bound trackers have produced a live pose.
            The lock stays asserted until the teleop core acknowledges tracking,
            so a slow startup cannot miss a short pulse.
    """

    def __init__(
        self,
        source: TrackerSource,
        *,
        left: str | None,
        right: str | None,
        host: str = "localhost",
        port: int = VR_PORT,
        hz: float = 120.0,
        controls: BridgeControls | None = None,
        left_trigger: TriggerReader | None = None,
        right_trigger: TriggerReader | None = None,
        allow_single_side: bool = False,
        auto_engage: bool = False,
    ) -> None:
        if left is None and right is None:
            raise ValueError(
                "no tracker is bound to either side — run `axol tracker.identify` first"
            )
        if (left is None or right is None) and not allow_single_side:
            bound = "left" if right is None else "right"
            raise ValueError(
                f"only the {bound} side has a tracker bound. Absolute-mode (Mantis) "
                "engagement solves the shared world→robot base transform from "
                "BOTH controller positions, so streaming the built-in placeholder "
                "pose for the unbound side corrupts the engage base fit. Bind "
                "both trackers (`axol tracker.identify`), or pass "
                "--allow-single-side if you accept the corrupted fit."
            )
        self._source = source
        self._keys = {"left": left, "right": right}
        self._host = host
        self._port = port
        self._hz = hz
        self._controls = controls or StdinControls()

        self._triggers = {"left": left_trigger, "right": right_trigger}
        # Last grip streamed per side; held across trigger dropouts so a stale
        # node never commands a jump. Open until the first frame arrives.
        self._grip_held = {"left": 1.0, "right": 1.0}
        self._warned_trigger: dict[str, bool] = {"left": False, "right": False}
        self._gesture = {
            "left": TriggerGestureRecognizer(),
            "right": TriggerGestureRecognizer(),
        }

        self._seq = 0
        self._engaged = False
        self._auto_engage_pending = auto_engage
        self._auto_engage_waiting_ack = False
        self._lock_pulse = 0
        self._reset_pulse = 0
        self._outcome_pulse = 0
        self._episode_outcome: VREpisodeOutcome | None = None
        self._held: dict[str, TrackerPose] = {}
        self._warned_stale: dict[str, bool] = {"left": False, "right": False}
        # The server announces whether this connection belongs to teleop or
        # data collection. TELEOP is the safe default until that message lands.
        self._state = VRState.TELEOP

    # -- Frame composition ---------------------------------------------------

    def _side_pose(self, side: str, now: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``(pos, quat)`` to stream for one side.

        The last good pose wins over anything stale/untracked; a side with
        no data yet (or no tracker assigned) streams the fixed placeholder.
        """
        key = self._keys[side]
        if key is not None:
            sample = self._source.poses().get(key)
            if sample is not None and sample.tracking and now - sample.t <= _STALE_S:
                self._held[side] = sample
                if self._warned_stale[side]:
                    self._warned_stale[side] = False
                    _logger.info("%s tracker (%s) is tracking again", side, key)
            elif not self._warned_stale[side]:
                self._warned_stale[side] = True
                _logger.warning(
                    "%s tracker (%s) is %s — holding its last pose",
                    side,
                    key,
                    "not tracking" if sample is not None else "not reporting",
                )
        held = self._held.get(side)
        if held is not None:
            return held.pos, held.quat
        return _DEFAULT_POSE[side]

    def _side_grip(self, side: str) -> float:
        """Return the grip command to stream for one side.

        Fresh trigger data wins; a configured-but-stale trigger holds the
        last streamed grip (with a rate-limited warning) so a dropout never
        commands a jump. A side with no trigger configured streams 1.0
        (open), matching the pre-PCB behaviour.
        """
        reader = self._triggers[side]
        if reader is None:
            return 1.0
        grip = reader.grip()
        if grip is not None and not reader.is_stale():
            self._grip_held[side] = grip
            if self._warned_trigger[side]:
                self._warned_trigger[side] = False
                _logger.info("%s trigger node is reporting again", side)
        elif not self._warned_trigger[side]:
            self._warned_trigger[side] = True
            _logger.warning(
                "%s trigger node is %s — holding grip %.2f",
                side,
                "stale" if grip is not None else "not reporting",
                self._grip_held[side],
            )
        return self._grip_held[side]

    def compose_frame(self) -> dict:
        """Build one VRFrame JSON object from the latest tracker poses."""
        toggle, reset = self._controls.consume()
        if toggle and self._lock_pulse == 0:
            self._auto_engage_pending = False
            self._auto_engage_waiting_ack = False
            self._engaged = not self._engaged
            self._lock_pulse = _PULSE_FRAMES
            _logger.info(
                "engage toggle → %s", "engaged" if self._engaged else "disengaged"
            )
        if reset and self._reset_pulse == 0:
            self._reset_pulse = _PULSE_FRAMES
            _logger.info("reset requested")

        now = time.perf_counter()
        frame: dict = {}
        for side, ee_key, elbow_key in (
            ("left", "l_ee", "l_elbow"),
            ("right", "r_ee", "r_elbow"),
        ):
            pos, quat = self._side_pose(side, now)
            frame[ee_key] = {
                "position": {
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "z": float(pos[2]),
                },
                "quaternion": {
                    "x": float(quat[0]),
                    "y": float(quat[1]),
                    "z": float(quat[2]),
                    "w": float(quat[3]),
                },
            }
            # Elbow hints are ignored in absolute (Mantis) mode; stream the
            # tracker position so the field is well-formed.
            frame[elbow_key] = {
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
            }

        required_sides = [side for side, key in self._keys.items() if key is not None]
        if self._auto_engage_pending and all(
            side in self._held for side in required_sides
        ):
            self._auto_engage_pending = False
            self._auto_engage_waiting_ack = True
            _logger.info("trackers live — engaging Mantis tracking automatically")

        lock = self._lock_pulse > 0 or self._auto_engage_waiting_ack
        if self._lock_pulse > 0:
            self._lock_pulse -= 1
        frame["l_lock"] = lock
        frame["r_lock"] = lock

        frame["reset"] = self._reset_pulse > 0
        if self._reset_pulse > 0:
            self._reset_pulse -= 1

        grips = {
            "left": self._side_grip("left"),
            "right": self._side_grip("right"),
        }
        frame["l_grip"] = grips["left"]
        frame["r_grip"] = grips["right"]

        # A gesture may be issued on either side. Failure wins if two sides
        # happen to complete different gestures on the same frame.
        outcomes = [
            self._gesture[side].update(grips[side], now)
            for side in ("left", "right")
            if self._triggers[side] is not None
        ]
        gesture = (
            VREpisodeOutcome.FAILURE
            if VREpisodeOutcome.FAILURE in outcomes
            else VREpisodeOutcome.SUCCESS
            if VREpisodeOutcome.SUCCESS in outcomes
            else None
        )
        episode_outcome: VREpisodeOutcome | None = None
        if gesture == VREpisodeOutcome.SUCCESS:
            if self._state == VRState.DATA_COLLECTION:
                self._state = VRState.RECORDING
                _logger.info("triple trigger press → start recording")
            elif self._state == VRState.RECORDING:
                self._state = VRState.DATA_COLLECTION
                episode_outcome = VREpisodeOutcome.SUCCESS
                _logger.info("triple trigger press → end episode successfully")
        elif gesture == VREpisodeOutcome.FAILURE and self._state == VRState.RECORDING:
            self._state = VRState.DATA_COLLECTION
            episode_outcome = VREpisodeOutcome.FAILURE
            _logger.info("quadruple trigger press → end episode as failure")

        if episode_outcome is not None:
            # Repeat the transition tag just like reset/lock pulses: if the
            # first packet drops, the next DATA_COLLECTION frame still carries
            # the outcome while the receiver's previous state is RECORDING.
            self._episode_outcome = episode_outcome
            self._outcome_pulse = _PULSE_FRAMES
        frame["state"] = self._state.value
        frame["episode_outcome"] = (
            self._episode_outcome.value if self._outcome_pulse > 0 else None
        )
        if self._outcome_pulse > 0:
            self._outcome_pulse -= 1
            if self._outcome_pulse == 0:
                self._episode_outcome = None

        self._seq += 1
        frame["seq"] = self._seq
        # Stamp with the tracker sample's *capture* time (monotonic ms, like
        # performance.now()), not compose time: the server's interpolator
        # reconstructs this instant as ``t_host``, and Mantis recording aligns
        # dataset rows and camera exposures on it — stamping compose time
        # would fold the tracker→bridge latency into every recorded pose.
        # With two trackers the freshest capture stands in for both (the
        # sides sample within a driver poll of each other).
        cap_ts = [p.t for p in self._held.values()]
        frame["t"] = (max(cap_ts) if cap_ts else now) * 1000.0
        return frame

    # -- Streaming -----------------------------------------------------------

    async def run(self) -> None:
        """Stream frames until stdin quits, reconnecting on socket loss."""
        import websockets

        self._controls.start()
        uri = f"wss://{self._host}:{self._port}/ws"
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE  # the VR server's cert is self-signed

        if isinstance(self._controls, StdinControls):
            print(
                "Streaming tracker poses. Controls: Enter = engage/disengage, "
                "r = reset, q = quit; trigger x3 = start/success, "
                "trigger x4 = failure."
            )
        else:
            print(
                "Streaming tracker poses (managed by control panel); "
                "trigger x3 = start/success, trigger x4 = failure."
            )
        while not self._controls.quit.is_set():
            try:
                async with websockets.connect(uri, ssl=ssl_ctx, max_queue=4) as ws:
                    _logger.info("connected to %s", uri)
                    await self._stream(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect on any drop
                if self._controls.quit.is_set():
                    break
                _logger.warning("connection to %s lost (%s); retrying in 2s", uri, exc)
                await asyncio.sleep(2.0)

    async def _stream(self, ws) -> None:
        """Send frames at the configured rate over one connection."""
        drain = asyncio.create_task(self._drain(ws))
        interval = 1.0 / self._hz
        deadline = time.perf_counter()
        try:
            while not self._controls.quit.is_set():
                deadline += interval
                await ws.send(json.dumps(self.compose_frame()))
                await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
        finally:
            drain.cancel()

    async def _drain(self, ws) -> None:
        """Consume server broadcasts and mirror its authoritative state."""
        try:
            async for msg in ws:
                _logger.debug("server: %s", msg)
                try:
                    payload = json.loads(msg)
                except (TypeError, json.JSONDecodeError):
                    continue
                msg_type = payload.get("type")
                value = payload.get("value")
                if msg_type == "mode":
                    if value == "data_collection":
                        self._state = VRState.DATA_COLLECTION
                    elif value == "teleop":
                        self._state = VRState.TELEOP
                elif msg_type == "state":
                    try:
                        self._state = VRState(value)
                    except ValueError:
                        _logger.warning("server sent unknown VR state %r", value)
                elif msg_type == "tracking":
                    self._engaged = bool(value)
                    if self._engaged:
                        self._auto_engage_waiting_ack = False
        except Exception:  # noqa: BLE001 - connection teardown ends the drain
            pass
