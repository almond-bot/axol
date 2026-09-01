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
or ends one successfully; four presses ends one as a failure. A managed
plain-teleop/data-collection bridge waits for both trackers and triggers,
then requires both triggers released and squeezed together to align and
engage. Other flows use :class:`StdinControls` (the trigger frame carries no
buttons — session controls arrive with a later PCB revision): Enter toggles
tracking engage, ``r`` triggers a reset. A manual toggle is realised as a
short pulse of both lock bits — the shared teleop core enables on a rising
edge of both locks together and disables on a rising edge of either.

A side whose tracker stops reporting (occlusion, SLAM relocalising)
holds its last good pose rather than going quiet, so IK never chases a
glitch. Managed bridges freeze and wait for both sides to recover; the
operator must then release and squeeze both triggers together to re-anchor.
Standalone bridges retain manual engage control.
A stale trigger node likewise holds its last grip command, never jumping
on a dropout. Managed bridges treat trigger freshness as a safety input:
either trigger dropping out disengages tracking and recovery requires a new
two-trigger release→squeeze at the alignment pose.
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
import uuid
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from ..utils.ports import VR_PORT
from ..vr.models import VREpisodeOutcome, VRState
from .base import (
    TRACKER_PAIR_MAX_SKEW_S,
    TRACKER_POSE_MAX_AGE_S,
    TrackerPose,
    TrackerSource,
    TrackerSourceError,
    valid_tracker_pose,
)
from .trigger import TriggerReader

_logger = logging.getLogger(__name__)

# Keep direct ``q`` shutdown inside the managed context's five-second join
# budget even if a local server is stuck mid-handshake or close handshake.
_SOCKET_LIFECYCLE_TIMEOUT_S = 2.0
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


class TrackerBridgeError(RuntimeError):
    """A non-network bridge failure that cannot be fixed by reconnecting."""


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

    def cancel_sequence(self) -> None:
        """Forget pending presses while preserving the current press latch.

        Preserving ``_pressed`` matters when a four-press gesture resolves on
        the press edge: the held fourth squeeze must not become the first
        press of a new gesture on the next sample.
        """
        self._clear_sequence()

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
    queue carrying control-panel reset requests. Managed Mantis operations
    always track, so engage/disengage commands are deliberately ignored; the
    stdin-driven standalone bridge remains available for manual testing.
    """

    def __init__(self, stop_event: Any, command_queue: Any = None) -> None:
        self.quit = stop_event
        self._commands = command_queue

    def start(self) -> None:
        pass

    def consume(self) -> tuple[bool, bool]:
        reset = False
        if self._commands is None:
            return False, reset
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                break
            if command == "reset":
                reset = True
        return False, reset


class ManagedStdinControls:
    """Reset/quit controls for a direct managed Mantis operation.

    Unlike :class:`StdinControls`, Enter never toggles engagement: the managed
    release→squeeze safety handshake owns that state.  ``q`` sets the bridge's
    shared stop event and asks the context manager to interrupt the owning
    teleop/collection loop, while EOF is ignored so redirected stdin does not
    unexpectedly stop a robot operation.
    """

    def __init__(
        self,
        stop_event: Any,
        on_quit: Callable[[], None],
        *,
        input_stream: Any = None,
        activation_event: Any = None,
    ) -> None:
        self.quit = stop_event
        self._on_quit = on_quit
        self._input_stream = sys.stdin if input_stream is None else input_stream
        self._activation_event = activation_event
        self._reset_requests = 0
        self._lock = threading.Lock()
        self._started = False
        self._thread = threading.Thread(
            target=self._read_loop,
            daemon=True,
            name="mantis-stdin",
        )

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        self._thread.start()

    def _read_loop(self) -> None:
        if self._activation_event is not None:
            while not self.quit.is_set():
                if self._activation_event.wait(0.1):
                    break
            if self.quit.is_set():
                return
        for line in self._input_stream:
            if self.quit.is_set():
                return
            command = line.strip().lower()
            if command == "r":
                with self._lock:
                    self._reset_requests += 1
            elif command == "q":
                self.quit.set()
                try:
                    self._on_quit()
                except Exception:  # noqa: BLE001 - stdin thread must stay contained
                    _logger.exception("managed Mantis quit callback failed")
                return

    def consume(self) -> tuple[bool, bool]:
        """Return a reset request; managed stdin never toggles engagement."""
        with self._lock:
            reset = self._reset_requests > 0
            self._reset_requests = 0
        return False, reset


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
        auto_engage: Engage once all bound trackers have produced a live pose,
            and re-engage after the teleop core reports a forced disengage. The
            lock request stays asserted until the core acknowledges tracking,
            and every release stays low until the core echoes its transaction
            ID, so slow startup/reset/IK handling cannot miss either edge.
        confirm_auto_engage: When ``auto_engage`` is enabled and both trigger
            readers exist, require a deliberate release then simultaneous
            squeeze before each engage/re-engage. This makes the operator hold
            both rigs at the intended rest/alignment pose before the absolute
            world→base transform is fitted.
        pose_source_id: Logical producer ID placed on every frame. Managed
            operations supply the server's one-run token; standalone bridges
            generate their own stable ID.
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
        confirm_auto_engage: bool = False,
        pose_source_id: str | None = None,
    ) -> None:
        if left is None and right is None:
            raise ValueError(
                "no tracker is bound to either side — run `axol tracker.identify` first"
            )
        if left is not None and left == right:
            raise ValueError(
                f"left and right are both bound to {left!r}. Bind two distinct "
                "trackers with `axol tracker.identify`."
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
        # A missing reader is "live" for standalone/legacy bridges where the
        # input is intentionally optional. Managed Mantis bridges require two
        # readers at construction and update both values from CAN freshness.
        self._trigger_fresh: dict[str, bool] = {
            "left": left_trigger is None,
            "right": right_trigger is None,
        }
        self._warned_trigger: dict[str, bool] = {"left": False, "right": False}
        self._gesture = {
            "left": TriggerGestureRecognizer(),
            "right": TriggerGestureRecognizer(),
        }

        self._seq = 0
        # Stable across WebSocket reconnects for this bridge instance. The VR
        # server de-duplicates sequence numbers per logical source and gives a
        # tracker source exclusive control during managed Mantis runs.
        if pose_source_id is not None and (
            not isinstance(pose_source_id, str)
            or not pose_source_id.strip()
            or len(pose_source_id) > 128
        ):
            raise ValueError(
                "pose_source_id must be a non-empty string of at most 128 characters"
            )
        self._pose_source_id = pose_source_id or f"tracker-{uuid.uuid4()}"
        self._engaged = False
        self._auto_engage_enabled = auto_engage
        self._confirm_auto_engage = auto_engage and confirm_auto_engage
        if self._confirm_auto_engage and (
            left_trigger is None or right_trigger is None
        ):
            raise ValueError(
                "managed Mantis engagement requires fresh left and right trigger "
                "inputs; configure both trigger CAN channels"
            )
        self._engage_confirmation_needed = self._confirm_auto_engage
        self._engage_confirmation_armed = False
        self._auto_engage_pending = auto_engage and not self._confirm_auto_engage
        self._auto_engage_waiting_ack = False
        self._auto_engage_withdrawn = False
        self._auto_disengage_pending = False
        self._auto_disengage_waiting_ack = False
        self._auto_lock_release_id: int | None = None
        self._auto_lock_release_seq = 0
        self._lock_pulse = 0
        self._reset_pulse = 0
        self._outcome_pulse = 0
        self._episode_outcome: VREpisodeOutcome | None = None
        self._held: dict[str, TrackerPose] = {}
        self._fresh: dict[str, bool] = {"left": False, "right": False}
        self._warned_stale: dict[str, bool] = {"left": False, "right": False}
        self._warned_skew: dict[str, bool] = {"left": False, "right": False}
        # The server announces whether this connection belongs to teleop or
        # data collection. TELEOP is the safe default until that message lands.
        self._state = VRState.TELEOP

    def _request_auto_lock_release(self) -> None:
        """Hold managed lock bits low until the teleop core consumes them."""
        if self._auto_lock_release_id is None:
            self._auto_lock_release_seq += 1
            self._auto_lock_release_id = self._auto_lock_release_seq

    def _require_engage_confirmation(self) -> None:
        """Require a fresh two-trigger release→squeeze before auto-engaging."""
        if self._confirm_auto_engage:
            self._engage_confirmation_needed = True
            self._engage_confirmation_armed = False
            self._auto_engage_pending = False
        else:
            self._auto_engage_pending = True

    # -- Frame composition ---------------------------------------------------

    def _side_pose(
        self,
        side: str,
        now: float,
        poses: dict[str, TrackerPose],
        *,
        hold_updates: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``(pos, quat)`` to stream for one side.

        The last good pose wins over anything stale/untracked; a side with
        no data yet (or no tracker assigned) streams the fixed placeholder.
        ``hold_updates`` still observes freshness but latches the pre-loss
        pose until the core confirms it has disengaged; a recovered pose must
        first appear on the fresh re-engage edge, never under the old base fit.
        """
        key = self._keys[side]
        self._fresh[side] = False
        if key is not None:
            sample = poses.get(key)
            if (
                sample is not None
                and valid_tracker_pose(sample)
                and sample.tracking
                and now - sample.t <= TRACKER_POSE_MAX_AGE_S
            ):
                self._fresh[side] = True
                if not hold_updates:
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
            self._trigger_fresh[side] = True
            return 1.0
        grip = reader.grip()
        fresh = grip is not None and not reader.is_stale()
        self._trigger_fresh[side] = fresh
        if fresh:
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
            # A manual control source takes ownership of engage state. Managed
            # bridges never emit toggles, while this keeps a custom/standalone
            # auto-engage bridge from immediately undoing its operator's
            # explicit disengage request.
            self._auto_engage_enabled = False
            self._auto_engage_pending = False
            self._auto_engage_waiting_ack = False
            self._auto_engage_withdrawn = False
            self._auto_disengage_pending = False
            self._auto_disengage_waiting_ack = False
            self._auto_lock_release_id = None
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
        # One atomic-ish backend snapshot per frame. Calling ``poses()`` once
        # per side lets an intervening callback pair unrelated instants and
        # hides real left/right skew.
        poses = self._source.poses()
        hold_pose_updates = self._auto_engage_enabled and (
            self._auto_engage_withdrawn
            or self._auto_disengage_pending
            or self._auto_disengage_waiting_ack
        )
        for side, ee_key, elbow_key in (
            ("left", "l_ee", "l_elbow"),
            ("right", "r_ee", "r_elbow"),
        ):
            pos, quat = self._side_pose(
                side, now, poses, hold_updates=hold_pose_updates
            )
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
            frame[f"{side[0]}_tracked"] = self._fresh[side]

        left_held = self._held.get("left")
        right_held = self._held.get("right")
        if (
            self._fresh["left"]
            and self._fresh["right"]
            and left_held is not None
            and right_held is not None
            and abs(left_held.t - right_held.t) > TRACKER_PAIR_MAX_SKEW_S
        ):
            older = "left" if left_held.t < right_held.t else "right"
            self._fresh[older] = False
            frame[f"{older[0]}_tracked"] = False
            if not self._warned_skew[older]:
                self._warned_skew[older] = True
                _logger.warning(
                    "%s tracker sample is %.0f ms behind the other side — "
                    "holding and requiring a fresh Mantis alignment",
                    older,
                    abs(left_held.t - right_held.t) * 1000.0,
                )
        else:
            self._warned_skew = {"left": False, "right": False}

        required_sides = [side for side, key in self._keys.items() if key is not None]
        all_required_fresh = all(self._fresh[side] for side in required_sides)
        grips = {
            "left": self._side_grip("left"),
            "right": self._side_grip("right"),
        }
        frame["l_grip"] = grips["left"]
        frame["r_grip"] = grips["right"]
        frame["l_trigger_live"] = self._trigger_fresh["left"]
        frame["r_trigger_live"] = self._trigger_fresh["right"]
        all_required_triggers_fresh = all(
            self._trigger_fresh[side] for side in ("left", "right")
        )
        all_required_inputs_fresh = all_required_fresh and (
            all_required_triggers_fresh or not self._confirm_auto_engage
        )

        # A partial gesture cannot straddle a CAN dropout. In particular, a
        # held stale grip must never resolve a triple press and start/finish a
        # recording after its trigger node has stopped reporting.
        for side, recognizer in self._gesture.items():
            if self._triggers[side] is not None and not self._trigger_fresh[side]:
                recognizer.cancel_sequence()

        confirmation_press = False
        if not all_required_inputs_fresh and self._engage_confirmation_needed:
            # A release observed before a tracker or trigger dropout cannot
            # authorize a later engage. Recovery starts a new cycle.
            self._engage_confirmation_armed = False
        if (
            self._auto_engage_enabled
            and self._engage_confirmation_needed
            and not self._engaged
            and all_required_inputs_fresh
        ):
            if all(value >= _GESTURE_RELEASE_GRIP for value in grips.values()):
                self._engage_confirmation_armed = True
            elif self._engage_confirmation_armed and all(
                value <= _GESTURE_PRESS_GRIP for value in grips.values()
            ):
                self._engage_confirmation_needed = False
                self._engage_confirmation_armed = False
                self._auto_engage_pending = True
                confirmation_press = True
                for side, recognizer in self._gesture.items():
                    # Consume the alignment squeeze in each recognizer so a
                    # still-held trigger cannot become press #1 of the episode
                    # triple-click gesture on the next frame.
                    recognizer.update(grips[side], now)
                    recognizer.cancel_sequence()
                _logger.info("alignment confirmed with both triggers — engaging Mantis")
        if self._auto_engage_waiting_ack and not all_required_inputs_fresh:
            # Withdraw an unacknowledged engage request if tracking drops in
            # the meantime. Keeping the level asserted would let the core
            # eventually engage against historical held poses.
            self._auto_engage_waiting_ack = False
            self._auto_engage_withdrawn = True
            self._require_engage_confirmation()
            self._request_auto_lock_release()
        if (
            self._auto_engage_enabled
            and self._engaged
            and not all_required_inputs_fresh
            and not self._auto_disengage_pending
            and not self._auto_disengage_waiting_ack
        ):
            # Held poses are safe only as a stationary fallback. Toggle the
            # core out of tracking before a returning tracker can jump under
            # the old world→base fit; once both sides are fresh, the normal
            # auto-engage path takes a new alignment snapshot.
            self._require_engage_confirmation()
            self._auto_disengage_pending = True
            _logger.warning(
                "tracker or trigger freshness lost — freezing Mantis until "
                "both sides recover"
            )
        if (
            self._auto_disengage_pending
            and self._engaged
            and self._auto_lock_release_id is None
            and not self._auto_engage_waiting_ack
            and not self._auto_disengage_waiting_ack
        ):
            self._auto_disengage_pending = False
            self._auto_disengage_waiting_ack = True
        elif self._auto_disengage_pending and not self._engaged:
            self._auto_disengage_pending = False
        if (
            self._auto_engage_enabled
            and self._auto_engage_pending
            and not self._engaged
            and all_required_inputs_fresh
            and not self._auto_disengage_pending
            and not self._auto_disengage_waiting_ack
            and self._auto_lock_release_id is None
        ):
            self._auto_engage_pending = False
            self._auto_engage_waiting_ack = True
            _logger.info("trackers live and aligned — engaging Mantis tracking")

        lock = (
            self._lock_pulse > 0
            or self._auto_engage_waiting_ack
            or self._auto_disengage_waiting_ack
        )
        if self._auto_lock_release_id is not None:
            # Toggle-mode engagement needs a genuine low→high edge. Keep the
            # wire low until the core explicitly confirms it consumed this
            # release; an IK solve can block far longer than a timed pulse.
            lock = False
            frame["lock_release_id"] = self._auto_lock_release_id
        if self._lock_pulse > 0:
            self._lock_pulse -= 1
        frame["l_lock"] = lock
        frame["r_lock"] = lock

        frame["reset"] = self._reset_pulse > 0
        if self._reset_pulse > 0:
            self._reset_pulse -= 1

        # A gesture may be issued on either side. Failure wins if two sides
        # happen to complete different gestures on the same frame.
        outcomes = (
            []
            if confirmation_press
            or (self._confirm_auto_engage and not all_required_inputs_fresh)
            else [
                self._gesture[side].update(grips[side], now)
                for side in ("left", "right")
                if self._triggers[side] is not None and self._trigger_fresh[side]
            ]
        )
        gesture = (
            VREpisodeOutcome.FAILURE
            if VREpisodeOutcome.FAILURE in outcomes
            else VREpisodeOutcome.SUCCESS
            if VREpisodeOutcome.SUCCESS in outcomes
            else None
        )
        if gesture is not None:
            # The operator commonly squeezes both Mantis triggers together.
            # Their CAN samples can cross the 0.6 s resolution deadline one
            # frame apart: without consuming both pending sequences, the
            # first triple starts an episode and the other immediately ends
            # it. Treat contemporaneous two-handed presses as one gesture.
            for recognizer in self._gesture.values():
                recognizer.cancel_sequence()
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
        frame["pose_source_id"] = self._pose_source_id
        frame["pose_source_kind"] = "tracker"
        # Stamp the time this latest-known pair became available: the newest
        # contributing sample. Using the older side's time would place a pose
        # update from the future into an earlier frame. The other side is a
        # bounded (<= TRACKER_PAIR_MAX_SKEW_S) zero-order hold. Backends use
        # native capture time when exposed and earliest host receipt otherwise;
        # see TrackerPose.timestamp_is_capture. Invalid/stale frames use compose
        # time so the tracking-loss control edge remains monotonic.
        required_held = [
            self._held[side]
            for side in required_sides
            if side in self._held and self._fresh[side]
        ]
        all_required_held_fresh = len(required_held) == len(required_sides)
        pair_t = max(p.t for p in required_held) if all_required_held_fresh else now
        frame["t"] = pair_t * 1000.0
        return frame

    def _handle_tracking_state(self, engaged: bool) -> None:
        """Adopt a server tracking acknowledgement for managed auto-engage."""
        was_engaged = self._engaged
        self._engaged = engaged

        if not self._auto_engage_enabled:
            return

        if engaged:
            if self._auto_disengage_waiting_ack:
                # A reconnect can seed the current true state while our
                # disengage toggle is still waiting to be consumed.
                return
            if self._auto_engage_waiting_ack:
                self._auto_engage_waiting_ack = False
                self._auto_engage_pending = False
                self._auto_engage_withdrawn = False
            elif self._auto_engage_withdrawn:
                # The core consumed an engage just before freshness loss made
                # us withdraw it. Complete a deliberate off→on cycle even if
                # the tracker already recovered, so absolute IK re-anchors.
                self._auto_engage_withdrawn = False
                self._auto_disengage_pending = True
                self._require_engage_confirmation()
            else:
                # Seed from an already-engaged core (for example after bridge
                # reconnect). Adopt it instead of toggling it off.
                self._auto_engage_pending = False
            self._request_auto_lock_release()
            return

        if self._auto_engage_waiting_ack and not was_engaged:
            # The server seeds its existing false state when this WebSocket
            # connects. It is not an acknowledgement of our newly asserted
            # engage request, so keep that request high until a true arrives.
            return

        self._auto_engage_waiting_ack = False
        self._auto_disengage_waiting_ack = False
        self._auto_disengage_pending = False
        self._require_engage_confirmation()
        if was_engaged:
            # Whether this was our stale-tracker toggle or an out-of-band
            # reset/contact stop, make the core prove it consumed a low frame
            # before the next engage edge is allowed.
            self._request_auto_lock_release()

    def _handle_lock_release(self, value: object) -> None:
        """Complete the managed low-level handshake for the matching ID."""
        try:
            release_id = int(value)
        except (TypeError, ValueError):
            return
        if release_id != self._auto_lock_release_id:
            return
        self._auto_lock_release_id = None
        if self._auto_engage_withdrawn and not self._engaged:
            # If the high frame had engaged the core, its ordered tracking=true
            # broadcast would precede this release acknowledgement.
            self._auto_engage_withdrawn = False

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
        elif isinstance(self._controls, ManagedStdinControls):
            print(
                "Streaming managed Mantis tracker poses. Controls: r = reset, "
                "q = stop; engagement uses the two-trigger alignment handshake; "
                "trigger x3 = start/success, x4 = failure."
            )
        else:
            if self._confirm_auto_engage:
                print(
                    "Streaming tracker poses. Hold both rigs at the alignment "
                    "pose, release both triggers, then squeeze both together "
                    "to engage; trigger x3 = start/success, x4 = failure."
                )
            else:
                print(
                    "Streaming tracker poses (tracking managed automatically); "
                    "trigger x3 = start/success, trigger x4 = failure."
                )
        while not self._controls.quit.is_set():
            try:
                async with websockets.connect(
                    uri,
                    ssl=ssl_ctx,
                    max_queue=4,
                    open_timeout=_SOCKET_LIFECYCLE_TIMEOUT_S,
                    close_timeout=_SOCKET_LIFECYCLE_TIMEOUT_S,
                ) as ws:
                    _logger.info("connected to %s", uri)
                    await self._stream(ws)
            except asyncio.CancelledError:
                raise
            except (TrackerSourceError, TrackerBridgeError):
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect on any drop
                if self._controls.quit.is_set():
                    break
                # A socket failure and a backend failure can race. Probe the
                # source before entering the reconnect delay so a dead reader
                # cannot leave its owning operation alive indefinitely.
                try:
                    self._source.poses()
                except TrackerSourceError:
                    raise
                except Exception as source_exc:  # noqa: BLE001 - backend API varies
                    raise TrackerBridgeError(
                        "tracker backend health check failed"
                    ) from source_exc
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
                try:
                    payload = json.dumps(self.compose_frame())
                except TrackerSourceError:
                    raise
                except Exception as exc:  # noqa: BLE001 - composition is fatal
                    raise TrackerBridgeError(
                        f"could not compose tracker frame ({type(exc).__name__}: {exc})"
                    ) from exc
                await ws.send(payload)
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
                    self._handle_tracking_state(bool(value))
                elif msg_type == "lock_release":
                    self._handle_lock_release(value)
        except Exception:  # noqa: BLE001 - connection teardown ends the drain
            pass
