"""Shared VR-teleoperation state machine (engage + smoothing + reset).

:class:`VRTeleopCore` is the single source of truth for the per-step control
logic that *both* teleop flows drive:

  - native :class:`almond_axol.teleop.teleop.VRTeleop` (``axol teleop``), and
  - the LeRobot adapter
    :class:`almond_axol.lerobot.teleop.teleop_vr.AxolVRTeleop` (``axol
    collect-data`` / ``run-policy``).

The two classes differ only in *transport* — native returns ``(left, right)``
joint tuples, the LeRobot adapter returns a ``RobotAction`` dict — and in how
they own the VR server, robot link, and IK subprocess. The engage toggle, the
EMA + trapezoidal smoothing (with gripper bypass), the post-engage velocity
restore, and the reset / startup-trajectory handling all live here so the two
flows cannot drift apart.

Threading contract (identical for both adapters):
  - The IK-dispatch thread calls :meth:`run_ik_loop`, which advances the engage
    state, dispatches resets, and publishes raw IK targets (:meth:`set_target`).
  - The control-loop thread calls :meth:`compute_output` once per step — the
    *only* place the smoothing filters are advanced, so it must run at the
    steady control rate, not the slower/jittery IK rate.
The few cross-thread float writes (``max_vel`` / ``engage_time`` set on the IK
thread and read on the control loop; ``reset_interp`` touched by both; the
pose heartbeat ``_last_frame_time`` set on the VR thread and read on the IK
thread) are single-word and benign — this matches the original design of both
adapters.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing.connection
import threading
import time
from collections.abc import Awaitable, Callable

import numpy as np

from ..robot.control import ContactWatchdog
from .config import VRTeleopConfig
from .filter import AlphaSmoothFilter, ResetInterpolator, TrapezoidalFilter
from .recorder import make as _recorder_make

_IK_RECV_TIMEOUT = 5.0  # seconds; avoid blocking forever if IK process hangs

# How long a guarded-return contact hold keeps waiting for a reset press
# after the VR frame stream has gone dead (headset exited XR or died —
# nobody is left to press reset). Long enough to free a hooked gripper
# calmly; then the hold settles into a position hold where the arms are.
_HOLD_ORPHAN_GRACE_S = 30.0


def recv_with_timeout(
    conn: multiprocessing.connection.Connection,
    timeout: float,
    stop_event: threading.Event | None = None,
) -> object | None:
    """Return ``conn.recv()`` if data arrives within ``timeout``, else ``None``.

    Polls in short intervals so ``stop_event`` can interrupt a long wait.
    """
    poll_interval = 0.05
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        if stop_event is not None and stop_event.is_set():
            return None
        if conn.poll(min(poll_interval, remaining)):
            return conn.recv()


class VRTeleopCore:
    """Engage + smoothing + reset state machine shared by both teleop flows.

    Args:
        config: Teleop loop parameters (frequency, velocity limits, EMA alpha).
        logger: Logger for engage/reset messages, so they appear under the
            owning adapter's module name (parity with the pre-split logs).
        broadcast_tracking: Callback ``(enabled: bool) -> None`` that pushes the
            engage state to the headset. Safe to call before the VR server
            exists (the adapter's implementation guards that).
    """

    def __init__(
        self,
        config: VRTeleopConfig,
        logger: logging.Logger,
        broadcast_tracking: Callable[[bool], None],
    ) -> None:
        self.config = config
        self._logger = logger
        self._broadcast = broadcast_tracking

        dt = 1.0 / config.frequency
        # ik_alpha is specified as a per-tick blend at the historical 120 Hz
        # control rate; convert it to this rate's per-tick alpha so the EMA's
        # *time constant* — the thing that was tuned — is rate-invariant.
        alpha = 1.0 - (1.0 - config.ik_alpha) ** (120.0 * dt)
        self.ema_left = AlphaSmoothFilter(alpha)
        self.ema_right = AlphaSmoothFilter(alpha)
        self.smooth_left = TrapezoidalFilter(
            config.teleop_max_vel, config.teleop_max_accel, dt
        )
        self.smooth_right = TrapezoidalFilter(
            config.teleop_max_vel, config.teleop_max_accel, dt
        )
        self.reset_interp = ResetInterpolator()

        # Raw IK solution (full URDF vector) + per-arm joint indices into it.
        self.q: np.ndarray | None = None
        self.left_indices: list[int] = []
        self.right_indices: list[int] = []
        self.l_grip: float = 0.0
        self.r_grip: float = 0.0

        # Target playback segment: IK solutions arrive at the (irregular)
        # solve cadence, and tracking the newest one latest-wins makes the
        # control loop chase a staircase — each new solution lands as a step
        # of hand-speed x solve-interval, which the EMA/trapezoid bound but
        # don't hide (acceleration can flip in one tick). Instead, each new
        # solution starts a short linear segment from the currently *rendered*
        # target to the new solution, played over roughly one solve interval,
        # so the control loop tracks a continuous, C0 target. Costs about one
        # solve interval (~10 ms) of extra latency. The tuple is replaced
        # atomically (single reference write) for the control thread.
        self._segment: tuple[np.ndarray, np.ndarray, float, float] | None = None
        self._seg_last_arrival: float | None = None

        # Output guard: the hard contract that no two consecutive arm
        # commands differ by more than teleop_max_vel / frequency per joint
        # (~52 mrad at defaults — double any human-driven joint speed, so a
        # larger step is by definition an upstream bug or bypass). The
        # trapezoid already enforces this on the tracking path; the guard
        # extends it to *every* path (reset/startup playback included),
        # spreading an oversized step across ticks and warning when it had to
        # — violations become impossible and visible. Cleared at the points
        # where the command is legitimately re-adopted to match the measured
        # arm (connect seeding, resync after hand-guiding).
        self._last_cmd: np.ndarray | None = None
        self._guard_warn_time: float = 0.0

        # Engage state (mutated on the IK thread). Per-arm: a disengaged arm
        # freezes where it is (pose + gripper) while the other keeps
        # tracking. From rest both arms always engage together (both grips);
        # once engaged, grips act per-arm — as rising-edge toggles by
        # default, or as dead-man holds with ``config.hold_to_engage``.
        self.left_enabled: bool = False
        self.right_enabled: bool = False
        self._prev_both: bool = False
        self._prev_l_lock: bool = False
        self._prev_r_lock: bool = False
        # Both-grips gate for the next engage. Set at startup, at rest, and by
        # every forced disengage (reset, stale stream, contact) so resuming is
        # always deliberate; cleared once engaged, so an operator who froze
        # both arms mid-task can resume either with a single click.
        self._require_both_engage: bool = True
        self._at_rest: bool = True
        self._engage_time: float | None = None

        # Reset latch (set from the VR frame callback / programmatically).
        self._prev_reset: bool = False
        self._reset_latched: bool = False
        # True from reset dispatch (latch consumed) until the planned
        # trajectory is adopted, so ``is_resetting`` has no false window
        # while the IK worker plans (planning can take seconds).
        self._reset_dispatching: bool = False
        # Set by cancel_reset() while a dispatch is in flight: the planning
        # round trip can't be interrupted, so the dispatch drops its result
        # instead of adopting a trajectory the caller no longer wants.
        self._reset_cancel: bool = False

        # IK pipeline pause (set while the arms are moved out-of-band, e.g.
        # hand-guided under gravity compensation). While paused the IK loop
        # idles: no frames are dispatched, grips can't engage tracking, and a
        # latched reset is *held* — the owner re-syncs ``q`` to the measured
        # positions (see :meth:`resync_to_positions`) before resuming, so the
        # reset trajectory plans from where the arms actually are.
        self._ik_paused: bool = False

        # Arrival time of the most recent raw VR frame (written on the VR
        # server thread in note_frame_reset, read on the IK thread). Drives
        # the stale-stream auto-disengage: identity-stable render frames make
        # "no new frame from get_frame()" indistinguishable from an operator
        # holding perfectly still, so staleness must be measured at ingest.
        self._last_frame_time: float | None = None

        # Jitter flight recorder (--teleop.record, see .recorder):
        # taps the smoothing stages this class owns per control tick — the
        # segment-rendered raw target, the EMA output, and the final guarded
        # command (7 left + 7 right arm joints each).
        self._rec = _recorder_make(
            self.config.record, "cmd", {"tgt": 14, "ema": 14, "out": 14}
        )

    # ------------------------------------------------------------------
    # Seeding (called once at connect, before the IK loop starts)
    # ------------------------------------------------------------------

    def set_initial_grips(self, l_grip: float | None, r_grip: float | None) -> None:
        """Seed gripper targets from the robot's current positions."""
        if l_grip is not None:
            self.l_grip = float(l_grip)
        if r_grip is not None:
            self.r_grip = float(r_grip)

    def set_solution(
        self, q_init: np.ndarray, left_indices: list[int], right_indices: list[int]
    ) -> None:
        """Adopt the IK worker's initial solution + per-arm joint index maps."""
        self._hold_target(np.asarray(q_init, dtype=np.float32))
        self.left_indices = left_indices
        self.right_indices = right_indices

    def seed_filters(
        self, cur_left: np.ndarray | None, cur_right: np.ndarray | None
    ) -> None:
        """Reset the EMA + trapezoidal filters to the current arm positions.

        ``cur_left`` / ``cur_right`` are ``(>=8,)`` position vectors (7 arm
        joints + gripper); ``None`` skips that arm. Seeding here means the first
        step produces no transient.
        """
        if cur_left is not None:
            seed_l = np.append(cur_left[:7], self.l_grip)
            self.ema_left.reset(seed=seed_l)
            self.smooth_left.reset(seed=seed_l[:7])
        if cur_right is not None:
            seed_r = np.append(cur_right[:7], self.r_grip)
            self.ema_right.reset(seed=seed_r)
            self.smooth_right.reset(seed=seed_r[:7])
        # The command is being re-adopted to the measured arm: the output
        # guard must not rate-limit across the adoption (the arm is already
        # there; clamping would command it back toward the stale position).
        self._last_cmd = None

    def set_startup_trajectory(self, trajectory: object) -> None:
        """Queue the IK worker's startup trajectory for playback in compute_output."""
        if trajectory:
            self.reset_interp.set_trajectory(trajectory, self.l_grip, self.r_grip)

    def resync_to_positions(
        self, cur_left: np.ndarray | None, cur_right: np.ndarray | None
    ) -> None:
        """Re-adopt the measured arm positions after an out-of-band move.

        Call while the IK loop is paused (:meth:`pause_ik`), after the arms
        were moved outside teleop (e.g. hand-guided under gravity comp):
        writes the measured joints into the raw IK solution ``q`` — so the
        next reset plans from where the arms actually are — and re-seeds the
        grips and smoothing filters so the first command back out produces no
        transient. ``cur_left`` / ``cur_right`` are ``(>=8,)`` position
        vectors (7 arm joints + gripper, gripper normalized [0, 1]); ``None``
        skips that arm.
        """
        if self.q is not None:
            q = self.q.copy()
            if cur_left is not None and self.left_indices:
                q[self.left_indices] = np.asarray(cur_left[:7], dtype=np.float32)
            if cur_right is not None and self.right_indices:
                q[self.right_indices] = np.asarray(cur_right[:7], dtype=np.float32)
            self._hold_target(q)
        self.set_initial_grips(
            float(cur_left[7]) if cur_left is not None and len(cur_left) > 7 else None,
            float(cur_right[7])
            if cur_right is not None and len(cur_right) > 7
            else None,
        )
        self.seed_filters(cur_left, cur_right)

    # ------------------------------------------------------------------
    # Reset control
    # ------------------------------------------------------------------

    def note_frame_reset(self, reset: bool) -> None:
        """Latch a reset on the rising edge of the VR reset button.

        Called from the VR frame callback for every frame so a short press
        can't be missed while the IK loop blocks on ``conn.recv``. Doubles as
        the pose-stream heartbeat for the stale-stream auto-disengage (see
        :meth:`_maybe_disengage_stale`).
        """
        self._last_frame_time = time.perf_counter()
        if reset and not self._prev_reset:
            self._reset_latched = True
        self._prev_reset = reset

    def request_reset(self) -> None:
        """Programmatically trigger a return-to-rest move. Safe from any thread."""
        self._reset_latched = True

    def clear_reset_request(self) -> None:
        """Consume a pending reset latch without acting on it.

        Used when a discard/rerecord already returns to rest, so the latched
        reset press shouldn't also trigger a rest-pose move.
        """
        self._reset_latched = False

    def cancel_reset(self) -> None:
        """Abandon a pending, planning, or playing reset move.

        Clears the latch, cancels any trajectory playback, and marks an
        in-flight planning dispatch so its result is dropped when it returns
        (the blocking round trip itself can't be interrupted). Used by
        guarded moves that bail out on contact: call :meth:`pause_ik` first
        so no new dispatch can start, then this. Safe from any thread.
        """
        self._reset_latched = False
        self._reset_cancel = True
        self.reset_interp.clear()

    @property
    def reset_pending(self) -> bool:
        """True while a reset press is latched but not yet dispatched.

        Distinct from :attr:`is_resetting`, which also covers planning and
        playback: a paused IK loop holds the latch (see :meth:`pause_ik`),
        so this is the "operator asked to go home" signal a hold loop waits
        on.
        """
        return self._reset_latched

    @property
    def is_resetting(self) -> bool:
        """True while a reset is pending, being planned, or playing back."""
        return (
            self._reset_latched
            or self._reset_dispatching
            or self.reset_interp.is_active()
        )

    # ------------------------------------------------------------------
    # IK pipeline pause (out-of-band moves)
    # ------------------------------------------------------------------

    def pause_ik(self) -> None:
        """Idle the IK loop while the arms are moved out-of-band.

        While paused, VR frames are not dispatched (grips can't engage
        tracking) and a latched reset press is held instead of dispatched, so
        it can't plan from a solution that no longer matches the hand-guided
        arms. Re-sync with :meth:`resync_to_positions`, then
        :meth:`resume_ik`. Safe to call from any thread.
        """
        self._ik_paused = True

    def resume_ik(self) -> None:
        """Resume the IK loop after :meth:`pause_ik`.

        A reset latched during the pause dispatches on the next IK loop
        iteration, planning from the (re-synced) current ``q``.
        """
        self._ik_paused = False

    # ------------------------------------------------------------------
    # Engage toggle + IK target (IK thread)
    # ------------------------------------------------------------------

    @property
    def teleop_enabled(self) -> bool:
        """True while at least one arm is engaged (tracking)."""
        return self.left_enabled or self.right_enabled

    def _disengage_all(self, log_message: str | None = None) -> None:
        """Disengage both arms and clear the edge/ramp state (IK thread).

        Used by every non-toggle disengage path — reset adoption, the
        stale-stream watchdog, a tracking contact trip — so re-engaging
        afterwards always requires a deliberate both-grips engage.
        """
        was_enabled = self.teleop_enabled
        self.left_enabled = False
        self.right_enabled = False
        self._prev_both = False
        self._prev_l_lock = False
        self._prev_r_lock = False
        self._require_both_engage = True
        self._engage_time = None
        if log_message is not None and was_enabled:
            self._logger.info(log_message)
        self._broadcast(False)

    def update_engage(self, frame: object) -> None:
        """Advance the per-arm engage state and grip tracking for one VR frame.

        From rest — and after any forced disengage (reset, stale stream,
        contact) — both grips together are required to engage, and both arms
        start tracking. Once engaged, grips act per-arm:

          - Toggle scheme (default): a rising edge on a grip toggles *that*
            arm between tracking and frozen. A frozen arm holds its pose and
            gripper while the other keeps tracking; a click re-engages it.
          - Dead-man scheme (``config.hold_to_engage``): each arm tracks only
            while its grip is held; releasing a grip freezes that arm, and
            releasing both disengages the session (both grips must be held
            again to resume).

        On the first engage out of rest, the velocity cap starts at
        ``engage_max_vel`` and smoothsteps up to ``teleop_max_vel`` across
        ``engage_duration`` (advanced in :meth:`compute_output`).
        """
        l_lock = bool(frame.l_lock)
        r_lock = bool(frame.r_lock)
        both = l_lock and r_lock
        was_left = self.left_enabled
        was_right = self.right_enabled
        was_enabled = was_left or was_right

        if not was_enabled and (
            self._require_both_engage or self.config.hold_to_engage
        ):
            # Engage gate: both grips together. Rising-edge in toggle mode so
            # a held-over press can't re-engage; level-triggered in hold mode
            # (the operator is *holding*, there is no release to edge on).
            engage = (
                both if self.config.hold_to_engage else (both and not self._prev_both)
            )
            if engage:
                self.left_enabled = True
                self.right_enabled = True
                self._require_both_engage = False
        elif self.config.hold_to_engage:
            # Dead-man: each arm tracks exactly while its grip is held.
            self.left_enabled = l_lock
            self.right_enabled = r_lock
        else:
            # Toggle: a rising edge on a grip flips that arm. Also reached
            # fully-disengaged when the operator froze both arms themselves
            # (no forced disengage in between): a single click resumes.
            if l_lock and not self._prev_l_lock:
                self.left_enabled = not self.left_enabled
            if r_lock and not self._prev_r_lock:
                self.right_enabled = not self.right_enabled

        now_enabled = self.left_enabled or self.right_enabled
        if now_enabled and not was_enabled:
            self._logger.info("Teleop enabled")
            self._broadcast(True)
            if self._at_rest:
                self.smooth_left.max_vel = self.config.engage_max_vel
                self.smooth_right.max_vel = self.config.engage_max_vel
                self._engage_time = time.perf_counter()
                self._at_rest = False
        elif was_enabled and not now_enabled:
            self._logger.info("Teleop disabled")
            self._broadcast(False)
        elif was_enabled:
            if self.left_enabled != was_left:
                self._logger.info(
                    "Left arm %s", "engaged" if self.left_enabled else "frozen"
                )
            if self.right_enabled != was_right:
                self._logger.info(
                    "Right arm %s", "engaged" if self.right_enabled else "frozen"
                )

        self._prev_both = both
        self._prev_l_lock = l_lock
        self._prev_r_lock = r_lock

        # Only track a gripper while its arm is engaged, so a frozen arm's
        # grasp (and a disengaged session) can't be actuated by the trigger.
        if self.left_enabled:
            self.l_grip = frame.l_grip
        if self.right_enabled:
            self.r_grip = frame.r_grip

    def set_target(self, q_raw: object) -> None:
        """Publish a fresh raw IK solution (consumed by compute_output).

        Starts a playback segment from the currently rendered target to the
        new solution (see ``_segment`` in ``__init__``) so the control loop
        never sees a step. Late solutions (a dispatch stall) play back over a
        capped window instead of landing at once.
        """
        now = time.perf_counter()
        q_new = np.asarray(q_raw, dtype=np.float32)
        start = self._eval_target(now)
        if start is None:
            start = q_new
        gap = (
            now - self._seg_last_arrival
            if self._seg_last_arrival is not None
            else 1.0 / self.config.frequency
        )
        self._seg_last_arrival = now
        # Play over the observed solve gap (floor: a couple control ticks) so
        # normal cadence adds ~one solve interval of lag; cap the window so a
        # stalled solve's accumulated motion is spread, not slow-motion.
        dur = min(max(gap, 2.0 / self.config.frequency), 0.04)
        self.q = q_new
        self._segment = (start, q_new, now, dur)

    def _eval_target(self, now: float) -> np.ndarray | None:
        """Render the interpolated raw target at time ``now`` (any thread)."""
        seg = self._segment
        if seg is None:
            return None
        start, end, t0, dur = seg
        u = (now - t0) / dur
        if u >= 1.0:
            return end
        if u <= 0.0:
            return start.copy()
        return (start + u * (end - start)).astype(np.float32)

    def _hold_target(self, q: np.ndarray) -> None:
        """Adopt ``q`` as a stationary target (reset/startup adoption paths)."""
        self.q = np.asarray(q, dtype=np.float32)
        self._segment = None
        self._seg_last_arrival = None

    # ------------------------------------------------------------------
    # Smoothing (control-loop thread)
    # ------------------------------------------------------------------

    def compute_output(self) -> np.ndarray | None:
        """Return the smoothed 16-DOF command, or ``None`` if no target yet.

        Layout: ``[0:7]`` left arm, ``[7]`` left grip, ``[8:15]`` right arm,
        ``[15]`` right grip. Advances the EMA + trapezoidal filters (and any
        active reset/startup trajectory) by one step, so it must be called once
        per control step at the steady control rate — applied at the slower IK
        rate the setpoint stair-cases and the velocity feedforward turns each
        jump into a torque spike (jerk).
        """
        # Flight recorder covers engaged segments only: gate before any early
        # return so the disengage edge writes the _cmd file even while no
        # target exists yet or a reset trajectory is playing.
        if self._rec is not None:
            self._rec.set_engaged(self.teleop_enabled)

        # Post-engage velocity ramp: smoothstep the cap from engage_max_vel to
        # teleop_max_vel across engage_duration. The old behaviour held the
        # low cap for the whole window and then stepped to full speed — error
        # accumulated during the slow phase was released all at once, so
        # moving immediately after engage felt dead and then lurched.
        if self._engage_time is not None:
            u = (time.perf_counter() - self._engage_time) / max(
                self.config.engage_duration, 1e-6
            )
            if u >= 1.0:
                v = self.config.teleop_max_vel
                self._engage_time = None
            else:
                s = u * u * (3.0 - 2.0 * u)
                v = self.config.engage_max_vel + s * (
                    self.config.teleop_max_vel - self.config.engage_max_vel
                )
            self.smooth_left.max_vel = v
            self.smooth_right.max_vel = v

        # Track the interpolated target segment (see set_target), not the
        # latest-wins staircase; fall back to the raw solution before the
        # first segment exists.
        q = self._eval_target(time.perf_counter())
        if q is None:
            q = self.q
        if q is None:
            return None

        if self.reset_interp.is_active():
            new_q, l_grip, r_grip, done = self.reset_interp.step()
            if new_q is None:
                return None
            q = np.asarray(new_q, dtype=np.float32)
            if done:
                self._hold_target(q.copy())
                self.l_grip = l_grip
                self.r_grip = r_grip
                self._at_rest = True
                seed_l = np.append(q[self.left_indices], l_grip)
                seed_r = np.append(q[self.right_indices], r_grip)
                self.ema_left.reset(seed=seed_l)
                self.ema_right.reset(seed=seed_r)
                self.smooth_left.reset(seed=q[self.left_indices])
                self.smooth_right.reset(seed=q[self.right_indices])
            out = np.empty(16, dtype=np.float32)
            out[:7] = q[self.left_indices]
            out[7] = l_grip
            out[8:15] = q[self.right_indices]
            out[15] = r_grip
            return self._guard_output(out)

        l_grip = self.l_grip
        r_grip = self.r_grip
        ema_l = self.ema_left.update(np.append(q[self.left_indices], l_grip))
        ema_r = self.ema_right.update(np.append(q[self.right_indices], r_grip))

        # Arm joints go through the trapezoidal filter; the gripper bypasses it
        # so it responds immediately (limited only by the EMA) rather than being
        # throttled by the rad/s velocity limit designed for arm joints.
        smoothed_l_arm = self.smooth_left.update(ema_l[:7])
        smoothed_r_arm = self.smooth_right.update(ema_r[:7])

        out = np.empty(16, dtype=np.float32)
        out[:7] = smoothed_l_arm
        out[7] = ema_l[7]
        out[8:15] = smoothed_r_arm
        out[15] = ema_r[7]
        out = self._guard_output(out)
        if self._rec is not None:
            self._rec.record(
                tgt=np.concatenate([q[self.left_indices], q[self.right_indices]]),
                ema=np.concatenate([ema_l[:7], ema_r[:7]]),
                out=np.concatenate([out[:7], out[8:15]]),
            )
        return out

    def _guard_output(self, out: np.ndarray) -> np.ndarray:
        """Enforce the per-tick command-step contract on the arm joints.

        Clamps each arm's step from the previous command to
        ``teleop_max_vel / frequency`` per joint (whole-vector scaling, so a
        clamped step keeps its direction and is simply spread over the next
        ticks). Grippers (normalized units, EMA-bounded) pass through. Warns
        (rate-limited) whenever it engages: the guard exists to make an
        upstream discontinuity harmless *and* loud, not to silently absorb it.
        """
        last = self._last_cmd
        if last is None:
            self._last_cmd = out
            return out
        bound = self.config.teleop_max_vel / self.config.frequency
        worst = 0.0
        for sl in (slice(0, 7), slice(8, 15)):
            delta = out[sl] - last[sl]
            m = float(np.max(np.abs(delta)))
            if m > bound:
                out[sl] = last[sl] + delta * (bound / m)
                worst = max(worst, m)
        self._last_cmd = out
        if worst > 0.0:
            now = time.perf_counter()
            if now - self._guard_warn_time >= 2.0:
                self._guard_warn_time = now
                self._logger.warning(
                    "Output guard clamped a %.1f mrad command step to the "
                    "%.1f mrad/tick contract — an upstream stage produced a "
                    "discontinuity (spread over the following ticks)",
                    1e3 * worst,
                    1e3 * bound,
                )
        return out

    # ------------------------------------------------------------------
    # Guarded return-to-rest (hardware flows)
    # ------------------------------------------------------------------

    async def guarded_return(
        self,
        *,
        send_step: Callable[[], Awaitable[object]],
        gravity_step: Callable[[], Awaitable[object]],
        torque_residuals: Callable[[], tuple[object, object]],
        reset_command_state: Callable[[], None],
        get_positions: Callable[[], tuple[np.ndarray, np.ndarray]],
        stopped: Callable[[], bool],
        announce: Callable[[str], None],
        on_contact: Callable[[], None] | None = None,
        hold_tick: Callable[[], None] | None = None,
        vr_alive: Callable[[], bool] | None = None,
        move_timeout_s: float = 30.0,
    ) -> None:
        """Play the pending return-to-rest guarded by the contact watchdog.

        The single engine behind every VR-flow rest move (native ``axol
        teleop`` and ``collect-data``), so the sequencing cannot drift
        between them. Await with a reset already pending or playing
        (``is_resetting``); returns once the arms are home or the flow
        stopped. The move plays at the normal session gains — accurate
        tracking of the collision-checked path — and a torque residual
        sustained above ``config.reset_torque_threshold`` (see
        :class:`~almond_axol.robot.control.ContactWatchdog`) cancels it
        where it is and drops the arms into a limp gravity-comp hold. The
        operator hand-guides them clear and presses reset, which replans
        from wherever they were left.

        Args:
            send_step: Advance the flow's control pipeline by one step —
                compute the smoothed output (which plays the trajectory) and
                command the robot. Must call :meth:`compute_output` exactly
                once.
            gravity_step: Apply one gravity-compensation cycle (the hold).
            torque_residuals: ``(left, right)`` per-joint measured-minus-
                gravity torques, ``None`` for an absent arm.
            reset_command_state: Clear the robot's command history after the
                arms moved out-of-band (max-step safety check).
            get_positions: Measured ``(left, right)`` arm positions used to
                re-sync the pipeline after hand-guiding.
            stopped: Flow shutdown flag; checked every cycle.
            announce: Operator-facing status line (e.g. ``log_say``).
            on_contact: Extra flow hook run once per trip, before the hold
                (e.g. unblock the headset's reset button).
            hold_tick: Flow hook run every hold cycle (e.g. consume teleop
                events so a stray record press is answered).
            vr_alive: Whether VR frames are still arriving. When given, a
                contact hold whose frame stream has been dead for a grace
                period is *orphaned* — the reset press that ends it can
                never come (a Y-exit return, a headset that died) — so the
                hold settles into a position hold where the arms are and
                the guarded return ends; reconnecting and pressing reset
                resumes the normal path. ``None`` waits indefinitely.
            move_timeout_s: Cap on each individual play attempt.
        """
        interval = 1.0 / self.config.frequency

        async def _play() -> str:
            """One guarded play attempt: ``done`` | ``contact`` | ``stopped``."""
            watchdog = ContactWatchdog(self.config.reset_torque_threshold)
            cap = time.perf_counter() + move_timeout_s
            deadline = time.perf_counter()
            while self.is_resetting and time.perf_counter() < cap:
                if stopped():
                    return "stopped"
                deadline += interval
                await send_step()
                tripped = watchdog.update(torque_residuals())
                if tripped is not None:
                    joint, residual = tripped
                    self._logger.warning(
                        "return-to-rest contact: %s torque residual %.1f "
                        "exceeds %.1f — going limp",
                        joint,
                        residual,
                        self.config.reset_torque_threshold,
                    )
                    return "contact"
                await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
            return "stopped" if stopped() else "done"

        while not stopped():
            outcome = await _play()
            if outcome != "contact":
                return
            hold = await self._contact_hold_until_reset(
                gravity_step=gravity_step,
                reset_command_state=reset_command_state,
                get_positions=get_positions,
                stopped=stopped,
                announce=announce,
                on_contact=on_contact,
                hold_tick=hold_tick,
                vr_alive=vr_alive,
            )
            if hold != "reset":
                return
            announce("Returning to rest pose.")

    async def contact_hold(
        self,
        *,
        gravity_step: Callable[[], Awaitable[object]],
        reset_command_state: Callable[[], None],
        get_positions: Callable[[], tuple[np.ndarray, np.ndarray]],
        stopped: Callable[[], bool],
        announce: Callable[[str], None],
        on_contact: Callable[[], None] | None = None,
        hold_tick: Callable[[], None] | None = None,
        vr_alive: Callable[[], bool] | None = None,
    ) -> None:
        """Limp gravity-comp hold after a *tracking-time* contact trip.

        The tracking-phase counterpart of :meth:`guarded_return`'s contact
        fallback: the flow's control loop tripped its
        ``config.teleop_torque_threshold`` watchdog while the operator was
        driving the arms. Disengages tracking (re-engaging afterwards needs a
        deliberate both-grips engage), drops the arms into the limp hold, and
        waits for a reset press. On reset, the pipeline is re-synced to the
        hand-guided positions and the latched reset is left pending, so the
        caller's normal reset path plans the return-to-rest from wherever the
        arms were left. Arguments as in :meth:`guarded_return`.
        """
        self.pause_ik()
        self.cancel_reset()
        self._disengage_all("Teleop disabled (contact)")
        hold = await self._contact_hold_until_reset(
            gravity_step=gravity_step,
            reset_command_state=reset_command_state,
            get_positions=get_positions,
            stopped=stopped,
            announce=announce,
            on_contact=on_contact,
            hold_tick=hold_tick,
            vr_alive=vr_alive,
        )
        if hold == "reset":
            # The reset press that ended the hold is still latched
            # (dispatch was paused): the caller's reset path picks it up.
            announce("Returning to rest pose.")

    async def _contact_hold_until_reset(
        self,
        *,
        gravity_step: Callable[[], Awaitable[object]],
        reset_command_state: Callable[[], None],
        get_positions: Callable[[], tuple[np.ndarray, np.ndarray]],
        stopped: Callable[[], bool],
        announce: Callable[[str], None],
        on_contact: Callable[[], None] | None,
        hold_tick: Callable[[], None] | None,
        vr_alive: Callable[[], bool] | None,
    ) -> str:
        """Shared limp-hold engine: ``reset`` | ``orphaned`` | ``stopped``.

        Freezes the IK pipeline, holds the arms limp until a reset press
        (or the flow stops / the hold is orphaned), then hands position
        control back re-synced to wherever the operator left the arms.
        """
        interval = 1.0 / self.config.frequency
        # Freeze the IK pipeline before the arms go limp — a reset
        # latched during the hold must not dispatch from a solution
        # the hand-guided arms no longer match — then drop the
        # interrupted plan.
        self.pause_ik()
        self.cancel_reset()
        if on_contact is not None:
            on_contact()
        announce("Contact. Arms are free — press reset to continue.")
        deadline = time.perf_counter()
        orphan_since: float | None = None
        orphaned = False
        while not stopped() and not self.reset_pending:
            deadline += interval
            await gravity_step()
            if hold_tick is not None:
                hold_tick()
            # Orphaned hold: the VR frame stream is dead (Y-exit
            # return, headset died), so the reset press that ends
            # this hold can never arrive. After the grace period,
            # stop waiting and settle where the arms are.
            if vr_alive is not None:
                now = time.perf_counter()
                if vr_alive():
                    orphan_since = None
                elif orphan_since is None:
                    orphan_since = now
                elif now - orphan_since >= _HOLD_ORPHAN_GRACE_S:
                    orphaned = True
                    break
            await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
        if stopped():
            return "stopped"
        # Hand position control back from wherever the operator left
        # the arms: clear the stale command history (max-step safety
        # check) and re-seed the pipeline from the measured
        # positions, so the first commands hold the true pose with
        # no transient — and, on a reset press, the replan starts
        # from it.
        reset_command_state()
        pos_l, pos_r = get_positions()
        self.resync_to_positions(pos_l, pos_r)
        self.resume_ik()
        if orphaned:
            announce(
                "No headset connected — holding position here. "
                "Reconnect and press reset to return to rest."
            )
            return "orphaned"
        return "reset"

    # ------------------------------------------------------------------
    # IK dispatch loop (IK thread)
    # ------------------------------------------------------------------

    def run_ik_loop(
        self,
        conn: multiprocessing.connection.Connection,
        get_frame: Callable[[], object],
        stop_event: threading.Event,
        process_alive: Callable[[], bool],
        on_ik_sample: Callable[[float], None],
    ) -> None:
        """Dispatch VR frames to the IK subprocess and publish raw targets.

        Runs in a dedicated daemon thread owned by the adapter: its blocking
        pipe reads release the GIL while the solver works, so the caller's CAN
        control loop isn't starved by cross-thread GIL contention. Only updates
        the raw target / engage / reset state — the smoothing runs in
        :meth:`compute_output` on the control loop.

        Args:
            conn: Pipe to the IK worker subprocess.
            get_frame: Returns the latest VR frame (or ``None``).
            stop_event: Set to stop the loop.
            process_alive: Returns ``False`` if the IK subprocess has died.
            on_ik_sample: Called with ``time.perf_counter()`` after each solve,
                for the adapter's IK-rate readout.
        """
        ik_interval = 1.0 / self.config.ik_frequency
        last_frame = None
        recv_timeout_count = 0
        last_dead_warn = 0.0

        while not stop_event.is_set():
            # Paused (arms moved out-of-band, e.g. gravity comp): idle without
            # dispatching frames or a latched reset — see :meth:`pause_ik`.
            if self._ik_paused:
                time.sleep(0.02)
                continue

            t0 = time.perf_counter()

            # Reset (return-to-rest) takes priority over normal tracking. Leave
            # the latch set if we can't dispatch yet (no solution, or a
            # trajectory already playing) so a press is never dropped.
            if (
                self._reset_latched
                and self.q is not None
                and not self.reset_interp.is_active()
            ):
                self._reset_latched = False
                self._reset_cancel = False
                # Keep is_resetting true across the (possibly seconds-long)
                # planning round trip so callers waiting on it don't observe a
                # false gap between the latch and the trajectory playback.
                self._reset_dispatching = True
                try:
                    # Plan from the last *commanded* joints, not the raw IK
                    # solution: the smoothed command lags raw q by the EMA +
                    # trapezoid lag (tens of mrad mid-motion), and a trajectory
                    # planned from raw q would join the command stream with a
                    # step the playback path never filters. Holding the target
                    # there also freezes the output for the (possibly
                    # seconds-long) planning window, so playback starts exactly
                    # where the arm stopped.
                    q_plan = self.q.copy()
                    pos_l = self.smooth_left.position
                    pos_r = self.smooth_right.position
                    if pos_l is not None and self.left_indices:
                        q_plan[self.left_indices] = pos_l
                    if pos_r is not None and self.right_indices:
                        q_plan[self.right_indices] = pos_r
                    self._hold_target(q_plan)
                    conn.send(("reset", q_plan.copy()))
                    result = conn.recv()
                    if self._reset_cancel:
                        # cancel_reset() fired mid-planning: drop the plan —
                        # the owner took the arms out-of-band and re-syncs q
                        # before any next dispatch.
                        pass
                    elif isinstance(result, tuple) and result[0] == "reset_traj":
                        _, q_default, trajectory = result
                        if trajectory:
                            self.reset_interp.set_trajectory(
                                trajectory, self.l_grip, self.r_grip
                            )
                            self._disengage_all()
                            self._at_rest = True
                        self._hold_target(np.asarray(q_default, dtype=np.float32))
                except Exception as e:  # noqa: BLE001 - keep the loop alive
                    self._logger.error("Reset error: %s", e)
                finally:
                    self._reset_dispatching = False
                    self._reset_cancel = False
                self._pace(t0, ik_interval)
                continue

            # A reset/startup trajectory is playing back; compute_output advances
            # it at the control rate, so just don't dispatch new IK meanwhile.
            if self.reset_interp.is_active():
                time.sleep(0.001)
                continue

            # Guard the sampling + engage steps: an exception here previously
            # killed this thread silently, freezing teleop at the last target
            # with nothing in the logs.
            try:
                frame = get_frame()
            except Exception:
                self._logger.exception("VR frame sampling failed; keeping last target")
                self._pace(t0, ik_interval)
                continue
            if frame is None or frame is last_frame:
                self._maybe_disengage_stale(conn, last_frame, process_alive)
                time.sleep(0.001)
                continue
            last_frame = frame

            try:
                self.update_engage(frame)
            except Exception:
                self._logger.exception("Engage update failed; keeping last target")
                self._pace(t0, ik_interval)
                continue

            if not process_alive():
                # Rate-limited: this used to log per iteration (~120 Hz).
                if t0 - last_dead_warn >= 5.0:
                    last_dead_warn = t0
                    self._logger.error(
                        "IK process is not alive — teleop is holding its last "
                        "target; restart teleop to recover"
                    )
                self._pace(t0, ik_interval)
                continue

            try:
                # Synthesize lock state so the IK worker tracks our per-arm
                # engage state rather than the raw button state.
                frame_to_send = frame.model_copy(
                    update={
                        "l_lock": self.left_enabled,
                        "r_lock": self.right_enabled,
                    }
                )
                conn.send(frame_to_send)
                result = recv_with_timeout(conn, _IK_RECV_TIMEOUT, stop_event)
                if result is not None:
                    self.set_target(result)
                    recv_timeout_count = 0
                    on_ik_sample(time.perf_counter())
                else:
                    recv_timeout_count += 1
                    if recv_timeout_count <= 3 or recv_timeout_count % 100 == 0:
                        self._logger.warning(
                            "IK recv timeout (no response in %.1fs)", _IK_RECV_TIMEOUT
                        )
            except Exception as e:  # noqa: BLE001 - keep the loop alive
                self._logger.error("IK dispatch error: %s", e)

            self._pace(t0, ik_interval)

    def _maybe_disengage_stale(
        self,
        conn: multiprocessing.connection.Connection,
        last_frame: object,
        process_alive: Callable[[], bool],
    ) -> None:
        """Force-disengage when the pose stream stops (``disengage_timeout``).

        The pose stream only flows while the headset is presenting, so a gap
        means the operator left VR (or the link died) — through *any* exit
        path, including ones that never deliver the Y-press reset frame
        (system menu, headset doffed, app crash, WiFi drop).

        Staying engaged across the gap is what makes the arms jerk on the
        next VR entry: the resumed frames are measured against the old engage
        snapshot, so the target jumps by however far the controllers moved in
        the meantime. Disengaging freezes the arms in place and makes
        re-entry inert until the operator deliberately re-engages (fresh
        engage snapshot — no jump).

        That freeze is deliberately where it ends: the arms *hold position*
        until an operator acts. A lost link, a quit app, or a crashed headset
        must never move the robot on its own — the operator reconnects and
        either re-engages (tracking resumes from where the arms are) or
        presses reset (X) to send them home.

        Runs on the IK thread. Frame arrivals are timestamped at ingest
        (:meth:`note_frame_reset`) rather than via ``get_frame`` identity,
        which cannot distinguish "stream stopped" from "operator perfectly
        still" (the interpolator's render frame is identity-stable while the
        pose holds).
        """
        last_seen = self._last_frame_time
        if last_seen is None:
            return
        gap = time.perf_counter() - last_seen

        timeout = self.config.disengage_timeout
        if self.teleop_enabled and 0 < timeout <= gap:
            self._logger.info(
                "Teleop disabled: no VR poses for %.2fs (operator left VR or "
                "the link dropped); holding position — re-engage with both "
                "grips, or press reset (X) to return to rest",
                gap,
            )
            self._disengage_all()
            # Also deactivate the IK worker so its next engaged frame performs
            # a fresh engage-snap. Without this, a client that resumes already
            # holding both grips (or a reconnecting SDK client streaming
            # l_lock/r_lock true) would have its first frame solved against the
            # stale snapshot — the exact jump this disengage exists to prevent.
            if last_frame is not None and process_alive():
                try:
                    conn.send(
                        last_frame.model_copy(  # type: ignore[attr-defined]
                            update={"l_lock": False, "r_lock": False}
                        )
                    )
                    result = recv_with_timeout(conn, _IK_RECV_TIMEOUT)
                    if result is not None:
                        self.set_target(result)
                    else:
                        self._logger.warning(
                            "IK recv timeout during stale-stream disengage"
                        )
                except Exception as e:  # noqa: BLE001 - keep the loop alive
                    self._logger.error("IK disengage dispatch error: %s", e)

    @staticmethod
    def _pace(t0: float, interval: float) -> None:
        rem = interval - (time.perf_counter() - t0)
        if rem > 0.0:
            time.sleep(rem)
