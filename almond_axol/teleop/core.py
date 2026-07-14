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
thread and read on the control loop; ``reset_interp`` touched by both) are
single-word and benign — this matches the original design of both adapters.
"""

from __future__ import annotations

import logging
import multiprocessing.connection
import threading
import time
from collections.abc import Callable

import numpy as np

from .config import VRTeleopConfig
from .filter import AlphaSmoothFilter, ResetInterpolator, TrapezoidalFilter

_IK_RECV_TIMEOUT = 5.0  # seconds; avoid blocking forever if IK process hangs


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
        self.ema_left = AlphaSmoothFilter(config.ik_alpha)
        self.ema_right = AlphaSmoothFilter(config.ik_alpha)
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

        # Engage toggle state (mutated on the IK thread).
        self.teleop_enabled: bool = False
        self._prev_both: bool = False
        self._prev_either: bool = False
        self._at_rest: bool = True
        self._engage_time: float | None = None

        # Reset latch (set from the VR frame callback / programmatically).
        self._prev_reset: bool = False
        self._reset_latched: bool = False
        # Target of the next latched reset — "rest" (the configured rest pose,
        # the default) or "zero" (all arm joints at 0, used by the
        # soft-shutdown park; see request_zero). Consumed and restored to
        # "rest" when the move is dispatched. _dispatching_reset holds
        # is_resetting True across the blocking plan round-trip so a poller
        # never sees a false "done" between the latch clearing and the
        # trajectory becoming active.
        self._reset_target: str = "rest"
        self._dispatching_reset: bool = False

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
        self.q = np.asarray(q_init, dtype=np.float32)
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

    def set_startup_trajectory(self, trajectory: object) -> None:
        """Queue the IK worker's startup trajectory for playback in compute_output."""
        if trajectory:
            self.reset_interp.set_trajectory(trajectory, self.l_grip, self.r_grip)

    # ------------------------------------------------------------------
    # Reset control
    # ------------------------------------------------------------------

    def note_frame_reset(self, reset: bool) -> None:
        """Latch a reset on the rising edge of the VR reset button.

        Called from the VR frame callback for every frame so a short press
        can't be missed while the IK loop blocks on ``conn.recv``.
        """
        if reset and not self._prev_reset:
            self._reset_latched = True
        self._prev_reset = reset

    def request_reset(self) -> None:
        """Programmatically trigger a return-to-rest move. Safe from any thread."""
        self._reset_target = "rest"  # see request_zero
        self._reset_latched = True

    def request_zero(self) -> None:
        """Programmatically trigger a return-to-zero move. Safe from any thread.

        Like :meth:`request_reset`, but the collision-aware trajectory targets
        the all-zeros arm pose instead of the configured rest pose — used to
        park the arms at the end of a session so they descend gradually rather
        than dropping when the motors release. The IK loop picks up the latch
        on its next iteration; poll :attr:`is_resetting` to know when the move
        completes. The grippers hold their current value for the whole move (a
        park must not drop a held object).
        """
        self._reset_target = "zero"
        self._reset_latched = True

    def clear_reset_request(self) -> None:
        """Consume a pending reset latch without acting on it.

        Used when a discard/rerecord already returns to rest, so the latched
        reset press shouldn't also trigger a rest-pose move.
        """
        self._reset_latched = False

    @property
    def is_resetting(self) -> bool:
        """True while a reset is pending, dispatching, or playing back."""
        return (
            self._reset_latched
            or self._dispatching_reset  # see __init__
            or self.reset_interp.is_active()
        )

    # ------------------------------------------------------------------
    # Engage toggle + IK target (IK thread)
    # ------------------------------------------------------------------

    def update_engage(self, frame: object) -> None:
        """Advance the engage toggle and grip tracking for one VR frame.

        Toggle logic (rising-edge): both grips together → enable; either grip
        alone → disable. On the first engage out of rest, ramp at the reduced
        ``engage_max_vel`` for ``engage_duration`` (restored in
        :meth:`compute_output`).
        """
        both = frame.l_lock and frame.r_lock
        either = frame.l_lock or frame.r_lock
        if not self.teleop_enabled:
            if both and not self._prev_both:
                self.teleop_enabled = True
                self._logger.info("Teleop enabled")
                self._broadcast(True)
                if self._at_rest:
                    self.smooth_left.max_vel = self.config.engage_max_vel
                    self.smooth_right.max_vel = self.config.engage_max_vel
                    self._engage_time = time.perf_counter()
                    self._at_rest = False
        else:
            if either and not self._prev_either:
                self.teleop_enabled = False
                self._logger.info("Teleop disabled")
                self._broadcast(False)
        self._prev_both = both
        self._prev_either = either

        # Only track gripper position while engaged so it can't be actuated
        # independently of the toggle.
        if self.teleop_enabled:
            self.l_grip = frame.l_grip
            self.r_grip = frame.r_grip

    def set_target(self, q_raw: object) -> None:
        """Publish a fresh raw IK solution (consumed by compute_output)."""
        self.q = np.asarray(q_raw, dtype=np.float32)

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
        # Restore full velocity once the post-engage ramp window expires.
        if (
            self._engage_time is not None
            and time.perf_counter() - self._engage_time >= self.config.engage_duration
        ):
            self.smooth_left.max_vel = self.config.teleop_max_vel
            self.smooth_right.max_vel = self.config.teleop_max_vel
            self._engage_time = None

        q = self.q
        if q is None:
            return None

        if self.reset_interp.is_active():
            new_q, l_grip, r_grip, done = self.reset_interp.step()
            if new_q is None:
                return None
            q = np.asarray(new_q, dtype=np.float32)
            if done:
                self.q = q.copy()
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
            return out

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
        return out

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
        ik_interval = 1.0 / self.config.frequency
        last_frame = None
        recv_timeout_count = 0

        while not stop_event.is_set():
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
                # Consume the reset target (default "rest"; "zero" parks the
                # arms at all-joints-0, computed here on the thread that owns
                # self.q — see request_zero). _dispatching_reset holds
                # is_resetting True across the blocking plan round-trip.
                target = self._reset_target
                self._reset_target = "rest"
                self._dispatching_reset = True
                try:
                    if target == "zero":
                        q_target = self.q.copy()
                        q_target[self.left_indices] = 0.0
                        q_target[self.right_indices] = 0.0
                    else:
                        q_target = None
                    conn.send(("reset", self.q.copy(), q_target))
                    result = conn.recv()
                    if isinstance(result, tuple) and result[0] == "reset_traj":
                        _, q_default, trajectory = result
                        if trajectory:
                            # A zero-target move is the soft-shutdown park:
                            # hold the grippers at their current value instead
                            # of ramping them open, so a held object isn't
                            # dropped as the arms ease down.
                            self.reset_interp.set_trajectory(
                                trajectory,
                                self.l_grip,
                                self.r_grip,
                                hold_grippers=(target == "zero"),
                            )
                            self.teleop_enabled = False
                            self._broadcast(False)
                            self._prev_both = False
                            self._prev_either = False
                            self._engage_time = None
                            self._at_rest = True
                        self.q = np.asarray(q_default, dtype=np.float32)
                except Exception as e:  # noqa: BLE001 - keep the loop alive
                    self._logger.error("Reset error: %s", e)
                finally:
                    self._dispatching_reset = False
                self._pace(t0, ik_interval)
                continue

            # A reset/startup trajectory is playing back; compute_output advances
            # it at the control rate, so just don't dispatch new IK meanwhile.
            if self.reset_interp.is_active():
                time.sleep(0.001)
                continue

            frame = get_frame()
            if frame is None or frame is last_frame:
                time.sleep(0.001)
                continue
            last_frame = frame

            self.update_engage(frame)

            if not process_alive():
                self._logger.warning("IK process is not alive")
                self._pace(t0, ik_interval)
                continue

            try:
                # Synthesize lock state so the IK worker tracks our toggle
                # rather than the raw button state.
                frame_to_send = frame.model_copy(
                    update={
                        "l_lock": self.teleop_enabled,
                        "r_lock": self.teleop_enabled,
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

    @staticmethod
    def _pace(t0: float, interval: float) -> None:
        rem = interval - (time.perf_counter() - t0)
        if rem > 0.0:
            time.sleep(rem)
