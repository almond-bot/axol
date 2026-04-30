"""
VR teleoperator as a LeRobot Teleoperator.

AxolVRTeleop wraps the VRServer and IK subprocess behind LeRobot's synchronous
Teleoperator interface. A background thread runs a dedicated asyncio event loop
so the VR WebSocket server and IK dispatch loop keep running while get_action()
and send_feedback() block synchronously on the calling thread.

Episode control is exposed via get_teleop_events(), mapped from VRState
transitions:
  - DATA_COLLECTION → RECORDING:         start recording (no event)
  - RECORDING → DATA_COLLECTION + reset: RERECORD_EPISODE (discard)
  - RECORDING → DATA_COLLECTION:         TERMINATE_EPISODE + SUCCESS

After a TERMINATE_EPISODE the caller should push SAVING to the headset via
send_feedback_state(VRState.SAVING) to block controls while writing the
episode, then send_feedback_state(VRState.DATA_COLLECTION) when done.

Typical usage::

    from almond_axol.lerobot.robot import AxolRobot, AxolRobotConfig
    from almond_axol.lerobot.teleop import AxolVRTeleop, AxolVRTeleopConfig

    with AxolRobot(AxolRobotConfig()) as robot, AxolVRTeleop(AxolVRTeleopConfig()) as teleop:
        while True:
            obs = robot.get_observation()
            teleop.send_feedback(obs)
            action = teleop.get_action()
            robot.send_action(action)
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import multiprocessing.connection
import multiprocessing.context
import threading
import time
from typing import Any

import numpy as np
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ...shared import Joint
from ...teleop.filter import AlphaSmoothFilter, ResetInterpolator, TrapezoidalFilter
from ...teleop.worker import run_ik_worker
from ...vr.models import VRFrame, VRState
from ...vr.server import VRServer
from .config_vr import AxolVRTeleopConfig

_logger = logging.getLogger(__name__)

_IK_RECV_TIMEOUT = 5.0

_JOINTS = list(Joint)
_LEFT_POS_KEYS = [f"left_{j.value}.pos" for j in _JOINTS]
_RIGHT_POS_KEYS = [f"right_{j.value}.pos" for j in _JOINTS]


def _recv_with_timeout(
    conn: multiprocessing.connection.Connection, timeout: float
) -> object | None:
    if not conn.poll(timeout):
        return None
    return conn.recv()


class AxolVRTeleop(Teleoperator):
    """LeRobot Teleoperator wrapping the Axol VR teleoperation stack.

    Connects a VR headset (via VRServer) and an IK subprocess to produce
    joint position actions compatible with AxolRobot. Episode control signals
    are derived from VRState transitions and exposed via get_teleop_events().

    Args:
        config: Teleop session and IK solver parameters.
    """

    config_class = AxolVRTeleopConfig
    name = "axol_vr"

    def __init__(self, config: AxolVRTeleopConfig) -> None:
        super().__init__(config)
        self.config = config

        # Async bridge
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None

        # VR + IK
        self._vr_server: VRServer | None = None
        self._parent_conn: multiprocessing.connection.Connection | None = None
        self._ik_process: multiprocessing.context.SpawnProcess | None = None
        self._ik_task: asyncio.Task | None = None

        # Joint state — all access to _q_out protected by _q_lock
        self._q: np.ndarray | None = None  # full URDF vector from IK subprocess
        self._left_indices: list[int] = []
        self._right_indices: list[int] = []
        self._l_grip: float = 0.0
        self._r_grip: float = 0.0
        self._q_out = np.zeros(16, dtype=np.float32)
        self._q_lock = threading.Lock()

        # Signal processing (accessed only from IK loop thread)
        cfg = config.vr_teleop_config
        dt = 1.0 / cfg.frequency
        self._ema_left = AlphaSmoothFilter(cfg.ik_alpha)
        self._ema_right = AlphaSmoothFilter(cfg.ik_alpha)
        self._smooth_left = TrapezoidalFilter(
            cfg.teleop_max_vel, cfg.teleop_max_accel, dt
        )
        self._smooth_right = TrapezoidalFilter(
            cfg.teleop_max_vel, cfg.teleop_max_accel, dt
        )
        self._reset_interp = ResetInterpolator()

        # Reset latch
        self._prev_reset: bool = False
        self._reset_latched: bool = False

        # Deadman toggle (mirrors native VRTeleop behaviour):
        #   both grips rising edge → enable, either grip rising edge → disable
        self._teleop_enabled: bool = False
        self._prev_both: bool = False
        self._prev_either: bool = False
        # After a startup/reset trajectory completes, engage at reduced velocity
        # so the first move toward the IK target is gentle.
        self._at_rest: bool = True
        self._engage_time: float | None = None

        # Episode state
        self._prev_state: VRState = VRState.TELEOP
        self._rerecord_latch: bool = False
        self._terminate_latch: bool = False
        self._start_recording_latch: bool = False

        self._ik_loop_times: list[float] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._vr_server is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def action_features(self) -> dict:
        return {key: float for key in _LEFT_POS_KEYS + _RIGHT_POS_KEYS}

    @property
    def feedback_features(self) -> dict:
        return {key: float for key in _LEFT_POS_KEYS + _RIGHT_POS_KEYS}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @check_if_already_connected
    def connect(
        self,
        calibrate: bool = True,
        q_start_left: np.ndarray | None = None,
        q_start_right: np.ndarray | None = None,
    ) -> None:
        """Start the VR server and IK subprocess.

        Args:
            q_start_left:  Shape (7,) current left arm positions (rad) in ARM_JOINTS
                           order. Used as the startup trajectory start so the arm ramps
                           from its actual position rather than zeros. Optional.
            q_start_right: Same for the right arm.
        """
        loop = asyncio.new_event_loop()
        self._loop = loop
        self._loop_thread = threading.Thread(
            target=loop.run_forever, name="vr-axol-event-loop", daemon=True
        )
        self._loop_thread.start()
        asyncio.run_coroutine_threadsafe(
            self._connect_async(q_start_left, q_start_right), loop
        ).result(timeout=60)
        _logger.info("AxolVRTeleop connected.")

    async def _connect_async(
        self,
        q_start_left: np.ndarray | None = None,
        q_start_right: np.ndarray | None = None,
    ) -> None:
        self._vr_server = VRServer(self.config.vr_server_config)
        self._vr_server.set_on_frame(self._on_vr_frame)
        await self._vr_server.enable()

        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self._parent_conn = parent_conn

        process = ctx.Process(
            target=run_ik_worker,
            args=(
                child_conn,
                self.config.vr_teleop_config,
                self.config.kinematics_config,
                q_start_left,
                q_start_right,
            ),
            daemon=True,
        )
        process.start()
        child_conn.close()
        self._ik_process = process

        # Receive ready message: ("ready", q_init, left_indices, right_indices, startup_traj)
        loop = asyncio.get_running_loop()
        msg = await loop.run_in_executor(None, parent_conn.recv)
        assert isinstance(msg, tuple) and msg[0] == "ready"
        _, q_init, left_indices, right_indices, startup_traj = msg
        self._q = np.asarray(q_init, dtype=np.float32)
        self._left_indices = left_indices
        self._right_indices = right_indices
        if q_start_left is not None and len(q_start_left) > 7:
            self._l_grip = float(q_start_left[7])
        if q_start_right is not None and len(q_start_right) > 7:
            self._r_grip = float(q_start_right[7])

        # Seed signal-processing filters from current arm positions so there
        # is no transient on the first step (mirrors native VRTeleop.enable).
        if q_start_left is not None:
            seed_l = np.append(q_start_left[:7], self._l_grip)
            self._ema_left.reset(seed=seed_l)
            self._smooth_left.reset(seed=seed_l[:7])
        if q_start_right is not None:
            seed_r = np.append(q_start_right[:7], self._r_grip)
            self._ema_right.reset(seed=seed_r)
            self._smooth_right.reset(seed=seed_r[:7])

        # Prime _q_out from the seeded filters so get_action() returns correct
        # starting positions immediately rather than the all-zeros default.
        with self._q_lock:
            self._q_out = self._compute_output()

        if startup_traj:
            self._reset_interp.set_trajectory(startup_traj, self._l_grip, self._r_grip)

        self._ik_task = asyncio.create_task(self._ik_loop())

    def disconnect(self) -> None:
        """Stop the IK subprocess and VR server."""
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._disconnect_async(), self._loop).result(
            timeout=15
        )
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5)
        self._loop = None
        self._loop_thread = None
        self._vr_server = None
        _logger.info("AxolVRTeleop disconnected.")

    async def _disconnect_async(self) -> None:
        if self._ik_task is not None:
            self._ik_task.cancel()
            try:
                await self._ik_task
            except asyncio.CancelledError:
                pass
            self._ik_task = None

        if self._parent_conn is not None:
            try:
                self._parent_conn.send(None)
            except Exception:
                pass
            self._parent_conn.close()
            self._parent_conn = None

        if self._ik_process is not None:
            self._ik_process.join(timeout=3.0)
            if self._ik_process.is_alive():
                self._ik_process.terminate()
            self._ik_process = None

        if self._vr_server is not None:
            await self._vr_server.disable()

    # ------------------------------------------------------------------
    # Calibration / configuration (no-ops)
    # ------------------------------------------------------------------

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Reset control
    # ------------------------------------------------------------------

    def request_reset(self) -> None:
        """Programmatically trigger a rest-pose return move.

        Safe to call from any thread. The IK loop will pick up the latch on
        its next iteration and send a reset request to the IK subprocess,
        which plans a collision-aware trajectory back to the rest pose.
        Poll :attr:`is_resetting` to know when the move completes.
        """
        self._reset_latched = True

    @property
    def is_resetting(self) -> bool:
        """True while a reset is pending or a reset trajectory is playing back."""
        return self._reset_latched or self._reset_interp.is_active()

    # ------------------------------------------------------------------
    # Teleoperator interface
    # ------------------------------------------------------------------

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Accept robot observation. Currently a no-op — IK worker maintains its own state."""
        pass

    def send_feedback_state(self, state: VRState) -> None:
        """Broadcast a state override to all connected VR clients.

        Used to push server-driven states (e.g. ``VRState.SAVING``,
        ``VRState.DATA_COLLECTION``) back to the headset so the UI can block
        controls appropriately. Safe to call from any thread.
        """
        if self._vr_server is None or self._loop is None:
            return
        text = json.dumps({"type": "state", "value": state.value})
        asyncio.run_coroutine_threadsafe(
            self._vr_server.broadcast_text(text), self._loop
        )

    def send_feedback_error(self, timeout: float = 2.0) -> None:
        """Broadcast the error state to all connected VR clients.

        Blocks until the broadcast completes (or ``timeout`` seconds elapse) so
        the state update is delivered before ``disconnect()`` shuts down the
        event loop. Safe to call from any thread.

        Args:
            timeout: Maximum seconds to wait for delivery (default: 2.0).
        """
        if self._vr_server is None or self._loop is None:
            return
        text = json.dumps({"type": "state", "value": "error"})
        future = asyncio.run_coroutine_threadsafe(
            self._vr_server.broadcast_text(text), self._loop
        )
        try:
            future.result(timeout=timeout)
        except Exception:
            pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """Return the latest solved and smoothed joint positions."""
        with self._q_lock:
            q = self._q_out.copy()
        action: RobotAction = {}
        for i, key in enumerate(_LEFT_POS_KEYS):
            action[key] = float(q[i])
        for i, key in enumerate(_RIGHT_POS_KEYS):
            action[key] = float(q[8 + i])
        return action

    def get_teleop_events(self) -> dict[TeleopEvents | str, Any]:
        """Return episode control events derived from VRState transitions.

        Consumes and clears any latched events. Call once per control step.
        """
        rerecord = self._rerecord_latch
        terminate = self._terminate_latch
        start_recording = self._start_recording_latch
        self._rerecord_latch = False
        self._terminate_latch = False
        self._start_recording_latch = False
        return {
            TeleopEvents.IS_INTERVENTION: False,
            TeleopEvents.TERMINATE_EPISODE: terminate,
            TeleopEvents.SUCCESS: terminate,
            TeleopEvents.RERECORD_EPISODE: rerecord,
            "start_recording": start_recording,
        }

    # ------------------------------------------------------------------
    # VR frame callback (asyncio thread)
    # ------------------------------------------------------------------

    def _on_vr_frame(self, frame: VRFrame) -> None:
        """Latch reset rising edge and detect episode state transitions.

        Runs on the event-loop thread for every incoming WebSocket frame.
        """
        # Reset rising edge
        if frame.reset and not self._prev_reset:
            self._reset_latched = True
        self._prev_reset = frame.reset

        # Episode state transitions
        prev = self._prev_state
        curr = frame.state
        if prev == VRState.DATA_COLLECTION and curr == VRState.RECORDING:
            self._start_recording_latch = True

        if prev == VRState.RECORDING and curr == VRState.DATA_COLLECTION:
            if frame.reset:
                # Discard episode — consume the reset so it doesn't trigger rest-pose move
                self._rerecord_latch = True
                self._reset_latched = False
            else:
                self._terminate_latch = True
        self._prev_state = curr

    # ------------------------------------------------------------------
    # IK loop (background asyncio task)
    # ------------------------------------------------------------------

    def _compute_output(self) -> np.ndarray:
        """Compute 16-DOF output from current state. Call from IK loop thread only."""
        q = self._q
        if q is None:
            return self._q_out

        l_grip = self._l_grip
        r_grip = self._r_grip

        if self._reset_interp.is_active():
            new_q, l_grip, r_grip, done = self._reset_interp.step()
            if new_q is not None:
                q = np.asarray(new_q, dtype=np.float32)
                if done:
                    self._q = q.copy()
                    self._l_grip = l_grip
                    self._r_grip = r_grip
                    self._at_rest = True

        ema_l = self._ema_left.update(np.append(q[self._left_indices], l_grip))
        ema_r = self._ema_right.update(np.append(q[self._right_indices], r_grip))

        # Arm joints go through the trapezoidal filter; the gripper bypasses it
        # so it responds immediately (limited only by the EMA) rather than being
        # throttled by the rad/s velocity limit designed for arm joints.
        smoothed_l_arm = self._smooth_left.update(ema_l[:7])
        smoothed_r_arm = self._smooth_right.update(ema_r[:7])

        out = np.empty(16, dtype=np.float32)
        out[:7] = smoothed_l_arm
        out[7] = ema_l[7]
        out[8:15] = smoothed_r_arm
        out[15] = ema_r[7]
        return out

    async def _ik_loop(self) -> None:
        """Dispatch VR frames to the IK subprocess and update _q_out."""
        loop = asyncio.get_running_loop()
        assert self._parent_conn is not None
        conn = self._parent_conn
        ik_interval = 1.0 / self.config.vr_teleop_config.frequency
        last_frame = None
        ik_recv_timeout_count = 0

        while True:
            t0 = time.perf_counter()

            # Process a pending reset before checking for VR frames so it fires
            # even without a live headset.  _reset_latched is cleared only in the
            # finally block — after the IK subprocess responds and the trajectory is
            # armed — so is_resetting() stays True throughout the async IK call and
            # the collect_data reset loop cannot exit prematurely.
            if (
                self._reset_latched
                and not self._reset_interp.is_active()
                and self._q is not None
            ):
                try:
                    conn.send(("reset", self._q.copy()))
                    result = await loop.run_in_executor(None, conn.recv)
                    if isinstance(result, tuple) and result[0] == "reset_traj":
                        _, q_default, trajectory = result
                        if trajectory:
                            self._reset_interp.set_trajectory(
                                trajectory, self._l_grip, self._r_grip
                            )
                            self._ema_left.reset()
                            self._ema_right.reset()
                            self._smooth_left.reset()
                            self._smooth_right.reset()
                            self._teleop_enabled = False
                            self._prev_both = False
                            self._prev_either = False
                            self._at_rest = True
                        self._q = np.asarray(q_default, dtype=np.float32)
                except Exception as e:
                    _logger.error("Reset error: %s", e)
                finally:
                    self._reset_latched = False
                out = self._compute_output()
                with self._q_lock:
                    self._q_out = out
                await asyncio.sleep(max(0.0, ik_interval - (time.perf_counter() - t0)))
                continue

            # Service an active reset/startup trajectory (runs at IK frequency,
            # no VR frame needed).
            if self._reset_interp.is_active():
                out = self._compute_output()
                with self._q_lock:
                    self._q_out = out
                await asyncio.sleep(max(0.0, ik_interval - (time.perf_counter() - t0)))
                continue

            frame = self._vr_server.get_frame()  # type: ignore[union-attr]

            if frame is None or frame is last_frame:
                await asyncio.sleep(0.001)
                continue

            last_frame = frame
            self._l_grip = frame.l_grip
            self._r_grip = frame.r_grip

            # Deadman toggle — mirrors native VRTeleop._ik_loop logic:
            #   rising edge of BOTH grips pressed together → enable tracking
            #   rising edge of EITHER grip pressed alone   → disable tracking
            both = frame.l_lock and frame.r_lock
            either = frame.l_lock or frame.r_lock
            if not self._teleop_enabled:
                if both and not self._prev_both:
                    self._teleop_enabled = True
                    _logger.info("Teleop enabled")
                    if self._at_rest:
                        cfg = self.config.vr_teleop_config
                        self._smooth_left.max_vel = cfg.engage_max_vel
                        self._smooth_right.max_vel = cfg.engage_max_vel
                        self._engage_time = time.perf_counter()
                        self._at_rest = False
            else:
                if either and not self._prev_either:
                    self._teleop_enabled = False
                    _logger.info("Teleop disabled")
            self._prev_both = both
            self._prev_either = either

            # Restore full velocity after engage window expires
            if self._engage_time is not None:
                cfg = self.config.vr_teleop_config
                if time.perf_counter() - self._engage_time >= cfg.engage_duration:
                    self._smooth_left.max_vel = cfg.teleop_max_vel
                    self._smooth_right.max_vel = cfg.teleop_max_vel
                    self._engage_time = None

            if self._ik_process is not None and not self._ik_process.is_alive():
                _logger.warning("IK process is not alive")
                await asyncio.sleep(max(0.0, ik_interval - (time.perf_counter() - t0)))
                continue

            try:
                # Synthesize lock state so the IK worker tracks our toggle
                # rather than the raw button state (matches native VRTeleop).
                frame_to_send = frame.model_copy(
                    update={
                        "l_lock": self._teleop_enabled,
                        "r_lock": self._teleop_enabled,
                    }
                )
                conn.send(frame_to_send)
                result = await loop.run_in_executor(
                    None,
                    lambda: _recv_with_timeout(conn, _IK_RECV_TIMEOUT),
                )
                if result is not None:
                    self._q = np.asarray(result, dtype=np.float32)
                    ik_recv_timeout_count = 0
                    now = time.perf_counter()
                    self._ik_loop_times.append(now)
                    while (
                        len(self._ik_loop_times) > 1
                        and self._ik_loop_times[-1] - self._ik_loop_times[0] > 2.0
                    ):
                        self._ik_loop_times.pop(0)
                else:
                    ik_recv_timeout_count += 1
                    if ik_recv_timeout_count <= 3 or ik_recv_timeout_count % 100 == 0:
                        _logger.warning(
                            "IK recv timeout (no response in %.1fs)", _IK_RECV_TIMEOUT
                        )
            except Exception as e:
                _logger.error("IK process error: %s", e)
                ik_recv_timeout_count += 1

            out = self._compute_output()
            with self._q_lock:
                self._q_out = out

            await asyncio.sleep(max(0.0, ik_interval - (time.perf_counter() - t0)))
