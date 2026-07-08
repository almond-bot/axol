"""
VR teleoperator as a LeRobot Teleoperator.

AxolVRTeleop wraps the VRServer and IK subprocess behind LeRobot's synchronous
Teleoperator interface. A background thread runs a dedicated asyncio event loop
for the VR WebSocket server, and a *separate* dedicated thread runs the IK
dispatch loop (blocking on the solver pipe). Keeping IK dispatch off the event
loop — and off the caller's control loop — means its waits release the GIL
instead of contending with the robot's CAN I/O, mirroring native VRTeleop. Both
keep running while get_action() / send_feedback() block on the calling thread.

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

    robot_config = AxolRobotConfig()
    with AxolRobot(robot_config) as robot, AxolVRTeleop(AxolVRTeleopConfig()) as teleop:
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

from ...constants import Joint
from ...teleop.core import VRTeleopCore
from ...teleop.worker import run_ik_worker
from ...vr.models import VRFrame, VRState
from ...vr.server import VRServer
from .config_vr import AxolVRTeleopConfig

_logger = logging.getLogger(__name__)

_JOINTS = list(Joint)
_LEFT_POS_KEYS = [f"left_{j.value}.pos" for j in _JOINTS]
_RIGHT_POS_KEYS = [f"right_{j.value}.pos" for j in _JOINTS]


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
        # IK dispatch runs in a dedicated daemon thread (not an asyncio task on
        # the VR event loop) so its blocking pipe reads release the GIL while the
        # solver works — keeping the caller's CAN control loop free of
        # cross-thread GIL contention, exactly like native VRTeleop.
        self._ik_thread: threading.Thread | None = None
        self._ik_stop = threading.Event()

        # Engage toggle, EMA/trapezoidal smoothing, and reset handling all live
        # in the shared core so this flow and native `axol teleop` (VRTeleop)
        # cannot drift apart.
        self._core = VRTeleopCore(
            config.vr_teleop_config,
            _logger,
            self._broadcast_tracking,
            self._broadcast_json,
        )

        # Last smoothed command; protected by _q_lock so concurrent get_action
        # calls serialize (only the control loop calls it, so uncontended).
        self._q_out = np.zeros(16, dtype=np.float32)
        self._q_lock = threading.Lock()

        # Episode state
        self._prev_state: VRState = VRState.TELEOP
        self._rerecord_latch: bool = False
        self._terminate_latch: bool = False
        self._start_recording_latch: bool = False

        # Rolling ~2s windows of IK-solve and VR-frame timestamps, for the
        # loop/vr/ik Hz readout (parity with the native VRTeleop). Both lists
        # are written on the event-loop thread and read from the caller's
        # control thread, so all access is guarded by ``_rate_lock``.
        self._ik_loop_times: list[float] = []
        self._vr_frame_times: list[float] = []
        self._rate_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @staticmethod
    def _window_hz(times: list[float]) -> float:
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        return (len(times) - 1) / span if span > 0 else 0.0

    def ik_hz(self) -> float:
        """Recent IK-solve rate (Hz) over a ~2s window, or 0.0 before warmup.

        Thread-safe; call from the control loop to monitor solver throughput.
        """
        with self._rate_lock:
            return self._window_hz(self._ik_loop_times)

    def vr_hz(self) -> float:
        """Recent VR-frame arrival rate (Hz) over a ~2s window, or 0.0.

        Thread-safe; reflects how fast the headset is streaming poses.
        """
        with self._rate_lock:
            return self._window_hz(self._vr_frame_times)

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
        ).result(timeout=120)
        _logger.info("AxolVRTeleop connected.")

    async def _connect_async(
        self,
        q_start_left: np.ndarray | None = None,
        q_start_right: np.ndarray | None = None,
    ) -> None:
        self._vr_server = VRServer(self.config.vr_server_config)
        self._vr_server.set_on_frame(self._on_vr_frame)
        # Lock the headset HUD to data collection: the operator can record
        # episodes but can't switch back to plain teleop.
        self._vr_server.set_mode("data_collection")
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
        self._core.set_solution(q_init, left_indices, right_indices)
        self._core.set_initial_grips(
            q_start_left[7]
            if q_start_left is not None and len(q_start_left) > 7
            else None,
            q_start_right[7]
            if q_start_right is not None and len(q_start_right) > 7
            else None,
        )
        # Seed filters from current arm positions so the first step has no
        # transient (the core also seeds gripper from the grips set above).
        self._core.seed_filters(q_start_left, q_start_right)

        # Prime _q_out from the seeded filters so get_action() returns correct
        # starting positions immediately rather than the all-zeros default.
        with self._q_lock:
            out = self._core.compute_output()
            if out is not None:
                self._q_out = out

        self._core.set_startup_trajectory(startup_traj)

        self._ik_stop.clear()
        self._ik_thread = threading.Thread(
            target=self._ik_loop, daemon=True, name="vr-axol-ik-loop"
        )
        self._ik_thread.start()

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
        if self._ik_thread is not None:
            self._ik_stop.set()
            await asyncio.get_running_loop().run_in_executor(
                None, self._ik_thread.join, 3.0
            )
            self._ik_thread = None

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
        self._core.request_reset()

    @property
    def is_resetting(self) -> bool:
        """True while a reset is pending or a reset trajectory is playing back."""
        return self._core.is_resetting

    # ------------------------------------------------------------------
    # Teleoperator interface
    # ------------------------------------------------------------------

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Accept robot observation. Currently a no-op — IK worker maintains its own state."""
        pass

    def set_video_sources(self, sources: dict[str, Any] | None) -> None:
        """Stream wrist-camera frames to the headset via WebRTC.

        Each value is a connected ``ZedCamera`` / stereo eye (registered
        directly) or any raw-frame source exposing ``width`` / ``height`` /
        ``fps`` + ``wait_next``; see
        :meth:`almond_axol.vr.server.VRServer.set_video_sources`. Must be
        called after :meth:`connect` (so the VR server exists). Safe to call
        from any thread. Requires the GStreamer NVENC stack (``axol
        gst.install``); without it video is silently disabled.
        """
        if self._vr_server is not None:
            self._vr_server.set_video_sources(sources)

    def set_video_manager(self, manager: Any | None) -> None:
        """Stream to the headset via a pre-built WebRTC manager (e.g. a relay).

        ``manager`` implements the ``WebRTCManager`` signaling interface and is
        normally an out-of-process :class:`~almond_axol.video.video_proc.VideoRelayProcess`,
        so all camera grab/encode and RTP traffic stays off the control process
        — see :meth:`almond_axol.vr.server.VRServer.set_video_manager`. Must be
        called after :meth:`connect`. Pass ``None`` to disable video.
        """
        if self._vr_server is not None:
            self._vr_server.set_video_manager(manager)

    def _broadcast_tracking(self, enabled: bool) -> None:
        """Push the engage-toggle state to the headset (fire-and-forget).

        The VR app uses it to allow screen repositioning (trigger grabs) only
        while the robot isn't being controlled. Safe to call from any thread.
        """
        if self._vr_server is None or self._loop is None:
            return
        text = json.dumps({"type": "tracking", "value": enabled})
        try:
            asyncio.run_coroutine_threadsafe(
                self._vr_server.broadcast_text(text), self._loop
            )
        except RuntimeError:
            pass  # event loop already shut down

    def _broadcast_json(self, obj: dict[str, Any]) -> None:
        """Push an arbitrary JSON message to the headset (fire-and-forget).

        Used by the shared core for the URDF overlay state in absolute (UMI)
        mode. Safe to call from any thread.
        """
        if self._vr_server is None or self._loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._vr_server.broadcast_text(json.dumps(obj)), self._loop
            )
        except RuntimeError:
            pass  # event loop already shut down

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
        """Return the latest IK target, smoothed at the control-loop rate.

        Mirrors native ``VRTeleop.step()``: the EMA + trapezoidal velocity
        smoothing (and any active reset/startup trajectory) is advanced *here*,
        on the caller's control loop, not in the IK loop. Running the smoothing
        at the fast, regular control rate instead of the slower, jittery
        IK-solve rate is what keeps motion smooth — applied at the IK rate the
        setpoint stair-cases, and the robot's velocity feedforward (which
        differentiates commanded positions) turns each step's jump into a torque
        spike (jerk). Call once per control step at a steady rate.
        """
        with self._q_lock:
            out = self._core.compute_output()
            if out is not None:
                self._q_out = out
            q = self._q_out
        action: RobotAction = {}
        for i, key in enumerate(_LEFT_POS_KEYS):
            action[key] = float(q[i])
        for i, key in enumerate(_RIGHT_POS_KEYS):
            action[key] = float(q[8 + i])
        return action

    def pose_capture_ts(self) -> float | None:
        """Host-clock capture time of the pose behind the latest action.

        ``time.perf_counter`` seconds, estimated by the VR server's pose
        interpolator (``VRFrame.t_host``). UMI data collection stamps dataset
        rows with this so image exposure and pose share one capture timeline.
        ``None`` before the first solve or when the client doesn't stamp
        capture times.
        """
        return self._core.last_pose_host_ts

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
        # Track the VR send rate (one frame per WebSocket message).
        now = time.perf_counter()
        with self._rate_lock:
            self._vr_frame_times.append(now)
            while self._vr_frame_times and now - self._vr_frame_times[0] > 2.0:
                self._vr_frame_times.pop(0)

        # Reset rising edge
        self._core.note_frame_reset(frame.reset)

        # Episode state transitions
        prev = self._prev_state
        curr = frame.state
        if prev == VRState.DATA_COLLECTION and curr == VRState.RECORDING:
            self._start_recording_latch = True

        if prev == VRState.RECORDING and curr == VRState.DATA_COLLECTION:
            if frame.reset:
                # Discard episode — consume the reset so it doesn't trigger
                # rest-pose move.
                self._rerecord_latch = True
                self._core.clear_reset_request()
            else:
                self._terminate_latch = True
        self._prev_state = curr

    # ------------------------------------------------------------------
    # IK loop (daemon thread)
    # ------------------------------------------------------------------

    def _note_ik_sample(self, now: float) -> None:
        with self._rate_lock:
            self._ik_loop_times.append(now)
            while (
                len(self._ik_loop_times) > 1
                and self._ik_loop_times[-1] - self._ik_loop_times[0] > 2.0
            ):
                self._ik_loop_times.pop(0)

    def _ik_loop(self) -> None:
        """Dispatch VR frames to the IK subprocess via the shared core.

        Runs in a dedicated daemon thread (not an asyncio task on the VR event
        loop): its blocking pipe reads release the GIL while the solver works, so
        the caller's CAN control loop isn't starved by cross-thread GIL
        contention — this is what keeps ``send`` flat under load. All the engage
        / reset / dispatch logic lives in :meth:`VRTeleopCore.run_ik_loop`, so it
        stays identical to native ``axol teleop``. The EMA/trapezoidal smoothing
        is *not* applied here — it runs in ``get_action`` at the control-loop
        rate so the commanded setpoint stays smooth even when this loop solves
        slower or with jitter (see ``get_action``).
        """
        assert self._parent_conn is not None
        self._core.run_ik_loop(
            self._parent_conn,
            self._vr_server.get_render_frame,  # type: ignore[union-attr]
            self._ik_stop,
            lambda: self._ik_process is None or self._ik_process.is_alive(),
            self._note_ik_sample,
        )
