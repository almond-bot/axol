"""
VR teleoperation for the Axol robot.

VRTeleop connects a VRServer (headset input) and a MotionControl implementation
(Axol hardware or Sim visualizer) into a single runnable teleop session. IK
runs in a separate subprocess so JAX/CUDA never blocks the asyncio event loop.

Typical usage::

    from almond_axol.robot import Sim
    from almond_axol.teleop import VRTeleop

    async def main():
        sim = Sim()
        async with VRTeleop(sim) as teleop:
            await teleop.run()

Or with custom components::

    async with VRTeleop(
        Axol(),
        config=VRTeleopConfig(teleop_max_vel=2.0),
        vr_server_config=VRServerConfig(port=9000),
    ) as teleop:
        await teleop.run()
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import multiprocessing.connection
import multiprocessing.context
import threading
import time

import numpy as np

from ..kinematics import KinematicsConfig
from ..robot.base import RobotBase
from ..vr.config import VRServerConfig
from ..vr.server import VRServer
from .config import VRTeleopConfig
from .filter import AlphaSmoothFilter, ResetInterpolator, TrapezoidalFilter
from .worker import run_ik_worker

_logger = logging.getLogger(__name__)

_IK_RECV_TIMEOUT = 5.0  # seconds; avoid blocking forever if IK process hangs


def _recv_with_timeout(
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


class VRTeleop:
    """Connects a VR headset and robot into a teleoperation session.

    IK runs in a dedicated subprocess; the main process handles frame
    dispatch, smoothing, reset trajectory playback, and robot I/O.

    Args:
        robot:             Hardware or simulation target implementing :class:`MotionControl`.
        config:            Teleop session parameters (rest poses, loop frequency).
        kinematics_config: IK solver parameters forwarded to the subprocess.
        vr_server_config:  VR WebSocket server parameters (port, TLS certs).
    """

    def __init__(
        self,
        robot: RobotBase,
        *,
        config: VRTeleopConfig = VRTeleopConfig(),
        kinematics_config: KinematicsConfig = KinematicsConfig(),
        vr_server_config: VRServerConfig = VRServerConfig(),
    ) -> None:
        self._robot = robot
        self._config = config
        self._kinematics_config = kinematics_config
        self._vr_server = VRServer(vr_server_config)
        self._vr_server.set_on_frame(self._on_vr_frame)

        # Full joint vector (radians), updated by _ik_loop
        self._q: np.ndarray | None = None
        self._left_indices: list[int] = []
        self._right_indices: list[int] = []

        self._l_grip: float = 0.0
        self._r_grip: float = 0.0
        self._prev_reset: bool = False
        # Latched by _on_vr_frame on every rising edge so the IK loop can't
        # miss a short reset press that arrives while blocked on conn.recv.
        self._reset_latched: bool = False

        self._reset_interp = ResetInterpolator()
        dt = 1.0 / config.frequency
        self._ema_left = AlphaSmoothFilter(config.ik_alpha)
        self._ema_right = AlphaSmoothFilter(config.ik_alpha)
        self._smooth_left = TrapezoidalFilter(
            config.teleop_max_vel, config.teleop_max_accel, dt
        )
        self._smooth_right = TrapezoidalFilter(
            config.teleop_max_vel, config.teleop_max_accel, dt
        )

        self._teleop_enabled: bool = False
        self._prev_both: bool = False
        self._prev_either: bool = False
        self._at_rest: bool = True
        self._engage_time: float | None = None
        self._startup_complete: bool = False

        self._parent_conn: multiprocessing.connection.Connection | None = None
        self._ik_process: multiprocessing.context.SpawnProcess | None = None
        self._ik_thread: threading.Thread | None = None
        self._ik_stop: threading.Event = threading.Event()

        self._ik_loop_times: list[float] = []
        self._ik_loop_times_lock: threading.Lock = threading.Lock()
        self._vr_frame_times: list[float] = []
        self._vr_frame_times_lock: threading.Lock = threading.Lock()

        self._vr_thread: threading.Thread | None = None
        self._vr_stop: threading.Event = threading.Event()
        self._vr_ready: threading.Event = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _run_vr_thread(self) -> None:
        """Run the VR WebSocket server in its own asyncio event loop.

        Keeping VR in a dedicated thread prevents burst WebSocket callbacks
        from contending with the IK thread for the GIL or CPU time.
        """

        async def _serve() -> None:
            await self._vr_server.enable()
            self._vr_ready.set()
            while not self._vr_stop.is_set():
                await asyncio.sleep(0.05)
            await self._vr_server.disable()

        asyncio.run(_serve())

    async def enable(self) -> None:
        """Start the VR server, robot, and IK subprocess."""
        self._vr_stop.clear()
        self._vr_ready.clear()
        self._vr_thread = threading.Thread(
            target=self._run_vr_thread, daemon=True, name="vr-server"
        )
        self._vr_thread.start()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._vr_ready.wait)

        await self._robot.enable()

        pos_l, pos_r = await self._robot.get_positions()
        if pos_l is not None:
            self._l_grip = float(pos_l[7])
        if pos_r is not None:
            self._r_grip = float(pos_r[7])

        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self._parent_conn = parent_conn

        process = ctx.Process(
            target=run_ik_worker,
            args=(
                child_conn,
                self._config,
                self._kinematics_config,
                pos_l[:7] if pos_l is not None else None,
                pos_r[:7] if pos_r is not None else None,
            ),
            daemon=True,
        )
        process.start()
        child_conn.close()
        self._ik_process = process

        # Receive ready message: (q_init, left_indices, right_indices, startup_traj)
        loop = asyncio.get_running_loop()
        msg = await loop.run_in_executor(None, parent_conn.recv)
        assert isinstance(msg, tuple) and msg[0] == "ready"
        _, q_init, left_indices, right_indices, startup_traj = msg
        self._q = np.asarray(q_init, dtype=np.float32)
        self._left_indices = left_indices
        self._right_indices = right_indices

        cur_l, cur_r = await self._robot.get_positions()
        if cur_l is not None:
            seed_l = np.append(cur_l[:7], self._l_grip)
            self._ema_left.reset(seed=seed_l)
            self._smooth_left.reset(seed=seed_l[:7])
        if cur_r is not None:
            seed_r = np.append(cur_r[:7], self._r_grip)
            self._ema_right.reset(seed=seed_r)
            self._smooth_right.reset(seed=seed_r[:7])

        if startup_traj:
            self._smooth_left.max_accel = self._config.startup_max_accel
            self._smooth_right.max_accel = self._config.startup_max_accel
            self._reset_interp.set_trajectory(startup_traj, self._l_grip, self._r_grip)

        self._ik_stop.clear()
        self._ik_thread = threading.Thread(
            target=self._ik_loop, daemon=True, name="ik-loop"
        )
        self._ik_thread.start()

    async def disable(self) -> None:
        """Stop IK subprocess and VR server. Does not disable motors."""
        if self._ik_thread is not None:
            self._ik_stop.set()
            self._ik_thread.join(timeout=3.0)
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

        if self._vr_thread is not None:
            self._vr_stop.set()
            self._vr_thread.join(timeout=5.0)
            self._vr_thread = None

    async def __aenter__(self) -> VRTeleop:
        await self.enable()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disable()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Run the teleop control loop until cancelled."""
        interval = 1.0 / self._config.frequency
        loop_times: list[float] = []
        last_log = time.perf_counter()

        _logger.info("VRTeleop loop started at %.0f Hz", self._config.frequency)
        # Track an absolute deadline so late wakeups are corrected in the next
        # cycle rather than accumulating as permanent drift.
        deadline = time.perf_counter()
        try:
            while True:
                deadline += interval
                if self._engage_time is not None:
                    if (
                        time.perf_counter() - self._engage_time
                        >= self._config.engage_duration
                    ):
                        self._smooth_left.max_vel = self._config.teleop_max_vel
                        self._smooth_right.max_vel = self._config.teleop_max_vel
                        self._engage_time = None

                left, right = self.step()
                if left is not None or right is not None:
                    try:
                        await self._robot.motion_control(left=left, right=right)
                    except Exception as e:
                        _logger.error("Motion control error: %s", e)

                now = time.perf_counter()
                loop_times.append(now)
                if now - last_log >= 1.0 and len(loop_times) > 1:
                    total = loop_times[-1] - loop_times[0]
                    rate = (len(loop_times) - 1) / total
                    with self._ik_loop_times_lock:
                        ik_times_snap = list(self._ik_loop_times)
                    if len(ik_times_snap) >= 2:
                        ik_total = ik_times_snap[-1] - ik_times_snap[0]
                        ik_hz = (
                            (len(ik_times_snap) - 1) / ik_total if ik_total > 0 else 0.0
                        )
                        with self._vr_frame_times_lock:
                            vr_times_snap = list(self._vr_frame_times)
                        if len(vr_times_snap) >= 2:
                            vr_total = vr_times_snap[-1] - vr_times_snap[0]
                            vr_hz = (
                                (len(vr_times_snap) - 1) / vr_total
                                if vr_total > 0
                                else 0.0
                            )
                            _logger.info(
                                "loop: %.1f Hz  vr: %.1f Hz  ik: %.1f Hz",
                                rate,
                                vr_hz,
                                ik_hz,
                            )
                        else:
                            _logger.info("loop: %.1f Hz  ik: %.1f Hz", rate, ik_hz)
                    else:
                        _logger.info("loop: %.1f Hz", rate)
                    loop_times.clear()
                    last_log = now

                await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
        except asyncio.CancelledError:
            await self._robot.disable()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return the latest smoothed joint positions.

        Returns ``(None, None)`` until the IK subprocess is ready.
        Once ready, always returns positions so the robot actively holds
        its commanded pose (matching the arm repo behaviour).

        Returns:
            Tuple ``(left, right)`` where each is a shape (8,) float32 array
            of joint positions in radians (Joint enum order), or ``None``
            if not yet ready.
        """
        if self._q is None:
            return None, None

        q = self._q

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
                    if not self._startup_complete:
                        self._startup_complete = True
                        self._smooth_left.max_accel = self._config.teleop_max_accel
                        self._smooth_right.max_accel = self._config.teleop_max_accel

        ema_l = self._ema_left.update(np.append(q[self._left_indices], l_grip))
        ema_r = self._ema_right.update(np.append(q[self._right_indices], r_grip))

        # Arm joints go through the trapezoidal filter; the gripper bypasses it
        # so it responds immediately (limited only by the EMA smoother) rather
        # than being throttled by the rad/s velocity limit designed for arm joints.
        smoothed_l_arm = self._smooth_left.update(ema_l[:7])
        smoothed_r_arm = self._smooth_right.update(ema_r[:7])

        left = np.empty(8, dtype=np.float32)
        left[:7] = smoothed_l_arm
        left[7] = ema_l[7]

        right = np.empty(8, dtype=np.float32)
        right[:7] = smoothed_r_arm
        right[7] = ema_r[7]

        return left, right

    # ------------------------------------------------------------------
    # VR frame callback (runs on every incoming frame)
    # ------------------------------------------------------------------

    def _on_vr_frame(self, frame) -> None:
        """Latch the reset rising edge as soon as the frame arrives.

        Called from the VR server thread; uses a lock for the frame-time list.
        """
        now = time.perf_counter()
        with self._vr_frame_times_lock:
            self._vr_frame_times.append(now)
            while (
                len(self._vr_frame_times) > 1
                and self._vr_frame_times[-1] - self._vr_frame_times[0] > 2.0
            ):
                self._vr_frame_times.pop(0)
        if frame.reset and not self._prev_reset:
            self._reset_latched = True
        self._prev_reset = frame.reset

    # ------------------------------------------------------------------
    # IK loop (daemon thread)
    # ------------------------------------------------------------------

    def _ik_loop(self) -> None:
        """Dispatch VR frames to the IK subprocess and receive results.

        Runs in a dedicated daemon thread so that asyncio event-loop activity
        (e.g. VR WebSocket burst callbacks) cannot delay IK scheduling.
        """
        assert self._parent_conn is not None
        conn = self._parent_conn
        ik_interval = 1.0 / self._config.frequency
        last_frame = None

        while not self._ik_stop.is_set():
            t0 = time.perf_counter()
            frame = self._vr_server.get_frame()

            if frame is None or frame is last_frame:
                time.sleep(0.001)
                continue
            last_frame = frame
            self._l_grip = frame.l_grip
            self._r_grip = frame.r_grip

            both = frame.l_lock and frame.r_lock
            either = frame.l_lock or frame.r_lock

            if not self._teleop_enabled:
                if both and not self._prev_both:
                    self._teleop_enabled = True
                    _logger.info("Teleop enabled")
                    if self._at_rest:
                        self._smooth_left.max_vel = self._config.engage_max_vel
                        self._smooth_right.max_vel = self._config.engage_max_vel
                        self._engage_time = time.perf_counter()
                        self._at_rest = False
            else:
                if either and not self._prev_either:
                    self._teleop_enabled = False
                    _logger.info("Teleop disabled")

            self._prev_both = both
            self._prev_either = either

            if self._reset_latched:
                if self._reset_interp.is_active() or self._q is None:
                    self._reset_latched = False
                else:
                    self._reset_latched = False
                    try:
                        conn.send(("reset", self._q.copy()))
                        result = conn.recv()
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
                            self._q = np.asarray(q_default, dtype=np.float32)
                    except Exception as e:
                        _logger.error("Reset error: %s", e)
                    _rem = ik_interval - (time.perf_counter() - t0)
                    if _rem > 0.0:
                        time.sleep(_rem)
                    continue

            if self._reset_interp.is_active():
                time.sleep(0.001)
                continue

            if self._ik_process is not None and not self._ik_process.is_alive():
                _logger.warning("IK process is not alive")
                _rem = ik_interval - (time.perf_counter() - t0)
                if _rem > 0.0:
                    time.sleep(_rem)
                continue

            try:
                # Synthesize lock state: worker sees both locks = enabled state,
                # so its _active flag tracks our toggle rather than the physical buttons.
                frame_to_send = frame.model_copy(
                    update={
                        "l_lock": self._teleop_enabled,
                        "r_lock": self._teleop_enabled,
                    }
                )
                conn.send(frame_to_send)
                result = _recv_with_timeout(conn, _IK_RECV_TIMEOUT, self._ik_stop)
                if result is not None:
                    self._q = np.asarray(result, dtype=np.float32)
                    now = time.perf_counter()
                    with self._ik_loop_times_lock:
                        self._ik_loop_times.append(now)
                        while (
                            len(self._ik_loop_times) > 1
                            and self._ik_loop_times[-1] - self._ik_loop_times[0] > 2.0
                        ):
                            self._ik_loop_times.pop(0)
            except Exception:
                pass

            _rem = ik_interval - (time.perf_counter() - t0)
            if _rem > 0.0:
                time.sleep(_rem)
