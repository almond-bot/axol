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
        config=VRTeleopConfig(smooth_alpha=0.3),
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
import time

import numpy as np

from ..kinematics import KinematicsConfig
from ..robot.base import RobotBase
from ..vr.config import VRServerConfig
from ..vr.server import VRServer
from .config import VRTeleopConfig
from .filter import AlphaSmoothFilter, ResetInterpolator
from .worker import run_ik_worker

_logger = logging.getLogger(__name__)

_IK_RECV_TIMEOUT = 5.0  # seconds; avoid blocking forever if IK process hangs


def _recv_with_timeout(
    conn: multiprocessing.connection.Connection, timeout: float
) -> object | None:
    """Return ``conn.recv()`` if data arrives within ``timeout``, else ``None``."""
    if not conn.poll(timeout):
        return None
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
        self._smooth_left = AlphaSmoothFilter(alpha=config.smooth_alpha)
        self._smooth_right = AlphaSmoothFilter(alpha=config.smooth_alpha)

        self._parent_conn: multiprocessing.connection.Connection | None = None
        self._ik_process: multiprocessing.context.SpawnProcess | None = None
        self._ik_task: asyncio.Task | None = None

        self._ik_loop_times: list[float] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def enable(self) -> None:
        """Start the VR server, robot, and IK subprocess."""
        await self._vr_server.enable()
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
        if startup_traj:
            self._reset_interp.set_trajectory(startup_traj, self._l_grip, self._r_grip)

        self._ik_task = asyncio.create_task(self._ik_loop())
        _logger.info("VRTeleop enabled")

    async def disable(self) -> None:
        """Stop IK subprocess and VR server. Does not disable motors."""
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

        await self._vr_server.disable()
        _logger.info("VRTeleop disabled")

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
        try:
            while True:
                t0 = time.perf_counter()
                left, right = self.step()
                if left is not None or right is not None:
                    try:
                        await self._robot.motion_control(left=left, right=right)
                    except Exception as e:
                        _logger.error("Motion control error: %s", e)
                        pass

                now = time.perf_counter()
                loop_times.append(now)
                if now - last_log >= 1.0 and len(loop_times) > 1:
                    total = loop_times[-1] - loop_times[0]
                    rate = (len(loop_times) - 1) / total
                    if len(self._ik_loop_times) >= 2:
                        ik_total = self._ik_loop_times[-1] - self._ik_loop_times[0]
                        ik_hz = (
                            (len(self._ik_loop_times) - 1) / ik_total
                            if ik_total > 0
                            else 0.0
                        )
                        _logger.info("loop: %.1f Hz  ik: %.1f Hz", rate, ik_hz)
                    else:
                        _logger.info("loop: %.1f Hz", rate)
                    loop_times.clear()
                    last_log = now

                elapsed = time.perf_counter() - t0
                await asyncio.sleep(max(0.0, interval - elapsed))
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

        smoothed_l = self._smooth_left.update(np.append(q[self._left_indices], l_grip))
        smoothed_r = self._smooth_right.update(
            np.append(q[self._right_indices], r_grip)
        )

        left = np.empty(8, dtype=np.float32)
        left[:] = smoothed_l

        right = np.empty(8, dtype=np.float32)
        right[:] = smoothed_r

        return left, right

    # ------------------------------------------------------------------
    # VR frame callback (runs on every incoming frame)
    # ------------------------------------------------------------------

    def _on_vr_frame(self, frame) -> None:
        """Latch the reset rising edge as soon as the frame arrives.

        This runs on the event-loop thread for every WebSocket frame, so it
        captures reset=True even if the IK loop is blocked in run_in_executor
        and the button is released before the loop next checks get_frame().
        """
        if frame.reset and not self._prev_reset:
            self._reset_latched = True
        self._prev_reset = frame.reset

    # ------------------------------------------------------------------
    # IK loop (background task)
    # ------------------------------------------------------------------

    async def _ik_loop(self) -> None:
        """Dispatch VR frames to the IK subprocess and receive results."""
        loop = asyncio.get_running_loop()
        assert self._parent_conn is not None
        conn = self._parent_conn
        ik_interval = 1.0 / self._config.frequency
        last_frame = None
        ik_recv_timeout_count = 0

        while True:
            t0 = time.perf_counter()
            frame = self._vr_server.get_frame()

            if frame is None or frame is last_frame:
                await asyncio.sleep(0.001)
                continue

            last_frame = frame
            self._l_grip = frame.l_grip
            self._r_grip = frame.r_grip

            if self._reset_latched:
                if self._reset_interp.is_active() or self._q is None:
                    self._reset_latched = False
                else:
                    self._reset_latched = False
                    try:
                        conn.send(("reset", self._q.copy()))
                        result = await loop.run_in_executor(None, conn.recv)
                        if isinstance(result, tuple) and result[0] == "reset_traj":
                            _, q_default, trajectory = result
                            if trajectory:
                                self._reset_interp.set_trajectory(
                                    trajectory, self._l_grip, self._r_grip
                                )
                                self._smooth_left.reset()
                                self._smooth_right.reset()
                            self._q = np.asarray(q_default, dtype=np.float32)
                    except Exception as e:
                        _logger.error("Reset error: %s", e)
                    await asyncio.sleep(
                        max(0.0, ik_interval - (time.perf_counter() - t0))
                    )
                    continue

            if self._reset_interp.is_active():
                await asyncio.sleep(0.001)
                continue

            if self._ik_process is not None and not self._ik_process.is_alive():
                _logger.warning("IK process is not alive")
                await asyncio.sleep(max(0.0, ik_interval - (time.perf_counter() - t0)))
                continue

            try:
                conn.send(frame)
                result = await loop.run_in_executor(
                    None,
                    lambda: _recv_with_timeout(conn, _IK_RECV_TIMEOUT),
                )
                if result is not None:
                    self._q = np.asarray(result, dtype=np.float32)
                    ik_recv_timeout_count = 0
                    now = time.perf_counter()
                    self._ik_loop_times.append(now)
                    # Keep a 2-second rolling window
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

            await asyncio.sleep(max(0.0, ik_interval - (time.perf_counter() - t0)))
