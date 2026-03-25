"""
VR teleoperation for the Axol robot.

VRTeleop connects a VRServer (headset input) and a RobotBase implementation
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
        vr_server=VRServer(port=9000),
        config=TeleopConfig(smooth_alpha=0.3),
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

from ..kinematics import KinematicsConfig, KinematicsSolver
from ..motor import Joint, JointValues
from ..robot.base import RobotBase
from ..shared import ARM_JOINTS, rad_to_rev
from ..vr.server import VRServer
from .config import TeleopConfig
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
        robot: Hardware or simulation target implementing :class:`RobotBase`.
        vr_server: WebSocket server that receives VR frame data.
        config: Teleop session parameters (rest poses, loop frequency).
        solver: Optional :class:`KinematicsSolver` whose config is forwarded to
            the IK subprocess. Pass a custom instance to tune IK weights.
    """

    def __init__(
        self,
        robot: RobotBase,
        *,
        vr_server: VRServer | None = None,
        config: TeleopConfig = TeleopConfig(),
        solver: KinematicsSolver | None = None,
    ) -> None:
        self._robot = robot
        self._vr_server = vr_server or VRServer()
        self._config = config
        self._kinematics_config = (
            solver.config if solver is not None else KinematicsConfig()
        )

        # Full joint vector (radians), updated by _ik_loop
        self._q: np.ndarray | None = None
        self._left_indices: list[int] = []
        self._right_indices: list[int] = []

        self._l_grip: float = 1.0
        self._r_grip: float = 1.0
        self._prev_reset: bool = False

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

        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self._parent_conn = parent_conn

        process = ctx.Process(
            target=run_ik_worker,
            args=(child_conn, self._config, self._kinematics_config),
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
            self._reset_interp.set_trajectory(startup_traj)

        self._ik_task = asyncio.create_task(self._ik_loop())
        _logger.info("VRTeleop enabled")

    async def disable(self) -> None:
        """Stop IK subprocess, robot, and VR server."""
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

        await self._robot.disable()
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
                    await self._robot.set_positions(left=left, right=right)

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
            pass

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self) -> tuple[JointValues | None, JointValues | None]:
        """Return the latest smoothed joint positions.

        Returns ``(None, None)`` until the IK subprocess is ready.
        Once ready, always returns positions so the robot actively holds
        its commanded pose (matching the arm repo behaviour).
        """
        if self._q is None:
            return None, None

        q = self._q

        if self._reset_interp.is_active():
            new_q, done = self._reset_interp.step()
            if new_q is not None:
                q = np.asarray(new_q, dtype=np.float32)
                if done:
                    self._q = q.copy()

        return self._q_to_joint_values(q)

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

            # Reset rising edge
            reset_rising = frame.reset and not self._prev_reset
            self._prev_reset = frame.reset

            if (
                reset_rising
                and self._q is not None
                and not self._reset_interp.is_active()
            ):
                try:
                    conn.send(("reset", self._q.copy()))
                    result = await loop.run_in_executor(None, conn.recv)
                    if isinstance(result, tuple) and result[0] == "reset_traj":
                        _, q_default, trajectory = result
                        if trajectory:
                            self._reset_interp.set_trajectory(trajectory)
                            self._smooth_left.reset()
                            self._smooth_right.reset()
                        self._q = np.asarray(q_default, dtype=np.float32)
                except Exception as e:
                    _logger.error("Reset error: %s", e)
                await asyncio.sleep(max(0.0, ik_interval - (time.perf_counter() - t0)))
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _q_to_joint_values(self, q: np.ndarray) -> tuple[JointValues, JointValues]:
        """Convert full (14,) radian array to smoothed JointValues dicts (revolutions)."""
        q_left = q[self._left_indices]
        q_right = q[self._right_indices]

        smoothed_l = self._smooth_left.update(q_left)
        smoothed_r = self._smooth_right.update(q_right)
        if smoothed_l is not None:
            q_left = smoothed_l
        if smoothed_r is not None:
            q_right = smoothed_r

        left: JointValues = {
            joint: rad_to_rev(float(q_left[i])) for i, joint in enumerate(ARM_JOINTS)
        }
        left[Joint.GRIPPER] = self._l_grip
        right: JointValues = {
            joint: rad_to_rev(float(q_right[i])) for i, joint in enumerate(ARM_JOINTS)
        }
        right[Joint.GRIPPER] = self._r_grip
        return left, right
