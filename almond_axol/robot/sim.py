"""Viser-based robot simulation with the same interface as the Axol hardware."""

from __future__ import annotations

import logging
import threading

import numpy as np

from ..constants import (
    ARM_JOINTS,
    GRIPPER_URDF_OPEN,
    URDF_PATH,
    urdf_arm_joint_names,
    urdf_gripper_joint_name,
)
from ..utils.ports import reclaim_port
from .base import RobotBase

_logger = logging.getLogger(__name__)


try:
    import viser
    import yourdfpy
    from viser.extras import ViserUrdf
except ImportError as e:
    raise ImportError(
        "viser is required for simulation. Install with: uv pip install almond-axol[sim]"
    ) from e


class Sim(RobotBase):
    """Viser-based robot simulation.

    Implements the same :class:`MotionControl` interface as :class:`Axol` so it
    can be used as a drop-in replacement for visualising motion without hardware.

    Args:
        joint_names: Ordered list of actuated joint names matching the URDF.
            Defaults to the order reported by the loaded URDF.
        default_q: Initial joint configuration in radians. Defaults to zeros.
        port: Port for the viser web server.

    Example::

        async with Sim() as sim:
            left_q, right_q = await sim.get_positions()
            await sim.motion_control(left=np.zeros(8, dtype=np.float32))
    """

    def __init__(
        self,
        *,
        joint_names: list[str] | None = None,
        default_q: np.ndarray | None = None,
        port: int = 8002,
    ) -> None:
        """Construct the simulation.

        The viser server is not started until :meth:`enable` is called.

        Args:
            joint_names: Ordered list of actuated joint names to look up in the URDF.
                Defaults to the hard-coded left-then-right arm order.
            default_q:   Initial joint configuration in radians; defaults to zeros.
            port:        Port for the viser web server.
        """
        self._joint_names = joint_names
        self._default_q = default_q
        self._port = port
        self._latest_q: np.ndarray | None = None
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._server: viser.ViserServer | None = None
        self._stop = threading.Event()
        # Shape (8,): 7 arm joints then gripper, in Joint enum order
        self._last_left: np.ndarray = np.zeros(len(ARM_JOINTS) + 1, dtype=np.float32)
        self._last_right: np.ndarray = np.zeros(len(ARM_JOINTS) + 1, dtype=np.float32)

    async def enable(self) -> None:
        """Start the viser server thread. No-op after the first call.

        Reclaims the viewer port first so a leftover viser server from a
        crashed/previous run (the URL is bookmarked, so the port is fixed)
        doesn't make the new one fail to bind.
        """
        if self._thread is not None:
            return
        import asyncio

        await asyncio.to_thread(reclaim_port, self._port)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        _logger.info("Simulation server started at http://localhost:%d", self._port)

    async def disable(self) -> None:
        """Stop the viser server and join its thread, freeing the viewer port.

        A daemon thread would free the port when the *process* exits, but the
        control panel runs sim teleop in-process and restarts it, so the server
        has to be torn down on stop too — otherwise the next run can't bind 8002.
        """
        if self._thread is None:
            return
        import asyncio

        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        # The worker stops its own server in _run's finally block, so we only
        # need to wait for it to exit. Don't clear _thread if the join times
        # out: the worker (and its in-process server on self._port) is still
        # alive, and reclaim_port can't kill the current PID, so letting enable
        # start a second server would collide on the port.
        await asyncio.to_thread(self._thread.join, 5.0)
        if self._thread.is_alive():
            _logger.warning(
                "viser worker did not exit within timeout; leaving it running "
                "so a restart doesn't collide on port %d",
                self._port,
            )
            return
        self._thread = None

    async def get_positions(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return the last commanded joint positions for both arms.

        Each array is shape (8,) in Joint enum order: 7 arm joints in radians,
        then gripper normalized to [0, 1] (0.0 = closed, 1.0 = fully open).
        """
        return self._last_left.copy(), self._last_right.copy()

    async def motion_control(
        self,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
    ) -> None:
        """Update the simulation to the given joint positions.

        Args:
            left:  Shape (8,) array — 7 arm joints in radians then gripper in [0, 1].
                   ``None`` skips the arm.
            right: Same for the right arm.
        """
        if left is not None:
            self._last_left = np.asarray(left, dtype=np.float32)
        if right is not None:
            self._last_right = np.asarray(right, dtype=np.float32)

        q = self._build_q()
        with self._condition:
            self._latest_q = q
            self._condition.notify()

    def _build_q(self) -> np.ndarray:
        """Build the command vector consumed by the viser worker.

        Layout (length 16): 7 left arm joints (rad), 7 right arm joints (rad),
        then the left and right gripper values normalized to ``[0, 1]`` (0 =
        closed, 1 = open). The arm joints map to the URDF revolute joints; the
        two gripper values are mapped to the actuated prismatic finger joints in
        :meth:`_run`.
        """
        n_arm = len(ARM_JOINTS)  # 7, no gripper
        return np.concatenate(
            [
                self._last_left[:n_arm].astype(float),
                self._last_right[:n_arm].astype(float),
                [float(self._last_left[n_arm]), float(self._last_right[n_arm])],
            ]
        )

    def _run(self) -> None:
        server = viser.ViserServer(port=self._port)
        self._server = server
        # Always tear the server down from the worker itself. disable() can't
        # reliably do it: a fast disable() may snapshot self._server before it's
        # assigned above, and a timed-out join would skip its stop() entirely.
        # Stopping here guarantees the in-process port is freed whenever the
        # loop exits, which is what makes the in-process restart reliable.
        try:
            urdf = yourdfpy.URDF.load(str(URDF_PATH), mesh_dir=str(URDF_PATH.parent))
            viser_urdf = ViserUrdf(
                server,
                urdf_or_path=urdf,
                root_node_name="/robot",
                load_meshes=True,
                load_collision_meshes=False,
            )

            # Build the robot-side joint ordering to match _build_q's output:
            # left arm joint1-N, then right arm joint1-N (the two trailing
            # gripper entries are handled separately below).
            robot_order = (
                self._joint_names or urdf_arm_joint_names(is_left=True)
            ) + urdf_arm_joint_names(is_left=False)
            n_arm_total = len(robot_order)

            # Map each viser joint to its index in robot_order (-1 for joints not
            # in robot_order). The gripper finger joints are not in robot_order;
            # they are driven from the two trailing gripper values instead.
            viser_order = viser_urdf.get_actuated_joint_names()
            viser_to_robot: list[int] = []
            for name in viser_order:
                try:
                    viser_to_robot.append(robot_order.index(name))
                except ValueError:
                    viser_to_robot.append(-1)

            # Viser indices of the actuated (non-mimic) prismatic gripper joints.
            # The opposing finger mimics these, so viser/yourdfpy moves it for us.
            def _grip_vi(is_left: bool) -> int:
                name = urdf_gripper_joint_name(is_left=is_left)
                return viser_order.index(name) if name in viser_order else -1

            left_grip_vi = _grip_vi(is_left=True)
            right_grip_vi = _grip_vi(is_left=False)

            def _to_viser(q_robot: np.ndarray) -> np.ndarray:
                q_out = np.zeros(len(viser_order), dtype=float)
                for vi, ri in enumerate(viser_to_robot):
                    if ri >= 0:
                        q_out[vi] = q_robot[ri]
                # Trailing entries n_arm_total, n_arm_total+1 are the left/right
                # gripper values in [0, 1]; scale to the prismatic travel limit
                # (0 = closed, GRIPPER_URDF_OPEN = open).
                if q_robot.size > n_arm_total + 1:
                    if left_grip_vi >= 0:
                        q_out[left_grip_vi] = (
                            float(q_robot[n_arm_total]) * GRIPPER_URDF_OPEN
                        )
                    if right_grip_vi >= 0:
                        q_out[right_grip_vi] = (
                            float(q_robot[n_arm_total + 1]) * GRIPPER_URDF_OPEN
                        )
                return q_out

            if self._default_q is not None:
                default_arm = np.asarray(self._default_q, dtype=float)
            else:
                default_arm = np.zeros(n_arm_total)
            # Start with grippers closed.
            q0 = np.concatenate([default_arm[:n_arm_total], [0.0, 0.0]])
            viser_urdf.update_cfg(_to_viser(q0))

            server.scene.add_grid(
                "/grid", width=2.0, height=2.0, position=(0.0, 0.0, 0.0)
            )

            while not self._stop.is_set():
                with self._condition:
                    # Time out so a stop set between notifications is still observed.
                    self._condition.wait(timeout=0.5)
                    q = self._latest_q
                if self._stop.is_set():
                    break
                if q is not None and q.size > 0:
                    viser_urdf.update_cfg(_to_viser(np.asarray(q, dtype=float)))
        finally:
            try:
                server.stop()
            except Exception as exc:  # noqa: BLE001 - teardown is best-effort
                _logger.debug("viser stop failed: %s", exc)
            self._server = None
