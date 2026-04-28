"""Viser-based robot simulation with the same interface as the Axol hardware."""

from __future__ import annotations

import logging
import threading

import numpy as np

from ..shared import ARM_JOINTS, URDF_PATH
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
        port: int = 8080,
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
        # Shape (8,): 7 arm joints then gripper, in Joint enum order
        self._last_left: np.ndarray = np.zeros(len(ARM_JOINTS) + 1, dtype=np.float32)
        self._last_right: np.ndarray = np.zeros(len(ARM_JOINTS) + 1, dtype=np.float32)

    async def enable(self) -> None:
        """Start the viser server thread. No-op after the first call."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        _logger.info("Simulation server started at http://localhost:%d", self._port)

    async def disable(self) -> None:
        """No-op — the daemon thread exits when the process ends."""
        pass

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
        """Build the arm joint angle array (radians), left then right, no gripper."""
        n_arm = len(ARM_JOINTS)  # 7, no gripper
        return np.concatenate(
            [
                self._last_left[:n_arm].astype(float),
                self._last_right[:n_arm].astype(float),
            ]
        )

    def _run(self) -> None:
        server = viser.ViserServer(port=self._port)

        urdf = yourdfpy.URDF.load(str(URDF_PATH), mesh_dir=str(URDF_PATH.parent))
        viser_urdf = ViserUrdf(
            server,
            urdf_or_path=urdf,
            root_node_name="/robot",
            load_meshes=True,
            load_collision_meshes=False,
        )

        # Build the robot-side joint ordering to match _build_q's output:
        # left arm joint1-N, then right arm joint1-N.
        left_names = [
            "left_s1_0",
            "left_s2_0",
            "left_s3_0",
            "left_e1_0",
            "left_e2_0",
            "left_w1_0",
            "left_w2_0",
        ]
        right_names = [
            "right_s1_0",
            "right_s2_0",
            "right_s3_0",
            "right_e1_0",
            "right_e2_0",
            "right_w1_0",
            "right_w2_0",
        ]
        robot_order = (self._joint_names or left_names) + right_names

        # Map each viser joint to its index in robot_order (-1 for joints not
        # in robot_order, e.g. finger joints, which stay at 0).
        viser_order = viser_urdf.get_actuated_joint_names()
        viser_to_robot: list[int] = []
        for name in viser_order:
            try:
                viser_to_robot.append(robot_order.index(name))
            except ValueError:
                viser_to_robot.append(-1)

        def _to_viser(q_robot: np.ndarray) -> np.ndarray:
            q_out = np.zeros(len(viser_order), dtype=float)
            for vi, ri in enumerate(viser_to_robot):
                if ri >= 0:
                    q_out[vi] = q_robot[ri]
            return q_out

        q0 = (
            np.asarray(self._default_q, dtype=float)
            if self._default_q is not None
            else np.zeros(len(robot_order))
        )
        viser_urdf.update_cfg(_to_viser(q0))

        server.scene.add_grid("/grid", width=2.0, height=2.0, position=(0.0, 0.0, 0.0))

        while True:
            with self._condition:
                self._condition.wait()
                q = self._latest_q
            if q is not None and q.size > 0:
                viser_urdf.update_cfg(_to_viser(np.asarray(q, dtype=float)))
