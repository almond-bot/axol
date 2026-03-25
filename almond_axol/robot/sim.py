"""Viser-based robot simulation with the same interface as the Axol hardware."""

from __future__ import annotations

import logging
import threading

import numpy as np

from ..motor import JointValues
from ..shared import ARM_JOINTS, URDF_PATH, rev_to_rad
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

    Implements the same :class:`RobotBase` interface as :class:`Axol` so it
    can be used as a drop-in replacement for visualising motion without hardware.

    Args:
        joint_names: Ordered list of actuated joint names matching the URDF.
            Defaults to the order reported by the loaded URDF.
        default_q: Initial joint configuration in radians. Defaults to zeros.
        port: Port for the viser web server.

    Example::

        async with Sim() as sim:
            left_q, right_q = await sim.get_positions()
            await sim.set_positions(left={Joint.ELBOW: 0.5})
    """

    def __init__(
        self,
        *,
        joint_names: list[str] | None = None,
        default_q: np.ndarray | None = None,
        port: int = 8080,
    ) -> None:
        self._joint_names = joint_names
        self._default_q = default_q
        self._port = port
        self._latest_q: np.ndarray | None = None
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._last_left: JointValues = {j: 0.0 for j in ARM_JOINTS}
        self._last_right: JointValues = {j: 0.0 for j in ARM_JOINTS}

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

    async def get_positions(self) -> tuple[JointValues, JointValues]:
        """Return the last commanded joint positions (rev) for both arms."""
        return dict(self._last_left), dict(self._last_right)

    async def set_positions(
        self,
        left: JointValues | None = None,
        right: JointValues | None = None,
    ) -> None:
        """Update the simulation to the given joint positions (rev).

        Args:
            left:  Target positions (rev) for the left arm.  ``None`` skips the arm.
            right: Target positions (rev) for the right arm. ``None`` skips the arm.
        """
        if left is not None:
            self._last_left.update(left)
        if right is not None:
            self._last_right.update(right)

        q = self._build_q()
        with self._condition:
            self._latest_q = q
            self._condition.notify()

    def _build_q(self) -> np.ndarray:
        """Build the arm joint angle array (radians), left then right, in ARM_JOINTS order."""
        q = np.zeros(len(ARM_JOINTS) * 2, dtype=float)
        for i, joint in enumerate(ARM_JOINTS):
            q[i] = rev_to_rad(self._last_left.get(joint, 0.0))
            q[i + len(ARM_JOINTS)] = rev_to_rad(self._last_right.get(joint, 0.0))
        return q

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
        n_arm = len(ARM_JOINTS)
        left_names = [f"openarm_left_joint{i + 1}" for i in range(n_arm)]
        right_names = [f"openarm_right_joint{i + 1}" for i in range(n_arm)]
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
