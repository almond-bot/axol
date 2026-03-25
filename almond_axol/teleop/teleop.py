"""
VR teleoperation for the Axol robot.

VRTeleop connects a VRServer (headset input), KinematicsSolver (IK), and a
RobotBase implementation (Axol hardware or Sim visualizer) into a single
runnable teleop session.

Typical usage::

    from almond_axol.robot import Sim
    from almond_axol.teleop import VRTeleop

    async def main():
        sim = Sim()
        await sim.enable()
        async with VRTeleop(sim) as teleop:
            await teleop.run()

Or with custom components::

    async with VRTeleop(
        Axol(),
        vr_server=VRServer(port=9000),
        solver=KinematicsSolver(config=KinematicsConfig(pos_weight=80.0)),
    ) as teleop:
        await teleop.run()
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np
import pyroki as pk

from ..kinematics.solver import KinematicsSolver, _clamp_reach
from ..motor import Joint, JointValues
from ..robot.base import RobotBase
from ..shared import ARM_JOINTS, rad_to_rev, rev_to_rad
from ..vr.models import VRFrame
from ..vr.server import VRServer
from .config import TeleopConfig
from .filter import AlphaSmoothFilter, ResetInterpolator

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def _quat_xyzw_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert an xyzw quaternion to a 3x3 rotation matrix (float32)."""
    x, y, z, w = float(qx), float(qy), float(qz), float(qw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _rub_to_flu(
    px: float,
    py: float,
    pz: float,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> jaxlie.SE3:
    """Convert a pose from VR RUB frame to robot FLU frame.

    Position mapping: ``(-z, -x, y)``.
    Rotation mapping: fast permute/negate equivalent to
    ``R_RUB2FLU @ M @ R_RUB2FLU.T``.
    """
    pos_flu = np.array((-pz, -px, py), dtype=np.float32)
    m = _quat_xyzw_to_matrix(qx, qy, qz, qw)
    rot_flu = np.empty((3, 3), dtype=np.float32)
    rot_flu[0, :] = (m[2, 2], m[2, 0], -m[2, 1])
    rot_flu[1, :] = (m[0, 2], m[0, 0], -m[0, 1])
    rot_flu[2, :] = (-m[1, 2], -m[1, 0], m[1, 1])
    return jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_matrix(rot_flu),
        jnp.asarray(pos_flu),
    )


def _relative_target(
    pose_curr: jaxlie.SE3,
    snap_ctrl: jaxlie.SE3,
    snap_fk: jaxlie.SE3,
) -> jaxlie.SE3:
    """Compute absolute EE target from controller delta relative to snap pose.

    The controller's displacement in its own frame is remapped to the EE frame
    using the ``(z, -y, x)`` axis permutation, then applied to the snapped FK
    EE pose.
    """
    pos_curr = np.asarray(pose_curr.translation(), dtype=np.float32)
    rot_curr = np.asarray(pose_curr.rotation().as_matrix(), dtype=np.float32)
    pos_snap_ctrl = np.asarray(snap_ctrl.translation(), dtype=np.float32)
    rot_snap_ctrl = np.asarray(snap_ctrl.rotation().as_matrix(), dtype=np.float32)
    pos_snap_fk = np.asarray(snap_fk.translation(), dtype=np.float32)
    rot_snap_fk = np.asarray(snap_fk.rotation().as_matrix(), dtype=np.float32)

    d = rot_snap_ctrl.T @ (pos_curr - pos_snap_ctrl)
    new_t = (
        pos_snap_fk
        + rot_snap_fk[:, 0] * d[2]
        - rot_snap_fk[:, 1] * d[1]
        + rot_snap_fk[:, 2] * d[0]
    )
    A = rot_snap_ctrl.T @ rot_curr
    R_delta = np.empty((3, 3), dtype=np.float32)
    R_delta[0, :] = (A[2, 2], -A[2, 1], A[2, 0])
    R_delta[1, :] = (-A[1, 2], A[1, 1], -A[1, 0])
    R_delta[2, :] = (A[0, 2], -A[0, 1], A[0, 0])
    new_R = (rot_snap_fk @ R_delta).astype(np.float32)
    return jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_matrix(new_R),
        jnp.asarray(new_t.astype(np.float32)),
    )


# ---------------------------------------------------------------------------
# JIT-compiled reset step
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=("max_iterations",))
def _solve_reset_step(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    q_interp: jax.Array,
    q_current: jax.Array,
    rest_weight: float,
    limit_weight: float,
    collision_margin: float,
    collision_weight: float,
    max_iterations: int,
) -> jax.Array:
    """One IK step toward ``q_interp`` with limit and self-collision costs only.

    No EE targets — used exclusively for computing smooth reset trajectories.
    """
    JointVar = robot.joint_var_cls
    costs = [
        pk.costs.rest_cost(JointVar(0), rest_pose=q_interp, weight=rest_weight),
        pk.costs.limit_cost(robot, JointVar(0), weight=limit_weight),
        pk.costs.self_collision_cost(
            robot,
            robot_coll,
            JointVar(0),
            margin=collision_margin,
            weight=collision_weight,
        ),
    ]
    var_joints = JointVar(jnp.array([0]))
    initial_vals = jaxls.VarValues.make(
        [var_joints.with_value(q_current[jnp.newaxis, :])]
    )
    problem = jaxls.LeastSquaresProblem(costs, [var_joints])
    analyzed = problem.analyze()
    solution_vals = analyzed.solve(
        initial_vals=initial_vals,
        verbose=False,
        linear_solver="dense_cholesky",
        trust_region=jaxls.TrustRegionConfig(),
        termination=jaxls.TerminationConfig(
            max_iterations=max_iterations,
            cost_tolerance=1e-2,
        ),
    )
    return solution_vals[var_joints][0]


# ---------------------------------------------------------------------------
# VRTeleop
# ---------------------------------------------------------------------------


class VRTeleop:
    """Connects a VR headset, IK solver, and robot into a teleoperation session.

    Uses relative tracking: controller displacement from the deadman snap point
    is mapped to an absolute EE target in the robot's world frame (FLU). Both
    ``l_lock`` and ``r_lock`` must be ``True`` simultaneously to activate tracking
    (deadman switch). A rising edge on ``frame.reset`` triggers a smooth return
    to the rest pose defined in ``config``.

    Args:
        robot: Hardware or simulation target implementing :class:`RobotBase`.
        vr_server: WebSocket server that receives :class:`VRFrame` data.
            Defaults to ``VRServer()`` with standard settings.
        solver: Bimanual IK solver. Defaults to ``KinematicsSolver()`` with
            default cost weights.
        config: Teleop session parameters (rest poses, loop frequency).
            Defaults to ``TeleopConfig()`` — all-zero rest poses at 100 Hz.

    Example::

        sim = Sim()
        await sim.enable()
        async with VRTeleop(sim) as teleop:
            await teleop.run()
    """

    def __init__(
        self,
        robot: RobotBase,
        *,
        vr_server: VRServer | None = None,
        solver: KinematicsSolver | None = None,
        config: TeleopConfig = TeleopConfig(),
    ) -> None:
        self._robot = robot
        self._vr_server = vr_server or VRServer()
        self._solver = solver or KinematicsSolver()
        self._config = config

        self._rest_pose_left = np.array(
            [rev_to_rad(config.rest_pose_left.get(j, 0.0)) for j in ARM_JOINTS],
            dtype=np.float32,
        )
        self._rest_pose_right = np.array(
            [rev_to_rad(config.rest_pose_right.get(j, 0.0)) for j in ARM_JOINTS],
            dtype=np.float32,
        )

        self._q_left = self._rest_pose_left.copy()
        self._q_right = self._rest_pose_right.copy()

        self._active: bool = False
        self._prev_reset: bool = False
        self._l_grip: float = 1.0
        self._r_grip: float = 1.0

        self._snap_ctrl: dict[str, jaxlie.SE3] = {}
        self._snap_fk: dict[str, jaxlie.SE3] = {}
        self._snap_elbow_ctrl: dict[str, np.ndarray] = {}
        self._snap_elbow_fk: dict[str, np.ndarray] = {}

        self._reset_interp = ResetInterpolator()
        self._smooth_left = AlphaSmoothFilter()
        self._smooth_right = AlphaSmoothFilter()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def enable(self) -> None:
        """Start the VR server and enable the robot."""
        await self._vr_server.enable()
        await self._robot.enable()
        self._warmup_reset_step()
        _logger.info("VRTeleop enabled")

    def _warmup_reset_step(self) -> None:
        """JIT-compile _solve_reset_step so the first reset is instant."""
        cfg = self._config
        q = jnp.zeros(self._solver.num_joints, dtype=jnp.float32)
        try:
            _solve_reset_step(
                self._solver.robot,
                self._solver.robot_coll,
                q,
                q,
                cfg.reset_rest_weight,
                cfg.reset_limit_weight,
                cfg.reset_collision_margin,
                cfg.reset_collision_weight,
                cfg.reset_max_iterations,
            )
        except Exception:
            pass

    async def disable(self) -> None:
        """Disable the robot and stop the VR server."""
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
        """Run the teleop control loop until cancelled.

        Calls :meth:`step` on each iteration and forwards the result to
        :meth:`robot.set_positions <RobotBase.set_positions>`. Loop rate
        is set by ``config.frequency``. Logs achieved rate every second.
        """
        interval = 1.0 / self._config.frequency
        loop_times: list[float] = []
        last_log = time.perf_counter()

        _logger.info("VRTeleop loop started at %.0f Hz", self._config.frequency)
        q_zeros = np.zeros(self._solver.num_joints, dtype=np.float32)
        self._reset_interp.set(q_zeros, self._build_rest_q_full())
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
                    _logger.debug("loop rate: %.1f Hz", rate)
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
        """Process the latest VRFrame and return joint positions for both arms.

        Returns ``(None, None)`` when there is nothing new to command (no VR
        frame received yet, or deadman switch not engaged).

        Returns:
            ``(left, right)`` as :class:`JointValues` dicts in revolutions
            (including ``Joint.GRIPPER``), or ``(None, None)`` when idle.
        """
        # 1. Reset interpolation in progress — advance one step
        if self._reset_interp.is_active():
            q_full, _ = self._reset_interp.step()
            if q_full is not None:
                q_full = np.asarray(q_full, dtype=np.float32)
                self._q_left = q_full[self._solver._left_indices]
                self._q_right = q_full[self._solver._right_indices]
            return self._q_to_joint_values(self._q_left, self._q_right)

        # 2. No frame yet
        frame = self._vr_server.get_frame()
        if frame is None:
            return None, None

        self._l_grip = frame.l_grip
        self._r_grip = frame.r_grip

        # 3. Reset rising edge → compute trajectory and start interpolation
        reset_rising = frame.reset and not self._prev_reset
        self._prev_reset = frame.reset
        if reset_rising:
            self._active = False
            self._snap_ctrl = {}
            self._snap_fk = {}
            self._smooth_left.reset()
            self._smooth_right.reset()
            q_full = self._build_q_full()
            q_target = self._build_rest_q_full()
            trajectory = self.compute_reset_trajectory(q_full, q_target)
            if trajectory:
                self._reset_interp.set_trajectory(trajectory)
            else:
                self._reset_interp.set(q_full, q_target)
            return None, None

        # 4. Deadman not engaged — hold current q
        deadman = frame.l_lock and frame.r_lock
        if not deadman:
            self._active = False
            return None, None

        # 5. Deadman just engaged — snap FK and controller poses
        if not self._active:
            self._active = True
            self._engage_snap(frame)
            return None, None

        # 6. Deadman active — solve relative IK
        left_pose = _rub_to_flu(
            frame.l_ee.position.x,
            frame.l_ee.position.y,
            frame.l_ee.position.z,
            frame.l_ee.quaternion.x,
            frame.l_ee.quaternion.y,
            frame.l_ee.quaternion.z,
            frame.l_ee.quaternion.w,
        )
        right_pose = _rub_to_flu(
            frame.r_ee.position.x,
            frame.r_ee.position.y,
            frame.r_ee.position.z,
            frame.r_ee.quaternion.x,
            frame.r_ee.quaternion.y,
            frame.r_ee.quaternion.z,
            frame.r_ee.quaternion.w,
        )
        left_e = np.array(
            (-frame.l_elbow.z, -frame.l_elbow.x, frame.l_elbow.y), dtype=np.float32
        )
        right_e = np.array(
            (-frame.r_elbow.z, -frame.r_elbow.x, frame.r_elbow.y), dtype=np.float32
        )

        tl = _relative_target(left_pose, self._snap_ctrl["left"], self._snap_fk["left"])
        tr = _relative_target(
            right_pose, self._snap_ctrl["right"], self._snap_fk["right"]
        )

        cfg = self._solver.config
        tl = jaxlie.SE3.from_rotation_and_translation(
            tl.rotation(),
            jnp.asarray(
                _clamp_reach(
                    np.asarray(tl.translation(), dtype=np.float32),
                    self._solver._L_shoulder_pos,
                    cfg.max_reach,
                )
            ),
        )
        tr = jaxlie.SE3.from_rotation_and_translation(
            tr.rotation(),
            jnp.asarray(
                _clamp_reach(
                    np.asarray(tr.translation(), dtype=np.float32),
                    self._solver._R_shoulder_pos,
                    cfg.max_reach,
                )
            ),
        )

        elbow_l = self._snap_elbow_fk["left"] + (left_e - self._snap_elbow_ctrl["left"])
        elbow_r = self._snap_elbow_fk["right"] + (
            right_e - self._snap_elbow_ctrl["right"]
        )

        left_jv, right_jv = self._solver.ik(
            left_pose=tl,
            right_pose=tr,
            q_current_left={
                j: rad_to_rev(float(self._q_left[i])) for i, j in enumerate(ARM_JOINTS)
            },
            q_current_right={
                j: rad_to_rev(float(self._q_right[i])) for i, j in enumerate(ARM_JOINTS)
            },
            left_elbow_pos=elbow_l,
            right_elbow_pos=elbow_r,
        )
        q_left_new = np.array(
            [rev_to_rad(left_jv[j]) for j in ARM_JOINTS], dtype=np.float32
        )
        q_right_new = np.array(
            [rev_to_rad(right_jv[j]) for j in ARM_JOINTS], dtype=np.float32
        )

        smoothed_l = self._smooth_left.update(q_left_new)
        smoothed_r = self._smooth_right.update(q_right_new)
        if smoothed_l is not None:
            q_left_new = smoothed_l
        if smoothed_r is not None:
            q_right_new = smoothed_r

        self._q_left = q_left_new
        self._q_right = q_right_new
        return self._q_to_joint_values(q_left_new, q_right_new)

    # ------------------------------------------------------------------
    # Reset trajectory
    # ------------------------------------------------------------------

    def compute_reset_trajectory(
        self, q_current: np.ndarray, q_target: np.ndarray
    ) -> list[np.ndarray]:
        """Compute a collision-aware smoothstep trajectory to ``q_target``.

        Uses ``_solve_reset_step`` (JIT-compiled) at each waypoint to keep
        the robot clear of self-collisions during the return to rest pose.

        Args:
            q_current: Current full joint configuration (radians, shape ``(14,)``).
            q_target: Target full joint configuration (radians, shape ``(14,)``).

        Returns:
            List of waypoint arrays (radians, shape ``(14,)``), empty if
            ``q_current`` and ``q_target`` are identical.
        """
        cfg = self._config
        max_dist_rev = rad_to_rev(float(np.max(np.abs(q_current - q_target))))
        duration = max_dist_rev / cfg.reset_speed
        n_steps = max(1, round(duration * cfg.frequency))
        trajectory: list[np.ndarray] = []
        q = np.array(q_current, dtype=np.float32)
        for i in range(n_steps):
            t = (i + 1) / n_steps
            alpha = t * t * (3.0 - 2.0 * t)  # smoothstep
            q_interp = (q_current * (1.0 - alpha) + q_target * alpha).astype(np.float32)
            result = _solve_reset_step(
                self._solver.robot,
                self._solver.robot_coll,
                jnp.asarray(q_interp),
                jnp.asarray(q),
                cfg.reset_rest_weight,
                cfg.reset_limit_weight,
                cfg.reset_collision_margin,
                cfg.reset_collision_weight,
                cfg.reset_max_iterations,
            )
            q = np.array(result, dtype=np.float32)
            trajectory.append(q)
        return trajectory

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _engage_snap(self, frame: VRFrame) -> None:
        """Save snap poses on first deadman activation frame."""
        left_pose = _rub_to_flu(
            frame.l_ee.position.x,
            frame.l_ee.position.y,
            frame.l_ee.position.z,
            frame.l_ee.quaternion.x,
            frame.l_ee.quaternion.y,
            frame.l_ee.quaternion.z,
            frame.l_ee.quaternion.w,
        )
        right_pose = _rub_to_flu(
            frame.r_ee.position.x,
            frame.r_ee.position.y,
            frame.r_ee.position.z,
            frame.r_ee.quaternion.x,
            frame.r_ee.quaternion.y,
            frame.r_ee.quaternion.z,
            frame.r_ee.quaternion.w,
        )
        left_e = np.array(
            (-frame.l_elbow.z, -frame.l_elbow.x, frame.l_elbow.y), dtype=np.float32
        )
        right_e = np.array(
            (-frame.r_elbow.z, -frame.r_elbow.x, frame.r_elbow.y), dtype=np.float32
        )

        q_full = self._build_q_full()
        fk = self._solver.robot.forward_kinematics(jnp.asarray(q_full))

        self._snap_ctrl = {
            "left": left_pose,
            "right": right_pose,
        }
        self._snap_fk = {
            "left": jaxlie.SE3(fk[self._solver._L_ee_idx]),
            "right": jaxlie.SE3(fk[self._solver._R_ee_idx]),
        }
        self._snap_elbow_ctrl = {
            "left": left_e,
            "right": right_e,
        }
        self._snap_elbow_fk = {
            "left": np.asarray(
                jaxlie.SE3(fk[self._solver._L_elbow_idx]).translation(),
                dtype=np.float32,
            ),
            "right": np.asarray(
                jaxlie.SE3(fk[self._solver._R_elbow_idx]).translation(),
                dtype=np.float32,
            ),
        }

    def _build_q_full(self) -> np.ndarray:
        """Assemble the full (14,) joint vector from current left/right q."""
        q_full = np.zeros(self._solver.num_joints, dtype=np.float32)
        for i, gi in enumerate(self._solver._left_indices):
            q_full[gi] = self._q_left[i]
        for i, gi in enumerate(self._solver._right_indices):
            q_full[gi] = self._q_right[i]
        return q_full

    def _build_rest_q_full(self) -> np.ndarray:
        """Assemble the full (14,) rest pose vector."""
        q_full = np.zeros(self._solver.num_joints, dtype=np.float32)
        for i, gi in enumerate(self._solver._left_indices):
            q_full[gi] = self._rest_pose_left[i]
        for i, gi in enumerate(self._solver._right_indices):
            q_full[gi] = self._rest_pose_right[i]
        return q_full

    def _q_to_joint_values(
        self, q_left: np.ndarray, q_right: np.ndarray
    ) -> tuple[JointValues, JointValues]:
        """Convert ``(7,)`` radian arrays to :class:`JointValues` dicts in revolutions.

        Appends :attr:`Joint.GRIPPER` from the latest frame gripper values.
        """
        left: JointValues = {
            joint: rad_to_rev(float(q_left[i])) for i, joint in enumerate(ARM_JOINTS)
        }
        left[Joint.GRIPPER] = self._l_grip
        right: JointValues = {
            joint: rad_to_rev(float(q_right[i])) for i, joint in enumerate(ARM_JOINTS)
        }
        right[Joint.GRIPPER] = self._r_grip
        return left, right
