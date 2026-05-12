"""
IK subprocess worker for VR teleoperation.

Runs in a separate process to keep JAX/CUDA off the main asyncio event loop.
All intermediate computations stay in NumPy; the single JAX boundary is the
``solver.ik`` call itself (matching the arm-repo pattern).
"""

from __future__ import annotations

import functools
import multiprocessing
import multiprocessing.connection
import os

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np
import pyroki as pk

from ..kinematics.config import KinematicsConfig
from ..kinematics.solver import KinematicsSolver
from ..vr.models import VRFrame
from .config import VRTeleopConfig
from .filter import OneEuroFilter

# ---------------------------------------------------------------------------
# NumPy-only helpers (no JAX dispatch overhead)
# ---------------------------------------------------------------------------


def _quat_xyzw_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
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


def _vr_to_flu_np(
    px: float,
    py: float,
    pz: float,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert VR pose (X=Down, Y=Left, Z=Forward) → robot FLU. Returns (pos_3, rot_3x3), float32."""
    pos = np.array((pz, py, -px), dtype=np.float32)
    m = _quat_xyzw_to_matrix(qx, qy, qz, qw)
    rot = np.empty((3, 3), dtype=np.float32)
    rot[0, :] = (m[2, 2], m[2, 1], -m[2, 0])
    rot[1, :] = (m[1, 2], m[1, 1], -m[1, 0])
    rot[2, :] = (-m[0, 2], -m[0, 1], m[0, 0])
    return pos, rot


def _relative_target_np(
    pos_curr: np.ndarray,
    rot_curr: np.ndarray,
    pos_snap_ctrl: np.ndarray,
    rot_snap_ctrl: np.ndarray,
    pos_snap_fk: np.ndarray,
    rot_snap_fk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute absolute EE target from controller delta. Returns (pos_3, rot_3x3)."""
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
    return new_t.astype(np.float32), (rot_snap_fk @ R_delta).astype(np.float32)


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
    """One IK step toward ``q_interp`` with limit and self-collision costs only."""
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
# IKWorker
# ---------------------------------------------------------------------------


class IKWorker:
    """Self-contained IK controller for the subprocess.

    Snap state is numpy-only. The single JAX boundary is the ``solver.ik``
    call inside :meth:`step`.
    """

    def __init__(
        self, config: VRTeleopConfig, kinematics_config: KinematicsConfig
    ) -> None:
        """Construct the IK worker.

        Instantiates the :class:`KinematicsSolver` (which triggers JAX JIT
        compilation) and initialises One Euro Filters for all VR pose streams.

        Args:
            config:            Teleop session parameters (rest poses, frequency, filter settings).
            kinematics_config: IK solver cost weights forwarded to :class:`KinematicsSolver`.
        """
        self._config = config
        self._solver = KinematicsSolver(kinematics_config)

        self._rest_pose_left = np.asarray(config.rest_pose_left, dtype=np.float32)
        self._rest_pose_right = np.asarray(config.rest_pose_right, dtype=np.float32)

        self._solver.set_posture_pose(self.get_rest_q())

        self._active: bool = False
        # Snap poses as (pos_3, rot_3x3) numpy tuples — no jaxlie overhead
        self._snap_ctrl: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._snap_fk: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._snap_elbow_ctrl: dict[str, np.ndarray] = {}
        self._snap_elbow_fk: dict[str, np.ndarray] = {}

        freq = config.frequency
        mc = config.pose_min_cutoff
        beta = config.pose_beta
        self._f_l_pos = OneEuroFilter(freq, mc, beta)
        self._f_l_quat = OneEuroFilter(freq, mc, beta)
        self._f_r_pos = OneEuroFilter(freq, mc, beta)
        self._f_r_quat = OneEuroFilter(freq, mc, beta)
        self._f_l_elbow = OneEuroFilter(freq, mc, beta)
        self._f_r_elbow = OneEuroFilter(freq, mc, beta)

        # Pre-settle the configured rest pose to the manipulability-balanced
        # IK fixed point. The configured pose has a non-zero manipulability
        # gradient, so a first engage there walks q in the EE null space
        # toward higher manipulability over the next ~10-30 frames. Baking the
        # settling in at startup means the trajectory ends at the fixed point
        # and the first engage produces no motion.
        q_settled = self._settle_rest_pose()
        self._rest_pose_left = q_settled[self._solver.left_indices].astype(np.float32)
        self._rest_pose_right = q_settled[self._solver.right_indices].astype(np.float32)
        self._solver.set_posture_pose(self.get_rest_q())

    # -- Properties the main process needs ----------------------------------

    @property
    def left_indices(self) -> list[int]:
        """Indices of the left arm joints within the full ``(N,)`` joint array, in ARM_JOINTS order."""
        return self._solver.left_indices

    @property
    def right_indices(self) -> list[int]:
        """Indices of the right arm joints within the full ``(N,)`` joint array, in ARM_JOINTS order."""
        return self._solver.right_indices

    def get_rest_q(self) -> np.ndarray:
        """Full (N,) rest pose vector in radians."""
        q = np.zeros(self._solver.num_joints, dtype=np.float32)
        for i, gi in enumerate(self._solver.left_indices):
            q[gi] = self._rest_pose_left[i]
        for i, gi in enumerate(self._solver.right_indices):
            q[gi] = self._rest_pose_right[i]
        return q

    # -- Core ---------------------------------------------------------------

    def step(self, frame: VRFrame, q_current: np.ndarray) -> np.ndarray:
        """Process one VRFrame. Returns updated full (N,) q in radians."""
        deadman = frame.l_lock and frame.r_lock
        if not deadman:
            self._active = False
            return q_current

        if not self._active:
            # OneEuroFilter ``_x_prev`` froze at the controller pose held when
            # the deadman was last released; reset so the engage-snap uses the
            # actual current pose instead of biasing toward stale state and
            # sweeping the IK target as the filter catches up.
            self._reset_pose_filters()
            # Pin posture to ``q_current`` so the held pose is itself the IK
            # fixed point. The default rest-pose attractor would otherwise pull
            # q in the EE null space at every frame, growing with distance from
            # rest; reset() restores the rest-pose attractor.
            self._solver.set_posture_pose(q_current)

        # Filter raw VR poses before IK to remove tracking noise / tremor.
        lp = self._f_l_pos.update(
            np.array(
                [frame.l_ee.position.x, frame.l_ee.position.y, frame.l_ee.position.z]
            )
        )
        lq = self._f_l_quat.update(
            np.array(
                [
                    frame.l_ee.quaternion.x,
                    frame.l_ee.quaternion.y,
                    frame.l_ee.quaternion.z,
                    frame.l_ee.quaternion.w,
                ]
            )
        )
        lq = lq / np.linalg.norm(lq)

        rp = self._f_r_pos.update(
            np.array(
                [frame.r_ee.position.x, frame.r_ee.position.y, frame.r_ee.position.z]
            )
        )
        rq = self._f_r_quat.update(
            np.array(
                [
                    frame.r_ee.quaternion.x,
                    frame.r_ee.quaternion.y,
                    frame.r_ee.quaternion.z,
                    frame.r_ee.quaternion.w,
                ]
            )
        )
        rq = rq / np.linalg.norm(rq)

        # Convert filtered poses to FLU — pure numpy
        left_pos, left_rot = _vr_to_flu_np(*lp, *lq)
        right_pos, right_rot = _vr_to_flu_np(*rp, *rq)

        le = self._f_l_elbow.update(
            np.array([frame.l_elbow.x, frame.l_elbow.y, frame.l_elbow.z])
        )
        re = self._f_r_elbow.update(
            np.array([frame.r_elbow.x, frame.r_elbow.y, frame.r_elbow.z])
        )
        left_e = np.array((le[2], le[1], -le[0]), dtype=np.float32)
        right_e = np.array((re[2], re[1], -re[0]), dtype=np.float32)

        if not self._active:
            self._active = True
            self._engage_snap(
                left_pos, left_rot, right_pos, right_rot, left_e, right_e, q_current
            )
            return q_current

        # Relative targets — pure numpy
        tl_pos, tl_rot = _relative_target_np(
            left_pos,
            left_rot,
            *self._snap_ctrl["left"],
            *self._snap_fk["left"],
        )
        tr_pos, tr_rot = _relative_target_np(
            right_pos,
            right_rot,
            *self._snap_ctrl["right"],
            *self._snap_fk["right"],
        )

        elbow_l = self._snap_elbow_fk["left"] + (left_e - self._snap_elbow_ctrl["left"])
        elbow_r = self._snap_elbow_fk["right"] + (
            right_e - self._snap_elbow_ctrl["right"]
        )

        return self._solver.ik(
            q_current,
            left_pose=(tl_pos, tl_rot),
            right_pose=(tr_pos, tr_rot),
            left_elbow_pos=elbow_l,
            right_elbow_pos=elbow_r,
        )

    def compute_reset_trajectory(
        self, q_current: np.ndarray, q_target: np.ndarray
    ) -> list[np.ndarray]:
        """Collision-aware trajectory. Each item is a full (N,) array in radians."""
        cfg = self._config
        max_dist_rad = float(np.max(np.abs(q_current - q_target)))
        duration = max_dist_rad / cfg.reset_speed
        n_steps = max(1, round(duration * cfg.frequency))
        trajectory: list[np.ndarray] = []
        q = np.array(q_current, dtype=np.float32)
        for i in range(n_steps):
            t = (i + 1) / n_steps
            alpha = t * t * (3.0 - 2.0 * t)
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
            trajectory.append(q.copy())
        return trajectory

    def reset(self) -> None:
        """Deactivate the deadman-switch state and clear snap poses and filter state.

        Call this before replaying a reset trajectory so the next deadman press
        performs a fresh engage-snap from the current IK pose.
        """
        self._active = False
        self._snap_ctrl = {}
        self._snap_fk = {}
        self._snap_elbow_ctrl = {}
        self._snap_elbow_fk = {}
        self._reset_pose_filters()
        # step() pins posture to q_current on each engage; an explicit reset
        # restores the default rest-pose attractor.
        self._solver.set_posture_pose(self.get_rest_q())

    # -- Internal -----------------------------------------------------------

    def _reset_pose_filters(self) -> None:
        """Clear the OneEuroFilter state for every controller and elbow stream."""
        self._f_l_pos.reset()
        self._f_l_quat.reset()
        self._f_r_pos.reset()
        self._f_r_quat.reset()
        self._f_l_elbow.reset()
        self._f_r_elbow.reset()

    def _settle_rest_pose(
        self, max_iterations: int = 200, tol: float = 1e-5
    ) -> np.ndarray:
        """Iterate the full teleop IK to the manipulability-balanced rest pose.

        EE and elbow targets are the configured rest pose's own FK, and posture
        is pinned to the current iterate, so all costs except manipulability
        have zero gradient at the starting q. The remaining manipulability
        gradient drives q in the EE null space until it stops changing — the
        same conditions the rising-edge posture pin in :meth:`step` produces
        at engage time.
        """
        q = self.get_rest_q()
        fk = self._solver.robot.forward_kinematics(jnp.asarray(q))

        def _pose(idx: int) -> tuple[np.ndarray, np.ndarray]:
            T = jaxlie.SE3(fk[idx])
            return (
                np.asarray(T.translation(), dtype=np.float32),
                np.asarray(T.rotation().as_matrix(), dtype=np.float32),
            )

        def _elbow(idx: int) -> np.ndarray:
            return np.asarray(jaxlie.SE3(fk[idx]).translation(), dtype=np.float32)

        l_pose = _pose(self._solver.l_ee_idx)
        r_pose = _pose(self._solver.r_ee_idx)
        l_elbow = _elbow(self._solver.l_elbow_idx)
        r_elbow = _elbow(self._solver.r_elbow_idx)

        for _ in range(max_iterations):
            self._solver.set_posture_pose(q)
            q_new = self._solver.ik(
                q,
                left_pose=l_pose,
                right_pose=r_pose,
                left_elbow_pos=l_elbow,
                right_elbow_pos=r_elbow,
            )
            if float(np.max(np.abs(q_new - q))) < tol:
                return q_new
            q = q_new
        return q

    def _engage_snap(
        self,
        left_pos: np.ndarray,
        left_rot: np.ndarray,
        right_pos: np.ndarray,
        right_rot: np.ndarray,
        left_e: np.ndarray,
        right_e: np.ndarray,
        q_current: np.ndarray,
    ) -> None:
        fk = self._solver.robot.forward_kinematics(jnp.asarray(q_current))

        def _fk_pos_rot(idx: int) -> tuple[np.ndarray, np.ndarray]:
            T = jaxlie.SE3(fk[idx])
            pos = np.asarray(T.translation(), dtype=np.float32)
            rot = np.asarray(T.rotation().as_matrix(), dtype=np.float32)
            return pos, rot

        self._snap_ctrl = {
            "left": (left_pos, left_rot),
            "right": (right_pos, right_rot),
        }
        self._snap_fk = {
            "left": _fk_pos_rot(self._solver.l_ee_idx),
            "right": _fk_pos_rot(self._solver.r_ee_idx),
        }
        self._snap_elbow_ctrl = {"left": left_e, "right": right_e}
        self._snap_elbow_fk = {
            "left": np.asarray(
                jaxlie.SE3(fk[self._solver.l_elbow_idx]).translation(), dtype=np.float32
            ),
            "right": np.asarray(
                jaxlie.SE3(fk[self._solver.r_elbow_idx]).translation(), dtype=np.float32
            ),
        }


# ---------------------------------------------------------------------------
# Subprocess entry point
# ---------------------------------------------------------------------------


def run_ik_worker(
    conn: multiprocessing.connection.Connection,
    config: VRTeleopConfig,
    kinematics_config: KinematicsConfig,
    q_current_left: np.ndarray | None = None,
    q_current_right: np.ndarray | None = None,
) -> None:
    """IK subprocess entry point."""
    try:
        os.nice(-10)
    except (AttributeError, OSError):
        pass

    worker = IKWorker(config, kinematics_config)
    q_rest = worker.get_rest_q()

    q_start = np.zeros_like(q_rest)
    if q_current_left is not None:
        for i, gi in enumerate(worker.left_indices):
            q_start[gi] = q_current_left[i]
    if q_current_right is not None:
        for i, gi in enumerate(worker.right_indices):
            q_start[gi] = q_current_right[i]

    startup_traj = worker.compute_reset_trajectory(q_start, q_rest)
    q = startup_traj[-1].copy() if startup_traj else q_rest.copy()

    conn.send(
        ("ready", q.copy(), worker.left_indices, worker.right_indices, startup_traj)
    )

    while True:
        try:
            msg = conn.recv()
            if msg is None:
                break
            if isinstance(msg, tuple) and msg[0] == "reset":
                q_current = np.asarray(msg[1], dtype=np.float32)
                traj = worker.compute_reset_trajectory(q_current, q_rest)
                worker.reset()
                q = traj[-1].copy() if traj else q_rest.copy()
                conn.send(("reset_traj", q_rest.copy(), traj))
            elif isinstance(msg, VRFrame):
                q = worker.step(msg, q)
                conn.send(q.copy())
        except (EOFError, KeyboardInterrupt):
            break
