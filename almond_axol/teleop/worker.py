"""
IK subprocess worker for VR teleoperation.

Runs in a separate process to keep JAX off the main asyncio event loop.
All intermediate computations stay in NumPy; the single JAX boundary is the
``solver.ik`` call itself (matching the arm-repo pattern).
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import multiprocessing.connection
import os
import time

import jax.numpy as jnp
import jaxlie
import numpy as np

from ..kinematics.config import KinematicsConfig
from ..kinematics.solver import KinematicsSolver
from ..vr.models import VRFrame
from .config import VRTeleopConfig
from .filter import OneEuroFilter
from .trajectory import plan_collision_aware_trajectory

_logger = logging.getLogger(__name__)

# Up direction of the raw VR world frame (WebXR reference space: +y is up).
_VR_UP = np.array([0.0, 1.0, 0.0])

# Freeze detection (see IKWorker._note_solve): warn when the solver keeps
# returning its seed unchanged while the tracked targets move away — the
# operator experiences the arm as stuck, then lurching once the solve breaks
# free. Thresholds: how long the output must be frozen before warning, how far
# the targets must have moved (so a still hand doesn't warn), and how often to
# re-warn while the freeze persists.
_FREEZE_WARN_AFTER_S = 0.5
_FREEZE_MIN_TARGET_DRIFT_M = 0.005
_FREEZE_REWARN_EVERY_S = 2.0

# ---------------------------------------------------------------------------
# NumPy-only helpers (no JAX dispatch overhead)
# ---------------------------------------------------------------------------


def _matrix_to_quat_xyzw(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to an ``(x, y, z, w)`` quaternion."""
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        return (
            float(R[2, 1] - R[1, 2]) / s,
            float(R[0, 2] - R[2, 0]) / s,
            float(R[1, 0] - R[0, 1]) / s,
            0.25 * s,
        )
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return (
            0.25 * s,
            float(R[0, 1] + R[1, 0]) / s,
            float(R[0, 2] + R[2, 0]) / s,
            float(R[2, 1] - R[1, 2]) / s,
        )
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return (
            float(R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            float(R[1, 2] + R[2, 1]) / s,
            float(R[0, 2] - R[2, 0]) / s,
        )
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return (
        float(R[0, 2] + R[2, 0]) / s,
        float(R[1, 2] + R[2, 1]) / s,
        0.25 * s,
        float(R[1, 0] - R[0, 1]) / s,
    )


def _quat_xyzw_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert an ``(x, y, z, w)`` quaternion to a 3x3 rotation matrix (float32)."""
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


def _scale_rotation_np(R: np.ndarray, scale: float) -> np.ndarray:
    """Scale the angle of a rotation matrix by ``scale`` (a power in SO(3)).

    Converts ``R`` to axis-angle, multiplies the angle by ``scale``, and maps
    back via Rodrigues' formula.  ``scale == 1.0`` and near-identity rotations
    are short-circuited.
    """
    if scale == 1.0:
        return R
    cos_theta = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) * 0.5))
    theta = math.acos(cos_theta)
    if theta < 1e-6:
        return R
    axis = np.array(
        (R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]),
        dtype=np.float64,
    ) / (2.0 * math.sin(theta))
    new_theta = theta * scale
    k = np.array(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        ),
        dtype=np.float64,
    )
    r_scaled = (
        np.eye(3) + math.sin(new_theta) * k + (1.0 - math.cos(new_theta)) * (k @ k)
    )
    return r_scaled.astype(np.float32)


def _relative_target_np(
    pos_curr: np.ndarray,
    rot_curr: np.ndarray,
    pos_snap_ctrl: np.ndarray,
    rot_snap_ctrl: np.ndarray,
    pos_snap_fk: np.ndarray,
    rot_snap_fk: np.ndarray,
    position_multiplier: float = 1.0,
    rotation_multiplier: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute absolute EE target from controller delta. Returns (pos_3, rot_3x3).

    ``position_multiplier`` scales only the translational displacement of the
    controller relative to its engage snapshot; ``rotation_multiplier`` scales
    only the angle of its orientation displacement.
    """
    d = (rot_snap_ctrl.T @ (pos_curr - pos_snap_ctrl)) * position_multiplier
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
    R_delta = _scale_rotation_np(R_delta, rotation_multiplier)
    return new_t.astype(np.float32), (rot_snap_fk @ R_delta).astype(np.float32)


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
        # Freeze detection state: when the last solve returned its seed
        # unchanged, `_freeze_since` holds the wall time the freeze started and
        # `_freeze_targets` the EE/elbow target positions at that moment, so a
        # warning fires only when the targets kept moving away (a still hand
        # legitimately produces an unchanged solution).
        self._freeze_since: float | None = None
        self._freeze_targets: np.ndarray | None = None
        self._freeze_next_warn: float = _FREEZE_WARN_AFTER_S
        # Snap poses as (pos_3, rot_3x3) numpy tuples — no jaxlie overhead
        self._snap_ctrl: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._snap_fk: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._snap_elbow_ctrl: dict[str, np.ndarray] = {}
        self._snap_elbow_fk: dict[str, np.ndarray] = {}

        # Absolute (UMI) mode state: the world-anchored base transform solved
        # at engage — ``(R_wb, t_wb)`` maps base-frame FLU coordinates into the
        # raw VR world frame — plus each controller's rigid controller→TCP
        # offset ``(p_off, R_off)`` expressed in the controller's local frame.
        self._abs_base: tuple[np.ndarray, np.ndarray] | None = None
        self._abs_offset: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        # Tracker→gripper transforms (the rig's factory design constants, or
        # per-unit file overrides — see almond_axol.umi.calibration), per
        # side as ``(p_off_3, R_off_3x3)`` in the tracker's local frame.
        # When present for a side, engage uses it verbatim instead of
        # absorbing the mount offset into the engage snapshot.
        self._tcp_transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for side, tf in (
            ("left", config.tcp_transform_left),
            ("right", config.tcp_transform_right),
        ):
            if tf is not None:
                self._tcp_transforms[side] = (
                    np.asarray(tf[:3], dtype=np.float64),
                    _quat_xyzw_to_matrix(*tf[3:]).astype(np.float64),
                )
        # Quaternion sign continuity for the calibrated pose mapping (see
        # :meth:`_apply_tcp_transform`).
        self._last_mapped_quat: dict[str, np.ndarray] = {}
        if self._tcp_transforms and config.absolute_mode:
            _logger.info(
                "absolute mode: using calibrated tracker→gripper transforms for %s",
                sorted(self._tcp_transforms),
            )
        # JSON-safe copy of the base transform for the headset (VR world
        # coords), so the web client can render the URDF at the engage-
        # calibrated base. ``None`` until the first engage.
        self.abs_base_msg: dict[str, list[float]] | None = None
        # Latest absolute-mode TCP target per side, in the robot base frame:
        # ``{"left": [x, y, z, qx, qy, qz, qw], "right": [...]}``. This is the
        # tracked ground-truth pose the IK solver chases — UMI data collection
        # records it per row so training can use raw TCP trajectories instead
        # of (or alongside) the IK joint solutions. Holds the last engaged
        # target while disengaged (mirroring the latched virtual joints);
        # seeded from rest FK so it is never ``None`` in absolute mode.
        self.last_tcp_msg: dict[str, list[float]] | None = None

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

        if config.absolute_mode:
            # Warm the no-elbow IK graph now: absolute mode never passes elbow
            # hints, and that distinct JAX graph would otherwise JIT-compile on
            # the first engage, stalling the session for the compile time.
            fk_l, fk_r = self._rest_fk_poses()
            self._solver.ik(self.get_rest_q(), left_pose=fk_l, right_pose=fk_r)
            self.last_tcp_msg = self._encode_tcp_msg(fk_l, fk_r)

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
        if self._config.absolute_mode:
            return self._step_absolute(frame, q_current)

        enabled = frame.l_lock and frame.r_lock
        if not enabled:
            self._active = False
            self._clear_freeze()
            return q_current

        if not self._active:
            # OneEuroFilter ``_x_prev`` froze at the controller pose held when
            # the toggle was last disabled; reset so the engage-snap uses the
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
            self._clear_freeze()
            self._engage_snap(
                left_pos, left_rot, right_pos, right_rot, left_e, right_e, q_current
            )
            return q_current

        pos_mult = self._config.position_multiplier
        rot_mult = self._config.rotation_multiplier
        tl_pos, tl_rot = _relative_target_np(
            left_pos,
            left_rot,
            *self._snap_ctrl["left"],
            *self._snap_fk["left"],
            position_multiplier=pos_mult,
            rotation_multiplier=rot_mult,
        )
        tr_pos, tr_rot = _relative_target_np(
            right_pos,
            right_rot,
            *self._snap_ctrl["right"],
            *self._snap_fk["right"],
            position_multiplier=pos_mult,
            rotation_multiplier=rot_mult,
        )

        elbow_l = self._snap_elbow_fk["left"] + pos_mult * (
            left_e - self._snap_elbow_ctrl["left"]
        )
        elbow_r = self._snap_elbow_fk["right"] + pos_mult * (
            right_e - self._snap_elbow_ctrl["right"]
        )

        q_new = self._solver.ik(
            q_current,
            left_pose=(tl_pos, tl_rot),
            right_pose=(tr_pos, tr_rot),
            left_elbow_pos=elbow_l,
            right_elbow_pos=elbow_r,
        )
        self._note_solve(
            bool(np.array_equal(q_new, q_current)),
            np.concatenate([tl_pos, tr_pos, elbow_l, elbow_r]),
        )
        return q_new

    def _step_absolute(self, frame: VRFrame, q_current: np.ndarray) -> np.ndarray:
        """Mantis UMI step: absolute world-anchored targets, no deltas.

        The engage rising edge solves the base transform + per-controller TCP
        offsets (:meth:`_engage_absolute`); every later frame maps each
        controller pose rigidly into the base frame and solves IK against the
        absolute target. Elbow hints are never passed — the operator's elbows
        say nothing about the robot's preferred null-space posture, which is
        instead anchored to the rest pose so joint solutions stay consistent
        across operators and episodes.
        """
        enabled = frame.l_lock and frame.r_lock
        if not enabled:
            self._active = False
            return q_current

        if not self._active:
            # Same rationale as the relative path: the One Euro state froze at
            # the pose held when tracking was last disabled.
            self._reset_pose_filters()

        # Apply the calibrated tracker→gripper transform (when present) to the
        # *raw* pose, before filtering: the signal of interest is the physical
        # gripper, and filtering the tracker pose instead would let the mount
        # lever arm leak filter lag into the gripper position during wrist
        # rotations (filtering does not commute with a rigid transform).
        lp_raw, lq_raw = self._apply_tcp_transform(
            "left",
            np.array(
                [frame.l_ee.position.x, frame.l_ee.position.y, frame.l_ee.position.z]
            ),
            np.array(
                [
                    frame.l_ee.quaternion.x,
                    frame.l_ee.quaternion.y,
                    frame.l_ee.quaternion.z,
                    frame.l_ee.quaternion.w,
                ]
            ),
        )
        rp_raw, rq_raw = self._apply_tcp_transform(
            "right",
            np.array(
                [frame.r_ee.position.x, frame.r_ee.position.y, frame.r_ee.position.z]
            ),
            np.array(
                [
                    frame.r_ee.quaternion.x,
                    frame.r_ee.quaternion.y,
                    frame.r_ee.quaternion.z,
                    frame.r_ee.quaternion.w,
                ]
            ),
        )
        lp = self._f_l_pos.update(lp_raw)
        lq = self._f_l_quat.update(lq_raw)
        lq = lq / np.linalg.norm(lq)
        rp = self._f_r_pos.update(rp_raw)
        rq = self._f_r_quat.update(rq_raw)
        rq = rq / np.linalg.norm(rq)

        l_pos, l_rot = (
            lp.astype(np.float64),
            _quat_xyzw_to_matrix(*lq).astype(np.float64),
        )
        r_pos, r_rot = (
            rp.astype(np.float64),
            _quat_xyzw_to_matrix(*rq).astype(np.float64),
        )

        if not self._active:
            self._active = True
            self._engage_absolute(l_pos, l_rot, r_pos, r_rot)
            # Anchor the null-space posture at rest (not q_current) so arm
            # configurations stay consistent across operators and episodes.
            self._solver.set_posture_pose(self.get_rest_q())
            # At engage the offsets make the controller poses coincide with
            # rest FK by construction, so the targets equal rest FK exactly.
            self.last_tcp_msg = self._encode_tcp_msg(
                self._absolute_target("left", l_pos, l_rot),
                self._absolute_target("right", r_pos, r_rot),
            )
            return q_current

        left_target = self._absolute_target("left", l_pos, l_rot)
        right_target = self._absolute_target("right", r_pos, r_rot)
        self.last_tcp_msg = self._encode_tcp_msg(left_target, right_target)
        return self._solver.ik(
            q_current,
            left_pose=left_target,
            right_pose=right_target,
        )

    def compute_reset_trajectory(
        self, q_current: np.ndarray, q_target: np.ndarray
    ) -> list[np.ndarray]:
        """Collision-aware trajectory. Each item is a full (N,) array in radians."""
        cfg = self._config
        return plan_collision_aware_trajectory(
            self._solver.robot,
            self._solver.robot_coll,
            q_current,
            q_target,
            speed=cfg.reset_speed,
            rate=cfg.frequency,
            min_duration=cfg.reset_min_duration,
            rest_weight=cfg.reset_rest_weight,
            limit_weight=cfg.reset_limit_weight,
            collision_margin=cfg.reset_collision_margin,
            collision_weight=cfg.reset_collision_weight,
            max_iterations=cfg.reset_max_iterations,
        )

    def reset(self) -> None:
        """Deactivate the engage-toggle state and clear snap poses and filter state.

        Call this before replaying a reset trajectory so the next engage
        performs a fresh engage-snap from the current IK pose.
        """
        self._active = False
        self._clear_freeze()
        self._snap_ctrl = {}
        self._snap_fk = {}
        self._snap_elbow_ctrl = {}
        self._snap_elbow_fk = {}
        self._abs_base = None
        self._abs_offset = {}
        self.abs_base_msg = None
        if self._config.absolute_mode:
            # The reset trajectory returns the arms to rest, so the recorded
            # TCP stream should land there too rather than hold the last
            # engaged target.
            self.last_tcp_msg = self._encode_tcp_msg(*self._rest_fk_poses())
        self._reset_pose_filters()
        # step() pins posture to q_current on each engage; an explicit reset
        # restores the default rest-pose attractor.
        self._solver.set_posture_pose(self.get_rest_q())

    # -- Internal -----------------------------------------------------------

    def _rest_fk_poses(
        self,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """Rest-pose FK gripper poses ``(left, right)`` in the base frame."""
        fk = self._solver.robot.forward_kinematics(jnp.asarray(self.get_rest_q()))

        def _pose(idx: int) -> tuple[np.ndarray, np.ndarray]:
            T = jaxlie.SE3(fk[idx])
            return (
                np.asarray(T.translation(), dtype=np.float64),
                np.asarray(T.rotation().as_matrix(), dtype=np.float64),
            )

        return _pose(self._solver.l_ee_idx), _pose(self._solver.r_ee_idx)

    def _engage_absolute(
        self,
        l_pos: np.ndarray,
        l_rot: np.ndarray,
        r_pos: np.ndarray,
        r_rot: np.ndarray,
    ) -> None:
        """Solve the world-anchored base transform + controller→TCP offsets.

        The operator is holding both grippers at the agreed start pose — the
        pose the robot's grippers occupy at rest, relative to the task scene.
        The base transform is gravity-aligned (base up = VR world up) with its
        yaw set so the base's left axis points from the right gripper to the
        left one, and its translation chosen so the rest-pose FK gripper
        midpoint lands on the measured gripper midpoint (``base_height``, when
        set, pins the vertical component to the robot's real mounting height
        instead).

        A side with a tracker→gripper transform (factory design constant or
        per-unit override) arrives here already mapped to the physical gripper
        pose (see :meth:`_apply_tcp_transform`), so its engage offset is
        identity — recorded poses are mount-independent, wrist rotations
        don't smear into position error, and the operator's residual
        alignment error at engage stays visible instead of being baked in.
        An uncalibrated side gets whatever offset makes the engage pose
        coincide exactly with rest FK — absorbing the physical mount
        transform, URDF frame conventions, and the operator's alignment
        error in one snapshot; its alignment quality at engage then bounds
        the episode's absolute accuracy.
        """
        fk_l, fk_r = self._rest_fk_poses()

        # The measured poses stand in for the gripper TCPs (exactly the TCPs
        # for calibrated sides; controller origins otherwise, their lever arm
        # absorbed into the per-side engage offsets below).
        fk_l_anchor, fk_r_anchor = fk_l[0], fk_r[0]

        # URDF base frame: +x = left, +y = forward, +z = up. Base up aligns
        # with world up; the yaw is set so the base-frame anchor-separation
        # direction (projected horizontal — pure ±x for the gripper origins)
        # maps onto the measured right→left direction.
        d = l_pos - r_pos
        d_h = d - np.dot(d, _VR_UP) * _VR_UP
        n = float(np.linalg.norm(d_h))
        if n < 1e-6:
            _logger.warning(
                "absolute engage: grippers are horizontally coincident; "
                "base yaw is arbitrary — re-engage with grippers apart."
            )
            d_h, n = np.array([1.0, 0.0, 0.0]), 1.0
        b = d_h / n
        d_b = fk_l_anchor - fk_r_anchor
        theta_a = math.atan2(float(d_b[1]), float(d_b[0]))
        x_axis = b * math.cos(theta_a) - np.cross(_VR_UP, b) * math.sin(theta_a)
        x_axis /= np.linalg.norm(x_axis)
        z_axis = _VR_UP
        y_axis = np.cross(z_axis, x_axis)
        R_wb = np.column_stack([x_axis, y_axis, z_axis])

        mid_w = 0.5 * (l_pos + r_pos)
        mid_b = 0.5 * (fk_l_anchor + fk_r_anchor)
        t_wb = mid_w - R_wb @ mid_b
        if self._config.base_height is not None:
            t_wb[1] = float(self._config.base_height)
        self._abs_base = (R_wb, t_wb)
        self.abs_base_msg = {
            "pos": [float(v) for v in t_wb],
            "quat": list(_matrix_to_quat_xyzw(R_wb)),
        }

        def _offset(
            side: str,
            ctrl_pos: np.ndarray,
            ctrl_rot: np.ndarray,
            fk_pose: tuple[np.ndarray, np.ndarray],
        ) -> tuple[np.ndarray, np.ndarray]:
            if side in self._tcp_transforms:
                # Pose already mapped to the gripper by _apply_tcp_transform.
                return np.zeros(3), np.eye(3)
            fk_pos, fk_rot = fk_pose
            r_off = ctrl_rot.T @ (R_wb @ fk_rot)
            p_w_tcp = R_wb @ fk_pos + t_wb
            return ctrl_rot.T @ (p_w_tcp - ctrl_pos), r_off

        self._abs_offset = {
            "left": _offset("left", l_pos, l_rot, fk_l),
            "right": _offset("right", r_pos, r_rot, fk_r),
        }

    def _apply_tcp_transform(
        self, side: str, pos: np.ndarray, quat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map a raw tracker pose to the gripper pose via the calibration.

        No-op when the side has no calibrated transform. Returns
        ``(pos_3, quat_xyzw)``; the output quaternion's sign is kept
        continuous with the previous frame so the One Euro quaternion filter
        never sees a representation flip.
        """
        tf = self._tcp_transforms.get(side)
        if tf is None:
            return pos, quat
        p_x, R_x = tf
        R_c = _quat_xyzw_to_matrix(*quat).astype(np.float64)
        out_pos = np.asarray(pos, dtype=np.float64) + R_c @ p_x
        out_quat = np.array(_matrix_to_quat_xyzw(R_c @ R_x))
        prev = self._last_mapped_quat.get(side)
        if prev is not None and float(np.dot(out_quat, prev)) < 0.0:
            out_quat = -out_quat
        self._last_mapped_quat[side] = out_quat
        return out_pos, out_quat

    @staticmethod
    def _encode_tcp_msg(
        left: tuple[np.ndarray, np.ndarray],
        right: tuple[np.ndarray, np.ndarray],
    ) -> dict[str, list[float]]:
        """Pack two base-frame ``(pos, rot)`` poses as JSON-safe pos+quat lists."""

        def _enc(pose: tuple[np.ndarray, np.ndarray]) -> list[float]:
            pos, rot = pose
            quat = _matrix_to_quat_xyzw(np.asarray(rot, dtype=np.float64))
            return [float(pos[0]), float(pos[1]), float(pos[2]), *quat]

        return {"left": _enc(left), "right": _enc(right)}

    def _absolute_target(
        self, side: str, pos: np.ndarray, rot: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map a controller pose rigidly into the base frame. Returns (pos_3, rot_3x3)."""
        assert self._abs_base is not None
        R_wb, t_wb = self._abs_base
        p_off, R_off = self._abs_offset[side]
        p_w = pos + rot @ p_off
        R_w = rot @ R_off
        p_b = R_wb.T @ (p_w - t_wb)
        R_b = R_wb.T @ R_w
        return p_b.astype(np.float32), R_b.astype(np.float32)

    def _clear_freeze(self) -> None:
        """Forget any in-progress freeze run (disengage / engage / reset)."""
        self._freeze_since = None
        self._freeze_targets = None
        self._freeze_next_warn = _FREEZE_WARN_AFTER_S

    def _note_solve(self, frozen: bool, targets: np.ndarray) -> None:
        """Track seed-returning solves; WARN when the output freezes.

        A single unchanged solution is normal (e.g. the hand is still, or the
        target is held against a constraint). The failure mode worth flagging is
        a *run* of solves that return the seed while the EE/elbow targets keep
        moving away — the operator sees the arm stop responding, and the
        accumulated error is released as a lurch once a solve finally makes
        progress (see ``KinematicsConfig`` for the tuning that minimises this).

        Args:
            frozen: True when this solve returned ``q_current`` bit-identically.
            targets: Concatenated EE + elbow target positions (m) of this solve,
                used to measure how far the targets drifted during the freeze.
        """
        if not frozen:
            self._clear_freeze()
            return
        now = time.monotonic()
        if self._freeze_since is None or self._freeze_targets is None:
            self._freeze_since = now
            self._freeze_targets = targets
            return
        duration = now - self._freeze_since
        drift = float(np.max(np.abs(targets - self._freeze_targets)))
        if duration >= self._freeze_next_warn and drift >= _FREEZE_MIN_TARGET_DRIFT_M:
            _logger.warning(
                "IK frozen for %.1fs: solver keeps returning its seed while the "
                "EE/elbow targets moved %.0f mm — it cannot make progress "
                "(target likely conflicts with the self-collision margin or "
                "joint limits); the arm holds, then catches up in a lurch when "
                "the solve breaks free.",
                duration,
                drift * 1e3,
            )
            self._freeze_next_warn = duration + _FREEZE_REWARN_EVERY_S

    def _reset_pose_filters(self) -> None:
        """Clear the OneEuroFilter state for every controller and elbow stream."""
        self._last_mapped_quat = {}
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
        """Snapshot controller and FK poses at toggle engage.

        These snapshots become the origin against which subsequent controller
        motion is measured to build relative EE and elbow targets in :meth:`step`.
        """
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
    # Confine the JAX solve to a single core's worth of compute. The per-arm IK
    # is tiny, but XLA's CPU backend fans its Eigen thread pool across *every*
    # core for each solve; combined with this process's nice(-10) boost, that
    # burst preempts the control loop's CAN round-trip and the video relay on
    # every step — exactly the engaged-only send/act latency spikes and grainy
    # frames seen in `collect-data`, which (unlike teleop) has no spare core
    # headroom once the relay's raw-frame branch is running. Single-threaded XLA
    # is no slower for a problem this small and leaves the real-time loop alone.
    # Must be set before the first JAX op (backend init reads XLA_FLAGS lazily).
    _xla = os.environ.get("XLA_FLAGS", "")
    if "xla_cpu_multi_thread_eigen" not in _xla:
        os.environ["XLA_FLAGS"] = f"{_xla} --xla_cpu_multi_thread_eigen=false".strip()
    for _var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(_var, "1")

    try:
        os.nice(-10)
    except (AttributeError, OSError):
        pass

    # IK affinity is applied in two phases. The one-time startup that follows —
    # JAX/XLA compile, the rest-pose settle, and the collision-aware startup
    # trajectory — is heavy and must finish inside the caller's 60s connect
    # handshake, so it runs *widened* across the control-side cores (safe: the
    # control loop and recording haven't started yet). Confining it to the single
    # dedicated IK core instead roughly triples its wall time and blows that
    # handshake. Only the steady-state solve loop is narrowed to the dedicated IK
    # core (below, right after the ready handshake) so recording load can't preempt
    # it mid-solve.
    from ..utils import affinity

    affinity.pin_ik_startup()

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

    # Startup compile/settle/trajectory are done and the handshake is sent: narrow
    # to the dedicated IK core so per-frame solves aren't preempted by recording
    # load (on <8-core hosts this collapses onto the realtime cores).
    affinity.pin_ik()

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
                if config.absolute_mode:
                    # Absolute (UMI) mode replies carry the engage-calibrated
                    # base transform (for the headset's URDF overlay) and the
                    # base-frame TCP targets (recorded per dataset row by UMI
                    # data collection) alongside the joint solution.
                    conn.send(("q", q.copy(), worker.abs_base_msg, worker.last_tcp_msg))
                else:
                    conn.send(q.copy())
        except (EOFError, KeyboardInterrupt):
            break
