"""
IK subprocess worker for VR teleoperation.

Runs in a separate process to keep the solver (JAX / QP) off the main asyncio
event loop. All intermediate computations stay in NumPy; the solver boundary
is the :meth:`IKBackend.ik` call itself. The backend implementation is
selected by ``KinematicsConfig.backend`` (see
:mod:`almond_axol.kinematics.backends`).
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import multiprocessing.connection
import os
import time

import numpy as np

from ..kinematics.backends import create_backend
from ..kinematics.conditioning import (
    ColumnKeepout,
    clamp_reach,
    clamp_target_error,
    clear_swivel_angle,
    elbow_circle,
    elevated_swivel_angle,
    swivel_direction,
    swivel_frame,
)
from ..kinematics.config import KinematicsConfig
from ..kinematics.pyroki_model import (
    canonical_to_pyroki,
    load_pyroki_model,
    to_canonical_order,
    to_pyroki_order,
)
from ..vr.models import VRFrame
from .config import VRTeleopConfig
from .filter import OneEuroFilter
from .trajectory import plan_collision_aware_trajectory

_logger = logging.getLogger(__name__)

# Freeze detection (see IKWorker._note_solve): warn when the solver keeps
# returning its seed unchanged while the tracked targets move away — the
# operator experiences the arm as stuck, then lurching once the solve breaks
# free. Thresholds: how long the output must be frozen before warning, how far
# the targets must have moved (so a still hand doesn't warn), and how often to
# re-warn while the freeze persists.
_FREEZE_WARN_AFTER_S = 0.5
_FREEZE_MIN_TARGET_DRIFT_M = 0.005
_FREEZE_REWARN_EVERY_S = 2.0

# Elbow swivel smoothing, per 120 Hz tick, applied to the swivel *angle*
# relative to the previous reference (see conditioning.swivel_frame). The EMA
# factor sets the lag toward a new operator swivel; the rate limit caps how
# fast the reference can move regardless. The rate limit is the important
# one: the swivel is the arm's only self-motion and near the singular
# shoulder its gain is unbounded, so reference noise (the inferred elbow
# whips up to ~22 deg/tick when the wrist target passes near the shoulder)
# becomes wild shoulder rotation. 2 deg/tick = 240 deg/s still tracks a
# deliberate 90-degree swivel in under half a second.
_SWIVEL_BLEND_ALPHA = 0.15
_SWIVEL_MAX_STEP_RAD = math.radians(2.0)

# How far (radians) the swivel reference may be rotated up from the operator's
# own elbow toward the highest elbow the arm could reach. The headset infers
# the operator's elbow rather than measuring it, and that inference sits at the
# bottom of the swivel range — measured over a captured session it asked for
# the 4th percentile of reachable elbow elevation, which is why the elbow never
# rose when the operator reached out. 40 degrees restores the elevation without
# taking the swivel far enough to fight the end-effector task. 0.0 follows the
# operator's elbow exactly.
_SWIVEL_LIFT_LIMIT_RAD = math.radians(40.0)

# Margin (m) added to the theoretical folded-arm distance |upper - forearm|
# for the inner reach clamp around each shoulder.
_MIN_REACH_MARGIN_M = 0.05

# Margin (m) subtracted from the arm's full extension (upper + forearm) for
# the outer reach clamp. Reach is a flat function of elbow angle near the
# straight arm, so a small radial margin buys a lot of bend: 30 mm keeps the
# elbow >= ~28 deg from its straight stop.
_MAX_REACH_MARGIN_M = 0.03

# Keep-out around the base column, used by the (opt-in) elbow-swivel
# reference so it never asks for an elbow inside the robot's own torso. The
# column collision hull spans x[-0.13, 0.13], y[-0.06, 0.06] up to z=0.80,
# with the s1 yoke extending it to z=0.92 (measured from the URDF convex
# hulls); semi-axes are inflated 45 mm for link thickness.
_COLUMN_KEEPOUT = ColumnKeepout(half_x=0.13 + 0.045, half_y=0.06 + 0.045, top_z=0.92)

# The headset's elbow positions are *inferred* by a generative body model from
# the headset + controller poses, not measured; when the model re-localises it
# teleports (up to 70 mm in a single 8 ms frame in captured sessions — 8+ m/s,
# physically impossible). The OneEuro pose filter passes fast steps by design,
# so those teleports must be rejected before it: a per-frame step larger than
# this (m) keeps the previous sample instead. A real elbow swing measures
# ~10 mm/frame at 120 Hz, so the gate never engages on genuine motion.
_ELBOW_JUMP_REJECT_M = 0.04
# A model flip can be persistent (the new estimate is the one that tracks);
# after this many consecutive rejected frames (~0.25 s) the new position is
# accepted so the hint can't stay latched to a stale estimate.
_ELBOW_JUMP_ACCEPT_AFTER = 30

# ---------------------------------------------------------------------------
# NumPy-only helpers (no JAX dispatch overhead)
# ---------------------------------------------------------------------------


def _wrap_pi(angle: float) -> float:
    """Wrap an angle (radians) into ``(-pi, pi]``."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


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

    Snap state is numpy-only. The solver boundary is the backend's ``ik``
    call inside :meth:`step`.
    """

    def __init__(
        self, config: VRTeleopConfig, kinematics_config: KinematicsConfig
    ) -> None:
        """Construct the IK worker.

        Instantiates the IK backend selected by ``kinematics_config.backend``
        and initialises One Euro Filters for all VR pose streams.

        Args:
            config:            Teleop session parameters (rest poses, frequency, filter settings).
            kinematics_config: IK solver parameters (backend selector + weights).
        """
        self._config = config
        self._kin = kinematics_config
        self._backend = create_backend(kinematics_config, dt=1.0 / config.frequency)

        self._rest_pose_left = np.asarray(config.rest_pose_left, dtype=np.float32)
        self._rest_pose_right = np.asarray(config.rest_pose_right, dtype=np.float32)

        self._backend.set_posture_pose(self.get_rest_q())

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

        freq = config.frequency
        mc = config.pose_min_cutoff
        beta = config.pose_beta
        self._f_l_pos = OneEuroFilter(freq, mc, beta)
        self._f_l_quat = OneEuroFilter(freq, mc, beta)
        self._f_r_pos = OneEuroFilter(freq, mc, beta)
        self._f_r_quat = OneEuroFilter(freq, mc, beta)
        self._f_l_elbow = OneEuroFilter(freq, mc, beta)
        self._f_r_elbow = OneEuroFilter(freq, mc, beta)

        # Pre-settle the configured rest pose to the solver's own fixed point
        # (identity for the differential backends; the pyroki-lm baseline
        # settles its manipulability-cost null-space drift). Baking the
        # settling in at startup means the startup trajectory ends at the
        # fixed point and the first engage produces no motion.
        q_settled = self._backend.settle_rest_pose(self.get_rest_q())
        self._rest_pose_left = q_settled[self._backend.left_indices].astype(np.float32)
        self._rest_pose_right = q_settled[self._backend.right_indices].astype(
            np.float32
        )
        self._backend.set_posture_pose(self.get_rest_q())

        # Robot arm segment lengths (m) for the elbow swivel projection,
        # measured at the rest pose (the shoulder->elbow and elbow->EE
        # distances vary only marginally with wrist configuration).
        rest_frames = self._backend.fk_frames(self.get_rest_q())
        self._l_upper = float(
            np.linalg.norm(rest_frames.left_elbow - self._backend.left_shoulder_pos)
        )
        self._l_fore = float(
            np.linalg.norm(rest_frames.left_ee[0] - rest_frames.left_elbow)
        )
        self._r_upper = float(
            np.linalg.norm(rest_frames.right_elbow - self._backend.right_shoulder_pos)
        )
        self._r_fore = float(
            np.linalg.norm(rest_frames.right_ee[0] - rest_frames.right_elbow)
        )
        # Inner reach clamp: closer than the folded-arm distance no feasible
        # configuration exists (the shoulder-singular region) and the solve
        # grinds against joint limits.
        self._l_min_reach = abs(self._l_upper - self._l_fore) + _MIN_REACH_MARGIN_M
        self._r_min_reach = abs(self._r_upper - self._r_fore) + _MIN_REACH_MARGIN_M
        # Outer reach clamp: the configured max_reach, additionally capped by
        # the arm's *physical* extension minus a margin. The config default
        # (0.8 m) predates this arm's geometry — full extension is only
        # ~0.73 m from the shoulder, so without the cap the clamp never
        # engaged and operators could command past full extension, grinding
        # the straight-elbow singularity and the e1 = 0 travel stop. The
        # 30 mm margin keeps the elbow at least ~28 deg bent, where the arm
        # still has authority to track radially.
        self._l_max_reach = min(
            kinematics_config.max_reach,
            self._l_upper + self._l_fore - _MAX_REACH_MARGIN_M,
        )
        self._r_max_reach = min(
            kinematics_config.max_reach,
            self._r_upper + self._r_fore - _MAX_REACH_MARGIN_M,
        )
        # Low-passed elbow swivel direction per arm (see _elbow_reference).
        self._swivel_dir: dict[str, np.ndarray | None] = {"left": None, "right": None}
        # Jump-rejection state for the inferred elbow hints (see _gate_elbow).
        self._elbow_prev_raw: dict[str, np.ndarray | None] = {
            "left": None,
            "right": None,
        }
        self._elbow_reject_count: dict[str, int] = {"left": 0, "right": 0}

        # Permutation for the pyroki-based reset planner (built lazily so the
        # non-pyroki backends only pay for it on the first reset/startup plan).
        self._pk_perm: np.ndarray | None = None

    # -- Properties the main process needs ----------------------------------

    @property
    def left_indices(self) -> list[int]:
        """Indices of the left arm joints within the full ``(N,)`` joint array, in ARM_JOINTS order."""
        return self._backend.left_indices

    @property
    def right_indices(self) -> list[int]:
        """Indices of the right arm joints within the full ``(N,)`` joint array, in ARM_JOINTS order."""
        return self._backend.right_indices

    def get_rest_q(self) -> np.ndarray:
        """Full (N,) rest pose vector in radians."""
        q = np.zeros(self._backend.num_joints, dtype=np.float32)
        for i, gi in enumerate(self._backend.left_indices):
            q[gi] = self._rest_pose_left[i]
        for i, gi in enumerate(self._backend.right_indices):
            q[gi] = self._rest_pose_right[i]
        return q

    # -- Core ---------------------------------------------------------------

    def step(self, frame: VRFrame, q_current: np.ndarray) -> np.ndarray:
        """Process one VRFrame. Returns updated full (N,) q in radians."""
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
            self._backend.set_posture_pose(q_current)

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
            self._gate_elbow(
                "left", np.array([frame.l_elbow.x, frame.l_elbow.y, frame.l_elbow.z])
            )
        )
        re = self._f_r_elbow.update(
            self._gate_elbow(
                "right", np.array([frame.r_elbow.x, frame.r_elbow.y, frame.r_elbow.z])
            )
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

        (tl_pos, tl_rot), (tr_pos, tr_rot), elbow_l, elbow_r = self._condition_targets(
            q_current, (tl_pos, tl_rot), (tr_pos, tr_rot), elbow_l, elbow_r
        )

        q_new = self._backend.ik(
            q_current,
            left_pose=(tl_pos, tl_rot),
            right_pose=(tr_pos, tr_rot),
            left_elbow_pos=elbow_l,
            right_elbow_pos=elbow_r,
        )
        self._note_solve(
            bool(np.array_equal(q_new, q_current)),
            np.concatenate(
                [
                    tl_pos,
                    tr_pos,
                    elbow_l if elbow_l is not None else np.zeros(3, np.float32),
                    elbow_r if elbow_r is not None else np.zeros(3, np.float32),
                ]
            ),
        )
        return q_new

    def _condition_targets(
        self,
        q_current: np.ndarray,
        left_pose: tuple[np.ndarray, np.ndarray],
        right_pose: tuple[np.ndarray, np.ndarray],
        elbow_l: np.ndarray,
        elbow_r: np.ndarray,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
        np.ndarray | None,
        np.ndarray | None,
    ]:
        """Backend-independent target conditioning (see kinematics.conditioning).

        1. Reach clamp: EE targets are pulled into the ``[min_reach,
           max_reach]`` annulus around each shoulder — outside the folded-arm
           (shoulder-singular) zone, inside the arm's reach.
        2. Error clamp: the SE(3) error between the current EE pose and its
           target is capped (jointly for position and orientation), so fast or
           unreachable targets become a bounded, direction-preserving pull.
        3. Elbow swivel (only when ``elbow_swivel`` is enabled): the
           operator's elbow hint keeps only its (smoothed, rate-limited)
           swivel angle about the shoulder->wrist axis, rotated to the
           nearest angle that also clears the column; the reference the
           solver sees lies exactly on the robot's own reachable elbow
           circle.
        """
        kin = self._kin
        frames = self._backend.fk_frames(q_current)

        tl_pos, tl_rot = left_pose
        tr_pos, tr_rot = right_pose
        tl_pos = clamp_reach(
            tl_pos,
            self._backend.left_shoulder_pos,
            self._l_max_reach,
            self._l_min_reach,
        )
        tr_pos = clamp_reach(
            tr_pos,
            self._backend.right_shoulder_pos,
            self._r_max_reach,
            self._r_min_reach,
        )
        tl_pos, tl_rot = clamp_target_error(
            *frames.left_ee,
            tl_pos,
            tl_rot,
            kin.max_target_err_lin,
            kin.max_target_err_ang,
        )
        tr_pos, tr_rot = clamp_target_error(
            *frames.right_ee,
            tr_pos,
            tr_rot,
            kin.max_target_err_lin,
            kin.max_target_err_ang,
        )

        out_l: np.ndarray | None = elbow_l
        out_r: np.ndarray | None = elbow_r
        if kin.elbow_swivel and kin.diff_elbow_cost > 0.0:
            out_l = self._elbow_reference(
                "left",
                self._backend.left_shoulder_pos,
                tl_pos,
                elbow_l,
                frames.left_elbow,
                self._l_upper,
                self._l_fore,
            )
            out_r = self._elbow_reference(
                "right",
                self._backend.right_shoulder_pos,
                tr_pos,
                elbow_r,
                frames.right_elbow,
                self._r_upper,
                self._r_fore,
            )
        return (tl_pos, tl_rot), (tr_pos, tr_rot), out_l, out_r

    def _elbow_reference(
        self,
        side: str,
        shoulder: np.ndarray,
        wrist_target: np.ndarray,
        elbow_raw: np.ndarray,
        elbow_fk: np.ndarray,
        upper_len: float,
        fore_len: float,
    ) -> np.ndarray | None:
        """Smoothed, column-clear swivel reference from the operator's elbow hint.

        The swivel angle is measured relative to the reference this method
        last produced, EMA-blended toward the hint and rate-limited, then
        rotated to the nearest angle that clears the base column. Working
        relative to the previous reference keeps the whole chain
        well-conditioned: "hint unchanged" is exactly angle zero, and a
        degenerate tick (hint collinear with the arm axis) simply holds the
        previous swivel rather than dropping the reference, which would leave
        the arm's only self-motion unconstrained.

        Seeded from the robot's current FK elbow, so the first engaged tick
        references the swivel the arm is already in and nothing snaps; the
        rate limit then walks it to the prior over a few tenths of a second.
        """
        circle = elbow_circle(shoulder, wrist_target, upper_len, fore_len)
        if circle is None:
            return None

        prev = self._swivel_dir[side]
        if prev is None:
            prev = swivel_direction(shoulder, wrist_target, elbow_fk)
            if prev is None:
                return None
        basis = swivel_frame(circle.axis, prev)
        if basis is None:
            return None
        e_a, e_b = basis

        # Follow the operator's swivel, then rotate it a bounded amount toward
        # the highest elbow the circle allows. The operator stays in control —
        # this only corrects the body model's documented downward bias — and
        # the bound is what keeps it honest: aiming *at* maximum elevation
        # instead drove the shoulder to its travel limit for a quarter of a
        # captured session and cost up to 100 mm of hand-tracking accuracy,
        # since an extreme swivel competes with the end-effector task.
        hint = swivel_direction(shoulder, wrist_target, elbow_raw)
        base = 0.0  # angle zero is "hold the previous swivel"
        if hint is not None:
            base = float(np.arctan2(np.dot(hint, e_b), np.dot(hint, e_a)))
        lift = _wrap_pi(elevated_swivel_angle(e_a, e_b) - base)
        target = base + float(
            np.clip(lift, -_SWIVEL_LIFT_LIMIT_RAD, _SWIVEL_LIFT_LIMIT_RAD)
        )
        step = float(
            np.clip(
                _SWIVEL_BLEND_ALPHA * _wrap_pi(target),
                -_SWIVEL_MAX_STEP_RAD,
                _SWIVEL_MAX_STEP_RAD,
            )
        )

        angle = clear_swivel_angle(
            circle, e_a, e_b, wrist_target, step, _COLUMN_KEEPOUT
        )

        direction = np.cos(angle) * e_a + np.sin(angle) * e_b
        self._swivel_dir[side] = direction
        return circle.point(direction)

    def compute_reset_trajectory(
        self, q_current: np.ndarray, q_target: np.ndarray
    ) -> list[np.ndarray]:
        """Collision-aware trajectory. Each item is a full (N,) array in radians.

        Reset/startup trajectories are offline plans and stay on the shared
        pyroki model regardless of the selected IK backend; joint vectors are
        permuted between the canonical and pyroki orders at this boundary.
        """
        cfg = self._config
        _, robot, robot_coll = load_pyroki_model()
        if self._pk_perm is None:
            self._pk_perm = canonical_to_pyroki(robot)
        traj = plan_collision_aware_trajectory(
            robot,
            robot_coll,
            to_pyroki_order(np.asarray(q_current, dtype=np.float32), self._pk_perm),
            to_pyroki_order(np.asarray(q_target, dtype=np.float32), self._pk_perm),
            speed=cfg.reset_speed,
            rate=cfg.frequency,
            min_duration=cfg.reset_min_duration,
            rest_weight=cfg.reset_rest_weight,
            limit_weight=cfg.reset_limit_weight,
            collision_margin=cfg.reset_collision_margin,
            collision_weight=cfg.reset_collision_weight,
            max_iterations=cfg.reset_max_iterations,
        )
        return [to_canonical_order(q, self._pk_perm) for q in traj]

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
        self._swivel_dir = {"left": None, "right": None}
        self._reset_pose_filters()
        # step() pins posture to q_current on each engage; an explicit reset
        # restores the default rest-pose attractor.
        self._backend.set_posture_pose(self.get_rest_q())

    # -- Internal -----------------------------------------------------------

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

    def _gate_elbow(self, side: str, raw: np.ndarray) -> np.ndarray:
        """Reject single-frame teleports in the inferred elbow position.

        The headset's body model re-localises in discrete jumps that no
        low-pass filter downstream is allowed to smooth away (OneEuro tracks
        fast steps by design). A step above ``_ELBOW_JUMP_REJECT_M`` holds the
        previous sample; a persistent jump (the model settled on a new
        estimate) is accepted after ``_ELBOW_JUMP_ACCEPT_AFTER`` frames.
        """
        prev = self._elbow_prev_raw[side]
        if (
            prev is not None
            and float(np.linalg.norm(raw - prev)) > _ELBOW_JUMP_REJECT_M
            and self._elbow_reject_count[side] < _ELBOW_JUMP_ACCEPT_AFTER
        ):
            self._elbow_reject_count[side] += 1
            return prev
        self._elbow_reject_count[side] = 0
        self._elbow_prev_raw[side] = raw
        return raw

    def _reset_pose_filters(self) -> None:
        """Clear the OneEuroFilter state for every controller and elbow stream."""
        self._f_l_pos.reset()
        self._f_l_quat.reset()
        self._f_r_pos.reset()
        self._f_r_quat.reset()
        self._f_l_elbow.reset()
        self._f_r_elbow.reset()
        self._elbow_prev_raw = {"left": None, "right": None}
        self._elbow_reject_count = {"left": 0, "right": 0}

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
        frames = self._backend.fk_frames(q_current)

        self._snap_ctrl = {
            "left": (left_pos, left_rot),
            "right": (right_pos, right_rot),
        }
        self._snap_fk = {
            "left": frames.left_ee,
            "right": frames.right_ee,
        }
        self._snap_elbow_ctrl = {"left": left_e, "right": right_e}
        self._snap_elbow_fk = {
            "left": frames.left_elbow,
            "right": frames.right_elbow,
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
                conn.send(q.copy())
        except (EOFError, KeyboardInterrupt, OSError):
            # OSError covers ConnectionResetError/BrokenPipeError when the
            # parent end closes abruptly (parent crash, or a shutdown that
            # left an in-flight response unread — the close then RSTs this
            # end). Exit cleanly instead of dying with a traceback.
            break
