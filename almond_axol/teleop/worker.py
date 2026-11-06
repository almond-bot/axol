"""
IK subprocess worker for VR teleoperation.

Runs in a separate process to keep IK off the main asyncio event loop.
Tracking and reset planning use the selected backend; Mink requires no JAX packages.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import multiprocessing.connection
import os
import signal
import time
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..kinematics.config import KinematicsConfig
from ..vr.models import VRFrame
from .absolute_pose import AbsoluteCalibration
from .absolute_pose import matrix_to_quat_xyzw as _matrix_to_quat_xyzw
from .box import (
    URDF_TOOL,
    BoxState,
    Faces,
    Pose,
    ToolGeometry,
    blend_pose,
    box_frame,
    box_targets,
    choose_faces,
    contact_width,
    elbow_swivel_hint,
    ideal_gripper_poses,
    pair_aligned,
    parcel_tool,
    right_stick_axis,
    rodrigues,
    rotation_angle,
    smoothstep,
    snap_box,
    squeeze_lean,
    swivel_step,
    tip_inward_sign,
    tipped_frame_holds,
    toe_out_sides,
)
from .config import VRTeleopConfig
from .filter import LagCompensatedLowPass
from .recorder import make as _recorder_make

_logger = logging.getLogger(__name__)


def _make_jax_solver(config: KinematicsConfig) -> Any:
    # Keep this import behind backend selection, including worker startup.
    from ..kinematics.solver import KinematicsSolver

    return KinematicsSolver(config)


# Named goals a ``("reset_path", …)`` request chains through: the configured
# rest pose, and joint zero — the arms' natural resting pose, where they
# settle once torqued off.
PATH_GOALS = ("rest", "zero")
# A path leg whose start is already this close to its goal on every joint is
# skipped: each planned leg plays for at least ``reset_min_duration``, which
# would otherwise hold the arms still for that long.
_PATH_LEG_SKIP_RAD = 1e-3

# Up direction of the raw VR world frame (WebXR reference space: +y is up).
_VR_UP = np.array([0.0, 1.0, 0.0])

# Absolute-engage side-swap guard (see IKWorker._side_swap_rejection): a
# side counts as facing away from the rest heading when its horizontal
# gripper heading is more than 120° from the rest FK heading, and only when
# that heading is far enough from vertical for the comparison to mean
# anything.
_SWAP_GUARD_COS = math.cos(math.radians(120.0))
_SWAP_GUARD_MIN_HORIZONTAL = 0.3

# Freeze handling (see IKWorker._note_solve): when one arm's solver output keeps
# returning its seed unchanged while that arm's tracked target moves away, the
# operator experiences a hold followed by a catch-up lurch. After a confirmed
# run, automatically clutch that controller at the held IK pose: the snap frame
# itself cannot move, future hand deltas retain the normal relative mapping, and
# only the unexecuted motion accumulated during the freeze is discarded.
_FREEZE_WARN_AFTER_S = 0.5
_FREEZE_MIN_TARGET_DRIFT_M = 0.005
_FREEZE_MIN_TARGET_DRIFT_RAD = math.radians(5.0)

# Tracking glitch rejection (see IKWorker._frame_snap_verdict): the VR pose
# stream carries two kinds of both-hand discontinuity that the operator's
# hands did not produce, both measured in recorded sessions:
#
#   * one-to-two-frame *blips* — the raw pose jumps 20-45 mm and bounces
#     straight back (one headset emitted these on a strict 10 s period);
#   * persistent world-frame *shifts* — a headset re-localization teleports
#     both controllers (up to 96 mm observed, right after a 46 ms tracking
#     dropout) and the offset never reverts.
#
# Followed naively, either kind lurches the arm (4.5° in 100 ms measured).
# Detection: both hands must miss a constant-velocity prediction by more
# than a noise floor plus what a generous hand acceleration could produce
# over the frame gap (a single occluded controller snapping back is a
# different failure with a different correct response — re-anchoring there
# would bake its error in). A trigger opens a short *suspect window* during
# which the arm holds and the frames are quarantined; the window then
# resolves to discard (blip reverted), genuine-motion resume (offset kept
# growing — e.g. a hard bimanual flick that beat the prediction), or a
# confirmed frame shift (offset stable), which slides the engage anchors by
# the measured offset so the EE targets stay exactly continuous.
_SNAP_FLOOR_M = 0.010
_SNAP_ACCEL_MAX = 25.0  # m/s², upper bound for genuine hand acceleration
_SNAP_CONFIRM_FRAMES = 8  # suspect window length (~65 ms at 120 Hz)
_SNAP_STABLE_RATIO = 0.5  # offset growth/size below this = shift, else motion

# Box-mode sticks (see IKWorker._integrate_sticks): stick deflections below
# this are ignored so a resting stick never creeps the grip width,
# and one integration step is capped so a stalled frame stream can't
# authorise a large jump.
_STICK_DEADZONE = 0.15
_STICK_MAX_DT_S = 0.1
# Box mode keeps each elbow bent at least this far forward of straight
# (rad, see IKWorker._keep_elbows_bent); ARM_JOINTS index of the elbow.
_ELBOW_BENT_RAD = 0.05
_ELBOW_INDEX = 3

# Box-mode stow (see IKWorker._start_stow). The stowed pose is planned
# once, offline, for the left arm and mirrored onto the right (the arms are
# exact mirror images: q_right = -q_left), so the stow ends symmetric from
# any start. The plan swings each elbow _STOW_ELBOW_DEG about its
# shoulder-wrist line (negative: inboard of straight down, toward the base;
# the arm/base collision model holds it off the body) over
# up to _STOW_PLAN_ITERS solves per candidate (stopping once no joint moves
# more than _STOW_PLAN_SETTLED_RAD in a solve: the angled grasp's elbow
# takes ~80 to swing in, the straight grasp's far fewer), trying box distances from
# box_stow_reach outward in _STOW_PLAN_STEP_M steps (up to _STOW_PLAN_SPAN_M
# further) and taking the first whose grippers land within
# _STOW_PLAN_POS_TOL_M of their slots with the elbows no more than
# _STOW_ELBOW_OUT_MAX_M outboard of the shoulders (a small box held right
# against the chest pushes the elbows back into the base, which spreads
# them, so it stows a little further out); failing that, the best of the
# rest (in reach first, then the tighter elbows).
_STOW_ELBOW_DEG = -30.0
_STOW_PLAN_ITERS = 200
_STOW_PLAN_MIN_ITERS = 10
_STOW_PLAN_SETTLED_RAD = 1e-3
_STOW_PLAN_STEP_M = 0.025
_STOW_PLAN_SPAN_M = 0.20
_STOW_PLAN_POS_TOL_M = 0.008
_STOW_ELBOW_OUT_MAX_M = 0.02
# The move carries the box on a straight line while the arms' posture and
# elbows blend into the planned pose; it lasts long enough for the box to
# move at box_stow_speed, turn at _STOW_TURN_RAD_S and every joint to move
# at _STOW_JOINT_RAD_S, and at least _STOW_MIN_S.
_STOW_TURN_RAD_S = math.radians(60.0)
_STOW_JOINT_RAD_S = 0.8
_STOW_MIN_S = 1.0
# The move stops short when the box is pressed against something (the
# chest, on hardware): the measured box (the midpoint of the measured
# grippers, so a squeeze on a clamped box cancels out) trails the commanded
# one (the midpoint of the solved grippers, so a target the IK can't reach
# isn't a stall) along the move by _STOW_STALL_M more than when the move
# began, and it has moved less than _STOW_STALL_STILL_M along the move in the
# last _STOW_STALL_WINDOW_S (an arm that lags at speed is still moving). The
# box then stops _STOW_BACKOFF_M short of the measured one. A move shorter
# than _STOW_STALL_MIN_MOVE_M is not watched.
_STOW_STALL_M = 0.03
_STOW_STALL_STILL_M = 0.005
_STOW_STALL_WINDOW_S = 0.5
_STOW_STALL_MIN_MOVE_M = 0.05
_STOW_BACKOFF_M = 0.01
# After the move the arms step onto the planned pose in joint space (at up to
# _STOW_SETTLE_RAD_S, see _settle_stow) and it is held until they are on it —
# no joint moves more than _STOW_STILL_RAD per solve and every joint is within
# _STOW_ARRIVED_RAD of the plan — after at least
# _STOW_HOLD_MIN_S and at most _STOW_HOLD_MAX_S. The frozen pair is never
# solved again, so a stow reported done mid-swing would keep the flare.
_STOW_STILL_RAD = 2e-3
_STOW_ARRIVED_RAD = 1e-3
_STOW_SETTLE_RAD_S = 0.3
_STOW_HOLD_MIN_S = 0.5
_STOW_HOLD_MAX_S = 3.0
# Plain teleop's stow (see IKWorker._step_arm_stow) carries the two grippers
# in as a rigid pair: the pair's midpoint is tried _ARM_STOW_REACH_M in front
# of the shoulders and then further out, in _STOW_PLAN_STEP_M steps up to
# _ARM_STOW_SPAN_M further (box mode's stow places the box centre, which the
# straight grasp holds ahead of the grippers; this places the grippers).
# A candidate counts as reached when both grippers land within
# _ARM_STOW_POS_TOL_M of their targets: a miss changes how far each runs
# into a box squeezed between them, and so the squeeze. Each height's
# search stops after _ARM_STOW_PLAN_PATIENCE candidates in a row that do no
# better (a pair held too wide to tuck the elbows in at any distance), and
# a height none reach is tried again _ARM_STOW_DROP_M lower (see
# IKWorker._plan_arm_stow).
_ARM_STOW_REACH_M = 0.10
_ARM_STOW_SPAN_M = 0.30
_ARM_STOW_POS_TOL_M = 0.003
_ARM_STOW_PLAN_PATIENCE = 3
_ARM_STOW_DROP_M = 0.05


@dataclass
class _DeckPlane:
    """An arm's hold on the height of Jelly's deck's top (see
    :meth:`IKWorker._hold_lift_plane`)."""

    # The solved gripper height (m, world) while it was on the top.
    z: float
    # The target's height then, after any raise.
    hand_z: float
    # How far the target has been raised since (m).
    raised: float = 0.0
    # When the arm last touched the deck.
    t: float = 0.0


@dataclass
class _ArmStow:
    """Plain teleop's stow in progress (see :meth:`IKWorker._step_arm_stow`)."""

    # The pair frame (midpoint between the grippers; y along the line from
    # the right gripper to the left, z as near world up as that allows) at
    # the press, and where it goes.
    start: tuple[np.ndarray, np.ndarray]
    goal: tuple[np.ndarray, np.ndarray]
    # Each gripper's pose in the pair frame, kept for the whole stow.
    rel: dict[str, tuple[np.ndarray, np.ndarray]]
    q_start: np.ndarray
    # The planned stowed joints (None without an elbow model: Mink).
    q_goal: np.ndarray | None
    elbows_start: dict[str, np.ndarray]
    elbows_goal: dict[str, np.ndarray]
    t0: float
    duration: float
    phase: str = "move"  # "move" -> "hold" -> "done"
    alpha: float = 0.0
    hold_t0: float = 0.0


@dataclass
class _Stow:
    """One box-mode stow in progress (see :meth:`IKWorker._start_stow`)."""

    start: tuple[np.ndarray, np.ndarray]
    goal: tuple[np.ndarray, np.ndarray]
    q_start: np.ndarray
    # The planned stowed joints (None once a stall cut the move short).
    q_goal: np.ndarray | None
    elbows_start: dict[str, np.ndarray]
    elbows_goal: dict[str, np.ndarray]
    t0: float
    duration: float
    phase: str = "move"  # "move" -> "hold" -> "done"
    alpha: float = 0.0
    hold_t0: float = 0.0
    stall_base: float | None = None
    # The measured box's progress along the move when it began.
    stall_p0: float = 0.0
    # (time, measured box progress along the move) over the last
    # _STOW_STALL_WINDOW_S of the move.
    stall_trace: list[tuple[float, float]] = field(default_factory=list)


# Squeeze lean (see IKWorker._squeeze_lean): low-pass time constant on the
# measured clamp depth. The depth is a difference of two positions, so it
# carries the encoders' noise and the servo lag of a move; a fifth of a
# second takes the flicker out and still follows a width jog as it lands.
_LEAN_TAU_S = 0.2
# A gap between depth samples longer than this restarts the filter.
_LEAN_RESET_S = 0.5
# Squeeze trim (see _squeeze_lean): integral gain (1/s) from the measured
# toe-out of the tips to the extra inward yaw of the targets — an error of
# 1° adds 1°/s — and the time constant (s) the trim bleeds away with once
# the grippers stop pressing.
_TRIM_GAIN = 1.0
_TRIM_DECAY_S = 1.0
# The trims are per arm, so a turn's servo lag would read as toe-out on
# one side and toe-in on the other: they only integrate while the pair's
# commanded motion has been under these rates (rad/s about up, m/s at the
# centre) for _TRIM_STILL_S. Between turns the lag has settled by then.
_TRIM_STILL_YAW_RATE = math.radians(3.0)
_TRIM_STILL_SPEED = 0.03
_TRIM_STILL_S = 0.3
# While the grippers press, a line of clamp diagnostics this often (s).
_CLAMP_LOG_PERIOD_S = 1.0

# Gripper-pair status (see IKWorker.pair_status): reported to the core every
# this many solved frames (~10 Hz at the 120 Hz cadence), and the tolerance
# (per gripper, from the rotation a box-mode engage would blend it to) for
# calling the two grippers "aligned" into the side-clamping pair.
_STATUS_EVERY_N = 12
_ALIGNED_TOL_DEG = 25.0

# Collision proximity for the headset's haptics (see
# IKWorker.collision_proximity): reported every this many solved frames
# (~30 Hz at the 120 Hz cadence) while either arm is engaged.
_PROXIMITY_EVERY_N = 4

# Re-engage ramp (config.reengage == "ramp", see IKWorker.step): the blend from
# the arm's pose at the grip to the controller-implied target is paced by the
# configured linear speed and, for the rotational part, by this angular rate.
_RAMP_ANG_SPEED = 0.6  # rad/s

# Lift clearance on the mobile Axol (see IKWorker.clear_lift): the fastest
# (m/s) an arm that isn't tracking is moved out of the way of Jelly's deck as
# the lift lowers. The deck rises at twice the lift's speed.
_LIFT_CLEAR_SPEED = 0.2
# An engaged arm held up by Jelly's deck keeps its height (see
# IKWorker._hold_lift_plane): the smallest height (m) its target is raised
# by (below it the gap is the solve's own lag), and how far up (the
# vertical component of the contact normal) the deck must push the arm
# for it to count. The lower column's sides push sideways and don't.
_LIFT_PLANE_MIN_M = 0.002
_LIFT_PLANE_MIN_UP = 0.2
# An arm held on the deck's top (a contact pushing at least _LIFT_PLANE_TOP_UP
# up) remembers that height. Off the top (round the edge and past it) the
# deck no longer holds it up, and the solve's pull toward its posture would
# let it sag below its target (a dip of up to ~1.5 cm in sim). Its target
# is raised by _LIFT_PLANE_GAIN of the sag each solve until it is back at
# the remembered height (and, once clear, lowered again by as much of that
# raise as left it above it). The hold lets go once the hand moves up or down
# _LIFT_PLANE_RELEASE_M from where it was when the arm left the top, or
# _LIFT_PLANE_HOLD_S after the arm last touched the deck.
_LIFT_PLANE_TOP_UP = 0.98
_LIFT_PLANE_GAIN = 0.2
_LIFT_PLANE_RELEASE_M = 0.01
_LIFT_PLANE_HOLD_S = 1.0

# ---------------------------------------------------------------------------
# NumPy-only helpers (no JAX dispatch overhead)
# ---------------------------------------------------------------------------


def _dz(v: float) -> float:
    """A thumbstick axis with the deadzone applied (``0.0`` when resting)."""
    return 0.0 if abs(v) < _STICK_DEADZONE else float(v)


def _pair_frame(
    left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """The frame of a gripper pair: ``(center, rot)`` (float64).

    Centred between the two grippers, its y axis along the line from the
    right gripper to the left, its z axis as near world up as that allows
    and x = y × z (forward, for a pair held in front). The plain-teleop stow
    (:meth:`IKWorker._step_arm_stow`) carries the grippers in this frame.
    """
    pl = np.asarray(left[0], np.float64)
    pr = np.asarray(right[0], np.float64)
    center = 0.5 * (pl + pr)
    y = pl - pr
    norm = float(np.linalg.norm(y))
    y = y / norm if norm > 1e-6 else np.array((0.0, 1.0, 0.0))
    up = np.array((0.0, 0.0, 1.0)) - y[2] * y
    if float(np.linalg.norm(up)) < 1e-6:  # the pair stands vertically
        up = np.array((1.0, 0.0, 0.0)) - y[0] * y
    up = up / float(np.linalg.norm(up))
    x = np.cross(y, up)
    return center, np.column_stack((x, y, up))


def _pair_targets(
    frame: tuple[np.ndarray, np.ndarray],
    rel: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Pose]:
    """Each gripper's world pose for the pair ``frame`` and its ``rel`` pose."""
    center = np.asarray(frame[0], np.float64)
    rot = np.asarray(frame[1], np.float64)
    return {
        side: (
            (center + rot @ np.asarray(pos, np.float64)).astype(np.float32),
            (rot @ np.asarray(r, np.float64)).astype(np.float32),
        )
        for side, (pos, r) in rel.items()
    }


# Clearance (m) kept between the two grippers' bodies when the width is
# closed with nothing between them (see ToolGeometry.min_width). The stock
# gripper's bodies are its wrists. The parcel gripper's are its open plates
# (straight grasp), which may close onto each other.
_BODY_CLEARANCE_M = 0.01
_PARCEL_BODY_CLEARANCE_M = 0.0
# Longest step (m) along its swivel an elbow hint asks for (swivel_step): the
# swivel is a curve, and a longer step's chord leaves it.
_ELBOW_HINT_STEP_M = 0.03
# Box mode's (position, orientation) IK weight multipliers. A clamp needs each
# gripper's yaw to a fraction of a degree — half a degree tip-out holds the
# angled grasp's tip a millimetre off the box — and at the arm weights the
# manipulability cost and the elbow hint settle against a pose error that
# size (0.3-0.5° tip-out in the angled grasp); these hold it under 0.05°.
_BOX_POSE_WEIGHT_SCALE = (2.0, 4.0)


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


# Controller axes → box-frame axes for box mode's turn (see _box_turn). The
# controller rotations out of ``_vr_to_flu_np`` hold the controller's own
# axes as (back, up, left); the box frame is (forward, left, up).
_CTRL_TO_BOX = np.array(
    ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)), dtype=np.float32
)


def _box_turn(
    rot_curr: np.ndarray, rot_snap_ctrl: np.ndarray, rotation_multiplier: float = 1.0
) -> np.ndarray:
    """The controller's turn since its snap, in the box frame's own axes.

    The controller-local rotation delta (what :func:`_relative_target_np`
    reads too), re-expressed as if the controller had been held like the
    box frame at the snap — pointing forward, upright — so turning the hand
    about its own vertical yaws the pair, about its left/right axis pitches
    it and about its pointing axis rolls it, wherever the operator stands
    and however the grippers were posed at the engage. Apply it on the
    right of the box rotation (``box.rot @ turn``).
    ``rotation_multiplier`` scales the angle.
    """
    local = np.asarray(rot_snap_ctrl, np.float32).T @ np.asarray(rot_curr, np.float32)
    turn = _CTRL_TO_BOX @ local @ _CTRL_TO_BOX.T
    return _scale_rotation_np(turn, rotation_multiplier).astype(np.float32)


# ---------------------------------------------------------------------------
# IKWorker
# ---------------------------------------------------------------------------


class IKWorker:
    """Self-contained IK controller for the subprocess.

    Snap state is numpy-only. The selected solver runs through ``solver.ik``
    inside :meth:`step`.
    """

    def __init__(
        self, config: VRTeleopConfig, kinematics_config: KinematicsConfig
    ) -> None:
        """Construct the IK worker.

        Instantiates the selected solver and initialises pose filters for VR streams.

        Args:
            config:            Teleop session parameters (rest poses, frequency, filter settings).
            kinematics_config: IK solver cost weights forwarded to :class:`KinematicsSolver`.
        """
        self._config = config
        self._mink_backend = kinematics_config.backend == "mink"
        if self._mink_backend:
            from ..kinematics.mink_backend import MinkKinematicsSolver

            self._solver = MinkKinematicsSolver(
                kinematics_config, solve_hz=config.ik_frequency
            )
        elif kinematics_config.backend == "jax":
            self._solver = _make_jax_solver(kinematics_config)
        else:
            raise ValueError("kinematics backend must be 'jax' or 'mink'")
        # Elbow hints are optional (kinematics.elbow_weight == 0 disables, the
        # default): skip the whole elbow pipeline — filters, engage snapshots,
        # target math — so the solve graph never carries the cost.
        # Mink has no elbow task, so it never consumes them.
        self._use_elbow = (
            kinematics_config.elbow_weight > 0.0 and not self._mink_backend
        )

        self._rest_pose_left = np.asarray(config.rest_pose_left, dtype=np.float32)
        self._rest_pose_right = np.asarray(config.rest_pose_right, dtype=np.float32)

        self._solver.set_posture_pose(self.get_rest_q())

        # Per-arm engage state: an arm is *active* while its (core-synthesized)
        # lock is held and tracks the controller; an inactive arm in an
        # otherwise-engaged session is *frozen* — held at the pose it had when
        # its lock dropped (see ``_hold_fk`` / ``_hold_elbow_fk``).
        self._active: dict[str, bool] = {"left": False, "right": False}
        self._hold_fk: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._hold_elbow_fk: dict[str, np.ndarray] = {}
        # Per-arm freeze state. A bimanual solve can leave one arm's joint slice
        # bit-identical while the other progresses, so whole-vector detection
        # both misses real freezes and cannot clutch only the affected mapping.
        # Each target snapshot is (EE position, EE rotation, optional elbow).
        self._freeze_since: dict[str, float] = {}
        self._freeze_targets: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray | None]
        ] = {}
        # Snap poses as (pos_3, rot_3x3) numpy tuples — no jaxlie overhead
        self._snap_ctrl: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._snap_fk: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._snap_elbow_ctrl: dict[str, np.ndarray] = {}
        self._snap_elbow_fk: dict[str, np.ndarray] = {}
        # Re-engage ramp (config.reengage == "ramp"): per arm, the EE pose it
        # had at its grip edge plus (t0, duration) of the blend from there to
        # the target implied by its *kept* snap. Absent while not ramping.
        self._ramp: dict[str, tuple[tuple[np.ndarray, np.ndarray], float, float]] = {}

        # Absolute (Mantis) mode state: the world-anchored base transform solved
        # at engage — ``(R_wb, t_wb)`` maps base-frame FLU coordinates into the
        # raw VR world frame — plus each controller's rigid controller→TCP
        # offset ``(p_off, R_off)`` expressed in the controller's local frame.
        # ``_abs_active`` is the whole-session engage toggle (absolute mode
        # has no per-arm freeze — both grips engage, both release). Seeded
        # here, not only in reset(): the first VR frame can arrive before any
        # reset or engage, and the absolute-mode reply reads this state.
        self._abs_active: bool = False
        self._abs_base: tuple[np.ndarray, np.ndarray] | None = None
        self._abs_offset: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        # Tracker→gripper transforms (the rig's factory design constants, or
        # per-unit file overrides — see almond_axol.mantis.calibration), per
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
        # tracked ground-truth pose the IK solver chases — Mantis data collection
        # records it per row so training can use raw TCP trajectories instead
        # of (or alongside) the IK joint solutions. Holds the last engaged
        # target while disengaged (mirroring the latched virtual joints);
        # seeded from rest FK so it is never ``None`` in absolute mode.
        self.last_tcp_msg: dict[str, list[float]] | None = None

        # Tracking glitch detection state (see _frame_snap_verdict): last good
        # raw controller positions, their (effective) timestamp, an EMA
        # velocity per hand, and the in-progress suspect window, if any.
        self._prev_raw: dict[str, np.ndarray] = {}
        self._prev_raw_t: float | None = None
        self._raw_vel: dict[str, np.ndarray] = {}
        self._suspect: dict | None = None
        # Wall time of the previous solve, for scaling the solver's per-call
        # step clamp by the actual solve cadence (see delta_scale in step()).
        self._last_solve_t: float | None = None
        # Lift clearance (see clear_lift): time of the previous step, and the
        # lift-safe rest goal with the lift height it was computed for.
        self._lift_clear_t: float | None = None
        # Per engaged arm held on Jelly's deck's top: the height it is held
        # at (see _hold_lift_plane).
        self._deck_plane: dict[str, _DeckPlane] = {}
        self._rest_goal: tuple[float, np.ndarray] | None = None
        # Box mode (bimanual carry, see .box): the pair's state from the engage
        # snap and which controller leads it. None while not in box tracking.
        self._box: BoxState | None = None
        # The grasp the pair still holds the box in after a grasp switch,
        # until the re-snap reads the box off it (see set_config).
        self._held_tool: ToolGeometry | None = None
        self._box_leader: str | None = None
        # The leader's anchor for the pair — its controller pose and its
        # gripper's FK at the box engage — kept apart from the per-arm snaps
        # above, which stay the session's controller↔arm mapping: leaving
        # box mode in "ramp" re-engage blends *both* arms back out to that
        # mapping, the one the operator engaged with before box mode.
        self._box_snap: (
            tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None
        ) = None
        # The pair's box rotation as last commanded (the snap's turned by the
        # leader, so possibly tipped). Kept across a freeze, a handover and a
        # grasp switch so their re-snap holds a tipped box at its tilt
        # (snap_box's ``rot``) instead of levelling it; dropped once the
        # session leaves box tracking or the arms are moved by something else.
        self._box_rot: np.ndarray | None = None
        # The box-mode stow in progress, if any (see _start_stow).
        self._stow: _Stow | None = None
        # Plain teleop's stow in progress, if any (see _step_arm_stow).
        self._arm_stow: _ArmStow | None = None
        # Squeeze lean (see _squeeze_lean): the arms' measured joint positions
        # and joint stiffness as last reported by the core (``("meas", ...)``,
        # hardware only — the sim reports none), the low-passed clamp depth
        # (m) with the time of its last sample, and the clamp force (N) the
        # last solve leaned for (0 while not pressing), for pair_status.
        self._measured: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = (
            None
        )
        self._measured_t = 0.0  # perf_counter of the last report
        self._lean_depth = 0.0
        self._lean_t: float | None = None
        self._lean_force = 0.0
        # Squeeze trim: per arm, the extra inward yaw (rad) of its gripper
        # target the measured toe-out of its tip has integrated to, see
        # _squeeze_lean; the previous frame's targets (for the pair's
        # commanded motion) and when the pair last moved; when the clamp
        # diagnostics were last logged.
        self._lean_trim: dict[str, float] = {"left": 0.0, "right": 0.0}
        self._trim_prev: tuple[float, dict[str, Pose]] | None = None
        self._trim_moving_t: float = -math.inf
        self._clamp_log_t: float = -math.inf
        # The arm model the lean reads its Jacobians from (the MuJoCo model
        # gravity compensation runs on); built up front so the first clamp
        # doesn't stall a solve. None if it can't be built (no lean then).
        self._lean_model = None
        try:
            from ..robot.gravity import GravityCompensator

            self._lean_model = GravityCompensator()
        except Exception:  # noqa: BLE001 - the lean is optional
            _logger.exception("arm model unavailable; box mode's squeeze lean is off")

        # Pose-stream smoothing (see LagCompensatedLowPass for why this is a
        # linear filter and not OneEuro). Nominal rate is the VR-frame / IK
        # dispatch cadence, not the (faster) CAN control rate.
        freq = config.ik_frequency
        fc = config.pose_cutoff
        self._f_l_pos = LagCompensatedLowPass(freq, fc)
        self._f_l_quat = LagCompensatedLowPass(freq, fc)
        self._f_r_pos = LagCompensatedLowPass(freq, fc)
        self._f_r_quat = LagCompensatedLowPass(freq, fc)
        self._f_l_elbow = LagCompensatedLowPass(freq, fc)
        self._f_r_elbow = LagCompensatedLowPass(freq, fc)

        # Pre-settle the configured rest pose to the manipulability-balanced
        # IK fixed point. The configured pose has a non-zero manipulability
        # gradient, so a first engage there walks q in the EE null space
        # toward higher manipulability over the next ~10-30 frames. Baking the
        # settling in at startup means the trajectory ends at the fixed point
        # and the first engage produces no motion.
        # Mink has no manipulability objective; keep the configured policy
        # rest posture unchanged across policy/operator handovers.
        q_settled = (
            self.get_rest_q() if self._mink_backend else self._settle_rest_pose()
        )
        self._rest_pose_left = q_settled[self._solver.left_indices].astype(np.float32)
        self._rest_pose_right = q_settled[self._solver.right_indices].astype(np.float32)
        self._solver.set_posture_pose(self.get_rest_q())
        # The haptics' collision proximity reads the rest pose as clear.
        set_reference = getattr(self._solver, "set_proximity_reference", None)
        if callable(set_reference):
            set_reference(self.get_rest_q())

        # Teleop flight recorder (--teleop.record, arriving here via
        # the pickled config, see .recorder): taps the solve path at every
        # stage boundary this process owns — raw VR pose, filtered pose,
        # world EE target, IK output.
        n = self._solver.num_joints
        self._rec = _recorder_make(
            config.record,
            "ik",
            {
                "raw_l": 3,
                "raw_r": 3,
                "filt_l": 3,
                "filt_r": 3,
                "tgt_l": 3,
                "tgt_r": 3,
                "q": n,
                "engaged": 2,
                "solve_ms": 1,
            },
        )

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
        """Process one VRFrame. Returns updated full (N,) q in radians.

        ``frame.l_lock`` / ``frame.r_lock`` carry the core's per-arm engage
        state: a locked arm tracks its controller, an unlocked arm in an
        otherwise-engaged frame is frozen at the pose it had when its lock
        dropped. A frame with neither lock leaves ``q_current`` untouched.
        """
        if self._config.absolute_mode:
            return self._step_absolute(frame, q_current)

        l_lock = bool(frame.l_lock)
        r_lock = bool(frame.r_lock)

        # Flight recorder covers engaged segments only: the falling edge
        # writes the _ik file, the rising edge starts a fresh segment.
        if self._rec is not None:
            self._rec.set_engaged(l_lock or r_lock)

        # Filter raw VR poses on *every* frame — engaged or not — so the
        # filters are always warm. They used to run only while engaged and be
        # reset on the engage rising edge, which fixed stale-state sweeps but
        # made every engage a cold start: a fresh pose filter's velocity
        # estimate is zero, so its lag-compensation feedforward is absent
        # for the first few hundred ms and moving immediately after
        # engaging felt heavily over-smoothed. Continuous filtering keeps
        # the state fresh (no stale sweep) and the velocity estimate already
        # tracking hand motion at the engage snap (no cold start).
        #
        # ``t`` is the frame's playout/capture stamp: frames reach this worker
        # at the irregular solve cadence, and timestamped updates keep that
        # timing jitter from being read as velocity jitter.
        t_s = (frame.t / 1000.0) if frame.t is not None else None
        raw_l_pos = np.array(
            [frame.l_ee.position.x, frame.l_ee.position.y, frame.l_ee.position.z]
        )
        raw_r_pos = np.array(
            [frame.r_ee.position.x, frame.r_ee.position.y, frame.r_ee.position.z]
        )
        raw_l_quat = np.array(
            [
                frame.l_ee.quaternion.x,
                frame.l_ee.quaternion.y,
                frame.l_ee.quaternion.z,
                frame.l_ee.quaternion.w,
            ]
        )
        raw_r_quat = np.array(
            [
                frame.r_ee.quaternion.x,
                frame.r_ee.quaternion.y,
                frame.r_ee.quaternion.z,
                frame.r_ee.quaternion.w,
            ]
        )

        verdict, off_l, off_r = self._frame_snap_verdict(raw_l_pos, raw_r_pos, t_s)
        if verdict == "hold":
            # Suspect frame: quarantine it (filters never see it) and hold the
            # arm until the window resolves — a few tens of ms at most.
            return q_current
        if verdict == "shift":
            # Confirmed world-frame shift: the hands didn't move, the VR world
            # did. Nudge each position filter's state by the measured offset
            # (its motion history is still valid — only the reference frame
            # moved) and slide each engaged anchor by the same offset. Target
            # math sees (filtered - anchor), so the EE targets stay *exactly*
            # continuous: no step, no filter cold start. Re-snapping against
            # FK instead would discard the servo lag (up to 60 mm during
            # motion, measured) and yank the target by that much. Any
            # rotational component of the shift is left to the quaternion
            # filters to absorb gradually (observed shifts are translation-
            # dominated).
            assert off_l is not None and off_r is not None
            self._f_l_pos.nudge(off_l)
            self._f_r_pos.nudge(off_r)
            if self._use_elbow:
                self._f_l_elbow.nudge(off_l)
                self._f_r_elbow.nudge(off_r)
            for side, off in (("left", off_l), ("right", off_r)):
                # VR (X=Down, Y=Left, Z=Forward) -> robot FLU, as in _vr_to_flu_np.
                delta = np.array((off[2], off[1], -off[0]), dtype=np.float32)
                if side in self._snap_ctrl:
                    pos, rot = self._snap_ctrl[side]
                    self._snap_ctrl[side] = (pos + delta, rot)
                if self._box_snap is not None and side == self._box_leader:
                    (pos, rot), fk = self._box_snap
                    self._box_snap = ((pos + delta, rot), fk)
                if self._snap_elbow_ctrl.get(side) is not None:
                    self._snap_elbow_ctrl[side] = self._snap_elbow_ctrl[side] + delta

        lp = self._f_l_pos.update(raw_l_pos, t=t_s)
        lq = self._f_l_quat.update(raw_l_quat, t=t_s)
        lq = lq / np.linalg.norm(lq)

        rp = self._f_r_pos.update(raw_r_pos, t=t_s)
        rq = self._f_r_quat.update(raw_r_quat, t=t_s)
        rq = rq / np.linalg.norm(rq)

        left_pos, left_rot = _vr_to_flu_np(*lp, *lq)
        right_pos, right_rot = _vr_to_flu_np(*rp, *rq)

        left_e: np.ndarray | None = None
        right_e: np.ndarray | None = None
        if self._use_elbow:
            le = self._f_l_elbow.update(
                np.array([frame.l_elbow.x, frame.l_elbow.y, frame.l_elbow.z]), t=t_s
            )
            re = self._f_r_elbow.update(
                np.array([frame.r_elbow.x, frame.r_elbow.y, frame.r_elbow.z]), t=t_s
            )
            left_e = np.array((le[2], le[1], -le[0]), dtype=np.float32)
            right_e = np.array((re[2], re[1], -re[0]), dtype=np.float32)

        if getattr(frame, "arm_stow", False) and frame.box_leader is None:
            return self._step_arm_stow(q_current)
        if getattr(self, "_arm_stow", None) is not None:
            # Done (the core froze both arms) or cancelled (a grip).
            self._end_arm_stow(q_current)

        if not (l_lock or r_lock):
            # Snap poses are deliberately kept: in "ramp" re-engage mode they
            # are the session anchor the next grip ramps the arm back to
            # (reset() / clear_engage() drop them; "clutch" mode re-snaps).
            self._active = {"left": False, "right": False}
            self._hold_fk = {}
            self._hold_elbow_fk = {}
            self._ramp = {}
            self._clear_freeze()
            self._box = None
            self._box_leader = None
            self._box_snap = None
            self._stow = None
            return self.clear_lift(q_current)

        if frame.box_leader is not None:
            return self._step_box(
                frame,
                q_current,
                {"left": (left_pos, left_rot), "right": (right_pos, right_rot)},
                lp,
                rp,
            )
        self._box_rot = None
        if self._box is not None:
            # Box tracking ended without a lock-less frame in between (mode
            # switched while engaged): the per-arm path below re-engages each
            # arm — in "ramp" mode against the session snaps box mode left
            # untouched, so both arms blend back out to the mapping the
            # operator had before the box; otherwise with a fresh snap.
            self._box = None
            self._box_leader = None
            self._box_snap = None
            self._active = {"left": False, "right": False}
            self._hold_fk = {}
            self._hold_elbow_fk = {}
            self._ramp = {}

        was_any = self._active["left"] or self._active["right"]
        if not was_any:
            self._clear_freeze()

        pos_mult = self._config.position_multiplier
        rot_mult = self._config.rotation_multiplier
        now = time.perf_counter()
        # Re-engage behaviour: "clutch" re-snaps the controller against the
        # arm's current pose (the arm stays; the controller "matches the
        # arm"), "ramp" keeps the arm's existing snap and blends the arm out
        # to where that mapping says the controller now is (the arm "matches
        # the controller"). The live mode rides on the frame (core-forwarded
        # HUD toggle); the config is the fallback.
        ramp_mode = (frame.reengage or self._config.reengage) == "ramp"

        # Per-arm activation. FK of q_current is needed to snapshot a rising
        # arm's EE pose and to capture a freezing/frozen arm's hold pose;
        # compute each (lazily) at most once per step.
        ee_fk: (
            tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None
        ) = None
        elbow_fk: tuple[np.ndarray, np.ndarray] | None = None

        def _ee(side: str) -> tuple[np.ndarray, np.ndarray]:
            nonlocal ee_fk
            if ee_fk is None:
                ee_fk = self._solver.fk(q_current)
            return ee_fk[0] if side == "left" else ee_fk[1]

        def _elbow(side: str) -> np.ndarray:
            nonlocal elbow_fk
            if elbow_fk is None:
                elbow_fk = self._solver.elbow_positions(q_current)
            return elbow_fk[0] if side == "left" else elbow_fk[1]

        snapped: list[list[int]] = []
        for side, lock, ctrl_pos, ctrl_rot, ctrl_e, indices in (
            ("left", l_lock, left_pos, left_rot, left_e, self._solver.left_indices),
            (
                "right",
                r_lock,
                right_pos,
                right_rot,
                right_e,
                self._solver.right_indices,
            ),
        ):
            if lock:
                if not self._active[side]:
                    self._active[side] = True
                    self._hold_fk.pop(side, None)
                    self._hold_elbow_fk.pop(side, None)
                    if ramp_mode and side in self._snap_ctrl:
                        self._start_ramp(side, ctrl_pos, ctrl_rot, _ee(side), now)
                    else:
                        self._snap_arm(
                            side,
                            ctrl_pos,
                            ctrl_rot,
                            ctrl_e,
                            _ee(side),
                            _elbow(side) if self._use_elbow else None,
                        )
                    snapped.append(indices)
            else:
                if self._active[side]:
                    self._active[side] = False
                if side not in self._hold_fk:
                    self._hold_fk[side] = _ee(side)
                    if self._use_elbow:
                        self._hold_elbow_fk[side] = _elbow(side)

        if snapped:
            # Pin posture to ``q_current`` so the held pose is itself the IK
            # fixed point (the rest-pose attractor would otherwise pull q in
            # the EE null space at every frame, growing with distance from
            # rest; reset() restores it). Re-pinned on *every* engage snap,
            # not just the first out of a full disengage: a single arm
            # re-engaging mid-session is no longer pinned to its seed, so a
            # posture pose left at the previous engage would drag it through
            # the null space — a visible settle over the first frames even
            # with a still controller.
            #
            # Only the snapping arm's joint slice is re-pinned. The other arm
            # may still be tracking, balanced between its EE target and the
            # posture pull toward wherever *its* slice was last pinned; moving
            # that pin to its current q drops the pull instantly and the arm
            # relaxes to the pure EE/manipulability solution on the next
            # solve — a visible twitch on the tracking arm every time the
            # frozen one was re-engaged, growing with how far it had travelled
            # since its own pin.
            posture = self._solver.posture_pose
            for indices in snapped:
                posture[indices] = q_current[indices]
            if not getattr(self, "_mink_backend", False):
                self._solver.set_posture_pose(posture)
            # An engage snap re-anchors that arm to q_current: return the
            # seed unchanged so the snap frame itself produces no motion
            # (matching the previous whole-session engage behaviour). A ramp
            # engage likewise starts from q_current (blend alpha 0).
            self._clear_freeze()
            return q_current

        def _target(
            side: str, ctrl_pos: np.ndarray, ctrl_rot: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            if not self._active[side]:
                return self._hold_fk[side]
            goal = _relative_target_np(
                ctrl_pos,
                ctrl_rot,
                *self._snap_ctrl[side],
                *self._snap_fk[side],
                position_multiplier=pos_mult,
                rotation_multiplier=rot_mult,
            )
            ramp = self._ramp.get(side)
            if ramp is None:
                return goal
            start, t0, duration = ramp
            alpha = smoothstep((now - t0) / duration)
            if alpha >= 1.0:
                del self._ramp[side]
                return goal
            # The goal is live: the hand may keep moving during the blend and
            # the arm converges on wherever it ends up, never on a stale pose.
            return blend_pose(start, goal, alpha)

        tl_pos, tl_rot = _target("left", left_pos, left_rot)
        tr_pos, tr_rot = _target("right", right_pos, right_rot)

        elbow_l: np.ndarray | None = None
        elbow_r: np.ndarray | None = None
        if self._use_elbow:
            elbow_l = (
                self._snap_elbow_fk["left"]
                + pos_mult * (left_e - self._snap_elbow_ctrl["left"])
                if self._active["left"]
                else self._hold_elbow_fk["left"]
            )
            elbow_r = (
                self._snap_elbow_fk["right"]
                + pos_mult * (right_e - self._snap_elbow_ctrl["right"])
                if self._active["right"]
                else self._hold_elbow_fk["right"]
            )

        # The solver's max_joint_delta is a per-call clamp — an implicit
        # velocity limit at the nominal cadence. Scale it by the actual time
        # since the last solve so a slow solve (fast motion, contended CPU)
        # doesn't silently strangle joint speed: with the clamp fixed, a
        # 30 ms solve capped joints at ~1.1 rad/s, the target fell behind
        # the hand, and the backlog released as a lurch — the "random
        # jitter" bursts seen during fast wrist rotations. Capped at 4x so
        # a multi-second stall can't authorize a giant step.
        delta_scale = 1.0
        if self._last_solve_t is not None:
            elapsed = now - self._last_solve_t
            delta_scale = float(np.clip(elapsed * self._config.ik_frequency, 1.0, 4.0))
        self._last_solve_t = now

        solve_t0 = time.perf_counter()
        solve_options = (
            {
                "active_sides": tuple(
                    side for side, active in self._active.items() if active
                )
            }
            if getattr(self, "_mink_backend", False)
            else {}
        )
        q_new = self._solver.ik(
            q_current,
            left_pose=(tl_pos, tl_rot),
            right_pose=(tr_pos, tr_rot),
            left_elbow_pos=elbow_l,
            right_elbow_pos=elbow_r,
            delta_scale=delta_scale,
            **solve_options,
        )
        solve_ms = (time.perf_counter() - solve_t0) * 1000.0
        # A frozen arm must not move at all: the hold-pose target keeps the
        # solve consistent (collision terms see the true pose), but the
        # joints themselves are pinned to the seed.
        q_new = np.asarray(q_new, dtype=np.float32).copy()
        if not self._active["left"]:
            q_new[self._solver.left_indices] = q_current[self._solver.left_indices]
        if not self._active["right"]:
            q_new[self._solver.right_indices] = q_current[self._solver.right_indices]
        # Detect and resolve seed-return stalls per arm. Re-snapshot only a
        # stalled arm whose target actually moved: a still controller can
        # legitimately sit at a collision or joint boundary, and the healthy
        # arm must retain both its solved output and its controller mapping.
        #
        # Re-anchoring uses the *current filtered* controller pose and FK of the
        # unchanged joint slice. At this sample _relative_target_np therefore
        # returns that FK exactly (zero translation, identity rotation), so the
        # clutch cannot introduce a command step. Motion after this sample is
        # once again relative 1:1 (or with the configured multipliers); motion
        # accumulated while the solver was unable to move is deliberately
        # discarded instead of being released later as a lurch.
        reanchor: list[
            tuple[
                str,
                np.ndarray,
                np.ndarray,
                np.ndarray | None,
                list[int],
            ]
        ] = []
        for (
            side,
            ctrl_pos,
            ctrl_rot,
            ctrl_e,
            target_pos,
            target_rot,
            target_e,
            indices,
        ) in (
            (
                "left",
                left_pos,
                left_rot,
                left_e,
                tl_pos,
                tl_rot,
                elbow_l,
                self._solver.left_indices,
            ),
            (
                "right",
                right_pos,
                right_rot,
                right_e,
                tr_pos,
                tr_rot,
                elbow_r,
                self._solver.right_indices,
            ),
        ):
            if not self._active[side]:
                self._clear_freeze(side)
                continue
            arm_frozen = bool(np.array_equal(q_new[indices], q_current[indices]))
            if self._note_solve(
                side,
                arm_frozen,
                target_pos,
                target_rot,
                target_e,
            ):
                reanchor.append((side, ctrl_pos, ctrl_rot, ctrl_e, indices))

        if reanchor:
            # Keep the persistent posture attractor consistent with the new
            # clutch origin, but update only re-anchored joint slices. Re-pinning
            # the whole vector would unnecessarily perturb a healthy arm that
            # made progress in this same bimanual solve.
            posture = self._solver.posture_pose
            for side, ctrl_pos, ctrl_rot, ctrl_e, indices in reanchor:
                self._snap_arm(
                    side,
                    ctrl_pos,
                    ctrl_rot,
                    ctrl_e,
                    _ee(side),
                    _elbow(side) if self._use_elbow else None,
                )
                posture[indices] = q_current[indices]
            if not getattr(self, "_mink_backend", False):
                self._solver.set_posture_pose(posture)
        if self._has_lift_contact():
            # An arm re-anchored above already has a fresh mapping.
            fresh = {side for side, *_rest in reanchor}
            q_new = self._lift_contact_per_arm(
                q_new,
                {
                    side: pose
                    for side, pose in (
                        ("left", (tl_pos, tl_rot)),
                        ("right", (tr_pos, tr_rot)),
                    )
                    if side not in fresh
                },
            )
        if self._rec is not None:
            self._rec.record(
                raw_l=np.array(
                    [
                        frame.l_ee.position.x,
                        frame.l_ee.position.y,
                        frame.l_ee.position.z,
                    ]
                ),
                raw_r=np.array(
                    [
                        frame.r_ee.position.x,
                        frame.r_ee.position.y,
                        frame.r_ee.position.z,
                    ]
                ),
                filt_l=lp,
                filt_r=rp,
                tgt_l=tl_pos,
                tgt_r=tr_pos,
                q=q_new,
                engaged=np.array(
                    [float(self._active["left"]), float(self._active["right"])]
                ),
                solve_ms=solve_ms,
            )
        return q_new

    def _step_absolute(self, frame: VRFrame, q_current: np.ndarray) -> np.ndarray:
        """Mantis step: absolute world-anchored targets, no deltas.

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
            self._abs_active = False
            if isinstance(self.abs_base_msg, dict) and "rejected" in self.abs_base_msg:
                # The core has seen the rejection (it disengaged in response);
                # stop repeating it.
                self.abs_base_msg = None
            return q_current

        if not self._abs_active:
            # Same rationale as the relative path: the pose-filter state froze
            # at the pose held when tracking was last disabled.
            self._reset_pose_filters()

        # Timestamped filter updates, as in the relative path: frames arrive at
        # the irregular solve cadence and the stamp keeps that jitter from
        # being read as hand velocity.
        t_s = (frame.t / 1000.0) if frame.t is not None else None

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
        lp = self._f_l_pos.update(lp_raw, t=t_s)
        lq = self._f_l_quat.update(lq_raw, t=t_s)
        lq = lq / np.linalg.norm(lq)
        rp = self._f_r_pos.update(rp_raw, t=t_s)
        rq = self._f_r_quat.update(rq_raw, t=t_s)
        rq = rq / np.linalg.norm(rq)

        l_pos, l_rot = (
            lp.astype(np.float64),
            _quat_xyzw_to_matrix(*lq).astype(np.float64),
        )
        r_pos, r_rot = (
            rp.astype(np.float64),
            _quat_xyzw_to_matrix(*rq).astype(np.float64),
        )

        if not self._abs_active:
            self._abs_active = True
            if not self._engage_absolute(l_pos, l_rot, r_pos, r_rot):
                # Rejected (rigs in the wrong hands): hold the current joints;
                # the core disengages when it sees the rejection reply.
                return q_current
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
        self.last_tcp_msg = self._encode_tcp_msg(
            left_target,
            right_target,
            out_of_reach=self._out_of_reach(left_target, right_target),
        )
        return self._solver.ik(
            q_current,
            left_pose=left_target,
            right_pose=right_target,
        )

    def _out_of_reach(
        self,
        left_target: tuple[np.ndarray, np.ndarray],
        right_target: tuple[np.ndarray, np.ndarray],
    ) -> tuple[str, ...]:
        """Sides whose absolute target lies beyond the arm's reach soft-clamp.

        On the Mantis rig the recorded pose is the tracked hand, not the
        (clamped) IK target, so a hand carried past ``reach_soft_start`` from
        the shoulder records a pose the robot cannot follow on replay. Mantis
        data collection counts these rows per episode and warns.
        """
        limit = float(self._solver.config.reach_soft_start)
        shoulders = self._solver.shoulder_positions
        sides: list[str] = []
        for side, (pos, _rot) in (("left", left_target), ("right", right_target)):
            dist = float(
                np.linalg.norm(np.asarray(pos, dtype=np.float64) - shoulders[side])
            )
            if dist > limit:
                sides.append(side)
        return tuple(sides)

    def compute_reset_trajectory(
        self, q_current: np.ndarray, q_target: np.ndarray
    ) -> list[np.ndarray]:
        """Collision-aware trajectory. Each item is a full (N,) array in radians."""
        cfg = self._config
        if getattr(self, "_mink_backend", False):
            from ..kinematics.mink_trajectory import plan_mink_trajectory

            return plan_mink_trajectory(
                self._solver,
                q_current,
                q_target,
                speed=cfg.reset_speed,
                rate=cfg.frequency,
                min_duration=cfg.reset_min_duration,
                collision_margin=cfg.mink_reset_collision_margin,
            )
        from .trajectory import plan_collision_aware_trajectory

        return plan_collision_aware_trajectory(
            self._solver,
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

    def path_goal(self, name: str) -> np.ndarray:
        """Full (N,) joint vector for a named path goal (see :data:`PATH_GOALS`)."""
        if name == "rest":
            return self.rest_goal()
        if name == "zero":
            return np.zeros(self._solver.num_joints, dtype=np.float32)
        raise ValueError(f"unknown path goal {name!r} (expected one of {PATH_GOALS})")

    def compute_path_trajectory(
        self, q_current: np.ndarray, goals: Sequence[str]
    ) -> list[np.ndarray]:
        """Chain collision-aware legs through named goals, in order.

        Every leg is planned before anything plays, so a caller that loses
        this worker mid-move (a stop that kills the IK subprocess) still has
        the whole path. Each leg starts where the previous one ends; a leg
        already at its goal is skipped rather than played as a still hold.
        """
        traj: list[np.ndarray] = []
        q = np.asarray(q_current, dtype=np.float32)
        for name in goals:
            goal = self.path_goal(name)
            if float(np.max(np.abs(goal - q))) < _PATH_LEG_SKIP_RAD:
                continue
            leg = self.compute_reset_trajectory(q, goal)
            traj.extend(leg)
            if leg:
                q = np.asarray(leg[-1], dtype=np.float32)
        return traj

    def set_config(self, key: str, value: object) -> None:
        """Live-update one :class:`VRTeleopConfig` field (``("set", …)`` message).

        Only fields this process reads at step time are meaningful here
        (multipliers, box width rate, ramp pacing); the core validates the
        key before forwarding, so an unknown one is logged and ignored rather
        than raised.
        """
        if not hasattr(self._config, key):
            _logger.warning("Ignoring live update of unknown config field %r", key)
            return
        current = getattr(self._config, key)
        try:
            coerced = (
                type(current)(value)
                if isinstance(current, (bool, int, float))
                else value
            )
        except (TypeError, ValueError):
            _logger.warning("Ignoring live update %s=%r (bad value)", key, value)
            return
        if key == "box_grasp" and coerced != current and self._box is not None:
            # Switch grasps under a live pair: re-snap, so the arms blend
            # into the new configuration. The two grasps touch the box at
            # different points, so the box is read off the grasp it is held
            # in (see snap_box's held_tool).
            self._held_tool = self._box.tool
            self._box = None
        setattr(self._config, key, coerced)

    def set_gripper(self, left: float, right: float) -> None:
        """The commanded gripper openings (``("grip", left, right)``
        message, 0 = closed, 1 = open): where the collision checks put the
        fingers (see :meth:`KinematicsSolver.set_gripper`). The Mink model
        has no sliding fingers: ignored there."""
        if not self._mink_backend:
            self._solver.set_gripper(left, right)

    def set_lift(self, raised: float) -> None:
        """The live lift height (``("lift", raised)`` message): where the
        collision checks put the deck and lower column (see
        :meth:`KinematicsSolver.set_lift`). The Mink model has no lift:
        ignored there."""
        if not self._mink_backend:
            self._solver.set_lift(raised)

    def _has_lift_contact(self) -> bool:
        """Whether the solver models a lift (the mobile Axol on the JAX
        backend, collision detection on): see :meth:`clear_lift`."""
        return (
            not getattr(self, "_mink_backend", False)
            and getattr(self._solver, "has_lift_contact", False) is True
        )

    def rest_goal(self) -> np.ndarray:
        """The return-to-rest goal: the rest pose, moved clear of Jelly's
        deck at the live lift height.

        With the lift lowered far enough that the hanging arms would sit in
        the deck, the arms come up and forward over it instead
        (:meth:`KinematicsSolver.lift_clear_pose`). The rest pose itself
        everywhere else. Cached per lift height.
        """
        rest = self.get_rest_q()
        if not self._has_lift_contact():
            return rest
        lowered = float(self._solver.lift_lowered)
        cached = self._rest_goal
        if cached is not None and abs(cached[0] - lowered) < 1e-4:
            return cached[1].copy()
        goal = np.asarray(self._solver.lift_clear_pose(rest), dtype=np.float32)
        self._rest_goal = (lowered, goal)
        if not np.array_equal(goal, rest):
            _logger.info(
                "Rest pose raised over Jelly's deck (lift lowered %.0f mm)",
                1e3 * lowered,
            )
        return goal.copy()

    def clear_lift(
        self, q: np.ndarray, sides: tuple[str, ...] = ("left", "right")
    ) -> np.ndarray:
        """``q`` with each of ``sides`` moved a step out of Jelly's deck.

        On the mobile Axol the deck and lower lift stage rise toward the arms
        as the lift lowers. An arm that isn't tracking (disengaged, or frozen
        while the other tracks) is moved out of their way, at up to
        ``_LIFT_CLEAR_SPEED`` (see :meth:`KinematicsSolver.lift_clear_step`).
        ``q`` unchanged when no arm is in their way, and everywhere but the
        mobile Axol.
        """
        if not self._has_lift_contact():
            return q
        now = time.perf_counter()
        nominal = 1.0 / float(self._config.ik_frequency)
        dt = nominal if self._lift_clear_t is None else now - self._lift_clear_t
        self._lift_clear_t = now
        dt = min(max(dt, nominal), 4.0 * nominal)
        q_new = self._solver.lift_clear_step(
            q, max_step=_LIFT_CLEAR_SPEED * dt, sides=sides
        )
        return q if q_new is None else np.asarray(q_new, dtype=np.float32)

    def _lift_contact_per_arm(
        self, q_new: np.ndarray, targets: dict[str, Pose]
    ) -> np.ndarray:
        """Per-arm teleop's handling of Jelly's deck after a solve.

        An engaged arm held up by the deck has its clutch anchor raised to
        the height it is held at (:meth:`_hold_lift_plane`). A frozen arm the
        deck rises into is moved out of its way (:meth:`clear_lift`), and its
        hold pose follows it.
        """
        contact = self._solver.lift_contact(q_new)
        engaged = {side: targets[side] for side in targets if self._active[side]}
        for side, shift in self._hold_lift_plane(q_new, engaged, contact).items():
            pos, rot = self._snap_fk[side]
            self._snap_fk[side] = (pos + shift, rot)
        if not contact:
            return q_new
        frozen = tuple(
            side
            for side in ("left", "right")
            if not self._active[side] and side in contact
        )
        if frozen:
            q_new = self.clear_lift(q_new, frozen)
            left, right = self._solver.fk(q_new)
            for side in frozen:
                self._hold_fk[side] = left if side == "left" else right
        return q_new

    def _hold_lift_plane(
        self,
        q_new: np.ndarray,
        targets: dict[str, Pose],
        contact: dict[str, tuple[float, np.ndarray]],
        shared: bool = False,
    ) -> dict[str, np.ndarray]:
        """How far to raise each engaged arm's target held up by Jelly's deck.

        While the collision cost holds an arm up off the deck, its target
        sits under the arm, inside the deck. Left there, it drops the arm to
        it the moment the arm clears the deck's edge. So per side in
        ``targets`` whose contact pushes the arm up, the target is raised to
        the height of the solved gripper (FK of ``q_new``). Pressing further
        down is given up the same way; raising the hand raises the arm at
        once. Horizontal motion is left alone, so pushing into the deck's
        edge still slides the arm up over it.

        That alone still lets the arm sag as it leaves the top: the deck held
        it up against the solve's pull toward its posture, and the target
        sits only where the arm was. So an arm on the top remembers its
        height (``_deck_plane``), and round the edge and past it its target
        is raised by ``_LIFT_PLANE_GAIN`` of any sag below that height each
        solve (clear of the deck, lowered the same way while above it, by no
        more than it was raised), until the hand moves up or down ``_LIFT_PLANE_RELEASE_M`` or
        ``_LIFT_PLANE_HOLD_S`` after the arm last touched the deck.

        Returns the shifts (world frame, vertical); those under
        ``_LIFT_PLANE_MIN_M`` (raise to the solved height) are left out.
        ``shared``: both targets move by the largest shift (box mode, which
        raises the box centre), so each hold counts that one.
        """
        planes = self._deck_plane
        for side in [side for side in planes if side not in targets]:
            del planes[side]
        if not contact and not planes:
            return {}
        now = time.perf_counter()
        left, right = self._solver.fk(q_new)
        solved = {"left": left, "right": right}
        raises: dict[str, float] = {}
        on_top: dict[str, float] = {}
        for side, target in targets.items():
            z = float(solved[side][0][2])
            z_target = float(target[0][2])
            up = float(contact[side][1][2]) if side in contact else None
            raise_by = 0.0
            if up is not None and up >= _LIFT_PLANE_MIN_UP:
                below = z - z_target
                if below > _LIFT_PLANE_MIN_M:
                    raise_by = below
            plane = planes.get(side)
            if up is not None and up >= _LIFT_PLANE_TOP_UP:
                planes[side] = _DeckPlane(z=z, hand_z=z_target, t=now)
                on_top[side] = z_target
            elif plane is not None:
                if up is not None:
                    plane.t = now
                hand = z_target - plane.raised - plane.hand_z
                if (
                    abs(hand) > _LIFT_PLANE_RELEASE_M
                    or now - plane.t > _LIFT_PLANE_HOLD_S
                ):
                    del planes[side]
                else:
                    sag = plane.z - z
                    if sag > _LIFT_PLANE_MIN_M:
                        raise_by = max(raise_by, _LIFT_PLANE_GAIN * sag)
                    elif up is None and sag < -_LIFT_PLANE_MIN_M:
                        # Clear of the deck and above its height: give back
                        # what the edge made the hold overshoot, no more.
                        raise_by = max(_LIFT_PLANE_GAIN * sag, -plane.raised)
            if raise_by != 0.0:
                raises[side] = raise_by
        if shared and raises:
            top = max(raises.values())
            raises = dict.fromkeys(raises, top)
            applied = dict.fromkeys(targets, top)
        else:
            applied = raises
        for side, plane in planes.items():
            if side in on_top:
                plane.hand_z = on_top[side] + applied.get(side, 0.0)
            else:
                plane.raised += applied.get(side, 0.0)
        return {
            side: np.array((0.0, 0.0, v), dtype=np.float32)
            for side, v in raises.items()
        }

    def pair_status(self, q: np.ndarray) -> dict:
        """Geometry of the gripper pair at ``q`` for the headset's cues.

        ``aligned`` is True when the two grippers already form the box-mode
        side-clamping pair — fingers forward, a flat face toward the other
        gripper, each within ``_ALIGNED_TOL_DEG`` of where a box-mode engage
        would blend it (see :func:`~almond_axol.teleop.box.pair_aligned`) —
        and the gap is inside the box-mode width range, so switching to box
        mode from here costs (almost) no alignment blend. ``width`` is the
        grip width in metres — the separation the fitted tool's contact
        faces would have with the mounts where they are (the mount
        separation itself for the URDF gripper) — ``grasp`` the grasp in
        force (``"straight"`` / ``"flush"``) and ``elbow`` the elbows-out
        angle (degrees, ``config.box_elbow_out``) the sticks may have jogged.
        ``squeeze`` is the clamp force (N, per arm) the last box-mode solve
        leaned for (:meth:`_squeeze_lean`; 0 while not pressing) and
        ``trim`` the mean extra inward yaw (degrees) the squeeze trims have
        added to the gripper targets on top of the lean, ``trims`` the two
        arms' own (``[left, right]``).
        """
        left, right = self._solver.fk(q)
        tool = self._box_tool()
        faces = self._box_faces()
        tilt = math.radians(self._config.box_grip_tilt)
        _c, rot, sep = box_frame(left[0], right[0])
        if sep <= 1e-3:
            width = 0.0
            aligned = False
        else:
            yaw = tilt + tool.flush_tilt
            chosen = choose_faces({"left": left[1], "right": right[1]}, rot, yaw, faces)
            width = contact_width(
                left,
                right,
                rot,
                {side: tool.foot(chosen[side]) for side in ("left", "right")},
            )
            aligned = pair_aligned(
                left,
                right,
                width_min=self._box_width_min(),
                width_max=self._config.box_width_max,
                tilt=tilt,
                tol_deg=_ALIGNED_TOL_DEG,
                tool=tool,
                faces=faces,
            )
        return {
            "aligned": bool(aligned),
            "width": round(width, 3),
            "grasp": self._box_grasp(),
            "elbow": round(float(self._config.box_elbow_out), 1),
            "squeeze": round(self._lean_force, 1),
            "trim": round(self.squeeze_trim_deg, 1),
            "trims": [round(v, 1) for v in self.squeeze_trims_deg.values()],
            "stow": (
                stow.phase
                if (
                    stow := getattr(self, "_stow", None)
                    or getattr(self, "_arm_stow", None)
                )
                else None
            ),
        }

    def collision_proximity(self, q: np.ndarray) -> dict[str, float] | None:
        """Per-arm closeness to the IK collision boundary at ``q``, 0 to 1.

        See :meth:`~almond_axol.kinematics.solver.KinematicsSolver.collision_proximity`.
        ``None`` when the solver has no collision model (detection off).
        """
        proximity = getattr(self._solver, "collision_proximity", None)
        if proximity is None:
            return None
        levels = proximity(q)
        if not isinstance(levels, dict):
            return None
        return {side: round(float(v), 3) for side, v in levels.items()}

    def _box_grasp(self) -> str:
        """The current box-mode grasp, ``"straight"`` or ``"flush"`` (``config.box_grasp``)."""
        raw = str(getattr(self._config, "box_grasp", "straight")).strip().lower()
        return "flush" if raw == "flush" else "straight"

    def _box_tool(self) -> ToolGeometry:
        """The box-mode contact geometry for the current grasp and tool.

        The fitted gripper's (``config.gripper``) in the current grasp: for
        the parcel gripper, its fixed blade's front face in ``"straight"``
        and its plate's facet and the blade's tip, turned ``box_flush_deg``
        inward, in ``"flush"`` (:func:`parcel_tool`). A grasp switch keeps
        the box and moves each gripper onto the new grasp's contacts (see
        :meth:`set_config`). Box mode only runs with the parcel
        gripper; anything else gets the trivial :data:`URDF_TOOL` (the
        mounts themselves, the same in both grasps).
        """
        cfg = self._config
        if getattr(cfg, "gripper", "parallel") == "parcel":
            return parcel_tool(
                float(getattr(cfg, "box_flush_deg", 39.0)), self._box_grasp()
            )
        return URDF_TOOL

    def _box_width_min(self) -> float:
        """The smallest grip width the sticks (and an engage snap) allow now.

        ``config.box_width_min`` — the operator's floor between the contact
        faces — or, if larger, the width at which the fitted tool's bodies
        would come within a clearance of each other in the current grasp
        (:meth:`ToolGeometry.min_width`). The parcel gripper's straight
        grasp closes until its open plates meet (``_PARCEL_BODY_CLEARANCE_M``,
        ~124 mm between the blade fronts); its flush grasp's contacts are
        proud of everything else, so they may close to the operator's floor
        (``0`` by default: the faces together). The stock gripper's face is
        modelled on the mount axis and its wrists meet first, kept
        ``_BODY_CLEARANCE_M`` apart (~77 mm).
        """
        cfg = self._config
        parcel = getattr(cfg, "gripper", "parallel") == "parcel"
        clearance = _PARCEL_BODY_CLEARANCE_M if parcel else _BODY_CLEARANCE_M
        return max(float(cfg.box_width_min), self._box_tool().min_width(clearance))

    def note_measured(
        self,
        pos_left: np.ndarray,
        pos_right: np.ndarray,
        kp_left: np.ndarray,
        kp_right: np.ndarray,
    ) -> None:
        """Record the arms' measured joint positions and joint stiffness.

        Sent by the core before each frame in box mode on hardware
        (``("meas", left, right, kp_left, kp_right)``, see
        ``VRTeleopCore.run_ik_loop``): the joint positions (rad, each
        ``(8,)`` in :data:`ARM_JOINTS` order plus the gripper) and each
        arm's impedance stiffness ``kp`` per arm joint (Nm/rad, ``(7,)``).
        Read by :meth:`_squeeze_lean`.
        """
        self._measured = (
            np.asarray(pos_left, dtype=np.float32),
            np.asarray(pos_right, dtype=np.float32),
            np.asarray(kp_left, dtype=np.float64),
            np.asarray(kp_right, dtype=np.float64),
        )
        self._measured_t = time.perf_counter()

    @property
    def squeeze_force(self) -> float:
        """Clamp force (N) per arm the last box-mode solve leaned for (0 when not pressing)."""
        return self._lean_force

    def _squeeze_lean(
        self,
        box: BoxState,
        targets: dict[str, Pose],
        q_current: np.ndarray,
        now: float,
        rot: np.ndarray | None = None,
    ) -> dict[str, Pose]:
        """Lean the gripper targets so the clamp presses evenly on the contacts.

        Jogging the width in past the box runs the targets ahead of where
        the box holds the grippers, and the arms' springs turn that
        run-ahead into the clamp. A plain lateral run-ahead, though, is a
        force at the gripper *mount* plus the moment it takes to hold the
        mount's orientation against the arm's stiffness coupling — and the
        parcel gripper touches the box with its fixed blade's front face,
        7.5 to 12.5 cm ahead of the mount, or its chamfer facet and the
        blade's tip 7 cm apart, so that moment can only be carried by the contacts loading
        unevenly: the facet digs in as the tip lifts, the pinch. :func:`squeeze_lean` gives, from the
        arm's Jacobian at the commanded pose and its joint stiffness, the
        target offset (a yaw of about a degree per centimetre of depth, a
        touch of roll for the tall face, a millimetre of translation) that
        turns the run-ahead into a pure force through the contacts'
        centroid, which they then share evenly; it is scaled by
        ``config.box_squeeze_lean`` (1 = the model) and applied to each
        gripper's target here.

        The one measurement is the clamp *depth*: how far the raw targets
        sit past the measured mounts along the pair's inward normals
        (:meth:`note_measured`, FK of the measured joints), averaged over
        the two arms — the servo lag of a move loads one arm's normal and
        unloads the other's by the same amount, so the mean is the clamp
        and a carry adds nothing — and low-passed (``_LEAN_TAU_S``). The
        lean is proportional to it, so nothing is added before contact,
        and it is an offset to the *target*: the arms are position
        controlled as ever, the command path is untouched, and neither arm
        is ever held back from a move (the measured-pose command shaping
        this replaces did exactly that and cost the pair its alignment).
        With ``config.box_squeeze_force`` > 0 the depth is capped at that
        force's, the targets pulled back out along the normals to hold it
        — the same on both arms, so the pair stays a pair. Otherwise (the
        default) nothing holds the depth back: the clamp keeps tightening
        as far as the operator jogs the width in.

        The model's lean is open loop, and the arm is not quite the model:
        gear backlash, the wrist's own compliance, and the gripper flexing
        under load all let the tip swing off the box further than the joint
        springs say, so a heavy box — more depth for the friction to carry
        it — still shows the pinch. The **squeeze trim** closes that gap
        from the one thing the encoders do see: the yaw between each
        gripper's measured mount rotation and its parallel slot, signed
        tip-out (:func:`toe_out_sides`). While the grippers press, each
        arm's toe-out is integrated (``_TRIM_GAIN``) into an extra inward
        yaw of *its* target — the two arms' give differs (the grippers are
        identical parts, not mirrored, so a clamp loads their hinges
        opposite ways), so one trim for both left the worse side short —
        bounded by ``config.box_squeeze_trim`` degrees (0 turns it off),
        and it bleeds away (``_TRIM_DECAY_S``) once they don't. A turn of
        the pair lags both arms the same way about up, which reads as
        toe-out on one side and toe-in on the other, so the trims only
        integrate once the pair's commanded motion has been still
        (``_TRIM_STILL_*``) long enough for the lag to settle. They settle
        where each measured face is parallel to the box — tip and root
        both on it — whatever the unmodelled give was, a second or so
        after the clamp. While pressing, a line of diagnostics — per arm
        depth, force, toe-out and trim — is logged every
        ``_CLAMP_LOG_PERIOD_S``.

        Off (targets returned as they are) without a measurement (the
        sim, or none reported for ``_LEAN_RESET_S``) or a pair still
        blending into alignment; the model's lean also needs the arm model
        and a nonzero lean scale or force cap. ``rot`` is the pair's box
        frame this frame (the snap's ``box.rot`` turned by the leader);
        omitted, the snap's.
        """
        cfg = self._config
        scale = float(getattr(cfg, "box_squeeze_lean", 0.0))
        cap = float(getattr(cfg, "box_squeeze_force", 0.0))
        trim_max = math.radians(max(float(getattr(cfg, "box_squeeze_trim", 0.0)), 0.0))
        meas = self._measured
        if meas is not None and now - self._measured_t > _LEAN_RESET_S:
            meas = None  # the core stopped reporting (no reading): stale
        if meas is None or not box.aligned:
            self._lean_depth, self._lean_t, self._lean_force = 0.0, None, 0.0
            self._lean_trim = {"left": 0.0, "right": 0.0}
            self._trim_prev = None
            return targets
        pos_l, pos_r, kp_l, kp_r = meas
        q_meas = np.asarray(q_current, dtype=np.float32).copy()
        for i, gi in enumerate(self.left_indices):
            q_meas[gi] = pos_l[i]
        for i, gi in enumerate(self.right_indices):
            q_meas[gi] = pos_r[i]
        if not np.all(np.isfinite(q_meas)):
            return targets
        measured = dict(zip(("left", "right"), self._solver.fk(q_meas)))
        frame = np.asarray(box.rot if rot is None else rot, dtype=np.float64)
        lateral = frame[:, 1]
        normals = {"left": -lateral, "right": lateral}
        up = frame[:, 2]
        depths = {
            side: float(
                (
                    np.asarray(targets[side][0], dtype=np.float64)
                    - np.asarray(measured[side][0], dtype=np.float64)
                )
                @ normals[side]
            )
            for side in ("left", "right")
        }
        raw_depth = 0.5 * (depths["left"] + depths["right"])
        dt = 0.0
        if self._lean_t is None or now - self._lean_t > _LEAN_RESET_S:
            self._lean_depth = raw_depth
        else:
            dt = now - self._lean_t
            alpha = 1.0 - math.exp(-dt / _LEAN_TAU_S)
            self._lean_depth += alpha * (raw_depth - self._lean_depth)
        self._lean_t = now
        depth = max(self._lean_depth, 0.0)

        # The trims: integrate each arm's measured toe-out while pressing
        # and the pair is still, bleed them off while not pressing; bounded
        # either way.
        toe = toe_out_sides(targets, measured, normals, up)
        still = self._pair_still(targets, now)
        if trim_max <= 0.0:
            self._lean_trim = {"left": 0.0, "right": 0.0}
        elif depth > 0.0:
            if still:
                for side in ("left", "right"):
                    self._lean_trim[side] = float(
                        np.clip(
                            self._lean_trim[side] + _TRIM_GAIN * toe[side] * dt,
                            -trim_max,
                            trim_max,
                        )
                    )
        elif dt > 0.0:
            decay = math.exp(-dt / _TRIM_DECAY_S)
            for side in ("left", "right"):
                self._lean_trim[side] *= decay

        if depth <= 0.0:
            self._lean_force = 0.0
            return self._apply_trim(targets, normals, up)

        model = self._lean_model if (scale > 0.0 or cap > 0.0) else None
        if model is None:
            # No arm model, or the model's lean and cap both off: the trim
            # alone.
            self._lean_force = 0.0
            self._log_clamp(now, depths, {}, toe, still)
            return self._apply_trim(targets, normals, up)
        tool = self._box_tool()
        kp = {"left": kp_l, "right": kp_r}
        out: dict[str, Pose] = {}
        forces: list[float] = []
        per_side_force: dict[str, float] = {}
        models: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for side, indices in (
            ("left", self.left_indices),
            ("right", self.right_indices),
        ):
            arm_q = np.asarray(q_current, dtype=np.float64)[indices]
            try:
                _p, rot_cmd, jac = model.mount_jacobian(arm_q, is_left=(side == "left"))
            except Exception:  # noqa: BLE001 - never let the lean stop the solve
                _logger.exception("squeeze lean failed; targets unchanged")
                self._lean_model = None
                return self._apply_trim(targets, normals, up)
            models[side] = (rot_cmd, jac, np.asarray(tool.contacts(box.face[side])))
        for side in ("left", "right"):
            pos, rot = targets[side]
            rot_cmd, jac, pts = models[side]
            lean = squeeze_lean(
                jac,
                kp[side],
                rot_cmd,
                normals[side],
                pts,
                depth,
                cap if cap > 0.0 else 0.0,
            )
            forces.append(lean.force)
            per_side_force[side] = lean.force
            shift = scale * lean.translation + lean.pullback * normals[side]
            new_pos = (np.asarray(pos, dtype=np.float64) + shift).astype(np.float32)
            angle = float(np.linalg.norm(lean.rotation)) * scale
            new_rot = rot
            if angle > 1e-9:
                axis = lean.rotation / np.linalg.norm(lean.rotation)
                new_rot = (rodrigues(axis, angle) @ rot).astype(np.float32)
            out[side] = (new_pos, new_rot)
        self._lean_force = 0.5 * sum(forces)
        self._log_clamp(now, depths, per_side_force, toe, still)
        return self._apply_trim(out, normals, up)

    def _pair_still(self, targets: dict[str, Pose], now: float) -> bool:
        """True once the pair's commanded motion has been slow for ``_TRIM_STILL_S``."""
        prev = self._trim_prev
        self._trim_prev = (now, targets)
        if prev is None:
            self._trim_moving_t = now
            return False
        t_prev, prev_targets = prev
        dt = now - t_prev
        if dt <= 0.0:
            return now - self._trim_moving_t >= _TRIM_STILL_S
        moving = False
        for side in ("left", "right"):
            (p0, r0), (p1, r1) = prev_targets[side], targets[side]
            speed = float(np.linalg.norm(np.asarray(p1, float) - np.asarray(p0, float)))
            speed /= dt
            yaw_rate = rotation_angle(r0, r1) / dt
            if speed > _TRIM_STILL_SPEED or yaw_rate > _TRIM_STILL_YAW_RATE:
                moving = True
        if moving:
            self._trim_moving_t = now
        return now - self._trim_moving_t >= _TRIM_STILL_S

    def _log_clamp(
        self,
        now: float,
        depths: dict[str, float],
        forces: dict[str, float],
        toe: dict[str, float],
        still: bool,
    ) -> None:
        """Rate-limited clamp diagnostics while the grippers press."""
        if now - self._clamp_log_t < _CLAMP_LOG_PERIOD_S:
            return
        self._clamp_log_t = now
        _logger.info(
            "box clamp: depth L %.1f / R %.1f mm, force L %.1f / R %.1f N, "
            "toe-out L %+.2f / R %+.2f°, trim L %+.2f / R %+.2f°%s",
            depths["left"] * 1e3,
            depths["right"] * 1e3,
            forces.get("left", 0.0),
            forces.get("right", 0.0),
            math.degrees(toe["left"]),
            math.degrees(toe["right"]),
            math.degrees(self._lean_trim["left"]),
            math.degrees(self._lean_trim["right"]),
            "" if still else " (pair moving: trims held)",
        )

    def _apply_trim(
        self, targets: dict[str, Pose], normals: dict[str, np.ndarray], up: np.ndarray
    ) -> dict[str, Pose]:
        """Yaw each gripper target its ``_lean_trim`` further into the box about ``up``."""
        if all(abs(t) <= 1e-9 for t in self._lean_trim.values()):
            return targets
        out: dict[str, Pose] = {}
        for side, (pos, rot) in targets.items():
            trim = self._lean_trim[side]
            if abs(trim) <= 1e-9:
                out[side] = (pos, rot)
                continue
            sign = tip_inward_sign(rot, normals[side], up)
            out[side] = (pos, (rodrigues(up, sign * trim) @ rot).astype(np.float32))
        return out

    @property
    def squeeze_trim_deg(self) -> float:
        """Mean extra inward yaw (degrees) of the gripper targets the squeeze trims have added."""
        return math.degrees(0.5 * sum(self._lean_trim.values()))

    @property
    def squeeze_trims_deg(self) -> dict[str, float]:
        """Per arm, the extra inward yaw (degrees) its squeeze trim has added."""
        return {side: math.degrees(t) for side, t in self._lean_trim.items()}

    def _box_faces(self) -> Faces:
        """Pinned clamping faces from ``config.box_face_left/right`` (0 = auto)."""
        out: Faces = {}
        for side in ("left", "right"):
            raw = str(getattr(self._config, f"box_face_{side}", "auto")).strip()
            out[side] = {"+x": 1.0, "-x": -1.0}.get(raw.lower(), 0.0)
        return out

    def clear_engage(self) -> None:
        """Drop engage state without touching the (warm) pose filters.

        Used when the arms are moved by something other than this worker
        (reset replay, external motion) so the next locked frame performs a
        fresh engage snap against the new pose — in either re-engage mode,
        since the snap poses (the "ramp" anchor) go too.
        """
        self._active = {"left": False, "right": False}
        self._hold_fk = {}
        self._hold_elbow_fk = {}
        self._ramp = {}
        self._clear_freeze()
        self._box = None
        self._held_tool = None
        self._deck_plane = {}
        self._arm_stow = None
        self._box_leader = None
        self._box_snap = None
        self._box_rot = None
        self._snap_ctrl = {}
        self._snap_fk = {}
        self._snap_elbow_ctrl = {}
        self._snap_elbow_fk = {}
        if self._rec is not None:
            self._rec.set_engaged(False)

    def reset(self) -> None:
        """Deactivate the engage-toggle state and clear snap poses and filter state.

        Call this before replaying a reset trajectory so the next engage
        performs a fresh engage-snap from the current IK pose.
        """
        # A forced disengage (reset, stale stream) may never deliver another
        # lock-less frame to step() — clear_engage closes the recording
        # segment too.
        self.clear_engage()
        self._abs_active = False
        self._abs_base = None
        self._abs_offset = {}
        self.abs_base_msg = None
        if self._config.absolute_mode:
            # The reset trajectory returns the arms to rest, so the recorded
            # TCP stream should land there too rather than hold the last
            # engaged target.
            self.last_tcp_msg = self._encode_tcp_msg(*self._rest_fk_poses())
        self._prev_raw = {}
        self._prev_raw_t = None
        self._raw_vel = {}
        self._suspect = None
        self._last_solve_t = None
        self._reset_pose_filters()
        # JAX pins posture on engage; Mink keeps its rest attractor throughout
        # a policy/teleop cycle. Both clear history at the reset boundary.
        self._solver.set_posture_pose(self.get_rest_q())
        if getattr(self, "_mink_backend", False):
            self._solver.reset_tracking_state()

    # -- Internal -----------------------------------------------------------

    def _step_box(
        self,
        frame: VRFrame,
        q_current: np.ndarray,
        ctrl: dict[str, tuple[np.ndarray, np.ndarray]],
        lp: np.ndarray,
        rp: np.ndarray,
    ) -> np.ndarray:
        """Box-mode solve: one controller carries both grippers as a level pair.

        On engage the pair is snapped from FK (:func:`snap_box`), the leader
        controller's pose is anchored, and the grippers are blended into the
        side-clamping grasp (level, the fitted tool's contact face along the
        box side — the grasp, read at each engage together with the pinned
        faces) over ``box_align_duration``. Afterwards the leader hand
        drives the pair through the usual per-arm clutch mapping
        (:func:`_relative_target_np`, so moving the hand feels exactly like
        normal teleop) in all six degrees of freedom: the pair follows the
        hand's translation, and the hand's rotation since the snap — yaw,
        pitch and roll, in the box frame's axes (:func:`_box_turn`) — turns
        the whole pair about its centre, so a wrist turn lines the pair up
        with (or tips) a box without moving it. The right thumbstick's forward/back sets the
        grip width (see :meth:`_integrate_sticks`); the rest of the sticks
        drive Jelly as outside box mode, and the grasp is switched from the
        settings (``box_grasp``), re-running the blend.
        """
        leader = frame.box_leader
        assert leader in ("left", "right")
        cfg = self._config
        now = time.perf_counter()
        if self._rec is not None:
            self._rec.set_engaged(True)

        engaged = self._active["left"] and self._active["right"]
        if self._box is None or not engaged or self._box_leader != leader:
            # Engage snap, or the leading hand changed: (re)anchor to the
            # live pose. A handover mid-align simply restarts the blend from
            # wherever the grippers are, so nothing jumps. A re-snap within
            # box mode (handover, freeze and lead again, grasp switch) keeps
            # the pair's last frame while the grippers still sit in it, so a
            # tipped box stays tipped; a fresh engage squares the pair level.
            l_fk, r_fk = self._solver.fk(q_current)
            keep = self._box_rot
            if keep is not None and not tipped_frame_holds(
                l_fk, r_fk, keep, _ALIGNED_TOL_DEG
            ):
                keep = None
            self._box = snap_box(
                l_fk,
                r_fk,
                now,
                align_duration=cfg.box_align_duration,
                width_min=self._box_width_min(),
                width_max=cfg.box_width_max,
                tilt=math.radians(cfg.box_grip_tilt),
                tool=self._box_tool(),
                faces=self._box_faces(),
                rot=keep,
                held_tool=self._held_tool,
            )
            self._held_tool = None
            self._deck_plane = {}
            self._box_leader = leader
            self._box_snap = (ctrl[leader], l_fk if leader == "left" else r_fk)
            self._stow = None
            self._ramp = {}
            self._active = {"left": True, "right": True}
            self._hold_fk = {}
            self._hold_elbow_fk = {}
            self._clear_freeze()
            self._solver.set_posture_pose(q_current)
            self._last_solve_t = None
            return q_current

        box = self._box
        ctrl_pos, ctrl_rot = ctrl[leader]
        assert self._box_snap is not None
        (snap_ctrl_pos, snap_ctrl_rot), snap_fk = self._box_snap
        lead_pos, _lead_rot = _relative_target_np(
            ctrl_pos,
            ctrl_rot,
            snap_ctrl_pos,
            snap_ctrl_rot,
            *snap_fk,
            position_multiplier=cfg.position_multiplier,
        )
        # The box centre is carried along with the leader gripper's
        # translation since the snap, and the pair is turned about that
        # centre by the hand's full rotation since the snap — yaw, pitch and
        # roll — read in the box frame's own axes (_box_turn), not through
        # the leader gripper's clutch mapping: that one is relative to the
        # gripper's pose at the snap, before the align blend, so a gripper
        # that engaged pointing down would turn a hand's yaw into a roll of
        # the box.
        snap_pos = snap_fk[0]
        center = (lead_pos + (box.center - snap_pos)).astype(np.float32)
        turn = _box_turn(ctrl_rot, snap_ctrl_rot, cfg.rotation_multiplier)
        rot = (box.rot @ turn).astype(np.float32)
        self._box_rot = rot
        self._integrate_sticks(frame, box, now)
        if getattr(frame, "box_stow", False) and getattr(self, "_stow", None) is None:
            self._stow = self._start_stow(box, center, rot, q_current, now)
        stow = getattr(self, "_stow", None)
        if stow is not None:
            # The stow drives the pair; the leading hand is ignored.
            center, rot = self._stow_pose(stow, now)
            self._box_rot = rot
        targets = box_targets(box, center, rot, now)
        targets = self._squeeze_lean(box, targets, q_current, now, rot)
        elbows = (
            self._stow_elbows(stow)
            if stow is not None
            else self._box_elbow_hints(q_current, targets)
        )
        # The posture attractor follows q for the whole of box mode. Pinned at
        # the engage pose (normal teleop's behaviour) it balances the pose
        # cost well short of the target — posture_weight 5 against pos_weight
        # 50 left the grippers 20-45 mm apart from their slots and the pair
        # twisted 5-13° in offline replays of a chest-height close, i.e. a
        # visibly non-parallel grasp; following, the same closes land within
        # 1 mm / 0.1°. The arms' free elbow swivel then has no attractor, and
        # doesn't need one: rest damping holds it where it is, the arm/torso
        # collision model keeps it off the base, and the optional elbow hint
        # (box_elbow_weight > 0) steers it explicitly.
        # A stow instead pulls the posture into its planned pose, so the
        # arms arrive there, symmetric, whatever they started from.
        self._solver.set_posture_pose(self._stow_posture(stow, q_current))

        delta_scale = 1.0
        if self._last_solve_t is not None:
            elapsed = now - self._last_solve_t
            delta_scale = float(np.clip(elapsed * cfg.ik_frequency, 1.0, 4.0))
        self._last_solve_t = now

        solve_t0 = time.perf_counter()
        q_new = self._solver.ik(
            q_current,
            left_pose=targets["left"],
            right_pose=targets["right"],
            left_elbow_pos=elbows["left"] if elbows else None,
            right_elbow_pos=elbows["right"] if elbows else None,
            delta_scale=delta_scale,
            elbow_weight=cfg.box_elbow_weight if elbows else None,
            pose_weight_scale=_BOX_POSE_WEIGHT_SCALE,
        )
        solve_ms = (time.perf_counter() - solve_t0) * 1000.0
        q_new = np.asarray(q_new, dtype=np.float32).copy()
        self._keep_elbows_bent(q_new, q_current)
        if stow is not None:
            self._settle_stow(stow, q_new, q_current, delta_scale)
            self._advance_stow(stow, q_new, q_current, targets, now)
        if self._has_lift_contact():
            # Held up by Jelly's deck, the pair keeps the height it is held
            # at (see _hold_lift_plane): the box centre rises with the arm
            # held furthest above its target, so the pair stays a pair.
            shifts = self._hold_lift_plane(
                q_new, targets, self._solver.lift_contact(q_new), shared=True
            )
            if shifts:
                shift = max(shifts.values(), key=lambda v: float(np.linalg.norm(v)))
                box.center = (box.center + shift).astype(np.float32)
        if self._rec is not None:
            self._rec.record(
                raw_l=np.array(
                    [
                        frame.l_ee.position.x,
                        frame.l_ee.position.y,
                        frame.l_ee.position.z,
                    ]
                ),
                raw_r=np.array(
                    [
                        frame.r_ee.position.x,
                        frame.r_ee.position.y,
                        frame.r_ee.position.z,
                    ]
                ),
                filt_l=lp,
                filt_r=rp,
                tgt_l=targets["left"][0],
                tgt_r=targets["right"][0],
                q=q_new,
                engaged=np.array([1.0, 1.0]),
                solve_ms=solve_ms,
            )
        return q_new

    def _keep_elbows_bent(self, q_new: np.ndarray, q_current: np.ndarray) -> None:
        """Keep box mode's elbows from bending backwards (in place).

        Each elbow can hyperextend a little past straight (the URDF allows
        about 24°). Led out of reach, the box-mode solve stretches an arm
        through straight into that backwards branch, and once there no box
        motion toward the body brings it back: with the gripper held, the
        arm would have to pass through straight again, at full reach. The
        stow then can't fold that arm into its plan. So in box mode an
        elbow may not move further past ``_ELBOW_BENT_RAD`` from straight
        toward the backwards side; one already past it may only come back.
        """
        left = getattr(self._solver, "left_indices", None)
        right = getattr(self._solver, "right_indices", None)
        if getattr(self, "_mink_backend", False) or left is None or right is None:
            return
        for indices, sign in ((left, 1.0), (right, -1.0)):
            i = indices[_ELBOW_INDEX]
            # Bent forward is +sign; "sign * q" < _ELBOW_BENT_RAD is backwards.
            floor = min(float(sign * q_current[i]), _ELBOW_BENT_RAD)
            if sign * q_new[i] < floor:
                q_new[i] = sign * floor

    def _start_stow(
        self,
        box: BoxState,
        center: np.ndarray,
        rot: np.ndarray,
        q: np.ndarray,
        now: float,
    ) -> _Stow:
        """Begin the box-mode stow from the pair's pose this frame.

        The stow (**A**) brings the box in to the chest: level, facing
        straight forward, centred between the shoulders,
        ``config.box_stow_height`` below them and as close in front as the
        arms reach (:meth:`_plan_stow`), with the elbows tucked in toward
        the base and the two arms mirror images of each other. The box moves
        there on a straight line while the arms' posture and elbows blend
        into the planned pose (:meth:`_stow_posture`,
        :meth:`_stow_elbows`), then the pose is held until the arms settle
        (:meth:`_advance_stow`). The core then freezes the pair for driving.
        """
        cfg = self._config
        t_plan = time.perf_counter()
        goal_center, q_goal = self._plan_stow(box, q)
        goal_rot = np.eye(3, dtype=np.float32)
        q = np.asarray(q, dtype=np.float32)
        speed = max(float(cfg.box_stow_speed), 1e-3)
        distance = float(np.linalg.norm(goal_center - np.asarray(center, np.float32)))
        turn = rotation_angle(np.asarray(rot, np.float32), goal_rot)
        joints = 0.0 if q_goal is None else float(np.max(np.abs(q_goal - q)))
        duration = max(
            distance / speed,
            turn / _STOW_TURN_RAD_S,
            joints / _STOW_JOINT_RAD_S,
            _STOW_MIN_S,
        )
        el0 = self._solver.elbow_positions(q)
        el1 = el0 if q_goal is None else self._solver.elbow_positions(q_goal)
        _logger.info(
            "Box stow: planned in %.1f s (box %.0f mm wide, %.0f mm ahead of "
            "the shoulders, elbows %+.0f mm outboard); moving it in (%.1f s)",
            time.perf_counter() - t_plan,
            box.width * 1000.0,
            (float(goal_center[0]) - float(self._shoulder_mid()[0])) * 1000.0,
            (self._stow_elbow_out(q_goal) if q_goal is not None else float("nan"))
            * 1000.0,
            duration,
        )
        return _Stow(
            start=(
                np.asarray(center, np.float32).copy(),
                np.asarray(rot, np.float32).copy(),
            ),
            goal=(goal_center, goal_rot),
            q_start=q.copy(),
            q_goal=q_goal,
            elbows_start={"left": np.asarray(el0[0]), "right": np.asarray(el0[1])},
            elbows_goal={"left": np.asarray(el1[0]), "right": np.asarray(el1[1])},
            # The plan blocked this solve for a while: the move starts now.
            t0=now + (time.perf_counter() - t_plan),
            duration=duration,
        )

    def _plan_stow(
        self, box: BoxState, q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """The stowed box centre and joints: ``(center, q)``.

        Tries box centres ``box_stow_reach`` in front of the shoulders and
        then further out (see ``_STOW_PLAN_*``), and returns the first the
        arms reach with the grippers in their slots; failing every one, the
        closest miss. Each candidate is solved for the left arm from the
        rest pose (warm-started from the previous candidate), with its elbow
        swung ``_STOW_ELBOW_DEG`` toward the base, and mirrored onto the
        right arm. The solver's posture, step history and collision-guard
        state are restored afterwards.
        """
        cfg = self._config
        solver = self._solver
        mid = self._shoulder_mid()
        rot = np.eye(3, dtype=np.float32)
        if getattr(self, "_mink_backend", False):
            # Mink has no elbow model to plan with: the box goes straight
            # in, the arms following it as they would a led box.
            center = np.array(
                (
                    float(mid[0]) + float(cfg.box_stow_reach),
                    mid[1],
                    mid[2] - float(cfg.box_stow_height),
                ),
                dtype=np.float32,
            )
            return center, None
        grip_rel, feet = box.grip_rel(), box.feet()
        saved = (
            getattr(solver, "_posture_pose", None),
            getattr(solver, "_q_prev", None),
            getattr(solver, "_base_collision_guard_active", None),
        )
        best: tuple[tuple[bool, bool, float], float, np.ndarray, np.ndarray] | None = (
            None
        )
        seed = self.get_rest_q()
        x0 = float(mid[0]) + float(cfg.box_stow_reach)
        n = int(round(_STOW_PLAN_SPAN_M / _STOW_PLAN_STEP_M)) + 1
        try:
            for k in range(n):
                center = np.array(
                    (
                        x0 + k * _STOW_PLAN_STEP_M,
                        mid[1],
                        mid[2] - float(cfg.box_stow_height),
                    ),
                    dtype=np.float32,
                )
                targets = ideal_gripper_poses(center, rot, box.width, grip_rel, feet)
                seed, miss = self._solve_stow_arms(seed, targets)
                out = self._stow_elbow_out(seed)
                reached = miss <= _STOW_PLAN_POS_TOL_M
                tucked = out <= _STOW_ELBOW_OUT_MAX_M
                key = (not reached, not tucked, miss + max(out, 0.0))
                if best is None or key < best[0]:
                    best = (key, miss, center, seed.copy())
                if reached and tucked:
                    break
        finally:
            for name, value in zip(
                ("_posture_pose", "_q_prev", "_base_collision_guard_active"), saved
            ):
                if value is not None:
                    setattr(solver, name, value)
        assert best is not None
        _key, miss, center, q_goal = best
        if miss > _STOW_PLAN_POS_TOL_M:
            _logger.warning(
                "Box stow: the arms can't reach the stowed box exactly "
                "(grippers %.0f mm off); stowing as close as they get",
                miss * 1000.0,
            )
        return center, q_goal

    def _step_arm_stow(self, q_current: np.ndarray) -> np.ndarray:
        """One solve of plain teleop's stow (**A** with the parcel gripper).

        Started on the first frame the core asks for it (``frame.arm_stow``),
        from the arms' joints then, whatever each arm was doing (tracking,
        frozen or at rest). The two grippers move in as a rigid pair: each
        keeps its pose in the pair frame (:func:`_pair_frame`) at the press,
        so a box squeezed between them stays squeezed by the same amount
        (each arm's command runs the same distance into it, so each presses
        with the same force), and the pair ends level along the lateral
        line, facing forward, in front of the chest with the elbows tucked
        (:meth:`_plan_arm_stow`). The move lasts as the box-mode stow's
        does and the pose is then held until the arms are still. The hands
        are ignored throughout. The core freezes both arms once the status
        reports ``"done"`` and stops asking; the next frame ends it
        (:meth:`_end_arm_stow`).
        """
        now = time.perf_counter()
        stow = self._arm_stow
        if stow is None:
            stow = self._arm_stow = self._start_arm_stow(q_current, now)
        if stow.phase == "move":
            u = (now - stow.t0) / stow.duration
            stow.alpha = smoothstep(min(max(u, 0.0), 1.0))
            if u >= 1.0:
                stow.phase = "hold"
                stow.hold_t0 = now
            frame_pose = blend_pose(stow.start, stow.goal, stow.alpha)
        else:
            frame_pose = stow.goal
        targets = _pair_targets(frame_pose, stow.rel)
        a = stow.alpha if stow.phase == "move" else 1.0
        elbows = None
        if stow.q_goal is not None:
            self._solver.set_posture_pose(
                ((1.0 - a) * stow.q_start + a * stow.q_goal).astype(np.float32)
            )
            elbows = {
                side: (
                    (1.0 - a) * stow.elbows_start[side] + a * stow.elbows_goal[side]
                ).astype(np.float32)
                for side in ("left", "right")
            }
        delta_scale = 1.0
        if self._last_solve_t is not None:
            delta_scale = float(
                np.clip(
                    (now - self._last_solve_t) * self._config.ik_frequency, 1.0, 4.0
                )
            )
        self._last_solve_t = now
        q_new = np.asarray(
            self._solver.ik(
                q_current,
                left_pose=targets["left"],
                right_pose=targets["right"],
                left_elbow_pos=elbows["left"] if elbows else None,
                right_elbow_pos=elbows["right"] if elbows else None,
                delta_scale=delta_scale,
                elbow_weight=(float(self._config.box_elbow_weight) or 10.0)
                if elbows
                else None,
                pose_weight_scale=_BOX_POSE_WEIGHT_SCALE,
            ),
            dtype=np.float32,
        ).copy()
        if stow.phase == "hold":
            held = now - stow.hold_t0
            still = (
                float(np.max(np.abs(q_new - np.asarray(q_current)))) < _STOW_STILL_RAD
            )
            if held >= _STOW_HOLD_MAX_S or (held >= _STOW_HOLD_MIN_S and still):
                if stow.phase != "done":
                    _logger.info("Arm stow: done")
                stow.phase = "done"
        return q_new

    def _start_arm_stow(self, q: np.ndarray, now: float) -> _ArmStow:
        """Plan plain teleop's stow from the arms' joints ``q``."""
        t_plan = time.perf_counter()
        q = np.asarray(q, dtype=np.float32)
        left, right = self._solver.fk(q)
        start = _pair_frame(left, right)
        center0, rot0 = start
        rel = {
            side: (
                (rot0.T @ (np.asarray(pose[0], np.float64) - center0)).astype(
                    np.float32
                ),
                (rot0.T @ np.asarray(pose[1], np.float64)).astype(np.float32),
            )
            for side, pose in (("left", left), ("right", right))
        }
        # Level and facing forward; arms crossed over keep their lateral
        # line's sense (a half turn), rather than swapping sides.
        goal_rot = np.eye(3, dtype=np.float32)
        if float(rot0[1, 1]) < 0.0:
            goal_rot = np.diag((-1.0, -1.0, 1.0)).astype(np.float32)
        goal_center, q_goal = self._plan_arm_stow(rel, goal_rot, q, float(center0[2]))
        speed = max(float(self._config.box_stow_speed), 1e-3)
        distance = float(np.linalg.norm(goal_center - center0))
        turn = rotation_angle(rot0.astype(np.float32), goal_rot)
        joints = 0.0 if q_goal is None else float(np.max(np.abs(q_goal - q)))
        duration = max(
            distance / speed,
            turn / _STOW_TURN_RAD_S,
            joints / _STOW_JOINT_RAD_S,
            _STOW_MIN_S,
        )
        el0 = self._solver.elbow_positions(q)
        el1 = el0 if q_goal is None else self._solver.elbow_positions(q_goal)
        sep = float(np.linalg.norm(rel["left"][0] - rel["right"][0]))
        _logger.info(
            "Arm stow: planned in %.1f s (grippers %.0f mm apart, %.0f mm ahead "
            "of the shoulders, elbows %+.0f mm outboard); moving in (%.1f s)",
            time.perf_counter() - t_plan,
            sep * 1000.0,
            (float(goal_center[0]) - float(self._shoulder_mid()[0])) * 1000.0,
            (self._arm_stow_elbow_out(q_goal) if q_goal is not None else float("nan"))
            * 1000.0,
            duration,
        )
        return _ArmStow(
            start=(center0.astype(np.float32), rot0.astype(np.float32)),
            goal=(goal_center, goal_rot),
            rel=rel,
            q_start=q.copy(),
            q_goal=q_goal,
            elbows_start={"left": np.asarray(el0[0]), "right": np.asarray(el0[1])},
            elbows_goal={"left": np.asarray(el1[0]), "right": np.asarray(el1[1])},
            # The plan blocked this solve for a while: the move starts now.
            t0=now + (time.perf_counter() - t_plan),
            duration=duration,
        )

    def _plan_arm_stow(
        self,
        rel: dict[str, tuple[np.ndarray, np.ndarray]],
        rot: np.ndarray,
        q: np.ndarray,
        start_z: float,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """The stowed pair midpoint and joints for plain teleop's stow.

        As :meth:`_plan_stow`, for the grippers' own pair: midpoints from
        ``_ARM_STOW_REACH_M`` in front of the shoulders outward, at
        ``box_stow_height`` below them, and the first both arms reach
        (within ``_ARM_STOW_POS_TOL_M``) with their elbows tucked
        (``_STOW_ELBOW_OUT_MAX_M``). A grasp the arms can't lift that high
        (the elbow folds into the shoulder first) is tried
        ``_ARM_STOW_DROP_M`` lower at a time, down to the height the pair is
        at (``start_z``). Each row stops once ``_ARM_STOW_PLAN_PATIENCE``
        candidates in a row do no better, and the first row with a reached
        candidate is the last. Failing all, the best of them (in reach
        first: the squeeze matters more than the elbows). The grippers
        needn't be mirror images, so each arm is solved with its own elbow
        hint (:meth:`_solve_arm_stow`).
        """
        cfg = self._config
        solver = self._solver
        mid = self._shoulder_mid()
        top = float(mid[2]) - float(cfg.box_stow_height)
        if getattr(self, "_mink_backend", False):
            center = np.array(
                (float(mid[0]) + _ARM_STOW_REACH_M, mid[1], top), dtype=np.float32
            )
            return center, None
        heights = [top]
        while heights[-1] - _ARM_STOW_DROP_M > start_z:
            heights.append(heights[-1] - _ARM_STOW_DROP_M)
        if start_z < heights[-1] - 1e-3:
            heights.append(start_z)
        saved = (
            getattr(solver, "_posture_pose", None),
            getattr(solver, "_q_prev", None),
            getattr(solver, "_base_collision_guard_active", None),
        )
        best: tuple[tuple[bool, bool, float], float, np.ndarray, np.ndarray] | None = (
            None
        )
        x0 = float(mid[0]) + _ARM_STOW_REACH_M
        n = int(round(_ARM_STOW_SPAN_M / _STOW_PLAN_STEP_M)) + 1
        try:
            for height in heights:
                seed = np.asarray(q, dtype=np.float32).copy()
                row_best: tuple[bool, bool, float] | None = None
                stale = 0
                done = False
                for k in range(n):
                    center = np.array(
                        (x0 + k * _STOW_PLAN_STEP_M, mid[1], height), dtype=np.float32
                    )
                    targets = _pair_targets((center, rot), rel)
                    seed, miss = self._solve_arm_stow(seed, targets)
                    out = self._arm_stow_elbow_out(seed)
                    reached = miss <= _ARM_STOW_POS_TOL_M
                    tucked = out <= _STOW_ELBOW_OUT_MAX_M
                    key = (not reached, not tucked, miss + max(out, 0.0))
                    if best is None or key < best[0]:
                        best = (key, miss, center, seed.copy())
                    if row_best is None or key < row_best:
                        row_best = key
                        stale = 0
                    else:
                        stale += 1
                    if reached and tucked:
                        done = True
                        break
                    if stale >= _ARM_STOW_PLAN_PATIENCE:
                        break
                if done or (best is not None and not best[0][0]):
                    break
        finally:
            for name, value in zip(
                ("_posture_pose", "_q_prev", "_base_collision_guard_active"), saved
            ):
                if value is not None:
                    setattr(solver, name, value)
        assert best is not None
        _key, miss, center, q_goal = best
        if miss > _ARM_STOW_POS_TOL_M:
            _logger.warning(
                "Arm stow: the arms can't reach the stowed pose exactly "
                "(grippers %.0f mm off); stowing as close as they get",
                miss * 1000.0,
            )
        return center, q_goal

    def _solve_arm_stow(
        self, seed: np.ndarray, targets: dict[str, Pose]
    ) -> tuple[np.ndarray, float]:
        """Solve both arms onto ``targets``, elbows tucked: ``(q, worst miss)``.

        Each arm's elbow is swung ``_STOW_ELBOW_DEG`` about its own
        shoulder-wrist line, iterated until no joint moves more than
        ``_STOW_PLAN_SETTLED_RAD`` (see :meth:`_solve_stow_arms`).
        """
        solver = self._solver
        q = np.asarray(seed, dtype=np.float32).copy()
        angle = math.radians(_STOW_ELBOW_DEG)
        for k in range(_STOW_PLAN_ITERS):
            q_prev = q
            swivel = solver.elbow_swivel(q)
            hints: dict[str, np.ndarray] = {}
            for side, sign in (("left", 1.0), ("right", -1.0)):
                arm = swivel[side]
                pos, rot = targets[side]
                wrist = np.asarray(pos, np.float64) + np.asarray(
                    rot, np.float64
                ) @ np.asarray(arm.wrist_in_mount, np.float64)
                hints[side] = elbow_swivel_hint(
                    arm.shoulder, arm.elbow, wrist, sign, angle
                ).astype(np.float32)
            solver.set_posture_pose(q)
            q = np.asarray(
                solver.ik(
                    q,
                    left_pose=targets["left"],
                    right_pose=targets["right"],
                    left_elbow_pos=hints["left"],
                    right_elbow_pos=hints["right"],
                    delta_scale=4.0,
                    elbow_weight=float(self._config.box_elbow_weight) or 10.0,
                    pose_weight_scale=_BOX_POSE_WEIGHT_SCALE,
                ),
                dtype=np.float32,
            ).copy()
            settled = float(np.max(np.abs(q - q_prev))) < _STOW_PLAN_SETTLED_RAD
            if settled and k + 1 >= _STOW_PLAN_MIN_ITERS:
                break
        fk = dict(zip(("left", "right"), solver.fk(q)))
        miss = max(
            float(np.linalg.norm(np.asarray(fk[s][0]) - np.asarray(targets[s][0])))
            for s in ("left", "right")
        )
        return q, miss

    def _arm_stow_elbow_out(self, q: np.ndarray) -> float:
        """How far (m) the further-out elbow sits outboard of its shoulder."""
        swivel = self._solver.elbow_swivel(q)
        left, right = swivel["left"], swivel["right"]
        return max(
            float(left.elbow[1] - left.shoulder[1]),
            float(right.shoulder[1] - right.elbow[1]),
        )

    def _end_arm_stow(self, q_current: np.ndarray) -> None:
        """End plain teleop's stow: both arms hold where they are.

        The core froze both arms (done) or a grip cancelled the stow; either
        way the next engage snaps that arm afresh from here.
        """
        if self._arm_stow is not None and self._arm_stow.phase != "done":
            _logger.info("Arm stow: cancelled; both arms hold where they are")
        self._arm_stow = None
        self._active = {"left": False, "right": False}
        self._ramp = {}
        left, right = self._solver.fk(q_current)
        self._hold_fk = {"left": left, "right": right}
        self._hold_elbow_fk = {}
        if self._use_elbow:
            el = self._solver.elbow_positions(q_current)
            self._hold_elbow_fk = {"left": el[0], "right": el[1]}
        self._clear_freeze()
        self._solver.set_posture_pose(np.asarray(q_current, dtype=np.float32).copy())
        self._last_solve_t = None

    def _shoulder_mid(self) -> np.ndarray:
        """The midpoint between the two shoulders (the stow's reference)."""
        shoulders = self._solver.shoulder_positions
        return (
            np.asarray(shoulders["left"], np.float64)
            + np.asarray(shoulders["right"], np.float64)
        ) * 0.5

    def _stow_elbow_out(self, q: np.ndarray) -> float:
        """How far (m) the left elbow sits outboard of its shoulder at ``q``.

        The stow plans mirrored arms, so the left one speaks for both.
        """
        arm = self._solver.elbow_swivel(q)["left"]
        return float(arm.elbow[1] - arm.shoulder[1])

    def _solve_stow_arms(
        self, seed: np.ndarray, targets: dict[str, Pose]
    ) -> tuple[np.ndarray, float]:
        """Solve the stowed pose for ``targets``: ``(q, worst gripper miss)``.

        The left arm is solved with its elbow swung toward the base and the
        right arm is its mirror image (``q_right = -q_left``). A tool whose
        mirrored right gripper doesn't land in its own slot (the miss says
        so) falls back to the closest of the two.
        """
        solver = self._solver
        li, ri = solver.left_indices, solver.right_indices
        q = np.asarray(seed, dtype=np.float32).copy()
        angle = math.radians(_STOW_ELBOW_DEG)
        pos, rot = targets["left"]
        for k in range(_STOW_PLAN_ITERS):
            q_prev = q
            arm = solver.elbow_swivel(q)["left"]
            wrist = np.asarray(pos, np.float64) + np.asarray(
                rot, np.float64
            ) @ np.asarray(arm.wrist_in_mount, np.float64)
            hint = elbow_swivel_hint(arm.shoulder, arm.elbow, wrist, 1.0, angle)
            solver.set_posture_pose(q)
            # Both arms, the right against the mirrored hint: the solver's
            # two-arm graph is the one already compiled (its warm-up), so the
            # first stow doesn't stall on a compile.
            q = np.asarray(
                solver.ik(
                    q,
                    left_pose=targets["left"],
                    right_pose=targets["right"],
                    left_elbow_pos=hint,
                    right_elbow_pos=(hint * np.array((1.0, -1.0, 1.0))).astype(
                        np.float32
                    ),
                    delta_scale=4.0,
                    elbow_weight=float(self._config.box_elbow_weight) or 10.0,
                    pose_weight_scale=_BOX_POSE_WEIGHT_SCALE,
                ),
                dtype=np.float32,
            ).copy()
            q[ri] = -q[li]
            settled = float(np.max(np.abs(q[li] - q_prev[li]))) < _STOW_PLAN_SETTLED_RAD
            if settled and k + 1 >= _STOW_PLAN_MIN_ITERS:
                break
        fk = dict(zip(("left", "right"), solver.fk(q)))
        miss = max(
            float(np.linalg.norm(np.asarray(fk[s][0]) - np.asarray(targets[s][0])))
            for s in ("left", "right")
        )
        return q, miss

    def _stow_pose(self, stow: _Stow, now: float) -> tuple[np.ndarray, np.ndarray]:
        """The box pose the stow commands this frame (see :meth:`_start_stow`)."""
        if stow.phase == "move":
            u = (now - stow.t0) / stow.duration
            stow.alpha = smoothstep(min(max(u, 0.0), 1.0))
            if u >= 1.0:
                stow.phase = "hold"
                stow.hold_t0 = now
            return blend_pose(stow.start, stow.goal, stow.alpha)
        return stow.goal

    def _stow_posture(self, stow: _Stow | None, q_current: np.ndarray) -> np.ndarray:
        """The posture the solve is pulled toward (``q_current`` outside a stow)."""
        if stow is None or stow.q_goal is None:
            return q_current
        a = stow.alpha if stow.phase == "move" else 1.0
        return ((1.0 - a) * stow.q_start + a * stow.q_goal).astype(np.float32)

    def _stow_elbows(self, stow: _Stow) -> dict[str, np.ndarray] | None:
        """Elbow hints during a stow: blending into the planned pose's elbows."""
        if self._config.box_elbow_weight <= 0.0 or stow.q_goal is None:
            return None
        a = stow.alpha if stow.phase == "move" else 1.0
        return {
            side: (
                (1.0 - a) * stow.elbows_start[side] + a * stow.elbows_goal[side]
            ).astype(np.float32)
            for side in ("left", "right")
        }

    def _settle_stow(
        self,
        stow: _Stow,
        q_new: np.ndarray,
        q_current: np.ndarray,
        delta_scale: float,
    ) -> None:
        """Once the box is in place, step the arms onto the planned pose.

        The live solve settles near the plan but not on it: the arm/base
        clearance cost nudges the elbows a centimetre or two outboard of
        where the plan put them. The plan already holds the grippers in
        their slots, so for the hold the arms move to it in joint space, at
        up to ``_STOW_SETTLE_RAD_S``, in place of the solve. Not after a
        stall (no plan to settle onto). It keeps holding the plan once the
        stow is done, until the core (which reads the phase off the status a
        few times a second) freezes the pair.
        """
        if stow.phase not in ("hold", "done") or stow.q_goal is None:
            return
        step = _STOW_SETTLE_RAD_S * delta_scale / float(self._config.ik_frequency)
        cur = np.asarray(q_current, dtype=np.float32)
        for idx in (self._solver.left_indices, self._solver.right_indices):
            q_new[idx] = cur[idx] + np.clip(stow.q_goal[idx] - cur[idx], -step, step)

    def _advance_stow(
        self,
        stow: _Stow,
        q_new: np.ndarray,
        q_current: np.ndarray,
        targets: dict[str, Pose],
        now: float,
    ) -> None:
        """Watch for a stall during the move; finish once the arms settle."""
        if stow.phase == "move":
            q_meas = self._measured_q(q_current)
            if q_meas is None:
                return
            start = np.asarray(stow.start[0], np.float64)
            path = np.asarray(stow.goal[0], np.float64) - start
            length = float(np.linalg.norm(path))
            if length < _STOW_STALL_MIN_MOVE_M:
                return
            along = path / length
            (ml, _), (mr, _) = self._solver.fk(q_meas)
            (cl, _), (cr, _) = self._solver.fk(q_new)
            meas = 0.5 * (np.asarray(ml, np.float64) + np.asarray(mr, np.float64))
            cmd = 0.5 * (np.asarray(cl, np.float64) + np.asarray(cr, np.float64))
            progress = float(np.dot(meas - start, along))
            trail = float(np.dot(cmd - start, along)) - progress
            if stow.stall_base is None:
                stow.stall_base = trail
                stow.stall_p0 = progress
            trace = stow.stall_trace
            trace.append((now, progress))
            while len(trace) > 2 and now - trace[1][0] >= _STOW_STALL_WINDOW_S:
                trace.pop(0)
            still = (
                now - trace[0][0] >= _STOW_STALL_WINDOW_S
                and progress - trace[0][1] < _STOW_STALL_STILL_M
            )
            if trail - stow.stall_base > _STOW_STALL_M and still:
                # Pressed against something: stop short of it and hold.
                _, rot = blend_pose(stow.start, stow.goal, stow.alpha)
                moved = progress - stow.stall_p0
                stop = max(moved - _STOW_BACKOFF_M, 0.0)
                center = (start + stop * along).astype(np.float32)
                stow.goal = (center, rot)
                stow.q_goal = None
                stow.phase = "hold"
                stow.hold_t0 = now
                _logger.info(
                    "Box stow: pressed against something %.0f mm into the "
                    "%.0f mm move; holding there",
                    moved * 1000.0,
                    length * 1000.0,
                )
            return
        if stow.phase == "hold":
            held = now - stow.hold_t0
            still = (
                float(np.max(np.abs(q_new - np.asarray(q_current)))) < _STOW_STILL_RAD
            )
            arrived = stow.q_goal is None or (
                float(np.max(np.abs(q_new - stow.q_goal))) < _STOW_ARRIVED_RAD
            )
            if held >= _STOW_HOLD_MAX_S or (
                held >= _STOW_HOLD_MIN_S and still and arrived
            ):
                stow.phase = "done"
                if arrived:
                    _logger.info("Box stow: done")
                else:
                    _logger.warning(
                        "Box stow: done, but the arms stopped short of the "
                        "planned pose (%.2f rad off)",
                        float(np.max(np.abs(q_new - stow.q_goal))),
                    )

    def _measured_q(self, q_current: np.ndarray) -> np.ndarray | None:
        """The arms' measured joints as a full ``q`` (``None`` if not reported)."""
        meas = self._measured
        if meas is None or time.perf_counter() - self._measured_t > _LEAN_RESET_S:
            return None
        pos_l, pos_r = meas[0], meas[1]
        q_meas = np.asarray(q_current, dtype=np.float32).copy()
        for i, gi in enumerate(self.left_indices):
            q_meas[gi] = pos_l[i]
        for i, gi in enumerate(self.right_indices):
            q_meas[gi] = pos_r[i]
        return q_meas if np.all(np.isfinite(q_meas)) else None

    def _box_elbow_hints(
        self, q_current: np.ndarray, targets: dict[str, Pose]
    ) -> dict[str, np.ndarray] | None:
        """Outward elbow hints for the box-mode solve, or ``None`` if disabled.

        Box mode's gripper targets fix each wrist but leave the elbow swivel
        free, and as the grippers close on the box the nearest solution
        swings the elbows into the torso. Each hint's goal is the arm's
        current elbow rotated about its shoulder-to-wrist-centre line
        (:meth:`KinematicsSolver.elbow_swivel`) to ``config.box_elbow_out``
        degrees outboard of straight down (:func:`elbow_swivel_hint`), with
        the wrist centre placed on the gripper *target*, not the measured
        pose, so the hint leads the motion the same way the pose target
        does. The hint fed to the solver (at ``config.box_elbow_weight``)
        is the step toward that goal along the arm's own swivel motion
        (:func:`swivel_step`): the circle is only approximate, and a hint
        on it — swung about the gripper mount, as it once was — is out of
        reach in the angled grasp, where at its weight it pulled each
        gripper half a degree tip-out of its target, the tip off the box
        from the first touch.
        """
        cfg = self._config
        if cfg.box_elbow_weight <= 0.0 or getattr(self, "_mink_backend", False):
            # Mink has no elbow task (nor elbow_swivel); its rest posture
            # attractor steers the swivel instead.
            return None
        angle = math.radians(cfg.box_elbow_out)
        swivel = self._solver.elbow_swivel(q_current)
        hints: dict[str, np.ndarray] = {}
        for side, sign in (("left", 1.0), ("right", -1.0)):
            arm = swivel[side]
            pos, rot = targets[side]
            wrist = np.asarray(pos, np.float64) + np.asarray(
                rot, np.float64
            ) @ np.asarray(arm.wrist_in_mount, np.float64)
            goal = elbow_swivel_hint(arm.shoulder, arm.elbow, wrist, sign, angle)
            hints[side] = swivel_step(
                arm.elbow, goal, arm.direction, _ELBOW_HINT_STEP_M
            )
        return hints

    def _integrate_sticks(self, frame: VRFrame, box: BoxState, now: float) -> None:
        """Accumulate this frame's thumbstick input into the grip width.

        Box mode takes one stick axis, the one Jelly leaves free: the
        right stick's forward/back sets the **width** between the grippers
        (push forward = wider, pull back = narrower, at
        ``box_width_speed``), clamped to ``box_width_min``..``box_width_max``
        — the floor raised, if need be, to where the grippers' bodies would
        meet in this grasp (:meth:`_box_width_min`). Everything else on the
        sticks — the left stick, the right stick's left/right, the stick
        clicks — drives Jelly's wheels and lift exactly as outside box
        mode, so the base moves while a grip leads the pair. The pair's
        position and rotation are the leader hand's job, not the sticks'.
        """
        cfg = self._config
        dt = (
            0.0
            if box.stick_t is None
            else min(max(now - box.stick_t, 0.0), _STICK_MAX_DT_S)
        )
        box.stick_t = now
        if dt <= 0.0:
            return
        # The raw stick axis reads negative pushed forward (WebXR gamepad
        # convention, as Jelly's drive reads it too): forward = wider.
        # Only while forward/back outweighs left/right: a sideways push
        # turns Jelly, and must not change the width (right_stick_axis).
        _, stick_y = right_stick_axis(frame.r_stick_x, frame.r_stick_y)
        width_rate = -_dz(stick_y) * cfg.box_width_speed
        if width_rate:
            box.width = float(
                np.clip(
                    box.width + dt * width_rate,
                    self._box_width_min(),
                    cfg.box_width_max,
                )
            )

    def _rest_fk_poses(
        self,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """Rest-pose FK gripper poses ``(left, right)`` in the base frame."""
        (l_pos, l_rot), (r_pos, r_rot) = self._solver.fk(self.get_rest_q())
        return (
            (np.asarray(l_pos, dtype=np.float64), np.asarray(l_rot, dtype=np.float64)),
            (np.asarray(r_pos, dtype=np.float64), np.asarray(r_rot, dtype=np.float64)),
        )

    def _engage_absolute(
        self,
        l_pos: np.ndarray,
        l_rot: np.ndarray,
        r_pos: np.ndarray,
        r_rot: np.ndarray,
    ) -> bool:
        """Solve the world-anchored base transform + controller→TCP offsets.

        Returns ``False`` (and publishes a ``{"rejected": reason}`` base
        message) when the fit is refused by :meth:`_side_swap_rejection`.

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

        # URDF base frame is FLU: +x = forward, +y = left, +z = up. Base up aligns
        # with world up; the yaw is set so the base-frame anchor-separation
        # direction (projected horizontal — along ±y for the gripper origins)
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

        rejection = self._side_swap_rejection(R_wb, l_rot, r_rot, fk_l[1], fk_r[1])
        if rejection is not None:
            # Leave the previous anchor untouched and let the core disengage:
            # a base fit from swapped rigs would record every pose yawed 180°
            # and column-swapped, which no later step could detect.
            _logger.error("absolute engage rejected: %s", rejection)
            self._abs_base = None
            self._abs_offset = {}
            self._abs_active = False
            self.abs_base_msg = {"rejected": rejection}
            return False

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
        return True

    def _side_swap_rejection(
        self,
        R_wb: np.ndarray,
        l_rot: np.ndarray,
        r_rot: np.ndarray,
        fk_l_rot: np.ndarray,
        fk_r_rot: np.ndarray,
    ) -> str | None:
        """Detect left/right rigs held in the opposite hands at engage.

        The base yaw comes from the measured right→left gripper direction, so
        if the tracker bound as "left" is on the rig in the operator's right
        hand the fitted base faces *backwards*: every recorded pose is yawed
        180° and the columns are swapped — and nothing downstream can tell.
        The gripper orientations expose it: with both sides mapped through a
        tracker→gripper transform, an operator holding the rigs like the
        robot's rest grippers has each gripper's horizontal heading (its +y
        axis, which points backwards at rest) well within 120° of the rest
        FK heading; a swap flips *both* headings by ~180°. Only both sides
        agreeing counts — one twisted wrist is the operator's business, and
        an uncalibrated side has its alignment error absorbed anyway.
        Headings too close to vertical (rig pointed straight up/down) are
        inconclusive and never reject.
        """
        if not ("left" in self._tcp_transforms and "right" in self._tcp_transforms):
            return None
        up = np.array([0.0, 0.0, 1.0])
        flipped: list[str] = []
        for side, rot, fk_rot in (
            ("left", l_rot, fk_l_rot),
            ("right", r_rot, fk_r_rot),
        ):
            heading = (R_wb.T @ rot)[:, 1]
            heading_h = heading - np.dot(heading, up) * up
            rest = np.asarray(fk_rot)[:, 1]
            rest_h = rest - np.dot(rest, up) * up
            n_h, n_r = float(np.linalg.norm(heading_h)), float(np.linalg.norm(rest_h))
            if n_h < _SWAP_GUARD_MIN_HORIZONTAL or n_r < 1e-6:
                return None
            if float(np.dot(heading_h, rest_h)) / (n_h * n_r) > _SWAP_GUARD_COS:
                return None
            flipped.append(side)
        if len(flipped) < 2:
            return None
        return (
            "both grippers face away from the robot's rest heading. The tracker "
            "bound as LEFT is most likely on the rig in your RIGHT hand (or the "
            "rigs' gripper channels are swapped): re-run `axol tracker.identify` "
            "with each tracker on the rig it is mounted to, or swap the rigs, "
            "then release both grips and engage again."
        )

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

    def absolute_reply_base_msg(self) -> dict[str, Any] | None:
        """:attr:`abs_base_msg` plus the live calibration while engaged.

        The ``"calibration"`` entry (:meth:`AbsoluteCalibration.to_message`)
        is present only while an engage fit is active, so a reply to any
        disengaged frame withdraws it: the core must never map later tracker
        poses through a fit the next engage is about to replace.
        """
        msg = self.abs_base_msg
        if not (
            self._abs_active
            and self._abs_base is not None
            and isinstance(msg, dict)
            and "rejected" not in msg
        ):
            return msg
        R_wb, t_wb = self._abs_base
        calibration = AbsoluteCalibration(
            base=(np.asarray(t_wb, dtype=np.float64), np.asarray(R_wb, np.float64)),
            tcp={side: self._tcp_transforms.get(side) for side in ("left", "right")},
            offset={
                side: (
                    np.asarray(self._abs_offset[side][0], dtype=np.float64),
                    np.asarray(self._abs_offset[side][1], dtype=np.float64),
                )
                for side in ("left", "right")
            },
        )
        return {**msg, "calibration": calibration.to_message()}

    @staticmethod
    def _encode_tcp_msg(
        left: tuple[np.ndarray, np.ndarray],
        right: tuple[np.ndarray, np.ndarray],
        *,
        out_of_reach: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Pack two base-frame ``(pos, rot)`` poses as JSON-safe pos+quat lists.

        ``out_of_reach`` lists the sides whose target exceeded the reach
        soft-clamp (see :meth:`_out_of_reach`); it is omitted when empty.
        """

        def _enc(pose: tuple[np.ndarray, np.ndarray]) -> list[float]:
            pos, rot = pose
            quat = _matrix_to_quat_xyzw(np.asarray(rot, dtype=np.float64))
            return [float(pos[0]), float(pos[1]), float(pos[2]), *quat]

        msg: dict[str, Any] = {"left": _enc(left), "right": _enc(right)}
        if out_of_reach:
            msg["out_of_reach"] = list(out_of_reach)
        return msg

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

    def raw_tcp_mapping(self) -> dict[str, Any] | None:
        """Export the current engage's rigid raw tracker→base TCP mapping.

        Collection applies it on receipt, before virtual IK and its filters.
        Ordinary robot teleoperation still consumes the usual solved reply.
        """
        if not self._abs_active or self._abs_base is None:
            return None
        rotation, position = self._abs_base
        offsets = {}
        for side in ("left", "right"):
            mount_pos, mount_rot = self._tcp_transforms.get(
                side, (np.zeros(3), np.eye(3))
            )
            engage_pos, engage_rot = self._abs_offset[side]
            offsets[side] = (
                mount_pos + mount_rot @ engage_pos,
                mount_rot @ engage_rot,
            )
        message = self._encode_tcp_msg(offsets["left"], offsets["right"])
        message["base"] = [*map(float, position), *_matrix_to_quat_xyzw(rotation)]
        for side in ("left", "right"):
            message[f"shoulder_{side}"] = list(
                map(float, self._solver.shoulder_positions[side])
            )
        message["reach_limit"] = float(self._solver.config.reach_soft_start)
        return message

    def _note_raw(self, raw_l: np.ndarray, raw_r: np.ndarray, t_eff: float) -> None:
        """Fold a good frame into the raw-tracking state (position + EMA velocity)."""
        if self._prev_raw and self._prev_raw_t is not None:
            dt = min(max(t_eff - self._prev_raw_t, 0.002), 0.1)
            for side, raw in (("left", raw_l), ("right", raw_r)):
                v = (raw - self._prev_raw[side]) / dt
                self._raw_vel[side] = 0.7 * self._raw_vel[side] + 0.3 * v
        else:
            self._raw_vel = {"left": np.zeros(3), "right": np.zeros(3)}
        self._prev_raw = {"left": raw_l.copy(), "right": raw_r.copy()}
        self._prev_raw_t = t_eff

    def _frame_snap_verdict(
        self, raw_l: np.ndarray, raw_r: np.ndarray, t_s: float | None
    ) -> tuple[str, np.ndarray | None, np.ndarray | None]:
        """Classify this frame's raw poses: ``("ok"|"hold"|"shift", off_l, off_r)``.

        Detection compares each hand against a constant-velocity prediction
        from its EMA velocity; *both* hands missing it by more than a noise
        floor plus plausible-acceleration displacement opens a suspect window
        (see the module constants). During the window every frame returns
        ``"hold"`` — the caller quarantines it — while the offsets against the
        pre-trigger prediction accumulate. The window resolves three ways:

        * the offset collapses back under the threshold → the glitch was a
          transient blip; the quarantined frames are discarded ("ok");
        * the offset kept *growing* → genuine motion that beat the predictor
          (hard bimanual flick); tracking resumes from the live pose ("ok");
        * the offset is *stable* → a persistent world-frame shift; returns
          ``"shift"`` with the per-hand offsets so the caller can slide the
          engage anchors and keep the EE targets continuous.
        """
        if t_s is not None:
            t_eff = t_s
        elif self._prev_raw_t is not None:
            t_eff = self._prev_raw_t + 1.0 / self._config.ik_frequency
        else:
            t_eff = 0.0

        if self._suspect is not None:
            s = self._suspect
            gap = min(max(t_eff - s["t0"], 0.002), 0.5)
            threshold = _SNAP_FLOOR_M + 0.5 * _SNAP_ACCEL_MAX * gap * gap
            off_l = raw_l - (s["pos"]["left"] + s["vel"]["left"] * gap)
            off_r = raw_r - (s["pos"]["right"] + s["vel"]["right"] * gap)
            if (
                float(np.linalg.norm(off_l)) < threshold
                or float(np.linalg.norm(off_r)) < threshold
            ):
                _logger.info(
                    "VR tracking blip (%d frames) reverted — discarded.", s["n"]
                )
                self._suspect = None
                self._note_raw(raw_l, raw_r, t_eff)
                return ("ok", None, None)
            s["offs"].append((off_l, off_r))
            s["n"] += 1
            if s["n"] < _SNAP_CONFIRM_FRAMES:
                return ("hold", None, None)

            # Window full: a stable offset means the world frame moved; a
            # growing one means the hands are genuinely accelerating beyond
            # the predictor. Both hands must agree for a shift.
            def _stable(first: np.ndarray, last: np.ndarray) -> bool:
                size = 0.5 * float(np.linalg.norm(first) + np.linalg.norm(last))
                return float(np.linalg.norm(last - first)) < _SNAP_STABLE_RATIO * size

            first_l, first_r = s["offs"][0]
            last_l, last_r = s["offs"][-1]
            is_shift = _stable(first_l, last_l) and _stable(first_r, last_r)
            vel = dict(s["vel"])
            self._suspect = None
            self._prev_raw = {"left": raw_l.copy(), "right": raw_r.copy()}
            self._prev_raw_t = t_eff
            if is_shift:
                # The hands continue their pre-shift motion in the new frame.
                self._raw_vel = vel
                _logger.warning(
                    "VR world frame shifted %.0f/%.0f mm (L/R) — headset "
                    "re-localization, not hand motion. Re-anchoring engaged "
                    "arms in place.",
                    float(np.linalg.norm(last_l)) * 1e3,
                    float(np.linalg.norm(last_r)) * 1e3,
                )
                return ("shift", last_l, last_r)
            self._raw_vel = {"left": np.zeros(3), "right": np.zeros(3)}
            _logger.info(
                "VR pose discontinuity resolved as genuine motion (offset "
                "grew %.0f→%.0f mm); resuming.",
                float(np.linalg.norm(first_l)) * 1e3,
                float(np.linalg.norm(last_l)) * 1e3,
            )
            return ("ok", None, None)

        prev_l = self._prev_raw.get("left")
        prev_r = self._prev_raw.get("right")
        if prev_l is not None and prev_r is not None and self._prev_raw_t is not None:
            dt = min(max(t_eff - self._prev_raw_t, 0.002), 0.1)
            threshold = _SNAP_FLOOR_M + 0.5 * _SNAP_ACCEL_MAX * dt * dt
            off_l = raw_l - (prev_l + self._raw_vel["left"] * dt)
            off_r = raw_r - (prev_r + self._raw_vel["right"] * dt)
            if (
                float(np.linalg.norm(off_l)) > threshold
                and float(np.linalg.norm(off_r)) > threshold
            ):
                self._suspect = {
                    "t0": self._prev_raw_t,
                    "pos": {"left": prev_l.copy(), "right": prev_r.copy()},
                    "vel": {
                        "left": self._raw_vel["left"].copy(),
                        "right": self._raw_vel["right"].copy(),
                    },
                    "offs": [(off_l, off_r)],
                    "n": 1,
                }
                return ("hold", None, None)
        self._note_raw(raw_l, raw_r, t_eff)
        return ("ok", None, None)

    def _clear_freeze(self, side: str | None = None) -> None:
        """Forget one or all in-progress freeze runs."""
        if side is None:
            self._freeze_since.clear()
            self._freeze_targets.clear()
            return
        self._freeze_since.pop(side, None)
        self._freeze_targets.pop(side, None)

    def _note_solve(
        self,
        side: str,
        frozen: bool,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        target_elbow: np.ndarray | None,
    ) -> bool:
        """Track one arm's seed-returning solves; request a safe clutch.

        A single unchanged solution is normal (e.g. the hand is still, or the
        target is held against a constraint). The failure mode worth handling
        is a *run* of solves that returns this arm's seed while its EE/elbow
        target keeps moving away. Once confirmed, the caller re-snapshots the
        controller against FK of the held joints, acting like an automatic
        clutch: no command step now, no accumulated catch-up later.

        Args:
            side: ``"left"`` or ``"right"``.
            frozen: True when this arm's solved joint slice returned its
                ``q_current`` slice bit-identically.
            target_pos: Current EE target position in metres.
            target_rot: Current EE target rotation matrix.
            target_elbow: Optional elbow target position in metres.

        Returns:
            True once the freeze duration and target-motion thresholds are met;
            the caller should re-anchor this arm at the current sample.
        """
        if not frozen:
            self._clear_freeze(side)
            return False
        now = time.monotonic()
        start = self._freeze_targets.get(side)
        if side not in self._freeze_since or start is None:
            self._freeze_since[side] = now
            self._freeze_targets[side] = (
                target_pos.copy(),
                target_rot.copy(),
                None if target_elbow is None else target_elbow.copy(),
            )
            return False

        duration = now - self._freeze_since[side]
        start_pos, start_rot, start_elbow = start
        pos_drift = float(np.linalg.norm(target_pos - start_pos))
        if target_elbow is not None and start_elbow is not None:
            pos_drift = max(
                pos_drift,
                float(np.linalg.norm(target_elbow - start_elbow)),
            )
        relative_rot = start_rot.T @ target_rot
        cos_angle = float(np.clip((np.trace(relative_rot) - 1.0) * 0.5, -1.0, 1.0))
        rot_drift = math.acos(cos_angle)
        moved = (
            pos_drift >= _FREEZE_MIN_TARGET_DRIFT_M
            or rot_drift >= _FREEZE_MIN_TARGET_DRIFT_RAD
        )
        if duration < _FREEZE_WARN_AFTER_S or not moved:
            return False

        _logger.warning(
            "IK %s arm frozen for %.1fs: its solver output stayed at the seed "
            "while the EE/elbow target moved %.0f mm / %.1f deg (likely a "
            "self-collision or joint-limit conflict). Re-anchoring the "
            "controller at the held pose; motion accumulated during the "
            "freeze is discarded to prevent a catch-up lurch.",
            side,
            duration,
            pos_drift * 1e3,
            math.degrees(rot_drift),
        )
        self._clear_freeze(side)
        return True

    def _reset_pose_filters(self) -> None:
        """Clear the pose-filter state for every controller and elbow stream."""
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
        l_pose, r_pose = self._solver.fk(q)
        l_elbow, r_elbow = self._solver.elbow_positions(q)

        for _ in range(max_iterations):
            self._solver.set_posture_pose(q)
            q_new = self._solver.ik(
                q,
                left_pose=l_pose,
                right_pose=r_pose,
                left_elbow_pos=l_elbow if self._use_elbow else None,
                right_elbow_pos=r_elbow if self._use_elbow else None,
            )
            if float(np.max(np.abs(q_new - q))) < tol:
                return q_new
            q = q_new
        return q

    def _snap_arm(
        self,
        side: str,
        ctrl_pos: np.ndarray,
        ctrl_rot: np.ndarray,
        ctrl_e: np.ndarray | None,
        ee_pose: tuple[np.ndarray, np.ndarray],
        elbow_pos: np.ndarray | None,
    ) -> None:
        """Snapshot one arm's controller and FK poses at its engage edge.

        These snapshots become the origin against which that controller's
        subsequent motion is measured to build relative EE and elbow targets
        in :meth:`step`. The elbow snapshots are ``None`` when elbow tracking
        is disabled (``kinematics.elbow_weight == 0``) and are never read.
        """
        self._snap_ctrl[side] = (ctrl_pos, ctrl_rot)
        self._snap_fk[side] = ee_pose
        getattr(self, "_deck_plane", {}).pop(side, None)
        self._snap_elbow_ctrl[side] = ctrl_e
        if elbow_pos is not None:
            self._snap_elbow_fk[side] = elbow_pos
        # A fresh snap makes the target equal FK: nothing left to ramp to.
        self._ramp.pop(side, None)

    def _start_ramp(
        self,
        side: str,
        ctrl_pos: np.ndarray,
        ctrl_rot: np.ndarray,
        ee_pose: tuple[np.ndarray, np.ndarray],
        now: float,
    ) -> None:
        """Begin a "ramp" re-engage: blend ``side`` from ``ee_pose`` to its target.

        The arm's existing snap is kept as the controller↔arm mapping, so the
        target is wherever that mapping says the controller is *now*; the
        blend duration is paced by the distance and turn to cover
        (``reengage_ramp_speed`` / :data:`_RAMP_ANG_SPEED`) with a floor of
        ``reengage_ramp_min_s`` so a short hop is still eased, not stepped.
        """
        cfg = self._config
        goal = _relative_target_np(
            ctrl_pos,
            ctrl_rot,
            *self._snap_ctrl[side],
            *self._snap_fk[side],
            position_multiplier=cfg.position_multiplier,
            rotation_multiplier=cfg.rotation_multiplier,
        )
        dist = float(np.linalg.norm(goal[0] - ee_pose[0]))
        angle = rotation_angle(ee_pose[1], goal[1])
        duration = max(
            cfg.reengage_ramp_min_s,
            dist / max(cfg.reengage_ramp_speed, 1e-3),
            angle / _RAMP_ANG_SPEED,
        )
        self._ramp[side] = (ee_pose, now, duration)
        _logger.info(
            "%s arm re-engaged (ramp): %.0f mm / %.0f° to the controller over %.1fs",
            side,
            dist * 1e3,
            math.degrees(angle),
            duration,
        )


# ---------------------------------------------------------------------------
# Subprocess entry point
# ---------------------------------------------------------------------------


def configure_ik_process(backend: str = "jax") -> None:
    """Set up this process the way the IK runs it (``backend``: the
    kinematics backend; the XLA flag only matters to JAX).

    Confine the JAX solve to a single core's worth of compute. The per-arm IK
    is tiny, but XLA's CPU backend fans its Eigen thread pool across *every*
    core for each solve; combined with the worker's nice(-10) boost, that
    burst preempts the control loop's CAN round-trip and the video relay on
    every step — exactly the engaged-only send/act latency spikes and grainy
    frames seen in `collect-data`, which (unlike teleop) has no spare core
    headroom once the relay's raw-frame branch is running. Single-threaded XLA
    is no slower for a problem this small and leaves the real-time loop alone.

    Must run before the first JAX op (backend init reads XLA_FLAGS lazily).
    The flags are part of the compilation cache's key, so anything that
    precompiles for the worker (:mod:`.warm`) calls this too.
    """
    xla = os.environ.get("XLA_FLAGS", "")
    if backend == "jax" and "xla_cpu_multi_thread_eigen" not in xla:
        os.environ["XLA_FLAGS"] = f"{xla} --xla_cpu_multi_thread_eigen=false".strip()
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


def run_ik_worker(
    conn: multiprocessing.connection.Connection,
    config: VRTeleopConfig,
    kinematics_config: KinematicsConfig,
    q_current_left: np.ndarray | None = None,
    q_current_right: np.ndarray | None = None,
    record_raw_tcp: bool = False,
) -> None:
    """IK subprocess entry point.

    Startup sends ``("ready", q, left_indices, right_indices, startup_traj)``,
    or ``("startup_error", reason)`` and exits when the planner rejects the
    startup move (nothing has moved).

    Message protocol (after the ``("ready", …)`` handshake):

    - ``VRFrame``                      → ``(q, status, near)`` (one solve
      step; ``status`` is :meth:`IKWorker.pair_status` every
      ``_STATUS_EVERY_N`` frames, else ``None``; ``near`` is
      :meth:`IKWorker.collision_proximity` every ``_PROXIMITY_EVERY_N``
      frames while an arm is engaged, else ``None``)
    - ``("set", key, value)``          → *(no reply)* — live update of one
      :class:`VRTeleopConfig` field on this process's copy of the config
      (``VRTeleopCore.set_live``).
    - ``("meas", left, right, kp_l, kp_r)`` → (no reply; see IKWorker.note_measured)
    - ``("clear",)``                   → ``("cleared", q, moved)`` — one
      :meth:`IKWorker.clear_lift` step on the worker's joint vector while
      the arms aren't tracking and no frames arrive (the lift lowered from
      the control panel); ``moved`` is whether it changed.
    - ``("reset", q_current)``         → ``("reset_traj", q_rest, traj)``;
      ``q_rest`` is :meth:`IKWorker.rest_goal` (the rest pose, clear of
      Jelly's deck at the live lift height)
    - ``("reset", q_current, goal)``   → ``("reset_traj", goal, traj)`` —
      an explicit joint target for the second (zero) leg of a guarded park.
    - ``("reset_path", q_current, goals)`` → ``("reset_traj", last, traj)`` —
      one trajectory chained through named goals (``"rest"`` / ``"zero"``,
      see :meth:`IKWorker.compute_path_trajectory`); ``last`` is the final
      goal's joint vector.
    - A ``reset`` / ``reset_path`` the planner rejects (Mink: limits,
      self-collision, unreached endpoint) replies ``("reset_error", reason)``
      without motion; the worker stays up for a retry.
    - ``("sync", pos_left, pos_right)`` → ``("synced", q)`` — seat the worker's
      joint vector at the robot's measured arm positions (7 arm joints per
      side; any gripper element past index 6 is ignored) and clear the engage
      state via :meth:`IKWorker.reset`, so the *next* engage snapshots FK at
      the robot's actual pose instead of wherever the worker last solved.
      Used by the DAgger takeover (see :mod:`almond_axol.teleop.dagger`):
      after a policy has moved the arms, the worker's own last solution is
      stale, and engaging against it would drag the robot back toward it.
    - ``None``                         → exit

    Absolute replies add the base transform and current TCP target. With
    ``record_raw_tcp``, a fifth item exports the engage calibration so Mantis
    collection can map each raw received sample without waiting for IK.
    """
    # Ctrl-C is owned by the parent, which first disables physical outputs and
    # then sends the pipe sentinel / terminate / kill sequence. Terminal and
    # process-group SIGINT otherwise interrupts JAX startup in this child and
    # emits a several-page traceback while racing the parent's ownership
    # checks. SIGTERM/SIGKILL retain their defaults for the bounded fallback.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # A spawned child starts with no logging set up, so only warnings would
    # reach stderr (the parent's log), bare. Route this module's info lines
    # (box stow plans and stops) there too, in the parent's format.
    if not logging.getLogger().handlers and not _logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        _logger.addHandler(handler)
        _logger.setLevel(logging.INFO)
        _logger.propagate = False

    configure_ik_process(getattr(kinematics_config, "backend", "jax"))

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

    # Start through zero — the arms' natural resting pose, which the session
    # start check has already confirmed they are near — so the move out to the
    # rest pose always follows the same planned path.
    try:
        startup_traj = worker.compute_path_trajectory(q_start, ("zero", "rest"))
    except (RuntimeError, ValueError) as exc:
        # Mink refuses paths it cannot validate (limits, self-collision, an
        # unreached endpoint). Nothing has moved: tell the parent why instead
        # of dying with a bare exit code.
        _logger.error("Startup move rejected: %s", exc)
        conn.send(("startup_error", str(exc)))
        return
    q = startup_traj[-1].copy() if startup_traj else q_rest.copy()

    conn.send(
        ("ready", q.copy(), worker.left_indices, worker.right_indices, startup_traj)
    )

    # Startup compile/settle/trajectory are done and the handshake is sent: narrow
    # to the dedicated IK core so per-frame solves aren't preempted by recording
    # load (on <8-core hosts this collapses onto the realtime cores).
    affinity.pin_ik()

    frames = 0
    while True:
        try:
            msg = conn.recv()
            if msg is None:
                break
            if isinstance(msg, tuple) and msg[0] == "set":
                worker.set_config(str(msg[1]), msg[2])
            elif isinstance(msg, tuple) and msg[0] == "lift":
                worker.set_lift(float(msg[1]))
            elif isinstance(msg, tuple) and msg[0] == "grip":
                worker.set_gripper(float(msg[1]), float(msg[2]))
            elif isinstance(msg, tuple) and msg[0] == "clear":
                q_clear = worker.clear_lift(q)
                moved = not np.array_equal(q_clear, q)
                q = np.asarray(q_clear, dtype=np.float32).copy()
                conn.send(("cleared", q.copy(), moved))
            elif isinstance(msg, tuple) and msg[0] == "meas":
                worker.note_measured(msg[1], msg[2], msg[3], msg[4])
            elif isinstance(msg, tuple) and msg[0] == "reset":
                try:
                    if len(msg) not in (2, 3):
                        raise ValueError(
                            "reset requires current joints and an optional goal"
                        )
                    q_current = np.asarray(msg[1], dtype=np.float32)
                    q_target = (
                        np.asarray(msg[2], dtype=np.float32)
                        if len(msg) == 3
                        else worker.rest_goal()
                    )
                    if (
                        q_current.shape != q_rest.shape
                        or q_target.shape != q_rest.shape
                        or not np.isfinite(q_current).all()
                        or not np.isfinite(q_target).all()
                    ):
                        raise ValueError(
                            "reset requires finite current/target joint vectors"
                        )
                    traj = worker.compute_reset_trajectory(q_current, q_target)
                except (RuntimeError, ValueError) as exc:
                    # Reject the entire plan while retaining the worker and its
                    # last command. The parent keeps torque and can retry from
                    # a hand-guided measured pose.
                    conn.send(("reset_error", str(exc)))
                    continue
                worker.reset()
                q = traj[-1].copy() if traj else q_target.copy()
                conn.send(("reset_traj", q_target.copy(), traj))
            elif isinstance(msg, tuple) and msg[0] == "reset_path":
                q_current = np.asarray(msg[1], dtype=np.float32)
                goals = tuple(msg[2]) if len(msg) == 3 else ()
                if (
                    len(msg) != 3
                    or not goals
                    or any(g not in PATH_GOALS for g in goals)
                    or q_current.shape != q_rest.shape
                    or not np.isfinite(q_current).all()
                ):
                    raise ValueError(
                        "reset_path requires a finite current joint vector and "
                        f"named goals from {PATH_GOALS}"
                    )
                try:
                    traj = worker.compute_path_trajectory(q_current, goals)
                except (RuntimeError, ValueError) as exc:
                    # Same contract as "reset": a rejected plan moves nothing.
                    conn.send(("reset_error", str(exc)))
                    continue
                worker.reset()
                q_target = worker.path_goal(goals[-1])
                q = traj[-1].copy() if traj else q_target.copy()
                conn.send(("reset_traj", q_target.copy(), traj))
            elif isinstance(msg, tuple) and msg[0] == "sync":
                pos_l = np.asarray(msg[1], dtype=np.float32)
                pos_r = np.asarray(msg[2], dtype=np.float32)
                for i, gi in enumerate(worker.left_indices):
                    q[gi] = pos_l[i]
                for i, gi in enumerate(worker.right_indices):
                    q[gi] = pos_r[i]
                # Deactivate the engage state and drop the stale snap and
                # frozen-hold poses so the next engage performs a fresh
                # engage-snap from the synced q. Deliberately NOT
                # worker.reset(): that would also clear the One Euro pose
                # filters, which step() keeps warm on every frame precisely
                # so an engage isn't a smoothing cold start — and a DAgger
                # takeover is exactly such an engage. The engage rising edge
                # in step() re-pins the posture pose and re-snaps from the
                # warm filtered poses, so nothing else from reset() is needed
                # here.
                worker.clear_engage()
                conn.send(("synced", q.copy()))
            elif isinstance(msg, VRFrame):
                q = worker.step(msg, q)
                if config.absolute_mode:
                    # Absolute (Mantis) mode replies carry the engage-calibrated
                    # base transform (for the headset's URDF overlay) and the
                    # base-frame TCP targets (recorded per dataset row by Mantis
                    # callers that consume solved targets) alongside joints.
                    # Raw collection instead uses the rigid mapping below.
                    reply = (
                        "q",
                        q.copy(),
                        worker.absolute_reply_base_msg(),
                        worker.last_tcp_msg,
                    )
                    if record_raw_tcp:
                        reply += (worker.raw_tcp_mapping(),)
                    conn.send(reply)
                else:
                    # Relative-mode replies pair the joint solution with the
                    # gripper-pair status (box-mode "aligned" cue), refreshed
                    # every _STATUS_EVERY_N frames and None in between, and
                    # the collision proximity the headset's haptics play.
                    frames += 1
                    status = (
                        worker.pair_status(q) if frames % _STATUS_EVERY_N == 0 else None
                    )
                    near = (
                        worker.collision_proximity(q)
                        if (msg.l_lock or msg.r_lock)
                        and frames % _PROXIMITY_EVERY_N == 0
                        else None
                    )
                    conn.send((q.copy(), status, near))
        except (EOFError, KeyboardInterrupt, OSError):
            # OSError covers ConnectionResetError/BrokenPipeError when the
            # parent end closes abruptly (parent crash, or a shutdown that
            # left an in-flight response unread — the close then RSTs this
            # end). Exit cleanly instead of dying with a traceback.
            break
