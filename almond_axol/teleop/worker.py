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

import numpy as np

from ..kinematics.config import KinematicsConfig
from ..kinematics.solver import KinematicsSolver
from ..vr.models import VRFrame
from .box import (
    BoxState,
    Pose,
    blend_pose,
    box_targets,
    elbow_swivel_hint,
    pair_aligned,
    rodrigues,
    rotation_angle,
    smoothstep,
    snap_box,
)
from .config import VRTeleopConfig
from .filter import LagCompensatedLowPass
from .recorder import make as _recorder_make
from .trajectory import plan_collision_aware_trajectory

_logger = logging.getLogger(__name__)

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

# Box-mode jog (see IKWorker._step_box): stick deflections below this are
# ignored so a resting stick never creeps the arm pair, and one integration
# step is capped so a stalled frame stream can't authorise a large jump.
_JOG_DEADZONE = 0.15
_JOG_MAX_DT_S = 0.1
_UP = np.array((0.0, 0.0, 1.0), dtype=np.float32)

# Gripper-pair status (see IKWorker.pair_status): reported to the core every
# this many solved frames (~10 Hz at the 120 Hz cadence), and the tolerance
# (per gripper, from the rotation a box-mode engage would blend it to) for
# calling the two grippers "aligned" into the side-clamping pair.
_STATUS_EVERY_N = 12
_ALIGNED_TOL_DEG = 25.0

# Re-engage ramp (config.reengage == "ramp", see IKWorker.step): the blend from
# the arm's pose at the grip to the controller-implied target is paced by the
# configured linear speed and, for the rotational part, by this angular rate.
_RAMP_ANG_SPEED = 0.6  # rad/s

# ---------------------------------------------------------------------------
# NumPy-only helpers (no JAX dispatch overhead)
# ---------------------------------------------------------------------------


def _dz(v: float) -> float:
    """A thumbstick axis with the jog deadzone applied (``0.0`` when resting)."""
    return 0.0 if abs(v) < _JOG_DEADZONE else float(v)


def _dominant_axis(x: float, y: float) -> tuple[float, float]:
    """Keep only the larger of a thumbstick's two axes (deadzoned).

    For sticks whose axes drive different things (width vs. height), so the
    off-axis leak of a thumb pushed "left" never also moves the other one.
    Ties go to ``x``.
    """
    x, y = _dz(x), _dz(y)
    if abs(x) >= abs(y):
        return x, 0.0
    return 0.0, y


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
        # Elbow hints are optional (kinematics.elbow_weight == 0 disables, the
        # default): skip the whole elbow pipeline — filters, engage snapshots,
        # target math — so the solve graph never carries the cost.
        self._use_elbow = kinematics_config.elbow_weight > 0.0

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
        # Box mode (bimanual carry, see .box): the pair's state from the engage
        # snap and which controller leads it. None while not in box tracking.
        self._box: BoxState | None = None
        self._box_leader: str | None = None

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
        q_settled = self._settle_rest_pose()
        self._rest_pose_left = q_settled[self._solver.left_indices].astype(np.float32)
        self._rest_pose_right = q_settled[self._solver.right_indices].astype(np.float32)
        self._solver.set_posture_pose(self.get_rest_q())

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
            return q_current

        if frame.box_leader is not None:
            return self._step_box(
                frame,
                q_current,
                {"left": (left_pos, left_rot), "right": (right_pos, right_rot)},
                lp,
                rp,
            )
        if self._box is not None:
            # Box tracking ended without a lock-less frame in between (mode
            # switched while engaged): the per-arm path below re-snaps.
            self._box = None
            self._box_leader = None
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
        q_new = self._solver.ik(
            q_current,
            left_pose=(tl_pos, tl_rot),
            right_pose=(tr_pos, tr_rot),
            left_elbow_pos=elbow_l,
            right_elbow_pos=elbow_r,
            delta_scale=delta_scale,
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
            self._solver.set_posture_pose(posture)
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

    def compute_reset_trajectory(
        self, q_current: np.ndarray, q_target: np.ndarray
    ) -> list[np.ndarray]:
        """Collision-aware trajectory. Each item is a full (N,) array in radians."""
        cfg = self._config
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

    def set_config(self, key: str, value: object) -> None:
        """Live-update one :class:`VRTeleopConfig` field (``("set", …)`` message).

        Only fields this process reads at step time are meaningful here
        (multipliers, box jog speeds, ramp pacing); the core validates the
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
        setattr(self._config, key, coerced)

    def pair_status(self, q: np.ndarray) -> dict:
        """Geometry of the gripper pair at ``q`` for the headset's cues.

        ``aligned`` is True when the two grippers already form the box-mode
        side-clamping pair — fingers forward, a flat face toward the other
        gripper, each within ``_ALIGNED_TOL_DEG`` of where a box-mode engage
        would blend it (see :func:`~almond_axol.teleop.box.pair_aligned`) —
        and the gap is inside the box-mode width range, so switching to box
        mode from here costs (almost) no alignment blend. ``width`` is the
        mount separation in metres and ``tilt`` the pair's current inward
        fingertip yaw in degrees (``config.box_grip_tilt``, which the jog
        writes back to).
        """
        left, right = self._solver.fk(q)
        width = float(
            np.linalg.norm(
                np.asarray(right[0], dtype=np.float64)
                - np.asarray(left[0], dtype=np.float64)
            )
        )
        aligned = width > 1e-3 and pair_aligned(
            left,
            right,
            width_min=self._config.box_width_min,
            width_max=self._config.box_width_max,
            tilt=math.radians(self._config.box_grip_tilt),
            tol_deg=_ALIGNED_TOL_DEG,
        )
        return {
            "aligned": bool(aligned),
            "width": round(width, 3),
            "tilt": round(float(self._config.box_grip_tilt), 1),
        }

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
        self._box_leader = None
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
        self._prev_raw = {}
        self._prev_raw_t = None
        self._raw_vel = {}
        self._suspect = None
        self._last_solve_t = None
        self._reset_pose_filters()
        # step() pins posture to q_current on each engage; an explicit reset
        # restores the default rest-pose attractor.
        self._solver.set_posture_pose(self.get_rest_q())

    # -- Internal -----------------------------------------------------------

    def _step_box(
        self,
        frame: VRFrame,
        q_current: np.ndarray,
        ctrl: dict[str, tuple[np.ndarray, np.ndarray]],
        lp: np.ndarray,
        rp: np.ndarray,
    ) -> np.ndarray:
        """Box-mode solve: one controller carries both grippers as a rigid pair.

        On engage the pair is snapped from FK (:func:`snap_box`), the leader
        controller's pose is anchored, and the grippers are blended into the
        side-clamping grasp (fingers forward, flat faces on the box) over
        ``box_align_duration``. Afterwards the leader
        gripper is driven by the usual per-arm clutch mapping
        (:func:`_relative_target_np`, so one hand feels exactly like normal
        teleop) and the box rides rigidly on it; the thumbsticks jog the pair
        in the box's own horizontal frame (see :meth:`_integrate_jog`).
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
            # wherever the grippers are, so nothing jumps.
            l_fk, r_fk = self._solver.fk(q_current)
            self._box = snap_box(
                l_fk,
                r_fk,
                now,
                align_duration=cfg.box_align_duration,
                width_min=cfg.box_width_min,
                width_max=cfg.box_width_max,
                tilt=math.radians(cfg.box_grip_tilt),
            )
            self._box_leader = leader
            self._snap_ctrl = {leader: ctrl[leader]}
            self._snap_fk = {leader: l_fk if leader == "left" else r_fk}
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
        lead_pos, lead_rot = _relative_target_np(
            ctrl_pos,
            ctrl_rot,
            *self._snap_ctrl[leader],
            *self._snap_fk[leader],
            position_multiplier=cfg.position_multiplier,
            rotation_multiplier=cfg.rotation_multiplier,
        )
        # The box is rigidly attached to the leader gripper's snap frame:
        # carry the snap-time box pose along with the leader's world motion.
        snap_pos, snap_rot = self._snap_fk[leader]
        r_delta = lead_rot @ snap_rot.T
        center = lead_pos + r_delta @ (box.center - snap_pos)
        rot = r_delta @ box.rot
        self._integrate_jog(frame, box, rot, now)
        center = (center + box.jog_pos).astype(np.float32)
        if box.jog_yaw:
            rot = rodrigues(_UP, box.jog_yaw) @ rot
        rot = rot.astype(np.float32)
        targets = box_targets(box, center, rot, now)
        elbows = self._box_elbow_hints(q_current, targets)
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
        self._solver.set_posture_pose(q_current)

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
        )
        solve_ms = (time.perf_counter() - solve_t0) * 1000.0
        q_new = np.asarray(q_new, dtype=np.float32).copy()
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

    def _box_elbow_hints(
        self, q_current: np.ndarray, targets: dict[str, Pose]
    ) -> dict[str, np.ndarray] | None:
        """Outward elbow hints for the box-mode solve, or ``None`` if disabled.

        Box mode's gripper targets fix each wrist but leave the elbow swivel
        free, and as the grippers close on the box the nearest solution
        swings the elbows into the torso. Each hint is the arm's current
        elbow rotated about its shoulder-wrist line to ``config.box_elbow_out``
        degrees outboard of straight down (:func:`elbow_swivel_hint`), fed to
        the solver at ``config.box_elbow_weight``. The wrist used is the
        gripper *target*, not the measured pose, so the hint leads the motion
        the same way the pose target does.
        """
        cfg = self._config
        if cfg.box_elbow_weight <= 0.0:
            return None
        angle = math.radians(cfg.box_elbow_out)
        shoulders = dict(zip(("left", "right"), self._solver.shoulder_positions))
        elbows = dict(zip(("left", "right"), self._solver.elbow_positions(q_current)))
        return {
            side: elbow_swivel_hint(
                shoulders[side], elbows[side], targets[side][0], sign, angle
            )
            for side, sign in (("left", 1.0), ("right", -1.0))
        }

    def _integrate_jog(
        self, frame: VRFrame, box: BoxState, rot: np.ndarray, now: float
    ) -> None:
        """Accumulate this frame's thumbstick jog into ``box``.

        Leader stick: forward/back and left/right translate the pair in the
        box's horizontal frame (forward = the box's ``+x``, perpendicular to
        the gripper-to-gripper line) — a free 2-D jog, so diagonals work.
        With the stick clicked in, forward/back moves the pair up/down and
        left/right yaws it about its centre. The *other* stick's forward/back
        moves the pair up/down and its left/right widens / narrows the grasp;
        with *that* stick clicked in, left/right instead tilts the fingertips
        inward (left) / outward (right) — the grippers' yaw toward the box
        centre, ``BoxState.tilt``, written back to ``config.box_grip_tilt`` so
        the next engage starts from it. Where a stick's two axes drive
        *different* things (the clicked leader stick, the other stick) only
        its dominant axis counts, so a thumb pushing "left" with a little
        forward in it changes the width alone and never lifts the pair (see
        :func:`_dominant_axis`). ``rot`` is the box rotation before this
        frame's jog, used to resolve the horizontal frame.
        """
        cfg = self._config
        dt = 0.0 if box.jog_t is None else min(max(now - box.jog_t, 0.0), _JOG_MAX_DT_S)
        box.jog_t = now
        if dt <= 0.0:
            return

        if self._box_leader == "right":
            sx, sy, click = (
                _dz(frame.r_stick_x),
                _dz(frame.r_stick_y),
                frame.r_stick_click,
            )
            ox, oy = _dominant_axis(frame.l_stick_x, frame.l_stick_y)
            o_click = frame.l_stick_click
        else:
            sx, sy, click = (
                _dz(frame.l_stick_x),
                _dz(frame.l_stick_y),
                frame.l_stick_click,
            )
            ox, oy = _dominant_axis(frame.r_stick_x, frame.r_stick_y)
            o_click = frame.r_stick_click
        if click:
            sx, sy = _dominant_axis(sx, sy)
        if not (sx or sy or ox or oy):
            return

        if box.jog_yaw:
            rot = rodrigues(_UP, box.jog_yaw) @ rot
        fwd = np.array((rot[0, 0], rot[1, 0], 0.0))
        n = float(np.linalg.norm(fwd))
        fwd = fwd / n if n > 1e-6 else np.array((1.0, 0.0, 0.0))
        lat = np.cross(_UP, fwd)

        # Sticks report pushed-forward as -1 (WebXR), so negate y for "forward".
        v_fwd = v_lat = v_up = yaw_rate = width_rate = tilt_rate = 0.0
        if click:
            v_up += -sy * cfg.box_jog_speed
            yaw_rate += -sx * cfg.box_jog_yaw_speed  # push right = clockwise
        else:
            v_fwd += -sy * cfg.box_jog_speed
            v_lat += -sx * cfg.box_jog_speed  # push right = move right
        if o_click:
            # Push left = fingertips inward (more pinch), right = outward —
            # the same sense as the width axis (right opens).
            tilt_rate += -ox * math.radians(cfg.box_tilt_speed)
        else:
            v_up += -oy * cfg.box_jog_speed
            width_rate += ox * cfg.box_width_speed  # push right = wider

        box.jog_pos = (
            box.jog_pos + dt * (v_fwd * fwd + v_lat * lat + v_up * _UP)
        ).astype(np.float32)
        box.jog_yaw += dt * yaw_rate
        if width_rate:
            box.width = float(
                np.clip(
                    box.width + dt * width_rate, cfg.box_width_min, cfg.box_width_max
                )
            )
        if tilt_rate:
            limit = math.radians(abs(cfg.box_tilt_max))
            box.tilt = float(np.clip(box.tilt + dt * tilt_rate, -limit, limit))
            # Carry the jogged tilt into the next engage (and pair_status).
            cfg.box_grip_tilt = math.degrees(box.tilt)

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


def run_ik_worker(
    conn: multiprocessing.connection.Connection,
    config: VRTeleopConfig,
    kinematics_config: KinematicsConfig,
    q_current_left: np.ndarray | None = None,
    q_current_right: np.ndarray | None = None,
) -> None:
    """IK subprocess entry point.

    Message protocol (after the ``("ready", …)`` handshake):

    - ``VRFrame``                      → ``(q, status)`` (one solve step;
      ``status`` is :meth:`IKWorker.pair_status` every
      ``_STATUS_EVERY_N`` frames, else ``None``)
    - ``("set", key, value)``          → *(no reply)* — live update of one
      :class:`VRTeleopConfig` field on this process's copy of the config
      (``VRTeleopCore.set_live``).
    - ``("reset", q_current)``         → ``("reset_traj", q_rest, traj)``
    - ``("sync", pos_left, pos_right)`` → ``("synced", q)`` — seat the worker's
      joint vector at the robot's measured arm positions (7 arm joints per
      side; any gripper element past index 6 is ignored) and clear the engage
      state via :meth:`IKWorker.reset`, so the *next* engage snapshots FK at
      the robot's actual pose instead of wherever the worker last solved.
      Used by the DAgger takeover (see :mod:`almond_axol.teleop.dagger`):
      after a policy has moved the arms, the worker's own last solution is
      stale, and engaging against it would drag the robot back toward it.
    - ``None``                         → exit
    """
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

    frames = 0
    while True:
        try:
            msg = conn.recv()
            if msg is None:
                break
            if isinstance(msg, tuple) and msg[0] == "set":
                worker.set_config(str(msg[1]), msg[2])
            elif isinstance(msg, tuple) and msg[0] == "reset":
                q_current = np.asarray(msg[1], dtype=np.float32)
                traj = worker.compute_reset_trajectory(q_current, q_rest)
                worker.reset()
                q = traj[-1].copy() if traj else q_rest.copy()
                conn.send(("reset_traj", q_rest.copy(), traj))
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
                frames += 1
                status = (
                    worker.pair_status(q) if frames % _STATUS_EVERY_N == 0 else None
                )
                conn.send((q.copy(), status))
        except (EOFError, KeyboardInterrupt, OSError):
            # OSError covers ConnectionResetError/BrokenPipeError when the
            # parent end closes abruptly (parent crash, or a shutdown that
            # left an in-flight response unread — the close then RSTs this
            # end). Exit cleanly instead of dying with a traceback.
            break
