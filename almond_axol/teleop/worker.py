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
from .config import VRTeleopConfig
from .filter import LagCompensatedLowPass
from .recorder import from_env as _recorder_from_env
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

# ---------------------------------------------------------------------------
# NumPy-only helpers (no JAX dispatch overhead)
# ---------------------------------------------------------------------------


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

        # Jitter flight recorder (AXOL_JITTER_RECORD, see .recorder): taps the
        # solve path at every stage boundary this process owns — raw VR pose,
        # filtered pose, world EE target, IK output.
        n = self._solver.num_joints
        self._rec = _recorder_from_env(
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
            self._active = {"left": False, "right": False}
            self._hold_fk = {}
            self._hold_elbow_fk = {}
            self._clear_freeze()
            return q_current

        was_any = self._active["left"] or self._active["right"]
        if not was_any:
            self._clear_freeze()

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

        snapped = False
        for side, lock, ctrl_pos, ctrl_rot, ctrl_e in (
            ("left", l_lock, left_pos, left_rot, left_e),
            ("right", r_lock, right_pos, right_rot, right_e),
        ):
            if lock:
                if not self._active[side]:
                    self._active[side] = True
                    self._hold_fk.pop(side, None)
                    self._hold_elbow_fk.pop(side, None)
                    self._snap_arm(
                        side,
                        ctrl_pos,
                        ctrl_rot,
                        ctrl_e,
                        _ee(side),
                        _elbow(side) if self._use_elbow else None,
                    )
                    snapped = True
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
            self._solver.set_posture_pose(q_current)
            # An engage snap re-anchors that arm to q_current: return the
            # seed unchanged so the snap frame itself produces no motion
            # (matching the previous whole-session engage behaviour).
            self._clear_freeze()
            return q_current

        pos_mult = self._config.position_multiplier
        rot_mult = self._config.rotation_multiplier

        def _target(
            side: str, ctrl_pos: np.ndarray, ctrl_rot: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            if self._active[side]:
                return _relative_target_np(
                    ctrl_pos,
                    ctrl_rot,
                    *self._snap_ctrl[side],
                    *self._snap_fk[side],
                    position_multiplier=pos_mult,
                    rotation_multiplier=rot_mult,
                )
            return self._hold_fk[side]

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
        now = time.perf_counter()
        delta_scale = 1.0
        if self._last_solve_t is not None:
            elapsed = now - self._last_solve_t
            delta_scale = float(np.clip(elapsed * self._config.ik_frequency, 1.0, 4.0))
        self._last_solve_t = now

        q_new = self._solver.ik(
            q_current,
            left_pose=(tl_pos, tl_rot),
            right_pose=(tr_pos, tr_rot),
            left_elbow_pos=elbow_l,
            right_elbow_pos=elbow_r,
            delta_scale=delta_scale,
        )
        # A frozen arm must not move at all: the hold-pose target keeps the
        # solve consistent (collision terms see the true pose), but the
        # joints themselves are pinned to the seed.
        q_new = np.asarray(q_new, dtype=np.float32).copy()
        if not self._active["left"]:
            q_new[self._solver.left_indices] = q_current[self._solver.left_indices]
        if not self._active["right"]:
            q_new[self._solver.right_indices] = q_current[self._solver.right_indices]
        targets = [tl_pos, tr_pos]
        if elbow_l is not None and elbow_r is not None:
            targets += [elbow_l, elbow_r]
        self._note_solve(
            bool(np.array_equal(q_new, q_current)),
            np.concatenate(targets),
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

    def reset(self) -> None:
        """Deactivate the engage-toggle state and clear snap poses and filter state.

        Call this before replaying a reset trajectory so the next engage
        performs a fresh engage-snap from the current IK pose.
        """
        self._active = {"left": False, "right": False}
        self._hold_fk = {}
        self._hold_elbow_fk = {}
        self._clear_freeze()
        self._snap_ctrl = {}
        self._snap_fk = {}
        self._snap_elbow_ctrl = {}
        self._snap_elbow_fk = {}
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
