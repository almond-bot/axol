"""AX = XB hand-eye solver for the UMI tracker→TCP transform.

The rig (gripper + rigidly mounted tracker) is bolted to the robot's wrist so
the rig gripper coincides with the robot's own gripper frame, and the arm is
swept through a slow trajectory while both streams record:

- ``A_i`` — robot FK gripper poses ``T^base_gripper`` (base frame), and
- ``B_i`` — tracker poses ``T^world_tracker`` (the tracking system's world
  frame, e.g. WebXR y-up).

The rigid mount means ``B_i · X = Y · A_i`` for all ``i``, where ``X =
T^tracker_gripper`` (the unknown we want — the gripper frame expressed in the
tracker's local frame) and ``Y = T^world_base`` (a nuisance transform).
Eliminating ``Y`` between two samples gives the classic hand-eye equation

    (B_i⁻¹ B_j) · X = X · (A_i⁻¹ A_j)

solved in the standard two steps (Park & Martin, 1994): the rotation from the
relative motions' rotation-log vectors via orthogonal Procrustes, then the
translation by linear least squares. Both world conventions cancel in the
relative motions, so no coordinate conversion between the tracker world and
the robot base is needed (or assumed).

Everything here is plain NumPy — no JAX, importable anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Pose = tuple[np.ndarray, np.ndarray]  # (R_3x3, t_3)

# Relative motions with less rotation than this contribute mostly noise to
# the Procrustes fit; skip them.
_MIN_PAIR_ROT_RAD = math.radians(3.0)


def _inv(pose: Pose) -> Pose:
    R, t = pose
    return R.T, -(R.T @ t)


def _mul(a: Pose, b: Pose) -> Pose:
    Ra, ta = a
    Rb, tb = b
    return Ra @ Rb, Ra @ tb + ta


def _rot_log(R: np.ndarray) -> np.ndarray:
    """SO(3) log map: rotation matrix → rotation vector (axis * angle)."""
    cos_theta = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) * 0.5))
    theta = math.acos(cos_theta)
    if theta < 1e-9:
        return np.zeros(3)
    if theta > math.pi - 1e-6:
        # Near-pi: extract the axis from the symmetric part.
        M = (R + np.eye(3)) * 0.5
        axis = np.sqrt(np.maximum(np.diag(M), 0.0))
        # Fix signs from the off-diagonals.
        k = int(np.argmax(axis))
        if axis[k] > 0:
            for i in range(3):
                if i != k and M[k, i] < 0:
                    axis[i] = -axis[i]
        axis /= np.linalg.norm(axis)
        return axis * theta
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (theta / (2.0 * math.sin(theta)))


def _quat_wxyz(R: np.ndarray) -> np.ndarray:
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w > 1e-6:
        return np.array(
            [
                w,
                (R[2, 1] - R[1, 2]) / (4 * w),
                (R[0, 2] - R[2, 0]) / (4 * w),
                (R[1, 0] - R[0, 1]) / (4 * w),
            ]
        )
    x = math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    y = math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    z = math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
    return np.array([0.0, x, y, z])


def _average_rotations(rotations: list[np.ndarray]) -> np.ndarray:
    """Chordal-L2 rotation average via the quaternion eigen method."""
    M = np.zeros((4, 4))
    q0 = _quat_wxyz(rotations[0])
    for R in rotations:
        q = _quat_wxyz(R)
        if float(np.dot(q, q0)) < 0.0:
            q = -q
        M += np.outer(q, q)
    _w, v = np.linalg.eigh(M)
    q = v[:, -1]
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass
class HandEyeResult:
    """Solution + fit diagnostics of :func:`solve_hand_eye`.

    Attributes:
        rotation: ``R_X`` — rotation of the gripper frame in the tracker frame.
        translation: ``t_X`` (m) — gripper origin in the tracker frame.
        world_base: The nuisance ``Y = T^world_base`` estimate ``(R, t)``.
        pos_rms: RMS position residual (m) of ``B_i X`` vs ``Y A_i`` over all
            samples — the end-to-end consistency of the calibration.
        ori_rms_deg: RMS orientation residual (deg), same comparison.
        n_pairs: Relative-motion pairs used by the fit.
        axis_spread: Second-largest singular value of the stacked relative
            rotation axes — 0 means all rotations shared one axis (degenerate:
            the component of ``t_X`` along that axis is unobservable).
    """

    rotation: np.ndarray
    translation: np.ndarray
    world_base: Pose
    pos_rms: float
    ori_rms_deg: float
    n_pairs: int
    axis_spread: float


def solve_hand_eye(
    fk_poses: list[Pose],
    tracker_poses: list[Pose],
    pair_stride: int | None = None,
) -> HandEyeResult:
    """Solve the tracker→gripper transform from time-aligned pose streams.

    Args:
        fk_poses: Robot FK gripper poses ``T^base_gripper`` as ``(R, t)``.
        tracker_poses: Tracker poses ``T^world_tracker``, one per FK sample
            (already time-aligned by the caller).
        pair_stride: Sample separation for the relative-motion pairs. Default
            picks ~an eighth of the stream so pairs carry real rotation.

    Raises:
        ValueError: On mismatched/short streams or degenerate motion (fewer
            than 3 usable pairs).
    """
    if len(fk_poses) != len(tracker_poses):
        raise ValueError(
            f"stream lengths differ: {len(fk_poses)} FK vs {len(tracker_poses)} tracker"
        )
    n = len(fk_poses)
    if n < 10:
        raise ValueError(f"need at least 10 aligned samples, got {n}")
    if pair_stride is None:
        pair_stride = max(1, n // 8)

    alphas: list[np.ndarray] = []  # log of tracker relative rotations
    betas: list[np.ndarray] = []  # log of FK relative rotations
    pairs: list[tuple[Pose, Pose]] = []  # (P, Q) relative motions
    for stride in {pair_stride, max(1, pair_stride // 2), pair_stride * 2}:
        for i in range(0, n - stride, max(1, stride // 2)):
            j = i + stride
            P = _mul(_inv(tracker_poses[i]), tracker_poses[j])
            Q = _mul(_inv(fk_poses[i]), fk_poses[j])
            alpha = _rot_log(P[0])
            beta = _rot_log(Q[0])
            if np.linalg.norm(alpha) < _MIN_PAIR_ROT_RAD:
                continue
            alphas.append(alpha)
            betas.append(beta)
            pairs.append((P, Q))

    if len(pairs) < 3:
        raise ValueError(
            "too little rotation in the sweep for a hand-eye fit "
            f"({len(pairs)} usable relative-motion pairs); sweep with more "
            "wrist rotation."
        )

    # Rotation: alpha_k = R_X beta_k -> orthogonal Procrustes (Kabsch).
    A = np.stack(alphas)  # (K, 3) targets
    Bm = np.stack(betas)  # (K, 3) sources
    H = Bm.T @ A
    U, _S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, float(np.linalg.det(Vt.T @ U.T))])
    R_x = Vt.T @ D @ U.T

    # Degeneracy diagnostic: the relative rotation axes must span >= 2
    # directions for t_X to be fully observable.
    axis_sv = np.linalg.svd(A, compute_uv=False)
    axis_spread = float(axis_sv[1] / axis_sv[0]) if axis_sv[0] > 0 else 0.0

    # Translation: (R_P - I) t_X = R_X t_Q - t_P, stacked over pairs.
    M = np.zeros((3 * len(pairs), 3))
    b = np.zeros(3 * len(pairs))
    for k, (P, Q) in enumerate(pairs):
        M[3 * k : 3 * k + 3] = P[0] - np.eye(3)
        b[3 * k : 3 * k + 3] = R_x @ Q[1] - P[1]
    t_x, *_ = np.linalg.lstsq(M, b, rcond=None)

    # Nuisance Y = T^world_base from every sample, then residuals against it.
    X: Pose = (R_x, t_x)
    y_rots: list[np.ndarray] = []
    y_ts: list[np.ndarray] = []
    for A_i, B_i in zip(fk_poses, tracker_poses):
        Y_i = _mul(_mul(B_i, X), _inv(A_i))
        y_rots.append(Y_i[0])
        y_ts.append(Y_i[1])
    R_y = _average_rotations(y_rots)
    t_y = np.mean(np.stack(y_ts), axis=0)
    Y: Pose = (R_y, t_y)

    pos_errs = []
    ori_errs = []
    for A_i, B_i in zip(fk_poses, tracker_poses):
        lhs = _mul(B_i, X)
        rhs = _mul(Y, A_i)
        pos_errs.append(float(np.linalg.norm(lhs[1] - rhs[1])))
        ori_errs.append(math.degrees(np.linalg.norm(_rot_log(lhs[0].T @ rhs[0]))))

    return HandEyeResult(
        rotation=R_x,
        translation=t_x,
        world_base=Y,
        pos_rms=float(np.sqrt(np.mean(np.square(pos_errs)))),
        ori_rms_deg=float(np.sqrt(np.mean(np.square(ori_errs)))),
        n_pairs=len(pairs),
        axis_spread=axis_spread,
    )
