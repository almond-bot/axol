"""Resample tracked EE poses onto camera exposure times at episode-save time.

The recorder pairs each dataset row's camera exposure with the nearest state
snapshot. On the Mantis rig that snapshot is stamped with the tracker pose's
capture time, so the pose written next to an image was sampled a few ms before
or after the exposure — ``observation.pose_lag = exposure - pose_time``.
Measured on recorded Mantis datasets the lag is zero-mean (no bias to remove)
but spreads ~±5 ms (std) with tails past 20 ms, which at fast hand speeds is
several mm and a few degrees of pose↔image misalignment.

Once the full episode is buffered, every row's pose time is known
(``row_capture_ts - pose_lag``), so the EE pose track can be interpolated at
each row's exposure instead: positions linearly, rotations by quaternion slerp.
Only the 6-axis EE pose groups of ``action`` and ``observation.state`` move
(on the rig both carry the same tracked pose); grippers and every other dim are
left untouched, because their sample time is not the tracker's.

Rules that keep the resample honest:

* A held pose (the recorder reused an older snapshot) repeats an earlier pose
  time; only strictly increasing pose times are interpolation sources, so a
  held row is filled from the real samples around it.
* A row whose bracketing samples are more than ``max_gap_s`` apart (a
  tracking dropout, a paused-then-resumed take) or that would need
  extrapolation (the first/last rows) keeps its original pose and lag.
* ``pose_lag`` is rewritten to the residual: 0 for a resampled row, the
  original value for a row left alone — so the column keeps meaning "skew
  between this row's exposure and the pose stored next to it", and a
  downstream consumer that corrects by it never double-corrects.

Runs in the recorder before :mod:`almond_axol.mantis.smoothing` (whose filter
assumes evenly spaced samples — exactly what this produces).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .relative import quat_xyzw_to_rotvec
from .smoothing import rotvec_to_quat_xyzw

_logger = logging.getLogger(__name__)

POSE_LAG_FEATURE = "observation.pose_lag"

# Bracketing pose samples further apart than this many dataset frame periods
# are a gap (dropout / pause), not a sampling jitter; rows inside one keep their
# recorded pose. One held row makes a 2-period bracket, two make 3.
_MAX_GAP_PERIODS = 3.0


@dataclass(frozen=True)
class AlignmentStats:
    rows: int
    aligned: int
    lag_abs_p95_ms: float
    lag_abs_max_ms: float
    max_shift_mm: float
    max_shift_deg: float


def _slerp_xyzw(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Row-wise shortest-path slerp of (N, 4) xyzw quaternions at (N,) ``t``."""
    dot = np.sum(q0 * q1, axis=-1)
    q1 = np.where((dot < 0.0)[:, None], -q1, q1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    small = sin_theta < 1e-9
    safe = np.where(small, 1.0, sin_theta)
    w0 = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / safe)
    w1 = np.where(small, t, np.sin(t * theta) / safe)
    q = w0[:, None] * q0 + w1[:, None] * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _scalar(value: object) -> float:
    return float(np.ravel(np.asarray(value, dtype=np.float64))[0])


def align_episode_ee_poses(
    episode_buffer: dict,
    features: dict,
    row_times: list[float],
    fps: float,
) -> AlignmentStats | None:
    """Resample a buffered episode's EE poses to its rows' exposure times, in place.

    ``episode_buffer`` is LeRobot's per-episode dict (feature key -> list of
    per-frame arrays); ``row_times`` the recorder's per-row camera exposure
    times (``perf_counter`` seconds, one per buffered row). Returns ``None``
    (buffer untouched) when the episode has no ``observation.pose_lag`` column,
    no EE pose groups, or a row count that doesn't match ``row_times``.
    """
    from lerobot.utils.constants import ACTION, OBS_STATE

    from .processor import ee_pose_groups

    lag_rows = episode_buffer.get(POSE_LAG_FEATURE)
    n = len(row_times)
    if not lag_rows or n == 0:
        return None
    if len(lag_rows) != n:
        _logger.warning(
            "pose alignment skipped: %d pose-lag rows but %d exposure times",
            len(lag_rows),
            n,
        )
        return None

    targets: list[tuple[list, list[list[int]]]] = []
    for key in (ACTION, OBS_STATE):
        rows = episode_buffer.get(key)
        names = features.get(key, {}).get("names")
        groups = ee_pose_groups(list(names) if names else None)
        if not rows or not groups:
            continue
        if len(rows) != n:
            _logger.warning(
                "pose alignment skipped: %d %s rows but %d exposure times",
                len(rows),
                key,
                n,
            )
            return None
        targets.append((rows, groups))
    if not targets:
        return None

    t_img = np.asarray(row_times, dtype=np.float64)
    lag = np.array([_scalar(v) for v in lag_rows], dtype=np.float64)
    t_pose = t_img - lag

    # Interpolation sources: rows whose pose time is strictly newer than every
    # earlier row's (drops held repeats and anything non-finite).
    finite = np.isfinite(t_pose) & np.isfinite(t_img)
    running = np.maximum.accumulate(np.where(finite, t_pose, -np.inf))
    prev_max = np.r_[-np.inf, running[:-1]]
    source = finite & (t_pose > prev_max)
    src_idx = np.flatnonzero(source)
    if src_idx.size < 2:
        return None
    src_t = t_pose[src_idx]

    hi = np.searchsorted(src_t, t_img, side="right")
    lo = hi - 1
    in_range = finite & (lo >= 0) & (hi < src_t.size)
    lo_c = np.clip(lo, 0, src_t.size - 1)
    hi_c = np.clip(hi, 0, src_t.size - 1)
    span = src_t[hi_c] - src_t[lo_c]
    max_gap_s = _MAX_GAP_PERIODS / float(fps) if fps > 0 else 0.0
    valid = in_range & (span > 0.0) & (span <= max_gap_s)
    rows_idx = np.flatnonzero(valid)

    abs_lag_ms = np.abs(lag[finite]) * 1e3
    if rows_idx.size == 0:
        return AlignmentStats(
            rows=n,
            aligned=0,
            lag_abs_p95_ms=float(np.percentile(abs_lag_ms, 95))
            if abs_lag_ms.size
            else 0.0,
            lag_abs_max_ms=float(abs_lag_ms.max()) if abs_lag_ms.size else 0.0,
            max_shift_mm=0.0,
            max_shift_deg=0.0,
        )

    a_lo = src_idx[lo[rows_idx]]
    a_hi = src_idx[hi[rows_idx]]
    alpha = (t_img[rows_idx] - t_pose[a_lo]) / (t_pose[a_hi] - t_pose[a_lo])

    max_shift_mm = 0.0
    max_shift_deg = 0.0
    for rows, groups in targets:
        stacked = np.stack(rows).astype(np.float64)
        out = stacked.copy()
        for dims in groups:
            pos_dims, rot_dims = dims[:3], dims[3:]
            p0 = stacked[a_lo][:, pos_dims]
            p1 = stacked[a_hi][:, pos_dims]
            pos = p0 + alpha[:, None] * (p1 - p0)
            q0 = rotvec_to_quat_xyzw(stacked[a_lo][:, rot_dims])
            q1 = rotvec_to_quat_xyzw(stacked[a_hi][:, rot_dims])
            quat = _slerp_xyzw(q0, q1, alpha)
            out[np.ix_(rows_idx, pos_dims)] = pos
            out[np.ix_(rows_idx, rot_dims)] = quat_xyzw_to_rotvec(quat)

            shift = np.linalg.norm(pos - stacked[rows_idx][:, pos_dims], axis=1)
            max_shift_mm = max(max_shift_mm, float(shift.max()) * 1e3)
            q_old = rotvec_to_quat_xyzw(stacked[rows_idx][:, rot_dims])
            cos_half = np.clip(np.abs(np.sum(q_old * quat, axis=-1)), 0.0, 1.0)
            ang = np.degrees(2.0 * np.arccos(cos_half))
            max_shift_deg = max(max_shift_deg, float(ang.max()))

        pose_dims = [d for g in groups for d in g]
        for i in rows_idx:
            rows[i][pose_dims] = out[i, pose_dims]

    for i in rows_idx:
        row = lag_rows[i]
        if isinstance(row, np.ndarray):
            row[...] = 0.0
        else:
            lag_rows[i] = np.zeros(1, dtype=np.float32)

    return AlignmentStats(
        rows=n,
        aligned=int(rows_idx.size),
        lag_abs_p95_ms=float(np.percentile(abs_lag_ms, 95)),
        lag_abs_max_ms=float(abs_lag_ms.max()),
        max_shift_mm=max_shift_mm,
        max_shift_deg=max_shift_deg,
    )
