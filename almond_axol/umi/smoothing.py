"""Zero-phase smoothing of tracked EE pose tracks at episode-save time.

The UMI rig's poses come from a VR tracker, whose broadband measurement noise
(~mm scale) is comparable in magnitude to the true frame-to-frame hand motion.
Intentional hand motion lives well below ~10 Hz, so once the full episode is
buffered we can low-pass the pose track *acausally* (``scipy.signal.filtfilt``
runs the filter forward and backward — zero phase shift, no lag) and remove
most of the noise without distorting the motion. This benefits every consumer
of the absolute stored poses — chunk-relative training (``axol umi.train``),
any externally-derived action representation, latency analysis — which is why
it runs at recording time rather than in a training pipeline.

Only the 6-axis EE pose groups (``*_ee.x ... *_ee.rz``) of ``action`` and
``observation.state`` are touched; grippers, torques, and ``pose_lag`` pass
through. Positions are filtered directly; rotations are filtered in quaternion
space (sign-continuous, renormalized) and converted back to rotation vectors,
so trajectories crossing the rotation-vector wrap stay clean.

Applied by the dataset recorder just before ``save_episode`` when the
``smooth_ee_hz`` recorder config is set (``collect-data --umi`` enables it;
on-robot FK poses come from joint encoders and don't need it). A paused and
resumed episode is filtered as one contiguous track, so a large pose jump
across the gap is smeared over ~1/cutoff seconds around it.
"""

from __future__ import annotations

import logging

import numpy as np

from .relative import quat_xyzw_to_rotvec

_logger = logging.getLogger(__name__)

_FILTER_ORDER = 2


def rotvec_to_quat_xyzw(rotvec: np.ndarray) -> np.ndarray:
    """(..., 4) xyzw quaternion(s) from (..., 3) rotation vector(s)."""
    rv = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rv, axis=-1, keepdims=True)
    half = 0.5 * angle
    # sin(x)/x -> 1 as x -> 0; guard the division, not the result.
    scale = np.where(
        angle > 1e-9, np.sin(half) / np.where(angle > 1e-9, angle, 1.0), 0.5
    )
    return np.concatenate([rv * scale, np.cos(half)], axis=-1)


def _zero_phase_lowpass(track: np.ndarray, fps: float, cutoff_hz: float) -> np.ndarray:
    """filtfilt a (T, D) track along time; identity if too short / cutoff too high."""
    from scipy.signal import butter, filtfilt

    nyquist = fps / 2.0
    if cutoff_hz >= nyquist:
        return track
    b, a = butter(_FILTER_ORDER, cutoff_hz / nyquist)
    padlen = 3 * max(len(a), len(b))
    if track.shape[0] <= padlen:
        return track
    return filtfilt(b, a, track, axis=0)


def smooth_pose6_track(track: np.ndarray, fps: float, cutoff_hz: float) -> np.ndarray:
    """Low-pass a (T, 6) pose track (position + rotation vector), zero phase.

    Positions filter directly. Rotations are lifted to quaternions with sign
    continuity enforced (q and -q are the same rotation; filtering across a
    sign flip would corrupt it), filtered per component, renormalized, and
    mapped back to shortest-representation rotation vectors — the same
    convention the recording path stores.
    """
    track = np.asarray(track, dtype=np.float64)
    pos = _zero_phase_lowpass(track[:, :3], fps, cutoff_hz)

    quat = rotvec_to_quat_xyzw(track[:, 3:6])
    flip = np.cumsum(np.sum(quat[1:] * quat[:-1], axis=-1) < 0.0) % 2
    quat[1:][flip == 1] *= -1.0
    quat = _zero_phase_lowpass(quat, fps, cutoff_hz)
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)

    return np.concatenate([pos, quat_xyzw_to_rotvec(quat)], axis=-1)


def smooth_episode_ee_poses(
    episode_buffer: dict, features: dict, fps: float, cutoff_hz: float
) -> None:
    """Smooth the EE pose dims of a buffered episode in place, pre-save.

    ``episode_buffer`` is LeRobot's per-episode dict (feature key -> list of
    per-frame arrays, plus bookkeeping); ``features`` the dataset feature
    specs (for the per-dimension names). Every 6-axis EE pose group found in
    ``action`` / ``observation.state`` is filtered as one track over the
    episode; all other dims are left untouched.
    """
    from lerobot.utils.constants import ACTION, OBS_STATE

    from .processor import ee_pose_groups

    for key in (ACTION, OBS_STATE):
        rows = episode_buffer.get(key)
        names = features.get(key, {}).get("names")
        groups = ee_pose_groups(list(names) if names else None)
        if not rows or not groups:
            continue
        stacked = np.stack(rows).astype(np.float64)
        pose_dims = [d for g in groups for d in g]
        for dims in groups:
            stacked[:, dims] = smooth_pose6_track(stacked[:, dims], fps, cutoff_hz)
        for i, row in enumerate(rows):
            row[pose_dims] = stacked[i, pose_dims]
        _logger.debug(
            "smoothed %d-frame %s track (%d pose group(s), cutoff %.1f Hz)",
            stacked.shape[0],
            key,
            len(groups),
            cutoff_hz,
        )
