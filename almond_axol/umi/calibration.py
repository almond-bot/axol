"""Persistence of the per-side, per-tracker UMI tracker→TCP transforms.

``axol umi.calibrate`` writes one full SE(3) transform per rig side *and
tracker identity* to ``~/.almond/umi/tcp_transform.json``::

    {
      "left": {
        "quest":        {"pos": [x, y, z], "quat": [qx, qy, qz, qw],
                         "time_offset_s": ...},
        "survive:T20":  {"pos": [...], "quat": [...], "time_offset_s": ...}
      },
      "right": { ... }
    }

The tracker key is the tracking backend name plus the device identifier when
one exists (``"quest"`` for the headset path, ``"survive:<codename>"`` /
``"ultimate:<mac>"`` for the Vive backends, see :func:`tracker_key_for_side`).
Keying by tracker identity matters because each tracker type has both a
different physical mount on the rig *and* a different device-local frame —
a transform calibrated for one tracker is silently wrong for another.

``time_offset_s`` is informational: the tracker↔FK timestamp offset found by
the calibration's time-offset search (seconds to add to the tracker's stamps
to land on the FK timeline; negative means the tracker stamps its poses late,
the usual driver-callback delay).

The transform is ``T^tracker_gripper``: the gripper (TCP) frame expressed in
the tracker's local frame, exactly the ``(p_off, R_off)`` shape the absolute-
mode IK worker applies as ``p_world_tcp = p_ctrl + R_ctrl @ pos`` /
``R_world_tcp = R_ctrl @ R(quat)``. ``apply_umi_teleop_profile`` loads the
entry for the active tracker into ``VRTeleopConfig`` so both ``teleop --umi``
and ``collect-data --umi`` pick a saved calibration up automatically.

The pre-keying legacy format (``{"left": {"pos": ..., "quat": ...}}``) is
still accepted on load: it surfaces under the :data:`LEGACY_TRACKER_KEY`
pseudo-key with a deprecation warning, and is preserved (nested under that
key) when a new calibration merges into an old file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

_logger = logging.getLogger(__name__)

UMI_TCP_TRANSFORM_FILE = Path.home() / ".almond" / "umi" / "tcp_transform.json"

# Pseudo tracker key under which entries from the legacy (per-side only) file
# format surface on load. Never produced by a fresh calibration.
LEGACY_TRACKER_KEY = "legacy"


def tracker_key_for_side(
    side: str,
    override: str | None = None,
    config_path: Path | None = None,
) -> tuple[str, str]:
    """Identity key of the tracker presumed active on ``side``.

    The key is what per-tracker calibrations are stored and looked up under:
    ``"quest"`` for the headset path (which has no tracker backend config at
    all), otherwise ``"<backend>"`` or ``"<backend>:<device>"`` from the
    saved tracker config (``~/.almond/tracker/config.json``, written by
    ``axol tracker.identify``) — e.g. ``"survive:T20"`` or
    ``"ultimate:<mac>"``.

    The frame source is not observable here (a Quest headset and an ``axol
    tracker.bridge`` connect to the VR server identically), so this is a
    presumption: a tracker config file on disk means the Vive path is in use,
    no file means Quest. ``override`` short-circuits it for operators who
    know better.

    Args:
        side: ``"left"`` or ``"right"`` (the two rigs bind different devices).
        override: Explicit key to use instead of deriving one, or ``None``.
        config_path: Tracker config file to read (default: the real one);
            for tests.

    Returns:
        ``(key, reason)`` — the key plus a human-readable one-liner of how it
        was chosen, for callers to log.
    """
    if override is not None:
        return override, "explicitly requested"
    from ..tracker.config import TRACKER_CONFIG_FILE, load_tracker_config

    path = TRACKER_CONFIG_FILE if config_path is None else config_path
    if not path.exists():
        return (
            "quest",
            f"no tracker backend configured ({path} missing) — "
            "assuming the Quest headset path",
        )
    config = load_tracker_config(path)
    device = config.left if side == "left" else config.right
    key = f"{config.backend}:{device}" if device else config.backend
    return key, f"'{config.backend}' backend configured in {path}"


def _is_legacy_side_entry(entry: object) -> bool:
    """True for a pre-keying side entry (``{"pos": ..., "quat": ...}``)."""
    return isinstance(entry, dict) and "pos" in entry and "quat" in entry


def save_tcp_transforms(
    transforms: dict[str, tuple[np.ndarray, np.ndarray]],
    tracker_keys: dict[str, str],
    time_offsets: dict[str, float] | None = None,
    path: Path = UMI_TCP_TRANSFORM_FILE,
) -> None:
    """Write per-side ``(R_3x3, t_3)`` tracker→TCP transforms as JSON.

    Merges into an existing file so the two sides — and different trackers —
    can be calibrated in separate sessions. Legacy (per-side only) entries
    already in the file are preserved under :data:`LEGACY_TRACKER_KEY`.

    Args:
        transforms: ``{side: (R, t)}`` solved transforms.
        tracker_keys: ``{side: tracker_key}`` identity key each transform was
            calibrated with (see :func:`tracker_key_for_side`).
        time_offsets: Optional ``{side: seconds}`` tracker↔FK time offset
            found during calibration, stored informationally.
        path: Target file (default: the real calibration file); for tests.
    """
    data: dict[str, dict] = {}
    if path.exists():
        try:
            raw = json.loads(path.read_text())
            if isinstance(raw, dict):
                data = raw
        except (OSError, ValueError):
            data = {}
    for side, entry in list(data.items()):
        if _is_legacy_side_entry(entry):
            data[side] = {LEGACY_TRACKER_KEY: entry}
    for side, (rot, pos) in transforms.items():
        entry: dict[str, object] = {
            "pos": [float(v) for v in pos],
            "quat": [float(v) for v in _quat_xyzw(np.asarray(rot))],
        }
        if time_offsets is not None and side in time_offsets:
            entry["time_offset_s"] = float(time_offsets[side])
        side_entries = data.setdefault(side, {})
        if not isinstance(side_entries, dict):
            side_entries = {}
            data[side] = side_entries
        side_entries[tracker_keys[side]] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_tcp_transforms(
    path: Path = UMI_TCP_TRANSFORM_FILE,
) -> dict[str, dict[str, list[float]]]:
    """Load saved transforms as ``{side: {tracker_key: [x, y, z, qx..qw]}}``.

    Entries in the legacy (per-side only) format are accepted under
    :data:`LEGACY_TRACKER_KEY` with a deprecation warning — they predate
    per-tracker keying, so which tracker they were measured with is unknown.

    Returns an empty dict when no calibration exists or the file is invalid
    (callers fall back to the engage-snapshot absorption).
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        _logger.warning("could not read %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, list[float]]] = {}
    legacy_seen = False
    for side in ("left", "right"):
        side_entries = data.get(side)
        if _is_legacy_side_entry(side_entries):
            side_entries = {LEGACY_TRACKER_KEY: side_entries}
            legacy_seen = True
        if not isinstance(side_entries, dict):
            continue
        for key, entry in side_entries.items():
            flat = _flatten_entry(entry)
            if flat is not None:
                out.setdefault(side, {})[key] = flat
    if legacy_seen:
        _logger.warning(
            "%s holds calibration(s) in the legacy per-side format (no tracker "
            "key — measured with an unknown tracker); re-run `axol "
            "umi.calibrate` to migrate to per-tracker entries.",
            path,
        )
    return out


def _flatten_entry(entry: object) -> list[float] | None:
    """Validate one entry into ``[x, y, z, qx, qy, qz, qw]``, else ``None``."""
    if not isinstance(entry, dict):
        return None
    pos = entry.get("pos")
    quat = entry.get("quat")
    if (
        isinstance(pos, list)
        and len(pos) == 3
        and isinstance(quat, list)
        and len(quat) == 4
    ):
        return [float(v) for v in (*pos, *quat)]
    return None


def _quat_xyzw(R: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix → ``(x, y, z, w)`` quaternion, Shepperd's method.

    Branches on the largest of the four squared quaternion components, so it
    is numerically stable for every rotation — the naive w-first formula
    divides by ~0 (and loses the component signs) near 180°.
    """
    import math

    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0  # s = 4w
        return (
            float(R[2, 1] - R[1, 2]) / s,
            float(R[0, 2] - R[2, 0]) / s,
            float(R[1, 0] - R[0, 1]) / s,
            0.25 * s,
        )
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + float(R[0, 0] - R[1, 1] - R[2, 2])) * 2.0  # s = 4x
        return (
            0.25 * s,
            float(R[0, 1] + R[1, 0]) / s,
            float(R[0, 2] + R[2, 0]) / s,
            float(R[2, 1] - R[1, 2]) / s,
        )
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + float(R[1, 1] - R[0, 0] - R[2, 2])) * 2.0  # s = 4y
        return (
            float(R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            float(R[1, 2] + R[2, 1]) / s,
            float(R[0, 2] - R[2, 0]) / s,
        )
    s = math.sqrt(1.0 + float(R[2, 2] - R[0, 0] - R[1, 1])) * 2.0  # s = 4z
    return (
        float(R[0, 2] + R[2, 0]) / s,
        float(R[1, 2] + R[2, 1]) / s,
        0.25 * s,
        float(R[1, 0] - R[0, 1]) / s,
    )
