"""The Mantis rig's tracker→TCP transforms: factory design constants + overrides.

The rig's tracker mounts are a fixed design, so the rigid tracker→gripper
transform is a **design constant** per tracker family, shipped in
:data:`DESIGN_TCP_TRANSFORMS` and applied out of the box. A per-unit override
file at ``~/.almond/mantis/tcp_transform.json`` takes precedence when present
(hand-measured refinements, non-standard mounts); its shape is one SE(3)
transform per rig side *and tracker identity*::

    {
      "left": {
        "quest":        {"pos": [x, y, z], "quat": [qx, qy, qz, qw]},
        "survive:T20":  {"pos": [...], "quat": [...]}
      },
      "right": { ... }
    }

The tracker key is the tracking backend name plus the device identifier when
one exists (``"quest"`` for the headset path, ``"survive:<codename>"`` /
``"ultimate:<mac>"`` for the Vive backends, see :func:`tracker_key_for_side`);
design defaults are keyed by the backend family alone. Keying by tracker
matters because each tracker type has both a different physical mount on the
rig *and* a different device-local frame — a transform for one tracker type
is silently wrong for another.

The transform is ``T^tracker_gripper``: the gripper (TCP) frame expressed in
the tracker's local frame, exactly the ``(p_off, R_off)`` shape the absolute-
mode IK worker applies as ``p_world_tcp = p_ctrl + R_ctrl @ pos`` /
``R_world_tcp = R_ctrl @ R(quat)``. ``apply_mantis_teleop_profile`` resolves the
entry for the active tracker into ``VRTeleopConfig`` so both ``teleop --mantis``
and ``collect-data --mantis`` pick it up automatically.

The pre-keying legacy file format (``{"left": {"pos": ..., "quat": ...}}``,
written by the retired ``axol mantis.calibrate`` robot-sweep command) is still
accepted on load: it surfaces under the :data:`LEGACY_TRACKER_KEY` pseudo-key
with a deprecation warning.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

_logger = logging.getLogger(__name__)

MANTIS_TCP_TRANSFORM_FILE = Path.home() / ".almond" / "mantis" / "tcp_transform.json"
_PRE_MANTIS_TCP_TRANSFORM_FILE = (
    Path.home() / ".almond" / ("u" + "mi") / "tcp_transform.json"
)

# Pseudo tracker key under which entries from the legacy (per-side only) file
# format surface on load. Never produced by a fresh calibration.
LEGACY_TRACKER_KEY = "legacy"

# Factory (CAD-derived) tracker→gripper transforms for the Mantis rig's
# standard tracker mounts, keyed by tracker backend family — the part of a
# tracker key before the ":" (``"survive:T20"`` → ``"survive"``). These are
# design constants of the rig, identical for every unit up to manufacturing
# tolerance, so they apply out of the box; a per-unit entry in the override
# file always wins over them.
#
# survive (Vive Tracker 3.0, standard mount): derived from the rig CAD
# (2026-08-10) — tracker seated flat, stabilizing-pin recess toward the jaws,
# gripper flange 92 mm forward / 35.5 mm below the tracker's mounting plane,
# gripper pointing forward, jaw travel lateral. Expressed in the bridge's
# tracker frame (libsurvive head frame with the z-up→y-up body relabel).
# The jaw-travel sign gives a 180°-roll alternate candidate
# ([0, 0.7071068, 0.7071068, 0]); the shipped choice is pending the one-time
# URDF-overlay bench check — flip here if the overlay shows the virtual
# gripper rolled half a turn.
#
# Each entry is ``[x, y, z, qx, qy, qz, qw]``: the gripper TCP frame
# expressed in that tracker's device-local frame as the bridge/headset
# reports it — the TCP origin in metres plus the rotation taking tracker
# axes to gripper axes, straight from the mount CAD.
#
# TODO(mantis-calibration): Measure the Quest 3 cradle transform empirically.
# The headset client deliberately streams WebXR ``targetRaySpace`` (the
# runtime-defined aim/pointer pose), NOT ``gripSpace``. Its origin is therefore
# a virtual point near the controller's top/front rather than a dimensioned
# shell datum, and Meta may move it across firmware. Cradle CAD alone cannot
# supply this transform: seat the controller, run the absolute-mode URDF
# overlay, iterate the per-unit ``quest`` pos/quat entry until the physical and
# rendered gripper TCPs coincide, then promote the result here. Until then the
# engage snapshot absorbs the whole offset (recorded TCP poses are
# mount-dependent; a loud warning is logged at session start).
#
# TODO(mantis-calibration): Derive the Vive Ultimate Tracker transform from
# the mount CAD. Use the centre of the tracker's bottom mounting surface on
# the 1/4-20 insert axis (the seating plane / screw centre) as the physical
# tracker-origin datum, and measure from there to the gripper TCP plus the
# tracker→gripper rotation. pyvut is reverse-engineered, so first verify its
# quaternion order and up-axis settings on the bench (see docs/cli/tracker.mdx),
# then confirm the finished overlay before shipping the constant. Same fallback
# as Quest until it ships.
DESIGN_TCP_TRANSFORMS: dict[str, dict[str, list[float]]] = {
    "survive": {
        "left": [0.0, 0.0355, -0.092, 0.7071068, 0.0, 0.0, 0.7071068],
        "right": [0.0, 0.0355, -0.092, 0.7071068, 0.0, 0.0, 0.7071068],
    },
}


def design_transform_for(side: str, tracker_key: str) -> list[float] | None:
    """The rig's factory transform for ``side``, or ``None`` for the family.

    ``tracker_key`` is matched by backend family only (device identity does
    not change the design constant — every Tracker 3.0 sits on the same
    mount). Returns ``[x, y, z, qx, qy, qz, qw]`` like a calibration entry.
    """
    family = tracker_key.split(":", 1)[0]
    return DESIGN_TCP_TRANSFORMS.get(family, {}).get(side)


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


def load_tcp_transforms(
    path: Path = MANTIS_TCP_TRANSFORM_FILE,
) -> dict[str, dict[str, list[float]]]:
    """Load saved transforms as ``{side: {tracker_key: [x, y, z, qx..qw]}}``.

    Entries in the legacy (per-side only) format are accepted under
    :data:`LEGACY_TRACKER_KEY` with a deprecation warning — they predate
    per-tracker keying, so which tracker they were measured with is unknown.

    Returns an empty dict when no calibration exists or the file is invalid
    (callers fall back to the engage-snapshot absorption).
    """
    if (
        path == MANTIS_TCP_TRANSFORM_FILE
        and not path.exists()
        and _PRE_MANTIS_TCP_TRANSFORM_FILE.exists()
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _PRE_MANTIS_TCP_TRANSFORM_FILE.replace(path)
            _logger.info("migrated Mantis TCP calibration to %s", path)
        except OSError as exc:
            _logger.warning("could not migrate Mantis TCP calibration: %s", exc)
            path = _PRE_MANTIS_TCP_TRANSFORM_FILE
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
            "key — measured with an unknown tracker); re-key the entries under "
            'the active tracker (e.g. "survive:<codename>") or delete them '
            "to use the rig's factory design transform.",
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
