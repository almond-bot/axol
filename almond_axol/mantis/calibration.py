"""The Mantis rig's tracker→TCP transforms: verified constants + overrides.

The rig's tracker mounts are a fixed design, so a bench-verified rigid
tracker→gripper transform can be shipped per tracker family in
:data:`DESIGN_TCP_TRANSFORMS` and applied out of the box. CAD-derived values
that have not been checked against the live tracker datum remain in
:data:`CANDIDATE_TCP_TRANSFORMS`; they are never applied automatically or
accepted for production collection. A per-unit override
file at ``~/.almond/mantis/tcp_transform.json`` takes precedence when present
(hand-measured refinements, non-standard mounts); its shape is one SE(3)
transform per rig side *and tracker identity*::

    {
      "left": {
        "quest:meta-quest-touch-plus:grip":
                        {"pos": [x, y, z], "quat": [qx, qy, qz, qw]},
        "survive:T20":  {"pos": [...], "quat": [...]}
      },
      "right": { ... }
    }

The tracker key is the tracking backend plus its exact device-local datum:
``"quest:<WebXR-profile>:grip"`` for the headset path, or
``"survive:<codename>"`` / ``"ultimate:<mac>"`` for Vive backends (see
:func:`tracker_key_for_side`). Hardware design defaults are keyed by backend
family; Quest defaults must remain profile/pose-space scoped. Keying matters
because each tracker type can have both a different physical mount and a
different device-local frame — a transform for one is silently wrong for
another.

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
import math
from numbers import Real
from pathlib import Path

from ..utils.paths import almond_path

_logger = logging.getLogger(__name__)

MANTIS_TCP_TRANSFORM_FILE = almond_path("mantis", "tcp_transform.json")
_PRE_MANTIS_TCP_TRANSFORM_FILE = almond_path("u" + "mi", "tcp_transform.json")

# Pseudo tracker key under which entries from the legacy (per-side only) file
# format surface on load. Never produced by a fresh calibration.
LEGACY_TRACKER_KEY = "legacy"
QUEST_POSE_SPACES = frozenset({"grip", "target-ray"})

# Bench-verified factory tracker→gripper transforms for the Mantis rig's
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
# The headset client streams WebXR ``gripSpace`` (with ``targetRaySpace`` only
# as a compatibility fallback on runtimes that omit gripSpace). Cradle CAD may
# therefore supply a starting physical transform, but the WebXR grip datum is
# still not a dimensioned shell origin. Seat the controller, run the URDF
# overlay, iterate the per-unit ``quest`` pos/quat entry until the physical and
# rendered gripper TCPs coincide, then promote the result here. Until then the
# engage snapshot aligns only the starting pose; later recorded TCP poses stay
# mount-dependent, so production collection rejects the missing transform.
#
# ultimate (Vive Ultimate Tracker, standard mount): a mechanical comparison
# reported 2026-08-31 places its device origin 11 mm higher than the Tracker 3.0
# origin along native physical z (vertical). The bridge relabels native z-up as
# WebXR y-up, and the Tracker 3.0 candidate represents the TCP's 35.5 mm
# downward separation as +0.0355 local y. Measuring the same TCP from an origin
# 11 mm higher therefore changes that stored offset to +0.0465 m; the -92 mm
# forward offset and mount orientation are shared. Because pyvut is
# reverse-engineered, its quaternion order/up-axis and the completed overlay
# still need a bench check before this candidate can become a design constant.
CANDIDATE_TCP_TRANSFORMS: dict[str, dict[str, list[float]]] = {
    "survive": {
        "left": [0.0, 0.0355, -0.092, 0.7071068, 0.0, 0.0, 0.7071068],
        "right": [0.0, 0.0355, -0.092, 0.7071068, 0.0, 0.0, 0.7071068],
    },
    "ultimate": {
        "left": [0.0, 0.0465, -0.092, 0.7071068, 0.0, 0.0, 0.7071068],
        "right": [0.0, 0.0465, -0.092, 0.7071068, 0.0, 0.0, 0.7071068],
    },
}

# Only values promoted after a live URDF-overlay bench check belong here.
# Collection treats absence from this mapping as uncalibrated even when a CAD
# candidate exists above.
DESIGN_TCP_TRANSFORMS: dict[str, dict[str, list[float]]] = {}


def validate_tcp_transform(transform: object) -> list[float]:
    """Return one safe tracker→TCP transform as seven floats.

    ``transform`` must be ``[x, y, z, qx, qy, qz, qw]`` with exactly seven
    finite numeric entries. As with the calibration-file loader's historical
    behavior, a finite non-zero quaternion is normalized before use. A zero
    (or numerically degenerate) quaternion is rejected rather than becoming an
    invalid rotation matrix in the teleop worker.

    Raises:
        ValueError: If the shape, values, or quaternion are unsafe.
    """
    if not isinstance(transform, (list, tuple)) or len(transform) != 7:
        raise ValueError("must contain exactly 7 values [x, y, z, qx, qy, qz, qw]")

    values: list[float] = []
    for index, value in enumerate(transform):
        # bool is an int subclass, but accepting it in a spatial transform is
        # almost certainly a malformed JSON/YAML or Advanced-field value.
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"value at index {index} must be numeric")
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"value at index {index} must be a finite float") from exc
        if not math.isfinite(converted):
            raise ValueError(f"value at index {index} must be finite")
        values.append(converted)

    quat_norm = math.hypot(*values[3:])
    if not math.isfinite(quat_norm) or quat_norm <= 1e-12:
        raise ValueError("quaternion must have a finite, non-zero norm")
    values[3:] = [value / quat_norm for value in values[3:]]
    return values


def design_transform_for(side: str, tracker_key: str) -> list[float] | None:
    """The rig's bench-verified transform for ``side``, or ``None``.

    ``tracker_key`` is matched by backend family only (device identity does
    not change the design constant — every Tracker 3.0 sits on the same
    mount). Returns ``[x, y, z, qx, qy, qz, qw]`` like a calibration entry.
    """
    family = tracker_key.split(":", 1)[0]
    # Quest controller frames are profile- and pose-space-specific. A future
    # verified constant must therefore use the full
    # ``quest:<profile>:<space>`` key; never fan a bare family value across
    # controller generations. Hardware trackers have a stable backend-local
    # datum, so their family default remains appropriate.
    lookup = tracker_key if family == "quest" else family
    if family == "quest" and parse_quest_tracker_key(tracker_key) is None:
        return None
    return DESIGN_TCP_TRANSFORMS.get(lookup, {}).get(side)


def candidate_transform_for(side: str, tracker_key: str) -> list[float] | None:
    """Return an unverified CAD candidate, never a production transform."""
    family = tracker_key.split(":", 1)[0]
    lookup = tracker_key if family == "quest" else family
    if family == "quest" and parse_quest_tracker_key(tracker_key) is None:
        return None
    return CANDIDATE_TCP_TRANSFORMS.get(lookup, {}).get(side)


def parse_quest_tracker_key(tracker_key: str) -> tuple[str, str] | None:
    """Parse ``quest:<WebXR profile>:<pose space>`` into its live datum.

    A bare ``quest`` key predates controller-profile reporting and is
    intentionally not accepted here: a Touch controller generation and
    WebXR's grip/aim spaces do not share an interchangeable local frame.
    """
    if not tracker_key.startswith("quest:"):
        return None
    profile_and_space = tracker_key[len("quest:") :]
    profile, separator, pose_space = profile_and_space.rpartition(":")
    if not separator or not profile or pose_space not in QUEST_POSE_SPACES:
        return None
    return profile, pose_space


def select_quest_transform_key(
    transforms: dict[str, dict[str, list[float]]],
) -> str | None:
    """Select the sole profile-scoped Quest key calibrated on both sides.

    Multiple common profiles are deliberately ambiguous; callers need an
    explicit ``tracker_key`` in that case instead of guessing which connected
    controller generation the operation will report.
    """
    common = set(transforms.get("left", {})) & set(transforms.get("right", {}))
    common.update(
        key
        for key, sides in DESIGN_TCP_TRANSFORMS.items()
        if "left" in sides and "right" in sides
    )
    scoped = sorted(key for key in common if parse_quest_tracker_key(key) is not None)
    return scoped[0] if len(scoped) == 1 else None


def tracker_key_for_side(
    side: str,
    override: str | None = None,
    source: str | None = None,
    config_path: Path | None = None,
) -> tuple[str, str]:
    """Identity key of the tracker presumed active on ``side``.

    The key is what per-tracker calibrations are stored and looked up under:
    ``"quest"`` for the headset path (which has no tracker backend config at
    all), otherwise ``"<backend>"`` or ``"<backend>:<device>"`` from the
    saved tracker config (``~/.almond/tracker/config.json``, written by
    ``axol tracker.identify``) — e.g. ``"survive:T20"`` or
    ``"ultimate:<mac>"``.

    ``source`` is the selected Mantis source (``quest``, ``lighthouse``, or
    ``ultimate``). Passing it is strongly preferred: a Quest headset and an
    ``axol tracker.bridge`` look identical to the VR server, and the tracker
    config may contain bindings for more than one backend. The historical
    file-existence inference remains only for callers that do not know the
    active source. ``override`` always takes precedence.

    Args:
        side: ``"left"`` or ``"right"`` (the two rigs bind different devices).
        override: Explicit key to use instead of deriving one, or ``None``.
        source: Selected Mantis source, or ``None`` for legacy inference.
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
    if source == "quest":
        return "quest", "Quest WebXR source explicitly selected"
    backend = {"lighthouse": "survive", "ultimate": "ultimate"}.get(source or "")
    if source is not None and backend is None:
        raise ValueError(
            f"tracker source must be quest, lighthouse, or ultimate; got {source!r}"
        )
    if not path.exists():
        if backend is not None:
            return backend, f"{source} source selected; no saved device binding"
        return (
            "quest",
            f"no tracker backend configured ({path} missing) — "
            "assuming the Quest headset path",
        )
    config = load_tracker_config(path)
    selected_backend = backend or config.backend
    binding = config.bindings.get(selected_backend, {})
    if selected_backend == config.backend:
        device = binding.get(side) or (config.left if side == "left" else config.right)
    else:
        device = binding.get(side)
    key = f"{selected_backend}:{device}" if device else selected_backend
    if source is not None:
        return key, f"{source} source explicitly selected; binding loaded from {path}"
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
    (teleop may use its explicitly warned, start-pose-only fallback; production
    collection rejects a missing transform).
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
    if not (
        isinstance(pos, list)
        and len(pos) == 3
        and isinstance(quat, list)
        and len(quat) == 4
    ):
        return None
    try:
        return validate_tcp_transform([*pos, *quat])
    except ValueError:
        return None
