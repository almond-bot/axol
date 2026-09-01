"""The Mantis rig's tracker→TCP transforms: factory constants + overrides.

The rig's tracker mounts are a fixed design, so an approved rigid
tracker→gripper transform can be shipped per tracker family in
:data:`DESIGN_TCP_TRANSFORMS` and applied out of the box. CAD-derived values
that have not been approved for the live tracker datum remain in
:data:`CANDIDATE_TCP_TRANSFORMS`; they are never applied automatically or
accepted for production collection. A per-unit override
file at ``~/.almond/mantis/tcp_transform.json`` takes precedence when present
(hand-measured refinements, non-standard mounts); its shape is one SE(3)
transform per rig side *and tracker identity*::

    {
      "left": {
        "quest:meta-quest-touch-plus:grip":
                        {"pos": [x, y, z], "quat": [qx, qy, qz, qw]},
        "survive:T20":  {"pos": [...], "quat": [...]},
        "ultimate:a:b:c:d:e:f": {
          "pos": [...], "quat": [...],
          "ultimate_pose_convention": {"quat_order": "wxyz", "up_axis": "z"}
        }
      },
      "right": { ... }
    }

The tracker key is the tracking backend plus its exact device-local datum:
``"quest:<WebXR-profile>:grip"`` for the headset path, or
``"survive:<codename>"`` / ``"ultimate:<mac>"`` for Vive backends (see
:func:`tracker_key_for_side`). Hardware design defaults are keyed by backend
family; Quest defaults must remain profile/pose-space scoped. Ultimate saved
measurements additionally carry its quaternion-order/up-axis parser convention,
because those settings define the bridge-reported tracker frame. Keying matters
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
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

from ..utils.paths import almond_path
from ..utils.state_files import (
    secure_atomic_write_text,
    secure_read_text,
    secure_unlink,
)

_logger = logging.getLogger(__name__)

MANTIS_TCP_TRANSFORM_FILE = almond_path("mantis", "tcp_transform.json")
_PRE_MANTIS_TCP_TRANSFORM_FILE = almond_path("u" + "mi", "tcp_transform.json")

# Pseudo tracker key under which entries from the legacy (per-side only) file
# format surface on load. Never produced by a fresh calibration.
LEGACY_TRACKER_KEY = "legacy"
QUEST_POSE_SPACES = frozenset({"grip", "target-ray"})
ULTIMATE_POSE_CONVENTION_FIELD = "ultimate_pose_convention"
ULTIMATE_QUAT_ORDERS = frozenset({"xyzw", "wxyz"})
ULTIMATE_UP_AXES = frozenset({"y", "z"})
# Convention under which the approved Ultimate factory transform is expressed.
# A different parser basis needs its own approved transform instead of silently
# reusing these numbers in a differently reported tracker frame.
ULTIMATE_FACTORY_POSE_CONVENTION = ("wxyz", "z")
CURRENT_TRANSFORM_ENTRY = "current"
STALE_TRANSFORM_ENTRY = "stale"
INVALID_TRANSFORM_ENTRY = "invalid"

# Tracker reference-origin positions reported in the shared gripper/CAD frame
# G, in millimetres. These are not tracker→TCP transforms by themselves: their
# difference establishes how the two device datums move relative to the
# otherwise unchanged Mantis mount geometry.
VIVE_TRACKER_CAD_ORIGINS_MM: dict[str, tuple[float, float, float]] = {
    "survive": (47.0, 0.0, 35.0),
    "ultimate": (47.0, 0.0, 46.0),
}
_ULTIMATE_CAD_ORIGIN_DELTA_Z_M = (
    VIVE_TRACKER_CAD_ORIGINS_MM["ultimate"][2]
    - VIVE_TRACKER_CAD_ORIGINS_MM["survive"][2]
) / 1000.0
# Let G be the shared gripper/CAD frame and T the bridge-reported tracker
# frame. The stored quaternion is R_TG = Rx(+90 deg), and the translation is
# the TCP origin expressed in T. Moving the tracker origin by delta_O in G
# therefore changes that translation by
#
#     delta_p_TG = -R_TG @ delta_O.
#
# Rx(+90 deg) maps CAD +z to tracker -y, so negating the +11 mm origin shift
# produces +11 mm on the Ultimate tracker-y TCP component. This is a mount-
# frame derivation, not the bridge's z-up-world -> y-up-world basis relabel.
_ULTIMATE_TRACKER_Y_TCP_DELTA_M = _ULTIMATE_CAD_ORIGIN_DELTA_Z_M
# A tracker mounted to a hand-held gripper cannot plausibly be farther than a
# metre from its TCP. This catches the dangerous and easy mm-as-m typo (for
# example entering 47 instead of 0.047) before it can authorize collection.
MAX_TCP_TRANSLATION_M = 1.0

# Tracker→gripper transform candidates and approved factory values for the
# Mantis rig's standard mounts are keyed by tracker backend
# family — the part of a tracker key before the ":" (``"survive:T20"`` →
# ``"survive"``). Only entries promoted into ``DESIGN_TCP_TRANSFORMS`` below
# are factory values that apply out of the box; a per-unit measured entry in
# the override file always wins over them.
#
# survive (Vive Tracker 3.0, standard mount): derived from the rig CAD
# (2026-08-10) — tracker seated flat, stabilizing-pin recess toward the jaws,
# gripper flange 92 mm forward / 35.5 mm below the tracker's mounting plane,
# gripper pointing forward, jaw travel lateral. Expressed in the bridge's
# tracker frame (libsurvive head frame with the z-up→y-up body relabel).
# The tracker and Mantis CAD frames use the approved standard mounting
# orientation; no additional axis flip or 180° correction is applied.
#
# Each entry is ``[x, y, z, qx, qy, qz, qw]``: the gripper TCP frame
# expressed in that tracker's device-local frame as the bridge/headset
# reports it — the TCP origin in metres plus ``R_TG``, whose columns are the
# gripper axes expressed in tracker coordinates (equivalently, it maps
# gripper-coordinate vectors into tracker coordinates), straight from the
# mount CAD.
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
# ultimate (Vive Ultimate Tracker, standard mount): starts from the Tracker 3.0
# factory transform above. :data:`VIVE_TRACKER_CAD_ORIGINS_MM` records their
# respective [47, 0, 35] mm and [47, 0, 46] mm reference origins. The common 47/0
# coordinates cancel and establish delta_O = [0, 0, +11] mm in the shared
# gripper/CAD frame. With the inherited R_TG = Rx(+90 deg),
# delta_p_TG = -R_TG @ delta_O = [0, +11, 0] mm. Applying that tracker-y delta
# to the V3 transform's independently derived 35.5 mm component gives 46.5 mm;
# the -92 mm forward offset and mount rotation are inherited. The two tracker
# families use the same local axis directions and the same flat-back mounting
# orientation, so Ultimate needs no additional rotation.
DESIGN_TCP_TRANSFORMS: dict[str, dict[str, list[float]]] = {
    "survive": {
        "left": [0.0, 0.0355, -0.092, 0.7071068, 0.0, 0.0, 0.7071068],
        "right": [0.0, 0.0355, -0.092, 0.7071068, 0.0, 0.0, 0.7071068],
    },
    "ultimate": {
        "left": [
            0.0,
            0.0355 + _ULTIMATE_TRACKER_Y_TCP_DELTA_M,
            -0.092,
            0.7071068,
            0.0,
            0.0,
            0.7071068,
        ],
        "right": [
            0.0,
            0.0355 + _ULTIMATE_TRACKER_Y_TCP_DELTA_M,
            -0.092,
            0.7071068,
            0.0,
            0.0,
            0.7071068,
        ],
    },
}

# Unapproved CAD starting points may be exposed here without making them
# usable for production collection. All currently known Vive transforms above
# are approved factory constants, so there are no remaining candidates.
CANDIDATE_TCP_TRANSFORMS: dict[str, dict[str, list[float]]] = {}


def validate_tcp_transform(transform: object) -> list[float]:
    """Return one safe tracker→TCP transform as seven floats.

    ``transform`` must be ``[x, y, z, qx, qy, qz, qw]`` with exactly seven
    finite numeric entries. The translation magnitude must be at most one
    metre, which is a deliberately generous physical bound for a tracker
    mounted on a hand-held rig. As with the calibration-file loader's
    historical behavior, a finite non-zero quaternion is normalized before
    use. A zero (or numerically degenerate) quaternion is rejected rather than
    becoming an invalid rotation matrix in the teleop worker.

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

    translation_m = math.hypot(*values[:3])
    if translation_m > MAX_TCP_TRANSLATION_M:
        raise ValueError(
            f"translation magnitude must be at most {MAX_TCP_TRANSLATION_M:g} metre; "
            "positions are entered in metres, not millimetres"
        )

    quat_norm = math.hypot(*values[3:])
    if not math.isfinite(quat_norm) or quat_norm <= 1e-12:
        raise ValueError("quaternion must have a finite, non-zero norm")
    values[3:] = [value / quat_norm for value in values[3:]]
    return values


def design_transform_for(
    side: str,
    tracker_key: str,
    *,
    tracker_config_path: Path | None = None,
) -> list[float] | None:
    """The rig's approved factory transform for ``side``, or ``None``.

    ``tracker_key`` is matched by backend family only (device identity does
    not change the design constant — every hardware tracker of one family
    sits on the same mount). Ultimate's factory value is returned only under
    the pose-parser convention for which it was approved. Returns
    ``[x, y, z, qx, qy, qz, qw]`` like a calibration entry.
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
    if (
        family == "ultimate"
        and current_ultimate_pose_convention(tracker_config_path)
        != ULTIMATE_FACTORY_POSE_CONVENTION
    ):
        return None
    return DESIGN_TCP_TRANSFORMS.get(lookup, {}).get(side)


def candidate_transform_for(side: str, tracker_key: str) -> list[float] | None:
    """Return an unverified CAD candidate, never a production transform."""
    family = tracker_key.split(":", 1)[0]
    lookup = tracker_key if family == "quest" else family
    if family == "quest" and parse_quest_tracker_key(tracker_key) is None:
        return None
    return CANDIDATE_TCP_TRANSFORMS.get(lookup, {}).get(side)


def has_conflicting_transform_override(
    side: str,
    tracker_key: str,
    transforms: Mapping[str, Mapping[str, object]],
    entry_statuses: Mapping[tuple[str, str], str] | None = None,
) -> bool:
    """Whether saved state must suppress a hardware factory fallback.

    An exact active-device entry is handled by the caller. Legacy, bare-family,
    and same-family entries for a different device may describe a non-standard
    mount, so silently replacing them after a rebind would be unsafe. Overrides
    for another tracker family do not conflict.
    """
    saved_keys: set[str] = {
        key for key in transforms.get(side, {}) if isinstance(key, str)
    }
    if entry_statuses is not None:
        saved_keys.update(
            key
            for (entry_side, key) in entry_statuses
            if entry_side == side and isinstance(key, str)
        )
    if LEGACY_TRACKER_KEY in saved_keys:
        return True
    family = tracker_key.split(":", 1)[0]
    if family not in {"survive", "ultimate"}:
        return False
    for saved_key in saved_keys:
        same_family = saved_key == family or saved_key.startswith(f"{family}:")
        if same_family and (saved_key != tracker_key or tracker_key == family):
            return True
    return False


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
    path: Path | None = None,
    *,
    tracker_config_path: Path | None = None,
    entry_statuses: dict[tuple[str, str], str] | None = None,
    document_errors: list[str] | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Load saved transforms as ``{side: {tracker_key: [x, y, z, qx..qw]}}``.

    Entries in the legacy (per-side only) format are accepted under
    :data:`LEGACY_TRACKER_KEY` with a deprecation warning — they predate
    per-tracker keying, so which tracker they were measured with is unknown.
    Ultimate entries are returned only when their saved pose convention exactly
    matches the active tracker config; convention-less and mismatched entries
    remain on disk for explicit adoption but are not authoritative. When
    ``entry_statuses`` is supplied, it receives the classification of every
    string-keyed entry so callers do not silently fall back to a factory value
    over an exact stale or malformed per-device override. ``document_errors``
    receives a safe diagnostic when a present file cannot be trusted; only an
    actually absent file permits an unconditional factory fallback.

    Returns an empty dict when no calibration exists or the file is invalid
    (teleop may use its explicitly warned, start-pose-only fallback; production
    collection rejects a missing transform).
    """
    path = MANTIS_TCP_TRANSFORM_FILE if path is None else path
    if (
        path == MANTIS_TCP_TRANSFORM_FILE
        and not path.exists()
        and _PRE_MANTIS_TCP_TRANSFORM_FILE.exists()
    ):
        try:
            legacy = secure_read_text(_PRE_MANTIS_TCP_TRANSFORM_FILE)
            secure_atomic_write_text(path, legacy)
            secure_unlink(_PRE_MANTIS_TCP_TRANSFORM_FILE)
            _logger.info("migrated Mantis TCP calibration to %s", path)
        except OSError as exc:
            _logger.warning("could not migrate Mantis TCP calibration: %s", exc)
            path = _PRE_MANTIS_TCP_TRANSFORM_FILE
    if not path.exists():
        return {}
    try:
        data = json.loads(secure_read_text(path))
    except (OSError, ValueError) as exc:
        _logger.warning("could not read %s: %s", path, exc)
        if document_errors is not None:
            document_errors.append("calibration file is unreadable or invalid JSON")
        return {}
    if not isinstance(data, dict):
        if document_errors is not None:
            document_errors.append("calibration file root is not an object")
        return {}
    out: dict[str, dict[str, list[float]]] = {}
    legacy_seen = False
    ultimate_convention_loaded = False
    ultimate_convention: tuple[str, str] | None = None
    for side in ("left", "right"):
        if side not in data:
            continue
        side_entries = data.get(side)
        if _is_legacy_side_entry(side_entries):
            side_entries = {LEGACY_TRACKER_KEY: side_entries}
            legacy_seen = True
        if not isinstance(side_entries, dict):
            if document_errors is not None:
                document_errors.append(f"calibration `{side}` section is not an object")
            continue
        if ("pos" in side_entries) != ("quat" in side_entries):
            if document_errors is not None:
                document_errors.append(
                    f"calibration `{side}` legacy section is incomplete"
                )
            continue
        for key, entry in side_entries.items():
            if _is_ultimate_transform_key(key) and not ultimate_convention_loaded:
                ultimate_convention = current_ultimate_pose_convention(
                    tracker_config_path
                )
                ultimate_convention_loaded = True
            flat, status = classify_tcp_transform_entry(
                key,
                entry,
                ultimate_convention=ultimate_convention,
            )
            if entry_statuses is not None and isinstance(key, str):
                entry_statuses[(side, key)] = status
            if flat is not None and status == CURRENT_TRANSFORM_ENTRY:
                out.setdefault(side, {})[key] = flat
            elif flat is not None and status == STALE_TRANSFORM_ENTRY:
                stored = ultimate_pose_convention_from_entry(entry)
                _logger.warning(
                    "%s %s calibration for %r is not authoritative: saved "
                    "Ultimate pose convention %r does not match active %r. "
                    "Bench-check and explicitly resave it under the active "
                    "convention before production collection.",
                    path,
                    side,
                    key,
                    stored,
                    ultimate_convention,
                )
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


def _is_ultimate_transform_key(tracker_key: object) -> bool:
    """Whether a saved transform key uses the Ultimate tracker-local frame."""
    return isinstance(tracker_key, str) and (
        tracker_key == "ultimate" or tracker_key.startswith("ultimate:")
    )


def _normalize_ultimate_pose_convention(
    quat_order: object, up_axis: object
) -> tuple[str, str] | None:
    if (
        not isinstance(quat_order, str)
        or quat_order not in ULTIMATE_QUAT_ORDERS
        or not isinstance(up_axis, str)
        or up_axis not in ULTIMATE_UP_AXES
    ):
        return None
    return quat_order, up_axis


def current_ultimate_pose_convention(
    config_path: Path | None = None,
) -> tuple[str, str] | None:
    """Return the Ultimate parser convention active for tracker pose reports.

    A tracker→TCP transform is expressed in the bridge-reported tracker-local
    frame. Both of these settings change that frame, so they are calibration
    provenance rather than incidental runtime options.
    """
    from ..tracker.config import TRACKER_CONFIG_FILE, load_tracker_config

    path = TRACKER_CONFIG_FILE if config_path is None else config_path
    config = load_tracker_config(path)
    return _normalize_ultimate_pose_convention(
        config.ultimate_quat_order,
        config.ultimate_up_axis,
    )


def ultimate_pose_convention_metadata(
    convention: tuple[str, str],
) -> dict[str, str]:
    """Serialize an already validated Ultimate convention for one entry."""
    normalized = _normalize_ultimate_pose_convention(*convention)
    if normalized is None:
        raise ValueError(f"invalid Ultimate pose convention: {convention!r}")
    quat_order, up_axis = normalized
    return {"quat_order": quat_order, "up_axis": up_axis}


def ultimate_pose_convention_from_entry(
    entry: object,
) -> tuple[str, str] | None:
    """Parse exact Ultimate convention metadata from one calibration entry."""
    if not isinstance(entry, dict):
        return None
    metadata = entry.get(ULTIMATE_POSE_CONVENTION_FIELD)
    if not isinstance(metadata, dict) or set(metadata) != {"quat_order", "up_axis"}:
        return None
    return _normalize_ultimate_pose_convention(
        metadata.get("quat_order"), metadata.get("up_axis")
    )


def classify_tcp_transform_entry(
    tracker_key: object,
    entry: object,
    *,
    ultimate_convention: tuple[str, str] | None,
) -> tuple[list[float] | None, str]:
    """Return a normalized transform and its authorization provenance state.

    Quest and Lighthouse entries keep their existing keyed-file behavior.
    Ultimate entries are current only when they record the exact quaternion
    order and up-axis used to interpret the device pose. Older convention-less
    entries remain readable for explicit operator adoption, but are stale and
    therefore omitted by :func:`load_tcp_transforms`.
    """
    flat = _flatten_entry(entry)
    if flat is None:
        return None, INVALID_TRANSFORM_ENTRY
    if not _is_ultimate_transform_key(tracker_key):
        return flat, CURRENT_TRANSFORM_ENTRY
    stored = ultimate_pose_convention_from_entry(entry)
    if stored is None or ultimate_convention is None or stored != ultimate_convention:
        return flat, STALE_TRANSFORM_ENTRY
    return flat, CURRENT_TRANSFORM_ENTRY
