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

# Where each tracker's reported origin sits on the standard Mantis mount, in
# the gripper frame (the URDF ``*_gripper`` link: origin at the centre of the
# gripper motor's mounting face, +x along the jaw travel, -z towards the
# fingertips), in millimetres. From the rig CAD (2026-10): both Vive trackers
# sit 47 mm behind the motor face (gripper +z) and above it (gripper -y), the
# Ultimate's origin 11 mm higher than the Tracker 3.0's.
VIVE_TRACKER_CAD_ORIGINS_MM: dict[str, tuple[float, float, float]] = {
    "survive": (0.0, -35.0, 47.0),
    "ultimate": (0.0, -46.0, 47.0),
}

# Tracker→gripper rotation for the flat-back Vive mounts (Tracker 3.0 and
# Ultimate), as a unit quaternion ``[qx, qy, qz, qw]``: Ry(180°). Field
# datasets recorded with the previous Rx(+90°) constant (axol <= 0.2.4)
# replayed on Axol with every gripper pitched ~90° away from where the
# operator held it; ``axol migrate-dataset --mantis-tcp-rotation`` repairs
# them (see :data:`LEGACY_VIVE_TCP_ROTATION_QUAT`).
VIVE_TCP_ROTATION_QUAT: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)
# The retired Rx(+90°) rotation, kept only so the dataset migration can undo
# it.
LEGACY_VIVE_TCP_ROTATION_QUAT: tuple[float, float, float, float] = (
    0.7071068,
    0.0,
    0.0,
    0.7071068,
)
# Identifier axol 0.2.5–0.2.16 wrote to a Mantis dataset's ``meta/axol.json``
# (as ``mantis_tcp_transform.id``) for the factory Vive constants of that era:
# the corrected Ry(180°) rotation with the retired ``[0, 0.0355, -0.092]``
# (Tracker 3.0) / ``[0, 0.0465, -0.092]`` (Ultimate) translation, which put
# the tracker 92 mm in front of the gripper instead of 47 mm behind it.
# ``migrate-dataset --mantis-tcp-rotation`` still stamps it, since that
# migration replaces only the rotation. Datasets without the field predate it
# and were recorded with :data:`LEGACY_VIVE_TCP_ROTATION_QUAT`.
LEGACY_DESIGN_TCP_TRANSFORM_ID = "vive-flat-back-ry180-v0.2.5"
# Marker id for a run whose transforms came from a per-unit measurement (the
# override file or an explicit CLI value) rather than the factory constants.
MEASURED_TCP_TRANSFORM_ID = "measured"
# Marker id for a bring-up run without any tracker→gripper transform.
UNCALIBRATED_TCP_TRANSFORM_ID = "uncalibrated"
# A tracker mounted to a hand-held gripper cannot plausibly be farther than a
# metre from its TCP. This catches the dangerous and easy mm-as-m typo (for
# example entering 47 instead of 0.047) before it can authorize collection.
MAX_TCP_TRANSLATION_M = 1.0


def _rotate(quat: tuple[float, ...], vector: tuple[float, ...]) -> list[float]:
    """Rotate ``vector`` by the unit ``(x, y, z, w)`` quaternion ``quat``."""
    qx, qy, qz, qw = quat
    vx, vy, vz = vector
    # t = 2 q_vec × v;  v' = v + w t + q_vec × t
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return [
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    ]


def _tcp_transform_from_mount(
    tracker_origin_mm: tuple[float, float, float],
    tracker_to_gripper_quat: tuple[float, float, float, float],
) -> list[float]:
    """Stored ``[x, y, z, qx, qy, qz, qw]`` for a tracker placed on the rig.

    ``tracker_origin_mm`` is the tracker origin in the gripper frame (what the
    mount CAD measures); the stored translation is the inverse: the gripper
    origin in the tracker frame, ``-R_TG @ origin``.
    """
    # CAD quaternions arrive rounded; store the unit one validation would use.
    norm = math.sqrt(sum(value * value for value in tracker_to_gripper_quat))
    quat = tuple(value / norm for value in tracker_to_gripper_quat)
    origin_m = tuple(value / 1000.0 for value in tracker_origin_mm)
    position = [-value for value in _rotate(quat, origin_m)]
    # Drop float noise (-0.0, 1e-18) so the constants read as designed.
    return [round(value, 9) + 0.0 for value in position] + list(quat)


# Tracker→gripper transforms for the Mantis rig's standard mounts. Vive
# entries are keyed by tracker backend family — the part of a tracker key
# before the ":" (``"survive:T20"`` → ``"survive"``); Quest entries by their
# full ``quest:<WebXR profile>:<pose space>`` key, since controller
# generations and WebXR pose spaces do not share a local frame. Only entries
# promoted into ``DESIGN_TCP_TRANSFORMS`` below are factory values that apply
# out of the box; a per-unit measured entry in the override file always wins
# over them.
#
# Each entry is ``[x, y, z, qx, qy, qz, qw]``: the gripper TCP frame
# expressed in that tracker's device-local frame as the bridge/headset
# reports it — the TCP origin in metres plus ``R_TG``, whose columns are the
# gripper axes expressed in tracker coordinates (equivalently, it maps
# gripper-coordinate vectors into tracker coordinates).
#
# Vive (Tracker 3.0 and Ultimate, flat-back mount): the tracker sits flat on
# top of the rig, behind the gripper motor (see
# :data:`VIVE_TRACKER_CAD_ORIGINS_MM`). Rotation: Ry(180°)
# (:data:`VIVE_TCP_ROTATION_QUAT`) — the bridge-reported tracker +z runs along
# the gripper's -z (the finger direction, see ``constants.GRIPPER_TIP_OFFSET``)
# and its +x along the gripper's -x, with the y axis shared. The original
# derivation shipped Rx(+90°) here (axol <= 0.2.4): every recorded gripper
# orientation was pitched ~90° from where the operator held the rig (field
# report, isolate-5, 2026-09). Rx(180°), which also puts the fingers along
# tracker +z, was ruled out on the same data: under it the tracker side of the
# gripper tilted down in 95-100% of frames and every rig would have been held
# with its jaw axis reversed (handle away from the operator).
#
# Quest 3 (Touch Plus, WebXR grip space): the controller seated in the right
# Quest cradle has its grip origin 62 mm behind and 45 mm above the motor
# face, with the grip axes (x right, y up, z back when held tips-forward) at
# Rz(180°) from the gripper axes plus a ~10° cradle tilt. From the rig CAD
# (2026-10), given as the controller pose in the gripper frame and inverted
# here. The left cradle and controller are mirror images of the right ones
# across the gripper's y-z plane, while both WebXR grip frames keep +x to the
# right; reflecting the right pose (x -> -x in both frames) gives the left:
# the origin keeps x = 0 and the tilt's y/z quaternion components flip.
_QUEST_3_GRIP_ORIGIN_MM = (0.0, -45.0, 62.0)
_QUEST_3_RIGHT_GRIP_IN_GRIPPER_QUAT = (-0.066552, 0.053497, -0.996292, 0.010540)
QUEST_3_TRACKER_KEY = "quest:meta-quest-touch-plus:grip"


def _mirror_across_gripper_yz(
    origin_mm: tuple[float, float, float],
    quat: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """The mirror-image mount's pose: ``M p`` and ``M R M`` for ``M = diag(-1, 1, 1)``."""
    x, y, z = origin_mm
    qx, qy, qz, qw = quat
    return (-x + 0.0, y, z), (qx, -qy, -qz, qw)


def _quest_tcp_transform(
    origin_mm: tuple[float, float, float],
    grip_in_gripper_quat: tuple[float, float, float, float],
) -> list[float]:
    qx, qy, qz, qw = grip_in_gripper_quat
    # The inverse rotation: the conjugate of the grip-in-gripper quaternion.
    return _tcp_transform_from_mount(origin_mm, (-qx, -qy, -qz, qw))


_SURVIVE_DESIGN_TCP_TRANSFORM = _tcp_transform_from_mount(
    VIVE_TRACKER_CAD_ORIGINS_MM["survive"], VIVE_TCP_ROTATION_QUAT
)
_ULTIMATE_DESIGN_TCP_TRANSFORM = _tcp_transform_from_mount(
    VIVE_TRACKER_CAD_ORIGINS_MM["ultimate"], VIVE_TCP_ROTATION_QUAT
)
_QUEST_3_RIGHT_DESIGN_TCP_TRANSFORM = _quest_tcp_transform(
    _QUEST_3_GRIP_ORIGIN_MM, _QUEST_3_RIGHT_GRIP_IN_GRIPPER_QUAT
)
_QUEST_3_LEFT_DESIGN_TCP_TRANSFORM = _quest_tcp_transform(
    *_mirror_across_gripper_yz(
        _QUEST_3_GRIP_ORIGIN_MM, _QUEST_3_RIGHT_GRIP_IN_GRIPPER_QUAT
    )
)
DESIGN_TCP_TRANSFORMS: dict[str, dict[str, list[float]]] = {
    "survive": {
        "left": list(_SURVIVE_DESIGN_TCP_TRANSFORM),
        "right": list(_SURVIVE_DESIGN_TCP_TRANSFORM),
    },
    "ultimate": {
        "left": list(_ULTIMATE_DESIGN_TCP_TRANSFORM),
        "right": list(_ULTIMATE_DESIGN_TCP_TRANSFORM),
    },
    QUEST_3_TRACKER_KEY: {
        "left": list(_QUEST_3_LEFT_DESIGN_TCP_TRANSFORM),
        "right": list(_QUEST_3_RIGHT_DESIGN_TCP_TRANSFORM),
    },
}
# Identifier written to a Mantis dataset's ``meta/axol.json`` (as
# ``mantis_tcp_transform.id``) when both sides were recorded with one family's
# factory constants above. Bump an id whenever its constant changes, so a
# dataset recorded under the old value refuses to resume under the new one.
DESIGN_TCP_TRANSFORM_IDS: dict[str, str] = {
    "survive": "vive3-flat-back-v0.2.17",
    "ultimate": "vive-ultimate-flat-back-v0.2.17",
    QUEST_3_TRACKER_KEY: "quest3-touch-plus-grip-v0.2.17",
}

# Unapproved CAD starting points may be exposed here without making them
# usable for production collection. All currently known transforms above
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


def same_tcp_transform(actual: object, reference: object) -> bool:
    """True when two 7-vector transforms describe the same rigid pose mapping.

    Positions must match to within floating-point noise; the rotations are
    compared as rotations, so a unit quaternion and its negation (``q`` and
    ``-q`` encode the same rotation) count as equal. Anything that is not a
    7-element numeric list is simply "not the same" — callers use this to
    compare a value read back from ``meta/axol.json`` against a live one.
    """
    if not isinstance(actual, list) or not isinstance(reference, list):
        return False
    if len(actual) != 7 or len(reference) != 7:
        return False
    try:
        a = [float(v) for v in actual]
        b = [float(v) for v in reference]
    except (TypeError, ValueError):
        return False
    position_matches = all(
        math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9)
        for x, y in zip(a[:3], b[:3], strict=True)
    )
    quat_dot = sum(x * y for x, y in zip(a[3:], b[3:], strict=True))
    return position_matches and math.isclose(
        abs(quat_dot), 1.0, rel_tol=1e-9, abs_tol=1e-6
    )


def tcp_transform_provenance(
    left: list[float] | None,
    right: list[float] | None,
    *,
    source: str | None,
) -> dict[str, object]:
    """Describe the tracker→gripper transforms a Mantis dataset was recorded with.

    The result is written verbatim into the dataset's ``meta/axol.json`` so a
    later constant change can be undone by ``migrate-dataset`` without
    guessing. ``id`` is the matching :data:`DESIGN_TCP_TRANSFORM_IDS` entry
    when both sides use a family's current factory constants (two ids joined
    by ``+`` when the sides carry different families),
    :data:`MEASURED_TCP_TRANSFORM_ID` when either side carries a per-unit
    value, and :data:`UNCALIBRATED_TCP_TRANSFORM_ID` when a side has no
    transform at all (``--mantis_allow_uncalibrated`` bring-up capture).
    """
    transforms = {"left": left, "right": right}
    if any(value is None for value in transforms.values()):
        transform_id = UNCALIBRATED_TCP_TRANSFORM_ID
    else:
        side_ids = [
            next(
                (
                    DESIGN_TCP_TRANSFORM_IDS[key]
                    for key, family in DESIGN_TCP_TRANSFORMS.items()
                    if same_tcp_transform(
                        validate_tcp_transform(transforms[side]),
                        validate_tcp_transform(family[side]),
                    )
                ),
                None,
            )
            for side in ("left", "right")
        ]
        transform_id = (
            "+".join(dict.fromkeys(side_ids))
            if None not in side_ids
            else MEASURED_TCP_TRANSFORM_ID
        )
    return {
        "id": transform_id,
        "source": source,
        "left": None if left is None else [float(v) for v in left],
        "right": None if right is None else [float(v) for v in right],
    }


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
    """Whether saved state must suppress a factory fallback.

    An exact active-device entry is handled by the caller. Legacy, bare-family,
    and same-family entries for a different device may describe a non-standard
    mount, so silently replacing them after a rebind would be unsafe. Overrides
    for another tracker family do not conflict. For Quest, a bare ``quest``
    entry or the same controller profile under another pose space conflicts.
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
    if family == "quest":
        # A bare ``quest`` entry predates profile scoping and may describe a
        # non-standard cradle; the same profile under another pose space is a
        # measurement of this controller against a different datum. Either
        # must suppress the Quest factory value. Other controller generations
        # are different devices and do not conflict.
        datum = parse_quest_tracker_key(tracker_key)
        return "quest" in saved_keys or (
            datum is not None
            and any(
                (saved := parse_quest_tracker_key(key)) is not None
                and saved[0] == datum[0]
                and key != tracker_key
                for key in saved_keys
            )
        )
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
    controller generation the operation will report. A factory Quest profile
    stands in only when no Quest entry is saved at all: a bare ``quest`` key
    or a one-sided measurement may describe a non-standard cradle, so it
    must fail closed rather than be silently replaced by the factory value.
    """
    saved = [set(transforms.get(side, {})) for side in ("left", "right")]
    if any(
        key == "quest" or key.startswith("quest:") for keys in saved for key in keys
    ):
        common = saved[0] & saved[1]
        scoped = sorted(
            key for key in common if parse_quest_tracker_key(key) is not None
        )
    else:
        scoped = sorted(
            key
            for key, sides in DESIGN_TCP_TRANSFORMS.items()
            if "left" in sides
            and "right" in sides
            and parse_quest_tracker_key(key) is not None
        )
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
