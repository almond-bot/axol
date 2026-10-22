"""Tracker configuration persisted at ``~/.almond/tracker/config.json``.

Written by ``axol tracker.identify`` (backend + left/right device binding)
and read by ``axol tracker.bridge``. Kept as a plain JSON file — like the
Mantis TCP-offset calibration — so it survives reinstalls and is trivially
editable by hand.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..constants import CAN_MANTIS_LEFT, CAN_MANTIS_RIGHT
from ..utils.paths import almond_path
from ..utils.state_files import secure_atomic_write_json, secure_read_text
from .lighthouse_survey import validate_expected_stations
from .trigger import validate_trigger_full_scale

TRACKER_CONFIG_FILE = almond_path("tracker", "config.json")

LIGHTHOUSE_QUALITY_MODES = ("standard", "strict")


def validate_lighthouse_quality(value: object) -> str:
    """Return a supported Lighthouse quality mode or raise ``ValueError``."""
    if value not in LIGHTHOUSE_QUALITY_MODES:
        raise ValueError(
            f"lighthouse_quality must be one of {', '.join(LIGHTHOUSE_QUALITY_MODES)}"
        )
    return str(value)


def lighthouse_strict(config: TrackerConfig | None = None) -> bool:
    """Whether Lighthouse runs use the strict fixed-installation policy."""
    if config is None:
        config = load_tracker_config()
    return config.lighthouse_quality == "strict"


@dataclass
class TrackerConfig:
    """Backend selection and left/right device binding.

    Attributes:
        backend: ``"survive"`` (Vive Tracker 3.0 via libsurvive),
            ``"ultimate"`` (Vive Ultimate Tracker via the dongle), or
            ``"synthetic"`` (generated motion for tests).
        left:  Device key of the left-rig tracker (Lighthouse tracker serial /
            Ultimate MAC), or ``None`` if unassigned.
        right: Device key of the right-rig tracker, or ``None``.
        bindings: Per-backend device bindings. This lets the control panel
            switch between Lighthouse and Ultimate without applying device
            identifiers from one tracker system to the other. ``left`` and
            ``right`` remain the active backend's binding for compatibility.
        ultimate_quat_order: Component order of the quaternion in the
            Ultimate dongle's pose reports (``"xyzw"`` or ``"wxyz"``).
            The pinned pyvut runtime reports ``"wxyz"``.
            Verify at bring-up: hold a tracker still and level; the
            streamed orientation must be near-identity after conversion.
        ultimate_up_axis: Up axis of the Ultimate tracker's SLAM world
            frame (``"z"`` or ``"y"``). ``"z"`` converts through the z-up →
            y-up basis change; ``"y"`` passes through. Verify at bring-up.
        trigger_can_left:  SocketCAN interface of the left rig's trigger
            node, defaulting to the Mantis rig's left gripper bus — the
            only bus a trigger node ever sits on. Set it to ``None`` for a
            rig with no trigger; grip then streams as 1.0 (open), as it
            does when the interface is absent.
        trigger_can_right: SocketCAN interface of the right rig's trigger
            node, or ``None``.
        trigger_full_scale_left: Normalized left trigger position reached at
            full physical squeeze; values below 1.0 extend the input to a
            fully closed grip command.
        trigger_full_scale_right: Equivalent endpoint for the right trigger.
        allow_single_side: Let the bridge run with only one side's
            tracker bound. Off by default: absolute-mode (Mantis) engagement
            fits the base transform from BOTH controller positions, so
            the placeholder pose streamed for an unbound side corrupts it.
        expected_base_stations: Number of installed Lighthouse base stations.
        lighthouse_quality: How strictly Lighthouse tracking is policed (see
            :data:`LIGHTHOUSE_QUALITY_MODES`). ``"standard"`` runs libsurvive
            with its own background calibration and leaves take acceptance to
            the episode QA gate. ``"strict"`` is for fixed installations: the
            room geometry is frozen outside an explicit Recalibrate, collection
            requires a passed, unexpired Lighthouse validation, optical support
            and pose discontinuities are monitored live, and a tracking fault
            marks the take for retake regardless of ``qa_gate``.
    """

    backend: str = "survive"
    left: str | None = None
    right: str | None = None
    bindings: dict[str, dict[str, str | None]] = field(default_factory=dict)
    ultimate_quat_order: str = "wxyz"
    ultimate_up_axis: str = "z"
    trigger_can_left: str | None = CAN_MANTIS_LEFT
    trigger_can_right: str | None = CAN_MANTIS_RIGHT
    trigger_full_scale_left: float = 1.0
    trigger_full_scale_right: float = 1.0
    allow_single_side: bool = False
    expected_base_stations: int = 2
    lighthouse_quality: str = "standard"


# A libsurvive codename (``WM0``, ``T20``…) rather than a tracker serial
# (``LHR-…``). Bindings saved by earlier releases used these, but a codename
# names the dongle slot in open order and can swap between runs, which bound
# each rig to the other's tracker.
_SURVIVE_CODENAME = re.compile(r"[A-Z]+[0-9]*")


def _is_survive_codename(key: object) -> bool:
    return isinstance(key, str) and _SURVIVE_CODENAME.fullmatch(key) is not None


def load_tracker_config(path: Path = TRACKER_CONFIG_FILE) -> TrackerConfig:
    """Load the saved config, tolerating a missing file or unknown keys."""
    try:
        data = json.loads(secure_read_text(path))
    except (OSError, ValueError):
        return TrackerConfig()
    if not isinstance(data, dict):
        return TrackerConfig()
    old_stem = "u" + "mi"
    old_channels = {
        f"can_alm_{old_stem}_l": CAN_MANTIS_LEFT,
        f"can_alm_{old_stem}_r": CAN_MANTIS_RIGHT,
    }
    migrated = False
    for key in ("trigger_can_left", "trigger_can_right"):
        if data.get(key) in old_channels:
            data[key] = old_channels[data[key]]
            migrated = True

    known = {f for f in TrackerConfig.__dataclass_fields__}
    config = TrackerConfig(**{k: v for k, v in data.items() if k in known})
    config.expected_base_stations = validate_expected_stations(
        config.expected_base_stations
    )
    config.trigger_full_scale_left = validate_trigger_full_scale(
        config.trigger_full_scale_left
    )
    config.trigger_full_scale_right = validate_trigger_full_scale(
        config.trigger_full_scale_right
    )
    if config.lighthouse_quality not in LIGHTHOUSE_QUALITY_MODES:
        # A hand-edited typo must not silently select the strict gate.
        config.lighthouse_quality = "standard"
        migrated = True
    if not isinstance(config.bindings, dict):
        config.bindings = {}
        migrated = True
    else:
        for backend, binding in list(config.bindings.items()):
            if not isinstance(binding, dict):
                config.bindings.pop(backend)
                migrated = True
    # Codename bindings can no longer be trusted (see _SURVIVE_CODENAME):
    # drop them so setup asks for Identify again instead of engaging each rig
    # with the other's tracker.
    survive_binding = config.bindings.get("survive")
    if survive_binding is not None:
        for side in ("left", "right"):
            if _is_survive_codename(survive_binding.get(side)):
                survive_binding[side] = None
                migrated = True
    if config.backend == "survive":
        for side in ("left", "right"):
            if _is_survive_codename(getattr(config, side)):
                setattr(config, side, None)
                migrated = True
    if config.backend in {"survive", "ultimate"} and (
        config.left is not None or config.right is not None
    ):
        binding = {"left": config.left, "right": config.right}
        if config.bindings.get(config.backend) != binding:
            config.bindings[config.backend] = binding
            migrated = True
    if migrated:
        save_tracker_config(config, path)
    return config


def save_tracker_config(
    config: TrackerConfig, path: Path = TRACKER_CONFIG_FILE
) -> None:
    """Persist the config as pretty JSON, creating parent directories."""
    validate_expected_stations(config.expected_base_stations)
    validate_lighthouse_quality(config.lighthouse_quality)
    config.trigger_full_scale_left = validate_trigger_full_scale(
        config.trigger_full_scale_left
    )
    config.trigger_full_scale_right = validate_trigger_full_scale(
        config.trigger_full_scale_right
    )
    if config.backend in {"survive", "ultimate"}:
        config.bindings[config.backend] = {
            "left": config.left,
            "right": config.right,
        }
    secure_atomic_write_json(path, asdict(config), sort_keys=False)


def select_tracker_backend(config: TrackerConfig, backend: str) -> None:
    """Activate ``backend`` and restore its saved left/right binding."""
    if backend == config.backend:
        # A hand-edited or early multi-backend config may hold the binding only
        # in ``bindings``. Treat that as authoritative when the compatibility
        # top-level fields are absent instead of reporting a false incomplete
        # setup for the already-active backend.
        binding = config.bindings.get(backend, {})
        if config.left is None:
            config.left = binding.get("left")
        if config.right is None:
            config.right = binding.get("right")
        return
    binding = config.bindings.get(backend, {})
    config.backend = backend
    config.left = binding.get("left")
    config.right = binding.get("right")
