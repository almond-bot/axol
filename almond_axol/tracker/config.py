"""Tracker configuration persisted at ``~/.almond/tracker/config.json``.

Written by ``axol tracker.identify`` (backend + left/right device binding)
and read by ``axol tracker.bridge``. Kept as a plain JSON file — like the
UMI TCP-offset calibration — so it survives reinstalls and is trivially
editable by hand.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

TRACKER_CONFIG_FILE = Path.home() / ".almond" / "tracker" / "config.json"


@dataclass
class TrackerConfig:
    """Backend selection and left/right device binding.

    Attributes:
        backend: ``"survive"`` (Vive Tracker 3.0 via libsurvive),
            ``"ultimate"`` (Vive Ultimate Tracker via the dongle), or
            ``"synthetic"`` (generated motion for tests).
        left:  Device key of the left-rig tracker (libsurvive codename /
            Ultimate MAC), or ``None`` if unassigned.
        right: Device key of the right-rig tracker, or ``None``.
        ultimate_quat_order: Component order of the quaternion in the
            Ultimate dongle's pose reports (``"xyzw"`` or ``"wxyz"``).
            Verify at bring-up: hold a tracker still and level; the
            streamed orientation must be near-identity after conversion.
        ultimate_up_axis: Up axis of the Ultimate tracker's SLAM world
            frame (``"z"`` or ``"y"``). ``"z"`` converts through the z-up →
            y-up basis change; ``"y"`` passes through. Verify at bring-up.
    """

    backend: str = "survive"
    left: str | None = None
    right: str | None = None
    ultimate_quat_order: str = "xyzw"
    ultimate_up_axis: str = "z"


def load_tracker_config(path: Path = TRACKER_CONFIG_FILE) -> TrackerConfig:
    """Load the saved config, tolerating a missing file or unknown keys."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return TrackerConfig()
    known = {f for f in TrackerConfig.__dataclass_fields__}
    return TrackerConfig(**{k: v for k, v in data.items() if k in known})


def save_tracker_config(
    config: TrackerConfig, path: Path = TRACKER_CONFIG_FILE
) -> None:
    """Persist the config as pretty JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2) + "\n")
