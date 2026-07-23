"""Headset-free pose sources for teleop: Vive trackers on the UMI rig.

This package feeds the existing VR teleop stack from HTC Vive trackers
instead of a Quest headset. A :class:`TrackerSource` backend produces
per-device 6-DOF poses in the WebXR convention (y-up, metres) and the
bridge (``axol tracker.bridge``) composes them into standard ``VRFrame``
JSON streamed to the VR WebSocket server — so teleop, IK, and
collect-data run unchanged.

Backends:
  - ``survive``   — Vive Tracker 3.0 via libsurvive (lighthouse tracking).
  - ``ultimate``  — Vive Ultimate Tracker via the wireless dongle (pyvut).
  - ``synthetic`` — generated motion for end-to-end tests without hardware.
"""

from .base import TrackerPose, TrackerSource
from .config import TRACKER_CONFIG_FILE, TrackerConfig, load_tracker_config
from .synthetic import SyntheticSource

__all__ = [
    "TRACKER_CONFIG_FILE",
    "SyntheticSource",
    "TrackerConfig",
    "TrackerPose",
    "TrackerSource",
    "create_source",
    "load_tracker_config",
]


def create_source(config: TrackerConfig) -> TrackerSource:
    """Instantiate the configured backend (imported lazily — the survive and
    ultimate backends require optional system/pip dependencies)."""
    if config.backend == "survive":
        from .survive import SurviveSource

        return SurviveSource()
    if config.backend == "ultimate":
        from .ultimate import UltimateSource

        return UltimateSource(
            quat_order=config.ultimate_quat_order,
            up_axis=config.ultimate_up_axis,
        )
    if config.backend == "synthetic":
        return SyntheticSource()
    raise ValueError(
        f"unknown tracker backend {config.backend!r} "
        "(expected survive, ultimate, or synthetic)"
    )
