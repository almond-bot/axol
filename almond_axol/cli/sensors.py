"""
Stream the cameras to the control panel's Sensors page.

The Sensors page shows the lidar and the live map on its own. The cameras
can only be opened by one program, so the page starts this operation to
stream them when no other operation (teleop, Jog, data collection) is
streaming them already. Nothing moves: it opens no motors, Jelly or lift.

The session is started and stopped by the control panel (``axol serve``),
which fills in the cameras from the camera settings. Run it from the panel.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field

from .config import LogLevel

_logger = logging.getLogger(__name__)


@dataclass
class SensorsCmdConfig:
    """Config for the ``sensors`` operation."""

    cameras: dict[str, int] = field(default_factory=dict)
    """Cameras to stream, by slot (filled from the camera settings by the
    control panel, as for teleop)."""
    camera_eyes: dict[str, str] = field(default_factory=dict)
    """Which eyes a stereo camera streams, by slot."""
    resolution: str | None = None
    """Camera stream resolution (the camera settings' stream resolution)."""
    log_level: LogLevel = "INFO"


class _Idle:
    """A session that waits to be stopped while the cameras stream."""

    async def run(self, stop_event: threading.Event | None) -> None:
        while stop_event is None or not stop_event.is_set():
            await asyncio.sleep(0.2)


def _run(cfg: SensorsCmdConfig, stop_event: threading.Event | None = None) -> None:
    """Stream the configured cameras until the panel stops the operation."""
    from .jog import _run_with_cameras

    if not cfg.cameras:
        raise SystemExit(
            "No cameras are set to stream. Assign them in Settings > Cameras."
        )
    asyncio.run(_run_with_cameras(cfg, _Idle(), stop_event, "sensors"))


def main(argv: list[str]) -> None:
    from .config import parse

    cfg = parse(SensorsCmdConfig, argv)
    logging.basicConfig(level=getattr(logging, cfg.log_level), force=True)
    try:
        _run(cfg)
    except KeyboardInterrupt:
        print("\nExiting sensors ...")
