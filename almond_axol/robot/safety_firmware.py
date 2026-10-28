"""
Almond Safety board firmware update over CAN.

The safety board (firmware 0.3+, installed once over USB with its A/B
partition table) runs the jelly_legs update protocol on its own IDs: commands
on **0x440**, replies on **0x443**, image data on **0x444**. The firmware
README in the circuits-py repo (``designs/almond_safety/firmware``, "Firmware
update over CAN") is the spec, and its ``safety_update.py`` the reference
host tool. The shared client lives in :mod:`almond_axol.robot.lift_firmware`.

Both boards share the lift bus without getting in each other's way:

- Each board only reacts to its own command and data IDs (0x420 / 0x424 for
  the legs, 0x440 / 0x444 here), and each client only listens to its own
  reply ID.
- Each image carries its board's identity block. :meth:`FirmwareImage.parse`
  and the board itself both reject the other board's firmware.

The safety board only takes an update with its output off, so
:class:`SafetyFirmwareUpdater` cuts power (a software stop) before it starts.
From then until the new image is confirmed the board keeps its software
enable locked off, so neither the green on-switch nor a remote start can
switch power back on mid-update. Power stays off afterwards until a start.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .lift_firmware import (
    BoardFirmwareUpdater,
    BoardSpec,
    FirmwareImage,
    FirmwareUpdateError,
)
from .lift_firmware import store_upload as _store_upload

__all__ = [
    "SAFETY",
    "UPLOAD_DIR",
    "FirmwareImage",
    "FirmwareUpdateError",
    "SafetyFirmwareUpdater",
    "parse_image",
    "store_upload",
]

SAFETY = BoardSpec(
    name="almond_safety",
    board="the safety board",
    file_kind="safety board firmware file",
    ident_magic=b"alm_safety-fwid\0",
    min_version="0.3",
    id_cmd=0x440,
    id_reply=0x443,
    id_data=0x444,
    busy="busy (power is on, or a remote start is running)",
    trial_lock="power locked off",
)

_OP_SW_STOP = 0x01

# Where the control panel keeps uploaded images for `safety.update --firmware`.
UPLOAD_DIR = Path.home() / ".almond" / "safety-firmware"


def parse_image(data: bytes) -> FirmwareImage:
    """Validate a safety board ``firmware.bin``; raises :class:`FirmwareUpdateError`."""
    return FirmwareImage.parse(data, SAFETY)


def store_upload(image: FirmwareImage, directory: Path | None = None) -> Path:
    """Save a validated upload under :data:`UPLOAD_DIR`; returns its path."""
    return _store_upload(image, UPLOAD_DIR if directory is None else directory)


class SafetyFirmwareUpdater(BoardFirmwareUpdater):
    """Firmware updates for the Almond Safety board. Cuts power first."""

    spec = SAFETY

    async def _before_begin(self, log: Callable[[str], None]) -> None:
        log("cutting power to Axol for the update (software stop)")
        await self._cmd(_OP_SW_STOP)

    async def _while_busy(self) -> None:
        # Still on: the stop frame may have been lost.
        await self._cmd(_OP_SW_STOP)
