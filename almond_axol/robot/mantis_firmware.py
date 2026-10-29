"""
Mantis trigger board firmware update over CAN.

The Mantis trigger board (firmware 0.2+, installed once over USB with its A/B
partition table) runs the jelly_legs update protocol on its own IDs: commands
on **0x460**, replies on **0x463**, image data on **0x464**. The firmware
README in the circuits-py repo (``designs/mantis/firmware``, "Firmware update
over CAN") is the spec, and its ``mantis_update.py`` the reference host tool.
The shared client lives in :mod:`almond_axol.robot.lift_firmware`.

Each Mantis sits alone on its own gripper bus (``can_mantis_l`` /
``can_mantis_r``) next to its Damiao gripper motor (0x008 / 0x018 / 0x7FF),
so both sides use the same IDs and are updated one bus at a time.

The board drives nothing, so there is nothing to lock: it never answers
UPDATE_BEGIN as busy, and it has no status broadcast to turn off. The trigger
keeps publishing on 0x009 during the transfer and goes quiet for the reboot
into the new image, which measures the trigger's rest position again.
"""

from __future__ import annotations

from pathlib import Path

from .lift_firmware import (
    BoardFirmwareUpdater,
    BoardSpec,
    FirmwareImage,
    FirmwareUpdateError,
)
from .lift_firmware import store_upload as _store_upload

__all__ = [
    "MANTIS",
    "UPLOAD_DIR",
    "FirmwareImage",
    "FirmwareUpdateError",
    "MantisFirmwareUpdater",
    "parse_image",
    "store_upload",
]

MANTIS = BoardSpec(
    name="mantis",
    board="the Mantis rig",
    file_kind="Mantis firmware file",
    # firmware.c's "mantis-fwid", zero-padded to the 16-byte magic field.
    ident_magic=b"mantis-fwid".ljust(16, b"\0"),
    min_version="0.2",
    id_cmd=0x460,
    id_reply=0x463,
    id_data=0x464,
    busy="busy",  # never sent: the board has nothing to wait for
    trial_lock="trigger still publishing",
)

# Where the control panel keeps uploaded images for `mantis.update --firmware`.
UPLOAD_DIR = Path.home() / ".almond" / "mantis-firmware"


def parse_image(data: bytes) -> FirmwareImage:
    """Validate a Mantis ``firmware.bin``; raises :class:`FirmwareUpdateError`."""
    return FirmwareImage.parse(data, MANTIS)


def store_upload(image: FirmwareImage, directory: Path | None = None) -> Path:
    """Save a validated upload under :data:`UPLOAD_DIR`; returns its path."""
    return _store_upload(image, UPLOAD_DIR if directory is None else directory)


class MantisFirmwareUpdater(BoardFirmwareUpdater):
    """Firmware updates for one Mantis trigger board, on its gripper bus."""

    spec = MANTIS

    async def quiet_status(self) -> None:
        """No-op: the trigger board has no status broadcast to turn off."""
