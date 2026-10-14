"""The Axol arm hub: the RP2350 USB-to-CAN bridge for both arms.

Its firmware lives in ``firmware/axol_hub`` in the repository; the built image
ships here as ``axol_hub.uf2`` and :func:`ensure_hub_firmware` keeps attached
hubs on it (see :mod:`almond_axol.hub.firmware`).
"""

from .firmware import (
    FIRMWARE_VERSION,
    Hub,
    HubFirmwareError,
    ensure_hub_firmware,
    flash,
    format_version,
    hubs_for_channels,
)

__all__ = [
    "FIRMWARE_VERSION",
    "Hub",
    "HubFirmwareError",
    "ensure_hub_firmware",
    "flash",
    "format_version",
    "hubs_for_channels",
]
