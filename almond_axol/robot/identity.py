"""Resolve the Axol hub USB serial used as the robot's identity.

This module is intentionally lightweight so robot configuration can validate
per-robot caches without importing the CLI package. The dual-channel arm hub
is distinguishable from otherwise-identical single-channel CAN adapters by
its two ``dev_id`` values.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from ..constants import CAN_LEFT, CAN_RIGHT

_VID = "1d50"
_PID = "606f"
_UDEV_RULES_FILE = Path("/etc/udev/rules.d/90-can.rules")


def _udev_attr(info: str, attr: str) -> str:
    """First value of ``attr`` in ``udevadm info -a`` output."""
    return next(
        (line.split('"')[1] for line in info.splitlines() if attr in line),
        "",
    )


def scan_adapters() -> dict[str, dict[str, Any]]:
    """Return attached gs_usb CAN adapters keyed by USB serial."""
    adapters: dict[str, dict[str, Any]] = {}
    for iface_path in Path("/sys/class/net").glob("can*"):
        info = subprocess.run(
            ["udevadm", "info", "-a", "-p", str(iface_path)],
            capture_output=True,
            text=True,
        ).stdout
        vid = _udev_attr(info, "ATTRS{idVendor}").lower()
        pid = _udev_attr(info, "ATTRS{idProduct}").lower()
        if 'DRIVERS=="gs_usb"' not in info and (vid, pid) != (_VID, _PID):
            continue
        serial = _udev_attr(info, "ATTRS{serial}")
        if not serial:
            continue
        try:
            dev_id = int(_udev_attr(info, "ATTR{dev_id}"), 16)
        except ValueError:
            continue
        entry = adapters.setdefault(serial, {"vid": vid, "pid": pid, "dev_ids": set()})
        entry["dev_ids"].add(dev_id)
    return adapters


def attached_hub_serials() -> list[str]:
    """Serials of attached dual-channel Axol arm hubs."""
    return [
        serial
        for serial, adapter in scan_adapters().items()
        if len(adapter["dev_ids"]) >= 2
        and (adapter["vid"], adapter["pid"]) == (_VID, _PID)
    ]


def serial_of_interface(name: str) -> str | None:
    """The USB serial behind a named CAN interface, when it is attached."""
    iface_path = Path("/sys/class/net") / name
    if not iface_path.exists():
        return None
    info = subprocess.run(
        ["udevadm", "info", "-a", "-p", str(iface_path)],
        capture_output=True,
        text=True,
    ).stdout
    return next(
        (line.split('"')[1] for line in info.splitlines() if "ATTRS{serial}" in line),
        None,
    )


def rules_serial_for(name: str) -> str | None:
    """The serial pinned to ``name`` in the persistent udev rules."""
    try:
        rules = _UDEV_RULES_FILE.read_text()
    except OSError:
        return None
    match = re.search(
        r'ATTRS\{serial\}=="([^"]+)"[^\n]*NAME="' + re.escape(name) + '"',
        rules,
    )
    return match.group(1) if match else None


def configured_hub_serial() -> str | None:
    """The live or persisted hub serial selected by a previous setup."""
    for iface in (CAN_LEFT, CAN_RIGHT):
        serial = serial_of_interface(iface)
        if serial:
            return serial
    return rules_serial_for(CAN_LEFT) or rules_serial_for(CAN_RIGHT)


def select_hub_serial(configured: str | None, attached: list[str]) -> str | None:
    """Choose an attachment-aware hub identity from discovered serials."""
    if configured and (configured in attached or not attached):
        return configured
    if len(attached) == 1:
        return attached[0]
    if not attached:
        return None
    raise RuntimeError(
        "Multiple CAN adapters found — run `axol can.setup` once to pick the Axol's"
    )


def hub_serial() -> str | None:
    """Return the attachment-aware Axol hub identity.

    A stale persisted pin never wins over a different attached hub. The pin
    remains an offline fallback when no dual-channel hub is currently visible.
    """
    return select_hub_serial(configured_hub_serial(), attached_hub_serials())
