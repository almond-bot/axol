"""Enumeration of locally connected ZED cameras.

Mono ZED-X One cameras live in ``sl.CameraOne``'s device list and stereo
ZED X cameras in ``sl.Camera``'s, so both are queried. The ZED X daemon only
enumerates GMSL cameras when it starts — a camera plugged in after boot stays
invisible until the daemon restarts (see :func:`..daemon.restart_zed_daemon`).
"""

from __future__ import annotations

import logging
from typing import TypedDict

_logger = logging.getLogger(__name__)


class ZedDevice(TypedDict):
    """One detected ZED camera."""

    serial: int
    model: str
    kind: str  # "mono" (ZED-X One) | "stereo" (ZED X)


def list_zed_devices() -> list[ZedDevice]:
    """Return every locally connected ZED camera (mono + stereo).

    Raises:
        ImportError: If pyzed is not installed (run ``axol zed.install``).
    """
    import pyzed.sl as sl

    devices: dict[int, ZedDevice] = {}
    for d in sl.CameraOne.get_device_list():
        serial = int(d.serial_number)
        devices[serial] = {
            "serial": serial,
            "model": str(d.camera_model),
            "kind": "mono",
        }
    for d in sl.Camera.get_device_list():
        serial = int(d.serial_number)
        # A camera present in both lists is a stereo-capable ZED X.
        devices[serial] = {
            "serial": serial,
            "model": str(d.camera_model),
            "kind": "stereo",
        }
    out = sorted(devices.values(), key=lambda d: d["serial"])
    _logger.debug(
        "Detected %d ZED camera(s): %s",
        len(out),
        ", ".join(str(d["serial"]) for d in out) or "<none>",
    )
    return out


def stereo_serials() -> set[int]:
    """Serials of locally connected stereo (ZED X) cameras.

    Best-effort companion to :func:`list_zed_devices` used to auto-detect
    whether a configured camera is stereo from its serial, so operators never
    have to flag it manually. Any enumeration failure (pyzed missing, daemon
    down) returns an empty set, in which case callers treat cameras as mono.
    """
    try:
        return {d["serial"] for d in list_zed_devices() if d["kind"] == "stereo"}
    except Exception:  # noqa: BLE001 - detection is best-effort
        _logger.debug("stereo_serials: ZED enumeration failed", exc_info=True)
        return set()
