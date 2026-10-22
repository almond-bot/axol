"""Keep the Axol arm hub on the firmware this package ships.

The hub is an RP2350 that bridges
the two arm CAN buses to USB as a gs_usb device. Its firmware decides what
happens to motor commands when the bus dies — whether a setpoint parked behind
an e-stop can still reach the motors when they are powered back on — so a
robot must not keep running an old build. :func:`ensure_hub_firmware` runs on
every bring-up (``Axol.connect()`` and the realtime ``enable()``): it reads
the hub's version from sysfs (its USB ``bcdDevice``, readable unprivileged)
and reflashes it over USB when it is older than :data:`FIRMWARE_VERSION`.

Flashing needs root (usbfs control transfer, mounting the bootloader's disk).
The hosted ``axol serve`` is root; an interactive session escalates through
``sudo`` like the CAN bring-up does, and a session that cannot escalate keeps
running on the old firmware with a warning naming ``sudo axol hub.flash``.

Reflashing reboots the hub, so both arm buses go silent for a few seconds.
Motors keep executing their last command meanwhile (the arms set no CAN
timeout), and the reboot itself discards anything the old firmware had
queued. A hub whose firmware predates the reflash request (before 2026-08-17)
cannot be rebooted into its bootloader from the host; it needs one manual
flash (hold BOOTSEL while plugging it in, then ``sudo axol hub.flash``).
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import glob
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

_logger = logging.getLogger(__name__)

#: BCD firmware version of the bundled image (0x0200 = 2.00): the
#: ``AXOL_HUB_FW_VERSION`` it was built with. The test suite checks the image
#: itself reports it.
FIRMWARE_VERSION = 0x0201

#: The bundled image. The source lives in the circuits-py repository
#: (``designs/axol_hub/firmware``); this is its built ``firmware.uf2``, copied
#: here when the firmware changes, together with a :data:`FIRMWARE_VERSION`
#: bump to the new ``AXOL_HUB_FW_VERSION``.
FIRMWARE_PATH = Path(__file__).resolve().parent / "axol_hub.uf2"

HUB_VID, HUB_PID = 0x1D50, 0x606F  # gs_usb IDs the hub enumerates with
BOOT_VID, BOOT_PID = 0x2E8A, 0x000F  # RP2350 bootrom
# 1d50:606f is the generic candleLight identity; only these strings are ours.
HUB_MANUFACTURER = "Almond"
HUB_PRODUCT = "Axol CAN Bus"

# Vendor control requests (firmware.c). Device recipient, so they work while
# gs_usb has the interface claimed.
AXOL_BREQ_RESET_INFO = 0xFE
AXOL_BREQ_BOOTLOADER = 0xFF
_VENDOR_IN = 0xC0
_VENDOR_OUT = 0x40

_SYSFS = Path("/sys")

# Warn once per hub per process when an outdated hub cannot be flashed, not on
# every reconnect.
_warned: set[str] = set()


class HubFirmwareError(RuntimeError):
    """The hub could not be brought onto the bundled firmware."""


@dataclass(frozen=True)
class Hub:
    """An attached Axol hub, located through sysfs."""

    port: str  # USB port path, e.g. "1-3" — stable across re-enumeration
    serial: str
    version: int  # bcdDevice

    @property
    def outdated(self) -> bool:
        return self.version < FIRMWARE_VERSION


def format_version(version: int) -> str:
    """``0x0200`` -> ``"2.00"`` (the BCD bcdDevice convention)."""
    return f"{version >> 8:x}.{version & 0xFF:02x}"


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _usb_ids(device: Path) -> tuple[int, int] | None:
    vid, pid = _read(device / "idVendor"), _read(device / "idProduct")
    if vid is None or pid is None:
        return None
    try:
        return int(vid, 16), int(pid, 16)
    except ValueError:
        return None


def _hub_at(device: Path) -> Hub | None:
    """The Axol hub at USB device directory ``device``, or None if it is not one."""
    if _usb_ids(device) != (HUB_VID, HUB_PID):
        return None
    if (
        _read(device / "manufacturer") != HUB_MANUFACTURER
        or _read(device / "product") != HUB_PRODUCT
    ):
        return None
    try:
        version = int(_read(device / "bcdDevice") or "", 16)
    except ValueError:
        return None
    return Hub(port=device.name, serial=_read(device / "serial") or "", version=version)


def hubs_for_channels(channels: list[str], *, sysfs: Path = _SYSFS) -> list[Hub]:
    """The Axol hubs behind the given CAN interfaces (deduplicated).

    Interfaces that are missing, or that sit on some other adapter (a bench
    candleLight, the Jetson's mttcan), are skipped.
    """
    hubs: dict[str, Hub] = {}
    for channel in channels:
        # /sys/class/net/<if>/device is the USB interface; its parent is the device.
        interface = sysfs / "class" / "net" / channel / "device"
        try:
            device = interface.resolve(strict=True).parent
        except OSError:
            continue
        hub = _hub_at(device)
        if hub is not None:
            hubs.setdefault(hub.port, hub)
    return list(hubs.values())


def _usb_device(port: str, sysfs: Path = _SYSFS) -> Path:
    return sysfs / "bus" / "usb" / "devices" / port


def _devnode(device: Path) -> str:
    bus, num = _read(device / "busnum"), _read(device / "devnum")
    if bus is None or num is None:
        raise HubFirmwareError(f"USB device {device.name} disappeared")
    return f"/dev/bus/usb/{int(bus):03d}/{int(num):03d}"


class _CtrlTransfer(ctypes.Structure):
    _fields_ = [
        ("bRequestType", ctypes.c_uint8),
        ("bRequest", ctypes.c_uint8),
        ("wValue", ctypes.c_uint16),
        ("wIndex", ctypes.c_uint16),
        ("wLength", ctypes.c_uint16),
        ("timeout", ctypes.c_uint32),
        ("data", ctypes.c_void_p),
    ]


# _IOWR('U', 0, struct usbdevfs_ctrltransfer)
_USBDEVFS_CONTROL = (3 << 30) | (ctypes.sizeof(_CtrlTransfer) << 16) | (ord("U") << 8)


def _control(devnode: str, request_type: int, request: int, length: int = 0) -> bytes:
    """One usbfs control transfer (root). Raises OSError as the kernel reports it."""
    buf = ctypes.create_string_buffer(max(length, 1))
    xfer = _CtrlTransfer(
        request_type, request, 0, 0, length, 1000, ctypes.cast(buf, ctypes.c_void_p)
    )
    fd = os.open(devnode, os.O_RDWR)
    try:
        n = fcntl.ioctl(fd, _USBDEVFS_CONTROL, xfer)
    finally:
        os.close(fd)
    return buf.raw[:n]


def reset_info(hub: Hub, *, sysfs: Path = _SYSFS) -> dict[str, int] | None:
    """Why the hub last reset and how long it has been up (root), if it can say."""
    import struct

    try:
        raw = _control(
            _devnode(_usb_device(hub.port, sysfs)),
            _VENDOR_IN,
            AXOL_BREQ_RESET_INFO,
            64,
        )
        version, chip_reset, wd_reason, bod, uptime_us = struct.unpack_from(
            "<IIIIQ", raw
        )
    except (OSError, struct.error):
        return None
    return {
        "info_version": version,
        "chip_reset": chip_reset,
        "watchdog_reason": wd_reason,
        "bod": bod,
        "uptime_us": uptime_us,
    }


def _wait_for(predicate, timeout_s: float, what: str):
    deadline = time.monotonic() + timeout_s
    while True:
        result = predicate()
        if result:
            return result
        if time.monotonic() >= deadline:
            raise HubFirmwareError(f"timed out waiting for {what}")
        time.sleep(0.2)


def _in_bootloader(port: str, sysfs: Path) -> bool:
    return _usb_ids(_usb_device(port, sysfs)) == (BOOT_VID, BOOT_PID)


def _bootloader_disk(port: str, sysfs: Path) -> str | None:
    """The block device the bootrom at ``port`` exposes (its partition if any)."""
    device = _usb_device(port, sysfs)
    for block in sorted(glob.glob(f"{device}/{port}:*/host*/target*/*/block/*")):
        name = os.path.basename(block)
        part = Path(block) / f"{name}1"
        return f"/dev/{name}1" if part.exists() else f"/dev/{name}"
    return None


def _copy_uf2(disk: str, image: Path) -> None:
    """Write ``image`` to the bootloader's disk.

    The bootrom reboots the moment the last block lands, so the disk can
    vanish under the tail of the write, the fsync, or the close. An I/O error
    here therefore says nothing about whether the flash took; the caller
    decides that by waiting for the hub to come back (:func:`flash`).
    """
    mounted = subprocess.run(
        ["findmnt", "-n", "-o", "TARGET", "-S", disk], capture_output=True, text=True
    ).stdout.strip()
    own_mount = None
    if not mounted:
        own_mount = tempfile.mkdtemp(prefix="axol-hub-uf2-")
        result = subprocess.run(
            ["mount", disk, own_mount], capture_output=True, text=True
        )
        if result.returncode != 0:
            os.rmdir(own_mount)
            raise HubFirmwareError(
                f"could not mount the bootloader disk {disk}: "
                f"{(result.stderr or '').strip() or result.returncode}"
            )
        mounted = own_mount
    try:
        target = os.path.join(mounted, image.name)
        with open(image, "rb") as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
    except OSError as exc:
        _logger.debug("bootloader disk went away during the copy: %s", exc)
    finally:
        if own_mount:
            # The disk may already be gone; a lazy unmount cleans up either way.
            subprocess.run(["umount", "-l", own_mount], capture_output=True)
            try:
                os.rmdir(own_mount)
            except OSError:
                pass


def flash(port: str, *, image: Path = FIRMWARE_PATH, sysfs: Path = _SYSFS) -> int:
    """Reflash the hub at USB port ``port`` with ``image`` (root). Returns its new version.

    Works from a running hub (it is asked to reboot into its bootloader) or
    from one already sitting in the bootloader — e.g. after an interrupted
    flash, or the manual BOOTSEL flash an old hub needs.
    """
    if os.geteuid() != 0:
        raise HubFirmwareError("flashing the Axol hub needs root")
    if not image.is_file():
        raise HubFirmwareError(f"hub firmware image missing: {image}")
    device = _usb_device(port, sysfs)
    if not _in_bootloader(port, sysfs):
        if _hub_at(device) is None:
            raise HubFirmwareError(f"no Axol hub at USB port {port}")
        try:
            _control(_devnode(device), _VENDOR_OUT, AXOL_BREQ_BOOTLOADER)
        except OSError as exc:
            if exc.errno == errno.EPIPE:
                # Stalled: the firmware does not know the request.
                raise HubFirmwareError(
                    f"the Axol hub at USB port {port} runs firmware too old to "
                    "update over USB; flash it once by hand: hold its BOOTSEL "
                    "button while plugging it in, then run `sudo axol hub.flash "
                    f"--port {port}`"
                ) from exc
            # Otherwise the hub most likely rebooted before the status stage.
        _wait_for(lambda: _in_bootloader(port, sysfs), 10, "the hub's bootloader")
    disk = _wait_for(
        lambda: _bootloader_disk(port, sysfs), 15, "the bootloader's USB disk"
    )
    _copy_uf2(disk, image)
    try:
        hub = _wait_for(lambda: _hub_at(device), 20, "the hub to restart")
    except HubFirmwareError:
        if _in_bootloader(port, sysfs):
            # The bootrom reboots once the image lands; with BOOTSEL still
            # held (a manual flash) it comes straight back to the bootloader.
            raise HubFirmwareError(
                f"the hub at USB port {port} restarted into its bootloader "
                "again; if BOOTSEL is still held, release it and replug the "
                "hub to boot the new firmware, otherwise rerun the flash"
            ) from None
        raise
    return hub.version


def _wait_for_channels(channels: list[str], timeout_s: float = 15.0) -> None:
    """After a hub restart, wait for its interfaces to be back up.

    The hotplug unit ``axol can.setup`` installs brings them up on
    re-enumeration; configure them directly if it has not within the timeout.
    """
    from ..cli.can.setup import bring_up_interfaces, iface_up

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if all(iface_up(ch) for ch in channels):
            return
        time.sleep(0.25)
    bring_up_interfaces(channels)


def ensure_hub_firmware(channels: list[str], *, sysfs: Path = _SYSFS) -> list[Hub]:
    """Reflash any Axol hub behind ``channels`` that runs older firmware.

    Must run while nobody owns the interfaces (the same window as the stale
    queue purge): the hub reboots and its netdevs are recreated. Returns the
    hubs that were reflashed (as they were before the flash).

    Raises:
        HubFirmwareError: A flash started and the hub did not come back as a
            CAN adapter (stuck in its bootloader, or gone), so the buses are
            unusable. An outdated hub that is still running — it could not be
            flashed from here (no root, firmware too old to reboot itself) or
            came back on the old image — is logged instead, not retried in
            this process, and the bring-up continues on the old firmware.
    """
    if os.environ.get("AXOL_HUB_FLASH", "1") == "0":
        return []
    outdated = [
        hub
        for hub in hubs_for_channels(channels, sysfs=sysfs)
        # A hub that already failed to update in this process is not retried
        # on every reconnect (each attempt may prompt for sudo).
        if hub.outdated and hub.port not in _warned
    ]
    flashed: list[Hub] = []
    attempted = False
    for hub in outdated:
        _logger.warning(
            "Axol hub %s (USB %s) runs firmware %s; updating to %s — the arm "
            "buses go quiet for a few seconds",
            hub.serial,
            hub.port,
            format_version(hub.version),
            format_version(FIRMWARE_VERSION),
        )
        # From here the hub may have rebooted (and its netdevs been
        # recreated) whatever the outcome.
        attempted = True
        try:
            if os.geteuid() == 0:
                new_version = flash(hub.port, sysfs=sysfs)
            else:
                new_version = _flash_via_sudo(hub, sysfs=sysfs)
        except HubFirmwareError as exc:
            if not _in_bootloader(hub.port, sysfs) and _hub_at(
                _usb_device(hub.port, sysfs)
            ):
                # Never left the old firmware: the buses still work.
                _warned.add(hub.port)
                _logger.error("Axol hub firmware update skipped: %s", exc)
                continue
            raise
        if new_version < FIRMWARE_VERSION:
            # Back up and working, just not on the new image (a wrong image,
            # an interrupted copy): carry on rather than refuse the robot.
            _warned.add(hub.port)
            _logger.error(
                "Axol hub at USB %s came back on firmware %s after flashing %s",
                hub.port,
                format_version(new_version),
                format_version(FIRMWARE_VERSION),
            )
            continue
        _logger.warning(
            "Axol hub %s updated to firmware %s",
            hub.serial,
            format_version(new_version),
        )
        flashed.append(hub)
    if attempted:
        _wait_for_channels(channels)
    return flashed


def _flash_via_sudo(hub: Hub, *, sysfs: Path) -> int:
    """Run the flash as root through ``sudo`` (prompts on a terminal)."""
    from ..utils.sudo import run_root

    proc = run_root(
        [sys.executable, "-m", "almond_axol.hub", "flash", "--port", hub.port]
    )
    if proc.returncode != 0:
        # The flash reports its failure on stderr (``ERROR: ...``), as does
        # sudo when it cannot escalate; either way the last line says why.
        lines = [ln for ln in (proc.stderr or "").splitlines() if ln.strip()]
        detail = lines[-1].removeprefix("ERROR: ") if lines else ""
        raise HubFirmwareError(
            detail or f"`sudo axol hub.flash` exited {proc.returncode}"
        )
    updated = _hub_at(_usb_device(hub.port, sysfs))
    if updated is None:
        raise HubFirmwareError(f"the hub at USB {hub.port} did not come back")
    return updated.version
