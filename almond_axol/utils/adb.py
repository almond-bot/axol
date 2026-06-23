"""adb (Android Debug Bridge) helpers for Quest-over-USB teleop.

The Quest headset can stream controller poses to the robot over a USB cable
instead of WiFi, sidestepping the 802.11 power-save buffering behind the
~150 ms pose gaps. The mechanism is ``adb reverse``: the headset's
``localhost:8000`` is forwarded over the cable to the robot's VR server, so the
WebXR app reaches it at ``wss://localhost:8000``. Camera video still rides the
LAN (WebRTC can't cross the TCP port-forward), so USB is pose-only.

This module is the single place that knows how to install adb (used by
``axol provision``) and how to query/establish the reverse tunnel (used by the
``axol serve`` control panel).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass

from .sudo import prime_sudo, run_root

_logger = logging.getLogger(__name__)

# The VR WebSocket server port; we forward the headset's loopback copy of it.
VR_PORT = 8000

# Meta/Oculus USB vendor id. The udev rule lets a non-root user in ``plugdev``
# open the headset; the root ``axol serve`` process doesn't need it, but it
# keeps interactive ``adb`` working too.
_OCULUS_RULE_PATH = "/etc/udev/rules.d/51-oculus.rules"
_OCULUS_RULE = 'SUBSYSTEM=="usb", ATTR{idVendor}=="2833", MODE="0660", GROUP="plugdev"'

# ``adb`` is the binary; ``android-sdk-platform-tools-common`` ships the base
# Android udev rules (plugdev-group device nodes).
_APT_PACKAGES = ("adb", "android-sdk-platform-tools-common")


def _adb() -> str | None:
    return shutil.which("adb")


def install() -> None:
    """Install adb + the Oculus udev rule (idempotent, best-effort, needs root).

    No-op with a hint when apt-get is missing or root can't be obtained.
    """
    if _adb() is not None and os.path.isfile(_OCULUS_RULE_PATH):
        _logger.info("adb already installed; Quest-over-USB ready")
        return
    if shutil.which("apt-get") is None:
        _logger.info("apt-get not found; skipping adb install")
        return
    if not prime_sudo():
        _logger.warning(
            "adb needs root to install; run manually: sudo apt-get install -y %s",
            " ".join(_APT_PACKAGES),
        )
        return
    if _adb() is None:
        run_root(["apt-get", "install", "-y", *_APT_PACKAGES])
    run_root(["tee", _OCULUS_RULE_PATH], input_text=_OCULUS_RULE + "\n")
    run_root(["udevadm", "control", "--reload-rules"])
    run_root(["udevadm", "trigger"])
    _logger.info("adb installed; Quest-over-USB ready")


@dataclass(frozen=True)
class AdbStatus:
    """Snapshot of the adb device + reverse-tunnel state for the control panel."""

    installed: bool
    serial: str | None
    # "none" | "device" | "unauthorized" | "offline" | (raw adb state string)
    state: str
    reverse_active: bool

    @property
    def ready(self) -> bool:
        """True when a headset is authorized and the pose tunnel is live."""
        return self.state == "device" and self.reverse_active


def _run(
    args: list[str], timeout: float = 10.0
) -> subprocess.CompletedProcess[str] | None:
    adb = _adb()
    if adb is None:
        return None
    try:
        return subprocess.run(
            [adb, *args], capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _logger.warning("adb %s failed: %s", " ".join(args), exc)
        return None


def _first_device() -> tuple[str | None, str]:
    """Return (serial, state) of the first attached device, or (None, "none")."""
    proc = _run(["devices"])
    if proc is None or proc.returncode != 0:
        return None, "none"
    for line in proc.stdout.splitlines()[1:]:  # skip "List of devices attached"
        line = line.strip()
        if not line or "\t" not in line:
            continue
        serial, _, state = line.partition("\t")
        return serial.strip(), state.strip()
    return None, "none"


def _reverse_active(port: int) -> bool:
    proc = _run(["reverse", "--list"])
    if proc is None or proc.returncode != 0:
        return False
    needle = f"tcp:{port} tcp:{port}"
    return any(needle in line for line in proc.stdout.splitlines())


def status(port: int = VR_PORT) -> AdbStatus:
    """Return the current adb device + reverse-tunnel status."""
    if _adb() is None:
        return AdbStatus(
            installed=False, serial=None, state="none", reverse_active=False
        )
    serial, state = _first_device()
    return AdbStatus(
        installed=True,
        serial=serial,
        state=state,
        reverse_active=_reverse_active(port),
    )


def connect(port: int = VR_PORT) -> AdbStatus:
    """Establish the reverse tunnel (headset localhost:port → this host:port).

    The first adb command against a freshly connected headset also triggers the
    USB-debugging authorization popup on the device. Returns the resulting
    status so the caller can surface "authorize on headset" vs "ready".
    """
    _run(["reverse", f"tcp:{port}", f"tcp:{port}"])
    return status(port)
