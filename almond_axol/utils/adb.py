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

import grp
import logging
import os
import pwd
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .sudo import prime_sudo, run_root

_logger = logging.getLogger(__name__)

# The VR WebSocket server port; we forward the headset's loopback copy of it.
VR_PORT = 8000

# Meta/Oculus USB vendor id. The udev rule lets a non-root user open the
# headset for interactive ``adb`` (the root ``axol serve`` process opens it
# directly and doesn't need the rule).
#
# We hand the device node to ``dialout`` rather than the conventional
# ``plugdev`` group: operators are already in ``dialout`` for CAN/serial
# access, so ``adb`` works immediately without adding them to a new group and
# forcing a re-login. ``install()`` still adds the operator to the group as a
# safety net for hosts where they aren't a member yet.
_QUEST_GROUP = "dialout"
_OCULUS_RULE_PATH = "/etc/udev/rules.d/51-oculus.rules"
_OCULUS_RULE = (
    'SUBSYSTEM=="usb", ATTR{idVendor}=="2833", '
    'MODE="0660", GROUP="' + _QUEST_GROUP + '"'
)

# ``adb`` is the binary; ``android-sdk-platform-tools-common`` ships the base
# Android udev rules. Those use ``plugdev``; our Oculus rule above supersedes
# them for the Quest's vendor id.
_APT_PACKAGES = ("adb", "android-sdk-platform-tools-common")


def _adb() -> str | None:
    return shutil.which("adb")


def _read_rule() -> str:
    """Return the current Oculus udev rule contents (empty if absent)."""
    try:
        return Path(_OCULUS_RULE_PATH).read_text().strip()
    except OSError:
        return ""


def _operator_user() -> str | None:
    """Best-effort operator login to grant headset (``dialout``) access.

    ``axol serve`` runs as root under systemd with no ``SUDO_USER``, so fall
    back to the owner of the first ``/home/*`` entry — the same heuristic the
    hosted installer uses to locate the operator's account.
    """
    user = os.environ.get("SUDO_USER")
    if user and user != "root":
        return user
    try:
        homes = sorted(Path("/home").iterdir())
    except OSError:
        return None
    for home in homes:
        try:
            return home.owner()
        except (KeyError, OSError):
            continue
    return None


def _in_group(user: str, group: str) -> bool:
    """True when ``user`` already belongs to ``group`` (primary or secondary)."""
    try:
        gr = grp.getgrnam(group)
    except KeyError:
        return False
    if user in gr.gr_mem:
        return True
    try:
        return pwd.getpwnam(user).pw_gid == gr.gr_gid
    except KeyError:
        return False


def _grant_operator_access() -> None:
    """Add the operator to the headset group so interactive ``adb`` works.

    No-op (and no ``sudo`` prompt) when the operator is already a member; the
    change otherwise takes effect on their next login. Best-effort.
    """
    user = _operator_user()
    if user is None or _in_group(user, _QUEST_GROUP):
        return
    run_root(["usermod", "-aG", _QUEST_GROUP, user])


def install() -> None:
    """Install adb + the Oculus udev rule and grant the operator headset access.

    Idempotent and best-effort. No-ops with a hint when apt-get is missing or
    root can't be obtained. Rewrites the udev rule when its contents drift
    (e.g. an older rule that used a different group).
    """
    rule_ok = _read_rule() == _OCULUS_RULE
    if _adb() is not None and rule_ok:
        _grant_operator_access()
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
        # `update` refreshes a stale/empty package index (best-effort, like the
        # GStreamer step); `install` is checked so a failure doesn't fall
        # through to claiming success with adb still missing.
        run_root(["apt-get", "update"])
        try:
            run_root(["apt-get", "install", "-y", *_APT_PACKAGES], check=True)
        except RuntimeError as exc:
            _logger.warning(
                "adb install failed (%s); Quest-over-USB unavailable. "
                "Retry with: sudo apt-get install -y %s",
                exc,
                " ".join(_APT_PACKAGES),
            )
            return
    if not rule_ok:
        run_root(["tee", _OCULUS_RULE_PATH], input_text=_OCULUS_RULE + "\n")
        run_root(["udevadm", "control", "--reload-rules"])
        run_root(["udevadm", "trigger"])
    _grant_operator_access()
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
