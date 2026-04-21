"""
axol can.setup

Sets persistent CAN interface names for the Almond Axol arm CAN bus adapter
and registers a root crontab @reboot entry to bring up the interfaces.

The Almond Axol adapter (VID 0x1D50 / PID 0x606F) exposes two CAN channels
on a single USB device:
  channel 0 (dev_id 0x0) -> can_alm_axol_l  (left arm)
  channel 1 (dev_id 0x1) -> can_alm_axol_r  (right arm)
"""

import subprocess
import sys
from pathlib import Path

from ...shared import CAN_LEFT, CAN_RIGHT

_VID = "1d50"
_PID = "606f"
_CAN_L = CAN_LEFT
_CAN_R = CAN_RIGHT
_BITRATE = 1_000_000
_TXQUEUELEN = 512

_UDEV_RULES_FILE = Path("/etc/udev/rules.d/90-can.rules")
_CAN_DIR = Path.home() / ".almond" / "can"
_CRON_SCRIPT = _CAN_DIR / "startup.sh"


def _die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _find_serial() -> str:
    print(f"Scanning for Almond Axol CAN adapter ({_VID}:{_PID})...")

    serials: list[str] = []
    for iface_path in Path("/sys/class/net").glob("can*"):
        info = subprocess.run(
            ["udevadm", "info", "-a", "-p", str(iface_path)],
            capture_output=True,
            text=True,
        ).stdout

        vid = next(
            (
                line.split('"')[1]
                for line in info.splitlines()
                if "ATTRS{idVendor}" in line
            ),
            "",
        )
        pid = next(
            (
                line.split('"')[1]
                for line in info.splitlines()
                if "ATTRS{idProduct}" in line
            ),
            "",
        )

        if vid.lower() == _VID and pid.lower() == _PID:
            serial = next(
                (
                    line.split('"')[1]
                    for line in info.splitlines()
                    if "ATTRS{serial}" in line
                ),
                "",
            )
            if serial:
                serials.append(serial)

    unique = list(dict.fromkeys(serials))

    if not unique:
        print(
            "\n  No adapter found. Enter the serial number manually (blank to abort):"
        )
        serial = input("  Serial: ").strip()
        if not serial:
            _die("No serial provided. Connect the device and re-run.")
        return serial

    if len(unique) == 1:
        print(f"  Found adapter — serial: {unique[0]}")
        return unique[0]

    print("  Multiple adapters found:")
    for i, s in enumerate(unique):
        print(f"    [{i}] {s}")
    idx = input("  Select adapter index [0]: ").strip() or "0"
    return unique[int(idx)]


def _write_udev_rules(serial: str) -> None:
    print(f"Writing udev rules to {_UDEV_RULES_FILE} (requires sudo)...")
    content = (
        f"# Almond Axol dual-channel CAN adapter\n"
        f"# Adapter serial: {serial}\n"
        f"# Channel 0 -> left arm\n"
        f'SUBSYSTEM=="net", ACTION=="add", ATTRS{{idVendor}}=="{_VID}", ATTRS{{idProduct}}=="{_PID}", ATTRS{{serial}}=="{serial}", ATTR{{dev_id}}=="0x0", NAME="{_CAN_L}"\n'
        f"# Channel 1 -> right arm\n"
        f'SUBSYSTEM=="net", ACTION=="add", ATTRS{{idVendor}}=="{_VID}", ATTRS{{idProduct}}=="{_PID}", ATTRS{{serial}}=="{serial}", ATTR{{dev_id}}=="0x1", NAME="{_CAN_R}"\n'
    )
    subprocess.run(
        ["sudo", "tee", str(_UDEV_RULES_FILE)],
        input=content,
        text=True,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    print("  Done.")


def _reload_udev() -> None:
    print("Reloading udev rules (requires sudo)...")
    subprocess.run(["sudo", "udevadm", "control", "--reload-rules"], check=True)
    subprocess.run(["sudo", "systemctl", "restart", "systemd-udevd"], check=True)
    print("  Done.")


def _rename_interfaces(serial: str) -> None:
    """Rename existing canX interfaces to their target names without replug."""
    print("Renaming CAN interfaces (requires sudo)...")
    target = {0: _CAN_L, 1: _CAN_R}

    for iface_path in Path("/sys/class/net").glob("can*"):
        iface = iface_path.name
        info = subprocess.run(
            ["udevadm", "info", "-a", "-p", str(iface_path)],
            capture_output=True,
            text=True,
        ).stdout

        iface_serial = next(
            (
                line.split('"')[1]
                for line in info.splitlines()
                if "ATTRS{serial}" in line
            ),
            "",
        )
        if iface_serial != serial:
            continue

        dev_id_str = next(
            (
                line.split('"')[1]
                for line in info.splitlines()
                if "ATTR{dev_id}" in line
            ),
            "",
        )
        try:
            dev_id = int(dev_id_str, 16)
        except ValueError:
            continue

        new_name = target.get(dev_id)
        if new_name is None or iface == new_name:
            continue

        print(f"  {iface} -> {new_name}")
        subprocess.run(["sudo", "ip", "link", "set", iface, "down"], check=True)
        subprocess.run(
            ["sudo", "ip", "link", "set", iface, "name", new_name], check=True
        )

    print("  Done.")


def _write_cron_script() -> None:
    print(f"Writing CAN startup script to {_CRON_SCRIPT}...")
    _CAN_DIR.mkdir(parents=True, exist_ok=True)
    _CRON_SCRIPT.write_text(
        f"#!/bin/bash\n"
        f"# Bring up Almond Axol CAN interfaces\n"
        f"set -euo pipefail\n\n"
        f"for IFACE in {_CAN_L} {_CAN_R}; do\n"
        f'    ip link set "${{IFACE}}" down 2>/dev/null || true\n'
        f'    ip link set "${{IFACE}}" type can bitrate {_BITRATE}\n'
        f'    ip link set "${{IFACE}}" txqueuelen {_TXQUEUELEN}\n'
        f'    ip link set "${{IFACE}}" up\n'
        f"done\n"
    )
    _CRON_SCRIPT.chmod(0o755)
    print("  Done.")


def _register_cron() -> None:
    print("Registering @reboot cron entry in root crontab (requires sudo)...")
    cron_entry = f"@reboot {_CRON_SCRIPT}"
    existing = subprocess.run(
        ["sudo", "crontab", "-l"], capture_output=True, text=True
    ).stdout
    if str(_CRON_SCRIPT) in existing:
        print("  Entry already present — skipping.")
    else:
        new_crontab = existing.rstrip("\n") + "\n" + cron_entry + "\n"
        subprocess.run(
            ["sudo", "crontab", "-"], input=new_crontab, text=True, check=True
        )
        print(f"  Added: {cron_entry}")


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    subparsers.add_parser(
        "can.setup",
        help="Configure CAN interfaces for the Axol arm.",
    ).set_defaults(func=run)


def _bring_up_can() -> None:
    print("Bringing up CAN interfaces (requires sudo)...")
    subprocess.run(["sudo", "bash", str(_CRON_SCRIPT)], check=True)
    print("  Done.")


def run(_args: object = None) -> None:
    serial = _find_serial()
    _write_udev_rules(serial)
    _reload_udev()
    _rename_interfaces(serial)
    _write_cron_script()
    _register_cron()
    _bring_up_can()

    print()
    print("Setup complete.")
    print(f"  Left arm : {_CAN_L}")
    print(f"  Right arm: {_CAN_R}")
    print(f"  Startup  : {_CRON_SCRIPT} (runs at @reboot via root crontab)")
