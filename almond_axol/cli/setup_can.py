"""
almond-axol setup-can

Sets persistent CAN interface names for the Almond Axol arm CAN bus adapter
and registers a root crontab @reboot entry to bring up the interfaces.

The Almond Axol adapter (VID 0x1D50 / PID 0x606F) exposes two CAN channels
on a single USB device:
  channel 0 (dev_id 0x0) -> can_alm_axol_l  (left arm)
  channel 1 (dev_id 0x1) -> can_alm_axol_r  (right arm)
"""

import os
import subprocess
import sys
from pathlib import Path

_VID      = "1d50"
_PID      = "606f"
_CAN_L    = "can_alm_axol_l"
_CAN_R    = "can_alm_axol_r"
_BITRATE  = 1000000
_TXQUEUELEN = 512

_UDEV_RULES_FILE = Path("/etc/udev/rules.d/90-can.rules")
_CAN_DIR         = Path.home() / ".almond" / "can"
_CRON_SCRIPT     = _CAN_DIR / "startup.sh"


def _die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _require_root() -> None:
    if os.geteuid() != 0:
        _die("This command must be run as root (sudo).")


def _find_serial() -> str:
    print(f"Scanning for Almond Axol CAN adapter ({_VID}:{_PID})...")

    serials: list[str] = []
    for iface_path in Path("/sys/class/net").glob("can*"):
        info = subprocess.run(
            ["udevadm", "info", "-a", "-p", str(iface_path)],
            capture_output=True, text=True,
        ).stdout

        vid = next((l.split('"')[1] for l in info.splitlines() if "ATTRS{idVendor}" in l), "")
        pid = next((l.split('"')[1] for l in info.splitlines() if "ATTRS{idProduct}" in l), "")

        if vid.lower() == _VID and pid.lower() == _PID:
            serial = next((l.split('"')[1] for l in info.splitlines() if "ATTRS{serial}" in l), "")
            if serial:
                serials.append(serial)

    unique = list(dict.fromkeys(serials))

    if not unique:
        print("\n  No adapter found. Enter the serial number manually (blank to abort):")
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
    print(f"Writing udev rules to {_UDEV_RULES_FILE}...")
    _UDEV_RULES_FILE.write_text(
        f"# Almond Axol dual-channel CAN adapter\n"
        f"# Adapter serial: {serial}\n"
        f"# Channel 0 -> left arm\n"
        f'SUBSYSTEM=="net", ACTION=="add", ATTRS{{idVendor}}=="{_VID}", ATTRS{{idProduct}}=="{_PID}", ATTRS{{serial}}=="{serial}", ATTR{{dev_id}}=="0x0", NAME="{_CAN_L}"\n'
        f"# Channel 1 -> right arm\n"
        f'SUBSYSTEM=="net", ACTION=="add", ATTRS{{idVendor}}=="{_VID}", ATTRS{{idProduct}}=="{_PID}", ATTRS{{serial}}=="{serial}", ATTR{{dev_id}}=="0x1", NAME="{_CAN_R}"\n'
    )
    print("  Done.")


def _reload_udev() -> None:
    print("Reloading udev rules...")
    subprocess.run(["udevadm", "control", "--reload-rules"], check=True)
    subprocess.run(["systemctl", "restart", "systemd-udevd"], check=True)
    subprocess.run(["udevadm", "trigger"], check=True)
    print("  Done. Unplug and replug the adapter to apply the new names.")


def _write_cron_script() -> None:
    print(f"Writing CAN startup script to {_CRON_SCRIPT}...")
    _CAN_DIR.mkdir(parents=True, exist_ok=True)
    _CRON_SCRIPT.write_text(
        f"#!/bin/bash\n"
        f"# Bring up Almond Axol CAN interfaces\n"
        f"set -euo pipefail\n\n"
        f"for IFACE in {_CAN_L} {_CAN_R}; do\n"
        f"    ip link set \"${{IFACE}}\" down 2>/dev/null || true\n"
        f"    ip link set \"${{IFACE}}\" type can bitrate {_BITRATE}\n"
        f"    ip link set \"${{IFACE}}\" txqueuelen {_TXQUEUELEN}\n"
        f"    ip link set \"${{IFACE}}\" up\n"
        f"done\n"
    )
    _CRON_SCRIPT.chmod(0o755)
    print("  Done.")


def _register_cron() -> None:
    print("Registering @reboot cron entry in root crontab...")
    cron_entry = f"@reboot {_CRON_SCRIPT}"
    existing = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
    if str(_CRON_SCRIPT) in existing:
        print("  Entry already present — skipping.")
    else:
        new_crontab = existing.rstrip("\n") + "\n" + cron_entry + "\n"
        subprocess.run(["crontab", "-"], input=new_crontab, text=True, check=True)
        print(f"  Added: {cron_entry}")


def run() -> None:
    _require_root()
    serial = _find_serial()
    _write_udev_rules(serial)
    _reload_udev()
    _write_cron_script()
    _register_cron()

    print()
    print("Setup complete.")
    print(f"  Left arm : {_CAN_L}")
    print(f"  Right arm: {_CAN_R}")
    print(f"  Startup  : {_CRON_SCRIPT} (runs at @reboot via root crontab)")
