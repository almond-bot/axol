"""
axol can.setup

Sets persistent CAN interface names for the Almond Axol arm CAN bus adapter
and registers a root crontab @reboot entry to bring up the interfaces.

The Almond Axol adapter (VID 0x1D50 / PID 0x606F) exposes two CAN channels
on a single USB device:
  channel 0 (dev_id 0x0) -> can_alm_axol_l  (left arm)
  channel 1 (dev_id 0x1) -> can_alm_axol_r  (right arm)
"""

import re
import subprocess
import sys
from pathlib import Path

from ...constants import CAN_LEFT, CAN_RIGHT
from ...utils.sudo import run_root
from . import driver

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


def _detect_serials() -> list[str]:
    """Return the serials of every attached Almond Axol CAN adapter (no prompts)."""
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

    return list(dict.fromkeys(serials))


def _serial_of_interface(iface: str) -> str | None:
    """The USB serial behind a named CAN interface, or None if it's absent."""
    iface_path = Path("/sys/class/net") / iface
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


def _configured_serial() -> str | None:
    """The Axol adapter's serial as pinned by a *previous* setup, if any.

    Preferred over live adapter detection: other candlelight devices (e.g. a
    UMI rig's CAN adapter) share the same generic VID/PID, so a host with
    several attached is ambiguous to a fresh scan — but not to a machine
    that has already named its Axol interfaces or written its udev rules.
    """
    for iface in (_CAN_L, _CAN_R):
        serial = _serial_of_interface(iface)
        if serial:
            return serial
    try:
        rules = _UDEV_RULES_FILE.read_text()
    except OSError:
        return None
    match = re.search(r'ATTRS\{serial\}=="([^"]+)"', rules)
    return match.group(1) if match else None


def _resolve_serial() -> str:
    """Pick the adapter serial without prompting (for headless ``ensure_setup``).

    A previously configured serial (named ``can_alm_axol_*`` interfaces, or
    the pinned serial in the udev rules) wins outright, so re-running setup on
    an already-configured host works no matter how many other candlelight
    adapters are attached. Only a genuinely fresh machine falls back to live
    detection — and raises when zero or several adapters are present, since
    that needs the interactive ``axol can.setup`` flow to disambiguate.
    """
    configured = _configured_serial()
    if configured:
        return configured
    unique = _detect_serials()
    if len(unique) == 1:
        return unique[0]
    if not unique:
        raise RuntimeError("Robot not detected")
    raise RuntimeError(
        "Multiple CAN adapters found — run `axol can.setup` once to pick the Axol's"
    )


def _find_serial() -> str:
    print(f"Scanning for Almond Axol CAN adapter ({_VID}:{_PID})...")

    unique = _detect_serials()

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
    run_root(["tee", str(_UDEV_RULES_FILE)], input_text=content, check=True)
    print("  Done.")


def _reload_udev() -> None:
    print("Reloading udev rules (requires sudo)...")
    run_root(["udevadm", "control", "--reload-rules"], check=True)
    run_root(["systemctl", "restart", "systemd-udevd"], check=True)
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
        run_root(["ip", "link", "set", iface, "down"], check=True)
        run_root(["ip", "link", "set", iface, "name", new_name], check=True)

    print("  Done.")


def _write_cron_script() -> None:
    print(f"Writing CAN startup script to {_CRON_SCRIPT}...")
    _CAN_DIR.mkdir(parents=True, exist_ok=True)
    _CRON_SCRIPT.write_text(
        f"#!/bin/bash\n"
        f"# Bring up Almond Axol CAN interfaces\n"
        f"#\n"
        f"# Both interfaces are channels of one dual-channel gs_usb adapter.\n"
        f"# Bring them down together, configure, then up together — flapping\n"
        f"# the channels one at a time (down/up L, then down/up R) toggles the\n"
        f"# adapter into a state where TX works but no RX frame is delivered.\n"
        f"set -euo pipefail\n\n"
        f"for IFACE in {_CAN_L} {_CAN_R}; do\n"
        f'    ip link set "${{IFACE}}" down 2>/dev/null || true\n'
        f"done\n"
        f"for IFACE in {_CAN_L} {_CAN_R}; do\n"
        f'    ip link set "${{IFACE}}" type can bitrate {_BITRATE}\n'
        f'    ip link set "${{IFACE}}" txqueuelen {_TXQUEUELEN}\n'
        f"done\n"
        f"for IFACE in {_CAN_L} {_CAN_R}; do\n"
        f'    ip link set "${{IFACE}}" up\n'
        f"done\n"
    )
    _CRON_SCRIPT.chmod(0o755)
    print("  Done.")


def _register_cron() -> None:
    print("Registering @reboot cron entry in root crontab (requires sudo)...")
    cron_entry = f"@reboot {_CRON_SCRIPT}"
    existing = run_root(["crontab", "-l"]).stdout or ""
    if str(_CRON_SCRIPT) in existing:
        print("  Entry already present — skipping.")
    else:
        new_crontab = existing.rstrip("\n") + "\n" + cron_entry + "\n"
        run_root(["crontab", "-"], input_text=new_crontab, check=True)
        print(f"  Added: {cron_entry}")


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``can.setup`` subcommand."""
    subparsers.add_parser(
        "can.setup",
        help="Configure CAN interfaces for the Axol arm.",
    ).set_defaults(func=run)


def _rx_alive() -> bool:
    """True when at least one motor answers on either arm.

    Verifies the adapter's receive path, not just the interface state: the
    dual-channel gs_usb adapter can come out of a down/up cycle in a state
    where TX still works but no received frame is ever delivered (kernel-side
    everything looks healthy — UP, ERROR-ACTIVE, correct bitrate).
    """
    import asyncio

    from ...constants import Joint
    from ...motor import CanBus, Motor

    async def probe(channel: str) -> bool:
        try:
            async with CanBus(channel) as bus:
                await asyncio.wait_for(
                    Motor(bus, Joint.SHOULDER_1).get_error_code(), timeout=0.7
                )
                return True
        except Exception:  # noqa: BLE001 - silence means "no RX", whatever the cause
            return False

    async def probe_all() -> bool:
        await asyncio.sleep(0.5)  # let the freshly-upped interfaces settle
        results = await asyncio.gather(probe(_CAN_L), probe(_CAN_R))
        return any(results)

    return asyncio.run(probe_all())


def _bring_up_can() -> None:
    """Run the bring-up script, then verify RX and re-flap once if it's dead.

    Every down/up cycle of the adapter's channels toggles it between a healthy
    state and the TX-only wedge described in :func:`_rx_alive`, so a bring-up
    that lands in the wedge is recovered by exactly one more cycle. A robot
    with its motors powered off is indistinguishable from the wedge, hence the
    bounded retries and the warning instead of an error.
    """
    print("Bringing up CAN interfaces (requires sudo)...")
    for attempt in range(3):
        run_root(["bash", str(_CRON_SCRIPT)], check=True)
        if _rx_alive():
            print("  Done — motors responding.")
            return
        if attempt < 2:
            print("  No motor responses (adapter RX may be wedged) — cycling again...")
    print(
        "  WARNING: no motor responded after bring-up. If the robot is powered "
        "on, re-run this command; otherwise this is expected."
    )


def is_configured() -> bool:
    """True when persistent CAN config has been written by a prior setup.

    Used by the control panel to decide whether connecting needs to run the
    full :func:`ensure_setup` (first time on a machine) or can just bring the
    already-named interfaces up.
    """
    return _UDEV_RULES_FILE.exists() and _CRON_SCRIPT.exists()


def ensure_setup(*, serial: str | None = None) -> None:
    """Run the full CAN configuration non-interactively (for the control panel).

    Mirrors :func:`run` but resolves the adapter serial without prompting.
    Each step is idempotent, so this is safe to call on a partially-configured
    machine.
    """
    driver.ensure_driver()
    serial = serial or _resolve_serial()
    _write_udev_rules(serial)
    _reload_udev()
    _rename_interfaces(serial)
    _write_cron_script()
    _register_cron()
    _bring_up_can()


def run(_args: object = None) -> None:
    """Configure persistent CAN interfaces and a @reboot bring-up entry."""
    driver.ensure_driver()
    serial = _find_serial()
    ensure_setup(serial=serial)

    print()
    print("Setup complete.")
    print(f"  Left arm : {_CAN_L}")
    print(f"  Right arm: {_CAN_R}")
    print(f"  Startup  : {_CRON_SCRIPT} (runs at @reboot via root crontab)")
