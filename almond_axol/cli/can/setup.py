"""
axol can.setup

Sets persistent CAN interface names for the Almond Axol arm CAN bus adapter
and registers a root crontab @reboot entry to bring up the interfaces.

The Almond Axol adapter (VID 0x1D50 / PID 0x606F) exposes two CAN channels
on a single USB device:
  channel 0 (dev_id 0x0) -> can_alm_axol_l  (left arm)
  channel 1 (dev_id 0x1) -> can_alm_axol_r  (right arm)

``axol can.setup --umi`` configures a second adapter of the same model for the
handheld UMI data-collection rig instead:
  channel 0 (dev_id 0x0) -> can_alm_umi_l   (left gripper)
  channel 1 (dev_id 0x1) -> can_alm_umi_r   (right gripper)
The two profiles use separate udev rule files and startup scripts, so a machine
can have both the robot and the UMI rig configured at once.
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ...constants import CAN_LEFT, CAN_RIGHT, CAN_UMI_LEFT, CAN_UMI_RIGHT, Joint
from ...utils.sudo import run_root
from . import driver

_VID = "1d50"
_PID = "606f"
_BITRATE = 1_000_000
_TXQUEUELEN = 512

_CAN_DIR = Path.home() / ".almond" / "can"


@dataclass(frozen=True)
class _Profile:
    """One adapter's persistent-naming setup (rule file, names, bring-up script)."""

    label: str
    left: str
    right: str
    rules_file: Path
    cron_script: Path
    # Joint whose motor the post-bring-up RX probe queries (see _rx_alive).
    probe_joint: Joint


_AXOL_PROFILE = _Profile(
    label="Almond Axol arm",
    left=CAN_LEFT,
    right=CAN_RIGHT,
    rules_file=Path("/etc/udev/rules.d/90-can.rules"),
    cron_script=_CAN_DIR / "startup.sh",
    probe_joint=Joint.SHOULDER_1,
)

_UMI_PROFILE = _Profile(
    label="Almond UMI rig",
    left=CAN_UMI_LEFT,
    right=CAN_UMI_RIGHT,
    rules_file=Path("/etc/udev/rules.d/91-can-umi.rules"),
    cron_script=_CAN_DIR / "startup_umi.sh",
    probe_joint=Joint.GRIPPER,
)


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


def _serial_in_rules(rules_file: Path) -> str | None:
    """The adapter serial a previously-written rules file is keyed on, if any."""
    try:
        content = rules_file.read_text()
    except OSError:
        return None
    for line in content.splitlines():
        if 'ATTRS{serial}=="' in line:
            return line.split('ATTRS{serial}=="')[1].split('"')[0]
    return None


def _resolve_serial() -> str:
    """Pick the adapter serial without prompting (for headless ``ensure_setup``).

    Raises ``RuntimeError`` when zero or several adapters are present, since
    that needs the interactive ``axol can.setup`` flow to disambiguate.
    """
    unique = _detect_serials()
    if len(unique) == 1:
        return unique[0]
    if not unique:
        raise RuntimeError("Robot not detected")
    raise RuntimeError("Multiple CAN adapters")


def _find_serial(profile: _Profile) -> str:
    print(f"Scanning for {profile.label} CAN adapter ({_VID}:{_PID})...")

    unique = _detect_serials()

    # When setting up a second profile, hide the serial the *other* profile's
    # rules already claim so the obvious single-adapter case stays promptless.
    other = _AXOL_PROFILE if profile is _UMI_PROFILE else _UMI_PROFILE
    claimed = _serial_in_rules(other.rules_file)
    if claimed and len(unique) > 1:
        unique = [s for s in unique if s != claimed]

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


def _write_udev_rules(serial: str, profile: _Profile) -> None:
    print(f"Writing udev rules to {profile.rules_file} (requires sudo)...")
    content = (
        f"# {profile.label} dual-channel CAN adapter\n"
        f"# Adapter serial: {serial}\n"
        f"# Channel 0 -> left\n"
        f'SUBSYSTEM=="net", ACTION=="add", ATTRS{{idVendor}}=="{_VID}", ATTRS{{idProduct}}=="{_PID}", ATTRS{{serial}}=="{serial}", ATTR{{dev_id}}=="0x0", NAME="{profile.left}"\n'
        f"# Channel 1 -> right\n"
        f'SUBSYSTEM=="net", ACTION=="add", ATTRS{{idVendor}}=="{_VID}", ATTRS{{idProduct}}=="{_PID}", ATTRS{{serial}}=="{serial}", ATTR{{dev_id}}=="0x1", NAME="{profile.right}"\n'
    )
    run_root(["tee", str(profile.rules_file)], input_text=content, check=True)
    print("  Done.")


def _reload_udev() -> None:
    print("Reloading udev rules (requires sudo)...")
    run_root(["udevadm", "control", "--reload-rules"], check=True)
    run_root(["systemctl", "restart", "systemd-udevd"], check=True)
    print("  Done.")


def _rename_interfaces(serial: str, profile: _Profile) -> None:
    """Rename existing canX interfaces to their target names without replug."""
    print("Renaming CAN interfaces (requires sudo)...")
    target = {0: profile.left, 1: profile.right}

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


def _write_cron_script(profile: _Profile) -> None:
    print(f"Writing CAN startup script to {profile.cron_script}...")
    _CAN_DIR.mkdir(parents=True, exist_ok=True)
    profile.cron_script.write_text(
        f"#!/bin/bash\n"
        f"# Bring up {profile.label} CAN interfaces\n"
        f"#\n"
        f"# Both interfaces are channels of one dual-channel gs_usb adapter.\n"
        f"# Bring them down together, configure, then up together — flapping\n"
        f"# the channels one at a time (down/up L, then down/up R) toggles the\n"
        f"# adapter into a state where TX works but no RX frame is delivered.\n"
        f"set -euo pipefail\n\n"
        f"for IFACE in {profile.left} {profile.right}; do\n"
        f'    ip link set "${{IFACE}}" down 2>/dev/null || true\n'
        f"done\n"
        f"for IFACE in {profile.left} {profile.right}; do\n"
        f'    ip link set "${{IFACE}}" type can bitrate {_BITRATE}\n'
        f'    ip link set "${{IFACE}}" txqueuelen {_TXQUEUELEN}\n'
        f"done\n"
        f"for IFACE in {profile.left} {profile.right}; do\n"
        f'    ip link set "${{IFACE}}" up\n'
        f"done\n"
    )
    profile.cron_script.chmod(0o755)
    print("  Done.")


def _register_cron(profile: _Profile) -> None:
    print("Registering @reboot cron entry in root crontab (requires sudo)...")
    cron_entry = f"@reboot {profile.cron_script}"
    existing = run_root(["crontab", "-l"]).stdout or ""
    if str(profile.cron_script) in existing:
        print("  Entry already present — skipping.")
    else:
        new_crontab = existing.rstrip("\n") + "\n" + cron_entry + "\n"
        run_root(["crontab", "-"], input_text=new_crontab, check=True)
        print(f"  Added: {cron_entry}")


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``can.setup`` subcommand."""
    parser = subparsers.add_parser(
        "can.setup",
        help="Configure CAN interfaces for the Axol arm (or the UMI rig with --umi).",
    )
    parser.add_argument(
        "--umi",
        action="store_true",
        help=f"Configure the handheld UMI rig adapter ({CAN_UMI_LEFT} / {CAN_UMI_RIGHT}).",
    )
    parser.set_defaults(func=run)


def _rx_alive(profile: _Profile) -> bool:
    """True when at least one motor answers on either channel.

    Verifies the adapter's receive path, not just the interface state: the
    dual-channel gs_usb adapter can come out of a down/up cycle in a state
    where TX still works but no received frame is ever delivered (kernel-side
    everything looks healthy — UP, ERROR-ACTIVE, correct bitrate). Probes the
    profile's ``probe_joint`` — the shoulder on the robot arm, the gripper on
    the UMI rig (its buses carry nothing else).
    """
    import asyncio

    from ...motor import CanBus, Motor

    async def probe(channel: str) -> bool:
        try:
            async with CanBus(channel) as bus:
                await asyncio.wait_for(
                    Motor(bus, profile.probe_joint).get_error_code(), timeout=0.7
                )
                return True
        except Exception:  # noqa: BLE001 - silence means "no RX", whatever the cause
            return False

    async def probe_all() -> bool:
        await asyncio.sleep(0.5)  # let the freshly-upped interfaces settle
        results = await asyncio.gather(probe(profile.left), probe(profile.right))
        return any(results)

    return asyncio.run(probe_all())


def _bring_up_can(profile: _Profile) -> None:
    """Run the bring-up script, then verify RX and re-flap once if it's dead.

    Every down/up cycle of the adapter's channels toggles it between a healthy
    state and the TX-only wedge described in :func:`_rx_alive`, so a bring-up
    that lands in the wedge is recovered by exactly one more cycle. A device
    with its motors powered off is indistinguishable from the wedge, hence the
    bounded retries and the warning instead of an error.
    """
    print("Bringing up CAN interfaces (requires sudo)...")
    for attempt in range(3):
        run_root(["bash", str(profile.cron_script)], check=True)
        if _rx_alive(profile):
            print("  Done — motors responding.")
            return
        if attempt < 2:
            print("  No motor responses (adapter RX may be wedged) — cycling again...")
    print(
        "  WARNING: no motor responded after bring-up. If the device is powered "
        "on, re-run this command; otherwise this is expected."
    )


def is_configured() -> bool:
    """True when persistent CAN config has been written by a prior setup.

    Used by the control panel to decide whether connecting needs to run the
    full :func:`ensure_setup` (first time on a machine) or can just bring the
    already-named interfaces up. Refers to the robot-arm profile; the UMI rig
    is configured explicitly via ``axol can.setup --umi``.
    """
    return _AXOL_PROFILE.rules_file.exists() and _AXOL_PROFILE.cron_script.exists()


def ensure_setup(*, serial: str | None = None) -> None:
    """Run the full CAN configuration non-interactively (for the control panel).

    Mirrors :func:`run` but resolves the adapter serial without prompting.
    Each step is idempotent, so this is safe to call on a partially-configured
    machine. Configures the robot-arm profile only.
    """
    driver.ensure_driver()
    serial = serial or _resolve_serial()
    _configure(serial, _AXOL_PROFILE)


def _configure(serial: str, profile: _Profile) -> None:
    _write_udev_rules(serial, profile)
    _reload_udev()
    _rename_interfaces(serial, profile)
    _write_cron_script(profile)
    _register_cron(profile)
    _bring_up_can(profile)


def run(args: object = None) -> None:
    """Configure persistent CAN interfaces and a @reboot bring-up entry."""
    profile = _UMI_PROFILE if getattr(args, "umi", False) else _AXOL_PROFILE
    driver.ensure_driver()
    serial = _find_serial(profile)
    _configure(serial, profile)

    print()
    print("Setup complete.")
    print(f"  Left  : {profile.left}")
    print(f"  Right : {profile.right}")
    print(f"  Startup  : {profile.cron_script} (runs at @reboot via root crontab)")
