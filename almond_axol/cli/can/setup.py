"""
axol can.setup

Sets persistent CAN interface names for the Almond Axol arm CAN bus adapter
and registers a root crontab @reboot entry to bring up the interfaces.

The Almond Axol adapter (VID 0x1D50 / PID 0x606F) exposes two CAN channels
on a single USB device:
  channel 0 (dev_id 0x0) -> can_alm_axol_l  (left arm)
  channel 1 (dev_id 0x1) -> can_alm_axol_r  (right arm)

``axol can.setup --umi`` configures the handheld UMI data-collection rig
instead: **two off-the-shelf single-channel CANable adapters** (candleLight
1d50:606f or CANable 2.0 16d0:117e), one per gripper, each keyed by its USB
serial:
  adapter A -> can_alm_umi_l   (left gripper)
  adapter B -> can_alm_umi_r   (right gripper)
Which adapter is left is chosen interactively at setup. The two profiles use
separate udev rule files and startup scripts, so a machine can have both the
robot and the UMI rig configured at once.
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


# USB IDs the UMI rig accepts: any gs_usb-compatible single-channel CANable —
# candleLight-flashed (1d50:606f), CANable 2.0 stock firmware (16d0:117e), or
# original candleLight (1209:2323). The vendored driver claims all of these.
_UMI_USB_IDS = {("1d50", "606f"), ("16d0", "117e"), ("1209", "2323")}


def _usb_attr(info: str, attr: str) -> str:
    """First value of ``ATTRS{attr}`` (or ``ATTR{attr}``) in udevadm output."""
    for line in info.splitlines():
        if f"{{{attr}}}" in line and '=="' in line:
            return line.split('=="')[1].split('"')[0]
    return ""


def _detect_umi_adapters() -> list[tuple[str, str, str]]:
    """``(serial, vid, pid)`` of every attached UMI-compatible CAN adapter.

    Scans the bound CAN network interfaces, so an adapter the running
    ``gs_usb`` driver doesn't claim (e.g. a CANable 2.0 against an old driver
    build) is invisible here — ``driver.ensure_driver()`` runs first to
    prevent that. The Jetson's built-in mttcan controller has no USB vendor
    attributes and is skipped naturally.
    """
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for iface_path in sorted(Path("/sys/class/net").glob("can*")):
        info = subprocess.run(
            ["udevadm", "info", "-a", "-p", str(iface_path)],
            capture_output=True,
            text=True,
        ).stdout
        vid = _usb_attr(info, "idVendor").lower()
        pid = _usb_attr(info, "idProduct").lower()
        serial = _usb_attr(info, "serial")
        if (vid, pid) in _UMI_USB_IDS and serial and serial not in seen:
            seen.add(serial)
            out.append((serial, vid, pid))
    return out


def _serials_in_rules(rules_file: Path) -> set[str]:
    """Every adapter serial a previously-written rules file is keyed on."""
    try:
        content = rules_file.read_text()
    except OSError:
        return set()
    serials: set[str] = set()
    for line in content.splitlines():
        if 'ATTRS{serial}=="' in line:
            serials.add(line.split('ATTRS{serial}=="')[1].split('"')[0])
    return serials


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

    # Hide serials the UMI profile's rules already claim (a candleLight
    # CANable shares the hub's USB ID) so the obvious single-adapter case
    # stays promptless.
    claimed = _serials_in_rules(_UMI_PROFILE.rules_file)
    if claimed and len(unique) > 1:
        unique = [s for s in unique if s not in claimed]

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


def _find_umi_assignment() -> dict[str, tuple[str, str, str]]:
    """Interactively map the two UMI channel names to attached adapters.

    Returns ``{interface_name: (serial, vid, pid)}`` for the left and right
    grippers. Requires two UMI-compatible adapters to be attached (the serial
    the robot-arm rules claim is excluded).
    """
    adapters = _detect_umi_adapters()
    claimed = _serials_in_rules(_AXOL_PROFILE.rules_file)
    adapters = [a for a in adapters if a[0] not in claimed]

    if len(adapters) < 2:
        _die(
            f"Found {len(adapters)} UMI-compatible CAN adapter(s); need 2 "
            "(one per gripper). Check both CANables are plugged in and the "
            "gs_usb driver claims them (`axol can.driver`, then replug)."
        )

    print("  Found adapters:")
    for i, (serial, vid, pid) in enumerate(adapters):
        print(f"    [{i}] {serial}  ({vid}:{pid})")
    idx_l = int(input("  Index of the LEFT gripper's adapter [0]: ").strip() or "0")
    left = adapters[idx_l]
    remaining = [a for i, a in enumerate(adapters) if i != idx_l]
    if len(remaining) == 1:
        right = remaining[0]
        print(f"  Right gripper: {right[0]} ({right[1]}:{right[2]})")
    else:
        for i, (serial, vid, pid) in enumerate(remaining):
            print(f"    [{i}] {serial}  ({vid}:{pid})")
        idx_r = int(
            input("  Index of the RIGHT gripper's adapter [0]: ").strip() or "0"
        )
        right = remaining[idx_r]
    return {_UMI_PROFILE.left: left, _UMI_PROFILE.right: right}


def _write_umi_udev_rules(assign: dict[str, tuple[str, str, str]]) -> None:
    """Write per-serial rules naming each single-channel adapter's interface."""
    print(f"Writing udev rules to {_UMI_PROFILE.rules_file} (requires sudo)...")
    lines = [f"# {_UMI_PROFILE.label}: one single-channel CANable per gripper"]
    for name, (serial, vid, pid) in assign.items():
        side = "left" if name == _UMI_PROFILE.left else "right"
        lines.append(f"# {side} gripper — adapter serial {serial}")
        lines.append(
            f'SUBSYSTEM=="net", ACTION=="add", ATTRS{{idVendor}}=="{vid}", '
            f'ATTRS{{idProduct}}=="{pid}", ATTRS{{serial}}=="{serial}", '
            f'NAME="{name}"'
        )
    run_root(
        ["tee", str(_UMI_PROFILE.rules_file)],
        input_text="\n".join(lines) + "\n",
        check=True,
    )
    print("  Done.")


def _rename_umi_interfaces(assign: dict[str, tuple[str, str, str]]) -> None:
    """Rename the assigned adapters' interfaces to their target names now."""
    print("Renaming CAN interfaces (requires sudo)...")
    by_serial = {serial: name for name, (serial, _vid, _pid) in assign.items()}
    for iface_path in Path("/sys/class/net").glob("can*"):
        iface = iface_path.name
        info = subprocess.run(
            ["udevadm", "info", "-a", "-p", str(iface_path)],
            capture_output=True,
            text=True,
        ).stdout
        new_name = by_serial.get(_usb_attr(info, "serial"))
        if new_name is None or iface == new_name:
            continue
        print(f"  {iface} -> {new_name}")
        run_root(["ip", "link", "set", iface, "down"], check=True)
        run_root(["ip", "link", "set", iface, "name", new_name], check=True)
    print("  Done.")


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
    installed = driver.ensure_driver()
    if installed:
        # The freshly-loaded driver may claim adapters the old one ignored
        # (CANable 2.0); give their interfaces a moment to appear.
        import time

        time.sleep(2.0)

    if profile is _UMI_PROFILE:
        assign = _find_umi_assignment()
        _write_umi_udev_rules(assign)
        _reload_udev()
        _rename_umi_interfaces(assign)
        _write_cron_script(profile)
        _register_cron(profile)
        _bring_up_can(profile)
    else:
        serial = _find_serial(profile)
        _configure(serial, profile)

    print()
    print("Setup complete.")
    print(f"  Left  : {profile.left}")
    print(f"  Right : {profile.right}")
    print(f"  Startup  : {profile.cron_script} (runs at @reboot via root crontab)")
