"""
axol can.setup

Sets persistent CAN interface names for the Almond Axol CAN bus adapters,
registers a root crontab @reboot entry to bring up the interfaces, and
installs a udev-triggered systemd unit that re-runs the bring-up whenever the
adapter (re-)enumerates — so a mid-session USB drop of the hub (EMI from the
arms can kick it off the bus, most visibly on Raspberry Pi 5 hosts) heals
itself without operator action.

The Almond Axol arm hub adapter (VID 0x1D50 / PID 0x606F) exposes two CAN
channels on a single USB device:
  channel 0 (dev_id 0x0) -> can_alm_axol_l  (left arm)
  channel 1 (dev_id 0x1) -> can_alm_axol_r  (right arm)

Robots on the powered cart additionally carry a single-channel candlelight
adapter (same generic VID/PID) for the wheel bus, named can_alm_axol_b. The
channel count tells the two apart: the hub always enumerates both channels
under one serial, the cart adapter exactly one.

On Raspberry Pi 5 hosts the setup additionally raises the RP1 USB
controllers' EMI tolerance (see :func:`_setup_rp1_usb_quirk`), which targets
the disconnects at their source; the hotplug bring-up covers whatever still
gets through.
"""

import re
import subprocess
import sys
from pathlib import Path

from ...constants import CAN_BASE, CAN_LEFT, CAN_RIGHT
from ...utils.sudo import run_root
from . import driver

_VID = "1d50"
_PID = "606f"
_CAN_L = CAN_LEFT
_CAN_R = CAN_RIGHT
_CAN_B = CAN_BASE
_BITRATE = 1_000_000
_TXQUEUELEN = 512

_UDEV_RULES_FILE = Path("/etc/udev/rules.d/90-can.rules")
_CAN_DIR = Path.home() / ".almond" / "can"
_CRON_SCRIPT = _CAN_DIR / "startup.sh"

# Hotplug bring-up: pulled in via the udev rules (SYSTEMD_WANTS) whenever an
# adapter (re-)enumerates, so interfaces recreated by a mid-session USB drop
# come back configured and up without operator action.
_HOTPLUG_UNIT = "axol-can-up.service"
_HOTPLUG_UNIT_FILE = Path("/etc/systemd/system") / _HOTPLUG_UNIT

# Raspberry Pi 5 RP1 USB EMI-tolerance quirk (see _setup_rp1_usb_quirk).
_RP1_QUIRK_SCRIPT = _CAN_DIR / "rp1-usb-quirk.sh"
_RP1_QUIRK_UNIT = "axol-rp1-usb-quirk.service"
_RP1_QUIRK_UNIT_FILE = Path("/etc/systemd/system") / _RP1_QUIRK_UNIT
# GUCTL1 register of each RP1 xHCI controller (DWC3 global registers live at
# base + 0xc100; the Pi 5 has one controller per USB-A port pair).
_RP1_GUCTL1_REGS = ("0x1f0020c11c", "0x1f0030c11c")


def _die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _udev_attr(info: str, attr: str) -> str:
    """First value of ``attr`` in ``udevadm info -a`` output ('' if absent)."""
    return next(
        (line.split('"')[1] for line in info.splitlines() if attr in line),
        "",
    )


def _scan_adapters() -> dict[str, dict]:
    """Every attached gs_usb CAN adapter: serial -> {vid, pid, dev_ids}.

    The dual-channel Axol arm hub shows up as one serial with dev_ids {0, 1};
    a single-channel adapter (the cart's wheel-bus CANable, a UMI rig, ...)
    as one serial with {0}. Matched on the gs_usb driver rather than a VID/PID
    so CANable firmware variants that don't use the candlelight 1d50:606f IDs
    still count; the Jetson's built-in mttcan controller has no USB serial and
    is excluded either way.
    """
    adapters: dict[str, dict] = {}
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


def _detect_serials() -> list[str]:
    """Serials of every attached *dual-channel* Axol adapter — hub candidates.

    Single-channel devices (the cart's wheel-bus adapter, UMI rigs) share the
    generic VID/PID but can never be the hub, so they are excluded rather
    than left to make the scan ambiguous.
    """
    return [
        serial
        for serial, a in _scan_adapters().items()
        if len(a["dev_ids"]) >= 2 and (a["vid"], a["pid"]) == (_VID, _PID)
    ]


def _detect_base_serials(exclude: str) -> list[str]:
    """Serials of attached single-channel adapters — cart wheel-bus candidates."""
    return [
        serial
        for serial, a in _scan_adapters().items()
        if len(a["dev_ids"]) == 1 and serial != exclude
    ]


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


def _rules_serial_for(name: str) -> str | None:
    """The serial pinned to interface ``name`` in the written udev rules."""
    try:
        rules = _UDEV_RULES_FILE.read_text()
    except OSError:
        return None
    match = re.search(
        r'ATTRS\{serial\}=="([^"]+)"[^\n]*NAME="' + re.escape(name) + '"', rules
    )
    return match.group(1) if match else None


def _configured_serial() -> str | None:
    """The Axol hub adapter's serial as pinned by a *previous* setup, if any.

    Preferred over live adapter detection: other candlelight devices (e.g. a
    UMI rig's CAN adapter) share the same generic VID/PID, so a host with
    several attached is ambiguous to a fresh scan — but not to a machine
    that has already named its Axol interfaces or written its udev rules.
    """
    for iface in (_CAN_L, _CAN_R):
        serial = _serial_of_interface(iface)
        if serial:
            return serial
    return _rules_serial_for(_CAN_L) or _rules_serial_for(_CAN_R)


def _configured_base_serial() -> str | None:
    """The cart wheel-bus adapter's serial as pinned by a previous setup.

    Never auto-detected outside the interactive ``axol can.setup`` flow: a
    single-channel candlelight adapter is indistinguishable from unrelated
    hardware (UMI rigs), so only a serial the operator has already confirmed
    — a live ``can_alm_axol_b`` interface or a written udev rule — counts.
    """
    return _serial_of_interface(_CAN_B) or _rules_serial_for(_CAN_B)


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
        # An unplugged hub on an already-configured host (e.g. re-running
        # setup on a cart-only session) keeps its pinned serial — the udev
        # rule and startup script stay valid for whenever it's reattached.
        configured = _configured_serial()
        if configured:
            print(f"  No hub attached — keeping configured serial {configured}.")
            return configured
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


def _write_udev_rules(serial: str, base_serial: str | None = None) -> None:
    print(f"Writing udev rules to {_UDEV_RULES_FILE} (requires sudo)...")
    content = (
        f"# Almond Axol dual-channel CAN adapter\n"
        f"# Adapter serial: {serial}\n"
        f"# Channel 0 -> left arm\n"
        f'SUBSYSTEM=="net", ACTION=="add", ATTRS{{idVendor}}=="{_VID}", ATTRS{{idProduct}}=="{_PID}", ATTRS{{serial}}=="{serial}", ATTR{{dev_id}}=="0x0", NAME="{_CAN_L}"\n'
        f"# Channel 1 -> right arm\n"
        f'SUBSYSTEM=="net", ACTION=="add", ATTRS{{idVendor}}=="{_VID}", ATTRS{{idProduct}}=="{_PID}", ATTRS{{serial}}=="{serial}", ATTR{{dev_id}}=="0x1", NAME="{_CAN_R}"\n'
        f"# Every (re-)enumeration — boot or a mid-session USB drop — pulls in\n"
        f"# the bring-up service so the channels come back configured and up.\n"
        f"# Tagged on the USB device rather than the net interfaces: the NAME=\n"
        f"# rules above put every real hotplug add mid-rename, and systemd\n"
        f'# skips SYSTEMD_WANTS on renaming devices ("device is renaming").\n'
        f'SUBSYSTEM=="usb", ENV{{DEVTYPE}}=="usb_device", ACTION=="add", ATTR{{idVendor}}=="{_VID}", ATTR{{idProduct}}=="{_PID}", ATTR{{serial}}=="{serial}", TAG+="systemd", ENV{{SYSTEMD_WANTS}}+="{_HOTPLUG_UNIT}"\n'
    )
    if base_serial:
        # Matched by serial alone: CANable firmware variants ship various
        # VID/PIDs, and the serial already identifies the exact adapter.
        content += (
            f"# Powered-cart wheel bus (single-channel adapter)\n"
            f"# Adapter serial: {base_serial}\n"
            f'SUBSYSTEM=="net", ACTION=="add", ATTRS{{serial}}=="{base_serial}", NAME="{_CAN_B}"\n'
            f'SUBSYSTEM=="usb", ENV{{DEVTYPE}}=="usb_device", ACTION=="add", ATTR{{serial}}=="{base_serial}", TAG+="systemd", ENV{{SYSTEMD_WANTS}}+="{_HOTPLUG_UNIT}"\n'
        )
    run_root(["tee", str(_UDEV_RULES_FILE)], input_text=content, check=True)
    print("  Done.")


def _reload_udev() -> None:
    print("Reloading udev rules (requires sudo)...")
    run_root(["udevadm", "control", "--reload-rules"], check=True)
    run_root(["systemctl", "restart", "systemd-udevd"], check=True)
    print("  Done.")


def _rename_interfaces(serial: str, base_serial: str | None = None) -> None:
    """Rename existing canX interfaces to their target names without replug."""
    print("Renaming CAN interfaces (requires sudo)...")
    # (adapter serial, channel dev_id) -> persistent name. The cart adapter is
    # single-channel, so its only interface is dev_id 0.
    target = {(serial, 0): _CAN_L, (serial, 1): _CAN_R}
    if base_serial:
        target[(base_serial, 0)] = _CAN_B

    for iface_path in Path("/sys/class/net").glob("can*"):
        iface = iface_path.name
        info = subprocess.run(
            ["udevadm", "info", "-a", "-p", str(iface_path)],
            capture_output=True,
            text=True,
        ).stdout

        iface_serial = _udev_attr(info, "ATTRS{serial}")
        try:
            dev_id = int(_udev_attr(info, "ATTR{dev_id}"), 16)
        except ValueError:
            continue

        new_name = target.get((iface_serial, dev_id))
        if new_name is None or iface == new_name:
            continue

        print(f"  {iface} -> {new_name}")
        run_root(["ip", "link", "set", iface, "down"], check=True)
        run_root(["ip", "link", "set", iface, "name", new_name], check=True)

    print("  Done.")


def _write_cron_script(*, with_base: bool = False) -> None:
    print(f"Writing CAN startup script to {_CRON_SCRIPT}...")
    _CAN_DIR.mkdir(parents=True, exist_ok=True)
    script = (
        f"#!/bin/bash\n"
        f"# Bring up Almond Axol CAN interfaces\n"
        f"#\n"
        f"# Runs at boot (@reboot root crontab) and on every (re-)enumeration\n"
        f"# of the adapter (udev -> {_HOTPLUG_UNIT}), so a mid-session USB\n"
        f"# drop of the hub comes back configured without operator action.\n"
        f"#\n"
        f"# The arm interfaces are channels of one dual-channel gs_usb adapter.\n"
        f"# Bring them down together, configure, then up together — flapping\n"
        f"# the channels one at a time (down/up L, then down/up R) toggles the\n"
        f"# adapter into a state where TX works but no RX frame is delivered.\n"
        f"# Skipped entirely when the hub is unplugged (a cart-only session\n"
        f"# must still bring up the wheel bus below).\n"
        f"set -euo pipefail\n\n"
        f"# Boot and hotplug triggers can race (the hub's two channels fire one\n"
        f"# udev add event each) — serialize whole runs.\n"
        f'exec 9>"/run/lock/axol-can-up.lock"\n'
        f"flock 9\n\n"
        f"# The two channels enumerate a beat apart, so the trigger for the\n"
        f"# first can run before the second exists. Give the pair a moment.\n"
        f"for _ in $(seq 1 30); do\n"
        f"    if ip link show {_CAN_L} >/dev/null 2>&1 "
        f"&& ip link show {_CAN_R} >/dev/null 2>&1; then\n"
        f"        break\n"
        f"    fi\n"
        f"    sleep 0.1\n"
        f"done\n\n"
        f"if ip link show {_CAN_L} >/dev/null 2>&1 "
        f"&& ip link show {_CAN_R} >/dev/null 2>&1; then\n"
        f"    for IFACE in {_CAN_L} {_CAN_R}; do\n"
        f'        ip link set "${{IFACE}}" down 2>/dev/null || true\n'
        f"    done\n"
        f"    for IFACE in {_CAN_L} {_CAN_R}; do\n"
        f'        ip link set "${{IFACE}}" type can bitrate {_BITRATE}\n'
        f'        ip link set "${{IFACE}}" txqueuelen {_TXQUEUELEN}\n'
        f"    done\n"
        f"    for IFACE in {_CAN_L} {_CAN_R}; do\n"
        f'        ip link set "${{IFACE}}" up\n'
        f"    done\n"
        f"else\n"
        f'    echo "arm hub interfaces not present — skipping arm bring-up"\n'
        f"fi\n"
    )
    if with_base:
        script += (
            f"\n# Powered-cart wheel bus: its own single-channel adapter, so no\n"
            f"# flap-together dance — and skipped when absent, so an unplugged\n"
            f"# cart never blocks the arm bring-up.\n"
            f"if ip link show {_CAN_B} >/dev/null 2>&1; then\n"
            f'    ip link set "{_CAN_B}" down 2>/dev/null || true\n'
            f'    ip link set "{_CAN_B}" type can bitrate {_BITRATE}\n'
            f'    ip link set "{_CAN_B}" txqueuelen {_TXQUEUELEN}\n'
            f'    ip link set "{_CAN_B}" up\n'
            f"fi\n"
        )
    _CRON_SCRIPT.write_text(script)
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


def _write_hotplug_unit() -> None:
    """Install the systemd unit the udev rules pull in on adapter hotplug.

    udev tags the adapter's net devices with ``SYSTEMD_WANTS=axol-can-up.service``
    (see :func:`_write_udev_rules`), so every (re-)enumeration — boot or a
    mid-session USB drop — runs the startup script and the interfaces come
    back configured and up within a second, no operator action needed.
    """
    print(f"Writing hotplug bring-up unit to {_HOTPLUG_UNIT_FILE} (requires sudo)...")
    content = (
        f"# Installed by `axol can.setup`. Pulled in by {_UDEV_RULES_FILE}\n"
        f"# whenever an Axol CAN adapter (re-)enumerates: a mid-session USB\n"
        f"# drop recreates the interfaces down and unconfigured, and this\n"
        f"# service brings them back up without operator action.\n"
        f"[Unit]\n"
        f"Description=Bring up Almond Axol CAN interfaces on adapter hotplug\n"
        f"\n"
        f"[Service]\n"
        f"Type=oneshot\n"
        f"ExecStart=/bin/bash {_CRON_SCRIPT}\n"
    )
    run_root(["tee", str(_HOTPLUG_UNIT_FILE)], input_text=content, check=True)
    run_root(["systemctl", "daemon-reload"], check=True)
    print("  Done.")


def _is_raspberry_pi_5() -> bool:
    """True on Raspberry Pi 5 family boards (the RP1-southbridge USB ports)."""
    try:
        model = Path("/proc/device-tree/model").read_text()
    except OSError:
        return False
    return "Raspberry Pi 5" in model


def _setup_rp1_usb_quirk() -> None:
    """Raise the RP1 xHCI's EMI tolerance on Raspberry Pi 5 hosts.

    The Pi 5's RP1 USB controllers ship with a hair-trigger loss-of-activity /
    babble detector. Electrical noise from the arms' motor drivers coupling
    into the hub's full-speed USB link trips it, and the kernel disables the
    port (``usb usb3-port1: disabled by hub (EMI?), re-enabling...``) or the
    adapter drops off the bus entirely — disconnects that the same robot never
    shows on ZED Box / Jetson hosts. Setting bit 0 (the LOA filter) of each
    controller's GUCTL1 register makes the detector ride out those glitches;
    it's the workaround recommended by Raspberry Pi engineers
    (forums.raspberrypi.com/viewtopic.php?t=363780).

    The register resets on reboot, so this installs a boot-time oneshot unit
    and also applies it immediately. No-op on non-Pi-5 hosts.
    """
    if not _is_raspberry_pi_5():
        return
    if not Path("/usr/bin/busybox").exists():
        print(
            "WARNING: busybox not found — skipping the RP1 USB EMI quirk "
            "(install busybox and re-run `axol can.setup`)."
        )
        return
    print("Applying RP1 USB EMI-tolerance quirk (Pi 5 only, requires sudo)...")
    _CAN_DIR.mkdir(parents=True, exist_ok=True)
    regs = " ".join(_RP1_GUCTL1_REGS)
    script = (
        f"#!/bin/bash\n"
        f"# Raspberry Pi 5 (RP1) USB EMI-tolerance quirk for the Axol CAN hub.\n"
        f"#\n"
        f"# Sets bit 0 (LOA filter enable) of GUCTL1 in each RP1 xHCI\n"
        f"# controller so EMI glitches on the hub's full-speed link no longer\n"
        f'# disable the port ("disabled by hub (EMI?)" disconnects in dmesg).\n'
        f"# Recommended by Raspberry Pi engineers:\n"
        f"#   https://forums.raspberrypi.com/viewtopic.php?t=363780\n"
        f"#\n"
        f"# The register resets on reboot; {_RP1_QUIRK_UNIT} re-applies it.\n"
        f"set -euo pipefail\n\n"
        f"for reg in {regs}; do\n"
        f'    val=$(busybox devmem "$reg")\n'
        f'    busybox devmem "$reg" 32 $(( val | 1 ))\n'
        f"done\n"
    )
    _RP1_QUIRK_SCRIPT.write_text(script)
    _RP1_QUIRK_SCRIPT.chmod(0o755)
    unit = (
        f"# Installed by `axol can.setup` on Raspberry Pi 5 hosts.\n"
        f"[Unit]\n"
        f"Description=RP1 USB EMI-tolerance quirk for the Axol CAN hub\n"
        f"\n"
        f"[Service]\n"
        f"Type=oneshot\n"
        f"ExecStart=/bin/bash {_RP1_QUIRK_SCRIPT}\n"
        f"\n"
        f"[Install]\n"
        f"WantedBy=multi-user.target\n"
    )
    run_root(["tee", str(_RP1_QUIRK_UNIT_FILE)], input_text=unit, check=True)
    run_root(["systemctl", "daemon-reload"], check=True)
    run_root(["systemctl", "enable", "--now", _RP1_QUIRK_UNIT], check=True)
    print("  Done.")


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``can.setup`` subcommand."""
    subparsers.add_parser(
        "can.setup",
        help="Configure CAN interfaces for the Axol arm.",
    ).set_defaults(func=run)


def rx_alive() -> bool:
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


def bring_up_can() -> None:
    """Run the bring-up script, then verify RX and re-flap once if it's dead.

    Every down/up cycle of the adapter's channels toggles it between a healthy
    state and the TX-only wedge described in :func:`rx_alive`, so a bring-up
    that lands in the wedge is recovered by exactly one more cycle. A robot
    with its motors powered off is indistinguishable from the wedge, hence the
    bounded retries and the warning instead of an error.
    """
    print("Bringing up CAN interfaces (requires sudo)...")
    hub_present = (Path("/sys/class/net") / _CAN_L).exists() and (
        Path("/sys/class/net") / _CAN_R
    ).exists()
    if not hub_present:
        # Cart-only host state: the script still brings up the wheel bus; the
        # RX-wedge probe/re-flap is hub-specific, so there's nothing to verify.
        run_root(["bash", str(_CRON_SCRIPT)], check=True)
        print("  Done — arm hub not attached, arm interfaces skipped.")
        return
    for attempt in range(3):
        run_root(["bash", str(_CRON_SCRIPT)], check=True)
        if rx_alive():
            print("  Done — motors responding.")
            return
        if attempt < 2:
            print("  No motor responses (adapter RX may be wedged) — cycling again...")
    print(
        "  WARNING: no motor responded after bring-up. If the robot is powered "
        "on, re-run this command; otherwise this is expected."
    )


def iface_up(channel: str) -> bool:
    """True when the interface exists and is administratively up (IFF_UP)."""
    try:
        flags = int(
            (Path("/sys/class/net") / channel / "flags").read_text().strip(), 16
        )
    except (OSError, ValueError):
        return False
    return bool(flags & 0x1)


def bring_up_interfaces(channels: list[str]) -> None:
    """Configure and bring up arbitrary SocketCAN interfaces.

    The non-Axol-hub counterpart of :func:`bring_up_can`, for setups running
    on some other CAN adapter: no startup script, no udev naming, no RX-wedge
    cycling — just per-interface bitrate / txqueuelen / up. Interfaces already
    up are left untouched; a missing one raises ``RuntimeError`` naming it so
    callers (CLI, control panel) can surface which channel to fix.
    """
    missing = [ch for ch in channels if not (Path("/sys/class/net") / ch).exists()]
    if missing:
        raise RuntimeError(f"CAN interface not found: {', '.join(missing)}")
    for channel in channels:
        if iface_up(channel):
            print(f"  {channel}: already up.")
            continue
        print(f"  {channel}: bringing up at {_BITRATE} bit/s (requires sudo)...")
        run_root(
            ["ip", "link", "set", channel, "type", "can", "bitrate", str(_BITRATE)],
            check=True,
        )
        run_root(
            ["ip", "link", "set", channel, "txqueuelen", str(_TXQUEUELEN)],
            check=True,
        )
        run_root(["ip", "link", "set", channel, "up"], check=True)
    print("  Done.")


def is_configured() -> bool:
    """True when persistent CAN config has been written by a prior setup.

    Used by the control panel to decide whether connecting needs to run the
    full :func:`ensure_setup` (first time on a machine) or can just bring the
    already-named interfaces up.
    """
    return (
        _UDEV_RULES_FILE.exists()
        and _CRON_SCRIPT.exists()
        and _HOTPLUG_UNIT_FILE.exists()
    )


def ensure_setup(*, serial: str | None = None, base_serial: str | None = None) -> None:
    """Run the full CAN configuration non-interactively (for the control panel).

    Mirrors :func:`run` but resolves the adapter serials without prompting.
    Each step is idempotent, so this is safe to call on a partially-configured
    machine. The cart wheel-bus adapter is only ever *re*-pinned here (from a
    previous setup's rules or a live interface); confirming a new one needs
    the interactive flow — see :func:`_configured_base_serial`.
    """
    driver.ensure_driver()
    serial = serial or _resolve_serial()
    base_serial = base_serial or _configured_base_serial()
    _write_udev_rules(serial, base_serial)
    _write_cron_script(with_base=base_serial is not None)
    _write_hotplug_unit()
    _reload_udev()
    _rename_interfaces(serial, base_serial)
    _register_cron()
    try:
        _setup_rp1_usb_quirk()
    except Exception as exc:  # noqa: BLE001 - the quirk must never block CAN setup
        print(f"  WARNING: RP1 USB quirk setup failed: {exc}")
    bring_up_can()


def _find_base_serial(hub_serial: str) -> str | None:
    """Interactively pick the cart wheel-bus adapter, or None for no cart.

    A previously pinned adapter is kept without prompting; otherwise any
    attached single-channel candlelight adapter is offered. Opt-in ([y/N])
    because a single-channel adapter isn't necessarily a cart — it could be
    any other candlelight device on the host.
    """
    configured = _configured_base_serial()
    if configured:
        print(f"Cart wheel bus: keeping configured adapter (serial {configured}).")
        return configured
    candidates = _detect_base_serials(exclude=hub_serial)
    if not candidates:
        return None
    if len(candidates) == 1:
        prompt = (
            f"Found a single-channel CAN adapter (serial {candidates[0]}). "
            f"Use it as the powered cart's wheel bus ({_CAN_B})? [y/N]: "
        )
        return candidates[0] if input(prompt).strip().lower() == "y" else None
    print("  Multiple single-channel CAN adapters found:")
    for i, s in enumerate(candidates):
        print(f"    [{i}] {s}")
    choice = input(
        f"  Index of the powered cart's wheel-bus adapter ({_CAN_B}), blank for none: "
    ).strip()
    return candidates[int(choice)] if choice else None


def run(_args: object = None) -> None:
    """Configure persistent CAN interfaces and a @reboot bring-up entry."""
    driver.ensure_driver()
    serial = _find_serial()
    base_serial = _find_base_serial(serial)
    ensure_setup(serial=serial, base_serial=base_serial)

    print()
    print("Setup complete.")
    print(f"  Left arm : {_CAN_L}")
    print(f"  Right arm: {_CAN_R}")
    if base_serial:
        print(f"  Cart     : {_CAN_B}")
    print(f"  Startup  : {_CRON_SCRIPT} (runs at @reboot via root crontab)")
    print(
        f"  Hotplug  : {_HOTPLUG_UNIT} (re-runs the startup script whenever "
        f"the adapter re-enumerates, e.g. after a mid-session USB drop)"
    )
    if _is_raspberry_pi_5():
        print(f"  Pi 5     : {_RP1_QUIRK_UNIT} (RP1 USB EMI-tolerance quirk)")
