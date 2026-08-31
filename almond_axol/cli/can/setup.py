"""
axol can.setup

Sets persistent CAN interface names for the Almond Axol CAN bus adapters,
registers a root crontab @reboot entry to bring up the interfaces, and
installs a udev-triggered systemd unit that re-runs the bring-up whenever an
adapter (re-)enumerates — so a mid-session USB drop of the hub (EMI from the
arms can kick it off the bus, most visibly on Raspberry Pi 5 hosts) heals
itself without operator action.

Up to four Axol buses, every one of them optional and independent:

  - The Almond Axol arm hub adapter (VID 0x1D50 / PID 0x606F) exposes two
    CAN channels on a single USB device:
      channel 0 (dev_id 0x0) -> can_alm_axol_l  (left arm)
      channel 1 (dev_id 0x1) -> can_alm_axol_r  (right arm)
  - The powered cart's wheel bus: a single-channel candlelight adapter
    (same generic VID/PID) carrying the four Damiao wheel motors at CAN
    IDs 0x01-0x04, named can_alm_axol_b.
  - The chest bus: another single-channel adapter, carrying the jelly_legs
    lift controller (listens on 0x420, answers on 0x421 — see
    ``almond_axol.robot.lift``), named can_alm_axol_c.

The hub is told apart from the single-channel adapters by channel count: it
always enumerates both channels under one serial, the others exactly one.
The two single-channel adapters are physically identical, so they are told
apart by *probing*: a bus whose jelly_legs board answers a GET_STATUS is the
chest, one whose Damiao motors answer a register read is the wheels; a bus
where nothing answers (devices unpowered) falls back to asking the operator.
A Jetson host's built-in system CAN controller (mttcan) has no USB serial
and is never touched.

On Raspberry Pi 5 hosts the setup additionally raises the RP1 USB
controllers' EMI tolerance (see :func:`_setup_rp1_usb_quirk`), which targets
the disconnects at their source; the hotplug bring-up covers whatever still
gets through.
"""

import re
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

from ...constants import CAN_BASE, CAN_BRINGUP_SCRIPT, CAN_CHEST, CAN_LEFT, CAN_RIGHT
from ...utils.sudo import run_root
from . import driver

_VID = "1d50"
_PID = "606f"
_CAN_L = CAN_LEFT
_CAN_R = CAN_RIGHT
_CAN_B = CAN_BASE
_CAN_C = CAN_CHEST
_BITRATE = 1_000_000
_TXQUEUELEN = 512

_UDEV_RULES_FILE = Path("/etc/udev/rules.d/90-can.rules")
_CAN_DIR = CAN_BRINGUP_SCRIPT.parent
_CRON_SCRIPT = CAN_BRINGUP_SCRIPT

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


def _detect_single_serials(exclude: set[str]) -> list[str]:
    """Serials of attached single-channel adapters — wheel/chest bus candidates."""
    return [
        serial
        for serial, a in _scan_adapters().items()
        if len(a["dev_ids"]) == 1 and serial not in exclude
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


def _configured_named_serial(name: str) -> str | None:
    """A single-channel adapter's serial as pinned by a previous setup.

    Never auto-detected outside the interactive ``axol can.setup`` flow: a
    single-channel candlelight adapter is indistinguishable from unrelated
    hardware (UMI rigs) without probing, so only a serial the operator has
    already confirmed — a live named interface or a written udev rule —
    counts here.
    """
    return _serial_of_interface(name) or _rules_serial_for(name)


def _resolve_hub_serial() -> str | None:
    """Pick the hub serial without prompting (for headless ``ensure_setup``).

    A previously configured serial (named ``can_alm_axol_*`` interfaces, or
    the pinned serial in the udev rules) wins while its adapter is attached —
    or while no dual-channel candidate is attached at all (an unplugged hub
    keeps its pin for whenever it returns) — so re-running setup on an
    already-configured host works no matter how many other candlelight
    adapters are attached.

    A configured serial that is *absent* while a different hub is attached is
    stale — this host last ran on another Axol, or the hub was replaced — and
    must not win: preferring it would re-pin the missing adapter and leave the
    attached hub unnamed on ``canX``, which is exactly what the control
    panel's Connect used to trip over. The attached hub is registered instead
    (when it's unambiguous), matching what the interactive ``axol can.setup``
    picks in the same situation. Several attached candidates still raise,
    since that needs the interactive flow to disambiguate.
    """
    configured = _configured_serial()
    attached = _detect_serials()
    if configured and (configured in attached or not attached):
        return configured
    if len(attached) == 1:
        return attached[0]
    if not attached:
        return None
    raise RuntimeError(
        "Multiple CAN adapters found — run `axol can.setup` once to pick the Axol's"
    )


def _find_serial() -> str | None:
    """Interactively pick the arm hub adapter, or None when there is no hub."""
    print(f"Scanning for the Almond Axol arm hub adapter ({_VID}:{_PID})...")

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
            "\n  No arm hub found. Enter its serial manually, or leave blank "
            "for a robot without one (cart/chest only):"
        )
        return input("  Serial: ").strip() or None

    if len(unique) == 1:
        print(f"  Found adapter — serial: {unique[0]}")
        return unique[0]

    print("  Multiple adapters found:")
    for i, s in enumerate(unique):
        print(f"    [{i}] {s}")
    idx = input("  Select adapter index [0]: ").strip() or "0"
    return unique[int(idx)]


# --------------------------------------------------------------------------
# Single-channel adapter identification (wheels vs chest)
# --------------------------------------------------------------------------

# The jelly_legs lift controller listens on 0x420 and answers on 0x421
# (see almond_axol.robot.lift); a GET_STATUS (opcode 0x04) provokes exactly
# one status frame. Both IDs are clear of every Damiao/MyActuator range.
_JELLY_CMD_ID = 0x420
_JELLY_STATUS_ID = 0x421
_JELLY_GET_STATUS = bytes([0x04])
# SET_RATE 0 (opcode 0x05, uint16 period 0): turns the 50 ms status
# broadcast off. The board starts broadcasting as soon as it has received
# *any* frame, and that stream starves the CANable's gs_usb TX path (see
# the firmware README bench note) — so every probe sequence must silence
# the board before expecting its own transmissions to get through.
_JELLY_SET_RATE_OFF = bytes([0x05, 0x00, 0x00])
# Damiao register read: 0x7FF [id_lo, id_hi, 0x33, rid, ...]; the motor
# echoes a 0x33 reply on its feedback ID. Register 60 (VBUS) is read-only.
_DAMIAO_CFG_ID = 0x7FF
_DAMIAO_WHEEL_IDS = (0x01, 0x02, 0x03, 0x04)
_PROBE_ATTEMPTS = 3
_PROBE_WINDOW_S = 0.4


def _probe(iface: str, frames: list[tuple[int, bytes]], match) -> bool:  # noqa: ANN001
    """Send probe frames on ``iface`` and wait briefly for a matching reply.

    Raw SocketCAN (no python-can machinery needed for a one-shot probe).
    Unanswered probes are harmless: the IDs used command nothing, and a
    frame left queued behind an unpowered bus is dropped by the bring-up
    script's interface flap.
    """
    try:
        s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        s.bind((iface,))
    except OSError:
        return False
    try:
        for _ in range(_PROBE_ATTEMPTS):
            for can_id, data in frames:
                try:
                    s.send(
                        struct.pack("<IB3x8s", can_id, len(data), data.ljust(8, b"\0"))
                    )
                except OSError:
                    return False  # interface down / TX queue wedged
            deadline = time.monotonic() + _PROBE_WINDOW_S
            while (remaining := deadline - time.monotonic()) > 0:
                s.settimeout(remaining)
                try:
                    frame = s.recv(16)
                except (TimeoutError, OSError):
                    break
                can_id, dlc = struct.unpack("<IB3x", frame[:8])
                if match(can_id & 0x7FF, frame[8 : 8 + dlc]):
                    return True
    finally:
        s.close()
    return False


def _send_once(iface: str, can_id: int, data: bytes) -> None:
    """Fire one frame on ``iface`` and return; no reply expected."""
    try:
        s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        s.bind((iface,))
    except OSError:
        return
    try:
        s.send(struct.pack("<IB3x8s", can_id, len(data), data.ljust(8, b"\0")))
    except OSError:
        pass
    finally:
        s.close()


def _probe_chest(iface: str) -> bool:
    """True when a jelly_legs board answers a GET_STATUS on ``iface``.

    Each attempt sends SET_RATE 0 before the GET_STATUS so the board's
    status broadcast stays off — both for the probe's own reply and for
    whatever runs on this bus next.
    """
    return _probe(
        iface,
        [
            (_JELLY_CMD_ID, _JELLY_SET_RATE_OFF),
            (_JELLY_CMD_ID, _JELLY_GET_STATUS),
        ],
        lambda can_id, _data: can_id == _JELLY_STATUS_ID,
    )


def _probe_wheels(iface: str) -> bool:
    """True when a Damiao wheel motor (ID 0x01-0x04) answers on ``iface``."""
    frames = [
        (_DAMIAO_CFG_ID, bytes([mid, 0x00, 0x33, 60, 0, 0, 0, 0]))
        for mid in _DAMIAO_WHEEL_IDS
    ]

    def is_reply(_can_id: int, data: bytes) -> bool:
        return (
            len(data) == 8
            and data[2] == 0x33
            and data[1] == 0x00
            and data[0] in _DAMIAO_WHEEL_IDS
        )

    return _probe(iface, frames, is_reply)


def _iface_for_serial(serial: str) -> str | None:
    """The current interface name of a single-channel adapter, or None."""
    for iface_path in Path("/sys/class/net").glob("can*"):
        info = subprocess.run(
            ["udevadm", "info", "-a", "-p", str(iface_path)],
            capture_output=True,
            text=True,
        ).stdout
        if _udev_attr(info, "ATTRS{serial}") == serial:
            return iface_path.name
    return None


def _identify_adapter(serial: str) -> str | None:
    """Probe a single-channel adapter's bus: ``"wheels"``, ``"chest"``, or None.

    Brings the interface up first (root); a bus where nothing answers —
    devices unpowered, or unrelated hardware like a UMI rig — stays None and
    is left to the operator.
    """
    iface = _iface_for_serial(serial)
    if iface is None:
        return None
    try:
        bring_up_interfaces([iface])
    except RuntimeError:
        return None
    # Silence any jelly_legs board before the wheel probe: the board starts
    # its 50 ms broadcast after the first frame it sees (the wheel probe's
    # own Damiao reads would wake it), and that stream starves the CANable's
    # TX path — on a combined bus the wheel probe would then go deaf and the
    # bus would be misclassified as chest-only. Harmless where no board is
    # listening; a frame queued on a dead bus is dropped by the bring-up flap.
    _send_once(iface, _JELLY_CMD_ID, _JELLY_SET_RATE_OFF)
    time.sleep(0.05)
    wheels = _probe_wheels(iface)
    chest = _probe_chest(iface)
    if chest and wheels:
        # The pre-split combined cart bus (jelly_legs next to the wheels).
        print(
            f"  WARNING: both the wheel motors and the jelly_legs board "
            f"answer on {iface} — treating it as the wheel bus. Point the "
            f"lift at it explicitly (cart.lift_channel={_CAN_B}) or move "
            f"the lift onto its own chest bus."
        )
        return "wheels"
    if chest:
        return "chest"
    if wheels:
        return "wheels"
    return None


def _write_udev_rules(
    hub_serial: str | None,
    wheels_serial: str | None = None,
    chest_serial: str | None = None,
) -> None:
    print(f"Writing udev rules to {_UDEV_RULES_FILE} (requires sudo)...")
    content = ""
    if hub_serial:
        content += (
            f"# Almond Axol dual-channel CAN adapter\n"
            f"# Adapter serial: {hub_serial}\n"
            f"# Channel 0 -> left arm\n"
            f'SUBSYSTEM=="net", ACTION=="add", ATTRS{{idVendor}}=="{_VID}", ATTRS{{idProduct}}=="{_PID}", ATTRS{{serial}}=="{hub_serial}", ATTR{{dev_id}}=="0x0", NAME="{_CAN_L}"\n'
            f"# Channel 1 -> right arm\n"
            f'SUBSYSTEM=="net", ACTION=="add", ATTRS{{idVendor}}=="{_VID}", ATTRS{{idProduct}}=="{_PID}", ATTRS{{serial}}=="{hub_serial}", ATTR{{dev_id}}=="0x1", NAME="{_CAN_R}"\n'
            f"# Every (re-)enumeration — boot or a mid-session USB drop — pulls in\n"
            f"# the bring-up service so the channels come back configured and up.\n"
            f"# Tagged on the USB device rather than the net interfaces: the NAME=\n"
            f"# rules above put every real hotplug add mid-rename, and systemd\n"
            f'# skips SYSTEMD_WANTS on renaming devices ("device is renaming").\n'
            f'SUBSYSTEM=="usb", ENV{{DEVTYPE}}=="usb_device", ACTION=="add", ATTR{{idVendor}}=="{_VID}", ATTR{{idProduct}}=="{_PID}", ATTR{{serial}}=="{hub_serial}", TAG+="systemd", ENV{{SYSTEMD_WANTS}}+="{_HOTPLUG_UNIT}"\n'
        )
    # Single-channel adapters, matched by serial alone: CANable firmware
    # variants ship various VID/PIDs, and the serial already identifies the
    # exact adapter.
    for label, name, serial in (
        ("Powered-cart wheel bus", _CAN_B, wheels_serial),
        ("Chest bus (jelly_legs lift controller)", _CAN_C, chest_serial),
    ):
        if not serial:
            continue
        content += (
            f"# {label} (single-channel adapter)\n"
            f"# Adapter serial: {serial}\n"
            f'SUBSYSTEM=="net", ACTION=="add", ATTRS{{serial}}=="{serial}", NAME="{name}"\n'
            f'SUBSYSTEM=="usb", ENV{{DEVTYPE}}=="usb_device", ACTION=="add", ATTR{{serial}}=="{serial}", TAG+="systemd", ENV{{SYSTEMD_WANTS}}+="{_HOTPLUG_UNIT}"\n'
        )
    run_root(["tee", str(_UDEV_RULES_FILE)], input_text=content, check=True)
    print("  Done.")


def _reload_udev() -> None:
    print("Reloading udev rules (requires sudo)...")
    run_root(["udevadm", "control", "--reload-rules"], check=True)
    run_root(["systemctl", "restart", "systemd-udevd"], check=True)
    print("  Done.")


def _validate_adapter_assignments(
    hub_serial: str | None,
    wheels_serial: str | None,
    chest_serial: str | None,
) -> None:
    """Reject one physical adapter being assigned to incompatible roles."""
    seen: dict[str, str] = {}
    for role, serial in (
        ("arm hub", hub_serial),
        ("wheel bus", wheels_serial),
        ("chest bus", chest_serial),
    ):
        if not serial:
            continue
        previous_role = seen.get(serial)
        if previous_role:
            raise RuntimeError(
                f"Adapter {serial} cannot be assigned to both the "
                f"{previous_role} and {role}."
            )
        seen[serial] = role


def _rename_interfaces(
    hub_serial: str | None,
    wheels_serial: str | None = None,
    chest_serial: str | None = None,
) -> None:
    """Rename existing canX interfaces to their target names without replug.

    Every move is staged through a temporary ``can_*`` name before any final
    name is assigned.  That extra phase matters when live probing corrects a
    stale wheel/chest assignment: Linux will not directly rename B to C while
    C still exists (or vice versa), even when both interfaces are down.
    """
    print("Renaming CAN interfaces (requires sudo)...")
    _validate_adapter_assignments(hub_serial, wheels_serial, chest_serial)

    # (adapter serial, channel dev_id) -> persistent name. The wheel/chest
    # adapters are single-channel, so their only interface is dev_id 0.
    target: dict[tuple[str, int], str] = {}
    if hub_serial:
        target[(hub_serial, 0)] = _CAN_L
        target[(hub_serial, 1)] = _CAN_R
    if wheels_serial:
        target[(wheels_serial, 0)] = _CAN_B
    if chest_serial:
        target[(chest_serial, 0)] = _CAN_C

    net_dir = Path("/sys/class/net")
    iface_paths = list(net_dir.glob("can*"))
    records: dict[str, tuple[str, int]] = {}
    for iface_path in iface_paths:
        iface = iface_path.name
        info = subprocess.run(
            ["udevadm", "info", "-a", "-p", str(iface_path)],
            capture_output=True,
            text=True,
        ).stdout

        iface_serial = _udev_attr(info, "ATTRS{serial}")
        if not iface_serial:
            continue
        try:
            dev_id = int(_udev_attr(info, "ATTR{dev_id}"), 16)
        except ValueError:
            continue
        records[iface] = (iface_serial, dev_id)

    # Assigned adapters that are not already at their destination must move.
    moves: dict[str, str] = {
        iface: target[identity]
        for iface, identity in records.items()
        if identity in target and iface != target[identity]
    }

    # A stale adapter can still occupy a managed name after probing reassigns
    # or removes that role. Stage the old occupant out of the way too; it stays
    # down under a discoverable temporary name until replugged.
    managed_names = {_CAN_L, _CAN_R, _CAN_B, _CAN_C}
    for name in managed_names:
        if name not in records or name in moves:
            continue
        occupant = records[name]
        if target.get(occupant) == name:
            continue
        moves[name] = ""

    # Refuse to rename an unrelated interface. This should be impossible for
    # the reserved can_alm_axol_* names, but detecting it before any mutation
    # is much safer than failing halfway through the staging phase.
    existing_names = {path.name for path in net_dir.iterdir()}
    for destination in set(target.values()):
        if (
            destination in existing_names
            and destination not in records
            and destination not in moves
        ):
            raise RuntimeError(
                f"Cannot rename a CAN interface to {destination}: that name "
                "is already used by an unrelated network interface."
            )

    reserved_names = existing_names | set(target.values())

    def temporary_name(index: int) -> str:
        while True:
            name = f"can_ax_tmp{index}"
            index += 1
            if name not in reserved_names:
                reserved_names.add(name)
                return name

    staged: list[tuple[str, str]] = []
    next_temp = 0
    for iface, new_name in moves.items():
        temp_name = temporary_name(next_temp)
        next_temp += 1
        if new_name:
            print(f"  {iface} -> {new_name}")
        else:
            print(f"  {iface} -> {temp_name} (no longer assigned)")
        run_root(["ip", "link", "set", iface, "down"], check=True)
        run_root(["ip", "link", "set", iface, "name", temp_name], check=True)
        staged.append((temp_name, new_name))

    for temp_name, new_name in staged:
        if new_name:
            run_root(["ip", "link", "set", temp_name, "name", new_name], check=True)

    print("  Done.")


def _write_cron_script() -> None:
    """Write the bring-up script covering all four interfaces.

    Every interface is optional and checked for presence at runtime, so one
    script serves every hardware combination — arm-only, cart-only, chest-
    only, or all of them — and an unplugged adapter never blocks the rest.
    """
    print(f"Writing CAN startup script to {_CRON_SCRIPT}...")
    _CAN_DIR.mkdir(parents=True, exist_ok=True)
    script = (
        f"#!/bin/bash\n"
        f"# Bring up Almond Axol CAN interfaces\n"
        f"#\n"
        f"# Runs at boot (@reboot root crontab) and on every (re-)enumeration\n"
        f"# of an adapter (udev -> {_HOTPLUG_UNIT}), so a mid-session USB\n"
        f"# drop of the hub comes back configured without operator action.\n"
        f"#\n"
        f"# The arm interfaces are channels of one dual-channel gs_usb adapter.\n"
        f"# Bring them down together, configure, then up together — flapping\n"
        f"# the channels one at a time (down/up L, then down/up R) toggles the\n"
        f"# adapter into a state where TX works but no RX frame is delivered.\n"
        f"# Skipped entirely when the hub is unplugged (a hub-less session\n"
        f"# must still bring up the wheel/chest buses below).\n"
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
        f"\n# Wheel and chest buses: their own single-channel adapters, so no\n"
        f"# flap-together dance — and each skipped when absent, so an\n"
        f"# unplugged adapter never blocks the others' bring-up.\n"
        f"for IFACE in {_CAN_B} {_CAN_C}; do\n"
        f'    if ip link show "${{IFACE}}" >/dev/null 2>&1; then\n'
        f'        ip link set "${{IFACE}}" down 2>/dev/null || true\n'
        f'        ip link set "${{IFACE}}" type can bitrate {_BITRATE}\n'
        f'        ip link set "${{IFACE}}" txqueuelen {_TXQUEUELEN}\n'
        f'        ip link set "${{IFACE}}" up\n'
        f"    fi\n"
        f"done\n"
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
        help="Configure CAN interfaces (arm hub, wheel bus, chest bus).",
    ).set_defaults(func=run)


def rx_alive_per_arm() -> tuple[bool, bool]:
    """(left, right) — True where a motor answers on that arm's bus.

    Verifies the adapter's receive path, not just the interface state: the
    dual-channel gs_usb adapter can come out of a down/up cycle in a state
    where TX still works but no received frame is ever delivered (kernel-side
    everything looks healthy — UP, ERROR-ACTIVE, correct bitrate).

    Per-arm results matter because one healthy arm must not mask the other:
    a bus with no responding motors (arm powered off, harness fault, dead
    transceiver channel) looks identical to the adapter wedge on that side.
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

    async def probe_all() -> tuple[bool, bool]:
        await asyncio.sleep(0.5)  # let the freshly-upped interfaces settle
        left, right = await asyncio.gather(probe(_CAN_L), probe(_CAN_R))
        return left, right

    return asyncio.run(probe_all())


def rx_alive() -> bool:
    """True when at least one motor answers on either arm."""
    return any(rx_alive_per_arm())


def bring_up_can() -> None:
    """Run the bring-up script, then verify RX and re-flap once if it's dead.

    Every down/up cycle of the adapter's channels toggles it between a healthy
    state and the TX-only wedge described in :func:`rx_alive_per_arm`, so a
    bring-up that lands in the wedge is recovered by exactly one more cycle.
    A robot with its motors powered off is indistinguishable from the wedge,
    hence the bounded retries and the warning instead of an error. Results are
    reported per arm: one answering arm proves the adapter is healthy (no
    retry), while the other side staying silent is called out instead of being
    masked by it.
    """
    print("Bringing up CAN interfaces (requires sudo)...")
    hub_present = (Path("/sys/class/net") / _CAN_L).exists() and (
        Path("/sys/class/net") / _CAN_R
    ).exists()
    if not hub_present:
        # Hub-less host state: the script still brings up the wheel/chest
        # buses; the RX-wedge probe/re-flap is hub-specific, so there's
        # nothing to verify.
        run_root(["bash", str(_CRON_SCRIPT)], check=True)
        print("  Done — arm hub not attached, arm interfaces skipped.")
        return
    for attempt in range(3):
        run_root(["bash", str(_CRON_SCRIPT)], check=True)
        left, right = rx_alive_per_arm()
        if left and right:
            print("  Done — motors responding on both arms.")
            return
        if left or right:
            # One arm answering proves the adapter's RX path is healthy, so
            # more cycling can't help the silent side — that's an arm-level
            # problem (power, harness, or the hub's transceiver channel).
            alive, silent = ("left", "right") if left else ("right", "left")
            print(f"  Done — motors responding on the {alive} arm.")
            print(
                f"  WARNING: no motor answered on the {silent} arm. Check that "
                f"it is powered and its CAN connection is intact."
            )
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


def _apply_setup(
    hub_serial: str | None,
    wheels_serial: str | None,
    chest_serial: str | None,
) -> None:
    """Write the persistent config for the given adapters and bring them up.

    Each step is idempotent, so this is safe to re-run on a
    partially-configured machine.
    """
    _validate_adapter_assignments(hub_serial, wheels_serial, chest_serial)
    _write_udev_rules(hub_serial, wheels_serial, chest_serial)
    _write_cron_script()
    _write_hotplug_unit()
    _reload_udev()
    _rename_interfaces(hub_serial, wheels_serial, chest_serial)
    _register_cron()
    try:
        _setup_rp1_usb_quirk()
    except Exception as exc:  # noqa: BLE001 - the quirk must never block CAN setup
        print(f"  WARNING: RP1 USB quirk setup failed: {exc}")
    bring_up_can()


def ensure_setup(
    *,
    hub_serial: str | None = None,
    wheels_serial: str | None = None,
    chest_serial: str | None = None,
) -> None:
    """Run the full CAN configuration non-interactively (for the control panel).

    Mirrors :func:`run` but resolves the adapter serials without prompting.
    The wheel-bus and chest adapters are only ever *re*-pinned here (from a
    previous setup's rules or a live interface); identifying a new one needs
    the interactive flow's probing — see :func:`_identify_adapter`.
    """
    driver.ensure_driver()
    hub_serial = hub_serial or _resolve_hub_serial()
    wheels_serial = wheels_serial or _configured_named_serial(_CAN_B)
    chest_serial = chest_serial or _configured_named_serial(_CAN_C)
    if not (hub_serial or wheels_serial or chest_serial):
        raise RuntimeError("Robot not detected")
    _apply_setup(hub_serial, wheels_serial, chest_serial)


def _find_single_serials(hub_serial: str | None) -> tuple[str | None, str | None]:
    """Interactively assign single-channel adapters to the wheel/chest buses.

    Every attached adapter is probed, including adapters pinned by a previous
    setup. A positive device response wins over a stale pin; an attached but
    silent (or currently unplugged) configured adapter keeps its previous role
    as an unverified fallback. A new silent adapter is offered to the operator
    instead of being guessed at, and may explicitly replace such a fallback.

    Returns ``(wheels_serial, chest_serial)``, either of which may be None.
    """
    configured = {
        "wheels": _configured_named_serial(_CAN_B),
        "chest": _configured_named_serial(_CAN_C),
    }
    exclude = {hub_serial} if hub_serial else set()
    attached = sorted(_detect_single_serials(exclude))
    detected: dict[str, str | None] = {}
    if attached:
        configured_note = (
            ", including previously configured adapters"
            if any(configured.values())
            else ""
        )
        print(
            f"Identifying {len(attached)} single-channel CAN adapter(s) by "
            f"probing{configured_note} (wheel motors / jelly_legs board must "
            "be powered)..."
        )
        for serial in attached:
            detected[serial] = _identify_adapter(serial)

    conflicting = configured["wheels"]
    if conflicting and conflicting == configured["chest"]:
        observed = detected.get(conflicting)
        live_roles = {role for role in detected.values() if role is not None}
        if observed is None and not live_roles:
            _die(
                f"Adapter {conflicting} is pinned as both the wheel and chest "
                "buses, and no device answered to resolve the conflict. Power "
                "the cart hardware or remove the conflicting CAN udev rule, "
                "then re-run setup."
            )

    response_labels = {
        "wheels": f"Damiao wheel motors answered -> {_CAN_B}",
        "chest": f"jelly_legs board answered -> {_CAN_C}",
    }

    def detected_for(role: str) -> str | None:
        matches = sorted(serial for serial, found in detected.items() if found == role)
        if not matches:
            return None
        old_serial = configured[role]
        selected = old_serial if old_serial in matches else matches[0]
        print(f"  {selected}: {response_labels[role]}")
        for serial in matches:
            if serial != selected:
                print(
                    f"  {serial}: also identified as the {role} bus, but that "
                    "role is already assigned — skipping."
                )
        return selected

    assigned = {
        "wheels": detected_for("wheels"),
        "chest": detected_for("chest"),
    }
    source = {
        role: ("detected" if serial else None) for role, serial in assigned.items()
    }

    # Keep old pins only when no live response contradicts them. A later
    # operator choice may replace these unverified fallbacks.
    for role, other_role, label in (
        ("wheels", "chest", "Cart wheel bus"),
        ("chest", "wheels", "Chest bus"),
    ):
        old_serial = configured[role]
        if assigned[role] or not old_serial or old_serial == assigned[other_role]:
            continue
        observed = detected.get(old_serial) if old_serial in detected else None
        if old_serial in detected and observed is not None:
            continue
        assigned[role] = old_serial
        source[role] = "configured"
        if old_serial in detected:
            print(
                f"  {label}: no identifying device answered; keeping "
                f"configured adapter (serial {old_serial}) unverified."
            )
        else:
            print(
                f"  {label}: configured adapter {old_serial} is not attached; "
                "keeping its assignment for when it returns."
            )

    unidentified = [
        serial
        for serial, role in sorted(detected.items())
        if role is None and serial not in assigned.values()
    ]
    for serial in unidentified:
        print(f"  {serial}: nothing answered on this adapter's bus.")
        choice = (
            input(
                f"    Assign it to the [w]heel bus ({_CAN_B}), the [c]hest "
                f"bus ({_CAN_C}), or leave blank to skip: "
            )
            .strip()
            .lower()
        )
        selected_role = {"w": "wheels", "c": "chest"}.get(choice)
        if selected_role is None:
            continue
        previous = assigned[selected_role]
        if previous is not None and source[selected_role] != "configured":
            print("    That bus is already assigned — skipping.")
            continue
        if previous is not None:
            print(
                f"    Replacing unverified configured adapter {previous} with {serial}."
            )
        assigned[selected_role] = serial
        source[selected_role] = "operator"

    wheels = assigned["wheels"]
    chest = assigned["chest"]
    if wheels and wheels == chest:
        _die(f"Adapter {wheels} cannot be assigned to both wheel and chest buses.")
    return wheels, chest


def run(_args: object = None) -> None:
    """Configure persistent CAN interfaces and a @reboot bring-up entry."""
    driver.ensure_driver()
    hub_serial = _find_serial()
    wheels_serial, chest_serial = _find_single_serials(hub_serial)
    if not (hub_serial or wheels_serial or chest_serial):
        _die(
            "No CAN adapters found or configured. Connect the arm hub, "
            "wheel-bus, or chest adapter and re-run."
        )
    _apply_setup(hub_serial, wheels_serial, chest_serial)

    print()
    print("Setup complete.")
    if hub_serial:
        print(f"  Left arm : {_CAN_L}")
        print(f"  Right arm: {_CAN_R}")
    if wheels_serial:
        print(f"  Wheels   : {_CAN_B}")
    if chest_serial:
        print(f"  Chest    : {_CAN_C} (jelly_legs lift)")
    print(f"  Startup  : {_CRON_SCRIPT} (runs at @reboot via root crontab)")
    print(
        f"  Hotplug  : {_HOTPLUG_UNIT} (re-runs the startup script whenever "
        f"an adapter re-enumerates, e.g. after a mid-session USB drop)"
    )
    if _is_raspberry_pi_5():
        print(f"  Pi 5     : {_RP1_QUIRK_UNIT} (RP1 USB EMI-tolerance quirk)")
