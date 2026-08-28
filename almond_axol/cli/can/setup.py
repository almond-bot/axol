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

The Mantis handheld data-collection rig uses the **same dual-channel board**
as the arm hub (one USB device, keyed by its serial), with one gripper bus per
channel:
  channel 0 (dev_id 0x0) -> can_mantis_l  (left gripper: motor + trigger)
  channel 1 (dev_id 0x1) -> can_mantis_r  (right gripper)
``can.setup`` distinguishes them by probing the attached devices: a Mantis
trigger publishes valid frames on CAN ID 0x009, while an Axol arm has a
MyActuator shoulder motor at ID 0x001. An unpowered/unidentified adapter is
offered to the operator rather than guessed. The two profiles use separate
udev rule files, startup scripts, and hotplug units, so a machine can have both
configured at once.
"""

import re
import socket
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ...constants import (
    CAN_BASE,
    CAN_BRINGUP_SCRIPT,
    CAN_CHEST,
    CAN_LEFT,
    CAN_RIGHT,
    CAN_MANTIS_LEFT,
    CAN_MANTIS_RIGHT,
    Joint,
)
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


@dataclass(frozen=True)
class _Profile:
    """One adapter's persistent-naming setup (rule file, names, bring-up script).

    Both profiles describe the same dual-channel gs_usb board: channel 0
    (``dev_id`` 0x0) is the left interface, channel 1 the right.
    """

    label: str
    left: str
    right: str
    rules_file: Path
    cron_script: Path
    # Joint whose motor the post-bring-up RX probe queries (see rx_alive).
    probe_joint: Joint
    # Lock file serializing runs of the bring-up script (boot and hotplug
    # triggers can race; see _write_cron_script).
    lock_name: str
    # Hotplug bring-up: pulled in via the udev rules (SYSTEMD_WANTS) whenever
    # the adapter (re-)enumerates, so interfaces recreated by a mid-session
    # USB drop come back configured and up without operator action.
    hotplug_unit: str

    @property
    def hotplug_unit_file(self) -> Path:
        return Path("/etc/systemd/system") / self.hotplug_unit


_AXOL_PROFILE = _Profile(
    label="Almond Axol arm",
    left=_CAN_L,
    right=_CAN_R,
    rules_file=_UDEV_RULES_FILE,
    cron_script=_CRON_SCRIPT,
    probe_joint=Joint.SHOULDER_1,
    lock_name="axol-can-up.lock",
    hotplug_unit="axol-can-up.service",
)

_MANTIS_PROFILE = _Profile(
    label="Almond Mantis",
    left=CAN_MANTIS_LEFT,
    right=CAN_MANTIS_RIGHT,
    rules_file=Path("/etc/udev/rules.d/91-can-mantis.rules"),
    cron_script=_CAN_DIR / "startup_mantis.sh",
    probe_joint=Joint.GRIPPER,
    lock_name="axol-can-mantis-up.lock",
    hotplug_unit="axol-can-mantis-up.service",
)

# Configuration written before the product was consistently named Mantis.
# These values are assembled so the retired name is not exposed in source,
# logs, help, or generated configuration. They are read only for one-way
# migration when ``can.setup`` next runs.
_PRE_MANTIS_NAME = "u" + "mi"
_PRE_MANTIS_LEFT = f"can_alm_{_PRE_MANTIS_NAME}_l"
_PRE_MANTIS_RIGHT = f"can_alm_{_PRE_MANTIS_NAME}_r"
_PRE_MANTIS_RULES_FILE = Path(f"/etc/udev/rules.d/91-can-{_PRE_MANTIS_NAME}.rules")
_PRE_MANTIS_CRON_SCRIPT = _CAN_DIR / f"startup_{_PRE_MANTIS_NAME}.sh"
_PRE_MANTIS_HOTPLUG_UNIT = f"axol-can-{_PRE_MANTIS_NAME}-up.service"
_PRE_MANTIS_HOTPLUG_UNIT_FILE = Path("/etc/systemd/system") / _PRE_MANTIS_HOTPLUG_UNIT

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

    Dual-channel boards (the Axol arm hub, the Mantis) show up as one
    serial with dev_ids {0, 1}; a single-channel adapter (the cart's
    wheel-bus CANable) as one serial with {0}. Matched on the gs_usb driver
    rather than a VID/PID so CANable firmware variants that don't use the
    candlelight 1d50:606f IDs still count; the Jetson's built-in mttcan
    controller has no USB serial and is excluded either way.
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
    """Serials of every attached *dual-channel* Axol adapter.

    Candidates for the arm hub or the Mantis (same board — the caller
    disambiguates via the other profile's claimed serials). Single-channel
    devices (the cart's wheel-bus adapter) share the generic VID/PID but can
    never be either, so they are excluded rather than left to make the scan
    ambiguous.
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


def _rules_serial_for(name: str, rules_file: Path | None = None) -> str | None:
    """The serial pinned to interface ``name`` in a profile's udev rules."""
    try:
        rules = (rules_file or _UDEV_RULES_FILE).read_text()
    except OSError:
        return None
    match = re.search(
        r'ATTRS\{serial\}=="([^"]+)"[^\n]*NAME="' + re.escape(name) + '"', rules
    )
    return match.group(1) if match else None


def _configured_serial(profile: _Profile = _AXOL_PROFILE) -> str | None:
    """A profile's adapter serial as pinned by a *previous* setup, if any.

    Preferred over live adapter detection: the arm hub and the Mantis use
    the same dual-channel board (same VID/PID), so a host with both attached
    is ambiguous to a fresh scan — but not to a machine that has already
    named the profile's interfaces or written its udev rules.
    """
    for iface in (profile.left, profile.right):
        serial = _serial_of_interface(iface)
        if serial:
            return serial
    configured = _rules_serial_for(
        profile.left, profile.rules_file
    ) or _rules_serial_for(profile.right, profile.rules_file)
    if configured or profile is not _MANTIS_PROFILE:
        return configured

    # One-way migration from the former Mantis interface/rule names.
    for iface in (_PRE_MANTIS_LEFT, _PRE_MANTIS_RIGHT):
        serial = _serial_of_interface(iface)
        if serial:
            return serial
    return _rules_serial_for(
        _PRE_MANTIS_LEFT, _PRE_MANTIS_RULES_FILE
    ) or _rules_serial_for(_PRE_MANTIS_RIGHT, _PRE_MANTIS_RULES_FILE)


def _mantis_claimed_serials() -> set[str]:
    """Serials claimed by current or pre-Mantis persistent rules."""
    return _serials_in_rules(_MANTIS_PROFILE.rules_file) | _serials_in_rules(
        _PRE_MANTIS_RULES_FILE
    )


def _configured_named_serial(name: str) -> str | None:
    """A single-channel adapter's serial as pinned by a previous setup.

    Never auto-detected outside the interactive ``axol can.setup`` flow: a
    single-channel candlelight adapter is indistinguishable from unrelated
    hardware without probing, so only a serial the operator has already
    confirmed — a live named interface or a written udev rule — counts here.
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
    picks in the same situation. Serials the Mantis rig's rules already claim
    are excluded from live detection — the rig uses the same dual-channel
    board as the hub (1d50:606f), so with the rig plugged in it would
    otherwise make the robot's adapter ambiguous. Several attached candidates
    still raise, since that needs the interactive flow to disambiguate.
    """
    configured = _configured_serial()
    claimed = _mantis_claimed_serials()
    attached = [s for s in _detect_serials() if s not in claimed]
    if configured and (configured in attached or not attached):
        return configured
    if len(attached) == 1:
        return attached[0]
    if not attached:
        return None
    raise RuntimeError(
        "Multiple CAN adapters found — run `axol can.setup` once to pick the Axol's"
    )


def _find_dual_serials() -> tuple[str | None, str | None]:
    """Automatically assign attached dual-channel hubs to Axol/Mantis.

    Previously pinned, still-attached serials win. New or replacement hubs are
    probed just like the single-channel peripherals; silence is presented for
    manual assignment. Returns ``(axol_serial, mantis_serial)``.
    """
    print(f"Scanning for dual-channel CAN adapters ({_VID}:{_PID})...")
    attached = set(_detect_serials())
    configured_axol = _configured_serial(_AXOL_PROFILE)
    configured_mantis = _configured_serial(_MANTIS_PROFILE)

    # Preserve an unplugged configured hub only when there is no possible
    # replacement attached. Otherwise probe the live candidates and replace a
    # stale pin, matching the old one-profile setup behavior.
    axol = configured_axol if configured_axol in attached or not attached else None
    mantis = (
        configured_mantis if configured_mantis in attached or not attached else None
    )
    if axol:
        print(f"  Axol: keeping configured adapter (serial {axol}).")
    if mantis:
        print(f"  Mantis: keeping configured adapter (serial {mantis}).")
    if axol is not None and axol == mantis:
        _die(
            f"Adapter {axol} is pinned as both Axol and Mantis. Remove the "
            "conflicting CAN udev rule and re-run setup."
        )

    candidates = sorted(attached - {s for s in (axol, mantis) if s})
    unidentified: list[str] = []
    for serial in candidates:
        print(f"  {serial}: probing Mantis trigger / Axol shoulder...")
        role = _identify_dual_adapter(serial)
        if role == "axol" and axol is None:
            axol = serial
            print(f"  {serial}: Axol shoulder answered -> arm hub")
        elif role == "mantis" and mantis is None:
            mantis = serial
            print(f"  {serial}: Mantis trigger answered -> handheld rig")
        elif role is not None:
            print(
                f"  {serial}: identified as {role}, but that role is already "
                "pinned to another adapter — skipping."
            )
        else:
            unidentified.append(serial)

    for serial in unidentified:
        print(f"  {serial}: no identifying device answered.")
        choice = (
            input(
                "    Assign it to the [a]xol arm hub, [m]antis rig, "
                "or leave blank to skip: "
            )
            .strip()
            .lower()
        )
        if choice == "a" and axol is None:
            axol = serial
        elif choice == "m" and mantis is None:
            mantis = serial
        elif choice in ("a", "m"):
            print("    That role is already assigned — skipping.")
    return axol, mantis


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

# Dual-channel hub identity probes. The Mantis trigger is the strongest
# discriminator because it does not exist on an Axol arm. Its firmware emits
# ``<fH`` (normalised position + 12-bit raw ADC) at 100 Hz on 0x009. The arm
# fallback asks its MyActuator shoulder-1 (ID 0x01) for status frame 1.
_MANTIS_TRIGGER_ID = 0x009
_MANTIS_TRIGGER_DLC = 6
_AXOL_SHOULDER_REQ_ID = 0x141
_AXOL_SHOULDER_RESP_ID = 0x241
_MYACTUATOR_STATUS1 = 0x9A


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


def _probe_mantis_trigger(iface: str) -> bool:
    """True when a valid Mantis trigger broadcast is observed on ``iface``."""

    def is_trigger(can_id: int, data: bytes) -> bool:
        if can_id != _MANTIS_TRIGGER_ID or len(data) != _MANTIS_TRIGGER_DLC:
            return False
        try:
            position, raw = struct.unpack("<fH", data)
        except struct.error:
            return False
        # The range checks reject NaN too. Validate the otherwise-unused ADC
        # field so unrelated traffic sharing 0x009 cannot identify a hub.
        return 0.0 <= position <= 1.0 and raw <= 0x0FFF

    # No request is necessary: the trigger publishes continuously at 100 Hz.
    return _probe(iface, [], is_trigger)


def _probe_axol_shoulder(iface: str) -> bool:
    """True when the Axol arm's shoulder-1 motor answers on ``iface``."""
    request = bytes([_MYACTUATOR_STATUS1, 0, 0, 0, 0, 0, 0, 0])
    return _probe(
        iface,
        [(_AXOL_SHOULDER_REQ_ID, request)],
        lambda can_id, data: (
            can_id == _AXOL_SHOULDER_RESP_ID
            and len(data) == 8
            and data[0] == _MYACTUATOR_STATUS1
        ),
    )


def _ifaces_for_serial(serial: str) -> list[str]:
    """Every live CAN interface belonging to ``serial``, ordered by dev_id."""
    found: list[tuple[int, str]] = []
    for iface_path in Path("/sys/class/net").glob("can*"):
        info = subprocess.run(
            ["udevadm", "info", "-a", "-p", str(iface_path)],
            capture_output=True,
            text=True,
        ).stdout
        if _udev_attr(info, "ATTRS{serial}") != serial:
            continue
        try:
            dev_id = int(_udev_attr(info, "ATTR{dev_id}"), 16)
        except ValueError:
            continue
        found.append((dev_id, iface_path.name))
    return [name for _, name in sorted(found)]


def _identify_dual_adapter(serial: str) -> str | None:
    """Probe a dual-channel hub: ``"axol"``, ``"mantis"``, or ``None``.

    Both products use identical USB hardware, so identity comes from the CAN
    devices behind it. Silence leaves the decision to the operator instead of
    turning an unpowered Mantis into an arm hub (or vice versa).
    """
    ifaces = _ifaces_for_serial(serial)
    if len(ifaces) < 2:
        return None
    try:
        bring_up_interfaces(ifaces)
    except RuntimeError:
        return None

    mantis = any(_probe_mantis_trigger(iface) for iface in ifaces)
    axol = any(_probe_axol_shoulder(iface) for iface in ifaces)
    if mantis and axol:
        print(
            f"  WARNING: both a Mantis trigger and an Axol shoulder answered "
            f"behind {serial}; refusing to guess."
        )
        return None
    if mantis:
        return "mantis"
    if axol:
        return "axol"
    return None


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
    devices unpowered, or unrelated hardware — stays None and is left to the
    operator.
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


def _dual_channel_rules(serial: str, profile: _Profile) -> str:
    """udev rules block for a profile's dual-channel adapter."""
    return (
        f"# {profile.label} dual-channel CAN adapter\n"
        f"# Adapter serial: {serial}\n"
        f"# Channel 0 -> left\n"
        f'SUBSYSTEM=="net", ACTION=="add", ATTRS{{idVendor}}=="{_VID}", ATTRS{{idProduct}}=="{_PID}", ATTRS{{serial}}=="{serial}", ATTR{{dev_id}}=="0x0", NAME="{profile.left}"\n'
        f"# Channel 1 -> right\n"
        f'SUBSYSTEM=="net", ACTION=="add", ATTRS{{idVendor}}=="{_VID}", ATTRS{{idProduct}}=="{_PID}", ATTRS{{serial}}=="{serial}", ATTR{{dev_id}}=="0x1", NAME="{profile.right}"\n'
        f"# Every (re-)enumeration — boot or a mid-session USB drop — pulls in\n"
        f"# the bring-up service so the channels come back configured and up.\n"
        f"# Tagged on the USB device rather than the net interfaces: the NAME=\n"
        f"# rules above put every real hotplug add mid-rename, and systemd\n"
        f'# skips SYSTEMD_WANTS on renaming devices ("device is renaming").\n'
        f'SUBSYSTEM=="usb", ENV{{DEVTYPE}}=="usb_device", ACTION=="add", ATTR{{idVendor}}=="{_VID}", ATTR{{idProduct}}=="{_PID}", ATTR{{serial}}=="{serial}", TAG+="systemd", ENV{{SYSTEMD_WANTS}}+="{profile.hotplug_unit}"\n'
    )


def _write_udev_rules(
    hub_serial: str | None,
    wheels_serial: str | None = None,
    chest_serial: str | None = None,
    profile: _Profile = _AXOL_PROFILE,
) -> None:
    print(f"Writing udev rules to {profile.rules_file} (requires sudo)...")
    content = ""
    if hub_serial:
        content += _dual_channel_rules(hub_serial, profile)
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
            f'SUBSYSTEM=="usb", ENV{{DEVTYPE}}=="usb_device", ACTION=="add", ATTR{{serial}}=="{serial}", TAG+="systemd", ENV{{SYSTEMD_WANTS}}+="{profile.hotplug_unit}"\n'
        )
    run_root(["tee", str(profile.rules_file)], input_text=content, check=True)
    print("  Done.")


def _reload_udev() -> None:
    print("Reloading udev rules (requires sudo)...")
    run_root(["udevadm", "control", "--reload-rules"], check=True)
    run_root(["systemctl", "restart", "systemd-udevd"], check=True)
    print("  Done.")


def _rename_interfaces(
    hub_serial: str | None,
    wheels_serial: str | None = None,
    chest_serial: str | None = None,
    profile: _Profile = _AXOL_PROFILE,
) -> None:
    """Rename existing canX interfaces to their target names without replug."""
    print("Renaming CAN interfaces (requires sudo)...")
    # (adapter serial, channel dev_id) -> persistent name. The wheel/chest
    # adapters are single-channel, so their only interface is dev_id 0.
    target: dict[tuple[str, int], str] = {}
    if hub_serial:
        target[(hub_serial, 0)] = profile.left
        target[(hub_serial, 1)] = profile.right
    if wheels_serial:
        target[(wheels_serial, 0)] = _CAN_B
    if chest_serial:
        target[(chest_serial, 0)] = _CAN_C

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


def _write_cron_script(profile: _Profile = _AXOL_PROFILE) -> None:
    """Write a profile's bring-up script.

    On the robot profile the script also covers the wheel and chest buses;
    every interface is optional and checked for presence at runtime, so one
    script serves every hardware combination — arm-only, cart-only, chest-
    only, or all of them — and an unplugged adapter never blocks the rest.
    """
    print(f"Writing CAN startup script to {profile.cron_script}...")
    _CAN_DIR.mkdir(parents=True, exist_ok=True)
    script = (
        f"#!/bin/bash\n"
        f"# Bring up {profile.label} CAN interfaces\n"
        f"#\n"
        f"# Runs at boot (@reboot root crontab) and on every (re-)enumeration\n"
        f"# of an adapter (udev -> {profile.hotplug_unit}), so a mid-session USB\n"
        f"# drop of the hub comes back configured without operator action.\n"
        f"#\n"
        f"# The left/right interfaces are channels of one dual-channel gs_usb\n"
        f"# adapter. Bring them down together, configure, then up together —\n"
        f"# flapping the channels one at a time (down/up L, then down/up R)\n"
        f"# toggles the adapter into a state where TX works but no RX frame is\n"
        f"# delivered. Skipped entirely when the adapter is unplugged (other\n"
        f"# buses below must still come up).\n"
        f"set -euo pipefail\n\n"
        f"# Boot and hotplug triggers can race (the hub's two channels fire one\n"
        f"# udev add event each) — serialize whole runs.\n"
        f'exec 9>"/run/lock/{profile.lock_name}"\n'
        f"flock 9\n\n"
        f"# The two channels enumerate a beat apart, so the trigger for the\n"
        f"# first can run before the second exists. Give the pair a moment.\n"
        f"for _ in $(seq 1 30); do\n"
        f"    if ip link show {profile.left} >/dev/null 2>&1 "
        f"&& ip link show {profile.right} >/dev/null 2>&1; then\n"
        f"        break\n"
        f"    fi\n"
        f"    sleep 0.1\n"
        f"done\n\n"
        f"if ip link show {profile.left} >/dev/null 2>&1 "
        f"&& ip link show {profile.right} >/dev/null 2>&1; then\n"
        f"    for IFACE in {profile.left} {profile.right}; do\n"
        f'        ip link set "${{IFACE}}" down 2>/dev/null || true\n'
        f"    done\n"
        f"    for IFACE in {profile.left} {profile.right}; do\n"
        f'        ip link set "${{IFACE}}" type can bitrate {_BITRATE}\n'
        f'        ip link set "${{IFACE}}" txqueuelen {_TXQUEUELEN}\n'
        f"    done\n"
        f"    for IFACE in {profile.left} {profile.right}; do\n"
        f'        ip link set "${{IFACE}}" up\n'
        f"    done\n"
        f"else\n"
        f'    echo "{profile.left}/{profile.right} not present — skipping bring-up"\n'
        f"fi\n"
    )
    if profile is _AXOL_PROFILE:
        script += (
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
    profile.cron_script.write_text(script)
    profile.cron_script.chmod(0o755)
    print("  Done.")


def _register_cron(profile: _Profile = _AXOL_PROFILE) -> None:
    print("Registering @reboot cron entry in root crontab (requires sudo)...")
    cron_entry = f"@reboot {profile.cron_script}"
    existing = run_root(["crontab", "-l"]).stdout or ""
    if str(profile.cron_script) in existing:
        print("  Entry already present — skipping.")
    else:
        new_crontab = existing.rstrip("\n") + "\n" + cron_entry + "\n"
        run_root(["crontab", "-"], input_text=new_crontab, check=True)
        print(f"  Added: {cron_entry}")


def _remove_pre_mantis_config() -> None:
    """Remove superseded Mantis rule/script/unit names during migration."""
    old_paths = (_PRE_MANTIS_RULES_FILE, _PRE_MANTIS_HOTPLUG_UNIT_FILE)
    if _PRE_MANTIS_HOTPLUG_UNIT_FILE.exists():
        run_root(["systemctl", "disable", _PRE_MANTIS_HOTPLUG_UNIT], check=False)
    for path in old_paths:
        if path.exists():
            run_root(["rm", "-f", str(path)], check=True)

    if _PRE_MANTIS_CRON_SCRIPT.exists():
        _PRE_MANTIS_CRON_SCRIPT.unlink()
    existing = run_root(["crontab", "-l"]).stdout or ""
    kept = [
        line
        for line in existing.splitlines()
        if str(_PRE_MANTIS_CRON_SCRIPT) not in line
    ]
    if len(kept) != len(existing.splitlines()):
        new_crontab = "\n".join(kept).rstrip("\n") + "\n"
        run_root(["crontab", "-"], input_text=new_crontab, check=True)


def _write_hotplug_unit(profile: _Profile = _AXOL_PROFILE) -> None:
    """Install the systemd unit the udev rules pull in on adapter hotplug.

    udev tags the adapter's USB device with ``SYSTEMD_WANTS=<hotplug unit>``
    (see :func:`_write_udev_rules`), so every (re-)enumeration — boot or a
    mid-session USB drop — runs the startup script and the interfaces come
    back configured and up within a second, no operator action needed. The
    Mantis rig gets its own unit: the handheld is unplugged far more often
    than the arm hub.
    """
    print(
        f"Writing hotplug bring-up unit to {profile.hotplug_unit_file} "
        "(requires sudo)..."
    )
    content = (
        f"# Installed by `axol can.setup`. Pulled in by {profile.rules_file}\n"
        f"# whenever the adapter (re-)enumerates: a mid-session USB drop\n"
        f"# recreates the interfaces down and unconfigured, and this\n"
        f"# service brings them back up without operator action.\n"
        f"[Unit]\n"
        f"Description=Bring up {profile.label} CAN interfaces on adapter hotplug\n"
        f"\n"
        f"[Service]\n"
        f"Type=oneshot\n"
        f"ExecStart=/bin/bash {profile.cron_script}\n"
    )
    run_root(["tee", str(profile.hotplug_unit_file)], input_text=content, check=True)
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
    parser = subparsers.add_parser(
        "can.setup",
        help="Auto-detect and configure Axol/Mantis CAN interfaces.",
    )
    parser.set_defaults(func=run)


def rx_alive_per_arm(profile: _Profile = _AXOL_PROFILE) -> tuple[bool, bool]:
    """(left, right) — True where a motor answers on that side's bus.

    Verifies the adapter's receive path, not just the interface state: the
    dual-channel gs_usb adapter can come out of a down/up cycle in a state
    where TX still works but no received frame is ever delivered (kernel-side
    everything looks healthy — UP, ERROR-ACTIVE, correct bitrate). Probes the
    profile's ``probe_joint`` — the shoulder on the robot arm, the gripper on
    the Mantis (its buses carry nothing else).

    Per-side results matter because one healthy side must not mask the other:
    a bus with no responding motors (arm powered off, harness fault, dead
    transceiver channel) looks identical to the adapter wedge on that side.
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

    async def probe_all() -> tuple[bool, bool]:
        await asyncio.sleep(0.5)  # let the freshly-upped interfaces settle
        left, right = await asyncio.gather(probe(profile.left), probe(profile.right))
        return left, right

    return asyncio.run(probe_all())


def rx_alive(profile: _Profile = _AXOL_PROFILE) -> bool:
    """True when at least one motor answers on either side."""
    return any(rx_alive_per_arm(profile))


def bring_up_can(profile: _Profile = _AXOL_PROFILE) -> None:
    """Run the bring-up script, then verify RX and re-flap once if it's dead.

    Every down/up cycle of the adapter's channels toggles it between a healthy
    state and the TX-only wedge described in :func:`rx_alive_per_arm`, so a
    bring-up that lands in the wedge is recovered by exactly one more cycle.
    A robot with its motors powered off is indistinguishable from the wedge,
    hence the bounded retries and the warning instead of an error. Results are
    reported per side: one answering side proves the adapter is healthy (no
    retry), while the other side staying silent is called out instead of being
    masked by it.
    """
    print("Bringing up CAN interfaces (requires sudo)...")
    noun = "arm" if profile is _AXOL_PROFILE else "gripper"
    adapter_present = (Path("/sys/class/net") / profile.left).exists() and (
        Path("/sys/class/net") / profile.right
    ).exists()
    if not adapter_present:
        # Adapter-less host state: the script still brings up the wheel/chest
        # buses; the RX-wedge probe/re-flap is adapter-specific, so there's
        # nothing to verify.
        run_root(["bash", str(profile.cron_script)], check=True)
        print(f"  Done — {profile.label} not attached, its interfaces skipped.")
        return
    for attempt in range(3):
        run_root(["bash", str(profile.cron_script)], check=True)
        left, right = rx_alive_per_arm(profile)
        if left and right:
            print(f"  Done — motors responding on both {noun}s.")
            return
        if left or right:
            # One side answering proves the adapter's RX path is healthy, so
            # more cycling can't help the silent side — that's a device-level
            # problem (power, harness, or the adapter's transceiver channel).
            alive, silent = ("left", "right") if left else ("right", "left")
            print(f"  Done — motors responding on the {alive} {noun}.")
            print(
                f"  WARNING: no motor answered on the {silent} {noun}. Check "
                f"that it is powered and its CAN connection is intact."
            )
            return
        if attempt < 2:
            print("  No motor responses (adapter RX may be wedged) — cycling again...")
    print(
        "  WARNING: no motor responded after bring-up. If the device is powered "
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
    already-named interfaces up. Refers only to the robot-arm profile because
    it serves the control panel's idle robot connection; interactive
    ``can.setup`` discovers and configures the Mantis profile independently.
    """
    return (
        _AXOL_PROFILE.rules_file.exists()
        and _AXOL_PROFILE.cron_script.exists()
        and _AXOL_PROFILE.hotplug_unit_file.exists()
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


def _configure_mantis(serial: str) -> None:
    """Write the Mantis rig's persistent config and bring its buses up.

    Same dual-channel board as the arm hub: channel 0 -> left gripper,
    channel 1 -> right. No wheel/chest/RP1 handling on the rig profile.
    """
    _remove_pre_mantis_config()
    _write_udev_rules(serial, profile=_MANTIS_PROFILE)
    _write_cron_script(_MANTIS_PROFILE)
    _write_hotplug_unit(_MANTIS_PROFILE)
    _reload_udev()
    _rename_interfaces(serial, profile=_MANTIS_PROFILE)
    _register_cron(_MANTIS_PROFILE)
    bring_up_can(_MANTIS_PROFILE)


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
    the interactive flow's probing — see :func:`_identify_adapter`. This helper
    configures the robot-arm profile only; the interactive ``can.setup`` flow
    discovers both Axol and Mantis hubs.
    """
    driver.ensure_driver()
    hub_serial = hub_serial or _resolve_hub_serial()
    wheels_serial = wheels_serial or _configured_named_serial(_CAN_B)
    chest_serial = chest_serial or _configured_named_serial(_CAN_C)
    if not (hub_serial or wheels_serial or chest_serial):
        raise RuntimeError("Robot not detected")
    _apply_setup(hub_serial, wheels_serial, chest_serial)


def ensure_mantis_setup() -> None:
    """Configure a detected Mantis hub non-interactively for the web panel.

    A previously pinned serial wins. Otherwise every unattached dual-channel
    hub is probed and exactly one Mantis response must be found; silent or
    ambiguous adapters are left for interactive :func:`run` instead of being
    guessed. This is the Mantis counterpart to :func:`ensure_setup` used by
    the idle diagnostics link.
    """
    installed = driver.ensure_driver()
    if installed:
        time.sleep(2.0)

    serial = _configured_serial(_MANTIS_PROFILE)
    if serial is None:
        axol_serial = _configured_serial(_AXOL_PROFILE)
        matches = [
            candidate
            for candidate in _detect_serials()
            if candidate != axol_serial
            and _identify_dual_adapter(candidate) == "mantis"
        ]
        if len(matches) != 1:
            if not matches:
                raise RuntimeError(
                    "Mantis not detected — power the grippers and triggers, "
                    "then run `axol can.setup`"
                )
            raise RuntimeError(
                "Multiple Mantis hubs detected — run `axol can.setup` to assign them"
            )
        serial = matches[0]
    _configure_mantis(serial)


def _find_single_serials(hub_serial: str | None) -> tuple[str | None, str | None]:
    """Interactively assign single-channel adapters to the wheel/chest buses.

    Previously pinned adapters are kept without prompting. Every other
    attached single-channel adapter is identified by probing its bus (see
    :func:`_identify_adapter`); one where nothing answers — devices
    unpowered, or unrelated hardware — is offered to the operator instead of
    guessed at. Serials the Mantis rig's rules already claim are excluded
    outright (a belt-and-braces guard; the rig's board is dual-channel, so it
    should never appear here).

    Returns ``(wheels_serial, chest_serial)``, either of which may be None.
    """
    wheels = _configured_named_serial(_CAN_B)
    chest = _configured_named_serial(_CAN_C)
    if wheels:
        print(f"Cart wheel bus: keeping configured adapter (serial {wheels}).")
    if chest:
        print(f"Chest bus: keeping configured adapter (serial {chest}).")

    exclude = {s for s in (hub_serial, wheels, chest) if s}
    exclude |= _mantis_claimed_serials()
    candidates = _detect_single_serials(exclude)
    if not candidates:
        return wheels, chest

    print(
        f"Identifying {len(candidates)} single-channel CAN adapter(s) by "
        f"probing (wheel motors / jelly_legs board must be powered)..."
    )
    unidentified: list[str] = []
    for serial in candidates:
        role = _identify_adapter(serial)
        if role == "wheels" and wheels is None:
            wheels = serial
            print(f"  {serial}: Damiao wheel motors answered -> {_CAN_B}")
        elif role == "chest" and chest is None:
            chest = serial
            print(f"  {serial}: jelly_legs board answered -> {_CAN_C}")
        elif role is not None:
            print(
                f"  {serial}: identified as the {role} bus, but that bus is "
                f"already pinned to another adapter — skipping."
            )
        else:
            unidentified.append(serial)

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
        if choice == "w" and wheels is None:
            wheels = serial
        elif choice == "c" and chest is None:
            chest = serial
        elif choice in ("w", "c"):
            print("    That bus is already assigned — skipping.")
    return wheels, chest


def run(args: object = None) -> None:
    """Configure persistent CAN interfaces and a @reboot bring-up entry."""
    installed = driver.ensure_driver()
    if installed:
        # The freshly-loaded driver may claim adapters the old one ignored
        # (CANable 2.0); give their interfaces a moment to appear.
        time.sleep(2.0)

    hub_serial, mantis_serial = _find_dual_serials()
    wheels_serial, chest_serial = _find_single_serials(hub_serial)
    if not (hub_serial or mantis_serial or wheels_serial or chest_serial):
        _die(
            "No CAN adapters found or configured. Connect the Axol/Mantis hub, "
            "wheel-bus, or chest adapter and re-run."
        )
    if hub_serial or wheels_serial or chest_serial:
        _apply_setup(hub_serial, wheels_serial, chest_serial)
    if mantis_serial:
        _configure_mantis(mantis_serial)

    print()
    print("Setup complete.")
    if hub_serial:
        print(f"  Left arm : {_CAN_L}")
        print(f"  Right arm: {_CAN_R}")
    if wheels_serial:
        print(f"  Wheels   : {_CAN_B}")
    if chest_serial:
        print(f"  Chest    : {_CAN_C} (jelly_legs lift)")
    if hub_serial or wheels_serial or chest_serial:
        print(f"  Startup  : {_CRON_SCRIPT} (runs at @reboot via root crontab)")
        print(
            f"  Hotplug  : {_AXOL_PROFILE.hotplug_unit} (re-runs the startup script "
            "whenever an adapter re-enumerates, e.g. after a mid-session USB drop)"
        )
        if _is_raspberry_pi_5():
            print(f"  Pi 5     : {_RP1_QUIRK_UNIT} (RP1 USB EMI-tolerance quirk)")
    if mantis_serial:
        print(f"  Mantis L : {_MANTIS_PROFILE.left}")
        print(f"  Mantis R : {_MANTIS_PROFILE.right}")
        print(
            f"  Mantis startup : {_MANTIS_PROFILE.cron_script} "
            "(runs at @reboot via root crontab)"
        )
