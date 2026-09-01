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
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from ...constants import (
    CAN_BASE,
    CAN_BRINGUP_SCRIPT,
    CAN_CHEST,
    CAN_LEFT,
    CAN_MANTIS_LEFT,
    CAN_MANTIS_RIGHT,
    CAN_RIGHT,
    Joint,
)
from ...tracker.trigger import TRIGGER_CAN_ID, decode_trigger_payload
from ...utils.paths import almond_path
from ...utils.sudo import run_root
from . import driver

_VID = "1d50"
_PID = "606f"
_USB_DEVICES = Path("/sys/bus/usb/devices")
_HUB_ENUMERATION_TIMEOUT_S = 2.0
_HUB_ENUMERATION_POLL_S = 0.05
_CAN_L = CAN_LEFT
_CAN_R = CAN_RIGHT
_CAN_B = CAN_BASE
_CAN_C = CAN_CHEST
_BITRATE = 1_000_000
_TXQUEUELEN = 512

_UDEV_RULES_FILE = Path("/etc/udev/rules.d/90-can.rules")
_CAN_DIR = CAN_BRINGUP_SCRIPT.parent
_CRON_SCRIPT = CAN_BRINGUP_SCRIPT
_LEGACY_CAN_DIRS = {almond_path("can"), Path("/root/.almond/can")}


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


def _dual_channel_rule_serial(
    rules_file: Path, left_name: str, right_name: str
) -> str | None:
    """Serial claimed by one complete, generated dual-channel rule block.

    This is intentionally narrower than :func:`_rules_serial_for`: load-time
    hardware discovery may use it as authority while the driver is unloaded
    and no netdev exists yet, so both expected names, VID/PID, and dev IDs must
    agree on one serial. Partial or hand-written rules fail closed.
    """
    try:
        lines = rules_file.read_text().splitlines()
    except OSError:
        return None

    serials: list[str] = []
    for dev_id, name in (("0x0", left_name), ("0x1", right_name)):
        matches: list[str] = []
        generated_rule = re.compile(
            r'\s*SUBSYSTEM=="net",\s*ACTION=="add",\s*'
            rf'ATTRS\{{idVendor\}}=="{re.escape(_VID)}",\s*'
            rf'ATTRS\{{idProduct\}}=="{re.escape(_PID)}",\s*'
            r'ATTRS\{serial\}=="([^"\r\n]+)",\s*'
            rf'ATTR\{{dev_id\}}=="{re.escape(dev_id)}",\s*'
            rf'NAME="{re.escape(name)}"\s*'
        )
        for line in lines:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            match = generated_rule.fullmatch(line)
            if match is not None:
                matches.append(match.group(1))
        if len(matches) != 1:
            return None
        serials.append(matches[0])
    return serials[0] if serials[0] == serials[1] else None


def _configured_profile_usb_serials() -> dict[str, str | None]:
    """Strict persisted USB identity for each dual-channel hardware profile."""
    axol = _dual_channel_rule_serial(
        _AXOL_PROFILE.rules_file, _AXOL_PROFILE.left, _AXOL_PROFILE.right
    )
    mantis = _dual_channel_rule_serial(
        _MANTIS_PROFILE.rules_file, _MANTIS_PROFILE.left, _MANTIS_PROFILE.right
    )
    if mantis is None and not _MANTIS_PROFILE.rules_file.exists():
        # A legacy Mantis install is still an explicit operator-confirmed role.
        # Its next normal setup migrates the retired names to the current ones.
        mantis = _dual_channel_rule_serial(
            _PRE_MANTIS_RULES_FILE, _PRE_MANTIS_LEFT, _PRE_MANTIS_RIGHT
        )
    return {"axol": axol, "mantis": mantis}


def _attached_supported_usb_serials() -> set[str]:
    """Attached supported hub serials, including before ``gs_usb`` is loaded."""
    serials: set[str] = set()
    for vendor_file in _USB_DEVICES.glob("*/idVendor"):
        device = vendor_file.parent
        try:
            vendor = vendor_file.read_text().strip().lower()
            product = device.joinpath("idProduct").read_text().strip().lower()
            serial = device.joinpath("serial").read_text().strip()
        except OSError:
            continue
        if (vendor, product) == (_VID, _PID) and serial:
            serials.add(serial)
    return serials


def _attached_configured_hub_serials() -> dict[str, str]:
    """Exact persisted profile claims whose supported USB device is attached."""
    claims = _configured_profile_usb_serials()
    attached = _attached_supported_usb_serials()
    conflicts = {
        serial
        for serial in claims.values()
        if serial is not None and list(claims.values()).count(serial) > 1
    }
    return {
        profile: serial
        for profile, serial in claims.items()
        if serial is not None and serial in attached and serial not in conflicts
    }


def attached_configured_hub_profiles() -> set[str]:
    """Profiles whose exact persisted dual-channel USB hub is attached.

    This supports hosted ``axol serve`` recovery before the CAN driver has
    created or renamed netdevs. A raw ``can0`` is never classified. If stale
    rules claim one serial for both Axol and Mantis, neither claim is trusted.
    """
    return set(_attached_configured_hub_serials())


def _wait_for_dual_channel_serial(
    serial: str,
    *,
    timeout: float = _HUB_ENUMERATION_TIMEOUT_S,
    poll_interval: float = _HUB_ENUMERATION_POLL_S,
) -> bool:
    """Wait boundedly for both gs_usb netdevs of one attached hub serial.

    ``modprobe`` may return before the USB probe and udev add events have
    exposed both channels. Polling the exact persisted serial avoids a blind
    delay and, crucially, cannot mistake another newly-enumerated hub for the
    requested Axol/Mantis profile.
    """
    deadline = time.monotonic() + timeout
    while True:
        adapter = _scan_adapters().get(serial)
        if (
            adapter is not None
            and (adapter.get("vid"), adapter.get("pid")) == (_VID, _PID)
            and {0, 1} <= set(adapter.get("dev_ids", set()))
        ):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(poll_interval, remaining))


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

    Every attached hub is probed, including hubs already pinned by udev rules.
    A positive device response overrides a stale pin. If a configured hub is
    still silent after bounded RX recovery, the normal interactive flow offers
    Axol/Mantis assignment with Enter preserving its previous role. New silent
    hardware is likewise presented. Returns ``(axol_serial, mantis_serial)``.
    """
    print(f"Scanning for dual-channel CAN adapters ({_VID}:{_PID})...")
    attached = set(_detect_serials())
    # Role-specific scans deliberately exclude other CAN topologies. Keep the
    # full set too so a serial that now enumerates as (say) a wheel adapter is
    # not retained as an allegedly unplugged Axol hub as well.
    all_attached = set(_scan_adapters()) | attached
    configured_axol = _configured_serial(_AXOL_PROFILE)
    configured_mantis = _configured_serial(_MANTIS_PROFILE)

    # Probe configured adapters too. Otherwise an adapter incorrectly pinned
    # as Axol once is excluded forever and can.setup never sees its Mantis
    # trigger, so every subsequent run preserves the wrong interface names.
    roles: dict[str, str | None] = {}
    for serial in sorted(attached):
        print(f"  {serial}: probing Mantis trigger / Axol shoulder...")
        roles[serial] = _identify_dual_adapter(serial, reset=True)

    def detected_for(role: str, configured: str | None) -> str | None:
        matches = sorted(serial for serial, found in roles.items() if found == role)
        if not matches:
            return None
        selected = configured if configured in matches else matches[0]
        label = "Axol shoulder" if role == "axol" else "Mantis trigger"
        print(f"  {selected}: {label} answered -> {role}")
        for serial in matches:
            if serial != selected:
                print(
                    f"  {serial}: also identified as {role}, but that role is "
                    "already assigned — skipping."
                )
        return selected

    axol = detected_for("axol", configured_axol)
    mantis = detected_for("mantis", configured_mantis)

    unidentified = [
        serial
        for serial, role in sorted(roles.items())
        if role is None and serial not in (axol, mantis)
    ]
    for serial in unidentified:
        print(f"  {serial}: no identifying device answered.")
        previous_roles = [
            role
            for role, configured in (
                ("axol", configured_axol),
                ("mantis", configured_mantis),
            )
            if serial == configured
        ]
        if len(previous_roles) == 1:
            previous = previous_roles[0]
            prompt = (
                f"    Assign it to the [a]xol arm hub or [m]antis rig "
                f"([Enter] keeps {previous}): "
            )
        elif len(previous_roles) > 1:
            previous = None
            prompt = (
                "    It is currently assigned to both products. Assign it to "
                "the [a]xol arm hub or [m]antis rig (blank skips): "
            )
        else:
            previous = None
            prompt = (
                "    Assign it to the [a]xol arm hub, [m]antis rig, "
                "or leave blank to skip: "
            )
        choice = input(prompt).strip().lower()
        if not choice and previous is not None:
            choice = previous[0]
        if choice == "a" and axol is None:
            axol = serial
        elif choice == "m" and mantis is None:
            mantis = serial
        elif choice in ("a", "m"):
            print("    That role is already assigned — skipping.")

    # If no positive probe or explicit operator choice replaced an unplugged
    # configured role, keep its pin. In particular, leaving a different silent
    # adapter blank means "skip this adapter", not "forget the hub that is
    # temporarily unplugged". Otherwise a configured wheel/chest bus later in
    # run() would cause _apply_setup() to rewrite the rules without that hub.
    for role, configured in (
        ("axol", configured_axol),
        ("mantis", configured_mantis),
    ):
        current = axol if role == "axol" else mantis
        other = mantis if role == "axol" else axol
        if (
            current is None
            and configured is not None
            and configured not in all_attached
            and configured != other
        ):
            if role == "axol":
                axol = configured
            else:
                mantis = configured
            print(
                f"  {role.capitalize()}: adapter {configured} is not attached; "
                "keeping its configured assignment."
            )
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
# a ``<fH`` core (normalised position + 12-bit raw ADC), optionally followed
# by one opaque byte, at 100 Hz on 0x009. The arm fallback asks its
# MyActuator shoulder-1 (ID 0x01) for status frame 1.
_AXOL_SHOULDER_REQ_ID = 0x141
_AXOL_SHOULDER_RESP_ID = 0x241
_MYACTUATOR_STATUS1 = 0x9A

# Linux ORs these flags into ``can_frame.can_id``. Identity probes only accept
# standard 11-bit data frames: masking first would let an extended, RTR, or
# error frame whose low bits happen to match a device ID create a false role.
_CAN_FRAME_TYPE_FLAGS = 0xE0000000
_CAN_SFF_MASK = 0x7FF


def _standard_data_can_id(raw_can_id: int) -> int | None:
    """Return an 11-bit data-frame ID, rejecting EFF/RTR/error frames."""
    if raw_can_id & _CAN_FRAME_TYPE_FLAGS:
        return None
    return raw_can_id & _CAN_SFF_MASK


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
                raw_can_id, dlc = struct.unpack("<IB3x", frame[:8])
                can_id = _standard_data_can_id(raw_can_id)
                if can_id is not None and match(can_id, frame[8 : 8 + dlc]):
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
        if can_id != TRIGGER_CAN_ID:
            return False
        decoded = decode_trigger_payload(data)
        if decoded is None:
            return False
        position, raw = decoded
        # Validate the otherwise-unused ADC field too so unrelated traffic
        # sharing 0x009 cannot identify a hub.
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


def _identify_dual_adapter(serial: str, *, reset: bool = False) -> str | None:
    """Probe a dual-channel hub: ``"axol"``, ``"mantis"``, or ``None``.

    Both products use identical USB hardware, so identity comes from the CAN
    devices behind it. Silence leaves the decision to the operator instead of
    turning an unpowered Mantis into an arm hub (or vice versa). Explicit
    ``can.setup`` passes ``reset=True`` so every attached supported hub is
    pair-reset before identification; runtime callers can retain a healthy
    first pass while still receiving bounded recovery when it is silent.
    """
    ifaces = _ifaces_for_serial(serial)
    if len(ifaces) < 2:
        return None
    try:
        # An UP gs_usb interface can still be TX-only with wedged RX. Preserve
        # a healthy first pass, but if either half is down recover the adapter
        # pair together before probing. Pair-wide ordering matters for this
        # dual-channel firmware: both channels go down before either comes up.
        bring_up_interfaces(
            ifaces,
            force_cycle=reset or not all(iface_up(iface) for iface in ifaces),
        )
    except RuntimeError:
        return None

    # Initial probe plus two bounded pair recoveries. A powered Mantis/Axol
    # therefore corrects a stale udev role during ordinary ``can.setup``;
    # genuinely unpowered identical USB hardware still falls back to the
    # operator instead of being guessed.
    for attempt in range(3):
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
        if attempt < 2:
            print(
                "    No identity response; recovering both CAN channels and retrying..."
            )
            try:
                bring_up_interfaces(ifaces, force_cycle=True)
            except RuntimeError:
                return None
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


def _identify_adapter(
    serial: str, *, reset: bool = False, recover_silence: bool = True
) -> str | None:
    """Probe a single-channel adapter's bus: ``"wheels"``, ``"chest"``, or None.

    Explicit ``can.setup`` passes ``reset=True`` for a previously identified
    wheel/cart or chest/lift adapter. Unknown generic gs_usb devices get a
    non-disruptive first probe with ``recover_silence=False``; a positive match
    is reset later during final setup, while an unrelated device is not flapped.
    """
    iface = _iface_for_serial(serial)
    if iface is None:
        return None
    was_up = iface_up(iface)
    try:
        bring_up_interfaces([iface], force_cycle=reset or not was_up)
    except RuntimeError:
        return None

    attempts = 3 if recover_silence else 1
    for attempt in range(attempts):
        # Silence any jelly_legs board before the wheel probe: the board starts
        # its 50 ms broadcast after the first frame it sees (the wheel probe's
        # own Damiao reads would wake it), and that stream starves the CANable's
        # TX path — on a combined bus the wheel probe would then go deaf and the
        # bus would be misclassified as chest-only. Harmless where no board is
        # listening; a frame queued on a dead bus is dropped by the reset.
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
        if attempt < attempts - 1:
            print(f"    No wheel/cart response on {iface}; resetting and retrying...")
            try:
                bring_up_interfaces([iface], force_cycle=True)
            except RuntimeError:
                return None
    if not recover_silence and not was_up:
        # Probing needed the unknown interface UP, but silence means it may be
        # unrelated to Axol. Restore the administrative state we found.
        run_root(["ip", "link", "set", iface, "down"], check=False)
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


def _interface_rename_plan(
    records: list[tuple[str, str, int]],
    target: dict[tuple[str, int], str],
    *,
    managed_names: set[str] | None = None,
    reserved_names: set[str] | None = None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return collision-free ``(temporary, final)`` interface rename stages.

    ``records`` contains ``(current_name, USB serial, dev_id)``. Every source
    that needs a new final name is first moved to a short temporary name. Any
    *other* interface currently occupying one of those final names is staged
    too, which is what lets two attached hubs recover when their stale
    Axol/Mantis assignments are exactly swapped.

    The second stage includes only interfaces claimed by ``target``. An
    unclaimed stale occupant remains under its temporary name until a later
    profile pass gives it the right final name (or until it is replugged).
    ``managed_names`` also evicts names whose role was removed entirely, and
    ``reserved_names`` prevents temporary names colliding with unrelated live
    network interfaces.
    """
    by_name = {name: (serial, dev_id) for name, serial, dev_id in records}
    final_by_source = {
        name: target[(serial, dev_id)]
        for name, serial, dev_id in records
        if (serial, dev_id) in target and target[(serial, dev_id)] != name
    }
    participants = set(final_by_source)
    participants.update(
        final
        for final in final_by_source.values()
        if final in by_name and final not in final_by_source
    )
    for name in managed_names or set():
        if name in by_name and target.get(by_name[name]) != name:
            participants.add(name)
    if not participants:
        return [], []

    # Linux IFNAMSIZ leaves 15 visible characters. Keep the generated names
    # well below that and avoid every live/final name so a partial run remains
    # diagnosable and the second profile pass can still discover ``can*``.
    used = set(by_name) | set(target.values()) | (reserved_names or set())
    temporary_by_source: dict[str, str] = {}
    next_index = 0
    for source in sorted(participants):
        while True:
            temporary = f"can_tmp{next_index}"
            next_index += 1
            if temporary not in used:
                break
        temporary_by_source[source] = temporary
        used.add(temporary)

    temporary_stage = [
        (source, temporary_by_source[source]) for source in sorted(participants)
    ]
    final_stage = [
        (temporary_by_source[source], final)
        for source, final in sorted(final_by_source.items())
    ]
    return temporary_stage, final_stage


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
    profile: _Profile = _AXOL_PROFILE,
) -> None:
    """Rename existing CAN interfaces to their target names without replug.

    All participants are staged through temporary names first. This recovers a
    swapped Axol/Mantis pin or wheel/chest pin even while each desired final
    name is occupied by the other adapter. Stale managed-name occupants are
    evicted even when their replacement is absent.
    """
    print("Renaming CAN interfaces (requires sudo)...")
    _validate_adapter_assignments(hub_serial, wheels_serial, chest_serial)

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

    net_dir = Path("/sys/class/net")
    records: list[tuple[str, str, int]] = []
    for iface_path in list(net_dir.glob("can*")):
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
        records.append((iface, iface_serial, dev_id))

    by_name = {name: (serial, dev_id) for name, serial, dev_id in records}
    existing_names = {path.name for path in net_dir.iterdir()}

    # Refuse to mutate anything if a requested destination is occupied by a
    # network interface that is not one of the serial-bearing CAN adapters we
    # scanned. This catches reserved-name collisions before the all-down phase.
    for destination in set(target.values()):
        if destination in existing_names and destination not in by_name:
            raise RuntimeError(
                f"Cannot rename a CAN interface to {destination}: that name "
                "is already used by an unrelated network interface."
            )

    # A Mantis setup must never evict the Axol profile (or its wheel/chest
    # buses), and vice versa. Wheel/chest names belong only to the Axol pass.
    managed_names = {profile.left, profile.right}
    if profile == _AXOL_PROFILE:
        managed_names.update({_CAN_B, _CAN_C})

    temporary_stage, final_stage = _interface_rename_plan(
        records,
        target,
        managed_names=managed_names,
        reserved_names=existing_names,
    )
    final_by_temporary = dict(final_stage)
    for source, temporary in temporary_stage:
        final = final_by_temporary.get(temporary)
        if final is None:
            print(f"  {source} -> {temporary} (clearing a stale occupied name)")
        else:
            print(f"  {source} -> {final}")
        run_root(["ip", "link", "set", source, "down"], check=True)

    for source, temporary in temporary_stage:
        run_root(["ip", "link", "set", source, "name", temporary], check=True)
    for temporary, final in final_stage:
        run_root(["ip", "link", "set", temporary, "name", final], check=True)

    print("  Done.")


def _install_privileged_script(path: Path, content: str) -> None:
    """Install root-executed shell content outside operator-writable state.

    ``axol can.setup`` may itself be running as the non-root operator and use
    sudo one command at a time.  Stage the generated bytes in a private
    temporary file, then let ``install`` create a root-owned 0755 directory and
    root-owned 0755 destination.  In particular, never put a systemd/cron
    executable below ``ALMOND_HOME``: that tree is intentionally writable by
    the operator and would turn the hotplug unit into a root-code-execution
    primitive.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="axol-can-", delete=False
    ) as staged:
        staged.write(content)
        staged_path = Path(staged.name)
    try:
        run_root(
            [
                "install",
                "-d",
                "-o",
                "root",
                "-g",
                "root",
                "-m",
                "0755",
                str(path.parent),
            ],
            check=True,
        )
        run_root(
            [
                "install",
                "-o",
                "root",
                "-g",
                "root",
                "-m",
                "0755",
                str(staged_path),
                str(path),
            ],
            check=True,
        )
    finally:
        staged_path.unlink(missing_ok=True)


def _legacy_profile_scripts(profile: _Profile) -> set[Path]:
    """Old operator-state executables that must no longer appear in cron."""
    names = {profile.cron_script.name}
    if profile is _MANTIS_PROFILE:
        names.add(f"startup_{_PRE_MANTIS_NAME}.sh")
    return {directory / name for directory in _LEGACY_CAN_DIRS for name in names}


def _write_cron_script(profile: _Profile = _AXOL_PROFILE) -> None:
    """Write a profile's bring-up script.

    On the robot profile the script also covers the wheel and chest buses;
    every interface is optional and checked for presence at runtime, so one
    script serves every hardware combination — arm-only, cart-only, chest-
    only, or all of them — and an unplugged adapter never blocks the rest.
    """
    print(f"Writing CAN startup script to {profile.cron_script}...")
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
    _install_privileged_script(profile.cron_script, script)
    print("  Done.")


def _read_root_crontab() -> str:
    """Read root's crontab, distinguishing an empty table from inspection errors."""
    result = run_root(["env", "LC_ALL=C", "crontab", "-l"])
    if result.returncode == 0:
        return result.stdout or ""
    if "no crontab" in (result.stderr or "").lower():
        return ""
    detail = (result.stderr or "").strip() or f"exit {result.returncode}"
    raise RuntimeError(f"could not inspect root crontab: {detail}")


def _register_cron(profile: _Profile = _AXOL_PROFILE) -> None:
    print("Registering @reboot cron entry in root crontab (requires sudo)...")
    cron_entry = f"@reboot {profile.cron_script}"
    existing = _read_root_crontab()
    # Every historical profile lived below operator-writable ALMOND_HOME.
    # Remove all exact generated entries on either profile's next setup, not
    # merely the profile being configured: a hub reclassification can leave
    # the other profile absent while its old root cron entry still exists.
    legacy_entries = {
        f"@reboot {legacy}"
        for configured_profile in (_AXOL_PROFILE, _MANTIS_PROFILE)
        for legacy in _legacy_profile_scripts(configured_profile)
    }
    kept: list[str] = []
    current_seen = False
    for line in existing.splitlines():
        stripped = line.strip()
        if stripped in legacy_entries:
            continue
        if stripped == cron_entry:
            if current_seen:
                continue
            current_seen = True
        kept.append(line)
    if current_seen and kept == existing.splitlines():
        print("  Entry already present — skipping.")
    else:
        if not current_seen:
            kept.append(cron_entry)
        new_crontab = "\n".join(kept).rstrip("\n") + "\n"
        run_root(["crontab", "-"], input_text=new_crontab, check=True)
        print(f"  Registered: {cron_entry}")


def _remove_pre_mantis_config() -> None:
    """Remove superseded Mantis rule/script/unit names during migration."""
    # Inspect and rewrite the root scheduler first. Older cron entries pointed
    # directly into an operator-writable ~/.almond tree; deleting a generated
    # script before a failed crontab update would let the operator recreate it
    # and retain root execution at the next reboot.
    existing = _read_root_crontab()
    obsolete = {
        f"@reboot {directory / f'startup_{_PRE_MANTIS_NAME}.sh'}"
        for directory in _LEGACY_CAN_DIRS | {_CAN_DIR}
    }
    kept = [line for line in existing.splitlines() if line.strip() not in obsolete]
    if len(kept) != len(existing.splitlines()):
        new_crontab = "\n".join(kept).rstrip("\n") + "\n"
        run_root(["crontab", "-"], input_text=new_crontab, check=True)

    unit_exists = _PRE_MANTIS_HOTPLUG_UNIT_FILE.exists()
    if unit_exists:
        # The retired unit may itself execute that writable script. Prove it is
        # stopped and disabled before removing its definition.
        run_root(["systemctl", "stop", _PRE_MANTIS_HOTPLUG_UNIT], check=True)
        run_root(["systemctl", "disable", _PRE_MANTIS_HOTPLUG_UNIT], check=True)
        run_root(["rm", "-f", str(_PRE_MANTIS_HOTPLUG_UNIT_FILE)], check=True)
        run_root(["systemctl", "daemon-reload"], check=True)

    # Only after every scheduler reference is neutralized may the superseded
    # udev rule and generated root-owned script be deleted.
    for path in (_PRE_MANTIS_RULES_FILE, _PRE_MANTIS_CRON_SCRIPT):
        if path.exists():
            run_root(["rm", "-f", str(path)], check=True)


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


def _remove_rp1_quirk_unit() -> None:
    """Remove a stale quirk unit, including legacy operator-script references."""
    if not _RP1_QUIRK_UNIT_FILE.exists():
        return
    # Prove the old service is no longer running before deleting its
    # definition.  Older releases pointed this root service at an
    # operator-writable script; ignoring a stop failure would leave those bytes
    # executing while also removing the most direct way to retry or diagnose
    # the service teardown.
    run_root(["systemctl", "stop", _RP1_QUIRK_UNIT], check=True)
    run_root(["systemctl", "disable", _RP1_QUIRK_UNIT], check=True)
    run_root(["rm", "-f", str(_RP1_QUIRK_UNIT_FILE)], check=True)
    run_root(["systemctl", "daemon-reload"], check=True)


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
        _remove_rp1_quirk_unit()
        return
    if not Path("/usr/bin/busybox").exists():
        _remove_rp1_quirk_unit()
        print(
            "WARNING: busybox not found — skipping the RP1 USB EMI quirk "
            "(install busybox and re-run `axol can.setup`)."
        )
        return
    print("Applying RP1 USB EMI-tolerance quirk (Pi 5 only, requires sudo)...")
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
    _install_privileged_script(_RP1_QUIRK_SCRIPT, script)
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
    """Run the bring-up script, then verify RX and recover its hub pair.

    Every down/up cycle of the adapter's channels toggles it between a healthy
    state and the TX-only wedge described in :func:`rx_alive_per_arm`, so a
    bring-up that lands in the wedge is recovered by another pair cycle. The
    full startup script runs once so the Axol wheel/cart and chest/lift buses
    are reset once too; hub RX retries never re-flap those healthy single buses.
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
    run_root(["bash", str(profile.cron_script)], check=True)
    for attempt in range(3):
        if attempt:
            bring_up_interfaces([profile.left, profile.right], force_cycle=True)
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


def bring_up_interfaces(channels: list[str], *, force_cycle: bool = False) -> None:
    """Configure and bring up arbitrary SocketCAN interfaces.

    The non-Axol-hub counterpart of :func:`bring_up_can`, for setups running
    on some other CAN adapter: no startup script or udev naming, just
    per-interface bitrate / txqueuelen / up. Interfaces already up are left
    untouched unless ``force_cycle`` requests paired recovery. A missing one
    raises ``RuntimeError`` naming it so callers can surface which channel to
    fix.
    """
    missing = [ch for ch in channels if not (Path("/sys/class/net") / ch).exists()]
    if missing:
        raise RuntimeError(f"CAN interface not found: {', '.join(missing)}")
    if force_cycle:
        reset_label = "paired CAN recovery" if len(channels) > 1 else "CAN recovery"
        for channel in channels:
            print(f"  {channel}: cycling down for {reset_label}...")
            run_root(["ip", "link", "set", channel, "down"], check=True)

    pending = channels if force_cycle else [ch for ch in channels if not iface_up(ch)]
    for channel in channels:
        if channel not in pending:
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
    # Configure the full reset group before raising any channel. For a dual
    # hub this avoids the half-up transition that can wedge gs_usb RX.
    for channel in pending:
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
    _validate_adapter_assignments(hub_serial, wheels_serial, chest_serial)
    _write_udev_rules(hub_serial, wheels_serial, chest_serial)
    _write_cron_script()
    _write_hotplug_unit()
    _reload_udev()
    _rename_interfaces(hub_serial, wheels_serial, chest_serial)
    _register_cron()
    _setup_rp1_usb_quirk()
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
    configured_usb = _attached_configured_hub_serials().get("axol")
    if configured_usb is not None:
        _wait_for_dual_channel_serial(configured_usb)
    hub_serial = hub_serial or _resolve_hub_serial()
    wheels_serial = wheels_serial or _configured_named_serial(_CAN_B)
    chest_serial = chest_serial or _configured_named_serial(_CAN_C)
    if not (hub_serial or wheels_serial or chest_serial):
        raise RuntimeError("Robot not detected")
    _apply_setup(hub_serial, wheels_serial, chest_serial)


def ensure_mantis_setup() -> None:
    """Configure a detected Mantis hub non-interactively for the web panel.

    Every attached dual-channel hub is probed, including a serial previously
    pinned as Axol. Exactly one Mantis response must be found; silent or
    ambiguous adapters are left for interactive :func:`run` instead of being
    guessed. This is the Mantis counterpart to :func:`ensure_setup` used by
    the idle diagnostics link.
    """
    installed = driver.ensure_driver()
    configured_usb = _attached_configured_hub_serials().get("mantis")
    if configured_usb is not None:
        _wait_for_dual_channel_serial(configured_usb)
    elif installed:
        # First-time setup has no persisted serial to poll deterministically.
        time.sleep(2.0)

    configured_mantis = _configured_serial(_MANTIS_PROFILE)
    configured_axol = _configured_serial(_AXOL_PROFILE)
    configured_wheels = _configured_named_serial(_CAN_B)
    configured_chest = _configured_named_serial(_CAN_C)
    matches = [
        candidate
        for candidate in _detect_serials()
        if _identify_dual_adapter(candidate) == "mantis"
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
    serial = configured_mantis if configured_mantis in matches else matches[0]

    if serial in {configured_axol, configured_wheels, configured_chest}:
        # A live trigger response proved that any Axol role pinned to this
        # serial is stale. Clear every such match before writing the Mantis
        # profile; otherwise a former single-bus rule (which matches by serial
        # alone) could try to rename both Mantis channels to one Axol name.
        _write_udev_rules(
            None if serial == configured_axol else configured_axol,
            None if serial == configured_wheels else configured_wheels,
            None if serial == configured_chest else configured_chest,
        )
    _configure_mantis(serial)


def _find_single_serials(
    hub_serial: str | None, mantis_serial: str | None = None
) -> tuple[str | None, str | None]:
    """Interactively assign single-channel adapters to the wheel/chest buses.

    Every attached candidate is probed. Previously pinned wheel/cart and
    chest/lift buses are reset first so stale assignments correct themselves;
    an unknown generic gs_usb adapter is probed without flapping it and gets
    reset in final setup only after a positive match or operator assignment.
    A positive response wins over a stale pin. A configured adapter that is
    silent or unplugged remains as an explicitly unverified fallback, which a
    newly attached silent adapter may replace by operator choice. Duplicate
    unresolved pins are rejected. Serials claimed by a dual hub are excluded.

    Returns ``(wheels_serial, chest_serial)``, either of which may be None.
    """
    configured = {
        "wheels": _configured_named_serial(_CAN_B),
        "chest": _configured_named_serial(_CAN_C),
    }
    selected_dual = {serial for serial in (hub_serial, mantis_serial) if serial}
    exclude = set(selected_dual)
    exclude |= _mantis_claimed_serials()
    attached = sorted(_detect_single_serials(exclude))
    all_attached = set(_scan_adapters()) | set(attached)

    if attached:
        configured_note = (
            ", including previously configured adapters"
            if any(configured.values())
            else ""
        )
        print(
            f"Identifying {len(attached)} single-channel CAN adapter(s) by "
            f"probing{configured_note} (wheel motors / cart lift must be "
            "powered)..."
        )
    detected: dict[str, str | None] = {}
    for serial in attached:
        print(f"  {serial}: probing wheel drive / cart lift controller...")
        known = serial in configured.values()
        detected[serial] = (
            _identify_adapter(serial, reset=True)
            if known
            else _identify_adapter(serial, recover_silence=False)
        )

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

    def detected_for(role: str) -> str | None:
        matches = sorted(serial for serial, found in detected.items() if found == role)
        if not matches:
            return None
        configured_serial = configured[role]
        selected = configured_serial if configured_serial in matches else matches[0]
        label = "Damiao wheel motors" if role == "wheels" else "cart lift controller"
        target = _CAN_B if role == "wheels" else _CAN_C
        print(f"  {selected}: {label} answered -> {target}")
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
        # Do not preserve a single-bus pin when that serial is currently
        # attached under a different topology (especially a selected hub).
        if old_serial in selected_dual or (
            old_serial in all_attached and old_serial not in detected
        ):
            continue
        observed = detected.get(old_serial)
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
                f"    Assign it to the [w]heel/cart bus ({_CAN_B}), the "
                f"[c]hest/lift bus ({_CAN_C}), or leave blank to skip: "
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


def run(args: object = None) -> None:
    """Configure persistent CAN interfaces and a @reboot bring-up entry."""
    try:
        installed = driver.ensure_driver()
    except RuntimeError as exc:
        _die(str(exc))
    if installed:
        # The freshly-loaded driver may claim adapters the old one ignored
        # (CANable 2.0); give their interfaces a moment to appear.
        time.sleep(2.0)

    configured_axol = _configured_serial(_AXOL_PROFILE)
    configured_mantis = _configured_serial(_MANTIS_PROFILE)
    configured_wheels = _configured_named_serial(_CAN_B)
    configured_chest = _configured_named_serial(_CAN_C)
    hub_serial, mantis_serial = _find_dual_serials()
    wheels_serial, chest_serial = _find_single_serials(hub_serial, mantis_serial)
    if not (hub_serial or mantis_serial or wheels_serial or chest_serial):
        _die(
            "No CAN adapters found or configured. Connect the Axol/Mantis hub, "
            "wheel-bus, or chest adapter and re-run."
        )
    assignments = {
        "Axol": hub_serial,
        "Mantis": mantis_serial,
        "wheels": wheels_serial,
        "chest/lift": chest_serial,
    }
    for serial in {value for value in assignments.values() if value is not None}:
        claimed = [name for name, value in assignments.items() if value == serial]
        if len(claimed) > 1:
            _die(
                f"Adapter {serial} resolved to multiple CAN roles "
                f"({', '.join(claimed)}). Power the attached hardware and "
                "re-run `axol can.setup` to resolve it before rules are written."
            )
    axol_role_reclassified = mantis_serial is not None and mantis_serial in {
        configured_axol,
        configured_wheels,
        configured_chest,
    }
    mantis_reclassified = (
        configured_mantis is not None and configured_mantis == hub_serial
    )
    if hub_serial or wheels_serial or chest_serial:
        _apply_setup(hub_serial, wheels_serial, chest_serial)
    elif axol_role_reclassified:
        # _configure_mantis below reloads udev and renames the live interfaces;
        # erase every stale Axol role for this serial first. In particular, a
        # former single-bus rule matches by serial alone and would otherwise try
        # to rename both Mantis channels to the same wheel/chest interface on the
        # next hotplug. Running the full Axol bring-up here would pointlessly
        # probe the Mantis grippers as shoulders.
        _write_udev_rules(
            None if configured_axol == mantis_serial else configured_axol,
            None if configured_wheels == mantis_serial else configured_wheels,
            None if configured_chest == mantis_serial else configured_chest,
        )
    if mantis_serial:
        _configure_mantis(mantis_serial)
    elif mantis_reclassified:
        # The live Axol response overrode a stale Mantis pin. Leave its startup
        # assets harmlessly installed, but remove the serial-matching udev rule
        # so the adapter cannot be renamed back to Mantis on its next hotplug.
        _remove_pre_mantis_config()
        _write_udev_rules(None, profile=_MANTIS_PROFILE)
        _reload_udev()

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
