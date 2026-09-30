"""Configuration parameter tables for both motor families.

Each driver owns its own table. The two enums must never share a dict: both
are :class:`enum.IntEnum`, so they hash as plain ints, and their value ranges
overlap (Damiao registers run 0-36, MyActuator indices 0x1C-0x55) — merged,
they would silently alias one another.

MyActuator
    Reached through the undocumented 0xC0 command; see :class:`MyActuatorParam`.

Damiao
    Reached through the documented 0x7FF register protocol (0x33 read, 0x55
    write, 0xAA store), which the driver already implements.
"""

from dataclasses import dataclass
from enum import IntEnum


class Access(IntEnum):
    """How freely a parameter may be written."""

    READ_WRITE = 0
    """Ordinary setting: written by a restore, shown as editable."""

    PROTECTED = 1
    """Writable, but not casually.

    Covers factory/calibration values (pole pairs, encoder calibration, phase
    order) and identity/comm settings (CAN IDs, baud rate). Getting one wrong
    can leave a motor unable to commutate or unreachable on the bus, and the
    identity ones would change the address mid-conversation, so a restore
    skips them unless explicitly told otherwise.
    """

    READ_ONLY = 2
    """Reported by the motor but not writable — firmware versions, serial
    number, measured winding constants. Always skipped by a write."""


@dataclass(frozen=True)
class ParamSpec:
    """Display and safety metadata for one configuration parameter."""

    unit: str
    """Unit of the *displayed* value; empty for unitless flags and codes."""

    access: Access = Access.READ_WRITE

    integer: bool = False
    """Whether the value is a whole number, so it is shown without decimals."""

    scale: float = 1.0
    """Multiplier from the motor's raw units to the displayed unit.

    Only Damiao's timeout needs this today: the register counts 50 µs ticks,
    which is a needless footgun to expose, so it is shown and set in ms.
    """

    min_firmware: int | None = None
    """Oldest firmware VersionDate that implements the parameter, if not all do.

    On older firmware the index still answers, but with junk: the reply echoes
    the request and carries whatever bytes the motor last transmitted. The
    driver skips such parameters in dumps and restores and refuses to read or
    write them individually.
    """


# ---------------------------------------------------------------------------
# MyActuator — 0xC0 parameter indices
# ---------------------------------------------------------------------------

# First MyActuator firmware (VersionDate) implementing protocol V4.4 — the
# MIT-range change in myactuator.py and the 0x46-0x4A parameters below.
MYACTUATOR_FW_V44 = 2026042402

# Index range worth sweeping in a raw dump. The vendor software never addresses
# anything above 0x55, so this covers its whole parameter space.
PARAM_SWEEP_RANGE = range(0x00, 0x60)


class MyActuatorParam(IntEnum):
    """Parameter index carried in byte 2 of a 0xC0 frame.

    None of these appear in MyActuator's published protocol (V4.4). The command
    and the index assignments were recovered from the vendor setup software,
    whose entire "advanced parameter" UI is built on 0xC0 — 158 of the ~180 CAN
    calls in that binary are 0xC0, with 0xC1 committing batches to ROM.

    This table is the mapping the **fleet firmware actually uses**
    (VersionDate 2025070202 and 2026042402), pinned two independent ways that
    agree exactly:

    * Static: every 0xC0 call site in the setup binary (``myactuator.exe``,
      Setup Software V4.0) was decoded through its shared frame builder
      (``sub_0x4039e0``: ``[0xC0, 0x00, index, rw, v0..v3]``), which fixes the
      whole index space at 0x01-0x55 and shows which index each GUI field reads
      (and, from the Save handlers, which ones it writes and how).
    * Live: the value the GUI shows for every field was matched against a raw
      ``dump_config(raw_range=...)`` sweep of the five left-arm motors. The
      per-motor values that differ (Factory Time, Motor Position Zero, KT_OUT,
      the current and speed limits, Current/Voltage Sample Res, Max Torque)
      each match exactly one index on every motor, so the mapping is not a
      coincidence. Every write the GUI makes to these indices sends the four
      little-endian bytes of a float32, the same encoding ``_write_param`` uses.

    An earlier revision of this table was mis-indexed for the current firmware —
    it named, e.g., 0x54/0x55 "over/low voltage" when the real voltage
    thresholds are at 0x13/0x14 (the setup software uses 0x54/0x55 for the MIT
    KP/KD ceilings, which no fleet firmware implements). Every
    entry below was re-derived from the GUI ground truth; if you ever move to a
    firmware line whose ``dump_config`` no longer matches, re-verify before
    trusting a write.

    A handful of factory/identity fields the GUI's "Settings" tab exposes (pole
    pairs, single-turn resolution, calibration current, phase order, encoder
    calibration value, powerdown-save-multiturn, change-direction) are
    intentionally absent: their indices are not in the captured screenshots and
    could not be pinned by value. Reading an unknown index is harmless, so
    recover them with ``dump_config(raw_range=...)`` against a motor rather than
    guessing. The motor's zero is set by the documented 0x64 command;
    ``MOTOR_POSITION_ZERO`` here is the raw calibration field and is protected.

    The Motor-panel selectors were disambiguated from the read handler's widget
    calls (a read feeding a checkbox is float-tested against zero and stored as
    a byte; one feeding a combo box is stored as a value) and from the combo item
    strings, which give each selector's value meaning — see ``MYACTUATOR_PARAMS``.

    Not every index is implemented on every firmware. An unimplemented one still
    answers, echoing the request but carrying whatever value bytes the motor
    last sent, so it reads as plausible junk (often ~0) rather than failing.
    Probed on the fleet by priming the motor with a known reply and reading the
    index straight after: 0x54/0x55 ("MIT model max KP/KD") are unimplemented on
    both firmware lines and so are left out, and 0x46-0x4A are implemented on
    2026042402 (the X8s) but not 2025070202 (the X6s) — see ``min_firmware``.
    Firmware and model coincide on this fleet, so the gate is by firmware, as
    the driver's other V4.4 behavior is.

    "Enable CAN Filter" is deliberately absent. The GUI reads the checkbox from
    0x32 and writes it through function-control 0x20/0x02 rather than 0xC0, but
    0x32 reads 1 on every fleet motor while the GUI shows the filter off, so the
    label cannot be confirmed. 0x32 stays reachable through a raw sweep.
    """

    # Motor Information panel (Basic Parameters tab).
    MOTOR_NUMBER = 0x01
    FACTORY_TIME = 0x02
    REDUCTION_RATIO = 0x03
    # Protect Parameters panel.
    OVER_VOLTAGE = 0x13
    LOW_VOLTAGE = 0x14
    STALL_TIME_LIMIT = 0x15
    EBRAKE_START_DUTY = 0x16
    CURRENT_SAMPLE_RES = 0x17
    EBRAKE_HOLD_DUTY = 0x18
    BRAKE_MODE = 0x19
    # Plan Parameters panel.
    MAX_POSITIVE_POSITION = 0x1A
    MIN_NEGATIVE_POSITION = 0x1B
    POSITION_PLAN_MAX_ACC = 0x1C
    POSITION_PLAN_MAX_DEC = 0x1D
    SPEED_PLAN_MAX_ACC = 0x1F
    SPEED_PLAN_MAX_DEC = 0x20
    MOTOR_POSITION_ZERO = 0x21
    # Motor Parameters panel.
    RATED_CURRENT = 0x22
    MAX_CURRENT = 0x23
    STALL_CURRENT = 0x24
    SHUTDOWN_TEMP = 0x25
    RESUME_TEMP = 0x26
    MAX_SPEED = 0x27
    NOMINAL_SPEED = 0x28
    ENABLE_2ND_ENCODER = 0x29
    ENABLE_ETHERCAT = 0x3B
    SECOND_ENCODER_RESOLUTION = 0x3C
    THERMISTOR = 0x3D
    KT_OUT = 0x3E
    ENCODER2_ABNORMAL_VALUE = 0x3F
    ENCODER2_ABNORMAL_SPEED = 0x40
    AUTOMATIC_ERROR_RECOVERY = 0x41
    MAX_TORQUE = 0x46
    VOLTAGE_SAMPLE_RES = 0x47
    LPF_CF_FOR_CURRENT = 0x48
    LPF_CF_FOR_SPEED = 0x49
    ERROR_DETECTION_LPF_CF = 0x4A


_RW = Access.READ_WRITE
_PROT = Access.PROTECTED
_RO = Access.READ_ONLY
_V44 = MYACTUATOR_FW_V44

MYACTUATOR_PARAMS: dict[MyActuatorParam, ParamSpec] = {
    # Motor Information (Basic Parameters tab). The GUI's Save writes all three,
    # but they are factory identity — and the reduction ratio scales the output
    # position — so they are protected.
    MyActuatorParam.MOTOR_NUMBER: ParamSpec("", _PROT),
    MyActuatorParam.FACTORY_TIME: ParamSpec("", _PROT),
    MyActuatorParam.REDUCTION_RATIO: ParamSpec("", _PROT),
    # Protect Parameters.
    MyActuatorParam.OVER_VOLTAGE: ParamSpec("V"),
    MyActuatorParam.LOW_VOLTAGE: ParamSpec("V"),
    MyActuatorParam.STALL_TIME_LIMIT: ParamSpec("s"),
    MyActuatorParam.EBRAKE_START_DUTY: ParamSpec("%"),
    MyActuatorParam.CURRENT_SAMPLE_RES: ParamSpec("mOhm", _PROT),
    MyActuatorParam.EBRAKE_HOLD_DUTY: ParamSpec("%"),
    # 0 = holding brake ("E-Brake"), 1 = braking resistor.
    MyActuatorParam.BRAKE_MODE: ParamSpec(""),
    MyActuatorParam.ERROR_DETECTION_LPF_CF: ParamSpec("", min_firmware=_V44),
    MyActuatorParam.AUTOMATIC_ERROR_RECOVERY: ParamSpec(""),
    # Second-encoder (OUTENCODER2) deviation thresholds. No unit suffix in the
    # GUI and stored unscaled, so none is claimed.
    MyActuatorParam.ENCODER2_ABNORMAL_VALUE: ParamSpec(""),
    MyActuatorParam.ENCODER2_ABNORMAL_SPEED: ParamSpec(""),
    # Plan Parameters.
    MyActuatorParam.MAX_POSITIVE_POSITION: ParamSpec("deg"),
    MyActuatorParam.MIN_NEGATIVE_POSITION: ParamSpec("deg"),
    MyActuatorParam.POSITION_PLAN_MAX_ACC: ParamSpec("dps/s"),
    MyActuatorParam.POSITION_PLAN_MAX_DEC: ParamSpec("dps/s"),
    MyActuatorParam.SPEED_PLAN_MAX_ACC: ParamSpec("dps/s"),
    MyActuatorParam.SPEED_PLAN_MAX_DEC: ParamSpec("dps/s"),
    MyActuatorParam.MOTOR_POSITION_ZERO: ParamSpec("pulses", _PROT),
    # The GUI gives KT_OUT no unit, so none is claimed.
    MyActuatorParam.KT_OUT: ParamSpec("", _PROT),
    MyActuatorParam.MAX_TORQUE: ParamSpec("Nm", _PROT, min_firmware=_V44),
    MyActuatorParam.VOLTAGE_SAMPLE_RES: ParamSpec("kOhm", _PROT, min_firmware=_V44),
    # Motor Parameters (limits).
    MyActuatorParam.RATED_CURRENT: ParamSpec("A"),
    MyActuatorParam.MAX_CURRENT: ParamSpec("A"),
    MyActuatorParam.STALL_CURRENT: ParamSpec("A"),
    MyActuatorParam.SHUTDOWN_TEMP: ParamSpec("C"),
    MyActuatorParam.RESUME_TEMP: ParamSpec("C"),
    MyActuatorParam.MAX_SPEED: ParamSpec("rpm"),
    MyActuatorParam.NOMINAL_SPEED: ParamSpec("rpm"),
    MyActuatorParam.LPF_CF_FOR_CURRENT: ParamSpec("", min_firmware=_V44),
    MyActuatorParam.LPF_CF_FOR_SPEED: ParamSpec("", min_firmware=_V44),
    # Second-encoder / fieldbus / thermistor configuration. The GUI's Save writes
    # all four over 0xC0; they are hardware/identity settings, so protected.
    #
    # ENABLE_2ND_ENCODER is a mode: 0 = disabled, 1 = "Encoder1 262144",
    # 2 = "Encoder2 16384", 3 = "Encoder3 131072". The GUI derives
    # SECOND_ENCODER_RESOLUTION (pulses/rev) from that mode and always writes the
    # pair together — change them together too. THERMISTOR: 0 = "Thermistor1",
    # 1 = "Thermistor2".
    MyActuatorParam.ENABLE_2ND_ENCODER: ParamSpec("", _PROT),
    MyActuatorParam.SECOND_ENCODER_RESOLUTION: ParamSpec("pulses", _PROT),
    MyActuatorParam.ENABLE_ETHERCAT: ParamSpec("", _PROT),
    MyActuatorParam.THERMISTOR: ParamSpec("", _PROT),
}


# ---------------------------------------------------------------------------
# Damiao — 0x7FF register IDs
# ---------------------------------------------------------------------------


class DamiaoParam(IntEnum):
    """Damiao register ID (RID), carried in byte 3 of a 0x7FF frame.

    This table is published by Damiao and matches the register list embedded in
    the DMTool setup software, where the fields appear in RID order. It is
    corroborated by the driver's own ``_DM_UINT32_REGS``: every register this
    table types as an integer is exactly one the driver already packs as
    uint32.
    """

    UV_VALUE = 0  # undervoltage threshold
    KT_VALUE = 1  # torque constant
    OT_VALUE = 2  # overtemperature threshold
    OC_VALUE = 3  # overcurrent threshold
    ACC = 4
    DEC = 5
    MAX_SPD = 6
    MST_ID = 7  # feedback CAN ID
    ESC_ID = 8  # receive CAN ID
    TIMEOUT = 9  # CAN loss-of-comms alarm time
    CTRL_MODE = 10  # 1=MIT, 2=POS_VEL, 3=VEL, 4=FORCE_POS
    DAMP = 11
    INERTIA = 12
    HW_VER = 13
    SW_VER = 14
    SN = 15
    NPP = 16  # pole pairs
    RS = 17  # phase resistance
    LS = 18  # phase inductance
    FLUX = 19
    GR = 20  # gear ratio
    PMAX = 21  # position scaling for the MIT protocol
    VMAX = 22  # velocity scaling for the MIT protocol
    TMAX = 23  # torque scaling for the MIT protocol
    I_BW = 24  # current loop bandwidth
    KP_ASR = 25  # speed loop Kp
    KI_ASR = 26  # speed loop Ki
    KP_APR = 27  # position loop Kp
    KI_APR = 28  # position loop Ki
    OV_VALUE = 29  # overvoltage threshold
    GREF = 30
    DETA = 31
    V_BW = 32  # velocity loop bandwidth
    IQ_CL = 33
    VL_CL = 34
    CAN_BR = 35  # CAN baud rate code (0-4)
    SUB_VER = 36


# The register counts 50 µs ticks; ms is the unit every human uses for it.
DAMIAO_TIMEOUT_MS_PER_UNIT = 0.05

DAMIAO_PARAMS: dict[DamiaoParam, ParamSpec] = {
    DamiaoParam.UV_VALUE: ParamSpec("V"),
    DamiaoParam.KT_VALUE: ParamSpec("Nm/A"),
    DamiaoParam.OT_VALUE: ParamSpec("C"),
    DamiaoParam.OC_VALUE: ParamSpec("A"),
    DamiaoParam.ACC: ParamSpec("rad/s^2"),
    DamiaoParam.DEC: ParamSpec("rad/s^2"),
    DamiaoParam.MAX_SPD: ParamSpec("rad/s"),
    DamiaoParam.MST_ID: ParamSpec("", _PROT, integer=True),
    DamiaoParam.ESC_ID: ParamSpec("", _PROT, integer=True),
    DamiaoParam.TIMEOUT: ParamSpec(
        "ms", _RW, integer=False, scale=DAMIAO_TIMEOUT_MS_PER_UNIT
    ),
    DamiaoParam.CTRL_MODE: ParamSpec("", _RW, integer=True),
    DamiaoParam.DAMP: ParamSpec("", _RO),
    DamiaoParam.INERTIA: ParamSpec("kg*m^2", _RO),
    DamiaoParam.HW_VER: ParamSpec("", _RO, integer=True),
    DamiaoParam.SW_VER: ParamSpec("", _RO, integer=True),
    DamiaoParam.SN: ParamSpec("", _RO, integer=True),
    DamiaoParam.NPP: ParamSpec("", _RO, integer=True),
    DamiaoParam.RS: ParamSpec("mOhm", _RO),
    DamiaoParam.LS: ParamSpec("uH", _RO),
    DamiaoParam.FLUX: ParamSpec("Wb", _RO),
    DamiaoParam.GR: ParamSpec("", _RO),
    DamiaoParam.PMAX: ParamSpec("rad"),
    DamiaoParam.VMAX: ParamSpec("rad/s"),
    DamiaoParam.TMAX: ParamSpec("Nm"),
    DamiaoParam.I_BW: ParamSpec("Hz"),
    DamiaoParam.KP_ASR: ParamSpec(""),
    DamiaoParam.KI_ASR: ParamSpec(""),
    DamiaoParam.KP_APR: ParamSpec(""),
    DamiaoParam.KI_APR: ParamSpec(""),
    DamiaoParam.OV_VALUE: ParamSpec("V"),
    DamiaoParam.GREF: ParamSpec(""),
    DamiaoParam.DETA: ParamSpec(""),
    DamiaoParam.V_BW: ParamSpec("Hz"),
    DamiaoParam.IQ_CL: ParamSpec(""),
    DamiaoParam.VL_CL: ParamSpec(""),
    DamiaoParam.CAN_BR: ParamSpec("", _PROT, integer=True),
    DamiaoParam.SUB_VER: ParamSpec("", _RO, integer=True),
}


MotorParam = MyActuatorParam | DamiaoParam

# Union of every parameter name, for CLI/UI dropdowns. Names are unique across
# the two families, so a name resolves to at most one parameter; the driver
# still validates that the name belongs to *its* table.
ALL_PARAM_NAMES: list[str] = [p.name for p in MyActuatorParam] + [
    p.name for p in DamiaoParam
]
