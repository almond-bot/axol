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


# ---------------------------------------------------------------------------
# MyActuator — 0xC0 parameter indices
# ---------------------------------------------------------------------------

# Index range worth sweeping in a raw dump. The vendor software never addresses
# anything above 0x55, so this covers its whole parameter space.
PARAM_SWEEP_RANGE = range(0x00, 0x60)


class MyActuatorParam(IntEnum):
    """Parameter index carried in byte 2 of a 0xC0 frame.

    None of these appear in MyActuator's published protocol (V4.4). The command
    and the index assignments were recovered from the vendor setup software,
    whose entire "advanced parameter" UI is built on 0xC0 — 158 of the ~180 CAN
    calls in that binary are 0xC0, with 0xC1 committing batches to ROM.

    Indices were recovered by matching each 0xC0 call site to the field it
    loads from or stores into, then aligning those fields against the parameter
    list the GUI renders. The fields form one contiguous run, and four of them
    (:attr:`ENCODER_CALIBRATION_VALUE`, :attr:`OVER_VOLTAGE`,
    :attr:`LOW_VOLTAGE`, :attr:`STALL_TIME_LIMIT`) were confirmed directly,
    which pins the alignment for the rest.

    Parameters the GUI shows past the end of this run — stall current, shutdown
    and resume temperature, max and nominal speed — are deliberately absent:
    their call sites could not be resolved, so their indices are unknown.
    Reading an unknown index is harmless, so use ``dump_config(raw_range=...)``
    to identify them against a motor rather than guessing here. "Motor Position
    Zero" is likewise omitted because two candidate indices fit it equally well;
    the documented 0x64 command already sets the zero position.
    """

    POWERDOWN_SAVE_MULTITURN = 0x1C
    POLE_PAIRS = 0x1D
    SINGLE_TURN_RESOLUTION = 0x1F
    CALIBRATION_CURRENT = 0x20
    EXCHANGE_PHASE = 0x21
    EBRAKE_HOLD_DUTY = 0x22
    BRAKE_MODE = 0x23
    MAX_POSITIVE_POSITION = 0x24
    MIN_NEGATIVE_POSITION = 0x25
    POSITION_PLAN_MAX_ACC = 0x26
    POSITION_PLAN_MAX_DEC = 0x27
    POSITION_PLAN_MAX_SPEED = 0x28
    SPEED_PLAN_MAX_ACC = 0x29
    SPEED_PLAN_MAX_DEC = 0x3C
    CHANGE_MOTOR_DIRECTION = 0x3E
    ENCODER_CALIBRATION_VALUE = 0x46
    STALL_TIME_LIMIT = 0x47
    EBRAKE_START_DUTY = 0x48
    CURRENT_SAMPLE_RES = 0x49
    OVER_VOLTAGE = 0x54
    LOW_VOLTAGE = 0x55


_RW = Access.READ_WRITE
_PROT = Access.PROTECTED
_RO = Access.READ_ONLY

MYACTUATOR_PARAMS: dict[MyActuatorParam, ParamSpec] = {
    MyActuatorParam.POWERDOWN_SAVE_MULTITURN: ParamSpec(""),
    MyActuatorParam.POLE_PAIRS: ParamSpec("", _PROT),
    MyActuatorParam.SINGLE_TURN_RESOLUTION: ParamSpec("pulses", _PROT),
    MyActuatorParam.CALIBRATION_CURRENT: ParamSpec("A", _PROT),
    MyActuatorParam.EXCHANGE_PHASE: ParamSpec("", _PROT),
    MyActuatorParam.EBRAKE_HOLD_DUTY: ParamSpec("%"),
    MyActuatorParam.BRAKE_MODE: ParamSpec(""),
    MyActuatorParam.MAX_POSITIVE_POSITION: ParamSpec("deg"),
    MyActuatorParam.MIN_NEGATIVE_POSITION: ParamSpec("deg"),
    MyActuatorParam.POSITION_PLAN_MAX_ACC: ParamSpec("dps/s"),
    MyActuatorParam.POSITION_PLAN_MAX_DEC: ParamSpec("dps/s"),
    MyActuatorParam.POSITION_PLAN_MAX_SPEED: ParamSpec("rpm"),
    MyActuatorParam.SPEED_PLAN_MAX_ACC: ParamSpec("dps/s"),
    MyActuatorParam.SPEED_PLAN_MAX_DEC: ParamSpec("dps/s"),
    MyActuatorParam.CHANGE_MOTOR_DIRECTION: ParamSpec("", _PROT),
    MyActuatorParam.ENCODER_CALIBRATION_VALUE: ParamSpec("", _PROT),
    MyActuatorParam.STALL_TIME_LIMIT: ParamSpec("s"),
    MyActuatorParam.EBRAKE_START_DUTY: ParamSpec("%"),
    MyActuatorParam.CURRENT_SAMPLE_RES: ParamSpec("mOhm", _PROT),
    MyActuatorParam.OVER_VOLTAGE: ParamSpec("V"),
    MyActuatorParam.LOW_VOLTAGE: ParamSpec("V"),
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
