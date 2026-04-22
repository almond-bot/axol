from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum


class ControlMode(IntEnum):
    """Active control mode for a motor.

    Integer values match the Damiao register encoding (register 10).
    """

    IMPEDANCE = 1
    POSITION_VELOCITY = 2
    VELOCITY = 3
    POSITION_FORCE = 4


class MotorStatus(Enum):
    """Unified motor status / error code returned by both driver types."""

    OK = "ok"
    DISABLED = "disabled"
    OVER_VOLTAGE = "over_voltage"
    UNDER_VOLTAGE = "under_voltage"
    OVER_CURRENT = "over_current"
    OVER_TEMPERATURE = "over_temperature"
    MOS_OVER_TEMP = "mos_over_temp"
    ROTOR_OVER_TEMP = "rotor_over_temp"
    LOST_COMM = "lost_comm"
    OVERLOAD = "overload"
    MOTOR_STALL = "motor_stall"  # MyActuator only
    ENCODER_ERROR = "encoder_error"  # MyActuator only
    POWER_OVERRUN = "power_overrun"  # MyActuator only
    SPEEDING = "speeding"  # MyActuator only
    UNKNOWN = "unknown"


@dataclass
class MotorGains:
    """
    PID gains for the motor's internal speed and position control loops.

    Used with Motor.get_gains() and Motor.set_gains(). Only relevant when
    using set_position_velocity() or set_velocity(); set_impedance() accepts gains
    directly per command and does not use these stored values.

    Attributes:
        speed_kp:    Speed loop proportional gain.
        speed_ki:    Speed loop integral gain.
        position_kp: Position loop proportional gain.
        position_ki: Position loop integral gain.
        current_kp:  Current loop proportional gain (MyActuator only, uint8).
        current_ki:  Current loop integral gain (MyActuator only, uint8).
    """

    speed_kp: float = 0.0
    speed_ki: float = 0.0
    position_kp: float = 0.0
    position_ki: float = 0.0
    current_kp: float | None = field(default=None)  # MyActuator only
    current_ki: float | None = field(default=None)  # MyActuator only
