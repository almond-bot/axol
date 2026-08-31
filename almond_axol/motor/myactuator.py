"""MyActuator RMD motor driver implementing the 0x140-series CAN protocol."""

# Protocol source: myactuator_rmd/include/myactuator_rmd/driver/can_address_offset.hpp
#                  myactuator_rmd/include/myactuator_rmd/protocol/command_type.hpp
#                  myactuator_rmd/include/myactuator_rmd/protocol/motion_control_request.hpp
#                  myactuator_rmd/include/myactuator_rmd/actuator_state/error_code.hpp
#                  myactuator_rmd/include/myactuator_rmd/actuator_state/acceleration_type.hpp

from __future__ import annotations

import asyncio
import logging
import math
import re
import struct
from typing import Callable

import can

from .bus import CanBus
from .config import MYACTUATOR_PARAMS, PARAM_SWEEP_RANGE, MyActuatorParam
from .driver import MotorDriver
from .errors import MotorError
from .types import ControlMode, MotorGains, MotorStatus

_logger = logging.getLogger(__name__)

_MA_REQ = 0x140  # standard command request  → 0x140 + motor_id
_MA_RESP = 0x240  # standard command response ← 0x240 + motor_id
_MA_MC_REQ = 0x400  # motion control request    → 0x400 + motor_id
_MA_MC_RESP = 0x500  # motion control response   ← 0x500 + motor_id

_MA_SHUTDOWN = 0x80
_MA_RELEASE_BRAKE = 0x77
_MA_RESET = 0x76  # system reset; no response — motor restarts immediately
_MA_READ_STATUS1 = 0x9A  # temperature, voltage, error flags
_MA_READ_VERSION = 0xB2  # system software VersionDate (uint32, e.g. 2026042402)
_MA_READ_MODEL = 0xB5  # motor model string (5 ASCII chars per index)
_MA_MULTI_TURN_ANGLE = 0x92
_MA_MOTOR_STATUS_2 = 0x9C  # temperature, current, velocity, encoder
_MA_SET_ENCODER_ZERO = 0x64
_MA_POS_CONTROL = 0xA4  # absolute position closed-loop control
_MA_VELOCITY_CONTROL = 0xA2  # speed closed-loop control
_MA_FUNCTION_CONTROL = 0x20  # function control; byte 1 = index, bytes 4-7 = value
_MA_FC_SET_CANID = 0x05  # function control index: set CAN ID
_MA_READ_GAINS = 0x30  # read all PID gains (uint8, bulk)
_MA_WRITE_GAINS_ROM = (
    0x32  # write all PID gains to ROM (uint8, bulk); persistent by command
)
_MA_SET_ACCELERATION = 0x43  # write acceleration to RAM and ROM; persistent by command

# Configuration-parameter access. These two commands are absent from MyActuator's
# published protocol (V4.4) — they were recovered from the vendor setup software,
# which uses them for every "advanced" / protection parameter the GUI exposes.
#
#   request: [0xC0, 0x00, index, rw, v0, v1, v2, v3]
#
# where ``rw`` selects read or write and ``v0..v3`` are a little-endian float32
# (the value in the parameter's display unit — Volts for the threshold below).
# A read echoes the frame back with the stored value in bytes 4-7. Writes only
# land in RAM; 0xC1 commits every parameter written since the last commit to ROM.
_MA_PARAM_ACCESS = 0xC0
_MA_PARAM_SAVE = 0xC1  # commit 0xC0 writes to ROM; sent with an empty payload
_MA_PARAM_READ = 0x01  # byte 3 of a 0xC0 frame
_MA_PARAM_WRITE = 0x00

# Undervoltage threshold applied to every motor on enable(). Negative, so the
# bus voltage can never fall below it and the motor never latches the level-2
# undervoltage fault (status-1 bit 0x0004).
_MA_LOW_VOLTAGE_THRESHOLD_V = -2.0

# Match tolerance (V) when checking the stored threshold against the target.
# Only a mismatch triggers a write, so an already-provisioned motor doesn't
# burn a ROM write cycle on every enable().
_MA_LOW_VOLTAGE_TOL_V = 0.05

# Seconds to wait after a 0x76 system reset before the motor answers again.
# Measured reboot time is ~1.12s; 2.0s leaves margin across motors/temperature.
_MA_RESET_SETTLE_S = 2.0

# AccelerationType enum values from acceleration_type.hpp
_MA_ACC_POS_PLAN = 0x00  # position planning acceleration
_MA_DEC_POS_PLAN = 0x01  # position planning deceleration
_MA_ACC_VEL_PLAN = 0x02  # velocity planning acceleration
_MA_DEC_VEL_PLAN = 0x03  # velocity planning deceleration

# MIT / motion-control (0x400) parameter ranges.
#
# MyActuator firmware dated 2026042402 (VersionDate) and later implements
# protocol V4.4, which widened several MIT-command ranges. The motor scales the
# fixed-point command/feedback fields against whichever range its firmware uses,
# so the host MUST match the motor's firmware or the applied torque/damping is
# silently wrong — e.g. a t_ff encoded for the legacy ±24 Nm range decodes to
# ~2.5x on V4.4 firmware (±60 Nm on an X6). The active firmware is detected
# per-motor in enable() via the 0xB2 version read.
_MA_FW_V44_VERSION = 2026042402

# Unchanged across firmware versions.
_MA_V_MAX = 45.0  # rad/s
_MA_KP_MAX = 500.0

# Legacy firmware (< V4.4).
_MA_P_MAX_LEGACY = 12.5  # rad
_MA_KD_MAX_LEGACY = 5.0
_MA_T_MAX_LEGACY = 24.0  # Nm — fixed range for both t_ff and feedback torque

# V4.4 firmware (>= _MA_FW_V44_VERSION). p_des and kd widened; the t_ff /
# feedback-torque range becomes ±(motor max torque) read from the model.
_MA_P_MAX_V44 = 12.566  # rad
_MA_KD_MAX_V44 = 50.0

# Motor max torque (Nm) keyed by model series — the "X<n>" token found in the
# 0xB5 model string (e.g. "RMD-X8-P20" -> 8). Used as the V4.4 t_ff range and
# to decode feedback torque. Extend as new series are introduced.
_MA_SERIES_MAX_TORQUE: dict[int, float] = {
    6: 60.0,
    8: 129.0,
}

# Fallback when the model can't be read or its series isn't in the table above.
_MA_DEFAULT_MAX_TORQUE = 24.0

# Acceleration range in dps/s²; [100, 60000] — 0 disables ramping.
_MA_ACC_MIN_DPS_S2 = 100
_MA_ACC_MAX_DPS_S2 = 60000

# Status-1 (0x9A) error bitmask → MotorStatus, ordered by severity (first match
# wins). Bit definitions per protocol V4.4 §2.13: level-1 (non-recoverable)
# faults are listed first so they win over recoverable level-2 faults.
_MA_ERROR_MAP: list[tuple[int, MotorStatus]] = [
    (0x0010, MotorStatus.OVER_CURRENT),  # phase overcurrent (L1)
    (0x0002, MotorStatus.MOTOR_STALL),  # (L1)
    (0x0080, MotorStatus.CALIBRATION_ERROR),  # calibration parameter write error (L1)
    (0x2000, MotorStatus.ENCODER_ERROR),  # encoder calibration error (L1)
    (0x4000, MotorStatus.ENCODER_ERROR),  # encoder data abnormal (L1)
    (0x0800, MotorStatus.OVER_TEMPERATURE),  # component (PCB) overtemp (L2)
    (0x1000, MotorStatus.OVER_TEMPERATURE),  # motor overtemp (L2)
    (0x0008, MotorStatus.OVER_VOLTAGE),  # (L2)
    (0x0004, MotorStatus.UNDER_VOLTAGE),  # (L2)
    (0x0040, MotorStatus.POWER_OVERRUN),  # (L2)
    (0x0100, MotorStatus.SPEEDING),  # overspeed (L2)
]


def _float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    """Encode a clamped float into a fixed-point uint for the MIT protocol byte layout."""
    if not math.isfinite(x):
        raise ValueError("cannot encode a non-finite motor command")
    x = max(x_min, min(x_max, x))
    return int((x - x_min) * ((1 << bits) - 1) / (x_max - x_min))


def _uint_to_float(x_int: int, x_min: float, x_max: float, bits: int) -> float:
    """Decode a fixed-point uint from the MIT protocol byte layout back into a float."""
    return float(x_int) * (x_max - x_min) / ((1 << bits) - 1) + x_min


def _ma_error_to_status(error_code: int) -> MotorStatus:
    """Map a MyActuator status-1 error bitmask to a MotorStatus (first match by severity)."""
    if error_code == 0:
        return MotorStatus.OK
    for bit, status in _MA_ERROR_MAP:
        if error_code & bit:
            return status
    return MotorStatus.UNKNOWN


def _model_max_torque(model: str | None) -> float:
    """Return the motor's max torque (Nm) inferred from its 0xB5 model string.

    The model string contains an ``X<n>`` series token (e.g. ``"RMD-X8-P20"`` ->
    ``8``); ``n`` selects the rated max torque from ``_MA_SERIES_MAX_TORQUE``.
    Falls back to ``_MA_DEFAULT_MAX_TORQUE`` for unknown or unreadable models.
    """
    if model:
        match = re.search(r"X(\d+)", model.strip().upper())
        if match is not None:
            return _MA_SERIES_MAX_TORQUE.get(
                int(match.group(1)), _MA_DEFAULT_MAX_TORQUE
            )
    return _MA_DEFAULT_MAX_TORQUE


class MyActuatorMotor(MotorDriver):
    """MotorDriver implementation for MyActuator RMD motors using the 0x140-series protocol."""

    MOTOR_TYPE = "myactuator"
    PARAMS = MYACTUATOR_PARAMS
    # The 0xC0 table is still incomplete, so a raw sweep has something to find.
    PARAM_SWEEP_RANGE = PARAM_SWEEP_RANGE

    def __init__(self, bus: CanBus, motor_id: int, kt: float) -> None:
        """Construct a MyActuator driver.

        Args:
            bus:      Shared CAN bus.
            motor_id: Motor CAN ID; request/response arbitration IDs are derived
                      from it (0x140/0x240/0x400/0x500 + motor_id).
            kt:       Torque constant (Nm/A) used to convert current to torque.
        """
        self._bus = bus
        self._motor_id = motor_id
        self._kt = kt
        self._pending: dict[tuple[int, int], asyncio.Future[bytes]] = {}
        self._on_feedback: Callable[[float, float], None] | None = None

        # Firmware-dependent MIT-command ranges. Default to the conservative
        # legacy ranges until enable() reads the firmware version and model.
        self._fw_version: int | None = None
        self._model: str | None = None
        self._p_max = _MA_P_MAX_LEGACY
        self._kd_max = _MA_KD_MAX_LEGACY
        self._t_max = _MA_T_MAX_LEGACY  # range for t_ff and feedback torque
        self._max_torque = _MA_DEFAULT_MAX_TORQUE  # motor's rated max torque (Nm)

        bus._add_listener(self._on_message)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _on_message(self, msg: can.Message) -> None:
        mc_resp_id = _MA_MC_RESP + self._motor_id
        if msg.arbitration_id == mc_resp_id:
            if self._on_feedback is not None:
                data = bytes(msg.data)
                pos_int = (data[1] << 8) | data[2]
                torq_int = ((data[4] & 0x0F) << 8) | data[5]
                position = _uint_to_float(pos_int, -self._p_max, self._p_max, 16)
                torque = _uint_to_float(torq_int, -self._t_max, self._t_max, 12)
                self._on_feedback(position, torque)
            return
        key = (msg.arbitration_id, msg.data[0])
        fut = self._pending.pop(key, None)
        if fut is not None and not fut.done():
            fut.set_result(bytes(msg.data))

    async def _request(self, data: bytes, timeout: float = 0.1) -> bytes:
        resp_id = _MA_RESP + self._motor_id
        key = (resp_id, data[0])
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bytes] = loop.create_future()
        self._pending[key] = fut
        await self._bus._send(_MA_REQ + self._motor_id, data)
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            raise MotorError(
                f"MyActuator motor {self._motor_id:#04x} did not respond within {timeout}s"
            )

    @staticmethod
    def _cmd(byte: int) -> bytes:
        return bytes([byte, 0, 0, 0, 0, 0, 0, 0])

    async def _get_status1(self) -> bytes:
        """Request status frame 1 (temperature, voltage, error flags)."""
        return await self._request(self._cmd(_MA_READ_STATUS1))

    async def _running_state(self) -> tuple[bool, int]:
        """Return ``(running, error_bits)`` from a status-1 read.

        ``running`` is status-1 byte 3. The protocol labels it the brake
        release state; observed on fleet firmware (2025070202) it reads 1
        only while the motor is actively executing commands — 0 when
        disabled, freshly enabled but never commanded, or just reset — which
        makes it exactly the "enabled and holding" signal attach/is_holding
        need.
        """
        resp = await self._get_status1()
        error_bits = struct.unpack_from("<H", resp, 6)[0]
        return resp[3] == 0x01, int(error_bits)

    async def _get_status2(self) -> bytes:
        """Request status frame 2 (temperature, current, velocity, encoder)."""
        return await self._request(self._cmd(_MA_MOTOR_STATUS_2))

    async def _read_firmware_version(self) -> int:
        """Read the firmware VersionDate (uint32, e.g. 2026042402) via 0xB2."""
        resp = await self._request(self._cmd(_MA_READ_VERSION))
        return int(struct.unpack_from("<I", resp, 4)[0])

    async def _read_model(self) -> str:
        """Read the first 10 characters of the motor model string via 0xB5.

        The full model is up to 15 characters (read 5 at a time by index).
        Two blocks are read so that the ``X<series>`` digit (e.g. ``"X6"`` in
        ``"RMD-X6-P20"``) is always captured regardless of its position.
        """
        block1 = await self._request(bytes([_MA_READ_MODEL, 0x01, 0x01, 0, 0, 0, 0, 0]))
        block2 = await self._request(bytes([_MA_READ_MODEL, 0x01, 0x02, 0, 0, 0, 0, 0]))
        raw = bytes(block1[3:8]) + bytes(block2[3:8])
        chars = raw.split(b"\x00", 1)[0]
        return chars.decode("ascii", errors="ignore")

    async def _read_param(self, index: int) -> float:
        """Read configuration parameter ``index`` via 0xC0.

        Every parameter shares one response CAN ID and command byte, so
        concurrent reads on the same motor would race — keep them sequential.
        """
        data = bytes([_MA_PARAM_ACCESS, 0x00, index, _MA_PARAM_READ, 0, 0, 0, 0])
        resp = await self._request(data)
        return float(struct.unpack_from("<f", resp, 4)[0])

    async def _write_param(
        self, index: int, value: float, *, commit: bool = True
    ) -> None:
        """Write configuration parameter ``index`` via 0xC0.

        A 0xC0 write only lands in RAM. Pass ``commit=False`` to defer the 0xC1
        that flushes it to ROM, so a batch of writes costs one ROM cycle
        instead of one per parameter.
        """
        data = bytes([_MA_PARAM_ACCESS, 0x00, index, _MA_PARAM_WRITE]) + struct.pack(
            "<f", value
        )
        await self._request(data)
        if commit:
            await self._commit_params()

    async def _commit_params(self) -> None:
        """Commit every 0xC0 write made since the last commit to ROM."""
        await self._request(self._cmd(_MA_PARAM_SAVE))

    async def _apply_low_voltage_threshold(self) -> None:
        """Bring the undervoltage threshold to :data:`_MA_LOW_VOLTAGE_THRESHOLD_V`.

        Reads first so an already-provisioned motor is left untouched. A motor
        that doesn't answer keeps whatever threshold it has — that only costs
        the (optional) undervoltage suppression, so warn rather than fail the
        whole enable().
        """
        try:
            current = await self.get_low_voltage_threshold()
            if math.isclose(
                current, _MA_LOW_VOLTAGE_THRESHOLD_V, abs_tol=_MA_LOW_VOLTAGE_TOL_V
            ):
                return
            await self.set_low_voltage_threshold(_MA_LOW_VOLTAGE_THRESHOLD_V)
        except MotorError as exc:
            _logger.warning(
                "MyActuator motor %#04x: could not set undervoltage threshold to %.1f V (%s)",
                self._motor_id,
                _MA_LOW_VOLTAGE_THRESHOLD_V,
                exc,
            )

    async def _detect_capabilities(self) -> None:
        """Read firmware version + model once and configure MIT-command ranges.

        Cached after the first success; raises MotorError if the motor doesn't
        answer (callers fall back to the conservative legacy ranges).
        """
        if self._fw_version is not None:
            return
        version = await self._read_firmware_version()
        model = await self._read_model()
        self._fw_version = version
        self._model = model
        self._max_torque = _model_max_torque(model)
        if version >= _MA_FW_V44_VERSION:
            self._p_max = _MA_P_MAX_V44
            self._kd_max = _MA_KD_MAX_V44
            self._t_max = self._max_torque
        else:
            self._p_max = _MA_P_MAX_LEGACY
            self._kd_max = _MA_KD_MAX_LEGACY
            self._t_max = _MA_T_MAX_LEGACY

    # ------------------------------------------------------------------ #
    # Public API (implements MotorDriver)                                  #
    # ------------------------------------------------------------------ #

    async def enable(self) -> None:
        # Detect firmware version + model so the MIT command and feedback decode
        # use the ranges this motor's firmware actually implements. If the motor
        # doesn't answer, keep the conservative legacy ranges set in __init__.
        try:
            await self._detect_capabilities()
        except MotorError:
            pass
        await self._apply_low_voltage_threshold()
        await self._request(self._cmd(_MA_RELEASE_BRAKE))

    async def attach(self) -> None:
        # Reads only. The capability detection must succeed here (unlike
        # enable(), which tolerates a silent motor still booting): a motor
        # that doesn't answer can't be verified as holding. Crucially there
        # is no 0x76 reset (that reboots the motor, killing torque for ~2 s)
        # and no 0x77 brake release (that would drop an arm parked on its
        # brake) — the whole point of attach is to leave torque untouched.
        await self._detect_capabilities()
        running, error_bits = await self._running_state()
        if error_bits:
            raise MotorError(
                f"MyActuator motor {self._motor_id:#04x} has a latched fault "
                f"({_ma_error_to_status(error_bits).value}) — attach is not "
                f"possible; use enable() for a full bring-up"
            )
        if not running:
            raise MotorError(
                f"MyActuator motor {self._motor_id:#04x} is not running "
                f"(not enabled and holding) — attach is not possible; "
                f"use enable() for a full bring-up"
            )

    async def is_holding(self) -> bool:
        running, error_bits = await self._running_state()
        return running and not error_bits

    async def get_firmware_version(self) -> int | None:
        return await self._read_firmware_version()

    async def get_model(self) -> str | None:
        return await self._read_model()

    async def disable(self) -> None:
        await self._request(self._cmd(_MA_SHUTDOWN))

    async def set_control_mode(self, mode: ControlMode) -> None:
        # MyActuator has no persistent control mode register; the active mode is
        # determined by which command is sent. Reset the motor to clear internal
        # state so it comes back ready for the next command type.
        await self._bus._send(_MA_REQ + self._motor_id, self._cmd(_MA_RESET))
        await asyncio.sleep(_MA_RESET_SETTLE_S)

    async def clear_errors(self) -> None:
        pass  # MyActuator has no clear-errors command

    async def set_zero_position(self) -> None:
        await self._request(self._cmd(_MA_SET_ENCODER_ZERO))
        await self._bus._send(_MA_REQ + self._motor_id, self._cmd(_MA_RESET))
        await asyncio.sleep(_MA_RESET_SETTLE_S)

    async def get_position(self) -> float:
        resp = await self._request(self._cmd(_MA_MULTI_TURN_ANGLE))
        raw = struct.unpack_from("<i", resp, 4)[0]
        return raw * (0.01 * math.pi / 180.0)  # 0.01 deg/LSB → rad

    async def get_velocity(self) -> float:
        resp = await self._get_status2()
        speed_dps = struct.unpack_from("<h", resp, 4)[0]
        return speed_dps * (math.pi / 180.0)  # dps → rad/s

    async def get_torque(self) -> float:
        resp = await self._get_status2()
        return struct.unpack_from("<h", resp, 2)[0] * 0.01 * self._kt  # 0.01 A/LSB → Nm

    async def get_telemetry(
        self,
        on_position: Callable[[float], None],
        on_torque: Callable[[float], None] | None = None,
    ) -> None:
        def _on_pos(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception() is None:
                on_position(task.result())

        pos_task = asyncio.create_task(self.get_position())
        pos_task.add_done_callback(_on_pos)
        tasks: list[asyncio.Task] = [pos_task]

        if on_torque is not None:

            def _on_torq(task: asyncio.Task) -> None:
                if not task.cancelled() and task.exception() is None:
                    on_torque(task.result())

            torq_task = asyncio.create_task(self.get_torque())
            torq_task.add_done_callback(_on_torq)
            tasks.append(torq_task)

        await asyncio.gather(*tasks, return_exceptions=True)

    async def get_temperature(self) -> float:
        resp = await self._get_status2()
        return float(struct.unpack_from("b", resp, 1)[0])  # int8 °C

    async def get_voltage(self) -> float:
        resp = await self._get_status1()
        raw = struct.unpack_from("<H", resp, 4)[0]
        return raw * 0.1  # 0.1 V/LSB

    async def get_low_voltage_threshold(self) -> float:
        return await self._read_param(MyActuatorParam.LOW_VOLTAGE)

    async def set_low_voltage_threshold(self, volts: float) -> None:
        await self._write_param(MyActuatorParam.LOW_VOLTAGE, volts)

    async def _config_read(self, index: int) -> float:
        return await self._read_param(index)

    async def _config_write(self, index: int, value: float) -> None:
        await self._write_param(index, value, commit=False)

    async def _config_commit(self) -> None:
        await self._commit_params()

    async def get_error_code(self) -> MotorStatus:
        resp = await self._get_status1()
        error_bits = struct.unpack_from("<H", resp, 6)[0]
        return _ma_error_to_status(error_bits)

    async def set_position_velocity(self, position: float, max_speed: float) -> None:
        # bytes 2-3: uint16 max speed in dps; bytes 4-7: int32 position in 0.01 degree units
        speed_dps = int(max_speed * (180.0 / math.pi))
        pos_centideg = int(position * (18000.0 / math.pi))  # rad → 0.01 deg units
        data = (
            bytes([_MA_POS_CONTROL, 0x00])
            + struct.pack("<H", speed_dps)
            + struct.pack("<i", pos_centideg)
        )
        await self._request(data)

    async def set_velocity(self, velocity: float) -> None:
        # bytes 4-7: int32 in centidps (dps × 100); rad/s → dps → centidps
        centidps = int(velocity * (18000.0 / math.pi))
        data = bytes([_MA_VELOCITY_CONTROL, 0, 0, 0]) + struct.pack("<i", centidps)
        await self._request(data)

    async def set_acceleration(
        self, acceleration: float, deceleration: float | None = None
    ) -> None:
        # Command 0x43 writes to both RAM and ROM — no separate store step needed.
        dec = deceleration if deceleration is not None else acceleration

        async def _send(accel_type: int, value_rad_s2: float) -> None:
            dps_s2 = max(
                _MA_ACC_MIN_DPS_S2,
                min(_MA_ACC_MAX_DPS_S2, int(value_rad_s2 * (180.0 / math.pi))),
            )
            data = bytes([_MA_SET_ACCELERATION, accel_type, 0, 0]) + struct.pack(
                "<I", dps_s2
            )
            await self._request(data)

        # All four types share the same response CAN ID — must be sequential.
        await _send(_MA_ACC_POS_PLAN, acceleration)
        await _send(_MA_DEC_POS_PLAN, dec)
        await _send(_MA_ACC_VEL_PLAN, acceleration)
        await _send(_MA_DEC_VEL_PLAN, dec)

    async def get_gains(self) -> MotorGains:
        resp = await self._request(self._cmd(_MA_READ_GAINS))
        # Response bytes 2-7: current_kp, current_ki, speed_kp, speed_ki, pos_kp, pos_ki (uint8)
        return MotorGains(
            speed_kp=float(resp[4]),
            speed_ki=float(resp[5]),
            position_kp=float(resp[6]),
            position_ki=float(resp[7]),
            current_kp=float(resp[2]),
            current_ki=float(resp[3]),
        )

    async def set_gains(self, gains: MotorGains) -> None:
        # Command 0x32 writes directly to ROM — no separate store step needed.
        current_kp = int(max(0, min(255, gains.current_kp or 0)))
        current_ki = int(max(0, min(255, gains.current_ki or 0)))
        speed_kp = int(max(0, min(255, gains.speed_kp)))
        speed_ki = int(max(0, min(255, gains.speed_ki)))
        pos_kp = int(max(0, min(255, gains.position_kp)))
        pos_ki = int(max(0, min(255, gains.position_ki)))
        # SDK byte layout: [cmd, 0, cur_kp, cur_ki, spd_kp, spd_ki, pos_kp, pos_ki]
        data = bytes(
            [
                _MA_WRITE_GAINS_ROM,
                0,
                current_kp,
                current_ki,
                speed_kp,
                speed_ki,
                pos_kp,
                pos_ki,
            ]
        )
        await self._request(data)

    async def set_can_id(self, can_id: int) -> None:
        # no response; motor must be reset for the new ID to take effect
        data = bytes(
            [
                _MA_FUNCTION_CONTROL,
                _MA_FC_SET_CANID,
                0x00,
                0x00,
                can_id & 0xFF,
                (can_id >> 8) & 0xFF,
                0x00,
                0x00,
            ]
        )
        await self._bus._send(_MA_REQ + self._motor_id, data)
        await asyncio.sleep(0.1)
        await self._bus._send(_MA_REQ + self._motor_id, self._cmd(_MA_RESET))
        await asyncio.sleep(_MA_RESET_SETTLE_S)
        self._motor_id = can_id

    async def set_impedance(
        self,
        p_des: float,
        v_des: float,
        kp: float,
        kd: float,
        t_ff: float,
    ) -> None:
        p_u = _float_to_uint(p_des, -self._p_max, self._p_max, 16)
        v_u = _float_to_uint(v_des, -_MA_V_MAX, _MA_V_MAX, 12)
        kp_u = _float_to_uint(kp, 0.0, _MA_KP_MAX, 12)
        kd_u = _float_to_uint(kd, 0.0, self._kd_max, 12)
        t_u = _float_to_uint(t_ff, -self._t_max, self._t_max, 12)

        data = bytes(
            [
                (p_u >> 8) & 0xFF,
                p_u & 0xFF,
                (v_u >> 4) & 0xFF,
                ((v_u & 0xF) << 4) | ((kp_u >> 8) & 0xF),
                kp_u & 0xFF,
                (kd_u >> 4) & 0xFF,
                ((kd_u & 0xF) << 4) | ((t_u >> 8) & 0xF),
                t_u & 0xFF,
            ]
        )

        await self._bus._send(_MA_MC_REQ + self._motor_id, data)
