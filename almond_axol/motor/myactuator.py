# Protocol source: myactuator_rmd/include/myactuator_rmd/driver/can_address_offset.hpp
#                  myactuator_rmd/include/myactuator_rmd/protocol/command_type.hpp
#                  myactuator_rmd/include/myactuator_rmd/protocol/motion_control_request.hpp
#                  myactuator_rmd/include/myactuator_rmd/actuator_state/error_code.hpp
#                  myactuator_rmd/include/myactuator_rmd/actuator_state/acceleration_type.hpp

from __future__ import annotations

import asyncio
import math
import struct
from typing import Callable

import can

from .bus import CanBus
from .driver import MotorDriver
from .errors import MotorError
from .types import ControlMode, MotorGains, MotorStatus

_MA_REQ = 0x140  # standard command request  → 0x140 + motor_id
_MA_RESP = 0x240  # standard command response ← 0x240 + motor_id
_MA_MC_REQ = 0x400  # motion control request    → 0x400 + motor_id
_MA_MC_RESP = 0x500  # motion control response   ← 0x500 + motor_id

_MA_SHUTDOWN = 0x80
_MA_RELEASE_BRAKE = 0x77
_MA_RESET = 0x76  # system reset; no response — motor restarts immediately
_MA_READ_STATUS1 = 0x9A  # temperature, voltage, error flags
_MA_MULTI_TURN_ANGLE = 0x92
_MA_MOTOR_STATUS_2 = 0x9C  # temperature, current, velocity, encoder
_MA_SET_ENCODER_ZERO = 0x64
_MA_POS_CONTROL = 0xA4  # absolute position closed-loop control
_MA_VELOCITY_CONTROL = 0xA2  # speed closed-loop control
_MA_FUNCTION_CONTROL = 0x20  # function control; byte 1 = index, bytes 4-7 = value
_MA_FC_SET_CANID = 0x05  # function control index: set CAN ID
_MA_SET_CAN_BAUD = 0xB4  # change CAN baud rate; no response — motor restarts
_MA_READ_GAINS = 0x30  # read all PID gains (uint8, bulk)
_MA_WRITE_GAINS_ROM = (
    0x32  # write all PID gains to ROM (uint8, bulk); persistent by command
)
_MA_SET_ACCELERATION = 0x43  # write acceleration to RAM and ROM; persistent by command

# AccelerationType enum values from acceleration_type.hpp
_MA_ACC_POS_PLAN = 0x00  # position planning acceleration
_MA_DEC_POS_PLAN = 0x01  # position planning deceleration
_MA_ACC_VEL_PLAN = 0x02  # velocity planning acceleration
_MA_DEC_VEL_PLAN = 0x03  # velocity planning deceleration

_MA_P_MIN, _MA_P_MAX = -12.5, 12.5  # rad
_MA_V_MIN, _MA_V_MAX = -45.0, 45.0  # rad/s
_MA_T_MIN, _MA_T_MAX = -24.0, 24.0  # Nm
_MA_KP_MIN, _MA_KP_MAX = 0.0, 500.0
_MA_KD_MIN, _MA_KD_MAX = 0.0, 5.0

_MA_BAUD_MAP: dict[int, int] = {
    500_000: 0,
    1_000_000: 1,
}

# Acceleration range in dps/s²; [100, 60000] — 0 disables ramping.
_MA_ACC_MIN_DPS_S2 = 100
_MA_ACC_MAX_DPS_S2 = 60000

# Error bitmask → MotorStatus, ordered by severity (first match wins).
_MA_ERROR_MAP: list[tuple[int, MotorStatus]] = [
    (0x0002, MotorStatus.MOTOR_STALL),
    (0x0004, MotorStatus.UNDER_VOLTAGE),
    (0x0008, MotorStatus.OVER_VOLTAGE),
    (0x0010, MotorStatus.OVER_CURRENT),
    (0x0040, MotorStatus.POWER_OVERRUN),
    (0x0100, MotorStatus.SPEEDING),
    (0x1000, MotorStatus.OVER_TEMPERATURE),
    (0x2000, MotorStatus.ENCODER_ERROR),
]


def _float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    x = max(x_min, min(x_max, x))
    return int((x - x_min) * ((1 << bits) - 1) / (x_max - x_min))


def _ma_error_to_status(error_code: int) -> MotorStatus:
    if error_code == 0:
        return MotorStatus.OK
    for bit, status in _MA_ERROR_MAP:
        if error_code & bit:
            return status
    return MotorStatus.UNKNOWN


class MyActuatorMotor(MotorDriver):
    def __init__(self, bus: CanBus, motor_id: int) -> None:
        self._bus = bus
        self._motor_id = motor_id
        self._pending: dict[tuple[int, int], asyncio.Future[bytes]] = {}
        bus._add_listener(self._on_message)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _on_message(self, msg: can.Message) -> None:
        # Motion control responses (0x500+id): data[0] is the echoed actuator CAN ID,
        # not a command byte — key by (arb_id, motor_id) rather than (arb_id, data[0]).
        mc_resp_id = _MA_MC_RESP + self._motor_id
        if msg.arbitration_id == mc_resp_id:
            key: tuple[int, int] = (mc_resp_id, self._motor_id)
        else:
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

    async def _get_status2(self) -> bytes:
        """Request status frame 2 (temperature, current, velocity, encoder)."""
        return await self._request(self._cmd(_MA_MOTOR_STATUS_2))

    # ------------------------------------------------------------------ #
    # Public API (implements MotorDriver)                                  #
    # ------------------------------------------------------------------ #

    async def enable(self) -> None:
        await self._request(self._cmd(_MA_RELEASE_BRAKE))

    async def disable(self) -> None:
        await self._request(self._cmd(_MA_SHUTDOWN))

    async def set_control_mode(self, mode: ControlMode) -> None:
        # MyActuator has no persistent control mode register; the active mode is
        # determined by which command is sent.
        pass

    async def clear_errors(self) -> None:
        pass  # MyActuator has no clear-errors command

    async def set_zero_position(self) -> None:
        await self._request(self._cmd(_MA_SET_ENCODER_ZERO))
        await self._bus._send(_MA_REQ + self._motor_id, self._cmd(_MA_RESET))
        await asyncio.sleep(0.5)

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
        return struct.unpack_from("<h", resp, 2)[0] * 0.01  # 0.01 A/LSB

    async def get_telemetry(
        self,
        on_position: Callable[[float], None],
        on_torque: Callable[[float], None],
    ) -> None:
        def _on_pos(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception() is None:
                on_position(task.result())

        def _on_torq(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception() is None:
                on_torque(task.result())

        pos_task = asyncio.create_task(self.get_position())
        torq_task = asyncio.create_task(self.get_torque())
        pos_task.add_done_callback(_on_pos)
        torq_task.add_done_callback(_on_torq)
        await asyncio.gather(pos_task, torq_task, return_exceptions=True)

    async def get_temperature(self) -> float:
        resp = await self._get_status2()
        return float(struct.unpack_from("b", resp, 1)[0])  # int8 °C

    async def get_voltage(self) -> float:
        resp = await self._get_status1()
        raw = struct.unpack_from("<H", resp, 4)[0]
        return raw * 0.1  # 0.1 V/LSB

    async def get_error_code(self) -> MotorStatus:
        resp = await self._get_status1()
        error_bits = struct.unpack_from("<H", resp, 6)[0]
        return _ma_error_to_status(error_bits)

    async def set_position(self, position: float, max_speed: float) -> None:
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
        await asyncio.sleep(0.5)
        self._motor_id = can_id

    async def set_can_baud_rate(self, baud_rate: int) -> None:
        code = _MA_BAUD_MAP.get(baud_rate)
        if code is None:
            raise MotorError(
                f"Unsupported baud rate {baud_rate}. Supported: {sorted(_MA_BAUD_MAP)}"
            )
        data = bytes([_MA_SET_CAN_BAUD, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, code])
        await self._bus._send(_MA_REQ + self._motor_id, data)

    async def motion_control(
        self,
        p_des: float,
        v_des: float,
        kp: float,
        kd: float,
        t_ff: float,
        timeout: float = 0.05,
    ) -> None:
        p_u = _float_to_uint(p_des, _MA_P_MIN, _MA_P_MAX, 16)
        v_u = _float_to_uint(v_des, _MA_V_MIN, _MA_V_MAX, 12)
        kp_u = _float_to_uint(kp, _MA_KP_MIN, _MA_KP_MAX, 12)
        kd_u = _float_to_uint(kd, _MA_KD_MIN, _MA_KD_MAX, 12)
        t_u = _float_to_uint(t_ff, _MA_T_MIN, _MA_T_MAX, 12)

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

        resp_id = _MA_MC_RESP + self._motor_id
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bytes] = loop.create_future()
        self._pending[(resp_id, self._motor_id)] = fut
        await self._bus._send(_MA_MC_REQ + self._motor_id, data)
        try:
            await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            raise MotorError(
                f"MyActuator motor {self._motor_id:#04x} motion control timed out"
            )
