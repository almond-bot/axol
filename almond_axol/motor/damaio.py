# Protocol source: python-damiao-driver/damiao_motor/motor.py

from __future__ import annotations

import asyncio
import math
import struct
from dataclasses import dataclass
from enum import Enum

import can

from .bus import CanBus
from .driver import MotorDriver
from .errors import MotorError


class _ControlMode(Enum):
    MIT       = "mit"
    POS_VEL   = "pos_vel"
    VEL       = "vel"
    FORCE_POS = "force_pos"


class _DamaioStatus(Enum):
    DISABLED        = 0x0
    ENABLED         = 0x1
    OVER_VOLTAGE    = 0x8
    UNDER_VOLTAGE   = 0x9
    OVER_CURRENT    = 0xA
    MOS_OVER_TEMP   = 0xB
    ROTOR_OVER_TEMP = 0xC
    LOST_COMM       = 0xD
    OVERLOAD        = 0xE


@dataclass
class _MotorFeedback:
    status:   _DamaioStatus
    position: float  # rad (motor internal)
    velocity: float  # rad/s
    torque:   float  # Nm
    t_mos:    float  # °C
    t_rotor:  float  # °C


_DM_UINT32_REGS = {7, 8, 9, 10, 13, 14, 15, 16, 35, 36}
_DM_REG_PMAX    = 21
_DM_REG_VMAX    = 22
_DM_REG_TMAX    = 23


def _float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    x = max(x_min, min(x_max, x))
    return int((x - x_min) * ((1 << bits) - 1) / (x_max - x_min))


def _uint_to_float(x_int: int, x_min: float, x_max: float, bits: int) -> float:
    return float(x_int) * (x_max - x_min) / ((1 << bits) - 1) + x_min


class DamaioMotor(MotorDriver):
    def __init__(self, bus: CanBus, motor_id: int, feedback_id: int) -> None:
        self._bus = bus
        self._motor_id = motor_id
        self._feedback_id = feedback_id

        self._feedback: _MotorFeedback | None = None
        self._registers: dict[int, float | int] = {}
        self._feedback_waiters: list[asyncio.Future[_MotorFeedback]] = []
        self._register_waiters: dict[int, asyncio.Future[float | int]] = {}

        self._p_max = 12.5
        self._v_max = 45.0
        self._t_max = 18.0

        bus._add_listener(self._on_message)

    def _on_message(self, msg: can.Message) -> None:
        if len(msg.data) != 8:
            return
        data = bytes(msg.data)

        if data[2] == 0x33 and data[3] <= 81:
            if (data[0] | (data[1] << 8)) == self._motor_id:
                self._handle_register_reply(data)
            return

        if (data[0] & 0x0F) == (self._motor_id & 0x0F):
            self._handle_feedback(data)

    def _handle_register_reply(self, data: bytes) -> None:
        rid = data[3]
        value: float | int
        if rid in _DM_UINT32_REGS:
            value = struct.unpack("<I", data[4:8])[0]
        else:
            value = struct.unpack("<f", data[4:8])[0]
        self._registers[rid] = value

        fut = self._register_waiters.pop(rid, None)
        if fut is not None and not fut.done():
            fut.set_result(value)

    def _handle_feedback(self, data: bytes) -> None:
        status_code = data[0] >> 4
        pos_int     = (data[1] << 8) | data[2]
        vel_int     = (data[3] << 4) | (data[4] >> 4)
        torq_int    = ((data[4] & 0xF) << 8) | data[5]

        try:
            status = _DamaioStatus(status_code)
        except ValueError:
            status = _DamaioStatus.DISABLED

        self._feedback = _MotorFeedback(
            status=  status,
            position=_uint_to_float(pos_int,  -self._p_max, self._p_max, 16),
            velocity=_uint_to_float(vel_int,  -self._v_max, self._v_max, 12),
            torque=  _uint_to_float(torq_int, -self._t_max, self._t_max, 12),
            t_mos=   float(data[6]),
            t_rotor= float(data[7]),
        )

        for fut in self._feedback_waiters:
            if not fut.done():
                fut.set_result(self._feedback)
        self._feedback_waiters.clear()

    def _canid_bytes(self) -> tuple[int, int]:
        return self._motor_id & 0xFF, (self._motor_id >> 8) & 0xFF

    async def _raw_send(self, data: bytes, arb_id: int | None = None) -> None:
        await self._bus._send(arb_id if arb_id is not None else self._motor_id, data)

    async def _read_register(self, rid: int, timeout: float = 0.2) -> float | int:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[float | int] = loop.create_future()
        self._register_waiters[rid] = fut
        canid_l, canid_h = self._canid_bytes()
        await self._bus._send(0x7FF, bytes([canid_l, canid_h, 0x33, rid, 0, 0, 0, 0]))
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            raise MotorError(f"Damaio motor {self._motor_id:#04x} register {rid} read timed out")

    async def _request_feedback(self, timeout: float = 0.1) -> _MotorFeedback:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[_MotorFeedback] = loop.create_future()
        self._feedback_waiters.append(fut)
        canid_l, canid_h = self._canid_bytes()
        await self._bus._send(0x7FF, bytes([canid_l, canid_h, 0xCC, 0, 0, 0, 0, 0]))
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            raise MotorError(f"Damaio motor {self._motor_id:#04x} feedback timed out")

    async def enable(self) -> None:
        pmax, vmax, tmax = await asyncio.gather(
            self._read_register(_DM_REG_PMAX),
            self._read_register(_DM_REG_VMAX),
            self._read_register(_DM_REG_TMAX),
        )
        self._p_max = float(pmax)
        self._v_max = float(vmax)
        self._t_max = float(tmax)
        await self._raw_send(bytes([0xFF] * 7 + [0xFC]))

    async def disable(self) -> None:
        await self._raw_send(bytes([0xFF] * 7 + [0xFD]))

    async def clear_errors(self) -> None:
        await self._raw_send(bytes([0xFF] * 7 + [0xFB]))

    async def get_position(self) -> float:
        feedback = await self._request_feedback()
        return feedback.position / (2 * math.pi)

    async def set_control_mode(self, mode: _ControlMode) -> None:
        canid_l, canid_h = self._canid_bytes()
        data = struct.pack("<I", list(_ControlMode).index(mode) + 1)
        await self._bus._send(0x7FF, bytes([canid_l, canid_h, 0x55, 10]) + data)

    async def send_cmd(
        self,
        target_position: float = 0.0,
        target_velocity: float = 0.0,
        stiffness: float = 0.0,
        damping: float = 0.0,
        feedforward_torque: float = 0.0,
        control_mode: _ControlMode = _ControlMode.MIT,
        velocity_limit: float = 0.0,
        current_limit: float = 0.0,
    ) -> None:
        if control_mode == _ControlMode.MIT:
            pos_u  = _float_to_uint(target_position,    -self._p_max, self._p_max, 16)
            vel_u  = _float_to_uint(target_velocity,    -self._v_max, self._v_max, 12)
            kp_u   = _float_to_uint(stiffness,          0.0, 500.0, 12)
            kd_u   = _float_to_uint(damping,            0.0,   5.0, 12)
            torq_u = _float_to_uint(feedforward_torque, -self._t_max, self._t_max, 12)
            data = bytes([
                (pos_u >> 8) & 0xFF,
                pos_u & 0xFF,
                (vel_u >> 4) & 0xFF,
                ((vel_u & 0xF) << 4) | ((kp_u >> 8) & 0xF),
                kp_u & 0xFF,
                (kd_u >> 4) & 0xFF,
                ((kd_u & 0xF) << 4) | ((torq_u >> 8) & 0xF),
                torq_u & 0xFF,
            ])
            await self._raw_send(data)
        elif control_mode == _ControlMode.POS_VEL:
            data = struct.pack("<ff", target_position, target_velocity)
            await self._raw_send(data, arb_id=0x100 + self._motor_id)
        elif control_mode == _ControlMode.VEL:
            data = struct.pack("<f", target_velocity) + b"\x00" * 4
            await self._raw_send(data, arb_id=0x200 + self._motor_id)
        elif control_mode == _ControlMode.FORCE_POS:
            v_scaled = int(max(0.0, min(100.0, velocity_limit)) * 100)
            i_scaled = int(max(0.0, min(1.0,   current_limit))  * 10000)
            data = struct.pack("<fHH", target_position, v_scaled, i_scaled)
            await self._raw_send(data, arb_id=0x300 + self._motor_id)
