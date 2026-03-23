# Protocol source: myactuator_rmd/include/myactuator_rmd/driver/can_address_offset.hpp
#                  myactuator_rmd/include/myactuator_rmd/protocol/command_type.hpp
#                  myactuator_rmd/include/myactuator_rmd/protocol/motion_control_request.hpp

from __future__ import annotations

import asyncio
import struct

import can

from .bus import CanBus
from .driver import _MotorDriver
from .errors import MotorError

_MA_REQ     = 0x140  # standard command request  → 0x140 + motor_id
_MA_RESP    = 0x240  # standard command response ← 0x240 + motor_id
_MA_MC_REQ  = 0x400  # motion control request    → 0x400 + motor_id
_MA_MC_RESP = 0x500  # motion control response   ← 0x500 + motor_id

_MA_SHUTDOWN         = 0x80
_MA_STOP             = 0x81
_MA_RELEASE_BRAKE    = 0x77
_MA_MULTI_TURN_ANGLE = 0x92

_MA_P_MIN,  _MA_P_MAX  = -12.5, 12.5   # rad
_MA_V_MIN,  _MA_V_MAX  = -45.0, 45.0   # rad/s
_MA_T_MIN,  _MA_T_MAX  = -24.0, 24.0   # Nm
_MA_KP_MIN, _MA_KP_MAX =   0.0, 500.0
_MA_KD_MIN, _MA_KD_MAX =   0.0,   5.0


def _float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    x = max(x_min, min(x_max, x))
    return int((x - x_min) * ((1 << bits) - 1) / (x_max - x_min))


class _MyActuatorMotor(_MotorDriver):
    def __init__(self, bus: CanBus, motor_id: int) -> None:
        self._bus = bus
        self._motor_id = motor_id
        self._pending: dict[int, asyncio.Future[bytes]] = {}
        bus._add_listener(self._on_message)

    def _on_message(self, msg: can.Message) -> None:
        fut = self._pending.pop(msg.arbitration_id, None)
        if fut is not None and not fut.done():
            fut.set_result(bytes(msg.data))

    async def _request(self, data: bytes, timeout: float = 0.1) -> bytes:
        resp_id = _MA_RESP + self._motor_id
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bytes] = loop.create_future()
        self._pending[resp_id] = fut
        await self._bus._send(_MA_REQ + self._motor_id, data)
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            raise MotorError(f"MyActuator motor {self._motor_id:#04x} did not respond within {timeout}s")

    @staticmethod
    def _cmd(byte: int) -> bytes:
        return bytes([byte, 0, 0, 0, 0, 0, 0, 0])

    async def enable(self) -> None:
        await self._request(self._cmd(_MA_RELEASE_BRAKE))

    async def disable(self) -> None:
        await self._request(self._cmd(_MA_SHUTDOWN))

    async def clear_errors(self) -> None:
        await self._request(self._cmd(_MA_STOP))

    async def get_position(self) -> float:
        resp = await self._request(self._cmd(_MA_MULTI_TURN_ANGLE))
        raw = struct.unpack_from("<i", resp, 4)[0]
        degrees = raw * 0.01  # raw is in 0.01 degree units
        return degrees / 360.0

    async def motion_control(
        self,
        p_des: float,
        v_des: float,
        kp: float,
        kd: float,
        t_ff: float,
        timeout: float = 0.05,
    ) -> None:
        """
        MIT-style impedance control (CAN ID 0x400 + motor_id).

        Args:
            p_des: Desired position  (rad,  [-12.5, 12.5])
            v_des: Desired velocity  (rad/s, [-45,  45])
            kp:    Position stiffness        [0, 500]
            kd:    Velocity damping          [0,   5]
            t_ff:  Feedforward torque (Nm,  [-24,  24])
        """
        p_u  = _float_to_uint(p_des, _MA_P_MIN,  _MA_P_MAX,  16)
        v_u  = _float_to_uint(v_des, _MA_V_MIN,  _MA_V_MAX,  12)
        kp_u = _float_to_uint(kp,   _MA_KP_MIN, _MA_KP_MAX, 12)
        kd_u = _float_to_uint(kd,   _MA_KD_MIN, _MA_KD_MAX, 12)
        t_u  = _float_to_uint(t_ff, _MA_T_MIN,  _MA_T_MAX,  12)

        data = bytes([
            (p_u >> 8) & 0xFF,
            p_u & 0xFF,
            (v_u >> 4) & 0xFF,
            ((v_u & 0xF) << 4) | ((kp_u >> 8) & 0xF),
            kp_u & 0xFF,
            (kd_u >> 4) & 0xFF,
            ((kd_u & 0xF) << 4) | ((t_u >> 8) & 0xF),
            t_u & 0xFF,
        ])

        resp_id = _MA_MC_RESP + self._motor_id
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bytes] = loop.create_future()
        self._pending[resp_id] = fut
        await self._bus._send(_MA_MC_REQ + self._motor_id, data)
        try:
            await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            raise MotorError(f"MyActuator motor {self._motor_id:#04x} motion control timed out")
