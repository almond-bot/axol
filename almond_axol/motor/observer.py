"""Passive CAN observer: read every motor's state without ever transmitting.

SocketCAN delivers each frame on an interface to every open socket, so a
socket that never sends can watch the traffic of whichever process currently
commands the motors and reconstruct each joint's state from it. This splits
bus *observation* from bus *command*: command access stays exclusive (two
processes issuing request/response reads to the same motor would cross-match
replies), but any number of processes may observe. ``axol serve`` uses this
to keep live telemetry streaming while a teleop session or diagnostic owns
command of the robot.

What each motor family gives an observer:

- MyActuator: motion-control feedback on ``0x500 + id`` — position, velocity,
  torque, emitted for every MIT command (teleop streams these at the control
  rate) — plus standard replies on ``0x240 + id``: the multi-turn angle
  (0x92), temperature/current/speed (0x9C, and the 0xA2/0xA4 control replies
  that share its layout), and temperature/voltage/error bits (0x9A).
- Damiao: feedback frames on the MST ID ``0x10 + id`` — status, position,
  velocity, torque, MOS/rotor temperatures — emitted for every impedance
  command and every 0xCC feedback request, plus 0x33 register-read replies
  (bus voltage, and the PMAX/VMAX/TMAX scaling registers themselves).

The fixed-point MIT fields decode against per-motor ranges the observer
cannot query for itself (it never sends). Defaults are the conservative
legacy ranges; a process that owns command syncs the true values in via
:meth:`BusObserver.set_myactuator_ranges` / :meth:`BusObserver.set_damiao_ranges`
(the serve robot link does this once on connect). Damiao ranges additionally
self-correct whenever *any* process reads the scaling registers on the wire.
"""

from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass
from typing import Callable

import can

from ..constants import Joint
from .bus import CanBus
from .damiao import (
    _DM_REG_PMAX,
    _DM_REG_TMAX,
    _DM_REG_VBUS,
    _DM_REG_VMAX,
    _DM_STATUS_MAP,
    DamiaoMotor,
    _DamiaoStatus,
)
from .damiao import _uint_to_float as _dm_uint_to_float
from .motor import _JOINT_CONFIG, _MotorType
from .myactuator import (
    _MA_MC_RESP,
    _MA_MOTOR_STATUS_2,
    _MA_MULTI_TURN_ANGLE,
    _MA_P_MAX_LEGACY,
    _MA_POS_CONTROL,
    _MA_READ_STATUS1,
    _MA_RESP,
    _MA_T_MAX_LEGACY,
    _MA_V_MAX,
    _MA_VELOCITY_CONTROL,
    _ma_error_to_status,
)
from .myactuator import _uint_to_float as _ma_uint_to_float
from .types import MotorStatus


@dataclass
class JointObservation:
    """Latest state reconstructed for one joint from observed traffic.

    ``fast_ts`` / ``slow_ts`` are the CAN receive timestamps (epoch s) of the
    newest frame that updated the fast (position/velocity/torque) and slow
    (temperature/voltage/status) fields — consumers use them to tell live
    data from a joint that simply stopped being commanded.
    """

    position: float | None = None
    velocity: float | None = None
    torque: float | None = None
    temperature: float | None = None
    voltage: float | None = None
    status: str | None = None
    fast_ts: float = 0.0
    slow_ts: float = 0.0


class _MyActuatorDecoder:
    """Decode one MyActuator motor's reply traffic into a JointObservation."""

    def __init__(self, obs: JointObservation, kt: float) -> None:
        self.obs = obs
        self.kt = kt
        # Conservative legacy MIT ranges until synced (see set_ranges).
        self.p_max = _MA_P_MAX_LEGACY
        self.t_max = _MA_T_MAX_LEGACY
        self.synced = False

    def set_ranges(self, p_max: float, t_max: float) -> None:
        self.p_max = p_max
        self.t_max = t_max
        self.synced = True

    def on_motion_feedback(self, data: bytes, ts: float) -> None:
        """MIT feedback on 0x500+id: fixed-point position/velocity/torque."""
        pos_int = (data[1] << 8) | data[2]
        vel_int = (data[3] << 4) | (data[4] >> 4)
        torq_int = ((data[4] & 0x0F) << 8) | data[5]
        self.obs.position = _ma_uint_to_float(pos_int, -self.p_max, self.p_max, 16)
        self.obs.velocity = _ma_uint_to_float(vel_int, -_MA_V_MAX, _MA_V_MAX, 12)
        self.obs.torque = _ma_uint_to_float(torq_int, -self.t_max, self.t_max, 12)
        self.obs.fast_ts = ts

    def on_standard_reply(self, data: bytes, ts: float) -> None:
        """Request/response replies on 0x240+id, dispatched by the echoed command byte."""
        cmd = data[0]
        if cmd == _MA_MULTI_TURN_ANGLE:  # 0x92: int32, 0.01 deg/LSB
            raw = struct.unpack_from("<i", data, 4)[0]
            self.obs.position = raw * (0.01 * math.pi / 180.0)
            self.obs.fast_ts = ts
        elif cmd in (_MA_MOTOR_STATUS_2, _MA_VELOCITY_CONTROL, _MA_POS_CONTROL):
            # 0x9C and the 0xA2/0xA4 control replies share one layout:
            # [cmd, temp °C, current 0.01A ×2, speed dps ×2, encoder ×2].
            self.obs.temperature = float(struct.unpack_from("b", data, 1)[0])
            self.obs.torque = struct.unpack_from("<h", data, 2)[0] * 0.01 * self.kt
            self.obs.velocity = struct.unpack_from("<h", data, 4)[0] * (math.pi / 180.0)
            self.obs.fast_ts = ts
            self.obs.slow_ts = ts
        elif cmd == _MA_READ_STATUS1:  # 0x9A: voltage + error bitmask
            self.obs.voltage = struct.unpack_from("<H", data, 4)[0] * 0.1
            bits = struct.unpack_from("<H", data, 6)[0]
            self.obs.status = _ma_error_to_status(int(bits)).name
            self.obs.slow_ts = ts


class _DamiaoDecoder:
    """Decode one Damiao motor's MST-ID traffic into a JointObservation."""

    # Float registers worth sniffing off 0x33 read replies: the bus voltage
    # for slow telemetry, and the MIT scaling ranges themselves — whenever any
    # process reads them, the observer's decode self-corrects.
    _FLOAT_REGS = frozenset({_DM_REG_VBUS, _DM_REG_PMAX, _DM_REG_VMAX, _DM_REG_TMAX})

    def __init__(self, obs: JointObservation, motor_id: int) -> None:
        self.obs = obs
        self.motor_id = motor_id
        # Driver defaults until synced from the motor's registers.
        self.p_max = 12.5
        self.v_max = 45.0
        self.t_max = 18.0
        self.synced = False

    def set_ranges(self, p_max: float, v_max: float, t_max: float) -> None:
        self.p_max = p_max
        self.v_max = v_max
        self.t_max = t_max
        self.synced = True

    def on_feedback(self, data: bytes, ts: float) -> None:
        """A frame on this motor's MST_ID: register traffic or MIT feedback."""
        # Same register-traffic guard as DamiaoMotor._on_message: acks and
        # read replies echo [id_lo, id_hi, cmd, rid, ...] on the MST_ID and
        # must never reach the feedback decoder.
        if (
            data[1] <= 0x0F
            and data[2] in DamiaoMotor._REGISTER_TRAFFIC_CMDS
            and data[3] <= 81
        ):
            if (data[0] | (data[1] << 8)) == self.motor_id and data[2] == 0x33:
                self._on_register_reply(data, ts)
            return
        if (data[0] & 0x0F) != (self.motor_id & 0x0F):
            return

        pos_int = (data[1] << 8) | data[2]
        vel_int = (data[3] << 4) | (data[4] >> 4)
        torq_int = ((data[4] & 0xF) << 8) | data[5]
        self.obs.position = _dm_uint_to_float(pos_int, -self.p_max, self.p_max, 16)
        self.obs.velocity = _dm_uint_to_float(vel_int, -self.v_max, self.v_max, 12)
        self.obs.torque = _dm_uint_to_float(torq_int, -self.t_max, self.t_max, 12)
        self.obs.temperature = float(max(data[6], data[7]))  # max(t_mos, t_rotor)
        try:
            status = _DamiaoStatus(data[0] >> 4)
        except ValueError:
            status = _DamiaoStatus.DISABLED
        self.obs.status = _DM_STATUS_MAP.get(status, MotorStatus.UNKNOWN).name
        self.obs.fast_ts = ts
        self.obs.slow_ts = ts

    def _on_register_reply(self, data: bytes, ts: float) -> None:
        rid = data[3]
        if rid not in self._FLOAT_REGS:
            return
        value = float(struct.unpack("<f", data[4:8])[0])
        if rid == _DM_REG_VBUS:
            self.obs.voltage = value
            self.obs.slow_ts = ts
        elif rid == _DM_REG_PMAX and value > 0:
            self.p_max = value
        elif rid == _DM_REG_VMAX and value > 0:
            self.v_max = value
        elif rid == _DM_REG_TMAX and value > 0:
            self.t_max = value


class BusObserver:
    """Always-listening, never-transmitting view of one arm's CAN channel.

    Opens its own SocketCAN socket (via :class:`CanBus`, reusing its
    lost-interface recovery) and decodes every recognizable feedback or reply
    frame into per-joint :class:`JointObservation` state — regardless of which
    process generated the traffic. Never sends a single frame, so it is safe
    to keep open while another process owns command of the bus.

    Use from one event loop: the CAN reader dispatches on the loop
    :meth:`start` ran on, and snapshots are read on the same loop.
    """

    def __init__(self, channel: str, joints: list[Joint]) -> None:
        self._channel = channel
        self._observations: dict[Joint, JointObservation] = {}
        self._decoders: dict[Joint, _MyActuatorDecoder | _DamiaoDecoder] = {}
        # arbitration id -> decode method for an 8-byte payload on that id.
        self._handlers: dict[int, Callable[[bytes, float], None]] = {}
        for joint in joints:
            cfg = _JOINT_CONFIG[joint]
            obs = JointObservation()
            if cfg.kind is _MotorType.MYACTUATOR:
                ma = _MyActuatorDecoder(obs, kt=cfg.kt)
                self._handlers[_MA_MC_RESP + cfg.motor_id] = ma.on_motion_feedback
                self._handlers[_MA_RESP + cfg.motor_id] = ma.on_standard_reply
                self._decoders[joint] = ma
            else:
                dm = _DamiaoDecoder(obs, motor_id=cfg.motor_id)
                self._handlers[0x10 + cfg.motor_id] = dm.on_feedback
                self._decoders[joint] = dm
            self._observations[joint] = obs
        self._bus: CanBus | None = None

    async def start(self) -> None:
        """Open the socket and start decoding. Idempotent."""
        if self._bus is not None:
            return
        self._bus = CanBus(self._channel)
        self._bus._add_listener(self._on_message)
        await self._bus.start()

    async def close(self) -> None:
        if self._bus is not None:
            await self._bus.close()
            self._bus = None

    @property
    def running(self) -> bool:
        return self._bus is not None

    def _on_message(self, msg: can.Message) -> None:
        handler = self._handlers.get(msg.arbitration_id)
        if handler is None or len(msg.data) != 8:
            return
        handler(bytes(msg.data), msg.timestamp or time.time())

    # -- range sync (called by a process that can query the motors) ---------

    def ranges_synced(self, joint: Joint) -> bool:
        """Whether *joint*'s fixed-point decode ranges were synced from the motor."""
        decoder = self._decoders.get(joint)
        return decoder is not None and decoder.synced

    def set_myactuator_ranges(self, joint: Joint, p_max: float, t_max: float) -> None:
        """Set the MIT position/torque ranges this joint's firmware scales against."""
        decoder = self._decoders[joint]
        assert isinstance(decoder, _MyActuatorDecoder), joint
        decoder.set_ranges(p_max, t_max)

    def set_damiao_ranges(
        self, joint: Joint, p_max: float, v_max: float, t_max: float
    ) -> None:
        """Set the PMAX/VMAX/TMAX register values this joint scales against."""
        decoder = self._decoders[joint]
        assert isinstance(decoder, _DamiaoDecoder), joint
        decoder.set_ranges(p_max, v_max, t_max)

    # -- snapshots ------------------------------------------------------------

    def fast_snapshot(
        self, max_age_s: float = 0.5
    ) -> dict[Joint, tuple[float, float, float]]:
        """Joints with complete position/velocity/torque fresher than ``max_age_s``."""
        now = time.time()
        out: dict[Joint, tuple[float, float, float]] = {}
        for joint, o in self._observations.items():
            if now - o.fast_ts > max_age_s:
                continue
            if o.position is None or o.velocity is None or o.torque is None:
                continue
            out[joint] = (o.position, o.velocity, o.torque)
        return out

    def slow_snapshot(self, max_age_s: float = 3.0) -> dict[Joint, dict]:
        """Temperature/voltage/status for joints heard from within ``max_age_s``.

        Entries carry ``reachable: True`` — observed traffic is proof the
        motor is alive — matching the health-ping sweep's shape so they can
        merge into the same telemetry stream.
        """
        now = time.time()
        out: dict[Joint, dict] = {}
        for joint, o in self._observations.items():
            if now - o.slow_ts > max_age_s:
                continue
            out[joint] = {
                "reachable": True,
                "status": o.status,
                "temperature": o.temperature,
                "voltage": o.voltage,
            }
        return out
