"""Drive a single MyActuator X8 motor with a constant feedforward torque.

Self-contained: uses only ``python-can`` and the standard library — no
almond_axol imports. The MyActuator MIT-style impedance encoding and the
status-2 torque read-back are inlined below.

Like gravity comp, this uses impedance control with ``kp=0, kd=0`` and only a
feedforward torque (``t_ff=TORQUE_NM``), so there is no position or velocity
servo — just a pure commanded torque on one motor.

Edit the globals at the top of the file to point at your CAN interface, motor
CAN ID, and the torque to command.

WARNING: a free-spinning motor under constant torque will accelerate. Keep the
output shaft loaded/blocked and be ready to Ctrl-C.

Run directly:
    uv run -m almond_axol.test.myactuator_x8
    python -m almond_axol.test.myactuator_x8
"""

from __future__ import annotations

import struct
import time

import can

# ----------------------------------------------------------------------------- #
# Configuration — edit these to match your setup.                               #
# ----------------------------------------------------------------------------- #
CAN_CHANNEL = "can0"  # SocketCAN interface name
MOTOR_ID = 1  # Motor CAN ID
TORQUE_NM = 1.0  # Constant feedforward torque to command (Nm)
RATE_HZ = 100  # Command rate (Hz)

# Torque constant for the MyActuator X8 motor (Nm/A); only used to convert the
# read-back current to torque for the status print.
KT = 2.4

# ----------------------------------------------------------------------------- #
# MyActuator 0x140-series protocol constants (inlined).                         #
# ----------------------------------------------------------------------------- #
_REQ = 0x140  # standard command request  → 0x140 + motor_id
_RESP = 0x240  # standard command response ← 0x240 + motor_id
_MC_REQ = 0x400  # motion control request    → 0x400 + motor_id

_RELEASE_BRAKE = 0x77
_SHUTDOWN = 0x80
_RESET = 0x76  # system reset; no response — motor restarts immediately
_MOTOR_STATUS_2 = 0x9C  # temperature, current, velocity, encoder

# Seconds to wait after a 0x76 system reset before the motor answers again.
_RESET_SETTLE_S = 2.0

# MIT impedance encoding ranges.
_P_MIN, _P_MAX = -12.5, 12.5  # rad
_V_MIN, _V_MAX = -45.0, 45.0  # rad/s
_T_MIN, _T_MAX = -24.0, 24.0  # Nm
_KP_MIN, _KP_MAX = 0.0, 500.0
_KD_MIN, _KD_MAX = 0.0, 5.0


def _float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    """Encode a clamped float into a fixed-point uint for the MIT byte layout."""
    x = max(x_min, min(x_max, x))
    return int((x - x_min) * ((1 << bits) - 1) / (x_max - x_min))


def _cmd(byte: int) -> bytes:
    """Build a single-byte standard command frame (rest zero-padded)."""
    return bytes([byte, 0, 0, 0, 0, 0, 0, 0])


def _encode_impedance(p: float, v: float, kp: float, kd: float, t: float) -> bytes:
    """Pack an impedance (MIT) motion-control frame."""
    p_u = _float_to_uint(p, _P_MIN, _P_MAX, 16)
    v_u = _float_to_uint(v, _V_MIN, _V_MAX, 12)
    kp_u = _float_to_uint(kp, _KP_MIN, _KP_MAX, 12)
    kd_u = _float_to_uint(kd, _KD_MIN, _KD_MAX, 12)
    t_u = _float_to_uint(t, _T_MIN, _T_MAX, 12)
    return bytes(
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


def _send(bus: can.BusABC, arbitration_id: int, data: bytes) -> None:
    """Send a standard 8-byte CAN frame."""
    bus.send(
        can.Message(arbitration_id=arbitration_id, data=data, is_extended_id=False)
    )


def _read_torque(bus: can.BusABC) -> float | None:
    """Request status-2 and decode the measured torque (current × KT), or None."""
    _send(bus, _REQ + MOTOR_ID, _cmd(_MOTOR_STATUS_2))
    deadline = time.monotonic() + 0.1
    while time.monotonic() < deadline:
        msg = bus.recv(timeout=0.1)
        if (
            msg is not None
            and msg.arbitration_id == _RESP + MOTOR_ID
            and msg.data[0] == _MOTOR_STATUS_2
        ):
            current = struct.unpack_from("<h", msg.data, 2)[0]  # 0.01 A/LSB
            return current * 0.01 * KT
    return None


def main() -> None:
    """Drive the configured motor with a constant torque until interrupted."""
    interval = 1.0 / RATE_HZ
    bus = can.Bus(channel=CAN_CHANNEL, bustype="socketcan")
    try:
        # Release the brake, then reset to clear any latched control state so the
        # motor is ready to accept motion-control (MIT) frames.
        _send(bus, _REQ + MOTOR_ID, _cmd(_RELEASE_BRAKE))
        _send(bus, _REQ + MOTOR_ID, _cmd(_RESET))
        time.sleep(_RESET_SETTLE_S)

        print(
            f"Sending {TORQUE_NM:+.3f} Nm to motor {MOTOR_ID:#04x} on {CAN_CHANNEL} "
            f"at {RATE_HZ} Hz (kp=0, kd=0 — pure torque). Ctrl-C to stop."
        )

        frame = _encode_impedance(0.0, 0.0, 0.0, 0.0, TORQUE_NM)
        last_print = 0.0
        while True:
            t0 = time.monotonic()
            # p_des/v_des are ignored because kp=kd=0; only t_ff is applied.
            _send(bus, _MC_REQ + MOTOR_ID, frame)

            if t0 - last_print >= 0.5:
                last_print = t0
                meas = _read_torque(bus)
                if meas is None:
                    print(f"  cmd={TORQUE_NM:+.3f} Nm  (read failed)", flush=True)
                else:
                    print(f"  cmd={TORQUE_NM:+.3f} Nm  meas={meas:+.3f} Nm", flush=True)

            spent = time.monotonic() - t0
            if spent < interval:
                time.sleep(interval - spent)
    except KeyboardInterrupt:
        pass
    finally:
        _send(bus, _REQ + MOTOR_ID, _cmd(_SHUTDOWN))
        bus.shutdown()
        print("\nMotor disabled.")


if __name__ == "__main__":
    main()
