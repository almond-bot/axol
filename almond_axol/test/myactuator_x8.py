"""Drive a single MyActuator X8 motor with a constant feedforward torque.

Like gravity comp, this uses impedance control with ``set_impedance(p_des=0,
v_des=0, kp=0, kd=0, t_ff=TORQUE)`` so only the feedforward torque acts — there
is no position or velocity servo, just a pure commanded torque on one motor.

Edit the globals at the top of the file to point at your CAN interface, motor
CAN ID, and the torque to command.

WARNING: a free-spinning motor under constant torque will accelerate. Keep the
output shaft loaded/blocked and be ready to Ctrl-C.

Run directly:
    uv run -m almond_axol.test.myactuator_x8
"""

from __future__ import annotations

import asyncio

from ..motor import CanBus
from ..motor.myactuator import MyActuatorMotor
from ..motor.types import ControlMode

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


async def _run() -> None:
    interval = 1.0 / RATE_HZ
    async with CanBus(CAN_CHANNEL) as bus:
        motor = MyActuatorMotor(bus, MOTOR_ID, kt=KT)

        # Release the brake and clear any latched control state so the motor is
        # ready to accept motion-control (MIT) frames.
        await motor.enable()
        await motor.set_control_mode(ControlMode.IMPEDANCE)

        print(
            f"Sending {TORQUE_NM:+.3f} Nm to motor {MOTOR_ID:#04x} on {CAN_CHANNEL} "
            f"at {RATE_HZ} Hz (kp=0, kd=0 — pure torque). Ctrl-C to stop."
        )

        last_print = 0.0
        loop = asyncio.get_running_loop()
        try:
            while True:
                t0 = loop.time()
                # p_des/v_des are ignored because kp=kd=0; only t_ff is applied.
                await motor.set_impedance(0.0, 0.0, 0.0, 0.0, TORQUE_NM)

                if t0 - last_print >= 0.5:
                    last_print = t0
                    try:
                        meas = await motor.get_torque()
                        print(
                            f"  cmd={TORQUE_NM:+.3f} Nm  meas={meas:+.3f} Nm",
                            flush=True,
                        )
                    except Exception as exc:  # noqa: BLE001 — diagnostic only
                        print(
                            f"  cmd={TORQUE_NM:+.3f} Nm  (read failed: {exc})",
                            flush=True,
                        )

                spent = loop.time() - t0
                if spent < interval:
                    await asyncio.sleep(interval - spent)
        finally:
            await motor.disable()
            print("\nMotor disabled.")


def main() -> None:
    """Drive the configured motor with a constant torque until interrupted."""
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
