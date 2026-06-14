"""Send a constant feedforward torque (no position hold) to one MyActuator motor.

Like gravity comp, this drives the motor with ``set_impedance(p_des=0, v_des=0,
kp=0, kd=0, t_ff=TORQUE)`` so only the feedforward torque acts — there is no
position or velocity servo. Defaults to 1 Nm on the motor at CAN ID 1.

WARNING: a free-spinning motor under constant torque will accelerate. Keep the
output shaft loaded/blocked and be ready to Ctrl-C.

Run directly:
    uv run -m almond_axol.test.can.torque                 # left arm, id 1, 1 Nm
    uv run -m almond_axol.test.can.torque --r             # right arm
    uv run -m almond_axol.test.can.torque --id 2 --torque 0.5
    uv run -m almond_axol.test.can.torque --hz 200
"""

from __future__ import annotations

import argparse
import asyncio

from ...motor import CanBus
from ...motor.myactuator import MyActuatorMotor
from ...motor.types import ControlMode
from ...utils.shared import CAN_LEFT, CAN_RIGHT

# Torque constant for the shoulder/elbow MyActuator motors (Nm/A); only used to
# convert the read-back current to torque for the status print.
_DEFAULT_KT = 2.4


async def _run(channel: str, motor_id: int, torque: float, hz: int) -> None:
    interval = 1.0 / hz
    async with CanBus(channel) as bus:
        motor = MyActuatorMotor(bus, motor_id, kt=_DEFAULT_KT)

        # Release the brake and clear any latched control state so the motor is
        # ready to accept motion-control (MIT) frames.
        await motor.enable()
        await motor.set_control_mode(ControlMode.IMPEDANCE)

        print(
            f"Sending {torque:+.3f} Nm to motor {motor_id:#04x} on {channel} "
            f"at {hz} Hz (kp=0, kd=0 — pure torque). Ctrl-C to stop."
        )

        last_print = 0.0
        loop = asyncio.get_running_loop()
        try:
            while True:
                t0 = loop.time()
                # p_des/v_des are ignored because kp=kd=0; only t_ff is applied.
                await motor.set_impedance(0.0, 0.0, 0.0, 0.0, torque)

                if t0 - last_print >= 0.5:
                    last_print = t0
                    try:
                        meas = await motor.get_torque()
                        print(f"  cmd={torque:+.3f} Nm  meas={meas:+.3f} Nm", flush=True)
                    except Exception as exc:  # noqa: BLE001 — diagnostic only
                        print(f"  cmd={torque:+.3f} Nm  (read failed: {exc})", flush=True)

                spent = loop.time() - t0
                if spent < interval:
                    await asyncio.sleep(interval - spent)
        finally:
            await motor.disable()
            print("\nMotor disabled.")


def main() -> None:
    """Parse CLI arguments and drive one motor with a constant torque."""
    parser = argparse.ArgumentParser(
        description="Send a constant feedforward torque to one MyActuator motor."
    )
    side = parser.add_mutually_exclusive_group()
    side.add_argument("--l", action="store_true", help="Use left arm (default)")
    side.add_argument("--r", action="store_true", help="Use right arm")
    parser.add_argument(
        "--id", type=lambda x: int(x, 0), default=1, help="Motor CAN ID (default: 1)"
    )
    parser.add_argument(
        "--torque", type=float, default=1.0, help="Feedforward torque in Nm (default: 1.0)"
    )
    parser.add_argument(
        "--hz", type=int, default=100, help="Command rate in Hz (default: 100)"
    )
    args = parser.parse_args()

    channel = CAN_RIGHT if args.r else CAN_LEFT

    try:
        asyncio.run(_run(channel, args.id, args.torque, args.hz))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
