"""
base.drive

Drive the powered Axol Cart (x-drive omni base + telescoping lift) with a
Logitech gamepad.

This is a thin gamepad frontend over :class:`almond_axol.robot.cart.Cart`,
which owns all the control logic (slew limiting, x-drive mixing, the MIT
park hold, PMAX widening, lift GPIO edges) — the same class VR teleop
drives, so bench behavior and teleop behavior cannot drift apart. See the
``cart`` module docstring for wheel CAN IDs, body-frame conventions, and
the parking details, and :mod:`almond_axol.robot.lift` for the JCB35N2
lift protocol and wiring.

Controls (Logitech F310/F710 in XInput mode):
    Left stick    translate (up = forward, left = strafe left)
    Right stick   rotate (left = counter-clockwise)
    D-pad up/down raise / lower the telescoping lift (hold to move)
    LB or RB      deadman — hold to drive; release for a smooth stop
    B             quit (wheels stopped, motors disabled)

The lift only moves while the deadman is held and the D-pad is pressed;
releasing either stops it (the box also has its own anti-collision stop).
Pass ``--lift-height-port`` (a serial device wired to lift RJ45 pin 2) to
display live height; ``--no-lift`` disables the lift entirely.

Run directly (pygame ships in the ``gamepad`` extra):
    uv run --extra gamepad -m almond_axol.diagnostics.base.drive
    uv run --extra gamepad -m almond_axol.diagnostics.base.drive --channel can0 --max-speed 5
    uv run --extra gamepad -m almond_axol.diagnostics.base.drive --no-can  # gamepad + lift only
"""

from __future__ import annotations

import argparse
import asyncio
import os

from ...robot.cart import DEFAULT_CHANNEL, WHEELS, Cart, CartConfig, deadzone
from ...robot.lift import DOWN, STOP, UP, HeightReader

# Logitech F310/F710 (XInput mode) under SDL/pygame.
_AXIS_LX = 0  # left stick x: left = -1
_AXIS_LY = 1  # left stick y: up = -1
_AXIS_RX = 3  # right stick x: left = -1
_BTN_B = 1
_BTN_LB = 4
_BTN_RB = 5
_HAT_DPAD = 0  # D-pad hat index; hat y: up = +1, down = -1

_DISPLAY_HZ = 50.0


def _init_gamepad(index: int):  # noqa: ANN202 — pygame typed lazily
    """Init pygame (headless) and return the joystick at ``index``."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    try:
        import pygame
    except ImportError:
        raise SystemExit(
            "pygame is not installed — run with the gamepad extra:\n"
            "  uv run --extra gamepad -m almond_axol.diagnostics.base.drive"
        )
    pygame.init()
    pygame.joystick.init()
    count = pygame.joystick.get_count()
    if count == 0:
        raise SystemExit("No gamepad detected — plug in the Logitech controller.")
    if index >= count:
        raise SystemExit(f"--joystick {index} out of range ({count} detected).")
    pad = pygame.joystick.Joystick(index)
    pad.init()
    print(
        f"Gamepad: {pad.get_name()}  "
        f"(axes={pad.get_numaxes()} buttons={pad.get_numbuttons()})"
    )
    return pad


def _status_line(cart: Cart, engaged: bool, lift_height_mm: int | None) -> str:
    if engaged:
        state = "DRIVE"
    elif cart.parked:
        state = "PARKED (hold LB/RB)"
    else:
        state = "hold LB/RB to drive"
    cmd = cart.body_cmd
    wheels = "  ".join(
        f"{w.name.split('_')[0][0]}{w.name.split('_')[1][0]}:{s:+6.2f}"
        for w, s in zip(WHEELS, cart.wheel_speeds)
    )
    lift = {UP: "up", DOWN: "down", STOP: "--"}[cart.lift_dir]
    height = f" {lift_height_mm}mm" if lift_height_mm is not None else ""
    warn = "  [CMD ERR]" if cart.send_failed else ""
    return (
        f"\r  {state:<22}  vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} wz={cmd[2]:+.2f}"
        f"  |  {wheels} rad/s  |  lift:{lift}{height}{warn}  \033[K"
    )


async def _input_loop(
    pad,  # noqa: ANN001 — pygame typed lazily
    cart: Cart,
    height: HeightReader | None,
    dz: float,
) -> None:
    """Poll the gamepad into ``cart.set_command`` until B is pressed."""
    import pygame

    interval = 1.0 / _DISPLAY_HZ
    while True:
        pygame.event.pump()

        if pad.get_button(_BTN_B):
            print("\nB pressed — stopping.")
            return

        engaged = bool(pad.get_button(_BTN_LB) or pad.get_button(_BTN_RB))
        vx = vy = wz = 0.0
        lift_dir = STOP
        if engaged:
            vx = -deadzone(pad.get_axis(_AXIS_LY), dz)  # up = forward
            vy = -deadzone(pad.get_axis(_AXIS_LX), dz)  # left = +
            wz = -deadzone(pad.get_axis(_AXIS_RX), dz)  # left = CCW
            if pad.get_numhats() > _HAT_DPAD:
                hat_y = pad.get_hat(_HAT_DPAD)[1]
                lift_dir = UP if hat_y > 0 else DOWN if hat_y < 0 else STOP
        cart.set_command(vx, vy, wz, lift_dir)

        lift_height = height.poll() if height is not None else None
        print(_status_line(cart, engaged, lift_height), end="", flush=True)
        await asyncio.sleep(interval)


async def _run(args: argparse.Namespace) -> None:
    pad = _init_gamepad(args.joystick)

    config = CartConfig(
        channel=None if args.no_can else args.channel,
        max_speed=args.max_speed,
        turn_scale=args.turn_scale,
        slew=args.slew,
        deadzone=args.deadzone,
        hold_kp=args.hold_kp,
        hold_kd=args.hold_kd,
        lift=not args.no_lift,
        lift_chip=args.lift_chip,
        lift_up_gpio=args.lift_up_gpio,
        lift_down_gpio=args.lift_down_gpio,
    )
    if args.no_can:
        print("--no-can: wheel motors disabled (gamepad + lift only).")

    height: HeightReader | None = None
    if not args.no_lift and args.lift_height_port:
        height = HeightReader(args.lift_height_port)

    cart = Cart(config)
    await cart.enable()
    print("Cart enabled. Hold LB/RB to drive, D-pad for the lift, B to quit.")
    try:
        await _input_loop(pad, cart, height, args.deadzone)
    finally:
        await cart.disable()
        if height is not None:
            height.close()
        print("Cart disabled.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Drive the powered Axol Cart with a Logitech gamepad.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--channel",
        default=DEFAULT_CHANNEL,
        help=f"SocketCAN interface for the base (default: {DEFAULT_CHANNEL})",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=10.0,
        help="Peak wheel speed in rad/s at full stick (default: 10)",
    )
    parser.add_argument(
        "--turn-scale",
        type=float,
        default=0.5,
        help="Rotation weight relative to translation, [0, 1] (default: 0.5)",
    )
    parser.add_argument(
        "--deadzone",
        type=float,
        default=0.12,
        help="Stick deadzone as a fraction of full deflection (default: 0.12)",
    )
    parser.add_argument(
        "--slew",
        type=float,
        default=2.0,
        help="Max change of the normalized body command per second (default: 2)",
    )
    parser.add_argument(
        "--hold-kp",
        type=float,
        default=60.0,
        help="Position stiffness (Nm/rad) of the parked MIT hold; "
        "0 disables parking (default: 60)",
    )
    parser.add_argument(
        "--hold-kd",
        type=float,
        default=1.5,
        help="Damping (Nm·s/rad) of the parked MIT hold; must be > 0 "
        "when hold-kp > 0 (default: 1.5)",
    )
    parser.add_argument(
        "--joystick",
        type=int,
        default=0,
        help="pygame joystick index if several are connected (default: 0)",
    )
    parser.add_argument(
        "--no-can",
        action="store_true",
        help="Skip the CAN bus (no wheel motion); gamepad and lift still work.",
    )
    parser.add_argument(
        "--lift-chip",
        default="/dev/gpiochip0",
        help="gpiochip device for the lift button lines (default: /dev/gpiochip0)",
    )
    parser.add_argument(
        "--lift-up-gpio",
        type=int,
        default=23,
        help="GPIO line offset wired to lift RJ45 pin 7 (HS1, up) (default: 23)",
    )
    parser.add_argument(
        "--lift-down-gpio",
        type=int,
        default=24,
        help="GPIO line offset wired to lift RJ45 pin 8 (HS0, down) (default: 24)",
    )
    parser.add_argument(
        "--lift-height-port",
        default=None,
        help="Serial device wired to lift RJ45 pin 2 (DTX) to display live "
        "height, e.g. /dev/ttyAMA0. Default: off.",
    )
    parser.add_argument(
        "--no-lift",
        action="store_true",
        help="Skip the lift entirely (no GPIOs are touched).",
    )
    args = parser.parse_args(argv)

    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
