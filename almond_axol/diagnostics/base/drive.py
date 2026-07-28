"""
base.drive

Drive the powered Axol Cart (x-drive omni base + telescoping lift) with a
Logitech gamepad.

This is a thin gamepad frontend over :class:`almond_axol.robot.cart.Cart`,
which owns all the control logic (slew limiting, x-drive mixing, the MIT
park hold, PMAX widening, lift commands) — the same class VR teleop
drives, so bench behavior and teleop behavior cannot drift apart. See the
``cart`` module docstring for wheel CAN IDs, body-frame conventions, and
the parking details.

Controls (Logitech F310/F710 in XInput mode):
    Left stick    translate (up = forward, left = strafe left)
    Right stick   rotate (left = counter-clockwise)
    D-pad up/down raise / lower the telescoping lift (hold to move)
    LB or RB      deadman — hold to drive; release for a smooth stop
    B             quit (wheels stopped, motors disabled)

The D-pad commands the lift only while the deadman is held; releasing
either stops it. Nothing physically moves until the lift PCB lands — the
driver behind it is a no-op stub (see :mod:`almond_axol.robot.lift`), and
the status line's height stays blank for the same reason. ``--no-lift``
skips the lift entirely.

Run directly (pygame ships in the ``gamepad`` extra):
    uv run --extra gamepad -m almond_axol.diagnostics.base.drive
    uv run --extra gamepad -m almond_axol.diagnostics.base.drive --channel can0 --max-speed 5
    uv run --extra gamepad -m almond_axol.diagnostics.base.drive --no-can  # gamepad + lift only
    uv run --extra gamepad -m almond_axol.diagnostics.base.drive --imu --yaw-log  # bench the heading hold
"""

from __future__ import annotations

import argparse
import asyncio
import logging
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
        imu=args.imu,
        yaw_log=args.yaw_log,
    )
    if args.no_can:
        print("--no-can: wheel motors disabled (gamepad + lift only).")

    height: HeightReader | None = None
    if not args.no_lift:
        height = HeightReader()

    cart = Cart(config)
    await cart.enable()

    # Same wiring as VR teleop (see cli/teleop.py's _wire_cart_imu): the board
    # BMI088 feeds the heading hold; on failure the hold is simply inert.
    imu_src = None
    if args.imu:
        try:
            from ...robot.gyro import BoardYawRateSource

            imu_src = BoardYawRateSource(cart.feed_yaw_rate)
            imu_src.open()
            print("--imu: board gyro feeding the heading hold.")
        except Exception as exc:  # noqa: BLE001 - heading hold is best-effort
            print(f"--imu: could not start the board gyro ({exc}); hold disabled.")

    print("Cart enabled. Hold LB/RB to drive, D-pad for the lift, B to quit.")
    try:
        await _input_loop(pad, cart, height, args.deadzone)
    finally:
        if imu_src is not None:
            imu_src.close()
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
        "--no-lift",
        action="store_true",
        help="Skip the lift entirely (the D-pad is ignored).",
    )
    parser.add_argument(
        "--imu",
        action="store_true",
        help="Feed the heading hold from the board BMI088 gyro, as VR teleop "
        "does with --cart.imu (see almond_axol.robot.gyro).",
    )
    parser.add_argument(
        "--yaw-log",
        action="store_true",
        help="Trace the heading hold (10 Hz state line + per-stroke drift "
        "summary) — for diagnosing drift, usually together with --imu.",
    )
    args = parser.parse_args(argv)

    # The Cart and its yaw trace (--yaw-log) report through logging; without
    # a handler those INFO lines would be dropped silently.
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
