"""
base.drive

Drive the omni-wheel mobile base with a Logitech gamepad.

The base is an x-drive: four omni wheels mounted at 45° on the corners,
each powered by a Damiao motor (same units as the wrist motors) in
VELOCITY mode. CAN IDs are fixed by convention:

    id 1  front-left      id 2  front-right
    id 3  back-left       id 4  back-right

Controls (Logitech F310/F710 in XInput mode):
    Left stick    translate (up = forward, left = strafe left)
    Right stick   rotate (left = counter-clockwise)
    D-pad up/down raise / lower the telescoping lift (hold to move)
    LB or RB      deadman — hold to drive; release for a smooth stop
    B             quit (wheels stopped, motors disabled)

The telescoping lift (a Jiecang JCB35N2 box driving two desk legs) is
commanded by emulating its wired handset on two open-drain GPIOs — its
RJ45 port is a button/UART interface, not Ethernet; see ``lift.py`` for
the protocol and wiring. The lift only moves while the deadman is held
and the D-pad is pressed; releasing either stops it (the box also has its
own anti-collision stop). Configure with ``--lift-up-gpio`` /
``--lift-down-gpio`` / ``--lift-chip``; pass ``--lift-height-port`` (a
serial device wired to RJ45 pin 2) to display live height; ``--no-lift``
disables the lift entirely.

When the deadman is released and the command has ramped to zero, the wheels
are parked: switched to MIT/impedance mode and held at their current
position by the motor's internal high-bandwidth position loop, so the wheel
does not give under load. Tune with ``--hold-kp`` / ``--hold-kd``;
``--hold-kp 0`` disables parking.

Damiao position commands/feedback are mapped into ±PMAX (12.5 rad from
factory — about two wheel turns), which drive wheels escape almost
immediately; anchoring at the reported position then means a phantom error
of several radians and instant overcurrent. Re-zeroing at park time doesn't
help either: on this firmware the 0xFE zero command only applies after a
power cycle. So at startup the PMAX register is raised (RAM only, reverts
on power-off) to keep multi-turn positions valid for a whole session, and
parking refuses (with a warning) if a wheel ever approaches the widened
limit.

Body-frame convention: +x forward, +y left, +wz counter-clockwise. The
mixing assumes each wheel's positive spin has a forward (+x) component;
if a wheel runs backwards on your base, flip its entry in
:data:`_WHEEL_SIGNS`.

Run directly (pygame ships in the ``gamepad`` extra):
    uv run --extra gamepad -m almond_axol.diagnostics.base.drive
    uv run --extra gamepad -m almond_axol.diagnostics.base.drive --channel can0 --max-speed 5
    uv run --extra gamepad -m almond_axol.diagnostics.base.drive --no-can  # gamepad check only
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from dataclasses import dataclass

from ...motor import CanBus, ControlMode, make_driver
from ...motor.damiao import _DM_REG_PMAX
from ...motor.driver import MotorDriver
from .lift import DOWN, STOP, UP, HeightReader, JiecangLift

# The base rides its own CAN interface, separate from the two arm buses.
DEFAULT_CHANNEL = "can_alm_axol_base"

# Logitech F310/F710 (XInput mode) under SDL/pygame.
_AXIS_LX = 0  # left stick x: left = -1
_AXIS_LY = 1  # left stick y: up = -1
_AXIS_RX = 3  # right stick x: left = -1
_BTN_B = 1
_BTN_LB = 4
_BTN_RB = 5
_HAT_DPAD = 0  # D-pad hat index; hat y: up = +1, down = -1

# Per-wheel spin-direction calibration: flip an entry to -1 if that wheel
# drives the wrong way with everything else correct.
_WHEEL_SIGNS: dict[int, float] = {1: 1.0, 2: -1.0, 3: 1.0, 4: -1.0}

# Position-mapping range (PMAX, register 21) written at startup, in rad.
# Wide enough that a session's accumulated wheel rotation stays in range
# (the factory 12.5 rad is ~2 wheel turns), narrow enough that the 16-bit
# MIT position encoding keeps sub-centidegree resolution (~12 mrad here).
_SESSION_PMAX = 400.0


@dataclass(frozen=True)
class _Wheel:
    """One wheel's CAN ID and its x-drive mixing coefficients.

    Wheel speed = ``mx·vx + my·vy + mw·wz`` (body frame: +x forward, +y
    left, +wz CCW), with each wheel's positive drive direction chosen to
    have a forward component. The common √2/2 translation factor is folded
    into the normalization in :func:`_mix`.
    """

    name: str
    motor_id: int
    mx: float
    my: float
    mw: float


_WHEELS: tuple[_Wheel, ...] = (
    _Wheel("front_left", 1, +1.0, -1.0, -1.0),
    _Wheel("front_right", 2, +1.0, +1.0, +1.0),
    _Wheel("back_left", 3, +1.0, +1.0, -1.0),
    _Wheel("back_right", 4, +1.0, -1.0, +1.0),
)


def _deadzone(value: float, threshold: float) -> float:
    """Zero the stick inside the deadzone and rescale the rest to [-1, 1]."""
    if abs(value) < threshold:
        return 0.0
    scaled = (abs(value) - threshold) / (1.0 - threshold)
    return scaled if value > 0 else -scaled


def _mix(
    vx: float, vy: float, wz: float, max_speed: float, turn_scale: float
) -> list[float]:
    """Map normalized body command ([-1, 1] each) to per-wheel rad/s.

    The raw mix can exceed 1 when translation and rotation combine, so the
    whole set is scaled down together to preserve the motion direction while
    keeping every wheel within ``max_speed``.
    """
    wz *= turn_scale
    raw = [
        _WHEEL_SIGNS[w.motor_id] * (w.mx * vx + w.my * vy + w.mw * wz) for w in _WHEELS
    ]
    scale = max(1.0, max(abs(r) for r in raw))
    return [r / scale * max_speed for r in raw]


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


def _status_line(
    engaged: bool,
    cmd: list[float],
    speeds: list[float],
    stale: bool,
    holding: bool,
    lift_dir: int,
    lift_height_mm: int | None,
) -> str:
    if engaged:
        state = "DRIVE"
    elif holding:
        state = "PARKED (hold LB/RB)"
    else:
        state = "hold LB/RB to drive"
    wheels = "  ".join(
        f"{w.name.split('_')[0][0]}{w.name.split('_')[1][0]}:{s:+6.2f}"
        for w, s in zip(_WHEELS, speeds)
    )
    lift = {UP: "up", DOWN: "down", STOP: "--"}[lift_dir]
    height = f" {lift_height_mm}mm" if lift_height_mm is not None else ""
    warn = "  [CMD ERR]" if stale else ""
    return (
        f"\r  {state:<22}  vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} wz={cmd[2]:+.2f}"
        f"  |  {wheels} rad/s  |  lift:{lift}{height}{warn}  \033[K"
    )


async def _park(motors: list[MotorDriver]) -> list[float] | None:
    """Switch the wheels to the MIT position hold at their current positions.

    Returns the per-wheel anchor positions, or None if any wheel reports a
    position too close to the widened ±PMAX mapping limit — holding there
    would risk a wrapped/clamped anchor and a phantom position error at full
    torque, so the caller falls back to velocity mode instead.
    """
    positions = await asyncio.gather(*[m.get_position() for m in motors])
    if any(abs(p) > 0.9 * _SESSION_PMAX for p in positions):
        return None
    await asyncio.gather(*[m.set_control_mode(ControlMode.IMPEDANCE) for m in motors])
    return list(positions)


async def _unpark(motors: list[MotorDriver]) -> None:
    """Return parked wheels to VELOCITY mode (clears the motors' command state)."""
    await asyncio.gather(*[m.set_control_mode(ControlMode.VELOCITY) for m in motors])


async def _control_loop(
    pad,  # noqa: ANN001 — pygame typed lazily
    motors: list[MotorDriver],
    lift: JiecangLift | None,
    height: HeightReader | None,
    args: argparse.Namespace,
):
    """Poll the gamepad and stream wheel commands until B is pressed.

    While the deadman is held the wheels track the stick in VELOCITY mode.
    Once it is released and the slew-limited command has ramped to zero, the
    wheels are parked (see :func:`_park`): held at their current positions by
    the motor's internal MIT position loop with ``--hold-kp``/``--hold-kd``.
    The hold command is re-sent every cycle to keep the lost-comm watchdog
    fed. Holding in the motor's own loop (rather than an outer software loop
    over CAN) is what makes the wheel rigid instead of giving first and
    correcting after.
    """
    import pygame

    interval = 1.0 / args.hz
    # Slew-limit the body command so releasing the deadman (or yanking the
    # stick) ramps the wheels instead of stepping them.
    max_delta = args.slew * interval
    cmd = [0.0, 0.0, 0.0]  # (vx, vy, wz), normalized [-1, 1]
    send_failed = False
    hold_pos: list[float] | None = None  # per-wheel park anchors (rad)
    park_failed = False  # anchor out of range — don't retry every cycle

    while True:
        t_iter = time.perf_counter()
        pygame.event.pump()

        if pad.get_button(_BTN_B):
            print("\nB pressed — stopping.")
            if hold_pos is not None:
                await _unpark(motors)
            return

        engaged = bool(pad.get_button(_BTN_LB) or pad.get_button(_BTN_RB))

        # Lift: hold-to-move on the D-pad, gated behind the same deadman as
        # the wheels. command() is edge-triggered internally, so calling it
        # every cycle is free and guarantees release on any state change.
        lift_dir = STOP
        if engaged and pad.get_numhats() > _HAT_DPAD:
            hat_y = pad.get_hat(_HAT_DPAD)[1]
            lift_dir = UP if hat_y > 0 else DOWN if hat_y < 0 else STOP
        if lift is not None:
            lift.command(lift_dir)
        lift_height = height.poll() if height is not None else None
        if engaged:
            target = (
                -_deadzone(pad.get_axis(_AXIS_LY), args.deadzone),  # vx: up = fwd
                -_deadzone(pad.get_axis(_AXIS_LX), args.deadzone),  # vy: left = +
                -_deadzone(pad.get_axis(_AXIS_RX), args.deadzone),  # wz: left = CCW
            )
        else:
            target = (0.0, 0.0, 0.0)

        for i in range(3):
            delta = target[i] - cmd[i]
            cmd[i] += max(-max_delta, min(max_delta, delta))

        speeds = _mix(cmd[0], cmd[1], cmd[2], args.max_speed, args.turn_scale)

        stopped = not engaged and all(abs(c) < 1e-3 for c in cmd)
        if motors:
            try:
                if engaged:
                    park_failed = False
                    if hold_pos is not None:
                        await _unpark(motors)
                        hold_pos = None

                if stopped and args.hold_kp > 0.0 and not park_failed:
                    if hold_pos is None:
                        hold_pos = await _park(motors)
                        if hold_pos is None:
                            park_failed = True
                            print(
                                "\n  WARNING: wheel position near the ±PMAX "
                                "mapping limit — parking disabled. Power-cycle "
                                "the base to reset wheel positions.\n"
                            )
                    if hold_pos is not None:
                        await asyncio.gather(
                            *[
                                m.set_impedance(p, 0.0, args.hold_kp, args.hold_kd, 0.0)
                                for m, p in zip(motors, hold_pos)
                            ]
                        )
                if hold_pos is None:
                    await asyncio.gather(
                        *[m.set_velocity(s) for m, s in zip(motors, speeds)]
                    )
                send_failed = False
            except Exception:
                # Transient send failures (buffer full, bus-off recovery) are
                # surfaced on the status line; the next cycle retries.
                send_failed = True

        print(
            _status_line(
                engaged,
                cmd,
                speeds,
                send_failed,
                hold_pos is not None,
                lift_dir if lift is not None else STOP,
                lift_height,
            ),
            end="",
            flush=True,
        )

        elapsed = time.perf_counter() - t_iter
        await asyncio.sleep(max(0.0, interval - elapsed))


async def _run(args: argparse.Namespace) -> None:
    pad = _init_gamepad(args.joystick)

    lift: JiecangLift | None = None
    height: HeightReader | None = None
    if not args.no_lift:
        lift = JiecangLift(args.lift_chip, args.lift_up_gpio, args.lift_down_gpio)
        print(
            f"Lift: {args.lift_chip} up=GPIO{args.lift_up_gpio} "
            f"down=GPIO{args.lift_down_gpio} — D-pad up/down (with deadman)."
        )
        if args.lift_height_port:
            height = HeightReader(args.lift_height_port)
    try:
        await _run_wheels(pad, lift, height, args)
    finally:
        if height is not None:
            height.close()
        if lift is not None:
            lift.close()


async def _run_wheels(
    pad,  # noqa: ANN001 — pygame typed lazily
    lift: JiecangLift | None,
    height: HeightReader | None,
    args: argparse.Namespace,
) -> None:
    if args.no_can:
        print("--no-can: gamepad check only, no wheel motors will move.")
        await _control_loop(pad, [], lift, height, args)
        return

    async with CanBus(args.channel) as bus:
        # IDs 1-4 collide with the arm-bus MyActuator IDs in the inference
        # table, so the Damiao protocol is forced explicitly.
        motors = [make_driver(bus, w.motor_id, motor_type="damiao") for w in _WHEELS]
        print(f"Enabling wheel motors on {args.channel} ...")
        # Widen the position-mapping range (RAM only) before enable() reads it
        # back, so multi-turn wheel positions stay valid for the MIT park hold.
        await asyncio.gather(
            *[m._write_register(_DM_REG_PMAX, _SESSION_PMAX) for m in motors]
        )
        await asyncio.gather(*[m.enable() for m in motors])
        for w, m in zip(_WHEELS, motors):
            if abs(m._p_max - _SESSION_PMAX) > 1.0:
                print(
                    f"  WARNING: {w.name} PMAX readback {m._p_max:.0f} != "
                    f"{_SESSION_PMAX:.0f} — parking may misbehave."
                )
        await asyncio.gather(
            *[m.set_control_mode(ControlMode.VELOCITY) for m in motors]
        )
        print("Motors enabled. Hold LB/RB to drive, B to quit.")
        try:
            await _control_loop(pad, motors, lift, height, args)
        finally:
            try:
                await asyncio.gather(*[m.set_velocity(0.0) for m in motors])
            except Exception:
                pass
            await asyncio.gather(*[m.disable() for m in motors])
            print("Motors disabled.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Drive the x-drive omni base with a Logitech gamepad.",
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
        "--hz", type=float, default=50.0, help="Command rate in Hz (default: 50)"
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
        help="Skip the CAN bus and just display the gamepad mixing (no motion).",
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
