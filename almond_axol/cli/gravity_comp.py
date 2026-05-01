"""
axol gravity_comp

Put the Axol arms into gravity-compensation mode so the operator can move them
by hand. Each free arm joint is sent ``set_impedance(p_des=current, v_des=0,
kp=0, kd=KD, t_ff=gravity)`` at the configured rate; joints not in the free
set are held rigidly at their current position with their configured
``ArmConfig`` gains; the gripper is held softly at its current position.

Examples:
    axol gravity_comp
    axol gravity_comp --no-right
    axol gravity_comp --kd 1.0
    axol gravity_comp --joints WRIST_3
    axol gravity_comp --no-right --joints SHOULDER_1,WRIST_3
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

from ..robot import Axol
from ..shared import ARM_JOINTS, Joint


def _parse_joints(spec: str) -> set[Joint]:
    """Parse a comma-separated joint name list into a set of ``Joint`` enums.

    Names are case-insensitive and match the ``Joint`` enum members
    (``SHOULDER_1``, ``SHOULDER_2``, ``SHOULDER_3``, ``ELBOW``, ``WRIST_1``,
    ``WRIST_2``, ``WRIST_3``). ``GRIPPER`` is rejected — gravity comp only
    applies to the 7 arm joints.
    """
    valid_names = [j.name for j in ARM_JOINTS]
    out: set[Joint] = set()
    for raw in spec.split(","):
        name = raw.strip().upper()
        if not name:
            continue
        try:
            j = Joint[name]
        except KeyError:
            raise argparse.ArgumentTypeError(
                f"unknown joint {name!r}; valid: {', '.join(valid_names)}"
            )
        if j not in ARM_JOINTS:
            raise argparse.ArgumentTypeError(
                f"{name!r} cannot be gravity-compensated; valid: {', '.join(valid_names)}"
            )
        out.add(j)
    if not out:
        raise argparse.ArgumentTypeError("--joints is empty")
    return out


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "gravity_comp",
        help="Hold the Axol in gravity-compensation mode (move by hand).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument(
        "--no-left",
        action="store_true",
        help="Disable the left arm.",
    )
    p.add_argument(
        "--no-right",
        action="store_true",
        help="Disable the right arm.",
    )
    p.add_argument(
        "--joints",
        type=_parse_joints,
        default=None,
        metavar="J1,J2,...",
        help=(
            "Comma-separated list of joints to gravity-compensate "
            "(e.g. WRIST_3 or SHOULDER_1,ELBOW). All other arm joints are "
            "held rigidly at their current position using their configured "
            "kp/kd. Default: all 7 arm joints free."
        ),
    )
    p.add_argument(
        "--kd",
        type=float,
        default=0.5,
        help="Velocity damping coefficient on *free* joints (Nm·s/rad). Higher = less floppy. Default 0.5.",
    )
    p.add_argument(
        "--rate",
        type=float,
        default=100.0,
        help="Control loop rate in Hz (default: 100).",
    )
    p.add_argument(
        "--telemetry-rate",
        type=float,
        default=500.0,
        help="Joint telemetry poll rate in Hz (default: 500).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level))
    try:
        asyncio.run(
            _run(
                no_left=args.no_left,
                no_right=args.no_right,
                free_joints=args.joints,
                kd=args.kd,
                rate_hz=args.rate,
                telemetry_hz=args.telemetry_rate,
            )
        )
    except KeyboardInterrupt:
        print("\nExiting gravity comp ...")


async def _run(
    *,
    no_left: bool,
    no_right: bool,
    free_joints: set[Joint] | None,
    kd: float,
    rate_hz: float,
    telemetry_hz: float,
) -> None:
    if no_left and no_right:
        raise SystemExit("Both arms disabled — nothing to do.")

    kwargs: dict = {}
    if no_left:
        kwargs["left_channel"] = None
    if no_right:
        kwargs["right_channel"] = None

    free_str = (
        "all 7 joints"
        if free_joints is None
        else ", ".join(j.name for j in ARM_JOINTS if j in free_joints)
    )
    print(
        f"Gravity comp: free={free_str}; kd={kd:.2f} Nm·s/rad, rate={rate_hz:.0f} Hz "
        f"(telemetry={telemetry_hz:.0f} Hz). Press Ctrl-C to exit."
    )

    async with Axol(**kwargs) as axol:
        # ``enable()`` (called by ``__aenter__``) leaves arm joints in IMPEDANCE
        # and the gripper in POSITION_FORCE — both of which are the modes
        # ``gravity_compensate`` expects, so we don't touch control modes here.
        await axol.start_telemetry(telemetry_hz)
        # Settle a few cycles so positions cache is populated before we drive.
        await asyncio.sleep(max(0.05, 5.0 / telemetry_hz))

        dt = 1.0 / rate_hz
        while True:
            loop_start = time.monotonic()
            await axol.gravity_compensate(kd=kd, free_joints=free_joints)
            spent = time.monotonic() - loop_start
            if spent < dt:
                await asyncio.sleep(dt - spent)
