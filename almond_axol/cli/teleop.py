"""
axol teleop --robot [axol|sim]

Run a VR teleoperation session with default parameters.
"""

import argparse
import asyncio
import logging
import socket

from ..shared import ARM_JOINTS, parse_stiffness


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("teleop", help="Run a VR teleoperation session.")
    p.add_argument(
        "--robot",
        choices=["axol", "sim"],
        required=True,
        help="Robot backend: 'axol' for hardware, 'sim' for visualizer.",
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
        "--left-gripper-torque-limit",
        type=float,
        default=0.5,
        help="Max output torque (Nm) for the left gripper in POSITION_FORCE mode (default: 0.5).",
    )
    p.add_argument(
        "--right-gripper-torque-limit",
        type=float,
        default=0.5,
        help="Max output torque (Nm) for the right gripper in POSITION_FORCE mode (default: 0.5).",
    )
    stiffness_help = (
        "Compliance ↔ stiffness blend in [0, 1] for the {side} arm. "
        f"Either a single value applied to all {len(ARM_JOINTS)} joints, "
        f"or {len(ARM_JOINTS)} comma-separated values (one per joint, in "
        f"order: {', '.join(j.value for j in ARM_JOINTS)}; gripper "
        "excluded). 0 is fully compliant; 1 restores the pre-tuning "
        "industrial gains; 0.5 (default) is the geometric mean. See "
        "AxolConfig.{attr}."
    )
    stiffness_metavar = "S|" + ",".join("S" for _ in ARM_JOINTS)
    p.add_argument(
        "--left-stiffness",
        type=parse_stiffness,
        default=0.5,
        metavar=stiffness_metavar,
        help=stiffness_help.format(side="left", attr="left_stiffness"),
    )
    p.add_argument(
        "--right-stiffness",
        type=parse_stiffness,
        default=0.5,
        metavar=stiffness_metavar,
        help=stiffness_help.format(side="right", attr="right_stiffness"),
    )
    p.add_argument(
        "--motion-scale",
        type=float,
        default=1.0,
        metavar="S",
        help=(
            "Server-side fallback for the VR→arm position multiplier (default: 1.0). "
            "1.0 is identity (VR motion == arm motion). Values <1.0 magnify "
            "VR motion (small controller move → larger arm move), useful when "
            "the operator's controllers drift toward the edge of headset "
            "tracking. Applied only when the incoming VRFrame leaves "
            "motion_scale at its 1.0 default — a client that sends an "
            "explicit motion_scale always wins. Orientation is never scaled."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    p.set_defaults(func=run)


def _get_local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level))

    hostname = socket.gethostname()
    local_ip = _get_local_ip()
    print("Connect the VR app (https://axol.almond.bot) to this machine:")
    print(f"  Hostname : {hostname}.local")
    print(f"  IP       : {local_ip}")

    asyncio.run(
        _run(
            args.robot,
            no_left=args.no_left,
            no_right=args.no_right,
            left_gripper_torque_limit=args.left_gripper_torque_limit,
            right_gripper_torque_limit=args.right_gripper_torque_limit,
            left_stiffness=args.left_stiffness,
            right_stiffness=args.right_stiffness,
            motion_scale=args.motion_scale,
        )
    )


async def _run(
    robot_type: str,
    *,
    no_left: bool = False,
    no_right: bool = False,
    left_gripper_torque_limit: float = 0.5,
    right_gripper_torque_limit: float = 0.5,
    left_stiffness: float | tuple[float, ...] = 0.5,
    right_stiffness: float | tuple[float, ...] = 0.5,
    motion_scale: float = 1.0,
) -> None:
    from ..robot import Axol, Sim
    from ..robot.config import AxolConfig
    from ..teleop import VRTeleop
    from ..teleop.config import VRTeleopConfig

    if robot_type == "sim":
        robot = Sim()
    else:
        kwargs = {}
        if no_left:
            kwargs["left_channel"] = None
        if no_right:
            kwargs["right_channel"] = None
        axol_config = AxolConfig(
            left_stiffness=left_stiffness,
            right_stiffness=right_stiffness,
        )
        axol_config.left.gripper.torque_limit = left_gripper_torque_limit
        axol_config.right.gripper.torque_limit = right_gripper_torque_limit
        robot = Axol(config=axol_config, **kwargs)
    teleop_config = VRTeleopConfig(motion_scale=motion_scale)
    async with VRTeleop(robot, config=teleop_config) as teleop:
        await teleop.run()
