"""
axol teleop --robot [axol|sim]

Run a VR teleoperation session with default parameters.

While teleop runs, a torque-telemetry tap writes one JSON line per cycle to
``/dev/shm/axol_torque.jsonl`` so the dashboard's force HUD can read joint
torques without contending with this process for the CAN bus. The tap is a
no-op on the Sim backend (no torque sensors) and is silent on transient
read errors so it cannot crash teleop.
"""

import argparse
import asyncio
import json
import logging
import os
import socket
import time

from ..shared import ARM_JOINTS, parse_stiffness

_TORQUE_JSONL_PATH = "/dev/shm/axol_torque.jsonl"
_TORQUE_HZ = 30.0


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
        )
    )


async def _torque_writer(robot, fh, hz: float) -> None:
    """Periodically fetch joint torques and append one JSON line per cycle.

    Schema matches /home/twolabs/axol-ui/torque_publisher.py: one line per
    sample with ``ts``, ``left[8]``, ``right[8]``. The dashboard reads the
    file via SSE; the FastAPI side trims it as needed.

    Silent on any per-cycle error so a transient telemetry hiccup never
    crashes teleop. Exits on :class:`asyncio.CancelledError`.
    """
    period = 1.0 / hz
    while True:
        cycle_start = time.perf_counter()
        try:
            left_t, right_t = await robot.get_torques()
        except asyncio.CancelledError:
            raise
        except Exception:
            # Don't kill teleop on a telemetry hiccup.
            left_t = None
            right_t = None

        try:
            sample = {
                "ts": time.time(),
                "left": [float(v) for v in (left_t if left_t is not None else [])],
                "right": [float(v) for v in (right_t if right_t is not None else [])],
            }
            fh.write(json.dumps(sample) + "\n")
        except Exception:
            pass

        try:
            await asyncio.sleep(max(0.0, period - (time.perf_counter() - cycle_start)))
        except asyncio.CancelledError:
            raise


async def _run(
    robot_type: str,
    *,
    no_left: bool = False,
    no_right: bool = False,
    left_gripper_torque_limit: float = 0.5,
    right_gripper_torque_limit: float = 0.5,
    left_stiffness: float | tuple[float, ...] = 0.5,
    right_stiffness: float | tuple[float, ...] = 0.5,
) -> None:
    from ..robot import Axol, Sim
    from ..robot.config import AxolConfig
    from ..teleop import VRTeleop

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

    async with VRTeleop(robot) as teleop:
        # Robot is now enabled by VRTeleop.enable(). Start the torque tap.
        torque_fh = None
        torque_task: asyncio.Task[None] | None = None
        telemetry_started = False
        try:
            try:
                await robot.start_telemetry(hz=_TORQUE_HZ, torque=True)
                telemetry_started = True
            except (AttributeError, NotImplementedError):
                # Sim or any robot subclass without telemetry support.
                logging.getLogger(__name__).info(
                    "Robot does not support start_telemetry(torque=True); "
                    "skipping torque JSONL tap."
                )
            except Exception as exc:  # pragma: no cover - defensive
                logging.getLogger(__name__).warning(
                    "Failed to start torque telemetry: %s", exc
                )

            if telemetry_started:
                try:
                    os.makedirs(os.path.dirname(_TORQUE_JSONL_PATH), exist_ok=True)
                    # Line-buffered so each sample is immediately visible to
                    # the dashboard's SSE tailer.
                    torque_fh = open(_TORQUE_JSONL_PATH, "w", buffering=1)
                    torque_task = asyncio.create_task(
                        _torque_writer(robot, torque_fh, _TORQUE_HZ)
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logging.getLogger(__name__).warning(
                        "Failed to open torque JSONL %s: %s",
                        _TORQUE_JSONL_PATH,
                        exc,
                    )

            await teleop.run()
        finally:
            if torque_task is not None:
                torque_task.cancel()
                try:
                    await torque_task
                except (asyncio.CancelledError, Exception):
                    pass
            if torque_fh is not None:
                try:
                    torque_fh.close()
                except Exception:
                    pass
            if telemetry_started:
                try:
                    await robot.stop_telemetry()
                except Exception:
                    pass
