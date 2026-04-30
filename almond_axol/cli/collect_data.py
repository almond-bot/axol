"""
axol collect-data

Record teleoperation episodes with the Axol robot and three ZED cameras.
Episode boundaries are driven by VR controller commands:
  - DATA_COLLECTION → RECORDING:              start collecting frames
  - RECORDING → DATA_COLLECTION:              stop; save episode (success)
  - RECORDING → DATA_COLLECTION + reset btn:  stop; discard episode (rerecord)

While saving, the VR headset is pushed into the SAVING state so recording
controls are blocked until save_episode() completes.

Recording continues until Ctrl+C.
"""

import argparse
import logging
import socket
import time


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("collect-data", help="Record teleoperation episodes.")
    p.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace dataset repo ID (<user>/<dataset>).",
    )
    p.add_argument("--task", required=True, help="Natural language task description.")
    p.add_argument(
        "--fps", type=int, default=60, help="Recording frame rate (default: 60)."
    )
    p.add_argument(
        "--root",
        default=None,
        help="Local dataset root path (default: HF_LEROBOT_HOME).",
    )
    p.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push dataset to HuggingFace Hub when done.",
    )
    p.add_argument(
        "--zed-host",
        default="192.168.10.1",
        help="IP address of the ZED streamer (default: 192.168.10.1).",
    )
    p.add_argument(
        "--zed-iface",
        default=None,
        metavar="IFACE",
        help=(
            "Network interface to configure for the ZED link before connecting "
            "(e.g. eth0). Assigns 192.168.10.2/24 and requires sudo. "
            "Skip if the interface is already configured."
        ),
    )
    p.add_argument(
        "--gripper-torque-limit",
        type=float,
        default=1.0,
        help="Max output torque (Nm) for the gripper in POSITION_FORCE mode (default: 1.0).",
    )
    p.add_argument(
        "--rerun-ip",
        default=None,
        help=(
            "IP of a Rerun viewer running on your local machine. "
            "When set, streams live visualization to that viewer. "
            "On the local machine run: rerun --connect rerun+http://<robot-ip>:<port>/proxy"
        ),
    )
    p.add_argument(
        "--rerun-port",
        type=int,
        default=9876,
        help="Port of the Rerun viewer (default: 9876). Only used when --rerun-ip is set.",
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
    _run(
        repo_id=args.repo_id,
        task=args.task,
        fps=args.fps,
        root=args.root,
        push_to_hub=args.push_to_hub,
        zed_host=args.zed_host,
        zed_iface=args.zed_iface,
        gripper_torque_limit=args.gripper_torque_limit,
        rerun_ip=args.rerun_ip,
        rerun_port=args.rerun_port,
    )


def _run(
    repo_id: str,
    task: str,
    fps: int,
    root: str | None,
    push_to_hub: bool,
    zed_host: str = "192.168.10.1",
    zed_iface: str | None = None,
    gripper_torque_limit: float = 1.0,
    rerun_ip: str | None = None,
    rerun_port: int = 9876,
) -> None:
    from dataclasses import replace
    from pathlib import Path

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor import make_default_processors
    from lerobot.teleoperators.utils import TeleopEvents
    from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
    from lerobot.utils.feature_utils import (
        build_dataset_frame,
        hw_to_dataset_features,
    )
    from lerobot.utils.utils import log_say
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

    from ..lerobot.camera.configuration_zed import ZedCameraConfig
    from ..lerobot.robot.config_axol import AxolRobotConfig
    from ..lerobot.robot.robot_axol import AxolRobot
    from ..lerobot.teleop.config_vr import AxolVRTeleopConfig
    from ..lerobot.teleop.teleop_vr import AxolVRTeleop
    from ..robot.config import ArmConfig, AxolConfig
    from ..shared import setup_link_ip
    from ..vr.models import VRState

    if zed_iface:
        setup_link_ip(zed_iface, "192.168.10.2/24")

    left = ArmConfig()
    right = ArmConfig().mirror_gravity()
    gripper = replace(left.gripper, torque_limit=gripper_torque_limit)
    left = replace(left, gripper=gripper)
    right = replace(right, gripper=gripper)
    robot_config = AxolRobotConfig(
        cameras={
            "overhead": ZedCameraConfig(host=zed_host, port=30000),
            "left_arm": ZedCameraConfig(host=zed_host, port=30002),
            "right_arm": ZedCameraConfig(host=zed_host, port=30004),
        },
        axol_config=AxolConfig(left=left, right=right),
    )
    robot = AxolRobot(robot_config)
    teleop = AxolVRTeleop(AxolVRTeleopConfig())

    # Check resume eligibility before connecting (file check only)
    dataset_root = Path(root) if root else HF_LEROBOT_HOME / repo_id
    meta = dataset_root / "meta"
    has_info = (meta / "info.json").exists()
    is_complete = (
        has_info and (meta / "tasks.parquet").exists() and (meta / "episodes").is_dir()
    )
    if has_info and not is_complete:
        raise RuntimeError(
            f"Incomplete dataset found at {dataset_root} (missing tasks.parquet or episodes/). "
            f"Delete the directory and rerun to start fresh:\n"
            f"  rm -rf {dataset_root}"
        )

    hostname = socket.gethostname()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as _s:
        _s.connect(("8.8.8.8", 80))
        local_ip = _s.getsockname()[0]
    print("Connect the VR app (https://axol.almond.bot) to this machine:")
    print(f"  Hostname : {hostname}.local")
    print(f"  IP       : {local_ip}")

    if rerun_ip:
        init_rerun(session_name="axol_record", ip=rerun_ip, port=rerun_port)

    # Connect first — cameras auto-detect resolution and FPS from the stream,
    # which is then used to define the dataset observation features.
    robot.connect()

    if is_complete:
        log_say(f"Resuming existing dataset at {dataset_root}.")
        dataset = LeRobotDataset.resume(
            repo_id=repo_id,
            root=str(dataset_root),
            image_writer_threads=4,
        )
    else:
        action_features = hw_to_dataset_features(robot.action_features, ACTION)
        obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            features={**action_features, **obs_features},
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
        )
    pos_l, pos_r = robot.positions
    teleop.connect(q_start_left=pos_l, q_start_right=pos_r)
    teleop_action_proc, robot_action_proc, robot_obs_proc = make_default_processors()

    episodes_recorded = 0
    episode_idx = dataset.num_episodes  # global index; increments as episodes are saved
    try:
        while True:
            log_say(
                f"Episode {episode_idx + 1}: robot is at rest pose. Press record on the VR controller when ready."
            )
            dataset.clear_episode_buffer()
            recording = False
            rerecord = False

            while True:
                t0 = time.perf_counter()

                obs = robot.get_observation()
                obs_processed = robot_obs_proc(obs)

                teleop.send_feedback(obs)
                act = teleop.get_action()
                act_processed = teleop_action_proc((act, obs))
                robot.send_action(robot_action_proc((act_processed, obs)))

                events = teleop.get_teleop_events()

                if events.get("start_recording") and not recording:
                    recording = True
                    log_say("Recording started.")

                if recording:
                    obs_frame = build_dataset_frame(
                        dataset.features, obs_processed, prefix=OBS_STR
                    )
                    act_frame = build_dataset_frame(
                        dataset.features, act_processed, prefix=ACTION
                    )
                    dataset.add_frame({**obs_frame, **act_frame, "task": task})
                    if rerun_ip:
                        log_rerun_data(observation=obs_processed, action=act_processed)

                if events[TeleopEvents.TERMINATE_EPISODE]:
                    teleop.send_feedback_state(VRState.SAVING)
                    break
                if events[TeleopEvents.RERECORD_EPISODE]:
                    rerecord = True
                    break

                time.sleep(max(0.0, 1 / fps - (time.perf_counter() - t0)))

            # Return to rest pose before the next episode so the operator
            # starts each take from a consistent configuration.
            log_say("Returning to rest pose.")
            teleop.request_reset()
            reset_deadline = time.perf_counter() + 30.0
            while teleop.is_resetting and time.perf_counter() < reset_deadline:
                t0 = time.perf_counter()
                obs = robot.get_observation()
                act = teleop.get_action()
                robot.send_action(robot_action_proc((act, obs)))
                time.sleep(max(0.0, 1 / fps - (time.perf_counter() - t0)))
            # Drain any VR events that fired during the reset move.
            teleop.get_teleop_events()

            if rerecord:
                log_say("Re-recording episode.")
                continue

            if recording:
                log_say("Saving episode…")
                dataset.save_episode()
                episode_idx += 1
                episodes_recorded += 1
                log_say(
                    f"Saved episode {episode_idx} ({episodes_recorded} this session)."
                )
            else:
                log_say("Episode ended before recording started, skipping.")
            teleop.send_feedback_state(VRState.DATA_COLLECTION)

    except KeyboardInterrupt:
        pass
    finally:
        log_say("Stopping.")
        robot.disconnect()
        teleop.disconnect()
        dataset.finalize()
        if push_to_hub and episodes_recorded > 0:
            dataset.push_to_hub()
