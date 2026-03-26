"""
almond-axol collect-data

Record teleoperation episodes with the Axol robot and three ZED cameras.
Episode boundaries are driven by VR controller commands:
  - DATA_COLLECTION → RECORDING:  start collecting frames
  - RECORDING → DATA_COLLECTION:  save episode (success)
  - RECORDING → DATA_COLLECTION + reset button: discard episode (rerecord)

Recording continues until Ctrl+C.
"""

import argparse
import logging
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
        "--fps", type=int, default=30, help="Recording frame rate (default: 30)."
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
    )


def _run(
    repo_id: str,
    task: str,
    fps: int,
    root: str | None,
    push_to_hub: bool,
) -> None:
    from pathlib import Path

    from lerobot.datasets.feature_utils import (
        build_dataset_frame,
        hw_to_dataset_features,
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor import make_default_processors
    from lerobot.teleoperators.utils import TeleopEvents
    from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
    from lerobot.utils.utils import log_say
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

    from ..lerobot.camera.configuration_zed import ZedCameraConfig
    from ..lerobot.robot.config_axol import AxolRobotConfig
    from ..lerobot.robot.robot_axol import AxolRobot
    from ..lerobot.teleop.config_vr import AxolVRTeleopConfig
    from ..lerobot.teleop.teleop_vr import AxolVRTeleop

    robot_config = AxolRobotConfig(
        cameras={
            "overhead": ZedCameraConfig(port=30000, fps=fps, width=1280, height=720),
            "left_arm": ZedCameraConfig(port=30002, fps=fps, width=1280, height=720),
            "right_arm": ZedCameraConfig(port=30004, fps=fps, width=1280, height=720),
        }
    )
    robot = AxolRobot(robot_config)
    teleop = AxolVRTeleop(AxolVRTeleopConfig())

    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)

    dataset_root = Path(root) if root else HF_LEROBOT_HOME / repo_id
    if (dataset_root / "meta" / "info.json").exists():
        log_say(f"Resuming existing dataset at {dataset_root}.")
        dataset = LeRobotDataset(repo_id=repo_id, root=root)
        dataset.start_image_writer(num_threads=4)
    else:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            features={**action_features, **obs_features},
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
        )

    init_rerun(session_name="axol_record")
    robot.connect()
    teleop.connect()
    teleop_action_proc, robot_action_proc, robot_obs_proc = make_default_processors()

    episodes_recorded = 0
    episode_idx = dataset.num_episodes  # global index; increments as episodes are saved
    try:
        while True:
            log_say(
                f"Episode {episode_idx + 1}: move to start position, then press record on the VR controller."
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
                    log_rerun_data(observation=obs_processed, action=act_processed)

                if events[TeleopEvents.TERMINATE_EPISODE]:
                    break
                if events[TeleopEvents.RERECORD_EPISODE]:
                    rerecord = True
                    break

                time.sleep(max(0.0, 1 / fps - (time.perf_counter() - t0)))

            if rerecord:
                log_say("Re-recording episode.")
                continue

            if recording:
                dataset.save_episode()
                episode_idx += 1
                episodes_recorded += 1
                log_say(
                    f"Saved episode {episode_idx} ({episodes_recorded} this session)."
                )
            else:
                log_say("Episode ended before recording started, skipping.")

    except KeyboardInterrupt:
        pass
    finally:
        log_say("Stopping.")
        robot.disconnect()
        teleop.disconnect()
        dataset.finalize()
        if push_to_hub and episodes_recorded > 0:
            dataset.push_to_hub()
