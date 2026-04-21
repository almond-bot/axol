"""
axol run-policy

Run a trained policy on the Axol robot with three ZED cameras.
The policy drives actions autonomously for a fixed duration per episode.
After each episode the operator is prompted to save, rerecord, or quit,
giving them time to reset the scene. Runs until Ctrl+C or 'q'.
"""

import argparse
import logging


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("run-policy", help="Run a trained policy on the robot.")
    p.add_argument(
        "--policy",
        required=True,
        help="Local path or HuggingFace repo ID of the trained policy checkpoint.",
    )
    p.add_argument("--task", required=True, help="Natural language task description.")
    p.add_argument(
        "--episode-time-s",
        type=int,
        default=30,
        help="Max duration of each episode in seconds (default: 30).",
    )
    p.add_argument(
        "--fps", type=int, default=30, help="Control loop frame rate (default: 30)."
    )
    p.add_argument(
        "--repo-id",
        default=None,
        help="HuggingFace dataset repo ID to save rollouts (<user>/<dataset>). Optional.",
    )
    p.add_argument(
        "--root",
        default=None,
        help="Local dataset root path (default: HF_LEROBOT_HOME).",
    )
    p.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push rollout dataset to HuggingFace Hub when done.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Torch device for policy inference (default: cuda).",
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
        policy_path=args.policy,
        task=args.task,
        episode_time_s=args.episode_time_s,
        fps=args.fps,
        repo_id=args.repo_id,
        root=args.root,
        push_to_hub=args.push_to_hub,
        device=args.device,
    )


def _run(
    policy_path: str,
    task: str,
    episode_time_s: int,
    fps: int,
    repo_id: str | None,
    root: str | None,
    push_to_hub: bool,
    device: str,
) -> None:
    from lerobot.datasets.feature_utils import hw_to_dataset_features
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pretrained import PreTrainedPolicy
    from lerobot.processor import make_default_processors
    from lerobot.scripts.lerobot_record import record_loop
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.utils import log_say
    from lerobot.utils.visualization_utils import init_rerun

    from ..lerobot.camera.configuration_zed import ZedCameraConfig
    from ..lerobot.robot.config_axol import AxolRobotConfig
    from ..lerobot.robot.robot_axol import AxolRobot

    # Load policy
    policy = PreTrainedPolicy.from_pretrained(policy_path)
    policy.config.device = device
    policy.to(device)
    policy.eval()

    # Build robot with 3 ZED cameras
    robot_config = AxolRobotConfig(
        cameras={
            "overhead": ZedCameraConfig(port=30000, fps=fps, width=1280, height=720),
            "left_arm": ZedCameraConfig(port=30002, fps=fps, width=1280, height=720),
            "right_arm": ZedCameraConfig(port=30004, fps=fps, width=1280, height=720),
        }
    )
    robot = AxolRobot(robot_config)

    # Dataset (optional — only created if --repo-id is specified)
    dataset = None
    if repo_id:
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

    # Policy pre/post processors — normalization stats loaded from checkpoint
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    teleop_action_proc, robot_action_proc, robot_obs_proc = make_default_processors()

    robot.connect()

    # No keyboard listener — this command must work over SSH.
    # Episode control is handled via stdin prompts between episodes.
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    init_rerun(session_name="axol_run_policy")

    episodes_recorded = 0
    try:
        while True:
            log_say(f"Running episode {episodes_recorded + 1}.")

            record_loop(
                robot=robot,
                events=events,
                fps=fps,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                control_time_s=episode_time_s,
                single_task=task,
                display_data=True,
                teleop_action_processor=teleop_action_proc,
                robot_action_processor=robot_action_proc,
                robot_observation_processor=robot_obs_proc,
            )

            choice = (
                input("Episode done. [Enter]=save, r=rerecord, q=quit: ")
                .strip()
                .lower()
            )

            if choice == "q":
                break

            if choice == "r":
                log_say("Re-recording episode.")
                if dataset:
                    dataset.clear_episode_buffer()
                input("Reset the scene, then press Enter to start.")
                continue

            if dataset:
                dataset.save_episode()
            episodes_recorded += 1
            input("Reset the scene, then press Enter to start the next episode.")

    except KeyboardInterrupt:
        pass
    finally:
        log_say("Stopping.")
        robot.disconnect()
        if dataset:
            dataset.finalize()
            if push_to_hub and episodes_recorded > 0:
                dataset.push_to_hub()
