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
        "--gripper-torque-limit",
        type=float,
        default=1.0,
        help="Max output torque (Nm) for the gripper in POSITION_FORCE mode (default: 1.0).",
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
        gripper_torque_limit=args.gripper_torque_limit,
    )


def _move_to_rest(robot, fps: int, duration_s: float = 5.0) -> None:
    """Send the robot to the default rest pose for ``duration_s`` seconds.

    Uses the rest poses defined in VRTeleopConfig (arm joints) with gripper
    fully open. The robot's impedance controller smoothly tracks the target.
    """
    import time

    from ..shared import ARM_JOINTS, Joint
    from ..teleop.config import VRTeleopConfig

    cfg = VRTeleopConfig()
    rest: dict[str, float] = {}
    for j in Joint:
        if j in ARM_JOINTS:
            arm_i = ARM_JOINTS.index(j)
            rest[f"left_{j.value}.pos"] = float(cfg.rest_pose_left[arm_i])
            rest[f"right_{j.value}.pos"] = float(cfg.rest_pose_right[arm_i])
        else:
            rest[f"left_{j.value}.pos"] = 1.0
            rest[f"right_{j.value}.pos"] = 1.0

    steps = max(1, int(duration_s * fps))
    for _ in range(steps):
        t0 = time.perf_counter()
        robot.send_action(rest)
        time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - t0)))


def _run(
    policy_path: str,
    task: str,
    episode_time_s: int,
    fps: int,
    repo_id: str | None,
    root: str | None,
    push_to_hub: bool,
    device: str,
    gripper_torque_limit: float = 1.0,
) -> None:
    from dataclasses import replace

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
    from ..robot.config import ArmConfig, AxolConfig

    # Load policy
    policy = PreTrainedPolicy.from_pretrained(policy_path)
    policy.config.device = device
    policy.to(device)
    policy.eval()

    left = ArmConfig()
    right = ArmConfig().mirror_gravity()
    gripper = replace(left.gripper, torque_limit=gripper_torque_limit)
    left = replace(left, gripper=gripper)
    right = replace(right, gripper=gripper)

    # Build robot with 3 ZED cameras
    robot_config = AxolRobotConfig(
        cameras={
            "overhead": ZedCameraConfig(port=30000, fps=fps, width=1280, height=720),
            "left_arm": ZedCameraConfig(port=30002, fps=fps, width=1280, height=720),
            "right_arm": ZedCameraConfig(port=30004, fps=fps, width=1280, height=720),
        },
        axol_config=AxolConfig(left=left, right=right),
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
                log_say("Returning to rest pose.")
                _move_to_rest(robot, fps)
                input("Reset the scene, then press Enter to start.")
                continue

            if dataset:
                dataset.save_episode()
            episodes_recorded += 1
            log_say("Returning to rest pose.")
            _move_to_rest(robot, fps)
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
