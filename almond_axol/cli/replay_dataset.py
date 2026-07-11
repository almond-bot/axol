"""
axol replay-dataset

Replay a recorded LeRobot episode on the Axol robot: stream the episode's
recorded actions back to the arms frame by frame, then return to the rest pose
at the end (the same collision-aware return-to-rest ``run-policy`` uses).

This is the inverse of ``collect-data`` — instead of recording teleop actions,
it plays an already-recorded episode's ``action`` column straight onto the
hardware. No cameras, teleop, or policy server are involved: it only needs the
arms, so the robot config carries no cameras.

The robot is moved to the rest pose before playback so the arm starts from the
same place every episode does in ``collect-data`` (episodes are recorded from
rest, so the first replayed action is ~rest and there's no jump). Each frame's
action is sent at the dataset's recorded fps to reproduce the original timing,
then the soft-shutdown park leaves the arms eased down at the zero pose.

With ``--loop`` the episode replays continuously (returning to rest between
takes) until stopped with Ctrl+C, or Stop in the control panel. On a clean
stop the arms are parked — eased to the rest pose and then to the all-zeros
pose — before the motors release (``--park_on_exit false`` opts out); after an
error the possibly-faulted robot is never commanded, and the motors release
where they are.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig

from ..lerobot.robot.config_axol import AxolRobotConfig
from .config import LogLevel, parse

_logger = logging.getLogger(__name__)


def _default_robot_config() -> AxolRobotConfig:
    """Default Axol robot config for replay: arms only, no cameras.

    Replay neither records nor streams video — it just plays recorded
    actions back onto the arms — so no camera slots are seeded (an empty
    ``cameras`` dict opens the arms only). ``telemetry_hz=0`` skips the
    background poll loop: like ``collect-data``, a ``motion_control`` command
    is issued every step, whose feedback frames keep the position cache fresh,
    so the redundant telemetry transactions would only contend on the bus.
    """
    return AxolRobotConfig(telemetry_hz=0.0)


@dataclass
class ReplayDatasetConfig:
    """Config for ``axol replay-dataset``.

    Selects one episode of an existing LeRobot dataset and replays its recorded
    actions on the robot. ``robot_config`` is the full Axol robot config (CAN
    channels, per-joint gains); nest into it from the CLI (e.g.
    ``--robot_config.axol_config.left_stiffness 0.8``) or pass a whole-config
    file with ``--config_path``. Match the stiffness used at data-collection
    time so the arm tracks the recorded actions the same way.
    """

    repo_id: str
    episode: int
    robot_config: RobotConfig = field(default_factory=_default_robot_config)
    root: str | None = None
    # Playback rate. ``0`` (the default) replays at the dataset's recorded fps,
    # reproducing the original timing; set a positive value to override it.
    fps: int = 0
    # Replay the episode on a loop until stopped (Ctrl+C, or Stop in the UI),
    # returning to rest between takes. Off by default (a single replay). Either
    # way a clean stop parks the arms before the operation exits (see
    # park_on_exit).
    loop: bool = False
    # Park the arms on a clean stop (Ctrl+C or Stop in the control panel):
    # ease both arms to the rest pose and then to the all-zeros pose before
    # the motors release. Skipped after a propagating error — a
    # possibly-faulted robot is never commanded blind. A second Ctrl+C during
    # the park releases the motors immediately.
    park_on_exit: bool = True
    # Rest pose targeted by the pre-replay/between-take resets and the park's
    # first leg: 7 arm-joint angles (radians, in Joint enum order), one list
    # per arm. None keeps the built-in VRTeleopConfig rest pose.
    rest_pose_left: list[float] | None = None
    rest_pose_right: list[float] | None = None
    log_level: LogLevel = "INFO"


def main(argv: list[str]) -> None:
    """Parse the CLI config and replay the selected episode."""
    cfg = parse(ReplayDatasetConfig, argv)
    # force=True: importing lerobot (at module load) installs a root handler and
    # leaves the root level at WARNING, which would otherwise make this a no-op
    # and silently drop every log_say() status line.
    logging.basicConfig(level=getattr(logging, cfg.log_level), force=True)

    import sys

    import can

    from ..motor.errors import MotorError

    # Translate operator-actionable hardware faults into a clean non-zero exit
    # instead of a multi-frame traceback (mirrors run-policy).
    try:
        _run(cfg)
    except (MotorError, can.CanError) as exc:
        _logger.error("Robot hardware error: %s. Exiting.", exc)
        sys.exit(1)


def _run(cfg: ReplayDatasetConfig, stop_event: "threading.Event | None" = None) -> None:
    """Load the episode, return to rest, replay its actions, then park the arms."""
    from pathlib import Path

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME
    from lerobot.utils.utils import log_say

    from ..lerobot.robot.robot_axol import AxolRobot
    from ..lerobot.rollout import IKResetController

    if stop_event is None:
        stop_event = threading.Event()

    repo_id = cfg.repo_id
    episode = cfg.episode
    root = cfg.root

    # Verify the dataset is present and complete before loading (a clear error
    # beats LeRobotDataset's deeper failure, and mirrors collect-data's checks).
    dataset_root = Path(root) if root else HF_LEROBOT_HOME / repo_id
    meta = dataset_root / "meta"
    if not (meta / "info.json").exists():
        raise FileNotFoundError(
            f"No LeRobot dataset found at {dataset_root} (missing meta/info.json). "
            "Pass --repo_id (and --root if it isn't under $HF_LEROBOT_HOME)."
        )

    log_say(f"Loading episode {episode} from {dataset_root}.")
    try:
        dataset = LeRobotDataset(repo_id, root=str(dataset_root), episodes=[episode])
    except Exception as exc:  # noqa: BLE001 - surface a clean message
        raise RuntimeError(
            f"Could not load episode {episode} from the dataset at {dataset_root}: "
            f"{exc}. Check that the episode index exists."
        ) from exc

    num_frames = dataset.num_frames
    if num_frames == 0:
        raise ValueError(
            f"Episode {episode} in the dataset at {dataset_root} has no frames."
        )

    # Playback fps: the dataset's recorded fps unless overridden.
    fps = cfg.fps if cfg.fps and cfg.fps > 0 else dataset.fps

    # The recorded action layout must cover the robot's action keys. A joint
    # dataset stores one column per `{side}_{joint}.pos`; a Cartesian dataset
    # (observe_cartesian) stores per-arm `{side}_ee.{axis}` end-effector poses
    # plus gripper. Match the robot to whichever the dataset recorded so its
    # action_features line up and send_action picks the right path — joints go
    # straight out, Cartesian poses are resolved to joints via IK (as in
    # run-policy). Validate up front so a mismatched dataset fails clearly
    # instead of KeyError-ing inside send_action.
    action_names = list(dataset.features[ACTION]["names"])
    recorded_cartesian = any("_ee." in name for name in action_names)
    if isinstance(cfg.robot_config, AxolRobotConfig):
        cfg.robot_config.observe_cartesian = recorded_cartesian
    if recorded_cartesian:
        log_say("Cartesian dataset: replaying EE poses via inverse kinematics.")
    robot = AxolRobot(cfg.robot_config)
    missing = [k for k in robot.action_features if k not in action_names]
    if missing:
        raise ValueError(
            f"Dataset at {dataset_root} is missing action(s) {missing} the robot "
            f"expects (recorded actions: {action_names}). It wasn't recorded for "
            "this robot."
        )
    actions = dataset.select_columns(ACTION)

    # Spawn the IK worker now so its JAX JIT (~10-20 s) overlaps with the robot
    # connect, exactly as run-policy does before its policy load. A configured
    # rest pose (rest_pose_left / rest_pose_right) rides in on a VRTeleopConfig;
    # the defaults keep the stock config.
    vr_config = None
    if cfg.rest_pose_left is not None or cfg.rest_pose_right is not None:
        import numpy as np

        from ..teleop.config import VRTeleopConfig

        rest_overrides = {}
        for name, pose in (
            ("rest_pose_left", cfg.rest_pose_left),
            ("rest_pose_right", cfg.rest_pose_right),
        ):
            if pose is None:
                continue
            # Fail here with a clear message — a wrong-length pose otherwise
            # surfaces ~20 s later as a shape error inside the IK worker.
            if len(pose) != 7:
                raise ValueError(
                    f"--{name} must list exactly 7 arm-joint angles in "
                    f"radians (Joint enum order); got {len(pose)}."
                )
            rest_overrides[name] = np.asarray(pose, dtype=np.float32)
        vr_config = VRTeleopConfig(**rest_overrides)
    reset_controller = IKResetController(vr_config=vr_config)
    reset_controller.start()
    log_say("Started IK reset worker (collision-aware return-to-rest).")

    def _go_to_rest(message: str = "Returning to rest pose.") -> None:
        log_say(message)
        reset_controller.return_to_rest(robot)

    def _stopped() -> bool:
        return stop_event.is_set()

    try:
        log_say("Connecting robot...")
        robot.connect()

        # A Cartesian dataset resolves each recorded EE pose to joints via IK in
        # send_action. Build that solver now so its one-time JIT warmup overlaps
        # the return-to-rest below instead of stalling the first replayed frame.
        if recorded_cartesian:
            log_say("Preparing Cartesian action solver (IK)...")
            robot.prepare_cartesian_actions()

        # Start every take from rest, the same place collect-data records from,
        # so the first replayed action is ~rest and there's no jump.
        _go_to_rest()

        loop = bool(cfg.loop)
        period = 1.0 / fps
        iteration = 0
        # Replay once, or repeatedly when ``loop`` is set, until stopped (Ctrl+C
        # or the UI's Stop). On a clean finish and on a stop alike, the arms
        # are parked (rest pose, then zero) by the teardown below.
        while not _stopped():
            iteration += 1
            if loop:
                log_say(
                    f"Replaying episode {episode} (loop {iteration}): "
                    f"{num_frames} frames at {fps} fps."
                )
            else:
                log_say(
                    f"Replaying episode {episode}: {num_frames} frames at {fps} fps."
                )
            for idx in range(num_frames):
                if _stopped():
                    break
                t0 = time.perf_counter()
                action_array = actions[idx][ACTION]
                action = {
                    name: float(action_array[i]) for i, name in enumerate(action_names)
                }
                robot.send_action(action)
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))

            if not loop or _stopped():
                break
            # Looping: return to rest between takes so the next replay restarts
            # smoothly from the recorded start pose.
            _go_to_rest()
    except KeyboardInterrupt:
        pass
    finally:
        import signal
        import sys

        # Park on a clean stop: ease the arms to the rest pose and then to the
        # all-zeros pose BEFORE robot.disconnect() releases the motors. The
        # rest leg is a zero-length move when a loop iteration already ended at
        # rest (the planner handles it — no special case). Skipped after a
        # propagating error: the robot may be faulted and must not be commanded
        # blind. Runs before the SIG_IGN below so the park's own raising
        # handler lets a second Ctrl+C abort it; park() itself is best-effort
        # and never raises.
        clean = sys.exc_info()[0] is None
        try:
            if clean and cfg.park_on_exit and robot.is_connected:
                reset_controller.park(robot)
        except KeyboardInterrupt:
            # Ctrl+C before the park installed its raising handler: skip it.
            pass

        # Ignore SIGINT for the rest of cleanup so another Ctrl+C can't abort
        # partway through teardown (mirrors run-policy).
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except (ValueError, OSError):
            pass

        log_say("Stopping.")
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            pass
        try:
            reset_controller.stop()
        except Exception:  # noqa: BLE001
            pass

        try:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        except (ValueError, OSError):
            pass
