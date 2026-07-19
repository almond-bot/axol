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
then a final return-to-rest leaves the arm parked. Playback runs on the
robot's event loop with absolute-deadline pacing (like collect-data's control
loop) so command intervals stay regular; ``--interpolate`` additionally
upsamples the recorded actions to ~120 Hz commands for smoother tracking.

With ``--loop`` the episode replays continuously (returning to rest between
takes) until stopped with Ctrl+C, or Stop in the control panel; either way the
arm returns to the rest pose before the operation exits.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig

from ..lerobot.robot.config_axol import AxolRobotConfig
from .config import LogLevel, parse

_logger = logging.getLogger(__name__)

# Command rate interpolated playback upsamples to — the teleop control rate,
# so the arm receives setpoints at the same cadence it was driven with when
# the episode was recorded.
_INTERP_HZ = 120


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
    # Smooth playback by linearly interpolating between recorded actions and
    # commanding the arms at ~120 Hz (the teleop control rate) instead of the
    # dataset fps. Episode timing is unchanged; only the command granularity
    # increases. Off by default (each recorded action is sent once, as-is).
    interpolate: bool = False
    # Replay the episode on a loop until stopped (Ctrl+C, or Stop in the UI),
    # returning to rest between takes. Off by default (a single replay). Either
    # way the arm returns to the rest pose before the operation exits.
    loop: bool = False
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
    """Load the episode, return to rest, replay its actions, then return to rest."""
    from pathlib import Path

    import numpy as np
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
    # Pull the whole episode's actions into one numpy array up front: indexing
    # the Arrow-backed dataset per frame inside the timed playback loop has
    # variable latency (chunk decode), which would land directly in the command
    # interval and show up as jerk (see the pacing note in the playback loop).
    actions = dataset.select_columns(ACTION)
    action_matrix = np.stack(
        [np.asarray(actions[i][ACTION], dtype=np.float64) for i in range(num_frames)]
    )

    # Spawn the IK worker now so its JAX JIT (~10-20 s) overlaps with the robot
    # connect, exactly as run-policy does before its policy load.
    reset_controller = IKResetController()
    reset_controller.start()
    log_say("Started IK reset worker (collision-aware return-to-rest).")

    # Tracks whether the arm is currently parked at rest, so the teardown only
    # adds a return-to-rest when one is actually needed (and not a redundant one
    # right after a loop iteration already ended at rest).
    rested = False

    def _go_to_rest(message: str = "Returning to rest pose.") -> None:
        nonlocal rested
        log_say(message)
        reset_controller.return_to_rest(robot)
        rested = True

    def _stopped() -> bool:
        return stop_event.is_set()

    # Interpolated playback commands the arms at ~_INTERP_HZ (the teleop rate)
    # by linearly blending between consecutive recorded actions. Episode timing
    # is unchanged — substeps subdivide each recorded frame's period. Linear
    # blending is exact for joint targets and a good small-step approximation
    # for Cartesian poses (positions are linear; consecutive rotation vectors
    # are close enough that lerp ~= slerp at these deltas).
    substeps = max(1, round(_INTERP_HZ / fps)) if cfg.interpolate else 1

    async def _play_episode() -> None:
        """Stream the episode's actions from the robot's event loop.

        Runs *on* the robot's event loop so each command is dispatched inline
        via ``send_action_async`` — no per-frame cross-thread hop — and paces
        with absolute deadlines so a late wakeup is corrected on the next
        cycle instead of stretching the command interval (both mirror
        collect-data's hot loop). Regular command timing matters because
        ``motion_control`` derives its velocity/acceleration feedforward by
        differentiating commanded positions against wall time, so interval
        jitter comes out of the arm as torque jitter.
        """
        send_period = 1.0 / (fps * substeps)
        deadline = time.perf_counter()
        for idx in range(num_frames):
            base = action_matrix[idx]
            # Hold the last recorded action for its full frame; never
            # extrapolate past the end of the episode.
            nxt = action_matrix[idx + 1] if idx + 1 < num_frames else base
            for sub in range(substeps):
                if _stopped():
                    return
                deadline += send_period
                values = base if sub == 0 else base + (nxt - base) * (sub / substeps)
                action = {name: float(values[i]) for i, name in enumerate(action_names)}
                await robot.send_action_async(action)
                await asyncio.sleep(max(0.0, deadline - time.perf_counter()))

    def _play_episode_blocking() -> None:
        """Run the playback coroutine on the robot's loop; block until done.

        On Ctrl+C, signal the coroutine to unwind and wait for it to finish so
        it stops commanding the robot before teardown, then re-raise (the
        outer handler falls through to the return-to-rest teardown).
        """
        fut = asyncio.run_coroutine_threadsafe(_play_episode(), robot.event_loop)
        try:
            fut.result()
        except KeyboardInterrupt:
            stop_event.set()
            try:
                fut.result(timeout=5.0)
            except BaseException:  # noqa: BLE001 - best-effort unwind
                fut.cancel()
            raise

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
        interp_note = (
            f", interpolated to {fps * substeps} Hz commands" if substeps > 1 else ""
        )
        iteration = 0
        # Replay once, or repeatedly when ``loop`` is set, until stopped (Ctrl+C
        # or the UI's Stop). The arm is parked at rest before the op exits — on a
        # clean finish and on a stop alike — by the teardown below.
        while not _stopped():
            iteration += 1
            rested = False
            if loop:
                log_say(
                    f"Replaying episode {episode} (loop {iteration}): "
                    f"{num_frames} frames at {fps} fps{interp_note}."
                )
            else:
                log_say(
                    f"Replaying episode {episode}: {num_frames} frames at "
                    f"{fps} fps{interp_note}."
                )
            _play_episode_blocking()

            if not loop or _stopped():
                break
            # Looping: return to rest between takes so the next replay restarts
            # smoothly from the recorded start pose.
            _go_to_rest()
    except KeyboardInterrupt:
        pass
    finally:
        # Ignore SIGINT during cleanup so a second Ctrl+C can't abort partway
        # through the return-to-rest or teardown (mirrors run-policy).
        import signal

        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except (ValueError, OSError):
            pass

        # Park the arm at rest before killing the operation, unless it's already
        # there (a loop iteration just ended at rest) or never moved (connect
        # failed). The reset is planned by the IK worker (a quick round-trip) and
        # then played locally, so it still completes if a slow stop's watchdog
        # force-kills the worker mid-move.
        if robot.is_connected and not rested:
            try:
                _go_to_rest("Replay finished. Returning to rest pose.")
            except Exception:  # noqa: BLE001 - best-effort; still tear down
                _logger.warning("return-to-rest during teardown failed", exc_info=True)

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
