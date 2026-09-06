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
from typing import Any, Callable

import numpy as np
from lerobot.robots.config import RobotConfig

from ..lerobot.robot.config_axol import AxolRobotConfig
from ..mantis.relative import quat_xyzw_to_rotvec
from ..mantis.smoothing import rotvec_to_quat_xyzw
from ..robot.base import HardwareCleanupError, mark_hardware_cleanup_uncertain
from .config import LogLevel, parse

_logger = logging.getLogger(__name__)

# Command rate interpolated playback upsamples to — the teleop control rate,
# so the arm receives setpoints at the same cadence it was driven with when
# the episode was recorded.
_INTERP_HZ = 120


def _slerp_rotvec(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate two rotation vectors along the shortest path on SO(3)."""
    q0 = rotvec_to_quat_xyzw(np.asarray(start, dtype=np.float64))
    q1 = rotvec_to_quat_xyzw(np.asarray(end, dtype=np.float64))
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0.0:
        # q and -q encode the same orientation. Pick the representative in the
        # same quaternion hemisphere so interpolation follows the short arc.
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        quat = q0 + alpha * (q1 - q0)
        quat /= max(float(np.linalg.norm(quat)), 1e-12)
    else:
        theta = float(np.arccos(dot))
        sin_theta = float(np.sin(theta))
        quat = (
            np.sin((1.0 - alpha) * theta) / sin_theta * q0
            + np.sin(alpha * theta) / sin_theta * q1
        )
    return quat_xyzw_to_rotvec(quat)


def _interpolate_action_values(
    base: np.ndarray,
    nxt: np.ndarray,
    alpha: float,
    action_names: list[str],
) -> np.ndarray:
    """Interpolate an action, using SO(3) for Cartesian EE orientations.

    Joint targets, Cartesian positions, and grippers retain the existing
    componentwise interpolation. Each complete ``*_ee.rx/.ry/.rz`` group is
    then replaced with shortest-path orientation interpolation. This matters
    at the canonical rotation-vector branch cut: adjacent equivalent poses can
    be represented near ``+pi * axis`` and ``-pi * axis``, whose scalar midpoint
    is the identity rather than the intended 180-degree orientation.
    """
    values = base + (nxt - base) * alpha
    by_name = {name: i for i, name in enumerate(action_names)}
    for name, rx in by_name.items():
        if not name.endswith("_ee.rx"):
            continue
        prefix = name[: -len("rx")]
        rotation_indices = [
            by_name.get(f"{prefix}{axis}") for axis in ("rx", "ry", "rz")
        ]
        if any(index is None for index in rotation_indices):
            continue
        indices = [int(index) for index in rotation_indices if index is not None]
        values[indices] = _slerp_rotvec(base[indices], nxt[indices], alpha)
    return values


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

    # Dataset to replay: a repo id under $HF_LEROBOT_HOME (as recorded by
    # collect-data, e.g. ``almond/pick-place``), or a filesystem path to a
    # dataset directory (one containing ``meta/info.json``) anywhere on disk.
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
    # Contact watchdog for every return-to-rest: a joint torque residual
    # (measured minus modeled gravity, Nm) sustained above this drops the
    # arms into a limp gravity-comp hold instead of pulling through. Replay
    # has no interactive retry channel, so the hold lasts until the run is
    # stopped (Ctrl+C or the UI's Stop). 0 disables the watchdog.
    reset_torque_threshold: float = 4.0
    # Contact watchdog while the episode itself plays back: the same
    # sustained-torque-residual trip, checked on every command. On a trip
    # playback stops and the arms drop into the limp gravity-comp hold until
    # the run is stopped. 0 (the default) disables it — replayed episodes
    # touch the scene on purpose, so only the return-to-rest guard is always
    # on; set a threshold (16 is the control panel's suggested value) to
    # enable. Mirrors the teleop config's field of the same name (`axol
    # teleop` and collect-data share the same knob in the control panel).
    teleop_torque_threshold: float = 0.0
    # Velocity damping (Nm·s/rad) for that contact-fallback hold; same
    # semantics as `axol gravity-comp --kd`.
    reset_gravity_comp_kd: float = 0.25
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


def _wait_for_replay_exit(
    exit_event: threading.Event,
    *,
    timeout: float = 5.0,
) -> tuple[bool, HardwareCleanupError | None]:
    """Wait for the tracked robot-loop coroutine's own finalizer to run."""
    if exit_event.wait(timeout=timeout):
        return True, None
    return False, HardwareCleanupError(
        f"replay playback coroutine did not stop within {timeout:g}s; robot "
        "hardware access may still be active"
    )


def _cleanup_replay_robot(
    *,
    playback_stopped: bool,
    cleanup: Callable[[], None],
) -> BaseException | None:
    """Disconnect only after the robot-loop playback exit is proved."""
    if not playback_stopped:
        _logger.error(
            "skipping robot disconnect because replay playback exit was not proved"
        )
        return None
    try:
        cleanup()
    except BaseException as error:
        _logger.exception("robot disconnect failed")
        return error
    return None


def _finish_replay_cleanup(
    *,
    session_error: BaseException | None,
    playback_failure: BaseException | None,
    disconnect_failure: BaseException | None,
    reset_failure: BaseException | None,
) -> None:
    """Propagate teardown failures without replacing a replay's primary error."""
    if session_error is not None:
        if playback_failure is not None:
            session_error.add_note(
                "additional replay playback cleanup failure: "
                f"{type(playback_failure).__name__}: {playback_failure}"
            )
        if disconnect_failure is not None:
            session_error.add_note(
                "additional robot disconnect cleanup failure: "
                f"{type(disconnect_failure).__name__}: {disconnect_failure}"
            )
        if reset_failure is not None:
            session_error.add_note(
                "additional IK reset worker cleanup failure: "
                f"{type(reset_failure).__name__}: {reset_failure}"
            )
        uncertain = playback_failure or disconnect_failure or reset_failure
        if uncertain is not None:
            mark_hardware_cleanup_uncertain(session_error, uncertain)
        return

    if playback_failure is not None:
        if disconnect_failure is not None:
            playback_failure.add_note(
                "additional robot disconnect cleanup failure: "
                f"{type(disconnect_failure).__name__}: {disconnect_failure}"
            )
        if reset_failure is not None:
            playback_failure.add_note(
                "additional IK reset worker cleanup failure: "
                f"{type(reset_failure).__name__}: {reset_failure}"
            )
        if isinstance(playback_failure, HardwareCleanupError):
            raise playback_failure
        raise HardwareCleanupError(
            "replay playback did not stop; hardware ownership is uncertain"
        ) from playback_failure

    if disconnect_failure is not None:
        error = HardwareCleanupError(
            "robot disconnect failed; hardware ownership is uncertain"
        )
        if reset_failure is not None:
            error.add_note(
                "additional IK reset worker cleanup failure: "
                f"{type(reset_failure).__name__}: {reset_failure}"
            )
        raise error from disconnect_failure
    if reset_failure is not None:
        raise HardwareCleanupError(
            "IK reset worker did not stop; background ownership is uncertain"
        ) from reset_failure


def _run(cfg: ReplayDatasetConfig, stop_event: "threading.Event | None" = None) -> None:
    """Load the episode, return to rest, replay its actions, then return to rest."""
    from ..utils.state_files import require_service_dataset_configuration

    require_service_dataset_configuration()

    from ..lerobot.robot.config_mantis import MantisRobotConfig

    if isinstance(cfg.robot_config, MantisRobotConfig):
        raise ValueError("replay-dataset does not support Mantis hardware")

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

    # ``repo_id`` doubles as a filesystem path: a dataset directory anywhere
    # on disk (absolute, relative, or ~) is addressed directly, so operators
    # can point at datasets outside $HF_LEROBOT_HOME without a separate
    # --root. LeRobotDataset only needs a valid-looking repo id once a root
    # is given, so the directory name stands in for it.
    from ..recording.datasets import is_dataset_dir, list_datasets
    from ..utils.state_files import (
        confine_service_dataset_path,
        privileged_service_active,
    )

    repo_path = Path(repo_id).expanduser()
    hosted_service = privileged_service_active()
    # Plain CLI users may replay an arbitrary local dataset by path. The root
    # service must not probe an operator-supplied filesystem path before it has
    # been confined to the configured LeRobot tree, so it deliberately skips
    # this path shorthand and treats the value as a repo id instead.
    if root is None and not hosted_service and is_dataset_dir(repo_path):
        root = str(repo_path)
        repo_id = repo_path.name

    # Verify the dataset is present and complete before loading (a clear error
    # beats LeRobotDataset's deeper failure, and mirrors collect-data's checks).
    dataset_root = Path(root) if root else HF_LEROBOT_HOME / repo_id
    if hosted_service:
        dataset_root = confine_service_dataset_path(
            dataset_root,
            label="replay dataset root",
        )
        root = str(dataset_root)
    meta = dataset_root / "meta"
    if not (meta / "info.json").exists():
        available = list_datasets()
        listing = (
            "Datasets on this machine:\n"
            + "\n".join(
                f"  {d.repo_id}"
                + (f"  ({d.episodes} episodes)" if d.episodes is not None else "")
                for d in available
            )
            if available
            else "No datasets found under $HF_LEROBOT_HOME."
        )
        raise FileNotFoundError(
            f"No LeRobot dataset found at {dataset_root} (missing meta/info.json). "
            "Pass --repo_id as one of the ids below or as a path to a dataset "
            f"directory (and --root if it isn't under $HF_LEROBOT_HOME).\n{listing}"
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

    def _stopped() -> bool:
        return stop_event.is_set()

    def _go_to_rest(
        message: str = "Returning to rest pose.", *, final: bool = False
    ) -> None:
        # Guarded: a sustained torque residual (contact) drops the arms into
        # a limp gravity-comp hold. Replay has no interactive retry channel, so the
        # hold lasts until the run is stopped — `rested` then stays False and
        # the teardown skips its redundant return attempt. The teardown's own
        # return (``final=True``) must play even though the stop flag is
        # already set, so it passes no stop hook; on contact it aborts
        # immediately instead of holding (nothing could end that hold).
        nonlocal rested
        log_say(message)
        rested = reset_controller.return_to_rest(
            robot,
            torque_threshold=cfg.reset_torque_threshold,
            gravity_comp_kd=cfg.reset_gravity_comp_kd,
            stopped=None if final else _stopped,
            on_contact=None
            if final
            else lambda: log_say(
                "Contact during return to rest — arms are limp. Free them, "
                "then stop the run."
            ),
        )

    # Interpolated playback commands the arms at ~_INTERP_HZ (the teleop rate)
    # between consecutive recorded actions. Episode timing is unchanged —
    # substeps subdivide each recorded frame's period. Joint targets, Cartesian
    # positions, and grippers blend componentwise; Cartesian rotations follow
    # the shortest path on SO(3), including across the rotation-vector pi cut.
    substeps = max(1, round(_INTERP_HZ / fps)) if cfg.interpolate else 1

    async def _play_episode() -> tuple[str, float] | None:
        """Stream the episode's actions from the robot's event loop.

        Runs *on* the robot's event loop so each command is dispatched inline
        via ``send_action_async`` — no per-frame cross-thread hop — and paces
        with absolute deadlines so a late wakeup is corrected on the next
        cycle instead of stretching the command interval (both mirror
        collect-data's hot loop). Regular command timing matters because
        ``motion_control`` derives its velocity/acceleration feedforward by
        differentiating commanded positions against wall time, so interval
        jitter comes out of the arm as torque jitter.

        With ``teleop_torque_threshold`` set (> 0), a torque residual
        sustained above it (the scene changed since the recording —
        something is in the way, or a gripper caught) stops playback and
        returns the tripped ``(joint, residual)``; ``None`` on a clean
        finish or stop.
        """
        from ..robot.control import ContactWatchdog

        watchdog = (
            ContactWatchdog(cfg.teleop_torque_threshold)
            if cfg.teleop_torque_threshold > 0
            else None
        )
        send_period = 1.0 / (fps * substeps)
        deadline = time.perf_counter()
        for idx in range(num_frames):
            base = action_matrix[idx]
            # Hold the last recorded action for its full frame; never
            # extrapolate past the end of the episode.
            nxt = action_matrix[idx + 1] if idx + 1 < num_frames else base
            for sub in range(substeps):
                if _stopped():
                    return None
                deadline += send_period
                values = (
                    base
                    if sub == 0
                    else _interpolate_action_values(
                        base, nxt, sub / substeps, action_names
                    )
                )
                action = {name: float(values[i]) for i, name in enumerate(action_names)}
                await robot.send_action_async(action)
                if watchdog is not None:
                    tripped = watchdog.update(robot.torque_residuals())
                    if tripped is not None:
                        joint, residual = tripped
                        _logger.warning(
                            "replay contact: %s torque residual %.1f exceeds "
                            "%.1f — stopping playback",
                            joint,
                            residual,
                            cfg.teleop_torque_threshold,
                        )
                        return tripped
                await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
        return None

    playback_done = threading.Event()
    active_playback_future: Any | None = None
    playback_stopped = True

    async def _play_episode_tracked() -> tuple[str, float] | None:
        try:
            return await _play_episode()
        finally:
            # A concurrent Future can report cancellation before its event-loop
            # task has actually unwound.  This finalizer, not Future.cancel(),
            # is the ownership proof used by teardown.
            playback_done.set()

    def _play_episode_blocking() -> tuple[str, float] | None:
        """Run the playback coroutine on the robot's loop; block until done.

        On Ctrl+C, signal the coroutine to unwind and wait for it to finish so
        it stops commanding the robot before teardown, then re-raise (the
        outer handler falls through to the return-to-rest teardown). Returns
        the playback watchdog's trip, or ``None``.
        """
        nonlocal active_playback_future, playback_stopped
        playback_done.clear()
        coroutine = _play_episode_tracked()
        try:
            fut = asyncio.run_coroutine_threadsafe(coroutine, robot.event_loop)
        except BaseException:
            coroutine.close()
            raise
        # Retain the exact Future until the tracked coroutine's finalizer
        # proves exit. A timed-out interrupt deliberately does not cancel and
        # discard it: teardown must stay away from the robot while it is live.
        active_playback_future = fut
        playback_stopped = False
        try:
            result = fut.result()
        except KeyboardInterrupt:
            stop_event.set()
            try:
                fut.result(timeout=5.0)
            except TimeoutError:
                _logger.error(
                    "replay playback did not acknowledge stop within 5s; "
                    "retaining its Future and deferring robot teardown"
                )
            except BaseException as error:
                # The operator's interrupt remains the primary outcome, but a
                # completed playback fault is useful diagnostic context.
                _logger.warning(
                    "replay playback exited with an error while stopping: %s", error
                )
            if playback_done.is_set():
                playback_stopped = True
                active_playback_future = None
            raise
        except BaseException:
            if playback_done.is_set():
                playback_stopped = True
                active_playback_future = None
            raise
        if not playback_done.is_set():
            raise HardwareCleanupError(
                "replay Future completed before its coroutine exit could be proved"
            )
        playback_stopped = True
        active_playback_future = None
        return result

    session_error: BaseException | None = None
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
            contact = _play_episode_blocking()

            if contact is not None:
                # Playback hit something the recording didn't expect: hold
                # the arms limp so the operator can clear them by hand.
                # Replay has no interactive continue channel, so the hold
                # lasts until the run is stopped (Ctrl+C or the UI's Stop);
                # the teardown's final return then parks the arm.
                log_say(
                    "Contact during playback — the arms are limp and free "
                    "to move. Free them, then stop the run."
                )
                reset_controller.hold_limp(
                    robot,
                    gravity_comp_kd=cfg.reset_gravity_comp_kd,
                    wait=None,
                    stopped=_stopped,
                )
                # The arms were hand-guided during the hold: clear the stale
                # command history so the teardown's return-to-rest isn't
                # rejected by the max-step safety check.
                robot.reset_command_state()
                break

            if not loop or _stopped():
                break
            # Looping: return to rest between takes so the next replay restarts
            # smoothly from the recorded start pose.
            _go_to_rest()
    except KeyboardInterrupt:
        pass
    except BaseException as error:
        session_error = error
        raise
    finally:
        # Ignore SIGINT during cleanup so a second Ctrl+C can't abort partway
        # through the return-to-rest or teardown (mirrors run-policy).
        import signal

        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except (ValueError, OSError):
            pass

        playback_failure: BaseException | None = None
        if not playback_stopped:
            playback_stopped, playback_failure = _wait_for_replay_exit(playback_done)
            if playback_stopped:
                active_playback_future = None

        # Keep this exact reference reachable through the final ownership
        # decision. When exit is unproved, the robot event loop also retains
        # the submitted coroutine and OperationRunner locks the process after
        # the HardwareCleanupError below.
        _ = active_playback_future

        # Park the arm at rest before killing the operation, unless it's already
        # there (a loop iteration just ended at rest) or never moved (connect
        # failed). The reset is planned by the IK worker (a quick round-trip) and
        # then played locally, so it still completes if a slow stop's watchdog
        # force-kills the worker mid-move.
        if playback_stopped and robot.is_connected and not rested:
            try:
                _go_to_rest("Replay finished. Returning to rest pose.", final=True)
            except Exception:  # noqa: BLE001 - best-effort; still tear down
                _logger.warning("return-to-rest during teardown failed", exc_info=True)

        log_say("Stopping.")
        disconnect_failure = _cleanup_replay_robot(
            playback_stopped=playback_stopped,
            cleanup=robot.disconnect,
        )
        reset_failure: BaseException | None = None
        try:
            reset_controller.stop()
        except BaseException as exc:
            _logger.exception("IK reset worker cleanup failed")
            reset_failure = exc

        try:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        except (ValueError, OSError):
            pass
        _finish_replay_cleanup(
            session_error=session_error,
            playback_failure=playback_failure,
            disconnect_failure=disconnect_failure,
            reset_failure=reset_failure,
        )
