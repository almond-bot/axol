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

The teleop loop runs at ``--teleop_hz`` and publishes the latest
``(joint_obs, action)`` to a single-slot ``_SnapshotPublisher``. A separate
``_CaptureThread`` ticks at ``--fps`` and, for each tick, blocks on
``ZedCamera.read_at_or_after(T_n)`` per camera so every recorded frame
shares the capture instant ``T_n`` with the joint sample, then writes
the dataset row off the hot control loop.
"""

import logging
import shutil
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig

from ..lerobot.camera.configuration_zed import ZedCameraConfig
from ..lerobot.robot.config_axol import AxolRobotConfig
from ..lerobot.teleop.config_vr import AxolVRTeleopConfig
from ..utils.jetson import pin_realtime_clocks
from .config import LogLevel, parse

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.types import RobotAction, RobotObservation

    from ..lerobot.robot.robot_axol import AxolRobot

_logger = logging.getLogger(__name__)


def _default_robot_config() -> AxolRobotConfig:
    """Default Axol robot config for data collection: three local ZED cameras.

    Each camera's serial number is **required** — draccus takes dict
    fields as one inline YAML/JSON value, so pass
    ``--robot_config.cameras "{overhead: {serial: 41234567}, left_arm:
    {serial: 41234568}, right_arm: {serial: 41234569}}"`` (the zero
    placeholders below are stripped from the config overlay so draccus
    enforces the input). Other fields are overridable too, e.g.
    ``--robot_config.axol_config.left.elbow.kp 60``.
    """
    return AxolRobotConfig(
        cameras={
            "overhead": ZedCameraConfig(serial=0),
            "left_arm": ZedCameraConfig(serial=0),
            "right_arm": ZedCameraConfig(serial=0),
        },
    )


def _register_camera_video(robot: "AxolRobot", teleop: Any) -> None:
    """Register the ZED cameras as WebRTC video sources for the headset.

    Relays every camera the robot exposes (overhead — or ``overhead_left`` /
    ``overhead_right`` when stereo — plus both wrist cameras) so the headset can
    show them. Frame-driven (see ``_ZedFrameSource``): each relayed frame is
    encoded as soon as it's captured. Reads only consume the latest frame each
    camera already keeps, so the dataset capture pipeline is never blocked.
    """
    from .teleop import _ZedFrameSource

    sources: dict[str, Callable[[], Any]] = {
        name: _ZedFrameSource(cam) for name, cam in robot.cameras.items()
    }

    if not sources:
        return

    try:
        teleop.set_video_sources(sources)
    except Exception as exc:
        _logger.warning("failed to enable camera video: %s", exc)


@dataclass
class CollectDataConfig:
    """Config for ``axol collect-data``.

    ``robot_config`` and ``teleop_config`` are the full lerobot subsystem
    configs (cameras, per-joint gains, IK, VR server); nest into
    them from the CLI (e.g. ``--robot_config.axol_config.left_stiffness
    0.8``) or supply a whole-config file with ``--config_path``.
    """

    repo_id: str
    task: str
    robot_config: RobotConfig = field(default_factory=_default_robot_config)
    teleop_config: TeleoperatorConfig = field(default_factory=AxolVRTeleopConfig)
    fps: int = 60
    teleop_hz: int = 120
    root: str | None = None
    push_to_hub: bool = False
    rerun_ip: str | None = None
    rerun_port: int = 9876
    log_level: LogLevel = "INFO"


@dataclass
class _Snapshot:
    """``(joint_obs, action, ts)`` bundle from one teleop tick."""

    joint_obs: "RobotObservation"
    action: "RobotAction"
    ts: float


class _SnapshotPublisher:
    """Single-slot publisher shared between the teleop loop and capture thread.

    The teleop loop rebuilds fresh ``joint_obs`` / ``action`` dicts every
    tick and calls :meth:`publish`; the capture thread reads the latest
    slot via :meth:`latest`. The lock protects the slot pointer only — the
    contained dicts are never mutated in place.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: _Snapshot | None = None
        self._first_event = threading.Event()

    def publish(
        self,
        joint_obs: "RobotObservation",
        action: "RobotAction",
        ts: float,
    ) -> None:
        snap = _Snapshot(joint_obs=joint_obs, action=action, ts=ts)
        with self._lock:
            self._latest = snap
        self._first_event.set()

    def latest(self) -> _Snapshot | None:
        with self._lock:
            return self._latest

    def wait_for_first(self, timeout: float) -> bool:
        return self._first_event.wait(timeout=timeout)


class _CaptureThread(threading.Thread):
    """Capture dataset frames at ``fps`` Hz, decoupled from the teleop loop.

    Each tick the thread sleeps until ``T_n = recording_start + n / fps``,
    waits for a frame with ``capture_perf_ts >= T_n`` from every camera,
    pulls the latest joint+action snapshot from ``publisher``, and appends
    one dataset row. If any camera read times out the previous frame for
    that camera is reused (logged at DEBUG); if no frame has ever arrived
    for it the tick is skipped.
    """

    def __init__(
        self,
        *,
        publisher: _SnapshotPublisher,
        robot: "AxolRobot",
        dataset: "LeRobotDataset",
        robot_obs_proc: Callable[[Any], Any],
        fps: int,
        task: str,
        rerun_ip: str | None,
    ) -> None:
        super().__init__(name="axol-capture", daemon=True)
        self.publisher = publisher
        self.robot = robot
        self.dataset = dataset
        self.robot_obs_proc = robot_obs_proc
        self.fps = fps
        self.task = task
        self.rerun_ip = rerun_ip
        self.stop_event = threading.Event()

    def run(self) -> None:
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame
        from lerobot.utils.visualization_utils import log_rerun_data

        if not self.publisher.wait_for_first(timeout=5.0):
            _logger.warning(
                "Capture thread saw no publisher snapshot within 5s; exiting."
            )
            return
        if self.stop_event.is_set():
            return

        frame_interval = 1.0 / self.fps
        timeout_ms = int(2 * frame_interval * 1000 + 200)
        recording_start = time.perf_counter()
        last_frames: dict[str, tuple[Any, float, float]] = {}
        tick = 0

        while not self.stop_event.is_set():
            target_perf_ts = recording_start + tick * frame_interval

            wait_s = target_perf_ts - time.perf_counter()
            if wait_s > 0 and self.stop_event.wait(timeout=wait_s):
                return

            frames: dict[str, tuple[Any, float, float]] = {}
            skip_tick = False
            for cam_key, cam in self.robot.cameras.items():
                try:
                    frame, cap_ts, recv_ts = cam.read_at_or_after(  # type: ignore[attr-defined]
                        target_perf_ts, timeout_ms=timeout_ms
                    )
                except (TimeoutError, RuntimeError) as exc:
                    cached = last_frames.get(cam_key)
                    if cached is None:
                        _logger.debug(
                            "Capture tick %d: %s read failed (%s) and no "
                            "cached frame; skipping tick.",
                            tick,
                            cam_key,
                            exc,
                        )
                        skip_tick = True
                        break
                    _logger.debug(
                        "Capture tick %d: %s read failed (%s); reusing cached frame.",
                        tick,
                        cam_key,
                        exc,
                    )
                    frame, cap_ts, recv_ts = cached
                frames[cam_key] = (frame, cap_ts, recv_ts)
                last_frames[cam_key] = (frame, cap_ts, recv_ts)

            if skip_tick:
                tick += 1
                continue

            snap = self.publisher.latest()
            if snap is None:
                tick += 1
                continue

            obs: dict[str, Any] = dict(snap.joint_obs)
            for cam_key, (frame, _cap_ts, _recv_ts) in frames.items():
                obs[cam_key] = frame
            obs_processed = self.robot_obs_proc(obs)

            obs_frame = build_dataset_frame(
                self.dataset.features, obs_processed, prefix=OBS_STR
            )
            act_frame = build_dataset_frame(
                self.dataset.features, snap.action, prefix=ACTION
            )
            if self.stop_event.is_set():
                return
            self.dataset.add_frame({**obs_frame, **act_frame, "task": self.task})

            if self.rerun_ip:
                log_rerun_data(observation=obs_processed, action=snap.action)

            if _logger.isEnabledFor(logging.DEBUG) and tick % 30 == 0:
                cam_skews = ", ".join(
                    f"{k}: cap-T={1e3 * (cap_ts - target_perf_ts):+.1f}ms"
                    for k, (_, cap_ts, _) in frames.items()
                )
                _logger.debug(
                    "Capture tick %d skews — %s, T-snap.ts=%+.1fms",
                    tick,
                    cam_skews,
                    1e3 * (target_perf_ts - snap.ts),
                )

            tick += 1


def main(argv: list[str]) -> None:
    """Parse the CLI config and run a data-collection session."""
    cfg = parse(CollectDataConfig, argv)
    # force=True: importing lerobot (at module load) installs a root handler
    # and leaves the root level at WARNING, which would otherwise make this a
    # no-op and silently drop every log_say() status line.
    logging.basicConfig(level=getattr(logging, cfg.log_level), force=True)

    # Pin the Jetson clocks now, while a sudo prompt can still reach the tty
    # (mid-session escalation is non-interactive): the CPU governor for the
    # IK loop rate, and — when cameras stream — the NVENC/VIC engines for
    # encode latency.
    pin_realtime_clocks(interactive=True)

    _run(cfg)


def _run(cfg: CollectDataConfig, stop_event: "threading.Event | None" = None) -> None:
    from pathlib import Path

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor import make_default_processors
    from lerobot.teleoperators.utils import TeleopEvents
    from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
    from lerobot.utils.feature_utils import (
        hw_to_dataset_features,
    )
    from lerobot.utils.utils import log_say
    from lerobot.utils.visualization_utils import init_rerun

    from ..lerobot.robot.robot_axol import AxolRobot
    from ..lerobot.teleop.teleop_vr import AxolVRTeleop
    from ..vr.models import VRState

    repo_id = cfg.repo_id
    task = cfg.task
    fps = cfg.fps
    teleop_hz = cfg.teleop_hz
    root = cfg.root
    push_to_hub = cfg.push_to_hub
    rerun_ip = cfg.rerun_ip
    rerun_port = cfg.rerun_port

    robot = AxolRobot(cfg.robot_config)
    teleop = AxolVRTeleop(cfg.teleop_config)

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

    # Connect first — cameras auto-detect resolution and FPS on open, which
    # is then used to define the dataset observation features.
    robot.connect()

    if is_complete:
        log_say(f"Resuming existing dataset at {dataset_root}.")
        dataset = LeRobotDataset.resume(
            repo_id=repo_id,
            root=str(dataset_root),
            image_writer_threads=4,
            streaming_encoding=True,
            encoder_threads=4,
            vcodec="auto",
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
            streaming_encoding=True,
            encoder_threads=4,
            vcodec="auto",
        )
    pos_l, pos_r = robot.positions
    teleop.connect(q_start_left=pos_l, q_start_right=pos_r)

    # Relay the overhead + wrist cameras to the headset so the operator can see
    # the scene and grippers. Best-effort: reuses the frames ZedCamera decodes.
    _register_camera_video(robot, teleop)

    teleop_action_proc, robot_action_proc, robot_obs_proc = make_default_processors()

    episodes_recorded = 0
    episode_idx = dataset.num_episodes
    teleop_interval = 1.0 / teleop_hz
    publisher = _SnapshotPublisher()
    capture: _CaptureThread | None = None

    def _stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    try:
        while not _stopped():
            log_say(
                f"Episode {episode_idx + 1}: robot is at rest pose. Press record on the VR controller when ready."
            )
            dataset.clear_episode_buffer()
            recording = False
            rerecord = False

            while True:
                if _stopped():
                    break
                t0 = time.perf_counter()

                # Camera reads happen on the capture thread; the teleop loop
                # only ever touches joint state.
                joint_obs = robot.get_joint_observation()
                teleop.send_feedback(joint_obs)
                act = teleop.get_action()
                act_processed = teleop_action_proc((act, joint_obs))
                robot.send_action(robot_action_proc((act_processed, joint_obs)))

                publisher.publish(joint_obs, act_processed, t0)

                events = teleop.get_teleop_events()

                if events.get("start_recording") and not recording:
                    recording = True
                    capture = _CaptureThread(
                        publisher=publisher,
                        robot=robot,
                        dataset=dataset,
                        robot_obs_proc=robot_obs_proc,
                        fps=fps,
                        task=task,
                        rerun_ip=rerun_ip,
                    )
                    capture.start()
                    log_say("Recording started.")

                if events[TeleopEvents.TERMINATE_EPISODE]:
                    teleop.send_feedback_state(VRState.SAVING)
                    break
                if events[TeleopEvents.RERECORD_EPISODE]:
                    rerecord = True
                    break

                time.sleep(max(0.0, teleop_interval - (time.perf_counter() - t0)))

            if capture is not None:
                capture.stop_event.set()
                capture.join()
                capture = None

            if _stopped():
                break

            log_say("Returning to rest pose.")
            teleop.request_reset()
            reset_deadline = time.perf_counter() + 30.0
            while teleop.is_resetting and time.perf_counter() < reset_deadline:
                t0 = time.perf_counter()
                joint_obs = robot.get_joint_observation()
                act = teleop.get_action()
                robot.send_action(robot_action_proc((act, joint_obs)))
                time.sleep(max(0.0, teleop_interval - (time.perf_counter() - t0)))
            # Drain VR events fired during the reset move.
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
    except Exception:
        teleop.send_feedback_error()
        raise
    finally:
        if capture is not None:
            capture.stop_event.set()
            capture.join()

        log_say("Stopping.")

        robot.disconnect()
        teleop.disconnect()

        dataset.finalize()

        if push_to_hub and episodes_recorded > 0:
            dataset.push_to_hub()

        if not is_complete and episodes_recorded == 0 and dataset_root.exists():
            try:
                shutil.rmtree(dataset_root)
                log_say(f"No episodes saved — removed empty dataset at {dataset_root}.")
            except OSError as exc:
                _logger.warning(
                    "Failed to remove empty dataset at %s: %s", dataset_root, exc
                )
