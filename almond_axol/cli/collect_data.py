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

The teleop loop runs at ``--teleop-hz`` and publishes the latest
``(joint_obs, action)`` to a single-slot ``_SnapshotPublisher``. A separate
``_CaptureThread`` ticks at ``--fps`` and, for each tick, blocks on
``ZedCamera.read_at_or_after(T_n)`` per camera so every recorded frame
shares the sender-clock instant ``T_n`` with the joint sample, then writes
the dataset row off the hot control loop.
"""

import argparse
import logging
import shutil
import socket
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from ..shared import ARM_JOINTS, parse_stiffness

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.types import RobotAction, RobotObservation

    from ..lerobot.robot.robot_axol import AxolRobot

_logger = logging.getLogger(__name__)


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


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("collect-data", help="Record teleoperation episodes.")
    p.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace dataset repo ID (<user>/<dataset>).",
    )
    p.add_argument("--task", required=True, help="Natural language task description.")
    p.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Dataset recording frame rate (default: 60).",
    )
    p.add_argument(
        "--teleop-hz",
        type=int,
        default=120,
        help=(
            "Motor command rate in Hz (default: 120, matching the IK loop). "
            "Teleop runs at this rate while dataset frames are captured at --fps."
        ),
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
        "--left-gripper-torque-limit",
        type=float,
        default=1.0,
        help="Max output torque (Nm) for the left gripper in POSITION_FORCE mode (default: 1.0).",
    )
    p.add_argument(
        "--right-gripper-torque-limit",
        type=float,
        default=1.0,
        help="Max output torque (Nm) for the right gripper in POSITION_FORCE mode (default: 1.0).",
    )
    stiffness_help = (
        "Compliance ↔ stiffness blend in [0, 1] for the {side} arm. "
        f"Either a single value applied to all {len(ARM_JOINTS)} joints, "
        f"or {len(ARM_JOINTS)} comma-separated values (one per joint, in "
        f"order: {', '.join(j.value for j in ARM_JOINTS)}; gripper "
        "excluded). 0 (default) is fully compliant; 1 restores the "
        "pre-tuning industrial gains. See AxolConfig.{attr}."
    )
    stiffness_metavar = "S|" + ",".join("S" for _ in ARM_JOINTS)
    p.add_argument(
        "--left-stiffness",
        type=parse_stiffness,
        default=0.0,
        metavar=stiffness_metavar,
        help=stiffness_help.format(side="left", attr="left_stiffness"),
    )
    p.add_argument(
        "--right-stiffness",
        type=parse_stiffness,
        default=0.0,
        metavar=stiffness_metavar,
        help=stiffness_help.format(side="right", attr="right_stiffness"),
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
        teleop_hz=args.teleop_hz,
        root=args.root,
        push_to_hub=args.push_to_hub,
        zed_host=args.zed_host,
        zed_iface=args.zed_iface,
        left_gripper_torque_limit=args.left_gripper_torque_limit,
        right_gripper_torque_limit=args.right_gripper_torque_limit,
        left_stiffness=args.left_stiffness,
        right_stiffness=args.right_stiffness,
        rerun_ip=args.rerun_ip,
        rerun_port=args.rerun_port,
    )


def _run(
    repo_id: str,
    task: str,
    fps: int,
    teleop_hz: int = 120,
    root: str | None = None,
    push_to_hub: bool = False,
    zed_host: str = "192.168.10.1",
    zed_iface: str | None = None,
    left_gripper_torque_limit: float = 1.0,
    right_gripper_torque_limit: float = 1.0,
    left_stiffness: float | tuple[float, ...] = 0.0,
    right_stiffness: float | tuple[float, ...] = 0.0,
    rerun_ip: str | None = None,
    rerun_port: int = 9876,
) -> None:
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

    from ..lerobot.camera.configuration_zed import ZedCameraConfig
    from ..lerobot.robot.config_axol import AxolRobotConfig
    from ..lerobot.robot.robot_axol import AxolRobot
    from ..lerobot.teleop.config_vr import AxolVRTeleopConfig
    from ..lerobot.teleop.teleop_vr import AxolVRTeleop
    from ..robot.config import AxolConfig
    from ..shared import setup_link_ip
    from ..vr.models import VRState

    if zed_iface:
        setup_link_ip(zed_iface, "192.168.10.2/24")

    axol_config = AxolConfig(
        left_stiffness=left_stiffness,
        right_stiffness=right_stiffness,
    )
    axol_config.left.gripper.torque_limit = left_gripper_torque_limit
    axol_config.right.gripper.torque_limit = right_gripper_torque_limit
    robot_config = AxolRobotConfig(
        cameras={
            "overhead": ZedCameraConfig(host=zed_host, port=30000),
            "left_arm": ZedCameraConfig(host=zed_host, port=30002),
            "right_arm": ZedCameraConfig(host=zed_host, port=30004),
        },
        axol_config=axol_config,
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
    teleop_action_proc, robot_action_proc, robot_obs_proc = make_default_processors()

    episodes_recorded = 0
    episode_idx = dataset.num_episodes
    teleop_interval = 1.0 / teleop_hz
    publisher = _SnapshotPublisher()
    capture: _CaptureThread | None = None
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
