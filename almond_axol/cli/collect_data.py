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

import asyncio
import logging
import os
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
from ..utils.proc_diag import SystemDiag
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
        # The control loop runs motion_control every step, whose command replies
        # keep the joint cache fresh — so the background telemetry poll loop is
        # redundant CAN/CPU load. Skipping it (telemetry_hz=0) matches `axol
        # teleop` and keeps the control rate from sagging when teleop engages.
        telemetry_hz=0.0,
    )


def _register_camera_video(robot: "AxolRobot", teleop: Any) -> None:
    """Register the ZED cameras as WebRTC video sources for the headset.

    Relays every camera the robot exposes (overhead — or ``overhead_left`` /
    ``overhead_right`` when stereo — plus both wrist cameras) so the headset can
    show them. Each camera is registered bare and the relay picks the right
    WebRTC track per source (see :func:`almond_axol.vr.video._track_for_source`):
    a gst camera/eye already produces GPU-encoded H.264 access units (its
    ``subscribe()`` feeds a pre-encoded track — the same grab/encode serves the
    dataset), while an SDK camera is adapted to a frame-driven source that
    encodes each frame as soon as it's captured. Reads only consume the latest
    frame each camera already keeps, so the dataset capture pipeline is never
    blocked.
    """
    if not robot.cameras:
        return

    try:
        teleop.set_video_sources(dict(robot.cameras))
    except Exception as exc:
        _logger.warning("failed to enable camera video: %s", exc)


def _start_video_relay(cfg: "CollectDataConfig") -> Any | None:
    """Start the out-of-process video relay for data collection.

    The relay subprocess opens the ZED cameras on the GPU-resident gst pipeline,
    streams the headset view over WebRTC (aiortc), **and** publishes each
    camera's raw RGB frames back to this process through shared memory for the
    dataset (see :mod:`almond_axol.vr.shm_frames`). This keeps the control
    process off the camera grab/encode/RTP path entirely, so the teleop and IK
    loops stay as fast as ``axol teleop`` — even while recording.

    Returns the :class:`VideoRelayProcess`, or ``None`` when it can't be used
    (no cameras or aiortc unavailable), in which case the caller uses the
    in-process camera path. The caller must still verify the relay exported raw
    frames for every observation camera before relying on it.
    """
    cameras = getattr(cfg.robot_config, "cameras", {})
    if not cameras:
        return None
    try:
        from ..vr.video import webrtc_available
        from ..vr.video_proc import VideoRelayProcess
    except Exception as exc:  # noqa: BLE001 - aiortc / gst module missing
        _logger.debug("video relay unavailable: %s", exc)
        return None
    if not webrtc_available():
        return None

    specs: dict[str, dict[str, Any]] = {}
    for name, camcfg in cameras.items():
        spec: dict[str, Any] = {
            "serial": int(camcfg.serial),
            "fps": camcfg.fps or 60,
            "stereo": bool(getattr(camcfg, "stereo", False)),
        }
        res = camcfg.resolution_name() if hasattr(camcfg, "resolution_name") else None
        if res:
            spec["resolution"] = res
        specs[name] = spec

    relay = VideoRelayProcess(specs, want_raw=True)
    if not relay.has_sources:
        relay.shutdown()
        return None
    return relay


def _default_vcodec() -> str:
    """Pick a video codec that can actually open on this machine.

    LeRobot's "auto" prefers the NVIDIA hardware encoder (``h264_nvenc``)
    whenever the codec is compiled into ffmpeg, but on Jetson/Tegra (aarch64)
    there's no desktop ``libnvidia-encode`` to back it, so it fails to open and
    kills the encoder thread mid episode. Default to CPU "h264" (software
    libx264) on aarch64 and let "auto" pick the HW encoder everywhere else.
    """
    import platform

    return "h264" if platform.machine() == "aarch64" else "auto"


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
    # Video codec for the recorded LeRobot dataset; defaults per-platform (see
    # _default_vcodec). Override with any of LeRobot's VALID_VIDEO_CODECS
    # (e.g. auto, h264, libsvtav1).
    vcodec: str = field(default_factory=_default_vcodec)
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

    # System setup (Jetson clock pinning, the GStreamer NVENC stack) is handled
    # by the host installer + its boot service, not here — see
    # `axol jetson.setup` / `axol gst.install`. This entry point just runs.

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
    vcodec = cfg.vcodec
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

    # Prefer the out-of-process video relay: it owns the cameras (gst grab +
    # NVENC + WebRTC) in a subprocess and ships raw frames back via shared
    # memory, so the control loops stay as fast as `axol teleop`. Only use it
    # when it exported raw frames for every observation camera; otherwise tear
    # it down and fall back to the in-process camera path (robot owns cameras).
    relay = _start_video_relay(cfg)
    expected = set(cfg.robot_config.observation_cameras().keys())
    use_relay = relay is not None and expected <= set(relay.raw_cameras)
    if use_relay:
        robot.set_external_cameras({k: relay.raw_cameras[k] for k in expected})
    elif relay is not None:
        _logger.info(
            "video relay exported raw frames for %s but the dataset needs %s; "
            "using the in-process camera path instead.",
            sorted(relay.raw_cameras),
            sorted(expected),
        )
        relay.shutdown()
        relay = None

    # Connect first — cameras auto-detect resolution and FPS on open, which
    # is then used to define the dataset observation features. With the relay
    # the robot's cameras are shared-memory proxies, so this only opens the arms.
    # If any of this setup fails, tear the relay subprocess down so it doesn't
    # leak a held camera (it is daemonic, but a long-lived parent could outlive
    # the failure).
    try:
        robot.connect()

        if is_complete:
            log_say(f"Resuming existing dataset at {dataset_root}.")
            dataset = LeRobotDataset.resume(
                repo_id=repo_id,
                root=str(dataset_root),
                image_writer_threads=4,
                streaming_encoding=True,
                encoder_threads=4,
                vcodec=vcodec,
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
                vcodec=vcodec,
            )
        pos_l, pos_r = robot.positions
        teleop.connect(q_start_left=pos_l, q_start_right=pos_r)

        # Stream the overhead + wrist cameras to the headset so the operator can
        # see the scene and grippers. With the relay this is the subprocess's
        # WebRTC manager (out of process); otherwise fall back to the in-process
        # relay, which reuses the frames the robot's cameras already decode.
        if use_relay:
            teleop.set_video_manager(relay)
        else:
            _register_camera_video(robot, teleop)
    except BaseException:
        if relay is not None:
            relay.shutdown()
        raise

    # Background /proc sampler: shows whether the system saturates when engaged
    # and which process/thread is the aggressor. Labels the known subprocesses
    # (mp spawn children all report comm=python, so pid mapping is what makes the
    # IK solver and video relay legible in the output).
    diag_labels: dict[int, str] = {os.getpid(): "main"}
    ik_proc = getattr(teleop, "_ik_process", None)
    if ik_proc is not None and getattr(ik_proc, "pid", None):
        diag_labels[ik_proc.pid] = "ik"
    if relay is not None and getattr(relay, "_proc", None) is not None:
        relay_pid = getattr(relay._proc, "pid", None)
        if relay_pid:
            diag_labels[relay_pid] = "relay"
    diag = SystemDiag(diag_labels, _logger)
    diag.start()

    # Keep the relay's raw dataset branch closed until an episode records: the
    # raw VIC convert + shared-memory copy for every camera is the bulk of the
    # relay's CPU (~2 cores), and nothing reads raw frames during the pre-record
    # teleop phase. Closing it there makes that phase as light as `axol teleop`.
    if relay is not None:
        relay.set_raw_enabled(False)

    teleop_action_proc, robot_action_proc, robot_obs_proc = make_default_processors()

    episodes_recorded = 0
    episode_idx = dataset.num_episodes
    teleop_interval = 1.0 / teleop_hz
    publisher = _SnapshotPublisher()
    capture: _CaptureThread | None = None

    # Rolling control-loop rate readout, mirroring `axol teleop`: the loop rate
    # is measured here; vr/ik come from the teleop's ~2s windows. Logged once a
    # second at INFO so collection-time perf can be compared against teleop.
    loop_times: list[float] = []
    last_rate_log = time.perf_counter()
    # Also break the per-step body into its sections so we can see where the
    # time goes (joint read, action read, robot send).
    time_sections = True
    # `proc` isolates the LeRobot action processors (which native teleop does
    # not run) from `send`, so `send` is now the pure CAN motion_control
    # round-trip — directly comparable to teleop's `send`.
    sect = {"obs": 0.0, "act": 0.0, "proc": 0.0, "send": 0.0}

    # Set when the on-loop coroutines must unwind (Ctrl+C): the hot loop now
    # runs on the robot's event loop, so a KeyboardInterrupt on the main thread
    # can't break it directly — it has to exit via this flag before teardown.
    loop_stop = threading.Event()

    def _stopped() -> bool:
        return (stop_event is not None and stop_event.is_set()) or loop_stop.is_set()

    # Worst single-iteration stall and scheduler slip within each window. `gap`
    # is the longest time between consecutive loop iterations (a starved control
    # thread shows up as gaps >> the 1/teleop_hz period); `slip` is how late the
    # loop woke past its absolute deadline. Both isolate "the thread lost the
    # CPU" from "the CAN call itself was slow" — jerk tracks the former.
    max_gap = {"v": 0.0}
    max_slip = {"v": 0.0}
    prev_t0 = {"v": 0.0}

    def _maybe_log_rate(t0: float) -> None:
        nonlocal last_rate_log, sect
        loop_times.append(t0)
        if prev_t0["v"]:
            gap = t0 - prev_t0["v"]
            if gap > max_gap["v"]:
                max_gap["v"] = gap
        prev_t0["v"] = t0
        if t0 - last_rate_log < 1.0 or len(loop_times) <= 1:
            return
        span = loop_times[-1] - loop_times[0]
        n = len(loop_times)
        loop_hz = (n - 1) / span if span > 0 else 0.0
        _logger.info(
            "loop: %.1f Hz  vr: %.1f Hz  ik: %.1f Hz",
            loop_hz,
            teleop.vr_hz(),
            teleop.ik_hz(),
        )
        if time_sections:
            _logger.debug(
                "loop sections (mean ms): obs=%.2f act=%.2f proc=%.2f send=%.2f  "
                "maxgap=%.1fms maxslip=%.1fms",
                1e3 * sect["obs"] / n,
                1e3 * sect["act"] / n,
                1e3 * sect["proc"] / n,
                1e3 * sect["send"] / n,
                1e3 * max_gap["v"],
                1e3 * max_slip["v"],
            )
            sect = {"obs": 0.0, "act": 0.0, "proc": 0.0, "send": 0.0}
        loop_times.clear()
        max_gap["v"] = 0.0
        max_slip["v"] = 0.0
        last_rate_log = t0

    # The hot control loop runs *on the robot's event loop* (see
    # AxolRobot.event_loop) so motion_control is awaited inline — cooperatively
    # interleaved with CAN telemetry on one thread, exactly like `axol teleop`.
    # The main thread drives the episode lifecycle (dataset writes, rest-pose
    # moves) and blocks on each coroutine until the episode (or reset) finishes.
    async def _episode_loop() -> tuple[bool, bool]:
        nonlocal capture
        recording = False
        rerecord = False

        # Absolute-deadline pacing (mirrors `axol teleop`): late wakeups are
        # corrected on the next cycle instead of stretching the command interval.
        # Regular command timing matters because motion_control derives its
        # velocity feedforward by differentiating commanded positions, so a
        # jittery interval shows up as jerk.
        deadline = time.perf_counter()
        while not _stopped():
            deadline += teleop_interval
            t0 = time.perf_counter()
            _maybe_log_rate(t0)

            # Camera reads happen on the capture thread; the control loop only
            # ever touches joint state.
            joint_obs = robot.get_joint_observation()
            t_obs = time.perf_counter()
            teleop.send_feedback(joint_obs)
            act = teleop.get_action()
            t_act = time.perf_counter()
            act_processed = teleop_action_proc((act, joint_obs))
            robot_act = robot_action_proc((act_processed, joint_obs))
            t_proc = time.perf_counter()
            await robot.send_action_async(robot_act)
            t_send = time.perf_counter()
            sect["obs"] += t_obs - t0
            sect["act"] += t_act - t_obs
            sect["proc"] += t_proc - t_act
            sect["send"] += t_send - t_proc

            publisher.publish(joint_obs, act_processed, t0)

            events = teleop.get_teleop_events()

            if events.get("start_recording") and not recording:
                recording = True
                # Open the relay's raw branch so the capture thread has frames.
                if relay is not None:
                    relay.set_raw_enabled(True)
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

            await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
            slip = time.perf_counter() - deadline
            if slip > max_slip["v"]:
                max_slip["v"] = slip

        return recording, rerecord

    async def _reset_loop() -> None:
        reset_deadline = time.perf_counter() + 30.0
        deadline = time.perf_counter()
        while teleop.is_resetting and time.perf_counter() < reset_deadline:
            if _stopped():
                break
            deadline += teleop_interval
            joint_obs = robot.get_joint_observation()
            act = teleop.get_action()
            await robot.send_action_async(robot_action_proc((act, joint_obs)))
            await asyncio.sleep(max(0.0, deadline - time.perf_counter()))

    def _run_on_robot_loop(coro: Any) -> Any:
        """Run ``coro`` on the robot's event loop and block until it returns.

        On Ctrl+C, signal the coroutine to unwind and wait for it to finish so
        it stops commanding the robot before teardown, then re-raise.
        """
        fut = asyncio.run_coroutine_threadsafe(coro, robot.event_loop)
        try:
            return fut.result()
        except KeyboardInterrupt:
            loop_stop.set()
            try:
                fut.result(timeout=5.0)
            except BaseException:
                fut.cancel()
            raise

    try:
        while not _stopped():
            log_say(
                f"Episode {episode_idx + 1}: robot is at rest pose. Press record on the VR controller when ready."
            )
            dataset.clear_episode_buffer()

            recording, rerecord = _run_on_robot_loop(_episode_loop())

            if capture is not None:
                capture.stop_event.set()
                capture.join()
                capture = None
            # Recording done — close the raw branch so the rest-pose/reset and
            # next pre-record phase stay light.
            if relay is not None:
                relay.set_raw_enabled(False)

            if _stopped():
                break

            log_say("Returning to rest pose.")
            teleop.request_reset()
            _run_on_robot_loop(_reset_loop())
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

        diag.stop()

        robot.disconnect()
        teleop.disconnect()
        if relay is not None:
            relay.shutdown()

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
