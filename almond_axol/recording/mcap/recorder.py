"""The MCAP recorder: a subprocess that writes one ``.mcap`` file per episode.

Collect data with **Recording format** set to ``mcap`` uses
:class:`McapRecorderProcess` in place of the LeRobot
:class:`~almond_axol.recording.record_proc.DatasetRecorderProcess`. It has the
same episode methods (``start_episode``, ``finish_episode``, ``save_episode``,
``cancel_episode``, ``close`` ...), so the control loop does not change.

Data flow:

- The video relay encodes every recording camera to all-intra H.265 on the
  GPU and ships each access unit, with its exposure timestamp, over shared
  memory (:class:`~almond_axol.video.shm_frames.EncodedAuReader`). The
  recorder writes those bytes unchanged.
- The Rust core's 240 Hz joint feedback reaches the **joint ring** through
  :meth:`McapRecorderProcess.publish_joint_state`.
- The 120 Hz control loop publishes commands, headset input and Jelly
  readings to the **tick ring** through :meth:`McapRecorderProcess.publish`.

Both rings and the camera exposures are on ``perf_counter`` (the system's
monotonic clock), so every message in a file shares one timeline. The capture
thread drains all three in order while an episode records; see
:class:`~almond_axol.recording.mcap.episode.EpisodeRecording` for the topics.

Episodes are ``<root>/episode_NNNNNN.mcap``. Each is written as
``.mcap.partial`` and renamed only when saved, so a crash or a discarded take
never leaves a file that looks complete.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import os
import re
import shutil
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from ...camera_calibration.file import DEFAULT_PATH as DEFAULT_CALIBRATION_PATH
from ...robot import joint_zero
from ...video.shm_frames import SnapshotReader, SnapshotWriter
from .episode import FORMAT_VERSION, EpisodeRecording, iso_utc
from .rings import (
    JOINT_RING_CAPACITY,
    TICK_RING_CAPACITY,
    joint_ring_keys,
    tick_ring_keys,
)
from .imu import ImuReceiver
from .mounts import EpisodeFrames, episode_frames
from .scene import CameraSource, Scene
from .topics import format_topic_names, frame_names_problem

_logger = logging.getLogger(__name__)

_READY_TIMEOUT_S = 60.0
_CMD_TIMEOUT_S = 10.0
_SAVE_TIMEOUT_S = 60.0
# The capture thread's idle wait between polls of the rings (s).
_POLL_S = 0.010
# Refuse to start an episode with less free disk than this (bytes). Four
# 1920x1200 cameras at 60 fps record about 20 MB/s.
_MIN_FREE_BYTES = 2 * 1024**3
_EPISODE_RE = re.compile(r"^episode_(\d+)\.mcap$")


def episode_indices(root: Path) -> list[int]:
    """Indices of the saved episodes in ``root``."""
    try:
        names = os.listdir(root)
    except OSError:
        return []
    out = []
    for name in names:
        match = _EPISODE_RE.match(name)
        if match:
            out.append(int(match.group(1)))
    return sorted(out)


def episode_path(root: Path, index: int) -> Path:
    return Path(root) / f"episode_{index:06d}.mcap"


def compute_serial() -> str:
    """The onboard computer's serial number (a Jetson's), or ``""``."""
    for path in (
        "/proc/device-tree/serial-number",
        "/sys/firmware/devicetree/base/serial-number",
    ):
        try:
            return Path(path).read_bytes().strip(b"\0\n ").decode(errors="replace")
        except OSError:
            continue
    return ""


def clock_offset_ns() -> int:
    """``unix_ns - perf_counter_ns`` from the tightest of a few paired reads."""
    best: tuple[int, int] | None = None
    for _ in range(5):
        before = time.perf_counter_ns()
        wall = time.time_ns()
        after = time.perf_counter_ns()
        width = after - before
        offset = wall - (before + after) // 2
        if best is None or width < best[0]:
            best = (width, offset)
    assert best is not None
    return best[1]


# ---------------------------------------------------------------------------
# Child side
# ---------------------------------------------------------------------------


class McapSession:
    """The recorder process's state: readers, scene, and the current episode."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        cameras: dict[str, Any],
        scene: Scene,
        tick_reader: Any,
        joint_reader: Any,
    ) -> None:
        self.config = config
        self.root = Path(config["dataset_root"])
        self.readers = cameras
        self.scene = scene
        self.tick_reader = tick_reader
        self.joint_reader = joint_reader
        # Jelly's per-wheel spin signs (almond_axol.robot.jelly.WHEEL_SIGNS),
        # passed in so this process need not import the robot package.
        self.wheel_signs: dict[int, float] = dict(config.get("wheel_signs") or {})
        # Renamed topics (default name -> customer's name), see .topics.
        self.topic_names: dict[str, str] = dict(config.get("topic_names") or {})
        # Renamed frames (default name -> customer's name), see .topics.
        self.frame_names: dict[str, str] = dict(config.get("frame_names") or {})
        problem = frame_names_problem(self.frame_names, scene.frame_ids())
        if problem is not None:
            raise ValueError(problem)
        # One receiver per camera IMU: {topic: (receiver, left optical frame)}.
        self.imus: dict[str, tuple[Any, str]] = {}
        # The camera calibration file (see almond_axol.camera_calibration),
        # read again at the start of every episode.
        self.calibration_path = Path(
            config.get("camera_calibration_path") or DEFAULT_CALIBRATION_PATH
        ).expanduser()
        self.frames: EpisodeFrames | None = None
        # The joint zeroing record (almond_axol.robot.joint_zero), read again
        # at the start of every episode. ``None`` is ~/.almond/joint_zero.json.
        zero_path = config.get("joint_zero_path")
        self.joint_zero_path = Path(zero_path).expanduser() if zero_path else None
        self.zero: joint_zero.ZeroState | None = None
        self._zero_note: str | None = None
        # The frame each IMU topic is written in this episode.
        self.imu_frames: dict[str, str] = {}
        self._calibration_note: str | None = None
        self._imu_pose_warned: set[str] = set()
        self.episode: EpisodeRecording | None = None
        self.episode_index = 0
        self.episode_uuid = ""
        self.compute_serial = compute_serial()
        self.task = ""
        self.capture_error: str | None = None
        self.on_error: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._pauses: list[tuple[float, float | None]] = []
        self._tick_after = 0
        self._joint_after = 0
        self._trim_after: float | None = None
        self._lost_exposures: dict[str, int] = {}

    # -- construction ----------------------------------------------------

    @classmethod
    def open(cls, config: dict[str, Any]) -> McapSession:
        from ...video.shm_frames import EncodedAuReader

        raw_meta: dict[str, dict[str, Any]] = config["raw_meta"]
        sources = {
            src: CameraSource.from_meta(src, meta) for src, meta in raw_meta.items()
        }
        readers: dict[str, Any] = {}
        tick_reader = joint_reader = None
        try:
            for src, meta in raw_meta.items():
                if not str(meta.get("transport", "")).startswith("gstshm-h26"):
                    raise RuntimeError(
                        f"camera {src} is not encoded in the relay "
                        f"(transport {meta.get('transport')!r})"
                    )
                reader = EncodedAuReader(
                    meta["socket_path"],
                    meta["width"],
                    meta["height"],
                    meta["fps"],
                    name=src,
                    pts_perf_offset_s=meta["pts_perf_offset_s"],
                    capture_fps=meta.get("capture_fps", meta["fps"]),
                    codec=str(meta.get("codec", "h264")),
                )
                readers[src] = reader
                reader.connect()
            tick_reader = SnapshotReader(config["tick_ring"], tick_ring_keys(), [])
            joint_reader = SnapshotReader(config["joint_ring"], joint_ring_keys(), [])
            scene = Scene.build(
                Path(config["urdf_path"]),
                has_gripper=bool(config.get("has_gripper", True)),
                jelly_wheels=bool(config.get("jelly_wheels", False)),
                jelly_lift=bool(config.get("jelly_lift", False)),
                cameras=sources,
                calibration_dir=Path(config["calibration_dir"]),
                gripper=config.get("gripper"),
                software_version=str(config.get("software_version") or ""),
            )
        except BaseException:
            for reader in readers.values():
                with contextlib.suppress(Exception):
                    reader.disconnect()
            for ring in (tick_reader, joint_reader):
                if ring is not None:
                    with contextlib.suppress(Exception):
                        ring.close()
            raise
        session = cls(
            config,
            cameras=readers,
            scene=scene,
            tick_reader=tick_reader,
            joint_reader=joint_reader,
        )
        for topic, (path, frame_id) in scene.imus().items():
            try:
                session.imus[topic] = (ImuReceiver(path), frame_id)
            except OSError as exc:
                _logger.warning("can't record %s: %s", topic, exc)
        return session

    @property
    def episode_count(self) -> int:
        return len(episode_indices(self.root))

    # -- episode lifecycle -----------------------------------------------

    def start(self, task: str) -> None:
        if self.episode is not None:
            raise RuntimeError("an episode is already open")
        self.root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.root).free
        if free < _MIN_FREE_BYTES:
            raise RuntimeError(
                f"only {free / 1024**3:.1f} GB free in {self.root}. Free up disk "
                "space, then start recording again."
            )
        # Drop the transport tail of the previous take while the valves are
        # shut; the parent opens them once this returns.
        for reader in self.readers.values():
            reader.begin_flush()
        for reader in self.readers.values():
            reader.finish_flush()
            reset = getattr(reader, "reset_lost_exposures", None)
            if callable(reset):
                reset()
        indices = episode_indices(self.root)
        self.episode_index = indices[-1] + 1 if indices else 0
        self.episode_uuid = str(uuid.uuid4())
        self.task = task
        start_perf = time.perf_counter()
        self.capture_error = None
        self._trim_after = None
        self._pauses = []
        self._paused.clear()
        self._tick_after = self.tick_reader.published()
        self._joint_after = self.joint_reader.published()
        for receiver, _frame_id in self.imus.values():
            receiver.drain()  # samples from between takes
        self.frames = self._episode_frames()
        self.episode = EpisodeRecording(
            episode_path(self.root, self.episode_index),
            scene=self.scene,
            start_perf=start_perf,
            clock_offset_ns=clock_offset_ns(),
            wheel_signs={int(k): float(v) for k, v in self.wheel_signs.items()},
            topic_names=self.topic_names,
            frames=self.frames,
            frame_names=self.frame_names,
        )
        self.zero = self._joint_zero()
        if self.zero.raw:
            # The record itself, so the zeroing history can be checked later.
            self.episode.file.attachment(
                joint_zero.FILE_NAME,
                "application/json",
                self.zero.raw,
                self.episode.stamp(start_perf),
            )
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name="mcap-capture", daemon=True
        )
        self._thread.start()

    def _episode_frames(self) -> EpisodeFrames:
        """This episode's static frames, from the calibration file if usable."""
        imu_poses: dict[str, np.ndarray] = {}
        self.imu_frames = {}
        for topic, (receiver, optical) in self.imus.items():
            frame = self.scene.imu_frame_for(topic)
            pose = getattr(receiver, "pose", None)
            if frame is not None and pose is not None:
                imu_poses[frame] = pose
                self.imu_frames[topic] = frame
            else:
                self.imu_frames[topic] = optical
                if (
                    getattr(receiver, "received", 0)
                    and topic not in self._imu_pose_warned
                ):
                    self._imu_pose_warned.add(topic)
                    _logger.warning(
                        "%s is recorded in %s: the ZED camera plugins don't send "
                        "the IMU's position. Run `axol gst.build-zed` to update them.",
                        topic,
                        optical,
                    )
        frames = episode_frames(
            self.scene,
            self.calibration_path,
            self.config.get("robot_serial") or None,
            imu_poses,
            frame_names=self.frame_names,
        )
        note = frames.problem or None
        if note != self._calibration_note:
            self._calibration_note = note
            if note:
                _logger.warning(
                    "Camera calibration not used: %s. Episodes record the camera "
                    "positions from CAD. Run `axol calibration.cameras` to "
                    "calibrate.",
                    note,
                )
            elif frames.calibration is not None:
                _logger.info(
                    "Camera calibration %s (%s): cameras %s.",
                    frames.calibration.calibration_id,
                    frames.calibration.calibrated_at or "undated",
                    frames.status,
                )
        return frames

    def _calibration_zero(self) -> dict[str, str] | None:
        """The zeroing each arm was on when the cameras were calibrated:
        ``None`` without a camera calibration in use, ``{}`` when the
        calibration didn't record it."""
        calibration = self.frames.calibration if self.frames is not None else None
        if calibration is None:
            return None
        return dict(calibration.joint_zero)

    def _joint_zero(self) -> joint_zero.ZeroState:
        """The zeroing each arm joint is on, from the zeroing record."""
        zero = joint_zero.state(
            self.config.get("robot_serial") or "", self.joint_zero_path
        )
        match = joint_zero.calibration_match(zero, self._calibration_zero())
        note = match if match.startswith("changed") else (zero.problem or None)
        if note != self._zero_note:
            self._zero_note = note
            if match.startswith("changed"):
                _logger.warning(
                    "Joint zero %s since the camera calibration. The wrist camera "
                    "positions may be off. Run `axol calibration.cameras` again.",
                    match.removeprefix("changed: "),
                )
            elif zero.problem:
                _logger.warning(
                    "Joint zero not recorded: %s. Episodes can't name the zeroing "
                    "the arms are on until you zero them with `axol "
                    "motor.set-zero-pos --guided`.",
                    zero.problem,
                )
        return zero

    def _fail(self, message: str) -> None:
        if self.capture_error is None:
            self.capture_error = message
            _logger.error("MCAP capture failed: %s", message)
            if self.on_error is not None:
                with contextlib.suppress(Exception):
                    self.on_error(message)

    def _pump(self) -> bool:
        """Move everything that has arrived into the episode. True if any did."""
        episode = self.episode
        assert episode is not None
        moved = False
        paused = self._paused.is_set()
        for source, reader in self.readers.items():
            for au, exposure, _recv in reader.drain_aus():
                moved = True
                if not paused:
                    episode.add_video(source, au, exposure)
        rows, self._joint_after, lost = self.joint_reader.read_rows_since(
            self._joint_after
        )
        episode.joint_lost += lost
        if len(rows):
            moved = True
            if not paused:
                episode.add_joint_rows(rows)
        rows, self._tick_after, lost = self.tick_reader.read_rows_since(
            self._tick_after
        )
        episode.tick_lost += lost
        if len(rows):
            moved = True
            if not paused:
                episode.add_tick_rows(rows)
        for topic, (receiver, optical) in self.imus.items():
            samples = receiver.drain()
            if samples:
                moved = True
                if not paused:
                    episode.add_imu(topic, self.imu_frames.get(topic, optical), samples)
        return moved

    def _capture_loop(self) -> None:
        try:
            while not self._stop.is_set():
                if not self._pump():
                    self._stop.wait(_POLL_S)
            # One last pass for whatever arrived before the stop.
            self._pump()
        except Exception as exc:  # noqa: BLE001 - reported to the parent
            self._fail(f"{type(exc).__name__}: {exc}")

    def stop_capture(self, trim_after: float | None = None) -> tuple[int, str | None]:
        """Stop writing samples; returns ``(frames, capture_error)``."""
        thread = self._thread
        if thread is not None:
            self._stop.set()
            thread.join(timeout=_CMD_TIMEOUT_S)
            if thread.is_alive():
                raise RuntimeError("MCAP capture thread did not stop")
            self._thread = None
            for source, reader in self.readers.items():
                self._lost_exposures[source] = int(getattr(reader, "lost_exposures", 0))
        if self.episode is None:
            return 0, self.capture_error
        if self.episode.end_perf is None:
            self.episode.end_perf = time.perf_counter()
            if self._pauses and self._pauses[-1][1] is None:
                self._pauses[-1] = (self._pauses[-1][0], self.episode.end_perf)
        if trim_after is not None:
            self._trim_after = float(trim_after)
        return self.episode.frame_count(), self.capture_error

    def pause(self) -> int:
        if not self._paused.is_set():
            self._paused.set()
            self._pauses.append((time.perf_counter(), None))
        return self.frame_count()

    def resume(self) -> int:
        if self._paused.is_set():
            self._paused.clear()
            if self._pauses and self._pauses[-1][1] is None:
                self._pauses[-1] = (self._pauses[-1][0], time.perf_counter())
        return self.frame_count()

    def frame_count(self) -> int:
        return self.episode.frame_count() if self.episode is not None else 0

    def save(self, outcome: str | None, info: dict[str, Any]) -> int:
        """Finish the open episode's file; returns the saved episode count."""
        self.stop_capture()
        episode = self.episode
        if episode is None:
            raise RuntimeError("no episode to save")
        if self.capture_error is not None:
            self.cancel()
            raise RuntimeError(f"camera capture failed ({self.capture_error})")
        try:
            episode.flush()
            path = episode.finish(self._metadata(episode, outcome, info))
        except BaseException:
            episode.discard()
            self.episode = None
            raise
        self.episode = None
        _logger.info("Saved %s", path)
        return self.episode_count

    def cancel(self) -> None:
        with contextlib.suppress(RuntimeError):
            self.stop_capture()
        if self.episode is not None:
            self.episode.discard()
            self.episode = None

    def close(self) -> None:
        self.cancel()
        for reader in self.readers.values():
            with contextlib.suppress(Exception):
                reader.disconnect()
        for ring in (self.tick_reader, self.joint_reader):
            with contextlib.suppress(Exception):
                ring.close()
        for receiver, _frame_id in self.imus.values():
            with contextlib.suppress(Exception):
                receiver.close()

    # -- metadata ----------------------------------------------------------

    def _metadata(
        self, episode: EpisodeRecording, outcome: str | None, info: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        config = self.config
        scene = self.scene
        start_ns = episode.stamp(episode.start_perf)
        end_perf = (
            episode.end_perf if episode.end_perf is not None else time.perf_counter()
        )
        paused = sum(
            (end if end is not None else end_perf) - begin
            for begin, end in self._pauses
        )
        custom = {str(k): v for k, v in (config.get("episode_metadata") or {}).items()}
        episode_meta = {
            "index": self.episode_index,
            "uuid": self.episode_uuid,
            "task": self.task,
            "operator": config.get("operator") or "",
            "outcome": outcome or "success",
            "start_time": iso_utc(start_ns),
            "end_time": iso_utc(episode.stamp(end_perf)),
            "duration_s": f"{end_perf - episode.start_perf:.3f}",
            "paused_s": f"{paused:.3f}",
            "end_gesture_time": (
                iso_utc(episode.stamp(self._trim_after))
                if self._trim_after is not None
                else ""
            ),
            **info,
        }
        robot_meta = {
            "serial": config.get("robot_serial") or "",
            "model": config.get("robot_model") or "",
            "gripper": config.get("gripper") or "",
            "jelly": "yes" if (scene.jelly_wheels or scene.jelly_lift) else "no",
            "software_version": config.get("software_version") or "",
            "hostname": config.get("hostname") or socket.gethostname(),
            "compute_serial": self.compute_serial,
        }
        recording_meta: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "clock": "host monotonic clock (perf_counter), as Unix time",
            "clock_offset_ns": episode.file.clock_offset_ns,
            "video_codec": ",".join(sorted({c.codec for c in scene.cameras.values()})),
            "video_keyframes": "every frame",
            "images": "unrectified sensor images",
            "joint_rate_hz": 240,
            "command_rate_hz": config.get("teleop_hz", 120),
            "ee_frames": ",".join(
                episode.ee_frames[side] for side in sorted(episode.ee_frames)
            ),
            "joint_samples": episode.joint_samples,
            "joint_samples_lost": episode.joint_lost,
            "control_ticks": episode.tick_samples,
            "control_ticks_lost": episode.tick_lost,
            "commands_applied": episode.commands_applied,
            "commands_without_application_time": episode.commands_unapplied,
            "odometry": (
                "Jelly wheel angles, dead reckoned"
                if episode.odometry is not None
                else ""
            ),
            "odometry_rejected_samples": (
                episode.odometry.rejected if episode.odometry is not None else 0
            ),
            "assumed_joints": ",".join(scene.assumed_joints()),
            "topic_names": format_topic_names(self.topic_names),
            "frame_names": format_topic_names(self.frame_names),
            **episode.camera_metadata(),
        }
        for source, camera in scene.cameras.items():
            prefix = camera.topic.strip("/").replace("/", "_")
            recording_meta[f"{prefix}.lost_in_transport"] = self._lost_exposures.get(
                source, 0
            )
        duration = max(end_perf - episode.start_perf, 1e-9)
        for topic in self.imus:
            key = topic.strip("/").replace("/", "_")
            count = episode.imu_samples.get(topic, 0)
            recording_meta[f"{key}.samples"] = count
            recording_meta[f"{key}.rate_hz"] = f"{count / duration:.1f}"
            frame = self.imu_frames.get(topic, "")
            recording_meta[f"{key}.frame_id"] = episode.frame(frame) if frame else ""
            imu_camera = scene.imu_camera(topic)
            recording_meta[f"{key}.serial"] = (
                imu_camera.serial if imu_camera is not None else ""
            )
        missing_imu = sorted(
            {
                c.camera
                for c in scene.cameras.values()
                if c.imu_requested and not c.imu_socket
            }
        )
        if missing_imu:
            recording_meta["imu_missing"] = (
                f"{', '.join(missing_imu)}: the ZED camera plugins need updating "
                "(axol gst.build-zed)"
            )
        return {
            "episode": episode_meta,
            "robot": robot_meta,
            "recording": recording_meta,
            "calibration": calibration_metadata(scene, self.frames, self.frame_names),
            "joint_zero": joint_zero.episode_metadata(
                self.zero or self._joint_zero(), self._calibration_zero()
            ),
            **({"custom": custom} if custom else {}),
        }


def calibration_metadata(
    scene: Scene,
    frames: EpisodeFrames | None = None,
    frame_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Which calibration the episode used, and when it is from.

    Camera mounts come from the camera calibration file when it was used
    (``calibration_*``), else from the URDF (CAD); ``tf_sources`` names the
    source of every static transform. Intrinsics and the head stereo
    transform come from each camera's ZED factory calibration file; its date
    is when the ZED SDK saved that file on this robot.
    """
    import hashlib

    from .mounts import resolve

    if frames is None:
        frames = resolve(scene)
    files = {
        cam.calibration.serial: cam.calibration
        for cam in scene.cameras.values()
        if cam.calibration is not None
    }
    digest = hashlib.sha256(scene.urdf_sha256.encode())
    for serial in sorted(files):
        digest.update(files[serial].sha256.encode())
    newest = max((f.mtime for f in files.values()), default=None)
    used = frames.calibration
    sources = frames.sources()
    renamed = frame_names or {}
    out: dict[str, Any] = {
        "calibration_status": frames.status,
        "calibration_id": used.calibration_id if used is not None else "",
        "calibrated_at": used.calibrated_at if used is not None else "",
        "due_at": used.due_at if used is not None else "",
        "calibration_file": used.path.name if used is not None else "",
        "calibration_file_sha256": used.sha256 if used is not None else "",
        "calibration_problem": frames.problem,
        "calibration_skipped": "; ".join(frames.skipped),
        "tf_sources": ", ".join(
            f"{renamed.get(child, child)}={src}" for child, src in sources.items()
        ),
        "urdf_sha256": scene.urdf_sha256,
        "intrinsics": "ZED factory calibration",
        "intrinsics_id": digest.hexdigest()[:16],
        "intrinsics_date": (iso_utc(int(newest * 1e9)) if newest is not None else ""),
        "head_stereo": scene.stereo_source or "CAD baseline from the URDF",
    }
    for cam in scene.cameras.values():
        prefix = cam.topic.strip("/").replace("/", "_")
        out[f"{prefix}.extrinsics"] = sources.get(cam.frame_id, "")
        if cam.calibration is None:
            out[f"{prefix}.file"] = "missing"
            continue
        out[f"{prefix}.file"] = cam.calibration.path.name
        out[f"{prefix}.file_sha256"] = cam.calibration.sha256
        out[f"{prefix}.intrinsics"] = (
            cam.intrinsics.distortion_model if cam.intrinsics else "missing"
        )
    return out


def _mcap_recorder_main(conn: Any, error_conn: Any, config: dict[str, Any]) -> None:
    """Recorder subprocess entry point."""
    logging.basicConfig(level=config.get("log_level", "INFO"))
    from ...utils import affinity

    # Off the control loop's cores, like the LeRobot recorder.
    if affinity.pin_background_and_ik():
        affinity.pin_recorder()
    else:
        with contextlib.suppress(AttributeError, OSError):
            os.nice(5)

    try:
        session = McapSession.open(config)
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        _logger.exception("MCAP recorder failed to start")
        with contextlib.suppress(Exception):
            conn.send(("error", f"{type(exc).__name__}: {exc}"))
        return

    def report(message: str) -> None:
        with contextlib.suppress(OSError, EOFError, ValueError):
            error_conn.send(message)

    session.on_error = report
    conn.send(("ready", session.episode_count))
    try:
        while True:
            try:
                msg = conn.recv()
            except (EOFError, KeyboardInterrupt):
                break
            if msg is None or msg[0] == "shutdown":
                break
            kind = msg[0]
            try:
                if kind == "start_episode":
                    session.start(msg[1])
                    conn.send(("started",))
                elif kind == "finish_episode":
                    frames, error = session.stop_capture(
                        msg[1] if len(msg) > 1 else None
                    )
                    conn.send(("finished", frames, error))
                elif kind == "pause_episode":
                    conn.send(("paused", session.pause()))
                elif kind == "resume_episode":
                    conn.send(("resumed", session.resume()))
                elif kind == "frame_count":
                    conn.send(("frame_count", session.frame_count()))
                elif kind == "save_episode":
                    count = session.save(msg[1], msg[2] if len(msg) > 2 else {})
                    conn.send(("saved", count))
                elif kind == "cancel_episode":
                    session.cancel()
                    conn.send(("cancelled",))
                else:
                    conn.send(("error", f"unknown command {kind!r}"))
            except Exception as exc:  # noqa: BLE001 - per-command failure
                _logger.exception("MCAP recorder command %s failed", kind)
                conn.send(("error", f"{exc}"))
    finally:
        session.close()
        with contextlib.suppress(Exception):
            error_conn.close()


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------


class McapRecorderProcess:
    """Control-process handle for the MCAP recorder subprocess.

    ``raw_meta`` is the video relay's per-source description (every source
    must use an encoded transport). ``config`` keys: ``dataset_root``,
    ``urdf_path``, ``calibration_dir``, ``has_gripper``, ``jelly_wheels``,
    ``jelly_lift``, ``wheel_signs``, ``topic_names``, ``operator``,
    ``robot_serial``, ``robot_model``, ``gripper``, ``software_version``,
    ``teleop_hz`` and ``log_level``.
    """

    def __init__(
        self, *, raw_meta: dict[str, dict[str, Any]], config: dict[str, Any]
    ) -> None:
        self._tick_ring = SnapshotWriter(
            tick_ring_keys(), [], capacity=TICK_RING_CAPACITY
        )
        self._joint_ring = SnapshotWriter(
            joint_ring_keys(), [], capacity=JOINT_RING_CAPACITY
        )
        self._joint_values = np.zeros(len(joint_ring_keys()), dtype=np.float64)
        self._lock = threading.Lock()
        self._capture_error: str | None = None
        self._outcome: str | None = None
        self._closed = False
        self._joint_samples = 0
        ctx = multiprocessing.get_context("spawn")
        conn, child_conn = ctx.Pipe()
        error_conn, child_error_conn = ctx.Pipe(duplex=False)
        full_config = {
            **config,
            "raw_meta": raw_meta,
            "tick_ring": self._tick_ring.name,
            "joint_ring": self._joint_ring.name,
        }
        proc = ctx.Process(
            target=_mcap_recorder_main,
            args=(child_conn, child_error_conn, full_config),
            daemon=True,
            name="mcap-recorder",
        )
        try:
            proc.start()
            child_conn.close()
            child_error_conn.close()
            if not conn.poll(_READY_TIMEOUT_S):
                raise RuntimeError("the MCAP recorder did not start in time")
            try:
                msg = conn.recv()
            except (EOFError, OSError) as exc:
                proc.join(timeout=2.0)
                raise RuntimeError(
                    "the MCAP recorder exited during startup "
                    f"(exit code {proc.exitcode})"
                ) from exc
            if not isinstance(msg, tuple) or msg[0] != "ready":
                detail = msg[1] if isinstance(msg, tuple) and len(msg) > 1 else msg
                raise RuntimeError(f"the MCAP recorder failed to start: {detail}")
        except BaseException:
            from ..record_proc import _shutdown_process

            _shutdown_process(proc, graceful_timeout=2.0)
            for closer in (conn.close, error_conn.close, self._close_rings):
                with contextlib.suppress(Exception):
                    closer()
            raise
        self._proc = proc
        self._conn = conn
        self._error_conn = error_conn
        self._episode_count = int(msg[1])

    def _close_rings(self) -> None:
        self._tick_ring.close()
        self._joint_ring.close()

    @property
    def pid(self) -> int | None:
        return self._proc.pid

    # -- hot path --------------------------------------------------------

    def publish(
        self, joint_obs: dict, action: dict, ts: float, intervention: bool = False
    ) -> None:
        """One control tick: a :func:`~.rings.pack_tick` row (``action`` unused)."""
        if not self._closed:
            self._tick_ring.write(joint_obs, {}, ts)

    def publish_joint_state(
        self,
        perf_ts: float,
        positions: np.ndarray,
        velocities: np.ndarray,
        torques: np.ndarray,
        applied: tuple[float, float, float, float] | None = None,
    ) -> None:
        """One 240 Hz feedback sample (a Rust core state listener)."""
        if self._closed:
            return
        values = self._joint_values
        n = len(positions)
        values[:n] = positions
        values[n : 2 * n] = velocities
        values[2 * n : 3 * n] = torques
        values[3 * n : 3 * n + 4] = applied if applied is not None else np.nan
        self._joint_ring.write_values(perf_ts, values)
        self._joint_samples += 1

    @property
    def joint_samples(self) -> int:
        """How many 240 Hz samples have arrived (a liveness check)."""
        return self._joint_samples

    # -- commands --------------------------------------------------------

    def _command(
        self, msg: tuple, expect: str, timeout: float = _CMD_TIMEOUT_S
    ) -> tuple:
        with self._lock:
            self._conn.send(msg)
            if not self._conn.poll(timeout):
                raise RuntimeError(f"the MCAP recorder did not answer {msg[0]} in time")
            reply = self._conn.recv()
        if not isinstance(reply, tuple) or not reply:
            raise RuntimeError(f"the MCAP recorder sent {reply!r}")
        if reply[0] != expect:
            detail = reply[1] if len(reply) > 1 else reply[0]
            raise RuntimeError(str(detail))
        return reply

    def episode_count(self) -> int:
        return self._episode_count

    def start_episode(self, task: str) -> None:
        with contextlib.suppress(EOFError, OSError, ValueError):
            while self._error_conn.poll():
                self._error_conn.recv()
        self._capture_error = None
        self._outcome = None
        self._command(("start_episode", task), "started")

    def finish_episode(self, trim_after: float | None = None) -> int:
        from ..record_proc import RecorderCaptureError

        reply = self._command(("finish_episode", trim_after), "finished")
        if reply[2] is not None:
            self._capture_error = str(reply[2])
            raise RecorderCaptureError(f"recorder capture failed: {reply[2]}")
        return int(reply[1])

    def stop_capture(self, trim_after: float | None = None) -> tuple[int, str | None]:
        from ..record_proc import RecorderCaptureError

        try:
            return self.finish_episode(trim_after), None
        except RecorderCaptureError as exc:
            return 0, str(exc)

    def poll_capture_error(self) -> str | None:
        with contextlib.suppress(EOFError, OSError, ValueError):
            while self._error_conn.poll():
                self._capture_error = str(self._error_conn.recv())
        return self._capture_error

    def pause_episode(self) -> int:
        return int(self._command(("pause_episode",), "paused")[1])

    def resume_episode(self) -> int:
        return int(self._command(("resume_episode",), "resumed")[1])

    def frame_count(self) -> int:
        return int(self._command(("frame_count",), "frame_count")[1])

    def set_outcome(self, outcome: str) -> None:
        """How the take ended (``success`` or ``failure``), for its metadata."""
        self._outcome = outcome

    def save_episode(self, info: dict[str, Any] | None = None) -> None:
        reply = self._command(
            ("save_episode", self._outcome, dict(info or {})), "saved", _SAVE_TIMEOUT_S
        )
        self._episode_count = int(reply[1])

    def cancel_episode(self) -> None:
        self._command(("cancel_episode",), "cancelled", _SAVE_TIMEOUT_S)

    def close(self) -> None:
        if self._closed:
            return
        from ..record_proc import _shutdown_process

        with contextlib.suppress(OSError, ValueError), self._lock:
            self._conn.send(("shutdown",))
        alive, forced, failures = _shutdown_process(
            self._proc, graceful_timeout=_SAVE_TIMEOUT_S
        )
        self._closed = True
        for closer in (self._conn.close, self._error_conn.close, self._close_rings):
            with contextlib.suppress(Exception):
                closer()
        if alive:
            raise RuntimeError("the MCAP recorder did not shut down")
        if forced:
            _logger.warning("the MCAP recorder was stopped forcibly")
        for label, error in failures:
            _logger.warning("MCAP recorder shutdown %s: %s", label, error)
