"""Dataset recording, off the control loop.

``collect-data``'s control loop runs at 120 Hz on the robot's event loop. Writing
the LeRobot dataset — capturing camera frames, assembling rows, ``add_frame``,
NVENC encoding, ``save_episode`` — is heavy per-frame Python work. Running it as
threads *inside* the control process makes it share one GIL with the control
loop, so even with spare CPU cores the loop stutters during recording (the
remaining jitter after the SVGA downscale + stats-off-hot-path fixes).

This module moves all of that into a dedicated **recorder subprocess**
(:class:`DatasetRecorderProcess`) so the control process only writes a tiny
joint/action snapshot per tick (via
:class:`~almond_axol.vr.shm_frames.SnapshotWriter`) and sends episode-lifecycle
commands. The recorder pulls each camera's raw frames from the relay — via a gst
``shmsrc`` consumer (the relay's ``shmsink`` exports frames in C, so the relay
does no Python per frame and its WebRTC send keeps its GIL), or, when gst's shm
plugin is absent, via a :class:`RawFrameReader` over a shared-memory block the
relay's Python pull loop fills. Either way the recorder owns the
``LeRobotDataset`` end to end.

When the video relay is unavailable (no gst stack — a degraded, non-Jetson path),
:class:`InProcessRecorder` keeps the old behavior: dataset + capture thread in
the control process. Both expose the same interface, so the control loop is
single-path; only the construction differs.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import multiprocessing.connection
import os
import platform
import shutil
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

_logger = logging.getLogger("almond_axol.cli.collect_data")

# Per-stream encoder thread count (three cameras each spawn their own encoder
# thread); a small inner libx264 pool leaves cores for the control loop.
_ENCODER_THREADS = 2
# Keyframe interval for the software encoder. LeRobot hardcodes GOP 2 (a keyframe
# almost every frame), which tanks Tegra encode throughput; 30 (0.5 s at 60 fps)
# is plenty fine-grained for timestamp-tolerant dataset decode.
_ENCODER_GOP = 30

# How long the recorder subprocess may take to open cameras' shm + the dataset.
_READY_TIMEOUT_S = 60.0
# How long a save_episode (encoder flush + parquet write + post-episode stats)
# may take.
_SAVE_TIMEOUT_S = 180.0


# ---------------------------------------------------------------------------
# Encoder selection (applied in whichever process owns the dataset)
# ---------------------------------------------------------------------------


def default_vcodec() -> str:
    """Pick a video codec that can actually open on this machine.

    LeRobot's "auto" prefers ``h264_nvenc``, but on Jetson/Tegra (aarch64) there
    is no desktop ``libnvidia-encode`` to back it, so it fails to open mid
    episode. Default to CPU "h264" on aarch64 and let "auto" pick the HW encoder
    elsewhere.
    """
    return "h264" if platform.machine() == "aarch64" else "auto"


def _tune_software_encoder() -> None:
    """Patch LeRobot's libx264 options: ``preset=veryfast g=30`` (idempotent)."""
    import lerobot.datasets.video_utils as _vu

    if getattr(_vu, "_axol_encoder_tuned", False):
        return
    _orig_get_codec_options = _vu._get_codec_options

    def _tuned(
        vcodec: str, g: int | None = 2, crf: int | None = 30, preset=None
    ) -> dict:
        options = _orig_get_codec_options(vcodec, g=g, crf=crf, preset=preset)
        if vcodec in ("h264", "hevc"):
            options["preset"] = "veryfast"
            options["g"] = str(_ENCODER_GOP)
        return options

    _vu._get_codec_options = _tuned
    _vu._axol_encoder_tuned = True
    _logger.info(
        "tuned software video encoder: preset=veryfast g=%d threads=%d",
        _ENCODER_GOP,
        _ENCODER_THREADS,
    )


def install_dataset_encoder() -> bool:
    """Prefer the Jetson NVENC encoder for dataset video; else tune libx264.

    Module-level monkeypatch of ``LeRobotDataset._build_streaming_encoder`` — must
    be applied in whatever process creates the dataset (the recorder subprocess,
    or the control process for the in-process fallback). Returns True when NVENC
    is in use.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from ..lerobot.nvenc_encoder import (
        NvencStreamingEncoder,
        hw_dataset_encoder_available,
    )

    if getattr(LeRobotDataset, "_axol_nvenc_installed", False):
        return True

    if not hw_dataset_encoder_available():
        _tune_software_encoder()
        return False

    def _build_nvenc(fps, vcodec, encoder_queue_maxsize, encoder_threads):
        return NvencStreamingEncoder(fps=fps, queue_maxsize=encoder_queue_maxsize)

    LeRobotDataset._build_streaming_encoder = staticmethod(_build_nvenc)
    LeRobotDataset._axol_nvenc_installed = True
    _logger.info("using Jetson NVENC hardware video encoder for dataset recording")
    return True


# ---------------------------------------------------------------------------
# Joint/action snapshot (in-process publisher, mirrors the cross-process one)
# ---------------------------------------------------------------------------


class _SnapshotPublisher:
    """Single-slot in-process publisher (the no-relay fallback's snapshot sink).

    The control loop calls :meth:`write` every tick; the capture thread reads the
    latest via :meth:`read_latest`. Returns ``None`` before the first write. The
    method names mirror :class:`~almond_axol.vr.shm_frames.SnapshotReader` so the
    capture loop is identical in both paths.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: tuple[dict, dict, float] | None = None

    def write(self, joint_obs: dict, action: dict, ts: float) -> None:
        with self._lock:
            self._latest = (joint_obs, action, ts)

    def read_latest(self) -> tuple[dict, dict, float] | None:
        with self._lock:
            return self._latest


# ---------------------------------------------------------------------------
# Capture loop (shared by both recorders, runs on its own thread)
# ---------------------------------------------------------------------------


def run_capture_loop(
    *,
    cameras: dict[str, Any],
    read_snapshot: Callable[[], tuple[dict, dict, float] | None],
    dataset: "LeRobotDataset",
    robot_obs_proc: Callable[[Any], Any],
    fps: int,
    task: str,
    rerun_ip: str | None,
    stop_event: threading.Event,
    on_error: Callable[[str], None] | None = None,
) -> None:
    """Capture dataset rows at ``fps`` Hz until ``stop_event`` is set.

    Each tick sleeps until ``T_n = recording_start + n/fps``, waits for a frame
    with ``capture_perf_ts >= T_n`` from every camera, pulls the latest
    joint+action snapshot, and appends one dataset row. A camera read timeout
    reuses the previous frame for that camera (or skips the tick if none yet).
    Any fatal error is reported via ``on_error`` instead of dying silently.
    """
    try:
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame
        from lerobot.utils.visualization_utils import log_rerun_data

        # Wait for the first snapshot (the control loop publishes every tick).
        first_deadline = time.perf_counter() + 5.0
        while read_snapshot() is None:
            if stop_event.wait(0.02):
                return
            if time.perf_counter() > first_deadline:
                _logger.warning("capture loop saw no snapshot within 5s; exiting.")
                return
        if stop_event.is_set():
            return

        frame_interval = 1.0 / fps
        timeout_ms = int(2 * frame_interval * 1000 + 200)
        recording_start = time.perf_counter()
        last_frames: dict[str, tuple[Any, float, float]] = {}
        tick = 0

        tick_cost_sum = 0.0
        reuse_count = 0
        skip_count = 0
        frames_added = 0
        ticks_window = 0
        cap_last_log = recording_start

        while not stop_event.is_set():
            now = time.perf_counter()
            if now - cap_last_log >= 1.0:
                dt = now - cap_last_log
                _logger.debug(
                    "capture: %.1f fps  tick=%.1fms  added=%d reused=%d skipped=%d",
                    ticks_window / dt,
                    1e3 * tick_cost_sum / ticks_window if ticks_window else 0.0,
                    frames_added,
                    reuse_count,
                    skip_count,
                )
                tick_cost_sum = 0.0
                reuse_count = 0
                skip_count = 0
                frames_added = 0
                ticks_window = 0
                cap_last_log = now

            target_perf_ts = recording_start + tick * frame_interval
            wait_s = target_perf_ts - time.perf_counter()
            if wait_s > 0 and stop_event.wait(timeout=wait_s):
                return

            body_t0 = time.perf_counter()
            frames: dict[str, tuple[Any, float, float]] = {}
            skip_tick = False
            for cam_key, cam in cameras.items():
                try:
                    frame, cap_ts, recv_ts = cam.read_at_or_after(
                        target_perf_ts, timeout_ms=timeout_ms
                    )
                except (TimeoutError, RuntimeError) as exc:
                    cached = last_frames.get(cam_key)
                    if cached is None:
                        _logger.debug(
                            "Capture tick %d: %s read failed (%s) and no cached "
                            "frame; skipping tick.",
                            tick,
                            cam_key,
                            exc,
                        )
                        skip_tick = True
                        break
                    reuse_count += 1
                    frame, cap_ts, recv_ts = cached
                frames[cam_key] = (frame, cap_ts, recv_ts)
                last_frames[cam_key] = (frame, cap_ts, recv_ts)

            if skip_tick:
                skip_count += 1
                tick += 1
                continue

            snap = read_snapshot()
            if snap is None:
                tick += 1
                continue
            joint_obs, action, _snap_ts = snap

            obs: dict[str, Any] = dict(joint_obs)
            for cam_key, (frame, _cap_ts, _recv_ts) in frames.items():
                obs[cam_key] = frame
            obs_processed = robot_obs_proc(obs)

            obs_frame = build_dataset_frame(
                dataset.features, obs_processed, prefix=OBS_STR
            )
            act_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
            if stop_event.is_set():
                return
            dataset.add_frame({**obs_frame, **act_frame, "task": task})
            frames_added += 1
            tick_cost_sum += time.perf_counter() - body_t0
            ticks_window += 1

            if rerun_ip:
                log_rerun_data(observation=obs_processed, action=action)

            tick += 1
    except Exception as exc:  # noqa: BLE001 - surface instead of dying silently
        _logger.error("capture loop failed: %s", exc)
        if on_error is not None:
            on_error(str(exc))


# ---------------------------------------------------------------------------
# Dataset open + finalize (shared by both recorders)
# ---------------------------------------------------------------------------


def _open_dataset(config: dict) -> "LeRobotDataset":
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if config["is_complete"]:
        return LeRobotDataset.resume(
            repo_id=config["repo_id"],
            root=config["dataset_root"],
            image_writer_threads=4,
            streaming_encoding=True,
            encoder_threads=_ENCODER_THREADS,
            vcodec=config["vcodec"],
        )
    return LeRobotDataset.create(
        repo_id=config["repo_id"],
        fps=config["fps"],
        root=config["root"],
        features=config["features"],
        robot_type=config["robot_type"],
        use_videos=True,
        image_writer_threads=4,
        streaming_encoding=True,
        encoder_threads=_ENCODER_THREADS,
        vcodec=config["vcodec"],
    )


def _finalize_dataset(
    dataset: "LeRobotDataset", config: dict, episodes_recorded: int
) -> None:
    from lerobot.utils.utils import log_say

    dataset.finalize()
    if config["push_to_hub"] and episodes_recorded > 0:
        dataset.push_to_hub()
    dataset_root = Path(config["dataset_root"])
    if not config["is_complete"] and episodes_recorded == 0 and dataset_root.exists():
        try:
            shutil.rmtree(dataset_root)
            log_say(f"No episodes saved — removed empty dataset at {dataset_root}.")
        except OSError as exc:
            _logger.warning(
                "Failed to remove empty dataset at %s: %s", dataset_root, exc
            )


# ---------------------------------------------------------------------------
# In-process recorder (no-relay fallback)
# ---------------------------------------------------------------------------


class InProcessRecorder:
    """Dataset + capture thread in the control process (no-relay fallback).

    Used only when the gst video relay is unavailable. Owns the dataset, reads
    the robot's own (SDK) cameras, and runs the capture loop on a thread here.
    """

    def __init__(self, config: dict, robot: Any, robot_obs_proc: Callable) -> None:
        install_dataset_encoder()
        self._config = config
        self._robot = robot
        self._robot_obs_proc = robot_obs_proc
        self._dataset = _open_dataset(config)
        self._publisher = _SnapshotPublisher()
        self._thread: threading.Thread | None = None
        self._stop: threading.Event | None = None
        self._episodes_recorded = 0

    def publish(self, joint_obs: dict, action: dict, ts: float) -> None:
        self._publisher.write(joint_obs, action, ts)

    def episode_count(self) -> int:
        return self._dataset.num_episodes

    def start_episode(self, task: str) -> None:
        self._stop_capture()  # defensive: never overlap two capture threads
        self._dataset.clear_episode_buffer()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=run_capture_loop,
            kwargs=dict(
                cameras=self._robot.cameras,
                read_snapshot=self._publisher.read_latest,
                dataset=self._dataset,
                robot_obs_proc=self._robot_obs_proc,
                fps=self._config["fps"],
                task=task,
                rerun_ip=self._config["rerun_ip"],
                stop_event=self._stop,
            ),
            name="axol-capture",
            daemon=True,
        )
        self._thread.start()

    def _stop_capture(self) -> None:
        if self._thread is not None and self._stop is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None

    def save_episode(self) -> None:
        self._stop_capture()
        self._dataset.save_episode()
        self._episodes_recorded += 1

    def cancel_episode(self) -> None:
        self._stop_capture()
        self._dataset.clear_episode_buffer()

    def close(self) -> None:
        self._stop_capture()
        _finalize_dataset(self._dataset, self._config, self._episodes_recorded)


# ---------------------------------------------------------------------------
# Recorder subprocess (relay path)
# ---------------------------------------------------------------------------


def _recorder_main(
    conn: multiprocessing.connection.Connection,
    raw_cond: Any,
    config: dict,
) -> None:
    """Recorder subprocess entry: own the dataset, capture from shared memory."""
    logging.basicConfig(level=config["log_level"])

    # Keep the recorder (+ its NVENC gst children, which inherit this) off the
    # control loop's cores; fall back to a positive nice where affinity isn't
    # available so it still never preempts the control loop / IK.
    from ..utils import affinity

    if not affinity.pin_background():
        try:
            os.nice(5)
        except (AttributeError, OSError):
            pass

    from lerobot.processor import make_default_processors

    from ..vr.shm_frames import GstShmFrameReader, RawFrameReader, SnapshotReader

    install_dataset_encoder()
    _, _, robot_obs_proc = make_default_processors()

    # Build a per-source raw-frame reader matching the relay's chosen transport.
    # gstshm: a shmsrc → appsink consumer pulling on THIS process's GIL (so the
    # relay's send is never starved — the fix). pyshm: the older RawFrameReader
    # over a shared-memory block the relay's Python pull loop fills. Started here
    # (before "ready") and torn down in the finally; the relay's rawvalve gates
    # episode on/off, so the consumers can run continuously and just idle when
    # the valve is closed.
    cameras: dict[str, Any] = {}
    for source, meta in config["raw_meta"].items():
        if meta["transport"] == "gstshm":
            cam = GstShmFrameReader(
                meta["socket_path"],
                meta["caps"],
                meta["width"],
                meta["height"],
                meta["fps"],
                meta["latency_s"],
            )
            cam.connect()
            cameras[source] = cam
        else:
            cameras[source] = RawFrameReader(
                meta["shm_name"],
                meta["width"],
                meta["height"],
                meta["fps"],
                raw_cond,
            )
    snap_reader = SnapshotReader(
        config["snapshot_shm_name"], config["obs_keys"], config["action_keys"]
    )

    if config["rerun_ip"]:
        from lerobot.utils.visualization_utils import init_rerun

        init_rerun(
            session_name="axol_record", ip=config["rerun_ip"], port=config["rerun_port"]
        )

    dataset = _open_dataset(config)
    conn.send(("ready", dataset.num_episodes))

    thread: threading.Thread | None = None
    stop: threading.Event | None = None
    capture_error: dict[str, str | None] = {"v": None}
    episodes_recorded = 0

    def stop_capture() -> None:
        nonlocal thread
        if thread is not None and stop is not None:
            stop.set()
            thread.join(timeout=10.0)
            thread = None

    try:
        while True:
            try:
                msg = conn.recv()
            except (EOFError, KeyboardInterrupt):
                break
            if msg is None or msg[0] == "shutdown":
                break
            kind = msg[0]
            if kind == "start_episode":
                task = msg[1]
                stop_capture()  # defensive: never overlap two capture threads
                dataset.clear_episode_buffer()
                capture_error["v"] = None
                stop = threading.Event()
                thread = threading.Thread(
                    target=run_capture_loop,
                    kwargs=dict(
                        cameras=cameras,
                        read_snapshot=snap_reader.read_latest,
                        dataset=dataset,
                        robot_obs_proc=robot_obs_proc,
                        fps=config["fps"],
                        task=task,
                        rerun_ip=config["rerun_ip"],
                        stop_event=stop,
                        on_error=lambda m: capture_error.__setitem__("v", m),
                    ),
                    name="axol-capture",
                    daemon=True,
                )
                thread.start()
            elif kind == "save_episode":
                stop_capture()
                if capture_error["v"] is not None:
                    conn.send(("error", capture_error["v"]))
                else:
                    try:
                        dataset.save_episode()
                        episodes_recorded += 1
                        conn.send(("saved", dataset.num_episodes))
                    except Exception as exc:  # noqa: BLE001 - report to control proc
                        _logger.error("recorder save_episode failed: %s", exc)
                        conn.send(("error", str(exc)))
            elif kind == "cancel_episode":
                stop_capture()
                dataset.clear_episode_buffer()
                conn.send(("cancelled",))
    finally:
        stop_capture()
        with contextlib.suppress(Exception):
            _finalize_dataset(dataset, config, episodes_recorded)
        for cam in cameras.values():
            with contextlib.suppress(Exception):
                cam.close()
        with contextlib.suppress(Exception):
            snap_reader.close()


class DatasetRecorderProcess:
    """Parent-side handle for the recorder subprocess.

    Creates the cross-process :class:`SnapshotWriter`, spawns the recorder, and
    exposes the same interface as :class:`InProcessRecorder`. ``publish`` is the
    only hot-path call (one ~40-float shm write per control tick); the episode
    commands are rare and run on the main thread between episodes.
    """

    def __init__(
        self,
        *,
        raw_cond: Any,
        raw_meta: dict[str, dict],
        obs_keys: list[str],
        action_keys: list[str],
        config: dict,
    ) -> None:
        from ..vr.shm_frames import SnapshotWriter

        self._snap = SnapshotWriter(obs_keys, action_keys)
        ctx = multiprocessing.get_context("spawn")
        self._conn, child_conn = ctx.Pipe()
        full_config = {
            **config,
            "raw_meta": raw_meta,
            "obs_keys": obs_keys,
            "action_keys": action_keys,
            "snapshot_shm_name": self._snap.name,
        }
        self._proc = ctx.Process(
            target=_recorder_main,
            args=(child_conn, raw_cond, full_config),
            daemon=True,
            name="dataset-recorder",
        )
        self._proc.start()
        child_conn.close()
        self._lock = threading.Lock()
        self._episode_count = 0

        if self._conn.poll(_READY_TIMEOUT_S):
            msg = self._conn.recv()
            if isinstance(msg, tuple) and msg[0] == "ready":
                self._episode_count = int(msg[1])
            else:
                raise RuntimeError(f"recorder sent unexpected ready message: {msg!r}")
        else:
            raise RuntimeError("recorder subprocess did not become ready in time")

    @property
    def pid(self) -> int | None:
        return self._proc.pid

    def publish(self, joint_obs: dict, action: dict, ts: float) -> None:
        self._snap.write(joint_obs, action, ts)

    def episode_count(self) -> int:
        return self._episode_count

    def start_episode(self, task: str) -> None:
        with self._lock:
            self._conn.send(("start_episode", task))

    def save_episode(self) -> None:
        with self._lock:
            self._conn.send(("save_episode",))
            if not self._conn.poll(_SAVE_TIMEOUT_S):
                raise RuntimeError("recorder did not finish save_episode in time")
            msg = self._conn.recv()
        if msg[0] == "saved":
            self._episode_count = int(msg[1])
        elif msg[0] == "error":
            raise RuntimeError(f"recorder save_episode failed: {msg[1]}")

    def cancel_episode(self) -> None:
        with self._lock:
            self._conn.send(("cancel_episode",))
            if self._conn.poll(_SAVE_TIMEOUT_S):
                self._conn.recv()  # ("cancelled",)

    def close(self) -> None:
        try:
            with self._lock:
                self._conn.send(("shutdown",))
        except (OSError, ValueError):
            pass
        self._proc.join(timeout=_SAVE_TIMEOUT_S)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=5.0)
        with contextlib.suppress(Exception):
            self._conn.close()
        with contextlib.suppress(Exception):
            self._snap.close()
