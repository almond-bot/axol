"""Out-of-process NVENC video encoder for LeRobot dataset recording (Jetson).

A drop-in replacement for LeRobot's in-process ``StreamingVideoEncoder`` that
offloads H.264 encoding to the Jetson hardware encoder (``nvv4l2h264enc``) via a
per-camera ``gst-launch-1.0`` subprocess.

Why this exists
---------------
LeRobot encodes recorded video with libx264 **in the control process**. On the
Jetson that is fatal during ``collect-data``: encoding three camera streams at
60 fps not only burns CPU cores but does per-frame Python work (PIL conversion,
running stats) that holds the GIL, starving the asyncio control loop. The loop
collapses from 120 Hz to ~50 Hz (jerky arm) and the encoder still can't keep up,
overflowing its queue and dropping frames. Tuning the libx264 preset/GOP is not
enough — the work is simply in the wrong place.

The Jetson's hardware encoder is only reachable through GStreamer
(``nvv4l2h264enc``); the PyAV encoders LeRobot can use either don't open on Tegra
(``h264_nvenc``, ``h264_v4l2m2m``) or are software (libx264). So each camera gets
its own ``gst-launch`` pipeline:

    fdsrc -> rawvideoparse(rgba) -> nvvidconv -> NV12(NVMM) ->
    nvv4l2h264enc -> h264parse -> mp4mux -> filesink

The control process only writes raw frame bytes to a pipe (the ``os.write``
releases the GIL), so the encode cost — colorspace convert on the VIC and H.264
on NVENC — lands on dedicated hardware blocks, not the CPU cores the control loop
needs. ``nvvidconv`` rejects packed 24-bit RGB, so frames are padded RGB->RGBA
(the alpha byte is ignored by the BGRx/RGBA path) before being fed in.
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..vr.hw_video import hw_h264_available

if TYPE_CHECKING:
    from numpy.typing import NDArray

_logger = logging.getLogger(__name__)

# Stored-video bitrate budget, in bits per pixel-second (width*height*fps*bpp),
# clamped to a sane range. Higher than the headset relay's ~0.12 bpp since this
# is the training data we keep, not a bandwidth-constrained LAN stream.
_BITRATE_BPP = 0.25
_BITRATE_MIN = 6_000_000
_BITRATE_MAX = 40_000_000

# Sample every Nth frame for image stats. Stats feed dataset normalization and
# are robust to subsampling. They are accumulated *during* the episode, on the
# encoder's writer thread in the recorder subprocess — not by decoding the
# finished mp4 afterwards (which made save_episode slow). The recorder has its
# own GIL/cores, away from the control loop and the relay's WebRTC send, so this
# light per-frame work no longer harms either; computing on the source frames
# (pre-encode) also matches LeRobot's software path more closely than decoding.
_STATS_EVERY = 4


def hw_dataset_encoder_available() -> bool:
    """True when the Jetson NVENC GStreamer encoder is usable."""
    return hw_h264_available()


def _bitrate_for(width: int, height: int, fps: int) -> int:
    return max(
        _BITRATE_MIN, min(_BITRATE_MAX, int(width * height * fps * _BITRATE_BPP))
    )


def _gst_argv(
    width: int, height: int, fps: int, bitrate: int, out_path: Path
) -> list[str]:
    """``gst-launch-1.0`` argv: packed RGBA on stdin -> H.264 mp4 file.

    ``nvvidconv`` does the RGBA->NV12 colorspace conversion on the VIC and
    ``nvv4l2h264enc`` encodes on NVENC, so no CPU ``videoconvert`` is involved.
    ``h264parse`` + ``mp4mux`` produce a standard mp4; closing stdin sends EOS so
    ``mp4mux`` finalizes the moov atom and ``filesink`` flushes the file.
    """
    pipeline = [
        f"fdsrc fd=0 blocksize={width * height * 4}",
        (
            "rawvideoparse use-sink-caps=false format=rgba "
            f"width={width} height={height} framerate={fps}/1"
        ),
        "nvvidconv",
        "video/x-raw(memory:NVMM),format=NV12",
        (
            f"nvv4l2h264enc control-rate=1 bitrate={bitrate} preset-level=1 "
            f"insert-sps-pps=true idrinterval={fps} maxperf-enable=true"
        ),
        "h264parse",
        "mp4mux",
        f"filesink location={out_path} sync=false",
    ]
    return ["gst-launch-1.0", "-q", *" ! ".join(pipeline).split()]


def _to_hwc_rgb_uint8(frame: "NDArray") -> "NDArray":
    """Normalize an observation image to HWC uint8 (mirrors LeRobot)."""
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = frame.transpose(1, 2, 0)
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


class _CameraNvencEncoder:
    """One camera's NVENC pipeline plus the thread that feeds it.

    ``feed`` (called from the capture thread) only enqueues; a dedicated writer
    thread pads RGB->RGBA, writes the bytes to the gst subprocess, and folds every
    ``_STATS_EVERY``-th frame into a running image-stats accumulator. ``finish()``
    just returns the accumulated stats, so save_episode no longer re-decodes the
    whole mp4. This work runs in the recorder subprocess, off the control loop and
    the relay's WebRTC send.
    """

    def __init__(self, video_path: Path, fps: int, queue_maxsize: int) -> None:
        self.video_path = video_path
        self._fps = fps
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, queue_maxsize))
        self._proc: subprocess.Popen | None = None
        self._stdin_fd: int | None = None
        self._rgba: NDArray | None = None
        self._dims: tuple[int, int] | None = None
        self._frame_count = 0
        self._dropped = 0
        self._error: str | None = None

        # Running image stats, accumulated inline as frames are encoded (mirrors
        # LeRobot's auto_downsample + RunningQuantileStats so the result matches
        # the software path). Disabled if lerobot's compute_stats can't import.
        self._stats: object | None = None
        self._downsample = None
        self._stats_samples = 0
        try:
            from lerobot.datasets.compute_stats import (
                RunningQuantileStats,
                auto_downsample_height_width,
            )

            self._stats = RunningQuantileStats()
            self._downsample = auto_downsample_height_width
        except Exception as exc:  # noqa: BLE001 - no lerobot -> record without stats
            _logger.warning(
                "video stats unavailable for %s, recording without them: %s",
                video_path.name,
                exc,
            )

        # Rolling per-second writer-thread cost, logged at INFO so the record
        # path is visible: how much GIL-holding work each camera's encode does
        # (the RGB->RGBA copy; the os.write itself releases the GIL) and whether
        # its queue is backing up — the signal for control-loop starvation.
        self._t_copy = 0.0
        self._frames_window = 0
        self._last_log = time.perf_counter()

        self._thread = threading.Thread(
            target=self._run, name=f"nvenc-{video_path.stem}", daemon=True
        )
        self._thread.start()

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def feed(self, image: "NDArray") -> None:
        if not self._thread.is_alive():
            raise RuntimeError(
                f"NVENC encoder for {self.video_path.name} is not alive: {self._error}"
            )
        try:
            self._queue.put_nowait(image)
        except queue.Full:
            self._dropped += 1
            if self._dropped == 1 or self._dropped % 10 == 0:
                _logger.warning(
                    "NVENC encoder queue full for %s, dropped %d frame(s).",
                    self.video_path.name,
                    self._dropped,
                )

    def finish(self) -> tuple[Path, dict | None]:
        """Signal end of episode, wait for the mp4 to finalize, return stats."""
        self._queue.put(None)
        self._thread.join(timeout=120)
        if self._thread.is_alive():
            _logger.error(
                "NVENC encoder for %s did not finish in time", self.video_path.name
            )
            return self.video_path, None
        if self._error is not None:
            raise RuntimeError(
                f"NVENC encoder for {self.video_path.name} failed: {self._error}"
            )
        # Stats were accumulated inline during the episode, so finishing is just
        # the mp4 finalize (moov flush) above — no re-decode of the whole file.
        stats = (
            self._stats.get_statistics()  # type: ignore[attr-defined]
            if self._stats is not None and self._stats_samples
            else None
        )
        return self.video_path, stats

    def cancel(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5)
        self._kill()

    # -- writer thread -------------------------------------------------------

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                self._encode(item)
            self._finalize()
        except Exception as exc:  # noqa: BLE001 - surface via _error on finish()
            self._error = str(exc)
            _logger.error(
                "NVENC encoder thread for %s failed: %s", self.video_path.name, exc
            )
            self._kill()

    def _encode(self, image: "NDArray") -> None:
        t0 = time.perf_counter()
        frame = _to_hwc_rgb_uint8(image)
        h, w = frame.shape[:2]
        if self._proc is None:
            self._start(w, h)
        assert self._rgba is not None and self._stdin_fd is not None
        self._rgba[:, :, :3] = frame
        self._write(self._rgba)
        if self._stats is not None and self._frame_count % _STATS_EVERY == 0:
            self._update_stats(frame)
        self._t_copy += time.perf_counter() - t0
        self._frame_count += 1
        self._frames_window += 1
        self._maybe_log()

    def _update_stats(self, frame: "NDArray") -> None:
        """Fold one (HWC RGB uint8) frame into the running image stats.

        Mirrors LeRobot's ``compute_stats`` pipeline (downsample then quantile
        update) so the values match the software encoder path.
        """
        try:
            ds = self._downsample(frame.transpose(2, 0, 1))  # type: ignore[misc]
            self._stats.update(ds.transpose(1, 2, 0).reshape(-1, ds.shape[0]))  # type: ignore[attr-defined]
            self._stats_samples += 1
        except Exception as exc:  # noqa: BLE001 - never kill encode over stats
            _logger.debug("stats update failed for %s: %s", self.video_path.name, exc)

    def _maybe_log(self) -> None:
        now = time.perf_counter()
        dt = now - self._last_log
        if dt < 1.0:
            return
        n = self._frames_window
        copy_ms = 1e3 * self._t_copy / n if n else 0.0
        _logger.info(
            "nvenc %s: %.1f fps  copy+write=%.1fms  qdepth=%d/%d dropped=%d",
            self.video_path.stem,
            n / dt,
            copy_ms,
            self._queue.qsize(),
            self._queue.maxsize,
            self._dropped,
        )
        self._t_copy = 0.0
        self._frames_window = 0
        self._last_log = now

    def _start(self, width: int, height: int) -> None:
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        bitrate = _bitrate_for(width, height, self._fps)
        self._proc = subprocess.Popen(
            _gst_argv(width, height, self._fps, bitrate, self.video_path),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        assert self._proc.stdin is not None
        self._stdin_fd = self._proc.stdin.fileno()
        self._dims = (width, height)
        self._rgba = np.empty((height, width, 4), dtype=np.uint8)
        self._rgba[:, :, 3] = 255
        _logger.info(
            "NVENC mp4 pipeline started: %s %dx%d@%dfps %.1f Mbps (pid %d)",
            self.video_path.name,
            width,
            height,
            self._fps,
            bitrate / 1e6,
            self._proc.pid,
        )

    def _write(self, buf: "NDArray") -> None:
        """Write the whole frame to the pipe (os.write releases the GIL)."""
        assert self._stdin_fd is not None
        mv = memoryview(buf).cast("B")
        n = 0
        total = len(mv)
        while n < total:
            n += os.write(self._stdin_fd, mv[n:])

    def _finalize(self) -> None:
        """Close stdin so mp4mux writes the moov atom, then reap the process."""
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except OSError:
            pass
        rc = self._proc.wait(timeout=30)
        if rc != 0:
            err = b""
            if self._proc.stderr is not None:
                with contextlib.suppress(Exception):
                    err = self._proc.stderr.read() or b""
            raise RuntimeError(f"gst pipeline exited with {rc}: {err.decode()[-300:]}")

    def _kill(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            with contextlib.suppress(Exception):
                self._proc.wait(timeout=2)


class NvencStreamingEncoder:
    """LeRobot ``StreamingVideoEncoder``-compatible encoder backed by Jetson NVENC.

    Implements the subset of the interface that ``DatasetWriter`` uses:
    ``start_episode`` / ``feed_frame`` / ``finish_episode`` / ``cancel_episode``
    / ``close``. One ``_CameraNvencEncoder`` (and one gst subprocess) per camera.
    """

    def __init__(self, fps: int, queue_maxsize: int = 30) -> None:
        self.fps = fps
        self.queue_maxsize = queue_maxsize
        self._cams: dict[str, _CameraNvencEncoder] = {}
        self._episode_active = False
        self._closed = False

    def start_episode(self, video_keys: list[str], temp_dir: Path) -> None:
        if self._episode_active:
            self.cancel_episode()
        temp_dir = Path(temp_dir)
        self._cams = {}
        for video_key in video_keys:
            ep_dir = Path(tempfile.mkdtemp(dir=temp_dir))
            video_path = ep_dir / f"{video_key.replace('/', '_')}_streaming.mp4"
            self._cams[video_key] = _CameraNvencEncoder(
                video_path, self.fps, self.queue_maxsize
            )
        self._episode_active = True

    def feed_frame(self, video_key: str, image: "NDArray") -> None:
        if not self._episode_active:
            raise RuntimeError("No active episode. Call start_episode() first.")
        self._cams[video_key].feed(image)

    def finish_episode(self) -> dict[str, tuple[Path, dict | None]]:
        if not self._episode_active:
            raise RuntimeError("No active episode to finish.")
        results: dict[str, tuple[Path, dict | None]] = {}
        for video_key, cam in self._cams.items():
            results[video_key] = cam.finish()
        self._cams = {}
        self._episode_active = False
        return results

    def cancel_episode(self) -> None:
        if not self._episode_active:
            return
        for cam in self._cams.values():
            cam.cancel()
            video_path = cam.video_path
            if video_path.exists() or video_path.parent.exists():
                shutil.rmtree(str(video_path.parent), ignore_errors=True)
        self._cams = {}
        self._episode_active = False

    def close(self) -> None:
        if self._closed:
            return
        if self._episode_active:
            self.cancel_episode()
        self._closed = True
