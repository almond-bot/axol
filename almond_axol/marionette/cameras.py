"""Camera frames for the phones and the desktop page, compressed once.

The phone screen declares three streams (``head``, ``left``, ``right``) and
each phone picks a view of them. :class:`CameraHub` grabs only the streams
someone is watching (a phone's view or the desktop page), downscales each to
``width`` and JPEG-encodes it at ``quality``, ``fps`` times a second. The
gateway client and the desktop WebSocket both read the newest encoded frame
of a stream with :meth:`CameraHub.latest`, so each frame is compressed once
however many clients watch it.

Sources:

- :class:`ZedCameraSource`: the robot's ZED cameras, through the same video
  relay subprocess data collection uses, with only its raw shared-memory
  branch (no headset stream). The in-process ZED SDK is the fallback.
- :class:`~almond_axol.marionette.sim_cameras.SimCameraSource`: the
  simulated robot rendered from its head and wrist camera mounts.
- :class:`SyntheticCameraSource`: a labelled test picture per stream, for
  sim when rendering is unavailable.

A camera failure never stops teleoperation; it is reported in the state.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

_logger = logging.getLogger(__name__)

STREAMS: tuple[str, ...] = ("head", "left", "right")
STREAM_LABELS = {"head": "Head", "left": "Left wrist", "right": "Right wrist"}
# The phone screen's camera views (camera.view values).
VIEWS: dict[str, tuple[str, ...]] = {
    "head": ("head",),
    "wrists": ("left", "right"),
    "left": ("left",),
    "right": ("right",),
    "all": ("head", "left", "right"),
}
# The control panel's camera slots for each stream.
SLOTS = {"head": "overhead", "left": "left_arm", "right": "right_arm"}
# A watcher that stops renewing its demand is forgotten after this long.
DEMAND_TTL_S = 3.0
# A watched stream with no new frame for this long is reported as stalled,
# and its last frame is no longer shown.
STALE_S = 2.0


def streams_for_view(view: str) -> tuple[str, ...]:
    return VIEWS.get(view, VIEWS["head"])


def encode_jpeg(rgb: np.ndarray, width: int, quality: int) -> bytes:
    """Downscale an RGB frame to ``width`` (never up) and JPEG-encode it."""
    import cv2

    frame = np.asarray(rgb)
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]
    h, w = frame.shape[:2]
    if w > width:
        height = max(2, round(h * width / w))
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(
        ".jpg",
        cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return data.tobytes()


class CameraSource(Protocol):
    """Where frames come from."""

    name: str
    available: tuple[str, ...]
    error: str | None

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def grab(
        self, streams: Iterable[str], width: int, quality: int
    ) -> dict[str, bytes]:
        """The newest JPEG of each stream that has one."""
        ...


@dataclass(frozen=True)
class EncodedFrame:
    stream: str
    jpeg: bytes
    captured_ms: float  # wall clock, milliseconds
    seq: int


class CameraHub:
    """Grab, compress and hold the newest frame of every watched stream."""

    def __init__(
        self,
        source: CameraSource | None,
        *,
        fps: float = 12.0,
        width: int = 640,
        quality: int = 60,
        error: str | None = None,
    ) -> None:
        self.source = source
        self._error = error
        self.fps = max(1.0, float(fps))
        self.width = max(64, int(width))
        self.quality = min(max(int(quality), 10), 95)
        self._lock = threading.Lock()
        self._demand: dict[str, tuple[frozenset[str], float]] = {}
        self._frames: dict[str, EncodedFrame] = {}
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failure: str | None = None
        # When each watched stream started being watched (wall clock, ms).
        self._watched_since: dict[str, float] = {}

    # -- demand --------------------------------------------------------------

    def want(self, watcher: str, streams: Iterable[str]) -> None:
        """``watcher`` wants these streams; renew within DEMAND_TTL_S."""
        wanted = frozenset(s for s in streams if s in STREAMS)
        with self._lock:
            self._demand[watcher] = (wanted, time.monotonic())

    def drop(self, watcher: str) -> None:
        with self._lock:
            self._demand.pop(watcher, None)

    def wanted(self) -> set[str]:
        now = time.monotonic()
        with self._lock:
            for key, (_, at) in list(self._demand.items()):
                if now - at > DEMAND_TTL_S:
                    del self._demand[key]
            return set().union(*(s for s, _ in self._demand.values()))

    # -- frames --------------------------------------------------------------

    def latest(self, stream: str) -> EncodedFrame | None:
        """The newest frame of a stream; None once it is older than STALE_S."""
        with self._lock:
            frame = self._frames.get(stream)
        if frame is None or time.time() * 1000.0 - frame.captured_ms > STALE_S * 1000:
            return None
        return frame

    def stalled(self) -> list[str]:
        """Watched streams whose frames stopped coming."""
        now = time.time() * 1000.0
        with self._lock:
            since = dict(self._watched_since)
            frames = dict(self._frames)
        late = now - STALE_S * 1000
        return [
            s
            for s, at in sorted(since.items())
            if at < late and (s not in frames or frames[s].captured_ms < late)
        ]

    @property
    def available(self) -> list[str]:
        source = self.source
        return list(source.available) if source is not None else []

    @property
    def error(self) -> str | None:
        source = self.source
        if source is None:
            return self._error
        error = self._failure or source.error
        if error is None:
            stalled = self.stalled()
            if stalled:
                names = ", ".join(STREAM_LABELS[s] for s in stalled)
                error = f"{names} camera stopped sending frames."
        return error

    def describe(self) -> dict[str, Any]:
        source = self.source
        return {
            "available": self.available,
            "source": None if source is None else source.name,
            "error": self.error,
        }

    # -- worker --------------------------------------------------------------

    def start(self) -> None:
        if self.source is None or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="marionette-cameras", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            # Closing the cameras can take a while; the next operation must
            # not find them still open.
            thread.join(timeout=60.0)
            if thread.is_alive():
                _logger.warning("marionette: the cameras are still closing")
        self._thread = None

    def _run(self) -> None:
        source = self.source
        assert source is not None
        try:
            source.start()
        except Exception as exc:  # noqa: BLE001 - teleop goes on without video
            _logger.warning("marionette: cameras unavailable (%s)", exc)
            self._failure = f"The cameras could not start: {exc}"
            try:
                source.stop()
            except Exception:  # noqa: BLE001
                _logger.warning("marionette: camera cleanup failed", exc_info=True)
            return
        period = 1.0 / self.fps
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                wanted = self.wanted() & set(source.available)
                with self._lock:
                    now_ms = time.time() * 1000.0
                    self._watched_since = {
                        s: self._watched_since.get(s, now_ms) for s in wanted
                    }
                if wanted:
                    try:
                        frames = source.grab(sorted(wanted), self.width, self.quality)
                        self._failure = None
                    except Exception as exc:  # noqa: BLE001 - keep trying
                        _logger.debug("marionette: camera grab failed", exc_info=True)
                        self._failure = f"Camera frames stopped: {exc}"
                        frames = {}
                    captured = time.time() * 1000.0
                    with self._lock:
                        for stream, jpeg in frames.items():
                            self._seq += 1
                            self._frames[stream] = EncodedFrame(
                                stream, jpeg, captured, self._seq
                            )
                self._stop.wait(max(0.0, period - (time.monotonic() - started)))
        finally:
            try:
                source.stop()
            except Exception:  # noqa: BLE001
                _logger.warning("marionette: camera source stop failed", exc_info=True)


class SyntheticCameraSource:
    """A labelled moving test picture per stream (no rendering needed)."""

    name = "synthetic"
    available = STREAMS
    error: str | None = None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def grab(
        self, streams: Iterable[str], width: int, quality: int
    ) -> dict[str, bytes]:
        import cv2

        out: dict[str, bytes] = {}
        t = time.monotonic()
        height = round(width * 9 / 16)
        for stream in streams:
            frame = np.zeros((height, width, 3), np.uint8)
            frame[:] = (40, 52, 64)
            x = int((0.5 + 0.4 * math.sin(t * 0.8)) * width)
            cv2.circle(frame, (x, height // 2), height // 6, (90, 170, 220), -1)
            cv2.putText(
                frame,
                f"{STREAM_LABELS[stream]} (sim)",
                (16, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )
            out[stream] = encode_jpeg(frame, width, quality)
        return out


class ZedCameraSource:
    """The robot's ZED cameras, read from shared memory or the ZED SDK.

    ``cameras`` maps the control panel's slots (``overhead``, ``left_arm``,
    ``right_arm``) to serials, as the panel passes them to teleop.
    """

    name = "zed"

    def __init__(
        self,
        cameras: dict[str, int],
        *,
        resolution: str | None = None,
        fps: float = 15.0,
    ) -> None:
        self._cameras = {
            s: int(cameras[SLOTS[s]]) for s in STREAMS if SLOTS[s] in cameras
        }
        self._resolution = resolution or "SVGA"
        self._fps = fps
        self._relay: Any = None
        self._readers: dict[str, Any] = {}
        self._sdk: list[Any] = []
        self.available: tuple[str, ...] = ()
        self.error: str | None = None

    def start(self) -> None:
        if not self._cameras:
            self.error = "No cameras are assigned. Assign them in Settings, Cameras."
            return
        try:
            self._start_relay()
        except Exception as exc:  # noqa: BLE001 - try the SDK next
            _logger.info("marionette: video relay unavailable (%s)", exc)
        missing = [s for s in self._cameras if s not in self._readers]
        if missing:
            self._start_sdk(missing)
        self.available = tuple(s for s in STREAMS if s in self._readers)
        missing = [STREAM_LABELS[s] for s in self._cameras if s not in self._readers]
        if missing:
            self.error = (
                f"{', '.join(missing)} camera not available. Check the camera "
                "cables and serials in Settings, Cameras."
            )

    def _start_relay(self) -> None:
        from ..video.video import webrtc_available
        from ..video.video_proc import VideoRelayProcess
        from ..zed import stereo_serials

        if not webrtc_available():
            raise RuntimeError("the video relay is not installed")
        stereo = stereo_serials()
        specs: dict[str, dict[str, Any]] = {}
        for stream, serial in self._cameras.items():
            spec: dict[str, Any] = {
                "serial": serial,
                "resolution": self._resolution,
                # Capture at the rate the cameras support everywhere; the
                # raw branch is decimated to dataset_fps.
                "fps": 60,
                "dataset_fps": int(round(self._fps)),
                "dataset_resolution": "SVGA",
                "record": True,
                "stream": False,
                # Frames readable by this process, like collect-dagger's
                # policy observations.
                "raw_transport": "pyshm",
            }
            if serial in stereo:
                spec.update(stereo=True, record_eyes=["left"], record_suffix=False)
            specs[stream] = spec
        relay = VideoRelayProcess(specs, want_raw=True)
        try:
            readers = dict(relay.readable_raw_cameras)
        except BaseException:
            relay.shutdown()
            raise
        if not readers:
            # Nothing readable: free the cameras for the SDK fallback.
            relay.shutdown()
            raise RuntimeError("the video relay exported no readable frames")
        self._relay = relay
        self._readers.update(readers)

    def _start_sdk(self, streams: list[str]) -> None:
        from types import SimpleNamespace

        from ..cli.teleop import _connect_zed_cameras
        from ..zed import stereo_serials

        cfg = SimpleNamespace(
            cameras={s: self._cameras[s] for s in streams},
            camera_eyes={s: "left" for s in streams},
            resolution=self._resolution,
        )
        stereo = {s for s in stereo_serials() if s in cfg.cameras.values()}
        for name, camera in _connect_zed_cameras(cfg, stereo):
            self._sdk.append(camera)
            self._readers[name] = camera

    def stop(self) -> None:
        for camera in self._sdk:
            try:
                camera.disconnect()
            except Exception:  # noqa: BLE001
                _logger.warning("marionette: camera disconnect failed", exc_info=True)
        self._sdk.clear()
        if self._relay is not None:
            try:
                self._relay.shutdown()
            except Exception:  # noqa: BLE001
                _logger.warning("marionette: relay shutdown failed", exc_info=True)
            self._relay = None
        self._readers.clear()

    def grab(
        self, streams: Iterable[str], width: int, quality: int
    ) -> dict[str, bytes]:
        out: dict[str, bytes] = {}
        for stream in streams:
            reader = self._readers.get(stream)
            if reader is None:
                continue
            try:
                frame = reader.read_latest()
            except Exception:  # noqa: BLE001 - no frame yet, or a stale one
                continue
            out[stream] = encode_jpeg(frame, width, quality)
        return out
