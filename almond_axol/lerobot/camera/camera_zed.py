"""
ZED stream receiver camera for LeRobot.

ZedCamera connects to a single ZED video stream produced by ZedStreamer and
exposes it as a standard LeRobot Camera. One instance per camera — instantiate
three to cover overhead, left_arm, and right_arm.

Each grabbed frame carries two timestamps, both on the receiver's
``time.perf_counter`` clock:

* ``capture_perf_ts`` — when the sender exposed the frame, derived from
  the SDK's ``TIME_REFERENCE.IMAGE`` (sender wall clock) plus a per-frame
  wall→perf offset. Used by ``collect_data`` so dataset rows record the
  moment of capture, not the moment of decode. Requires the two machines'
  ``CLOCK_REALTIME`` to be aligned — see ``axol zed.sync-clocks``.
* ``receive_perf_ts`` — when this process decoded the frame.

Typical usage::

    from almond_axol.lerobot.zed import ZedCamera, ZedCameraConfig

    overhead  = ZedCamera(ZedCameraConfig(host="192.168.1.10", port=30000))
    left_arm  = ZedCamera(ZedCameraConfig(host="192.168.1.10", port=30002))
    right_arm = ZedCamera(ZedCameraConfig(host="192.168.1.10", port=30004))

    with overhead, left_arm, right_arm:
        frame = overhead.read()  # uint8 numpy array (1080, 1920, 3) RGB
"""

from __future__ import annotations

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
import pyzed.sl as sl
from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import ColorMode
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError
from numpy.typing import NDArray

from .configuration_zed import ZedCameraConfig

_logger = logging.getLogger(__name__)


class ZedCamera(Camera):
    """LeRobot camera that receives a ZED video stream over the local network.

    Connects to a stream started by ZedStreamer using the ZED SDK's local
    streaming API. A background thread continuously calls grab() and stores the
    latest frame so read() and async_read() never block on the network.

    Args:
        config: Host, port, color mode, and warmup duration.
    """

    def __init__(self, config: ZedCameraConfig) -> None:
        super().__init__(config)
        self.config = config

        self.zed: sl.CameraOne | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_capture_perf_ts: float | None = None
        self.latest_receive_perf_ts: float | None = None
        self.new_frame_event: Event = Event()

    def __str__(self) -> str:
        return f"ZedCamera({self.config.host}:{self.config.port})"

    @property
    def is_connected(self) -> bool:
        return self.zed is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Stream receivers do not enumerate hardware — returns empty list."""
        return []

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        """Connect to the ZED stream and start the background grab thread.

        Args:
            warmup: If True, reads frames for `config.warmup_s` seconds before
                    returning so the frame buffer is primed.

        Raises:
            ConnectionError: If the stream cannot be opened.
        """
        zed = sl.CameraOne()
        init_params = sl.InitParametersOne()
        init_params.set_from_stream(self.config.host, self.config.port)

        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise ConnectionError(
                f"{self} failed to open stream at {self.config.host}:{self.config.port}: {err}"
            )

        self.zed = zed

        # Read actual resolution and FPS from the stream and validate against config
        info = zed.get_camera_information()
        params = info.camera_configuration.resolution
        stream_fps = int(info.camera_configuration.fps)
        stream_width = int(params.width)
        stream_height = int(params.height)

        mismatches = []
        if self.config.fps is not None and stream_fps != self.config.fps:
            mismatches.append(f"fps: expected {self.config.fps}, got {stream_fps}")
        if self.config.width is not None and stream_width != self.config.width:
            mismatches.append(
                f"width: expected {self.config.width}, got {stream_width}"
            )
        if self.config.height is not None and stream_height != self.config.height:
            mismatches.append(
                f"height: expected {self.config.height}, got {stream_height}"
            )
        if mismatches:
            zed.close()
            raise RuntimeError(
                f"{self} stream parameters do not match config — "
                + ", ".join(mismatches)
                + ". Update ZedCameraConfig or the sender settings."
            )

        self.fps = stream_fps
        self.width = stream_width
        self.height = stream_height

        self._start_read_thread()

        if warmup:
            start = time.time()
            while time.time() - start < self.config.warmup_s:
                try:
                    self.async_read(timeout_ms=self.config.warmup_s * 1000)
                except TimeoutError:
                    pass
                time.sleep(0.05)

        self._log_pipeline_latency()

        _logger.info(f"{self} connected ({self.width}x{self.height} @ {self.fps}fps).")

    def _log_pipeline_latency(self, num_samples: int = 30) -> None:
        """Log mean/max ``receive_perf_ts - capture_perf_ts`` over ~N frames.

        Acts as a startup canary for PTP: if the sender and receiver wall
        clocks are out of sync the latency looks wildly negative or huge.
        """
        samples: list[float] = []
        deadline = time.perf_counter() + 5.0
        while len(samples) < num_samples and time.perf_counter() < deadline:
            self.new_frame_event.clear()
            if not self.new_frame_event.wait(timeout=0.5):
                continue
            with self.frame_lock:
                cap = self.latest_capture_perf_ts
                recv = self.latest_receive_perf_ts
            if cap is None or recv is None:
                continue
            samples.append(recv - cap)

        if not samples:
            _logger.warning(
                "%s: no frames captured during pipeline-latency probe; "
                "skipping startup PTP check.",
                self,
            )
            return

        mean_ms = sum(samples) / len(samples) * 1e3
        max_ms = max(samples) * 1e3
        _logger.info(
            "%s pipeline latency over %d frames: mean=%.1fms max=%.1fms.",
            self,
            len(samples),
            mean_ms,
            max_ms,
        )

        if mean_ms < 0.0 or mean_ms > 200.0:
            _logger.warning(
                "%s pipeline latency looks unhealthy (mean=%.1fms). "
                "Looks like the sender/receiver wall-clocks aren't synced — "
                "is `axol zed.sync-clocks` running on both machines?",
                self,
                mean_ms,
            )

    def _start_read_thread(self) -> None:
        self._stop_read_thread()
        self.stop_event = Event()
        self.thread = Thread(
            target=self._read_loop, name=f"{self}_read_loop", daemon=True
        )
        self.thread.start()

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.stop_event = None
        with self.frame_lock:
            self.latest_frame = None
            self.latest_capture_perf_ts = None
            self.latest_receive_perf_ts = None
            self.new_frame_event.clear()

    def _read_loop(self) -> None:
        if self.stop_event is None or self.zed is None:
            return

        image = sl.Mat()
        failure_count = 0

        while not self.stop_event.is_set():
            try:
                err = self.zed.grab()
                if err != sl.ERROR_CODE.SUCCESS:
                    _logger.debug(f"{self} grab returned {err}, skipping frame.")
                    continue

                self.zed.retrieve_image(image)
                raw = image.get_data()  # BGRA uint8 (height, width, 4)

                if self.config.color_mode == ColorMode.RGB:
                    frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
                else:
                    frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

                cap_wall = (
                    self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
                    * 1e-9
                )
                # Recompute the wall→perf offset per frame so PTP step
                # adjustments don't accumulate as silent skew.
                recv_wall = time.time()
                recv_perf = time.perf_counter()
                cap_perf = recv_perf - (recv_wall - cap_wall)

                with self.frame_lock:
                    self.latest_frame = frame
                    self.latest_capture_perf_ts = cap_perf
                    self.latest_receive_perf_ts = recv_perf
                self.new_frame_event.set()
                failure_count = 0

            except DeviceNotConnectedError:
                break
            except Exception as exc:
                failure_count += 1
                if failure_count <= 10:
                    _logger.warning(f"{self} read loop error: {exc}")
                else:
                    raise RuntimeError(
                        f"{self} exceeded maximum consecutive read failures."
                    ) from exc

    @check_if_not_connected
    def read(self) -> NDArray[Any]:
        """Return a single frame, blocking until one is available."""
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")
        self.new_frame_event.clear()
        return self.async_read(timeout_ms=10000)

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """Return the latest unconsumed frame, waiting up to timeout_ms for one."""
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"{self} timed out waiting for frame after {timeout_ms}ms. "
                f"Thread alive: {self.thread.is_alive()}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"{self}: event set but no frame available.")

        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """Return the most recent frame immediately without waiting.

        Raises:
            TimeoutError: If the latest frame is older than max_age_ms
                (measured against ``receive_perf_ts``).
            RuntimeError: If no frame has been captured yet.
        """
        frame, _cap_ts, recv_ts = self.read_latest_with_ts()
        age_ms = (time.perf_counter() - recv_ts) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f}ms (max {max_age_ms}ms)."
            )
        return frame

    @check_if_not_connected
    def read_latest_with_ts(self) -> tuple[NDArray[Any], float, float]:
        """Return ``(frame, capture_perf_ts, receive_perf_ts)`` for the latest frame.

        Both timestamps are on the receiver's ``perf_counter`` clock (see the
        module docstring).

        Raises:
            RuntimeError: If the grab thread is not running or no frame has
                been captured yet.
        """
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        with self.frame_lock:
            frame = self.latest_frame
            cap_ts = self.latest_capture_perf_ts
            recv_ts = self.latest_receive_perf_ts

        if frame is None or cap_ts is None or recv_ts is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        return frame, cap_ts, recv_ts

    @check_if_not_connected
    def read_at_or_after(
        self,
        target_capture_perf_ts: float,
        timeout_ms: float = 500,
    ) -> tuple[NDArray[Any], float, float]:
        """Block until a frame with ``capture_perf_ts >= target`` is available.

        Used by ``collect-data`` so every camera and the joint sample share
        the same sender-side timeline.

        Args:
            target_capture_perf_ts: Earliest acceptable ``capture_perf_ts``.
            timeout_ms:             Maximum time to wait for a qualifying frame.

        Returns:
            ``(frame, capture_perf_ts, receive_perf_ts)``.

        Raises:
            TimeoutError: If no qualifying frame arrives within ``timeout_ms``.
            RuntimeError: If the grab thread is not running.
        """
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        deadline = time.perf_counter() + timeout_ms / 1000.0

        while True:
            self.new_frame_event.clear()
            with self.frame_lock:
                frame = self.latest_frame
                cap_ts = self.latest_capture_perf_ts
                recv_ts = self.latest_receive_perf_ts
            if (
                frame is not None
                and cap_ts is not None
                and recv_ts is not None
                and cap_ts >= target_capture_perf_ts
            ):
                return frame, cap_ts, recv_ts

            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(
                    f"{self} timed out waiting for frame at "
                    f"capture_perf_ts >= {target_capture_perf_ts:.6f} "
                    f"after {timeout_ms:.1f}ms "
                    f"(latest cap_ts={cap_ts!r})."
                )
            self.new_frame_event.wait(timeout=remaining)

    def disconnect(self) -> None:
        """Stop the grab thread and close the ZED stream."""
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it is already disconnected."
            )

        self._stop_read_thread()

        if self.zed is not None:
            self.zed.close()
            self.zed = None

        _logger.info(f"{self} disconnected.")
