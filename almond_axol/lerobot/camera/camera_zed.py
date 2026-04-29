"""
ZED stream receiver camera for LeRobot.

ZedCamera connects to a single ZED video stream produced by ZedStreamer and
exposes it as a standard LeRobot Camera. One instance per camera — instantiate
three to cover overhead, left_arm, and right_arm.

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
        self.latest_timestamp: float | None = None
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

        # Always read resolution and FPS from the stream — do not use config values
        info = zed.get_camera_information()
        params = info.camera_configuration.resolution
        self.fps = int(info.camera_configuration.fps)
        self.width = int(params.width)
        self.height = int(params.height)

        self._start_read_thread()

        if warmup:
            start = time.time()
            while time.time() - start < self.config.warmup_s:
                try:
                    self.async_read(timeout_ms=self.config.warmup_s * 1000)
                except TimeoutError:
                    pass
                time.sleep(0.05)

        _logger.info(f"{self} connected ({self.width}x{self.height} @ {self.fps}fps).")

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
            self.latest_timestamp = None
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

                capture_time = time.perf_counter()
                with self.frame_lock:
                    self.latest_frame = frame
                    self.latest_timestamp = capture_time
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
            TimeoutError: If the latest frame is older than max_age_ms.
            RuntimeError: If no frame has been captured yet.
        """
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        with self.frame_lock:
            frame = self.latest_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f}ms (max {max_age_ms}ms)."
            )

        return frame

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
