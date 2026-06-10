"""
Local ZED camera for LeRobot.

ZedCamera opens a GMSL-attached ZED camera by serial number via the ZED SDK
and exposes it as a standard LeRobot Camera. One instance per camera —
instantiate three to cover overhead, left_arm, and right_arm.

Each grabbed frame carries two timestamps, both on ``time.perf_counter``:

* ``capture_perf_ts`` — when the sensor exposed the frame, derived from the
  SDK's ``TIME_REFERENCE.IMAGE`` wall-clock timestamp plus a per-frame
  wall→perf offset. Used by ``collect_data`` so dataset rows record the
  moment of capture, not the moment of decode.
* ``receive_perf_ts`` — when this process retrieved the frame.

The ZED X daemon only enumerates GMSL cameras when it starts, so a camera
plugged in after boot is invisible until the daemon restarts — see
``almond_axol.zed.restart_zed_daemon``.

Typical usage::

    from almond_axol.lerobot.camera import ZedCamera, ZedCameraConfig

    overhead  = ZedCamera(ZedCameraConfig(serial=41234567))
    left_arm  = ZedCamera(ZedCameraConfig(serial=41234568))
    right_arm = ZedCamera(ZedCameraConfig(serial=41234569))

    with overhead, left_arm, right_arm:
        frame = overhead.read()  # uint8 numpy array (600, 960, 3) RGB
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
    """LeRobot camera that captures from a locally connected ZED camera.

    Opens the camera by serial number via the ZED SDK. A background thread
    continuously calls grab() and stores the latest frame so read() and
    async_read() never block on the sensor.

    Args:
        config: Serial, resolution, fps, color mode, and warmup duration.
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
        return f"ZedCamera(serial={self.config.serial})"

    @property
    def is_connected(self) -> bool:
        return self.zed is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Enumerate locally connected mono ZED cameras."""
        return [
            {"serial": int(d.serial_number), "model": str(d.camera_model)}
            for d in sl.CameraOne.get_device_list()
        ]

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        """Open the ZED camera and start the background grab thread.

        Args:
            warmup: If True, reads frames for `config.warmup_s` seconds before
                    returning so the frame buffer is primed.

        Raises:
            ConnectionError: If the camera cannot be opened.
        """
        zed = sl.CameraOne()
        init_params = sl.InitParametersOne()
        init_params.set_from_serial_number(self.config.serial)
        resolution = self.config.resolution_name()
        if resolution is not None:
            init_params.camera_resolution = getattr(sl.RESOLUTION, resolution)
        if self.config.fps is not None:
            init_params.camera_fps = self.config.fps
        # Without async recovery, a briefly disrupted GMSL link makes
        # ``grab()`` block indefinitely until the camera recovers, which
        # silently freezes ``latest_*_perf_ts``. With async recovery
        # ``grab()`` returns CAMERA_REBOOTING quickly and the SDK reconnects
        # in the background, so our read loop can keep retrying.
        init_params.async_grab_camera_recovery = True

        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise ConnectionError(
                f"{self} failed to open camera: {err}. Is the camera connected "
                "(a camera plugged in after boot needs a zed_x_daemon restart)?"
            )

        info = zed.get_camera_information()
        opened_serial = int(info.serial_number)
        if opened_serial != self.config.serial:
            zed.close()
            raise ConnectionError(
                f"{self} requested serial {self.config.serial} but the SDK "
                f"opened serial {opened_serial}."
            )

        self.zed = zed

        params = info.camera_configuration.resolution
        live_fps = int(info.camera_configuration.fps)
        live_width = int(params.width)
        live_height = int(params.height)

        mismatches = []
        if self.config.fps is not None and live_fps != self.config.fps:
            mismatches.append(f"fps: expected {self.config.fps}, got {live_fps}")
        if self.config.width is not None and live_width != self.config.width:
            mismatches.append(f"width: expected {self.config.width}, got {live_width}")
        if self.config.height is not None and live_height != self.config.height:
            mismatches.append(
                f"height: expected {self.config.height}, got {live_height}"
            )
        if mismatches:
            zed.close()
            self.zed = None
            raise RuntimeError(
                f"{self} live camera parameters do not match config — "
                + ", ".join(mismatches)
                + ". Update ZedCameraConfig (the camera may not support the "
                "requested resolution/fps combination)."
            )

        self.fps = live_fps
        self.width = live_width
        self.height = live_height

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

        Acts as a startup canary for the capture pipeline: a healthy camera
        delivers frames within a few milliseconds of exposure, so a large or
        negative latency points at a wedged daemon or a flaky GMSL link.
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
                "skipping startup latency check.",
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
                "The capture pipeline may be stalled — check the GMSL cable "
                "and try restarting zed_x_daemon.",
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

    def _stop_read_thread(self) -> bool:
        """Stop the read loop; return True if the thread actually exited.

        When the camera is down, ``grab()`` can block in native code for a
        long time. We must never call ``zed.close()`` while ``grab()`` is in
        flight on another thread — that races inside the SDK and segfaults
        the whole process — so callers use the return value to decide whether
        closing is safe.
        """
        thread = self.thread
        if self.stop_event is not None:
            self.stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        stopped = thread is None or not thread.is_alive()
        self.thread = None
        self.stop_event = None
        with self.frame_lock:
            self.latest_frame = None
            self.latest_capture_perf_ts = None
            self.latest_receive_perf_ts = None
            self.new_frame_event.clear()
        return stopped

    def _read_loop(self) -> None:
        if self.stop_event is None or self.zed is None:
            return

        image = sl.Mat()
        failure_count = 0
        grab_failure_streak = 0
        last_grab_warning_perf = 0.0

        while not self.stop_event.is_set():
            try:
                err = self.zed.grab()
                if err != sl.ERROR_CODE.SUCCESS:
                    grab_failure_streak += 1
                    # Throttled WARN so silent freezes are visible at INFO level.
                    now = time.perf_counter()
                    if now - last_grab_warning_perf >= 1.0:
                        _logger.warning(
                            "%s grab returned %s (%d consecutive failures); "
                            "camera is recovering in the background.",
                            self,
                            err,
                            grab_failure_streak,
                        )
                        last_grab_warning_perf = now
                    # Backoff so a persistent failure doesn't pin a CPU.
                    if self.stop_event.wait(timeout=0.05):
                        return
                    continue

                if grab_failure_streak > 0:
                    _logger.info(
                        "%s grab recovered after %d failed attempts.",
                        self,
                        grab_failure_streak,
                    )
                    grab_failure_streak = 0

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
                # Recompute the wall→perf offset per frame so wall-clock step
                # adjustments (NTP) don't accumulate as silent skew.
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
        the same capture timeline.

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
        """Stop the grab thread and close the ZED camera."""
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it is already disconnected."
            )

        stopped = self._stop_read_thread()

        if self.zed is not None:
            if stopped:
                try:
                    self.zed.close()
                except Exception as exc:  # noqa: BLE001 - best-effort close
                    _logger.warning(f"{self} close failed: {exc}")
            else:
                _logger.error(
                    "%s read thread is stuck in grab() (camera down?); leaking "
                    "the SDK handle instead of closing it concurrently, which "
                    "would crash the process.",
                    self,
                )
            self.zed = None

        _logger.info(f"{self} disconnected.")


class _EyeBuffer:
    """Latest-frame slot for one eye of a stereo camera (thread-safe)."""

    def __init__(self) -> None:
        self.lock: Lock = Lock()
        self.frame: NDArray[Any] | None = None
        self.cap_ts: float | None = None
        self.recv_ts: float | None = None
        self.event: Event = Event()

    def set(self, frame: NDArray[Any], cap_ts: float, recv_ts: float) -> None:
        with self.lock:
            self.frame = frame
            self.cap_ts = cap_ts
            self.recv_ts = recv_ts
        self.event.set()

    def clear(self) -> None:
        with self.lock:
            self.frame = None
            self.cap_ts = None
            self.recv_ts = None
        self.event.clear()


class ZedStereoCamera:
    """Local stereo ZED X camera with a single shared grab.

    Opens one ``sl.Camera`` by serial number and, on every grab, retrieves
    both eyes into separate :class:`_EyeBuffer` slots. The left/right eyes are
    exposed as :class:`_StereoEyeView` objects (``left_view`` / ``right_view``)
    that present the same read API as :class:`ZedCamera`, so collect-data /
    run-policy / teleop can treat the two eyes as ordinary cameras while only
    grabbing the sensor once.

    Args:
        config: Serial, resolution, fps, color mode, and warmup duration
            (``stereo`` set).
    """

    def __init__(self, config: ZedCameraConfig) -> None:
        self.config = config
        self.zed: sl.Camera | None = None
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.fps: int = config.fps or 60
        self.width: int | None = config.width
        self.height: int | None = config.height
        self._left = _EyeBuffer()
        self._right = _EyeBuffer()
        self.left_view = _StereoEyeView(self, self._left, "left")
        self.right_view = _StereoEyeView(self, self._right, "right")

    def __str__(self) -> str:
        return f"ZedStereoCamera(serial={self.config.serial})"

    @property
    def is_connected(self) -> bool:
        return self.zed is not None

    def connect(self, warmup: bool = True) -> None:
        """Open the stereo camera and start the shared grab thread."""
        if self.is_connected:
            return
        zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.set_from_serial_number(self.config.serial)
        resolution = self.config.resolution_name()
        if resolution is not None:
            init_params.camera_resolution = getattr(sl.RESOLUTION, resolution)
        if self.config.fps is not None:
            init_params.camera_fps = self.config.fps
        # We only need the rectified images; skip depth to save GPU.
        init_params.depth_mode = sl.DEPTH_MODE.NONE
        # See ZedCamera.connect for why async recovery matters on a flaky link.
        init_params.async_grab_camera_recovery = True

        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise ConnectionError(
                f"{self} failed to open camera: {err}. Is the camera connected "
                "(a camera plugged in after boot needs a zed_x_daemon restart)?"
            )

        info = zed.get_camera_information()
        opened_serial = int(info.serial_number)
        if opened_serial != self.config.serial:
            zed.close()
            raise ConnectionError(
                f"{self} requested serial {self.config.serial} but the SDK "
                f"opened serial {opened_serial}."
            )

        self.zed = zed

        params = info.camera_configuration.resolution
        live_fps = int(info.camera_configuration.fps)
        # For a stereo sl.Camera the SDK reports per-eye resolution, which is
        # exactly what retrieve_image(LEFT/RIGHT) returns.
        live_width = int(params.width)
        live_height = int(params.height)

        # Enforce config == live camera (per eye) like the mono ZedCamera, so
        # the dataset features built from config before connect stay valid.
        mismatches = []
        if self.config.fps is not None and live_fps != self.config.fps:
            mismatches.append(f"fps: expected {self.config.fps}, got {live_fps}")
        if self.config.width is not None and live_width != self.config.width:
            mismatches.append(f"width: expected {self.config.width}, got {live_width}")
        if self.config.height is not None and live_height != self.config.height:
            mismatches.append(
                f"height: expected {self.config.height}, got {live_height}"
            )
        if mismatches:
            zed.close()
            self.zed = None
            raise RuntimeError(
                f"{self} live camera parameters do not match config (per eye) — "
                + ", ".join(mismatches)
                + ". Update ZedCameraConfig (the camera may not support the "
                "requested resolution/fps combination)."
            )

        self.fps = live_fps
        self.width = live_width
        self.height = live_height

        self._start_read_thread()

        if warmup:
            start = time.time()
            while time.time() - start < self.config.warmup_s:
                if self._left.event.wait(timeout=self.config.warmup_s):
                    break

        _logger.info(
            f"{self} connected ({self.width}x{self.height} @ {self.fps}fps, stereo)."
        )

    def _start_read_thread(self) -> None:
        self._stop_read_thread()
        self.stop_event = Event()
        self.thread = Thread(
            target=self._read_loop, name=f"{self}_read_loop", daemon=True
        )
        self.thread.start()

    def _stop_read_thread(self) -> bool:
        """Stop the read loop; return True if the thread actually exited.

        See :meth:`ZedCamera._stop_read_thread` — closing the SDK handle
        while ``grab()`` is in flight on another thread segfaults.
        """
        thread = self.thread
        if self.stop_event is not None:
            self.stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        stopped = thread is None or not thread.is_alive()
        self.thread = None
        self.stop_event = None
        self._left.clear()
        self._right.clear()
        return stopped

    def _read_loop(self) -> None:
        if self.stop_event is None or self.zed is None:
            return

        left_mat = sl.Mat()
        right_mat = sl.Mat()
        grab_failure_streak = 0
        last_grab_warning_perf = 0.0

        while not self.stop_event.is_set():
            try:
                err = self.zed.grab()
                if err != sl.ERROR_CODE.SUCCESS:
                    grab_failure_streak += 1
                    now = time.perf_counter()
                    if now - last_grab_warning_perf >= 1.0:
                        _logger.warning(
                            "%s grab returned %s (%d consecutive failures); "
                            "camera is recovering in the background.",
                            self,
                            err,
                            grab_failure_streak,
                        )
                        last_grab_warning_perf = now
                    if self.stop_event.wait(timeout=0.05):
                        return
                    continue

                if grab_failure_streak > 0:
                    _logger.info(
                        "%s grab recovered after %d failed attempts.",
                        self,
                        grab_failure_streak,
                    )
                    grab_failure_streak = 0

                self.zed.retrieve_image(left_mat, sl.VIEW.LEFT)
                self.zed.retrieve_image(right_mat, sl.VIEW.RIGHT)

                # One grab → both eyes share the same capture instant.
                cap_wall = (
                    self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
                    * 1e-9
                )
                recv_wall = time.time()
                recv_perf = time.perf_counter()
                cap_perf = recv_perf - (recv_wall - cap_wall)

                for mat, buf in ((left_mat, self._left), (right_mat, self._right)):
                    raw = mat.get_data()  # BGRA uint8 (height, width, 4)
                    if self.config.color_mode == ColorMode.RGB:
                        frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
                    else:
                        frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                    buf.set(frame, cap_perf, recv_perf)

            except DeviceNotConnectedError:
                break
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                _logger.warning(f"{self} read loop error: {exc}")
                if self.stop_event.wait(timeout=0.05):
                    return

    def disconnect(self) -> None:
        """Stop the grab thread and close the stereo camera."""
        if not self.is_connected and self.thread is None:
            return
        stopped = self._stop_read_thread()
        if self.zed is not None:
            if stopped:
                try:
                    self.zed.close()
                except Exception as exc:  # noqa: BLE001 - best-effort close
                    _logger.warning(f"{self} close failed: {exc}")
            else:
                _logger.error(
                    "%s read thread is stuck in grab() (camera down?); leaking "
                    "the SDK handle instead of closing it concurrently, which "
                    "would crash the process.",
                    self,
                )
            self.zed = None
        _logger.info(f"{self} disconnected.")


class _StereoEyeView:
    """One eye of a :class:`ZedStereoCamera`, presented as a camera.

    Implements the subset of the :class:`ZedCamera` read API that the Axol
    robot, collect-data, and teleop use (``read_latest`` / ``read_latest_with_ts``
    / ``read_at_or_after`` plus ``fps`` / ``width`` / ``height``). ``connect`` /
    ``disconnect`` defer to the shared parent so the camera is opened and closed
    exactly once regardless of iteration order.
    """

    def __init__(self, parent: ZedStereoCamera, buf: _EyeBuffer, eye: str) -> None:
        self._parent = parent
        self._buf = buf
        self._eye = eye

    def __str__(self) -> str:
        c = self._parent.config
        return f"ZedStereoEye({self._eye}@serial={c.serial})"

    @property
    def fps(self) -> int:
        return self._parent.fps

    @property
    def width(self) -> int | None:
        return self._parent.width

    @property
    def height(self) -> int | None:
        return self._parent.height

    @property
    def is_connected(self) -> bool:
        return self._parent.is_connected

    def connect(self, warmup: bool = True) -> None:
        if not self._parent.is_connected:
            self._parent.connect(warmup=warmup)

    def disconnect(self) -> None:
        if self._parent.is_connected:
            self._parent.disconnect()

    def _running(self) -> bool:
        return self._parent.thread is not None and self._parent.thread.is_alive()

    def read_latest_with_ts(self) -> tuple[NDArray[Any], float, float]:
        if not self._running():
            raise RuntimeError(f"{self} read thread is not running.")
        with self._buf.lock:
            frame = self._buf.frame
            cap_ts = self._buf.cap_ts
            recv_ts = self._buf.recv_ts
        if frame is None or cap_ts is None or recv_ts is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")
        return frame, cap_ts, recv_ts

    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        frame, _cap_ts, recv_ts = self.read_latest_with_ts()
        age_ms = (time.perf_counter() - recv_ts) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f}ms (max {max_age_ms}ms)."
            )
        return frame

    def read_at_or_after(
        self,
        target_capture_perf_ts: float,
        timeout_ms: float = 500,
    ) -> tuple[NDArray[Any], float, float]:
        if not self._running():
            raise RuntimeError(f"{self} read thread is not running.")
        deadline = time.perf_counter() + timeout_ms / 1000.0
        while True:
            self._buf.event.clear()
            with self._buf.lock:
                frame = self._buf.frame
                cap_ts = self._buf.cap_ts
                recv_ts = self._buf.recv_ts
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
                    f"after {timeout_ms:.1f}ms (latest cap_ts={cap_ts!r})."
                )
            self._buf.event.wait(timeout=remaining)
