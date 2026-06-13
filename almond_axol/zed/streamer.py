"""
ZED camera streamer for the Axol robot.

ZedStreamer opens up to three cameras (overhead, left_arm, right_arm) by serial
number and streams each over HEVC on the local network using the ZED SDK's
built-in streaming API. The wrist cameras are mono ZED-X One (``sl.CameraOne``);
the overhead may optionally be a stereo ZED X (``sl.Camera``, both eyes on one
stream) via ``ZedConfig.overhead_stereo``.

Typical usage::

    from almond_axol.zed import ZedConfig, ZedStreamer

    async with ZedStreamer(ZedConfig(
        overhead_serial=12345678,
        left_arm_serial=12345679,
        right_arm_serial=12345680,
    )):
        await asyncio.sleep(float("inf"))

Receivers can connect with::

    init = sl.InitParameters()
    init.set_from_stream("host_ip", 30000)  # or 30002 / 30004
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass

import pyzed.sl as sl

from .config import ZedConfig, auto_bitrate

_logger = logging.getLogger(__name__)


@dataclass
class _CameraState:
    """Per-camera runtime state: SDK handle, grab thread, and stop signal."""

    name: str
    serial: int
    port: int
    zed: sl.CameraOne | sl.Camera
    stop_event: threading.Event
    thread: threading.Thread | None = None


class ZedStreamer:
    """Streams ZED-X One cameras over the local network using HEVC.

    Each camera runs in a background grab thread that drives the encoder.
    Use as an async context manager or call ``enable()`` / ``disable()`` directly.

    Args:
        config: Serial numbers, ports, resolution, fps, and bitrate for all cameras.
    """

    def __init__(self, config: ZedConfig) -> None:
        """Construct the streamer.

        Cameras are not opened until :meth:`enable` (or ``async with``) is called.

        Args:
            config: Serial numbers, ports, resolution, fps, and bitrate for all cameras.
        """
        self._config = config
        self._cameras: list[_CameraState] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def enable(self) -> None:
        """Open all cameras and start streaming."""
        if self._cameras:
            return

        cfg = self._config
        # The overhead may be a stereo ZED X (sl.Camera); the wrist cameras are
        # always mono ZED-X One (sl.CameraOne).
        all_specs = [
            ("overhead", cfg.overhead_serial, cfg.overhead_port, cfg.overhead_stereo),
            ("left_arm", cfg.left_arm_serial, cfg.left_arm_port, False),
            ("right_arm", cfg.right_arm_serial, cfg.right_arm_port, False),
        ]
        specs = [
            (name, serial, port, stereo)
            for name, serial, port, stereo in all_specs
            if serial is not None
        ]

        # Stereo (ZED X) and mono (ZED-X One) cameras live in separate device
        # lists, so union both so either kind of serial validates.
        mono_devices = sl.CameraOne.get_device_list()
        stereo_devices = sl.Camera.get_device_list()
        available_serials = {int(d.serial_number) for d in mono_devices} | {
            int(d.serial_number) for d in stereo_devices
        }
        _logger.info(
            "Detected %d mono + %d stereo ZED camera(s): %s",
            len(mono_devices),
            len(stereo_devices),
            ", ".join(
                str(int(d.serial_number)) for d in (*mono_devices, *stereo_devices)
            )
            or "<none>",
        )

        failures: list[str] = []
        resolved_specs: list[tuple[str, int, int, bool]] = []
        for name, serial, port, stereo in specs:
            if int(serial) not in available_serials:
                _logger.error(
                    "Requested %s serial %d not found in device list.", name, serial
                )
                failures.append(f"{name} (serial {serial}): not connected")
                continue
            resolved_specs.append((name, serial, port, stereo))

        # Open cameras sequentially: the ZED SDK's open() + enable_streaming()
        # path touches shared NVENC state on Jetson and isn't safe to call
        # concurrently across camera instances.
        loop = asyncio.get_running_loop()
        for name, serial, port, stereo in resolved_specs:
            state = await loop.run_in_executor(
                None, self._open_camera, name, serial, port, stereo
            )
            if state is None:
                failures.append(f"{name} (serial {serial}): failed to open")
                continue
            self._cameras.append(state)

        # A requested camera that won't open means the receiving host would wait
        # forever for a stream port that never opens. Fail loudly instead so the
        # caller (and the control-panel UI) sees the error immediately rather
        # than a stream that's silently down.
        if failures:
            await self.disable()
            raise RuntimeError(
                "could not start all requested ZED cameras: " + "; ".join(failures)
            )

        _logger.info(
            "ZedStreamer enabled (%d/%d cameras)", len(self._cameras), len(specs)
        )

    async def disable(self) -> None:
        """Stop streaming and close all cameras."""
        cameras, self._cameras = self._cameras, []
        loop = asyncio.get_running_loop()
        await asyncio.gather(
            *[
                loop.run_in_executor(None, self._close_camera, state)
                for state in cameras
            ]
        )
        _logger.info("ZedStreamer disabled")

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> ZedStreamer:
        await self.enable()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disable()

    # ------------------------------------------------------------------
    # Internal (runs in thread-pool executor)
    # ------------------------------------------------------------------

    def _open_camera(
        self, name: str, serial: int, port: int, stereo: bool = False
    ) -> _CameraState | None:
        # A stereo ZED X uses the full sl.Camera API; the mono ZED-X One uses
        # the lighter sl.CameraOne API. Both expose open/enable_streaming/grab.
        if stereo:
            zed = sl.Camera()
            init_params = sl.InitParameters()
            # We only stream images, so skip depth to save Jetson compute.
            init_params.depth_mode = sl.DEPTH_MODE.NONE
        else:
            zed = sl.CameraOne()
            init_params = sl.InitParametersOne()
        init_params.camera_resolution = self._config.resolution
        init_params.camera_fps = self._config.fps
        init_params.set_from_serial_number(serial)

        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            _logger.error("Failed to open %s (serial %d): %s", name, serial, err)
            return None

        opened_serial = int(zed.get_camera_information().serial_number)
        if opened_serial != serial:
            _logger.warning(
                "%s: requested serial %d but SDK opened serial %d",
                name,
                serial,
                opened_serial,
            )
            zed.close()
            return None

        # An explicit config bitrate applies to every camera; otherwise pick a
        # recommended bitrate from the resolution (the stereo overhead carries
        # both eyes side-by-side, so it gets a higher one).
        bitrate = self._config.bitrate
        if bitrate is None:
            bitrate = auto_bitrate(self._config.resolution, stereo=stereo)

        stream_params = sl.StreamingParameters()
        stream_params.codec = sl.STREAMING_CODEC.H265
        stream_params.bitrate = bitrate
        stream_params.port = port
        stream_params.target_framerate = self._config.fps

        err = zed.enable_streaming(stream_params)
        if err != sl.ERROR_CODE.SUCCESS:
            _logger.error(
                "Failed to start streaming %s (serial %d): %s", name, serial, err
            )
            zed.close()
            return None

        stop_event = threading.Event()
        state = _CameraState(
            name=name, serial=serial, port=port, zed=zed, stop_event=stop_event
        )

        thread = threading.Thread(
            target=self._grab_loop,
            args=(state,),
            name=f"zed-grab-{name}",
            daemon=True,
        )
        thread.start()
        state.thread = thread

        _logger.info(
            "Streaming %s (serial %d) on port %d at %s %dfps %dkbps",
            name,
            serial,
            port,
            self._config.resolution,
            self._config.fps,
            bitrate,
        )
        return state

    def _close_camera(self, state: _CameraState) -> None:
        state.stop_event.set()
        if state.thread is not None:
            state.thread.join(timeout=3.0)

        try:
            state.zed.disable_streaming()
        except Exception as exc:
            _logger.warning("Error disabling streaming for %s: %s", state.name, exc)

        try:
            state.zed.close()
        except Exception as exc:
            _logger.warning("Error closing camera %s: %s", state.name, exc)

        _logger.info("Closed %s (serial %d)", state.name, state.serial)

    @staticmethod
    def _grab_loop(state: _CameraState) -> None:
        while not state.stop_event.is_set():
            state.zed.grab()
