"""
ZED-X One camera streamer for the Axol robot.

ZedStreamer opens up to three ZED-X One cameras (overhead, left_arm, right_arm)
by serial number and streams each over HEVC on the local network using the ZED
SDK's built-in streaming API.

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

from .config import ZedConfig

_logger = logging.getLogger(__name__)


@dataclass
class _CameraState:
    name: str
    serial: int
    port: int
    zed: sl.CameraOne
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
        all_specs = [
            ("overhead", cfg.overhead_serial, cfg.overhead_port),
            ("left_arm", cfg.left_arm_serial, cfg.left_arm_port),
            ("right_arm", cfg.right_arm_serial, cfg.right_arm_port),
        ]
        specs = [
            (name, serial, port)
            for name, serial, port in all_specs
            if serial is not None
        ]

        loop = asyncio.get_running_loop()
        states = await asyncio.gather(
            *[
                loop.run_in_executor(None, self._open_camera, name, serial, port)
                for name, serial, port in specs
            ]
        )

        self._cameras = [s for s in states if s is not None]
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

    def _open_camera(self, name: str, serial: int, port: int) -> _CameraState | None:
        zed = sl.CameraOne()

        init_params = sl.InitParametersOne()
        init_params.camera_resolution = self._config.resolution
        init_params.camera_fps = self._config.fps
        init_params.input.set_from_serial_number(serial)

        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            _logger.error("Failed to open %s (serial %d): %s", name, serial, err)
            return None

        stream_params = sl.StreamingParameters()
        stream_params.codec = sl.STREAMING_CODEC.H265
        stream_params.bitrate = self._config.bitrate
        stream_params.port = port

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
            self._config.bitrate,
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
