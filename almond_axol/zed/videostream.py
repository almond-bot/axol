"""
MJPEG HTTP video stream server for ZED-X One cameras.

Serves each configured camera as a live MJPEG stream over HTTP so any
browser, VLC, or OpenCV client on the workstation can view it without
needing the ZED SDK installed.

Typical usage::

    from almond_axol.zed import ZedConfig, ZedVideoStream

    async with ZedVideoStream(ZedConfig(
        overhead_serial=12345678,
        left_arm_serial=12345679,
    )):
        await asyncio.sleep(float("inf"))

Then open in a browser on the workstation::

    http://<robot-ip>:8080/              — index page with all cameras
    http://<robot-ip>:8080/overhead      — live MJPEG feed
    http://<robot-ip>:8080/left_arm      — live MJPEG feed
    http://<robot-ip>:8080/right_arm     — live MJPEG feed

Or with VLC::

    vlc http://<robot-ip>:8080/overhead
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

import cv2
import pyzed.sl as sl

from .config import ZedConfig

_logger = logging.getLogger(__name__)

_BOUNDARY = b"frame"
_JPEG_QUALITY = 80


class _CameraFeed:
    """Owns one ZED camera and keeps the latest JPEG frame in memory."""

    def __init__(self, name: str, serial: int, config: ZedConfig) -> None:
        self.name = name
        self.serial = serial
        self._config = config
        self._latest: bytes = b""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._zed: sl.CameraOne | None = None

    def start(self) -> bool:
        zed = sl.CameraOne()
        init = sl.InitParametersOne()
        init.camera_resolution = self._config.resolution
        init.camera_fps = self._config.fps
        init.input.set_from_serial_number(self.serial)

        err = zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            _logger.error(
                "Failed to open %s (serial %d): %s", self.name, self.serial, err
            )
            return False

        self._zed = zed
        self._thread = threading.Thread(
            target=self._grab_loop,
            name=f"zed-mjpeg-{self.name}",
            daemon=True,
        )
        self._thread.start()
        _logger.info("MJPEG feed started for %s (serial %d)", self.name, self.serial)
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._zed is not None:
            try:
                self._zed.close()
            except Exception as exc:
                _logger.warning("Error closing %s: %s", self.name, exc)
        _logger.info("MJPEG feed stopped for %s", self.name)

    def latest_jpeg(self) -> bytes:
        with self._lock:
            return self._latest

    def _grab_loop(self) -> None:
        assert self._zed is not None
        image = sl.Mat()
        fps_display = 0.0
        frame_count = 0
        t0 = time.monotonic()
        while not self._stop.is_set():
            if self._zed.grab() != sl.ERROR_CODE.SUCCESS:
                time.sleep(0.01)
                continue
            self._zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()[:, :, :3].copy()

            frame_count += 1
            elapsed = time.monotonic() - t0
            if elapsed >= 0.5:
                fps_display = frame_count / elapsed
                frame_count = 0
                t0 = time.monotonic()

            cv2.putText(
                frame,
                f"{self.name}  {fps_display:.1f} fps",
                (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{self.name}  {fps_display:.1f} fps",
                (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            ok, jpg = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]
            )
            if ok:
                with self._lock:
                    self._latest = jpg.tobytes()


def _make_handler(feeds: dict[str, _CameraFeed]) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            pass

        def do_GET(self) -> None:
            path = self.path.lstrip("/")

            if path == "":
                self._serve_index()
            elif path in feeds:
                self._serve_stream(feeds[path])
            else:
                self.send_error(404)

        def _serve_index(self) -> None:
            items = "".join(
                f"<li><a href='/{n}'>{n}</a> — "
                f"<img src='/{n}' style='max-height:240px;vertical-align:middle'/></li>"
                for n in feeds
            )
            body = (
                b"<html><body style='background:#111;color:#eee;font-family:sans-serif'>"
                b"<h2>ZED Camera Streams</h2><ul>"
                + items.encode()
                + b"</ul></body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)

        def _serve_stream(self, feed: _CameraFeed) -> None:
            self.send_response(200)
            self.send_header(
                "Content-Type",
                f"multipart/x-mixed-replace; boundary={_BOUNDARY.decode()}",
            )
            self.end_headers()
            try:
                while True:
                    frame = feed.latest_jpeg()
                    if frame:
                        self.wfile.write(
                            b"--" + _BOUNDARY + b"\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                        )
                    time.sleep(1 / 30)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return _Handler


class ZedVideoStream:
    """Serves ZED-X One cameras as MJPEG over HTTP.

    Each camera is accessible at ``http://<host>:<port>/<camera_name>``.
    An index page at ``http://<host>:<port>/`` shows all feeds.

    Args:
        config: Serial numbers, resolution, and fps for all cameras.
        host:   Address to listen on (default ``"0.0.0.0"``).
        port:   HTTP port (default ``8080``).
    """

    def __init__(
        self, config: ZedConfig, host: str = "0.0.0.0", port: int = 8080
    ) -> None:
        self._config = config
        self._host = host
        self._port = port
        self._feeds: dict[str, _CameraFeed] = {}
        self._server: HTTPServer | None = None
        self._server_thread: threading.Thread | None = None

    async def enable(self) -> None:
        """Open cameras and start the HTTP server."""
        if self._feeds:
            return

        cfg = self._config
        specs = [
            ("overhead", cfg.overhead_serial),
            ("left_arm", cfg.left_arm_serial),
            ("right_arm", cfg.right_arm_serial),
        ]

        loop = asyncio.get_running_loop()
        results = []
        for name, serial in specs:
            if serial is not None:
                result = await loop.run_in_executor(
                    None, self._start_feed, name, serial
                )
                results.append(result)
                await asyncio.sleep(2.0)

        self._feeds = {name: feed for name, feed in results if feed is not None}

        if not self._feeds:
            _logger.error("No cameras opened — HTTP server not started")
            return

        handler = _make_handler(self._feeds)
        self._server = ThreadingHTTPServer((self._host, self._port), handler)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="zed-mjpeg-http",
            daemon=True,
        )
        self._server_thread.start()
        _logger.info(
            "ZedVideoStream serving %d camera(s) at http://%s:%d/",
            len(self._feeds),
            self._host,
            self._port,
        )

    async def disable(self) -> None:
        """Stop the HTTP server and close all cameras."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None

        loop = asyncio.get_running_loop()
        await asyncio.gather(
            *[loop.run_in_executor(None, feed.stop) for feed in self._feeds.values()]
        )
        self._feeds = {}
        _logger.info("ZedVideoStream disabled")

    async def __aenter__(self) -> ZedVideoStream:
        await self.enable()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disable()

    def _start_feed(self, name: str, serial: int) -> tuple[str, _CameraFeed | None]:
        feed = _CameraFeed(name, serial, self._config)
        return (name, feed) if feed.start() else (name, None)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stream ZED cameras as MJPEG over HTTP."
    )
    parser.add_argument("--overhead", type=int, default=None, metavar="SERIAL")
    parser.add_argument("--left-arm", type=int, default=None, metavar="SERIAL")
    parser.add_argument("--right-arm", type=int, default=None, metavar="SERIAL")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    if not any([args.overhead, args.left_arm, args.right_arm]):
        parser.error("provide at least one of --overhead, --left-arm, --right-arm")

    import pyzed.sl as sl

    logging.basicConfig(level=logging.INFO)
    config = ZedConfig(
        overhead_serial=args.overhead,
        left_arm_serial=args.left_arm,
        right_arm_serial=args.right_arm,
        resolution=sl.RESOLUTION.HD1080,
        fps=args.fps,
    )

    async def _main() -> None:
        async with ZedVideoStream(config, port=args.port):
            print(f"Streaming at http://0.0.0.0:{args.port}/ — Ctrl+C to stop")
            await asyncio.sleep(float("inf"))

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass
