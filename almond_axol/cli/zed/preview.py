"""
Preview one ZED camera via MJPEG in a browser.

Run on the ZED box:
    python almond_axol/cli/zed/preview.py

Then open in your browser:
    http://<zed-box-ip>:8080

Press Ctrl+C to stop.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import pyzed.sl as sl

SERIAL = 305042468  # left_arm
PORT = 8080

_latest_frame: bytes = b""
_lock = threading.Lock()


def _grab_loop() -> None:
    global _latest_frame

    zed = sl.CameraOne()
    init = sl.InitParametersOne()
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.camera_fps = 30
    init.input.set_from_serial_number(SERIAL)

    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open camera (serial {SERIAL}): {err}")
        return

    print(f"Camera opened. View at http://0.0.0.0:{PORT}")
    image = sl.Mat()

    while True:
        if zed.grab() != sl.ERROR_CODE.SUCCESS:
            time.sleep(0.01)
            continue
        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame = image.get_data()[:, :, :3]
        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with _lock:
            _latest_frame = jpg.tobytes()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_: object) -> None:
        pass  # suppress per-request logs

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body style='background:#000;margin:0'>"
                b"<img src='/stream' style='width:100%'/>"
                b"</body></html>"
            )
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            try:
                while True:
                    with _lock:
                        frame = _latest_frame
                    if frame:
                        self.wfile.write(
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                        )
                    time.sleep(1 / 30)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


def main() -> None:
    t = threading.Thread(target=_grab_loop, daemon=True)
    t.start()

    server = HTTPServer(("0.0.0.0", PORT), _Handler)
    print(f"Serving at http://0.0.0.0:{PORT} — press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
