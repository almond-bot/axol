"""
Display a live MJPEG stream from a ZED video server.

Usage:
    python -m almond_axol.test.getvideo
    python -m almond_axol.test.getvideo --host 192.168.50.28 --camera right_arm

The ZED box must be running:
    python -m almond_axol.zed.videostream --right-arm <serial>
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time

if not os.environ.get("DISPLAY"):
    os.environ["DISPLAY"] = ":0"

import cv2

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

_latest_frame = None
_frame_lock = threading.Lock()


def _capture_loop(url: str) -> None:
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while not cap.isOpened():
        _logger.info("Waiting for stream at %s ...", url)
        time.sleep(0.5)
        cap.open(url)

    _logger.info("Stream opened: %s", url)

    global _latest_frame
    while True:
        ok, frame = cap.read()
        if not ok:
            _logger.warning("Stream read failed, retrying...")
            time.sleep(0.5)
            cap.open(url)
            continue
        with _frame_lock:
            _latest_frame = frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display a live MJPEG stream from a ZED video server."
    )
    parser.add_argument(
        "--host",
        default="192.168.50.28",
        help="IP of the ZED box (default: 192.168.50.28).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP port of the MJPEG server (default: 8080).",
    )
    parser.add_argument(
        "--camera",
        default="right_arm",
        help="Camera name to stream (default: right_arm).",
    )
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/{args.camera}"
    _logger.info("Connecting to %s", url)

    t = threading.Thread(target=_capture_loop, args=(url,), daemon=True)
    t.start()

    while True:
        with _frame_lock:
            frame = _latest_frame

        if frame is not None:
            cv2.imshow(f"ZED — {args.camera}", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
