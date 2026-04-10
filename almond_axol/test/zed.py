"""
Test script: connect to a ZED stream and save one frame as a PNG.

Usage:
    python -m almond_axol.test.zed --host 192.168.10.1 --port 30000
    python -m almond_axol.test.zed --host 192.168.10.1 --port 30000 --output logs/frame.png
    python -m almond_axol.test.zed --host 192.168.10.1 --port 30000 --setup-ip eth0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

_RECEIVER_IP = "192.168.10.2/24"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture one frame from a ZED stream and save as PNG."
    )
    parser.add_argument(
        "--host",
        default="192.168.10.1",
        help="IP address of the ZedStreamer host (default: 192.168.10.1).",
    )
    parser.add_argument(
        "--port", type=int, default=30000, help="Streaming port (default: 30000)."
    )
    parser.add_argument(
        "--output",
        default="logs/zed_frame.png",
        help="Output PNG file path (default: logs/zed_frame.png).",
    )
    parser.add_argument(
        "--setup-ip",
        metavar="IFACE",
        default=None,
        help=f"Assign the receiver IP ({_RECEIVER_IP}) to IFACE before connecting (e.g. eth0). Requires sudo.",
    )
    args = parser.parse_args()

    if args.setup_ip:
        from ..shared import setup_link_ip

        setup_link_ip(args.setup_ip, _RECEIVER_IP)

    import pyzed.sl as sl

    zed = sl.CameraOne()
    init_params = sl.InitParametersOne()
    init_params.set_from_stream(args.host, args.port)

    _logger.info("Connecting to %s:%d ...", args.host, args.port)
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"Failed to open stream at {args.host}:{args.port}: {err}")

    info = zed.get_camera_information()
    res = info.camera_configuration.resolution
    fps = int(info.camera_configuration.fps)
    _logger.info("Connected: %dx%d @ %dfps", res.width, res.height, fps)

    image = sl.Mat()
    _logger.info("Grabbing frame...")
    for _ in range(30):  # drain a few frames so the buffer is fresh
        err = zed.grab()
        if err == sl.ERROR_CODE.SUCCESS:
            break
    else:
        zed.close()
        raise SystemExit("Failed to grab a frame after 30 attempts.")

    zed.retrieve_image(image)
    raw = image.get_data()  # BGRA uint8

    bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, bgr)
    _logger.info("Saved frame to %s", args.output)

    zed.close()


if __name__ == "__main__":
    main()
