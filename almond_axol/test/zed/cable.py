"""
Test script: verify a ZED camera cable by capturing and validating frames.

Run this on a ZED box with exactly one ZED-X One camera connected. The test
restarts the ZED X daemon, opens the camera, and captures frames across a
5-second window, checking that each frame is a valid image. Any failure raises
an exception; if it returns cleanly the cable is good.

Usage:
    python -m almond_axol.test.zed.cable
    python -m almond_axol.test.zed.cable --output logs/cable_frame.png
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

# A live frame from a working sensor varies across pixels and sits in a sane
# brightness band. A disconnected/garbled cable typically yields a flat frame
# (all black, all white, or a constant value), which these bounds reject.
_MIN_STD = 1.0
_MIN_MEAN = 1.0
_MAX_MEAN = 254.0

# The ZED X daemon must be restarted before opening the camera; give it time to
# enumerate the sensor on the GMSL link before we connect.
_DAEMON_RESTART_WAIT_S = 5.0
# How long to keep capturing and validating frames.
_CAPTURE_DURATION_S = 5.0


class CableTestError(RuntimeError):
    """Raised when the ZED cable test fails at any step."""


def _restart_zed_daemon() -> None:
    """Restart the ZED X daemon and wait for it to come up.

    Raises:
        CableTestError: If the ``systemctl restart`` command fails.
    """
    _logger.info("Restarting zed_x_daemon...")
    result = subprocess.run(
        ["sudo", "systemctl", "restart", "zed_x_daemon"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise CableTestError(
            f"Failed to restart zed_x_daemon: {result.stderr.strip() or result.stdout.strip()}"
        )
    _logger.info("Waiting %.0fs for daemon to settle...", _DAEMON_RESTART_WAIT_S)
    time.sleep(_DAEMON_RESTART_WAIT_S)


def _validate_frame(bgr: np.ndarray) -> None:
    """Raise :class:`CableTestError` unless ``bgr`` looks like a real frame.

    Checks shape, dtype, that the image is not empty, and that it carries actual
    image content (non-trivial variation and a plausible brightness level).
    """
    if bgr is None or bgr.size == 0:
        raise CableTestError("Retrieved frame is empty.")
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise CableTestError(f"Frame has unexpected shape {bgr.shape}; expected HxWx3.")
    if bgr.dtype != np.uint8:
        raise CableTestError(f"Frame has unexpected dtype {bgr.dtype}; expected uint8.")

    height, width = bgr.shape[:2]
    if height <= 0 or width <= 0:
        raise CableTestError(f"Frame has invalid dimensions {width}x{height}.")

    std = float(bgr.std())
    mean = float(bgr.mean())

    if std < _MIN_STD:
        raise CableTestError(
            f"Frame is nearly uniform (std={std:.2f} < {_MIN_STD}); "
            "camera may be disconnected or the lens is covered."
        )
    if not (_MIN_MEAN <= mean <= _MAX_MEAN):
        raise CableTestError(
            f"Frame brightness out of range (mean={mean:.2f}); "
            "expected a non-black, non-saturated image."
        )


def run(output: str | None = None) -> None:
    """Restart the daemon, open the camera, and validate frames over 5 seconds.

    Args:
        output: Optional path to save the last captured frame as PNG for inspection.

    Raises:
        CableTestError: If the daemon cannot be restarted, the camera cannot be
            opened, no frames can be grabbed, or any captured frame fails validation.
    """
    import pyzed.sl as sl

    _restart_zed_daemon()

    zed = sl.CameraOne()
    init_params = sl.InitParametersOne()

    _logger.info("Opening connected ZED camera...")
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise CableTestError(f"Failed to open camera: {err}")

    last_bgr: np.ndarray | None = None
    try:
        info = zed.get_camera_information()
        serial = int(info.serial_number)
        res = info.camera_configuration.resolution
        fps = int(info.camera_configuration.fps)
        _logger.info(
            "Camera opened: serial=%d  %dx%d @ %dfps",
            serial,
            res.width,
            res.height,
            fps,
        )

        image = sl.Mat()
        _logger.info(
            "Capturing and validating frames for %.0fs...", _CAPTURE_DURATION_S
        )
        valid_frames = 0
        deadline = time.monotonic() + _CAPTURE_DURATION_S
        while time.monotonic() < deadline:
            if zed.grab() != sl.ERROR_CODE.SUCCESS:
                continue
            if zed.retrieve_image(image) != sl.ERROR_CODE.SUCCESS:
                raise CableTestError("Failed to retrieve image from camera.")

            raw = image.get_data()  # BGRA uint8
            bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
            _validate_frame(bgr)
            last_bgr = bgr
            valid_frames += 1

        if valid_frames == 0:
            raise CableTestError(
                f"No frames grabbed in {_CAPTURE_DURATION_S:.0f}s; "
                "camera may be disconnected."
            )
        _logger.info(
            "Validated %d frames over %.0fs.", valid_frames, _CAPTURE_DURATION_S
        )

        if output and last_bgr is not None:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output, last_bgr)
            _logger.info("Saved frame to %s", output)
    finally:
        zed.close()

    _logger.info(
        "Cable test PASSED: captured valid frames across %.0fs.", _CAPTURE_DURATION_S
    )


def main() -> None:
    """Parse CLI arguments and run the ZED cable test."""
    parser = argparse.ArgumentParser(
        description="Test a ZED camera cable by validating frames across 5 seconds."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the last captured frame as PNG (e.g. logs/cable_frame.png).",
    )
    args = parser.parse_args()
    run(output=args.output)


if __name__ == "__main__":
    main()
