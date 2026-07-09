"""
Test script: verify a ZED camera cable by capturing and validating frames.

Run this on a ZED box with exactly one ZED camera connected — either a mono
ZED-X One or a stereo ZED X. The test restarts the ZED X daemon, opens the
camera, and captures frames across a 5-second window, checking that each frame
is a valid image. Any failure raises an exception; if it returns cleanly the
cable is good.

The camera kind (mono vs stereo) is auto-detected by default; pass ``--kind``
to force one if detection is ambiguous.

Usage:
    axol diag.zed-cable
    axol diag.zed-cable --kind stereo
    uv run -m almond_axol.diagnostics.zed.cable
    uv run -m almond_axol.diagnostics.zed.cable --output logs/cable_frame.png
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from almond_axol.zed.daemon import restart_zed_daemon

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

# A live frame from a working sensor varies across pixels and sits in a sane
# brightness band. A disconnected/garbled cable typically yields a flat frame
# (all black, all white, or a constant value), which these bounds reject.
_MIN_STD = 1.0
_MIN_MEAN = 1.0
_MAX_MEAN = 254.0

# How long to keep capturing and validating frames.
_CAPTURE_DURATION_S = 5.0
# Minimum fraction of the expected fps * duration frames we tolerate before
# treating the capture as a dropped-frame failure. Per-frame validation (full
# std/mean scans) costs enough that even a healthy link only sustains ~24 of the
# nominal 30fps, so leave headroom below that and still catch a serious drop.
_MIN_FRAME_FRACTION = 0.6
# A flaky GMSL link drops frames in bursts. Too many grab() failures in a row
# points at a marginal cable even if the overall count looks acceptable.
_MAX_CONSECUTIVE_GRAB_FAILURES = 5
# A stuck link can keep handing back the same buffer. Allow this many repeats of
# a frame back-to-back before declaring the stream frozen rather than live.
_MAX_DUPLICATE_FRAMES = 3


class CableTestError(RuntimeError):
    """Raised when the ZED cable test fails at any step."""


def _detect_kind() -> str:
    """Auto-detect whether the single connected camera is mono or stereo.

    Enumerates locally connected ZED cameras (in a fresh subprocess) and returns
    the kind of the one that is present. The test expects exactly one camera, so
    zero or multiple cameras is treated as a setup error the operator must
    resolve (or disambiguate with ``--kind``).

    Raises:
        CableTestError: If no camera, or more than one, is detected.
    """
    from almond_axol.zed.devices import list_zed_devices

    try:
        devices = list_zed_devices()
    except ImportError as exc:
        raise CableTestError("pyzed is not installed; run `axol zed.install`.") from exc
    except Exception as exc:  # noqa: BLE001 - detection failure is a test failure
        raise CableTestError(f"Failed to enumerate ZED cameras: {exc}") from exc

    if not devices:
        raise CableTestError(
            "No ZED camera detected (a camera plugged in after boot needs a "
            "zed_x_daemon restart)."
        )
    if len(devices) > 1:
        summary = ", ".join(f"{d['serial']} ({d['kind']})" for d in devices)
        raise CableTestError(
            f"Detected {len(devices)} ZED cameras [{summary}]; connect exactly "
            "one, or pass --kind to choose which pipeline to test."
        )
    return devices[0]["kind"]


def _open_camera(sl, kind: str):  # type: ignore[no-untyped-def]
    """Open the connected camera with the API for ``kind`` and return the handle.

    Mono ZED-X One cameras use ``sl.CameraOne`` and stereo ZED X cameras
    ``sl.Camera``; the two have parallel but distinct open/retrieve APIs.

    Raises:
        CableTestError: If the camera cannot be opened.
    """
    if kind == "stereo":
        zed = sl.Camera()
        init_params = sl.InitParameters()
        # We only need the rectified images; skip depth to save GPU.
        init_params.depth_mode = sl.DEPTH_MODE.NONE
    else:
        zed = sl.CameraOne()
        init_params = sl.InitParametersOne()

    _logger.info("Opening connected ZED camera (%s)...", kind)
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise CableTestError(f"Failed to open camera: {err}")
    return zed


def _retrieve_frame(sl, zed, image, kind: str):  # type: ignore[no-untyped-def]
    """Retrieve one frame into ``image``; return the SDK error code.

    For a stereo camera the left eye is retrieved — a marginal cable drops or
    freezes frames on both eyes equally, so validating one eye is sufficient to
    exercise the link.
    """
    if kind == "stereo":
        return zed.retrieve_image(image, sl.VIEW.LEFT)
    return zed.retrieve_image(image)


def _validate_frame(bgr: np.ndarray, expected_width: int, expected_height: int) -> None:
    """Raise :class:`CableTestError` unless ``bgr`` looks like a real frame.

    Checks shape, dtype, that the image is not empty, that its resolution matches
    the camera's reported configuration, and that it carries actual image content
    (non-trivial variation and a plausible brightness level).
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
    if width != expected_width or height != expected_height:
        raise CableTestError(
            f"Frame resolution {width}x{height} does not match the camera's "
            f"reported {expected_width}x{expected_height}."
        )

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


def run(output: str | None = None, kind: str = "auto") -> None:
    """Restart the daemon, open the camera, and validate frames over 5 seconds.

    Args:
        output: Optional path to save the last captured frame as PNG for inspection.
        kind: Camera pipeline to test — ``"mono"`` (ZED-X One), ``"stereo"``
            (ZED X), or ``"auto"`` to detect it from the single connected camera.

    Raises:
        CableTestError: If the daemon cannot be restarted, the camera kind cannot
            be resolved, the camera cannot be opened, no frames can be grabbed, or
            any captured frame fails validation.
    """
    import pyzed.sl as sl

    try:
        restart_zed_daemon()
    except RuntimeError as exc:
        raise CableTestError(str(exc)) from exc

    if kind == "auto":
        kind = _detect_kind()
        _logger.info("Auto-detected %s camera.", kind)

    zed = _open_camera(sl, kind)

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
        grab_failures = 0
        consecutive_failures = 0
        max_consecutive_failures = 0
        duplicate_run = 0
        max_duplicate_run = 0
        prev_bgr: np.ndarray | None = None
        deadline = time.monotonic() + _CAPTURE_DURATION_S
        while time.monotonic() < deadline:
            grab_err = zed.grab()
            if grab_err != sl.ERROR_CODE.SUCCESS:
                grab_failures += 1
                consecutive_failures += 1
                max_consecutive_failures = max(
                    max_consecutive_failures, consecutive_failures
                )
                if consecutive_failures > _MAX_CONSECUTIVE_GRAB_FAILURES:
                    raise CableTestError(
                        f"{consecutive_failures} consecutive grab failures "
                        f"(last: {grab_err}); cable is dropping frames."
                    )
                continue
            consecutive_failures = 0

            if _retrieve_frame(sl, zed, image, kind) != sl.ERROR_CODE.SUCCESS:
                raise CableTestError("Failed to retrieve image from camera.")

            raw = image.get_data()  # BGRA uint8
            bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
            _validate_frame(bgr, res.width, res.height)

            if prev_bgr is not None and np.array_equal(bgr, prev_bgr):
                duplicate_run += 1
                max_duplicate_run = max(max_duplicate_run, duplicate_run)
                if duplicate_run >= _MAX_DUPLICATE_FRAMES:
                    raise CableTestError(
                        f"{duplicate_run + 1} consecutive identical frames; "
                        "stream appears frozen rather than live."
                    )
            else:
                duplicate_run = 0

            prev_bgr = bgr
            last_bgr = bgr
            valid_frames += 1

        if valid_frames == 0:
            raise CableTestError(
                f"No frames grabbed in {_CAPTURE_DURATION_S:.0f}s; "
                "camera may be disconnected."
            )

        # A healthy link delivers roughly fps * duration frames. A large shortfall
        # signals dropped frames from a flaky cable, so require at least a fraction
        # of the expected count.
        expected_frames = int(fps * _CAPTURE_DURATION_S)
        min_frames = int(expected_frames * _MIN_FRAME_FRACTION)
        if valid_frames < min_frames:
            raise CableTestError(
                f"Only grabbed {valid_frames} frames in {_CAPTURE_DURATION_S:.0f}s; "
                f"expected ~{expected_frames} ({fps}fps), at least {min_frames}. "
                "Cable may be dropping frames."
            )
        _logger.info(
            "Validated %d frames over %.0fs (expected ~%d); "
            "%d grab failures (max %d in a row), max %d duplicate frames.",
            valid_frames,
            _CAPTURE_DURATION_S,
            expected_frames,
            grab_failures,
            max_consecutive_failures,
            max_duplicate_run,
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


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the last captured frame as PNG (e.g. logs/cable_frame.png).",
    )
    parser.add_argument(
        "--kind",
        choices=("auto", "mono", "stereo"),
        default="auto",
        help="Camera pipeline to test: mono (ZED-X One), stereo (ZED X), or "
        "auto to detect from the connected camera (default: auto).",
    )


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``diag.zed-cable`` subcommand (for serve introspection)."""
    p = subparsers.add_parser(
        "diag.zed-cable",
        help="Verify a ZED camera cable by validating captured frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    _add_arguments(p)
    p.set_defaults(func=run_cli)


def run_cli(args: argparse.Namespace) -> None:
    """Run the ZED cable test from parsed arguments."""
    run(output=args.output, kind=args.kind)


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run the ZED cable test."""
    parser = argparse.ArgumentParser(
        description="Test a ZED camera cable (mono or stereo) by validating "
        "frames across 5 seconds."
    )
    _add_arguments(parser)
    run_cli(parser.parse_args(argv))


if __name__ == "__main__":
    main()
