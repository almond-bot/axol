"""
axol lidar.record

Record lidar frames and IMU samples to a file. Build a map from it later
with ``axol nav.map NAME --from FILE``.

Usage:
    axol lidar.record drive.npz                # until Ctrl-C
    axol lidar.record drive.npz --seconds 60
"""

from __future__ import annotations

import argparse
import time

from ...lidar import LidarConfig, LidarStream
from ...lidar.recording import LidarRecording


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("lidar.record", help="Record the lidar to a file.")
    p.add_argument("path", help="File to write (.npz).")
    p.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="Stop after this long (default: Ctrl-C).",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    rec = LidarRecording()
    start = time.monotonic()
    last_imu = None
    print("Recording the lidar (Ctrl-C to stop)...")
    with LidarStream(LidarConfig.load()) as stream:
        try:
            for frame in stream.frames():
                end = frame.stamp + frame.duration
                rec.add_frame(frame)
                rec.add_imu(
                    stream.imu_between(
                        last_imu if last_imu is not None else frame.stamp - 0.5, end
                    )
                )
                last_imu = end
                if args.seconds and time.monotonic() - start >= args.seconds:
                    break
                if len(rec) % 50 == 0:
                    print(f"  {len(rec) / 10:.0f} s recorded")
        except KeyboardInterrupt:
            pass
    path = rec.save(args.path)
    print(f"Saved {len(rec)} frames ({rec.duration:.0f} s) to {path}.")
