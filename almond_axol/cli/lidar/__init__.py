"""``axol lidar.*`` commands: set up, inspect and view the Livox lidar."""

from __future__ import annotations

import time

from ...lidar import LidarConfig, LidarStatus
from ...lidar.stream import LidarStream, StreamStats


def format_status(status: LidarStatus) -> str:
    lines = [
        f"  Model        {status.product}",
        f"  Serial       {status.serial}",
        f"  Firmware     {status.firmware}",
        f"  Address      {status.lidar_ip} (mask {status.netmask})",
        f"  Streams to   {status.point_host[0]}:{status.point_host[1]} (points), "
        f"{status.imu_host[0]}:{status.imu_host[1]} (IMU)",
        f"  State        {status.work_state_name}",
        f"  Temperature  {status.temperature_c:.1f} °C",
        f"  Time sync    {status.time_sync_name}",
        f"  Faults       {', '.join(f'0x{c:08x}' for c in status.faults) or 'none'}",
    ]
    return "\n".join(lines)


def format_stats(stats: StreamStats) -> str:
    return (
        f"  {stats.points_per_s / 1000:.0f}k points/s, "
        f"{stats.frames_per_s:.1f} frames/s, IMU {stats.imu_per_s:.0f} Hz"
        + (f", {stats.bad_packets} bad packets" if stats.bad_packets else "")
    )


def measure(config: LidarConfig, seconds: float = 2.0) -> StreamStats:
    """Stream for ``seconds`` and return the measured rates."""
    with LidarStream(config) as stream:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            stream.get(0.5)
        return stream.stats()
