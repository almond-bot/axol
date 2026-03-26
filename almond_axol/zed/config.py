from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ZedConfig:
    """Configuration for the ZED camera streamer.

    Args:
        overhead_serial:  Serial number of the overhead camera.
        left_arm_serial:  Serial number of the left-arm camera.
        right_arm_serial: Serial number of the right-arm camera.
        overhead_port:    Streaming port for the overhead camera (default 30000).
        left_arm_port:    Streaming port for the left-arm camera (default 30002).
        right_arm_port:   Streaming port for the right-arm camera (default 30004).
        bitrate:          Encoding bitrate in kbits/s (default 6000, recommended for HEVC HD720).
    """

    overhead_serial: int
    left_arm_serial: int
    right_arm_serial: int
    overhead_port: int = 30000
    left_arm_port: int = 30002
    right_arm_port: int = 30004
    bitrate: int = 6000
