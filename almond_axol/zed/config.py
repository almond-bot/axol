from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ZedConfig:
    """Configuration for the ZED camera streamer.

    At least one serial number must be provided.

    Args:
        overhead_serial:  Serial number of the overhead camera (optional).
        left_arm_serial:  Serial number of the left-arm camera (optional).
        right_arm_serial: Serial number of the right-arm camera (optional).
        overhead_port:    Streaming port for the overhead camera (default 30000).
        left_arm_port:    Streaming port for the left-arm camera (default 30002).
        right_arm_port:   Streaming port for the right-arm camera (default 30004).
        bitrate:          Encoding bitrate in kbits/s (default 6000, recommended for HEVC HD720).
    """

    overhead_serial: int | None = None
    left_arm_serial: int | None = None
    right_arm_serial: int | None = None
    overhead_port: int = 30000
    left_arm_port: int = 30002
    right_arm_port: int = 30004
    bitrate: int = 6000

    def __post_init__(self) -> None:
        if self.overhead_serial is None and self.left_arm_serial is None and self.right_arm_serial is None:
            raise ValueError("At least one camera serial number must be provided.")
