from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig

from ...kinematics.config import KinematicsConfig
from ...teleop.config import TeleopConfig


@TeleoperatorConfig.register_subclass("axol_vr")
@dataclass
class AxolVRTeleopConfig(TeleoperatorConfig):
    """Configuration for the VR-based Axol teleoperator.

    Args:
        teleop_config:      VR teleop session parameters (rest poses, frequency, smoothing).
        kinematics_config:  IK solver parameters forwarded to the subprocess.
    """

    teleop_config: TeleopConfig = field(default_factory=TeleopConfig)
    kinematics_config: KinematicsConfig = field(default_factory=KinematicsConfig)
