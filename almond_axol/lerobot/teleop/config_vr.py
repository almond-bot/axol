from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig

from ...kinematics.config import KinematicsConfig
from ...teleop.config import VRTeleopConfig
from ...vr.config import VRServerConfig


@TeleoperatorConfig.register_subclass("axol_vr")
@dataclass
class AxolVRTeleopConfig(TeleoperatorConfig):
    """Configuration for the VR-based Axol teleoperator.

    Args:
        vr_teleop_config:      VR teleop session parameters (rest poses, frequency, smoothing).
        kinematics_config:  IK solver parameters forwarded to the subprocess.
        vr_server_config:   VR WebSocket server parameters (port, TLS certs).
    """

    vr_teleop_config: VRTeleopConfig = field(default_factory=VRTeleopConfig)
    kinematics_config: KinematicsConfig = field(default_factory=KinematicsConfig)
    vr_server_config: VRServerConfig = field(default_factory=VRServerConfig)
