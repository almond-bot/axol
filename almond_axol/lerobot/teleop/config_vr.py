"""Configuration dataclass for the VR-based Axol teleoperator."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig

from ...kinematics.config import KinematicsConfig
from ...robot.jelly import JellyConfig
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
        jelly:               Jelly (x-drive base + telescoping lift) for
                            robots that have one; ``jelly.enabled`` gates it.
                            Operator-only mobility: the thumbsticks reposition
                            the base/lift during a session, but Jelly state is
                            never recorded into the dataset and policies never
                            control it.
        has_gripper:        Whether the robot has grippers. ``False`` (the
                            gripperless SKU) drops the gripper keys from the
                            emitted actions so they match the robot's action
                            features. ``collect-data`` propagates this from
                            ``AxolRobotConfig.axol_config.has_gripper``
                            automatically.
    """

    vr_teleop_config: VRTeleopConfig = field(default_factory=VRTeleopConfig)
    kinematics_config: KinematicsConfig = field(default_factory=KinematicsConfig)
    vr_server_config: VRServerConfig = field(default_factory=VRServerConfig)
    jelly: JellyConfig = field(default_factory=JellyConfig)
    has_gripper: bool = True
