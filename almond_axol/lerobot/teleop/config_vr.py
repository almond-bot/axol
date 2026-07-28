"""Configuration dataclass for the VR-based Axol teleoperator."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig

from ...kinematics.config import KinematicsConfig
from ...robot.cart import CartConfig
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
        cart:               Powered cart (x-drive base + telescoping lift) for
                            robots that have one; ``cart.enabled`` gates it.
                            Operator-only mobility: the thumbsticks reposition
                            the base/lift during a session, but cart state is
                            never recorded into the dataset and policies never
                            control it.
    """

    vr_teleop_config: VRTeleopConfig = field(default_factory=VRTeleopConfig)
    kinematics_config: KinematicsConfig = field(default_factory=KinematicsConfig)
    vr_server_config: VRServerConfig = field(default_factory=VRServerConfig)
    cart: CartConfig = field(default_factory=CartConfig)
