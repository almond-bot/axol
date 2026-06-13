"""LeRobot Axol VR teleoperator adapter: the Teleoperator interface and its config."""

from .config_vr import AxolVRTeleopConfig
from .teleop_vr import AxolVRTeleop

__all__ = ["AxolVRTeleop", "AxolVRTeleopConfig"]
