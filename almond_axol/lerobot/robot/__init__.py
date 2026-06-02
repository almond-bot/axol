"""LeRobot Axol robot adapter: the dual-arm Robot interface and its config."""

from .config_axol import AxolRobotConfig
from .robot_axol import AxolRobot

__all__ = ["AxolRobot", "AxolRobotConfig"]
