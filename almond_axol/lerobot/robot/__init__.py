"""LeRobot Axol robot adapter: the dual-arm Robot interface and its config."""

from .config_axol import AxolRobotConfig
from .config_mantis import MantisRobotConfig
from .robot_axol import AxolRobot
from .robot_mantis import MantisRobot

__all__ = ["AxolRobot", "AxolRobotConfig", "MantisRobot", "MantisRobotConfig"]
