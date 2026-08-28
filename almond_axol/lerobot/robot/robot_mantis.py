"""The Mantis as a LeRobot Robot.

Same synchronous Robot surface as :class:`AxolRobot` — observations, actions,
cameras, event loop — but the hardware is :class:`~almond_axol.robot.mantis.Mantis`:
a pair of handheld Damiao grippers on their own CAN buses, with virtual arm
joints latched from the commanded IK targets. ``collect-data`` drives it with
the exact same control loop it uses for the robot, so Mantis datasets are
schema-identical to robot-collected ones.
"""

from __future__ import annotations

import logging

from ...robot.mantis import Mantis
from .config_mantis import MantisRobotConfig
from .robot_axol import AxolRobot

_logger = logging.getLogger(__name__)


class MantisRobot(AxolRobot):
    """LeRobot Robot wrapping the Mantis.

    Observation state is 16 joint positions like the robot's: the 14 arm
    values echo the commanded IK solution (there is no physical arm to
    measure), the 2 gripper values are real motor feedback. Actions are the
    same joint-position dicts teleop produces; only the grippers actuate.
    """

    config_class = MantisRobotConfig
    name = "axol_mantis"

    def _build_hardware(self) -> Mantis:
        return Mantis(
            self.config.axol_config,
            left_channel=self.config.left_channel,
            right_channel=self.config.right_channel,
        )
