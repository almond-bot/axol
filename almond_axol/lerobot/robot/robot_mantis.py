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
from typing import TYPE_CHECKING

from ...robot.mantis import Mantis
from .config_mantis import MantisRobotConfig
from .robot_axol import AxolRobot

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ...kinematics.config import KinematicsConfig


class MantisRobot(AxolRobot):
    """LeRobot Robot wrapping the Mantis.

    Observation state is 16 joint positions like the robot's: the 14 arm
    values echo the commanded IK solution (there is no physical arm to
    measure), the 2 gripper values are real motor feedback. Actions are the
    same joint-position dicts teleop produces; only the grippers actuate.
    """

    config_class = MantisRobotConfig
    name = "axol_mantis"

    def __init__(
        self,
        config: MantisRobotConfig,
        *,
        ik_config: KinematicsConfig | None = None,
        defer_gripper_enable: bool = False,
    ) -> None:
        """Build a Mantis robot, optionally deferring gripper motor torque.

        The default preserves standalone/teleop behavior.  Data collection
        passes ``defer_gripper_enable=True`` so :meth:`connect` opens the CAN
        buses and cameras without actuating either gripper; episode control
        then calls :meth:`enable_grippers_async` and
        :meth:`disable_grippers_async` on the robot event loop.
        """
        self._defer_gripper_enable = defer_gripper_enable
        super().__init__(config, ik_config=ik_config)

    def _build_hardware(self) -> Mantis:
        return Mantis(
            self.config.axol_config,
            left_channel=self.config.left_channel,
            right_channel=self.config.right_channel,
            defer_gripper_enable=self._defer_gripper_enable,
        )

    async def enable_grippers_async(self) -> None:
        """Enable and, on first use, calibrate both Mantis grippers."""
        assert isinstance(self._axol, Mantis), "connect() first"
        await self._axol.enable_grippers()

    async def disable_grippers_async(self) -> None:
        """Disable both grippers, retaining calibration for the next episode."""
        assert isinstance(self._axol, Mantis), "connect() first"
        await self._axol.disable_grippers()
