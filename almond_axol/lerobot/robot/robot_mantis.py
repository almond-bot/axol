"""The Mantis as a LeRobot Robot.

Same synchronous Robot surface as :class:`AxolRobot` — observations, actions,
cameras, event loop — but the hardware is the Mantis rig: a pair of handheld
Damiao grippers on their own CAN buses, with virtual arm joints latched from
the commanded IK targets. Like the robot it is driven through the Rust
realtime core (:class:`~almond_axol.rt.RtMantis` wrapping
:class:`~almond_axol.robot.mantis.Mantis`): ``axol-rt`` owns the gripper
buses and streams the POSITION_FORCE commands for the duration of each take.
``collect-data`` drives it with the exact same control loop it uses for the
robot, so Mantis datasets are schema-identical to robot-collected ones.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ...robot.mantis import Mantis
from ...rt import RtMantis
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
        :meth:`disable_grippers_async` on the robot event loop, which arm and
        disarm the realtime core on the gripper buses per take.
        """
        self._defer_gripper_enable = defer_gripper_enable
        super().__init__(config, ik_config=ik_config)

    def _build_hardware(self) -> RtMantis:
        return RtMantis(
            Mantis(
                self.config.axol_config,
                left_channel=self.config.left_channel,
                right_channel=self.config.right_channel,
                defer_gripper_enable=self._defer_gripper_enable,
            ),
            record=self._control_trace,
        )

    async def open_grippers_async(self) -> None:
        """Open both grippers fully, then release torque (pre-record prep).

        See :meth:`~almond_axol.rt.RtMantis.open_grippers`: this also
        performs the one-time hard-stop calibration, so the later
        :meth:`enable_grippers_async` at take start is immediate.
        """
        assert isinstance(self._axol, RtMantis), "connect() first"
        await self._axol.open_grippers()

    async def enable_grippers_async(self) -> None:
        """Enable (calibrating on first use) both grippers and arm the core."""
        assert isinstance(self._axol, RtMantis), "connect() first"
        await self._axol.enable_grippers()

    async def disable_grippers_async(self) -> None:
        """Disarm the core and disable both grippers, retaining calibration."""
        assert isinstance(self._axol, RtMantis), "connect() first"
        await self._axol.disable_grippers()
