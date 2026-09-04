"""Configuration dataclass for the Mantis as a LeRobot Robot."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.robots.config import RobotConfig

from ...constants import CAN_MANTIS_LEFT, CAN_MANTIS_RIGHT
from .config_axol import AxolRobotConfig


@RobotConfig.register_subclass("axol_mantis")
@dataclass
class MantisRobotConfig(AxolRobotConfig):
    """Configuration for the Mantis handheld data-collection rig.

    Identical to :class:`AxolRobotConfig` — same camera slots, gains, and
    observation options, so recorded datasets keep the robot schema — but the
    hardware behind it is :class:`~almond_axol.robot.mantis.Mantis`: one Damiao
    gripper per CAN bus and virtual arm joints that echo the commanded IK
    targets. Cameras are the wrist slots only (``left_arm`` / ``right_arm``,
    mounted on the handheld grippers); there is no overhead camera.
    """

    left_channel: str = CAN_MANTIS_LEFT
    right_channel: str = CAN_MANTIS_RIGHT

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.axol_config.has_gripper:
            raise ValueError(
                "Mantis always has two physical grippers; has_gripper cannot be false"
            )
