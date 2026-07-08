"""Configuration dataclass for the handheld UMI rig as a LeRobot Robot."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.robots.config import RobotConfig

from ...constants import CAN_UMI_LEFT, CAN_UMI_RIGHT
from .config_axol import AxolRobotConfig


@RobotConfig.register_subclass("axol_umi")
@dataclass
class UmiRobotConfig(AxolRobotConfig):
    """Configuration for the handheld UMI data-collection rig.

    Identical to :class:`AxolRobotConfig` — same camera slots, gains, and
    observation options, so recorded datasets keep the robot schema — but the
    hardware behind it is :class:`~almond_axol.robot.umi.Umi`: one Damiao
    gripper per CAN bus and virtual arm joints that echo the commanded IK
    targets. Cameras are the wrist slots only (``left_arm`` / ``right_arm``,
    mounted on the handheld grippers); there is no overhead camera.
    """

    left_channel: str = CAN_UMI_LEFT
    right_channel: str = CAN_UMI_RIGHT
