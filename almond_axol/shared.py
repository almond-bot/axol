"""Shared constants for the Almond Axol robot."""

from enum import Enum
from pathlib import Path


class Joint(Enum):
    SHOULDER_1 = "shoulder_1"
    SHOULDER_2 = "shoulder_2"
    SHOULDER_3 = "shoulder_3"
    ELBOW = "elbow"
    WRIST_1 = "wrist_1"
    WRIST_2 = "wrist_2"
    WRIST_3 = "wrist_3"
    GRIPPER = "gripper"


CAN_LEFT = "can_alm_axol_l"
CAN_RIGHT = "can_alm_axol_r"

ARM_JOINTS: list[Joint] = [j for j in Joint if j != Joint.GRIPPER]

URDF_PATH: Path = (
    Path(__file__).resolve().parent / "kinematics" / "urdf" / "openarm.urdf"
)
