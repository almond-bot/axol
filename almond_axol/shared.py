"""Shared constants for the Almond Axol robot."""

import math
from pathlib import Path

from .motor import Joint

CAN_LEFT = "can_alm_axol_l"
CAN_RIGHT = "can_alm_axol_r"

ARM_JOINTS: list[Joint] = [j for j in Joint if j != Joint.GRIPPER]

URDF_PATH: Path = Path(__file__).resolve().parent / "kinematics" / "axol.urdf"


def rev_to_rad(rev: float) -> float:
    """Convert revolutions to radians."""
    return rev * 2.0 * math.pi


def rad_to_rev(rad: float) -> float:
    """Convert radians to revolutions."""
    return rad / (2.0 * math.pi)


def rev_to_deg(rev: float) -> float:
    """Convert revolutions to degrees."""
    return rev * 360.0


def deg_to_rev(deg: float) -> float:
    """Convert degrees to revolutions."""
    return deg / 360.0
