from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TeleopConfig:
    """Configuration for a :class:`VRTeleop` session.

    Attributes:
        rest_pose_left: Left arm rest configuration in radians, shape (7,) in
            ARM_JOINTS order (no gripper). Used as the reset target.
        rest_pose_right: Right arm rest configuration in radians, shape (7,) in
            ARM_JOINTS order (no gripper). Used as the reset target.
        frequency: Control loop rate in Hz used by :meth:`VRTeleop.run` and
            as waypoint density for reset trajectories.
        reset_speed: Speed of the reset move in rad/s. Determines the number
            of trajectory waypoints based on the distance to the rest pose.
        reset_rest_weight: Cost weight penalising deviation from the reset
            target pose during collision-aware trajectory generation.
        reset_limit_weight: Cost weight penalising joint-limit violations
            during reset trajectory generation.
        reset_collision_margin: Minimum clearance (m) enforced between
            collision bodies during reset trajectory generation.
        reset_collision_weight: Cost weight on self-collision penalty during
            reset trajectory generation.
        reset_max_iterations: Maximum solver iterations per reset waypoint.
        smooth_alpha: Exponential smoothing factor for IK output in (0, 1].
            ``1.0`` disables smoothing. Lower values = smoother but more lag.
    """

    rest_pose_left: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                0.05 * 2 * math.pi,
                0.0,
                0.0,
                0.1 * 2 * math.pi,
                0.0,
                0.0,
                0.05 * 2 * math.pi,
            ],
            dtype=np.float32,
        )
    )
    rest_pose_right: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                -0.05 * 2 * math.pi,
                0.0,
                0.0,
                0.1 * 2 * math.pi,
                0.0,
                0.0,
                -0.05 * 2 * math.pi,
            ],
            dtype=np.float32,
        )
    )
    frequency: float = 120.0
    reset_speed: float = 0.1 * 2 * math.pi
    reset_rest_weight: float = 20.0
    reset_limit_weight: float = 100.0
    reset_collision_margin: float = 0.01
    reset_collision_weight: float = 10.0
    reset_max_iterations: int = 10
    smooth_alpha: float = 0.45
