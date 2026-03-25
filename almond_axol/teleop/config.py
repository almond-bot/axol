from __future__ import annotations

from dataclasses import dataclass, field

from ..motor import JointValues
from ..shared import ARM_JOINTS


@dataclass
class TeleopConfig:
    """Configuration for a :class:`VRTeleop` session.

    Attributes:
        rest_pose_left: Left arm rest configuration in revolutions, keyed by
            :class:`Joint`. Used as the reset target. Defaults to all zeros
            (Axol default rest pose).
        rest_pose_right: Right arm rest configuration in revolutions, keyed by
            :class:`Joint`. Used as the reset target. Defaults to all zeros
            (Axol default rest pose).
        frequency: Control loop rate in Hz used by :meth:`VRTeleop.run` and
            as waypoint density for reset trajectories.
        reset_speed: Speed of the reset move in rev/s. Determines the number
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
    """

    rest_pose_left: JointValues = field(
        default_factory=lambda: dict(
            zip(ARM_JOINTS, [0.05, 0.0, 0.0, 0.1, 0.0, 0.0, 0.05])
        )
    )
    rest_pose_right: JointValues = field(
        default_factory=lambda: dict(
            zip(ARM_JOINTS, [-0.05, 0.0, 0.0, 0.1, 0.0, 0.0, -0.05])
        )
    )
    frequency: float = 100.0
    reset_speed: float = 0.1
    reset_rest_weight: float = 20.0
    reset_limit_weight: float = 100.0
    reset_collision_margin: float = 0.01
    reset_collision_weight: float = 10.0
    reset_max_iterations: int = 10
