from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class VRTeleopConfig:
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
        startup_max_accel: Maximum joint acceleration (rad/s²) used only during
            the initial startup trajectory (current pose → rest pose).  Motors
            are cold at this point and respond sharply; a gentler ramp avoids
            the initial jerk.  Restored to ``teleop_max_accel`` once the
            startup trajectory completes.  Defaults to 0.3 rev/s².
        engage_max_vel: Maximum joint velocity (rad/s) used by the
            trapezoidal filter when the deadman switch is first pressed after a
            rest-pose trajectory (startup or reset).  Slows the transition from
            rest pose to the first IK target.  Restored to ``teleop_max_vel``
            after ``engage_duration`` seconds.  Defaults to
            ``reset_speed`` for a consistent feel.
        engage_duration: Seconds to hold ``engage_max_vel`` after the
            post-rest deadman rising edge before restoring ``teleop_max_vel``.
        teleop_max_vel: Maximum joint velocity (rad/s) enforced by the
            trapezoidal filter during normal teleoperation.  Limits how fast
            any single joint can move toward a new IK target.  Defaults to
            0.5 rev/s (~180 °/s).
        teleop_max_accel: Maximum joint acceleration (rad/s²) enforced by the
            trapezoidal filter.  Controls how quickly the commanded velocity
            ramps up or down.  Defaults to 1.5 rev/s² (~540 °/s²), giving a
            ~0.2 s ramp from rest to full speed.
    """

    rest_pose_left: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                -0.025 * 2 * math.pi,
                0.0,
                0.0,
                0.05 * 2 * math.pi,
                0.0,
                0.0,
                -0.025 * 2 * math.pi,
            ],
            dtype=np.float32,
        )
    )
    rest_pose_right: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                0.025 * 2 * math.pi,
                0.0,
                0.0,
                -0.05 * 2 * math.pi,
                0.0,
                0.0,
                0.025 * 2 * math.pi,
            ],
            dtype=np.float32,
        )
    )
    frequency: float = 120.0
    reset_speed: float = 0.1 * 2 * math.pi
    reset_rest_weight: float = 50.0
    reset_limit_weight: float = 100.0
    reset_collision_margin: float = 0.025
    reset_collision_weight: float = 100.0
    reset_max_iterations: int = 10
    startup_max_accel: float = 0.3 * 2 * math.pi
    engage_max_vel: float = 0.1 * 2 * math.pi
    engage_duration: float = 1.0
    teleop_max_vel: float = 0.5 * 2 * math.pi
    teleop_max_accel: float = 1.5 * 2 * math.pi
