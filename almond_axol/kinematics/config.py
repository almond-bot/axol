from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class KinematicsConfig:
    """Cost weights and solver parameters for :class:`KinematicsSolver`.

    All weights are unitless scale factors passed directly to pyroki cost
    functions. Higher values make the solver prioritise that term more strongly.

    Attributes:
        pos_weight: Weight on end-effector position error.
        ori_weight: Weight on end-effector orientation error.
        elbow_weight: Weight on elbow position hints (position only, no orientation).
        rest_weight: Weight penalising deviation from the current joint configuration.
        manipulability_weight: Weight rewarding configurations with high manipulability.
        limit_weight: Weight penalising joint-limit violations.
        self_collision_margin: Minimum clearance (m) enforced between collision bodies.
        self_collision_weight: Weight on the self-collision penalty.
        max_iterations: Maximum solver iterations per call.
        cost_tolerance: Solver convergence tolerance.
        max_joint_delta: Maximum joint change per :meth:`KinematicsSolver.ik` call, in radians.
        max_reach: Maximum allowed distance (m) from shoulder to end-effector target.
    """

    pos_weight: float = 50.0
    ori_weight: float = 10.0
    elbow_weight: float = 25.0
    rest_weight: float = 4.5
    manipulability_weight: float = 0.05
    limit_weight: float = 75.0
    self_collision_margin: float = 0.075
    self_collision_weight: float = 50.0
    max_iterations: int = 25
    cost_tolerance: float = 1e-2
    max_joint_delta: float = 0.0055 * 2 * math.pi
    max_reach: float = 0.8
