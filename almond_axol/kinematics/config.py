"""KinematicsConfig dataclass with cost weights and solver parameters for KinematicsSolver."""

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
            Acts as a per-step damping term; uses q_current as the target.
        posture_weight: Weight penalising deviation from the global preferred posture.
            Acts as a persistent attractor toward the home/rest configuration,
            preventing slow null-space drift (e.g. unnecessary shoulder twist).
        manipulability_weight: Weight rewarding configurations with high manipulability.
        limit_weight: Weight penalising joint-limit violations.
        lower_arm_collision_margin: Extra clearance (m), measured relative to the
            home-pose distance, kept between the lower forearm/elbow links
            (``*_s3``/``*_e1``/``*_e2``) and the torso (``base``/``s1``). This is
            the knob that keeps the elbow off the base. Because it is referenced
            to the home pose it only bites as the elbow approaches the base and
            does not push the whole arm outward.
        lower_arm_collision_weight: Weight on the lower-arm self-collision penalty.
        distal_collision_margin: Extra clearance (m), relative to the home-pose
            distance, for the distal links (``*_w0``/``*_w1``/``*_w2``/``*_gripper``)
            vs the torso. Kept small so the arm's outward range of motion is
            preserved.
        distal_collision_weight: Weight on the distal self-collision penalty.
        max_iterations: Maximum solver iterations per call.
        cost_tolerance: Solver convergence tolerance.
        max_joint_delta: Maximum joint change per :meth:`KinematicsSolver.ik` call, in radians.
        max_reach: Maximum allowed distance (m) from shoulder to end-effector target.
    """

    pos_weight: float = 50.0
    ori_weight: float = 10.0
    elbow_weight: float = 5.0
    rest_weight: float = 7.5
    posture_weight: float = 5.0
    manipulability_weight: float = 0.05
    limit_weight: float = 75.0
    lower_arm_collision_margin: float = 0.04
    lower_arm_collision_weight: float = 100.0
    distal_collision_margin: float = 0.02
    distal_collision_weight: float = 75.0
    max_iterations: int = 8
    cost_tolerance: float = 1e-2
    max_joint_delta: float = 0.0055 * 2 * math.pi
    max_reach: float = 0.8
