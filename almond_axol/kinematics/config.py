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
        self_collision_margin: Minimum clearance (m) enforced between collision bodies.
        self_collision_weight: Weight on the self-collision penalty.
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
    self_collision_margin: float = 0.1
    self_collision_weight: float = 75.0
    max_iterations: int = 8
    cost_tolerance: float = 1e-2
    max_joint_delta: float = 0.0055 * 2 * math.pi
    max_reach: float = 0.8


# Solver values the UMI handheld-rig profile forces (see
# :func:`apply_umi_kinematics_profile`). Tuned with ``scripts/umi_ik_bench.py``
# against the feasibility floor of a near-unconstrained solve: within the
# reachable workspace this tracks the hand with ~2 mm mean / <6 mm p95 excess
# jaw-tip error over that floor, with no null-space drift while the hand is
# still and a per-tick solve that fits a 30 Hz collection tick on the Jetson.
#
# Rationale per field:
#   pos/ori weight   Raised ~4x/12x over the arm defaults so the pose target
#                    dominates the (default-strength) rest/posture
#                    regularizers — raising the ratio shrinks tracking error;
#                    weakening the regularizers instead unanchors the elbow
#                    null space and destabilises the solve.
#   manipulability   Off: it biases q away from the exact pose solution at
#                    every tick (~7 mm at rest). Protecting a physical arm
#                    from singular configs doesn't apply to a virtual one.
#   collision margin The 10 cm default is already active at the folded rest
#                    pose, shoving targets ~9 mm off; 2 cm still separates
#                    the capsules while freeing exact tracking. The virtual
#                    arms must stay collision-free (recorded joints replay on
#                    the real robot), only the standoff shrinks.
#   max_joint_delta  ~1.4x the fastest bench hand motion so tracking never
#                    saturates the per-tick clamp; still bounded so a bad
#                    frame can't teleport the solution.
#   max_iterations   12 (from 8): recovers convergence headroom the higher
#                    weights consume, within the 30 Hz solve budget.
UMI_KINEMATICS_OVERRIDES: dict[str, float | int] = {
    "pos_weight": 200.0,
    "ori_weight": 120.0,
    "manipulability_weight": 0.0,
    "self_collision_margin": 0.02,
    "max_joint_delta": 0.05,
    "max_iterations": 12,
}


def apply_umi_kinematics_profile(config: KinematicsConfig) -> None:
    """Apply the UMI handheld-rig solver profile in place.

    Shared by ``teleop --umi`` and ``collect-data --umi`` (the same way
    :func:`almond_axol.teleop.config.apply_umi_teleop_profile` is). Each
    override is applied only when the field still holds its dataclass
    default, so explicit ``--kinematics.<field>`` flags win over the profile.
    """
    defaults = KinematicsConfig()
    for name, value in UMI_KINEMATICS_OVERRIDES.items():
        if getattr(config, name) == getattr(defaults, name):
            setattr(config, name, value)
