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
            The residual is a reciprocal barrier on the Yoshikawa index, bounded
            near singularities (saturating at index ~1e-2 rather than pyroki's
            1e-6): the unbounded barrier gives ~1e9-scale Jacobian rows at a
            singular seed (e.g. an arm at the all-zero straight pose), whose
            float32 normal matrix is indefinite after rounding — cuSolver's
            Cholesky then NaNs, every LM step is rejected, and ik() silently
            returns its seed on GPU while CPU LAPACK survives by rounding luck.
        limit_weight: Weight penalising joint-limit violations.
        self_collision_margin: Minimum clearance (m) enforced between collision
            bodies. Keep this below the arm-torso clearance of normal teleop
            poses: a margin that is already violated at the folded rest pose
            (e.g. the old 0.1 default) keeps the nonsmooth collision hinge
            active during ordinary tracking, which makes the trust-region
            solver reject steps — the arm freezes while small target errors
            accumulate, then lurches once the error is large enough to punch
            through (the teleop "stuck, then jumps" failure). 0.025 matches
            ``VRTeleopConfig.reset_collision_margin``.
        self_collision_weight: Weight on the self-collision penalty.
        max_iterations: Maximum solver iterations per call.
        cost_tolerance: Solver convergence tolerance — terminate when the
            relative cost change of an iteration falls below this. jaxls also
            applies it to *rejected* proposals (whose cost change is ~0), so a
            loose value (the old 1e-2) makes one rejected step read as
            "converged" and return the seed unchanged. Keep it small enough
            that only genuine convergence trips it.
        lambda_initial: Initial Levenberg-Marquardt damping. The per-tick solve
            re-seeds from the previous solution, so the first proposal is taken
            in a region where the linearisation may be poor (e.g. with the
            self-collision cost active); a moderately damped start gets an
            acceptable step in fewer rejections than the jaxls default (5e-4).
        lambda_factor: Multiplier applied to the LM damping after each rejected
            step. With the jaxls default (2.0) and ``max_iterations`` of 8,
            damping can only grow 256x per solve — not enough to recover from a
            rejected near-Gauss-Newton step near an active collision
            constraint, so the solver used to return its seed unchanged for
            many consecutive ticks (teleop froze) and then lurch once the
            accumulated target error finally made a full step acceptable. 10.0
            spans the useful damping range within the iteration budget.
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
    self_collision_margin: float = 0.025
    self_collision_weight: float = 75.0
    max_iterations: int = 8
    cost_tolerance: float = 1e-4
    lambda_initial: float = 1e-2
    lambda_factor: float = 10.0
    max_joint_delta: float = 0.0055 * 2 * math.pi
    max_reach: float = 0.8


# Solver values the Mantis UMI profile forces (see
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
    """Apply the Mantis UMI solver profile in place.

    Shared by ``teleop --umi`` and ``collect-data --umi`` (the same way
    :func:`almond_axol.teleop.config.apply_umi_teleop_profile` is). Each
    override is applied only when the field still holds its dataclass
    default, so explicit ``--kinematics.<field>`` flags win over the profile.
    """
    defaults = KinematicsConfig()
    for name, value in UMI_KINEMATICS_OVERRIDES.items():
        if getattr(config, name) == getattr(defaults, name):
            setattr(config, name, value)
