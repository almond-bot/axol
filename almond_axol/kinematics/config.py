"""KinematicsConfig dataclass with cost weights and solver parameters for KinematicsSolver."""

from __future__ import annotations

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
            that only genuine convergence trips it: 1e-6 halves the residual
            freeze rate at the reach boundary versus 1e-4 (the iteration
            budget, not this tolerance, bounds the solve time).
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
        max_joint_delta: Maximum joint change per :meth:`KinematicsSolver.ik` call,
            in radians. This bounds the release velocity of any residual solver
            stall (stall ticks accumulate target error that is paid back at
            this rate), so it directly caps command-velocity spikes at
            kinematic boundaries and near shoulder singularities: 0.02 rad at
            the ~72 Hz frame rate is ~1.4 rad/s per joint, comfortably above
            normal tracking demand but half the old 0.0345-rad cap that let
            catch-up sweeps hit 2.5+ rad/s on s1/s2.
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
    cost_tolerance: float = 1e-6
    lambda_initial: float = 1e-2
    lambda_factor: float = 10.0
    max_joint_delta: float = 0.02
    max_reach: float = 0.8
