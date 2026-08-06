"""KinematicsConfig dataclass with cost weights and solver parameters for KinematicsSolver."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class KinematicsConfig:
    """Cost weights and solver parameters for the teleop IK backends.

    The ``backend`` field selects the solver implementation; the ``pos_weight``
    … ``max_reach`` block parametrises the original pyroki Levenberg-Marquardt
    solver (``pyroki-lm``), the ``diff_*`` block parametrises the differential
    (velocity-level) backends, and the ``elbow_swivel`` / ``max_target_err_*``
    fields tune the backend-independent target conditioning.

    All weights are unitless scale factors passed directly to pyroki cost
    functions. Higher values make the solver prioritise that term more strongly.

    Attributes:
        backend: IK solver implementation. ``pink-qp`` (default) is Pinocchio
            QP differential IK with hard joint/velocity limits — the winner of
            the 2026-07 offline solver bake-off (see ``bench.py``). The other
            candidates are kept for on-hardware A/B comparison: ``pyroki-lm``
            (the original per-frame Levenberg-Marquardt full solve),
            ``pyroki-diff`` (single damped Gauss-Newton step per tick on the
            pyroki/JAX model), ``mink-qp`` (MuJoCo QP differential IK, same
            structure as pink-qp), and ``dls`` (custom NumPy damped least
            squares with null-space posture and limit clamping).
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
        max_reach: Maximum allowed distance (m) from shoulder to end-effector
            target. The worker additionally caps this at the arm's physical
            extension minus a small margin (see ``_MAX_REACH_MARGIN_M``), so
            the effective limit is ~0.70 m: this arm fully extends at
            ~0.73 m, and a value beyond that (like this 0.8 default on its
            own) would let operators command past full extension and grind
            the straight-elbow singularity.
        diff_position_cost: Differential backends — weight on end-effector
            position error (per metre).
        diff_orientation_cost: Differential backends — weight on end-effector
            orientation error (per radian).
        diff_elbow_cost: Differential backends — weight on the elbow (swivel)
            reference, per metre. Position only, and below the EE position
            cost so end-effector tracking always wins. **Default 0.0 — the
            elbow task is off**, on direct operator feedback: every reference
            tried (the headset's inferred elbow raw, gated and smoothed, and
            an elevation prior with a singularity-gated weight) felt worse on
            hardware than leaving the swivel to the solver's damping and
            posture pull, because any elbow pull strong enough to matter
            competes with hand tracking in exactly the poses (reaches above
            the shoulder) where the shoulder is nearly singular. The swivel
            conditioning machinery stays available: setting a positive cost
            re-enables it (see ``elbow_swivel``).
        diff_posture_cost: Differential backends — weight of the preferred-
            posture attractor. Acts purely in the task null space when small.
            Must stay well below ``diff_elbow_cost``: the attractor pulls the
            swivel back toward the rest pose (elbow-down) and at 0.5 it wins
            outright — the same sweep showed the elbow pinned near the rest
            swivel (~89 deg mean error) regardless of the operator's hint.
        diff_damping: Tikhonov regularisation added to the QP Hessian.
            The primary singularity-robustness knob: larger values slow the
            solution near singular configurations instead of letting joint
            velocities blow up. 1.0 cuts jerk ~30x at the reach boundary
            (bench ``singularity`` scenario) with no measurable tracking cost
            on nominal motion.
        diff_lm_damping: Levenberg-Marquardt damping scaled by task error
            (pink ``lm_damping``). Stabilises far-from-target steps; tuned
            together with ``diff_damping`` on the bench.
        diff_max_joint_vel: Hard per-joint velocity limit (rad/s) enforced by
            the differential backends (QP inequality / DLS clamp). Do not
            lower it to make constraints better behaved: halving it to pi was
            tried and made the whole arm lag fast operator motion (a wrist
            flip alone wants 3-6 rad/s), which reads as the robot being
            "messed up" long before any constraint does.
        diff_iters: Integration sub-steps per :meth:`ik` call. More sub-steps
            converge closer to the target per tick at proportional CPU cost.
        diff_collision_margin: Minimum arm<->torso clearance (m) enforced as a
            hard constraint by ``pink-qp`` (collision barrier) and ``mink-qp``
            (collision-avoidance limit); ``pyroki-diff`` and ``dls`` have no
            collision support. Pairs are arm links versus the base column and
            torso yoke only — never arm<->arm. The URDF collision meshes are
            tight boxes (the base is a 0.26 x 0.12 x 0.80 m column), so the
            constraint only activates on genuine near-contact; the arms hang
            ~27 mm from the column at rest, which bounds how large the margin
            plus prune buffer can be before the forearm pairs get pruned as
            "close by construction" (see ``_PRUNE_THRESHOLD`` in the pink
            backend; the pair set is fixed and does not change with the
            margin). The margin trades protection against how often ordinary
            motion rides the constraint, and riding is what operators report
            as jitter and "the arm fighting me": on a captured session,
            0.02 produced 2.8x the whole-run jerk of 0.01 with no difference
            in penetration (both zero — the hulls the barrier acts on are
            already conservative convex boxes). Set to 0.0 to disable
            entirely; on the same session that meant ~11% of ticks in
            contact with the column hulls, i.e. real metal contact.
        elbow_swivel: When true, re-project the operator's elbow hint onto the
            robot's own upper-arm/forearm geometry (the swivel circle around
            the shoulder->wrist axis) so the hint is always reachable. Fixes
            the elbow-down bias of tracking the human elbow position directly.
        max_target_err_lin: Cap (m) on the position error between the current
            end-effector pose and the commanded target fed to the solver. The
            target direction is preserved (Drake-style feasibility scaling),
            so an out-of-reach or lagging target degrades into a smooth,
            bounded pull instead of a solver-dependent lurch.
        max_target_err_ang: Cap (radians) on the orientation error fed to the
            solver, scaled jointly with ``max_target_err_lin``.
    """

    backend: str = "pink-qp"

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

    # Differential (velocity-level) backend parameters: pink-qp, mink-qp,
    # pyroki-diff, dls. Damping values are tuned for pink-qp (the default).
    diff_position_cost: float = 50.0
    diff_orientation_cost: float = 10.0
    diff_elbow_cost: float = 0.0
    diff_posture_cost: float = 0.1
    diff_damping: float = 1.0
    diff_lm_damping: float = 4.0
    diff_max_joint_vel: float = 2 * math.pi
    diff_iters: int = 2
    diff_collision_margin: float = 0.01

    # Backend-independent target conditioning (applied in IKWorker).
    elbow_swivel: bool = True
    max_target_err_lin: float = 0.20
    max_target_err_ang: float = 1.5
