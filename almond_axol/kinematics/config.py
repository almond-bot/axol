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
        elbow_weight: Weight on elbow position hints (position only, no
            orientation). ``0`` disables elbow tracking entirely (the default):
            the headset's body-model elbow is inferred rather than tracked and
            proved unreliable on hardware, so the arm's swivel is left to the
            posture attractor, manipulability, and rest damping instead —
            the standard setup for stacks without operator-elbow data. When
            enabled, targets are projected onto the robot's reachable elbow
            sphere (current shoulder-to-elbow radius) so only the swivel
            direction survives — the raw target's radial component is
            unreachable by construction (human arm proportions) and would
            inject a permanent residual that fights the EE cost through the
            shared shoulder joints — and the hint fades to zero for overhead
            targets (see ``elbow_fade_band``).
        rest_weight: Weight penalising deviation from the current joint configuration.
            Acts as a per-step damping term; uses q_current as the target.
            This is what keeps millimetre-scale target noise from churning
            the arm's null space: recorded teleop showed 1.5 mm of hand
            tremor coming out of the solver as up to 1° of anti-correlated
            joint ripple (elbow vs shoulder_1 at r=-0.96) that cancelled at
            the hand — internal motion the task cost is blind to. Replaying
            those bursts through the solver: 15 cut the churn 20-35% with no
            measurable end-effector tracking cost; 30 cut it 2-5x but added
            ~30% EE lag during fast reaches, so 15 is the sweet spot.
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
        self_collision: Whether the arm<->torso (``base`` / ``s1``) collision
            model is part of the solve at all. ``False`` (the default) turns
            base collision detection off completely: no capsule model is
            built, the soft self-collision cost is left out of the IK cost
            graph, the hard upper-arm/base final-step guard is skipped, and
            the return-to-rest planner interpolates in joint space
            with joint limits only. Joint limits are still clamped on every
            step. Set ``True`` to restore the collision-aware solve described
            under ``self_collision_margin`` / ``self_collision_weight``.
        self_collision_margin: Minimum clearance (m) enforced between collision
            bodies. Keep this below the arm-torso clearance of normal teleop
            poses: a margin that is already violated at the folded rest pose
            (e.g. the old 0.1 default) keeps the nonsmooth collision hinge
            active during ordinary tracking, which makes the trust-region
            solver reject steps — the arm freezes while small target errors
            accumulate, then lurches once the error is large enough to punch
            through (the teleop "stuck, then jumps" failure). 0.025 matches
            ``VRTeleopConfig.reset_collision_margin``. This value is the
            *ceiling*: each pair's actual activation distance is derived from
            its home-pose clearance (``min(home_clearance - 2 mm, this
            value)``, possibly *negative* — see
            :func:`almond_axol.kinematics.model.collision_cost_params`), so
            pairs that normally operate near their shell — the wrists and
            grippers passing 10-17 mm from the base during close-over-table
            work, the elbow capsules whose conservative fits interpenetrate
            by 1.4 mm at rest — stay silent in their ordinary envelope
            instead of holding the cost hinge active there (measured at up
            to 59% of frames in recorded deburring sessions, which pulsed
            weight-150 gradients into the arm as path-specific jitter; the
            formerly floor-clamped elbow pair tripled per-tick solver output
            acceleration during front-of-torso reaches and kicked the elbow
            at every shell crossing). Distal arm/base pairs remain active even
            when their conservative capsules overlap at the physically safe
            home pose. The upper-arm/base contact pair additionally has an
            independent final-step guard calibrated from that safe reference,
            so the pose task cannot push either arm through the base.
        self_collision_weight: Weight on the self-collision penalty. 150 was
            set by replaying recorded close-to-body teleop episodes through
            the solver offline: at 75 the task cost dragged the elbow up to
            7 mm inside the capsule surface during deburring-style work; at
            150 the worst dip is +0.3 mm at a cost of +0.7% jerk and no
            change in EE error, freezes, or solve time.
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
        max_joint_delta: Maximum joint change per :meth:`KinematicsSolver.ik` call,
            in radians (infinity-norm bound). The clamp is direction-preserving:
            an over-limit step is scaled as a whole vector, never clipped per
            joint, so the commanded configuration stays on the solver's path.
        max_reach: Asymptotic cap (m) on the distance from shoulder to
            end-effector target. Targets are *soft*-clamped: unchanged below
            ``reach_soft_start``, then smoothly saturated so they approach but
            never reach ``max_reach``. This is a runaway-target guard only:
            the arm's true straight-elbow boundary measured from the shoulder
            frame is *direction-dependent* (~0.69 m in the worst directions,
            ~0.78 m reaching forward, because the shoulder joint axes do not
            intersect the s2 link origin), so no fixed sphere can hug it —
            a cap tight enough to protect one direction amputates several cm
            of legitimate reach in another. Smoothness at the extension
            boundary is instead handled by the manipulability-keyed adaptive
            damping, which reads the actual Jacobian and therefore knows the
            real boundary in every direction.
        reach_soft_start: Distance (m) from the shoulder at which the reach
            soft-clamp starts to engage. The mapping is C1-smooth (identity
            slope at the knee), so crossing it introduces no velocity
            discontinuity in the target. Set at the arm's maximum physical
            straight-elbow distance so it only ever engages on targets beyond
            any reachable pose.
        manip_damping_threshold: Translational (Yoshikawa) manipulability below
            which per-arm adaptive damping ramps in. Calibrated values for this
            arm: ~0.027 at rest, ~0.037 mid-workspace, ~0.009 at 7 degrees from
            elbow-straight, ~0 at the singularity — 0.015 keeps damping fully
            off in normal use and fully on at the boundary.
        manip_damping_boost: Extra rest-cost weight added to an arm's joints at
            zero manipulability. Ramps quadratically from 0 at
            ``manip_damping_threshold`` (Chiaverini-style damped least
            squares), slowing the solver smoothly near singular configurations
            instead of letting it thrash or lurch. ``0`` disables.
        limit_damping_margin: Distance (rad) from a joint limit inside which
            that joint's rest-cost damping ramps in (quadratically, reaching
            ``manip_damping_boost`` at the limit). Gated on approach: only a
            joint *moving toward* its nearby limit is damped — one parked
            against a limit by task pressure (a wrist during ordinary
            manipulation) stays free, so the damping can't fight the task.
            Per-joint, so one joint brushing a limit never slows the rest of
            the arm.
        elbow_fade_band: Height band (m) above the shoulder over which the
            elbow-hint weight fades to zero. The headset's body-model elbow is
            inferred (not tracked) and becomes unreliable when the hands work
            at or above shoulder height — fading it out stops garbage swivel
            targets from thrashing the null space exactly where the shoulder
            nears its joint limits.
    """

    pos_weight: float = 50.0
    ori_weight: float = 10.0
    elbow_weight: float = 0.0
    rest_weight: float = 15.0
    posture_weight: float = 5.0
    manipulability_weight: float = 0.05
    limit_weight: float = 75.0
    self_collision: bool = False
    self_collision_margin: float = 0.025
    self_collision_weight: float = 150.0
    max_iterations: int = 8
    cost_tolerance: float = 1e-4
    lambda_initial: float = 1e-2
    lambda_factor: float = 10.0
    max_joint_delta: float = 0.0055 * 2 * math.pi
    max_reach: float = 0.82
    reach_soft_start: float = 0.78
    manip_damping_threshold: float = 0.015
    manip_damping_boost: float = 60.0
    limit_damping_margin: float = 0.12
    elbow_fade_band: float = 0.15
