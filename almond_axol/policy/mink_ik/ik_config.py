"""Configuration for Axol Mink IK.

Solver parameters live in a lightweight dataclass so the numerical tracking
stack can be imported without JAX. The defaults describe a 30 Hz controller;
callers with another cadence must pass the actual solve rate for the posture
speed gate. Joint displacement limits are per call, independent of cadence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

# Keep collision geometry and root-frame conventions stable for the solver.
# Public Axol world poses are converted to this model frame at the boundary.
PINNED_URDF: Path = Path(__file__).resolve().parent / "assets" / "axol_mink.urdf"

# Per-call joint displacement budget in radians: approximately 0.0346 rad,
# or 1.04 rad/s at the default 30 Hz cadence. The QP reserves a small margin
# within this limit so the componentwise clamp is only a backstop.
SERVING_STEP_BUDGET_RAD: float = 0.0055 * 2 * math.pi


@dataclass
class MinkIKConfig:
    """Cost weights and parameters for :class:`~almond_axol.policy.mink_ik.MinkIK`.

    Attributes:
        mink_position_cost: QP weight on end-effector position error.
        mink_orientation_cost: QP weight on end-effector orientation error.
        mink_elbow_cost: QP weight on the operator-elbow hint (0 disables it).
        mink_posture_cost: QP weight of the null-space posture attractor.
        mink_posture_speed_gate: EE-target speed (m/s) at which the posture
            attractor's effective weight halves (0 disables gating).
        mink_posture_gate_tau: Time constant (s) of the one-pole low-pass on
            the speed-gate scale (0 uses the raw gate).
        mink_posture_target: ``"rest"`` uses the settled rest pose; ``"engage"``
            updates the attractor at each :meth:`MinkTracker.set_posture` call.
        mink_lm_damping: Per-task Levenberg-Marquardt damping.
        mink_damping: Global Tikhonov damping on joint velocity.
        mink_iterations: Solve/integrate iterations per call.
        mink_dt: Integration step per iteration. Keep 1.0 for the displacement
            formulation used by these weights and velocity limits.
        mink_collision: Enable hard torso-to-arm self-collision constraints.
        mink_collision_min_distance: Minimum allowed geometry clearance (m).
        mink_collision_detection_distance: Clearance (m) at which collision
            constraints activate.
        mink_solve_hz: Actual solve cadence used to estimate target speed.
        max_joint_delta: Public per-call displacement bound, initialized to
            the same value as ``mink_max_joint_delta``.
        mink_max_joint_delta: Per-call joint displacement budget (rad) of the
            QP velocity constraints and the componentwise backstop clamp.
        max_reach: Shoulder-to-EE radial reach clamp (m), applied before IK.
    """

    mink_position_cost: float = 1.0
    mink_orientation_cost: float = 0.25
    mink_elbow_cost: float = 0.0
    mink_posture_cost: float = 0.05
    mink_posture_speed_gate: float = 0.15
    mink_posture_gate_tau: float = 1.0
    mink_posture_target: str = "rest"
    mink_lm_damping: float = 0.01
    mink_damping: float = 0.8
    mink_iterations: int = 4
    mink_dt: float = 1.0
    mink_collision: bool = True
    mink_collision_min_distance: float = 0.01
    mink_collision_detection_distance: float = 0.06
    mink_solve_hz: float = 30.0
    max_joint_delta: float = SERVING_STEP_BUDGET_RAD
    mink_max_joint_delta: float = SERVING_STEP_BUDGET_RAD
    max_reach: float = 0.8
