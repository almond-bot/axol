"""JAX-free, prevalidated Cartesian reset trajectories for the Mink backend.

Both tool tips follow straight lines with smooth timing and SO(3) orientation
interpolation. A separate QP state leaves the live tracker's history untouched.
Planning fails before playback when the line, posture, or collision constraints
cannot be satisfied; collision constraints are never discarded to get a result.

Collision checks cover the robot's torso/arm geometry, as in the tracking stack.
They do not model objects in the environment.
"""

from __future__ import annotations

import copy
import math
import time
from typing import TYPE_CHECKING

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..constants import GRIPPER_TIP_OFFSET, Joint, urdf_body_name
from ..policy.mink_ik.ik_mujoco_model import body_geom_ids

if TYPE_CHECKING:
    from .mink_backend import MinkKinematicsSolver


# Measured elbows can straddle the zero-angle limit after a supported park.
# Recover at most 0.57 degrees without snapping the measured starting command.
_START_JOINT_TOLERANCE = 0.01
_CLEARANCE_TOLERANCE = 5e-5


class MinkPlanningError(RuntimeError):
    """A complete reset satisfying the configured limits could not be planned."""


def _ease(t: float) -> float:
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


class _ResetProblem:
    """Strict QP and independent geometry checks in the model's own world frame."""

    def __init__(
        self,
        solver: MinkKinematicsSolver,
        speed: float,
        dt: float,
        collision_margin: float = 0.01,
    ) -> None:
        self.model = copy.copy(solver._ik.model)
        # MuJoCo 3.11 native CCD can report zero distance for separated mesh
        # faces at near-zero arm angles. Use libccd on this private planning
        # model; the live tracker and its validated numerics are unchanged.
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        model = self.model
        self.names = solver.joint_names
        joints = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.names]
        )
        self.qadr = model.jnt_qposadr[joints]
        dofs = model.jnt_dofadr[joints]
        self.dofs = dofs
        self.frozen = np.array([], dtype=int)
        self.frozen_q = np.array([])
        self.ranges = model.jnt_range[joints]
        self.validation_ranges = self.ranges.copy()
        self.configuration = mink.Configuration(model)
        self.check_data = mujoco.MjData(model)
        self.dt = dt
        self._fromto = np.zeros(6)
        self.max_velocity = 1.875 * speed
        self.tasks = [
            mink.FrameTask(
                frame_name=urdf_body_name(Joint.GRIPPER, is_left=is_left),
                frame_type="body",
                position_cost=20.0,
                orientation_cost=5.0,
                lm_damping=1e-6,
            )
            for is_left in (True, False)
        ]
        self.body_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, task.frame_name)
            for task in self.tasks
        ]
        self.posture = mink.PostureTask(model, cost=0.1)
        self.constraints = [
            mink.DofFreezingTask(
                model, dof_indices=sorted(set(range(model.nv)) - set(dofs))
            )
        ]
        self.limits = [
            mink.ConfigurationLimit(model, gain=1.0),
            mink.VelocityLimit(model, dict.fromkeys(self.names, self.max_velocity)),
        ]
        # Keep near-home pairs using a home-relative clearance instead of
        # deleting them. The mesh fits include fixed overlap at some joints;
        # never authorize deeper overlap than that known reference plus 2 mm.
        torso = body_geom_ids(model, ["base", "s1"])
        # Shoulder mounts s2/s3 rotate captive in the torso and are excluded,
        # matching the vendor reset model. Their mesh overlap is intentional.
        arms = body_geom_ids(
            model,
            [
                f"{side}_{suffix}"
                for side in ("left", "right")
                for suffix in ("e1", "e2", "w0", "w1", "w2", "gripper")
            ],
        )
        self.check_data.qpos[:] = 0.0
        mujoco.mj_forward(model, self.check_data)
        groups: dict[float, list[tuple[list[int], list[int]]]] = {}
        for a in torso:
            for b in arms:
                home_distance = mujoco.mj_geomDistance(
                    model, self.check_data, a, b, 1.0, self._fromto
                )
                floor = min(collision_margin, home_distance - 0.002)
                groups.setdefault(floor, []).append(([a], [b]))
        self.pairs: list[tuple[int, int, float]] = []
        for floor, pairs in groups.items():
            limit = mink.CollisionAvoidanceLimit(
                model,
                geom_pairs=pairs,
                minimum_distance_from_collisions=floor,
                collision_detection_distance=max(0.06, floor + 0.02),
            )
            self.limits.append(limit)
            # Use Mink's filtered pair set for both QP and post-solve checks.
            self.pairs.extend((a, b, floor) for a, b in limit.geom_id_pairs)
        self.clearance_floors = {(a, b): floor for a, b, floor in self.pairs}

    def allow_start_recovery(self, q: np.ndarray) -> None:
        """Permit measured starts inside the clearance buffer, never deeper contact.

        The QP keeps its full nominal collision limits and must separate such
        pairs. Validation permits only non-decreasing clearance until the buffer
        is restored, accounting for measured/tracking error at the start.
        """
        if q.shape != (14,) or not np.isfinite(q).all():
            raise MinkPlanningError("reset start: expected 14 finite arm joint angles")
        if np.any(q < self.ranges[:, 0] - _START_JOINT_TOLERANCE) or np.any(
            q > self.ranges[:, 1] + _START_JOINT_TOLERANCE
        ):
            raise MinkPlanningError(
                "reset start: joint limit recovery exceeds 0.01 rad"
            )
        self.validation_ranges[:, 0] = np.minimum(self.ranges[:, 0], q)
        self.validation_ranges[:, 1] = np.maximum(self.ranges[:, 1], q)
        self.geometry(q)
        for a, b, floor in self.pairs:
            distance = mujoco.mj_geomDistance(
                self.model, self.check_data, a, b, 0.1, self._fromto
            )
            if distance < min(0.0, floor) - _CLEARANCE_TOLERANCE:
                raise MinkPlanningError("reset start: torso/arm collision")
            self.clearance_floors[a, b] = min(floor, distance)
        self.validate(q, "reset start")

    def freeze_unchanged_arms(self, q_from: np.ndarray, q_to: np.ndarray) -> None:
        self.frozen = np.array(
            [
                i
                for arm in (range(7), range(7, 14))
                if np.array_equal(q_from[list(arm)], q_to[list(arm)])
                for i in arm
            ],
            dtype=int,
        )
        if self.frozen.size:
            self.frozen_q = q_from[self.frozen].copy()
            self.constraints.append(
                mink.DofFreezingTask(
                    self.model, dof_indices=self.dofs[self.frozen].tolist()
                )
            )

    def advance_clearance(self, q: np.ndarray) -> None:
        self.validation_ranges[:, 0] = np.maximum(
            self.validation_ranges[:, 0], np.minimum(self.ranges[:, 0], q)
        )
        self.validation_ranges[:, 1] = np.minimum(
            self.validation_ranges[:, 1], np.maximum(self.ranges[:, 1], q)
        )
        self.geometry(q)
        for a, b, floor in self.pairs:
            distance = mujoco.mj_geomDistance(
                self.model, self.check_data, a, b, 0.1, self._fromto
            )
            self.clearance_floors[a, b] = max(
                self.clearance_floors[a, b], min(floor, distance)
            )

    def full_q(self, q: np.ndarray) -> np.ndarray:
        full = np.zeros(self.model.nq)
        full[self.qadr] = q
        return full

    def geometry(self, q: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        self.check_data.qpos[:] = self.full_q(q)
        mujoco.mj_forward(self.model, self.check_data)
        offset = np.asarray(GRIPPER_TIP_OFFSET)
        result = []
        for body in self.body_ids:
            rotation = self.check_data.xmat[body].reshape(3, 3).copy()
            tip = self.check_data.xpos[body] + rotation @ offset
            result.append((tip.copy(), rotation))
        return result

    def validate(self, q: np.ndarray, label: str) -> None:
        if q.shape != (14,) or not np.isfinite(q).all():
            raise MinkPlanningError(f"{label}: expected 14 finite arm joint angles")
        if np.any(q < self.validation_ranges[:, 0] - 1e-6) or np.any(
            q > self.validation_ranges[:, 1] + 1e-6
        ):
            raise MinkPlanningError(f"{label}: arm joint limit exceeded")
        self.geometry(q)
        for a, b, _nominal_floor in self.pairs:
            floor = self.clearance_floors[a, b]
            distance = mujoco.mj_geomDistance(
                self.model, self.check_data, a, b, max(0.1, floor + 0.01), self._fromto
            )
            if distance < floor - _CLEARANCE_TOLERANCE:
                raise MinkPlanningError(
                    f"{label}: torso/arm clearance violated "
                    f"(geoms {a}/{b}: {distance:.6f} m < {floor:.6f} m)"
                )

    def step(
        self,
        tips: list[tuple[np.ndarray, np.ndarray]],
        posture: np.ndarray,
    ) -> np.ndarray:
        offset = np.asarray(GRIPPER_TIP_OFFSET)
        for task, (tip, rotation) in zip(self.tasks, tips, strict=True):
            task.set_target(
                mink.SE3.from_rotation_and_translation(
                    mink.SO3.from_matrix(rotation), tip - rotation @ offset
                )
            )
        self.posture.set_target(self.full_q(posture))
        try:
            velocity = mink.solve_ik(
                self.configuration,
                [*self.tasks, self.posture],
                self.dt,
                "daqp",
                damping=1e-5,
                # Only the explicitly bounded measured-start recovery can be
                # outside the model limits; independent checks enforce monotonic
                # recovery. ConfigurationLimit remains present in every QP.
                safety_break=False,
                limits=self.limits,
                constraints=self.constraints,
            )
        except Exception as exc:
            raise MinkPlanningError(
                "Mink reset QP failed with collision limits enabled"
            ) from exc
        if not np.isfinite(velocity).all():
            raise MinkPlanningError("Mink reset QP returned non-finite velocity")
        self.configuration.integrate_inplace(velocity, self.dt)
        result = self.configuration.q[self.qadr].copy()
        # Remove equality-solver roundoff on arms frozen inside the QP.
        result[self.frozen] = self.frozen_q
        self.configuration.update(self.full_q(result))
        return result


def plan_mink_trajectory(
    solver: MinkKinematicsSolver,
    q_from: np.ndarray,
    q_to: np.ndarray,
    *,
    speed: float,
    rate: float,
    min_duration: float,
    linear_speed: float = 0.1,
    angular_speed: float = 0.5,
    collision_margin: float = 0.01,
    position_tolerance: float = 0.005,
    orientation_tolerance: float = 0.03,
    max_planning_time: float = 20.0,
) -> list[np.ndarray]:
    """Return a fully checked reset ending at the requested joint posture.

    ``speed`` retains teleop's average joint-speed unit (rad/s). Cartesian
    translation and rotation also bound duration. Quintic timing has a 1.875x
    peak speed. Every output tick and each inter-tick midpoint is checked for
    joint limits and torso/arm clearance. Samples must stay within the requested
    Cartesian tolerances; an unreachable straight line raises before playback.

    Redundant arms may need a short final posture settle while their tool poses
    remain at the goal within tolerance. The final returned joint vector is
    exactly ``q_to``; a solver hold or partial plan is never treated as success.
    """
    values = (
        speed,
        rate,
        min_duration,
        linear_speed,
        angular_speed,
        collision_margin,
        position_tolerance,
        orientation_tolerance,
        max_planning_time,
    )
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(
            "Mink reset speeds, timing, and tolerances must be positive and finite"
        )
    q_from = np.asarray(q_from, dtype=np.float64)
    q_to = np.asarray(q_to, dtype=np.float64)
    if q_from.shape != (14,) or q_to.shape != (14,):
        raise ValueError("Mink reset requires 14 arm joint angles at each endpoint")
    deadline = time.monotonic() + max_planning_time
    problem = _ResetProblem(solver, speed, 1.0 / rate, collision_margin)
    problem.validate(q_to, "reset goal")
    problem.allow_start_recovery(q_from)
    problem.freeze_unchanged_arms(q_from, q_to)
    if np.array_equal(q_from, q_to):
        return [q_from.astype(np.float32), q_to.astype(np.float32)]
    start = problem.geometry(q_from)
    goal = problem.geometry(q_to)
    rotations = [
        Rotation.from_matrix(a[1].T @ b[1]).as_rotvec()
        for a, b in zip(start, goal, strict=True)
    ]
    duration = max(min_duration, float(np.max(np.abs(q_to - q_from))) / speed)
    for (p0, _), (p1, _), rotation in zip(start, goal, rotations, strict=True):
        duration = max(
            duration,
            np.linalg.norm(p1 - p0) / linear_speed,
            np.linalg.norm(rotation) / angular_speed,
        )
    n_steps = max(2, math.ceil(duration * rate))
    if n_steps > 10000:
        raise MinkPlanningError("Mink reset exceeds the trajectory size budget")
    problem.configuration.update(problem.full_q(q_from))
    trajectory = [q_from.astype(np.float32)]
    previous_tips = start

    def append(q: np.ndarray, tips: list[tuple[np.ndarray, np.ndarray]]) -> None:
        nonlocal previous_tips
        if time.monotonic() > deadline:
            raise MinkPlanningError("Mink reset exceeded its planning time budget")
        previous = trajectory[-1].astype(np.float64)
        if np.max(np.abs(q - previous)) > problem.max_velocity / rate + 1e-6:
            raise MinkPlanningError("Mink reset exceeds its joint velocity limit")
        midpoint_tips = [
            (
                (p0 + p1) * 0.5,
                r0
                @ Rotation.from_rotvec(
                    0.5 * Rotation.from_matrix(r0.T @ r1).as_rotvec()
                ).as_matrix(),
            )
            for (p0, r0), (p1, r1) in zip(previous_tips, tips, strict=True)
        ]
        for sample, targets, label in (
            ((previous + q) * 0.5, midpoint_tips, "reset between samples"),
            (q, tips, "reset waypoint"),
        ):
            problem.validate(sample, label)
            actual = problem.geometry(sample)
            for (position, rotation), (target, target_rotation) in zip(
                actual, targets, strict=True
            ):
                angle = np.linalg.norm(
                    Rotation.from_matrix(rotation.T @ target_rotation).as_rotvec()
                )
                if (
                    np.linalg.norm(position - target) > position_tolerance
                    or angle > orientation_tolerance
                ):
                    raise MinkPlanningError(
                        "Mink reset cannot follow the straight Cartesian path"
                    )
        problem.advance_clearance(q)
        trajectory.append(q.astype(np.float32))
        previous_tips = tips

    for i in range(1, n_steps + 1):
        alpha = _ease(i / n_steps)
        tips = [
            (
                (1 - alpha) * p0 + alpha * p1,
                rotation @ Rotation.from_rotvec(alpha * delta).as_matrix(),
            )
            for (p0, rotation), (p1, _), delta in zip(
                start, goal, rotations, strict=True
            )
        ]
        posture = (1 - alpha) * q_from + alpha * q_to
        q = problem.step(tips, posture)
        append(q, tips)

    # Resolve null-space posture while holding the final tool pose. Starting
    # from an achieved Cartesian target avoids a discontinuous final joint snap.
    for _ in range(math.ceil(3.0 * rate)):
        if np.max(np.abs(q - q_to)) <= 0.003:
            break
        q = problem.step(goal, q_to)
        append(q, goal)
    if np.max(np.abs(q - q_to)) > 0.003:
        raise MinkPlanningError(
            "Mink reset reached the tool pose but not the requested joint posture"
        )
    # A short smooth joint finish makes the endpoint exact. Its entire path is
    # still checked against the goal tool pose and the same collision limits.
    final_start = q.copy()
    n_finish = max(2, math.ceil(max(0.1, np.max(np.abs(q_to - q)) / speed) * rate))
    for i in range(1, n_finish + 1):
        q = final_start + _ease(i / n_finish) * (q_to - final_start)
        append(q, goal)
    trajectory[-1] = q_to.astype(np.float32)
    return trajectory
