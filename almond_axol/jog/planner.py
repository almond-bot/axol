"""IK and joint-space planning for jog moves.

A jog move names a gripper-tip pose for one arm. :class:`JogPlanner` runs
Axol's IK solver (the same one teleop uses) to convergence from the arm's
current configuration, so the result is the reachable pose closest to the
target, then plans a collision-aware joint-space trajectory to it. The other
arm stays where it is.

Poses are in the torso frame IK uses (+x forward, +y left, +z up, the URDF's
``root`` link), at the gripper tips (:data:`~almond_axol.constants.GRIPPER_TIP_OFFSET`
out from the mount).
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..constants import GRIPPER_TIP_OFFSET
from ..kinematics.path import _apply_offset, _remove_offset, _rot_angle

Arm = Literal["left", "right"]
Pose = tuple[np.ndarray, np.ndarray]

# Convergence of the closest-pose search. Each IK call moves a bounded step,
# so the search repeats it until the tip lands within these tolerances or
# stops getting closer.
_POS_TOLERANCE_M = 5e-4
_ORI_TOLERANCE_RAD = np.radians(0.5)
_MAX_ITERATIONS = 400
_STALL_ITERATIONS = 15
# Per-call step multiplier: 4x the solver's per-tick joint bound, the most
# teleop ever lets a single solve move.
_DELTA_SCALE = 4.0
# A solution counts as reaching the target within these errors.
REACH_POS_TOLERANCE_M = 2e-3
REACH_ORI_TOLERANCE_RAD = np.radians(2.0)
# Path preview resolution: tip positions sampled along the trajectory.
_PATH_POINTS = 48

_plan_ids = itertools.count(1)


@dataclass
class IkSolution:
    """The closest reachable configuration to a target tip pose."""

    q: np.ndarray
    tip: Pose
    position_error: float
    orientation_error: float

    @property
    def reached(self) -> bool:
        return bool(
            self.position_error <= REACH_POS_TOLERANCE_M
            and self.orientation_error <= REACH_ORI_TOLERANCE_RAD
        )


@dataclass
class JogPlan:
    """A planned move from the current configuration to an IK solution."""

    arm: Arm
    target: Pose
    solution: IkSolution
    q_from: np.ndarray
    trajectory: list[np.ndarray]
    rate: float
    path: list[np.ndarray]
    id: int = field(default_factory=lambda: next(_plan_ids))

    @property
    def duration(self) -> float:
        return len(self.trajectory) / self.rate


class JogPlanner:
    """Thread-safe IK and trajectory planning on one :class:`KinematicsSolver`.

    The solver keeps per-call state, so every use goes through one lock.
    """

    def __init__(self, solver: Any, tool_offset=GRIPPER_TIP_OFFSET) -> None:
        self._solver = solver
        self._offset = np.asarray(tool_offset, dtype=np.float32)
        self._lock = threading.Lock()

    @property
    def solver(self) -> Any:
        return self._solver

    def indices(self, arm: Arm) -> list[int]:
        return list(
            self._solver.left_indices if arm == "left" else self._solver.right_indices
        )

    def tips(self, q: np.ndarray, *, blocking: bool = True) -> tuple[Pose, Pose] | None:
        """``(left, right)`` gripper-tip poses for a full joint vector.

        With ``blocking=False`` this returns None instead of waiting for a
        plan in progress, so a control loop never stalls on it.
        """
        if not self._lock.acquire(blocking=blocking):
            return None
        try:
            left, right = self._solver.fk(np.asarray(q, dtype=np.float32))
        finally:
            self._lock.release()
        return _apply_offset(left, self._offset), _apply_offset(right, self._offset)

    def _tip_position(self, q: np.ndarray, side: int) -> np.ndarray:
        tips = self.tips(q)
        assert tips is not None
        return tips[side][0]

    def solve(
        self,
        q_now: np.ndarray,
        arm: Arm,
        target: Pose,
        lift_raised: float | None = None,
    ) -> IkSolution:
        """Closest reachable configuration putting ``arm``'s tip at ``target``.

        Repeats the solver's bounded step from ``q_now`` until the tip is on
        the target or stops getting closer, keeping the best configuration
        seen. The posture attractor follows the solution so it does not pull
        the arm back toward where it started.
        """
        solver = self._solver
        q_now = np.asarray(q_now, dtype=np.float32)
        position = np.asarray(target[0], dtype=np.float32)
        rotation = np.asarray(target[1], dtype=np.float32)
        mount = _remove_offset((position, rotation), self._offset)
        own = self.indices(arm)
        other = [i for i in range(len(q_now)) if i not in own]
        with self._lock:
            if lift_raised is not None:
                solver.set_lift(lift_raised)
            # The other arm aims at where it already is, which keeps the solve
            # on the warmed-up two-pose graph; its joints are pinned after.
            other_mount = solver.fk(q_now)[1 if arm == "left" else 0]
            poses = (mount, other_mount) if arm == "left" else (other_mount, mount)
            q = q_now.copy()
            # (score, position error, orientation error, q); the score weighs
            # a radian of orientation like 5 cm of position.
            best: tuple[float, float, float, np.ndarray] | None = None
            stalled = 0
            for _ in range(_MAX_ITERATIONS):
                solver.set_posture_pose(q)
                q = solver.ik(
                    q, left_pose=poses[0], right_pose=poses[1], delta_scale=_DELTA_SCALE
                )
                q[other] = q_now[other]
                reached = solver.fk(q)[0 if arm == "left" else 1]
                tip = _apply_offset(reached, self._offset)
                pos_err = float(np.linalg.norm(tip[0] - position))
                ori_err = _rot_angle(tip[1], rotation)
                score = pos_err + 0.05 * ori_err
                if best is None or score < best[0] - 1e-6:
                    best = (score, pos_err, ori_err, q.copy())
                    stalled = 0
                else:
                    stalled += 1
                if pos_err <= _POS_TOLERANCE_M and ori_err <= _ORI_TOLERANCE_RAD:
                    break
                if stalled >= _STALL_ITERATIONS:
                    break
            solver.set_posture_pose(q_now)
            assert best is not None
            _, pos_err, ori_err, q_best = best
            reached = solver.fk(q_best)[0 if arm == "left" else 1]
        return IkSolution(
            q=q_best,
            tip=_apply_offset(reached, self._offset),
            position_error=pos_err,
            orientation_error=ori_err,
        )

    def joint_trajectory(
        self,
        q_from: np.ndarray,
        q_to: np.ndarray,
        *,
        speed: float,
        rate: float,
        min_duration: float,
    ) -> list[np.ndarray]:
        """Collision-aware joint-space trajectory between two configurations."""
        from ..teleop.trajectory import plan_collision_aware_trajectory

        with self._lock:
            trajectory = plan_collision_aware_trajectory(
                self._solver,
                np.asarray(q_from, dtype=np.float32),
                np.asarray(q_to, dtype=np.float32),
                speed=speed,
                rate=rate,
                min_duration=min_duration,
            )
        return [np.asarray(q, dtype=np.float32) for q in trajectory]

    def plan(
        self,
        q_now: np.ndarray,
        arm: Arm,
        target: Pose,
        *,
        speed: float,
        rate: float,
        min_duration: float,
        lift_raised: float | None = None,
    ) -> JogPlan:
        """IK to the closest pose, then a collision-aware trajectory to it."""
        q_now = np.asarray(q_now, dtype=np.float32)
        solution = self.solve(q_now, arm, target, lift_raised)
        trajectory = self.joint_trajectory(
            q_now, solution.q, speed=speed, rate=rate, min_duration=min_duration
        )
        own = self.indices(arm)
        other = [i for i in range(len(q_now)) if i not in own]
        pinned = []
        for q in trajectory:
            q = np.asarray(q, dtype=np.float32).copy()
            q[other] = q_now[other]
            pinned.append(q)
        if not pinned:
            pinned = [solution.q.copy()]
        # End exactly on the solution the preview shows.
        pinned[-1] = solution.q.copy()
        side = 0 if arm == "left" else 1
        picks = np.unique(
            np.linspace(0, len(pinned) - 1, min(_PATH_POINTS, len(pinned))).astype(int)
        )
        path = [self._tip_position(pinned[i], side) for i in picks]
        return JogPlan(
            arm=arm,
            target=(
                np.asarray(target[0], dtype=np.float32),
                np.asarray(target[1], dtype=np.float32),
            ),
            solution=solution,
            q_from=q_now.copy(),
            trajectory=pinned,
            rate=rate,
            path=path,
        )
