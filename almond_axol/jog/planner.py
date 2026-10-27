"""IK and joint-space planning for jog moves.

A jog move names a gripper-tip pose for one arm. :class:`JogPlanner` solves
IK for it with :class:`~almond_axol.kinematics.arm_ik.ArmKinematics` (single
arm, NumPy, about a millisecond), so the result is the reachable pose
closest to the target, then plans a joint-space path to it that keeps the
arms clear of the body and of each other. The other arm stays where it is.

Poses are in the torso frame IK uses (+x forward, +y left, +z up, the URDF's
``root`` link), at the gripper tips (:data:`~almond_axol.constants.GRIPPER_TIP_OFFSET`
out from the mount).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..kinematics.arm_ik import ArmKinematics, rotation_angle

Arm = Literal["left", "right"]
Pose = tuple[np.ndarray, np.ndarray]

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

    @property
    def score(self) -> float:
        """Closeness to the target: a radian of orientation weighs like 5 cm."""
        return self.position_error + 0.05 * self.orientation_error


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
    """IK, previews and joint-space paths on one :class:`ArmKinematics`.

    Thread-safe without a lock: the control loop, previews and plans may
    call it at once.
    """

    def __init__(self, kinematics: ArmKinematics) -> None:
        self._kin = kinematics

    @property
    def kinematics(self) -> ArmKinematics:
        return self._kin

    def indices(self, arm: Arm) -> list[int]:
        return self._kin.indices(arm)

    def tips(self, q: np.ndarray, *, blocking: bool = True) -> tuple[Pose, Pose]:
        """``(left, right)`` gripper-tip poses for a full joint vector.

        ``blocking`` is accepted for callers written against a locked
        solver; nothing here waits.
        """
        del blocking
        left, right = self._kin.tips(np.asarray(q, dtype=np.float64))
        return (
            (left[0].astype(np.float32), left[1].astype(np.float32)),
            (right[0].astype(np.float32), right[1].astype(np.float32)),
        )

    def _solution(self, q: np.ndarray, arm: Arm, target: Pose) -> IkSolution:
        tip = self._kin.tip(q, arm)
        return IkSolution(
            q=np.asarray(q, dtype=np.float32).copy(),
            tip=(tip[0].astype(np.float32), tip[1].astype(np.float32)),
            position_error=float(np.linalg.norm(tip[0] - target[0])),
            orientation_error=rotation_angle(tip[1], np.asarray(target[1], np.float64)),
        )

    def solve(
        self,
        q_now: np.ndarray,
        arm: Arm,
        target: Pose,
        lift_raised: float | None = None,
        *,
        seed: np.ndarray | None = None,
    ) -> IkSolution:
        """Closest reachable configuration putting ``arm``'s tip at ``target``.

        Solves from ``seed`` (the last preview's solution, so what the page
        showed is what gets planned) when given, else from ``q_now``. A seed
        that ends short of the target is checked against a fresh solve from
        ``q_now``, and the closer one wins. The other arm stays at ``q_now``.
        """
        kin = self._kin
        kin.set_lift(lift_raised)
        q_now = np.asarray(q_now, dtype=np.float64)
        position = np.asarray(target[0], dtype=np.float64)
        rotation = np.asarray(target[1], dtype=np.float64)
        own = self.indices(arm)
        starts = [q_now]
        if seed is not None:
            start = q_now.copy()
            start[own] = np.asarray(seed, dtype=np.float64)[own]
            starts.insert(0, start)
        best: IkSolution | None = None
        for start in starts:
            result = kin.solve(start, arm, (position, rotation))
            solution = IkSolution(
                q=result.q,
                tip=result.tip,
                position_error=result.position_error,
                orientation_error=result.orientation_error,
            )
            if best is None or solution.score < best.score - 1e-6:
                best = solution
            if best.reached:
                break
        assert best is not None
        return best

    def joint_trajectory(
        self,
        q_from: np.ndarray,
        q_to: np.ndarray,
        *,
        speed: float,
        rate: float,
        min_duration: float,
    ) -> list[np.ndarray]:
        """Joint-space trajectory between two configurations at ``rate`` Hz.

        A smoothstep whose worst joint averages ``speed`` (peak ``1.5 *
        speed``), at least ``min_duration`` long, kept clear of the body and
        the other arm (see :meth:`ArmKinematics.path`).
        """
        q_from = np.asarray(q_from, dtype=np.float64)
        q_to = np.asarray(q_to, dtype=np.float64)
        distance = float(np.max(np.abs(q_to - q_from))) if q_from.size else 0.0
        steps = max(2, round(max(distance / speed, min_duration) * rate))
        return self._kin.path(q_from, q_to, steps)

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
        seed: np.ndarray | None = None,
    ) -> JogPlan:
        """IK to the closest pose, then a collision-aware trajectory to it."""
        q_now = np.asarray(q_now, dtype=np.float32)
        target = (
            np.asarray(target[0], dtype=np.float32),
            np.asarray(target[1], dtype=np.float32),
        )
        solution = self.solve(q_now, arm, target, lift_raised, seed=seed)
        trajectory = self.joint_trajectory(
            q_now, solution.q, speed=speed, rate=rate, min_duration=min_duration
        )
        own = self.indices(arm)
        other = [i for i in range(len(q_now)) if i not in own]
        for q in trajectory:
            q[other] = q_now[other]
        if float(np.max(np.abs(trajectory[-1] - solution.q))) > 1e-4:
            # The path could not clear the body all the way: the move ends
            # where it stopped, and the plan says so.
            solution = self._solution(trajectory[-1], arm, target)
        else:
            trajectory[-1] = solution.q.copy()
        side = 0 if arm == "left" else 1
        picks = np.unique(
            np.linspace(0, len(trajectory) - 1, min(_PATH_POINTS, len(trajectory)))
            .round()
            .astype(int)
        )
        path = [self.tips(trajectory[i])[side][0] for i in picks]
        return JogPlan(
            arm=arm,
            target=target,
            solution=solution,
            q_from=q_now.copy(),
            trajectory=trajectory,
            rate=rate,
            path=path,
        )
