"""Arm motions for synthetic data: inverse kinematics and timed paths.

:class:`ArmKinematics` solves the URDF's arms for a TCP pose (the SDK's
tool point, :data:`~almond_axol.constants.GRIPPER_TIP_OFFSET` out from the
gripper mount) with damped least squares, seeded from the last solution so
a path's joint values move smoothly. :class:`Path` is a list of timed TCP
poses with minimum-jerk blends between them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..constants import GRIPPER_TIP_OFFSET, urdf_arm_joint_names
from ..recording.mcap.urdf_tree import UrdfTree


def rotation_log(rotation: np.ndarray) -> np.ndarray:
    """Axis times angle of a rotation matrix."""
    cos = (np.trace(rotation) - 1.0) / 2.0
    angle = math.acos(min(1.0, max(-1.0, cos)))
    if angle < 1e-9:
        return np.zeros(3)
    if math.pi - angle < 1e-6:
        # Near 180 degrees: the axis from the symmetric part.
        m = (rotation + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(m), 0.0))
        axis *= np.sign(np.where(m[0] != 0, m[0], 1.0))
        return axis / max(np.linalg.norm(axis), 1e-12) * angle
    w = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    return w * (angle / (2.0 * math.sin(angle)))


def rotation_exp(vector: np.ndarray) -> np.ndarray:
    """Rotation matrix of an axis-angle vector."""
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    k = np.asarray(vector) / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * kx @ kx


def pose(xyz: Sequence[float], rotation: np.ndarray | None = None) -> np.ndarray:
    out = np.eye(4)
    out[:3, 3] = xyz
    if rotation is not None:
        out[:3, :3] = rotation
    return out


class ArmKinematics:
    """TCP forward and inverse kinematics of both arms, in ``base`` frame."""

    def __init__(
        self,
        tree: UrdfTree,
        *,
        base: str = "root",
        tool: Sequence[float] = GRIPPER_TIP_OFFSET,
    ) -> None:
        self.tree = tree
        self.base = base
        self.tool = np.eye(4)
        self.tool[:3, 3] = tool
        self.joints = {
            side: urdf_arm_joint_names(is_left=side == "left")
            for side in ("left", "right")
        }
        self.limits = {
            side: np.array(
                [
                    (
                        tree.joints[n].lower
                        if tree.joints[n].lower is not None
                        else -math.pi,
                        tree.joints[n].upper
                        if tree.joints[n].upper is not None
                        else math.pi,
                    )
                    for n in names
                ]
            )
            for side, names in self.joints.items()
        }

    def tcp(self, side: str, q: Mapping[str, float]) -> np.ndarray:
        return self.tree.fk(self.base, f"{side}_gripper", q) @ self.tool

    def solve(
        self,
        side: str,
        target: np.ndarray,
        seed: Mapping[str, float],
        *,
        iterations: int = 60,
        rotation_weight: float = 0.4,
        tolerance: float = 1e-5,
    ) -> tuple[dict[str, float], float]:
        """Joint values reaching ``target`` (4x4 in ``base``), and the
        remaining position error (m)."""
        names = self.joints[side]
        lo, hi = self.limits[side][:, 0], self.limits[side][:, 1]
        q = dict(seed)
        x = np.array([q.get(n, 0.0) for n in names])
        margin = 1e-3
        x = np.clip(x, lo + margin, hi - margin)
        weights = np.array([1, 1, 1, rotation_weight, rotation_weight, rotation_weight])
        damping = 1e-3
        err_norm = math.inf
        for _ in range(iterations):
            q.update(zip(names, x, strict=True))
            current = self.tcp(side, q)
            err = np.concatenate(
                [
                    target[:3, 3] - current[:3, 3],
                    rotation_log(target[:3, :3] @ current[:3, :3].T),
                ]
            )
            err_norm = float(np.linalg.norm(err[:3]))
            if err_norm < tolerance and np.linalg.norm(err[3:]) < 10 * tolerance:
                break
            jac = np.zeros((6, len(names)))
            step = 1e-6
            for i in range(len(names)):
                q2 = dict(q)
                q2[names[i]] = x[i] + step
                moved = self.tcp(side, q2)
                jac[:3, i] = (moved[:3, 3] - current[:3, 3]) / step
                jac[3:, i] = rotation_log(moved[:3, :3] @ current[:3, :3].T) / step
            jw = jac * weights[:, None]
            ew = err * weights
            dx = jw.T @ np.linalg.solve(jw @ jw.T + damping * np.eye(6), ew)
            # Stay away from the joint limits.
            x = np.clip(x + dx, lo + margin, hi - margin)
        q.update(zip(names, x, strict=True))
        return {n: float(q[n]) for n in names}, err_norm


def min_jerk(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return s * s * s * (10 - 15 * s + 6 * s * s)


@dataclass
class Waypoint:
    t: float
    pose: np.ndarray  # TCP pose, 4x4
    grip: float  # gripper opening, 0 closed .. 1 open


class Path:
    """Timed TCP waypoints; between them, a minimum-jerk blend."""

    def __init__(self, waypoints: Sequence[Waypoint]) -> None:
        self.waypoints = sorted(waypoints, key=lambda w: w.t)

    def at(self, t: float) -> tuple[np.ndarray, float]:
        points = self.waypoints
        if t <= points[0].t:
            return points[0].pose, points[0].grip
        for a, b in zip(points, points[1:], strict=False):
            if t <= b.t:
                s = min_jerk((t - a.t) / max(b.t - a.t, 1e-9))
                out = np.eye(4)
                out[:3, 3] = a.pose[:3, 3] + s * (b.pose[:3, 3] - a.pose[:3, 3])
                delta = rotation_log(b.pose[:3, :3] @ a.pose[:3, :3].T)
                out[:3, :3] = rotation_exp(s * delta) @ a.pose[:3, :3]
                return out, a.grip + s * (b.grip - a.grip)
        return points[-1].pose, points[-1].grip
