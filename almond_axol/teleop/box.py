"""Box-mode (bimanual carry) target geometry.

Pure-NumPy helpers for the IK worker's box-mode tracking (the pose blending
ones double for the per-arm re-engage ramp). Everything is expressed in the
robot's world frame (FLU:
+x forward, +y left, +z up) using the ``(pos (3,), rot (3, 3))`` pose format
:class:`~almond_axol.kinematics.solver.KinematicsSolver` speaks.

The **box frame** sits at the midpoint of the two gripper mount frames. Its
axes are ``x`` forward, ``y`` *lateral* — the horizontal direction from the
right gripper to the left one — and ``z`` up. The grippers live at
``center ± y * width / 2`` with their approach axes (the gripper link's
``-Z``, the direction the fingers point) turned toward each other, so the
two palms face like a pair of hands around a box. Each gripper's roll about
its approach axis is *not* canonicalised: the alignment applies the smallest
rotation that turns the current approach axis onto the target, which keeps
the wrist from flipping and leaves the fingers wherever they were.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

Pose = tuple[np.ndarray, np.ndarray]

_UP = np.array((0.0, 0.0, 1.0), dtype=np.float32)
_LEFT = np.array((0.0, 1.0, 0.0), dtype=np.float32)
# Below this horizontal separation the lateral axis is undefined; fall back
# to world +y (the grippers are stacked vertically or coincident).
_MIN_LATERAL_M = 1e-3


def rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix for ``angle`` radians about the unit vector ``axis``."""
    x, y, z = (float(v) for v in axis)
    k = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64)
    r = np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)
    return r.astype(np.float32)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector ``a`` onto unit vector ``b``.

    Anti-parallel inputs have no unique axis; any axis perpendicular to ``a``
    works, so one is picked deterministically (world up, or world +y when
    ``a`` is itself vertical).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    axis = np.cross(a, b)
    s = float(np.linalg.norm(axis))
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if s < 1e-9:
        if c > 0.0:
            return np.eye(3, dtype=np.float32)
        perp = np.cross(a, _UP.astype(np.float64))
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, _LEFT.astype(np.float64))
        return rodrigues(perp / np.linalg.norm(perp), math.pi)
    return rodrigues(axis / s, math.atan2(s, c))


def approach_axis(rot: np.ndarray) -> np.ndarray:
    """Direction the fingers point for a gripper mount rotation (its ``-Z``)."""
    return -np.asarray(rot, dtype=np.float32)[:, 2]


def align_to_approach(rot: np.ndarray, target_approach: np.ndarray) -> np.ndarray:
    """Turn ``rot`` so its approach axis points along ``target_approach``.

    The smallest such rotation is applied (see :func:`rotation_between`), so
    the roll about the approach axis is carried over from ``rot``.
    """
    r = rotation_between(approach_axis(rot), target_approach)
    return (r @ np.asarray(rot, dtype=np.float32)).astype(np.float32)


def box_frame(
    left_pos: np.ndarray, right_pos: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """``(center, rotation, width)`` of the level box frame between two grippers.

    ``width`` is the full 3-D separation of the two mount frames, so two
    grippers that start at different heights keep their spacing when the
    frame levels them; ``rotation`` is yaw-only (``z`` up).
    """
    left_pos = np.asarray(left_pos, dtype=np.float64)
    right_pos = np.asarray(right_pos, dtype=np.float64)
    center = 0.5 * (left_pos + right_pos)
    d = left_pos - right_pos
    width = float(np.linalg.norm(d))
    lat = d.copy()
    lat[2] = 0.0
    if np.linalg.norm(lat) < _MIN_LATERAL_M:
        lat = _LEFT.astype(np.float64)
    lat = lat / np.linalg.norm(lat)
    up = _UP.astype(np.float64)
    fwd = np.cross(lat, up)
    rot = np.stack([fwd, lat, up], axis=1)
    return center.astype(np.float32), rot.astype(np.float32), width


def rotation_angle(r0: np.ndarray, r1: np.ndarray) -> float:
    """Angle (rad) of the rotation taking ``r0`` onto ``r1``."""
    rel = np.asarray(r0, dtype=np.float64).T @ np.asarray(r1, dtype=np.float64)
    cos_theta = max(-1.0, min(1.0, (float(np.trace(rel)) - 1.0) * 0.5))
    return math.acos(cos_theta)


def smoothstep(u: float) -> float:
    """C1 ease ``3u² - 2u³`` on ``[0, 1]`` (clamped)."""
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def blend_pose(start: Pose, goal: Pose, alpha: float) -> Pose:
    """Interpolate ``start`` → ``goal``: linear position, geodesic rotation."""
    if alpha <= 0.0:
        return start
    if alpha >= 1.0:
        return goal
    p0, r0 = start
    p1, r1 = goal
    pos = ((1.0 - alpha) * p0 + alpha * p1).astype(np.float32)
    rel = r0.T @ r1
    cos_theta = max(-1.0, min(1.0, (float(np.trace(rel)) - 1.0) * 0.5))
    theta = math.acos(cos_theta)
    if theta < 1e-6:
        return pos, r1.astype(np.float32)
    axis = np.array(
        (rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]),
        dtype=np.float64,
    ) / (2.0 * math.sin(theta))
    return pos, (r0 @ rodrigues(axis, alpha * theta)).astype(np.float32)


@dataclass
class BoxState:
    """Box-mode tracking state, established at the engage snap.

    ``center`` / ``rot`` are the box pose *at the snap*; the leader
    controller's motion since its own snap is applied to them every frame
    (the box rides rigidly on the leader gripper's clutch mapping, see
    ``IKWorker``), then the accumulated stick jog — a world-frame offset
    ``jog_pos`` and a yaw ``jog_yaw`` about the box centre — on top.
    ``grip_rel`` holds each gripper's rotation relative to the box frame,
    fixed at the snap so the pair turns rigidly with the box.
    ``align_start`` holds where each gripper actually was at the snap,
    expressed in the box frame, for the blend into the parallel
    configuration.
    """

    center: np.ndarray
    rot: np.ndarray
    width: float
    grip_rel: dict[str, np.ndarray]
    align_start: dict[str, Pose]
    align_t0: float
    align_duration: float
    jog_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    jog_yaw: float = 0.0
    # Wall time of the previous jog integration step (None before the first).
    jog_t: float | None = None

    @property
    def aligned(self) -> bool:
        """True once the align blend has finished (1:1 tracking from here)."""
        return self.align_duration <= 0.0

    def align_alpha(self, now: float) -> float:
        if self.align_duration <= 0.0:
            return 1.0
        u = (now - self.align_t0) / self.align_duration
        if u >= 1.0:
            self.align_duration = 0.0
            return 1.0
        return smoothstep(u)


def parallel_gripper_rotations(
    left: Pose, right: Pose, lateral: np.ndarray
) -> dict[str, np.ndarray]:
    """World rotations putting both grippers' approach axes on the lateral line.

    The left gripper (on the ``+lateral`` side) points ``-lateral``, toward
    the right one, and vice versa — palms facing.
    """
    lateral = np.asarray(lateral, dtype=np.float32)
    return {
        "left": align_to_approach(left[1], -lateral),
        "right": align_to_approach(right[1], lateral),
    }


def snap_box(
    left: Pose,
    right: Pose,
    now: float,
    align_duration: float,
    width_min: float,
    width_max: float,
) -> BoxState:
    """Build the box state for an engage snap from the current gripper poses."""
    center, rot, width = box_frame(left[0], right[0])
    width = float(np.clip(width, width_min, width_max))
    lateral = rot[:, 1]
    aligned = parallel_gripper_rotations(left, right, lateral)
    grip_rel = {side: (rot.T @ r).astype(np.float32) for side, r in aligned.items()}
    align_start = {
        side: (
            (rot.T @ (pose[0] - center)).astype(np.float32),
            (rot.T @ pose[1]).astype(np.float32),
        )
        for side, pose in (("left", left), ("right", right))
    }
    return BoxState(
        center=center,
        rot=rot,
        width=width,
        grip_rel=grip_rel,
        align_start=align_start,
        align_t0=now,
        align_duration=max(align_duration, 0.0),
    )


def ideal_gripper_poses(
    center: np.ndarray, rot: np.ndarray, width: float, grip_rel: dict[str, np.ndarray]
) -> dict[str, Pose]:
    """The parallel-gripper pair for a box pose: ``{"left": pose, "right": pose}``."""
    half = 0.5 * width * rot[:, 1]
    return {
        "left": (
            (center + half).astype(np.float32),
            (rot @ grip_rel["left"]).astype(np.float32),
        ),
        "right": (
            (center - half).astype(np.float32),
            (rot @ grip_rel["right"]).astype(np.float32),
        ),
    }


def box_targets(
    state: BoxState, center: np.ndarray, rot: np.ndarray, now: float
) -> dict[str, Pose]:
    """Per-gripper EE targets for the current box pose ``(center, rot)``.

    While the align blend runs, each gripper is eased from where it was at the
    snap (carried along with the box) into its parallel slot; afterwards the
    parallel pair is returned directly.
    """
    ideal = ideal_gripper_poses(center, rot, state.width, state.grip_rel)
    alpha = state.align_alpha(now)
    if alpha >= 1.0:
        return ideal
    out: dict[str, Pose] = {}
    for side, (p_rel, r_rel) in state.align_start.items():
        start = (
            (center + rot @ p_rel).astype(np.float32),
            (rot @ r_rel).astype(np.float32),
        )
        out[side] = blend_pose(start, ideal[side], alpha)
    return out
