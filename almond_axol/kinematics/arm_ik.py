"""Fast single-arm kinematics, IK and Cartesian servoing in NumPy.

Jogging asks for IK as fast as an operator drags a gizmo and for a Cartesian
step on every control tick, one arm at a time. The Pyroki solver
(:mod:`.solver`) is built for bimanual teleop instead: every call runs a
fixed Levenberg-Marquardt budget over both arms, every link and every
collision pair inside XLA (about 2 ms per iteration on a desktop CPU), and
its per-call joint bound makes a long move take dozens of calls.

This module solves the same kinematics directly:

- Forward kinematics and Jacobians come from the Pyroki robot's own joint
  arrays (:func:`.model.shared_robot`), so they match
  :meth:`.solver.KinematicsSolver.fk` to float precision. Only the seven
  joints of the arm being moved are evaluated.
- Self-collision uses the same shapes (:func:`.model.shared_robot_collision`:
  capsules on the arms, capsules and boxes on the body) and arm/body pairs,
  plus arm/arm pairs. The lift and the gripper openings move them as they
  move the teleop solver's (:meth:`ArmKinematics.set_lift`,
  :meth:`ArmKinematics.set_gripper`). Every pair has a hard floor (see
  ``ArmKinematics.floors``): no step may take a pair below it, or closer
  than it already is when it starts inside.
- Each step is a small QP (seven variables, solved by DAQP in microseconds)
  with the joint limits, a per-step joint bound and linearised clearance
  rows, and every step is checked against the exact shape distances.

Joint vectors use the robot's layout, the left arm's seven joints in
:data:`~almond_axol.constants.ARM_JOINTS` order then the right arm's. Poses
are ``(position (3,), rotation (3, 3))`` in the torso frame (the URDF's
``root`` link: +x forward, +y left, +z up), at the gripper tips
(``tool_offset`` out from the gripper mount).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

import daqp
import numpy as np

from ..constants import (
    ARM_JOINTS,
    GRIPPER_TIP_OFFSET,
    AxolModel,
    Joint,
    upper_arm_guard_links,
    urdf_arm_joint_names,
    urdf_body_name,
)

_logger = logging.getLogger(__name__)

Side = Literal["left", "right"]
Pose = tuple[np.ndarray, np.ndarray]
SIDES: tuple[Side, Side] = ("left", "right")
ARM_DOF = 7

# Clearance pairs farther than this beyond their floor add no QP row.
_INFLUENCE_M = 0.06
# A step may sink a pair this far below its floor before it is rejected:
# the QP rows are linear, the capsule distances are not.
_FLOOR_SLACK_M = 5e-4
# Same activation rule as .model.collision_cost_params.
_MARGIN_REST_BUFFER = 0.002
# The closest a pair other than an upper arm may come (capsule distance; the
# capsules enclose the meshes, so the real gap is at least this).
_MIN_CLEARANCE_M = 0.005
# The elbow's index in a seven-joint arm vector, and how near straight it
# may come once bent (see ArmKinematics._box).
_ELBOW = ARM_JOINTS.index(Joint.ELBOW)
_ELBOW_STRAIGHT_RAD = 0.05
# Metres of tip position a radian of orientation weighs like: with 0.3, one
# degree weighs like 5 mm.
_ROTATION_WEIGHT_M = 0.3
_WEIGHTS = np.array([1.0, 1.0, 1.0, *([_ROTATION_WEIGHT_M**2] * 3)])
_NO_ROWS = (np.zeros((0, ARM_DOF)), np.zeros(0))


# ---------------------------------------------------------------------------
# Small rigid-body helpers
# ---------------------------------------------------------------------------


def _quat_matrix(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in wxyz)
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        )
    )


def _transform(wxyz_xyz: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = _quat_matrix(wxyz_xyz[:4])
    t[:3, 3] = wxyz_xyz[4:7]
    return t


def _skew(axes: np.ndarray) -> np.ndarray:
    """``(n, 3, 3)`` cross-product matrices of ``(n, 3)`` vectors."""
    x, y, z = axes[:, 0], axes[:, 1], axes[:, 2]
    zero = np.zeros_like(x)
    return np.stack(
        (
            np.stack((zero, -z, y), -1),
            np.stack((z, zero, -x), -1),
            np.stack((-y, x, zero), -1),
        ),
        -2,
    )


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product over the last axis (broadcasting), without np.cross's
    per-call overhead."""
    out = np.empty(np.broadcast_shapes(a.shape, b.shape))
    out[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    out[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    out[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return out


def exp_rotation(vector: np.ndarray) -> np.ndarray:
    """Rotation matrix of an axis-angle vector."""
    vector = np.asarray(vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    k = _skew((vector / angle)[None])[0]
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a rotation matrix (stable near 0 and pi)."""
    cos = (float(np.trace(rotation)) - 1.0) * 0.5
    cos = min(1.0, max(-1.0, cos))
    angle = math.acos(cos)
    skew = np.array(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        )
    )
    if angle < 1e-6:
        return 0.5 * skew
    if angle < math.pi - 1e-4:
        return skew * (angle / (2.0 * math.sin(angle)))
    # Near pi the skew part vanishes: take the axis from the symmetric part.
    sym = (rotation + rotation.T) * 0.5 - cos * np.eye(3)
    axis = sym[int(np.argmax(np.diag(sym)))]
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    if float(axis @ skew) < 0.0:
        axis = -axis
    return axis * angle


def rotation_angle(r_from: np.ndarray, r_to: np.ndarray) -> float:
    """Angle (rad) between two rotations."""
    cos = (float(np.trace(r_from.T @ r_to)) - 1.0) * 0.5
    return float(math.acos(min(1.0, max(-1.0, cos))))


def _segment_points(
    a1: np.ndarray, b1: np.ndarray, a2: np.ndarray, b2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Closest points between segment batches ``[a1, b1]`` and ``[a2, b2]``.

    A NumPy port of Pyroki's ``closest_segment_to_segment_points`` (Ericson
    5.1.9), so capsule distances match the Pyroki collision model.
    """
    eps = 1e-9
    d1 = b1 - a1
    d2 = b2 - a2
    r = a1 - a2
    a = (d1 * d1).sum(1)
    e = (d2 * d2).sum(1)
    f = (d2 * r).sum(1)
    c = (d1 * r).sum(1)
    b = (d1 * d2).sum(1)
    denom = a * e - b * b
    general = denom > 1e-6 * np.maximum(a * e, eps)
    s = np.where(
        general, np.clip((b * f - c * e) / np.where(general, denom, 1.0), 0, 1), 0.0
    )
    t = (b * s + f) / np.maximum(e, eps)
    s = np.where(
        t < 0.0,
        np.clip(-c / np.maximum(a, eps), 0.0, 1.0),
        np.where(t > 1.0, np.clip((b - c) / np.maximum(a, eps), 0.0, 1.0), s),
    )
    t = np.clip(t, 0.0, 1.0)
    return a1 + d1 * s[:, None], a2 + d2 * t[:, None]


def _box_sdf(p: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Signed distance from points ``p`` (box frame) to boxes of
    half-extents ``half`` centred at the origin."""
    q = np.abs(p) - half
    outside = np.sqrt((np.maximum(q, 0.0) ** 2).sum(-1))
    return outside + np.minimum(q.max(-1), 0.0)


# Each pair of box axes, with each sign of each: where two face distances
# can tie along a capsule's axis (see _capsule_box).
_TIES = tuple(
    np.array(v)
    for v in zip(
        *(
            (i, j, si, sj)
            for i, j in ((0, 1), (0, 2), (1, 2))
            for si in (1.0, -1.0)
            for sj in (1.0, -1.0)
        )
    )
)


def _capsule_box(
    a: np.ndarray,
    b: np.ndarray,
    centre: np.ndarray,
    rotation: np.ndarray,
    half: np.ndarray,
    *,
    distance_only: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Closest approach of capsule axes ``[a, b]`` to boxes (``(k, 3)`` each,
    the boxes' ``rotation`` ``(k, 3, 3)``, all in the torso frame).

    Returns the box's signed distance at the closest axis point (subtract
    the capsule's radius for the gap), that point, and the box's outward
    unit normal there (None for both with ``distance_only``). It matches :func:`.collision.capsule_box_distance`,
    which searches for the same minimum, to well under 0.1 mm.

    The box's signed distance along the axis is convex, and its minimum is
    at one of a few candidate points, all evaluated at once:

    - outside the box it is the root of a piecewise quadratic, with a piece
      between each pair of points where a coordinate crosses a face plane:
      each piece's stationary point;
    - inside it is the largest of the three face distances, piecewise linear:
      where two of them tie, or a coordinate crosses zero;
    - the axis ends and the face-plane crossings.
    """
    k = len(a)
    p0 = np.einsum("kji,kj->ki", rotation, a - centre)
    d = np.einsum("kji,kj->ki", rotation, b - a)
    tiny = 1e-12
    safe = np.where(np.abs(d) > tiny, d, tiny)
    # Face-plane crossings (p = +-half) and zero crossings, per coordinate.
    planes = np.concatenate([(half - p0) / safe, (-half - p0) / safe], axis=1)
    zeros = -p0 / safe
    # Inside: two face distances tie, |p_i| - h_i = |p_j| - h_j.
    i, j, si, sj = _TIES
    den = si * d[:, i] - sj * d[:, j]
    num = half[:, i] - half[:, j] - si * p0[:, i] + sj * p0[:, j]
    ties = num / np.where(np.abs(den) > tiny, den, tiny)
    # Outside: on each piece between sorted plane crossings, the stationary
    # point of the sum of squares over the coordinates outside their face.
    edges = np.sort(
        np.concatenate([np.zeros((k, 1)), np.ones((k, 1)), np.clip(planes, 0, 1)], 1),
        axis=1,
    )
    lo, hi = edges[:, :-1], edges[:, 1:]
    mid = p0[:, None, :] + d[:, None, :] * ((lo + hi) / 2.0)[..., None]
    sign = np.where(mid < 0.0, -1.0, 1.0)
    active = np.abs(mid) > half[:, None, :]
    dd = d[:, None, :]
    num = (active * dd * (sign * half[:, None, :] - p0[:, None, :])).sum(-1)
    den = (active * dd * dd).sum(-1)
    stationary = np.where(den > tiny, num / np.where(den > tiny, den, 1.0), lo)
    stationary = np.clip(stationary, lo, hi)
    ts = np.clip(
        np.concatenate([edges, planes, zeros, ties, stationary], axis=1),
        0.0,
        1.0,
    )
    values = _box_sdf(p0[:, None, :] + d[:, None, :] * ts[..., None], half[:, None])
    if distance_only:
        return values.min(1), None, None
    best = values.argmin(1)
    rows = np.arange(k)
    t = ts[rows, best]
    sdf = values[rows, best]
    local = p0 + d * t[:, None]
    # The box SDF's gradient: outside, toward the point from the nearest
    # surface point; inside, out through the nearest face.
    q = np.abs(local) - half
    out = np.maximum(q, 0.0)
    norm = np.sqrt((out * out).sum(-1))
    face = np.zeros_like(q)
    face[rows, q.argmax(-1)] = 1.0
    grad = np.where(
        (norm > 1e-12)[:, None], out / np.maximum(norm, 1e-12)[:, None], face
    )
    grad = grad * np.where(local < 0.0, -1.0, 1.0)
    normal = np.einsum("kij,kj->ki", rotation, grad)
    return sdf, a + (b - a) * t[:, None], normal


def _qp(
    hessian: np.ndarray,
    gradient: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rows: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray | None:
    """``min ½ xᵀHx + gᵀx`` s.t. ``lower <= x <= upper``, ``rows @ x <= bounds``.

    None when DAQP finds no solution (the caller then takes no step: a zero
    step is always feasible for the problems built here).
    """
    upper_all = np.concatenate([upper, bounds])
    lower_all = np.concatenate([lower, np.full(len(bounds), -1e30)])
    x, _, flag, _ = daqp.solve(
        np.ascontiguousarray(hessian),
        np.ascontiguousarray(gradient),
        np.ascontiguousarray(rows),
        upper_all,
        lower_all,
        np.zeros(len(upper_all), dtype=np.int32),
    )
    if flag < 1 or not np.all(np.isfinite(x)):
        return None
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Arm:
    """One arm's chain, collapsed onto its seven joints."""

    fixed_r: np.ndarray  # (7, 3, 3): joint k-1 frame -> joint k frame at zero
    fixed_p: np.ndarray  # (7, 3)
    axes: np.ndarray  # (7, 3): each joint's axis in its own frame
    skew: np.ndarray  # (7, 3, 3): the axes' cross-product matrices
    outer: np.ndarray  # (7, 3, 3): the axes' outer products
    lower: np.ndarray  # (7,)
    upper: np.ndarray  # (7,)
    mount_r: np.ndarray  # wrist_3 joint frame -> gripper mount
    mount_p: np.ndarray
    cap_joint: np.ndarray  # (c,): the joint (0-6) each capsule rides on
    cap_a: np.ndarray  # (c, 3): capsule axis ends, in that joint's frame
    cap_b: np.ndarray
    cap_slide: np.ndarray  # (c, 3): a finger's slide per metre of its joint
    cap_finger: np.ndarray  # (c,): which capsules are this gripper's fingers


@dataclass
class ArmFrames:
    """One arm's forward kinematics at a configuration (torso frame)."""

    q: np.ndarray  # (7,)
    origins: np.ndarray  # (7, 3) joint frame origins
    axes: np.ndarray  # (7, 3) joint axes
    tip: Pose
    cap_a: np.ndarray  # (c, 3) capsule axis ends
    cap_b: np.ndarray

    def jacobian(self, point: np.ndarray) -> np.ndarray:
        """``(6, 7)`` geometric Jacobian of a point carried by the last joint."""
        return np.concatenate([_cross(self.axes, point - self.origins).T, self.axes.T])


@dataclass
class IkResult:
    """The closest configuration found for a tip target."""

    q: np.ndarray  # (14,) float32
    tip: Pose
    position_error: float
    orientation_error: float
    iterations: int


@dataclass
class StepResult:
    """One Cartesian servo step."""

    q: np.ndarray  # (14,) float32
    blocked: bool  # a joint limit or a clearance floor cut the step short


@dataclass(frozen=True)
class _Moving:
    """How a move of one arm changes the pairs it is part of."""

    pairs: np.ndarray  # pair indices
    second: np.ndarray  # True where the moving capsule is the pair's second
    joint: np.ndarray  # the joint (0-6) the moving capsule rides on


class ArmKinematics:
    """Thread-safe single-arm FK, IK and servoing for one Axol model.

    Everything is read-only after construction except the lift height and
    the gripper openings (:meth:`set_lift`, :meth:`set_gripper`), which swap
    whole arrays, so the control loop, previews and plans can share an
    instance without a lock.
    """

    def __init__(
        self,
        robot_model: str | None = None,
        *,
        gripper: str | None = None,
        mount: str | None = None,
        collision: bool = True,
        collision_margin: float = 0.02,
        tool_offset: tuple[float, float, float] = GRIPPER_TIP_OFFSET,
    ) -> None:
        """
        Args:
            robot_model:      Axol version (see :func:`~almond_axol.settings.resolve_robot_model`).
            gripper:          Fitted gripper on the mobile Axol (settings if None).
            mount:            Mount when ``robot_model`` is None.
            collision:        Keep the arms clear of the body and each other.
            collision_margin: The Pyroki solver's ``self_collision_margin``:
                              where its collision cost starts (see ``floors``).
            tool_offset:      Gripper tip in the gripper mount frame.
        """
        from ..settings import resolve_gripper, resolve_robot_model
        from .model import (
            finger_slide,
            lift_link_shift,
            shared_robot,
            shared_robot_collision,
            upper_arm_guard_floor,
        )

        model = resolve_robot_model(robot_model, mount=mount)
        gripper = resolve_gripper(gripper) if model is AxolModel.MOBILE else "parallel"
        self.robot_model = model
        self.gripper = gripper
        self._offset = np.asarray(tool_offset, dtype=np.float64)
        robot = shared_robot(model, gripper=gripper)
        joints = robot.joints
        parents = list(joints.parent_indices)
        twists = np.asarray(joints.twists, dtype=np.float64)
        parent_tf = [
            _transform(np.asarray(t, np.float64)) for t in joints.parent_transforms
        ]
        zero: dict[int, np.ndarray] = {}

        def joint_zero(i: int) -> np.ndarray:
            """Joint ``i``'s frame at the zero configuration (Pyroki's FK)."""
            if i not in zero:
                p = parents[i]
                zero[i] = parent_tf[i] if p < 0 else joint_zero(p) @ parent_tf[i]
            return zero[i]

        def ancestors(i: int) -> list[int]:
            chain = []
            while i >= 0:
                chain.append(i)
                i = parents[i]
            return chain

        names = list(joints.names)
        actuated = list(joints.actuated_names)
        lower_all = np.asarray(joints.lower_limits, dtype=np.float64)
        upper_all = np.asarray(joints.upper_limits, dtype=np.float64)
        link_names = list(robot.links.names)
        link_parent = list(robot.links.parent_joint_indices)

        def link_zero(link: int) -> np.ndarray:
            p = link_parent[link]
            return np.eye(4) if p < 0 else joint_zero(p)

        arm_joints = {
            side: [names.index(n) for n in urdf_arm_joint_names(is_left=side == "left")]
            for side in SIDES
        }

        def riding(link: int) -> tuple[Side, int] | None:
            """The arm joint a link rides on (the deepest of its ancestors)."""
            p = link_parent[link]
            chain = ancestors(p) if p >= 0 else []
            for side in SIDES:
                for k in range(ARM_DOF - 1, -1, -1):
                    if arm_joints[side][k] in chain:
                        return side, k
            return None

        # The shapes (.collision.ArmBodyCollision): capsules on the arms,
        # capsules and boxes on the body, any number per link. Capsule axis
        # ends and box poses are in their link's frame.
        coll = shared_robot_collision(model, gripper=gripper) if collision else None
        if coll is not None:
            if list(coll.link_names) != link_names:
                raise RuntimeError("The collision model's links don't match the robot.")
            cap_pose = np.asarray(coll.capsules.pose.wxyz_xyz, np.float64)
            cap_radius = np.asarray(coll.capsules.radius, np.float64).reshape(-1)
            cap_height = np.asarray(coll.capsules.height, np.float64).reshape(-1)
            cap_link = [int(k) for k in coll.capsule_link]
            box_link = [int(k) for k in coll.box_link]
            box_pose = (
                np.asarray(coll.boxes.pose.wxyz_xyz, np.float64)
                if coll.boxes is not None
                else np.zeros((0, 7))
            )
            box_half = (
                np.asarray(coll.boxes.half_extent, np.float64)
                if coll.boxes is not None
                else np.zeros((0, 3))
            )
            capsule_pairs = [(int(i), int(j)) for i, j in coll.capsule_pairs]
            box_pairs = [(int(i), int(j)) for i, j in coll.box_pairs]
        else:
            cap_pose = np.zeros((0, 7))
            cap_radius = cap_height = np.zeros(0)
            cap_link, box_link = [], []
            box_pose, box_half = np.zeros((0, 7)), np.zeros((0, 3))
            capsule_pairs, box_pairs = [], []

        def cap_ends(c: int, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Capsule ``c``'s axis ends, with its link at ``frame``."""
            t = frame @ _transform(cap_pose[c])
            half = t[:3, 2] * cap_height[c] * 0.5
            return t[:3, 3] - half, t[:3, 3] + half

        # Every pair's first shape is an arm capsule; the second a body
        # capsule or box. The arm capsules in any pair take part.
        arm_caps = sorted({i for i, _ in capsule_pairs} | {i for i, _ in box_pairs})
        for c in arm_caps:
            if riding(cap_link[c]) is None:
                raise RuntimeError(f"{link_names[cap_link[c]]} is not on an arm.")
        arm_caps_of: dict[Side, list[int]] = {
            side: [c for c in arm_caps if riding(cap_link[c])[0] == side]  # type: ignore[index]
            for side in SIDES
        }
        body_caps = sorted({j for _, j in capsule_pairs})
        body_boxes = sorted({j for _, j in box_pairs})
        # The capsule table: left arm, right arm, then the body.
        table = arm_caps_of["left"] + arm_caps_of["right"] + body_caps
        slot = {c: k for k, c in enumerate(table)}
        box_slot = {b: k for k, b in enumerate(body_boxes)}
        finger_axis, finger_side, finger_closed, finger_stroke = finger_slide(
            model, gripper=gripper
        )

        self._arms: dict[Side, _Arm] = {}
        for side in SIDES:
            idx = arm_joints[side]
            for i in idx:
                if np.any(twists[i, :3] != 0.0):
                    raise RuntimeError(f"Arm joint {names[i]} is not revolute.")
            fixed = [joint_zero(idx[0])]
            for prev, cur in zip(idx, idx[1:]):
                if prev not in ancestors(cur):
                    raise RuntimeError(f"The {side} arm's joints are not one chain.")
                fixed.append(np.linalg.inv(joint_zero(prev)) @ joint_zero(cur))
            axes = twists[idx, 3:]
            axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
            act = [actuated.index(names[i]) for i in idx]
            ee = link_names.index(urdf_body_name(Joint.GRIPPER, is_left=side == "left"))
            if riding(ee) != (side, ARM_DOF - 1):
                raise RuntimeError(f"The {side} gripper does not ride on wrist_3.")
            mount = np.linalg.inv(joint_zero(idx[-1])) @ link_zero(ee)
            cap_joint, cap_a, cap_b, cap_slide = [], [], [], []
            for c in arm_caps_of[side]:
                link = cap_link[c]
                ride = riding(link)
                assert ride is not None
                local = np.linalg.inv(joint_zero(idx[ride[1]])) @ link_zero(link)
                a, b = cap_ends(c, local)
                cap_joint.append(ride[1])
                cap_a.append(a)
                cap_b.append(b)
                # A finger slides along its joint axis with the opening.
                cap_slide.append(local[:3, :3] @ finger_axis[link])
            fixed_arr = np.stack(fixed)
            self._arms[side] = _Arm(
                fixed_r=fixed_arr[:, :3, :3].copy(),
                fixed_p=fixed_arr[:, :3, 3].copy(),
                axes=axes,
                skew=_skew(axes),
                outer=axes[:, :, None] * axes[:, None, :],
                lower=lower_all[act],
                upper=upper_all[act],
                mount_r=mount[:3, :3].copy(),
                mount_p=mount[:3, 3].copy(),
                cap_joint=np.asarray(cap_joint, dtype=np.intp),
                cap_a=np.asarray(cap_a, dtype=np.float64).reshape(-1, 3),
                cap_b=np.asarray(cap_b, dtype=np.float64).reshape(-1, 3),
                cap_slide=np.asarray(cap_slide, dtype=np.float64).reshape(-1, 3),
                cap_finger=np.asarray(
                    [
                        finger_side[cap_link[c]] == (0 if side == "left" else 1)
                        for c in arm_caps_of[side]
                    ],
                    dtype=bool,
                ),
            )

        # Pairs: Pyroki's arm/body pairs in its order (capsule pairs, then
        # box pairs, so the first distances match .collision's), then every
        # left/right arm pair (the teleop model leaves those out, but a
        # jogged gripper can be driven into the other arm). The first shape
        # is always an arm capsule.
        first = [slot[i] for i, _ in capsule_pairs] + [slot[i] for i, _ in box_pairs]
        second = [slot[j] for _, j in capsule_pairs] + [
            box_slot[j] for _, j in box_pairs
        ]
        is_box = [False] * len(capsule_pairs) + [True] * len(box_pairs)
        n_body = len(first)
        for left in arm_caps_of["left"]:
            for right in arm_caps_of["right"]:
                first.append(slot[left])
                second.append(slot[right])
                is_box.append(False)
        self._pair_i = np.asarray(first, dtype=np.intp)
        self._pair_j = np.asarray(second, dtype=np.intp)
        self._pair_box = np.asarray(is_box, dtype=bool)
        self._body_mask = np.arange(len(first)) < n_body
        n_left, n_right = len(arm_caps_of["left"]), len(arm_caps_of["right"])
        self._n_arm = n_left + n_right
        self._radius = cap_radius[table] if table else np.zeros(0)

        # Body shapes in the torso frame, as frozen (the lift fully raised).
        ends = [cap_ends(c, link_zero(cap_link[c])) for c in body_caps]
        self._body_a0 = np.asarray([e[0] for e in ends], np.float64).reshape(-1, 3)
        self._body_b0 = np.asarray([e[1] for e in ends], np.float64).reshape(-1, 3)
        boxes = [link_zero(box_link[b]) @ _transform(box_pose[b]) for b in body_boxes]
        self._box_r = np.asarray([t[:3, :3] for t in boxes], np.float64).reshape(
            -1, 3, 3
        )
        self._box_c0 = np.asarray([t[:3, 3] for t in boxes], np.float64).reshape(-1, 3)
        self._box_half = box_half[body_boxes] if body_boxes else np.zeros((0, 3))
        self._box_reach = np.linalg.norm(self._box_half, axis=1)
        self._body = (self._body_a0, self._body_b0, self._box_c0)
        shift, travel = lift_link_shift(model, gripper=gripper)
        self._body_shift = np.asarray(
            [shift[cap_link[c]] for c in body_caps], np.float64
        ).reshape(-1, 3)
        self._box_shift = np.asarray(
            [shift[box_link[b]] for b in body_boxes], np.float64
        ).reshape(-1, 3)
        self._lift_travel = float(travel)
        self._lift_raised = 1.0
        self._finger_closed = float(finger_closed)
        self._finger_stroke = float(finger_stroke)
        self._grip: tuple[float, float] | None = None
        self._slide: dict[Side, np.ndarray] = {}
        self._other_cache: dict[Side, ArmFrames] = {}

        def moving(side: Side) -> _Moving:
            lo = 0 if side == "left" else n_left
            hi = lo + (n_left if side == "left" else n_right)
            on_i = (self._pair_i >= lo) & (self._pair_i < hi)
            on_j = ~self._pair_box & (self._pair_j >= lo) & (self._pair_j < hi)
            pairs = np.flatnonzero(on_i | on_j)
            is_second = on_j[pairs]
            cap = np.where(is_second, self._pair_j[pairs], self._pair_i[pairs]) - lo
            return _Moving(pairs, is_second, self._arms[side].cap_joint[cap])

        self._moving = {side: moving(side) for side in SIDES}
        self.lower = np.concatenate([self._arms[s].lower for s in SIDES])
        self.upper = np.concatenate([self._arms[s].upper for s in SIDES])
        self.left_indices = list(range(ARM_DOF))
        self.right_indices = list(range(ARM_DOF, 2 * ARM_DOF))

        # Floors, from each pair's clearance at the straight-down home pose
        # (the shapes are conservative, and some overlap there), the closest
        # of the fingers as frozen, closed and open:
        # - an upper arm against the body keeps the Pyroki solver's hard
        #   guard (.model.upper_arm_guard_floor);
        # - every other pair stays _MIN_CLEARANCE_M apart, or as close as
        #   the Pyroki solver's collision cost starts pushing back
        #   (.model.collision_cost_params) where that is closer.
        zero = np.zeros(2 * ARM_DOF)
        home = self._distances(zero)
        for grip in ((0.0, 0.0), (1.0, 1.0)):
            self.set_gripper(*grip)
            home = np.minimum(home, self._distances(zero))
        # Until told otherwise the grippers are taken as fully open, like the
        # teleop solver.
        start = np.minimum(home - _MARGIN_REST_BUFFER, collision_margin)
        self.floors = np.minimum(start, _MIN_CLEARANCE_M)
        guard_links = set(upper_arm_guard_links(model))
        for k, (arm_cap, shape) in enumerate(capsule_pairs + box_pairs):
            body_link = box_link[shape] if self._pair_box[k] else cap_link[shape]
            arm_link = link_names[cap_link[arm_cap]]
            if link_names[body_link] in guard_links and arm_link.endswith("_e1"):
                self.floors[k] = upper_arm_guard_floor(float(home[k]))
        _logger.info(
            "Arm IK ready: %d clearance pairs (%d against the body).",
            self.num_pairs,
            n_body,
        )

    # -- configuration -----------------------------------------------------

    @property
    def num_pairs(self) -> int:
        return len(self._pair_i)

    @property
    def body_pairs(self) -> np.ndarray:
        """Which clearance pairs are an arm against the body (the rest are
        arm against arm). They come first, in the order of
        :meth:`.collision.ArmBodyCollision.distances`."""
        return self._body_mask

    @property
    def joint_names(self) -> list[str]:
        """URDF joint names in joint-vector order."""
        return urdf_arm_joint_names(is_left=True) + urdf_arm_joint_names(is_left=False)

    def indices(self, side: Side) -> list[int]:
        return self.left_indices if side == "left" else self.right_indices

    def set_lift(self, raised: float | None) -> None:
        """Place the parts the lift leaves behind for its live height.

        ``raised`` is the height as a fraction of travel (1 = fully raised),
        as the Jelly lift reports it. IK solves in the torso frame, which
        rides the lift, so lowering it brings the deck and lower column up
        toward the arms. ``None``, or a version without a lift: no effect.
        """
        if raised is None or self._lift_travel <= 0.0:
            return
        raised = min(max(float(raised), 0.0), 1.0)
        if abs(raised - self._lift_raised) < 1e-4:
            return
        lowered = (1.0 - raised) * self._lift_travel
        offset = self._body_shift * lowered
        self._body = (
            self._body_a0 + offset,
            self._body_b0 + offset,
            self._box_c0 + self._box_shift * lowered,
        )
        self._lift_raised = raised

    def set_gripper(self, left: float, right: float) -> None:
        """Place the fingers for the live gripper openings.

        ``left`` / ``right`` are fractions (0 = closed, 1 = fully open), as
        the arms report them. IK freezes the finger joints at zero, and each
        finger slides along its joint axis to where the opening puts it
        (:func:`.model.finger_slide`). Until the first call the grippers are
        taken as fully open. No effect on a gripper without sliding fingers.
        """
        if self._finger_stroke == 0.0:
            return
        grip = (min(max(float(left), 0.0), 1.0), min(max(float(right), 0.0), 1.0))
        if (
            self._grip is not None
            and max(abs(a - b) for a, b in zip(grip, self._grip)) < 1e-3
        ):
            return
        slide = {}
        for side, g in zip(SIDES, grip):
            arm = self._arms[side]
            joint = self._finger_closed + g * self._finger_stroke
            slide[side] = arm.cap_slide * (joint * arm.cap_finger)[:, None]
        # Swapped whole, with the held-arm cache, so readers see one or the
        # other.
        self._slide = slide
        self._other_cache = {}
        self._grip = grip

    # -- kinematics --------------------------------------------------------

    def frames(self, q_arm: np.ndarray, side: Side) -> ArmFrames:
        """Forward kinematics of one arm's seven joints."""
        arm = self._arms[side]
        # A copy: the frames remember their joints (see _held).
        q_arm = np.array(q_arm, dtype=np.float64)
        c = np.cos(q_arm)[:, None, None]
        s = np.sin(q_arm)[:, None, None]
        spins = c * np.eye(3) + s * arm.skew + (1.0 - c) * arm.outer
        rotations = np.empty((ARM_DOF, 3, 3))
        origins = np.empty((ARM_DOF, 3))
        r = np.eye(3)
        p = np.zeros(3)
        for k in range(ARM_DOF):
            p = p + r @ arm.fixed_p[k]
            r = r @ arm.fixed_r[k] @ spins[k]
            rotations[k] = r
            origins[k] = p
        axes = (rotations @ arm.axes[:, :, None])[:, :, 0]
        mount_r = r @ arm.mount_r
        mount_p = p + r @ arm.mount_p
        tip = (mount_p + mount_r @ self._offset, mount_r)
        jr = rotations[arm.cap_joint]
        jo = origins[arm.cap_joint]
        slide = self._slide.get(side)
        a_local = arm.cap_a if slide is None else arm.cap_a + slide
        b_local = arm.cap_b if slide is None else arm.cap_b + slide
        cap_a = (jr @ a_local[:, :, None])[:, :, 0] + jo
        cap_b = (jr @ b_local[:, :, None])[:, :, 0] + jo
        return ArmFrames(q_arm, origins, axes, tip, cap_a, cap_b)

    def _held(self, q_arm: np.ndarray, side: Side) -> ArmFrames:
        """:meth:`frames` for an arm that isn't moving, cached across calls."""
        q_arm = np.asarray(q_arm, dtype=np.float64)
        cache = self._other_cache
        cached = cache.get(side)
        if cached is None or not np.array_equal(cached.q, q_arm):
            cached = self.frames(q_arm, side)
            cache[side] = cached
        return cached

    def tip(self, q: np.ndarray, side: Side) -> Pose:
        """``side``'s gripper-tip pose for a ``(14,)`` joint vector."""
        q = np.asarray(q, dtype=np.float64)
        return self.frames(q[self.indices(side)], side).tip

    def tips(self, q: np.ndarray) -> tuple[Pose, Pose]:
        return self.tip(q, "left"), self.tip(q, "right")

    # -- clearance ---------------------------------------------------------

    def _geometry(
        self,
        left: ArmFrames,
        right: ArmFrames,
        pairs: np.ndarray | None = None,
        *,
        distance_only: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Signed distances of ``pairs``, the closest points on the first and
        second shape's axis (a box's: its surface), and the unit normals
        along which moving the first point apart opens the gap. With
        ``distance_only`` only the distances are filled in."""
        body_a, body_b, box_c = self._body
        a = np.concatenate([left.cap_a, right.cap_a, body_a])
        b = np.concatenate([left.cap_b, right.cap_b, body_b])
        if pairs is None:
            pairs = np.arange(self.num_pairs)
        i, j = self._pair_i[pairs], self._pair_j[pairs]
        box = self._pair_box[pairs]
        n = len(pairs)
        d = np.empty(n)
        p1, p2, normal = np.empty((n, 3)), np.empty((n, 3)), np.empty((n, 3))
        cap = ~box
        if cap.any():
            ci, cj = i[cap], j[cap]
            q1, q2 = _segment_points(a[ci], b[ci], a[cj], b[cj])
            # Pyroki's sphere distance adds 1e-6 to each component before the
            # norm.
            gap = q2 - q1 + 1e-6
            length = np.sqrt((gap * gap).sum(1))
            d[cap] = length - self._radius[ci] - self._radius[cj]
            p1[cap], p2[cap] = q1, q2
            normal[cap] = (q1 - q2) / np.maximum(length, 1e-9)[:, None]
        if box.any():
            bi, bj = i[box], j[box]
            sdf, point, grad = _capsule_box(
                a[bi],
                b[bi],
                box_c[bj],
                self._box_r[bj],
                self._box_half[bj],
                distance_only=distance_only,
            )
            d[box] = sdf - self._radius[bi]
            if point is not None and grad is not None:
                p1[box] = point
                p2[box] = point - grad * sdf[:, None]
                normal[box] = grad
        return d, p1, p2, normal

    def _distances(self, q: np.ndarray) -> np.ndarray:
        if not self.num_pairs:
            return np.zeros(0)
        q = np.asarray(q, dtype=np.float64)
        left = self.frames(q[:ARM_DOF], "left")
        right = self.frames(q[ARM_DOF:], "right")
        return self._geometry(left, right, distance_only=True)[0]

    def _clear(self, q: np.ndarray, floors: np.ndarray) -> bool:
        """Whether every pair at ``q`` stays above ``floors`` (less the
        slack). A box pair whose bounding sphere is already clear skips the
        exact search."""
        if not self.num_pairs:
            return True
        q = np.asarray(q, dtype=np.float64)
        left = self.frames(q[:ARM_DOF], "left")
        right = self.frames(q[ARM_DOF:], "right")
        limit = floors - _FLOOR_SLACK_M
        box = np.flatnonzero(self._pair_box)
        check = np.ones(self.num_pairs, dtype=bool)
        if len(box):
            a = np.concatenate([left.cap_a, right.cap_a])
            b = np.concatenate([left.cap_b, right.cap_b])
            i, j = self._pair_i[box], self._pair_j[box]
            centre = self._body[2][j]
            axis = b[i] - a[i]
            t = ((centre - a[i]) * axis).sum(1) / np.maximum(
                (axis * axis).sum(1), 1e-12
            )
            near = a[i] + axis * np.clip(t, 0.0, 1.0)[:, None]
            bound = (
                np.linalg.norm(near - centre, axis=1)
                - self._box_reach[j]
                - self._radius[i]
            )
            check[box[bound >= limit[box]]] = False
        pairs = np.flatnonzero(check)
        d = self._geometry(left, right, pairs, distance_only=True)[0]
        return bool(np.all(d >= limit[pairs]))

    def clearances(self, q: np.ndarray) -> np.ndarray:
        """Each pair's distance less its floor (negative: inside the floor)."""
        return self._distances(q) - self.floors

    def _rows(
        self,
        side: Side,
        own: ArmFrames,
        geometry: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        floors: np.ndarray,
        rate: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Linearised ``rows @ dq <= bounds`` keeping ``side``'s pairs above
        ``floors``: each may close at most ``rate`` of its distance to the
        floor, and one already inside may not close at all."""
        d, p1, p2, normals = geometry
        near = d < floors + _INFLUENCE_M
        if not near.any():
            return _NO_ROWS
        moving = self._moving[side]
        second = moving.second[near]
        point = np.where(second[:, None], p2[near], p1[near])
        normal = normals[near].copy()
        normal[second] *= -1.0  # the moving capsule's outward direction
        # d(distance)/dq_k = normal . (axis_k x (point - origin_k)) for the
        # joints upstream of the moving capsule.
        lever = point[:, None, :] - own.origins[None, :, :]
        grad = (_cross(own.axes[None], lever) * normal[:, None, :]).sum(-1)
        grad[np.arange(ARM_DOF)[None, :] > moving.joint[near][:, None]] = 0.0
        bounds = np.maximum(d[near] - floors[near], 0.0) * rate
        return -grad, bounds

    def _box(
        self, side: Side, q_arm: np.ndarray, step: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-joint step bounds: inside the limits, at most ``step``.

        A joint already outside its range (a measured offset, a hand-moved
        arm) may only move back toward it.
        """
        arm = self._arms[side]
        lo = np.minimum(arm.lower, q_arm)
        hi = np.maximum(arm.upper, q_arm)
        # The elbow stays bent the way it is. Straight is a singularity, and
        # least squares happily drives a reaching arm through it onto the
        # elbow's short far-side range, where it pins on that limit short of
        # the target. Within _ELBOW_STRAIGHT_RAD of straight it may only bend
        # further.
        elbow = float(q_arm[_ELBOW])
        if elbow < 0.0:
            hi[_ELBOW] = min(hi[_ELBOW], max(elbow, -_ELBOW_STRAIGHT_RAD))
        elif elbow > 0.0:
            lo[_ELBOW] = max(lo[_ELBOW], min(elbow, _ELBOW_STRAIGHT_RAD))
        return np.maximum(lo - q_arm, -step), np.minimum(hi - q_arm, step)

    @staticmethod
    def _error(tip: Pose, target: Pose) -> np.ndarray:
        return np.concatenate(
            [target[0] - tip[0], rotation_vector(target[1] @ tip[1].T)]
        )

    def _start(
        self, q: np.ndarray, side: Side
    ) -> tuple[np.ndarray, ArmFrames, ArmFrames, np.ndarray, tuple | None, np.ndarray]:
        """Shared set-up for a move of ``side`` from ``q``: the arm's joints,
        both arms' frames, the clearance geometry and the floors (a pair
        already inside its floor may not get any closer)."""
        q = np.asarray(q, dtype=np.float64)
        idx = self.indices(side)
        q_arm = q[idx].copy()
        other_side: Side = "right" if side == "left" else "left"
        own = self.frames(q_arm, side)
        other = self._held(q[self.indices(other_side)], other_side)
        pairs = self._moving[side].pairs
        if not len(pairs):
            return q_arm, own, other, pairs, None, np.zeros(0)
        geometry = self._pair_geometry(side, own, other)
        floors = np.minimum(self.floors[pairs], geometry[0])
        return q_arm, own, other, pairs, geometry, floors

    def _pair_geometry(
        self,
        side: Side,
        own: ArmFrames,
        other: ArmFrames,
        *,
        distance_only: bool = False,
    ):
        left, right = (own, other) if side == "left" else (other, own)
        return self._geometry(
            left, right, self._moving[side].pairs, distance_only=distance_only
        )

    # -- IK -----------------------------------------------------------------

    def solve(
        self,
        q: np.ndarray,
        side: Side,
        target: Pose,
        *,
        max_iterations: int = 80,
        position_tolerance: float = 1e-4,
        orientation_tolerance: float = math.radians(0.05),
        max_step: float = 0.35,
    ) -> IkResult:
        """The configuration of ``side`` closest to a tip ``target``.

        Damped least squares (Levenberg-Marquardt with error-scaled damping,
        which also converges toward targets out of reach) from ``q``, within
        the joint limits and the clearance floors. Only ``side``'s joints
        move. Each step is the smallest joint motion that reduces the error,
        so the solution stays on the seed's branch: seeding from the last
        solution while a target is dragged keeps the arm's pose continuous.
        """
        q_full = np.asarray(q, dtype=np.float64).copy()
        target = (
            np.asarray(target[0], dtype=np.float64),
            np.asarray(target[1], dtype=np.float64),
        )
        q_arm, own, other, pairs, geometry, floors = self._start(q_full, side)
        error = self._error(own.tip, target)
        cost = 0.5 * float(error @ (_WEIGHTS * error))
        damping = 1.0
        iterations = rejected = stalled = 0
        for iterations in range(1, max_iterations + 1):
            if (
                float(np.linalg.norm(error[:3])) <= position_tolerance
                and float(np.linalg.norm(error[3:])) <= orientation_tolerance
            ):
                break
            jac = own.jacobian(own.tip[0])
            jw = jac.T * _WEIGHTS
            hessian = jw @ jac + (damping * cost + 1e-6) * np.eye(ARM_DOF)
            lo, hi = self._box(side, q_arm, max_step)
            rows, bounds = (
                self._rows(side, own, geometry, floors, 0.8) if geometry else _NO_ROWS
            )
            step = _qp(hessian, -(jw @ error), lo, hi, rows, bounds)
            if step is None:
                break
            trial = self.frames(q_arm + step, side)
            trial_error = self._error(trial.tip, target)
            trial_cost = 0.5 * float(trial_error @ (_WEIGHTS * trial_error))
            trial_geometry = None
            safe = True
            if geometry:
                trial_geometry = self._pair_geometry(side, trial, other)
                safe = bool(np.all(trial_geometry[0] >= floors - _FLOOR_SLACK_M))
            if safe and trial_cost < cost:
                improvement = cost - trial_cost
                q_arm, own, error, cost = q_arm + step, trial, trial_error, trial_cost
                geometry = trial_geometry
                damping = max(damping * 0.3, 1e-3)
                rejected = 0
                # Out of reach, the error shrinks ever more slowly toward the
                # closest pose: stop once it has stopped paying off.
                stalled = stalled + 1 if improvement < 1e-3 * cost else 0
                if stalled >= 4 or float(np.max(np.abs(step))) < 1e-7:
                    break
            else:
                damping *= 10.0
                rejected += 1
                if rejected >= 8:
                    break
        q_full[self.indices(side)] = q_arm
        return IkResult(
            q=q_full.astype(np.float32),
            tip=(own.tip[0].astype(np.float32), own.tip[1].astype(np.float32)),
            position_error=float(np.linalg.norm(own.tip[0] - target[0])),
            orientation_error=rotation_angle(own.tip[1], target[1]),
            iterations=iterations,
        )

    def servo_step(
        self,
        q: np.ndarray,
        side: Side,
        target: Pose,
        max_joint_step: float,
        *,
        approach_rate: float = 0.3,
    ) -> StepResult:
        """One control tick of ``side`` toward a nearby tip ``target``.

        A single damped least-squares QP step: each joint moves at most
        ``max_joint_step`` and stays inside its limits, and each clearance
        pair closes at most ``approach_rate`` of its distance to the floor.
        A step the exact capsule distances reject is halved, and dropped
        after four halvings, so the arm stops short of a collision rather
        than entering it.
        """
        q_full = np.asarray(q, dtype=np.float64).copy()
        target = (np.asarray(target[0], np.float64), np.asarray(target[1], np.float64))
        q_arm, own, other, pairs, geometry, floors = self._start(q_full, side)
        error = self._error(own.tip, target)
        jac = own.jacobian(own.tip[0])
        jw = jac.T * _WEIGHTS
        hessian = jw @ jac + 1e-5 * np.eye(ARM_DOF)
        gradient = jw @ error
        lo, hi = self._box(side, q_arm, max_joint_step)
        rows, bounds = (
            self._rows(side, own, geometry, floors, approach_rate)
            if geometry
            else _NO_ROWS
        )
        step = _qp(hessian, -gradient, lo, hi, rows, bounds)
        if step is None:
            return StepResult(q_full.astype(np.float32), True)
        # Blocked: the bounds or a clearance row cut the free step short.
        free = np.linalg.solve(hessian, gradient)
        size = float(np.linalg.norm(free))
        blocked = size > 1e-7 and float(np.linalg.norm(free - step)) > 1e-3 * size
        if geometry:
            for _ in range(5):
                d = self._pair_geometry(
                    side,
                    self.frames(q_arm + step, side),
                    other,
                    distance_only=True,
                )[0]
                if np.all(d >= floors - _FLOOR_SLACK_M):
                    break
                step = step * 0.5
                blocked = True
            else:
                step = np.zeros(ARM_DOF)
        q_full[self.indices(side)] = q_arm + step
        return StepResult(q_full.astype(np.float32), blocked)

    # -- joint-space paths --------------------------------------------------

    def path(
        self,
        q_from: np.ndarray,
        q_to: np.ndarray,
        steps: int,
        *,
        max_extra: int = 250,
    ) -> list[np.ndarray]:
        """A smoothstep joint-space path that keeps the clearance floors.

        Waypoints follow the smoothstep from ``q_from`` to ``q_to``. One that
        would take a pair below its floor (or closer than at the start, for a
        pair already inside it) is replaced by a step pushed out along the
        clearance gradients, each joint moving at most twice the nominal
        step; extra waypoints then finish the move if the detour left it
        short (at most ``max_extra``). Returns at least ``steps`` ``(14,)``
        waypoints; the last is ``q_to`` unless no clear way there was found.
        """
        q_from = np.asarray(q_from, dtype=np.float64)
        q_to = np.asarray(q_to, dtype=np.float64)
        alphas = [(i + 1) / steps for i in range(steps)]
        alphas = [t * t * (3.0 - 2.0 * t) for t in alphas]
        if not self.num_pairs:
            return [(q_from * (1 - a) + q_to * a).astype(np.float32) for a in alphas]
        floors = np.minimum(self.floors, self._distances(q_from))
        nominal = float(np.max(np.abs(q_to - q_from))) * 1.5 / max(steps, 1)
        bound = max(2.0 * nominal, 2e-3)
        out: list[np.ndarray] = []
        q = q_from.copy()
        extra = 0
        for k in range(len(alphas) + max_extra):
            if k < len(alphas):
                goal = q_from * (1 - alphas[k]) + q_to * alphas[k]
            elif float(np.max(np.abs(q - q_to))) > 1e-6:
                goal = q_to
                extra += 1
            else:
                break
            if float(np.max(np.abs(goal - q))) <= bound + 1e-9 and self._clear(
                goal, floors
            ):
                q = goal
            else:
                q = self._project(q, goal, floors, bound)
            out.append(q.astype(np.float32))
        if extra:
            _logger.debug(
                "Arm path needed %d extra waypoints to clear the body.", extra
            )
        return out

    def _project(
        self, q: np.ndarray, goal: np.ndarray, floors: np.ndarray, bound: float
    ) -> np.ndarray:
        """Move from ``q`` toward ``goal`` by at most ``bound`` per joint,
        keeping every pair above its floor (one arm, then the other)."""
        q = q.copy()
        for side in SIDES:
            idx = self.indices(side)
            if float(np.max(np.abs(goal[idx] - q[idx]))) < 1e-12:
                continue
            other_side: Side = "right" if side == "left" else "left"
            own = self.frames(q[idx], side)
            other = self.frames(q[self.indices(other_side)], other_side)
            pairs = self._moving[side].pairs
            side_floors = floors[pairs]
            geometry = self._pair_geometry(side, own, other)
            rows, bounds = self._rows(side, own, geometry, side_floors, 0.8)
            lo, hi = self._box(side, q[idx], bound)
            step = _qp(np.eye(ARM_DOF), -(goal[idx] - q[idx]), lo, hi, rows, bounds)
            if step is None:
                continue
            for _ in range(5):
                trial = self.frames(q[idx] + step, side)
                d = self._pair_geometry(side, trial, other, distance_only=True)[0]
                if np.all(d >= side_floors - _FLOOR_SLACK_M):
                    q[idx] += step
                    break
                step = step * 0.5
        return q
