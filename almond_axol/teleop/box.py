"""Box-mode (bimanual carry) target geometry.

Pure-NumPy helpers for the IK worker's box-mode tracking (the pose blending
ones double for the per-arm re-engage ramp). Everything is expressed in the
robot's world frame (FLU:
+x forward, +y left, +z up) using the ``(pos (3,), rot (3, 3))`` pose format
:class:`~almond_axol.kinematics.solver.KinematicsSolver` speaks.

The **box frame** sits at the midpoint of the two gripper mount frames. Its
axes are ``x`` forward, ``y`` *lateral* — the horizontal direction from the
right gripper to the left one — and ``z`` up. The grippers live at
``center ± y * width / 2`` and hold the box the way two flat hands clamp its
sides: the fingers point *forward* (the gripper link's ``-Z``, the direction
the fingers point, goes along the box ``+x``) and the flat outer face of the
closed fingers — the gripper link's ``±X`` side, the jaw's open/close axis —
faces the box centre, so the box is held between the sides of the two
grippers by friction. Which of the two flat faces (``+X`` or ``-X``) is
turned toward the box is chosen per gripper as the one closest to its
current rotation, so the wrist never flips through 180° to get there. An
optional *tilt* yaws each gripper inward by a few degrees so the wedge-shaped
finger face lies flush on the box side instead of touching along its heel.
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


def approach_axis(rot: np.ndarray) -> np.ndarray:
    """Direction the fingers point for a gripper mount rotation (its ``-Z``)."""
    return -np.asarray(rot, dtype=np.float32)[:, 2]


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
    ``face`` records which flat face (``±1``, the gripper's ``±X`` side) each
    gripper turns toward the box, chosen at the snap; ``tilt`` is the
    grippers' inward yaw (rad), seeded from the config and jogged live.
    Together they give each gripper's rotation relative to the box frame
    (:meth:`grip_rel`), so the pair turns rigidly with the box.
    ``align_start`` holds where each gripper actually was at the snap,
    expressed in the box frame, for the blend into the parallel
    configuration.
    """

    center: np.ndarray
    rot: np.ndarray
    width: float
    face: dict[str, float]
    tilt: float
    align_start: dict[str, Pose]
    align_t0: float
    align_duration: float
    jog_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    jog_yaw: float = 0.0
    # Wall time of the previous jog integration step (None before the first).
    jog_t: float | None = None

    def grip_rel(self) -> dict[str, np.ndarray]:
        """Each gripper's rotation relative to the box frame (see :func:`side_clamp_rotation`)."""
        return {
            side: side_clamp_rotation(sign, self.face[side], self.tilt)
            for side, sign in _SIDE_SIGN.items()
        }

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


# Which side of the box each gripper sits on, as the sign of its lateral
# coordinate: the left gripper is at +y (the box's lateral axis runs from the
# right gripper to the left one).
_SIDE_SIGN = {"left": 1.0, "right": -1.0}


def side_clamp_rotation(sign: float, face: float, tilt: float) -> np.ndarray:
    """Box-frame rotation of a gripper clamping the box side with a flat face.

    ``sign`` is the gripper's side (+1 left, -1 right; see ``_SIDE_SIGN``),
    ``face`` which of its flat faces is turned toward the box (+1: the
    gripper's ``+X`` side, -1: its ``-X`` side) and ``tilt`` (rad) the inward
    yaw: 0 points the fingers straight along the box ``+x``; a positive tilt
    turns the fingertips toward the box centre so a finger face that narrows
    toward the tip (a wedge with half-angle ``tilt``) lies flush on the side.
    """
    # Fingers along the box +x (the gripper's -Z), the chosen flat face
    # toward the centre (-sign * lateral), Y completing a right-handed frame.
    x_g = np.array((0.0, -sign * face, 0.0), dtype=np.float64)
    z_g = np.array((-1.0, 0.0, 0.0), dtype=np.float64)
    y_g = np.cross(z_g, x_g)
    r0 = np.stack([x_g, y_g, z_g], axis=1)
    if tilt:
        r0 = rodrigues(_UP, -sign * tilt).astype(np.float64) @ r0
    return r0.astype(np.float32)


def parallel_grip_rel(
    current: dict[str, np.ndarray], rot: np.ndarray, tilt: float
) -> dict[str, np.ndarray]:
    """Box-relative rotations of the side-clamping gripper pair.

    ``current`` holds each gripper's world rotation, ``rot`` the box rotation
    and ``tilt`` the inward yaw (rad, see :func:`side_clamp_rotation`). Both
    flat faces of a gripper clamp equally well, so the one needing the
    smaller turn from ``current`` is used (:func:`choose_faces`) — the wrist
    never has to roll through 180° to reach the grasp.
    """
    faces = choose_faces(current, rot, tilt)
    return {
        side: side_clamp_rotation(sign, faces[side], tilt)
        for side, sign in _SIDE_SIGN.items()
    }


def choose_faces(
    current: dict[str, np.ndarray], rot: np.ndarray, tilt: float
) -> dict[str, float]:
    """Per gripper, the flat face (``±1``) nearest its ``current`` world rotation."""
    out: dict[str, float] = {}
    for side, sign in _SIDE_SIGN.items():
        out[side] = min(
            (1.0, -1.0),
            key=lambda face: rotation_angle(
                current[side], rot @ side_clamp_rotation(sign, face, tilt)
            ),
        )
    return out


def pair_aligned(
    left: Pose,
    right: Pose,
    width_min: float,
    width_max: float,
    tilt: float,
    tol_deg: float,
) -> bool:
    """True when the grippers already form the side-clamping pair.

    Each gripper is within ``tol_deg`` of the rotation a box-mode engage
    would blend it to (fingers forward, a flat face toward the other
    gripper; see :func:`parallel_grip_rel`) and their separation is inside
    ``[width_min, width_max]`` — so switching to box mode from here costs
    (almost) no alignment blend.
    """
    _center, rot, width = box_frame(left[0], right[0])
    if not (width_min <= width <= width_max):
        return False
    rel = parallel_grip_rel({"left": left[1], "right": right[1]}, rot, tilt)
    tol = math.radians(tol_deg)
    return all(
        rotation_angle(pose[1], rot @ rel[side]) <= tol
        for side, pose in (("left", left), ("right", right))
    )


def snap_box(
    left: Pose,
    right: Pose,
    now: float,
    align_duration: float,
    width_min: float,
    width_max: float,
    tilt: float = 0.0,
) -> BoxState:
    """Build the box state for an engage snap from the current gripper poses.

    ``tilt`` is the grippers' starting inward yaw in radians (see
    :func:`side_clamp_rotation`); the jog changes it live afterwards.
    """
    center, rot, width = box_frame(left[0], right[0])
    width = float(np.clip(width, width_min, width_max))
    face = choose_faces({"left": left[1], "right": right[1]}, rot, tilt)
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
        face=face,
        tilt=float(tilt),
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
    ideal = ideal_gripper_poses(center, rot, state.width, state.grip_rel())
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


def elbow_swivel_hint(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
    side_sign: float,
    out_angle: float,
) -> np.ndarray:
    """Where the elbow should be for an "elbows out" carry of the box.

    An arm with a fixed shoulder and wrist still has one free motion: the
    elbow swings on a circle about the shoulder-to-wrist line. Box mode's
    gripper poses — parallel, fingers forward, pulled toward each other — are
    ones a person never makes with a controller in hand, and from an ordinary
    reach the solver's nearest solution folds the elbows *inward*, into the
    torso. This hint pins the swivel instead: it is the current elbow
    position rotated about the shoulder-wrist axis so that it sits
    ``out_angle`` radians from straight down toward the arm's outboard side
    (``side_sign`` +1 for the left arm, whose outboard is +y, -1 for the
    right). ``0`` hangs the elbow directly under the axis like a relaxed arm;
    ``pi/2`` holds it out level with the shoulder. Because only the swivel
    changes — same shoulder-to-elbow radius, same elbow angle — the hint is
    exactly reachable, so the IK's elbow cost pulls the free motion without
    fighting the gripper pose (see ``KinematicsSolver.ik(elbow_weight=...)``).

    Degenerate cases return the current elbow: a wrist at the shoulder, a
    perfectly straight arm (no swivel to speak of), or an axis parallel to the
    wanted direction (no outboard component to project).
    """
    shoulder = np.asarray(shoulder, dtype=np.float64)
    elbow = np.asarray(elbow, dtype=np.float64)
    axis = np.asarray(wrist, dtype=np.float64) - shoulder
    n = float(np.linalg.norm(axis))
    if n < 1e-6:
        return elbow.astype(np.float32)
    a = axis / n
    e = elbow - shoulder
    along = float(e @ a)
    radial = e - along * a
    r = float(np.linalg.norm(radial))
    if r < 1e-6:
        return elbow.astype(np.float32)
    want = math.cos(out_angle) * -_UP + math.sin(out_angle) * side_sign * _LEFT
    want = want - float(want @ a) * a
    w = float(np.linalg.norm(want))
    if w < 1e-6:
        return elbow.astype(np.float32)
    return (shoulder + along * a + r * (want / w)).astype(np.float32)
