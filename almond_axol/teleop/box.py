"""Box-mode (bimanual carry) target geometry.

Pure-NumPy helpers for the IK worker's box-mode tracking (the pose blending
ones double for the per-arm re-engage ramp). Everything is expressed in the
robot's world frame (FLU:
+x forward, +y left, +z up) using the ``(pos (3,), rot (3, 3))`` pose format
:class:`~almond_axol.kinematics.solver.KinematicsSolver` speaks.

The **box frame** sits at the midpoint of the two gripper mount frames. Its
axes are ``x`` forward, ``y`` *lateral* — the horizontal direction from the
right gripper to the left one — and ``z`` up: the frame is always level, and
turns only about the vertical (yaw), so the grippers are always held level
with the hands straight out and the one thing the operator steers besides
the pair's position is its heading on the table plane, to line up with the
box. The grippers live at ``center ± y * width / 2`` and hold the box the
way two flat hands
clamp its sides: the fingers point *forward* (the gripper link's ``-Z``, the
direction the fingers point, goes along ``+x``) and the flat outer face of
the closed fingers — the gripper link's ``±X`` side, the jaw's open/close
axis — faces the box centre, so the box is held between the sides of the two
grippers by friction. Which of the two flat faces (``+X`` or ``-X``) is
turned toward the box is chosen per gripper as the one closest to its
current rotation, so the wrist never flips through 180° to get there (or
pinned by the config, for a tool that only clamps with one side). An
optional *tilt* yaws each gripper inward by a few degrees so the wedge-shaped
finger face lies flush on the box side instead of touching along its heel.

Where on the gripper the box is actually touched is the **tool geometry**
(:class:`ToolGeometry`): the yaw at which the tool's contact face is
parallel to the box side and where that face sits relative to the mount.
``width`` is the separation of the two *contact faces* — the box size — and
the mounts are placed behind them; ``tilt`` is a trim on top of the tool's
flush yaw and pivots about the contact face, so trimming keeps the face
where it is. The stock URDF gripper has a trivial geometry (the mount
separation is the width, the flat side faces the box at tilt 0); the parcel
gripper's is derived from its mechanism (:func:`parcel_tool`).
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


@dataclass(frozen=True)
class ToolGeometry:
    """Where a gripper touches the box side, in its own mount frame.

    Attributes:
        flush_tilt: Inward yaw (rad) at which the tool's contact face is
            parallel to the box side. Box mode holds each gripper at this
            yaw plus the operator's tilt trim.
        foot_fwd: Distance (m) along the fingers (the mount's ``-Z``) from
            the mount origin to the *foot* — the point of the contact face
            nearest the mount origin, with the gripper at ``flush_tilt``.
        foot_in: The foot's distance (m) toward the box, along the flat
            face turned toward it (the mount's ``±X``, see
            :func:`side_clamp_rotation`).

    The foot is what box mode places at ``±width / 2``: the two contact
    faces are then ``width`` apart whatever tool is fitted, and the tilt
    trim rotates the gripper about its foot so the face stays put.
    """

    flush_tilt: float = 0.0
    foot_fwd: float = 0.0
    foot_in: float = 0.0

    def foot(self, face: float) -> np.ndarray:
        """Mount-frame vector from the mount origin to the contact foot.

        ``face`` (``±1``) is which flat side (``±X``) faces the box.
        """
        return np.array((face * self.foot_in, 0.0, -self.foot_fwd), dtype=np.float32)


# The stock URDF gripper as box mode always modelled it: the mount
# separation is the width and a flat side faces the box at tilt 0.
URDF_TOOL = ToolGeometry()

# Parcel gripper mechanism (from its CAD, metres): the moving blade's hinge
# sits this far ahead of the mount flange, on the flange axis, and the blade's
# contact face is this far from the hinge axis.
PARCEL_PIVOT_FWD_M = 0.036
PARCEL_FACE_R_M = 0.029


def parcel_tool(
    open_deg: float,
    pivot_fwd: float = PARCEL_PIVOT_FWD_M,
    face_r: float = PARCEL_FACE_R_M,
) -> ToolGeometry:
    """Contact geometry of the parcel gripper with its blade folded open.

    The parcel gripper is two flat blades side by side along the fingers
    direction: a fixed one on the mount axis and a hinged one that swings
    toward the box side, about a vertical hinge ``pivot_fwd`` ahead of the
    flange, until it meets its mechanical stop ``open_deg`` from closed.
    Folded that far back it lies alongside the wrist, and the box side is
    clamped by that blade's flat face — a large patch *behind* the hinge,
    roughly centred on the wrist, so a straight lateral squeeze needs
    almost no wrist moment to keep it flat. For the face to lie on the box
    side the fixed blade must point ``180° - open_deg`` inward: that is the
    flush tilt (38.5° at the CAD's 141.5° stop). The foot is the point of
    the folded face nearest the mount origin: the face plane is ``face_r``
    from the hinge, so its distance from the mount origin along its normal
    is ``pivot_fwd * sin(open) + face_r``.

    The fixed blade's tip is ahead of the hinge and, at the CAD stop, ~9 mm
    *past* the face plane (it would sit exactly on it at ~146°), so with the
    face flush it hooks the box's front corner or digs in; the tilt trim
    backs the face off to touch tip-and-heel instead. Both flush at once
    needs the stop at ~146° (see the config docs).
    """
    phi = math.radians(open_deg)
    # Face normal toward the box, in (forward, inboard) mount coordinates:
    # the blade closed has its face normal pointing outboard (toward the
    # fixed blade); the fold turns it through ``phi`` toward the box.
    n_fwd, n_in = math.sin(phi), -math.cos(phi)
    c = pivot_fwd * math.sin(phi) + face_r
    return ToolGeometry(flush_tilt=math.pi - phi, foot_fwd=c * n_fwd, foot_in=c * n_in)


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

    ``rotation`` is yaw-only (``z`` up, ``y`` along the horizontal direction
    from the right gripper to the left), so a pair whose hands are staggered
    fore/aft engages with that heading rather than being swung square.
    ``width`` is the full 3-D separation of the two mount frames (the grip
    width box mode tracks is the lateral separation of the tool's contact
    faces instead, :func:`contact_width`).
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


def twist_about(rot: np.ndarray, axis: np.ndarray) -> float:
    """The part of a rotation that is about the unit vector ``axis`` (rad).

    Swing-twist split of ``rot``: the twist is the angle left once the swing
    that moves ``axis`` is taken out (``2 * atan2(q_v . axis, q_w)`` for the
    rotation's quaternion). For a rotation purely about ``axis`` it is that
    angle; for a hand that also pitches or rolls it is how far the hand
    turned about the room's up axis — the only component of the leader's
    rotation box mode follows. Right-handed: positive is counter-clockwise
    looking down ``axis``.
    """
    r = np.asarray(rot, dtype=np.float64)
    a = np.asarray(axis, dtype=np.float64)
    v = np.array((r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]))
    return 2.0 * math.atan2(float(v @ a), 1.0 + float(np.trace(r)))


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

    ``center`` / ``rot`` are the box pose *at the snap* (``rot`` is level,
    yaw only); every frame the leader controller's translation since its own
    snap is applied to the centre and the vertical-axis part of its rotation
    (:func:`twist_about` the room's up) turns the pair about that centre — the box rides on
    the leader gripper's clutch mapping, position and heading only, the
    hand's pitch and roll are ignored (see ``IKWorker``). The thumbsticks own
    the other two numbers:
    ``width``, the separation of the two contact faces (the box size), and
    ``tilt``, the grippers' inward yaw trim (rad, seeded from the config) on
    top of the tool's flush yaw. ``face`` records which flat face (``±1``,
    the gripper's ``±X`` side) each gripper turns toward the box, chosen at
    the snap; with ``tilt`` and ``tool`` it gives each gripper's rotation
    relative to the box frame (:meth:`grip_rel`) and where its mount sits
    behind the contact face (:meth:`feet`).
    ``align_start`` holds where each gripper actually was at the snap,
    expressed in the box frame, for the blend into the parallel
    configuration. ``squeeze_tilt`` is a further per-gripper inward yaw
    (rad) the worker adds while the arm is pressing on the box — the
    fingertips leaning in as the squeeze builds so the tip keeps its share
    of the contact (see ``IKWorker._squeeze_tilt``); like the trim it pivots
    about the contact face.
    """

    center: np.ndarray
    rot: np.ndarray
    width: float
    face: dict[str, float]
    tilt: float
    align_start: dict[str, Pose]
    align_t0: float
    align_duration: float
    # Wall time of the previous stick integration step (None before the first).
    stick_t: float | None = None
    tool: ToolGeometry = URDF_TOOL
    # Stick-click state for the grasp toggle (``IKWorker._stick_click_toggle``):
    # last frame's (left, right) click flags and whether a single press is
    # armed to toggle on its release.
    click_prev: tuple[bool, bool] = (False, False)
    click_armed: bool = False
    squeeze_tilt: dict[str, float] = field(
        default_factory=lambda: {"left": 0.0, "right": 0.0}
    )

    def grip_rel(self) -> dict[str, np.ndarray]:
        """Each gripper's rotation relative to the box frame (see :func:`side_clamp_rotation`)."""
        yaw = self.tilt + self.tool.flush_tilt
        return {
            side: side_clamp_rotation(
                sign, self.face[side], yaw + self.squeeze_tilt.get(side, 0.0)
            )
            for side, sign in _SIDE_SIGN.items()
        }

    def feet(self) -> dict[str, np.ndarray]:
        """Each gripper's mount-frame contact-foot vector (see :class:`ToolGeometry`)."""
        return {side: self.tool.foot(self.face[side]) for side in _SIDE_SIGN}

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


Faces = dict[str, float]


def parallel_grip_rel(
    current: dict[str, np.ndarray],
    rot: np.ndarray,
    tilt: float,
    faces: Faces | None = None,
) -> dict[str, np.ndarray]:
    """Box-relative rotations of the side-clamping gripper pair.

    ``current`` holds each gripper's world rotation, ``rot`` the box rotation
    and ``tilt`` the inward yaw (rad, see :func:`side_clamp_rotation`). Both
    flat faces of a gripper clamp equally well, so the one needing the
    smaller turn from ``current`` is used (:func:`choose_faces`) — the wrist
    never has to roll through 180° to reach the grasp — unless ``faces``
    pins a side's face (see :func:`choose_faces`).
    """
    chosen = choose_faces(current, rot, tilt, faces)
    return {
        side: side_clamp_rotation(sign, chosen[side], tilt)
        for side, sign in _SIDE_SIGN.items()
    }


def choose_faces(
    current: dict[str, np.ndarray],
    rot: np.ndarray,
    tilt: float,
    faces: Faces | None = None,
) -> Faces:
    """Per gripper, the flat face (``±1``) nearest its ``current`` world rotation.

    ``faces`` optionally pins a side: a nonzero entry (``±1``) is used as is
    (a tool that clamps with one particular side, like the parcel gripper's
    hinged blade); ``0`` or a missing side picks the nearest face.
    """
    out: Faces = {}
    for side, sign in _SIDE_SIGN.items():
        pinned = (faces or {}).get(side, 0.0)
        if pinned:
            out[side] = 1.0 if pinned > 0 else -1.0
            continue
        out[side] = min(
            (1.0, -1.0),
            key=lambda face: rotation_angle(
                current[side], rot @ side_clamp_rotation(sign, face, tilt)
            ),
        )
    return out


def contact_width(
    left_pos: np.ndarray,
    right_pos: np.ndarray,
    rot: np.ndarray,
    grip_rel: dict[str, np.ndarray],
    feet: dict[str, np.ndarray],
) -> float:
    """Lateral separation of the two contact faces for mounts at these positions.

    Each mount's foot is ``rot @ grip_rel[side] @ feet[side]`` from it (the
    pair held in its box-mode rotations); the width is the distance between
    the two feet along the box frame's lateral axis. With trivial feet
    (:data:`URDF_TOOL`) it is the mounts' own lateral separation.
    """
    lat = np.asarray(rot, dtype=np.float64)[:, 1]
    foot = {
        side: np.asarray(pos, dtype=np.float64)
        + np.asarray(rot, dtype=np.float64)
        @ np.asarray(grip_rel[side], dtype=np.float64)
        @ np.asarray(feet[side], dtype=np.float64)
        for side, pos in (("left", left_pos), ("right", right_pos))
    }
    return float((foot["left"] - foot["right"]) @ lat)


def pair_aligned(
    left: Pose,
    right: Pose,
    width_min: float,
    width_max: float,
    tilt: float,
    tol_deg: float,
    tool: ToolGeometry = URDF_TOOL,
    faces: Faces | None = None,
) -> bool:
    """True when the grippers already form the side-clamping pair.

    Each gripper is within ``tol_deg`` of the rotation a box-mode engage
    would blend it to (fingers forward, a flat face toward the other
    gripper; see :func:`parallel_grip_rel`) and their contact-face
    separation is inside ``[width_min, width_max]`` — so switching to box
    mode from here costs (almost) no alignment blend.
    """
    _center, rot, _width = box_frame(left[0], right[0])
    yaw = tilt + tool.flush_tilt
    chosen = choose_faces({"left": left[1], "right": right[1]}, rot, yaw, faces)
    rel = {
        side: side_clamp_rotation(sign, chosen[side], yaw)
        for side, sign in _SIDE_SIGN.items()
    }
    feet = {side: tool.foot(chosen[side]) for side in _SIDE_SIGN}
    width = contact_width(left[0], right[0], rot, rel, feet)
    if not (width_min <= width <= width_max):
        return False
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
    tool: ToolGeometry = URDF_TOOL,
    faces: Faces | None = None,
) -> BoxState:
    """Build the box state for an engage snap from the current gripper poses.

    ``tilt`` is the grippers' starting inward yaw trim in radians (see
    :func:`side_clamp_rotation`; the thumbsticks change it live afterwards),
    ``tool`` the fitted gripper's contact geometry and ``faces`` any pinned
    clamping faces (:func:`choose_faces`). The starting width is the
    separation the contact faces would have with the mounts where they are,
    so a pair already holding a box keeps its grip through the align blend.
    """
    center, rot, _width = box_frame(left[0], right[0])
    yaw = tilt + tool.flush_tilt
    face = choose_faces({"left": left[1], "right": right[1]}, rot, yaw, faces)
    rel = {
        side: side_clamp_rotation(sign, face[side], yaw)
        for side, sign in _SIDE_SIGN.items()
    }
    feet = {side: tool.foot(face[side]) for side in _SIDE_SIGN}
    width = contact_width(left[0], right[0], rot, rel, feet)
    width = float(np.clip(width, width_min, width_max))
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
        tool=tool,
    )


def ideal_gripper_poses(
    center: np.ndarray,
    rot: np.ndarray,
    width: float,
    grip_rel: dict[str, np.ndarray],
    feet: dict[str, np.ndarray] | None = None,
) -> dict[str, Pose]:
    """The parallel-gripper pair for a box pose: ``{"left": pose, "right": pose}``.

    The contact feet (see :class:`ToolGeometry`) sit at ``center ± lateral *
    width / 2``; each mount is its foot vector (``feet``, mount frame)
    behind that, so a nonzero foot puts the *contact face*, not the mount,
    ``width / 2`` from the centre. ``feet`` omitted means trivial feet (the
    mounts themselves are the slots).
    """
    half = 0.5 * width * rot[:, 1]
    out: dict[str, Pose] = {}
    for side, sign in _SIDE_SIGN.items():
        r = (rot @ grip_rel[side]).astype(np.float32)
        slot = center + sign * half
        if feet is not None:
            slot = slot - r @ feet[side]
        out[side] = (slot.astype(np.float32), r)
    return out


def box_targets(
    state: BoxState, center: np.ndarray, rot: np.ndarray, now: float
) -> dict[str, Pose]:
    """Per-gripper EE targets for the current box pose ``(center, rot)``.

    While the align blend runs, each gripper is eased from where it was at the
    snap (carried along with the box) into its parallel slot; afterwards the
    parallel pair is returned directly.
    """
    ideal = ideal_gripper_poses(
        center, rot, state.width, state.grip_rel(), state.feet()
    )
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
