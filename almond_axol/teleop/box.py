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
(:class:`ToolGeometry`, one per grasp): where the tool's contact face sits
relative to the mount, and how far the grasp turns the gripper inward.
``width`` is the separation of the two *contact faces* — the box size —
and the mounts are placed behind them; the grasp's turn and the ``tilt``
trim on top of it pivot about the contact face's foot, and so do the align
blend and a switch between grasps, so turning keeps the face where it is.
The stock URDF gripper has a trivial geometry (the mount separation is the
width, the flat side faces the box at tilt 0); the parcel gripper's is
measured off its CAD (:func:`parcel_tool`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Pose = tuple[np.ndarray, np.ndarray]

_UP = np.array((0.0, 0.0, 1.0), dtype=np.float32)
_LEFT = np.array((0.0, 1.0, 0.0), dtype=np.float32)
# Below this horizontal separation the lateral axis is undefined; fall back
# to world +y (the grippers are stacked vertically or coincident).
_MIN_LATERAL_M = 1e-3


@dataclass(frozen=True)
class ToolGeometry:
    """Where a gripper touches the box side in one grasp, in its mount frame.

    Points are given as ``(forward, in, up)`` on the ``face = +1`` side:
    ``forward`` along the fingers (the mount's ``-Z``), ``in`` toward the
    box along the flat face turned toward it (the mount's ``±X``, see
    :func:`side_clamp_rotation`) and ``up`` along the mount's ``Y`` (the
    parcel gripper's hinge, vertical in box mode). The ``face = -1`` side
    is the same tool rolled 180° about its fingers, so ``in`` and ``up``
    both flip there (:meth:`point`).

    Attributes:
        flush_tilt: Inward yaw (rad) of this grasp, about the foot: the
            angled (``"flush"``) grasp's turn, ``0`` for the straight one.
            Box mode holds each gripper at this yaw plus the operator's
            tilt trim.
        foot_fwd: Distance (m) along the fingers from the mount origin to
            the *foot* — the point of the contact face box mode places on
            the box side and turns the gripper about.
        foot_in: The foot's distance (m) toward the box.
        patch: The points (``(forward, in, up)``, m) the tool presses the
            box with in this grasp — the corners of its contact face and,
            on a tool with one, the far tip — that the squeeze lean puts the
            clamp force through (:meth:`contacts`). Empty: the foot alone.
        body_in: How far (m) the gripper's body — housing, wrist link —
            reaches past the contact face toward the other gripper: half
            the gap left between the faces when the two bodies meet
            (:meth:`min_width`). The stock gripper's face is modelled on
            the mount axis and its 67 mm wrist link sticks out 33.5 mm.

    The foot is what box mode places at ``±width / 2``: the two contact
    faces are then ``width`` apart whatever tool is fitted, and the grasp's
    turn, the tilt trim and the align blend rotate the gripper about its
    foot so the face stays put.
    """

    flush_tilt: float = 0.0
    foot_fwd: float = 0.0
    foot_in: float = 0.0
    patch: tuple[tuple[float, float, float], ...] = ()
    body_in: float = 0.0335

    def min_width(self, clearance: float = 0.0) -> float:
        """Smallest grip width (m) that keeps the two grippers' bodies apart.

        The width is between the contact faces and each body sits
        :attr:`body_in` past its face, so this is the width at which the
        bodies are ``clearance`` apart, never less than ``0``; the
        operator's ``box_width_min`` is applied on top.
        """
        return max(0.0, 2.0 * self.body_in + clearance)

    @staticmethod
    def point(fwd: float, in_: float, up: float, face: float) -> np.ndarray:
        """Mount-frame vector for a ``(forward, in, up)`` point on side ``face``."""
        return np.array((face * in_, face * up, -fwd), dtype=np.float32)

    def foot(self, face: float) -> np.ndarray:
        """Mount-frame vector from the mount origin to the contact foot.

        ``face`` (``±1``) is which flat side (``±X``) faces the box.
        """
        return self.point(self.foot_fwd, self.foot_in, 0.0, face)

    def contacts(self, face: float = 1.0) -> list[np.ndarray]:
        """Where this tool touches the box side, mount frame (:attr:`patch`).

        A tall face is given by its top and bottom corners rather than its
        centre line: with one point per face the lean fixes the force's
        line along the fingers but leaves the *roll* about them to the
        arm's stiffness coupling, so the face presses along one edge and
        the other lifts — the thumb and index finger of a hand pinching
        while the pinky comes off the box. Put through the centroid of the
        corners (:func:`squeeze_lean`), the force has no roll and every
        contact carries the same share.
        """
        if not self.patch:
            return [self.foot(face)]
        return [self.point(f, i, u, face) for f, i, u in self.patch]


# The stock URDF gripper as box mode always modelled it: the mount
# separation is the width and a flat side faces the box at tilt 0.
URDF_TOOL = ToolGeometry()

# Parcel gripper, measured off its CAD (Parcel_Gripper.step), in metres as
# ``(forward, in, up)`` from the URDF gripper mount (on the flange axis, 3 mm
# behind the flange face). The hinged plate stays at its open stop in every
# mode, folded back along the wrist on the box side, 41.5 mm out: its outer
# face is what the straight grasp presses flat on the box, from beside the
# wrist to just past the hinge, full height behind the hinge and cut down
# to 12.5 mm above the axis ahead of it.
PARCEL_PLATE_IN_M = 0.0415
PARCEL_PLATE_REAR_M = -0.0524
PARCEL_PLATE_FRONT_M = 0.0534
PARCEL_PLATE_HALF_HEIGHT_M = 0.0305
PARCEL_PLATE_FRONT_TOP_M = 0.0125
# The plate's front corner is chamfered at 39.2° to its face: turned in by
# that much, the chamfer facet lies flat on the box side together with the
# fixed blade's tip (within 0.4 mm). The facet's centre, and its height (the
# plate's front section, bottom flush with the face's).
PARCEL_FACET_DEG = 39.2
PARCEL_FACET_FWD_M = 0.0609
PARCEL_FACET_IN_M = 0.0360
# The edge where the face meets the facet: on the box side in both grasps,
# so it is the foot either grasp is placed and turned about — the angled
# grasp rolls the gripper from its face onto the facet about this edge.
PARCEL_EDGE_FWD_M = 0.0542
# The fixed blade's tip, across the mount axis from the box side: the edge
# the angled grasp touches with, ±14.5 mm about the axis.
PARCEL_TIP_FWD_M = 0.1350
PARCEL_TIP_IN_M = -0.0240
PARCEL_TIP_HALF_HEIGHT_M = 0.0145


def parcel_tool(flush_deg: float, grasp: str = "flush") -> ToolGeometry:
    """Contact geometry of the parcel gripper, its plate at the open stop.

    ``grasp`` ``"straight"``: the fingers straight along the box, the
    folded plate's outer face flat on it (its four corners). ``"flush"``:
    each gripper turned ``flush_deg`` inward (``VRTeleopConfig.box_flush_deg``,
    39° — the chamfer's 39.2° puts the facet flat on the box), pressing with
    the chamfer facet and the fixed blade's tip, half the clamp each (the
    facet's top and bottom and the tip edge's). Both grasps share the foot,
    the face-to-facet edge, so the turn between them keeps it on the box
    side, and nothing else on the gripper reaches past either contact plane
    — the faces may close all but together.
    """
    up = PARCEL_PLATE_HALF_HEIGHT_M
    if grasp == "straight":
        patch = (
            (PARCEL_PLATE_REAR_M, PARCEL_PLATE_IN_M, up),
            (PARCEL_PLATE_REAR_M, PARCEL_PLATE_IN_M, -up),
            (PARCEL_PLATE_FRONT_M, PARCEL_PLATE_IN_M, PARCEL_PLATE_FRONT_TOP_M),
            (PARCEL_PLATE_FRONT_M, PARCEL_PLATE_IN_M, -up),
        )
        tilt = 0.0
    else:
        tip = PARCEL_TIP_HALF_HEIGHT_M
        patch = (
            (PARCEL_FACET_FWD_M, PARCEL_FACET_IN_M, PARCEL_PLATE_FRONT_TOP_M),
            (PARCEL_FACET_FWD_M, PARCEL_FACET_IN_M, -up),
            (PARCEL_TIP_FWD_M, PARCEL_TIP_IN_M, tip),
            (PARCEL_TIP_FWD_M, PARCEL_TIP_IN_M, -tip),
        )
        tilt = math.radians(flush_deg)
    return ToolGeometry(
        flush_tilt=tilt,
        foot_fwd=PARCEL_EDGE_FWD_M,
        foot_in=PARCEL_PLATE_IN_M,
        patch=patch,
        body_in=0.0,
    )


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
    # Wall time of the previous stick integration step (None before the first).
    stick_t: float | None = None
    tool: ToolGeometry = URDF_TOOL
    # Stick-click state for the grasp toggle (``IKWorker._stick_click_toggle``):
    # last frame's (left, right) click flags and whether a single press is
    # armed to toggle on its release.
    click_prev: tuple[bool, bool] = (False, False)
    click_armed: bool = False

    def grip_rel(self) -> dict[str, np.ndarray]:
        """Each gripper's rotation relative to the box frame (see :func:`side_clamp_rotation`)."""
        yaw = self.tilt + self.tool.flush_tilt
        return {
            side: side_clamp_rotation(sign, self.face[side], yaw)
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


def contact_feet(
    left: Pose, right: Pose, feet: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """World positions of the two contact feet for grippers at these poses.

    Each foot is ``rot @ feet[side]`` from its mount, with the gripper's own
    rotation: where its contact face's foot is now, which a snap keeps — a
    pair holding a box keeps it there through the align blend, and a grasp
    switch turns each gripper about it.
    """
    return {
        side: np.asarray(pose[0], dtype=np.float64)
        + np.asarray(pose[1], dtype=np.float64)
        @ np.asarray(feet[side], dtype=np.float64)
        for side, pose in (("left", left), ("right", right))
    }


def contact_width(
    left: Pose, right: Pose, rot: np.ndarray, feet: dict[str, np.ndarray]
) -> float:
    """Lateral separation of the two contact faces for grippers at these poses.

    The distance between the two :func:`contact_feet` along the box frame's
    lateral axis. With trivial feet (:data:`URDF_TOOL`) it is the mounts'
    own lateral separation.
    """
    lat = np.asarray(rot, dtype=np.float64)[:, 1]
    foot = contact_feet(left, right, feet)
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
    width = contact_width(left, right, rot, feet)
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
    separation the contact feet have with the grippers where they are, and
    the box centre their midpoint, where :func:`ideal_gripper_poses` puts
    them — so each gripper turns into the grasp about its foot (the align
    blend keeps it on a straight line, :func:`box_targets`): a pair already
    holding a box keeps its grip, and a switch between the parcel
    gripper's grasps rolls it from its face onto its facet about the edge
    between them instead of swinging the facet into the box.
    """
    _mounts, rot, _width = box_frame(left[0], right[0])
    yaw = tilt + tool.flush_tilt
    face = choose_faces({"left": left[1], "right": right[1]}, rot, yaw, faces)
    feet = {side: tool.foot(face[side]) for side in _SIDE_SIGN}
    foot = contact_feet(left, right, feet)
    center = (0.5 * (foot["left"] + foot["right"])).astype(np.float32)
    width = float((foot["left"] - foot["right"]) @ np.asarray(rot, np.float64)[:, 1])
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
    snap (carried along with the box) into its parallel slot — about its
    contact foot, which moves in a straight line while the rotation turns,
    so a foot already on the box stays on it — and afterwards the parallel
    pair is returned directly.
    """
    feet = state.feet()
    ideal = ideal_gripper_poses(center, rot, state.width, state.grip_rel(), feet)
    alpha = state.align_alpha(now)
    if alpha >= 1.0:
        return ideal
    out: dict[str, Pose] = {}
    for side, (p_rel, r_rel) in state.align_start.items():
        r_start = (rot @ r_rel).astype(np.float32)
        foot = feet[side]
        start = ((center + rot @ p_rel + r_start @ foot).astype(np.float32), r_start)
        goal_pos, goal_rot = ideal[side]
        goal = ((goal_pos + goal_rot @ foot).astype(np.float32), goal_rot)
        f_pos, r = blend_pose(start, goal, alpha)
        out[side] = ((f_pos - r @ foot).astype(np.float32), r)
    return out


@dataclass(frozen=True)
class SqueezeLean:
    """One gripper's target offset that puts its clamp force through the contacts.

    Attributes:
        translation: Extra mount translation (m, world frame) on top of the
            run-ahead along the normal — the lean's own, perpendicular to
            the squeeze (the pullback is separate).
        rotation: Rotation vector (rad, world frame) to apply to the mount.
        force: Clamp force (N) the arm presses with at this depth.
        depth: Run-ahead (m) into the box along the inward normal the arm
            is commanded to — the operator's, or less once capped.
        pullback: How far (m, ≤ 0) the target is moved back out along the
            normal to hold the force at the cap; ``0`` under the cap.
        stiffness: Clamp force per metre of depth (N/m) along the lean.
    """

    translation: np.ndarray
    rotation: np.ndarray
    force: float
    depth: float
    pullback: float
    stiffness: float


_NO_LEAN = SqueezeLean(np.zeros(3), np.zeros(3), 0.0, 0.0, 0.0, 0.0)


def squeeze_lean(
    jac: np.ndarray,
    kp: np.ndarray,
    rotation: np.ndarray,
    normal: np.ndarray,
    contacts: np.ndarray,
    depth: float,
    force_cap: float = 0.0,
) -> SqueezeLean:
    """How to lean a gripper so its squeeze presses evenly on the tool's contacts.

    An impedance-controlled arm pressing on a box exerts, at the gripper
    mount, the wrench its joint springs produce from the run-ahead of the
    command over the pose the box holds it at: ``w = K δ`` with ``K`` the
    arm's Cartesian stiffness at the mount (``(J diag(1/kp) J^T)^-1``).
    Jogging the width in past the box is a run-ahead ``δ = depth * n`` —
    a pure lateral translation of the whole gripper — and the wrench it
    produces is a lateral force *plus the moment it takes to hold the
    mount's orientation fixed* against the arm's stiffness coupling: with
    the left arm's real Jacobian at a box-carrying pose, some 1.5 Nm per
    centimetre. But the tool does not touch the box at the mount: the
    parcel gripper's angled grasp presses with its plate's chamfer facet and
    its fixed blade's tip 7 cm further along the box side. That moment can only be
    carried by the contacts loading unevenly — the face digging in while
    the tip lifts (or the other way about), the pinch the operator sees —
    and it makes the plain jog feel three times stiffer than the clamp
    itself.

    The clamp the operator wants is the wrench ``w* = F [n; r_c × n]``: a
    force ``F`` along the inward normal through the *centroid* ``r_c`` of
    the contact points, which the contacts share evenly (the parcel
    gripper's facet and tip, half each) with no moment left to unbalance
    them. The run-ahead that produces it is ``δ* = C w*`` (``C`` the
    compliance ``J diag(1/kp) J^T``), and its component along ``n`` is the
    depth the operator jogged, which fixes ``F``. What this returns is the
    rest of ``δ*`` — the lean: the small yaw that brings the tip in as the
    face presses (about 1.3° per centimetre of depth at that pose), a
    touch of roll so the tall face stays flat, and the millimetre of
    translation that goes with them. Added to the gripper *target*, so the
    arm's springs render the clamp with no change to the command path:
    nothing is measured but the depth, and nothing is held back.

    Args:
        jac: ``(6, 7)`` geometric Jacobian at the mount origin, world
            frame, linear rows first, at the commanded pose.
        kp: Joint stiffness (Nm/rad, ``(7,)``).
        rotation: ``(3, 3)`` mount rotation at the commanded pose.
        normal: Unit vector, world frame, from this gripper into the box.
        contacts: ``(k, 3)`` contact points, mount frame, on the box side
            (:meth:`ToolGeometry.contacts` for the gripper's ``face``).
        depth: Run-ahead (m) into the box the operator has jogged — how
            far the raw target sits past where the box holds the gripper.
        force_cap: Clamp force limit (N); ``0`` or less for none. Past it
            the depth is pulled back to the cap's (``pullback``), so the
            gripper presses with the cap however far the width is jogged.

    Returns:
        :class:`SqueezeLean`. Nothing (all zeros) for a depth ≤ 0 — the
        gripper is not pressing — or a degenerate model.
    """
    if depth <= 0.0:
        return _NO_LEAN
    jac = np.asarray(jac, dtype=np.float64)
    kp = np.asarray(kp, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    pts = np.asarray(contacts, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0 or not np.all(kp > 0.0) or not np.all(np.isfinite(jac)):
        return _NO_LEAN
    compliance = (jac / kp) @ jac.T  # J diag(1/kp) J^T
    r_c = pts.mean(axis=0) @ np.asarray(rotation, dtype=np.float64).T
    unit_wrench = np.concatenate([n, np.cross(r_c, n)])
    per_newton = compliance @ unit_wrench  # run-ahead per newton of clamp
    along = float(n @ per_newton[:3])
    if not (along > 1e-9) or not np.all(np.isfinite(per_newton)):
        return _NO_LEAN
    stiffness = 1.0 / along
    force = stiffness * depth
    if force_cap > 0.0 and force > force_cap:
        force = force_cap
    capped = force / stiffness
    delta = force * per_newton
    return SqueezeLean(
        translation=delta[:3] - capped * n,
        rotation=delta[3:],
        force=force,
        depth=capped,
        pullback=capped - depth,
        stiffness=stiffness,
    )


def joint_force_limit(
    jac: np.ndarray,
    rotation: np.ndarray,
    normal: np.ndarray,
    contacts: np.ndarray,
    joint_caps: np.ndarray,
) -> float:
    """Largest even clamp (N) the arm's torque-capped joints can hold.

    The clamp :func:`squeeze_lean` renders is a force along ``normal``
    through the contacts' centroid; statics puts ``|J^T w|`` of it on each
    joint. A joint with a spring-torque cap (``joint_caps``, Nm, ``inf``
    for none — the wrists' 5 Nm) can't deliver more than that, and the
    joint that saturates first is the one holding the moment that keeps the
    far contact pressed (``wrist_2``, ~0.13 Nm per N for the parcel
    gripper's facet and tip): past it every extra newton goes where that
    joint adds no torque — its own axis, behind the facet — so the tip
    unloads and the facet takes it all, the pinch. Returns the force at
    which the first capped joint saturates (``inf`` if none is loaded).
    """
    pts = np.asarray(contacts, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return math.inf
    n = np.asarray(normal, dtype=np.float64)
    r_c = pts.mean(axis=0) @ np.asarray(rotation, dtype=np.float64).T
    unit_wrench = np.concatenate([n, np.cross(r_c, n)])
    per_newton = np.abs(np.asarray(jac, dtype=np.float64).T @ unit_wrench)
    caps = np.asarray(joint_caps, dtype=np.float64)
    loaded = np.isfinite(caps) & (per_newton > 1e-9)
    if not np.any(loaded):
        return math.inf
    return float(np.min(caps[loaded] / per_newton[loaded]))


def tip_inward_sign(rot: np.ndarray, normal: np.ndarray, up: np.ndarray) -> float:
    """``+1`` if a positive yaw about ``up`` swings this gripper's tip toward the box.

    The tip is where the fingers point (:func:`approach_axis`); a yaw
    about ``up`` moves that direction along ``up × fingers``. ``normal`` is
    the inward normal from the gripper into the box.
    """
    fingers = approach_axis(rot).astype(np.float64)
    swing = np.cross(np.asarray(up, dtype=np.float64), fingers)
    return 1.0 if float(swing @ np.asarray(normal, dtype=np.float64)) >= 0.0 else -1.0


def toe_out_sides(
    ideal: dict[str, Pose],
    measured: dict[str, Pose],
    normals: dict[str, np.ndarray],
    up: np.ndarray,
) -> dict[str, float]:
    """Per gripper, how far its tip has swung off the box (rad), from FK of the measured joints.

    Each gripper's measured mount rotation is compared with its ideal
    (parallel-slot) rotation and the yaw about ``up`` between them taken,
    signed so that a tip *away* from the box is positive. Under a clamp
    with the face pressing and the tip lifted — the pinch — that is the
    toe-out angle. Rigid gripper, flat box side: zero means both the face
    and the tip are on the box. A turn of the whole pair lags both arms
    with the *same* sense about ``up`` — opposite senses here — so the
    two sides' mean (:func:`toe_out`) is free of a carry's servo lag while
    the per-side values are not.
    """
    out: dict[str, float] = {}
    for side in _SIDE_SIGN:
        r_ideal = np.asarray(ideal[side][1], dtype=np.float64)
        r_meas = np.asarray(measured[side][1], dtype=np.float64)
        yaw = twist_about(r_meas @ r_ideal.T, up)
        out[side] = -tip_inward_sign(r_ideal, normals[side], up) * yaw
    return out


def toe_out(
    ideal: dict[str, Pose],
    measured: dict[str, Pose],
    normals: dict[str, np.ndarray],
    up: np.ndarray,
) -> float:
    """The two grippers' mean toe-out (rad, tip off the box positive); see :func:`toe_out_sides`."""
    sides = toe_out_sides(ideal, measured, normals, up)
    return 0.5 * sum(sides.values())


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
