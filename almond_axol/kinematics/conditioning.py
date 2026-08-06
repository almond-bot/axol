"""Backend-independent target conditioning for teleop IK.

Transforms applied to the raw Cartesian targets before they reach any IK
backend (see :class:`almond_axol.teleop.worker.IKWorker`):

- :func:`swivel_direction`, :func:`elbow_circle`, :func:`swivel_frame` and
  :func:`clear_swivel_angle` map the operator's elbow position onto the
  robot's own swivel circle — the locus of elbow positions consistent with
  the robot's upper-arm/forearm lengths for a given shoulder->wrist line. The
  human arm is shorter than the robot's, so tracking the raw (scaled) elbow
  position biases the solve toward unreachable references (in practice:
  elbow-down); preserving only the *swivel direction* keeps the intent
  ("elbow up/out") while making the reference exactly reachable.

  The steps are separate so the caller can smooth the swivel *as an angle*
  and keep it clear of the robot's own base column. Smoothing matters more
  than it looks: the swivel is the arm's only self-motion, and where the
  shoulder is near-singular (``|shoulder_2| ~ 90 deg``, which every
  reach above shoulder height passes through) that self-motion is a
  shoulder_1/shoulder_3 counter-rotation of unbounded gain. Reference noise
  is amplified straight into wild shoulder rotation there, so the reference
  has to be rate-limited, and it has to be authoritative enough to pin the
  swivel — otherwise the solver resolves the redundancy arbitrarily and the
  arm comes back down through a different, drastic configuration.

- :func:`clamp_target_error` caps the pose error between the current
  end-effector pose and the commanded target (Drake-style feasibility
  scaling). Out-of-reach or fast-moving targets degrade into a smooth,
  bounded pull in the commanded direction instead of a large error whose
  resolution depends on solver internals.

- :func:`clamp_reach` keeps targets inside the annulus the arm can actually
  reach: outside ``min_reach`` (the folded-arm inner zone around the
  shoulder, where every solve grinds against joint limits) and inside
  ``max_reach``.

- :func:`clamp_column_keepout` keeps wrist targets out of the base-column
  footprint, so a hands-toward-chest motion slides along the column face
  instead of commanding the forearm through the torso (which the collision
  barrier must then fight for as long as the pose is held).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation

_EPS = 1e-9


class ColumnKeepout(NamedTuple):
    """Elliptical-cylinder keep-out around the robot's central base column.

    An ellipse rather than the column's box hull: the clamp has to be
    continuous in its input, and projecting onto the nearest face of a
    rectangle is not (it flips across the diagonals). With the operator's
    hand hovering inside the footprint that discontinuity chattered, measured
    at 3x the whole-session jerk of the ellipse version.

    Attributes:
        half_x / half_y: Ellipse semi-axes (m), inflated past the column hull
            so links stop short of touching it.
        top_z: Height (m) above which targets are unconstrained.
    """

    half_x: float
    half_y: float
    top_z: float

    def radius(self, points: np.ndarray) -> np.ndarray:
        """Normalised radial distance of ``(N, 3)`` points from the column axis.

        Ignores height, so it is a smooth "how far outboard" score usable as
        an objective anywhere in the workspace.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return np.hypot(pts[:, 0] / self.half_x, pts[:, 1] / self.half_y)

    def clearance(self, points: np.ndarray) -> np.ndarray:
        """Normalised radial clearance of ``(N, 3)`` points; ``>= 1`` is clear.

        Points above :attr:`top_z` clear the column regardless of radius.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return np.where(pts[:, 2] >= self.top_z, np.inf, self.radius(pts))


def swivel_direction(
    shoulder: np.ndarray,
    wrist_target: np.ndarray,
    elbow_raw: np.ndarray,
) -> np.ndarray | None:
    """Unit component of an elbow hint perpendicular to the shoulder->wrist axis.

    Args:
        shoulder: ``(3,)`` world-frame shoulder position (m).
        wrist_target: ``(3,)`` world-frame wrist/EE target position (m).
        elbow_raw: ``(3,)`` raw (operator-derived) elbow hint position (m).

    Returns:
        ``(3,)`` unit direction, or ``None`` when degenerate (wrist at the
        shoulder, or elbow hint collinear with the arm axis).
    """
    shoulder = np.asarray(shoulder, dtype=np.float64)
    wrist = np.asarray(wrist_target, dtype=np.float64)
    elbow = np.asarray(elbow_raw, dtype=np.float64)

    axis_vec = wrist - shoulder
    d = float(np.linalg.norm(axis_vec))
    if d < _EPS:
        return None
    axis = axis_vec / d

    r = elbow - shoulder
    r_perp = r - float(np.dot(r, axis)) * axis
    norm = float(np.linalg.norm(r_perp))
    if norm < 1e-6:
        return None
    return (r_perp / norm).astype(np.float64)


class ElbowCircle(NamedTuple):
    """The locus of elbow positions consistent with one wrist target.

    With the shoulder fixed and the wrist at ``center + ...``, the robot's
    segment lengths put the elbow on a circle around the shoulder->wrist
    axis. Attributes:
        center: ``(3,)`` circle centre, on the shoulder->wrist axis (m).
        radius: Circle radius (m).
        axis: ``(3,)`` unit shoulder->wrist direction (the circle normal).
    """

    center: np.ndarray
    radius: float
    axis: np.ndarray

    def point(self, swivel_dir: np.ndarray) -> np.ndarray:
        """Circle point in the (unit, in-plane) direction ``swivel_dir``."""
        return (self.center + self.radius * swivel_dir).astype(np.float32)


def elbow_circle(
    shoulder: np.ndarray,
    wrist_target: np.ndarray,
    upper_arm_len: float,
    forearm_len: float,
) -> ElbowCircle | None:
    """Reachable elbow circle for a wrist target (law of cosines).

    Args:
        shoulder: ``(3,)`` world-frame shoulder position (m).
        wrist_target: ``(3,)`` world-frame wrist/EE target position (m).
        upper_arm_len: Robot shoulder->elbow segment length (m).
        forearm_len: Robot elbow->wrist segment length (m).

    Returns:
        The circle, or ``None`` when degenerate (wrist at the shoulder).
    """
    shoulder = np.asarray(shoulder, dtype=np.float64)
    wrist = np.asarray(wrist_target, dtype=np.float64)

    axis_vec = wrist - shoulder
    d = float(np.linalg.norm(axis_vec))
    if d < _EPS:
        return None
    axis = axis_vec / d
    # Clamp the shoulder->wrist distance to the annulus the two segments span
    # so the circle construction below is always well-defined.
    d = float(
        np.clip(
            d,
            abs(upper_arm_len - forearm_len) + 1e-3,
            upper_arm_len + forearm_len - 1e-3,
        )
    )

    a = (upper_arm_len**2 - forearm_len**2 + d * d) / (2.0 * d)
    h_sq = upper_arm_len**2 - a * a
    if h_sq <= 0.0:
        return None
    return ElbowCircle(shoulder + a * axis, float(np.sqrt(h_sq)), axis)


def swivel_frame(
    axis: np.ndarray, reference_dir: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Orthonormal basis of the swivel plane, with the first axis along a reference.

    Swivel angles are always measured *relative to the previous reference*
    rather than in a fixed world frame: no world frame is free of a
    degeneracy on the sphere of arm axes, and one placed anywhere in the
    reachable workspace would make the angle jump as the arm swept past it.
    Re-basing on the previous direction (parallel transport) keeps the angle
    well-conditioned everywhere and makes "no change" exactly angle zero.

    Args:
        axis: ``(3,)`` unit circle normal (shoulder->wrist direction).
        reference_dir: ``(3,)`` direction defining the zero angle; only its
            component perpendicular to ``axis`` is used.

    Returns:
        ``(e_a, e_b)`` unit vectors spanning the plane, or ``None`` when
        ``reference_dir`` is collinear with ``axis``.
    """
    axis = np.asarray(axis, dtype=np.float64)
    e_a = np.asarray(reference_dir, dtype=np.float64)
    e_a = e_a - float(np.dot(e_a, axis)) * axis
    norm = float(np.linalg.norm(e_a))
    if norm < 1e-6:
        return None
    e_a = e_a / norm
    return e_a, np.cross(axis, e_a)


def _arm_samples(
    circle: ElbowCircle,
    e_a: np.ndarray,
    e_b: np.ndarray,
    wrist_target: np.ndarray,
    angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Elbow positions and forearm interior points for candidate swivel angles.

    Returns ``(elbows, forearm)`` shaped ``(N, 3)`` and ``(N, 3, 3)``. The
    upper arm is not sampled: it is anchored at the shoulder, which sits at
    the column's edge by construction, so no swivel angle can clear it and
    including it would make every candidate score identically badly.
    """
    dirs = np.cos(angles)[:, None] * e_a + np.sin(angles)[:, None] * e_b
    elbows = circle.center + circle.radius * dirs
    fractions = np.array([0.25, 0.5, 0.75])
    reach = np.asarray(wrist_target, dtype=np.float64) - elbows
    return elbows, elbows[:, None, :] + fractions[None, :, None] * reach[:, None, :]


def elevated_swivel_angle(e_a: np.ndarray, e_b: np.ndarray) -> float:
    """Swivel angle lifting the elbow as high as the circle allows.

    A posture *prior*, and the term that actually makes the elbow rise when
    the operator reaches out — the reason the elbow otherwise stays down is
    that the headset's elbow hint is pinned at the bottom of the swivel
    range. Measured over a captured session, the hint asked for an elbow at
    the 4th percentile of the elevation the arm could have reached; it is
    inferred from the head and controller poses rather than measured, and
    inference biases elbow-down.

    Elevation is the right axis for a prior because it is exactly the
    operator's stated intent ("elbow up to reach the shelf") and because it
    is self-limiting: the circle's elevation span is only ~2 cm with the arm
    hanging (nothing to gain, and the prior degenerates to holding the
    current swivel) but ~33 cm with the hand above shoulder height, which is
    precisely where shelf and bin work needs it.

    Args:
        e_a / e_b: Orthonormal basis of the circle plane (see
            :func:`swivel_frame`); the returned angle is measured from ``e_a``.

    Returns:
        The angle, in radians, in ``(-pi, pi]`` relative to ``e_a``. Zero
        (hold the current swivel) when the circle is horizontal and elevation
        does not depend on the angle at all.
    """
    return float(np.arctan2(e_b[2], e_a[2]))


def clear_swivel_angle(
    circle: ElbowCircle,
    e_a: np.ndarray,
    e_b: np.ndarray,
    wrist_target: np.ndarray,
    desired: float,
    keepout: ColumnKeepout,
    samples: int = 180,
) -> float:
    """Swivel angle nearest ``desired`` that keeps the arm clear of the column.

    The elbow reference is free to sit anywhere on the elbow circle, and much
    of that circle passes through the robot's own base column — the shoulders
    are mounted at the column's edge, so an inward swivel points straight into
    it. Commanding such a reference asks the solver for a configuration that
    does not exist, which it can only answer by grinding against whatever
    protection is active (measured on a captured session: 13% of ticks
    referenced an elbow inside the column hull, and the upper arm penetrated
    it for 24% of ticks). Rotating the reference to the nearest clear angle
    keeps the operator's intent — swivel is a one-parameter family, so the
    nearest clear angle is on the same side they asked for — while staying
    reachable.

    Args:
        circle: The reachable elbow circle.
        e_a / e_b: Orthonormal basis of the circle plane (see
            :func:`swivel_frame`); angles are measured from ``e_a``.
        wrist_target: ``(3,)`` wrist target, the far end of the forearm (m).
        desired: Requested swivel angle (radians, measured from ``e_a``).
        keepout: Column keep-out region.
        samples: Angular resolution of the search (180 -> 2 degrees).

    Returns:
        The clear angle nearest ``desired``, or ``desired`` itself when no
        sampled angle is clear (fully boxed in — let the solver do its best).
    """
    offsets = np.linspace(-np.pi, np.pi, samples, endpoint=False)
    angles = desired + offsets
    elbows, forearm = _arm_samples(circle, e_a, e_b, wrist_target, angles)

    clear = keepout.clearance(elbows) >= 1.0
    clear &= (
        keepout.clearance(forearm.reshape(-1, 3)).reshape(samples, -1) >= 1.0
    ).all(axis=1)
    if not clear.any():
        return float(desired)
    idx = np.flatnonzero(clear)
    return float(angles[idx[np.argmin(np.abs(offsets[idx]))]])


def clamp_target_error(
    cur_pos: np.ndarray,
    cur_rot: np.ndarray,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    max_lin: float,
    max_ang: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cap the pose error between the current pose and the target.

    Scales the position and orientation error *jointly* (one factor for both)
    so the commanded direction in SE(3) is preserved — the same feasibility
    scaling Drake's differential IK applies to infeasible velocity commands.

    Args:
        cur_pos / cur_rot: Current EE pose (``(3,)``, ``(3,3)``).
        target_pos / target_rot: Desired EE pose.
        max_lin: Maximum allowed position error (m).
        max_ang: Maximum allowed orientation error (radians).

    Returns:
        ``(pos, rot_3x3)`` — the target, pulled toward the current pose when
        the error exceeded the caps, unchanged otherwise.
    """
    cur_pos = np.asarray(cur_pos, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)
    dp = target_pos - cur_pos
    lin = float(np.linalg.norm(dp))

    cur_rot = np.asarray(cur_rot, dtype=np.float64)
    target_rot = np.asarray(target_rot, dtype=np.float64)
    rotvec = Rotation.from_matrix(cur_rot.T @ target_rot).as_rotvec()
    ang = float(np.linalg.norm(rotvec))

    scale = min(
        1.0,
        max_lin / lin if lin > _EPS else 1.0,
        max_ang / ang if ang > _EPS else 1.0,
    )
    if scale >= 1.0:
        return (
            target_pos.astype(np.float32),
            target_rot.astype(np.float32),
        )
    pos = cur_pos + scale * dp
    rot = cur_rot @ Rotation.from_rotvec(scale * rotvec).as_matrix()
    return pos.astype(np.float32), rot.astype(np.float32)


def clamp_reach(
    pos: np.ndarray,
    center: np.ndarray,
    max_reach: float,
    min_reach: float = 0.0,
) -> np.ndarray:
    """Clamp a target position to the annulus ``[min_reach, max_reach]`` of ``center``.

    The inner clamp keeps the wrist target out of the folded-arm zone right
    around the shoulder, where no feasible configuration exists and the solve
    degenerates into grinding against joint limits (jerky on hardware).
    Targets closer than ``min_reach`` are pushed radially outward; a target
    exactly at the centre is left unchanged (no meaningful direction).
    """
    pos = np.asarray(pos, dtype=np.float32)
    d = pos - np.asarray(center, dtype=np.float32)
    dist = float(np.linalg.norm(d))
    if dist > max_reach:
        return (center + d * (max_reach / dist)).astype(np.float32)
    if dist < min_reach and dist > _EPS:
        return (center + d * (min_reach / dist)).astype(np.float32)
    return pos


def clamp_column_keepout(
    pos: np.ndarray,
    rot: np.ndarray,
    keepout: ColumnKeepout,
    hand_offsets: tuple[float, ...] = (0.0,),
    iterations: int = 3,
) -> np.ndarray:
    """Push an end-effector target out until its whole hand clears the column.

    The engage-relative mapping happily places wrist targets *inside* the
    base column (the rest EE hangs level with the column face, so any
    hand-toward-chest motion crosses it). Such a target is unreachable by
    construction; without this clamp the solve can only answer by grinding
    against whatever protection is active for as long as the operator holds
    the pose — measured as jitter and "the arm is stuck" on hardware.

    It is not enough to clamp the gripper point: the hand and wrist assembly
    extends from 145 mm ahead of the gripper frame to 230 mm behind it, and
    on a captured session those links reached 67 mm inside the column while
    the commanded point itself sat legally on the boundary. ``hand_offsets``
    are therefore sampled along the gripper frame's own axis, and the target
    is pushed radially until the worst of them clears. Clamping this way
    turns the conflict into a smooth slide along the column face, exactly as
    ``clamp_reach`` does at the workspace boundary.

    Args:
        pos: ``(3,)`` commanded end-effector position (m).
        rot: ``(3, 3)`` commanded end-effector rotation; its third column is
            the hand axis along which ``hand_offsets`` are measured.
        keepout: Column keep-out region.
        hand_offsets: Signed distances (m) along the hand axis to test.
            Negative is ahead of the gripper frame (the fingers).
        iterations: Fixed-point refinements. Each push moves every sample by
            the same vector, so one pass can expose a different worst sample;
            a few passes converge (the region is convex).

    Returns:
        The pushed target, or ``pos`` unchanged when the hand already clears.
    """
    pos = np.asarray(pos, dtype=np.float32)
    axis = np.asarray(rot, dtype=np.float64)[:, 2]
    offsets = np.asarray(hand_offsets, dtype=np.float64)
    out = pos.astype(np.float64)
    for _ in range(iterations):
        points = out + offsets[:, None] * axis
        radii = keepout.radius(points)
        # Only points inside the column's height band can be pushed usefully.
        radii = np.where(points[:, 2] < keepout.top_z, radii, np.inf)
        worst = int(np.argmin(radii))
        r = float(radii[worst])
        if r >= 1.0 or r < _EPS:
            break
        out[:2] += points[worst, :2] * (1.0 / r - 1.0)
    return out.astype(np.float32)
