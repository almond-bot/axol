"""Box mode's squeeze shaping: put the arm's clamp force where the tool touches.

An impedance-controlled arm pressing a gripper against a box exerts, at the
gripper mount, the wrench its joint springs produce from the run-ahead of
the command over the measured pose: ``w = (J^T)^+ (kp * (q_cmd - q_meas))``.
Jogging the grip width in past the box makes that run-ahead a lateral
translation of the whole gripper, and the wrench it produces is a lateral
force at the *mount* plus whatever moment the arm's stiffness coupling
happens to add. But the tool does not touch the box at the mount. The parcel
gripper's folded blade presses with its flat face right beside the wrist and
the fixed blade's tip 13 cm further along the box side; a flat pad touches
along its length. How the force divides between those contact points is set
entirely by the moment about the mount: a pure force there loads the point
on its line of action — the face at the wrist — and the far tip only as the
coupling moment allows (a fraction of a newton either way; it comes out
opposite on the two arms, and in one direction it lifts the face instead).
That is the pinch the operator sees: first touch is tip and face together,
a harder squeeze is one of them alone. The moment that shares the load
evenly — half the force times the tip's 13 cm lever — is about 0.07 Nm per
newton, and against the arm's yaw stiffness at the mount that is a hundredth
of a degree per newton, far too small and too sign-sensitive to trim by hand.

:func:`shape_squeeze` therefore rewrites the run-ahead's contact part: the
joint torque the run-ahead asks for is projected onto the torques the
contact points can produce (a lateral force at each, ``J^T [n; r_k x n]``),
which estimates the squeeze force the arm is about to apply; that part is
replaced by the *same* total force split evenly over the contacts — so the
mount gets the moment that puts the force through their centroid — and
saturated at a force cap. The rest of the run-ahead (servo lag, the box's
weight carried in friction at the contacts, the arm's posture motion) is
kept as is. Everything is expressed as joint torque and turned back into a
command with ``q_meas + tau / kp``, so the impedance springs render the
wrench without any change to the realtime control law. The Jacobian comes
from the same MuJoCo model gravity compensation runs on
(:meth:`GravityCompensator.mount_jacobian`).

The force cap is the tighter of the operator's force limit
(``box_squeeze_force``) and the force at which any spring-capped joint
(``box_squeeze_torque`` on the shoulders) would reach its cap under the
evenly split load — so the joint caps bound the force consistently across
poses instead of letting the arm's shape decide, and the per-joint back-off
and the realtime core's per-joint clamp underneath rarely have to act.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["SqueezeSpec", "SqueezeResult", "orient_contacts", "shape_squeeze"]


@dataclass(frozen=True)
class SqueezeSpec:
    """What one arm needs to shape its squeeze this tick.

    Attributes:
        normal: Unit vector, base frame, pointing from this gripper *into*
            the box (toward the other gripper).
        contacts: Contact points on the tool, in the gripper mount frame,
            for the tool's ``face = +1`` side; :func:`orient_contacts`
            mirrors them onto whichever side faces the box.
        force_cap: Squeeze force limit (N) for this arm; ``inf`` for none.
    """

    normal: np.ndarray
    contacts: tuple[np.ndarray, ...]
    force_cap: float = float("inf")


@dataclass(frozen=True)
class SqueezeResult:
    """Output of :func:`shape_squeeze`.

    Attributes:
        tau: Shaped run-ahead torque (Nm) per arm joint; ``q_meas + tau /
            kp`` is the command to send.
        force: Total squeeze force (N) the arm will apply after shaping —
            the estimate saturated at the limit; 0 while not pressing.
        estimate: Squeeze force (N) the unshaped run-ahead was asking for
            (negative when the gripper is being pulled off the box).
        limit: Force limit (N) in effect: the tighter of the force cap and
            the spring caps' equivalent under the even split.
        contact_forces: The estimated per-contact split of ``estimate``.
    """

    tau: np.ndarray
    force: float
    estimate: float
    limit: float
    contact_forces: np.ndarray


def orient_contacts(
    contacts: Sequence[np.ndarray], rotation: np.ndarray, normal: np.ndarray
) -> np.ndarray:
    """Mirror the tool's contact points onto the side of the mount facing the box.

    The tool geometry is given for ``face = +1`` (contacts on the mount
    frame's ``+x`` side); if the mount's ``+x`` axis points away from the
    box (``rotation[:, 0] . normal < 0``) the points' ``x`` is negated —
    the same ``face`` mirroring the IK's grasp geometry applies.
    """
    pts = np.asarray(contacts, dtype=np.float64).reshape(-1, 3).copy()
    if float(rotation[:, 0] @ normal) < 0.0:
        pts[:, 0] = -pts[:, 0]
    return pts


def shape_squeeze(
    tau: np.ndarray,
    kp: np.ndarray,
    jac: np.ndarray,
    rotation: np.ndarray,
    normal: np.ndarray,
    contacts: np.ndarray,
    spring_caps: Mapping[int, float] | None = None,
    force_cap: float = float("inf"),
) -> SqueezeResult:
    """Reshape a run-ahead torque so its squeeze is shared over the contacts.

    Args:
        tau: Run-ahead spring torque per arm joint, ``kp * (q_cmd -
            q_meas)`` (Nm, ``(7,)``).
        kp: Joint stiffness (Nm/rad, ``(7,)``); the fit weights torque by
            ``1 / kp`` (elastic energy) and the result is meant to be
            divided by it again.
        jac: ``(6, 7)`` geometric Jacobian at the mount origin, base frame
            (linear rows first), at the *measured* pose.
        rotation: ``(3, 3)`` mount rotation at the measured pose.
        normal: Unit inward normal (base frame) at this gripper.
        contacts: ``(k, 3)`` contact points in the mount frame, already
            on the box side (:func:`orient_contacts`).
        spring_caps: ``{joint index: cap Nm}`` spring-torque caps in force;
            each bounds the force at which that joint's share of the even
            split reaches the cap.
        force_cap: Operator's squeeze force limit (N).

    Returns:
        :class:`SqueezeResult`. With no squeeze estimated (the run-ahead is
        not pressing the contacts into the box) ``tau`` is returned
        unchanged.
    """
    tau = np.asarray(tau, dtype=np.float64)
    kp = np.asarray(kp, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    pts = np.asarray(contacts, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0 or not np.all(kp > 0.0):
        return SqueezeResult(tau, 0.0, 0.0, float("inf"), np.zeros(0))

    # Joint torque per newton of inward force at each contact: a lateral
    # force ``n`` at ``r_k`` (offset from the mount, base frame) is the
    # wrench ``[n; r_k x n]`` at the mount origin.
    r = pts @ rotation.T
    wrenches = np.hstack((np.tile(n, (len(pts), 1)), np.cross(r, n)))
    basis = jac.T @ wrenches.T  # (7, k)

    # Fit the run-ahead's contact part in the compliance metric: the
    # displacement ``tau / kp`` decomposed into what the contacts push back
    # with plus the rest (weighted least squares, weight 1/kp).
    w_sqrt = 1.0 / np.sqrt(kp)
    forces = np.linalg.pinv(basis * w_sqrt[:, None]) @ (tau * w_sqrt)
    estimate = float(forces.sum())

    even = basis.mean(axis=1)  # torque per newton of evenly split squeeze
    limit = float(force_cap) if force_cap > 0.0 else float("inf")
    for i, cap in (spring_caps or {}).items():
        lever = abs(float(even[i]))
        if cap > 0.0 and np.isfinite(cap) and lever > 1e-9:
            limit = min(limit, cap / lever)

    if estimate <= 0.0:
        return SqueezeResult(tau, 0.0, estimate, limit, forces)
    force = min(estimate, limit)
    shaped = tau - basis @ forces + force * even
    return SqueezeResult(shaped, force, estimate, limit, forces)
