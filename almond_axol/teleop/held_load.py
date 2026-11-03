"""What a gripper is holding, inferred from its arm's torque residuals.

The haptics' push cue (:meth:`VRTeleopCore.haptic_levels`) reads each arm's
torque residual — measured torque minus the gravity model — as contact. An
object in the gripper adds a torque the gravity model doesn't know about, so
a heavy enough one would rumble with nothing touched. :class:`HeldLoad`
explains that torque away while the gripper holds something.

The held load is modelled at the gripper mount, in terms that hold for any
tool (the parallel or the parcel gripper, whatever its fingers carry):

- a **point mass** ``m`` with its centre of mass at ``c`` in the mount frame
  (parameters ``m`` and ``p = m * c``). Its weight is constant in the world
  frame and its moment about the mount turns with the wrist, so the model
  stays right as the operator moves and rotates the object.
- in box mode, a **squeeze**: a force along the clamp, constant in the mount
  frame (box mode keeps each gripper level and turns it only about the
  vertical). It has two components, along the mount's ``X`` (the jaw axis,
  the flat face toward the box) and ``Z`` (the fingers), because the angled
  grasp turns the face inward; :func:`plausible` keeps it near ``X``.

The residual is linear in those parameters (:func:`load_regressor`). Each
sample is fit on its own first: when the fit explains the residual (a load
like a held object or the squeeze, see :func:`plausible`), that fit is the
held load right away, so lifting an object or tightening the squeeze stays
quiet, and the sample also refines a slow estimate (an exponentially
forgetting least-squares fit). When it doesn't — a push sideways or up from
below, which no held weight produces — the slow estimate stands in, so the
cue still feels the contact on top of what the gripper holds.
"""

from __future__ import annotations

import math

import numpy as np

from ..robot.gravity import MAX_PAYLOAD_KG

# Gravity in the robot base frame (m/s^2), as the gravity model uses it.
_GRAVITY = np.array((0.0, 0.0, -9.81))
# ``_GRAVITY_CROSS @ v == np.cross(_GRAVITY, v)``, as a matrix so the
# regressor is a few small matrix products (np.cross is slow per call).
_GRAVITY_CROSS = np.array(
    (
        (0.0, -_GRAVITY[2], _GRAVITY[1]),
        (_GRAVITY[2], 0.0, -_GRAVITY[0]),
        (-_GRAVITY[1], _GRAVITY[0], 0.0),
    )
)
# Slack below zero mass a fit may read and still count as a held load: the
# gravity model's error with nothing held.
_MASS_SLACK_KG = 0.2
# Largest centre-of-mass offset (m) from the mount a held load may have, and
# the mass below which that bound stops shrinking (a light object's offset is
# poorly observed).
_MAX_COM_M = 0.3
_COM_MASS_FLOOR_KG = 0.5
# The squeeze may turn this far from the mount's X toward its fingers (rad):
# the angled grasp's turn plus the tilt trim.
_SQUEEZE_CONE = math.radians(50.0)
# Slack (N) on that cone, so a squeeze near zero isn't judged by its angle.
_SQUEEZE_SLACK_N = 2.0
# Largest squeeze (N) counted as a held load.
_MAX_SQUEEZE_N = 80.0
# Ridge on the slow estimate's normal equations, relative to their scale:
# keeps the unobservable part of ``p`` (along gravity) at zero.
_RIDGE = 1e-6


def load_regressor(jac: np.ndarray, rot: np.ndarray, *, squeeze: bool) -> np.ndarray:
    """Joint torque per unit of each held-load parameter, ``(7, k)``.

    ``jac`` is the mount's ``(6, 7)`` Jacobian (linear rows first) and ``rot``
    its rotation in the base frame, from
    :meth:`~almond_axol.robot.gravity.GravityCompensator.mount_jacobian`.
    Columns are ``m``, ``p`` (3) and, with ``squeeze``, the squeeze along the
    mount's ``X`` and ``Z``. The torque is what holding the load costs each
    joint, the same sign as the gravity model's.
    """
    lin, ang = jac[:3], jac[3:]
    # p's moment about the mount is (R p) x g = -g x (R p).
    cols = [-(lin.T @ _GRAVITY)[:, None], ang.T @ (_GRAVITY_CROSS @ rot)]
    if squeeze:
        cols.append(-(lin.T @ rot[:, ::2]))
    return np.hstack(cols)


def plausible(theta: np.ndarray) -> bool:
    """Whether parameters ``theta`` describe something a gripper can hold."""
    mass = float(theta[0])
    if not -_MASS_SLACK_KG <= mass <= MAX_PAYLOAD_KG:
        return False
    if np.linalg.norm(theta[1:4]) > _MAX_COM_M * max(mass, _COM_MASS_FLOOR_KG):
        return False
    if theta.size > 4:
        along, across = abs(float(theta[4])), abs(float(theta[5]))
        if math.hypot(along, across) > _MAX_SQUEEZE_N:
            return False
        if across > math.tan(_SQUEEZE_CONE) * along + _SQUEEZE_SLACK_N:
            return False
    return True


class HeldLoad:
    """The load one gripper holds, learned from its arm's torque residuals.

    Args:
        squeeze: Model box mode's squeeze as well as a held weight.
    """

    def __init__(self, *, squeeze: bool) -> None:
        self.squeeze = squeeze
        k = 6 if squeeze else 4
        self._ata = np.zeros((k, k))
        self._atr = np.zeros(k)
        self.theta: np.ndarray | None = None

    @property
    def mass(self) -> float:
        """Estimated held mass (kg), ``0`` before the first estimate."""
        return 0.0 if self.theta is None else max(float(self.theta[0]), 0.0)

    def explain(
        self, a: np.ndarray, residual: np.ndarray, *, tol: float, learn: float
    ) -> np.ndarray:
        """The part of ``residual`` (``(7,)``, Nm) the held load accounts for.

        ``a`` is :func:`load_regressor` at the current pose. A sample whose
        own fit leaves no joint more than ``tol`` (Nm) unexplained and is
        :func:`plausible` is the held load as it is; it also refines the slow
        estimate with weight ``learn`` (0 to 1: pass 0 while the arm moves
        fast and its dynamics would bias it). Any other sample gets the slow
        estimate.
        """
        theta, *_ = np.linalg.lstsq(a, residual, rcond=None)
        fit = a @ theta
        if np.max(np.abs(residual - fit)) <= tol and plausible(theta):
            if learn > 0.0:
                self._ata += learn * (a.T @ a - self._ata)
                self._atr += learn * (a.T @ residual - self._atr)
                ridge = _RIDGE * max(float(np.trace(self._ata)), 1.0)
                self.theta = np.linalg.solve(
                    self._ata + ridge * np.eye(self._ata.shape[0]), self._atr
                )
            return fit
        if self.theta is None:
            return np.zeros_like(residual)
        return a @ self.theta
