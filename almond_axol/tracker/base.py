"""TrackerSource protocol and coordinate-conversion helpers.

Every backend reports poses in the **WebXR world convention** the teleop
stack assumes (see ``_VR_UP`` in :mod:`almond_axol.teleop.worker`): a
right-handed, gravity-aligned frame with **+y up**, positions in metres,
orientations as unit ``(x, y, z, w)`` quaternions. The absolute (UMI) IK
mode absorbs any world yaw/translation and the rigid tracker→gripper
mount at engage time, so gravity alignment and scale are the only things
a backend must get right.

Backends whose native world frame is z-up (libsurvive, and the Ultimate
tracker's SLAM frame) convert through :func:`zup_to_yup_pos` /
:func:`zup_to_yup_quat`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

# Basis change from a right-handed z-up world to the WebXR y-up world:
# x' = x, y' = z, z' = -y (a -90 deg rotation about x). As a quaternion
# (x, y, z, w) it conjugates orientations: q_yup = Q_C * q_zup * Q_C^-1.
_SQRT_HALF = float(np.sqrt(0.5))
_Q_ZUP_TO_YUP = np.array([-_SQRT_HALF, 0.0, 0.0, _SQRT_HALF])


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two ``(x, y, z, w)`` quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def zup_to_yup_pos(pos: np.ndarray) -> np.ndarray:
    """Map a position from a right-handed z-up world into the y-up world."""
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    return np.array([x, z, -y])


def zup_to_yup_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    """Map an ``(x, y, z, w)`` orientation from a z-up world into the y-up world."""
    q_conj = np.array(
        [-_Q_ZUP_TO_YUP[0], -_Q_ZUP_TO_YUP[1], -_Q_ZUP_TO_YUP[2], _Q_ZUP_TO_YUP[3]]
    )
    out = quat_multiply(quat_multiply(_Q_ZUP_TO_YUP, np.asarray(quat_xyzw)), q_conj)
    n = float(np.linalg.norm(out))
    return out / n if n > 0.0 else np.array([0.0, 0.0, 0.0, 1.0])


@dataclass
class TrackerPose:
    """One tracker's latest pose in the WebXR (y-up) world convention.

    Attributes:
        pos:      Position in metres, shape (3,).
        quat:     Unit orientation quaternion ``(x, y, z, w)``, shape (4,).
        t:        Host capture time in ``time.perf_counter()`` seconds.
        tracking: ``True`` while the backend considers the pose trustworthy
                  (lighthouse lock / SLAM converged). The bridge holds the
                  last good pose while this is ``False`` so IK never chases
                  an occlusion or relocalisation glitch.
    """

    pos: np.ndarray
    quat: np.ndarray
    t: float
    tracking: bool = True


class TrackerSource(abc.ABC):
    """A backend producing per-device 6-DOF poses in the y-up convention.

    Device keys are backend-native, stable identifiers (a libsurvive
    codename like ``T20``, an Ultimate tracker MAC). The operator binds
    them to the left/right rig sides once with ``axol tracker.identify``;
    the bridge then maps ``poses()[config.left]`` → ``l_ee`` etc.
    """

    @abc.abstractmethod
    def start(self) -> None:
        """Open the hardware and begin streaming poses (non-blocking)."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop streaming and release the hardware."""

    @abc.abstractmethod
    def poses(self) -> dict[str, TrackerPose]:
        """Latest pose per discovered device. Safe to call from any thread."""

    def __enter__(self) -> TrackerSource:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
