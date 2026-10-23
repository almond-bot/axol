"""Absolute (Mantis) calibration: raw tracker pose → base-frame TCP pose.

The IK worker owns the absolute-mode calibration: the calibrated
tracker→gripper transform per side and the world→base fit (plus per-side
offsets) solved at engage. It reports that calibration with every engaged
reply so Mantis data collection can map each raw tracker pose into the robot
base frame itself, at the pose's own capture time, instead of waiting for the
IK solve that consumed an older, interpolated frame.

The math mirrors ``IKWorker._apply_tcp_transform`` followed by
``IKWorker._absolute_target``, minus the worker's control-side pose filter:
the recorded pose is the calibrated tracker measurement itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

_SIDES = ("left", "right")


def matrix_to_quat_xyzw(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to an ``(x, y, z, w)`` quaternion."""
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        return (
            float(R[2, 1] - R[1, 2]) / s,
            float(R[0, 2] - R[2, 0]) / s,
            float(R[1, 0] - R[0, 1]) / s,
            0.25 * s,
        )
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return (
            0.25 * s,
            float(R[0, 1] + R[1, 0]) / s,
            float(R[0, 2] + R[2, 0]) / s,
            float(R[2, 1] - R[1, 2]) / s,
        )
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return (
            float(R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            float(R[1, 2] + R[2, 1]) / s,
            float(R[0, 2] - R[2, 0]) / s,
        )
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return (
        float(R[0, 2] + R[2, 0]) / s,
        float(R[1, 2] + R[2, 1]) / s,
        0.25 * s,
        float(R[1, 0] - R[0, 1]) / s,
    )


def _quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat / np.linalg.norm(quat))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rigid(value: object) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse a ``{"pos": [3], "rot": [[3]x3]}`` message into float64 arrays."""
    if not isinstance(value, dict):
        return None
    try:
        pos = np.asarray(value["pos"], dtype=np.float64)
        rot = np.asarray(value["rot"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if pos.shape != (3,) or rot.shape != (3, 3):
        return None
    if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(rot))):
        return None
    return pos, rot


def _encode_rigid(pos: np.ndarray, rot: np.ndarray) -> dict[str, list[Any]]:
    return {
        "pos": [float(v) for v in pos],
        "rot": [[float(v) for v in row] for row in rot],
    }


@dataclass(frozen=True)
class AbsoluteCalibration:
    """One engage's calibration, as the IK worker applied it.

    ``base`` is the world→base fit ``(t_wb, R_wb)``; ``tcp`` the calibrated
    tracker→gripper transform per side (``None`` when the side has none);
    ``offset`` the per-side engage offset (identity when ``tcp`` is set).
    """

    base: tuple[np.ndarray, np.ndarray]
    tcp: dict[str, tuple[np.ndarray, np.ndarray] | None]
    offset: dict[str, tuple[np.ndarray, np.ndarray]]

    def to_message(self) -> dict[str, Any]:
        """JSON-safe form, sent over the IK worker pipe."""
        return {
            "base": _encode_rigid(*self.base),
            **{
                side: {
                    "tcp": (
                        _encode_rigid(*self.tcp[side])
                        if self.tcp[side] is not None
                        else None
                    ),
                    "offset": _encode_rigid(*self.offset[side]),
                }
                for side in _SIDES
            },
        }

    @classmethod
    def from_message(cls, msg: object) -> AbsoluteCalibration | None:
        """Parse :meth:`to_message` output; ``None`` for anything malformed."""
        if not isinstance(msg, dict):
            return None
        base = _rigid(msg.get("base"))
        if base is None:
            return None
        tcp: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
        offset: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for side in _SIDES:
            entry = msg.get(side)
            if not isinstance(entry, dict):
                return None
            side_offset = _rigid(entry.get("offset"))
            if side_offset is None:
                return None
            offset[side] = side_offset
            if entry.get("tcp") is None:
                tcp[side] = None
            else:
                side_tcp = _rigid(entry["tcp"])
                if side_tcp is None:
                    return None
                tcp[side] = side_tcp
        return cls(base=base, tcp=tcp, offset=offset)

    def base_pose(
        self, side: str, pos: np.ndarray, quat_xyzw: np.ndarray
    ) -> tuple[float, ...]:
        """Map a raw world-frame tracker pose to the base-frame TCP pose.

        Returns ``(x, y, z, qx, qy, qz, qw)``, the layout of
        :class:`~almond_axol.teleop.core.TCPPoseSnapshot` sides.
        """
        pos = np.asarray(pos, dtype=np.float64)
        rot = _quat_xyzw_to_matrix(np.asarray(quat_xyzw, dtype=np.float64))
        tcp = self.tcp[side]
        if tcp is not None:
            p_x, R_x = tcp
            pos = pos + rot @ p_x
            rot = rot @ R_x
        p_off, R_off = self.offset[side]
        p_w = pos + rot @ p_off
        R_w = rot @ R_off
        t_wb, R_wb = self.base
        p_b = R_wb.T @ (p_w - t_wb)
        R_b = R_wb.T @ R_w
        return (*(float(v) for v in p_b), *matrix_to_quat_xyzw(R_b))
