"""Rigid raw-tracker TCP mapping, independent of the virtual IK solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..mantis.relative import quat_xyzw_to_matrix


def _multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    av, aw = a[:3], a[3]
    bv, bw = b[:3], b[3]
    return np.r_[aw * bv + bw * av + np.cross(av, bv), aw * bw - np.dot(av, bv)]


@dataclass(frozen=True)
class AbsoluteTCPMapping:
    """One engage's immutable tracker→TCP and world→base calibration.

    The worker composes mount calibration and any uncalibrated engage offset
    once. Mapping a new raw sample requires only these rigid transforms; it
    never depends on virtual joint solutions, pose filters, or solver timing.
    """

    base: tuple[float, ...]
    left: tuple[float, ...]
    right: tuple[float, ...]
    shoulder_left: tuple[float, ...]
    shoulder_right: tuple[float, ...]
    reach_limit: float

    @classmethod
    def from_message(cls, message: dict[str, Any]) -> AbsoluteTCPMapping:
        values = {}
        for name in ("base", "left", "right", "shoulder_left", "shoulder_right"):
            value = np.asarray(message[name], dtype=np.float64)
            size = 3 if name.startswith("shoulder_") else 7
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"invalid raw TCP calibration: {name}")
            if size == 7:
                norm = np.linalg.norm(value[3:])
                if norm < 1e-9:
                    raise ValueError(f"invalid raw TCP quaternion: {name}")
                value = value.copy()
                value[3:] /= norm
            values[name] = tuple(float(v) for v in value)
        limit = float(message["reach_limit"])
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError("invalid raw TCP reach limit")
        return cls(**values, reach_limit=limit)

    def map_frame(self, frame: Any) -> dict[str, Any]:
        """Map this frame's actual poses; the caller retains its capture time."""
        base = np.asarray(self.base)
        q_bw = base[3:].copy()
        q_bw[:3] *= -1
        r_bw = quat_xyzw_to_matrix(q_bw)
        result: dict[str, Any] = {}
        out_of_reach = []
        for side in ("left", "right"):
            pose = getattr(frame, f"{side[0]}_ee")
            pos = np.array([pose.position.x, pose.position.y, pose.position.z])
            quat = np.array(
                [
                    pose.quaternion.x,
                    pose.quaternion.y,
                    pose.quaternion.z,
                    pose.quaternion.w,
                ]
            )
            norm = np.linalg.norm(quat)
            if not np.isfinite(pos).all() or not np.isfinite(norm) or norm < 1e-9:
                raise ValueError("invalid raw tracker pose")
            quat /= norm
            offset = np.asarray(getattr(self, side))
            tcp_pos = r_bw @ (pos + quat_xyzw_to_matrix(quat) @ offset[:3] - base[:3])
            tcp_quat = _multiply(q_bw, _multiply(quat, offset[3:]))
            tcp_quat /= np.linalg.norm(tcp_quat)
            result[side] = [*map(float, tcp_pos), *map(float, tcp_quat)]
            shoulder = np.asarray(getattr(self, f"shoulder_{side}"))
            if np.linalg.norm(tcp_pos - shoulder) > self.reach_limit:
                out_of_reach.append(side)
        result["out_of_reach"] = out_of_reach
        return result
