"""The robot's URDF as a transform tree, in plain numpy.

The MCAP recorder publishes the tree the way ROS's ``robot_state_publisher``
does: each fixed joint once on ``/tf_static`` and each moving joint on
``/tf`` whenever its value changes, as the transform from the joint's parent
link to its child link. It also computes end-effector poses by chaining those
transforms. Doing it here keeps JAX and the IK model out of the recorder
process.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ElementTree
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def rpy_from_matrix(rotation: np.ndarray) -> tuple[float, float, float]:
    """URDF ``(roll, pitch, yaw)`` of a rotation matrix (inverse of
    :func:`rpy_matrix`)."""
    m = rotation
    sp = -float(m[2, 0])
    if abs(sp) < 1.0 - 1e-12:
        pitch = math.asin(sp)
        roll = math.atan2(float(m[2, 1]), float(m[2, 2]))
        yaw = math.atan2(float(m[1, 0]), float(m[0, 0]))
    else:  # pitch at +-90 degrees: only roll - yaw (or + yaw) is defined
        pitch = math.copysign(math.pi / 2, sp)
        roll = 0.0
        yaw = math.atan2(-float(m[0, 1]), float(m[1, 1]))
    return roll, pitch, yaw


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def quaternion_xyzw(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Unit quaternion ``(x, y, z, w)`` with ``w >= 0`` for a rotation matrix."""
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if w < 0.0:
        norm = -norm
    return (x / norm, y / norm, z / norm, w / norm)


def pose7(transform: np.ndarray) -> tuple[float, ...]:
    """``[x, y, z, qx, qy, qz, qw]`` for a 4x4 transform."""
    t = transform[:3, 3]
    return (float(t[0]), float(t[1]), float(t[2]), *quaternion_xyzw(transform[:3, :3]))


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray  # 4x4
    axis: np.ndarray  # unit 3-vector
    lower: float | None
    upper: float | None
    mimic: tuple[str, float, float] | None  # (joint, multiplier, offset)

    @property
    def moving(self) -> bool:
        return self.type in ("revolute", "continuous", "prismatic")

    def transform(self, q: float) -> np.ndarray:
        """Parent link to child link at joint value ``q``."""
        if not self.moving:
            return self.origin
        out = self.origin.copy()
        if self.type == "prismatic":
            out[:3, 3] += self.origin[:3, :3] @ (self.axis * q)
        else:
            out[:3, :3] = self.origin[:3, :3] @ axis_angle_matrix(self.axis, q)
        return out


def _floats(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if not text:
        return default
    return tuple(float(v) for v in text.split())


class UrdfTree:
    """Joints and links of one URDF."""

    def __init__(self, xml_text: str) -> None:
        root = ElementTree.fromstring(xml_text)
        self.xml = xml_text
        self.joints: dict[str, UrdfJoint] = {}
        self.parent_joint: dict[str, UrdfJoint] = {}
        links = {link.get("name") for link in root.findall("link")}
        for el in root.findall("joint"):
            origin_el = el.find("origin")
            xyz = _floats(
                origin_el.get("xyz") if origin_el is not None else None, (0.0,) * 3
            )
            rpy = _floats(
                origin_el.get("rpy") if origin_el is not None else None, (0.0,) * 3
            )
            origin = np.eye(4)
            origin[:3, :3] = rpy_matrix(*rpy)
            origin[:3, 3] = xyz
            axis_el = el.find("axis")
            axis = np.array(
                _floats(
                    axis_el.get("xyz") if axis_el is not None else None, (1.0, 0.0, 0.0)
                )
            )
            norm = float(np.linalg.norm(axis))
            axis = axis / norm if norm > 0 else np.array([1.0, 0.0, 0.0])
            limit = el.find("limit")
            lower = upper = None
            if limit is not None:
                lower = float(limit.get("lower")) if limit.get("lower") else None
                upper = float(limit.get("upper")) if limit.get("upper") else None
            mimic_el = el.find("mimic")
            mimic = None
            if mimic_el is not None:
                mimic = (
                    str(mimic_el.get("joint")),
                    float(mimic_el.get("multiplier", "1")),
                    float(mimic_el.get("offset", "0")),
                )
            joint = UrdfJoint(
                name=str(el.get("name")),
                type=str(el.get("type")),
                parent=str(el.find("parent").get("link")),  # type: ignore[union-attr]
                child=str(el.find("child").get("link")),  # type: ignore[union-attr]
                origin=origin,
                axis=axis,
                lower=lower,
                upper=upper,
                mimic=mimic,
            )
            self.joints[joint.name] = joint
            self.parent_joint[joint.child] = joint
        children = set(self.parent_joint)
        roots = sorted(link for link in links if link not in children)
        self.root = roots[0] if roots else ""
        self.links = links

    @classmethod
    def load(cls, path: Path) -> UrdfTree:
        return cls(Path(path).read_text())

    def has_link(self, link: str) -> bool:
        return link in self.links

    def value(self, joint: UrdfJoint, q: Mapping[str, float]) -> float:
        """A joint's value, following ``<mimic>``; zero when not given."""
        if joint.mimic is not None:
            source, multiplier, offset = joint.mimic
            return float(q.get(source, 0.0)) * multiplier + offset
        return float(q.get(joint.name, 0.0))

    def static_transforms(self) -> list[tuple[str, str, np.ndarray]]:
        """``(parent, child, transform)`` for every fixed joint."""
        return [
            (j.parent, j.child, j.origin) for j in self.joints.values() if not j.moving
        ]

    def moving_joints(self) -> list[UrdfJoint]:
        return [j for j in self.joints.values() if j.moving]

    def path(self, base: str, tip: str) -> list[UrdfJoint]:
        """Joints from ``base`` down to ``tip`` (``tip`` must hang below ``base``)."""
        chain: list[UrdfJoint] = []
        link = tip
        while link != base:
            joint = self.parent_joint.get(link)
            if joint is None:
                raise KeyError(f"{tip} does not hang below {base}")
            chain.append(joint)
            link = joint.parent
        chain.reverse()
        return chain

    def fk(self, base: str, tip: str, q: Mapping[str, float]) -> np.ndarray:
        """``tip``'s pose in ``base``'s frame for joint values ``q``."""
        out = np.eye(4)
        for joint in self.path(base, tip):
            out = out @ joint.transform(self.value(joint, q))
        return out
