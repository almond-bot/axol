"""MuJoCo forward kinematics and NumPy pose conversion without JAX.

The observation layout is the same as :mod:`almond_axol.kinematics.fk`:
the current URDF's gripper mount in world coordinates, with an axis-angle
rotation vector. Gripper openings do not change that mount's pose.
"""

from __future__ import annotations

import threading
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from ..constants import URDF_PATH, Joint, urdf_arm_joint_names, urdf_body_name

EE_AXES: tuple[str, ...] = ("x", "y", "z", "rx", "ry", "rz")


def pose6_to_pos_rot(pose6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode a finite ``[x, y, z, rx, ry, rz]`` pose into position/rotation."""
    pose = np.asarray(pose6, dtype=np.float32)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("pose6 must contain six finite values")
    vector = pose[3:].astype(np.float64)
    angle = float(np.linalg.norm(vector))
    # The unnormalized Rodrigues formula avoids loss of accuracy near zero.
    x, y, z = vector
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    if angle < 1e-4:
        squared = angle * angle
        sinc = 1 - squared / 6 + squared * squared / 120
        cosc = 0.5 - squared / 24 + squared * squared / 720
    else:
        sinc = np.sin(angle) / angle
        cosc = (1 - np.cos(angle)) / (angle * angle)
    rotation = np.eye(3) + sinc * skew + cosc * (skew @ skew)
    return pose[:3].copy(), rotation.astype(np.float32)


def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.asarray(matrix).reshape(9))
    if quaternion[0] < 0:
        quaternion *= -1
    length = float(np.linalg.norm(quaternion[1:]))
    scale = 2 * np.arctan2(length, quaternion[0]) / length if length > 1e-12 else 2.0
    return quaternion[1:] * scale


class AxolForwardKinematics:
    """Current-world gripper-mount FK with no solver or compiled JAX graph.

    One model and data pair are protected by a lock because observation and
    recording callers may share this helper across their threads.
    """

    def __init__(self) -> None:
        root = ET.fromstring(URDF_PATH.read_text())
        for link in root.findall("link"):
            for child in list(link):
                if child.tag in {"visual", "collision"}:
                    link.remove(child)
        extension = ET.SubElement(root, "mujoco")
        ET.SubElement(extension, "compiler", fusestatic="false")
        self._model = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode")
        )
        self._data = mujoco.MjData(self._model)
        self._lock = threading.Lock()
        self._indices = []
        self._bodies = []
        for is_left in (True, False):
            joint_ids = [
                mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in urdf_arm_joint_names(is_left=is_left)
            ]
            body = mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_BODY,
                urdf_body_name(Joint.GRIPPER, is_left=is_left),
            )
            if min(joint_ids) < 0 or body < 0:
                raise ValueError("Axol URDF is missing an arm joint or gripper frame")
            self._indices.append(self._model.jnt_qposadr[joint_ids])
            self._bodies.append(body)

    def ee_poses(
        self, left_pos: np.ndarray, right_pos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return two pose6 vectors; ignore optional trailing gripper values."""
        positions = [
            np.asarray(value, dtype=np.float64) for value in (left_pos, right_pos)
        ]
        if any(
            value.ndim != 1 or len(value) < 7 or not np.isfinite(value[:7]).all()
            for value in positions
        ):
            raise ValueError("Each arm needs seven finite joint angles")
        with self._lock:
            self._data.qpos[:] = 0
            for indices, value in zip(self._indices, positions, strict=True):
                self._data.qpos[indices] = value[:7]
            mujoco.mj_kinematics(self._model, self._data)
            poses = [
                np.concatenate(
                    (self._data.xpos[body], _rotation_vector(self._data.xmat[body]))
                ).astype(np.float32)
                for body in self._bodies
            ]
        return poses[0], poses[1]
