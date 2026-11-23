"""What the MCAP recorder knows about the robot before an episode starts.

:class:`Scene` holds the URDF tree, which joints are measured and how, and
the per-camera calibration. It is built once per session in the recorder
process and shared by every episode.

Camera naming. The control panel's camera slots are ``overhead`` (the ZED X
Mini head camera, usually stereo) and ``left_arm`` / ``right_arm`` (the wrist
cameras). Their topics and optical frames follow the customer-facing names:

=========================  =====================  ============================
camera                     topic prefix           optical frame (URDF link)
=========================  =====================  ============================
``overhead``, left eye     ``/head/left``         ``head_camera_left_optical``
``overhead``, right eye    ``/head/right``        ``head_camera_right_optical``
``left_arm``               ``/wrist/left``        ``left_wrist_camera_optical``
``right_arm``              ``/wrist/right``       ``right_wrist_camera_optical``
=========================  =====================  ============================

Any other camera name is used as-is (``/<name>``, ``<name>_optical``).
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...constants import urdf_arm_joint_names
from .rings import JOINT_NAMES, SIDES
from .urdf_tree import UrdfTree
from .zed_conf import (
    CalibrationFile,
    Intrinsics,
    camera_intrinsics,
    load_calibration,
    stereo_transform,
)

_WRIST_SLOTS = {"left_arm": "left", "right_arm": "right"}
LIFT_JOINT = "lift"
BASE_JOINTS = ("base_x", "base_y", "base_yaw")


def camera_topic(camera: str, eye: str | None) -> str:
    """Topic prefix for a recorded camera (``/head/left``, ``/wrist/right``)."""
    if camera == "overhead":
        return f"/head/{eye or 'left'}"
    if camera in _WRIST_SLOTS:
        return f"/wrist/{_WRIST_SLOTS[camera]}"
    return f"/{camera}" + (f"/{eye}" if eye else "")


def camera_base_topic(camera: str) -> str:
    """Topic prefix for a whole camera (both eyes), such as ``/head``."""
    if camera == "overhead":
        return "/head"
    return camera_topic(camera, None)


def imu_topic(camera: str) -> str:
    """The topic a camera's IMU samples go to (``/head/imu``)."""
    return f"{camera_base_topic(camera)}/imu"


def camera_frame_id(camera: str, eye: str | None) -> str:
    """The camera's optical frame: +z out of the lens, +x right, +y down."""
    if camera == "overhead":
        return f"head_camera_{eye or 'left'}_optical"
    if camera in _WRIST_SLOTS:
        return f"{_WRIST_SLOTS[camera]}_wrist_camera_optical"
    return f"{camera}_{eye}_optical" if eye else f"{camera}_optical"


@dataclass
class CameraSource:
    """One recorded image stream and its calibration."""

    source: str
    camera: str
    eye: str | None
    topic: str
    frame_id: str
    width: int
    height: int
    fps: int
    codec: str
    serial: int | None
    resolution: str | None
    stereo: bool
    intrinsics: Intrinsics | None = None
    calibration: CalibrationFile | None = None
    # The socket this camera's IMU samples arrive on (stereo cameras whose
    # ZED plugin has Axol's IMU patch), else None.
    imu_socket: str | None = None
    imu_requested: bool = False

    @classmethod
    def from_meta(cls, source: str, meta: dict[str, Any]) -> CameraSource:
        camera = str(meta.get("camera") or source)
        eye = meta.get("eye")
        serial = meta.get("serial")
        return cls(
            source=source,
            camera=camera,
            eye=str(eye) if eye else None,
            topic=camera_topic(camera, str(eye) if eye else None),
            frame_id=camera_frame_id(camera, str(eye) if eye else None),
            width=int(meta["width"]),
            height=int(meta["height"]),
            fps=int(meta["fps"]),
            codec=str(meta.get("codec", "h264")),
            serial=int(serial) if serial is not None else None,
            resolution=str(meta["resolution"]) if meta.get("resolution") else None,
            stereo=bool(meta.get("stereo", False)),
            imu_socket=str(meta["imu_socket"]) if meta.get("imu_socket") else None,
            imu_requested=bool(meta.get("imu_requested", False)),
        )

    def load_calibration(self, directory: Path) -> None:
        """Fill :attr:`intrinsics` from the camera's factory calibration file."""
        if self.serial is None or self.resolution is None:
            return
        self.calibration = load_calibration(self.serial, directory)
        if self.calibration is None:
            return
        self.intrinsics = camera_intrinsics(
            self.calibration,
            eye=self.eye or "left",
            capture_resolution=self.resolution,
            width=self.width,
            height=self.height,
            # The Camera Kit's mono cameras are the ZED X One S fisheye.
            fisheye=not self.stereo,
        )


def finger_travel(path: Path, tree: UrdfTree) -> tuple[float, float] | None:
    """``(closed, stroke)`` of the left finger joint, or ``None`` without one.

    At gripper opening ``g`` (0 closed, 1 open) each driving finger joint is
    at ``closed + g * stroke``: the same mapping IK uses
    (:func:`almond_axol.kinematics.model.finger_slide`). The closed end is the
    joint limit where the two fingers' collision hulls are nearest.
    """
    driver = next(
        (
            j
            for j in tree.joints.values()
            if j.type == "prismatic" and j.child == "left_finger_1" and j.mimic is None
        ),
        None,
    )
    if driver is None or driver.lower is None or driver.upper is None:
        return None
    try:
        import trimesh

        root = ElementTree.fromstring(tree.xml)
        centroids: dict[str, np.ndarray] = {}
        for link in root.findall("link"):
            name = link.get("name")
            if name not in ("left_finger_1", "left_finger_2"):
                continue
            points = []
            for col in link.findall("collision"):
                mesh_el = col.find("geometry/mesh")
                if mesh_el is None:
                    continue
                mesh = trimesh.load(
                    path.parent / str(mesh_el.get("filename")), force="mesh"
                )
                points.append(np.asarray(mesh.vertices))
            if points:
                centroids[str(name)] = np.vstack(points).mean(axis=0)
        if len(centroids) != 2:
            return None
        base = driver.parent

        def gap(q: float) -> float:
            values = {driver.name: q}
            ends = []
            for link_name, centroid in centroids.items():
                transform = tree.fk(base, link_name, values)
                ends.append(transform[:3, :3] @ centroid + transform[:3, 3])
            return float(np.linalg.norm(ends[0] - ends[1]))

        closed, opened = sorted((driver.lower, driver.upper), key=gap)
    except Exception:  # noqa: BLE001 - fingers are optional in the TF tree
        return None
    return closed, opened - closed


@dataclass
class Scene:
    """The robot model and cameras one recording session describes."""

    tree: UrdfTree
    urdf_name: str
    urdf_sha256: str
    has_gripper: bool
    jelly_wheels: bool
    jelly_lift: bool
    cameras: dict[str, CameraSource]
    arm_joints: dict[str, list[str]] = field(default_factory=dict)
    finger: tuple[float, float] | None = None
    lift_travel: float = 0.0
    stereo: np.ndarray | None = None
    stereo_source: str = ""

    @classmethod
    def build(
        cls,
        urdf_path: Path,
        *,
        has_gripper: bool,
        jelly_wheels: bool,
        jelly_lift: bool,
        cameras: dict[str, CameraSource],
        calibration_dir: Path,
    ) -> Scene:
        urdf_path = Path(urdf_path)
        raw = urdf_path.read_bytes()
        tree = UrdfTree(raw.decode())
        scene = cls(
            tree=tree,
            urdf_name=urdf_path.name,
            urdf_sha256=hashlib.sha256(raw).hexdigest(),
            has_gripper=has_gripper,
            jelly_wheels=jelly_wheels and all(j in tree.joints for j in BASE_JOINTS),
            jelly_lift=jelly_lift and LIFT_JOINT in tree.joints,
            cameras=cameras,
        )
        scene.arm_joints = {
            side: urdf_arm_joint_names(is_left=side == "left") for side in SIDES
        }
        if has_gripper:
            scene.finger = finger_travel(urdf_path, tree)
        lift = tree.joints.get(LIFT_JOINT)
        if lift is not None and lift.upper is not None:
            scene.lift_travel = float(lift.upper)
        for camera in cameras.values():
            camera.load_calibration(calibration_dir)
        scene._load_stereo()
        return scene

    def _load_stereo(self) -> None:
        for camera in self.cameras.values():
            if camera.camera != "overhead" or not camera.stereo:
                continue
            if camera.calibration is None or camera.resolution is None:
                continue
            transform = stereo_transform(camera.calibration, camera.resolution)
            if transform is not None:
                self.stereo = transform
                self.stereo_source = f"ZED factory calibration SN{camera.serial}"
                return

    # -- joint values ------------------------------------------------------

    def joint_state_names(self) -> list[str]:
        """``/joint_states`` names: URDF arm joints, then ``<side>_gripper``."""
        names: list[str] = []
        for side in SIDES:
            names += self.arm_joints[side]
            if self.has_gripper:
                names.append(f"{side}_gripper")
        return names

    def joint_columns(self) -> list[int]:
        """Index into a 16-joint (Joint order, left then right) vector per name."""
        columns: list[int] = []
        for s, _side in enumerate(SIDES):
            base = s * len(JOINT_NAMES)
            columns += [base + i for i in range(len(JOINT_NAMES) - 1)]
            if self.has_gripper:
                columns.append(base + len(JOINT_NAMES) - 1)
        return columns

    def urdf_values(self, joints16: np.ndarray) -> dict[str, float]:
        """URDF joint values for the arms (and fingers) from 16 joint readings."""
        values: dict[str, float] = {}
        n = len(JOINT_NAMES)
        for s, side in enumerate(SIDES):
            for i, name in enumerate(self.arm_joints[side]):
                values[name] = float(joints16[s * n + i])
            if self.finger is not None:
                opening = float(joints16[s * n + n - 1])
                if np.isfinite(opening):
                    closed, stroke = self.finger
                    g = min(max(opening, 0.0), 1.0)
                    values[f"{side}_finger_1_joint"] = closed + g * stroke
        return values

    def arm_tf_joints(self) -> list[str]:
        """Moving joints published with every joint sample."""
        names = [
            n for side in SIDES for n in self.arm_joints[side] if n in self.tree.joints
        ]
        if self.finger is not None:
            for joint in self.tree.joints.values():
                if not joint.moving:
                    continue
                driver = joint.mimic[0] if joint.mimic else joint.name
                if driver in ("left_finger_1_joint", "right_finger_1_joint"):
                    names.append(joint.name)
        return names

    def driven_by(self, driver: str) -> list[str]:
        """Moving joints whose value is ``driver``'s (itself and its mimics)."""
        return [
            j.name
            for j in self.tree.joints.values()
            if j.moving and (j.name == driver or (j.mimic and j.mimic[0] == driver))
        ]

    def dynamic_joints(self) -> set[str]:
        """Every moving joint this session publishes on ``/tf``."""
        names = set(self.arm_tf_joints())
        if self.jelly_lift:
            names.update(self.driven_by(LIFT_JOINT))
        if self.jelly_wheels:
            names.update(BASE_JOINTS)
        return names

    def assumed_joints(self) -> list[str]:
        """Moving joints nothing measures, written once at zero on ``/tf_static``.

        Jelly's wheels are left out: their spin does not move any other frame.
        """
        dynamic = self.dynamic_joints()
        return [
            j.name
            for j in self.tree.joints.values()
            if j.moving and j.name not in dynamic and not j.name.startswith("wheel_")
        ]

    def lift_value(self, raised: float) -> float:
        """URDF lift joint value (m lowered) for Jelly's raised fraction."""
        return (1.0 - min(max(raised, 0.0), 1.0)) * self.lift_travel

    def static_transforms(self) -> list[tuple[str, str, np.ndarray]]:
        """Every ``/tf_static`` transform: fixed joints and assumed joints.

        With a factory stereo calibration, the right head eye is placed from
        the left one by the measured transform instead of the CAD baseline.
        """
        out: list[tuple[str, str, np.ndarray]] = []
        tree = self.tree
        left_eye = tree.parent_joint.get("head_camera_left_optical")
        right_eye = tree.parent_joint.get("head_camera_right_optical")
        for parent, child, transform in tree.static_transforms():
            if (
                self.stereo is not None
                and child == "head_camera_right_optical"
                and left_eye is not None
                and right_eye is not None
                and left_eye.parent == parent
                and not left_eye.moving
            ):
                transform = left_eye.origin @ self.stereo
            out.append((parent, child, transform))
        for name in self.assumed_joints():
            joint = tree.joints[name]
            out.append(
                (joint.parent, joint.child, joint.transform(tree.value(joint, {})))
            )
        return out

    def imus(self) -> dict[str, tuple[str, str]]:
        """``{topic: (socket path, frame_id)}`` for every camera IMU recorded.

        A camera's IMU samples are given in its left eye's optical frame axes
        (the ZED plugin opens the camera in the SDK's IMAGE coordinate system).
        """
        out: dict[str, tuple[str, str]] = {}
        for camera in self.cameras.values():
            if camera.imu_socket:
                out.setdefault(
                    imu_topic(camera.camera),
                    (camera.imu_socket, camera_frame_id(camera.camera, "left")),
                )
        return out

    def ee_links(self) -> dict[str, str]:
        """End-effector link per side (the gripper mount), when the URDF has it."""
        return {
            side: f"{side}_gripper"
            for side in SIDES
            if self.tree.has_link(f"{side}_gripper")
        }

    def ee_base(self) -> str:
        """Frame the end-effector poses are given in: the torso (``root``)."""
        return "root" if self.tree.has_link("root") else self.tree.root
