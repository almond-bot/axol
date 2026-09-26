"""The Axol-on-Jelly simulation URDF (``axol_jelly_sim.urdf``).

Not loaded by the control stack; this pins what makes it usable in a physics
simulator (Isaac Sim, MuJoCo): the right joints move, the arm chain is the
IK model's, masses are AxolConfig's where they are known and physically
valid everywhere, mesh paths resolve without a ROS package, and the camera
optical frames sit where the cameras are.
"""

from __future__ import annotations

import unittest
from xml.etree import ElementTree

import mujoco
import numpy as np
import yourdfpy

from almond_axol.constants import (
    ARM_JOINTS,
    JELLY_SIM_URDF,
    AxolModel,
    urdf_arm_joint_names,
    urdf_body_name,
    urdf_path,
)
from almond_axol.robot.config import AxolConfig

ARM_JOINT_NAMES = urdf_arm_joint_names(is_left=True) + urdf_arm_joint_names(
    is_left=False
)
CAMERAS = (
    "left_wrist_camera_optical",
    "right_wrist_camera_optical",
    "head_camera_left_optical",
    "head_camera_right_optical",
)


def _load(path) -> yourdfpy.URDF:
    return yourdfpy.URDF.load(
        str(path), load_meshes=False, build_collision_scene_graph=False
    )


class JellySimUrdfTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = _load(JELLY_SIM_URDF)
        cls.arm = _load(urdf_path(AxolModel.MOBILE))
        cls.root = ElementTree.parse(JELLY_SIM_URDF).getroot()
        cls.zero = {n: 0.0 for n in cls.sim.actuated_joint_names}

    def test_moving_joints(self) -> None:
        extra = {
            "lift",
            "left_finger_1_joint",
            "right_finger_1_joint",
            "wheel_1",
            "wheel_2",
            "wheel_3",
            "wheel_4",
        }
        self.assertEqual(
            set(self.sim.actuated_joint_names), set(ARM_JOINT_NAMES) | extra
        )
        mimics = {
            j.name: j.mimic.joint for j in self.sim.robot.joints if j.mimic is not None
        }
        self.assertEqual(
            mimics,
            {
                "lift_2": "lift",
                "left_finger_2_joint": "left_finger_1_joint",
                "right_finger_2_joint": "right_finger_1_joint",
            },
        )
        self.assertEqual(self.sim.base_link, "jelly")

    def test_arm_chain_is_the_ik_models(self) -> None:
        rng = np.random.default_rng(1)
        for _ in range(5):
            q = dict(zip(ARM_JOINT_NAMES, rng.uniform(-1.0, 1.0, 14)))
            self.arm.update_cfg(q)
            self.sim.update_cfg({**self.zero, **q})
            for side in (True, False):
                for body in (urdf_body_name(j, is_left=side) for j in ARM_JOINTS):
                    np.testing.assert_allclose(
                        self.sim.get_transform(body, "root"),
                        self.arm.get_transform(body, "root"),
                        atol=1e-6,
                        err_msg=body,
                    )

    def test_fingers_open_together(self) -> None:
        # Driving the first finger's joint moves both fingers their full
        # stroke, in opposite directions (the second finger mimics).
        self.sim.update_cfg(self.zero)
        closed = [
            self.sim.get_transform(f"left_finger_{i}")[:3, 3].copy() for i in (1, 2)
        ]
        self.sim.update_cfg({**self.zero, "left_finger_1_joint": 0.0588})
        moved = [
            self.sim.get_transform(f"left_finger_{i}")[:3, 3] - c
            for i, c in zip((1, 2), closed)
        ]
        for step in moved:
            self.assertAlmostEqual(float(np.linalg.norm(step)), 0.0588, places=6)
        self.assertAlmostEqual(float(moved[0] @ moved[1]), -(0.0588**2), places=6)

    def test_arm_masses_are_axol_configs(self) -> None:
        cfg = AxolConfig()
        links = {link.name: link for link in self.sim.robot.links}
        for arm, side in ((cfg.left, True), (cfg.right, False)):
            prefix = "left" if side else "right"
            for joint in ARM_JOINTS[:-1]:
                inertial = links[urdf_body_name(joint, is_left=side)].inertial
                jc = getattr(arm, joint.value)
                self.assertAlmostEqual(inertial.mass, jc.mass, places=5)
                np.testing.assert_allclose(inertial.origin[:3, 3], jc.com, atol=1e-6)
            # wrist_3's tuned mass/CoM lumps the gripper assembly; the sim
            # model splits it across links but keeps the total.
            w3 = getattr(arm, ARM_JOINTS[-1].value)
            parts = [
                f"{prefix}_w2",
                f"{prefix}_gripper",
                f"{prefix}_finger_1",
                f"{prefix}_finger_2",
                f"{prefix}_wrist_camera",
            ]
            self.sim.update_cfg(self.zero)
            to_w2 = np.linalg.inv(self.sim.get_transform(f"{prefix}_w2"))
            total, moment = 0.0, np.zeros(3)
            for part in parts:
                inertial = links[part].inertial
                com = (to_w2 @ self.sim.get_transform(part) @ inertial.origin[:, 3])[:3]
                total += inertial.mass
                moment += inertial.mass * com
            self.assertAlmostEqual(total, w3.mass, places=5)
            np.testing.assert_allclose(moment / total, w3.com, atol=1e-5)

    def test_every_body_is_physically_valid(self) -> None:
        for link in self.sim.robot.links:
            if not (link.visuals or link.collisions):
                continue  # frame-only links (camera frames, the torso root)
            with self.subTest(link=link.name):
                self.assertIsNotNone(link.inertial)
                self.assertGreater(link.inertial.mass, 0.0)
                tensor = link.inertial.inertia
                self.assertTrue(np.all(np.linalg.eigvalsh(tensor) > 0))
                # Triangle inequality of principal moments (PhysX rejects
                # tensors that violate it).
                a, b, c = np.linalg.eigvalsh(tensor)
                self.assertLessEqual(c, a + b + 1e-12)

    def test_mesh_paths_are_relative_and_exist(self) -> None:
        refs = {m.attrib["filename"] for m in self.root.iter("mesh")}
        self.assertTrue(refs)
        for ref in refs:
            self.assertNotIn("://", ref)
            self.assertTrue((JELLY_SIM_URDF.parent / ref).is_file(), ref)

    def test_a_physics_engine_loads_it(self) -> None:
        xml = JELLY_SIM_URDF.read_text().replace(
            '<robot name="assembly">',
            '<robot name="assembly"><mujoco><compiler meshdir="%s" '
            'balanceinertia="false" discardvisual="true"/></mujoco>'
            % JELLY_SIM_URDF.parent,
            1,
        )
        model = mujoco.MjModel.from_xml_string(xml)
        # 21 driven joints + the three mimics (MuJoCo imports them as joints).
        self.assertEqual(model.nv, 24)


class CameraFramesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.models = {
            "arm": _load(urdf_path(AxolModel.MOBILE)),
            "whole_body": _load(urdf_path(AxolModel.MOBILE, whole_body=True)),
            "sim": _load(JELLY_SIM_URDF),
        }
        for urdf in cls.models.values():
            urdf.update_cfg({n: 0.0 for n in urdf.actuated_joint_names})

    def test_every_model_has_the_same_camera_frames(self) -> None:
        arm = self.models["arm"]
        for name, urdf in self.models.items():
            for cam in CAMERAS:
                with self.subTest(model=name, camera=cam):
                    joint = next(j for j in urdf.robot.joints if j.child == cam)
                    ref = next(j for j in arm.robot.joints if j.child == cam)
                    self.assertEqual(joint.parent, ref.parent)
                    np.testing.assert_allclose(joint.origin, ref.origin, atol=1e-9)

    def test_frames_are_right_handed_and_placed(self) -> None:
        sim = self.models["sim"]
        poses = {cam: sim.get_transform(cam) for cam in CAMERAS}
        for cam, pose in poses.items():
            rot = pose[:3, :3]
            np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-9)
            self.assertAlmostEqual(np.linalg.det(rot), 1.0, places=9)
        # Head: the right lens sits 50 mm along the left lens's +x.
        left, right = (
            poses["head_camera_left_optical"],
            poses["head_camera_right_optical"],
        )
        offset = np.linalg.inv(left) @ right
        np.testing.assert_allclose(offset[:3, 3], [0.05, 0.0, 0.0], atol=1e-4)
        # Both look forward and down; image-down has a downward component.
        self.assertGreater(left[0, 2], 0.0)
        self.assertLess(left[2, 2], 0.0)
        self.assertLess(left[2, 1], 0.0)
        # Wrists: each camera looks toward its own fingers.
        for side in ("left", "right"):
            cam = poses[f"{side}_wrist_camera_optical"]
            fingers = (
                sim.get_transform(f"{side}_finger_1")[:3, 3]
                + sim.get_transform(f"{side}_finger_2")[:3, 3]
            ) / 2
            self.assertGreater((fingers - cam[:3, 3]) @ cam[:3, 2], 0.0)


if __name__ == "__main__":
    unittest.main()
