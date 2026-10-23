"""The generated URDFs as full robot models (physics simulators: Isaac Sim,
MuJoCo).

Each generated version is one file that IK, the viewers and simulators all
load (see ``almond_axol.constants.urdf_path``). This pins what makes it usable
in a physics simulator: the right joints move, the arm chain is the one IK
solves, masses are AxolConfig's where they are known and physically valid
everywhere, mesh paths resolve without a ROS package, and the camera optical
frames sit where the cameras are.
"""

from __future__ import annotations

import unittest
from xml.etree import ElementTree

import mujoco
import numpy as np
import yourdfpy

from almond_axol.constants import (
    ARM_JOINTS,
    AxolModel,
    urdf_arm_joint_names,
    urdf_body_name,
    urdf_path,
)
from almond_axol.kinematics.model import ik_urdf
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


_PARALLEL_JOINTS = {"left_finger_1_joint", "right_finger_1_joint"}
_PARALLEL_MIMICS = {
    "left_finger_2_joint": "left_finger_1_joint",
    "right_finger_2_joint": "right_finger_1_joint",
}


class JellyFullUrdfTest(unittest.TestCase):
    # Per version: the model, its body joints (and their mimics), its root,
    # its gripper joints (and mimics), the links wrist_3's tuned mass is
    # split across, and MuJoCo's dof count (mimics import as joints).
    model = AxolModel.MOBILE
    gripper = "parallel"
    body_joints = {
        "base_x",
        "base_y",
        "base_yaw",
        "lift",
        "wheel_1",
        "wheel_2",
        "wheel_3",
        "wheel_4",
    }
    body_mimics = {"lift_2": "lift"}
    root_link = "world"
    gripper_joints = _PARALLEL_JOINTS
    gripper_mimics = _PARALLEL_MIMICS
    wrist_parts = ("w2", "gripper", "finger_1", "finger_2", "wrist_camera")
    # 14 arms + 8 body + 2 fingers driven, + the lift's and fingers' mimics.
    dofs = 27

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = urdf_path(cls.model, gripper=cls.gripper)
        cls.sim = _load(cls.path)
        cls.arm = ik_urdf(cls.path)
        cls.root = ElementTree.parse(cls.path).getroot()
        cls.zero = {n: 0.0 for n in cls.sim.actuated_joint_names}

    def test_moving_joints(self) -> None:
        self.assertEqual(
            set(self.sim.actuated_joint_names),
            set(ARM_JOINT_NAMES) | self.body_joints | self.gripper_joints,
        )
        mimics = {
            j.name: j.mimic.joint for j in self.sim.robot.joints if j.mimic is not None
        }
        self.assertEqual(mimics, {**self.body_mimics, **self.gripper_mimics})
        self.assertEqual(self.sim.base_link, self.root_link)

    def test_at_zero_the_torso_is_the_world_origin(self) -> None:
        # So IK's frame (and every recorded pose) is the classic one.
        self.sim.update_cfg(self.zero)
        np.testing.assert_allclose(
            self.sim.get_transform("root", self.root_link), np.eye(4), atol=1e-9
        )

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
        limit = self.sim.joint_map["left_finger_1_joint"].limit
        stroke = max((limit.lower, limit.upper), key=abs)
        self.assertGreater(abs(stroke), 0.03)
        self.sim.update_cfg(self.zero)
        closed = [
            self.sim.get_transform(f"left_finger_{i}")[:3, 3].copy() for i in (1, 2)
        ]
        self.sim.update_cfg({**self.zero, "left_finger_1_joint": stroke})
        moved = [
            self.sim.get_transform(f"left_finger_{i}")[:3, 3] - c
            for i, c in zip((1, 2), closed)
        ]
        for step in moved:
            self.assertAlmostEqual(float(np.linalg.norm(step)), abs(stroke), places=6)
        self.assertAlmostEqual(float(moved[0] @ moved[1]), -(stroke**2), places=6)

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
            parts = [f"{prefix}_{part}" for part in self.wrist_parts]
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
            self.assertTrue((self.path.parent / ref).is_file(), ref)

    def test_joint_friction_and_drive_gains_are_axol_configs(self) -> None:
        from almond_axol.constants import urdf_joint_name

        cfg = AxolConfig().resolved()
        joints = {j.name: j for j in self.sim.robot.joints}
        drives = {j.attrib["name"]: j.find("drive") for j in self.root.findall("joint")}
        self.assertEqual(
            {name for name, d in drives.items() if d is not None}, set(ARM_JOINT_NAMES)
        )
        for arm, side in ((cfg.left, True), (cfg.right, False)):
            for joint in ARM_JOINTS:
                name = urdf_joint_name(joint, is_left=side)
                jc = getattr(arm, joint.value)
                with self.subTest(joint=name):
                    dyn = joints[name].dynamics
                    self.assertAlmostEqual(
                        float(dyn.friction), jc.friction.fc, places=5
                    )
                    self.assertAlmostEqual(float(dyn.damping), jc.friction.fv, places=5)
                    drive = drives[name].attrib
                    self.assertAlmostEqual(float(drive["stiffness"]), jc.kp, places=5)
                    self.assertAlmostEqual(float(drive["damping"]), jc.kd, places=5)

    def test_a_physics_engine_loads_it(self) -> None:
        xml = self.path.read_text().replace(
            '<robot name="assembly">',
            '<robot name="assembly"><mujoco><compiler meshdir="%s" '
            'balanceinertia="false" discardvisual="true"/></mujoco>' % self.path.parent,
            1,
        )
        model = mujoco.MjModel.from_xml_string(xml)
        self.assertEqual(model.nv, self.dofs)
        # ...and reads the joints' measured friction.
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_s1_0")
        self.assertGreater(model.dof_frictionloss[model.jnt_dofadr[jid]], 0.5)


class ParcelFullUrdfTest(JellyFullUrdfTest):
    """Jelly with the parcel gripper: a hinged plate per gripper."""

    gripper = "parcel"
    gripper_joints = {"left_plate_joint", "right_plate_joint"}
    gripper_mimics: dict[str, str] = {}
    wrist_parts = ("w2", "gripper", "plate")
    # 14 arms + 8 body + 2 plates driven, + the lift's mimic.
    dofs = 25

    def test_fingers_open_together(self) -> None:
        self.skipTest("the parcel gripper has a plate, not fingers")

    def test_plate_swings_half_a_turn_about_its_hinge(self) -> None:
        for side, (lo, hi) in (("left", (-np.pi, 0.0)), ("right", (0.0, np.pi))):
            joint = self.sim.joint_map[f"{side}_plate_joint"]
            self.assertEqual(joint.type, "revolute")
            self.assertAlmostEqual(float(joint.limit.lower), lo, places=4)
            self.assertAlmostEqual(float(joint.limit.upper), hi, places=4)
            # The plate turns about its own origin, on the gripper.
            self.sim.update_cfg(self.zero)
            rest = self.sim.get_transform(f"{side}_plate", f"{side}_gripper")
            self.sim.update_cfg({**self.zero, joint.name: (lo + hi) / 2})
            turned = self.sim.get_transform(f"{side}_plate", f"{side}_gripper")
            np.testing.assert_allclose(turned[:3, 3], rest[:3, 3], atol=1e-9)
            angle = np.arccos(
                np.clip((np.trace(rest[:3, :3].T @ turned[:3, :3]) - 1) / 2, -1, 1)
            )
            self.assertAlmostEqual(float(angle), np.pi / 2, places=6)


class OxCartFullUrdfTest(JellyFullUrdfTest):
    """Axol on the Ox cart: the cart's one-stage lift."""

    model = AxolModel.OX_CART
    body_joints = {"lift"}
    body_mimics: dict[str, str] = {}
    # 14 arms + the lift + 2 fingers driven, + the fingers' mimics.
    dofs = 19

    def test_the_lift_lowers_the_torso(self) -> None:
        self.sim.update_cfg({**self.zero, "lift": 0.2})
        np.testing.assert_allclose(
            self.sim.get_transform("root", "world")[:3, 3], [0, 0, -0.2], atol=1e-9
        )


class OwlFullUrdfTest(JellyFullUrdfTest):
    """Axol on the Owl mount: only the arms and fingers move."""

    model = AxolModel.OWL
    body_joints: set[str] = set()
    body_mimics: dict[str, str] = {}
    root_link = "root"
    dofs = 18


class StandaloneFullUrdfTest(OwlFullUrdfTest):
    model = AxolModel.STANDALONE


class CameraFramesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.models = {
            (m.value, "parallel"): _load(urdf_path(m))
            for m in AxolModel
            if m is not AxolModel.CLASSIC
        }
        # The parcel gripper's model: the same head, no wrist cameras.
        cls.parcel = _load(urdf_path(AxolModel.MOBILE, gripper="parcel"))
        for urdf in (*cls.models.values(), cls.parcel):
            urdf.update_cfg({n: 0.0 for n in urdf.actuated_joint_names})

    def _frame(self, urdf: yourdfpy.URDF, cam: str):
        return next(j for j in urdf.robot.joints if j.child == cam)

    def test_every_version_has_the_same_camera_frames(self) -> None:
        # The exports' CAD poses differ by up to ~20 µm / µrad of rounding.
        ref = self.models[("mobile", "parallel")]
        for key, urdf in self.models.items():
            for cam in CAMERAS:
                with self.subTest(model=key, camera=cam):
                    joint, other = self._frame(urdf, cam), self._frame(ref, cam)
                    self.assertEqual(joint.parent, other.parent)
                    np.testing.assert_allclose(joint.origin, other.origin, atol=5e-5)

    def test_parcel_model_has_the_head_cameras_only(self) -> None:
        frames = {ln for ln in self.parcel.link_map if ln.endswith("_optical")}
        self.assertEqual(frames, set(CAMERAS[2:]))

    def test_frames_are_right_handed_and_placed(self) -> None:
        sim = self.models[("mobile", "parallel")]
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
            # ...with the gripper at the bottom of the image, as in real
            # wrist frames (+y is image down).
            self.assertGreater((fingers - cam[:3, 3]) @ cam[:3, 1], 0.0)


if __name__ == "__main__":
    unittest.main()
