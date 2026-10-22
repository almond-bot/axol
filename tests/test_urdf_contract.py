from __future__ import annotations

import unittest
from xml.etree import ElementTree

import numpy as np
import yourdfpy

from almond_axol.constants import (
    ARM_JOINTS,
    URDF_PATH,
    AxolModel,
    Joint,
    torso_links,
    urdf_arm_body_names,
    urdf_arm_joint_names,
    urdf_body_name,
    urdf_path,
)
from almond_axol.kinematics.model import ik_urdf, ik_urdf_xml


class UrdfContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ElementTree.parse(URDF_PATH).getroot()
        self.urdf = yourdfpy.URDF.load(str(URDF_PATH), mesh_dir=str(URDF_PATH.parent))

    def test_classic_assembly_has_only_arm_actuators(self) -> None:
        self.assertEqual(self.root.tag, "robot")
        self.assertEqual(self.root.attrib.get("name"), "assembly")

        movable = {
            joint.attrib["name"]
            for joint in self.root.findall("joint")
            if joint.attrib.get("type") != "fixed"
        }
        expected = set(
            urdf_arm_joint_names(is_left=True) + urdf_arm_joint_names(is_left=False)
        )
        self.assertEqual(movable, expected)
        self.assertEqual(len(movable), 14)
        self.assertEqual(set(self.urdf.actuated_joint_names), expected)

        parsed_joints = {joint.name: joint.type for joint in self.urdf.robot.joints}
        self.assertEqual(parsed_joints["left_gripper_0"], "fixed")
        self.assertEqual(parsed_joints["right_gripper_0"], "fixed")

    def test_classic_zero_pose_preserves_cartesian_frame(self) -> None:
        base = self.urdf.get_transform("base")
        left = self.urdf.get_transform("left_gripper")
        right = self.urdf.get_transform("right_gripper")

        np.testing.assert_allclose(base[:3, 3], [0.0, 0.0, 0.86], atol=1e-8)
        np.testing.assert_allclose(
            base[:3, :3],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            atol=1e-8,
        )
        np.testing.assert_allclose(left[:3, 3], [0.0, 0.19958, 0.148054], atol=1e-7)
        np.testing.assert_allclose(right[:3, 3], [0.0, -0.19958, 0.148054], atol=1e-7)
        expected_rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        np.testing.assert_allclose(left[:3, :3], expected_rotation, atol=1e-5)
        np.testing.assert_allclose(right[:3, :3], expected_rotation, atol=1e-5)

    def test_classic_mesh_set_is_complete(self) -> None:
        prefix = "package://assembly/"
        references = {
            mesh.attrib["filename"]
            for mesh in self.root.iter("mesh")
            if "filename" in mesh.attrib
        }
        self.assertTrue(references)
        self.assertTrue(all(reference.startswith(prefix) for reference in references))

        relative_paths = {reference.removeprefix(prefix) for reference in references}
        expected_meshes = {
            "meshes/Base.stl",
            "meshes/Left_E1.stl",
            "meshes/Left_E2.stl",
            "meshes/Left_Gripper.stl",
            "meshes/Left_S2.stl",
            "meshes/Left_S3.stl",
            "meshes/Left_W0.stl",
            "meshes/Left_W1.stl",
            "meshes/Left_W2.stl",
            "meshes/Right_E1.stl",
            "meshes/Right_E2.stl",
            "meshes/Right_Gripper.stl",
            "meshes/Right_S2.stl",
            "meshes/Right_S3.stl",
            "meshes/Right_W0.stl",
            "meshes/Right_W1.stl",
            "meshes/Right_W2.stl",
            "meshes/S1.stl",
        }
        self.assertEqual(relative_paths, expected_meshes)
        self.assertTrue(
            all((URDF_PATH.parent / relative).is_file() for relative in relative_paths)
        )

        packaged_meshes = {
            path.relative_to(URDF_PATH.parent).as_posix()
            for path in (URDF_PATH.parent / "meshes").glob("*.stl")
        }
        self.assertEqual(packaged_meshes, expected_meshes)


def _arm_bodies() -> list[str]:
    grippers = [urdf_body_name(Joint.GRIPPER, is_left=side) for side in (True, False)]
    return (
        urdf_arm_body_names(is_left=True)
        + urdf_arm_body_names(is_left=False)
        + grippers
    )


def _element_attrs(el: ElementTree.Element | None) -> dict[str, str]:
    return {} if el is None else dict(el.attrib)


class MobileUrdfContractTest(unittest.TestCase):
    """The generated URDFs are the classic arms on a different body.

    Everything downstream of the arm chain — FK, recorded Cartesian frames,
    gravity compensation and its tuned mass/CoM parameters — assumes every
    version's arms are the same, so this pins that each generated file only
    differs where it should: the elbow limits, the meshes, and the body —
    both as IK sees it (:func:`ik_urdf`) and as the full model.
    """

    model = AxolModel.MOBILE
    gripper = "parallel"

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = urdf_path(cls.model, gripper=cls.gripper)
        cls.root = ElementTree.parse(cls.path).getroot()
        cls.classic = ElementTree.parse(URDF_PATH).getroot()
        # What arm IK loads: every joint but the arms' frozen.
        cls.urdf = ik_urdf(cls.path)
        cls.ik_root = ElementTree.fromstring(ik_urdf_xml(cls.path))

    def _joints(self, root: ElementTree.Element) -> dict[str, ElementTree.Element]:
        return {j.attrib["name"]: j for j in root.findall("joint")}

    def test_arm_ik_moves_only_the_arm_joints(self) -> None:
        expected = set(
            urdf_arm_joint_names(is_left=True) + urdf_arm_joint_names(is_left=False)
        )
        self.assertEqual(set(self.urdf.actuated_joint_names), expected)

    def test_arm_chain_matches_classic_except_elbow_limits(self) -> None:
        mobile, classic = self._joints(self.root), self._joints(self.classic)
        for name, joint in classic.items():
            with self.subTest(joint=name):
                other = mobile[name]
                for tag in ("origin", "axis", "parent", "child"):
                    self.assertEqual(
                        _element_attrs(other.find(tag)), _element_attrs(joint.find(tag))
                    )
                lim, other_lim = joint.find("limit"), other.find("limit")
                if lim is None:
                    self.assertIsNone(other_lim)
                    continue
                lo, hi = float(lim.attrib["lower"]), float(lim.attrib["upper"])
                new_lo = float(other_lim.attrib["lower"])
                new_hi = float(other_lim.attrib["upper"])
                if name == "left_e1_0":
                    self.assertEqual((new_lo, new_hi), (-0.4189, hi))
                elif name == "right_e1_0":
                    self.assertEqual((new_lo, new_hi), (lo, 0.4189))
                else:
                    self.assertEqual((new_lo, new_hi), (lo, hi))

    def test_its_masses_give_the_tuned_gravity_torques(self) -> None:
        # The full model's own masses (AxolConfig's, with wrist_3's lump
        # split across the gripper parts) weigh the arms exactly as gravity
        # compensation's tuned model does.
        import mujoco

        from almond_axol.robot import gravity

        tuned = gravity.GravityCompensator()
        # fusestatic off: MuJoCo's fusing of fixed-jointed links misplaces
        # their CoMs, which would shift wrist_3's by ~3 mm.
        text = gravity._load_urdf_text(self.path).replace(
            '<robot name="assembly">',
            '<robot name="assembly"><mujoco><compiler fusestatic="false" '
            'balanceinertia="false"/></mujoco>',
            1,
        )
        model = mujoco.MjModel.from_xml_string(text)
        data = mujoco.MjData(model)
        rng = np.random.default_rng(0)
        for _ in range(10):
            q = rng.uniform(-1.5, 1.5, len(ARM_JOINTS))
            for is_left in (True, False):
                data.qpos[:] = 0.0
                data.qvel[:] = 0.0
                adr = [
                    model.jnt_dofadr[
                        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    ]
                    for name in urdf_arm_joint_names(is_left=is_left)
                ]
                for i, a in enumerate(adr):
                    data.qpos[model.jnt_qposadr[model.dof_jntid[a]]] = q[i]
                mujoco.mj_forward(model, data)
                np.testing.assert_allclose(
                    data.qfrc_bias[adr],
                    tuned.gravity_arm(q, is_left=is_left),
                    atol=1e-4,
                )

    def test_zero_pose_keeps_the_classic_grippers(self) -> None:
        classic = yourdfpy.URDF.load(str(URDF_PATH), mesh_dir=str(URDF_PATH.parent))
        for link in ("left_gripper", "right_gripper", "s1"):
            np.testing.assert_allclose(
                self.urdf.get_transform(link), classic.get_transform(link), atol=1e-9
            )

    def test_ik_collides_the_body_and_arms_only(self) -> None:
        links = {ln.attrib["name"]: ln for ln in self.ik_root.findall("link")}
        whole_body = self.model is AxolModel.MOBILE
        torso = set(torso_links(self.model, whole_body=whole_body))
        for link in sorted(torso) + _arm_bodies():
            with self.subTest(link=link):
                self.assertIsNotNone(links[link].find("collision"))
        # Every other link is visual or a frame only for IK: a torso link
        # without geometry would get a point capsule from pyroki. (The Jelly
        # deck and the Ox cart's caster frame are out of reach with the lift
        # raised.)
        for name, link in links.items():
            if not name.startswith(("left_", "right_")) and name not in torso:
                with self.subTest(link=name):
                    self.assertIsNone(link.find("collision"))
        self.assertIsNone(self.ik_root.find("link/ik_collision"))

    def test_every_part_has_contact_geometry_and_mass(self) -> None:
        # The full model, for physics: every link with a visual collides
        # and has a mass.
        for link in self.root.findall("link"):
            if link.find("visual") is None:
                continue
            with self.subTest(link=link.attrib["name"]):
                self.assertIsNotNone(link.find("collision"))
                self.assertGreater(float(link.find("inertial/mass").attrib["value"]), 0)

    def test_meshes_are_relative_and_exist(self) -> None:
        for mesh in self.root.iter("mesh"):
            ref = mesh.attrib["filename"]
            self.assertTrue(ref.startswith(_GENERATED_DIRS), ref)
            self.assertTrue((self.path.parent / ref).is_file(), ref)


class ParcelUrdfContractTest(MobileUrdfContractTest):
    """The mobile Axol with the parcel gripper: the same arms on the same body."""

    gripper = "parcel"

    def test_gripper_is_the_parcel_gripper(self) -> None:
        links = {ln.attrib["name"] for ln in self.root.findall("link")}
        for side in ("left", "right"):
            self.assertIn(f"{side}_plate", links)
            self.assertNotIn(f"{side}_finger_1", links)
            # No wrist camera on the parcel gripper.
            self.assertNotIn(f"{side}_wrist_camera", links)
            self.assertNotIn(f"{side}_wrist_camera_optical", links)

    def test_plate_sweep_is_one_capsule(self) -> None:
        # IK cannot see the plate's angle: its collision body covers the
        # whole sweep, fixed to the gripper.
        links = {ln.attrib["name"]: ln for ln in self.ik_root.findall("link")}
        for side in ("left", "right"):
            (col,) = links[f"{side}_fingers"].findall("collision")
            self.assertIsNotNone(col.find("geometry/cylinder"))
            joint = next(
                j
                for j in self.ik_root.findall("joint")
                if j.find("child").attrib["link"] == f"{side}_fingers"
            )
            self.assertEqual(joint.find("parent").attrib["link"], f"{side}_gripper")
            # The plate itself is frozen and collides only through the sweep.
            self.assertIsNone(links[f"{side}_plate"].find("collision"))


class StandaloneUrdfContractTest(MobileUrdfContractTest):
    """Axol standalone: the same arms under nothing but the shoulder bar."""

    model = AxolModel.STANDALONE


class OxCartUrdfContractTest(MobileUrdfContractTest):
    """Axol on the Ox cart: the same arms on its lift column."""

    model = AxolModel.OX_CART


class OwlUrdfContractTest(MobileUrdfContractTest):
    """Axol on the Owl mount: the same arms on its column and foot."""

    model = AxolModel.OWL


# Where scripts/build_urdfs.py writes meshes.
_GENERATED_DIRS = tuple(
    f"meshes/{d}/" for d in ("jelly", "jelly_parcel", "standalone", "owl", "ox_cart")
)


class GeneratedMeshesTest(unittest.TestCase):
    def test_one_urdf_per_version(self) -> None:
        # Legacy, standalone, Ox cart, Owl and Jelly — plus Jelly with the
        # parcel gripper — and nothing else.
        self.assertEqual(
            {p.name for p in URDF_PATH.parent.glob("*.urdf")},
            {urdf_path(m).name for m in AxolModel}
            | {urdf_path(AxolModel.MOBILE, gripper="parcel").name},
        )

    def test_every_generated_mesh_is_used(self) -> None:
        # Exports share meshes across directories (the body and arms are the
        # same in every export): together the generated URDFs use every mesh
        # the generator packaged, and nothing else.
        urdfs = [p for p in URDF_PATH.parent.glob("axol_*.urdf")]
        references = {
            mesh.attrib["filename"]
            for path in urdfs
            for mesh in ElementTree.parse(path).getroot().iter("mesh")
        }
        packaged = {
            p.relative_to(URDF_PATH.parent).as_posix()
            for d in _GENERATED_DIRS
            for p in (URDF_PATH.parent / d).glob("*.stl")
        }
        self.assertEqual(references, packaged)


if __name__ == "__main__":
    unittest.main()
