from __future__ import annotations

import unittest
from xml.etree import ElementTree

import numpy as np
import yourdfpy

from almond_axol.constants import URDF_PATH, urdf_arm_joint_names


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


if __name__ == "__main__":
    unittest.main()
