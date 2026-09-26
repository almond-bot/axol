"""Axol version selection: inferred from Jelly, and each version's collision model."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from almond_axol.constants import (
    URDF_PATH,
    AxolModel,
    torso_links,
    urdf_path,
)
from almond_axol.serve.commands import build_argv
from almond_axol.serve.settings import SettingsStore
from almond_axol.settings import resolve_robot_model


class JellyInfersRobotModelTest(unittest.TestCase):
    """Enabling Jelly (attached and not switched off) means the mobile Axol."""

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.store = SettingsStore(Path(self._dir.name) / "settings.json", strict=True)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def _resolve(self, attached: bool, **kwargs: object) -> AxolModel:
        with patch("almond_axol.robot.jelly._iface_exists", return_value=attached):
            return resolve_robot_model(store=self.store, **kwargs)

    def test_attached_jelly_is_the_mobile_axol(self) -> None:
        self.assertIs(self._resolve(True), AxolModel.MOBILE)
        self.assertIs(self._resolve(False), AxolModel.CLASSIC)

    def test_switched_off_jelly_is_the_classic_axol(self) -> None:
        self.store.update(values={"jelly.wheels": False, "jelly.lift": False})
        self.assertIs(self._resolve(True), AxolModel.CLASSIC)
        # Either half enabled is enough.
        self.store.update(values={"jelly.lift": True})
        self.assertIs(self._resolve(True), AxolModel.MOBILE)

    def test_a_session_decision_skips_detection(self) -> None:
        self.assertIs(self._resolve(False, jelly_enabled=True), AxolModel.MOBILE)
        self.assertIs(self._resolve(True, jelly_enabled=False), AxolModel.CLASSIC)

    def test_explicit_model_wins_and_is_validated(self) -> None:
        self.assertIs(self._resolve(True, model="classic"), AxolModel.CLASSIC)
        self.assertIs(self._resolve(False, model="Mobile"), AxolModel.MOBILE)
        self.assertIs(resolve_robot_model(AxolModel.CLASSIC), AxolModel.CLASSIC)
        with self.assertRaises(ValueError):
            resolve_robot_model("hover")

    def test_explicit_setting_reaches_the_ik_configs(self) -> None:
        # kinematics.robot_model is an ordinary (Advanced) kinematics leaf.
        self.store.update(values={"kinematics.robot_model": "classic"})
        teleop = build_argv("teleop", self.store.merged_args("teleop", {}))
        self.assertEqual(
            teleop[teleop.index("--kinematics.robot_model") + 1], "classic"
        )
        collect = self.store.merged_args("collect-data", {})
        self.assertEqual(
            collect["teleop_config.kinematics_config.robot_model"], "classic"
        )
        self.assertIs(self._resolve(True), AxolModel.CLASSIC)


class UrdfSelectionTest(unittest.TestCase):
    def test_each_version_has_its_own_urdf(self) -> None:
        self.assertEqual(urdf_path(), URDF_PATH)
        self.assertEqual(urdf_path("classic"), URDF_PATH)
        self.assertEqual(urdf_path(AxolModel.MOBILE).name, "axol_mobile.urdf")
        self.assertTrue(urdf_path(AxolModel.MOBILE).is_file())

    def test_collision_is_arm_against_body_on_both_versions(self) -> None:
        from almond_axol.kinematics.model import shared_robot_collision

        for model in AxolModel:
            with self.subTest(model=model.value):
                rc = shared_robot_collision(model)
                torso = set(torso_links(model))
                self.assertTrue(torso <= set(rc.link_names))
                pairs = {
                    frozenset((rc.link_names[int(i)], rc.link_names[int(j)]))
                    for i, j in zip(rc.active_idx_i, rc.active_idx_j)
                }
                arms = {
                    n
                    for n in rc.link_names
                    if n.startswith(("left_", "right_"))
                    and not n.endswith(("_s2", "_s3"))
                }
                expected = {frozenset((t, a)) for t in torso for a in arms}
                self.assertEqual(pairs, expected)


def _pair_distances(model: AxolModel, q_left, q_right) -> dict[tuple[str, str], float]:
    import jax.numpy as jnp
    import numpy as np

    from almond_axol.constants import urdf_arm_joint_names
    from almond_axol.kinematics.model import shared_robot, shared_robot_collision

    robot, rc = shared_robot(model), shared_robot_collision(model)
    names = list(robot.joints.actuated_names)
    q = np.zeros(len(names), dtype=np.float32)
    joints = urdf_arm_joint_names(is_left=True) + urdf_arm_joint_names(is_left=False)
    for name, value in zip(joints, list(q_left) + list(q_right)):
        q[names.index(name)] = value
    d = np.asarray(rc.compute_self_collision_distance(robot, jnp.asarray(q)))
    return {
        (rc.link_names[int(i)], rc.link_names[int(j)]): float(d[k])
        for k, (i, j) in enumerate(zip(rc.active_idx_i, rc.active_idx_j))
    }


class CapsuleDistanceTest(unittest.TestCase):
    def test_parallel_segments_pair_their_closest_points(self) -> None:
        import jax.numpy as jnp
        import numpy as np
        from pyroki.collision import _utils

        from almond_axol.kinematics import model

        # Two vertical segments 0.3 m apart, pointing opposite ways and
        # overlapping in height: the true gap is exactly 0.3 m.
        a1, b1 = jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 0.34])
        a2, b2 = jnp.array([0.3, 0.0, 0.3]), jnp.array([0.3, 0.0, -0.7])
        c1, c2 = model._closest_segment_to_segment_points(a1, b1, a2, b2)
        self.assertAlmostEqual(float(jnp.linalg.norm(c1 - c2)), 0.3, places=6)
        # And it is what pyroki's capsule pairs now call.
        self.assertIs(
            _utils.closest_segment_to_segment_points,
            model._closest_segment_to_segment_points,
        )

        # The stock almond-pyroki routine still gets this wrong; when this
        # fails, the fork has the fix and model._patch_pyroki can go.
        import importlib.util

        spec = importlib.util.spec_from_file_location("stock_utils", _utils.__file__)
        stock = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stock)
        s1, s2 = stock.closest_segment_to_segment_points(a1, b1, a2, b2)
        self.assertGreater(float(np.linalg.norm(s1 - s2)), 0.31)

    def test_both_arms_read_the_same_clearance(self) -> None:
        # The body is symmetric, so mirrored arms must be too; the parallel
        # bug read the right upper arm 165 mm further from the lift column.
        for model in AxolModel:
            with self.subTest(model=model.value):
                d = _pair_distances(model, [0.0] * 7, [0.0] * 7)
                for (a, b), value in d.items():
                    mirrored = (
                        a.replace("left_", "right_"),
                        b.replace("left_", "right_"),
                    )
                    if "left_" in a + b:
                        # Within CAD slop: mirrored parts differ by ~0.5 mm.
                        self.assertAlmostEqual(
                            d[mirrored], value, delta=1e-3, msg=(a, b)
                        )

    def test_upper_arm_guard_allows_the_rest_pose(self) -> None:
        # The solver hard-stops the upper arm at upper_arm_guard_floor
        # (KinematicsSolver.__init__); the default rest pose must clear it.
        from almond_axol.kinematics.model import upper_arm_guard_floor
        from almond_axol.teleop.config import VRTeleopConfig

        cfg = VRTeleopConfig()
        for model in AxolModel:
            with self.subTest(model=model.value):
                home = _pair_distances(model, [0.0] * 7, [0.0] * 7)
                rest = _pair_distances(model, cfg.rest_pose_left, cfg.rest_pose_right)
                torso = torso_links(model)
                for (a, b), value in rest.items():
                    arm = b if a in torso else a
                    if arm.endswith("_e1"):
                        floor = upper_arm_guard_floor(home[(a, b)])
                        self.assertGreaterEqual(value, floor, (a, b))
                        if model is AxolModel.CLASSIC:
                            self.assertAlmostEqual(floor, home[(a, b)] - 0.020)


class FloorFrameTest(unittest.TestCase):
    def test_mobile_wheels_stand_on_the_floor_frame(self) -> None:
        import trimesh
        import yourdfpy

        path = urdf_path(AxolModel.MOBILE)
        urdf = yourdfpy.URDF.load(str(path), mesh_dir=str(path.parent))
        floor_z = urdf.get_transform("floor")[2, 3]
        lowest = min(
            trimesh.load(
                path.parent
                / v.geometry.mesh.filename.removeprefix("package://assembly/")
            )
            .apply_transform(urdf.get_transform(name) @ v.origin)
            .bounds[0, 2]
            for name, link in urdf.link_map.items()
            for v in link.visuals
        )
        self.assertAlmostEqual(lowest, floor_z, delta=0.002)
        # The world frame is still the classic one: the floor is below it.
        self.assertLess(floor_z, -0.4)


if __name__ == "__main__":
    unittest.main()
