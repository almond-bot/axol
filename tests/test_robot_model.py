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


if __name__ == "__main__":
    unittest.main()
