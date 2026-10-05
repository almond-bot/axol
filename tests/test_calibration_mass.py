"""Link mass in the per-robot calibration (custom end-effectors).

Covers the ``mass`` field end to end: sanitized on load, written by
``update_joint_calibration``, overlaid by ``AxolConfig``, set from
``tune.factory --mass`` / ``--com``, and merged field-wise into the cloud
document so a custom gripper's gravity numbers follow the robot.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from almond_axol.cli.tune.factory import merge_cloud_document, parse_link_overrides
from almond_axol.robot import config as config_module
from almond_axol.robot.calibration import load_calibration, update_joint_calibration
from almond_axol.robot.config import AxolConfig

SERIAL = "004800345542501420373234"


class CalibrationMassFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "calibration.json"

    def _write(self, left: dict) -> None:
        self.path.write_text(
            json.dumps({"version": 1, "hub_serial": SERIAL, "left": left})
        )

    def test_load_keeps_positive_mass(self) -> None:
        self._write({"wrist_3": {"mass": 1.1, "com": [0.0, 0.0, -0.1]}})
        cal = load_calibration(self.path, expected_hub_serial=SERIAL)
        self.assertEqual(cal["left"]["wrist_3"]["mass"], 1.1)
        self.assertEqual(cal["left"]["wrist_3"]["com"], [0.0, 0.0, -0.1])

    def test_load_drops_invalid_mass_but_keeps_the_rest(self) -> None:
        for bad in (0, -0.5, "1.1", True, None, float("nan")):
            with self.subTest(mass=bad):
                self._write({"wrist_3": {"mass": bad, "com": [0.0, 0.0, -0.1]}})
                with self.assertLogs("almond_axol.robot.calibration", "WARNING"):
                    cal = load_calibration(self.path, expected_hub_serial=SERIAL)
                entry = cal["left"]["wrist_3"]
                self.assertNotIn("mass", entry)
                self.assertIn("com", entry)

    def test_update_merges_mass_without_clobbering(self) -> None:
        update_joint_calibration(
            "left",
            "wrist_3",
            friction={"fc": 0.1, "k": 900.0, "fv": 0.0, "fo": 0.0},
            hub_serial=SERIAL,
            path=self.path,
        )
        update_joint_calibration(
            "left", "wrist_3", mass=1.1, hub_serial=SERIAL, path=self.path
        )
        entry = json.loads(self.path.read_text())["left"]["wrist_3"]
        self.assertEqual(entry["mass"], 1.1)
        self.assertEqual(entry["friction"]["fc"], 0.1)

    def test_update_rejects_non_positive_mass(self) -> None:
        for bad in (0.0, -1.0, float("inf")):
            with self.subTest(mass=bad), self.assertRaises(ValueError):
                update_joint_calibration(
                    "left", "wrist_3", mass=bad, hub_serial=SERIAL, path=self.path
                )
        self.assertFalse(self.path.exists())


class AxolConfigMassOverlayTest(unittest.TestCase):
    def test_factory_mass_and_com_reach_the_gravity_model(self) -> None:
        factory = {
            "left": {"wrist_3": {"mass": 1.1, "com": [-0.03, 0.0, -0.14]}},
            "right": {"wrist_3": {"mass": 1.2}},
        }
        stock = AxolConfig(
            left=config_module.ArmConfig(),
            right=config_module.ArmConfig().mirror_to_right(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            factory_path = Path(tmp) / "factory_calibration.json"
            factory_path.write_text("{}")
            with (
                mock.patch.object(
                    config_module, "CALIBRATION_PATH", Path(tmp) / "none.json"
                ),
                mock.patch.object(
                    config_module, "FACTORY_CALIBRATION_PATH", factory_path
                ),
                mock.patch.object(config_module, "hub_serial", return_value=SERIAL),
                mock.patch.object(
                    config_module,
                    "load_factory_calibration",
                    return_value=factory,
                ),
                mock.patch.object(
                    config_module,
                    "load_calibration",
                    return_value={"left": {}, "right": {}},
                ),
            ):
                cfg = AxolConfig()
        self.assertEqual(cfg.left.wrist_3.mass, 1.1)
        self.assertEqual(cfg.left.wrist_3.com, (-0.03, 0.0, -0.14))
        self.assertEqual(cfg.right.wrist_3.mass, 1.2)
        # Mass alone leaves the (mirrored) CoM at its default.
        self.assertEqual(cfg.right.wrist_3.com, stock.right.wrist_3.com)
        self.assertEqual(cfg.left.elbow.mass, stock.left.elbow.mass)


class FactoryLinkOverridesTest(unittest.TestCase):
    def test_unsided_mass_applies_to_every_calibrated_arm(self) -> None:
        out = parse_link_overrides(["wrist_3=1.1"], [], ["left", "right"])
        self.assertEqual(out["left"]["wrist_3"], {"mass": 1.1})
        self.assertEqual(out["right"]["wrist_3"], {"mass": 1.1})

    def test_sided_mass_and_com_combine(self) -> None:
        out = parse_link_overrides(
            ["left.wrist_3=1.1"], ["left.wrist_3=-0.03,0,-0.14"], ["left", "right"]
        )
        self.assertEqual(
            out, {"left": {"wrist_3": {"mass": 1.1, "com": (-0.03, 0.0, -0.14)}}}
        )

    def test_rejects_bad_specs(self) -> None:
        cases = [
            (["wrist_3"], []),  # no value
            (["wrist_9=1.0"], []),  # unknown joint
            (["middle.wrist_3=1.0"], []),  # unknown side
            (["wrist_3=heavy"], []),  # not a number
            (["wrist_3=0"], []),  # non-positive
            (["right.wrist_3=1.0"], []),  # arm not being calibrated
            ([], ["wrist_3=0,0,0"]),  # com needs a side
            ([], ["left.wrist_3=0,0"]),  # com needs 3 components
        ]
        for mass, com in cases:
            with self.subTest(mass=mass, com=com), self.assertRaises(ValueError):
                parse_link_overrides(mass, com, ["left"])


class MergeCloudDocumentTest(unittest.TestCase):
    def test_merges_per_field_and_keeps_other_arm(self) -> None:
        existing = {
            "version": 1,
            "hub_serial": SERIAL,
            "left": {
                "wrist_3": {"mass": 1.1, "com": [0.0, 0.0, -0.14], "friction": {}},
                "elbow": {"com": [0.0, 0.0, -0.07]},
            },
            "right": {"wrist_3": {"mass": 1.2}},
        }
        # This run's wrist_3 gravity fit was rejected: only friction is new.
        document = {"left": {"wrist_3": {"friction": {"fc": 0.1}}}}
        merged = merge_cloud_document(existing, document, SERIAL)
        self.assertEqual(
            merged["left"]["wrist_3"],
            {"mass": 1.1, "com": [0.0, 0.0, -0.14], "friction": {"fc": 0.1}},
        )
        self.assertEqual(merged["left"]["elbow"], {"com": [0.0, 0.0, -0.07]})
        self.assertEqual(merged["right"], {"wrist_3": {"mass": 1.2}})
        self.assertEqual(merged["hub_serial"], SERIAL)


if __name__ == "__main__":
    unittest.main()
