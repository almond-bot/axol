"""The record of the motor models a hub's arms were last seen with, and the
X8-P20-150 shoulder defaults the config takes from it."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from almond_axol.constants import Joint
from almond_axol.robot import config as config_module
from almond_axol.robot import motor_models
from almond_axol.robot.config import AxolConfig
from almond_axol.robot.motor_models import (
    X8_150,
    load_motor_models,
    note_motor_models,
    record_motor_models,
    recorded_shoulder_variant,
    shoulder_variant,
)

SERIAL = "E4B3F34DF452C4E6"
OTHER = "44BDED79596A78C1"
M150 = "RMD-X8-P20-150"
M120 = "RMD-X8-P20-120"


class ShoulderVariantTest(unittest.TestCase):
    def test_only_a_motor_that_says_150_is_one(self) -> None:
        self.assertEqual(shoulder_variant(M150), X8_150)
        self.assertEqual(shoulder_variant(" rmd-x8-p20-150 "), X8_150)
        for model in (M120, "RMD-X8-P20", "RMD-X6-P20-60", "RMD-X6-P20-50", "", None):
            self.assertIsNone(shoulder_variant(model), model)


class RecordTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "motor_models.json"

    def test_round_trip_per_side(self) -> None:
        self.assertTrue(
            record_motor_models(
                "left", {"shoulder_1": M150}, hub_serial=SERIAL, path=self.path
            )
        )
        self.assertTrue(
            record_motor_models(
                "right", {"shoulder_1": M120}, hub_serial=SERIAL, path=self.path
            )
        )
        got = load_motor_models(self.path, expected_hub_serial=SERIAL)
        self.assertEqual(got["left"], {"shoulder_1": M150})
        self.assertEqual(got["right"], {"shoulder_1": M120})

    def test_same_models_again_change_nothing(self) -> None:
        models = {"shoulder_1": M150, "shoulder_2": M150}
        record_motor_models("left", models, hub_serial=SERIAL, path=self.path)
        before = self.path.read_text()
        self.assertFalse(
            record_motor_models("left", models, hub_serial=SERIAL, path=self.path)
        )
        self.assertEqual(self.path.read_text(), before)

    def test_a_joint_that_did_not_answer_keeps_its_model(self) -> None:
        record_motor_models(
            "left", {"shoulder_1": M150}, hub_serial=SERIAL, path=self.path
        )
        record_motor_models(
            "left",
            {"shoulder_1": None, "shoulder_2": M150},
            hub_serial=SERIAL,
            path=self.path,
        )
        got = load_motor_models(self.path, expected_hub_serial=SERIAL)["left"]
        self.assertEqual(got, {"shoulder_1": M150, "shoulder_2": M150})

    def test_a_swapped_arm_replaces_what_was_recorded(self) -> None:
        record_motor_models(
            "left", {"shoulder_1": M120}, hub_serial=SERIAL, path=self.path
        )
        self.assertTrue(
            record_motor_models(
                "left", {"shoulder_1": M150}, hub_serial=SERIAL, path=self.path
            )
        )
        got = load_motor_models(self.path, expected_hub_serial=SERIAL)["left"]
        self.assertEqual(got["shoulder_1"], M150)

    def test_another_hubs_record_reads_empty_and_is_replaced(self) -> None:
        record_motor_models(
            "left", {"shoulder_1": M150}, hub_serial=OTHER, path=self.path
        )
        empty = load_motor_models(self.path, expected_hub_serial=SERIAL)
        self.assertEqual(empty, {"left": {}, "right": {}})
        record_motor_models(
            "right", {"shoulder_1": M120}, hub_serial=SERIAL, path=self.path
        )
        raw = json.loads(self.path.read_text())
        self.assertEqual(raw["hub_serial"], SERIAL)
        self.assertEqual(raw["left"], {})

    def test_nothing_is_recorded_without_a_hub(self) -> None:
        with mock.patch.object(motor_models, "current_hub_serial", return_value=None):
            self.assertFalse(
                record_motor_models("left", {"shoulder_1": M150}, path=self.path)
            )
        self.assertFalse(self.path.exists())

    def test_a_corrupt_record_reads_empty(self) -> None:
        self.path.write_text("{not json")
        got = load_motor_models(self.path, expected_hub_serial=SERIAL)
        self.assertEqual(got, {"left": {}, "right": {}})


def _motor(model: Any) -> SimpleNamespace:
    return SimpleNamespace(_driver=SimpleNamespace(_model=model))


class NoteAtBringUpTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "motor_models.json"
        for name, value in (
            ("MOTOR_MODELS_PATH", self.path),
            ("current_hub_serial", lambda: SERIAL),
        ):
            patcher = mock.patch.object(motor_models, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _arm(self, s1: Any, s2: Any) -> dict[Joint, SimpleNamespace]:
        return {
            Joint.SHOULDER_1: _motor(s1),
            Joint.SHOULDER_2: _motor(s2),
            # A Damiao wrist: its driver reads no model.
            Joint.WRIST_2: SimpleNamespace(_driver=SimpleNamespace()),
        }

    def test_records_the_models_the_drivers_read(self) -> None:
        note_motor_models("left", self._arm(M150, M150))
        got = load_motor_models(self.path, expected_hub_serial=SERIAL)["left"]
        self.assertEqual(got, {"shoulder_1": M150, "shoulder_2": M150})
        self.assertEqual(recorded_shoulder_variant("left", "shoulder_2"), X8_150)
        self.assertIsNone(recorded_shoulder_variant("right", "shoulder_2"))

    def test_says_so_when_a_shoulder_is_not_what_the_session_assumed(self) -> None:
        with self.assertLogs(motor_models._logger, "WARNING") as logs:
            note_motor_models("left", self._arm(M150, M120))
        text = "\n".join(logs.output)
        self.assertIn("shoulder_1 is an RMD-X8-P20-150", text)
        self.assertNotIn("shoulder_2", text)

    def test_is_quiet_when_nothing_changed(self) -> None:
        note_motor_models("left", self._arm(M150, M150))
        with self.assertNoLogs(motor_models._logger, "WARNING"):
            note_motor_models("left", self._arm(M150, M150))

    def test_a_model_that_is_not_text_is_ignored(self) -> None:
        note_motor_models("left", self._arm(mock.MagicMock(), None))
        self.assertFalse(self.path.exists())

    def test_never_raises(self) -> None:
        with (
            mock.patch.object(
                motor_models, "record_motor_models", side_effect=OSError("disk")
            ),
            self.assertLogs(motor_models._logger, "WARNING"),
        ):
            note_motor_models("left", self._arm(M150, M150))


# What a slow-profile tune.friction run saves for one joint.
_SLOW_ENTRY: dict[str, Any] = {
    "friction": {"fc": 0.5, "k": 100.0, "fv": 0.0, "fo": 0.1, "fl": 0.05},
    "stribeck_gain": 0.0,
    "stribeck_dfs": 0.6,
    "stribeck_load_gain": 0.03,
    "stribeck_vs": 0.1,
}


def _config(
    models: dict[str, dict[str, str]],
    local: dict[str, dict[str, Any]] | None = None,
) -> AxolConfig:
    with tempfile.TemporaryDirectory() as tmp:
        record = Path(tmp) / "motor_models.json"
        record.write_text("{}")
        with (
            mock.patch.object(config_module, "CALIBRATION_PATH", Path(tmp) / "a.json"),
            mock.patch.object(
                config_module, "FACTORY_CALIBRATION_PATH", Path(tmp) / "b.json"
            ),
            mock.patch.object(config_module, "MOTOR_MODELS_PATH", record),
            mock.patch.object(config_module, "hub_serial", return_value=SERIAL),
            mock.patch.object(
                config_module,
                "load_motor_models",
                return_value={"left": {}, "right": {}, **models},
            ),
            mock.patch.object(
                config_module,
                "load_factory_calibration",
                return_value={"left": {}, "right": {}},
            ),
            mock.patch.object(
                config_module,
                "load_calibration",
                return_value={"left": {}, "right": {}, **(local or {})},
            ),
        ):
            return AxolConfig()


_BOTH_150 = {"shoulder_1": M150, "shoulder_2": M150}


class X8150DefaultsTest(unittest.TestCase):
    def test_150_shoulders_run_the_soft_damper_uncalibrated(self) -> None:
        cfg = _config({"left": _BOTH_150, "right": _BOTH_150})
        for arm in (cfg.left, cfg.right):
            for jc in (arm.shoulder_1, arm.shoulder_2):
                self.assertEqual((jc.kp, jc.kd), (250.0, 3.5))
                self.assertEqual((jc.kd_host, jc.kd_host_q), (10.0, 3.0))
                self.assertIsNone(jc.kd_host_hz)
                self.assertEqual(jc.stribeck_gain, 0.0)

    def test_150_shoulders_run_their_own_fallback_friction(self) -> None:
        cfg = _config({"left": _BOTH_150, "right": _BOTH_150})
        plain = _config({})
        for joint in ("shoulder_1", "shoulder_2"):
            want = config_module._X8_150_FRICTION[joint]
            for arm in (cfg.left, cfg.right):
                got = getattr(arm, joint).friction
                self.assertEqual(got, want)
                self.assertIsNot(got, want)  # a copy: mutating it must not leak
            self.assertGreater(want.fc, 0.0)
            self.assertGreater(want.fl, 0.0)
            self.assertNotEqual(getattr(plain.left, joint).friction, want)

    def test_every_other_joint_is_untouched(self) -> None:
        cfg = _config({"left": _BOTH_150, "right": _BOTH_150})
        plain = _config({})
        for joint in ("shoulder_3", "elbow", "wrist_1", "wrist_2", "wrist_3"):
            self.assertEqual(getattr(cfg.left, joint), getattr(plain.left, joint))
            self.assertEqual(getattr(cfg.right, joint), getattr(plain.right, joint))

    def test_x6_50_joints_run_the_x6_60_settings(self) -> None:
        models = {
            "shoulder_3": "RMD-X6-P20-50",
            "elbow": "RMD-X6-P20-50",
            "wrist_1": "RMD-X6-P20-50",
        }
        cfg = _config({"left": models, "right": models})
        plain = _config({})
        self.assertEqual(cfg.left, plain.left)
        self.assertEqual(cfg.right, plain.right)

    def test_the_variant_is_per_joint_and_per_arm(self) -> None:
        cfg = _config({"left": {"shoulder_2": M150, "shoulder_1": M120}})
        plain = _config({})
        self.assertEqual(cfg.left.shoulder_2.kd_host, 10.0)
        self.assertEqual(cfg.left.shoulder_1, plain.left.shoulder_1)
        self.assertEqual(cfg.right.shoulder_2, plain.right.shoulder_2)

    def test_120_and_unknown_shoulders_keep_the_long_standing_set(self) -> None:
        plain = _config({})
        for models in (
            {"left": {"shoulder_1": M120, "shoulder_2": M120}},
            {"left": {"shoulder_1": "RMD-X8-P20", "shoulder_2": ""}},
        ):
            cfg = _config(models)
            self.assertEqual(cfg.left, plain.left)
        self.assertEqual(
            (plain.left.shoulder_1.kd_host, plain.left.shoulder_2.kd_host), (40.0, 35.0)
        )

    def test_calibration_keeps_the_soft_set_and_brings_the_joints_own_friction(
        self,
    ) -> None:
        cfg = _config(
            {"left": _BOTH_150},
            local={"left": {"shoulder_1": _SLOW_ENTRY, "shoulder_2": _SLOW_ENTRY}},
        )
        for jc in (cfg.left.shoulder_1, cfg.left.shoulder_2):
            # Not the X8-P20-120 slow-motion set (kp 450 / 500, wide dampers).
            self.assertEqual((jc.kp, jc.kd, jc.kd_host), (250.0, 3.5, 10.0))
            self.assertEqual(jc.kd_host_q, 3.0)
            self.assertIsNone(jc.kd_host_hz)
            self.assertEqual(jc.stribeck_gain, 0.0)
            self.assertEqual((jc.friction.fc, jc.friction.fl), (0.5, 0.05))
            self.assertEqual(jc.stribeck_dfs, 0.6)

    def test_a_calibrated_120_shoulder_still_gets_the_slow_motion_set(self) -> None:
        entry = {**_SLOW_ENTRY, "stribeck_gain": 0.8}
        cfg = _config({}, local={"left": {"shoulder_2": entry}})
        self.assertEqual(cfg.left.shoulder_2.kp, 500.0)
        self.assertEqual(cfg.left.shoulder_2.kd_host, 70.0)

    def test_values_saved_in_the_entry_still_win(self) -> None:
        entry = {**_SLOW_ENTRY, "kp": 300.0, "kd_host": 20.0, "stribeck_gain": 0.4}
        cfg = _config({"left": _BOTH_150}, local={"left": {"shoulder_2": entry}})
        jc = cfg.left.shoulder_2
        self.assertEqual((jc.kp, jc.kd_host, jc.stribeck_gain), (300.0, 20.0, 0.4))

    def test_the_elbow_calibrates_as_on_any_robot(self) -> None:
        entry = {**_SLOW_ENTRY, "stribeck_gain": 0.8}
        cfg = _config({"left": _BOTH_150}, local={"left": {"elbow": entry}})
        self.assertEqual(cfg.left.elbow.kp, 200.0)
        self.assertEqual(cfg.left.elbow.stribeck_gain, 0.8)


class StribeckDefaultTest(unittest.TestCase):
    """What ``tune.friction`` / ``tune.factory`` save as the Stribeck gain."""

    def _gain(
        self, joint: Joint, variant: str | None, asked: float | None = None
    ) -> float:
        from almond_axol.cli.tune import friction

        args = argparse.Namespace(stribeck_gain=asked)
        with mock.patch.object(
            friction, "recorded_shoulder_variant", return_value=variant
        ):
            return friction._stribeck_gain(args, joint, "left")

    def test_150_shoulders_save_it_off(self) -> None:
        self.assertEqual(self._gain(Joint.SHOULDER_1, X8_150), 0.0)
        self.assertEqual(self._gain(Joint.SHOULDER_2, X8_150), 0.0)

    def test_120_shoulders_keep_their_default(self) -> None:
        self.assertEqual(self._gain(Joint.SHOULDER_1, None), 0.8)
        self.assertEqual(self._gain(Joint.SHOULDER_2, None), 0.8)

    def test_only_the_shoulders_follow_the_motor(self) -> None:
        self.assertEqual(self._gain(Joint.ELBOW, X8_150), 0.8)
        self.assertEqual(self._gain(Joint.WRIST_1, X8_150), 0.0)

    def test_a_gain_asked_for_wins(self) -> None:
        self.assertEqual(self._gain(Joint.SHOULDER_2, X8_150, asked=0.4), 0.4)


if __name__ == "__main__":
    unittest.main()
