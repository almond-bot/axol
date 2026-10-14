"""The slow-motion gains apply only to joints with a slow-profile friction fit.

The stiffer #317 set (kp 450 / 500 / 200, wider host dampers, Stribeck
cancellation) oscillates on a robot still running the coded fallback
friction, so ``AxolConfig`` keeps the long-standing gains until
``tune.factory`` (or ``tune.friction --profile slow --save``) has written
that joint's own fit.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from almond_axol.robot import config as config_module
from almond_axol.robot.config import AxolConfig

SERIAL = "004800345542501420373234"

# What a slow-profile tune.factory run saves for one joint.
_SLOW_ENTRY: dict[str, Any] = {
    "friction": {"fc": 0.5, "k": 100.0, "fv": 0.0, "fo": 0.0, "fl": 0.05},
    "stribeck_gain": 0.8,
    "stribeck_dfs": 0.9,
    "stribeck_load_gain": 0.05,
    "stribeck_vs": 0.1,
}
# A standard-profile (or pre-#317) factory entry: friction, no Stribeck fit.
_STANDARD_ENTRY: dict[str, Any] = {
    "friction": {"fc": 0.9, "k": 800.0, "fv": 0.0, "fo": 0.1},
}


def _config(
    factory: dict[str, dict[str, Any]],
    local: dict[str, dict[str, Any]] | None = None,
) -> AxolConfig:
    with tempfile.TemporaryDirectory() as tmp:
        factory_path = Path(tmp) / "factory_calibration.json"
        factory_path.write_text("{}")
        with (
            mock.patch.object(
                config_module, "CALIBRATION_PATH", Path(tmp) / "none.json"
            ),
            mock.patch.object(config_module, "FACTORY_CALIBRATION_PATH", factory_path),
            mock.patch.object(config_module, "hub_serial", return_value=SERIAL),
            mock.patch.object(
                config_module,
                "load_factory_calibration",
                return_value={"left": {}, "right": {}, **factory},
            ),
            mock.patch.object(
                config_module,
                "load_calibration",
                return_value={"left": {}, "right": {}, **(local or {})},
            ),
        ):
            return AxolConfig()


def _uncalibrated() -> AxolConfig:
    with tempfile.TemporaryDirectory() as tmp:
        with (
            mock.patch.object(config_module, "CALIBRATION_PATH", Path(tmp) / "a.json"),
            mock.patch.object(
                config_module, "FACTORY_CALIBRATION_PATH", Path(tmp) / "b.json"
            ),
        ):
            return AxolConfig()


def _gains(cfg: AxolConfig, side: str, joint: str) -> tuple:
    jc = getattr(getattr(cfg, side), joint)
    return (
        jc.kp,
        jc.kd,
        jc.kd_host,
        jc.kd_host_hz,
        jc.kd_host_q,
        jc.stribeck_gain,
    )


class SlowFrictionGainsTest(unittest.TestCase):
    def test_uncalibrated_robot_runs_the_long_standing_gains(self) -> None:
        cfg = _uncalibrated()
        for side in ("left", "right"):
            self.assertEqual(
                _gains(cfg, side, "shoulder_1"), (250.0, 3.5, 40.0, None, 3.0, 0.0)
            )
            self.assertEqual(
                _gains(cfg, side, "shoulder_2"), (250.0, 3.5, 35.0, None, None, 0.0)
            )
            self.assertEqual(
                _gains(cfg, side, "elbow"), (130.0, 5.0, 0.0, None, None, 0.0)
            )

    def test_uncalibrated_robot_runs_the_reference_friction(self) -> None:
        # The jelly robot's slow-profile fits are never a fallback: no load
        # term, and the Stribeck curve stays at the JointConfig defaults.
        cfg = _uncalibrated()
        self.assertEqual(
            (cfg.left.shoulder_1.friction.fc, cfg.left.shoulder_1.friction.k),
            (0.9091, 799.59),
        )
        self.assertEqual(cfg.right.shoulder_2.friction.fc, 1.3950)
        for side in ("left", "right"):
            for joint in ("shoulder_1", "shoulder_2", "elbow", "wrist_3"):
                jc = getattr(getattr(cfg, side), joint)
                self.assertEqual(jc.friction.fl, 0.0, (side, joint))
                self.assertEqual(
                    (jc.stribeck_dfs, jc.stribeck_load_gain, jc.stribeck_vs),
                    (0.3, 0.1, 0.1),
                    (side, joint),
                )

    def test_factory_slow_fit_unlocks_the_slow_motion_gains(self) -> None:
        cfg = _config(
            {
                "left": {
                    "shoulder_1": _SLOW_ENTRY,
                    "shoulder_2": _SLOW_ENTRY,
                    "elbow": _SLOW_ENTRY,
                }
            }
        )
        self.assertEqual(
            _gains(cfg, "left", "shoulder_1"), (450.0, 5.0, 110.0, 2.6, 1.0, 0.8)
        )
        self.assertEqual(
            _gains(cfg, "left", "shoulder_2"), (500.0, 5.0, 70.0, None, None, 0.8)
        )
        self.assertEqual(
            _gains(cfg, "left", "elbow"), (200.0, 5.0, 10.0, 1.6, 1.5, 0.8)
        )
        self.assertEqual(cfg.left.shoulder_1.stribeck_pole, 40.0)
        self.assertEqual(cfg.left.shoulder_1.stribeck_dfs, 0.9)
        # The other arm was not calibrated: it keeps the long-standing set.
        self.assertEqual(cfg.right.shoulder_1.kp, 250.0)
        self.assertEqual(cfg.right.shoulder_1.stribeck_gain, 0.0)

    def test_gate_is_per_joint(self) -> None:
        cfg = _config({"right": {"shoulder_1": _SLOW_ENTRY}})
        self.assertEqual(cfg.right.shoulder_1.kp, 450.0)
        self.assertEqual(cfg.right.shoulder_2.kp, 250.0)
        self.assertEqual(cfg.right.elbow.kp, 130.0)

    def test_local_slow_fit_counts_too(self) -> None:
        # tune.friction --profile slow --save writes only the local file.
        cfg = _config({}, local={"right": {"elbow": _SLOW_ENTRY}})
        self.assertEqual(cfg.right.elbow.kp, 200.0)

    def test_friction_without_a_stribeck_fit_keeps_the_long_standing_gains(
        self,
    ) -> None:
        cfg = _config({"left": {"shoulder_1": _STANDARD_ENTRY}})
        self.assertEqual(
            _gains(cfg, "left", "shoulder_1"), (250.0, 3.5, 40.0, None, 3.0, 0.0)
        )
        self.assertEqual(cfg.left.shoulder_1.friction.fc, 0.9)

    def test_gains_saved_in_the_entry_still_win(self) -> None:
        entry = {**_SLOW_ENTRY, "kp": 380.0, "stribeck_gain": 0.4}
        cfg = _config({}, local={"left": {"shoulder_1": entry}})
        self.assertEqual(cfg.left.shoulder_1.kp, 380.0)
        self.assertEqual(cfg.left.shoulder_1.stribeck_gain, 0.4)
        # The rest of the slow-motion set still comes along.
        self.assertEqual(cfg.left.shoulder_1.kd_host, 110.0)


if __name__ == "__main__":
    unittest.main()
