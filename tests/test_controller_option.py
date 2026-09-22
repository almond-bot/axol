"""``AxolConfig.controller``: impedance (MIT, 240 Hz) or the firmware position
loops (a4 / pv, 400 Hz) — baked into the per-joint wire modes at
construction, the core's loop rate and wire tokens derived from it."""

from __future__ import annotations

import argparse
import unittest
from typing import Any
from unittest.mock import patch

from almond_axol.cli.tune import motion as tune_motion
from almond_axol.constants import ARM_JOINTS, Joint
from almond_axol.robot.axol import AxolHardware
from almond_axol.robot.config import (
    CONTROLLER_LOOP_HZ,
    CONTROLLERS,
    AxolConfig,
    position_wire_mode,
)
from almond_axol.rt import Axol
from almond_axol.serve.introspect import _KNOWN_OPTIONS

_MYACTUATOR = (
    Joint.SHOULDER_1,
    Joint.SHOULDER_2,
    Joint.SHOULDER_3,
    Joint.ELBOW,
    Joint.WRIST_1,
)
_DAMIAO = (Joint.WRIST_2, Joint.WRIST_3)


class ConfigTest(unittest.TestCase):
    def test_impedance_is_the_default_and_leaves_wire_modes_alone(self) -> None:
        cfg = AxolConfig()
        self.assertEqual(cfg.controller, "impedance")
        self.assertEqual(cfg.loop_hz, 240.0)
        resolved = cfg.resolved()
        for arm in (resolved.left, resolved.right):
            for j in ARM_JOINTS:
                self.assertEqual(getattr(arm, j.value).wire_mode, "mit")

    def test_position_puts_the_myactuator_joints_on_a4_at_400_hz(self) -> None:
        cfg = AxolConfig(controller="position")
        self.assertEqual(cfg.loop_hz, 400.0)
        resolved = cfg.resolved()
        self.assertEqual(resolved.controller, "position")
        for arm in (resolved.left, resolved.right):
            for j in _MYACTUATOR:
                self.assertEqual(getattr(arm, j.value).wire_mode, "a4", j)
            # The Damiao wrists keep the impedance frame (their pv loop
            # stick-slips at creep and pumps the arm's 4 Hz sway).
            for j in _DAMIAO:
                self.assertEqual(getattr(arm, j.value).wire_mode, "mit", j)
        # Idempotent, like the stiffness blend.
        self.assertEqual(resolved.resolved(), resolved)

    def test_a_wrist_opted_into_pv_survives_the_position_controller(self) -> None:
        cfg = AxolConfig(controller="position")
        cfg.right.wrist_2.wire_mode = "pv"
        resolved = cfg.resolved()
        self.assertEqual(resolved.right.wrist_2.wire_mode, "pv")
        self.assertEqual(resolved.right.wrist_3.wire_mode, "mit")
        self.assertEqual(resolved.right.elbow.wire_mode, "a4")

    def test_position_wire_mode_follows_the_motor_vendor(self) -> None:
        for j in _MYACTUATOR:
            self.assertEqual(position_wire_mode(j), "a4")
        for j in _DAMIAO:
            self.assertEqual(position_wire_mode(j), "pv")

    def test_an_explicit_a4_joint_survives_under_impedance(self) -> None:
        cfg = AxolConfig()
        cfg.right.shoulder_1.wire_mode = "a4"
        resolved = cfg.resolved()
        self.assertEqual(resolved.right.shoulder_1.wire_mode, "a4")
        self.assertEqual(resolved.right.shoulder_2.wire_mode, "mit")
        self.assertEqual(resolved.loop_hz, 240.0)

    def test_unknown_controller_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            AxolConfig(controller="velocity").resolved()

    def test_rates_are_pinned(self) -> None:
        self.assertEqual(CONTROLLERS, ("impedance", "position"))
        self.assertEqual(CONTROLLER_LOOP_HZ, {"impedance": 240.0, "position": 400.0})
        # The dashboard's draccus forms render the field as a select.
        self.assertEqual(_KNOWN_OPTIONS["controller"], list(CONTROLLERS))


class _FakeBus:
    def __init__(self, channel: str) -> None:
        self._channel = channel

    async def close(self) -> None:
        return None


class _FakeDriver:
    kp_max = 500.0
    kd_max = 5.0

    def __init__(self, *_a: Any, **_k: Any) -> None:
        pass

    def set_feedback_callback(self, _cb: Any) -> None:
        pass


def _hardware(config: AxolConfig) -> AxolHardware:
    with (
        patch("almond_axol.robot.axol.CanBus", _FakeBus),
        patch(
            "almond_axol.motor.motor.make_driver",
            side_effect=lambda *_a, **_k: _FakeDriver(),
        ),
    ):
        return AxolHardware(config=config, left_channel="can0", right_channel=None)


class RealtimeConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        # Only the config text is exercised; no core is spawned, so CI needs
        # no built axol-rt binary.
        patcher = patch("almond_axol.rt.link.find_binary", return_value="/fake/axol-rt")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _joint_tokens(self, rt: Axol) -> dict[str, str]:
        out = {}
        for line in rt._config_text().splitlines():
            f = line.split()
            if f and f[0] == "joint":
                out[f[3]] = f[18]
        return out

    def test_impedance_core_runs_at_240_on_mit(self) -> None:
        rt = Axol._wrap(_hardware(AxolConfig()))
        lines = rt._config_text().splitlines()
        self.assertIn("loop_hz 240.0", lines)
        self.assertEqual(set(self._joint_tokens(rt).values()), {"mit"})

    def test_position_core_runs_at_400_with_a4_tokens_and_mit_wrists(self) -> None:
        rt = Axol._wrap(_hardware(AxolConfig(controller="position")))
        lines = rt._config_text().splitlines()
        self.assertIn("loop_hz 400.0", lines)
        tokens = self._joint_tokens(rt)
        for j in _MYACTUATOR:
            self.assertEqual(tokens[j.value], "a4")
        for j in _DAMIAO:
            self.assertEqual(tokens[j.value], "mit")

    def test_an_explicit_loop_rate_still_wins(self) -> None:
        rt = Axol._wrap(_hardware(AxolConfig(controller="position")), loop_hz=240.0)
        self.assertIn("loop_hz 240.0", rt._config_text().splitlines())

    def test_a_vendor_mismatched_wire_mode_is_refused(self) -> None:
        cfg = AxolConfig()
        cfg.left.wrist_2.wire_mode = "a4"  # a Damiao wrist has no 0xA4
        rt = Axol._wrap(_hardware(cfg))
        with self.assertRaisesRegex(ValueError, "wrist_2.*Damiao|not a MyActuator"):
            rt._config_text()
        cfg = AxolConfig()
        cfg.left.elbow.wire_mode = "pv"  # a MyActuator joint has no pos-vel frame
        rt = Axol._wrap(_hardware(cfg))
        with self.assertRaisesRegex(ValueError, "elbow"):
            rt._config_text()


class TuneMotionFlagTest(unittest.TestCase):
    def _parse(self, *argv: str) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        tune_motion.add_parser(sub)
        return parser.parse_args(["tune.motion", "--motion", "slow_osc", *argv])

    def test_loop_hz_override_and_firmware_gain_overrides_parse(self) -> None:
        ns = self._parse(
            "--loop-hz", "240", "--gain", "right.elbow.firmware.speed_kp=0.03"
        )
        self.assertEqual(ns.loop_hz, 240.0)
        overrides = tune_motion._parse_gain_overrides(ns.gain)
        self.assertEqual(overrides, {("right", "elbow", "firmware.speed_kp"): 0.03})
        both = tune_motion._parse_gain_overrides(["wrist_2.firmware.profile_acc=200"])
        self.assertEqual(
            both,
            {
                ("left", "wrist_2", "firmware.profile_acc"): 200.0,
                ("right", "wrist_2", "firmware.profile_acc"): 200.0,
            },
        )
        with self.assertRaises(SystemExit):
            tune_motion._parse_gain_overrides(["elbow.firmware.bogus=1"])

    def test_controller_flag_takes_the_two_laws_and_defaults_to_config(self) -> None:
        self.assertIsNone(self._parse().controller)
        self.assertEqual(self._parse("--controller", "position").controller, "position")
        self.assertEqual(
            self._parse("--controller", "impedance").controller, "impedance"
        )
        with self.assertRaises(SystemExit):
            self._parse("--controller", "velocity")


if __name__ == "__main__":
    unittest.main()
