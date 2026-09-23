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
    IMPEDANCE_LOOP_HZ,
    MIXED_LOOP_HZ,
    AxolConfig,
    check_loop_hz,
    impedance_joints,
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

    def test_position_puts_every_joint_on_its_vendors_loop_at_400_hz(self) -> None:
        cfg = AxolConfig(controller="position")
        self.assertEqual(cfg.loop_hz, 400.0)
        resolved = cfg.resolved()
        self.assertEqual(resolved.controller, "position")
        for arm in (resolved.left, resolved.right):
            for j in _MYACTUATOR:
                self.assertEqual(getattr(arm, j.value).wire_mode, "a4", j)
            for j in _DAMIAO:
                self.assertEqual(getattr(arm, j.value).wire_mode, "pv", j)
        # Idempotent, like the stiffness blend.
        self.assertEqual(resolved.resolved(), resolved)

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
        # A mixed arm runs the core at 480 Hz, impedance on alternate ticks.
        self.assertEqual(resolved.loop_hz, MIXED_LOOP_HZ)

    def test_the_rate_rule_counts_only_arm_joints_on_mit(self) -> None:
        self.assertEqual(IMPEDANCE_LOOP_HZ, 240.0)
        # The position controller puts every arm joint on a firmware loop;
        # the gripper, always MIT, does not count.
        self.assertEqual(impedance_joints(AxolConfig(controller="position")), [])
        check_loop_hz(AxolConfig(controller="position"), 400.0)
        check_loop_hz(AxolConfig(controller="position"), 240.0)
        check_loop_hz(AxolConfig(), 240.0)
        mit = impedance_joints(AxolConfig())
        self.assertEqual(len(mit), 2 * len(ARM_JOINTS))
        self.assertIn("right.shoulder_3", mit)
        with self.assertRaisesRegex(ValueError, "left.shoulder_1"):
            check_loop_hz(AxolConfig(), 400.0)
        check_loop_hz(AxolConfig(), MIXED_LOOP_HZ)
        self.assertEqual(MIXED_LOOP_HZ, 2 * IMPEDANCE_LOOP_HZ)

    def test_the_default_rate_follows_the_wire_mode_mix(self) -> None:
        self.assertEqual(AxolConfig().loop_hz, 240.0)
        self.assertEqual(AxolConfig(controller="position").loop_hz, 400.0)
        mixed = AxolConfig()
        mixed.right.shoulder_1.wire_mode = "a4"
        mixed.right.elbow.wire_mode = "a4"
        self.assertEqual(mixed.loop_hz, 480.0)
        check_loop_hz(mixed, mixed.loop_hz)

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

    def test_position_core_runs_at_400_with_a4_and_pv_tokens(self) -> None:
        rt = Axol._wrap(_hardware(AxolConfig(controller="position")))
        lines = rt._config_text().splitlines()
        self.assertIn("loop_hz 400.0", lines)
        tokens = self._joint_tokens(rt)
        for j in _MYACTUATOR:
            self.assertEqual(tokens[j.value], "a4")
        for j in _DAMIAO:
            self.assertEqual(tokens[j.value], "pv")

    def test_an_explicit_loop_rate_still_wins(self) -> None:
        rt = Axol._wrap(_hardware(AxolConfig(controller="position")), loop_hz=240.0)
        self.assertIn("loop_hz 240.0", rt._config_text().splitlines())

    def test_impedance_joints_are_held_to_240_hz(self) -> None:
        # The run that shook the arm: impedance with two joints on --a4, at
        # 400 Hz. Refused before any core starts.
        cfg = AxolConfig()
        cfg.right.shoulder_1.wire_mode = "a4"
        cfg.right.elbow.wire_mode = "a4"
        with self.assertRaisesRegex(ValueError, "impedance runs at 240 Hz only"):
            Axol._wrap(_hardware(cfg), loop_hz=400.0)
        # All-impedance at another rate is refused the same way.
        with self.assertRaisesRegex(ValueError, "240 Hz only"):
            Axol._wrap(_hardware(AxolConfig()), loop_hz=300.0)
        # At 240 the mixed split runs, and by default at 480 — every
        # impedance joint on alternate ticks, so still 240 Hz each.
        rt = Axol._wrap(_hardware(cfg), loop_hz=240.0)
        self.assertIn("loop_hz 240.0", rt._config_text().splitlines())
        rt = Axol._wrap(_hardware(cfg))
        self.assertIn("loop_hz 480.0", rt._config_text().splitlines())

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

    def test_hold_freezes_a_column_at_the_start_or_a_given_angle(self) -> None:
        import math

        import numpy as np

        holds = tune_motion._parse_holds(["right.elbow", "right.wrist_2=10"])
        elbow = tune_motion._COLUMNS.index("right.elbow")
        wrist_2 = tune_motion._COLUMNS.index("right.wrist_2")
        self.assertEqual(holds[elbow], None)
        self.assertAlmostEqual(holds[wrist_2], math.radians(10))
        rows = np.arange(3 * 14, dtype=float).reshape(3, 14)
        held = tune_motion._apply_holds(rows, holds, rows[0])
        np.testing.assert_array_equal(held[:, elbow], rows[0, elbow])
        np.testing.assert_allclose(held[:, wrist_2], math.radians(10))
        # Every other column still follows the motion; the input is untouched.
        other = [i for i in range(14) if i not in (elbow, wrist_2)]
        np.testing.assert_array_equal(held[:, other], rows[:, other])
        self.assertEqual(rows[1, elbow], 14 + elbow)
        self.assertEqual(self._parse("--hold", "right.elbow").hold, ["right.elbow"])

    def test_hold_refuses_what_it_cannot_do(self) -> None:
        for spec, message in {
            "elbow": "SIDE.JOINT",
            "right.hip": "SIDE.JOINT",
            "right.elbow=bent": "bad angle",
            "right.elbow=45": "outside",  # the right elbow is -150..0
        }.items():
            with self.subTest(spec=spec), self.assertRaisesRegex(SystemExit, message):
                tune_motion._parse_holds([spec])

    def test_an_override_touches_only_its_own_joint(self) -> None:
        # shoulder_1 and shoulder_2 share one firmware-gains object (and the
        # zero-friction joints one friction object) in the config; an
        # override on one joint must not reach the others, or the next config.
        cfg = AxolConfig()
        tune_motion._apply_gain_overrides(
            cfg,
            tune_motion._parse_gain_overrides(
                [
                    "right.shoulder_1.firmware.planner_accel=60000",
                    "right.elbow.friction.fc=0.3",
                    "right.shoulder_1.kd=2.0",
                ]
            ),
        )
        self.assertEqual(cfg.right.shoulder_1.firmware.planner_accel, 60000.0)
        self.assertEqual(cfg.right.shoulder_2.firmware.planner_accel, 0.0)
        self.assertEqual(cfg.left.shoulder_1.firmware.planner_accel, 0.0)
        self.assertEqual(cfg.right.elbow.friction.fc, 0.3)
        self.assertEqual(
            cfg.left.elbow.friction.fc, AxolConfig().left.elbow.friction.fc
        )
        self.assertEqual(
            cfg.right.wrist_2.friction.fc, AxolConfig().right.wrist_2.friction.fc
        )
        self.assertEqual(cfg.right.shoulder_1.kd, 2.0)
        self.assertEqual(AxolConfig().right.shoulder_1.firmware.planner_accel, 0.0)

    def test_planner_overrides_are_checked_before_the_bus(self) -> None:
        got = tune_motion._parse_gain_overrides(
            [
                "right.elbow.firmware.planner_accel=60000",
                "right.elbow.firmware.cap_track=1.2",
            ]
        )
        self.assertEqual(got[("right", "elbow", "firmware.planner_accel")], 60000.0)
        for spec, message in {
            "right.elbow.firmware.planner_accel=5000": "barely moves",
            "right.elbow.firmware.cap_track=0.5": "never keeps up",
            "right.wrist_2.firmware.planner_accel=60000": "Damiao wrist",
        }.items():
            with self.subTest(spec=spec), self.assertRaisesRegex(SystemExit, message):
                tune_motion._parse_gain_overrides([spec])

    def test_the_core_gets_each_joints_cap_track(self) -> None:
        cfg = AxolConfig()
        cfg.left.elbow.wire_mode = "a4"
        cfg.left.elbow.firmware.cap_track = 1.2
        with patch("almond_axol.rt.link.find_binary", return_value="/fake/axol-rt"):
            rt = Axol._wrap(_hardware(cfg))
        caps = {
            f[3]: f[25]
            for f in (ln.split() for ln in rt._config_text().splitlines())
            if f[0] == "joint"
        }
        self.assertEqual(caps["elbow"], "1.2")
        self.assertEqual(caps["shoulder_1"], "0.0")

    def test_repeat_defaults_to_one_pass(self) -> None:
        self.assertEqual(self._parse().repeat, 1)
        self.assertEqual(self._parse("--repeat", "5").repeat, 5)
        self.assertEqual(self._parse("--repeat", "0").repeat, 0)  # until Ctrl-C

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
