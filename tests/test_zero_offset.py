"""Per-joint ``zero_offset`` trim from the per-robot calibration.

Covers the field end to end: sanitized and bounded on load, written by
``update_joint_calibration``, overlaid onto ``JointConfig``, and folded into
``AxolArm``'s motor→joint offsets for both fixed-stop and either-stop joints,
so reported positions and commanded targets agree on the trimmed frame.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock
from unittest.mock import AsyncMock, patch

import numpy as np

from almond_axol.motor import Joint
from almond_axol.robot import axol as axol_module
from almond_axol.robot import config as config_module
from almond_axol.robot.axol import AxolHardware, closer_end_stop
from almond_axol.robot.calibration import (
    MAX_ZERO_OFFSET_RAD,
    load_calibration,
    update_joint_calibration,
)
from almond_axol.robot.config import AxolConfig

SERIAL = "004800345542501420373234"
_IDX = {j: i for i, j in enumerate(Joint)}


class ZeroOffsetFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "calibration.json"

    def _write(self, left: dict) -> None:
        self.path.write_text(
            json.dumps({"version": 1, "hub_serial": SERIAL, "left": left})
        )

    def test_load_keeps_a_valid_offset(self) -> None:
        self._write({"elbow": {"zero_offset": -0.0123, "kp": 40.0}})
        cal = load_calibration(self.path, expected_hub_serial=SERIAL)
        self.assertEqual(cal["left"]["elbow"], {"zero_offset": -0.0123, "kp": 40.0})

    def test_load_drops_invalid_offsets_but_keeps_the_rest(self) -> None:
        too_big = MAX_ZERO_OFFSET_RAD + 1e-3
        for bad in (too_big, -too_big, float("nan"), "0.01", True, None):
            with self.subTest(zero_offset=bad):
                self._write({"elbow": {"zero_offset": bad, "kp": 40.0}})
                with self.assertLogs("almond_axol.robot.calibration", "WARNING"):
                    cal = load_calibration(self.path, expected_hub_serial=SERIAL)
                self.assertEqual(cal["left"]["elbow"], {"kp": 40.0})

    def test_update_merges_offset_without_clobbering(self) -> None:
        update_joint_calibration(
            "left", "elbow", kp=40.0, hub_serial=SERIAL, path=self.path
        )
        update_joint_calibration(
            "left", "elbow", zero_offset=0.01, hub_serial=SERIAL, path=self.path
        )
        entry = json.loads(self.path.read_text())["left"]["elbow"]
        self.assertEqual(entry["zero_offset"], 0.01)
        self.assertEqual(entry["kp"], 40.0)

    def test_update_rejects_out_of_range_offsets(self) -> None:
        for bad in (math.radians(6.0), -math.radians(6.0), float("nan")):
            with self.subTest(zero_offset=bad), self.assertRaises(ValueError):
                update_joint_calibration(
                    "left",
                    "elbow",
                    zero_offset=bad,
                    hub_serial=SERIAL,
                    path=self.path,
                )
        self.assertFalse(self.path.exists())


class ZeroOffsetOverlayTest(unittest.TestCase):
    def test_local_offset_overrides_the_factory_one(self) -> None:
        factory = {
            "left": {"elbow": {"zero_offset": 0.02}, "wrist_1": {"zero_offset": 0.01}},
            "right": {},
        }
        local = {"left": {"elbow": {"zero_offset": -0.005}}, "right": {}}
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
                    config_module, "load_factory_calibration", return_value=factory
                ),
                mock.patch.object(
                    config_module, "load_calibration", return_value=local
                ),
            ):
                cfg = AxolConfig()
        self.assertEqual(cfg.left.elbow.zero_offset, -0.005)
        self.assertEqual(cfg.left.wrist_1.zero_offset, 0.01)
        self.assertEqual(cfg.left.shoulder_1.zero_offset, 0.0)
        self.assertEqual(cfg.right.elbow.zero_offset, 0.0)


def _trimmed_config(trims: dict[Joint, float]) -> AxolConfig:
    cfg = AxolConfig()
    left = cfg.left
    for joint, trim in trims.items():
        left = replace(
            left, **{joint.value: replace(getattr(left, joint.value), zero_offset=trim)}
        )
    return replace(cfg, left=left)


class ArmZeroOffsetTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.enterContext(patch.object(axol_module, "CanBus"))

    def _arm(self, trims: dict[Joint, float]):
        hardware = AxolHardware(
            _trimmed_config(trims), left_channel="can0", right_channel=None
        )
        assert hardware.left is not None
        return hardware.left

    async def test_offsets_carry_the_trim_after_resolution(self) -> None:
        trims = {Joint.ELBOW: 0.01, Joint.WRIST_2: -0.02}
        arm = self._arm(trims)
        # Fixed-stop joints carry it from construction; either-stop joints
        # are unresolved (NaN) until their first encoder reading.
        self.assertAlmostEqual(
            arm._joint_offsets[_IDX[Joint.ELBOW]],
            closer_end_stop(Joint.ELBOW, True)[0] + 0.01,
        )
        self.assertTrue(math.isnan(arm._joint_offsets[_IDX[Joint.WRIST_2]]))

        stop = 1.5
        arm._detect_stop_side = AsyncMock(return_value=(stop, 0.3))
        arm._verify_fixed_stop_zero = AsyncMock(return_value=0.0)
        await arm.resolve_joint_offsets()

        self.assertAlmostEqual(
            arm._joint_offsets[_IDX[Joint.ELBOW]],
            closer_end_stop(Joint.ELBOW, True)[0] + 0.01,
        )
        self.assertAlmostEqual(arm._joint_offsets[_IDX[Joint.WRIST_2]], stop - 0.02)
        # Untrimmed joints and the gripper are unaffected.
        self.assertAlmostEqual(
            arm._joint_offsets[_IDX[Joint.SHOULDER_1]],
            closer_end_stop(Joint.SHOULDER_1, True)[0],
        )
        self.assertEqual(arm._joint_offsets[_IDX[Joint.GRIPPER]], 0.0)

    async def test_wrap_correction_keeps_the_trim(self) -> None:
        arm = self._arm({Joint.ELBOW: 0.01})
        arm._detect_stop_side = AsyncMock(return_value=(1.5, 0.3))
        arm._verify_fixed_stop_zero = AsyncMock(return_value=2 * math.pi)
        await arm.resolve_joint_offsets()
        self.assertAlmostEqual(
            arm._joint_offsets[_IDX[Joint.ELBOW]],
            closer_end_stop(Joint.ELBOW, True)[0] + 2 * math.pi + 0.01,
        )

    async def test_reported_positions_include_the_trim(self) -> None:
        trimmed = self._arm({Joint.ELBOW: 0.01})
        stock = self._arm({})
        for arm in (trimmed, stock):
            arm._detect_stop_side = AsyncMock(return_value=(1.5, 0.3))
            arm._verify_fixed_stop_zero = AsyncMock(return_value=0.0)
            await arm.resolve_joint_offsets()
            for motor in arm.motors.values():
                motor._position = 0.2
        diff = trimmed.positions - stock.positions
        expected = np.zeros(len(Joint), dtype=np.float32)
        expected[_IDX[Joint.ELBOW]] = 0.01
        np.testing.assert_allclose(diff, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
