"""Firmware loop gains: config carriage, the idempotent ROM write on the
MyActuator driver, and the enable-time hook that applies them to cold joints."""

from __future__ import annotations

import struct
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from almond_axol.constants import Joint
from almond_axol.motor import MotorError
from almond_axol.motor.myactuator import _MA_PID_IDX, MyActuatorMotor
from almond_axol.robot import FirmwareGains, JointConfig
from almond_axol.robot.axol import apply_firmware_gains
from almond_axol.robot.config import AxolConfig, _calibrated_joint

_X8 = {"position_kp": 0.3, "position_kd": 0.1, "speed_kp": 0.1, "speed_ki": 1e-5}


class ConfigTest(unittest.TestCase):
    def test_shoulders_carry_the_x8_firmware_gains_on_both_arms(self) -> None:
        cfg = AxolConfig()
        for arm in (cfg.left, cfg.right):
            for joint in (arm.shoulder_1, arm.shoulder_2):
                self.assertEqual(joint.firmware.as_dict(), _X8)
                self.assertIsNone(joint.firmware.position_ki)

    def test_elbow_carries_its_own_set_on_both_arms(self) -> None:
        cfg = AxolConfig()
        for arm in (cfg.left, cfg.right):
            self.assertEqual(
                arm.elbow.firmware.as_dict(),
                {
                    "position_kp": 0.2,
                    "position_kd": 0.1,
                    "speed_kp": 0.1,
                    "speed_ki": 1e-5,
                },
            )

    def test_other_joints_leave_the_motor_alone(self) -> None:
        arm = AxolConfig().left
        for name in ("shoulder_3", "wrist_1", "wrist_2", "wrist_3"):
            self.assertEqual(getattr(arm, name).firmware.as_dict(), {})

    def test_defaults_survive_the_stiffness_blend(self) -> None:
        cfg = AxolConfig(left_stiffness=0.3).resolved()
        self.assertEqual(cfg.left.shoulder_1.firmware.as_dict(), _X8)

    def test_calibration_entry_overlays_firmware_block(self) -> None:
        base = AxolConfig().left.elbow
        out = _calibrated_joint(base, {"firmware": {"position_kp": 0.05}})
        self.assertEqual(out.firmware.as_dict(), {"position_kp": 0.05})
        # Untouched entries keep the config's block.
        self.assertIs(_calibrated_joint(base, {"kp": 100.0}).firmware, base.firmware)

    def test_firmware_gains_is_exported_and_replaceable(self) -> None:
        jc = replace(AxolConfig().left.elbow, firmware=FirmwareGains(speed_kp=0.05))
        self.assertIsInstance(jc, JointConfig)
        self.assertEqual(jc.firmware.as_dict(), {"speed_kp": 0.05})


class _FakeMotor(MyActuatorMotor):
    """A V4.2+ motor's gain store behind ``_request``; ROM writes take only
    while ``enabled`` is False, as on hardware."""

    def __init__(self, store: dict[int, float], *, enabled: bool = False) -> None:
        super().__init__(MagicMock(), 0x01, kt=2.0)
        self.store = store
        self.enabled = enabled
        self.writes: list[tuple[int, float]] = []
        self.resets = 0

    async def reset(self) -> None:  # type: ignore[override]
        self.resets += 1

    async def _request(self, data: bytes, *args, **kwargs) -> bytes:  # type: ignore[override]
        cmd, index = data[0], data[1]
        if cmd == 0x30:
            return bytes([0x30, index, 0, 0]) + struct.pack("<f", self.store[index])
        if cmd == 0x32:
            value = struct.unpack_from("<f", data, 4)[0]
            self.writes.append((index, value))
            if not self.enabled:
                self.store[index] = value
            return data
        raise AssertionError(f"unexpected frame {data.hex()}")


class _LegacyMotor(_FakeMotor):
    async def _request(self, data: bytes, *args, **kwargs) -> bytes:  # type: ignore[override]
        # Pre-V4.2 bulk reply: byte 1 is zero, six uint8 gains follow.
        return bytes([0x30, 0, 50, 50, 100, 5, 100, 5])


def _stock() -> dict[int, float]:
    return {
        _MA_PID_IDX["current_kp"]: 0.8,
        _MA_PID_IDX["current_ki"]: 0.08,
        _MA_PID_IDX["speed_kp"]: 0.03,
        _MA_PID_IDX["speed_ki"]: 1e-4,
        _MA_PID_IDX["position_kp"]: 0.008,
        _MA_PID_IDX["position_ki"]: 0.0,
        _MA_PID_IDX["position_kd"]: 0.1,
    }


class EnsureRomGainsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        # No flash settle in tests.
        patcher = patch("almond_axol.motor.myactuator._MA_ROM_SETTLE_S", 0.0)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_writes_only_the_gains_that_differ(self) -> None:
        motor = _FakeMotor(_stock())
        changed = await motor.ensure_rom_gains(_X8)
        # position_kd is already 0.1 in the motor: read, not written.
        self.assertEqual(set(changed), {"position_kp", "speed_kp", "speed_ki"})
        self.assertEqual(
            [i for i, _ in motor.writes],
            [_MA_PID_IDX[n] for n in ("position_kp", "speed_kp", "speed_ki")],
        )
        self.assertAlmostEqual(changed["position_kp"][0], 0.008)
        self.assertAlmostEqual(changed["position_kp"][1], 0.3, places=6)
        self.assertAlmostEqual(motor.store[_MA_PID_IDX["speed_ki"]], 1e-5, places=9)

    async def test_second_call_is_read_only(self) -> None:
        motor = _FakeMotor(_stock())
        await motor.ensure_rom_gains(_X8)
        motor.writes.clear()
        self.assertEqual(await motor.ensure_rom_gains(_X8), {})
        self.assertEqual(motor.writes, [])

    async def test_float32_rounding_counts_as_a_match(self) -> None:
        store = _stock()
        store[_MA_PID_IDX["speed_ki"]] = struct.unpack("<f", struct.pack("<f", 1e-5))[0]
        motor = _FakeMotor(store)
        self.assertEqual(await motor.ensure_rom_gains({"speed_ki": 1e-5}), {})

    async def test_readback_mismatch_raises(self) -> None:
        motor = _FakeMotor(_stock(), enabled=True)
        with self.assertRaisesRegex(MotorError, "reads back"):
            await motor.ensure_rom_gains({"position_kp": 0.3})

    async def test_legacy_firmware_is_refused_not_written(self) -> None:
        motor = _LegacyMotor(_stock())
        with self.assertRaisesRegex(MotorError, "V4.2"):
            await motor.ensure_rom_gains({"position_kp": 0.3})
        self.assertEqual(motor.writes, [])

    async def test_unknown_gain_name_is_a_programming_error(self) -> None:
        with self.assertRaises(ValueError):
            await _FakeMotor(_stock()).ensure_rom_gains({"current_kd": 1.0})


def _arm(drivers: dict[Joint, object], *, is_left: bool = True) -> SimpleNamespace:
    cfg = AxolConfig()
    return SimpleNamespace(
        _is_left=is_left,
        _arm_config=cfg.left if is_left else cfg.right,
        motors={j: SimpleNamespace(_driver=d) for j, d in drivers.items()},
    )


class ApplyFirmwareGainsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        patcher = patch("almond_axol.motor.myactuator._MA_ROM_SETTLE_S", 0.0)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_cold_configured_joints_get_their_gains_and_others_are_untouched(
        self,
    ) -> None:
        s1, s2, elbow, s3 = (_FakeMotor(_stock()) for _ in range(4))
        arm = _arm(
            {
                Joint.SHOULDER_1: s1,
                Joint.SHOULDER_2: s2,
                Joint.ELBOW: elbow,
                Joint.SHOULDER_3: s3,
            }
        )
        with self.assertLogs("almond_axol.robot.axol", level="INFO") as logs:
            await apply_firmware_gains(
                arm, [Joint.SHOULDER_1, Joint.SHOULDER_2, Joint.ELBOW, Joint.SHOULDER_3]
            )
        for motor in (s1, s2):
            self.assertAlmostEqual(motor.store[_MA_PID_IDX["position_kp"]], 0.3, 6)
        self.assertAlmostEqual(elbow.store[_MA_PID_IDX["position_kp"]], 0.2, 6)
        for motor in (s1, s2, elbow):
            self.assertAlmostEqual(motor.store[_MA_PID_IDX["speed_kp"]], 0.1, 6)
            self.assertAlmostEqual(motor.store[_MA_PID_IDX["speed_ki"]], 1e-5, 9)
            # position_kd 0.1 is already the stock value: read, never written.
            self.assertNotIn(_MA_PID_IDX["position_kd"], [i for i, _ in motor.writes])
        # shoulder_3 has no firmware block configured.
        self.assertEqual(s3.writes, [])
        self.assertEqual(sum("written to ROM" in m for m in logs.output), 3)
        # Every motor that took a write is rebooted so its loop loads the new
        # gains; the untouched one is not.
        self.assertEqual([s1.resets, s2.resets, elbow.resets, s3.resets], [1, 1, 1, 0])

    async def test_a_provisioned_motor_is_neither_written_nor_reset(self) -> None:
        s1 = _FakeMotor(_stock())
        await apply_firmware_gains(_arm({Joint.SHOULDER_1: s1}), [Joint.SHOULDER_1])
        s1.writes.clear()
        s1.resets = 0
        await apply_firmware_gains(_arm({Joint.SHOULDER_1: s1}), [Joint.SHOULDER_1])
        self.assertEqual((s1.writes, s1.resets), ([], 0))

    async def test_held_joints_are_not_in_the_list_so_nothing_is_written(self) -> None:
        s1 = _FakeMotor(_stock())
        await apply_firmware_gains(_arm({Joint.SHOULDER_1: s1}), [])
        self.assertEqual(s1.writes, [])

    async def test_a_refusing_motor_warns_and_does_not_fail_enable(self) -> None:
        s1 = _FakeMotor(_stock(), enabled=True)
        arm = _arm({Joint.SHOULDER_1: s1}, is_left=False)
        with self.assertLogs("almond_axol.robot.axol", level="WARNING") as logs:
            await apply_firmware_gains(arm, [Joint.SHOULDER_1])
        self.assertTrue(any("right.shoulder_1" in m for m in logs.output))

    async def test_gripper_and_non_myactuator_joints_are_skipped(self) -> None:
        arm = _arm({Joint.GRIPPER: object(), Joint.WRIST_2: object()})
        # Gripper config has no firmware block; wrist_2 has an empty one.
        await apply_firmware_gains(arm, [Joint.GRIPPER, Joint.WRIST_2])

    async def test_configured_gains_on_a_damiao_joint_warn(self) -> None:
        arm = _arm({Joint.WRIST_2: object()})
        arm._arm_config = replace(
            arm._arm_config,
            wrist_2=replace(
                arm._arm_config.wrist_2, firmware=FirmwareGains(speed_kp=1)
            ),
        )
        with self.assertLogs("almond_axol.robot.axol", level="WARNING") as logs:
            await apply_firmware_gains(arm, [Joint.WRIST_2])
        self.assertTrue(any("not a MyActuator" in m for m in logs.output))


if __name__ == "__main__":
    unittest.main()
