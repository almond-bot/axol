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
from almond_axol.motor.damiao import DamiaoMotor
from almond_axol.motor.myactuator import _MA_PID_IDX, MyActuatorMotor
from almond_axol.robot import FirmwareGains, JointConfig
from almond_axol.robot.axol import apply_firmware_gains
from almond_axol.robot.config import AxolConfig, _calibrated_joint

_X8 = {
    "position_kp": 1.0,
    "position_kd": 0.1,
    "speed_kp": 0.07,
    "speed_ki": 1e-5,
    "planner_accel": 0.0,
}


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
                    "position_kp": 1.4,
                    "position_kd": 0.1,
                    "speed_kp": 0.05,
                    "speed_ki": 1e-5,
                    "planner_accel": 0.0,
                },
            )

    def test_x6_roll_joints_share_a_set_on_both_arms(self) -> None:
        cfg = AxolConfig()
        for arm in (cfg.left, cfg.right):
            for joint in (arm.shoulder_3, arm.wrist_1):
                self.assertEqual(
                    joint.firmware.as_dict(),
                    {
                        "position_kp": 1.0,
                        "position_kd": 0.5,
                        "speed_kp": 0.05,
                        "speed_ki": 1e-5,
                        "planner_accel": 0.0,
                    },
                )

    def test_damiao_wrists_carry_the_position_gain_only(self) -> None:
        cfg = AxolConfig()
        for arm in (cfg.left, cfg.right):
            for name in ("wrist_2", "wrist_3"):
                self.assertEqual(
                    getattr(arm, name).firmware.as_dict(),
                    {"position_kp": 400.0, "profile_acc": 50.0},
                )

    def test_the_gripper_has_no_firmware_block(self) -> None:
        self.assertFalse(hasattr(AxolConfig().left.gripper, "firmware"))

    def test_defaults_survive_the_stiffness_blend(self) -> None:
        cfg = AxolConfig(left_stiffness=0.3).resolved()
        self.assertEqual(cfg.left.shoulder_1.firmware.as_dict(), _X8)

    def test_calibration_entry_overlays_firmware_block(self) -> None:
        base = AxolConfig().left.elbow
        out = _calibrated_joint(base, {"firmware": {"position_kp": 0.05}})
        self.assertEqual(out.firmware.as_dict(), {"position_kp": 0.05})
        # Untouched entries keep the config's block.
        self.assertEqual(_calibrated_joint(base, {"kp": 100.0}).firmware, base.firmware)

    def test_firmware_gains_is_exported_and_replaceable(self) -> None:
        jc = replace(AxolConfig().left.elbow, firmware=FirmwareGains(speed_kp=0.05))
        self.assertIsInstance(jc, JointConfig)
        self.assertEqual(jc.firmware.as_dict(), {"speed_kp": 0.05})


class _FakeMotor(MyActuatorMotor):
    """A V4.2+ motor's gain store behind ``_request``; ROM writes take only
    while ``enabled`` is False, as on hardware."""

    def __init__(
        self,
        store: dict[int, float],
        *,
        enabled: bool = False,
        planner: tuple[int, int] = (0, 0),
    ) -> None:
        super().__init__(MagicMock(), 0x01, kt=2.0)
        self.store = store
        self.enabled = enabled
        self.writes: list[tuple[int, float]] = []
        self.resets = 0
        # Position planner accel/decel (0x42 types 0/1), 0x43 writes.
        self.planner = {0: planner[0], 1: planner[1]}
        self.planner_writes: list[tuple[int, int]] = []

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
        if cmd == 0x42:
            return bytes([0x42, index, 0, 0]) + struct.pack("<i", self.planner[index])
        if cmd == 0x43:
            value = struct.unpack_from("<I", data, 4)[0]
            self.planner_writes.append((index, value))
            self.planner[index] = value
            return data
        raise AssertionError(f"unexpected frame {data.hex()}")


class _FakeDamiao(DamiaoMotor):
    """A Damiao register store: 0x33 reads and 0x55 writes against a dict, with
    0xAA stores counted; no bus."""

    def __init__(self, store: dict[int, float]) -> None:
        super().__init__(MagicMock(), 0x06, 0x16)
        self.store = store
        self.writes: list[tuple[int, float]] = []
        self.stores = 0

    async def _read_register(self, rid, timeout=0.2, attempts=5):  # type: ignore[override]
        return self.store[rid]

    async def _write_register(self, rid, value):  # type: ignore[override]
        self.writes.append((rid, float(value)))
        self.store[rid] = float(value)

    async def _store_parameters(self):  # type: ignore[override]
        self.stores += 1


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
        self.assertAlmostEqual(changed["position_kp"][1], 1.0, places=6)
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


def _arm(
    drivers: dict[Joint, object], *, is_left: bool = True, config: object = None
) -> SimpleNamespace:
    cfg = AxolConfig()
    return SimpleNamespace(
        _is_left=is_left,
        _arm_config=config or (cfg.left if is_left else cfg.right),
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
        s1, s2, elbow, w1 = (_FakeMotor(_stock()) for _ in range(4))
        arm = _arm(
            {
                Joint.SHOULDER_1: s1,
                Joint.SHOULDER_2: s2,
                Joint.ELBOW: elbow,
                Joint.WRIST_1: w1,
            }
        )
        with self.assertLogs("almond_axol.robot.axol", level="INFO") as logs:
            await apply_firmware_gains(
                arm, [Joint.SHOULDER_1, Joint.SHOULDER_2, Joint.ELBOW, Joint.WRIST_1]
            )
        for motor in (s1, s2):
            self.assertAlmostEqual(motor.store[_MA_PID_IDX["position_kp"]], 1.0, 6)
        self.assertAlmostEqual(elbow.store[_MA_PID_IDX["position_kp"]], 1.4, 6)
        for motor in (s1, s2):
            self.assertAlmostEqual(motor.store[_MA_PID_IDX["speed_kp"]], 0.07, 6)
        self.assertAlmostEqual(elbow.store[_MA_PID_IDX["speed_kp"]], 0.05, 6)
        for motor in (s1, s2, elbow):
            self.assertAlmostEqual(motor.store[_MA_PID_IDX["speed_ki"]], 1e-5, 9)
            # position_kd 0.1 is already the stock value: read, never written.
            self.assertNotIn(_MA_PID_IDX["position_kd"], [i for i, _ in motor.writes])
        # wrist_1 carries the X6 roll set: its position and speed gains change too.
        self.assertAlmostEqual(w1.store[_MA_PID_IDX["position_kp"]], 1.0, 6)
        self.assertAlmostEqual(w1.store[_MA_PID_IDX["speed_kp"]], 0.05, 6)
        self.assertEqual(sum("written to ROM" in m for m in logs.output), 4)
        # Every motor that took a write is rebooted so its loop loads the new
        # gains; the untouched one is not.
        self.assertEqual([s1.resets, s2.resets, elbow.resets, w1.resets], [1, 1, 1, 1])

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

    async def test_a_planner_left_on_is_put_back_to_direct_tracking(self) -> None:
        # A test run left the shoulder's planner at 60000 (decel 10 as found
        # on right shoulder_1): enable pins both back to the config's 0 and
        # reboots the motor, since the X6-P20's 2025-07 firmware ignores a 0
        # written into a running loop until the reset.
        s1 = _FakeMotor(_stock(), planner=(60000, 10))
        with self.assertLogs("almond_axol.robot.axol", level="INFO") as logs:
            await apply_firmware_gains(_arm({Joint.SHOULDER_1: s1}), [Joint.SHOULDER_1])
        self.assertEqual(s1.planner, {0: 0, 1: 0})
        self.assertEqual(s1.planner_writes, [(0, 0), (1, 0)])
        self.assertEqual(s1.resets, 1)
        self.assertTrue(any("planner_accel 60000 -> 0" in m for m in logs.output))

    async def test_the_planner_override_reaches_the_motor(self) -> None:
        cfg = AxolConfig()
        cfg.left.shoulder_1.firmware.planner_accel = 60000.0
        # Each joint owns its block: the shoulder_2 and a fresh config keep 0.
        self.assertEqual(cfg.left.shoulder_2.firmware.planner_accel, 0.0)
        self.assertEqual(AxolConfig().left.shoulder_1.firmware.planner_accel, 0.0)
        s1 = _FakeMotor(_stock())
        await apply_firmware_gains(
            _arm({Joint.SHOULDER_1: s1}, config=cfg.left), [Joint.SHOULDER_1]
        )
        self.assertEqual(s1.planner, {0: 60000, 1: 60000})

    async def test_gripper_is_skipped(self) -> None:
        # Gripper config has no firmware block at all.
        await apply_firmware_gains(_arm({Joint.GRIPPER: object()}), [Joint.GRIPPER])

    async def test_damiao_wrist_is_provisioned_through_its_registers_without_a_reset(
        self,
    ) -> None:
        w2 = _FakeDamiao({25: 0.0037, 26: 0.002, 27: 54.0, 28: 0.0, 4: 50.0, 5: -50.0})
        arm = _arm({Joint.WRIST_2: w2})
        with self.assertLogs("almond_axol.robot.axol", level="INFO") as logs:
            await apply_firmware_gains(arm, [Joint.WRIST_2])
        self.assertEqual(w2.store[27], 400.0)
        self.assertEqual(w2.stores, 1)
        self.assertTrue(any("written and stored" in m for m in logs.output))
        # Second pass: already provisioned — no write, no store.
        w2.writes.clear()
        w2.stores = 0
        await apply_firmware_gains(arm, [Joint.WRIST_2])
        self.assertEqual((w2.writes, w2.stores), ([], 0))

    async def test_damiao_profile_ramp_writes_acc_and_negative_dec(self) -> None:
        # Stock wrists: KP_APR 54, ramps ±2 rad/s². Config wants 400 and 50.
        w2 = _FakeDamiao({25: 0.0037, 26: 0.002, 27: 54.0, 28: 0.0, 4: 2.0, 5: -2.0})
        arm = _arm({Joint.WRIST_2: w2})
        with self.assertLogs("almond_axol.robot.axol", level="INFO"):
            await apply_firmware_gains(arm, [Joint.WRIST_2])
        self.assertEqual((w2.store[27], w2.store[4], w2.store[5]), (400.0, 50.0, -50.0))
        self.assertEqual(w2.stores, 1)
        w2.writes.clear()
        w2.stores = 0
        await apply_firmware_gains(arm, [Joint.WRIST_2])
        self.assertEqual((w2.writes, w2.stores), ([], 0))
        # A DEC that drifted alone is repaired too.
        w2.store[5] = -2.0
        await apply_firmware_gains(arm, [Joint.WRIST_2])
        self.assertEqual(w2.store[5], -50.0)
        self.assertEqual(w2.stores, 1)

    def test_wrists_carry_the_profile_ramp_and_myactuator_joints_do_not(self) -> None:
        cfg = AxolConfig()
        for arm in (cfg.left, cfg.right):
            self.assertEqual(arm.wrist_2.firmware.profile_acc, 50.0)
            self.assertEqual(arm.wrist_3.firmware.profile_acc, 50.0)
            self.assertIsNone(arm.elbow.firmware.profile_acc)
            self.assertIsNone(arm.shoulder_1.firmware.profile_acc)

    async def test_configured_gains_on_a_joint_without_a_loop_warn(self) -> None:
        arm = _arm({Joint.WRIST_2: object()})  # a driver of neither vendor
        arm._arm_config = replace(
            arm._arm_config,
            wrist_2=replace(
                arm._arm_config.wrist_2, firmware=FirmwareGains(speed_kp=1)
            ),
        )
        with self.assertLogs("almond_axol.robot.axol", level="WARNING") as logs:
            await apply_firmware_gains(arm, [Joint.WRIST_2])
        self.assertTrue(any("no firmware position loop" in m for m in logs.output))


if __name__ == "__main__":
    unittest.main()


class PlannerConfigTest(unittest.TestCase):
    """``planner_accel`` / ``cap_track``: the 0xA4 planner and its speed cap."""

    def test_only_the_two_accelerations_that_follow_a_stream(self) -> None:
        FirmwareGains(planner_accel=0.0)
        FirmwareGains(planner_accel=60000.0)
        with self.assertRaisesRegex(ValueError, "barely moves"):
            FirmwareGains(planner_accel=5000.0)

    def test_cap_track_must_keep_up_with_the_command(self) -> None:
        FirmwareGains(cap_track=1.2)
        FirmwareGains(cap_track=0.0)
        with self.assertRaisesRegex(ValueError, "never keeps up"):
            FirmwareGains(cap_track=0.8)

    def test_lead_is_bounded(self) -> None:
        FirmwareGains(planner_lead_ms=5.0)
        with self.assertRaisesRegex(ValueError, "0..50"):
            FirmwareGains(planner_lead_ms=80.0)

    def test_cap_track_is_the_cores_not_the_motors(self) -> None:
        gains = FirmwareGains(
            position_kp=1.0, planner_accel=60000.0, cap_track=1.2, planner_lead_ms=5.0
        )
        self.assertEqual(
            gains.as_dict(), {"position_kp": 1.0, "planner_accel": 60000.0}
        )
