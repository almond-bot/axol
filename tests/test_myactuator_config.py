"""MyActuator 0xC0 configuration access against a fake bus."""

from __future__ import annotations

import struct
import unittest
from unittest import mock

import can

from almond_axol.motor import myactuator
from almond_axol.motor.config import (
    MYACTUATOR_FW_ENCODER_524288,
    MYACTUATOR_FW_V44,
    MyActuatorParam,
)
from almond_axol.motor.errors import MotorError
from almond_axol.motor.myactuator import MyActuatorMotor

_OLD_FW = 2025070202


class _FakeBus:
    """Answers the handful of 0x140-series commands config access needs."""

    channel = "can_alm_axol_l"
    model = "RMD-X6-P20-60"

    def __init__(self, motor_id: int, firmware: int, params: dict[int, float]):
        self._id = motor_id
        self.firmware = firmware
        self.params = dict(params)
        self.echo_override: dict[int, int] = {}
        self.writes: list[tuple[int, float]] = []
        self.commits = 0
        self.position_centideg = 0  # what 0x92 reports
        self.status2 = (0, 0)  # (iq in 0.01 A, speed in dps) for 0x9C
        self.sent: list[tuple[int, bytes]] = []
        self.resets = 0
        self._listener = None

    def _add_listener(self, callback) -> None:  # type: ignore[no-untyped-def]
        self._listener = callback

    async def _send(self, arbitration_id: int, data: bytes) -> None:
        self.sent.append((arbitration_id, bytes(data)))
        if arbitration_id == 0x400 + self._id:  # MIT command: no reply here
            return
        assert arbitration_id == 0x140 + self._id
        cmd = data[0]
        if cmd == 0x76:  # system reset: no reply
            self.resets += 1
            return
        if cmd == 0x92:
            reply = bytes([0x92, 0, 0, 0]) + struct.pack("<i", self.position_centideg)
        elif cmd == 0x9C:
            iq, speed = self.status2
            reply = bytes([0x9C, 25]) + struct.pack("<hhh", iq, speed, 0)
        elif cmd == 0xA4:
            reply = bytes([0xA4, 25, 0, 0, 0, 0, 0, 0])
        elif cmd == 0xB2:
            reply = bytes([0xB2, 0, 0, 0]) + struct.pack("<I", self.firmware)
        elif cmd == 0xB5:
            model = self.model.encode().ljust(15, b"\x00")
            start = (data[2] - 1) * 5
            reply = bytes([0xB5, 0x01, data[2]]) + model[start : start + 5]
        elif cmd == 0x9A:  # status 1: 24 V, running, no faults
            reply = bytes([0x9A, 25, 0, 1, 0xF0, 0x00, 0, 0])
        elif cmd == 0x64:  # write the current position as zero
            reply = bytes(data)
        elif cmd == 0xC1:
            self.commits += 1
            reply = bytes(data)
        elif cmd == 0xC0:
            index, rw = data[2], data[3]
            if rw == 0x00:
                value = struct.unpack_from("<f", data, 4)[0]
                self.writes.append((index, value))
                self.params[index] = value
            echo = self.echo_override.get(index, index)
            reply = bytes([0xC0, 0, echo, rw]) + struct.pack(
                "<f", self.params.get(index, 0.0)
            )
        else:
            raise AssertionError(f"unexpected command {cmd:#04x}")
        assert self._listener is not None
        self._listener(can.Message(arbitration_id=0x240 + self._id, data=reply))


def _motor(firmware: int, **params: float) -> tuple[MyActuatorMotor, _FakeBus]:
    # Exchange Phase 1 is a motor that turns the fleet's way; the direction
    # tests pass 0 for one that turns the reverse.
    params.setdefault("EXCHANGE_PHASE", 1.0)
    # The fake motor is an X6 unless a test changes bus.model.
    params.setdefault("CURRENT_SAMPLE_RES", 3.0)
    values = {int(MyActuatorParam[k]): v for k, v in params.items()}
    bus = _FakeBus(0x01, firmware, values)
    return MyActuatorMotor(bus, 0x01, kt=1.0), bus  # type: ignore[arg-type]


class MyActuatorConfigTest(unittest.IsolatedAsyncioTestCase):
    async def test_low_voltage_reads_and_writes_the_real_threshold(self) -> None:
        motor, bus = _motor(_OLD_FW, LOW_VOLTAGE=20.0)
        self.assertEqual(await motor.get_low_voltage_threshold(), 20.0)
        await motor.set_low_voltage_threshold(-2.0)
        self.assertEqual(bus.writes, [(0x14, -2.0)])
        self.assertEqual(bus.commits, 1)

    async def test_mismatched_index_echo_is_rejected(self) -> None:
        motor, bus = _motor(_OLD_FW, LOW_VOLTAGE=20.0, OVER_VOLTAGE=55.0)
        bus.echo_override[0x14] = 0x13  # a late reply to the previous read
        with self.assertRaises(MotorError):
            await motor.read_config(MyActuatorParam.LOW_VOLTAGE)

    async def test_v44_only_parameters_are_skipped_on_older_firmware(self) -> None:
        motor, bus = _motor(_OLD_FW, MAX_TORQUE=129.0, OVER_VOLTAGE=55.0)
        dumped = await motor.dump_config()
        self.assertNotIn(MyActuatorParam.MAX_TORQUE, dumped)
        self.assertEqual(dumped[MyActuatorParam.OVER_VOLTAGE], 55.0)
        with self.assertRaises(MotorError):
            await motor.read_config(MyActuatorParam.MAX_TORQUE)
        with self.assertRaises(MotorError):
            await motor.write_config(MyActuatorParam.MAX_TORQUE, 60.0)
        written = await motor.restore_config(
            {MyActuatorParam.MAX_TORQUE: 60.0}, include_protected=True
        )
        self.assertEqual(written, [])
        self.assertEqual(bus.writes, [])

    async def test_v44_only_parameters_are_available_on_v44_firmware(self) -> None:
        motor, _ = _motor(MYACTUATOR_FW_V44, MAX_TORQUE=129.0)
        dumped = await motor.dump_config()
        self.assertEqual(dumped[MyActuatorParam.MAX_TORQUE], 129.0)

    async def test_2nd_encoder_mode_also_writes_its_resolution(self) -> None:
        motor, bus = _motor(_OLD_FW)
        await motor.write_config(MyActuatorParam.ENABLE_2ND_ENCODER, 3)
        self.assertEqual(bus.writes, [(0x29, 3.0), (0x3C, 131072.0)])
        bus.writes.clear()
        await motor.write_config(MyActuatorParam.ENABLE_2ND_ENCODER, 2)
        self.assertEqual(bus.writes, [(0x29, 2.0), (0x3C, 16384.0)])
        bus.writes.clear()
        # Disabled / mode 1 leave the resolution alone, as the setup software does.
        await motor.write_config(MyActuatorParam.ENABLE_2ND_ENCODER, 0)
        self.assertEqual(bus.writes, [(0x29, 0.0)])

    async def test_2nd_encoder_resolution_is_left_alone_on_newer_firmware(self) -> None:
        motor, bus = _motor(
            MYACTUATOR_FW_ENCODER_524288, SECOND_ENCODER_RESOLUTION=524288.0
        )
        await motor.write_config(MyActuatorParam.ENABLE_2ND_ENCODER, 3)
        self.assertEqual(bus.writes, [(0x29, 3.0)])
        self.assertEqual(
            await motor.read_config(MyActuatorParam.SECOND_ENCODER_RESOLUTION),
            524288.0,
        )


class MyActuatorModelTest(unittest.IsolatedAsyncioTestCase):
    async def test_model_includes_the_variant_suffix(self) -> None:
        # Same series, another variant: only the third block tells them apart.
        motor, bus = _motor(MYACTUATOR_FW_ENCODER_524288)
        bus.model = "RMD-X6-P20-50"
        self.assertEqual(await motor.get_model(), "RMD-X6-P20-50")
        bus.model = "RMD-X8-P20-120"
        self.assertEqual(await motor.get_model(), "RMD-X8-P20-120")


class MyActuatorTorqueScaleTest(unittest.IsolatedAsyncioTestCase):
    """V4.4 scales MIT torque against MAX_TORQUE; bring-up checks it."""

    async def test_fleet_v44_motor_at_its_rating_enables(self) -> None:
        motor, _ = _motor(MYACTUATOR_FW_V44, MAX_TORQUE=60.0)
        await motor.enable()
        await motor.attach()

    async def test_enable_sets_a_low_max_torque_to_the_rating(self) -> None:
        # Firmware 2026052501 ships X6-P20s with MAX_TORQUE at 1.5.
        motor, bus = _motor(MYACTUATOR_FW_ENCODER_524288, MAX_TORQUE=1.5)
        await motor.enable()
        self.assertIn((MyActuatorParam.MAX_TORQUE, 60.0), bus.writes)
        self.assertEqual(await motor.read_config(MyActuatorParam.MAX_TORQUE), 60.0)

    async def test_the_50_nm_variant_is_set_to_50(self) -> None:
        motor, bus = _motor(MYACTUATOR_FW_ENCODER_524288, MAX_TORQUE=60.0)
        bus.model = "RMD-X6-P20-50"
        await motor.enable()
        self.assertEqual(await motor.read_config(MyActuatorParam.MAX_TORQUE), 50.0)
        self.assertEqual(motor._t_max, 50.0)

    async def test_attach_refuses_a_low_max_torque(self) -> None:
        # A holding motor is not rescaled under its hold.
        motor, bus = _motor(MYACTUATOR_FW_ENCODER_524288, MAX_TORQUE=1.5)
        with self.assertRaises(MotorError) as ctx:
            await motor.attach()
        message = str(ctx.exception)
        self.assertIn("1.5 Nm", message)
        self.assertIn("rated 60 Nm", message)
        self.assertIn(
            "axol motor.set-config --channel can_alm_axol_l --id 0x01 "
            "--param MAX_TORQUE --value 60 --force-protected",
            message,
        )
        self.assertEqual(bus.writes, [])

    async def test_legacy_firmware_skips_the_check(self) -> None:
        motor, _ = _motor(_OLD_FW, MAX_TORQUE=1.5)
        await motor.enable()
        self.assertEqual(motor._t_max, 24.0)


class MyActuatorSampleResTest(unittest.IsolatedAsyncioTestCase):
    """CURRENT_SAMPLE_RES must match the model: 2 on an X8, 3 on an X6."""

    async def test_each_series_passes_with_its_own_value(self) -> None:
        for model, res in (("RMD-X6-P20-60", 3.0), ("RMD-X8-P20-120", 2.0)):
            motor, bus = _motor(MYACTUATOR_FW_V44, CURRENT_SAMPLE_RES=res)
            bus.model = model
            bus.params[MyActuatorParam.MAX_TORQUE] = 60.0 if "X6" in model else 129.0
            await motor.enable()
            await motor.attach()

    async def test_an_x8_with_an_x6_value_is_refused_and_not_written(self) -> None:
        motor, bus = _motor(MYACTUATOR_FW_V44, MAX_TORQUE=129.0)
        bus.model = "RMD-X8-P20-120"
        for bring_up in (motor.enable, motor.attach):
            with self.assertRaises(MotorError) as ctx:
                await bring_up()
            message = str(ctx.exception)
            self.assertIn("current sample resistor set to 3 mOhm", message)
            self.assertIn(
                "--param CURRENT_SAMPLE_RES --value 2 --force-protected", message
            )
        self.assertNotIn(MyActuatorParam.CURRENT_SAMPLE_RES, [i for i, _ in bus.writes])

    async def test_an_x6_with_an_x8_value_is_refused(self) -> None:
        motor, _ = _motor(MYACTUATOR_FW_V44, MAX_TORQUE=60.0, CURRENT_SAMPLE_RES=2.0)
        with self.assertRaises(MotorError):
            await motor.enable()

    async def test_unknown_models_pass(self) -> None:
        motor, bus = _motor(MYACTUATOR_FW_V44, CURRENT_SAMPLE_RES=7.0)
        bus.model = "RMD-L-7015"
        await motor.enable()


def _direction_writes(bus: _FakeBus) -> list[float]:
    return [v for i, v in bus.writes if i == MyActuatorParam.CHANGE_DIRECTION]


class MyActuatorDirectionTest(unittest.IsolatedAsyncioTestCase):
    """Exchange Phase 0 turns the reverse way until Change Motor Direction is on.

    Bring-up only refuses such a motor: the flag moves the joint's zero, so
    zeroing sets it, before it takes the zero.
    """

    def setUp(self) -> None:
        patcher = mock.patch.object(myactuator, "_MA_RESET_SETTLE_S", 0.0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _reversed(self, **params: float) -> tuple[MyActuatorMotor, _FakeBus]:
        return _motor(
            MYACTUATOR_FW_ENCODER_524288, MAX_TORQUE=60.0, EXCHANGE_PHASE=0.0, **params
        )

    async def test_enable_and_attach_refuse_the_flag_off(self) -> None:
        motor, bus = self._reversed()
        for bring_up in (motor.enable, motor.attach):
            with self.assertRaises(MotorError) as ctx:
                await bring_up()
            self.assertIn("axol motor.set-zero-pos --guided", str(ctx.exception))
        self.assertEqual((_direction_writes(bus), bus.resets), ([], 0))

    async def test_flag_on_enables_whatever_the_phase(self) -> None:
        for phase in (0.0, 1.0):
            motor, bus = self._reversed(CHANGE_DIRECTION=1.0)
            bus.params[MyActuatorParam.EXCHANGE_PHASE] = phase
            await motor.enable()
            await motor.attach()
            self.assertEqual((_direction_writes(bus), bus.resets), ([], 0))

    async def test_phase_1_is_never_flipped(self) -> None:
        for firmware in (_OLD_FW, MYACTUATOR_FW_V44, MYACTUATOR_FW_ENCODER_524288):
            motor, bus = _motor(firmware, MAX_TORQUE=60.0)
            self.assertFalse(await motor.set_direction_flag())
            await motor.set_zero_position()
            await motor.enable()
            self.assertEqual(_direction_writes(bus), [])

    async def test_zeroing_turns_the_flag_on_before_the_zero(self) -> None:
        motor, bus = self._reversed()
        await motor.set_zero_position()
        self.assertEqual(_direction_writes(bus), [1.0])
        commands = [data[0] for _, data in bus.sent]
        flag_write = next(
            i
            for i, (_, data) in enumerate(bus.sent)
            if data[0] == 0xC0
            and data[2] == MyActuatorParam.CHANGE_DIRECTION
            and data[3] == 0x00
        )
        # Write, commit, reset to apply, then the zero and its own reset.
        self.assertLess(flag_write, commands.index(0xC1))
        self.assertLess(commands.index(0xC1), commands.index(0x76))
        self.assertLess(commands.index(0x76), commands.index(0x64))
        self.assertEqual(bus.resets, 2)
        await motor.enable()

    async def test_flag_already_on_is_left_alone(self) -> None:
        motor, bus = self._reversed(CHANGE_DIRECTION=1.0)
        self.assertFalse(await motor.set_direction_flag())
        self.assertEqual((_direction_writes(bus), bus.resets), ([], 0))

    async def test_a_motor_without_the_parameters_passes(self) -> None:
        motor, bus = self._reversed()
        real_send = bus._send

        async def no_param_replies(arbitration_id: int, data: bytes) -> None:
            if data[0] == 0xC0 and data[2] in (0x0F, 0x10):
                return
            await real_send(arbitration_id, data)

        bus._send = no_param_replies  # type: ignore[method-assign]
        self.assertFalse(await motor.set_direction_flag())

    async def test_a_flag_that_does_not_stick_refuses(self) -> None:
        motor, bus = self._reversed()
        real_send = bus._send

        async def ignore_direction_writes(arbitration_id: int, data: bytes) -> None:
            if data[0] == 0xC0 and data[2] == 0x0F and data[3] == 0x00:
                data = bytes(data[:4]) + struct.pack("<f", 0.0)
            await real_send(arbitration_id, data)

        bus._send = ignore_direction_writes  # type: ignore[method-assign]
        with self.assertRaises(MotorError):
            await motor.set_zero_position()
        self.assertNotIn(0x64, [data[0] for _, data in bus.sent])


if __name__ == "__main__":
    unittest.main()
