"""MyActuator 0xC0 configuration access against a fake bus."""

from __future__ import annotations

import struct
import unittest

import can

from almond_axol.motor.config import MYACTUATOR_FW_V44, MyActuatorParam
from almond_axol.motor.errors import MotorError
from almond_axol.motor.myactuator import MyActuatorMotor

_OLD_FW = 2025070202


class _FakeBus:
    """Answers the handful of 0x140-series commands config access needs."""

    def __init__(self, motor_id: int, firmware: int, params: dict[int, float]):
        self._id = motor_id
        self.firmware = firmware
        self.params = dict(params)
        self.echo_override: dict[int, int] = {}
        self.writes: list[tuple[int, float]] = []
        self.commits = 0
        self._listener = None

    def _add_listener(self, callback) -> None:  # type: ignore[no-untyped-def]
        self._listener = callback

    async def _send(self, arbitration_id: int, data: bytes) -> None:
        assert arbitration_id == 0x140 + self._id
        cmd = data[0]
        if cmd == 0xB2:
            reply = bytes([0xB2, 0, 0, 0]) + struct.pack("<I", self.firmware)
        elif cmd == 0xB5:
            reply = bytes([0xB5, 0x01, data[2]]) + b"X6S2V"[:5]
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


if __name__ == "__main__":
    unittest.main()
