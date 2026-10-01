"""The robot link's motor-config read/write helpers (the parameter editor)."""

from __future__ import annotations

import asyncio
import json
import math
import unittest

from almond_axol.constants import Joint
from almond_axol.motor import Motor
from almond_axol.motor.config import MYACTUATOR_FW_V44, MyActuatorParam
from almond_axol.serve.robot_link import _read_motor_config, _write_motor_config
from tests.test_myactuator_config import _OLD_FW, _FakeBus


class _FakeArmLink:
    """Just what the config helpers touch: side, motors, and per-joint locks."""

    def __init__(self, motor: Motor, joint: Joint) -> None:
        self.side = "left"
        self.motors = {joint: motor}
        self._locks = {joint: asyncio.Lock()}

    def lock(self, joint: Joint) -> asyncio.Lock:
        return self._locks[joint]


def _link(firmware: int, **params: float) -> tuple[_FakeArmLink, _FakeBus]:
    values = {int(MyActuatorParam[k]): v for k, v in params.items()}
    bus = _FakeBus(0x01, firmware, values)
    motor = Motor(bus, Joint.SHOULDER_1)  # type: ignore[arg-type]
    return _FakeArmLink(motor, Joint.SHOULDER_1), bus


def _by_name(config: dict) -> dict[str, dict]:
    return {p["name"]: p for p in config["params"]}


class ReadMotorConfigTest(unittest.IsolatedAsyncioTestCase):
    async def test_lists_every_parameter_with_values_and_metadata(self) -> None:
        link, _ = _link(MYACTUATOR_FW_V44, LOW_VOLTAGE=20.0, MAX_TORQUE=129.0)
        config = await _read_motor_config(link, Joint.SHOULDER_1)  # type: ignore[arg-type]
        self.assertEqual(config["type"], "myactuator")
        self.assertEqual(config["firmware"], MYACTUATOR_FW_V44)
        params = _by_name(config)
        self.assertEqual(set(params), {p.name for p in MyActuatorParam})
        low = params["LOW_VOLTAGE"]
        self.assertEqual((low["index"], low["unit"], low["value"]), (0x14, "V", 20.0))
        self.assertEqual(low["access"], "read_write")
        self.assertEqual(params["MAX_TORQUE"]["value"], 129.0)
        self.assertEqual(params["MOTOR_POSITION_ZERO"]["access"], "protected")

    async def test_unsupported_parameters_have_no_value_on_older_firmware(self) -> None:
        link, _ = _link(_OLD_FW, MAX_TORQUE=129.0)
        params = _by_name(await _read_motor_config(link, Joint.SHOULDER_1))  # type: ignore[arg-type]
        self.assertFalse(params["MAX_TORQUE"]["supported"])
        self.assertIsNone(params["MAX_TORQUE"]["value"])
        self.assertTrue(params["LOW_VOLTAGE"]["supported"])

    async def test_non_finite_readback_is_json_safe(self) -> None:
        link, _ = _link(MYACTUATOR_FW_V44, LOW_VOLTAGE=math.nan)
        config = await _read_motor_config(link, Joint.SHOULDER_1)  # type: ignore[arg-type]
        low = _by_name(config)["LOW_VOLTAGE"]
        self.assertIsNone(low["value"])
        self.assertIsNotNone(low["error"])
        json.dumps(config, allow_nan=False)  # what JSONResponse requires

    async def test_one_bad_reply_does_not_fail_the_readout(self) -> None:
        link, bus = _link(MYACTUATOR_FW_V44, LOW_VOLTAGE=20.0, OVER_VOLTAGE=55.0)
        bus.echo_override[0x14] = 0x13  # every reply to 0x14 echoes the wrong index
        params = _by_name(await _read_motor_config(link, Joint.SHOULDER_1))  # type: ignore[arg-type]
        self.assertIsNone(params["LOW_VOLTAGE"]["value"])
        self.assertIn("echoed", params["LOW_VOLTAGE"]["error"])
        self.assertEqual(params["OVER_VOLTAGE"]["value"], 55.0)


class WriteMotorConfigTest(unittest.IsolatedAsyncioTestCase):
    async def test_writes_persists_and_reads_back(self) -> None:
        link, bus = _link(_OLD_FW, OVER_VOLTAGE=55.0)
        result = await _write_motor_config(
            link,  # type: ignore[arg-type]
            Joint.SHOULDER_1,
            "OVER_VOLTAGE",
            52.0,
            False,
        )
        self.assertEqual(
            result, {"name": "OVER_VOLTAGE", "requested": 52.0, "value": 52.0}
        )
        self.assertEqual(bus.writes, [(0x13, 52.0)])
        self.assertEqual(bus.commits, 1)

    async def test_protected_write_needs_explicit_confirmation(self) -> None:
        link, bus = _link(_OLD_FW, KT_OUT=2.0)
        with self.assertRaisesRegex(ValueError, "protected"):
            await _write_motor_config(link, Joint.SHOULDER_1, "KT_OUT", 2.5, False)  # type: ignore[arg-type]
        self.assertEqual(bus.writes, [])
        await _write_motor_config(link, Joint.SHOULDER_1, "KT_OUT", 2.5, True)  # type: ignore[arg-type]
        self.assertEqual(bus.writes, [(0x3E, 2.5)])

    async def test_refuses_bad_requests_without_touching_the_bus(self) -> None:
        link, bus = _link(_OLD_FW)
        cases = [
            ("NOT_A_PARAM", 1.0, True, "no configuration parameter"),
            ("OVER_VOLTAGE", math.inf, False, "finite"),
            ("MAX_TORQUE", 60.0, True, "not implemented"),  # V4.4-only on old firmware
            ("TIMEOUT", 1.0, True, "no configuration parameter"),  # Damiao's table
        ]
        for name, value, allow, message in cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                await _write_motor_config(link, Joint.SHOULDER_1, name, value, allow)  # type: ignore[arg-type]
        self.assertEqual(bus.writes, [])

    async def test_second_encoder_mode_carries_its_resolution(self) -> None:
        # Mirrors the setup software: modes 2/3 set the matching resolution,
        # modes 0/1 leave it alone.
        link, bus = _link(_OLD_FW, SECOND_ENCODER_RESOLUTION=131072.0)
        await _write_motor_config(
            link, Joint.SHOULDER_1, "ENABLE_2ND_ENCODER", 2.0, True
        )  # type: ignore[arg-type]
        self.assertEqual(bus.writes, [(0x29, 2.0), (0x3C, 16384.0)])
        bus.writes.clear()
        await _write_motor_config(
            link, Joint.SHOULDER_1, "ENABLE_2ND_ENCODER", 0.0, True
        )  # type: ignore[arg-type]
        self.assertEqual(bus.writes, [(0x29, 0.0)])


if __name__ == "__main__":
    unittest.main()
