from __future__ import annotations

import asyncio
import math
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from almond_axol.cli.can import setup as can_setup
from almond_axol.robot import lift as lift_module
from almond_axol.robot.battery import (
    CAPACITY_AH,
    CHARGING_VOLTS,
    LIFEPO4_8S_CURVE,
    BatteryEstimator,
    battery_percent,
    estimate_battery,
)
from almond_axol.robot.lift import (
    Lift,
    decode_power,
    decode_status,
    power_under_load,
    read_battery,
    read_power,
)


def _power_frame(
    volts: float = 25.693,
    leg_amps: tuple[float, float] = (0.026, 0.023),
    fault_mask: int = 0,
    state: int = 0x03,
) -> bytes:
    return struct.pack(
        "<HHHBB",
        round(volts * 1000),
        round(leg_amps[0] * 1000),
        round(leg_amps[1] * 1000),
        fault_mask,
        state,
    )


def _status_frame(flags: int = 0x01) -> bytes:
    return struct.pack("<HhBbBB", 500, 0, flags, 0, 0, 0x03)


class BatteryCurveTest(unittest.TestCase):
    def test_curve_points_and_interpolation(self) -> None:
        for volts, percent in LIFEPO4_8S_CURVE:
            self.assertAlmostEqual(battery_percent(volts), percent)
        # Halfway between 26.15 V (25 %) and 26.35 V (50 %).
        self.assertAlmostEqual(battery_percent(26.25), 37.5)

    def test_matches_the_bms_on_jelly(self) -> None:
        # Measured at rest on Jelly: 25.69 V while the LiTime app showed 18 %.
        self.assertAlmostEqual(battery_percent(25.69), 18.0, delta=3.0)

    def test_clamps_outside_the_curve(self) -> None:
        self.assertEqual(battery_percent(12.0), 0.0)
        self.assertEqual(battery_percent(29.2), 100.0)

    def test_curve_is_monotonic(self) -> None:
        volts = [v for v, _ in LIFEPO4_8S_CURVE]
        percents = [p for _, p in LIFEPO4_8S_CURVE]
        self.assertEqual(volts, sorted(volts))
        self.assertEqual(percents, sorted(percents))

    def test_rejects_non_finite_voltage(self) -> None:
        with self.assertRaises(ValueError):
            battery_percent(math.nan)

    def test_estimate(self) -> None:
        status = estimate_battery(26.35)
        assert status is not None
        self.assertAlmostEqual(status.percent, 50.0)
        self.assertAlmostEqual(status.remaining_ah, CAPACITY_AH / 2)
        self.assertFalse(status.charging)
        self.assertFalse(status.under_load)

    def test_charger_voltage_reports_charging_at_full(self) -> None:
        status = estimate_battery(29.0)
        assert status is not None
        self.assertTrue(status.charging)
        self.assertEqual(status.percent, 100.0)

    def test_no_pack_is_none_not_empty(self) -> None:
        # The board on USB power alone reads a few volts of nothing.
        self.assertIsNone(estimate_battery(0.4))
        self.assertIsNone(estimate_battery(math.nan))


class BatteryEstimatorTest(unittest.TestCase):
    def test_first_sample_is_taken_as_is_then_smoothed(self) -> None:
        est = BatteryEstimator()
        self.assertIsNone(est.status)
        first = est.update(26.0)
        assert first is not None
        self.assertAlmostEqual(first.voltage, 26.0)
        second = est.update(26.5)
        assert second is not None
        self.assertGreater(second.voltage, 26.0)
        self.assertLess(second.voltage, 26.5)

    def test_load_sample_never_displaces_a_resting_estimate(self) -> None:
        est = BatteryEstimator()
        est.update(26.2)
        sagged = est.update(24.5, under_load=True)
        assert sagged is not None
        self.assertAlmostEqual(sagged.voltage, 26.2)
        self.assertFalse(sagged.under_load)

    def test_load_only_estimate_is_flagged_and_replaced_at_rest(self) -> None:
        est = BatteryEstimator()
        loaded = est.update(25.0, under_load=True)
        assert loaded is not None
        self.assertTrue(loaded.under_load)
        rested = est.update(26.1)
        assert rested is not None
        self.assertFalse(rested.under_load)
        self.assertAlmostEqual(rested.voltage, 26.1)

    def test_charger_steps_are_not_averaged(self) -> None:
        est = BatteryEstimator()
        est.update(26.0)
        charging = est.update(29.1)
        assert charging is not None
        self.assertTrue(charging.charging)
        self.assertAlmostEqual(charging.voltage, 29.1)
        unplugged = est.update(27.0)
        assert unplugged is not None
        self.assertAlmostEqual(unplugged.voltage, 27.0)
        self.assertLess(unplugged.voltage, CHARGING_VOLTS)

    def test_pack_removed_clears_the_estimate(self) -> None:
        est = BatteryEstimator()
        est.update(26.0)
        self.assertIsNone(est.update(0.3))
        self.assertIsNone(est.status)


class PowerFrameTest(unittest.TestCase):
    def test_decodes_the_firmware_power_frame(self) -> None:
        power = decode_power(_power_frame(25.693, (1.25, 0.5), 0x02, 0x0B))
        self.assertAlmostEqual(power.supply_volts, 25.693)
        self.assertEqual(power.leg_currents, (1.25, 0.5))
        self.assertEqual(power.driver_fault_mask, 0x02)
        self.assertTrue(power.drivers_enabled)
        self.assertTrue(power.vm_present)
        self.assertFalse(power.flash_interlock)
        self.assertTrue(power.save_pending)

    def test_short_frame_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            decode_power(b"\x00" * 6)

    def test_leg_current_or_motion_means_under_load(self) -> None:
        idle = decode_power(_power_frame())
        drawing = decode_power(_power_frame(leg_amps=(2.0, 1.8)))
        self.assertFalse(power_under_load(idle, None))
        self.assertTrue(power_under_load(drawing, None))
        moving = decode_status(_status_frame(flags=0x01 | 0x02))
        self.assertTrue(power_under_load(idle, moving))


class LiftPowerTest(unittest.IsolatedAsyncioTestCase):
    def test_power_frames_feed_the_battery_estimate(self) -> None:
        lift = Lift("can-test")
        self.assertIsNone(lift.power)
        self.assertIsNone(lift.battery)
        lift._on_message(
            SimpleNamespace(arbitration_id=0x422, data=_power_frame(26.35))
        )
        assert lift.power is not None
        self.assertAlmostEqual(lift.power.supply_volts, 26.35)
        battery = lift.battery
        assert battery is not None
        self.assertAlmostEqual(battery.percent, 50.0)

    def test_stale_power_means_no_battery(self) -> None:
        lift = Lift("can-test")
        lift._on_message(SimpleNamespace(arbitration_id=0x422, data=_power_frame()))
        lift._last_power_monotonic = (
            lift_module.time.monotonic() - lift_module._POWER_FRESH_S - 1.0
        )
        self.assertIsNone(lift.battery)

    def test_external_load_holds_the_resting_estimate(self) -> None:
        lift = Lift("can-test")
        lift._on_message(SimpleNamespace(arbitration_id=0x422, data=_power_frame(26.2)))
        lift.external_load = True  # Jelly: the wheels are turning
        lift._on_message(SimpleNamespace(arbitration_id=0x422, data=_power_frame(25.0)))
        battery = lift.battery
        assert battery is not None
        self.assertAlmostEqual(battery.voltage, 26.2)

    async def test_polling_run_asks_for_power_once_a_second(self) -> None:
        lift = Lift("can-test")
        lift._send = AsyncMock()  # type: ignore[method-assign]
        clock = [100.0]
        loop = asyncio.get_running_loop()
        ticks = [0]

        async def sleep(seconds: float) -> None:
            clock[0] += seconds
            ticks[0] += 1
            if ticks[0] >= 50:  # 2.5 s of 50 ms ticks
                raise asyncio.CancelledError

        with (
            patch.object(loop, "time", lambda: clock[0]),
            patch.object(lift_module.asyncio, "sleep", sleep),
            self.assertRaises(asyncio.CancelledError),
        ):
            await lift._run()

        power_polls = [
            c
            for c in lift._send.await_args_list
            if c.args[0] == lift_module._OP_GET_POWER
        ]
        self.assertEqual(len(power_polls), 2)

    async def test_broadcast_mode_never_polls_power(self) -> None:
        lift = Lift("can-test", status_period_ms=200)
        lift._last_status_monotonic = lift_module.time.monotonic()
        lift._send = AsyncMock()  # type: ignore[method-assign]
        clock = [100.0]
        loop = asyncio.get_running_loop()
        ticks = [0]

        async def sleep(seconds: float) -> None:
            clock[0] += seconds
            ticks[0] += 1
            lift._last_status_monotonic = lift_module.time.monotonic()
            if ticks[0] >= 50:
                raise asyncio.CancelledError

        with (
            patch.object(loop, "time", lambda: clock[0]),
            patch.object(lift_module.asyncio, "sleep", sleep),
            self.assertRaises(asyncio.CancelledError),
        ):
            await lift._run()

        self.assertNotIn(
            lift_module._OP_GET_POWER,
            [c.args[0] for c in lift._send.await_args_list],
        )


class _AnsweringBus:
    """A CanBus whose jelly_legs board answers GET_POWER (or stays silent)."""

    instances: list[_AnsweringBus] = []

    def __init__(self, channel: str, *, answer: bytes | None = None) -> None:
        self.channel = channel
        self.answer = answer
        self.listener = None
        self.sent: list[bytes] = []
        self.closed = False
        _AnsweringBus.instances.append(self)

    def _add_listener(self, listener) -> None:  # noqa: ANN001
        self.listener = listener

    async def start(self) -> None:
        pass

    async def _send(self, arbitration_id: int, data: bytes) -> bool:
        self.sent.append(data)
        if data == bytes([lift_module._OP_GET_POWER]) and self.answer is not None:
            asyncio.get_running_loop().call_soon(
                self.listener, SimpleNamespace(arbitration_id=0x422, data=self.answer)
            )
        return True

    async def close(self) -> None:
        self.closed = True


class ReadPowerTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _AnsweringBus.instances.clear()

    async def test_one_shot_read_quiets_the_board_and_closes(self) -> None:
        with (
            patch.object(can_setup, "iface_up", return_value=True),
            patch.object(
                lift_module,
                "CanBus",
                lambda ch: _AnsweringBus(ch, answer=_power_frame(26.63)),
            ),
        ):
            power = await read_power("can-test")
            battery = await read_battery("can-test")

        assert power is not None and battery is not None
        self.assertAlmostEqual(power.supply_volts, 26.63)
        self.assertAlmostEqual(battery.percent, 75.0)
        bus = _AnsweringBus.instances[0]
        self.assertEqual(bus.sent[:2], [b"\x05\x00\x00", b"\x08"])
        # Never a motion opcode: no STOP / JOG / HOME / SET_POS.
        for bus in _AnsweringBus.instances:
            self.assertTrue(bus.closed)
            self.assertTrue(all(frame[0] in (0x05, 0x08) for frame in bus.sent))

    async def test_silent_board_is_none(self) -> None:
        with (
            patch.object(can_setup, "iface_up", return_value=True),
            patch.object(lift_module, "CanBus", lambda ch: _AnsweringBus(ch)),
        ):
            self.assertIsNone(await read_power("can-test", timeout=0.3))
        self.assertTrue(_AnsweringBus.instances[0].closed)
        self.assertGreater(len(_AnsweringBus.instances[0].sent), 2)


if __name__ == "__main__":
    unittest.main()
