from __future__ import annotations

import asyncio
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, call, patch

from almond_axol.cli.can import setup as can_setup
from almond_axol.robot import lift as lift_module
from almond_axol.robot.lift import Lift, LiftStatus, _decode_status


class LiftStatusTest(unittest.TestCase):
    def test_decodes_v08_driver_health(self) -> None:
        status = _decode_status(
            struct.pack("<HhBbBB", 750, -321, 0xFF, -12, 0x03, 0x0B)
        )

        self.assertEqual(status.position_permille, 750)
        self.assertEqual(status.velocity, -321)
        self.assertEqual(status.drift, -12)
        self.assertTrue(status.homed)
        self.assertTrue(status.moving)
        self.assertTrue(status.pos_move)
        self.assertTrue(status.stall_fault)
        self.assertTrue(status.at_lower)
        self.assertTrue(status.at_upper)
        self.assertTrue(status.homing)
        self.assertTrue(status.jog)
        self.assertEqual(status.driver_fault_mask, 0x03)
        self.assertTrue(status.drivers_enabled)
        self.assertTrue(status.vm_present)
        self.assertFalse(status.flash_interlock)
        self.assertTrue(status.save_pending)

    def test_legacy_status_keeps_driver_health_unknown(self) -> None:
        status = _decode_status(struct.pack("<HhBb", 0xFFFF, 0, 0, 0))

        self.assertIsNone(status.position_permille)
        self.assertIsNone(status.driver_fault_mask)
        self.assertIsNone(status.drivers_enabled)
        self.assertIsNone(status.vm_present)
        self.assertIsNone(status.flash_interlock)
        self.assertIsNone(status.save_pending)

    def test_new_fields_do_not_break_legacy_construction(self) -> None:
        status = LiftStatus(
            None,
            0,
            0,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        )

        self.assertIsNone(status.driver_fault_mask)
        self.assertIsNone(status.save_pending)

    def test_tracks_status_freshness(self) -> None:
        lift = Lift()
        message = SimpleNamespace(
            arbitration_id=lift_module._ID_STATUS,
            data=struct.pack("<HhBbBB", 1000, 0, 0x21, 1, 0, 0x03),
        )

        with patch.object(lift_module.time, "monotonic", return_value=10.0):
            lift._on_message(message)
        self.assertEqual(lift.last_status_monotonic, 10.0)
        with patch.object(lift_module.time, "monotonic", return_value=10.4):
            self.assertAlmostEqual(lift.status_age or 0.0, 0.4)
            self.assertTrue(lift.status_is_fresh(0.5))
            self.assertFalse(lift.status_is_fresh(0.3))

    def test_no_status_is_not_fresh(self) -> None:
        lift = Lift()

        self.assertIsNone(lift.status_age)
        self.assertFalse(lift.status_is_fresh(1.0))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            lift.status_is_fresh(-0.1)


class FakeCanBus:
    instances: list[FakeCanBus] = []

    def __init__(self, channel: str) -> None:
        self.channel = channel
        self.listener = None
        self.messages: list[tuple[int, bytes]] = []
        self.started = False
        self.closed = False
        self.instances.append(self)

    def _add_listener(self, listener) -> None:  # noqa: ANN001
        self.listener = listener

    async def start(self) -> None:
        self.started = True

    async def _send(self, arbitration_id: int, data: bytes) -> None:
        self.messages.append((arbitration_id, data))

    async def close(self) -> None:
        self.closed = True


class LiftStatusModeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        FakeCanBus.instances.clear()

    async def test_broadcast_mode_configures_200ms_and_close_quiets_board(
        self,
    ) -> None:
        with (
            patch.object(can_setup, "iface_up", return_value=True),
            patch.object(lift_module, "CanBus", FakeCanBus),
        ):
            lift = Lift("can-test", status_period_ms=200)
            await lift.start()
            bus = FakeCanBus.instances[-1]

            self.assertTrue(bus.started)
            self.assertEqual(
                bus.messages,
                [
                    (lift_module._ID_CMD, b"\x05\x00\x00"),
                    (lift_module._ID_CMD, b"\x04"),
                    (lift_module._ID_CMD, b"\x05\xc8\x00"),
                ],
            )

            await lift.close()

        self.assertTrue(bus.closed)
        self.assertEqual(
            bus.messages[-2:],
            [
                (lift_module._ID_CMD, b"\x06\x00\x00"),
                (lift_module._ID_CMD, b"\x05\x00\x00"),
            ],
        )

    async def test_default_run_keeps_explicit_status_polling(self) -> None:
        lift = Lift()
        lift._send = AsyncMock()  # type: ignore[method-assign]
        sleep = AsyncMock(side_effect=asyncio.CancelledError)

        with (
            patch.object(lift_module.asyncio, "sleep", sleep),
            self.assertRaises(asyncio.CancelledError),
        ):
            await lift._run()

        lift._send.assert_awaited_once_with(lift_module._OP_GET_STATUS)

    async def test_broadcast_run_does_not_poll(self) -> None:
        lift = Lift(status_period_ms=200)
        lift._last_status_monotonic = lift_module.time.monotonic()
        lift._send = AsyncMock()  # type: ignore[method-assign]
        sleep = AsyncMock(side_effect=asyncio.CancelledError)

        with (
            patch.object(lift_module.asyncio, "sleep", sleep),
            self.assertRaises(asyncio.CancelledError),
        ):
            await lift._run()

        lift._send.assert_not_awaited()

    async def test_stale_broadcast_mode_retries_rate_and_polls(self) -> None:
        lift = Lift(status_period_ms=200)
        lift._send = AsyncMock()  # type: ignore[method-assign]
        sleep = AsyncMock(side_effect=asyncio.CancelledError)

        with (
            patch.object(lift_module.asyncio, "sleep", sleep),
            self.assertRaises(asyncio.CancelledError),
        ):
            await lift._run()

        self.assertEqual(
            lift._send.await_args_list,
            [
                call(lift_module._OP_SET_RATE, struct.pack("<H", 200)),
                call(lift_module._OP_GET_STATUS),
            ],
        )

    async def test_status_period_can_be_changed_while_connected(self) -> None:
        lift = Lift()
        lift._bus = SimpleNamespace()
        lift._send = AsyncMock()  # type: ignore[method-assign]

        await lift.set_status_period(200)

        self.assertEqual(lift.status_period_ms, 200)
        lift._send.assert_awaited_once_with(
            lift_module._OP_SET_RATE, struct.pack("<H", 200)
        )

    async def test_close_still_disables_broadcast_if_stop_send_fails(self) -> None:
        bus = SimpleNamespace(close=AsyncMock())
        lift = Lift(status_period_ms=200)
        lift._bus = bus
        lift._send = AsyncMock(side_effect=[OSError("stop failed"), None])  # type: ignore[method-assign]

        await lift.close()

        self.assertEqual(
            lift._send.await_args_list,
            [
                call(lift_module._OP_JOG, struct.pack("<h", 0)),
                call(lift_module._OP_SET_RATE, struct.pack("<H", 0)),
            ],
        )
        bus.close.assert_awaited_once_with()

    async def test_rejects_out_of_range_status_periods(self) -> None:
        with self.assertRaisesRegex(ValueError, "0 and 65535"):
            Lift(status_period_ms=-1)
        with self.assertRaisesRegex(ValueError, "0 and 65535"):
            await Lift().set_status_period(65536)


if __name__ == "__main__":
    unittest.main()
