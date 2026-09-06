from __future__ import annotations

import asyncio
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

from almond_axol.cli.can import setup as can_setup
from almond_axol.motor import CanBus
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

    async def _send(self, arbitration_id: int, data: bytes) -> bool:
        self.messages.append((arbitration_id, data))
        return True

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
                (lift_module._ID_CMD, b"\x02"),
                (lift_module._ID_CMD, b"\x05\x00\x00"),
            ],
        )

    async def test_start_rejects_dropped_status_rate_configuration(self) -> None:
        bus = SimpleNamespace(
            _add_listener=Mock(),
            start=AsyncMock(),
            _send=AsyncMock(return_value=False),
            close=AsyncMock(),
        )
        with (
            patch.object(can_setup, "iface_up", return_value=True),
            patch.object(lift_module, "CanBus", return_value=bus),
        ):
            lift = Lift("can-test", status_period_ms=200)
            with self.assertRaisesRegex(OSError, "was not delivered"):
                await lift.start()

        self.assertIsNone(lift._bus)
        self.assertIsNone(lift._task)
        bus.start.assert_awaited_once_with()
        bus.close.assert_awaited_once_with()

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

    async def test_close_surfaces_stop_failure_and_remains_retryable(self) -> None:
        bus = SimpleNamespace(close=AsyncMock())
        lift = Lift(status_period_ms=200)
        lift._bus = bus
        lift._send = AsyncMock(side_effect=[OSError("stop failed"), None])  # type: ignore[method-assign]

        with self.assertRaisesRegex(OSError, "stop failed"):
            await lift.close()

        self.assertEqual(
            lift._send.await_args_list,
            [
                call(lift_module._OP_STOP),
                call(lift_module._OP_SET_RATE, struct.pack("<H", 0)),
            ],
        )
        bus.close.assert_not_awaited()
        self.assertIs(lift._bus, bus)

        lift._send = AsyncMock()  # type: ignore[method-assign]
        await lift.close()

        bus.close.assert_awaited_once_with()
        self.assertIsNone(lift._bus)

    async def test_can_bus_close_finishes_shutdown_then_propagates_cancellation(
        self,
    ) -> None:
        reader_cancelling = asyncio.Event()

        async def reader() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                reader_cancelling.set()
                await asyncio.Event().wait()

        reader_task = asyncio.create_task(reader())
        socket = SimpleNamespace(shutdown=Mock())
        bus = object.__new__(CanBus)
        bus._reader_task = reader_task
        bus._bus = socket

        close_task = asyncio.create_task(bus.close())
        await reader_cancelling.wait()
        close_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await close_task

        socket.shutdown.assert_called_once_with()
        self.assertIsNone(bus._bus)

    async def test_lift_close_finishes_cleanup_then_propagates_cancellation(
        self,
    ) -> None:
        command_cancelling = asyncio.Event()

        async def command_task() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                command_cancelling.set()
                await asyncio.Event().wait()

        lift = Lift()
        lift._task = asyncio.create_task(command_task())
        socket = SimpleNamespace(send=Mock(), shutdown=Mock())
        bus = object.__new__(CanBus)
        bus._reader_task = asyncio.create_task(asyncio.Event().wait())
        bus._bus = socket
        bus._lost = False
        bus._stalled = False
        bus._enobufs_since = None
        lift._bus = bus

        close_task = asyncio.create_task(lift.close())
        await command_cancelling.wait()
        close_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await close_task

        self.assertIsNone(lift._bus)
        socket.shutdown.assert_called_once_with()
        self.assertEqual(socket.send.call_count, 2)

    async def test_lost_or_stalled_can_bus_reports_dropped_frames(self) -> None:
        for lost, stalled, socket_missing in (
            (True, False, False),
            (False, True, False),
            (False, False, True),
        ):
            with self.subTest(
                lost=lost, stalled=stalled, socket_missing=socket_missing
            ):
                socket = None if socket_missing else SimpleNamespace(send=Mock())
                bus = object.__new__(CanBus)
                bus._lost = lost
                bus._stalled = stalled
                bus._bus = socket

                delivered = await bus._send(0x420, b"\x02")

                self.assertFalse(delivered)
                if socket is not None:
                    socket.send.assert_not_called()

    async def test_dropped_stop_is_reported_pending_and_keeps_bus_open(self) -> None:
        bus = SimpleNamespace(
            _send=AsyncMock(return_value=False),
            close=AsyncMock(),
        )
        lift = Lift()
        lift._bus = bus
        lift._one_shot_active = True

        with self.assertRaisesRegex(OSError, "was not delivered"):
            await lift.close()

        self.assertTrue(lift._stop_requested)
        self.assertTrue(lift._one_shot_active)
        self.assertIs(lift._bus, bus)
        bus.close.assert_not_awaited()

    async def test_command_stop_cancels_one_shot_with_canonical_stop(self) -> None:
        for command in ("home", "set_position"):
            with self.subTest(command=command):
                lift = Lift()
                lift._bus = SimpleNamespace()
                lift._send = AsyncMock()  # type: ignore[method-assign]
                if command == "home":
                    await lift.home()
                else:
                    await lift.set_position(500)
                lift._send.reset_mock()
                lift.command(lift_module.STOP)

                with (
                    patch.object(
                        lift_module.asyncio,
                        "sleep",
                        AsyncMock(side_effect=asyncio.CancelledError),
                    ),
                    self.assertRaises(asyncio.CancelledError),
                ):
                    await lift._run()

                self.assertEqual(
                    lift._send.await_args_list[0],
                    call(lift_module._OP_STOP),
                )

    async def test_set_position_rejects_invalid_values_instead_of_clamping(
        self,
    ) -> None:
        lift = Lift()
        lift._bus = SimpleNamespace()
        lift._send = AsyncMock()  # type: ignore[method-assign]
        lift._direction = lift_module.UP
        lift._last_jog_sent = lift_module.UP
        lift._stop_requested = True

        for position in (-1, 1001, 1.5, True):
            with self.subTest(position=position), self.assertRaises(ValueError):
                await lift.set_position(position)  # type: ignore[arg-type]
        for speed in (-1, 65536, 1.5, True):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                await lift.set_position(500, speed)  # type: ignore[arg-type]

        lift._send.assert_not_awaited()
        self.assertEqual(lift._direction, lift_module.UP)
        self.assertEqual(lift._last_jog_sent, lift_module.UP)
        self.assertTrue(lift._stop_requested)

    async def test_jog_inputs_cannot_reverse_overspeed_or_kill_driver(self) -> None:
        for speed in (-1, 32768, 1.5, True):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                Lift(jog_speed=speed)  # type: ignore[arg-type]

        lift = Lift()
        lift._direction = lift_module.UP
        for direction in (-2, 2, 1.5, True):
            with self.subTest(direction=direction), self.assertRaises(ValueError):
                lift.command(direction)  # type: ignore[arg-type]
            self.assertEqual(lift._direction, lift_module.UP)

    async def test_set_position_rechecks_interlock_after_stop_before_move(self) -> None:
        lift = Lift()
        lift._bus = SimpleNamespace()
        lift._send = AsyncMock()  # type: ignore[method-assign]
        interlock = AsyncMock(side_effect=RuntimeError("clearance lost"))

        with self.assertRaisesRegex(RuntimeError, "clearance lost"):
            await lift.set_position(500, before_send=interlock)

        interlock.assert_awaited_once_with()
        lift._send.assert_awaited_once_with(lift_module._OP_STOP)
        self.assertFalse(lift._one_shot_active)

    async def test_broadcast_proof_mode_never_solicits_status(self) -> None:
        lift = Lift(status_period_ms=200)
        lift._recover_stale_broadcasts = False
        lift._send = AsyncMock()  # type: ignore[method-assign]

        with (
            patch.object(
                lift_module.asyncio,
                "sleep",
                AsyncMock(side_effect=asyncio.CancelledError),
            ),
            self.assertRaises(asyncio.CancelledError),
        ):
            await lift._run()

        lift._send.assert_not_awaited()

    async def test_rejects_out_of_range_status_periods(self) -> None:
        with self.assertRaisesRegex(ValueError, "0 and 65535"):
            Lift(status_period_ms=-1)
        with self.assertRaisesRegex(ValueError, "0 and 65535"):
            await Lift().set_status_period(65536)


if __name__ == "__main__":
    unittest.main()
