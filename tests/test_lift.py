from __future__ import annotations

import asyncio
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

import can

from almond_axol.cli.can import setup as can_setup
from almond_axol.constants import CAN_BASE, CAN_CHEST
from almond_axol.motor import CanBus
from almond_axol.robot import lift as lift_module
from almond_axol.robot.lift import (
    Lift,
    LiftStatus,
    _decode_status,
    resolve_lift_channel,
)


class ResolveLiftChannelTest(unittest.TestCase):
    """The lift follows whichever bus ``axol can.setup`` pinned it to."""

    def _resolve(self, present: tuple[str, ...], channel: str | None = None) -> str:
        with tempfile.TemporaryDirectory() as directory:
            for name in present:
                (Path(directory) / name).mkdir()
            with patch.object(lift_module, "_SYS_NET", Path(directory)):
                return resolve_lift_channel(channel)

    def test_explicit_channel_wins(self) -> None:
        self.assertEqual(self._resolve((CAN_CHEST, CAN_BASE), "can9"), "can9")
        self.assertEqual(self._resolve((), "can9"), "can9")

    def test_own_chest_bus_when_present(self) -> None:
        self.assertEqual(self._resolve((CAN_CHEST,)), CAN_CHEST)
        self.assertEqual(self._resolve((CAN_CHEST, CAN_BASE)), CAN_CHEST)

    def test_shares_the_wheel_bus_when_there_is_no_chest_bus(self) -> None:
        self.assertEqual(self._resolve((CAN_BASE,)), CAN_BASE)

    def test_neither_present_names_the_canonical_chest_bus(self) -> None:
        # start() then fails with "interface not found: can_alm_axol_c",
        # pointing at can.setup rather than at the wheel bus.
        self.assertEqual(self._resolve(()), CAN_CHEST)

    def test_lift_resolves_its_channel_at_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / CAN_BASE).mkdir()
            with patch.object(lift_module, "_SYS_NET", Path(directory)):
                self.assertEqual(Lift().channel, CAN_BASE)
                self.assertEqual(Lift("can-test").channel, "can-test")


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


class _FakeProxyProcess:
    """A proxy child that has already exited; ``close`` must still reap it."""

    def __init__(self) -> None:
        self.reaped = False

    def poll(self) -> int:
        self.reaped = True
        return 0


def _proxy_bus(
    reader_task: asyncio.Task | None, proc: _FakeProxyProcess | None
) -> CanBus:
    """A ``CanBus`` in the open state without a spawned axol-rt proxy."""
    bus = object.__new__(CanBus)
    bus._channel = "can-test"
    bus._stalled = False
    bus._socket_path = "/tmp/axol-can-test-does-not-exist.sock"
    bus._proc = proc
    bus._reader = None
    bus._writer = None
    bus._reader_task = reader_task
    bus._listeners = []
    bus._ready = asyncio.Event()
    bus._closed_reason = None
    bus._timing = None
    bus._experiment_waiter = None
    bus._state = "open"
    return bus


_real_asyncio_wait = asyncio.wait


async def _immediate_wait(tasks, *, timeout=None):  # noqa: ANN001, ANN201
    """``asyncio.wait`` with the reader grace period collapsed to one tick."""
    return await _real_asyncio_wait(tasks, timeout=0)


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
        proc = _FakeProxyProcess()
        bus = _proxy_bus(reader_task, proc)

        with patch("almond_axol.motor.bus.asyncio.wait", _immediate_wait):
            close_task = asyncio.create_task(bus.close())
            await reader_cancelling.wait()
            close_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await close_task

        # The proxy child is reaped and the bus is closed before the
        # cancellation propagates.
        self.assertIsNone(bus._proc)
        self.assertIsNone(bus._reader_task)
        self.assertEqual(bus._state, "closed")
        self.assertTrue(proc.reaped)

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
        proc = _FakeProxyProcess()
        bus = _proxy_bus(asyncio.create_task(asyncio.Event().wait()), proc)
        sent = AsyncMock(return_value=True)
        bus._send = sent  # type: ignore[method-assign]
        lift._bus = bus

        with patch("almond_axol.motor.bus.asyncio.wait", _immediate_wait):
            close_task = asyncio.create_task(lift.close())
            await command_cancelling.wait()
            close_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await close_task

        self.assertIsNone(lift._bus)
        self.assertTrue(proc.reaped)
        self.assertEqual(bus._state, "closed")
        self.assertEqual(sent.await_count, 2)

    async def test_unusable_can_bus_never_reports_a_frame_as_delivered(self) -> None:
        # The Rust-proxy bus fails closed: a frame on a bus that is closed,
        # never opened, or whose proxy died raises instead of returning a
        # silent False, so a one-shot STOP is never mistaken for delivered.
        for state, closed_reason in (
            ("closed", None),
            ("unopened", None),
            ("open", "axol-rt proxy for can-test exited"),
        ):
            with self.subTest(state=state, closed_reason=closed_reason):
                bus = _proxy_bus(None, None)
                bus._state = state
                bus._closed_reason = closed_reason
                bus._writer = (
                    None
                    if state != "open"
                    else SimpleNamespace(is_closing=lambda: False, write=Mock())
                )

                with self.assertRaises(can.CanOperationError):
                    await bus._send(0x420, b"\x02")
                if bus._writer is not None:
                    bus._writer.write.assert_not_called()

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

    async def _run_ticks(self, lift: Lift, ticks: int) -> None:
        """Run ``_run`` for ``ticks`` iterations (the sleep ends the loop)."""
        remaining = [ticks]

        async def sleep(_seconds: float) -> None:
            remaining[0] -= 1
            if remaining[0] <= 0:
                raise asyncio.CancelledError

        with (
            patch.object(lift_module.asyncio, "sleep", sleep),
            self.assertRaises(asyncio.CancelledError),
        ):
            await lift._run()

    def _motion_frames(self, send: AsyncMock) -> list:
        """Sent opcodes with the status poll filtered out."""
        return [
            c for c in send.await_args_list if c.args[0] != lift_module._OP_GET_STATUS
        ]

    async def test_detached_lift_sends_no_motion_frames(self) -> None:
        """No command source yet → nothing but the status poll on the bus."""
        lift = Lift()
        lift._send = AsyncMock()  # type: ignore[method-assign]

        await self._run_ticks(lift, 3)

        self.assertEqual(self._motion_frames(lift._send), [])
        self.assertFalse(lift.streaming)

    async def test_attached_source_streams_a_motion_frame_every_tick(self) -> None:
        lift = Lift(jog_speed=500)
        lift._send = AsyncMock()  # type: ignore[method-assign]

        lift.command(lift_module.UP)
        await self._run_ticks(lift, 3)
        jog = call(lift_module._OP_JOG, struct.pack("<h", 500))
        self.assertEqual(self._motion_frames(lift._send), [jog, jog, jog])
        self.assertTrue(lift.streaming)

        # Release: the canonical STOP, then an idle STOP keepalive every tick
        # for as long as the source keeps saying "stopped".
        lift._send.reset_mock()
        lift.command(lift_module.STOP)
        await self._run_ticks(lift, 3)
        stop = call(lift_module._OP_STOP)
        self.assertEqual(self._motion_frames(lift._send), [stop, stop, stop])

    async def test_idle_stream_never_cancels_a_one_shot_move(self) -> None:
        lift = Lift()
        lift._bus = SimpleNamespace()
        lift._send = AsyncMock()  # type: ignore[method-assign]
        lift.command(lift_module.STOP)  # a source is attached and idle
        await self._run_ticks(lift, 1)
        lift._send.reset_mock()

        await lift.set_position(500)
        lift._send.reset_mock()
        await self._run_ticks(lift, 3)

        self.assertEqual(self._motion_frames(lift._send), [])

    async def test_suspend_stops_once_then_goes_silent(self) -> None:
        lift = Lift()
        lift._send = AsyncMock()  # type: ignore[method-assign]
        lift.command(lift_module.DOWN)
        await self._run_ticks(lift, 2)
        lift._send.reset_mock()

        lift.suspend()
        await self._run_ticks(lift, 4)

        self.assertEqual(self._motion_frames(lift._send), [call(lift_module._OP_STOP)])
        self.assertFalse(lift.streaming)

        # Suspending an already-idle, already-stopped lift sends nothing.
        lift._send.reset_mock()
        lift.suspend()
        await self._run_ticks(lift, 2)
        self.assertEqual(self._motion_frames(lift._send), [])

        # A new command re-attaches the source and the stream resumes.
        lift.command(lift_module.UP)
        await self._run_ticks(lift, 1)
        self.assertEqual(
            self._motion_frames(lift._send),
            [call(lift_module._OP_JOG, struct.pack("<h", lift_module.JOG_SPEED))],
        )

    async def test_suspend_aborts_an_active_one_shot(self) -> None:
        lift = Lift()
        lift._bus = SimpleNamespace()
        lift._send = AsyncMock()  # type: ignore[method-assign]
        await lift.home()
        lift._send.reset_mock()

        lift.suspend()
        await self._run_ticks(lift, 2)

        self.assertEqual(self._motion_frames(lift._send), [call(lift_module._OP_STOP)])
        self.assertFalse(lift._one_shot_active)

    async def test_rejects_out_of_range_status_periods(self) -> None:
        with self.assertRaisesRegex(ValueError, "0 and 65535"):
            Lift(status_period_ms=-1)
        with self.assertRaisesRegex(ValueError, "0 and 65535"):
            await Lift().set_status_period(65536)


if __name__ == "__main__":
    unittest.main()
