"""Jelly's safety board driver (``almond_axol.robot.safety``) and CLI.

The CAN bus is replaced by a small simulator of the almond_safety firmware's
command handling (circuits-py ``designs/almond_safety/firmware/firmware.c``):
every command but GET_VOLTAGES is answered with a status frame, a software
stop drops the enable relay and the latch, and a remote start pulses until
the contactor closes.
"""

from __future__ import annotations

import asyncio
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from almond_axol.cli import safety as safety_cli
from almond_axol.constants import CAN_BASE, CAN_CHEST
from almond_axol.robot import lift as lift_module
from almond_axol.robot import safety as safety_module
from almond_axol.robot.safety import (
    SafetyBoard,
    SafetyError,
    decode_status,
    decode_voltages,
    explain_start_failure,
    resolve_safety_channel,
)

CMD, STATUS, VOLTS = 0x440, 0x441, 0x442


class FakeFirmware:
    """The board side of the protocol, as the firmware implements it."""

    def __init__(self) -> None:
        self.estop_ok = True
        self.latched = True
        self.sw_enable = True
        self.cause = 0
        self.rate_ms = 50
        self.faults = 0
        self.supply_mv = 26400
        self.k2_stuck = False  # output stays on whatever K2 is told
        self.start_latches = True  # a remote-start pulse closes K1
        self.silent = False  # never answers (no board on this bus)
        self.drop_replies = 0  # lose this many replies first
        self.rstart_polls = 0  # status frames left before the pulse ends
        self.received: list[int] = []

    @property
    def output_on(self) -> bool:
        if self.k2_stuck:
            return self.estop_ok and self.latched
        return self.estop_ok and self.latched and self.sw_enable

    def status_frame(self) -> bytes:
        if self.rstart_polls:
            self.rstart_polls -= 1
            if not self.rstart_polls:
                if self.start_latches and self.estop_ok:
                    self.latched = True
                else:
                    self.faults |= 0x40
        state = (
            (0x01 if self.estop_ok else 0)
            | (0x02 if self.latched else 0)
            | (0x04 if self.output_on else 0)
            | (0x08 if self.sw_enable else 0)
            | (0x20 if self.rstart_polls else 0)
            | (0x80 if self.supply_mv >= 18000 else 0)
        )
        out_mv = 26000 if self.output_on else 30
        return bytes([state, self.faults, self.cause, 0x02]) + struct.pack(
            "<HH", self.supply_mv, out_mv
        )

    def handle(self, data: bytes) -> list[tuple[int, bytes]]:
        op = data[0]
        self.received.append(op)
        if op == 0x01:
            self.sw_enable, self.cause = False, 1
            self.latched = False
        elif op == 0x02:
            self.sw_enable, self.cause = True, 0
        elif op == 0x03:
            if self.estop_ok and not self.latched and not self.rstart_polls:
                self.sw_enable, self.cause = True, 0
                self.faults &= ~0x40
                self.rstart_polls = 2
        elif op == 0x05:
            self.rate_ms = struct.unpack_from("<H", data, 1)[0]
        elif op == 0x08:
            return [(VOLTS, struct.pack("<4H", self.supply_mv, 26300, 26200, 26000))]
        return [(STATUS, self.status_frame())]


class FakeBus:
    """Stands in for ``CanBus``: feeds commands to a FakeFirmware."""

    firmware: FakeFirmware

    def __init__(self, channel: str, *, publishes_stall: bool = True) -> None:
        self.channel = channel
        self.publishes_stall = publishes_stall
        self.listeners: list = []
        self.sent: list[tuple[int, bytes]] = []
        self.closed = False
        FakeBus.last = self

    def _add_listener(self, listener) -> None:  # noqa: ANN001
        self.listeners.append(listener)

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        self.closed = True

    async def _send(self, arbitration_id: int, data: bytes) -> bool:
        self.sent.append((arbitration_id, bytes(data)))
        fw = self.firmware
        if arbitration_id != CMD or fw.silent:
            return True
        replies = fw.handle(bytes(data))
        if fw.drop_replies:
            fw.drop_replies -= 1
            return True
        loop = asyncio.get_running_loop()
        for can_id, payload in replies:
            msg = SimpleNamespace(arbitration_id=can_id, data=payload)
            for listener in self.listeners:
                loop.call_soon(listener, msg)
        return True


class BoardTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.fw = FakeFirmware()
        FakeBus.firmware = self.fw
        patcher = patch.object(safety_module, "CanBus", FakeBus)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.board = SafetyBoard("can-test")
        await self.board.start(bring_up=False)
        self.bus: FakeBus = FakeBus.last

    async def asyncTearDown(self) -> None:
        await self.board.close()


class DecodeTest(unittest.TestCase):
    def test_status_bits(self) -> None:
        st = decode_status(
            bytes([0xEF, 0xC7, 0x02, 0x13]) + struct.pack("<HH", 26400, 25900)
        )
        self.assertTrue(st.estop_ok)
        self.assertTrue(st.latched)
        self.assertTrue(st.output_on)
        self.assertTrue(st.sw_enable)
        self.assertTrue(st.remote_start_active)
        self.assertTrue(st.heartbeat_on)
        self.assertTrue(st.supply_ok)
        self.assertTrue(st.k2_stuck and st.k2_no_close and st.k1_pole_b)
        self.assertTrue(st.relay_fault)
        self.assertTrue(st.remote_start_failed)
        self.assertTrue(st.heartbeat_lost)
        self.assertEqual(st.stop_cause, "heartbeat")
        self.assertEqual(st.firmware_version, "1.3")
        self.assertAlmostEqual(st.supply_volts, 26.4)
        self.assertAlmostEqual(st.output_volts, 25.9)

    def test_everything_off_reads_as_a_pressed_estop(self) -> None:
        st = decode_status(bytes(8))
        self.assertTrue(st.estop_pressed)
        self.assertTrue(st.software_stopped)
        self.assertFalse(st.output_on)
        self.assertFalse(st.relay_fault)
        self.assertEqual(st.stop_cause, "none")

    def test_unknown_stop_cause_keeps_its_code(self) -> None:
        self.assertEqual(decode_status(bytes([0, 0, 9, 0, 0, 0, 0, 0])).stop_cause, "9")

    def test_short_frames_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            decode_status(bytes(6))
        with self.assertRaises(ValueError):
            decode_voltages(bytes(4))

    def test_voltages(self) -> None:
        v = decode_voltages(struct.pack("<4H", 26400, 26300, 120, 50))
        self.assertAlmostEqual(v.supply, 26.4)
        self.assertAlmostEqual(v.estop_ok, 26.3)
        self.assertAlmostEqual(v.latch, 0.12)
        self.assertAlmostEqual(v.output, 0.05)


class ResolveChannelTest(unittest.TestCase):
    def _resolve(self, present: tuple[str, ...], channel: str | None = None) -> str:
        with tempfile.TemporaryDirectory() as directory:
            for name in present:
                (Path(directory) / name).mkdir()
            with patch.object(lift_module, "_SYS_NET", Path(directory)):
                return resolve_safety_channel(channel)

    def test_follows_the_lift_bus(self) -> None:
        self.assertEqual(self._resolve((CAN_CHEST, CAN_BASE)), CAN_CHEST)
        self.assertEqual(self._resolve((CAN_BASE,)), CAN_BASE)
        self.assertEqual(self._resolve((), "can9"), "can9")


class PollTest(BoardTestCase):
    async def test_start_quiets_the_broadcast_and_polls_with_set_rate(self) -> None:
        self.assertEqual(self.bus.sent[0], (CMD, bytes([0x05, 0, 0])))
        self.assertFalse(self.bus.publishes_stall)
        await asyncio.sleep(0.05)
        self.assertEqual(self.fw.rate_ms, 0)
        self.assertIsNotNone(self.board.status)
        self.assertTrue(self.board.status_is_fresh())
        # Polling never changes the board's state.
        self.assertEqual(set(self.fw.received), {0x05})

    async def test_close_sends_nothing(self) -> None:
        await asyncio.sleep(0.01)
        sent = len(self.bus.sent)
        await self.board.close()
        self.assertEqual(len(self.bus.sent), sent)
        self.assertTrue(self.bus.closed)
        self.assertFalse(self.board.running)

    async def test_poll_failure_is_reported(self) -> None:
        async def broken(*_args) -> bool:  # noqa: ANN002
            raise OSError("interface lost")

        self.bus._send = broken  # type: ignore[method-assign]
        for _ in range(100):
            if not self.board.running:
                break
            await asyncio.sleep(0.01)
        self.assertFalse(self.board.running)
        self.assertIsInstance(self.board.failure, OSError)


class StopTest(BoardTestCase):
    async def test_stop_is_confirmed(self) -> None:
        st = await self.board.stop()
        self.assertFalse(st.sw_enable)
        self.assertFalse(st.output_on)
        self.assertEqual(st.stop_cause, "can")
        self.assertIn(0x01, self.fw.received)

    async def test_lost_replies_are_covered_by_resending(self) -> None:
        self.fw.drop_replies = 3
        st = await self.board.stop()
        self.assertFalse(st.output_on)
        self.assertGreaterEqual(self.fw.received.count(0x01), 2)

    async def test_silent_board_raises(self) -> None:
        self.fw.silent = True
        with self.assertRaisesRegex(SafetyError, "did not confirm"):
            await self.board.stop(timeout=0.2)

    async def test_power_still_on_after_the_stop_raises(self) -> None:
        self.fw.k2_stuck = True
        # The stop unlatches K1 too; keep it latched so the output stays on.
        original = self.fw.handle

        def handle(data: bytes):  # noqa: ANN202
            replies = original(data)
            self.fw.latched = True
            return [
                (i, self.fw.status_frame() if i == STATUS else d) for i, d in replies
            ]

        self.fw.handle = handle  # type: ignore[method-assign]
        with self.assertRaisesRegex(SafetyError, "power is still on"):
            await self.board.stop(timeout=0.2)


class RemoteStartTest(BoardTestCase):
    async def test_start_after_a_software_stop(self) -> None:
        await self.board.stop()
        st = await self.board.remote_start()
        self.assertTrue(st.output_on)
        self.assertFalse(st.remote_start_active)
        self.assertTrue(st.sw_enable)
        self.assertEqual(self.fw.received.count(0x03), 1)

    async def test_already_on_sends_no_start(self) -> None:
        st = await self.board.remote_start()
        self.assertTrue(st.output_on)
        self.assertNotIn(0x03, self.fw.received)

    async def test_refused_while_the_estop_is_pressed(self) -> None:
        self.fw.estop_ok = False
        self.fw.latched = False
        with self.assertRaisesRegex(SafetyError, "E-Stop is pressed"):
            await self.board.remote_start()
        self.assertNotIn(0x03, self.fw.received)

    async def test_pulse_that_does_not_latch_explains_why(self) -> None:
        self.fw.latched = False
        self.fw.start_latches = False
        with self.assertRaisesRegex(SafetyError, "green on-switch"):
            await self.board.remote_start(timeout=0.3)

    async def test_low_battery_is_named(self) -> None:
        self.fw.latched = False
        self.fw.start_latches = False
        self.fw.supply_mv = 15000
        with self.assertRaisesRegex(SafetyError, "too low"):
            await self.board.remote_start(timeout=0.3)


class OtherCommandsTest(BoardTestCase):
    async def test_enable_rearms_without_switching_power_on(self) -> None:
        await self.board.stop()
        st = await self.board.enable()
        self.assertTrue(st.sw_enable)
        self.assertFalse(st.output_on)

    async def test_voltages(self) -> None:
        v = await self.board.read_voltages()
        self.assertAlmostEqual(v.supply, 26.4)

    async def test_heartbeat_range(self) -> None:
        await self.board.set_heartbeat(500)
        self.assertEqual(
            self.bus.sent[-1], (CMD, bytes([0x07]) + struct.pack("<H", 500))
        )
        with self.assertRaises(ValueError):
            await self.board.set_heartbeat(70000)


class ExplainTest(unittest.TestCase):
    def test_relay_fault_needs_service(self) -> None:
        st = decode_status(bytes([0x81, 0x02, 0, 0x02]) + struct.pack("<HH", 26000, 0))
        self.assertIn("relay fault", explain_start_failure(st))


class CliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fw = FakeFirmware()
        FakeBus.firmware = self.fw
        for target, value in ((safety_module, "CanBus"),):
            p = patch.object(target, value, FakeBus)
            p.start()
            self.addCleanup(p.stop)
        p = patch("almond_axol.cli.can.setup.iface_up", return_value=True)
        p.start()
        self.addCleanup(p.stop)

    def _run(self, command: str) -> None:
        import argparse

        parser = argparse.ArgumentParser()
        safety_cli.add_parser(parser.add_subparsers())
        args = parser.parse_args([command, "--channel", "can-test"])
        args.func(args)

    def test_stop_then_start(self) -> None:
        self._run("safety.stop")
        self.assertFalse(self.fw.sw_enable)
        self._run("safety.start")
        self.assertTrue(self.fw.output_on)

    def test_no_board_exits_with_a_message(self) -> None:
        self.fw.silent = True
        with self.assertRaisesRegex(SystemExit, "No safety board answered"):
            self._run("safety.status")

    def test_status_line(self) -> None:
        line = safety_cli.fmt_status(decode_status(FakeFirmware().status_frame()))
        self.assertIn("POWER ON", line)


if __name__ == "__main__":
    unittest.main()
