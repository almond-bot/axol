"""Jelly command-source safety on the Python side of the bridge.

The wheel loop itself lives in the Rust core (``axol-rt jelly``); what Python
owns is the latched target's freshness and the lift. A dead command source
must suspend the lift (one STOP, then CAN silence so the jelly_legs deadman is
the safety layer), and a frame handed over twice must not count as fresh.
"""

from __future__ import annotations

import asyncio
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from almond_axol.robot import jelly as jelly_module
from almond_axol.robot.jelly import Jelly, JellyConfig, _pack_config
from almond_axol.robot.lift import STOP, UP


def _fast_config(**overrides) -> JellyConfig:
    """A config the bridge loop can be exercised against with real sleeps."""
    base = dict(
        channel=None,  # no Rust core: the bridge only drives the lift
        lift=False,
        yaw_hold_gain=0.0,
        frequency=200.0,  # 5 ms cycles
        command_timeout=0.04,
    )
    base.update(overrides)
    return JellyConfig(**base)


class JellyBridgeLiftStreamTest(unittest.IsolatedAsyncioTestCase):
    """Drive the real bridge loop against a mock lift."""

    def _start(self, jelly: Jelly, lift: SimpleNamespace) -> asyncio.Task:
        jelly._lift = lift
        return asyncio.create_task(jelly._bridge_loop())

    async def _settle(self, cycles: float, jelly: Jelly) -> None:
        await asyncio.sleep(cycles / jelly.config.frequency)

    async def test_no_lift_frames_without_a_command_source(self) -> None:
        jelly = Jelly(_fast_config())
        lift = SimpleNamespace(command=Mock(), suspend=Mock())
        task = self._start(jelly, lift)
        try:
            await self._settle(4, jelly)
            # Never attached: the lift is neither streamed nor suspended.
            lift.command.assert_not_called()
            lift.suspend.assert_not_called()
            self.assertEqual(jelly.lift_dir, STOP)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        # Teardown releases the lift once, so a held jog cannot outlive the
        # bridge (Lift._run would otherwise retransmit it forever).
        lift.command.assert_called_once_with(STOP)

    async def test_stale_edge_suspends_the_lift_then_stays_quiet(self) -> None:
        jelly = Jelly(_fast_config())
        lift = SimpleNamespace(command=Mock(), suspend=Mock())
        task = self._start(jelly, lift)
        try:
            jelly.set_command(0.0, 0.0, 0.0, UP)
            await self._settle(2, jelly)
            lift.command.assert_called_with(UP)
            lift.suspend.assert_not_called()
            self.assertEqual(jelly.lift_dir, UP)

            await asyncio.sleep(
                jelly.config.command_timeout + 4 / jelly.config.frequency
            )
            lift.suspend.assert_called_once()
            calls_after_edge = lift.command.call_count
            await self._settle(4, jelly)
            # Silent: not even command(STOP) is repeated to the lift.
            self.assertEqual(lift.command.call_count, calls_after_edge)
            self.assertEqual(jelly.lift_dir, STOP)

            # A returning source re-attaches the lift and streams again.
            jelly.set_command(0.0, 0.0, 0.0, UP)
            await self._settle(2, jelly)
            self.assertGreater(lift.command.call_count, calls_after_edge)
            lift.command.assert_called_with(UP)
            lift.suspend.assert_called_once()
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task


class JellyVrFrameFreshnessTest(unittest.TestCase):
    def test_same_frame_object_does_not_refresh_the_deadline(self) -> None:
        jelly = Jelly(JellyConfig(lift=False, channel=None))
        frame = SimpleNamespace(
            reset=False,
            l_stick_x=0.0,
            l_stick_y=-1.0,
            r_stick_x=0.0,
            l_stick_click=False,
            r_stick_click=False,
        )
        with patch.object(jelly_module.time, "monotonic", return_value=100.0):
            jelly.apply_vr_frame(frame)
        self.assertEqual(jelly._target_time, 100.0)
        self.assertEqual(jelly._target[0], 1.0)

        # A poller handing the server's latest frame over again is not a
        # fresh command: the deadline stays where the real frame put it.
        with patch.object(jelly_module.time, "monotonic", return_value=100.5):
            jelly.apply_vr_frame(frame)
        self.assertEqual(jelly._target_time, 100.0)

        fresh = SimpleNamespace(**vars(frame))
        with patch.object(jelly_module.time, "monotonic", return_value=100.5):
            jelly.apply_vr_frame(fresh)
        self.assertEqual(jelly._target_time, 100.5)


class JellyCanTimeoutConfigTest(unittest.IsolatedAsyncioTestCase):
    """The wheel loss-of-comms alarm is the runaway safety layer: a config
    that disables it, or that the command stream cannot reliably feed, is
    refused before the Rust core (and any wheel) is touched."""

    async def _assert_refused(self, message: str, **overrides) -> None:
        cfg = JellyConfig(channel="can9", lift=False, **overrides)
        jelly = Jelly(cfg)
        with (
            patch.object(jelly_module.subprocess, "Popen") as popen,
            self.assertRaisesRegex(ValueError, message),
        ):
            await jelly.enable()
        popen.assert_not_called()

    async def test_zero_timeout_is_refused(self) -> None:
        await self._assert_refused("cannot be disabled", can_timeout_ms=0.0)

    async def test_negative_and_nan_timeouts_are_refused(self) -> None:
        await self._assert_refused("cannot be disabled", can_timeout_ms=-50.0)
        await self._assert_refused("cannot be disabled", can_timeout_ms=float("nan"))

    async def test_timeout_shorter_than_two_periods_is_refused(self) -> None:
        # 50 Hz → 20 ms period; 39 ms would trip on a single late cycle.
        await self._assert_refused(
            "at least twice the command period", frequency=50.0, can_timeout_ms=39.0
        )

    async def test_command_timeout_must_be_positive(self) -> None:
        await self._assert_refused("command_timeout", command_timeout=0.0)

    def test_default_timeout_is_200ms(self) -> None:
        self.assertEqual(JellyConfig().can_timeout_ms, 200.0)


class JellyRustWireTest(unittest.IsolatedAsyncioTestCase):
    """The IPC layout shared with ``rust/axol-rt/src/jelly.rs``."""

    def test_config_message_carries_eleven_values_ending_in_can_timeout(self) -> None:
        cfg = JellyConfig(frequency=50.0, command_timeout=0.2, can_timeout_ms=200.0)
        payload = _pack_config(cfg)
        self.assertEqual(payload[:1], b"C")
        # 11 f64 = 88 bytes, the length the Rust ``parse_config`` insists on.
        self.assertEqual(len(payload) - 1, 88)
        values = struct.unpack("<11d", payload[1:])
        self.assertEqual(values[8:], (50.0, 0.2, 200.0))

    async def test_status_packet_exposes_link_and_wheel_fault_flags(self) -> None:
        jelly = Jelly(JellyConfig(lift=False, channel=None))
        reader = asyncio.StreamReader()
        jelly._reader = reader

        def status(flags: int) -> bytes:
            payload = b"U" + struct.pack("<10dB", *([0.0] * 10), flags)
            return struct.pack("<I", len(payload)) + payload

        # parked | linked, then wheel_fault alone (source gone, wheel tripped).
        reader.feed_data(status(1 | 8))
        reader.feed_data(status(16))
        reader.feed_eof()
        await jelly._rust_reader_loop()
        self.assertFalse(jelly.parked)
        self.assertFalse(jelly.linked)
        self.assertTrue(jelly.wheel_fault)

        jelly._reader = reader = asyncio.StreamReader()
        reader.feed_data(status(1 | 8))
        reader.feed_eof()
        await jelly._rust_reader_loop()
        self.assertTrue(jelly.parked)
        self.assertTrue(jelly.linked)
        self.assertFalse(jelly.wheel_fault)


if __name__ == "__main__":
    unittest.main()
