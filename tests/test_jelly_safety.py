"""Jelly command-source safety on the Python side of the bridge.

The wheel loop itself lives in the Rust core (``axol-rt jelly``); what Python
owns is the latched target's freshness and the lift. A dead command source
must suspend the lift (one STOP, then CAN silence so the jelly_legs deadman is
the safety layer), and a frame handed over twice must not count as fresh.
"""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from almond_axol.robot import jelly as jelly_module
from almond_axol.robot.jelly import Jelly, JellyConfig
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


if __name__ == "__main__":
    unittest.main()
