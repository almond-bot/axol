"""The limp contact hold waits for a reset press — and nothing else.

A contact trip drops the arms into gravity comp. The hold used to time out
after the VR frame stream had been dead for a grace period and stiffen the
arms in place ("orphaned hold"); it no longer does. Without a headset the
arms simply stay limp, hand-guidable, until the operator is back in VR and
presses reset.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import unittest

import numpy as np

from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore


def _core() -> VRTeleopCore:
    # A fast hold cadence so the tests below spin many cycles quickly.
    return VRTeleopCore(
        VRTeleopConfig(frequency=2000.0),
        logging.getLogger("test"),
        broadcast_tracking=lambda _enabled: None,
    )


class ContactHoldTest(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_hold_has_no_timeout_without_a_headset(self) -> None:
        """No VR frames ever arrive; the hold keeps applying gravity comp."""
        core = _core()
        gravity_calls = 0
        resyncs: list[None] = []
        announced: list[str] = []

        async def gravity_step() -> None:
            nonlocal gravity_calls
            gravity_calls += 1

        async def scenario() -> str:
            task = asyncio.create_task(
                core._contact_hold_until_reset(
                    gravity_step=gravity_step,
                    reset_command_state=lambda: resyncs.append(None),
                    get_positions=lambda: (np.zeros(8), np.zeros(8)),
                    stopped=lambda: False,
                    announce=announced.append,
                    on_contact=None,
                    hold_tick=None,
                )
            )
            # Many hold cycles with no headset and no reset: still limp.
            while gravity_calls < 300:
                await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertEqual(resyncs, [])  # position control never resumed
            # The operator comes back and presses X.
            core.request_reset()
            return await task

        outcome = self._run(scenario())
        self.assertEqual(outcome, "reset")
        self.assertEqual(len(resyncs), 1)
        self.assertTrue(core.reset_pending)  # left latched for the reset path
        self.assertFalse(core._ik_paused)
        self.assertNotIn("holding position here", " ".join(announced).lower())

    def test_hold_ends_only_when_the_flow_stops(self) -> None:
        core = _core()
        stop = False
        resyncs: list[None] = []

        async def gravity_step() -> None:
            nonlocal stop
            stop = True  # the flow is torn down mid-hold

        outcome = self._run(
            core._contact_hold_until_reset(
                gravity_step=gravity_step,
                reset_command_state=lambda: resyncs.append(None),
                get_positions=lambda: (np.zeros(8), np.zeros(8)),
                stopped=lambda: stop,
                announce=lambda _msg: None,
                on_contact=None,
                hold_tick=None,
            )
        )
        self.assertEqual(outcome, "stopped")
        self.assertEqual(resyncs, [])

    def test_no_liveness_hook_is_accepted(self) -> None:
        """The orphan mechanism is gone: no caller can opt back into it."""
        for fn in (
            VRTeleopCore.guarded_return,
            VRTeleopCore.contact_hold,
            VRTeleopCore._contact_hold_until_reset,
        ):
            self.assertNotIn("vr_alive", inspect.signature(fn).parameters)


if __name__ == "__main__":
    unittest.main()
