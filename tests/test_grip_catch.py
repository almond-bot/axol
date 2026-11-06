"""Re-engaging a frozen arm keeps its gripper's grasp until the trigger catches it.

A frozen arm holds its grasp whatever the trigger does. Clicking its grip to
resume used to hand the gripper straight back to the trigger, so a released
trigger dropped what the arm held. Now the gripper keeps its grasp until the
trigger is squeezed back to it, then follows it (see
``VRTeleopCore._follow_trigger``).
"""

from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace

from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore


def _frame(**overrides) -> SimpleNamespace:
    fields = {
        "l_lock": False,
        "r_lock": False,
        "l_grip": 1.0,
        "r_grip": 1.0,
        "stow": False,
        "lock_release_id": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


class GripCatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.core = VRTeleopCore(
            VRTeleopConfig(),
            logging.getLogger("test"),
            broadcast_tracking=lambda _enabled: None,
        )

    def _step(self, **overrides) -> None:
        self.core.update_engage(_frame(**overrides))

    def _hold_with_left_frozen(self) -> None:
        self._step(l_lock=True, r_lock=True)  # engage both
        self._step()
        self._step(l_grip=0.2)  # close the left gripper on something
        self._step(l_lock=True, l_grip=0.2)  # freeze the left arm
        self._step(l_grip=1.0)  # let the trigger go: a frozen arm keeps it
        self.assertFalse(self.core.left_enabled)
        self.assertAlmostEqual(self.core.l_grip, 0.2)

    def test_the_first_engage_follows_the_trigger(self) -> None:
        self._step(l_lock=True, r_lock=True, l_grip=0.9)
        self.assertAlmostEqual(self.core.l_grip, 0.9)

    def test_resuming_keeps_the_grasp_until_the_trigger_catches_it(self) -> None:
        self._hold_with_left_frozen()
        self._step(l_lock=True, l_grip=1.0)  # resume, trigger released
        self.assertTrue(self.core.left_enabled)
        self._step(l_grip=1.0)
        self._step(l_grip=0.6)
        self.assertAlmostEqual(self.core.l_grip, 0.2)
        self._step(l_grip=0.21)  # caught
        self.assertAlmostEqual(self.core.l_grip, 0.21)
        self._step(l_grip=0.7)  # and it follows the trigger out
        self.assertAlmostEqual(self.core.l_grip, 0.7)

    def test_a_trigger_already_at_the_grasp_takes_over(self) -> None:
        self._hold_with_left_frozen()
        self._step(l_lock=True, l_grip=0.1)
        self.assertAlmostEqual(self.core.l_grip, 0.1)


if __name__ == "__main__":
    unittest.main()
