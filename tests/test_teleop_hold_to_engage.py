"""``hold_to_engage``: per-arm dead-man grips, and the shutdown park's gate.

With the dead-man scheme each arm moves exactly while its own grip is held —
either grip alone engages its arm, from rest too — and a grip still held when
a forced disengage lands must be released once before it engages again. The
shutdown park (``begin_park``) then takes the arms away from the grips and
the headset's reset button altogether.
"""

from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace

import numpy as np

from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore
from almond_axol.teleop.filter import ResetInterpolator


def _frame(left: bool, right: bool, l_grip: float = 0.5, r_grip: float = 0.5):
    return SimpleNamespace(
        l_lock=left, r_lock=right, l_grip=l_grip, r_grip=r_grip, lock_release_id=None
    )


def _core(**config: object) -> VRTeleopCore:
    return VRTeleopCore(
        VRTeleopConfig(**config), logging.getLogger(__name__), lambda _enabled: None
    )


class HoldToEngageTest(unittest.TestCase):
    def test_one_grip_engages_only_its_arm_from_rest(self) -> None:
        core = _core(hold_to_engage=True)
        core.update_engage(_frame(True, False))
        self.assertTrue(core.left_enabled)
        self.assertFalse(core.right_enabled)
        self.assertTrue(core.teleop_enabled)

    def test_releasing_a_grip_freezes_that_arm_only(self) -> None:
        core = _core(hold_to_engage=True)
        core.update_engage(_frame(True, True))
        core.update_engage(_frame(False, True))
        self.assertFalse(core.left_enabled)
        self.assertTrue(core.right_enabled)
        # Holding it again resumes the arm on its own, no both-grips gate.
        core.update_engage(_frame(True, True))
        self.assertTrue(core.left_enabled)

    def test_the_other_arm_engages_independently_later(self) -> None:
        core = _core(hold_to_engage=True)
        core.update_engage(_frame(True, False))
        core.update_engage(_frame(True, True))
        self.assertTrue(core.left_enabled)
        self.assertTrue(core.right_enabled)

    def test_a_grip_held_through_a_forced_disengage_is_inhibited(self) -> None:
        core = _core(hold_to_engage=True)
        core.update_engage(_frame(True, True))
        core._disengage_all("test")  # noqa: SLF001 - reset / stale stream / contact
        # Still holding both: neither arm may resume on its own.
        core.update_engage(_frame(True, True))
        self.assertFalse(core.teleop_enabled)
        # Release one grip and hold it again: only that arm re-engages.
        core.update_engage(_frame(False, True))
        core.update_engage(_frame(True, True))
        self.assertTrue(core.left_enabled)
        self.assertFalse(core.right_enabled)

    def test_grippers_follow_the_trigger_only_while_engaged(self) -> None:
        core = _core(hold_to_engage=True)
        core.update_engage(_frame(True, False, l_grip=0.2, r_grip=0.9))
        self.assertEqual(core.l_grip, 0.2)
        self.assertEqual(core.r_grip, 0.0)

    def test_toggle_scheme_still_needs_both_grips_from_rest(self) -> None:
        core = _core(hold_to_engage=False)
        core.update_engage(_frame(True, False))
        self.assertFalse(core.teleop_enabled)
        core.update_engage(_frame(False, False))
        core.update_engage(_frame(True, True))
        self.assertTrue(core.left_enabled)
        self.assertTrue(core.right_enabled)
        # A click on one grip freezes that arm; another resumes it.
        core.update_engage(_frame(False, False))
        core.update_engage(_frame(True, False))
        self.assertFalse(core.left_enabled)
        self.assertTrue(core.right_enabled)


class ParkGateTest(unittest.TestCase):
    def test_parking_ignores_grips_and_the_headset_reset(self) -> None:
        core = _core(hold_to_engage=True)
        core.update_engage(_frame(True, True))
        core.begin_park()
        self.assertTrue(core.is_parking)
        core.update_engage(_frame(True, True))
        self.assertFalse(core.teleop_enabled)
        core.note_frame_reset(True)
        self.assertFalse(core.reset_pending)
        # The owner's own legs still latch.
        core.request_reset()
        self.assertTrue(core.reset_pending)

    def test_zero_leg_needs_the_park(self) -> None:
        core = _core()
        with self.assertRaises(RuntimeError):
            core.request_zero()
        core.begin_park()
        core.request_zero()
        self.assertTrue(core.reset_pending)

    def test_park_snapshots_the_commanded_grasp(self) -> None:
        core = _core()
        core.set_solution(np.zeros(14, np.float32), list(range(7)), list(range(7, 14)))
        core.seed_filters(np.zeros(8, np.float32), np.zeros(8, np.float32))
        core.set_initial_grips(0.3, 0.6)
        out = core.compute_output()
        self.assertIsNotNone(out)
        core.begin_park()
        self.assertEqual(core._park_grippers, (float(out[7]), float(out[15])))  # noqa: SLF001

    def test_a_fresh_worker_handshake_ends_the_park(self) -> None:
        core = _core()
        core.begin_park()
        core.request_zero()
        core.set_solution(np.zeros(14, np.float32), list(range(7)), list(range(7, 14)))
        self.assertFalse(core.is_parking)
        core.note_frame_reset(True)
        self.assertTrue(core.reset_pending)


class ResetInterpolatorHoldTest(unittest.TestCase):
    def test_grippers_ramp_open_unless_held(self) -> None:
        traj = [np.zeros(2), np.ones(2)]
        ramp = ResetInterpolator()
        ramp.set_trajectory(traj, 0.3, 0.6)
        _q, l_end, r_end, _done = [ramp.step() for _ in traj][-1]
        self.assertAlmostEqual(l_end, 1.0)
        self.assertAlmostEqual(r_end, 1.0)
        hold = ResetInterpolator()
        hold.set_trajectory(traj, 0.3, 0.6, hold_grippers=True)
        grips = {(lg, rg) for _q, lg, rg, _done in (hold.step() for _ in traj)}
        self.assertEqual(grips, {(0.3, 0.6)})


class ParkToZeroSettingTest(unittest.TestCase):
    def test_park_to_zero_is_a_curated_teleop_setting(self) -> None:
        from almond_axol.serve.settings import settings_schema

        teleop = next(
            s
            for c in settings_schema()
            for s in c["settings"]
            if s["key"] == "teleop.park_to_zero"
        )
        self.assertEqual(teleop["type"], "boolean")
        self.assertIs(teleop["default"], True)


if __name__ == "__main__":
    unittest.main()
