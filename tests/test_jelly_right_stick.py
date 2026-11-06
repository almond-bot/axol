"""The headset's right stick: turn on left/right, lift on forward/back.

Pressing a stick in to run the lift also tipped it sideways and turned the
base. Outside box mode the lift is on the right stick's forward/back, and each
push is locked to one job (turn or lift), chosen once it is well under way;
only a push close to straight forward/back runs the lift.
Box mode keeps the lift on held stick clicks, its forward/back being the grip
width.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from almond_axol.robot.jelly import (
    DOWN,
    LIFT_STICK_THRESHOLD,
    STOP,
    UP,
    Jelly,
    JellyConfig,
)


def _frame(**sticks):
    frame = SimpleNamespace(
        reset=False,
        l_stick_x=0.0,
        l_stick_y=0.0,
        r_stick_x=0.0,
        r_stick_y=0.0,
        l_stick_click=False,
        r_stick_click=False,
    )
    for key, value in sticks.items():
        setattr(frame, key, value)
    return frame


class RightStickTest(unittest.TestCase):
    def setUp(self) -> None:
        self.jelly = Jelly(JellyConfig(lift=False, channel=None, stick_expo=0.0))

    def _feed(self, box_mode: bool = False, resetting: bool = False, **sticks):
        self.jelly.apply_vr_frame(
            _frame(**sticks), resetting=resetting, box_mode=box_mode
        )
        vx, vy, wz, lift = self.jelly._target
        return wz, lift

    def test_forward_raises_and_back_lowers(self) -> None:
        self.assertEqual(self._feed(r_stick_y=-1.0), (0.0, UP))
        self._feed()
        self.assertEqual(self._feed(r_stick_y=1.0), (0.0, DOWN))

    def test_a_lift_push_never_turns_the_base(self) -> None:
        # Starts straight forward, then drifts well sideways mid-push.
        self.assertEqual(self._feed(r_stick_y=-0.3), (0.0, STOP))
        self.assertEqual(self._feed(r_stick_x=0.2, r_stick_y=-0.8), (0.0, UP))
        self.assertEqual(self._feed(r_stick_x=0.7, r_stick_y=-0.6), (0.0, UP))

    def test_a_turn_never_runs_the_lift(self) -> None:
        wz, lift = self._feed(r_stick_x=0.4)
        self.assertLess(wz, 0.0)
        self.assertEqual(lift, STOP)
        wz, lift = self._feed(r_stick_x=0.5, r_stick_y=-0.9)
        self.assertLess(wz, 0.0)
        self.assertEqual(lift, STOP)

    def test_a_turn_that_leaves_centre_leaning_forward_still_turns(self) -> None:
        # Just outside the deadzone the thumb reads more forward than
        # sideways; by the time the push is decided it is a turn.
        self.assertEqual(self._feed(r_stick_x=0.1, r_stick_y=-0.16), (0.0, STOP))
        wz, lift = self._feed(r_stick_x=0.5, r_stick_y=-0.3)
        self.assertLess(wz, 0.0)
        self.assertEqual(lift, STOP)
        wz, lift = self._feed(r_stick_x=0.6, r_stick_y=-0.7)
        self.assertLess(wz, 0.0)
        self.assertEqual(lift, STOP)

    def test_a_diagonal_push_turns(self) -> None:
        wz, lift = self._feed(r_stick_x=-0.6, r_stick_y=-0.7)
        self.assertGreater(wz, 0.0)
        self.assertEqual(lift, STOP)

    def test_centring_the_stick_frees_it_for_the_other_job(self) -> None:
        self._feed(r_stick_x=0.8)
        self._feed(r_stick_x=0.05, r_stick_y=0.05)
        self.assertEqual(self._feed(r_stick_y=-0.9), (0.0, UP))

    def test_lift_needs_a_deliberate_push(self) -> None:
        below = -(LIFT_STICK_THRESHOLD - 0.05)
        self.assertEqual(self._feed(r_stick_y=below), (0.0, STOP))
        self.assertEqual(self._feed(r_stick_y=-LIFT_STICK_THRESHOLD), (0.0, UP))

    def test_clicks_do_not_run_the_lift_outside_box_mode(self) -> None:
        self.assertEqual(self._feed(r_stick_click=True), (0.0, STOP))
        self._feed()
        self.assertEqual(self._feed(l_stick_click=True), (0.0, STOP))

    def test_box_mode_runs_the_lift_on_clicks_and_ignores_forward_back(self) -> None:
        self.assertEqual(self._feed(box_mode=True, r_stick_y=-1.0), (0.0, STOP))
        self.assertEqual(
            self._feed(box_mode=True, r_stick_y=-1.0, r_stick_click=True), (0.0, UP)
        )
        self.assertEqual(self._feed(box_mode=True, l_stick_click=True), (0.0, DOWN))
        wz, lift = self._feed(box_mode=True, r_stick_x=-1.0, r_stick_y=0.4)
        self.assertAlmostEqual(wz, 1.0)
        self.assertEqual(lift, STOP)

    def test_a_width_push_held_out_of_box_mode_waits_for_centre(self) -> None:
        self._feed(box_mode=True, r_stick_y=-1.0)
        self.assertEqual(self._feed(r_stick_y=-1.0), (0.0, STOP))
        self._feed()
        self.assertEqual(self._feed(r_stick_y=-1.0), (0.0, UP))

    def test_a_push_held_through_a_reset_waits_for_centre(self) -> None:
        self.assertEqual(self._feed(resetting=True, r_stick_y=-1.0), (0.0, STOP))
        self.assertEqual(self._feed(r_stick_y=-1.0), (0.0, STOP))
        self.assertEqual(self._feed(r_stick_x=-1.0, r_stick_y=-1.0), (0.0, STOP))
        self._feed()
        self.assertEqual(self._feed(r_stick_y=-1.0), (0.0, UP))


if __name__ == "__main__":
    unittest.main()
