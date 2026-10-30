"""Startup progress an operation reports to the control panel."""

import unittest

from almond_axol import progress


class ProgressTest(unittest.TestCase):
    def tearDown(self) -> None:
        tracker = progress.begin()
        progress.end(tracker)

    def test_reports_are_ignored_without_a_run(self) -> None:
        progress.plan(["arms", "ik"])
        progress.step("arms")
        progress.ready()  # no tracker: nothing to raise or record

    def test_walks_the_plan_to_ready(self) -> None:
        tracker = progress.begin()
        progress.plan(["arms", "ik", "vr"])
        self.assertEqual(tracker.to_dict()["current"], -1)
        progress.step("arms")
        progress.step("ik")
        state = tracker.to_dict()
        self.assertEqual(state["steps"], ["arms", "ik", "vr"])
        self.assertEqual(state["current"], 1)
        self.assertIsNotNone(state["stepStartedAt"])
        self.assertFalse(state["ready"])
        progress.ready()
        state = tracker.to_dict()
        self.assertTrue(state["ready"])
        self.assertIsNone(state["stepStartedAt"])

    def test_an_unplanned_step_goes_after_the_current_one(self) -> None:
        tracker = progress.begin()
        progress.plan(["arms", "vr"])
        progress.step("arms")
        progress.step("cameras")
        state = tracker.to_dict()
        self.assertEqual(state["steps"], ["arms", "cameras", "vr"])
        self.assertEqual(state["current"], 1)

    def test_never_moves_backwards_or_past_ready(self) -> None:
        tracker = progress.begin()
        progress.plan(["arms", "ik"])
        progress.step("ik")
        progress.step("arms")
        self.assertEqual(tracker.to_dict()["current"], 1)
        progress.ready()
        progress.step("vr")
        self.assertNotIn("vr", tracker.to_dict()["steps"])

    def test_a_later_plan_keeps_the_steps_already_taken(self) -> None:
        tracker = progress.begin()
        progress.plan(["arms", "ik"])
        progress.step("arms")
        progress.plan(["arms", "cameras", "ik"])
        self.assertEqual(tracker.to_dict()["steps"], ["arms", "cameras", "ik"])

    def test_a_finished_run_stops_receiving_reports(self) -> None:
        first = progress.begin()
        progress.end(first)
        progress.step("arms")
        self.assertEqual(first.to_dict()["steps"], [])
        second = progress.begin()
        progress.end(first)  # ending an old run leaves the new one alone
        progress.step("arms")
        self.assertEqual(second.to_dict()["steps"], ["arms"])


if __name__ == "__main__":
    unittest.main()
