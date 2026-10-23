"""A take that is not saved keeps a persistent explanation on the panel."""

from __future__ import annotations

import threading
import unittest

from almond_axol.cli.collect_data import _NullCollectControl, _QueueCollectControl


class RejectionNoticeTest(unittest.TestCase):
    def test_rejection_survives_the_return_to_ready(self) -> None:
        control = _QueueCollectControl(threading.Event())
        control.note_recording()
        reason = "Episode 4 was REJECTED by the quality gate and discarded: stale."

        control.note_rejected(reason)
        control.note_returning()
        control.note_ready(4, "Episode 4: Mantis is ready.")

        snap = control.snapshot()
        self.assertEqual(snap["phase"], "ready")
        self.assertEqual(snap["message"], "Episode 4: Mantis is ready.")
        self.assertEqual(snap["notice"], {"level": "error", "message": reason})

    def test_next_take_clears_it(self) -> None:
        control = _QueueCollectControl(threading.Event())
        control.note_rejected("Episode 1 was NOT saved: encoder dropped frames")

        control.note_recording()

        self.assertNotIn("notice", control.snapshot())

    def test_a_saved_take_shows_no_notice(self) -> None:
        control = _QueueCollectControl(threading.Event())
        control.note_recording()
        control.note_saving()
        control.note_saved()
        control.note_ready(2)

        snap = control.snapshot()
        self.assertNotIn("notice", snap)
        self.assertEqual(snap["episodesRecorded"], 1)

    def test_plain_cli_control_accepts_it(self) -> None:
        _NullCollectControl().note_rejected("ignored")


if __name__ == "__main__":
    unittest.main()
