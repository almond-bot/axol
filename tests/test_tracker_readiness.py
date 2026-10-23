from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from almond_axol.cli import tracker_bridge
from almond_axol.cli.tracker_bridge import _wait_for_live_inputs
from almond_axol.tracker.base import (
    TRACKER_PAIR_MAX_SKEW_S,
    TRACKER_POSE_MAX_AGE_S,
)
from almond_axol.tracker.lighthouse_validation import LIGHTHOUSE_ACQUISITION_TIMEOUT_S


class _OffsetPoseSource:
    def __init__(
        self,
        samples: list[dict[str, tuple[float, bool]]],
    ) -> None:
        self._samples = samples
        self.calls = 0

    def poses(self):  # type: ignore[no-untyped-def]
        sample = self._samples[min(self.calls, len(self._samples) - 1)]
        self.calls += 1
        now = time.perf_counter()
        return {
            key: SimpleNamespace(
                pos=np.zeros(3),
                quat=np.array([0.0, 0.0, 0.0, 1.0]),
                t=now - age_s,
                tracking=tracking,
            )
            for key, (age_s, tracking) in sample.items()
        }


class ManagedTrackerReadinessTest(unittest.TestCase):
    def test_late_lighthouse_identities_can_acquire_with_progress(self) -> None:
        now = [0.0]
        source = _OffsetPoseSource([{"left": (0.0, True), "right": (0.0, True)}])
        source.tracking_quality = lambda: {
            "status": "ready" if now[0] >= 50.0 else "checking",
            "reasons": []
            if now[0] >= 50.0
            else ["Waiting for fresh station identities"],
        }

        def advance(delay: float) -> None:
            now[0] += delay

        with (
            patch.object(
                tracker_bridge.time, "perf_counter", side_effect=lambda: now[0]
            ),
            patch.object(tracker_bridge.time, "sleep", side_effect=advance),
            patch.object(tracker_bridge._logger, "info") as progress,
        ):
            _wait_for_live_inputs(
                source, "left", "right", {}, timeout_s=LIGHTHOUSE_ACQUISITION_TIMEOUT_S
            )
        self.assertGreaterEqual(now[0], 50.0)
        self.assertLess(now[0], 50.1)
        self.assertGreater(progress.call_count, 9)
        self.assertIn(
            "Waiting for fresh station identities", progress.call_args_list[0].args[-1]
        )

    def test_invalidated_calibration_fails_promptly_instead_of_waiting_two_minutes(
        self,
    ) -> None:
        source = _OffsetPoseSource([{"left": (0.0, True), "right": (0.0, True)}])
        source.tracking_quality = lambda: {
            "status": "error",
            "calibrationRequired": True,
            "reasons": ["Lighthouse calibration invalidated"],
        }
        with patch.object(tracker_bridge.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "calibration is no longer valid"):
                _wait_for_live_inputs(source, "left", "right", {}, timeout_s=120)
        sleep.assert_not_called()

    def test_requires_both_tracker_bindings(self) -> None:
        source = _OffsetPoseSource([{"right": (0.0, True)}])

        with self.assertRaisesRegex(RuntimeError, "left tracker is not bound"):
            _wait_for_live_inputs(
                source,
                None,
                "right",
                {},
                timeout_s=0.0,
            )

    def test_rejects_pose_accepted_by_old_half_second_window(self) -> None:
        source = _OffsetPoseSource([{"left": (0.2, True), "right": (0.0, True)}])

        with self.assertRaisesRegex(RuntimeError, "left tracker 'left' is stale"):
            _wait_for_live_inputs(
                source,
                "left",
                "right",
                {},
                timeout_s=0.0,
            )

    def test_rejects_fresh_but_unsynchronized_pair(self) -> None:
        skew_s = TRACKER_PAIR_MAX_SKEW_S + 0.02
        source = _OffsetPoseSource([{"left": (skew_s, True), "right": (0.0, True)}])

        with self.assertRaisesRegex(RuntimeError, "not synchronized.*maximum 50 ms"):
            _wait_for_live_inputs(
                source,
                "left",
                "right",
                {},
                timeout_s=0.0,
            )

    def test_waits_for_both_tracked_fresh_and_synchronized(self) -> None:
        source = _OffsetPoseSource(
            [
                {"left": (0.0, False), "right": (0.0, True)},
                {
                    "left": (TRACKER_POSE_MAX_AGE_S / 2.0, True),
                    "right": (TRACKER_POSE_MAX_AGE_S / 2.0, True),
                },
            ]
        )

        with patch("almond_axol.cli.tracker_bridge.time.sleep"):
            _wait_for_live_inputs(
                source,
                "left",
                "right",
                {},
                timeout_s=1.0,
            )

        self.assertEqual(source.calls, 2)


if __name__ == "__main__":
    unittest.main()
