from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from almond_axol.cli.tracker_bridge import _wait_for_live_inputs
from almond_axol.tracker.base import (
    TRACKER_PAIR_MAX_SKEW_S,
    TRACKER_POSE_MAX_AGE_S,
)


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
