"""Host-side calibration of Mantis trigger full-squeeze endpoints."""

from __future__ import annotations

import math
import queue
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from almond_axol.tracker.trigger import (
    TRIGGER_CAN_ID,
    TriggerReader,
    encode_trigger_frame,
    position_to_grip,
)


class _Bus:
    def __init__(self, position: float) -> None:
        self.frames: queue.Queue[SimpleNamespace] = queue.Queue()
        self.frames.put(
            SimpleNamespace(
                arbitration_id=TRIGGER_CAN_ID,
                is_extended_id=False,
                data=encode_trigger_frame(position),
            )
        )

    def recv(self, timeout: float) -> SimpleNamespace | None:
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def shutdown(self) -> None:
        pass


class TriggerFullScaleTest(unittest.TestCase):
    def test_default_mapping_and_calibrated_endpoint(self) -> None:
        self.assertAlmostEqual(position_to_grip(0.92), 0.08)
        self.assertAlmostEqual(position_to_grip(0.45, full_scale=0.90), 0.5)
        self.assertEqual(position_to_grip(0.90, full_scale=0.90), 0.0)
        self.assertEqual(position_to_grip(0.92, full_scale=0.90), 0.0)
        self.assertEqual(position_to_grip(0.0, full_scale=0.90), 1.0)

    def test_readers_can_use_different_endpoints_for_same_can_position(self) -> None:
        left = TriggerReader("can-left", full_scale=0.90, bus=_Bus(0.92))
        right = TriggerReader("can-right", bus=_Bus(0.92))
        try:
            deadline = time.monotonic() + 1.0
            while (
                left.grip() is None or right.grip() is None
            ) and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertEqual(left.grip(), 0.0)
            self.assertAlmostEqual(right.grip(), 0.08, places=5)
        finally:
            left.close()
            right.close()

    def test_invalid_endpoint_rejected_before_reader_start(self) -> None:
        for bad in (
            0,
            0.49,
            -0.1,
            1.1,
            math.inf,
            -math.inf,
            math.nan,
            None,
            "0.9",
            True,
        ):
            with self.subTest(full_scale=bad):
                with self.assertRaisesRegex(ValueError, "full_scale"):
                    position_to_grip(0.5, full_scale=bad)
                bus = Mock()
                with self.assertRaisesRegex(ValueError, "full_scale"):
                    TriggerReader("can-test", full_scale=bad, bus=bus)
                bus.recv.assert_not_called()


if __name__ == "__main__":
    unittest.main()
