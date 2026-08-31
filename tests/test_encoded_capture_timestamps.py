from __future__ import annotations

import math
import unittest

from almond_axol.video.shm_frames import _capture_perf_from_receive
from almond_axol.video.video_proc import _gsth264_meta


class EncodedCaptureTimestampTest(unittest.TestCase):
    def test_valid_pipeline_latency_is_subtracted_from_receive_time(self) -> None:
        self.assertAlmostEqual(_capture_perf_from_receive(123.5, 0.037), 123.463)

    def test_unavailable_or_invalid_latency_falls_back_to_receive_time(self) -> None:
        for latency in (
            None,
            False,
            True,
            "invalid",
            -0.001,
            5.001,
            math.nan,
            math.inf,
            -math.inf,
        ):
            with self.subTest(latency=latency):
                self.assertEqual(_capture_perf_from_receive(123.5, latency), 123.5)

    def test_latency_cannot_predate_the_perf_counter_epoch(self) -> None:
        self.assertEqual(_capture_perf_from_receive(0.25, 0.5), 0.25)

    def test_five_second_safety_ceiling_is_inclusive(self) -> None:
        self.assertEqual(_capture_perf_from_receive(10.0, 5.0), 5.0)

    def test_transport_metadata_carries_relay_pipeline_latency(self) -> None:
        meta = _gsth264_meta("/tmp/camera.sock", 640, 480, 60, 0.025)

        self.assertEqual(meta["latency_s"], 0.025)
        self.assertEqual(meta["transport"], "gstshm-h264")


if __name__ == "__main__":
    unittest.main()
