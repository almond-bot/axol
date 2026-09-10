from __future__ import annotations

import unittest
from unittest import mock

from almond_axol.video.gst_zed import _dataset_enc_shmsink
from almond_axol.video.shm_frames import EncodedAuReader
from almond_axol.video.video_proc import (
    _gsth264_meta,
    _gsth264_transport_available,
)


class EncodedCaptureTimestampTest(unittest.TestCase):
    def test_transport_metadata_carries_the_pts_to_perf_counter_mapping(self) -> None:
        meta = _gsth264_meta("/tmp/camera.sock", 640, 480, 30, 60, 8_765.25)

        self.assertEqual(meta["capture_fps"], 60)
        self.assertEqual(meta["pts_perf_offset_s"], 8_765.25)
        self.assertEqual(meta["transport"], "gstshm-h264")

    def test_producer_wraps_access_units_in_gdp_and_waits_for_reader(self) -> None:
        branch = _dataset_enc_shmsink("/tmp/camera.sock", 640, 480, 60, "dsenc")

        self.assertIn("! gdppay !", branch)
        self.assertIn("wait-for-connection=true", branch)
        self.assertNotIn("wait-for-connection=false", branch)

    def test_consumer_depays_gdp_before_h264_parse(self) -> None:
        gst = mock.Mock()
        with mock.patch(
            "almond_axol.video.gst_zed._require_gst", return_value=(gst, None)
        ):
            reader = EncodedAuReader(
                "/tmp/camera.sock",
                640,
                480,
                60,
                pts_perf_offset_s=8_765.25,
            )

        pipeline = gst.parse_launch.call_args.args[0]
        self.assertIn("shmsrc ", pipeline)
        self.assertIn("! gdpdepay !", pipeline)
        self.assertLess(pipeline.index("gdpdepay"), pipeline.index("h264parse"))
        self.assertEqual(reader._pts_perf_offset_s, 8_765.25)

    def test_reader_connect_failure_rolls_pipeline_back_to_null(self) -> None:
        gst = mock.Mock()
        gst.State.PLAYING = "playing"
        gst.State.NULL = "null"
        gst.StateChangeReturn.FAILURE = "failure"
        pipeline = mock.Mock()
        pipeline.get_by_name.return_value = mock.Mock()
        pipeline.set_state.return_value = "success"
        gst.parse_launch.return_value = pipeline
        failed_thread = mock.Mock()
        failed_thread.start.side_effect = RuntimeError("thread start failed")
        failed_thread.is_alive.return_value = False

        with (
            mock.patch(
                "almond_axol.video.gst_zed._require_gst", return_value=(gst, None)
            ),
            mock.patch(
                "almond_axol.video.shm_frames.threading.Thread",
                return_value=failed_thread,
            ),
        ):
            reader = EncodedAuReader(
                "/tmp/camera.sock", 640, 480, 60, pts_perf_offset_s=0.0
            )
            with self.assertRaisesRegex(RuntimeError, "thread start failed"):
                reader.connect()

        self.assertEqual(
            pipeline.set_state.call_args_list,
            [mock.call("playing"), mock.call("null")],
        )
        self.assertIsNone(reader._thread)
        self.assertIsNone(reader._pipeline)
        self.assertIsNone(reader._sink)

    def test_reader_disconnect_nulls_first_and_retains_live_thread(self) -> None:
        gst = mock.Mock()
        gst.State.NULL = "null"
        gst.StateChangeReturn.FAILURE = "failure"
        pipeline = mock.Mock()
        pipeline.set_state.return_value = "success"
        gst.parse_launch.return_value = pipeline
        events: list[str] = []
        pipeline.set_state.side_effect = (
            lambda _state: events.append("null") or "success"
        )
        thread = mock.Mock()
        thread.is_alive.return_value = True
        thread.join.side_effect = lambda **_kwargs: events.append("join")

        with mock.patch(
            "almond_axol.video.gst_zed._require_gst", return_value=(gst, None)
        ):
            reader = EncodedAuReader(
                "/tmp/camera.sock", 640, 480, 60, pts_perf_offset_s=0.0
            )
        reader._thread = thread
        reader._sink = object()

        with self.assertRaisesRegex(RuntimeError, "ownership remains uncertain"):
            reader.disconnect()

        self.assertEqual(events, ["null", "join"])
        self.assertIs(reader._thread, thread)
        self.assertIs(reader._pipeline, pipeline)
        self.assertIsNotNone(reader._sink)

    def test_transport_requires_gdp_and_both_shm_elements(self) -> None:
        with mock.patch(
            "almond_axol.video.gst_zed._element_available", return_value=True
        ) as available:
            self.assertTrue(_gsth264_transport_available())
        self.assertEqual(
            [call.args[0] for call in available.call_args_list],
            ["shmsink", "shmsrc", "gdppay", "gdpdepay"],
        )

        with mock.patch(
            "almond_axol.video.gst_zed._element_available",
            side_effect=lambda element: element != "gdpdepay",
        ):
            self.assertFalse(_gsth264_transport_available())


if __name__ == "__main__":
    unittest.main()
