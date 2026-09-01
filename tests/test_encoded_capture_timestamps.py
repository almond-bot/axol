from __future__ import annotations

import math
import threading
import time
import unittest
from unittest import mock

from almond_axol.recording import record_proc
from almond_axol.video.gst_zed import _dataset_enc_shmsink
from almond_axol.video.shm_frames import (
    EncodedAuReader,
    _capture_perf_from_gst_pts,
    _capture_perf_from_receive,
)
from almond_axol.video.video_proc import (
    _gsth264_meta,
    _gsth264_transport_available,
)


class EncodedCaptureTimestampTest(unittest.TestCase):
    def test_post_prime_timeout_discards_partial_row_and_rearms_idr_wait(
        self,
    ) -> None:
        prime_added = threading.Event()
        left_second_popped = threading.Event()
        release_timeout = threading.Event()
        stop = threading.Event()
        errors: list[str] = []
        frame_counter = {"n": 0}

        class Camera:
            def __init__(self, name: str) -> None:
                self.name = name
                self.reads = 0
                self.flushes = 0

            @property
            def pending(self) -> int:
                return 0

            def flush(self) -> None:
                self.flushes += 1

            def read_next_au(self, *, timeout_ms: float) -> tuple[bytes, float]:
                del timeout_ms
                self.reads += 1
                if self.reads == 1:
                    return f"{self.name}-idr".encode(), time.perf_counter()
                if self.name == "left":
                    left_second_popped.set()
                    return b"left-p-frame", time.perf_counter()
                if not release_timeout.wait(2.0):
                    raise AssertionError("test never released the stalled camera")
                raise TimeoutError("right camera stalled")

        left = Camera("left")
        right = Camera("right")
        dataset = mock.Mock(features={})
        dataset.add_frame.side_effect = lambda _row: prime_added.set()
        fresh_snapshot = (
            {"joint": 1.0},
            {"action": 2.0},
            time.perf_counter() + 60.0,
            False,
        )

        def build_frame(_features, values, *, prefix):
            return {f"{prefix}.{key}": value for key, value in values.items()}

        with (
            mock.patch.object(record_proc, "_ENCODED_START_TIMEOUT_S", 0.0),
            mock.patch.object(record_proc, "_ENCODED_ROW_TIMEOUT_S", 0.0),
            mock.patch(
                "lerobot.utils.feature_utils.build_dataset_frame",
                side_effect=build_frame,
            ),
        ):
            capture = threading.Thread(
                target=record_proc.run_encoded_capture_loop,
                kwargs={
                    "cameras": {"left": left, "right": right},
                    "read_snapshot": lambda: fresh_snapshot,
                    "dataset": dataset,
                    "robot_obs_proc": dict,
                    "fps": 60,
                    "task": "test",
                    "rerun_ip": None,
                    "stop_event": stop,
                    "frame_counter": frame_counter,
                    "on_error": errors.append,
                },
                daemon=True,
            )
            capture.start()
            try:
                self.assertTrue(prime_added.wait(2.0))
                self.assertTrue(left_second_popped.wait(2.0))
                # Only calls made after the first complete row matter here: the
                # second row has already popped the left P-frame, but must never
                # be appended with a replayed right access unit.
                dataset.add_frame.reset_mock()
                release_timeout.set()
                capture.join(2.0)
            finally:
                release_timeout.set()
                stop.set()
                capture.join(2.0)

        self.assertFalse(capture.is_alive())
        dataset.add_frame.assert_not_called()
        self.assertEqual(frame_counter["n"], 1)
        self.assertEqual(left.reads, 2)
        self.assertEqual(right.reads, 2)
        self.assertEqual(left.flushes, 2)
        self.assertEqual(right.flushes, 2)
        self.assertEqual(len(errors), 1)
        self.assertIn("'right'", errors[0])
        self.assertIn("after capture was primed", errors[0])
        self.assertIn("decoder reference chain", errors[0])

    def test_capture_base_exception_is_reported_and_reader_is_rearmed(self) -> None:
        errors: list[str] = []

        class Camera:
            def __init__(self) -> None:
                self.flushes = 0

            @property
            def pending(self) -> int:
                return 0

            def flush(self) -> None:
                self.flushes += 1

            def read_next_au(self, *, timeout_ms: float) -> tuple[bytes, float]:
                del timeout_ms
                return b"idr", time.perf_counter()

        camera = Camera()
        dataset = mock.Mock(features={})
        dataset.add_frame.side_effect = SystemExit("dataset writer exited")

        def build_frame(_features, values, *, prefix):
            return {f"{prefix}.{key}": value for key, value in values.items()}

        with mock.patch(
            "lerobot.utils.feature_utils.build_dataset_frame",
            side_effect=build_frame,
        ):
            record_proc.run_encoded_capture_loop(
                cameras={"left": camera},
                read_snapshot=lambda: ({}, {}, time.perf_counter() + 1.0, False),
                dataset=dataset,
                robot_obs_proc=dict,
                fps=60,
                task="test",
                rerun_ip=None,
                stop_event=threading.Event(),
                on_error=errors.append,
            )

        self.assertEqual(errors, ["dataset writer exited"])
        self.assertEqual(camera.flushes, 2)

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

    def test_valid_sensor_pts_maps_to_relay_perf_counter_timeline(self) -> None:
        capture = _capture_perf_from_gst_pts(
            recv_perf=123.5,
            pts_ns=3_463_000_000,
            pts_origin_perf=120.0,
            fallback_latency_s=0.025,
        )

        self.assertAlmostEqual(capture, 123.463)

    def test_tiny_future_sensor_pts_is_clamped_to_receipt(self) -> None:
        capture = _capture_perf_from_gst_pts(
            recv_perf=123.5,
            pts_ns=0,
            pts_origin_perf=123.51,
            fallback_latency_s=0.025,
        )

        self.assertEqual(capture, 123.5)

    def test_invalid_sensor_pts_or_origin_uses_latency_fallback(self) -> None:
        invalid_pairs = (
            (None, 120.0),
            (True, 120.0),
            ("invalid", 120.0),
            (-1, 120.0),
            ((1 << 64) - 1, 120.0),
            (math.nan, 120.0),
            (math.inf, 120.0),
            (3_463_000_000, None),
            (3_463_000_000, True),
            (3_463_000_000, "invalid"),
            (3_463_000_000, -1.0),
            (3_463_000_000, math.nan),
            (3_463_000_000, math.inf),
            (0, 123.551),  # over 50 ms in the future
            (0, 118.499),  # over the five-second stale-data bound
        )
        for pts, origin in invalid_pairs:
            with self.subTest(pts=pts, origin=origin):
                self.assertEqual(
                    _capture_perf_from_gst_pts(123.5, pts, origin, 0.025),
                    123.475,
                )

    def test_transport_metadata_carries_pts_origin_and_latency_fallback(self) -> None:
        meta = _gsth264_meta(
            "/tmp/camera.sock",
            640,
            480,
            60,
            0.025,
            8_765.25,
        )

        self.assertEqual(meta["latency_s"], 0.025)
        self.assertEqual(meta["pts_origin_perf"], 8_765.25)
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
                pts_origin_perf=8_765.25,
            )

        pipeline = gst.parse_launch.call_args.args[0]
        self.assertIn("! application/x-gdp ! gdpdepay ! h264parse", pipeline)
        self.assertEqual(reader._pts_origin_perf, 8_765.25)

    def test_reader_rejects_mid_episode_discontinuity_but_not_first_au(self) -> None:
        gst = mock.Mock()
        with mock.patch(
            "almond_axol.video.gst_zed._require_gst", return_value=(gst, None)
        ):
            reader = EncodedAuReader("/tmp/camera.sock", 640, 480, 60, name="left")

        reader.flush()
        reader._accept_access_unit(
            b"episode-idr",
            10.0,
            is_keyframe=True,
            discont=True,
        )
        self.assertEqual(reader.read_next_au(timeout_ms=0), (b"episode-idr", 10.0))

        reader._accept_access_unit(
            b"orphaned-p-frame",
            10.1,
            is_keyframe=False,
            discont=True,
        )
        with self.assertRaisesRegex(RuntimeError, "discontinuity.*left"):
            reader.read_next_au(timeout_ms=0)
        self.assertEqual(reader.pending, 0)

        # A new episode clears the fault and grants exactly one new boundary
        # exemption; its first IDR is readable even when shmsrc marks DISCONT.
        reader.flush()
        reader._accept_access_unit(
            b"next-idr",
            20.0,
            is_keyframe=True,
            discont=True,
        )
        self.assertEqual(reader.read_next_au(timeout_ms=0), (b"next-idr", 20.0))

    def test_reader_latches_excessive_keyframe_gap_until_flush(self) -> None:
        gst = mock.Mock()
        with mock.patch(
            "almond_axol.video.gst_zed._require_gst", return_value=(gst, None)
        ):
            reader = EncodedAuReader("/tmp/camera.sock", 640, 480, 60, name="right")

        reader._expected_gop = 1
        reader._gop_warn_at = 2
        reader.flush()
        reader._accept_access_unit(b"idr", 1.0, is_keyframe=True, discont=False)
        self.assertEqual(reader.read_next_au(timeout_ms=0), (b"idr", 1.0))
        reader._accept_access_unit(b"p1", 2.0, is_keyframe=False, discont=False)
        self.assertEqual(reader.read_next_au(timeout_ms=0), (b"p1", 2.0))
        reader._accept_access_unit(b"p2", 3.0, is_keyframe=False, discont=False)

        with self.assertRaisesRegex(RuntimeError, "keyframe gap.*right"):
            reader.read_next_au(timeout_ms=0)
        with self.assertRaisesRegex(RuntimeError, "keyframe gap.*right"):
            reader.read_next_au(timeout_ms=0)

        reader.flush()
        reader._accept_access_unit(
            b"recovered-idr", 4.0, is_keyframe=True, discont=False
        )
        self.assertEqual(
            reader.read_next_au(timeout_ms=0),
            (b"recovered-idr", 4.0),
        )

    def test_reader_backlog_is_capture_fatal_instead_of_unbounded_or_dropped(
        self,
    ) -> None:
        gst = mock.Mock()
        with mock.patch(
            "almond_axol.video.gst_zed._require_gst", return_value=(gst, None)
        ):
            reader = EncodedAuReader("/tmp/camera.sock", 640, 480, 60, name="left")

        reader._max_pending_aus = 2
        reader.flush()
        reader._accept_access_unit(b"idr", 1.0, is_keyframe=True, discont=False)
        reader._accept_access_unit(b"p1", 2.0, is_keyframe=False, discont=False)
        reader._accept_access_unit(b"p2", 3.0, is_keyframe=False, discont=False)

        with self.assertRaisesRegex(RuntimeError, "backlog.*left.*2 pending"):
            reader.read_next_au(timeout_ms=0)
        self.assertEqual(reader.pending, 0)

        reader.flush()
        reader._accept_access_unit(
            b"recovered-idr", 4.0, is_keyframe=True, discont=False
        )
        self.assertEqual(
            reader.read_next_au(timeout_ms=0),
            (b"recovered-idr", 4.0),
        )

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
            reader = EncodedAuReader("/tmp/camera.sock", 640, 480, 60)
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
            reader = EncodedAuReader("/tmp/camera.sock", 640, 480, 60)
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
