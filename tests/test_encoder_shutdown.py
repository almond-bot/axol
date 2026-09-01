from __future__ import annotations

import queue
import sys
import tempfile
import threading
import time
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from almond_axol.lerobot import h264_mux_encoder, nvenc_encoder
from almond_axol.recording import record_proc


class EncoderShutdownTest(unittest.TestCase):
    @staticmethod
    def _fake_gst():
        class Buffer:
            @staticmethod
            def new_wrapped(data: bytes):
                return SimpleNamespace(data=data, pts=None, dts=None, duration=None)

        return SimpleNamespace(
            SECOND=1_000_000_000,
            State=SimpleNamespace(PLAYING="playing", NULL="null"),
            FlowReturn=SimpleNamespace(OK="ok"),
            MessageType=SimpleNamespace(EOS=1, ERROR=2),
            Buffer=Buffer,
        )

    def _blocking_h264_muxer(self, directory: str):
        gst = self._fake_gst()
        entered = threading.Event()
        release = threading.Event()
        pipeline = mock.Mock()
        src = mock.Mock()

        def set_state(state: str) -> None:
            if state == gst.State.NULL:
                release.set()

        def emit(signal: str, *_args):
            if signal == "push-buffer":
                entered.set()
                if not release.wait(2.0):
                    raise AssertionError("test did not release blocking appsrc")
                return gst.FlowReturn.OK
            if signal == "end-of-stream":
                return gst.FlowReturn.OK
            raise AssertionError(f"unexpected appsrc signal {signal!r}")

        pipeline.set_state.side_effect = set_state
        src.emit.side_effect = emit
        patches = (
            mock.patch(
                "almond_axol.video.gst_zed._require_gst",
                return_value=(gst, None),
            ),
            mock.patch.object(
                h264_mux_encoder._CameraH264Muxer,
                "_build",
                return_value=(pipeline, src),
            ),
            mock.patch.object(h264_mux_encoder, "_FEED_QUEUE_MAX", 1),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        muxer = h264_mux_encoder._CameraH264Muxer(
            Path(directory) / "camera.mp4",
            60,
            want_stats=False,
        )
        return muxer, gst, pipeline, entered, release

    def test_blocking_appsrc_never_blocks_capture_and_full_queue_is_fatal(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            muxer, gst, pipeline, entered, release = self._blocking_h264_muxer(
                directory
            )
            try:
                started = time.perf_counter()
                muxer.feed(b"idr")
                self.assertLess(time.perf_counter() - started, 0.5)
                self.assertTrue(entered.wait(1.0))

                # The feeder owns the blocked push. One further AU fills its
                # bounded queue; the next row fails instead of blocking or
                # dropping a dependency-bearing picture.
                muxer.feed(b"p1")
                with self.assertRaisesRegex(RuntimeError, "queue filled"):
                    muxer.feed(b"p2")
            finally:
                muxer.cancel()
                release.set()

            self.assertFalse(muxer._feed_thread.is_alive())
            self.assertIn(
                mock.call(gst.State.NULL),
                pipeline.set_state.call_args_list,
            )

    def test_finish_cancels_blocked_appsrc_and_rejects_truncated_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            muxer, gst, pipeline, entered, release = self._blocking_h264_muxer(
                directory
            )
            muxer.feed(b"idr")
            self.assertTrue(entered.wait(1.0))
            try:
                with (
                    mock.patch.object(
                        h264_mux_encoder,
                        "_FEED_DRAIN_TIMEOUT_S",
                        0.0,
                    ),
                    mock.patch.object(
                        h264_mux_encoder,
                        "_FEED_ABORT_TIMEOUT_S",
                        1.0,
                    ),
                    self.assertRaisesRegex(RuntimeError, "did not drain"),
                ):
                    muxer.finish()
            finally:
                release.set()
                muxer._feed_thread.join(1.0)

            self.assertFalse(muxer._feed_thread.is_alive())
            self.assertIn(
                mock.call(gst.State.NULL),
                pipeline.set_state.call_args_list,
            )
            pipeline.get_bus.assert_not_called()

    def test_h264_feeder_base_exception_is_latched_and_finish_fails(self) -> None:
        gst = self._fake_gst()
        pipeline = mock.Mock()
        src = mock.Mock()
        src.emit.side_effect = SystemExit("native appsrc aborted")
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch(
                "almond_axol.video.gst_zed._require_gst",
                return_value=(gst, None),
            ),
            mock.patch.object(
                h264_mux_encoder._CameraH264Muxer,
                "_build",
                return_value=(pipeline, src),
            ),
        ):
            muxer = h264_mux_encoder._CameraH264Muxer(
                Path(directory) / "camera.mp4",
                60,
                want_stats=False,
            )
            muxer.feed(b"idr")
            muxer._feed_thread.join(1.0)

            with self.assertRaisesRegex(RuntimeError, "native appsrc aborted"):
                muxer.finish()

        self.assertFalse(muxer._feed_thread.is_alive())
        pipeline.get_bus.assert_not_called()

    def test_stats_shutdown_signal_never_blocks_on_full_queue(self) -> None:
        for method in ("result", "cancel"):
            with self.subTest(method=method):
                worker = object.__new__(h264_mux_encoder._StatsWorker)
                worker._queue = queue.Queue(maxsize=1)
                worker._queue.put_nowait(b"queued-idr")
                worker._stop = threading.Event()
                worker._cancelled = threading.Event()
                worker._thread = mock.Mock()
                worker._thread.is_alive.return_value = True

                started = time.perf_counter()
                with mock.patch.object(
                    h264_mux_encoder,
                    "_STATS_JOIN_TIMEOUT_S",
                    0.0,
                ):
                    result = getattr(worker, method)()
                self.assertLess(time.perf_counter() - started, 0.5)
                self.assertIsNone(result)
                self.assertTrue(worker._stop.is_set())
                self.assertEqual(worker._queue.qsize(), 1)
                if method == "cancel":
                    self.assertTrue(worker._cancelled.is_set())

    def test_h264_constructor_does_not_start_stats_before_pipeline_owns_state(
        self,
    ) -> None:
        gst = self._fake_gst()
        pipeline = mock.Mock()
        pipeline.set_state.side_effect = RuntimeError("PLAYING failed")
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch(
                "almond_axol.video.gst_zed._require_gst",
                return_value=(gst, None),
            ),
            mock.patch.object(
                h264_mux_encoder._CameraH264Muxer,
                "_build",
                return_value=(pipeline, mock.Mock()),
            ),
            mock.patch.object(h264_mux_encoder, "_StatsWorker") as stats_worker,
            self.assertRaisesRegex(RuntimeError, "PLAYING failed"),
        ):
            h264_mux_encoder._CameraH264Muxer(
                Path(directory) / "camera.mp4",
                60,
                want_stats=True,
            )

        stats_worker.assert_not_called()
        self.assertEqual(pipeline.set_state.call_count, 2)

    def test_h264_episode_start_rolls_back_prior_camera(self) -> None:
        encoder = h264_mux_encoder.H264MuxStreamingEncoder(60, want_stats=False)
        first_camera = mock.Mock()
        attempts = 0

        def construct(video_path: Path, *_args):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                first_camera.video_path = video_path
                return first_camera
            raise RuntimeError("second H264 camera failed")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(
                    h264_mux_encoder,
                    "_CameraH264Muxer",
                    side_effect=construct,
                ),
                mock.patch(
                    "almond_axol.utils.state_files.privileged_service_active",
                    return_value=False,
                ),
                self.assertRaisesRegex(RuntimeError, "second H264 camera failed"),
            ):
                encoder.start_episode(["left", "right"], root)

            first_camera.cancel.assert_called_once_with()
            self.assertFalse(encoder._episode_active)
            self.assertEqual(encoder._cams, {})
            self.assertEqual(list(root.iterdir()), [])

    def test_nvenc_finish_kills_wedged_writer_and_never_returns_path(self) -> None:
        encoder = object.__new__(nvenc_encoder._CameraNvencEncoder)
        encoder.video_path = Path("/tmp/never-return-truncated.mp4")
        encoder._queue = queue.Queue(maxsize=1)
        encoder._queue.put_nowait(object())
        encoder._stop = threading.Event()
        encoder._cancelled = threading.Event()
        encoder._thread = mock.Mock()
        encoder._thread.is_alive.return_value = True
        encoder._proc = mock.Mock()
        encoder._proc.poll.return_value = None
        encoder._error = None

        started = time.perf_counter()
        with (
            mock.patch.object(nvenc_encoder, "_WRITER_FINISH_TIMEOUT_S", 0.0),
            mock.patch.object(nvenc_encoder, "_WRITER_ABORT_TIMEOUT_S", 0.0),
            self.assertRaisesRegex(RuntimeError, "MP4 may be truncated"),
        ):
            encoder.finish()
        self.assertLess(time.perf_counter() - started, 0.5)
        self.assertTrue(encoder._stop.is_set())
        self.assertTrue(encoder._cancelled.is_set())
        self.assertEqual(encoder._queue.qsize(), 1)
        encoder._proc.kill.assert_called_once_with()
        encoder._proc.wait.assert_called_once_with(timeout=2)

    def test_nvenc_queue_overflow_is_immediately_capture_fatal(self) -> None:
        encoder = object.__new__(nvenc_encoder._CameraNvencEncoder)
        encoder.video_path = Path("camera.mp4")
        encoder._queue = queue.Queue(maxsize=1)
        encoder._queue.put_nowait(object())
        encoder._stop = threading.Event()
        encoder._thread = mock.Mock()
        encoder._thread.is_alive.return_value = True
        encoder._error = None
        encoder._dropped = 0
        nvenc_encoder.reset_dropped_frames()

        with self.assertRaisesRegex(RuntimeError, "shorter than its dataset rows"):
            encoder.feed(object())

        self.assertEqual(encoder.dropped, 1)
        self.assertEqual(nvenc_encoder.dropped_frames(), 1)
        self.assertIsNotNone(encoder._error)

    def test_nvenc_writer_base_exception_is_latched(self) -> None:
        encoder = object.__new__(nvenc_encoder._CameraNvencEncoder)
        encoder.video_path = Path("camera.mp4")
        encoder._queue = queue.Queue()
        encoder._queue.put_nowait(object())
        encoder._stop = threading.Event()
        encoder._cancelled = threading.Event()
        encoder._error = None
        encoder._encode = mock.Mock(side_effect=SystemExit("native writer aborted"))
        encoder._kill = mock.Mock()

        encoder._run()

        self.assertEqual(encoder._error, "native writer aborted")
        encoder._kill.assert_called_once_with()

    def test_nvenc_finish_episode_rejects_any_reported_drop(self) -> None:
        encoder = nvenc_encoder.NvencStreamingEncoder(60)
        camera = mock.Mock(dropped=1)
        camera.finish.return_value = (Path("camera.mp4"), None)
        encoder._cams = {"left": camera}
        encoder._episode_active = True

        with self.assertRaisesRegex(RuntimeError, "refusing to return"):
            encoder.finish_episode()

        self.assertTrue(encoder._episode_active)
        self.assertEqual(encoder._cams, {"left": camera})

    def test_nvenc_episode_start_rolls_back_prior_camera(self) -> None:
        encoder = nvenc_encoder.NvencStreamingEncoder(60)
        first_camera = mock.Mock()
        attempts = 0

        def construct(video_path: Path, *_args):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                first_camera.video_path = video_path
                return first_camera
            raise RuntimeError("second NVENC camera failed")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(
                    nvenc_encoder,
                    "_CameraNvencEncoder",
                    side_effect=construct,
                ),
                mock.patch(
                    "almond_axol.utils.state_files.privileged_service_active",
                    return_value=False,
                ),
                self.assertRaisesRegex(RuntimeError, "second NVENC camera failed"),
            ):
                encoder.start_episode(["left", "right"], root)

            first_camera.cancel.assert_called_once_with()
            self.assertFalse(encoder._episode_active)
            self.assertEqual(encoder._cams, {})
            self.assertEqual(list(root.iterdir()), [])

    def test_cancel_retains_camera_ownership_when_writer_will_not_stop(self) -> None:
        cases = (
            (
                h264_mux_encoder.H264MuxStreamingEncoder(60, want_stats=False),
                h264_mux_encoder,
                "_remove_h264_staging",
            ),
            (
                nvenc_encoder.NvencStreamingEncoder(60),
                nvenc_encoder,
                "_remove_nvenc_staging",
            ),
        )
        for encoder, module, cleanup_name in cases:
            with self.subTest(encoder=type(encoder).__name__):
                camera = mock.Mock(video_path=Path("camera.mp4"))
                camera.cancel.side_effect = RuntimeError("writer remains alive")
                encoder._cams = {"left": camera}
                encoder._episode_active = True
                with (
                    mock.patch.object(module, cleanup_name) as cleanup,
                    self.assertRaisesRegex(RuntimeError, "remains alive"),
                ):
                    encoder.cancel_episode()

                cleanup.assert_not_called()
                self.assertTrue(encoder._episode_active)
                self.assertEqual(encoder._cams, {"left": camera})

    def test_concat_rejects_short_h264_stream_without_replaying_packet(self) -> None:
        in_stream = SimpleNamespace(type="video", frames=2)
        packet = SimpleNamespace(
            dts=0,
            pts=0,
            duration=1,
            time_base=Fraction(1, 60),
            stream=in_stream,
        )
        source = SimpleNamespace(
            streams=[in_stream],
            demux=lambda _stream: [packet],
        )
        out_stream = SimpleNamespace(time_base=None)
        destination = mock.Mock()
        destination.add_stream_from_template.return_value = out_stream
        source_context = mock.MagicMock()
        source_context.__enter__.return_value = source
        destination_context = mock.MagicMock()
        destination_context.__enter__.return_value = destination
        fake_av = SimpleNamespace(Packet=mock.Mock())

        def open_container(_path, *, mode):
            return destination_context if mode == "w" else source_context

        fake_av.open = mock.Mock(side_effect=open_container)
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.dict(sys.modules, {"av": fake_av}),
            self.assertRaisesRegex(RuntimeError, "refusing to replay"),
        ):
            record_proc._concat_constant_fps(
                [Path(directory) / "short.mp4"],
                Path(directory) / "output.mp4",
                Fraction(60, 1),
            )

        destination.mux.assert_called_once_with(packet)
        fake_av.Packet.assert_not_called()

    def test_encoded_capture_failure_is_discarded_before_save_reply(self) -> None:
        dataset = mock.Mock(num_episodes=0)
        camera = mock.Mock()
        snapshot_reader = mock.Mock()
        verifier = mock.Mock()
        conn = mock.Mock()
        conn.recv.side_effect = [
            ("start_episode", "test"),
            ("save_episode",),
            None,
        ]
        config = {
            "log_level": "INFO",
            "raw_meta": {
                "left": {
                    "transport": "gstshm-h264",
                    "socket_path": "/tmp/left.sock",
                    "width": 640,
                    "height": 480,
                    "fps": 60,
                }
            },
            "snapshot_shm_name": "snapshot-test",
            "obs_keys": [],
            "action_keys": [],
            "rerun_ip": None,
            "dataset_root": "/tmp/unused-recorder-test",
            "smooth_ee_hz": 0.0,
            "fps": 60,
        }

        def fail_capture(**kwargs) -> None:
            kwargs["on_error"]("encoded camera stalled")

        with (
            mock.patch(
                "lerobot.processor.make_default_processors",
                return_value=(None, None, mock.Mock()),
            ),
            mock.patch(
                "almond_axol.video.shm_frames.EncodedAuReader",
                return_value=camera,
            ),
            mock.patch(
                "almond_axol.video.shm_frames.SnapshotReader",
                return_value=snapshot_reader,
            ),
            mock.patch("almond_axol.utils.affinity.pin_background", return_value=True),
            mock.patch.object(record_proc, "install_encoded_dataset_encoder"),
            mock.patch.object(record_proc, "_open_dataset", return_value=dataset),
            mock.patch.object(
                record_proc,
                "_EpisodeVideoVerifier",
                return_value=verifier,
            ),
            mock.patch.object(
                record_proc,
                "run_encoded_capture_loop",
                side_effect=fail_capture,
            ),
            mock.patch(
                "almond_axol.lerobot.nvenc_encoder.dropped_frames",
                return_value=0,
            ),
            mock.patch.object(record_proc, "_cleanup_recorder_session"),
        ):
            record_proc._recorder_main(conn, object(), config)

        self.assertEqual(dataset.clear_episode_buffer.call_count, 2)
        self.assertEqual(
            conn.send.call_args_list,
            [
                mock.call(("ready", 0)),
                mock.call(("started",)),
                mock.call(
                    (
                        "error",
                        "encoded camera stalled; episode discarded",
                    )
                ),
            ],
        )
        dataset.save_episode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
