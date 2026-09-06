from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

from almond_axol.cli import collect_data
from almond_axol.lerobot.teleop.teleop_vr import AxolVRTeleop
from almond_axol.recording import record_proc
from almond_axol.recording.record_proc import (
    DatasetRecorderProcess,
    InProcessRecorder,
    _cleanup_recorder_session,
    _shutdown_process,
    _stop_capture_thread,
)
from almond_axol.robot.base import is_hardware_cleanup_uncertain
from almond_axol.teleop.teleop import VRTeleop
from almond_axol.utils.state_files import UnsafeStatePathError
from almond_axol.video import video_proc


class CollectDataAffinityIntegrityTest(unittest.TestCase):
    def test_early_session_failure_restores_original_affinity(self) -> None:
        original = {2, 3, 4, 5}
        failure = ValueError("resume schema rejected")
        config = object()
        with (
            patch.object(collect_data.os, "sched_getaffinity", return_value=original),
            patch.object(collect_data.os, "sched_setaffinity") as restore,
            patch.object(collect_data.affinity, "pin_realtime") as pin,
            patch.object(collect_data, "_run_session", side_effect=failure),
            self.assertRaisesRegex(ValueError, "resume schema rejected") as raised,
        ):
            collect_data._run(config)  # type: ignore[arg-type]

        self.assertIs(raised.exception, failure)
        pin.assert_called_once_with()
        restore.assert_called_once_with(0, original)

    def test_normal_session_exit_restores_original_affinity(self) -> None:
        original = {0, 1, 2, 3}
        config = object()
        with (
            patch.object(collect_data.os, "sched_getaffinity", return_value=original),
            patch.object(collect_data.os, "sched_setaffinity") as restore,
            patch.object(collect_data.affinity, "pin_realtime") as pin,
            patch.object(collect_data, "_run_session") as run_session,
        ):
            collect_data._run(config)  # type: ignore[arg-type]

        run_session.assert_called_once_with(config, stop_event=None, control=None)
        pin.assert_called_once_with()
        restore.assert_called_once_with(0, original)

    def test_diagnostic_start_interrupt_cleans_live_collection_resources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            robot = Mock(
                action_features={},
                observation_features={},
                cameras={},
                positions=([], []),
                name="test-robot",
            )
            robot.get_joint_observation.return_value = {}
            teleop = Mock(cart=None)
            recorder = Mock(pid=None)
            diag = Mock()
            diag.start.side_effect = KeyboardInterrupt("diagnostic start interrupted")
            cfg = SimpleNamespace(
                mantis=False,
                repo_id="test/repo",
                task="test",
                fps=60,
                teleop_hz=120,
                vcodec="auto",
                root=directory,
                push_to_hub=False,
                rerun_ip=None,
                rerun_port=9876,
                dataset_resolution="SVGA",
                log_level="INFO",
                robot_config=SimpleNamespace(observation_cameras=lambda: {}),
                teleop_config=SimpleNamespace(),
            )

            with (
                patch.object(collect_data, "_prepare_recording_cameras"),
                patch.object(collect_data, "_start_video_relay", return_value=None),
                patch.object(collect_data, "InProcessRecorder", return_value=recorder),
                patch.object(collect_data, "SystemDiag", return_value=diag),
                patch(
                    "almond_axol.lerobot.robot.robot_axol.AxolRobot",
                    return_value=robot,
                ),
                patch(
                    "almond_axol.lerobot.teleop.teleop_vr.AxolVRTeleop",
                    return_value=teleop,
                ),
                patch(
                    "almond_axol.utils.state_files.require_service_dataset_configuration"
                ),
                patch(
                    "almond_axol.utils.state_files.privileged_service_active",
                    return_value=False,
                ),
                patch("almond_axol.utils.network.local_ip", return_value="127.0.0.1"),
                patch(
                    "lerobot.processor.make_default_processors",
                    return_value=(Mock(), Mock(), Mock()),
                ),
                patch.object(collect_data, "restore_dataset_ownership") as restore,
                self.assertRaisesRegex(
                    KeyboardInterrupt, "diagnostic start interrupted"
                ),
            ):
                collect_data._run_session(cfg)

        diag.stop.assert_called_once_with()
        teleop.disconnect.assert_called_once_with()
        robot.disconnect.assert_called_once_with()
        recorder.close.assert_called_once_with()
        restore.assert_called_once_with(Path(directory))


class VideoRelayLifecycleIntegrityTest(unittest.TestCase):
    def test_partial_process_start_is_stopped_before_constructor_raises(self) -> None:
        stopped = threading.Event()

        class Connection:
            def __init__(self) -> None:
                self.closed = False

            def send(self, message: object) -> None:
                if message is None:
                    stopped.set()

            def close(self) -> None:
                self.closed = True

        class PartialProcess:
            pid: int | None = None

            def __init__(self) -> None:
                self.alive = False
                self.join_calls = 0

            def start(self) -> None:
                self.pid = 8128
                self.alive = True
                raise KeyboardInterrupt("relay startup interrupted")

            def join(self, timeout: float | None = None) -> None:
                del timeout
                self.join_calls += 1
                if stopped.is_set():
                    self.alive = False

            def is_alive(self) -> bool:
                return self.alive

        parent_conn = Connection()
        child_conn = Connection()
        process = PartialProcess()
        context = SimpleNamespace(
            Pipe=Mock(return_value=(parent_conn, child_conn)),
            Condition=Mock(return_value=object()),
            Process=Mock(return_value=process),
        )

        with (
            patch.object(
                video_proc.multiprocessing, "get_context", return_value=context
            ),
            self.assertRaisesRegex(KeyboardInterrupt, "relay startup interrupted"),
        ):
            video_proc.VideoRelayProcess({}, want_raw=True)

        self.assertTrue(stopped.is_set())
        self.assertGreaterEqual(process.join_calls, 1)
        self.assertFalse(process.alive)
        self.assertTrue(parent_conn.closed)
        self.assertTrue(child_conn.closed)

    def test_shutdown_escalates_to_kill_and_proves_process_exit(self) -> None:
        class Process:
            pid = 8129

            def __init__(self) -> None:
                self.alive = True
                self.terminate_calls = 0
                self.kill_calls = 0

            def join(self, timeout: float | None = None) -> None:
                del timeout

            def is_alive(self) -> bool:
                return self.alive

            def terminate(self) -> None:
                self.terminate_calls += 1

            def kill(self) -> None:
                self.kill_calls += 1
                self.alive = False

        process = Process()
        relay = video_proc.VideoRelayProcess.__new__(video_proc.VideoRelayProcess)
        relay._shutdown_requested = threading.Event()
        relay._lock = threading.Lock()
        relay._conn = Mock()
        relay._proc = process
        relay.raw_cameras = {}

        relay.shutdown()

        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 1)
        self.assertFalse(process.alive)
        relay._conn.close.assert_called_once_with()


class VRStartupIntegrityTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _teleop(server: object) -> VRTeleop:
        teleop = object.__new__(VRTeleop)
        teleop._vr_server = server
        teleop._vr_thread = None
        teleop._vr_loop = None
        teleop._vr_stop = threading.Event()
        teleop._vr_ready = threading.Event()
        teleop._vr_start_error = None
        teleop._vr_cleanup_error = None
        teleop._ik_thread = None
        teleop._ik_stop = threading.Event()
        teleop._parent_conn = None
        teleop._ik_process = None
        teleop._cart = None
        teleop._robot = SimpleNamespace(enable=AsyncMock(), disable=AsyncMock())
        return teleop

    async def test_vr_startup_failure_wakes_propagates_and_is_retryable(self) -> None:
        first = ValueError("certificate setup failed")
        second = RuntimeError("listener bind failed")
        server = SimpleNamespace(
            enable=AsyncMock(side_effect=[first, second]),
            disable=AsyncMock(),
        )
        teleop = self._teleop(server)

        for expected in (first, second):
            with self.assertRaises(type(expected)) as raised:
                async with asyncio.timeout(1.0):
                    await VRTeleop.__aenter__(teleop)
            self.assertIs(raised.exception, expected)
            self.assertIsNone(teleop._vr_thread)
            self.assertIsNone(teleop._vr_loop)

        self.assertEqual(server.enable.await_count, 2)
        self.assertEqual(server.disable.await_count, 2)
        teleop._robot.enable.assert_not_awaited()
        self.assertEqual(teleop._robot.disable.await_count, 2)

    async def test_vr_startup_preserves_primary_after_cleanup_retry(self) -> None:
        startup = ValueError("listener bind failed")
        cleanup = RuntimeError("partial listener close failed")
        server = SimpleNamespace(
            enable=AsyncMock(side_effect=startup),
            # First call is on the VR thread; disable() retries after joining it.
            disable=AsyncMock(side_effect=[cleanup, None]),
        )
        teleop = self._teleop(server)

        with self.assertRaisesRegex(ValueError, "listener bind failed") as raised:
            async with asyncio.timeout(1.0):
                await VRTeleop.__aenter__(teleop)

        self.assertIs(raised.exception, startup)
        self.assertTrue(
            any("cleanup after failed startup" in note for note in startup.__notes__)
        )
        self.assertEqual(server.disable.await_count, 2)
        self.assertIsNone(teleop._vr_thread)

    async def test_vr_startup_retains_and_reaps_child_when_local_pipe_close_fails(
        self,
    ) -> None:
        startup = OSError("child pipe close failed")
        server = SimpleNamespace(enable=AsyncMock(), disable=AsyncMock())
        teleop = self._teleop(server)
        positions = [0.0] * 8
        teleop._robot.get_positions = AsyncMock(return_value=(positions, positions))
        teleop._config = object()
        teleop._kinematics_config = object()
        teleop._core = SimpleNamespace(set_initial_grips=Mock())

        parent_conn = Mock()
        parent_conn.poll.return_value = False
        child_conn = Mock()
        child_conn.close.side_effect = startup
        process = Mock(pid=8127)
        process.is_alive.side_effect = (True, False)
        context = Mock()
        context.Pipe.return_value = (parent_conn, child_conn)
        context.Process.return_value = process

        with (
            patch(
                "almond_axol.teleop.teleop.multiprocessing.get_context",
                return_value=context,
            ),
            self.assertRaisesRegex(OSError, "child pipe close failed") as raised,
        ):
            await teleop.__aenter__()

        self.assertIs(raised.exception, startup)
        process.start.assert_called_once_with()
        process.terminate.assert_called_once_with()
        self.assertEqual(
            process.join.call_args_list,
            [call(timeout=3.0), call(timeout=2.0)],
        )
        self.assertIsNone(teleop._ik_process)
        parent_conn.close.assert_called_once_with()
        server.disable.assert_awaited_once_with()
        teleop._robot.disable.assert_awaited_once_with()
        self.assertIsNone(teleop._vr_thread)


class LeRobotVRStartupIntegrityTest(unittest.TestCase):
    @staticmethod
    def _teleop() -> AxolVRTeleop:
        teleop = object.__new__(AxolVRTeleop)
        teleop.config = SimpleNamespace(
            vr_server_config=object(),
            vr_teleop_config=SimpleNamespace(absolute_mode=False),
            kinematics_config=object(),
        )
        teleop._loop = None
        teleop._loop_thread = None
        teleop._startup_done_event = None
        teleop._cleanup_pending = False
        teleop._vr_server = None
        teleop._video_expected = False
        teleop._parent_conn = None
        teleop._ik_process = None
        teleop._ik_thread = None
        teleop._ik_stop = threading.Event()
        teleop._cart = None
        return teleop

    @staticmethod
    def _server(
        *,
        enable_error: BaseException | None,
        disable_error: BaseException | None,
    ):
        return SimpleNamespace(
            set_on_frame=Mock(),
            set_mode=Mock(),
            set_pose_mode=Mock(),
            set_video_expected=Mock(),
            enable=AsyncMock(side_effect=enable_error),
            disable=AsyncMock(side_effect=disable_error),
        )

    def test_direct_connect_failure_releases_partial_server_and_loop(self) -> None:
        startup = ValueError("VR listener bind failed")
        server = self._server(enable_error=startup, disable_error=None)
        teleop = self._teleop()

        with (
            patch(
                "almond_axol.lerobot.teleop.teleop_vr.VRServer",
                return_value=server,
            ),
            self.assertRaisesRegex(ValueError, "listener bind failed") as raised,
        ):
            teleop.connect()

        self.assertIs(raised.exception, startup)
        server.disable.assert_awaited_once_with()
        self.assertIsNone(teleop._vr_server)
        self.assertIsNone(teleop._loop)
        self.assertIsNone(teleop._loop_thread)
        self.assertFalse(teleop._cleanup_pending)

    def test_direct_connect_preserves_primary_and_marks_cleanup_uncertain(
        self,
    ) -> None:
        startup = ValueError("VR listener bind failed")
        cleanup = RuntimeError("partial listener would not close")
        server = self._server(enable_error=startup, disable_error=cleanup)
        teleop = self._teleop()

        with (
            patch(
                "almond_axol.lerobot.teleop.teleop_vr.VRServer",
                return_value=server,
            ),
            self.assertRaisesRegex(ValueError, "listener bind failed") as raised,
        ):
            teleop.connect()

        self.assertIs(raised.exception, startup)
        self.assertTrue(is_hardware_cleanup_uncertain(startup))
        self.assertIs(teleop._vr_server, server)
        self.assertIsNone(teleop._loop)
        self.assertIsNone(teleop._loop_thread)
        self.assertTrue(teleop._cleanup_pending)

    def test_direct_connect_failure_after_spawn_reaps_retained_ik_child(
        self,
    ) -> None:
        startup = OSError("child pipe close failed")
        server = self._server(enable_error=None, disable_error=None)
        teleop = self._teleop()
        parent_conn = Mock()
        parent_conn.poll.return_value = False
        child_conn = Mock()
        child_conn.close.side_effect = startup
        process = Mock(pid=8123)
        process.is_alive.side_effect = [True, False]
        context = Mock()
        context.Pipe.return_value = (parent_conn, child_conn)
        context.Process.return_value = process

        with (
            patch(
                "almond_axol.lerobot.teleop.teleop_vr.VRServer",
                return_value=server,
            ),
            patch(
                "almond_axol.lerobot.teleop.teleop_vr.multiprocessing.get_context",
                return_value=context,
            ),
            self.assertRaisesRegex(OSError, "child pipe close failed") as raised,
        ):
            teleop.connect()

        self.assertIs(raised.exception, startup)
        process.start.assert_called_once_with()
        process.terminate.assert_called_once_with()
        self.assertEqual(
            process.join.call_args_list,
            [call(timeout=3.0), call(timeout=2.0)],
        )
        self.assertIsNone(teleop._ik_process)
        parent_conn.close.assert_called_once_with()
        server.disable.assert_awaited_once_with()
        self.assertIsNone(teleop._loop)
        self.assertIsNone(teleop._loop_thread)
        self.assertFalse(teleop._cleanup_pending)

    def test_direct_connect_timeout_requires_coroutine_exit_proof(self) -> None:
        teleop = self._teleop()
        timeout = TimeoutError("startup timed out")
        future = Mock()
        future.result.side_effect = timeout
        future.done.return_value = False
        loop = Mock()
        loop_thread = Mock()
        submitted: list[object] = []

        def submit(coroutine: object, submitted_loop: object) -> object:
            self.assertIs(submitted_loop, loop)
            submitted.append(coroutine)
            return future

        with (
            patch(
                "almond_axol.lerobot.teleop.teleop_vr.asyncio.new_event_loop",
                return_value=loop,
            ),
            patch(
                "almond_axol.lerobot.teleop.teleop_vr.threading.Thread",
                return_value=loop_thread,
            ),
            patch(
                "almond_axol.lerobot.teleop.teleop_vr.asyncio.run_coroutine_threadsafe",
                side_effect=submit,
            ),
            patch.object(teleop, "disconnect") as disconnect,
            patch("almond_axol.lerobot.teleop.teleop_vr.threading.Event") as event_type,
            self.assertRaisesRegex(TimeoutError, "startup timed out") as raised,
        ):
            event_type.return_value.wait.return_value = False
            event_type.return_value.is_set.return_value = False
            teleop.connect()

        self.assertIs(raised.exception, timeout)
        future.cancel.assert_called_once_with()
        disconnect.assert_called_once_with()
        self.assertTrue(is_hardware_cleanup_uncertain(timeout))
        self.assertTrue(any("startup coroutine" in note for note in timeout.__notes__))
        # The fake scheduler never owns the coroutine, so close it explicitly
        # after verifying connect retained the uncertainty instead of treating
        # Future.cancel() as an exit proof.
        submitted[0].close()  # type: ignore[attr-defined]

    def test_disconnect_rechecks_ik_thread_after_child_shutdown(self) -> None:
        teleop = self._teleop()
        server = self._server(enable_error=None, disable_error=None)
        thread = Mock()
        thread.is_alive.side_effect = [True, False]
        process = Mock(pid=8124)
        process.is_alive.side_effect = [True, False]
        parent_conn = Mock()
        parent_conn.poll.return_value = False
        teleop._vr_server = server
        teleop._ik_thread = thread
        teleop._ik_process = process
        teleop._parent_conn = parent_conn

        asyncio.run(teleop._disconnect_async())  # noqa: SLF001

        self.assertEqual(thread.join.call_args_list, [call(3.0), call(2.0)])
        process.terminate.assert_called_once_with()
        self.assertIsNone(teleop._ik_thread)
        self.assertIsNone(teleop._ik_process)
        server.disable.assert_awaited_once_with()
        self.assertFalse(teleop._cleanup_pending)

    def test_disconnect_retains_live_ik_thread_and_its_vr_dependency(self) -> None:
        teleop = self._teleop()
        server = self._server(enable_error=None, disable_error=None)
        thread = Mock()
        thread.is_alive.return_value = True
        process = Mock(pid=8125)
        process.is_alive.side_effect = [True, False]
        parent_conn = Mock()
        parent_conn.poll.return_value = False
        teleop._vr_server = server
        teleop._ik_thread = thread
        teleop._ik_process = process
        teleop._parent_conn = parent_conn

        with self.assertRaisesRegex(RuntimeError, "teleop IK thread failed"):
            asyncio.run(teleop._disconnect_async())  # noqa: SLF001

        self.assertEqual(thread.join.call_args_list, [call(3.0), call(2.0)])
        self.assertIs(teleop._ik_thread, thread)
        self.assertIs(teleop._vr_server, server)
        server.disable.assert_not_awaited()
        self.assertTrue(teleop._cleanup_pending)


class RecorderLifecycleIntegrityTest(unittest.TestCase):
    @staticmethod
    def _encoded_recorder_config(*sources: str) -> dict:
        return {
            "log_level": "INFO",
            "raw_meta": {
                source: {
                    "transport": "gstshm-h264",
                    "socket_path": f"/tmp/{source}.sock",
                    "width": 640,
                    "height": 480,
                    "fps": 60,
                }
                for source in sources
            },
            "snapshot_shm_name": "snapshot-test",
            "obs_keys": [],
            "action_keys": [],
            "rerun_ip": None,
            "dataset_root": "/tmp/unused-recorder-test",
            "smooth_ee_hz": 0.0,
            "fps": 60,
        }

    def test_subprocess_reader_connect_abort_rolls_back_every_camera(self) -> None:
        first = Mock()
        second = Mock()
        second.connect.side_effect = KeyboardInterrupt("startup interrupted")
        conn = Mock()
        config = self._encoded_recorder_config("left", "right")

        with (
            patch(
                "lerobot.processor.make_default_processors",
                return_value=(None, None, Mock()),
            ),
            patch(
                "almond_axol.video.shm_frames.EncodedAuReader",
                side_effect=[first, second],
            ),
            patch("almond_axol.utils.affinity.pin_background", return_value=True),
            patch.object(record_proc, "install_encoded_dataset_encoder"),
            patch("almond_axol.video.shm_frames.SnapshotReader") as snapshot,
            patch.object(record_proc, "_open_dataset") as open_dataset,
            self.assertRaisesRegex(KeyboardInterrupt, "startup interrupted"),
        ):
            record_proc._recorder_main(conn, object(), config)

        first.close.assert_called_once_with()
        second.close.assert_called_once_with()
        snapshot.assert_not_called()
        open_dataset.assert_not_called()
        conn.send.assert_not_called()

    def test_subprocess_ready_failure_closes_all_acquired_resources(self) -> None:
        camera = Mock()
        snap_reader = Mock()
        dataset = Mock()
        verifier = Mock()
        startup_error = RuntimeError("ready reply failed")
        conn = Mock()
        conn.send.side_effect = startup_error
        config = self._encoded_recorder_config("left")

        with (
            patch(
                "lerobot.processor.make_default_processors",
                return_value=(None, None, Mock()),
            ),
            patch(
                "almond_axol.video.shm_frames.EncodedAuReader",
                return_value=camera,
            ),
            patch(
                "almond_axol.video.shm_frames.SnapshotReader",
                return_value=snap_reader,
            ),
            patch("almond_axol.utils.affinity.pin_background", return_value=True),
            patch.object(record_proc, "install_encoded_dataset_encoder"),
            patch.object(record_proc, "_open_dataset", return_value=dataset),
            patch.object(
                record_proc,
                "_EpisodeVideoVerifier",
                return_value=verifier,
            ),
            patch.object(record_proc, "_finalize_dataset") as finalize,
            self.assertRaisesRegex(RuntimeError, "ready reply failed") as raised,
        ):
            record_proc._recorder_main(conn, object(), config)

        self.assertIs(raised.exception, startup_error)
        camera.close.assert_called_once_with()
        snap_reader.close.assert_called_once_with()
        verifier.close.assert_called_once_with()
        finalize.assert_called_once_with(dataset, config, episodes_recorded=0)
        conn.send.assert_called_once_with(("ready", dataset.num_episodes))

    @staticmethod
    def _durability_dataset(root: Path):
        latest_episode = {
            "episode_index": [0],
            "tasks": [["pick"]],
            "length": [12],
            "data/chunk_index": [1],
            "data/file_index": [2],
            "meta/episodes/chunk_index": [3],
            "meta/episodes/file_index": [4],
            "videos/observation.images.left/chunk_index": [5],
            "videos/observation.images.left/file_index": [6],
        }
        expected = [
            root / "data/chunk-001/file-002.parquet",
            root / "meta/episodes/chunk-003/file-004.parquet",
            root / "videos/observation.images.left/chunk-005/file-006.mp4",
            root / "meta/info.json",
            root / "meta/stats.json",
            root / "meta/tasks.parquet",
        ]
        for path in expected:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"durable episode fixture")

        meta = SimpleNamespace(
            root=root,
            data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            video_path=(
                "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
            ),
            latest_episode=latest_episode,
            episodes=None,
            _pq_writer=None,
            _close_writer=Mock(),
        )
        writer = SimpleNamespace(close_writer=Mock(), _latest_episode=object())
        dataset = SimpleNamespace(
            root=root,
            meta=meta,
            writer=writer,
            num_episodes=1,
            save_episode=Mock(),
            clear_episode_buffer=Mock(),
        )
        return dataset, expected

    def test_episode_durability_fsyncs_all_files_before_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, expected = self._durability_dataset(Path(directory) / "dataset")
            events: list[tuple[str, str | None]] = []
            dataset.writer.close_writer.side_effect = lambda: events.append(
                ("close-data", None)
            )
            dataset.meta._close_writer.side_effect = lambda: events.append(
                ("close-meta", None)
            )
            real_fsync = os.fsync

            def trace_fsync(descriptor: int) -> None:
                events.append(("fsync", os.readlink(f"/proc/self/fd/{descriptor}")))
                real_fsync(descriptor)

            with patch.object(record_proc.os, "fsync", side_effect=trace_fsync):
                episode = record_proc.make_episode_durable(dataset)

            self.assertEqual(events[:2], [("close-data", None), ("close-meta", None)])
            fsynced = [Path(target) for kind, target in events if kind == "fsync"]
            self.assertEqual(fsynced[: len(expected)], expected)
            self.assertTrue(all(path.is_dir() for path in fsynced[len(expected) :]))
            self.assertEqual(episode["data/chunk_index"], 1)
            self.assertIsNone(dataset.meta.latest_episode)
            self.assertIsNone(dataset.writer._latest_episode)

    def test_episode_durability_rejects_link_and_nonregular_payloads(self) -> None:
        for replacement in ("symlink", "hardlink", "directory"):
            with self.subTest(replacement=replacement):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory) / "dataset"
                    dataset, expected = self._durability_dataset(root)
                    victim = expected[2]
                    victim.unlink()
                    if replacement == "symlink":
                        outside = Path(directory) / "outside.mp4"
                        outside.write_bytes(b"not dataset content")
                        victim.symlink_to(outside)
                    elif replacement == "hardlink":
                        outside = Path(directory) / "outside.mp4"
                        outside.write_bytes(b"not dataset content")
                        os.link(outside, victim)
                    else:
                        victim.mkdir()

                    with self.assertRaisesRegex(
                        UnsafeStatePathError,
                        "non-regular or hard-linked",
                    ):
                        record_proc.make_episode_durable(dataset)

    def test_episode_durability_follows_links_above_the_dataset_root(self) -> None:
        # ``~/datasets -> /mnt/ssd/datasets`` is a normal operator layout; the
        # no-follow discipline is for paths *inside* the dataset, not for the
        # configured root itself, or every save on such a host fails durability.
        with tempfile.TemporaryDirectory() as directory:
            real_parent = Path(directory) / "ssd"
            real_parent.mkdir()
            linked_parent = Path(directory) / "datasets"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            dataset, expected = self._durability_dataset(linked_parent / "dataset")

            episode = record_proc.make_episode_durable(dataset)

            self.assertEqual(episode["data/chunk_index"], 1)
            self.assertTrue(all(path.exists() for path in expected))

    def test_episode_durability_requires_every_referenced_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, expected = self._durability_dataset(Path(directory) / "dataset")
            expected[2].unlink()

            with self.assertRaises(FileNotFoundError):
                record_proc.make_episode_durable(dataset)

    def test_in_process_durability_failure_is_terminal_but_finalizes_episode(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, _expected = self._durability_dataset(Path(directory) / "dataset")
            durability_error = OSError("metadata footer fsync failed")
            recorder = object.__new__(InProcessRecorder)
            recorder._thread = None
            recorder._stop = None
            recorder._capture_error = {"v": None}
            recorder._dataset = dataset
            recorder._config = {"smooth_ee_hz": 0.0}
            recorder._verifier = Mock()
            recorder._episodes_recorded = 0
            recorder._fatal_error = None

            with (
                patch(
                    "almond_axol.lerobot.nvenc_encoder.dropped_frames",
                    return_value=0,
                ),
                patch.object(
                    record_proc.os,
                    "fsync",
                    side_effect=durability_error,
                ),
                self.assertRaisesRegex(
                    record_proc.EpisodeDurabilityError,
                    "cannot continue safely",
                ) as raised,
            ):
                recorder.save_episode()

            self.assertIs(raised.exception.__cause__, durability_error)
            self.assertNotIsInstance(raised.exception, RuntimeError)
            recorder._dataset.save_episode.assert_called_once_with()
            recorder._verifier.submit.assert_not_called()
            # The episode was written, so cleanup must preserve it even though the
            # caller did not receive a successful save acknowledgement.
            self.assertEqual(recorder._episodes_recorded, 1)

            with self.assertRaises(record_proc.EpisodeDurabilityError) as retry:
                recorder.start_episode("unsafe retry")
            self.assertIs(retry.exception, raised.exception)
            recorder._dataset.clear_episode_buffer.assert_not_called()

            with patch.object(record_proc, "_finalize_dataset") as finalize:
                recorder.close()
            finalize.assert_called_once_with(
                recorder._dataset,
                recorder._config,
                1,
            )
            recorder._verifier.close.assert_called_once_with()

    def test_subprocess_durability_failure_sends_only_fatal_then_finalizes(
        self,
    ) -> None:
        durability_error = OSError("metadata footer fsync failed")
        dataset = Mock(num_episodes=7)
        verifier = Mock()
        snap_reader = Mock()
        conn = Mock()
        conn.recv.return_value = ("save_episode",)
        config = {
            "log_level": "INFO",
            "raw_meta": {},
            "snapshot_shm_name": "snapshot-test",
            "obs_keys": [],
            "action_keys": [],
            "rerun_ip": None,
            "dataset_root": "/tmp/unused-recorder-test",
            "smooth_ee_hz": 0.0,
        }

        with (
            patch(
                "lerobot.processor.make_default_processors",
                return_value=(None, None, Mock()),
            ),
            patch(
                "almond_axol.video.shm_frames.SnapshotReader",
                return_value=snap_reader,
            ),
            patch("almond_axol.utils.affinity.pin_background", return_value=True),
            patch.object(record_proc, "install_dataset_encoder"),
            patch.object(record_proc, "_open_dataset", return_value=dataset),
            patch.object(record_proc, "_EpisodeVideoVerifier", return_value=verifier),
            patch("almond_axol.lerobot.nvenc_encoder.dropped_frames", return_value=0),
            patch.object(
                record_proc,
                "make_episode_durable",
                side_effect=durability_error,
            ),
            patch.object(record_proc, "_cleanup_recorder_session") as cleanup,
            self.assertRaisesRegex(
                record_proc.EpisodeDurabilityError,
                "cannot continue safely",
            ) as raised,
        ):
            record_proc._recorder_main(conn, object(), config)

        self.assertIs(raised.exception.__cause__, durability_error)
        self.assertEqual(conn.send.call_args_list[0], call(("ready", 7)))
        self.assertEqual(conn.send.call_args_list[1][0][0][0], "fatal")
        self.assertFalse(
            any(args[0][0] == "saved" for args, _ in conn.send.call_args_list)
        )
        verifier.submit.assert_not_called()
        self.assertEqual(cleanup.call_args.kwargs["episodes_recorded"], 1)

    def test_parent_fatal_save_reply_poison_recorder_without_incrementing(self) -> None:
        recorder = object.__new__(DatasetRecorderProcess)
        recorder._lock = threading.Lock()
        recorder._conn = Mock()
        recorder._conn.poll.return_value = True
        recorder._conn.recv.return_value = (
            "fatal",
            "episode was written but could not be made crash-durable",
        )
        recorder._episode_count = 4
        recorder._fatal_error = None

        with self.assertRaises(record_proc.EpisodeDurabilityError) as raised:
            recorder.save_episode()

        self.assertNotIsInstance(raised.exception, RuntimeError)
        self.assertEqual(recorder.episode_count(), 4)
        recorder._conn.send.assert_called_once_with(("save_episode",))

        with self.assertRaises(record_proc.EpisodeDurabilityError) as retry:
            recorder.start_episode("must not reach child")
        self.assertIs(retry.exception, raised.exception)
        recorder._conn.send.assert_called_once_with(("save_episode",))

    def test_in_process_cancel_keeps_live_writer_and_buffer(self) -> None:
        recorder = object.__new__(record_proc.InProcessRecorder)
        live_thread = Mock()
        live_thread.is_alive.return_value = True
        recorder._thread = live_thread
        recorder._stop = threading.Event()
        recorder._dataset = Mock()

        with self.assertRaisesRegex(RuntimeError, "refusing to save, clear"):
            recorder.cancel_episode()

        self.assertIs(recorder._thread, live_thread)
        self.assertTrue(recorder._stop.is_set())
        recorder._dataset.clear_episode_buffer.assert_not_called()

    def test_capture_stop_retains_live_thread_until_a_retry_proves_exit(self) -> None:
        release = threading.Event()
        stop = threading.Event()
        thread = threading.Thread(target=release.wait, daemon=True)
        thread.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "refusing to save"):
                _stop_capture_thread(thread, stop, timeout=0.01)
            self.assertTrue(stop.is_set())
            self.assertTrue(thread.is_alive())

            release.set()
            self.assertIsNone(_stop_capture_thread(thread, stop, timeout=1.0))
            self.assertFalse(thread.is_alive())
        finally:
            release.set()
            thread.join(timeout=1.0)

    def test_finalize_failure_propagates_after_all_independent_cleanup(self) -> None:
        finalize_error = RuntimeError("episode parquet finalize failed")
        stop_capture = Mock()
        verifier = Mock()
        camera = Mock()
        snap_reader = Mock()

        with (
            patch.object(
                record_proc, "_finalize_dataset", side_effect=finalize_error
            ) as finalize,
            self.assertRaisesRegex(RuntimeError, "parquet finalize failed") as raised,
        ):
            _cleanup_recorder_session(
                stop_capture=stop_capture,
                dataset=object(),
                config={},
                episodes_recorded=3,
                verifier=verifier,
                cameras={"left": camera},
                snap_reader=snap_reader,
            )

        self.assertIs(raised.exception, finalize_error)
        finalize.assert_called_once()
        verifier.close.assert_called_once_with()
        camera.close.assert_called_once_with()
        snap_reader.close.assert_called_once_with()

    def test_live_capture_blocks_finalize_and_reader_close_retries_join(self) -> None:
        first = RuntimeError("capture thread still alive")
        second = RuntimeError("capture thread remains alive")
        stop_capture = Mock(side_effect=[first, second])
        verifier = Mock()
        camera = Mock()
        snap_reader = Mock()

        with (
            patch.object(record_proc, "_finalize_dataset") as finalize,
            self.assertRaisesRegex(RuntimeError, "still alive") as raised,
        ):
            _cleanup_recorder_session(
                stop_capture=stop_capture,
                dataset=object(),
                config={},
                episodes_recorded=1,
                verifier=verifier,
                cameras={"left": camera},
                snap_reader=snap_reader,
            )

        self.assertIs(raised.exception, first)
        self.assertEqual(stop_capture.call_count, 2)
        finalize.assert_not_called()
        verifier.close.assert_called_once_with()
        self.assertEqual(camera.close.call_count, 2)
        self.assertEqual(snap_reader.close.call_count, 2)

    def test_parent_close_raises_on_child_finalize_failure_after_local_cleanup(
        self,
    ) -> None:
        recorder = object.__new__(DatasetRecorderProcess)
        recorder._closed = False
        recorder._lock = threading.Lock()
        recorder._conn = Mock()
        recorder._snap = Mock()
        recorder._proc = Mock(exitcode=1)
        recorder._proc.is_alive.return_value = False

        with self.assertRaisesRegex(RuntimeError, "finalization failed"):
            recorder.close()

        recorder._conn.send.assert_called_once_with(("shutdown",))
        recorder._proc.join.assert_called_once_with(timeout=record_proc._SAVE_TIMEOUT_S)
        recorder._conn.close.assert_called_once_with()
        recorder._snap.close.assert_called_once_with()
        self.assertTrue(recorder._closed)

    def test_start_and_cancel_propagate_live_capture_rejection(self) -> None:
        recorder = object.__new__(DatasetRecorderProcess)
        recorder._lock = threading.Lock()
        recorder._conn = Mock()
        recorder._conn.poll.return_value = True
        recorder._conn.recv.side_effect = [
            ("error", "old capture thread is still alive"),
            ("error", "old capture thread is still alive"),
        ]

        with self.assertRaisesRegex(RuntimeError, "still alive"):
            recorder.start_episode("pick")
        with self.assertRaisesRegex(RuntimeError, "still alive"):
            recorder.cancel_episode()

        self.assertEqual(
            recorder._conn.send.call_args_list,
            [call(("start_episode", "pick")), call(("cancel_episode",))],
        )

    def test_in_process_finalize_failure_closes_verifier_and_stays_primary(
        self,
    ) -> None:
        finalize_error = RuntimeError("final metadata is unreadable")
        verifier_error = RuntimeError("verifier close failed")
        recorder = object.__new__(InProcessRecorder)
        recorder._stop_capture = Mock()
        recorder._dataset = object()
        recorder._config = {}
        recorder._episodes_recorded = 2
        recorder._verifier = Mock()
        recorder._verifier.close.side_effect = verifier_error

        with (
            patch.object(record_proc, "_finalize_dataset", side_effect=finalize_error),
            self.assertRaisesRegex(RuntimeError, "metadata is unreadable") as raised,
        ):
            recorder.close()

        self.assertIs(raised.exception, finalize_error)
        recorder._verifier.close.assert_called_once_with()
        self.assertTrue(
            any("verifier close failed" in note for note in finalize_error.__notes__)
        )

    def test_process_shutdown_attempts_kill_after_join_and_terminate_failures(
        self,
    ) -> None:
        proc = Mock()
        proc.join.side_effect = [RuntimeError("join failed"), None, None]
        proc.is_alive.side_effect = [True, True, False]
        proc.terminate.side_effect = RuntimeError("terminate failed")

        alive, forced, failures = _shutdown_process(proc, graceful_timeout=0.0)

        self.assertFalse(alive)
        self.assertTrue(forced)
        proc.terminate.assert_called_once_with()
        proc.kill.assert_called_once_with()
        self.assertEqual(len(failures), 2)

    def test_parent_close_retries_local_cleanup_that_did_not_complete(self) -> None:
        recorder = object.__new__(DatasetRecorderProcess)
        recorder._closed = False
        recorder._lock = threading.Lock()
        recorder._conn = Mock()
        recorder._snap = Mock()
        recorder._snap.close.side_effect = [RuntimeError("shm close failed"), None]
        recorder._proc = Mock(exitcode=0)
        recorder._proc.is_alive.return_value = False

        with self.assertRaisesRegex(RuntimeError, "shm close failed"):
            recorder.close()
        self.assertFalse(recorder._closed)

        recorder.close()
        self.assertTrue(recorder._closed)
        self.assertEqual(recorder._snap.close.call_count, 2)

    def test_constructor_ready_timeout_terminates_child_and_closes_ipc_shm(
        self,
    ) -> None:
        snap = Mock(name="snapshot_writer", name_attr="unused")
        snap.name = "snapshot-shm"
        parent_conn = Mock(name="parent_conn")
        parent_conn.poll.return_value = False
        child_conn = Mock(name="child_conn")
        proc = Mock(name="recorder_process")
        proc.is_alive.side_effect = [True, False]
        ctx = Mock()
        ctx.Pipe.return_value = (parent_conn, child_conn)
        ctx.Process.return_value = proc

        with (
            patch("almond_axol.video.shm_frames.SnapshotWriter", return_value=snap),
            patch.object(record_proc.multiprocessing, "get_context", return_value=ctx),
            patch.object(record_proc, "_READY_TIMEOUT_S", 0.0),
            self.assertRaisesRegex(RuntimeError, "did not become ready"),
        ):
            DatasetRecorderProcess(
                raw_cond=object(),
                raw_meta={},
                obs_keys=[],
                action_keys=[],
                config={},
            )

        proc.terminate.assert_called_once_with()
        proc.kill.assert_not_called()
        self.assertEqual(
            proc.join.call_args_list,
            [call(timeout=0), call(timeout=5.0)],
        )
        child_conn.close.assert_called()
        parent_conn.close.assert_called_once_with()
        snap.close.assert_called_once_with()

    def test_constructor_pipe_failure_closes_snapshot_shared_memory(self) -> None:
        snap = Mock(name="snapshot_writer")
        snap.name = "snapshot-shm"
        ctx = Mock()
        ctx.Pipe.side_effect = OSError("pipe allocation failed")

        with (
            patch("almond_axol.video.shm_frames.SnapshotWriter", return_value=snap),
            patch.object(record_proc.multiprocessing, "get_context", return_value=ctx),
            self.assertRaisesRegex(OSError, "pipe allocation failed"),
        ):
            DatasetRecorderProcess(
                raw_cond=object(),
                raw_meta={},
                obs_keys=[],
                action_keys=[],
                config={},
            )

        ctx.Process.assert_not_called()
        snap.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
