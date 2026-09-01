from __future__ import annotations

import asyncio
import io
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from almond_axol.cli.mantis_bridge import managed_mantis_bridge
from almond_axol.cli.tracker_bridge import run_configured_bridge
from almond_axol.cli import tracker_identify
from almond_axol.tracker import ultimate as tracker_ultimate
from almond_axol.tracker.base import (
    TrackerSourceError,
    zup_to_yup_pos,
    zup_to_yup_quat,
)
from almond_axol.tracker.bridge import (
    ManagedStdinControls,
    StopEventControls,
    TrackerBridge,
)
from almond_axol.tracker.survive import SurviveSource
from almond_axol.tracker.ultimate import UltimateSource


class _FatalSource:
    def poses(self):  # type: ignore[no-untyped-def]
        raise TrackerSourceError("hardware reader died")


class _FakeSocket:
    async def send(self, _payload: str) -> None:
        pass

    async def _messages(self):  # type: ignore[no-untyped-def]
        await asyncio.Future()
        yield ""

    def __aiter__(self):  # type: ignore[no-untyped-def]
        return self._messages()


class _FakeConnection:
    async def __aenter__(self) -> _FakeSocket:
        return _FakeSocket()

    async def __aexit__(self, *_args: object) -> None:
        pass


def _tracker_config() -> SimpleNamespace:
    return SimpleNamespace(
        backend="survive",
        left="tracker-left",
        right="tracker-right",
        allow_single_side=False,
        trigger_can_left="old-left",
        trigger_can_right="old-right",
    )


class ManagedControlsTest(unittest.TestCase):
    def test_managed_stdin_accepts_reset_and_quit_but_not_engage_toggle(self) -> None:
        stop = threading.Event()
        quit_called = threading.Event()
        owner_active = threading.Event()
        controls = ManagedStdinControls(
            stop,
            quit_called.set,
            input_stream=io.StringIO("\nr\nq\n"),
            activation_event=owner_active,
        )

        controls.start()
        self.assertFalse(quit_called.wait(0.02))
        owner_active.set()
        self.assertTrue(quit_called.wait(1.0))
        self.assertTrue(stop.is_set())
        self.assertEqual(controls.consume(), (False, True))
        self.assertEqual(controls.consume(), (False, False))

    def test_direct_q_interrupt_is_cleanly_consumed_by_owner_context(self) -> None:
        captured: list[ManagedStdinControls] = []

        def run_bridge(_config, **kwargs):  # type: ignore[no-untyped-def]
            controls = kwargs["controls"]
            captured.append(controls)
            kwargs["on_ready"]()
            controls.quit.wait(1.0)

        with (
            patch(
                "almond_axol.tracker.load_tracker_config",
                return_value=_tracker_config(),
            ),
            patch("almond_axol.tracker.config.select_tracker_backend"),
            patch("almond_axol.cli.mantis_bridge.require_mantis_tracker_readiness"),
            patch(
                "almond_axol.cli.tracker_bridge.run_configured_bridge",
                side_effect=run_bridge,
            ),
            patch("almond_axol.cli.mantis_bridge._thread.interrupt_main"),
        ):
            with managed_mantis_bridge(
                "lighthouse",
                left_channel="can-left",
                right_channel="can-right",
                port=8000,
                pose_source_id="managed-test",
            ):
                controls = captured[0]
                controls.quit.set()
                controls._on_quit()
                raise KeyboardInterrupt

    def test_post_ready_bridge_failure_stops_and_surfaces_to_owner(self) -> None:
        release_failure = threading.Event()
        owner_interrupted = threading.Event()

        def run_bridge(_config, **kwargs):  # type: ignore[no-untyped-def]
            kwargs["on_ready"]()
            release_failure.wait(1.0)
            raise TrackerSourceError("survive-cli exited unexpectedly (code 17)")

        with (
            patch(
                "almond_axol.tracker.load_tracker_config",
                return_value=_tracker_config(),
            ),
            patch("almond_axol.tracker.config.select_tracker_backend"),
            patch("almond_axol.cli.mantis_bridge.require_mantis_tracker_readiness"),
            patch(
                "almond_axol.cli.tracker_bridge.run_configured_bridge",
                side_effect=run_bridge,
            ),
            patch(
                "almond_axol.cli.mantis_bridge._thread.interrupt_main",
                side_effect=owner_interrupted.set,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "failed after startup.*survive-cli exited unexpectedly",
            ):
                with managed_mantis_bridge(
                    "lighthouse",
                    left_channel="can-left",
                    right_channel="can-right",
                    port=8000,
                    pose_source_id="managed-test",
                ):
                    release_failure.set()
                    self.assertTrue(owner_interrupted.wait(1.0))
                    # Model the KeyboardInterrupt delivered to the owning CLI
                    # without sending a real process signal in the test suite.
                    raise KeyboardInterrupt

    def test_bridge_thread_start_interrupt_is_stopped_and_joined(self) -> None:
        captured_stop: list[threading.Event] = []

        class InterruptingThread:
            ident = 123

            def __init__(self, **_kwargs: object) -> None:
                self.alive = True
                self.join = Mock(side_effect=self._join)

            def start(self) -> None:
                raise KeyboardInterrupt

            def _join(self, _timeout: float) -> None:
                self.alive = False

            def is_alive(self) -> bool:
                return self.alive

        def controls(stop: threading.Event, *_args: object, **_kwargs: object) -> Mock:
            captured_stop.append(stop)
            return Mock()

        with (
            patch(
                "almond_axol.tracker.load_tracker_config",
                return_value=_tracker_config(),
            ),
            patch("almond_axol.tracker.config.select_tracker_backend"),
            patch("almond_axol.cli.mantis_bridge.require_mantis_tracker_readiness"),
            patch(
                "almond_axol.tracker.bridge.ManagedStdinControls",
                side_effect=controls,
            ),
            patch(
                "almond_axol.cli.mantis_bridge.threading.Thread",
                side_effect=InterruptingThread,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            with managed_mantis_bridge(
                "lighthouse",
                left_channel="can-left",
                right_channel="can-right",
                port=8000,
                pose_source_id="managed-test",
            ):
                pass

        self.assertEqual(len(captured_stop), 1)
        self.assertTrue(captured_stop[0].is_set())


class SurviveLifecycleTest(unittest.TestCase):
    def test_pysurvive_uses_native_pose_epoch_on_host_monotonic_clock(self) -> None:
        source = SurviveSource()
        native_epoch = 1_700_000_000.0

        class Updated:
            def Name(self) -> bytes:
                return b"T20"

            def Pose(self) -> tuple[SimpleNamespace, float]:
                source._stop.set()
                return (
                    SimpleNamespace(
                        Pos=[1.0, 2.0, 3.0],
                        Rot=[1.0, 0.0, 0.0, 0.0],
                    ),
                    native_epoch,
                )

        class FakeContext:
            ptr = object()

            def __init__(self, _args: list[str]) -> None:
                pass

            def Running(self) -> bool:
                return True

            def NextUpdated(self) -> Updated:
                return Updated()

        pysurvive = ModuleType("pysurvive")
        pysurvive.SimpleContext = FakeContext  # type: ignore[attr-defined]
        pysurvive.simple_close = Mock()  # type: ignore[attr-defined]
        with (
            patch.dict(sys.modules, {"pysurvive": pysurvive}),
            patch(
                "almond_axol.tracker.survive.time.time",
                return_value=native_epoch + 0.025,
            ),
            patch("almond_axol.tracker.survive.time.perf_counter", return_value=42.0),
        ):
            source._run_pysurvive()
            source.stop()

        sample = source.poses()["T20"]
        self.assertAlmostEqual(sample.t, 41.975, places=5)
        self.assertTrue(sample.timestamp_is_capture)

    def test_pysurvive_context_closes_once_after_polling_exits(self) -> None:
        poll_entered = threading.Event()
        release_poll = threading.Event()
        poll_exited = threading.Event()
        close_calls: list[object] = []

        class FakeContext:
            def __init__(self, _args: list[str]) -> None:
                self.ptr = object()

            def Running(self) -> bool:
                return True

            def NextUpdated(self) -> None:
                poll_entered.set()
                release_poll.wait(1.0)
                poll_exited.set()

        pysurvive = ModuleType("pysurvive")
        pysurvive.SimpleContext = FakeContext  # type: ignore[attr-defined]

        def simple_close(ptr: object) -> None:
            self.assertTrue(poll_exited.is_set())
            self.assertIsNone(source._thread)
            close_calls.append(ptr)

        pysurvive.simple_close = simple_close  # type: ignore[attr-defined]
        source = SurviveSource()
        stop_errors: list[BaseException] = []

        def stop_source() -> None:
            try:
                source.stop()
            except BaseException as exc:  # surfaced on the test thread below
                stop_errors.append(exc)

        with patch.dict(sys.modules, {"pysurvive": pysurvive}):
            source._thread = threading.Thread(
                target=source._run_worker,
                args=(source._run_pysurvive,),
                daemon=True,
            )
            source._thread.start()
            self.assertTrue(poll_entered.wait(1.0))
            stopped = threading.Thread(target=stop_source)
            stopped.start()
            self.assertTrue(source._stop.wait(1.0))
            self.assertEqual(close_calls, [])
            release_poll.set()
            stopped.join(1.0)
            self.assertFalse(stopped.is_alive())
            source.stop()

        self.assertEqual(stop_errors, [])
        self.assertEqual(len(close_calls), 1)
        self.assertIsNone(source._simple_context)

    def test_uncertain_pysurvive_close_is_not_retried(self) -> None:
        poll_entered = threading.Event()
        release_poll = threading.Event()
        close = Mock(side_effect=RuntimeError("native close uncertain"))

        class FakeContext:
            ptr = object()

            def __init__(self, _args: list[str]) -> None:
                pass

            def Running(self) -> bool:
                return True

            def NextUpdated(self) -> None:
                poll_entered.set()
                release_poll.wait(1.0)

        pysurvive = ModuleType("pysurvive")
        pysurvive.SimpleContext = FakeContext  # type: ignore[attr-defined]
        pysurvive.simple_close = close  # type: ignore[attr-defined]
        source = SurviveSource()
        with patch.dict(sys.modules, {"pysurvive": pysurvive}):
            source._thread = threading.Thread(
                target=source._run_worker,
                args=(source._run_pysurvive,),
                daemon=True,
            )
            source._thread.start()
            self.assertTrue(poll_entered.wait(1.0))
            source._stop.set()
            release_poll.set()
            with self.assertRaisesRegex(TrackerSourceError, "ownership is uncertain"):
                source.stop()
            with self.assertRaisesRegex(TrackerSourceError, "ownership is uncertain"):
                source.stop()
            with self.assertRaisesRegex(TrackerSourceError, "cleanup is incomplete"):
                source.start()

        close.assert_called_once_with(FakeContext.ptr)
        self.assertIsNotNone(source._simple_context)

    def test_native_z_up_pose_is_relabelled_to_webxr_y_up(self) -> None:
        source = SurviveSource()
        half = np.sqrt(0.5)
        native_pos = np.array([1.0, 2.0, 3.0])
        native_quat_xyzw = np.array([0.0, 0.0, half, half])
        np.testing.assert_allclose(zup_to_yup_pos(native_pos), [1.0, 3.0, -2.0])
        np.testing.assert_allclose(
            zup_to_yup_quat(native_quat_xyzw), [0.0, half, 0.0, half]
        )
        source._publish(
            "tracker",
            native_pos,
            np.array([half, 0.0, 0.0, half]),  # wxyz, +90 deg around native z
        )

        sample = source.poses()["tracker"]
        np.testing.assert_allclose(sample.pos, [1.0, 3.0, -2.0])
        np.testing.assert_allclose(sample.quat, [0.0, half, 0.0, half])

    def test_malformed_pose_is_not_published_as_fresh_tracking(self) -> None:
        good_pos = np.array([0.1, 0.2, 0.3])
        good_quat = np.array([1.0, 0.0, 0.0, 0.0])
        invalid = (
            (np.array([np.nan, 0.0, 0.0]), good_quat),
            (good_pos, np.array([1.0, 0.0, np.inf, 0.0])),
            (good_pos, np.zeros(4)),
            (np.zeros(2), good_quat),
            (good_pos, np.zeros(3)),
        )

        for pos, quat in invalid:
            with self.subTest(pos=pos, quat=quat):
                source = SurviveSource()
                source._publish("tracker", pos, quat)
                self.assertEqual(source.poses(), {})

        source = SurviveSource()
        source._publish("tracker", good_pos, good_quat)
        self.assertIn("tracker", source.poses())

    def test_worker_exception_is_rethrown_from_pose_health_check(self) -> None:
        source = SurviveSource()

        def fail() -> None:
            raise ValueError("USB read failed")

        source._run_worker(fail)
        with self.assertRaisesRegex(
            TrackerSourceError, "libsurvive reader failed.*USB read failed"
        ):
            source.poses()

    def test_survive_cli_exit_code_is_rethrown_from_pose_health_check(self) -> None:
        source = SurviveSource()
        source._cli_executable = Path("/attested/survive-cli")
        process = SimpleNamespace(stdout=[], poll=lambda: 17)
        with patch(
            "almond_axol.tracker.survive.subprocess.Popen", return_value=process
        ):
            source._run_worker(source._run_cli)

        with self.assertRaisesRegex(
            TrackerSourceError, "survive-cli exited unexpectedly.*17"
        ):
            source.poses()

    def test_survive_cli_maps_output_clock_but_labels_it_as_receipt(self) -> None:
        source = SurviveSource()
        source._cli_executable = Path("/attested/survive-cli")

        def lines():  # type: ignore[no-untyped-def]
            yield "1.000 T20 POSE 0 0 0 1 0 0 0\n"
            yield "2.000 T21 POSE 0 0 0 1 0 0 0\n"
            yield "3.000 T20 POSE 0 0 0 1 0 0 0\n"
            source._stop.set()

        process = SimpleNamespace(stdout=lines(), poll=lambda: 0)
        with (
            patch(
                "almond_axol.tracker.survive.subprocess.Popen",
                return_value=process,
            ),
            patch(
                "almond_axol.tracker.survive.time.perf_counter",
                side_effect=(11.1, 12.03, 13.2),
            ),
        ):
            source._run_cli()

        poses = source.poses()
        self.assertAlmostEqual(poses["T20"].t, 13.03)
        self.assertAlmostEqual(poses["T21"].t, 12.03)
        self.assertFalse(poses["T20"].timestamp_is_capture)

    def test_start_selects_only_manifest_attested_cli(self) -> None:
        source = SurviveSource()
        fake_thread = SimpleNamespace(start=Mock(), is_alive=Mock(return_value=True))
        with (
            patch(
                "almond_axol.cli.tracker_install.verified_survive_cli",
                return_value=Path("/attested/survive-cli"),
            ) as verified,
            patch(
                "almond_axol.tracker.survive.threading.Thread",
                return_value=fake_thread,
            ),
        ):
            source.start()

        verified.assert_called_once_with()
        fake_thread.start.assert_called_once_with()
        self.assertEqual(source._cli_executable, Path("/attested/survive-cli"))

    def test_start_interrupt_stops_a_partially_started_reader(self) -> None:
        source = SurviveSource()
        reader = SimpleNamespace(
            start=Mock(side_effect=KeyboardInterrupt),
            join=Mock(),
            is_alive=Mock(side_effect=(True, False)),
            ident=123,
        )
        with (
            patch(
                "almond_axol.cli.tracker_install.verified_survive_cli",
                return_value=Path("/attested/survive-cli"),
            ),
            patch("almond_axol.tracker.survive.threading.Thread", return_value=reader),
            self.assertRaises(KeyboardInterrupt),
        ):
            source.start()

        self.assertTrue(source._stop.is_set())
        reader.join.assert_called_once_with(timeout=3.0)
        self.assertIsNone(source._thread)

    def test_start_retains_reader_when_interrupt_cleanup_is_uncertain(self) -> None:
        source = SurviveSource()
        reader = SimpleNamespace(
            start=Mock(side_effect=KeyboardInterrupt),
            join=Mock(),
            is_alive=Mock(return_value=True),
            ident=123,
        )
        with (
            patch(
                "almond_axol.cli.tracker_install.verified_survive_cli",
                return_value=Path("/attested/survive-cli"),
            ),
            patch("almond_axol.tracker.survive.threading.Thread", return_value=reader),
            self.assertRaisesRegex(TrackerSourceError, "ownership is uncertain"),
        ):
            source.start()

        self.assertIs(source._thread, reader)
        with self.assertRaisesRegex(TrackerSourceError, "cleanup is incomplete"):
            source.start()

    def test_expected_stop_does_not_create_a_backend_failure(self) -> None:
        source = SurviveSource()
        source._stop.set()
        source._run_worker(lambda: None)
        self.assertEqual(source.poses(), {})

    def test_stop_kills_and_reaps_survive_cli(self) -> None:
        source = SurviveSource()
        process = SimpleNamespace(
            terminate=Mock(),
            wait=Mock(side_effect=(subprocess.TimeoutExpired("survive-cli", 3.0), 9)),
            kill=Mock(),
        )
        source._proc = process

        source.stop()

        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(process.wait.call_count, 2)
        self.assertIsNone(source._proc)

    def test_start_refuses_an_uncertain_retained_process(self) -> None:
        source = SurviveSource()
        source._proc = SimpleNamespace()

        with self.assertRaisesRegex(
            TrackerSourceError, "process cleanup is incomplete"
        ):
            source.start()

    def test_stop_surfaces_and_retains_lingering_reader(self) -> None:
        source = SurviveSource()
        reader = SimpleNamespace(
            join=Mock(),
            is_alive=Mock(return_value=True),
        )
        source._thread = reader

        with self.assertRaisesRegex(TrackerSourceError, "ownership is uncertain"):
            source.stop()

        self.assertIs(source._thread, reader)


class UltimateLifecycleTest(unittest.TestCase):
    def test_runtime_passes_pyvut_a_private_pinned_wifi_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "ultimate_wifi.json"
            original = {
                "ssid": "private-map",
                "pass": "never-log-this-password",
                "country": "US",
                "freq": 5180,
            }
            config.write_text(json.dumps(original))
            config.chmod(0o600)
            captured: dict[str, object] = {}

            class FakeHidDevice:
                nonblocking = False

                def close(self) -> None:
                    captured["hid_closed"] = True

            class FakeAPI:
                def __init__(self, **kwargs: object) -> None:
                    path = Path(str(kwargs["wifi_info_path"]))
                    captured["path"] = path
                    captured["values"] = json.loads(path.read_text())
                    self._thread = None
                    self.hid_device = FakeHidDevice()
                    self.tracker_group = SimpleNamespace(
                        comms=SimpleNamespace(device_hid1=self.hid_device)
                    )

                def add_pose_callback(self, callback: object) -> None:
                    captured["callback"] = callback

                def start(self) -> None:
                    captured["started"] = True

                def stop(self) -> None:
                    captured["stopped"] = True

            hid = ModuleType("hid")
            hid.Device = object  # type: ignore[attr-defined]
            pyvut = ModuleType("pyvut")
            pyvut.UltimateTrackerAPI = FakeAPI  # type: ignore[attr-defined]
            tracker_core = ModuleType("pyvut.tracker_core")
            tracker_core.set_tracker_core_verbose = Mock()  # type: ignore[attr-defined]

            source = UltimateSource()
            with (
                patch.object(tracker_ultimate, "ULTIMATE_WIFI_CONFIG_FILE", config),
                patch.dict(
                    sys.modules,
                    {
                        "hid": hid,
                        "pyvut": pyvut,
                        "pyvut.tracker_core": tracker_core,
                    },
                ),
            ):
                source.start()

            snapshot = captured["path"]
            assert isinstance(snapshot, Path)
            self.assertNotEqual(snapshot, config)
            self.assertEqual(captured["values"], original)
            self.assertEqual(snapshot.stat().st_mode & 0o777, 0o600)
            api = source._api
            assert api is not None
            self.assertTrue(api.hid_device.nonblocking)

            config.write_text(json.dumps({**original, "pass": "replacement-password"}))
            self.assertEqual(json.loads(snapshot.read_text()), original)

            source.stop()
            self.assertTrue(captured["stopped"])
            self.assertTrue(captured["hid_closed"])
            self.assertFalse(snapshot.exists())

    def test_wifi_state_rejects_symlink_without_exposing_credential(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            victim = root / "victim.json"
            secret = "do-not-return-this-secret"
            victim.write_text(
                json.dumps(
                    {
                        "ssid": "private-map",
                        "pass": secret,
                        "country": "US",
                        "freq": 5180,
                    }
                )
            )
            config = root / "ultimate_wifi.json"
            config.symlink_to(victim)

            state = tracker_ultimate.ultimate_wifi_config_state(config)

            self.assertEqual(state.status, "invalid")
            self.assertIn("cannot be read", state.error or "")
            self.assertNotIn(secret, state.error or "")
            self.assertNotIn(secret, repr(state))

    def test_pose_callback_converts_axes_and_preserves_tracking_status(self) -> None:
        source = UltimateSource(quat_order="wxyz", up_axis="z")
        half = np.sqrt(0.5)
        source._on_pose(
            SimpleNamespace(
                mac="AA:BB",
                position=[1.0, 2.0, 3.0],
                rotation=[half, 0.0, 0.0, half],
                tracking_status=2,
            )
        )

        sample = source.poses()["AA:BB"]
        np.testing.assert_allclose(sample.pos, [1.0, 3.0, -2.0])
        np.testing.assert_allclose(sample.quat, [0.0, half, 0.0, half])
        self.assertTrue(sample.tracking)

        source._on_pose(
            SimpleNamespace(
                mac="CC:DD",
                position=[0.0, 0.0, 0.0],
                rotation=[1.0, 0.0, 0.0, 0.0],
                tracking_status="lost",
            )
        )
        self.assertFalse(source.poses()["CC:DD"].tracking)

        source._on_pose(
            SimpleNamespace(
                mac="invalid",
                position=[np.nan, 0.0, 0.0],
                rotation=[1.0, 0.0, 0.0, 0.0],
                tracking_status="tracking",
            )
        )
        self.assertNotIn("invalid", source.poses())

    def test_pose_callback_maps_pyvut_receipt_epoch_without_claiming_capture(
        self,
    ) -> None:
        source = UltimateSource(quat_order="xyzw", up_axis="y")
        with (
            patch(
                "almond_axol.tracker.ultimate.time.time", return_value=1_700_000_000.050
            ),
            patch("almond_axol.tracker.ultimate.time.perf_counter", return_value=80.0),
        ):
            source._on_pose(
                SimpleNamespace(
                    mac="AA:BB",
                    position=[0.0, 0.0, 0.0],
                    rotation=[0.0, 0.0, 0.0, 1.0],
                    tracking_status=2,
                    timestamp_ms=1_700_000_000_025,
                )
            )

        sample = source.poses()["AA:BB"]
        self.assertAlmostEqual(sample.t, 79.975, places=5)
        self.assertFalse(sample.timestamp_is_capture)

    def test_pinned_running_api_rejects_absent_or_dead_reader_thread(self) -> None:
        source = UltimateSource()
        running = threading.Event()
        running.set()

        for reader, detail in (
            (None, "reader thread is absent"),
            (SimpleNamespace(is_alive=lambda: False), "reader thread has stopped"),
        ):
            with self.subTest(detail=detail):
                source._api = SimpleNamespace(_running=running, _thread=reader)
                with self.assertRaisesRegex(TrackerSourceError, detail):
                    source.poses()

    def test_unknown_or_live_compatible_api_internals_do_not_false_fail(self) -> None:
        source = UltimateSource()
        source._api = SimpleNamespace()
        self.assertEqual(source.poses(), {})

        running = threading.Event()
        running.set()
        source._api = SimpleNamespace(
            _running=running,
            _thread=SimpleNamespace(is_alive=lambda: True),
        )
        self.assertEqual(source.poses(), {})

    def test_stop_surfaces_and_retains_lingering_pyvut_reader(self) -> None:
        source = UltimateSource()
        reader = SimpleNamespace(
            join=Mock(),
            is_alive=Mock(return_value=True),
        )
        api = SimpleNamespace(_thread=reader, stop=Mock())
        source._api = api

        with self.assertRaisesRegex(TrackerSourceError, "HID ownership is uncertain"):
            source.stop()

        api.stop.assert_called_once_with()
        reader.join.assert_called_once()
        self.assertIs(source._api, api)

    def test_uncertain_stop_is_latched_without_repeating_native_cleanup(
        self,
    ) -> None:
        source = UltimateSource()
        reader = SimpleNamespace(join=Mock(), is_alive=Mock(return_value=False))
        closed_device = SimpleNamespace(close=Mock())
        uncertain_device = SimpleNamespace(
            close=Mock(side_effect=OSError("close uncertain"))
        )
        api = SimpleNamespace(
            _thread=reader,
            stop=Mock(),
            tracker_group=SimpleNamespace(
                comms=SimpleNamespace(
                    device_hid1=closed_device,
                    device_hid3=uncertain_device,
                )
            ),
        )
        wifi_material = Mock()
        source._api = api
        source._wifi_material = wifi_material

        with self.assertRaisesRegex(
            TrackerSourceError, "HID ownership is uncertain"
        ) as first:
            source.stop()
        with self.assertRaises(TrackerSourceError) as second:
            source.stop()
        with self.assertRaisesRegex(
            TrackerSourceError, "cleanup is incomplete.*ownership is uncertain"
        ):
            source.start()

        self.assertIs(second.exception, first.exception)
        api.stop.assert_called_once_with()
        reader.join.assert_called_once_with(
            timeout=tracker_ultimate._READER_STOP_TIMEOUT_S
        )
        closed_device.close.assert_called_once_with()
        uncertain_device.close.assert_called_once_with()
        wifi_material.cleanup.assert_called_once_with()
        self.assertIsNone(source._wifi_material)
        self.assertIs(source._api, api)

    def test_failed_start_cleanup_failure_is_latched_without_retry(self) -> None:
        devices: list[object] = []
        api_instances: list[object] = []

        class FakeHidDevice:
            nonblocking = False

            def __init__(self) -> None:
                self.close = Mock()
                devices.append(self)

        class FakeAPI:
            def __init__(self, **_kwargs: object) -> None:
                self._thread = None
                self.stop = Mock(side_effect=OSError("stop uncertain"))
                self.device = FakeHidDevice()
                self.tracker_group = SimpleNamespace(
                    comms=SimpleNamespace(device_hid1=self.device)
                )
                api_instances.append(self)

            def add_pose_callback(self, _callback: object) -> None:
                pass

            def start(self) -> None:
                raise OSError("start failed")

        hid = ModuleType("hid")
        hid.Device = object  # type: ignore[attr-defined]
        pyvut = ModuleType("pyvut")
        pyvut.UltimateTrackerAPI = FakeAPI  # type: ignore[attr-defined]
        tracker_core = ModuleType("pyvut.tracker_core")
        tracker_core.set_tracker_core_verbose = Mock()  # type: ignore[attr-defined]
        source = UltimateSource()

        with patch.dict(
            sys.modules,
            {
                "hid": hid,
                "pyvut": pyvut,
                "pyvut.tracker_core": tracker_core,
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "dongle"):
                source.start()
            with self.assertRaisesRegex(TrackerSourceError, "cleanup is incomplete"):
                source.start()
            with self.assertRaises(TrackerSourceError) as stopped:
                source.stop()

        self.assertEqual(len(api_instances), 1)
        api = api_instances[0]
        api.stop.assert_called_once_with()
        self.assertEqual(len(devices), 1)
        devices[0].close.assert_called_once_with()
        self.assertIs(stopped.exception, source._teardown_failure)
        self.assertIs(source._api, api)

    def test_keyboard_interrupt_during_start_releases_hid_and_wifi_snapshot(
        self,
    ) -> None:
        devices: list[object] = []
        api_instances: list[object] = []

        class FakeHidDevice:
            nonblocking = False

            def __init__(self) -> None:
                self.close = Mock()
                devices.append(self)

        hid = ModuleType("hid")
        hid.Device = FakeHidDevice  # type: ignore[attr-defined]

        class FakeAPI:
            def __init__(self, **_kwargs: object) -> None:
                self._thread = None
                self.stop = Mock()
                self.device = hid.Device()  # type: ignore[attr-defined]
                self.tracker_group = SimpleNamespace(
                    comms=SimpleNamespace(device_hid1=self.device)
                )
                api_instances.append(self)

            def add_pose_callback(self, _callback: object) -> None:
                pass

            def start(self) -> None:
                raise KeyboardInterrupt

        pyvut = ModuleType("pyvut")
        pyvut.UltimateTrackerAPI = FakeAPI  # type: ignore[attr-defined]
        tracker_core = ModuleType("pyvut.tracker_core")
        tracker_core.set_tracker_core_verbose = Mock()  # type: ignore[attr-defined]
        wifi_material = Mock()
        source = UltimateSource()

        with (
            patch.dict(
                sys.modules,
                {
                    "hid": hid,
                    "pyvut": pyvut,
                    "pyvut.tracker_core": tracker_core,
                },
            ),
            patch.object(
                tracker_ultimate,
                "ultimate_wifi_config_state",
                return_value=tracker_ultimate.UltimateWifiConfigState(
                    "valid",
                    None,
                    values={
                        "ssid": "map",
                        "pass": "password",
                        "country": "US",
                        "freq": 5180,
                    },
                ),
            ),
            patch.object(
                tracker_ultimate,
                "_private_wifi_snapshot",
                return_value=(wifi_material, "/private/wifi_info.json"),
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            source.start()

        self.assertEqual(len(api_instances), 1)
        self.assertEqual(len(devices), 1)
        api_instances[0].stop.assert_called_once_with()
        devices[0].close.assert_called_once_with()
        wifi_material.cleanup.assert_called_once_with()
        self.assertIsNone(source._api)
        self.assertIsNone(source._wifi_material)
        self.assertIsNone(source._teardown_failure)

    def test_constructor_failure_closes_unpublished_hid_and_can_retry(self) -> None:
        devices: list[object] = []
        attempts = 0

        class FakeDevice:
            nonblocking = False

            def __init__(self, **_kwargs: object) -> None:
                self.close = Mock()
                devices.append(self)

        hid = ModuleType("hid")
        hid.Device = FakeDevice  # type: ignore[attr-defined]

        class FakeAPI:
            def __init__(self, **_kwargs: object) -> None:
                nonlocal attempts
                attempts += 1
                device = hid.Device(path="/dev/hidraw-test")  # type: ignore[attr-defined]
                if attempts == 1:
                    raise OSError("feature report failed after open")
                self._thread = None
                self.tracker_group = SimpleNamespace(
                    comms=SimpleNamespace(device_hid1=device)
                )

            def add_pose_callback(self, _callback: object) -> None:
                pass

            def start(self) -> None:
                pass

            def stop(self) -> None:
                pass

        pyvut = ModuleType("pyvut")
        pyvut.UltimateTrackerAPI = FakeAPI  # type: ignore[attr-defined]
        tracker_core = ModuleType("pyvut.tracker_core")
        tracker_core.set_tracker_core_verbose = Mock()  # type: ignore[attr-defined]
        source = UltimateSource()

        with patch.dict(
            sys.modules,
            {
                "hid": hid,
                "pyvut": pyvut,
                "pyvut.tracker_core": tracker_core,
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "dongle"):
                source.start()
            self.assertIs(hid.Device, FakeDevice)  # type: ignore[attr-defined]
            source.start()
            source.stop()

        self.assertEqual(attempts, 2)
        self.assertEqual(len(devices), 2)
        devices[0].close.assert_called_once_with()
        devices[1].close.assert_called_once_with()
        self.assertIsNone(source._teardown_failure)

    def test_uncertain_constructor_close_is_latched_without_retry(self) -> None:
        devices: list[object] = []
        attempts = 0

        class FakeDevice:
            def __init__(self, **_kwargs: object) -> None:
                self.close = Mock(side_effect=OSError("close uncertain"))
                devices.append(self)

        hid = ModuleType("hid")
        hid.Device = FakeDevice  # type: ignore[attr-defined]

        class FakeAPI:
            def __init__(self, **_kwargs: object) -> None:
                nonlocal attempts
                attempts += 1
                hid.Device(path="/dev/hidraw-test")  # type: ignore[attr-defined]
                raise OSError("feature report failed after open")

        pyvut = ModuleType("pyvut")
        pyvut.UltimateTrackerAPI = FakeAPI  # type: ignore[attr-defined]
        tracker_core = ModuleType("pyvut.tracker_core")
        tracker_core.set_tracker_core_verbose = Mock()  # type: ignore[attr-defined]
        source = UltimateSource()

        with patch.dict(
            sys.modules,
            {
                "hid": hid,
                "pyvut": pyvut,
                "pyvut.tracker_core": tracker_core,
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "dongle"):
                source.start()
            with self.assertRaisesRegex(TrackerSourceError, "cleanup is incomplete"):
                source.start()
            with self.assertRaises(TrackerSourceError) as stopped:
                source.stop()

        self.assertEqual(attempts, 1)
        self.assertEqual(len(devices), 1)
        devices[0].close.assert_called_once_with()
        self.assertEqual(source._orphaned_hid_devices, (devices[0],))
        self.assertIs(stopped.exception, source._teardown_failure)

    def test_stop_does_not_close_hid_under_lingering_reader(self) -> None:
        source = UltimateSource()
        reader = SimpleNamespace(join=Mock(), is_alive=Mock(return_value=True))
        device = SimpleNamespace(close=Mock())
        api = SimpleNamespace(
            _thread=reader,
            stop=Mock(),
            tracker_group=SimpleNamespace(comms=SimpleNamespace(device_hid1=device)),
        )
        source._api = api

        with self.assertRaisesRegex(TrackerSourceError, "HID ownership is uncertain"):
            source.stop()

        device.close.assert_not_called()

    def test_successful_stop_closes_hid_once_and_is_idempotent(self) -> None:
        source = UltimateSource()
        reader = SimpleNamespace(join=Mock(), is_alive=Mock(return_value=False))
        device = SimpleNamespace(close=Mock())
        api = SimpleNamespace(
            _thread=reader,
            stop=Mock(),
            tracker_group=SimpleNamespace(comms=SimpleNamespace(device_hid1=device)),
        )
        source._api = api

        source.stop()
        source.stop()

        api.stop.assert_called_once_with()
        device.close.assert_called_once_with()
        self.assertIsNone(source._api)


class BridgeFatalityTest(unittest.IsolatedAsyncioTestCase):
    async def test_source_failure_is_fatal_instead_of_reconnectable(self) -> None:
        bridge = TrackerBridge(
            _FatalSource(),  # type: ignore[arg-type]
            left="left",
            right="right",
            controls=StopEventControls(threading.Event()),
        )
        with (
            patch("websockets.connect", return_value=_FakeConnection()),
            self.assertRaisesRegex(TrackerSourceError, "hardware reader died"),
        ):
            await asyncio.wait_for(bridge.run(), timeout=1.0)


class IdentifyDiagnosticsTest(unittest.TestCase):
    def test_libsurvive_warnings_are_captured_once_without_transients(self) -> None:
        from almond_axol.tracker import survive

        clash = (
            "\x1b[0;31mWarning: Two or more lighthouses are on channel 0; "
            "tracking is most likely going to fail.\x1b[0m\n"
        )
        self.assertEqual(
            survive._setup_warning(clash),  # noqa: SLF001
            "Two or more lighthouses are on channel 0; tracking is most likely "
            "going to fail.",
        )
        self.assertIsNone(
            survive._setup_warning(  # noqa: SLF001
                "\x1b[0;31mWarning: Could not lighthouse more to T20 (5)\n"
            )
        )
        self.assertIsNone(
            survive._setup_warning(  # noqa: SLF001
                "1.018921 INFO LOG Two or more lighthouses are on channel 0\n"
            )
        )
        self.assertIsNone(
            survive._setup_warning("0.1 T20 POSE 0 0 0 1 0 0 0\n")  # noqa: SLF001
        )

        source = SurviveSource()
        source._note_warning("clash")  # noqa: SLF001
        source._note_warning("clash")  # noqa: SLF001
        source._note_warning("other")  # noqa: SLF001
        self.assertEqual(source.warnings(), ["clash", "other"])

    def test_failed_capture_names_every_device_and_its_motion(self) -> None:
        summary = tracker_identify._motion_summary(  # noqa: SLF001
            {"T20": 0.4123, "T21": 0.01},
            {"T20", "T21", "T22"},
            {"left": "T20"},
        )
        self.assertEqual(
            summary,
            "T20 0.41 m (already left), T21 0.01 m, T22 no fresh poses",
        )
        self.assertEqual(
            tracker_identify._motion_summary({}, set(), {}),  # noqa: SLF001
            "no devices reported",
        )


class BridgeCleanupTest(unittest.TestCase):
    def test_identify_start_interrupt_still_stops_partial_backend(self) -> None:
        source = SimpleNamespace(
            start=Mock(side_effect=KeyboardInterrupt),
            stop=Mock(),
        )
        config = SimpleNamespace(backend="synthetic")
        args = SimpleNamespace(backend=None, web_prompts=False)
        with (
            patch("almond_axol.tracker.load_tracker_config", return_value=config),
            patch("almond_axol.tracker.create_source", return_value=source),
            self.assertRaises(KeyboardInterrupt),
        ):
            tracker_identify.run(args)

        source.start.assert_called_once_with()
        source.stop.assert_called_once_with()

    def test_source_start_interrupt_still_stops_partial_backend(self) -> None:
        source = SimpleNamespace(
            start=Mock(side_effect=KeyboardInterrupt),
            stop=Mock(),
        )
        config = SimpleNamespace(
            backend="survive",
            left="left",
            right="right",
            allow_single_side=False,
            trigger_can_left=None,
            trigger_can_right=None,
        )

        with patch("almond_axol.tracker.create_source", return_value=source):
            run_configured_bridge(config)

        source.start.assert_called_once_with()
        source.stop.assert_called_once_with()

    def test_trigger_close_failure_does_not_skip_source_stop(self) -> None:
        source = SimpleNamespace(start=Mock(), stop=Mock())
        left_reader = SimpleNamespace(close=Mock(side_effect=OSError("left close")))
        right_reader = SimpleNamespace(close=Mock())
        bridge = SimpleNamespace(run=AsyncMock())
        config = SimpleNamespace(
            backend="survive",
            left="left",
            right="right",
            allow_single_side=False,
            trigger_can_left="can-left",
            trigger_can_right="can-right",
        )
        with (
            patch("almond_axol.tracker.create_source", return_value=source),
            patch(
                "almond_axol.tracker.trigger.TriggerReader",
                side_effect=(left_reader, right_reader),
            ),
            patch("almond_axol.tracker.bridge.TrackerBridge", return_value=bridge),
            self.assertRaisesRegex(TrackerSourceError, "trigger teardown failed"),
        ):
            run_configured_bridge(config)

        source.stop.assert_called_once_with()
        left_reader.close.assert_called_once_with()
        right_reader.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
