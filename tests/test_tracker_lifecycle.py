from __future__ import annotations

import asyncio
import io
import subprocess
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from almond_axol.cli.mantis_bridge import managed_mantis_bridge
from almond_axol.cli.tracker_bridge import run_configured_bridge
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


class SurviveLifecycleTest(unittest.TestCase):
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
        process = SimpleNamespace(stdout=[], poll=lambda: 17)
        with patch(
            "almond_axol.tracker.survive.subprocess.Popen", return_value=process
        ):
            source._run_worker(source._run_cli)

        with self.assertRaisesRegex(
            TrackerSourceError, "survive-cli exited unexpectedly.*17"
        ):
            source.poses()

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


class BridgeCleanupTest(unittest.TestCase):
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
