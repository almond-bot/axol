from __future__ import annotations

import asyncio
import io
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from almond_axol.cli.mantis_bridge import managed_mantis_bridge
from almond_axol.tracker.base import TrackerSourceError
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


class UltimateLifecycleTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
