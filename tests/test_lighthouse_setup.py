"""Lighthouse setup: calibration file, calibrating survey, already-paired dongles."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from almond_axol.cli import tracker_lighthouse, tracker_pair
from almond_axol.tracker import survive
from almond_axol.tracker.base import TrackerPose
from almond_axol.tracker.lighthouse_survey import LighthouseSurvey


class _Clock:
    """A perf_counter/monotonic stand-in that advances a fixed step per read."""

    def __init__(self, step: float) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


class SurviveConfigPathTest(unittest.TestCase):
    def test_follows_xdg_then_the_passwd_home(self) -> None:
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": "/xdg"}):
            self.assertEqual(
                survive.survive_config_path(), Path("/xdg/libsurvive/config.json")
            )
        env = {k: v for k, v in os.environ.items() if k not in ("XDG_CONFIG_HOME",)}
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(survive.Path, "home", return_value=Path("/root")),
        ):
            self.assertEqual(
                survive.survive_config_path(),
                Path("/root/.config/libsurvive/config.json"),
            )

    def test_root_adopts_the_service_era_calibration_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "config.json"
            legacy.write_text('"configed-lighthouse-gen":"2",\n"lighthouse0":{}\n')
            with (
                patch.dict(os.environ, {"XDG_CONFIG_HOME": f"{tmp}/xdg"}),
                patch.object(survive, "_LEGACY_ROOTLESS_CONFIG", legacy),
                patch.object(survive.os, "geteuid", return_value=0),
            ):
                path = survive.prepare_survive_config()
                self.assertEqual(path.read_text(), legacy.read_text())
                # An existing calibration is never overwritten.
                path.write_text("newer")
                survive.prepare_survive_config()
                self.assertEqual(path.read_text(), "newer")

    def test_operator_runs_never_adopt_the_root_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "config.json"
            legacy.write_text('"configed-lighthouse-gen":"2",\n')
            with (
                patch.dict(os.environ, {"XDG_CONFIG_HOME": f"{tmp}/xdg"}),
                patch.object(survive, "_LEGACY_ROOTLESS_CONFIG", legacy),
                patch.object(survive.os, "geteuid", return_value=1000),
            ):
                path = survive.prepare_survive_config()
            self.assertTrue(path.parent.is_dir())
            self.assertFalse(path.exists())

    def test_tracking_passes_the_calibration_file_explicitly(self) -> None:
        source = survive.SurviveSource()
        source._cli_executable = Path("/attested/survive-cli")
        source._stop.set()
        process = SimpleNamespace(stdout=iter(()), poll=lambda: 0)
        with patch.object(survive.subprocess, "Popen", return_value=process) as popen:
            source._run_cli()
        args = popen.call_args.args[0]
        self.assertEqual(
            args[1:3], ["--configfile", str(survive.survive_config_path())]
        )


class _SurveySource:
    """Live stations from the start; trackers report from ``tracking_at`` on."""

    def __init__(self, clock: _Clock, *, live: int, tracking_at: float) -> None:
        self.clock = clock
        self.live = live
        self.tracking_at = tracking_at

    def lighthouse_survey(self) -> LighthouseSurvey:
        survey = LighthouseSurvey()
        for channel in range(self.live):
            survey.note_channel(channel)
        return survey

    def poses(self) -> dict[str, TrackerPose]:
        if self.clock.now < self.tracking_at:
            return {}
        return {
            key: TrackerPose(
                pos=np.zeros(3), quat=np.array([0.0, 0.0, 0.0, 1.0]), t=self.clock.now
            )
            for key in ("WM0", "WM1")
        }


class CalibratingSurveyTest(unittest.TestCase):
    def _survey(
        self, *, live: int, tracking_at: float
    ) -> tuple[LighthouseSurvey, Mock, float]:
        clock = _Clock(0.5)
        source = _SurveySource(clock, live=live, tracking_at=tracking_at)
        on_calibrating = Mock()
        with (
            patch.object(tracker_lighthouse.time, "perf_counter", clock),
            patch.object(tracker_lighthouse.time, "sleep"),
        ):
            survey = tracker_lighthouse.survey_lighthouses(
                source, on_calibrating=on_calibrating
            )
        return survey, on_calibrating, clock.now

    def test_waits_for_a_calibration_that_outlasts_the_window(self) -> None:
        survey, on_calibrating, now = self._survey(live=4, tracking_at=92.0)

        self.assertEqual(survey.trackers, {"WM0", "WM1"})
        on_calibrating.assert_called_once_with(4)
        self.assertLess(now, 100.0)  # stops as soon as both trackers report

    def test_trackers_already_reporting_end_at_the_normal_window(self) -> None:
        survey, on_calibrating, now = self._survey(live=2, tracking_at=0.0)

        self.assertEqual(survey.trackers, {"WM0", "WM1"})
        on_calibrating.assert_not_called()
        self.assertLess(now, 25.0)

    def test_no_live_station_means_nothing_to_wait_for(self) -> None:
        survey, on_calibrating, now = self._survey(live=0, tracking_at=1e9)

        self.assertEqual(survey.trackers, set())
        on_calibrating.assert_not_called()
        self.assertLess(now, 25.0)

    def test_gives_up_after_the_calibration_limit(self) -> None:
        survey, on_calibrating, now = self._survey(live=4, tracking_at=1e9)

        self.assertEqual(survey.trackers, set())
        on_calibrating.assert_called_once_with(4)
        self.assertGreaterEqual(now, tracker_lighthouse._CALIBRATION_S)
        self.assertLess(now, tracker_lighthouse._CALIBRATION_S + 5.0)


class _PipeProcess:
    """A survive-cli stand-in whose stdout is a real pipe, as selectors needs."""

    def __init__(self, lines: list[str]) -> None:
        read_fd, write_fd = os.pipe()
        with os.fdopen(write_fd, "w") as writer:
            writer.writelines(f"{line}\n" for line in lines)
        self.stdout = os.fdopen(read_fd, "r")
        self.returncode = None

    def poll(self) -> None:
        return None


class AlreadyPairedTest(unittest.TestCase):
    def _pair(self, lines: list[str]) -> tuple[str, list[str]]:
        process = _PipeProcess(lines)
        with (
            patch("almond_axol.cli.mantis_bridge.require_mantis_tracker_readiness"),
            patch(
                "almond_axol.cli.tracker_install.verified_survive_cli",
                return_value=Path("/attested/survive-cli"),
            ),
            patch.object(
                tracker_pair.subprocess, "Popen", return_value=process
            ) as popen,
            patch.object(tracker_pair.time, "monotonic", _Clock(1.0)),
            patch.object(tracker_pair, "_stop_process"),
            patch("sys.stdout", io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            tracker_pair.run(SimpleNamespace(timeout=90.0))
        process.stdout.close()
        return str(raised.exception), popen.call_args.args[0]

    def test_powered_paired_trackers_are_reported_as_already_paired(self) -> None:
        message, args = self._pair(
            [
                "Info: Adding tracked object \x1b[0;31mWM0\x1b[0m from \x1b[0;34mHTC\x1b[0m",
                "Info: Adding tracked object \x1b[0;32mWM1\x1b[0m from \x1b[0;34mHTC\x1b[0m",
            ]
        )

        self.assertIn("Already paired", message)
        self.assertIn("WM0, WM1", message)
        self.assertNotIn("tracker.install", message)
        self.assertEqual(
            args[1:3], ["--configfile", str(survive.survive_config_path())]
        )

    def test_a_dongle_that_never_pairs_still_points_at_the_install(self) -> None:
        message, _ = self._pair(["Info: Successfully enumerated Watchman"])

        self.assertIn("did not enter pairing mode", message)
        self.assertIn("axol tracker.install", message)


if __name__ == "__main__":
    unittest.main()


def _config_line(codename: str, serial: str) -> str:
    return (
        f'0.5 {codename} CONFIG {{ "device_class": "generic_tracker", '
        f'"device_serial_number": "{serial}" }}\n'
    )


class TrackerSerialKeyTest(unittest.TestCase):
    def _poses(self, lines: list[str]) -> dict[str, TrackerPose]:
        source = survive.SurviveSource()
        source._cli_executable = Path("/attested/survive-cli")

        def stream():  # type: ignore[no-untyped-def]
            yield from lines
            source._stop.set()

        process = SimpleNamespace(stdout=stream(), poll=lambda: 0)
        with patch.object(survive.subprocess, "Popen", return_value=process):
            source._run_cli()
        return source.poses()

    def test_poses_follow_the_tracker_not_the_dongle_slot(self) -> None:
        first = self._poses(
            [
                _config_line("WM0", "LHR-E0FCD01B"),
                _config_line("WM1", "LHR-BDD65467"),
                "1.0 WM0 POSE 0.1 0 0 1 0 0 0\n",
                "1.0 WM1 POSE 0.2 0 0 1 0 0 0\n",
            ]
        )
        # The next run opens the dongles in the other order: the codenames
        # swap (seen on the AGX Orin), but each serial keeps its own pose.
        second = self._poses(
            [
                _config_line("WM0", "LHR-BDD65467"),
                _config_line("WM1", "LHR-E0FCD01B"),
                "1.0 WM0 POSE 0.2 0 0 1 0 0 0\n",
                "1.0 WM1 POSE 0.1 0 0 1 0 0 0\n",
            ]
        )

        for poses in (first, second):
            self.assertEqual(set(poses), {"LHR-E0FCD01B", "LHR-BDD65467"})
            self.assertAlmostEqual(float(poses["LHR-E0FCD01B"].pos[0]), 0.1)
            self.assertAlmostEqual(float(poses["LHR-BDD65467"].pos[0]), 0.2)

    def test_a_device_without_a_serial_is_never_published(self) -> None:
        poses = self._poses(["1.0 WM0 POSE 0 0 0 1 0 0 0\n"])
        self.assertEqual(poses, {})


class CodenameBindingMigrationTest(unittest.TestCase):
    def test_codename_bindings_are_dropped_so_setup_asks_for_identify(self) -> None:
        from almond_axol.tracker.config import load_tracker_config

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                '{"backend": "survive", "left": "WM1", "right": "WM0", '
                '"bindings": {"survive": {"left": "WM1", "right": "WM0"}, '
                '"ultimate": {"left": "aa:bb", "right": "cc:dd"}}}'
            )
            path.chmod(0o600)

            config = load_tracker_config(path)
            reloaded = load_tracker_config(path)

        for loaded in (config, reloaded):
            self.assertIsNone(loaded.left)
            self.assertIsNone(loaded.right)
            self.assertEqual(loaded.bindings["survive"], {"left": None, "right": None})
            self.assertEqual(
                loaded.bindings["ultimate"], {"left": "aa:bb", "right": "cc:dd"}
            )

    def test_serial_bindings_are_kept(self) -> None:
        from almond_axol.tracker.config import load_tracker_config

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                '{"backend": "survive", "left": "LHR-E0FCD01B", '
                '"right": "LHR-BDD65467"}'
            )
            path.chmod(0o600)
            config = load_tracker_config(path)

        self.assertEqual((config.left, config.right), ("LHR-E0FCD01B", "LHR-BDD65467"))
