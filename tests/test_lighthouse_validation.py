from __future__ import annotations

import argparse
import json
import math
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from almond_axol.cli import tracker_lighthouse_calibrate as command
from almond_axol.serve.commands import COMMANDS, get_schema
from almond_axol.tracker.base import TrackerPose
from almond_axol.tracker.config import TrackerConfig
from almond_axol.tracker.lighthouse_survey import LighthouseSurvey
from almond_axol.tracker.lighthouse_validation import (
    StationaryLimits,
    acquire_lighthouse_runtime_lock,
    collect_stationary_check,
    collect_workspace_check,
    invalidate_lighthouse_validation,
    new_validation_report,
    release_lighthouse_runtime_lock,
    require_lighthouse_validation,
    save_validation_report,
    stationary_metrics,
    validation_fingerprint,
    validation_snapshot,
)


_NATIVE_CONFIG = (
    Path(__file__).parent / "fixtures" / "libsurvive-config.native"
).read_bytes()
_UPDATED_NATIVE_CONFIG = _NATIVE_CONFIG.replace(b"-0.481664836407", b"-0.500000000000")


def _config() -> TrackerConfig:
    return TrackerConfig(left="LHR-left", right="LHR-right", expected_base_stations=3)


def _survey() -> LighthouseSurvey:
    return LighthouseSurvey(
        channels={0: {"aaaa"}, 2: {"bbbb"}, 4: {"cccc"}}, expected=3
    )


class _Clock:
    now = 10.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, delay: float) -> None:
        self.now += delay


class _Source:
    def __init__(
        self, clock: _Clock, *, held: bool = False, jitter: float = 0.0
    ) -> None:
        self.clock = clock
        self.held = held
        self.jitter = jitter
        self.quality_status = "ready"

    def poses(self) -> dict[str, TrackerPose]:
        stamp = 10.0 if self.held else self.clock()
        return {
            serial: TrackerPose(
                np.array([self.jitter * math.sin(stamp * 70), 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0, 1.0]),
                stamp,
            )
            for serial in ("LHR-left", "LHR-right")
        }

    def tracking_quality(self) -> dict:
        return {
            "status": self.quality_status,
            "reasons": [],
            "faultCount": 0,
            "trackers": {
                serial: {
                    "stations": {
                        str(n): {"serial": key, "sensorCount": 6, "opticalAgeMs": 1.0}
                        for n, key in enumerate(("aaaa", "bbbb", "cccc"))
                    }
                }
                for serial in ("LHR-left", "LHR-right")
            },
        }

    def lighthouse_survey(self) -> LighthouseSurvey:
        return _survey()


class AcquisitionReadinessTest(unittest.TestCase):
    def _run_wait(self, source: _Source, clock: _Clock, **kwargs: float) -> list[str]:
        with (
            patch.object(command.time, "perf_counter", clock),
            patch.object(command.time, "sleep", clock.sleep),
            patch("builtins.print") as output,
        ):
            command._wait_for_ready(source, 3, **kwargs)
        return [str(call.args[0]) for call in output.call_args_list]

    def test_late_serial_decoding_keeps_same_source_until_all_stations_ready(
        self,
    ) -> None:
        clock = _Clock()

        class LateIdentitySource(_Source):
            def lighthouse_survey(self) -> LighthouseSurvey:
                survey = _survey()
                if self.clock() < 60.0:  # 50 seconds after acquisition started
                    survey.channels[4] = set()
                return survey

        source = LateIdentitySource(clock)
        log = self._run_wait(source, clock)
        self.assertGreaterEqual(clock.now, 60.0)
        self.assertLess(clock.now, 60.2)
        self.assertIn("0/120 s", log[0])
        self.assertIn("3/3 station channels seen", log[0])
        self.assertIn("2/3 identities decoded", log[0])
        self.assertIn("identity pending on channels 5", log[0])
        self.assertIn("3/3 identities decoded", log[-1])
        self.assertGreater(len(log), 9)

    def test_undecoded_serial_has_bounded_timeout_and_specific_recovery(self) -> None:
        clock = _Clock()
        source = _Source(clock)
        survey = _survey()
        survey.channels[4] = set()
        source.lighthouse_survey = lambda: survey
        with self.assertRaisesRegex(
            RuntimeError, "identity broadcast decoding timed out"
        ) as error:
            self._run_wait(source, clock)
        self.assertAlmostEqual(
            clock.now, 10.0 + command.LIGHTHOUSE_ACQUISITION_TIMEOUT_S
        )
        message = str(error.exception)
        self.assertIn(
            "all 3 station channels were seen and live tracking was ready", message
        )
        self.assertIn("pending channels 5", message)
        self.assertIn("calibration was not replaced", message)
        self.assertIn("Reposition the rigs", message)

    def test_missing_optical_station_remains_a_failure(self) -> None:
        clock = _Clock()
        source = _Source(clock)
        survey = _survey()
        del survey.channels[4]
        source.lighthouse_survey = lambda: survey
        with self.assertRaisesRegex(RuntimeError, "only 2 of 3 base stations"):
            self._run_wait(source, clock, timeout_s=1.0)

    def test_all_identities_do_not_override_bad_tracking(self) -> None:
        clock = _Clock()
        source = _Source(clock)
        source.quality_status = "error"
        with self.assertRaisesRegex(RuntimeError, "Tracking has not stabilized"):
            self._run_wait(source, clock, timeout_s=1.0)

    def test_persistent_station_fault_requests_recalibration_without_waiting(
        self,
    ) -> None:
        clock = _Clock()
        source = _Source(clock)
        source.tracking_quality = lambda: {
            "status": "error",
            "calibrationRequired": True,
            "reasons": ["Base station estimate changed"],
        }
        with self.assertRaisesRegex(RuntimeError, "Run Recalibrate Lighthouse"):
            self._run_wait(source, clock)
        self.assertEqual(clock.now, 10.0)

    def test_serial_collision_is_not_reported_as_slow_decoding(self) -> None:
        clock = _Clock()
        source = _Source(clock)
        survey = _survey()
        survey.channels[0].add("dddd")
        survey.channels[4] = set()
        source.lighthouse_survey = lambda: survey
        with self.assertRaisesRegex(RuntimeError, "both set to channel 1"):
            self._run_wait(source, clock, timeout_s=1.0)


class ValidationMetricsTest(unittest.TestCase):
    def test_unidentified_pair_uses_neutral_labels_for_consistency(self) -> None:
        clock = _Clock()
        source = _Source(clock)
        metrics, problems, _, _ = collect_stationary_check(
            source,
            {"tracker1": "LHR-left", "tracker2": "LHR-right"},
            expected_stations=3,
            clock=clock,
            sleep=clock.sleep,
        )
        self.assertFalse(problems)
        self.assertEqual(set(metrics), {"tracker1", "tracker2"})

    def test_quaternion_sign_flip_is_not_orientation_noise(self) -> None:
        samples = [
            TrackerPose(np.zeros(3), np.array([0.0, 0.0, 0.0, (-1.0) ** n]), n / 100)
            for n in range(100)
        ]
        metrics = stationary_metrics(samples)
        self.assertEqual(metrics["anglePeakDeg"], 0)
        self.assertEqual(metrics["positionPeakMm"], 0)

    def test_stationary_samples_pass_but_held_and_noisy_samples_fail(self) -> None:
        for held, jitter, succeeds in (
            (False, 0.0, True),
            (True, 0.0, False),
            (False, 0.03, False),
        ):
            with self.subTest(held=held, jitter=jitter):
                clock = _Clock()
                source = _Source(clock, held=held, jitter=jitter)
                metrics, problems, _, _ = collect_stationary_check(
                    source,
                    {"left": "LHR-left", "right": "LHR-right"},
                    expected_stations=3,
                    limits=StationaryLimits(duration_s=1.0),
                    clock=clock,
                    sleep=clock.sleep,
                )
                self.assertEqual(not problems, succeeds, problems)
                if succeeds:
                    self.assertGreater(metrics["left"]["samples"], 50)

    def test_workspace_requires_motion_not_just_repeated_fresh_stationary_pose(
        self,
    ) -> None:
        clock = _Clock()
        _, problems = collect_workspace_check(
            _Source(clock),
            {"left": "LHR-left", "right": "LHR-right"},
            expected_stations=3,
            duration_s=1.0,
            clock=clock,
            sleep=clock.sleep,
        )
        self.assertTrue(any("wider slow sweep" in problem for problem in problems))

    def test_three_channels_need_three_distinct_identities(self) -> None:
        for channels in (
            {0: {"a"}, 1: {"b"}, 2: set()},
            {0: {"a"}, 1: {"b"}, 2: {"a"}},
        ):
            with self.subTest(channels=channels):
                self.assertTrue(
                    LighthouseSurvey(channels=channels, expected=3).problems(
                        require_serials=True
                    )
                )
        self.assertEqual(_survey().problems(require_serials=True), [])

    def test_quality_loss_at_end_of_hold_is_not_missed(self) -> None:
        clock = _Clock()

        class TerminalLossSource(_Source):
            def tracking_quality(self) -> dict:
                result = super().tracking_quality()
                if self.clock() >= 11.0:
                    result["status"] = "checking"
                return result

        _, problems, _, _ = collect_stationary_check(
            TerminalLossSource(clock),
            {"left": "LHR-left", "right": "LHR-right"},
            expected_stations=3,
            limits=StationaryLimits(duration_s=1.0),
            clock=clock,
            sleep=clock.sleep,
        )
        self.assertIn(
            "Tracking was not ready at the end of the stationary hold", problems
        )

    def test_slow_workspace_motion_with_three_station_coverage_passes(self) -> None:
        class MovingSource(_Source):
            def poses(self) -> dict[str, TrackerPose]:
                poses = super().poses()
                progress = self.clock() - 10.0
                for pose in poses.values():
                    pose.pos = np.array([progress * 0.2, 0.0, 0.0])
                    pose.quat = np.array(
                        [0.0, 0.0, math.sin(progress * 0.2), math.cos(progress * 0.2)]
                    )
                return poses

            def tracking_quality(self) -> dict:
                return {**super().tracking_quality(), **quality_updates}

        for quality_updates, passes in (
            ({}, True),
            (
                {
                    "status": "warning",
                    "reducedRedundancy": True,
                    "recentStationCount": 3,
                },
                True,
            ),
            (
                {
                    "status": "warning",
                    "reducedRedundancy": True,
                    "recentStationCount": 2,
                },
                False,
            ),
            (
                {
                    "status": "warning",
                    "reducedRedundancy": True,
                    "recentStationCount": 3,
                    "calibrationRequired": True,
                },
                False,
            ),
        ):
            with self.subTest(quality=quality_updates):
                clock = _Clock()
                metrics, problems = collect_workspace_check(
                    MovingSource(clock),
                    {"left": "LHR-left", "right": "LHR-right"},
                    expected_stations=3,
                    duration_s=1.0,
                    clock=clock,
                    sleep=clock.sleep,
                )
                self.assertEqual(not problems, passes, problems)
                self.assertEqual(len(metrics["left"]["stationSerials"]), 3)

    def test_changing_jump_storm_is_bounded_and_remains_failed_after_recovery(
        self,
    ) -> None:
        class FaultStormSource(_Source):
            calls = 0
            largest_jump = 0

            def tracking_quality(self) -> dict:
                result = super().tracking_quality()
                self.calls += 1
                result["faultCount"] = 17 if self.clock() > 10.01 else 0
                if self.clock() < 10.8:
                    self.largest_jump = self.calls
                    result.update(
                        status="error",
                        reasons=[
                            f"{serial}: pose discontinuity ({self.calls} mm, {self.calls / 2} degrees in 8 ms)"
                            for serial in ("LHR-left", "LHR-right")
                        ],
                    )
                return result

        for phase in ("workspace", "stationary"):
            with self.subTest(phase=phase):
                clock = _Clock()
                source = FaultStormSource(clock)
                bindings = {"left": "LHR-left", "right": "LHR-right"}
                if phase == "workspace":
                    metrics, problems = collect_workspace_check(
                        source,
                        bindings,
                        expected_stations=3,
                        duration_s=1.0,
                        clock=clock,
                        sleep=clock.sleep,
                    )
                    diagnostics = metrics["diagnostics"]
                else:
                    _, problems, _, quality = collect_stationary_check(
                        source,
                        bindings,
                        expected_stations=3,
                        limits=StationaryLimits(duration_s=1.0),
                        clock=clock,
                        sleep=clock.sleep,
                    )
                    diagnostics = quality["validationDiagnostics"]
                self.assertEqual(source.tracking_quality()["status"], "ready")
                self.assertTrue(problems)
                self.assertLess(len("; ".join(problems)), 1000)
                jumps = [
                    item
                    for item in diagnostics["groups"]
                    if item["category"] == "poseDiscontinuity"
                ]
                self.assertEqual(len(jumps), 2)
                for group in jumps:
                    self.assertGreater(group["observations"], 100)
                    self.assertEqual(group["maxPositionDeltaMm"], source.largest_jump)
                    self.assertEqual(group["maxAngleDeltaDeg"], source.largest_jump / 2)
                    self.assertEqual(len(group["examples"]), 3)
                self.assertEqual(diagnostics["nativeFaultCountDelta"], 17)
                self.assertEqual(
                    diagnostics["failureObservations"],
                    sum(item["observations"] for item in diagnostics["groups"]),
                )

    def test_unrecognized_failure_storm_has_bounded_evidence_and_exact_counts(
        self,
    ) -> None:
        class UnknownFaultSource(_Source):
            def tracking_quality(self) -> dict:
                return {
                    **super().tracking_quality(),
                    "status": "error",
                    "reasons": [
                        f"Unknown native error at {self.clock()}: " + "x" * 1000
                    ],
                }

        clock = _Clock()
        metrics, problems = collect_workspace_check(
            UnknownFaultSource(clock),
            {"left": "LHR-left", "right": "LHR-right"},
            expected_stations=3,
            duration_s=1.0,
            clock=clock,
            sleep=clock.sleep,
        )
        diagnostics = metrics["diagnostics"]
        self.assertLessEqual(len(problems), 9)
        self.assertLess(len("; ".join(problems)), 3000)
        self.assertEqual(len(diagnostics["groups"]), diagnostics["groupLimit"])
        self.assertGreater(diagnostics["omittedFailureObservations"], 100)
        self.assertEqual(
            diagnostics["failureObservations"],
            sum(item["observations"] for item in diagnostics["groups"])
            + diagnostics["omittedFailureObservations"],
        )

    def test_single_transient_failure_is_not_hidden_by_summary_or_recovery(
        self,
    ) -> None:
        class TransientFaultSource(_Source):
            calls = 0

            def tracking_quality(self) -> dict:
                result = super().tracking_quality()
                self.calls += 1
                if self.calls == 10:
                    result.update(
                        status="error",
                        reasons=[
                            "LHR-left: pose discontinuity (70 mm, 45 degrees in 5 ms)"
                        ],
                    )
                return result

        clock = _Clock()
        _, problems, _, quality = collect_stationary_check(
            TransientFaultSource(clock),
            {"left": "LHR-left", "right": "LHR-right"},
            expected_stations=3,
            limits=StationaryLimits(duration_s=1.0),
            clock=clock,
            sleep=clock.sleep,
        )
        self.assertEqual(quality["status"], "ready")
        self.assertEqual(len(problems), 1)
        self.assertIn("70 mm and 45 degrees", problems[0])
        self.assertEqual(quality["validationDiagnostics"]["failureObservations"], 1)


class ValidationProvenanceTest(unittest.TestCase):
    def test_report_requires_same_bindings_config_count_and_recent_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            native = Path(temporary) / "config.json"
            native.write_text('{"lighthouse": "original"}')
            path = Path(temporary) / "report.json"
            config = _config()
            report = new_validation_report(config, calibrated=False)
            report.update(
                status="passed",
                checkedAt=1000.0,
                expiresAt=1500.0,
                problems=[],
                fingerprint=validation_fingerprint(config, native),
            )
            save_validation_report(report, path)
            kwargs = {"config_path": native, "report_path": path, "now": 1100.0}
            self.assertTrue(require_lighthouse_validation(config, **kwargs)["valid"])
            self.assertEqual(
                validation_snapshot(config, **{**kwargs, "now": 1600.0})["status"],
                "stale",
            )
            config.expected_base_stations = 2
            with self.assertRaisesRegex(RuntimeError, "changed"):
                require_lighthouse_validation(config, **kwargs)
            config.expected_base_stations = 3
            config.left = "WM0"
            with self.assertRaisesRegex(RuntimeError, "hardware serial"):
                require_lighthouse_validation(config, **kwargs)
            config.left = "LHR-left"
            native.write_text('{"lighthouse": "moved"}')
            self.assertFalse(validation_snapshot(config, **kwargs)["valid"])

    def test_unfinished_check_overrides_previous_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            save_validation_report(
                new_validation_report(_config(), calibrated=True), path
            )
            report = validation_snapshot(_config(), report_path=path)
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["valid"])

    def test_old_solver_validation_requires_a_new_session_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            native = Path(temporary) / "config.json"
            native.write_bytes(_NATIVE_CONFIG)
            path = Path(temporary) / "report.json"
            config = _config()
            with patch(
                "almond_axol.tracker.lighthouse_validation.VALIDATION_RUNTIME_MODE",
                "gss-disabled-v1",
            ):
                report = new_validation_report(config, calibrated=False)
                report.update(
                    status="passed",
                    checkedAt=1000.0,
                    expiresAt=1500.0,
                    fingerprint=validation_fingerprint(config, native),
                )
                save_validation_report(report, path)
            result = validation_snapshot(
                config, config_path=native, report_path=path, now=1100.0
            )
            self.assertEqual(result["status"], "stale")
            self.assertFalse(result["valid"])
            self.assertIn("tracking runtime", " ".join(result["problems"]))

    def test_live_calibration_fault_invalidates_saved_check_across_restart(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            report = new_validation_report(_config(), calibrated=False)
            report.update(status="passed", valid=True)
            save_validation_report(report, path)
            invalidate_lighthouse_validation("Station geometry changed", path=path)
            after = validation_snapshot(_config(), report_path=path)
            self.assertFalse(after["valid"])
            self.assertEqual(after["status"], "failed")
            self.assertTrue(after["calibrationRequired"])
            self.assertIn("Station geometry changed", after["problems"])

    def test_lock_is_exclusive_and_releases_cleanly(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict("os.environ", {"ALMOND_HOME": temporary}),
        ):
            first = acquire_lighthouse_runtime_lock()
            try:
                with self.assertRaisesRegex(RuntimeError, "already in use"):
                    acquire_lighthouse_runtime_lock()
            finally:
                release_lighthouse_runtime_lock(first)
            next_owner = acquire_lighthouse_runtime_lock()
            release_lighthouse_runtime_lock(next_owner)


class CalibrationTransactionTest(unittest.TestCase):
    def _run(
        self,
        root: Path,
        *,
        problem: str | None = None,
        workspace_problem: str | None = None,
        final_problem: str | None = None,
        cancel: bool = False,
        stop_failure: bool = False,
        unbound: bool = False,
        discovery_problem: str | None = None,
        acquisition_problem: str | None = None,
        diagnostics_unavailable: bool = False,
        calibration_required: bool = False,
        recalibrate: bool = True,
        original: bytes = _NATIVE_CONFIG,
        updated: bytes = _UPDATED_NATIVE_CONFIG,
        expected_error: str | None = None,
    ) -> tuple[Path, Mock]:
        native = root / "native" / "config.json"
        native.parent.mkdir()
        native.write_bytes(original)
        source = Mock()
        source.lighthouse_survey.return_value = LighthouseSurvey(
            channels={0: {"aaaa"}, 2: set(), 4: set()},
            saved={0: {"aaaa"}, 2: {"bbbb"}, 4: {"cccc"}},
            expected=3,
        )
        source.tracking_quality.return_value = _Source(_Clock()).tracking_quality()
        source.warnings.return_value = ["Native diagnostic warning"]
        source.identity_diagnostics.return_value = {
            "elapsedSeconds": 120.0,
            "packetCount": 2,
            "acceptedPacketCount": 1,
            "packets": [
                {"name": "LH00_0000aaaa.ootx", "size": 43, "status": "accepted"},
                {
                    "name": "LH02_0000bbbb.ootx",
                    "size": 12,
                    "status": "pending",
                    "reason": "incomplete packet",
                },
            ],
        }
        if diagnostics_unavailable:
            source.lighthouse_survey.side_effect = RuntimeError("station read failed")
            source.tracking_quality.side_effect = RuntimeError("quality read failed")
            source.warnings.side_effect = RuntimeError("warning read failed")
            source.identity_diagnostics.side_effect = RuntimeError("packet read failed")

        created_sources: list[Mock] = []

        def make_source(**kwargs):
            nonlocal source
            candidate = kwargs["config_path"]
            self.assertNotEqual(candidate, native)
            self.assertEqual(
                kwargs["force_calibrate"], recalibrate and not created_sources
            )
            self.assertEqual(kwargs["expected_stations"], 3)
            if created_sources:
                self.assertTrue(recalibrate)
                source.stop.assert_called_once()
                self.assertEqual(candidate.read_bytes(), updated)
                preparation_source = source
                source = Mock()
                for method in (
                    "lighthouse_survey",
                    "tracking_quality",
                    "warnings",
                    "identity_diagnostics",
                ):
                    getattr(source, method).return_value = getattr(
                        preparation_source, method
                    ).return_value
                    getattr(source, method).side_effect = getattr(
                        preparation_source, method
                    ).side_effect
            else:
                self.assertEqual(candidate.read_bytes(), original)
                candidate.write_bytes(updated)
            created_sources.append(source)
            return source

        config = _config()
        if unbound:
            config.left, config.right = "WM0", "WM1"
        discovered = {"tracker1": "LHR-left", "tracker2": "LHR-right"}
        expected_bindings = (
            discovered if unbound else {"left": "LHR-left", "right": "LHR-right"}
        )
        metrics = {
            next(iter(expected_bindings)): {"positionP95Mm": 0.1, "angleP95Deg": 0.1}
        }
        measurement_count = 0

        def measure(_source, bindings, **_kwargs):
            nonlocal measurement_count
            self.assertEqual(bindings, expected_bindings)
            if recalibrate:
                prepare.assert_called_once()
                self.assertEqual(len(created_sources), 2)
                created_sources[0].stop.assert_called_once()
            else:
                prepare.assert_not_called()
                self.assertEqual(len(created_sources), 1)
            self.assertIs(_source, created_sources[-1])
            measurement_count += 1
            phase_problem = problem if measurement_count == 1 else final_problem
            return (
                metrics,
                [phase_problem] if phase_problem else [],
                _survey(),
                {
                    "trackers": {},
                    "validationDiagnostics": {"measurement": measurement_count},
                },
            )

        with ExitStack() as stack:
            stack.enter_context(patch.dict("os.environ", {"ALMOND_HOME": str(root)}))
            stack.enter_context(
                patch.object(command, "load_tracker_config", return_value=config)
            )
            stack.enter_context(
                patch.object(
                    command,
                    "validation_snapshot",
                    return_value={"calibrationRequired": calibration_required},
                )
            )
            stack.enter_context(
                patch(
                    "almond_axol.tracker.survive.default_survive_config_path",
                    return_value=native,
                )
            )
            stack.enter_context(
                patch(
                    "almond_axol.tracker.survive.SurviveSource", side_effect=make_source
                )
            )
            stack.enter_context(
                patch("almond_axol.cli.mantis_bridge.require_mantis_tracker_readiness")
            )
            prompt = stack.enter_context(patch.object(command, "_prompt"))
            prepare = stack.enter_context(patch.object(command, "_prepare_calibration"))
            stack.enter_context(
                patch.object(
                    command,
                    "_discover_calibration_trackers",
                    return_value=discovered,
                    side_effect=RuntimeError(discovery_problem)
                    if discovery_problem
                    else None,
                )
            )
            stack.enter_context(
                patch.object(
                    command,
                    "_wait_for_ready",
                    side_effect=KeyboardInterrupt
                    if cancel
                    else RuntimeError(acquisition_problem)
                    if acquisition_problem
                    else None,
                )
            )
            stack.enter_context(
                patch.object(command, "collect_stationary_check", side_effect=measure)
            )
            stack.enter_context(
                patch.object(
                    command,
                    "collect_workspace_check",
                    return_value=(
                        {
                            "diagnostics": {
                                "failureObservations": 500 if workspace_problem else 0
                            }
                        },
                        [workspace_problem] if workspace_problem else [],
                    ),
                )
            )
            stack.enter_context(patch.object(command, "save_lighthouse_survey"))
            if stop_failure:
                source.stop.side_effect = RuntimeError("native cleanup incomplete")
            args = argparse.Namespace(
                expected_stations=None, recalibrate=recalibrate, web_prompts=True
            )
            if expected_error:
                with self.assertRaisesRegex(SystemExit, expected_error):
                    command.run(args)
            elif (
                problem
                or workspace_problem
                or final_problem
                or cancel
                or stop_failure
                or discovery_problem
                or acquisition_problem
            ):
                with self.assertRaises(SystemExit):
                    command.run(args)
            else:
                command.run(args)
            if workspace_problem:
                self.assertEqual(measurement_count, 1)
                self.assertFalse(
                    any(
                        "final five-second hold" in call.args[0]
                        for call in prompt.call_args_list
                    )
                )
        return native, source

    def test_validate_does_not_repeat_calibration_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            native, source = self._run(Path(temporary), recalibrate=False)
            self.assertEqual(native.read_bytes(), _UPDATED_NATIVE_CONFIG)
            source.stop.assert_called_once()

    def test_failed_workspace_stops_before_final_hold_and_retains_phase_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            problem = "LHR-left: pose discontinuities (500 observations; maximum reported deltas 3236 mm and 175 degrees)"
            native, source = self._run(
                root,
                workspace_problem=problem,
                expected_error="Workspace sweep failed",
            )
            report = json.loads(
                (root / "tracker/lighthouse_validation.json").read_text()
            )
            self.assertEqual(report["failurePhase"], "workspace")
            self.assertEqual(set(report["phases"]), {"stationaryStart", "workspace"})
            self.assertEqual(report["phaseProblems"]["workspace"], [problem])
            self.assertEqual(
                report["phaseDiagnostics"]["stationaryStart"], {"measurement": 1}
            )
            self.assertEqual(
                report["phaseDiagnostics"]["workspace"], {"failureObservations": 500}
            )
            self.assertEqual(
                report["problems"],
                ["Workspace sweep failed. Calibration was not replaced.", problem],
            )
            # A recovered live snapshot cannot erase the failed phase.
            self.assertEqual(report["trackingQuality"]["status"], "ready")
            self.assertFalse(report["valid"])
            self.assertFalse(report["calibrationCommitted"])
            self.assertEqual(native.read_bytes(), _NATIVE_CONFIG)
            source.stop.assert_called_once()

    def test_failed_final_hold_preserves_both_stationary_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            problem = "left angleP95Deg 7.47 exceeds 1.00"
            native, source = self._run(
                root,
                final_problem=problem,
                expected_error="Final stationary check failed",
            )
            report = json.loads(
                (root / "tracker/lighthouse_validation.json").read_text()
            )
            self.assertEqual(report["failurePhase"], "stationaryEnd")
            self.assertEqual(report["phaseProblems"]["stationaryEnd"], [problem])
            self.assertEqual(report["phaseProblems"]["workspace"], [])
            self.assertEqual(
                report["phaseDiagnostics"]["stationaryStart"], {"measurement": 1}
            )
            self.assertEqual(
                report["phaseDiagnostics"]["stationaryEnd"], {"measurement": 2}
            )
            self.assertEqual(native.read_bytes(), _NATIVE_CONFIG)
            self.assertFalse(report["calibrationCommitted"])
            source.stop.assert_called_once()

    def test_acquisition_failure_retains_live_evidence_before_shutdown(self) -> None:
        problem = "Only 1 of 3 station identities decoded"
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("builtins.print") as output,
        ):
            root = Path(temporary)
            native, source = self._run(root, acquisition_problem=problem)
            self.assertEqual(native.read_bytes(), _NATIVE_CONFIG)
            report = json.loads(
                (root / "tracker/lighthouse_validation.json").read_text()
            )
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["problems"], [problem])
            self.assertFalse(report["calibrationCommitted"])
            self.assertEqual(
                report["baseStations"]["channels"],
                {"1": ["AAAA"], "3": [], "5": []},
            )
            self.assertEqual(
                report["baseStations"]["savedChannels"],
                {"1": ["AAAA"], "3": ["BBBB"], "5": ["CCCC"]},
            )
            self.assertEqual(report["trackingQuality"]["status"], "ready")
            self.assertEqual(set(report["trackers"]), {"LHR-left", "LHR-right"})
            self.assertEqual(report["nativeWarnings"], ["Native diagnostic warning"])
            self.assertEqual(
                report["identityDiagnostics"], source.identity_diagnostics.return_value
            )
            calls = [entry[0] for entry in source.mock_calls]
            self.assertLess(calls.index("warnings"), calls.index("stop"))
            self.assertLess(calls.index("identity_diagnostics"), calls.index("stop"))
            log = "\n".join(str(entry.args[0]) for entry in output.call_args_list)
            self.assertIn("3: identity pending", log)
            self.assertIn("Saved station identities (not live observations)", log)
            self.assertIn("Live tracking status: ready", log)
            self.assertIn("1 accepted of 2 observed packet files", log)
            self.assertIn("LH02_0000bbbb.ootx: incomplete packet", log)

    def test_unavailable_failure_diagnostics_do_not_mask_acquisition_error(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native, source = self._run(
                root,
                acquisition_problem="Original acquisition failure",
                diagnostics_unavailable=True,
                expected_error="Original acquisition failure",
            )
            report = json.loads(
                (root / "tracker/lighthouse_validation.json").read_text()
            )
            self.assertEqual(report["problems"], ["Original acquisition failure"])
            self.assertEqual(len(report["diagnosticErrors"]), 4)
            self.assertEqual(native.read_bytes(), _NATIVE_CONFIG)
            source.stop.assert_called_once()

    def test_bootstrap_calibration_preserves_unbound_sides_and_requires_validation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native, source = self._run(root, unbound=True, calibration_required=True)
            self.assertEqual(native.read_bytes(), _UPDATED_NATIVE_CONFIG)
            report_path = root / "tracker/lighthouse_validation.json"
            report = json.loads(report_path.read_text())
            self.assertEqual(report["status"], "needs_identification")
            self.assertFalse(report["valid"])
            self.assertFalse(report["calibrationRequired"])
            self.assertEqual(report["bindings"], {"left": "WM0", "right": "WM1"})
            self.assertEqual(
                report["discoveredTrackers"],
                {"tracker1": "LHR-left", "tracker2": "LHR-right"},
            )
            source.set_required_trackers.assert_called_once_with(
                {"LHR-left", "LHR-right"}
            )
            snapshot = validation_snapshot(
                _config(), report_path=report_path, config_path=native
            )
            self.assertEqual(snapshot["status"], "needs_identification")
            with self.assertRaisesRegex(RuntimeError, "Identify.*Validate"):
                require_lighthouse_validation(
                    _config(), report_path=report_path, config_path=native
                )

    def test_bootstrap_missing_pair_never_replaces_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native, source = self._run(
                root, unbound=True, discovery_problem="Only one tracker reporting"
            )
            self.assertEqual(native.read_bytes(), _NATIVE_CONFIG)
            source.stop.assert_called_once()

    def test_failed_recalibration_preserves_required_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._run(root, problem="tracker jumped", calibration_required=True)
            report = json.loads(
                (root / "tracker/lighthouse_validation.json").read_text()
            )
            self.assertTrue(report["calibrationRequired"])

    def test_validate_cannot_clear_persistent_calibration_fault(self) -> None:
        with (
            patch.object(command, "load_tracker_config", return_value=_config()),
            patch.object(
                command,
                "validation_snapshot",
                return_value={"calibrationRequired": True},
            ),
            patch.object(command, "save_validation_report") as save_report,
            patch("almond_axol.cli.mantis_bridge.require_mantis_tracker_readiness"),
            patch("almond_axol.tracker.survive.SurviveSource") as source,
            self.assertRaisesRegex(SystemExit, "requires Recalibrate"),
        ):
            command.run(
                argparse.Namespace(
                    expected_stations=None, recalibrate=False, web_prompts=True
                )
            )
        source.assert_not_called()
        self.assertTrue(save_report.call_args.args[0]["calibrationRequired"])

    def test_validate_still_requires_serial_bindings(self) -> None:
        config = _config()
        config.left = "WM0"
        with (
            patch.object(command, "load_tracker_config", return_value=config),
            patch.object(command, "validation_snapshot", return_value={}),
            patch.object(command, "save_validation_report"),
            patch("almond_axol.cli.mantis_bridge.require_mantis_tracker_readiness"),
            patch("almond_axol.tracker.survive.SurviveSource") as source,
            self.assertRaisesRegex(SystemExit, "hardware serials"),
        ):
            command.run(
                argparse.Namespace(
                    expected_stations=None, recalibrate=False, web_prompts=True
                )
            )
        source.assert_not_called()

    def test_success_only_publishes_after_native_stop_and_preserves_backup(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native, source = self._run(root)
            self.assertEqual(native.read_bytes(), _UPDATED_NATIVE_CONFIG)
            backups = list(native.parent.glob("*.backup"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), _NATIVE_CONFIG)
            report = json.loads(
                (root / "tracker/lighthouse_validation.json").read_text()
            )
            self.assertEqual(report["status"], "passed")
            self.assertEqual(
                set(report["phases"]), {"stationaryStart", "workspace", "stationaryEnd"}
            )
            source.stop.assert_called_once()
            source.set_required_trackers.assert_called_once_with(
                {"LHR-left", "LHR-right"}
            )

    def test_failed_cancelled_or_uncertain_cleanup_preserves_original(self) -> None:
        for case in (
            {"problem": "tracker jumped"},
            {"cancel": True},
            {"stop_failure": True},
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                native, source = self._run(root, **case)
                self.assertEqual(native.read_bytes(), _NATIVE_CONFIG)
                report = json.loads(
                    (root / "tracker/lighthouse_validation.json").read_text()
                )
                self.assertEqual(report["status"], "failed")
                source.stop.assert_called_once()

    def test_malformed_original_fails_before_start_with_path_and_keeps_original(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            original = b'"v":"0"\n"lighthouse0":{"pose":["0"'
            root = Path(temporary)
            native, source = self._run(
                root,
                original=original,
                expected_error="Cannot read native Lighthouse configuration .*config.json",
            )
            self.assertEqual(native.read_bytes(), original)
            source.start.assert_not_called()
            report = json.loads(
                (root / "tracker/lighthouse_validation.json").read_text()
            )
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["calibrationCommitted"])

    def test_malformed_native_output_cannot_replace_original_after_checks_pass(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native, source = self._run(
                root,
                updated=_UPDATED_NATIVE_CONFIG.rstrip()[:-1],
                expected_error="Cannot read native Lighthouse configuration .*config.json",
            )
            self.assertEqual(native.read_bytes(), _NATIVE_CONFIG)
            self.assertEqual(
                next(native.parent.glob("*.backup")).read_bytes(), _NATIVE_CONFIG
            )
            source.stop.assert_called_once()
            report = json.loads(
                (root / "tracker/lighthouse_validation.json").read_text()
            )
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["calibrationCommitted"])

    def test_guided_commands_are_launchable_with_typed_args(self) -> None:
        for name in ("tracker.lighthouse.validate", "tracker.lighthouse.calibrate"):
            self.assertIn(name, COMMANDS)
            schema = get_schema(name)
            self.assertIn("expected_stations", schema.emit)
            self.assertIn("web_prompts", schema.emit)


class CalibrationPreparationTest(unittest.TestCase):
    def test_preparation_waits_at_each_prompt_and_leaves_acceptance_to_checks(
        self,
    ) -> None:
        clock = _Clock()
        source = Mock()
        source.tracking_quality.return_value = {"status": "failed", "faultCount": 4}
        report = {"valid": False}
        times = []
        with (
            patch.object(command.time, "perf_counter", side_effect=clock),
            patch.object(command.time, "sleep", side_effect=clock.sleep),
            patch.object(
                command, "_prompt", side_effect=lambda *_: times.append(clock())
            ),
            patch("builtins.print"),
        ):
            command._prepare_calibration(source, report, web_prompts=True)
        preparation = report["calibrationPreparation"]
        self.assertTrue(preparation["completed"])
        self.assertEqual(len(preparation["holds"]), 3)
        self.assertEqual(times, [10.0, 18.0, 26.0])
        self.assertEqual(clock(), 34.0)
        for hold in preparation["holds"]:
            self.assertEqual(hold["durationSeconds"], 8.0)
            self.assertEqual(hold["trackingQuality"]["status"], "failed")
        self.assertFalse(report["valid"])
        self.assertGreater(source.poses.call_count, 3)

    def test_preparation_does_not_swallow_native_crash(self) -> None:
        clock = _Clock()
        source = Mock()
        source.poses.side_effect = RuntimeError("Native tracker exited")
        report = {"valid": False}
        with (
            patch.object(command.time, "perf_counter", side_effect=clock),
            patch.object(command.time, "sleep", side_effect=clock.sleep),
            patch.object(command, "_prompt"),
            patch("builtins.print"),
            self.assertRaisesRegex(RuntimeError, "Native tracker exited"),
        ):
            command._prepare_calibration(source, report, web_prompts=True)
        self.assertFalse(report["calibrationPreparation"]["completed"])
        self.assertFalse(report["valid"])


class CalibrationDiscoveryTest(unittest.TestCase):
    def test_discovers_physical_pair_before_tracking_is_ready(self) -> None:
        clock = _Clock()
        source = _Source(clock)
        original = source.poses

        def initializing():
            poses = original()
            for pose in poses.values():
                pose.tracking = False
            return poses

        with (
            patch.object(source, "poses", side_effect=initializing),
            patch.object(command.time, "perf_counter", side_effect=clock),
            patch.object(command.time, "sleep", side_effect=clock.sleep),
        ):
            self.assertEqual(
                command._discover_calibration_trackers(source),
                {"tracker1": "LHR-left", "tracker2": "LHR-right"},
            )

    def test_missing_or_stale_pair_cannot_satisfy_discovery(self) -> None:
        for mode in ("missing", "stale", "aliases"):
            with self.subTest(mode=mode):
                clock = _Clock()
                source = _Source(clock, held=mode == "stale")
                original = source.poses

                def incomplete():
                    poses = original()
                    if mode == "missing":
                        poses.pop("LHR-right")
                    elif mode == "aliases":
                        poses = {
                            f"WM{i}": pose for i, pose in enumerate(poses.values())
                        }
                    return poses

                with (
                    patch.object(source, "poses", side_effect=incomplete),
                    patch.object(command.time, "perf_counter", side_effect=clock),
                    patch.object(command.time, "sleep", side_effect=clock.sleep),
                    self.assertRaisesRegex(
                        RuntimeError, "two trackers supplied fresh poses"
                    ),
                ):
                    command._discover_calibration_trackers(source, timeout_s=1.5)

    def test_extra_connected_tracker_rejected_even_without_pose(self) -> None:
        clock = _Clock()
        source = _Source(clock)
        source.discovered_tracker_serials = lambda: {
            "LHR-left",
            "LHR-right",
            "LHR-extra",
        }
        with (
            patch.object(command.time, "perf_counter", side_effect=clock),
            self.assertRaisesRegex(RuntimeError, "More than two trackers"),
        ):
            command._discover_calibration_trackers(source)


if __name__ == "__main__":
    unittest.main()
