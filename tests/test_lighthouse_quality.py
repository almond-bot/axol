from __future__ import annotations

import json
import math
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from almond_axol.mantis.quality import EpisodeTrackingQuality
from almond_axol.tracker.bridge import StopEventControls, TrackerBridge
from almond_axol.tracker.quality import LighthouseQuality
from almond_axol.tracker.survive import (
    SurviveSource,
    _OutputClockMapper,
    default_survive_config_path,
)


class LighthouseQualityTest(unittest.TestCase):
    def setUp(self) -> None:
        invalidation = patch(
            "almond_axol.tracker.lighthouse_validation.invalidate_lighthouse_validation"
        )
        self.invalidate = invalidation.start()
        self.addCleanup(invalidation.stop)
        self.source = SurviveSource(expected_stations=3)
        self.clock = _OutputClockMapper()
        self.now = 0.0
        for alias, serial in (("T20", "LHR-LEFT"), ("T21", "LHR-RIGHT")):
            self.feed(f'0 {alias} CONFIG {{"device_serial_number": "{serial}"}}')
        for channel in range(3):
            self.feed(f"0 {channel} LH_POSE {channel} 0 2 1 0 0 0 {channel + 100}")

    def feed(self, line: str) -> None:
        self.source._consume_record(line, self.clock, 100 + self.now)

    def quality(self) -> dict:
        with patch(
            "almond_axol.tracker.survive.time.perf_counter", return_value=100 + self.now
        ):
            return self.source.tracking_quality()

    def samples(
        self,
        now: float,
        *,
        x: float = 0,
        optical: bool = True,
        channels: tuple[int, ...] = (0, 1, 2),
        quat: str = "1 0 0 0",
    ) -> dict:
        self.now = now
        for alias in ("T20", "T21"):
            if optical:
                for channel in channels:
                    for sensor in range(5):
                        tick = int((now + 0.001) * 48_000_000) & 0xFFFFFFFF
                        self.feed(f"{now} {alias} B {channel} {sensor} {tick} 0 .3")
            self.feed(f"{now} {alias} POSE {x} 0 0 {quat}")
        return self.quality()

    def ready(self) -> dict:
        for n in range(71):
            result = self.samples(n * 0.02)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["faultCount"], 0)
        return result

    def test_startup_waits_for_optical_support_and_counts_no_faults(self) -> None:
        result = self.samples(0.0, optical=False)
        self.assertEqual(result["status"], "checking")
        self.assertEqual(result["faultCount"], 0)
        with patch("almond_axol.tracker.survive.time.perf_counter", return_value=100):
            self.assertFalse(self.source.poses()["LHR-LEFT"].tracking)

    def test_three_stations_and_hardware_identities_are_reported(self) -> None:
        result = self.ready()
        self.assertEqual(set(result["trackers"]), {"LHR-LEFT", "LHR-RIGHT"})
        self.assertEqual(set(result["stations"]), {"1", "2", "3"})
        self.assertEqual(
            result["trackers"]["LHR-LEFT"]["stations"]["1"]["sensorCount"], 5
        )
        self.assertFalse(result["accuracyVerified"])
        json.dumps(result, allow_nan=False)

    def test_duplicate_station_identity_cannot_satisfy_expected_count(self) -> None:
        # Different channels are not proof of different physical stations.
        self.source._quality.stations[2]["serial"] = "00000065"
        for n in range(71):
            result = self.samples(n * 0.02)
        self.assertEqual(result["status"], "checking")
        self.assertIn("Only 2 of 3", " ".join(result["reasons"]))

    def test_same_serial_can_reconnect_under_another_alias(self) -> None:
        self.ready()
        self.now = 1.41
        self.feed("1.41 T20 DISCONNECT")
        self.feed('1.41 WM0 CONFIG {"device_serial_number":"LHR-LEFT"}')
        self.feed("1.42 WM0 POSE 0 0 0 1 0 0 0")
        self.assertEqual(self.source._serials["WM0"], "LHR-LEFT")
        self.assertNotIn("T20", self.source._serials)
        self.assertNotIn("device-identity", self.source._quality._events)
        self.assertGreater(self.quality()["faultCount"], 0)

    def test_raw_jump_invalidates_until_continuous_recovery(self) -> None:
        self.ready()
        result = self.samples(1.42, x=0.8)
        self.assertEqual(result["status"], "error")
        faults = result["faultCount"]
        self.assertGreater(faults, 0)
        with patch(
            "almond_axol.tracker.survive.time.perf_counter", return_value=101.42
        ):
            self.assertFalse(self.source.poses()["LHR-LEFT"].tracking)
        self.assertNotEqual(self.samples(1.44, x=0.8)["status"], "ready")
        for n in range(73, 130):
            result = self.samples(n * 0.02, x=0.8)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["faultCount"], faults)
        self.assertIn("discontinuity", result["lastFault"])

    def test_legitimate_fast_motion_and_quaternion_sign_do_not_fault(self) -> None:
        self.ready()
        for n in range(1, 31):
            result = self.samples(
                1.4 + n * 0.02, x=n * 0.1, quat="-1 0 0 0" if n % 2 else "1 0 0 0"
            )
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["faultCount"], 0)

    def test_rotation_jump_is_detected_before_smoothing(self) -> None:
        self.ready()
        result = self.samples(1.42, quat="0 0 0 1")
        self.assertEqual(result["status"], "error")
        self.assertGreater(result["faultCount"], 0)

    def test_fresh_pose_does_not_hide_optical_outage(self) -> None:
        self.ready()
        for n in range(71, 84):
            result = self.samples(n * 0.02, optical=False)
        self.assertEqual(result["status"], "error")
        self.assertGreater(result["trackers"]["LHR-LEFT"]["opticalAgeMs"], 150)
        faults = result["faultCount"]
        for n in range(84, 120):
            result = self.samples(n * 0.02, optical=False)
        self.assertEqual(result["faultCount"], faults)

    def test_missing_one_station_warns_without_rejecting_supported_motion(self) -> None:
        self.ready()
        for n in range(71, 190):
            result = self.samples(n * 0.02, channels=(0, 1))
        self.assertEqual(result["status"], "warning")
        self.assertEqual(result["faultCount"], 0)
        with patch(
            "almond_axol.tracker.survive.time.perf_counter", return_value=100 + self.now
        ):
            self.assertTrue(self.source.poses()["LHR-LEFT"].tracking)

    def test_per_tracker_single_station_support_warns_without_a_fault(self) -> None:
        self.ready()
        for n in range(71, 100):
            result = self.samples(n * 0.02, channels=(0,))
        self.assertEqual(result["observedBaseStations"], 3)
        self.assertEqual(result["status"], "warning")
        self.assertTrue(result["reducedRedundancy"])
        self.assertEqual(result["faultCount"], 0)
        self.assertEqual(result["trackers"]["LHR-LEFT"]["supportedStationCount"], 1)

    def test_channel_clash_stays_unsafe(self) -> None:
        self.ready()
        with patch("almond_axol.tracker.survive.time.perf_counter", return_value=101.4):
            self.source._note_warning(
                "Two or more lighthouses are on channel 0; tracking will fail"
            )
        for n in range(71, 150):
            result = self.samples(n * 0.02)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["faultCount"], 1)
        self.invalidate.assert_not_called()

    def test_native_calibration_invalidation_latches_until_recalibrated(self) -> None:
        self.ready()
        warning = (
            "Lighthouse calibration invalidated for station dd9e8fae; "
            "recalibrate before tracking"
        )
        with patch("time.perf_counter", return_value=101.4):
            self.source._note_warning(warning)
        for n in range(71, 300):
            quality = self.samples(n * 0.02)
        self.assertEqual(quality["status"], "error")
        self.assertTrue(quality["calibrationRequired"])
        self.assertEqual(quality["faultCount"], 1)
        self.assertEqual(quality["lastFault"], warning)
        self.assertIn(warning, quality["reasons"])
        self.invalidate.assert_called_once_with(warning)
        self.source._note_warning(warning)
        self.invalidate.assert_called_once_with(warning)

    def test_startup_native_calibration_invalidation_preserves_its_reason(self) -> None:
        warning = (
            "Lighthouse calibration invalidated for station DD9E8FAE; "
            "recalibrate before tracking"
        )
        self.source._note_warning(warning)
        quality = self.source.tracking_quality()
        self.assertEqual(quality["status"], "error")
        self.assertTrue(quality["calibrationRequired"])
        self.assertEqual(quality["lastFault"], warning)
        self.invalidate.assert_called_once_with(warning)

    def test_native_calibration_invalidation_persistence_failure_propagates(
        self,
    ) -> None:
        self.invalidate.side_effect = OSError("validation file is unwritable")
        with self.assertRaisesRegex(OSError, "unwritable"):
            self.source._note_warning(
                "Lighthouse calibration invalidated for station dd9e8fae; "
                "recalibrate before tracking"
            )
        self.assertFalse(self.source._validation_invalidated)
        self.assertTrue(self.source.tracking_quality()["calibrationRequired"])

    def test_rejected_background_refinement_does_not_interrupt_a_healthy_take(
        self,
    ) -> None:
        self.ready()
        bridge = TrackerBridge(
            self.source,
            left="LHR-LEFT",
            right="LHR-RIGHT",
            controls=StopEventControls(threading.Event()),
            pose_source_id="scene-refinement-regression",
        )
        monitor = EpisodeTrackingQuality(required=True)
        # Actual native global-scene rejection from the failed collection. The
        # per-tracker pose and optical observations remain unchanged and fresh.
        warning = (
            "MPFIT status failure 1457304.714584/62806.362928/0.000278 "
            "(1995 measurements, 5, MP_MAXITER)"
        )
        with patch("time.perf_counter", return_value=101.4):
            monitor.begin(bridge.compose_frame()["tracking_quality"])
            with self.assertLogs("almond_axol.tracker.survive", level="WARNING"):
                self.source._note_warning(warning)
            frame = bridge.compose_frame()
            self.assertTrue(frame["l_tracked"])
            self.assertTrue(frame["r_tracked"])
            self.assertEqual(frame["tracking_quality"]["status"], "ready")
            self.assertEqual(frame["tracking_quality"]["faultCount"], 0)
            self.assertFalse(
                monitor.update(frame["tracking_quality"], recording=True)[
                    "retakeRequired"
                ]
            )
        self.assertIn(warning, self.source.warnings())
        self.invalidate.assert_not_called()

        # Rejecting a background proposal must not disable the actual pose
        # guards: a subsequent discontinuity still invalidates this take.
        self.samples(1.42, x=0.3)
        with patch("time.perf_counter", return_value=101.42):
            frame = bridge.compose_frame()
            self.assertFalse(frame["l_tracked"])
            self.assertTrue(
                monitor.update(frame["tracking_quality"], recording=True)[
                    "retakeRequired"
                ]
            )

    def test_live_tracker_solver_failure_still_interrupts_the_take(self) -> None:
        self.ready()
        bridge = TrackerBridge(
            self.source,
            left="LHR-LEFT",
            right="LHR-RIGHT",
            controls=StopEventControls(threading.Event()),
            pose_source_id="tracker-solver-regression",
        )
        monitor = EpisodeTrackingQuality(required=True)
        warning = "MPFIT status failure T20 100.000000/25.000000 (25 measurements, -1)"
        with patch("time.perf_counter", return_value=101.4):
            monitor.begin(bridge.compose_frame()["tracking_quality"])
            self.source._note_warning(warning)
            frame = bridge.compose_frame()
            self.assertFalse(frame["l_tracked"])
            self.assertFalse(frame["r_tracked"])
            verdict = monitor.update(frame["tracking_quality"], recording=True)
            self.assertTrue(verdict["retakeRequired"])
            self.assertIn(warning, verdict["reasons"])

    def test_other_global_solver_errors_are_not_treated_as_rejected_refinement(
        self,
    ) -> None:
        self.ready()
        warning = (
            "MPFIT status failure 100.000000/25.000000/0.000278 "
            "(1995 measurements, -1, MP_ERR_INPUT)"
        )
        with patch("time.perf_counter", return_value=101.4):
            self.source._note_warning(warning)
            quality = self.source.tracking_quality()
            self.assertEqual(quality["status"], "error")
            self.assertIn(warning, quality["reasons"])

    def test_frame_fault_persistence_failure_is_not_silently_ignored(self) -> None:
        self.ready()
        self.now = 1.41
        self.invalidate.side_effect = OSError("validation file is unwritable")
        with self.assertRaisesRegex(OSError, "unwritable"):
            self.feed("1.41 0 LH_POSE .04 0 2 1 0 0 0 100")
        self.assertFalse(self.source._validation_invalidated)
        self.assertTrue(self.quality()["calibrationRequired"])

    def test_world_estimate_shift_records_fault_and_preserves_transform(self) -> None:
        self.ready()
        self.now = 1.41
        self.feed("1.41 0 LH_POSE .04 0 2 1 0 0 0 100")
        result = self.quality()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["faultCount"], 1)

        self.assertEqual(result["stations"]["1"]["position"], [0.04, 0.0, 2.0])
        self.assertTrue(result["calibrationRequired"])
        self.invalidate.assert_called_once()
        for n in range(71, 150):
            result = self.samples(n * 0.02)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["faultCount"], 1)

    def test_setup_frame_refinement_recovers_after_five_stable_seconds(self) -> None:
        self.source = SurviveSource(expected_stations=3, allow_frame_refinement=True)
        for alias, serial in (("T20", "LHR-LEFT"), ("T21", "LHR-RIGHT")):
            self.feed(f'0 {alias} CONFIG {{"device_serial_number": "{serial}"}}')
        for channel in range(3):
            self.feed(f"0 {channel} LH_POSE {channel} 0 2 1 0 0 0 {channel + 100}")
        for n in range(271):
            result = self.samples(n * 0.02)
        self.assertEqual(result["status"], "ready")

        self.now = 5.41
        self.feed("5.41 0 LH_POSE 2.5 0 2 0 0 0 1 100")
        result = self.quality()
        self.assertEqual(result["status"], "error")
        self.assertGreater(result["faultCount"], 0)
        self.assertFalse(result["calibrationRequired"])
        self.invalidate.assert_not_called()
        for n in range(271, 500):
            result = self.samples(n * 0.02)
        self.assertNotEqual(result["status"], "ready")
        for n in range(500, 540):
            result = self.samples(n * 0.02)
        self.assertEqual(result["status"], "ready")
        self.assertFalse(result["calibrationRequired"])

    def test_small_world_updates_accumulate_against_stable_baseline(self) -> None:
        self.feed("0 EXTERNAL_TO_WORLD 0 0 0 1 0 0 0")
        self.ready()
        for n in range(1, 4):
            self.now = 1.4 + n * 0.001
            self.feed(f"{self.now} EXTERNAL_TO_WORLD 0 0 {n * 0.009} 1 0 0 0")
        result = self.quality()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["faultCount"], 1)

    def test_live_serial_conflict_is_detected_without_lighthouse_pose(self) -> None:
        self.ready()
        with patch("almond_axol.tracker.survive.time.perf_counter", return_value=101.4):
            self.source._note_info("Got OOTX packet 0 aaaaaaaa")
            self.source._note_info("Got OOTX packet 0 bbbbbbbb")
        self.assertEqual(self.quality()["status"], "error")
        self.assertEqual(self.quality()["faultCount"], 1)

    def test_replayed_optical_ticks_do_not_refresh_visibility(self) -> None:
        self.ready()
        self.now = 1.7
        tick = int(1.401 * 48_000_000)
        for channel in range(3):
            for sensor in range(5):
                self.feed(f"1.7 T20 B {channel} {sensor} {tick} 0 .3")
        result = self.quality()
        self.assertGreater(result["trackers"]["LHR-LEFT"]["opticalAgeMs"], 290)

    def test_tick_wrap_is_accepted(self) -> None:
        quality = LighthouseQuality(1)
        quality.optical("serial", 0, 0, 0, 2**32 - 10, 1.0)
        quality.optical("serial", 0, 0, 0, 10, 1.02)
        self.assertEqual(quality.trackers["serial"].optical[0][0], 1.02)

    def test_unrelated_powered_tracker_does_not_block_bound_rigs(self) -> None:
        self.source.set_required_trackers({"LHR-LEFT", "LHR-RIGHT"})
        self.feed('0 WM0 CONFIG {"device_serial_number":"LHR-SPARE"}')
        self.feed("0 WM0 POSE 0 0 0 1 0 0 0")
        self.ready()
        self.now = 1.42
        self.feed("1.42 WM0 POSE 10 0 0 1 0 0 0")
        self.assertEqual(self.samples(1.42)["status"], "ready")

    def test_invalid_pose_and_timestamp_do_not_refresh_or_pass(self) -> None:
        self.ready()
        self.now = 1.42
        self.feed("1.42 T20 POSE nan 0 0 1 0 0 0")
        self.assertEqual(self.quality()["status"], "error")
        self.assertTrue(np.all(np.isfinite(self.source._poses["LHR-LEFT"].pos)))
        self.feed("nan T20 POSE 0 0 0 1 0 0 0")
        self.assertTrue(math.isfinite(self.source._poses["LHR-LEFT"].t))


class LighthouseIdentityTest(unittest.TestCase):
    def test_enumeration_change_keeps_real_serial_identity(self) -> None:
        for alias in ("T20", "T21", "WM0"):
            source = SurviveSource()
            clock = _OutputClockMapper()
            source._consume_record(f"0 {alias} POSE 0 0 0 1 0 0 0", clock, 100)
            self.assertEqual(source.poses(), {})
            source._consume_record(
                f'0 {alias} CONFIG {{"device_serial_number":"LHR-SAME"}}', clock, 100
            )
            source._consume_record(f".1 {alias} POSE 0 0 0 1 0 0 0", clock, 100.1)
            self.assertEqual(set(source.poses()), {"LHR-SAME"})

    def test_malformed_configuration_cannot_publish_alias(self) -> None:
        source = SurviveSource()
        clock = _OutputClockMapper()
        for config in ("{}", "[]", '{"device_serial_number":0}', "not json"):
            source._consume_record(f"0 T20 CONFIG {config}", clock, 100)
        source._consume_record(".1 T20 POSE 0 0 0 1 0 0 0", clock, 100.1)
        self.assertEqual(source.poses(), {})

    def test_explicit_config_and_protocol_ignore_native_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "candidate.json"
            source = SurviveSource(force_calibrate=True, config_path=destination)
            source._cli_executable = Path("/attested/survive-cli")

            def lines():
                source._stop.set()
                return iter(())

            process = SimpleNamespace(stdout=lines(), poll=lambda: 0)
            with (
                patch.dict(
                    "os.environ",
                    {"SURVIVE_CONFIGFILE": "/wrong.json", "SURVIVE_RECORD-ANGLE": "0"},
                ),
                patch(
                    "almond_axol.tracker.survive.subprocess.Popen", return_value=process
                ) as popen,
            ):
                source._run_cli()
            args = popen.call_args.args[0]
            self.assertEqual(args[args.index("--configfile") + 1], str(destination))
            self.assertEqual(args[args.index("--record-angle") + 1], "1")
            self.assertIn("--force-calibrate", args)
            self.assertNotIn("SURVIVE_CONFIGFILE", popen.call_args.kwargs["env"])

    def test_default_configuration_respects_xdg_and_native_override(self) -> None:
        with patch.dict(
            "os.environ",
            {"XDG_CONFIG_HOME": "/config", "SURVIVE_CONFIGFILE": "room.json"},
            clear=True,
        ):
            self.assertEqual(
                default_survive_config_path(), Path("/config/libsurvive/room.json")
            )
        with patch.dict("os.environ", {"SURVIVE_CONFIGFILE": "/room.json"}, clear=True):
            self.assertEqual(default_survive_config_path(), Path("/room.json"))


if __name__ == "__main__":
    unittest.main()
