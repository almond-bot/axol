from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from almond_axol.tracker.base import TrackerPose, TrackerSourceError
from almond_axol.tracker.config import TrackerConfig
from almond_axol.tracker.survive import SurviveSource, _OutputClockMapper


def _configuration() -> dict:
    return {
        "floor-offset": "-0.25",
        **{
            f"lighthouse{index}": {
                "mode": str(channel),
                "id": str(index + 100),
                "PositionSet": "1",
                "OOTXSet": "1",
                "pose": [str(index), "0", "2", "1", "0", "0", "0"],
            }
            for index, channel in enumerate((0, 1, 3))
        },
    }


@pytest.mark.parametrize("calibrate", [False, True])
@pytest.mark.parametrize("discover", [False, True])
def test_runtime_mode_is_explicit_and_environment_cannot_override_it(
    tmp_path: Path, calibrate: bool, discover: bool
) -> None:
    saved = tmp_path / "published.json"
    saved.write_text(json.dumps(_configuration()))
    original = saved.read_bytes()
    source = SurviveSource(
        force_calibrate=calibrate, allow_frame_refinement=discover, expected_stations=3
    )
    source._cli_executable = Path("/attested/survive-cli")
    calls: list[dict] = []

    def launch(args: list[str], **kwargs):
        native = Path(args[args.index("--configfile") + 1])
        assert native != saved
        assert native.parent == kwargs["cwd"]
        assert native.read_bytes() == original
        # Native config_save must never change the published validation baseline.
        native.write_text("native bookkeeping changed during the run")
        calls.append({"args": args, **kwargs})
        source._stop.set()
        return SimpleNamespace(stdout=[], poll=lambda: 0)

    with (
        patch(
            "almond_axol.tracker.survive.default_survive_config_path",
            return_value=saved,
        ),
        patch("almond_axol.tracker.survive.subprocess.Popen", side_effect=launch),
        patch.dict(
            "os.environ",
            {
                "SURVIVE_DISABLE-CALIBRATE": "unexpected",
                "SURVIVE_GLOBALSCENESOLVER": "unexpected",
                "SURVIVE_GSS-AUTO-FLOOR-HEIGHT": "unexpected",
                "SURVIVE_KALMAN-JOINT-MODEL-LIGHTCAP": "unexpected",
                "SURVIVE_LIGHT-ERROR-FOR-LH-CONFIDENCE": "unexpected",
                "SURVIVE_MPFIT-FILTER-INDIVIDUAL-OUTLIERS": "unexpected",
                "SURVIVE_FORCE-CALIBRATE": "unexpected",
            },
        ),
    ):
        source._run_cli()
    assert saved.read_bytes() == original
    call = calls[0]
    args = call["args"]
    for flag, value in (
        ("--force-calibrate", "1" if calibrate else "0"),
        ("--disable-calibrate", "0" if calibrate else "1"),
        ("--globalscenesolver", "1" if calibrate else "0"),
        ("--gss-auto-floor-height", "1" if calibrate else "0"),
        ("--kalman-joint-model-lightcap", "-1"),
        ("--light-error-for-lh-confidence", "0"),
        ("--mpfit-filter-individual-outliers", "0" if calibrate or discover else "1"),
    ):
        assert args[args.index(flag) + 1] == value
        assert f"SURVIVE_{flag[2:].upper()}" not in call["env"]
    assert not Path(call["cwd"]).exists()


@pytest.mark.parametrize(
    "change",
    [
        lambda config: config.pop("lighthouse2"),
        lambda config: config["lighthouse2"].update(PositionSet="0"),
        lambda config: config["lighthouse2"].update(OOTXSet="0"),
        lambda config: config["lighthouse2"].update(mode="0"),
        lambda config: config["lighthouse2"].update(id="100"),
        lambda config: config["lighthouse2"].update(pose=[0] * 7),
        lambda config: config["lighthouse2"].update(pose=["nan"] * 7),
        lambda config: config.update({"floor-offset": "nan"}),
    ],
)
def test_recording_rejects_incomplete_saved_geometry_before_native_launch(
    tmp_path: Path, change
) -> None:
    config = _configuration()
    change(config)
    saved = tmp_path / "candidate.json"
    saved.write_text(json.dumps(config))
    source = SurviveSource(config_path=saved, expected_stations=3)
    source._cli_executable = Path("/attested/survive-cli")
    with patch("almond_axol.tracker.survive.subprocess.Popen") as launch:
        with pytest.raises(TrackerSourceError, match="Recalibrate Lighthouse"):
            source._run_cli()
    launch.assert_not_called()
    assert source._ootx_directory is None


def test_missing_saved_geometry_does_not_silently_bootstrap_recording(
    tmp_path: Path,
) -> None:
    source = SurviveSource(config_path=tmp_path / "missing.json", expected_stations=3)
    source._cli_executable = Path("/attested/survive-cli")
    with patch("almond_axol.tracker.survive.subprocess.Popen") as launch:
        with pytest.raises(TrackerSourceError, match="calibration is missing"):
            source._run_cli()
    launch.assert_not_called()


def test_station_discovery_can_read_light_without_saved_geometry_or_publish(
    tmp_path: Path,
) -> None:
    saved = tmp_path / "missing.json"
    private = tmp_path / "private"
    private.mkdir()
    source = SurviveSource(allow_frame_refinement=True)
    with patch(
        "almond_axol.tracker.survive.default_survive_config_path", return_value=saved
    ):
        native = source._prepare_native_config(private)
    # An empty file, never "{}": survive-cli segfaults on an empty object.
    assert native.read_text() == ""
    assert not saved.exists()
    assert source._runtime_geometry == {}


def _fixed_source(tmp_path: Path) -> SurviveSource:
    saved = tmp_path / "candidate.json"
    saved.write_text(json.dumps(_configuration()))
    source = SurviveSource(config_path=saved, expected_stations=3)
    source._prepare_native_config(tmp_path)
    return source


def test_runtime_requires_fresh_station_identities_before_usable_poses(
    tmp_path: Path,
) -> None:
    source = _fixed_source(tmp_path)
    source._poses["LHR-left"] = TrackerPose(
        np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), 100.0
    )
    clock = _OutputClockMapper()
    for index, channel in enumerate((0, 1, 3)):
        source._consume_record(
            f"0 {channel} LH_POSE {index} 0 2.25 1 0 0 0 {index + 100}", clock, 100.0
        )
    with patch.object(
        source._quality,
        "snapshot",
        side_effect=lambda *_: {"status": "ready", "reasons": []},
    ):
        assert source.tracking_quality()["status"] == "checking"
        assert not source.poses()["LHR-left"].tracking
        for index, channel in enumerate((0, 1, 3)):
            source._note_info(f"Got OOTX packet {channel} {index + 100:08x}")
            packet = bytearray(43)
            packet[2:6] = (index + 100).to_bytes(4, "little")
            packet[31] = channel
            (tmp_path / f"LH{channel:02d}_{index + 100:08x}.ootx").write_bytes(packet)
        source._read_ootx_directory(tmp_path, receipt=100.0)
        assert not source.lighthouse_survey().problems(require_serials=True)
        # Both the old log and packet file precede native metadata checks.
        # They prove reception, but must not expose poses before a later
        # invalidation warning has been consumed from the same FIFO stream.
        assert source.tracking_quality()["status"] == "checking"
        assert not source.poses()["LHR-left"].tracking
        for index, channel in enumerate((0, 1)):
            source._note_info(f"Checked OOTX packet {channel} {index + 100:08x}")
        assert source.tracking_quality()["status"] == "checking"
        source._note_info("Checked OOTX packet 3 00000066")
        assert source.tracking_quality()["status"] == "ready"
        assert source.poses()["LHR-left"].tracking


def test_checked_ootx_marker_cannot_clear_a_preceding_invalidation(
    tmp_path: Path,
) -> None:
    source = _fixed_source(tmp_path)
    warning = (
        "Lighthouse calibration invalidated for station 00000066; "
        "recalibrate before tracking"
    )
    with patch(
        "almond_axol.tracker.lighthouse_validation.invalidate_lighthouse_validation"
    ) as invalidate:
        source._note_warning(warning)
        for index, channel in enumerate((0, 1, 3)):
            source._note_info(f"Checked OOTX packet {channel} {index + 100:08x}")
        # Markers complete identity acquisition; they never clear faults.
        with patch("time.perf_counter", return_value=1e12):
            report = source.tracking_quality()
        assert report["status"] == "error"
        assert report["calibrationRequired"] is True
        assert warning in report["reasons"]
        invalidate.assert_called_once_with(warning)


def test_startup_resolve_cannot_replace_fixed_geometry_before_quality_arms(
    tmp_path: Path,
) -> None:
    source = _fixed_source(tmp_path)
    clock = _OutputClockMapper()
    assert not source._quality._armed
    with patch(
        "almond_axol.tracker.lighthouse_validation.invalidate_lighthouse_validation"
    ) as invalidate:
        with pytest.raises(TrackerSourceError, match="changed saved geometry"):
            source._consume_record("0 0 LH_POSE 0.02 0 2.25 1 0 0 0 100", clock, 100.0)
    invalidate.assert_called_once()
    assert source.tracking_quality()["status"] == "error"
    assert source.tracking_quality()["calibrationRequired"]


def test_station_pose_rounding_and_quaternion_sign_do_not_change_geometry(
    tmp_path: Path,
) -> None:
    source = _fixed_source(tmp_path)
    clock = _OutputClockMapper()
    with patch(
        "almond_axol.tracker.lighthouse_validation.invalidate_lighthouse_validation"
    ) as invalidate:
        source._consume_record(
            "0 0 LH_POSE 0.000001 0 2.250001 -1 0 0 0 100", clock, 100.0
        )
    invalidate.assert_not_called()


def test_native_failure_is_explicit_in_quality_for_operator_recovery() -> None:
    source = SurviveSource()
    source._failure = TrackerSourceError("survive-cli exited unexpectedly")
    report = source.tracking_quality()
    assert report["status"] == "error"
    assert report["nativeFailure"] is True
    assert report["reasons"][0] == "survive-cli exited unexpectedly"


@pytest.mark.parametrize("backend, timeout", [("survive", 120.0), ("ultimate", 25.0)])
def test_managed_bridge_selects_the_acquisition_budget_for_its_backend(
    backend: str, timeout: float
) -> None:
    from almond_axol.cli.tracker_bridge import run_configured_bridge

    async def run() -> None:
        return None

    config = TrackerConfig(backend=backend, left="LHR-left", right="LHR-right")
    with (
        patch("almond_axol.tracker.create_source", return_value=Mock()),
        patch("almond_axol.tracker.trigger.TriggerReader", return_value=Mock()),
        patch(
            "almond_axol.tracker.bridge.TrackerBridge",
            return_value=SimpleNamespace(run=run),
        ),
        patch("almond_axol.cli.tracker_bridge._wait_for_live_inputs") as wait,
    ):
        run_configured_bridge(config, require_live_inputs=True, pose_source_id="test")
    assert wait.call_args.kwargs["timeout_s"] == timeout
