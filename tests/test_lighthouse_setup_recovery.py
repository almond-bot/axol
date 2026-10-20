from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from almond_axol.cli import (
    tracker_identify,
    tracker_lighthouse,
    tracker_lighthouse_calibrate,
)
from almond_axol.tracker import create_source
from almond_axol.tracker.base import TrackerPose, TrackerSourceError
from almond_axol.tracker.config import TrackerConfig
from almond_axol.tracker.lighthouse_survey import LighthouseSurvey
from almond_axol.tracker.survive import (
    SurviveSource,
    _OutputClockMapper,
    _serialized_ootx_identity,
)


def _ootx_packet(directory: Path, channel: int, serial: str) -> Path:
    packet = bytearray(43)
    packet[2:6] = int(serial, 16).to_bytes(4, "little")
    packet[31] = channel
    path = directory / f"LH{channel:02d}_{serial}.ootx"
    path.write_bytes(packet)
    return path


def _cached_stations(source: SurviveSource, channels: tuple[int, ...]) -> None:
    clock = _OutputClockMapper()
    for channel in channels:
        source._consume_record(
            f"0 {channel} LH_POSE {channel} 0 0 1 0 0 0 {channel + 1}", clock, 100
        )


def test_fresh_unchanged_ootx_satisfies_actual_acquisition_gate(tmp_path: Path) -> None:
    """Saved geometry and received light alone cannot prove today's identities."""
    source = SurviveSource(expected_stations=3)
    _cached_stations(source, (0, 1, 3))
    for channel in (0, 1, 3):
        source._note_info(f"Adding lighthouse ch {channel} (idx: 0, cnt: 3)")
        # A later pose refinement still uses the cached identity until OOTX
        # completes, so it must not accidentally satisfy the identity gate.
        _cached_stations(source, (channel,))
    now = [100.0]

    def advance(delay: float) -> None:
        now[0] += delay

    with (
        patch.object(source, "tracking_quality", return_value={"status": "ready"}),
        patch(
            "almond_axol.cli.tracker_lighthouse_calibrate.time.perf_counter",
            lambda: now[0],
        ),
        patch("almond_axol.cli.tracker_lighthouse_calibrate.time.sleep", advance),
    ):
        with pytest.raises(RuntimeError, match="only 0 of 3 station identities"):
            tracker_lighthouse_calibrate._wait_for_ready(source, 3, timeout_s=0.2)
        for channel in (0, 1, 3):
            _ootx_packet(tmp_path, channel, f"{channel + 1:08x}")
        source._read_ootx_directory(tmp_path)
        # No new LH_POSE or changed-identity Info log occurs after decoding.
        tracker_lighthouse_calibrate._wait_for_ready(source, 3, timeout_s=0.2)
    assert source.lighthouse_survey().channels == {
        0: {"00000001"},
        1: {"00000002"},
        3: {"00000004"},
    }


def test_current_packet_files_do_not_count_unseen_saved_stations(
    tmp_path: Path,
) -> None:
    source = SurviveSource(expected_stations=3)
    _cached_stations(source, (0, 1, 3))
    _ootx_packet(tmp_path, 0, "00000001")
    source._read_ootx_directory(tmp_path)
    survey = source.lighthouse_survey()
    assert survey.base_station_count == 1
    assert survey.channels == {0: {"00000001"}}
    assert len(survey.saved) == 3
    assert "only 1 of 3" in "; ".join(survey.problems(require_serials=True))


def test_fresh_identity_replacement_rejects_cached_geometry(tmp_path: Path) -> None:
    source = SurviveSource(expected_stations=1)
    _cached_stations(source, (0,))
    _ootx_packet(tmp_path, 0, "12345678")
    source._read_ootx_directory(tmp_path)
    assert source.lighthouse_survey().replaced_channels() == [0]
    assert source.tracking_quality()["status"] == "error"
    assert any(
        "identity changed" in text for text in source.tracking_quality()["reasons"]
    )


def test_two_fresh_identities_on_one_channel_reject_tracking(tmp_path: Path) -> None:
    source = SurviveSource()
    _ootx_packet(tmp_path, 0, "12345678")
    _ootx_packet(tmp_path, 0, "87654321")
    source._read_ootx_directory(tmp_path)
    assert source.lighthouse_survey().clashing_channels() == [0]
    assert source.tracking_quality()["status"] == "error"


def test_serialized_packet_requires_complete_matching_payload(tmp_path: Path) -> None:
    good = _ootx_packet(tmp_path, 3, "12345678")
    payload = good.read_bytes()
    assert _serialized_ootx_identity(good) == (3, "12345678")
    for malformed in (payload[:32], payload + b"extra", bytes(43)):
        good.write_bytes(malformed)
        assert _serialized_ootx_identity(good) is None
    good.write_bytes(payload)
    alias = tmp_path / "LH04_12345678.ootx"
    alias.write_bytes(payload)
    assert _serialized_ootx_identity(alias) is None
    alias.unlink()
    alias.symlink_to(good)
    assert _serialized_ootx_identity(alias) is None
    assert _serialized_ootx_identity(_ootx_packet(tmp_path, 16, "12345678")) is None
    assert _serialized_ootx_identity(_ootx_packet(tmp_path, 3, "00000000")) is None


def test_partial_packet_is_retried_when_native_write_completes(tmp_path: Path) -> None:
    source = SurviveSource(expected_stations=1)
    packet = _ootx_packet(tmp_path, 0, "12345678")
    payload = packet.read_bytes()
    packet.write_bytes(payload[:4])
    source._read_ootx_directory(tmp_path)
    assert source.lighthouse_survey().channels == {}
    packet.write_bytes(payload)
    source._read_ootx_directory(tmp_path)
    assert source.lighthouse_survey().channels == {0: {"12345678"}}


def test_real_native_ootx_captures_match_parser_and_identity_diagnostics() -> None:
    # Captured from the installed pinned native runtime, rather than created
    # from the offsets this parser assumes. They contain factory metadata only.
    directory = Path(__file__).parent / "fixtures" / "lighthouse-ootx"
    expected = {0: "dd9e8fae", 1: "229d2ce0", 3: "c0ef90c8"}
    for channel, serial in expected.items():
        packet = directory / f"LH{channel:02d}_{serial}.ootx"
        assert _serialized_ootx_identity(packet) == (channel, serial)
    source = SurviveSource(expected_stations=3)
    source._read_ootx_directory(directory)
    assert source.lighthouse_survey().channels == {
        channel: {serial} for channel, serial in expected.items()
    }
    report = source.identity_diagnostics()
    assert report["acceptedPacketCount"] == 3
    assert {packet["channel"] for packet in report["packets"]} == {1, 2, 4}
    assert all(packet["size"] == 43 for packet in report["packets"])
    assert all(packet["status"] == "accepted" for packet in report["packets"])


def test_packet_diagnostics_retain_incomplete_and_rejected_observations(
    tmp_path: Path,
) -> None:
    source = SurviveSource()
    source._identity_started_at = 100.0
    packet = _ootx_packet(tmp_path, 0, "12345678")
    complete = packet.read_bytes()
    packet.write_bytes(complete[:4])
    with patch("almond_axol.tracker.survive.time.perf_counter", return_value=101):
        source._read_ootx_directory(tmp_path)
    first = source.identity_diagnostics()
    assert first["acceptedPacketCount"] == 0
    assert first["packets"][0] == {
        "name": packet.name,
        "size": 4,
        "status": "pending",
        "reason": "incomplete packet",
        "channel": 1,
        "serial": "12345678",
        "firstSeenSeconds": 1.0,
        "firstAcceptedSeconds": None,
        "lastSeenSeconds": 1.0,
    }
    assert source.lighthouse_survey().channels == {}
    packet.write_bytes(complete)
    with patch("almond_axol.tracker.survive.time.perf_counter", return_value=103):
        source._read_ootx_directory(tmp_path)
    decoded = source.identity_diagnostics()
    assert decoded["acceptedPacketCount"] == 1
    assert decoded["packets"][0]["firstAcceptedSeconds"] == 3.0
    # A later partial overwrite does not erase proof from a complete packet.
    packet.write_bytes(complete[:4])
    with patch("almond_axol.tracker.survive.time.perf_counter", return_value=104):
        source._read_ootx_directory(tmp_path)
    retained = source.identity_diagnostics()
    assert retained["acceptedPacketCount"] == 1
    assert retained["packets"][0]["status"] == "pending"
    assert retained["packets"][0]["firstSeenSeconds"] == 1.0
    assert retained["packets"][0]["firstAcceptedSeconds"] == 3.0
    assert retained["packets"][0]["lastSeenSeconds"] == 4.0
    assert retained["pollCount"] == 3
    # Snapshots are independent and JSON serializable, with no raw payloads.
    retained["packets"][0]["status"] = "edited by consumer"
    assert source.identity_diagnostics()["packets"][0]["status"] == "pending"
    json.dumps(source.identity_diagnostics())

    for payload, reason in (
        (complete + b"extra", "unexpected packet size"),
        (bytes(43), "filename and packet identity do not match"),
    ):
        rejected = SurviveSource()
        packet.write_bytes(payload)
        rejected._read_ootx_directory(tmp_path)
        diagnostic = rejected.identity_diagnostics()["packets"][0]
        assert diagnostic["status"] == "rejected"
        assert diagnostic["reason"] == reason
        assert diagnostic["size"] == len(payload)
        assert rejected.lighthouse_survey().channels == {}


def test_identity_diagnostics_survive_cleanup_without_file_access_and_reset_on_start(
    tmp_path: Path,
) -> None:
    source = SurviveSource()
    directory = tmp_path / "native-packets"
    directory.mkdir()
    source._ootx_directory = directory
    _ootx_packet(directory, 0, "12345678")
    source._read_ootx_directory(directory)
    before = source.identity_diagnostics()
    source.stop()
    assert not directory.exists()
    with patch.object(Path, "open", side_effect=AssertionError("unexpected I/O")):
        assert source.identity_diagnostics() == before
    reader = Mock(ident=None)
    reader.is_alive.return_value = False
    with (
        patch(
            "almond_axol.cli.tracker_install.verified_survive_cli",
            return_value=Path("/attested/survive-cli"),
        ),
        patch("almond_axol.tracker.survive.threading.Thread", return_value=reader),
        patch(
            "almond_axol.tracker.lighthouse_validation.acquire_lighthouse_runtime_lock"
        ),
        patch(
            "almond_axol.tracker.lighthouse_validation.release_lighthouse_runtime_lock"
        ),
    ):
        source.start()
        assert source.identity_diagnostics() == {
            "elapsedSeconds": None,
            "pollCount": 0,
            "packetCount": 0,
            "acceptedPacketCount": 0,
            "packets": [],
        }
        source.stop()


def test_identity_diagnostics_are_bounded_even_if_packet_files_are_removed(
    tmp_path: Path,
) -> None:
    source = SurviveSource()
    with patch("almond_axol.tracker.survive._OOTX_MAX_FILES", 2):
        for number in (1, 2):
            packet = _ootx_packet(tmp_path, number, f"{number:08x}")
            source._read_ootx_directory(tmp_path)
            packet.unlink()
        _ootx_packet(tmp_path, 3, "00000003")
        with pytest.raises(TrackerSourceError, match="Too many Lighthouse identities"):
            source._read_ootx_directory(tmp_path)
    assert source.identity_diagnostics()["packetCount"] == 2


def test_native_packet_directory_is_unique_and_protocol_is_protected() -> None:
    directories: list[Path] = []
    sources: list[SurviveSource] = []

    def launch(args: list[str], **kwargs: object) -> SimpleNamespace:
        directory = Path(kwargs["cwd"])
        directories.append(directory)
        assert directory.stat().st_mode & 0o777 == 0o700
        assert list(directory.iterdir()) == []
        assert args[args.index("--serialize-ootx") + 1] == "1"
        assert args[args.index("--configfile") + 1] == str(Path("room.json").absolute())
        assert "SURVIVE_SERIALIZE-OOTX" not in kwargs["env"]

        def records():
            _ootx_packet(directory, 0, "12345678")
            yield "0 0 LH_POSE 0 0 0 1 0 0 0 305419896\n"
            sources[-1]._stop.set()

        return SimpleNamespace(stdout=records(), poll=lambda: 0)

    with (
        patch("almond_axol.tracker.survive.subprocess.Popen", side_effect=launch),
        patch.dict("os.environ", {"SURVIVE_SERIALIZE-OOTX": "0"}),
        patch.object(
            SurviveSource,
            "_prepare_native_config",
            return_value=Path("room.json").absolute(),
        ),
    ):
        for _ in range(2):
            source = SurviveSource(config_path="room.json")
            sources.append(source)
            source._cli_executable = Path("/attested/survive-cli")
            source._run_cli()
            assert source.lighthouse_survey().channels == {0: {"12345678"}}
            assert source._ootx_directory is None
    assert directories[0] != directories[1]
    assert not any(path.exists() for path in directories)


def test_packet_directory_survives_uncertain_native_exit() -> None:
    source = SurviveSource()
    source._cli_executable = Path("/attested/survive-cli")
    process = SimpleNamespace(
        stdout=[],
        poll=Mock(return_value=None),
        terminate=Mock(),
        kill=Mock(),
        wait=Mock(side_effect=subprocess.TimeoutExpired("survive-cli", 3)),
    )
    with (
        patch("almond_axol.tracker.survive.subprocess.Popen", return_value=process),
        patch.object(
            source, "_prepare_native_config", return_value=Path("/private/config.json")
        ),
    ):
        with pytest.raises(TrackerSourceError, match="exited unexpectedly"):
            source._run_cli()
    directory = source._ootx_directory
    assert directory is not None and directory.is_dir()
    with pytest.raises(TrackerSourceError, match="teardown failed"):
        source.stop()
    assert directory.is_dir()
    process.wait.side_effect = None
    process.wait.return_value = 0
    process.poll.return_value = 0
    source.stop()
    assert not directory.exists()
    assert source._ootx_directory is None


def test_stop_retains_directory_and_reservation_for_late_child(tmp_path: Path) -> None:
    source = SurviveSource()
    directory = tmp_path / "native-packets"
    directory.mkdir()
    source._ootx_directory = directory
    source._runtime_lock = Mock()
    process = SimpleNamespace(
        poll=Mock(return_value=None), terminate=Mock(), wait=Mock()
    )

    def late_launch(*_args: object, **_kwargs: object) -> None:
        source._proc = process

    # stop() sees no process initially. While it joins the worker, Popen
    # completes and the worker observes the stop flag before reading stdout.
    source._thread = SimpleNamespace(
        ident=123, is_alive=lambda: False, join=late_launch
    )
    with patch(
        "almond_axol.tracker.lighthouse_validation.release_lighthouse_runtime_lock"
    ) as release:
        with pytest.raises(TrackerSourceError, match="still running"):
            source.stop()
        assert directory.is_dir()
        assert source._runtime_lock is not None
        release.assert_not_called()
        process.poll.return_value = 0
        source.stop()
        release.assert_called_once()
    assert not directory.exists()


def test_factory_relaxes_frame_refinement_only_for_setup() -> None:
    config = TrackerConfig(expected_base_stations=3)
    source = create_source(config)
    assert not source._quality.allow_frame_refinement
    assert source._quality.stable_window_s == 1.0
    setup = create_source(config, setup=True)
    assert setup._quality.allow_frame_refinement
    assert setup._quality.stable_window_s == 5.0
    assert not setup._force_calibrate


def test_discovered_identity_does_not_require_optical_localization() -> None:
    source = SurviveSource()
    clock = _OutputClockMapper()
    source._consume_record(
        '0 WM0 CONFIG {"device_serial_number":"LHR-LEFT"}', clock, 100
    )
    assert source.discovered_tracker_serials() == {"LHR-LEFT"}
    assert source.poses() == {}
    source._consume_record(".1 WM0 DISCONNECT", clock, 100.1)
    assert source.discovered_tracker_serials() == set()


def test_identify_explains_received_but_untrusted_poses() -> None:
    now = [100.0]
    source = SimpleNamespace(
        poses=lambda: {
            "LHR-LEFT": TrackerPose(
                pos=np.zeros(3),
                quat=np.array([1.0, 0.0, 0.0, 0.0]),
                t=now[0],
                tracking=False,
            )
        }
    )

    def advance(delay: float) -> None:
        now[0] += delay

    rejected: dict[str, str] = {}
    with (
        patch("almond_axol.cli.tracker_identify.time.perf_counter", lambda: now[0]),
        patch("almond_axol.cli.tracker_identify.time.sleep", advance),
    ):
        travelled, seen = tracker_identify._motion(source, 0.05, rejected)
    assert travelled == {}
    assert seen == {"LHR-LEFT"}
    summary = tracker_identify._motion_summary(travelled, seen, {}, rejected)
    assert "poses received but tracking checks did not pass" in summary
    assert "no fresh poses" not in summary


def test_identify_prints_quality_failure_reason() -> None:
    source = SimpleNamespace(
        tracking_quality=lambda: {
            "reasons": ["Base station estimate changed; waiting for a stable window"]
        }
    )
    out = io.StringIO()
    with patch("sys.stdout", out):
        tracker_identify._report_backend_warnings(source)
    assert "tracking check: Base station estimate changed" in out.getvalue()


def test_survey_distinguishes_detected_tracker_from_fresh_localization() -> None:
    survey = LighthouseSurvey()
    survey.trackers = {"LHR-LEFT"}
    source = SimpleNamespace(
        discovered_tracker_serials=lambda: {"LHR-LEFT", "LHR-RIGHT"},
        warnings=lambda: ["MPFIT failed"],
        tracking_quality=lambda: {"reasons": ["LHR-RIGHT: pose stream is stale"]},
    )
    details = " ".join(tracker_lighthouse._source_details(source, survey))
    assert "Both trackers were detected" in details
    assert "does not by itself mean pairing was lost" in details
    assert "Backend warning: MPFIT failed" in details
    assert "Tracking check: LHR-RIGHT: pose stream is stale" in details


def test_survey_uses_and_saves_three_station_choice() -> None:
    config = TrackerConfig(expected_base_stations=2)
    survey = LighthouseSurvey(expected=2)
    survey.note_channel(0, "aaaaaaaa")
    survey.note_channel(1, "bbbbbbbb")
    survey.trackers = {"LHR-LEFT", "LHR-RIGHT"}
    source = SimpleNamespace(start=Mock(), stop=Mock())
    out = io.StringIO()
    with (
        patch("almond_axol.cli.mantis_bridge.require_mantis_tracker_readiness"),
        patch("almond_axol.tracker.config.load_tracker_config", return_value=config),
        patch("almond_axol.tracker.config.save_tracker_config") as save_config,
        patch(
            "almond_axol.tracker.survive.SurviveSource", return_value=source
        ) as factory,
        patch(
            "almond_axol.cli.tracker_lighthouse.survey_lighthouses", return_value=survey
        ),
        patch("almond_axol.cli.tracker_lighthouse.save_lighthouse_survey"),
        patch("sys.stdout", out),
        pytest.raises(SystemExit) as exc,
    ):
        tracker_lighthouse.run_check(SimpleNamespace(expected_stations=3))
    assert exc.value.code == 1
    assert config.expected_base_stations == 3
    save_config.assert_called_once_with(config)
    factory.assert_called_once_with(expected_stations=3, allow_frame_refinement=True)
    assert "2 of 3 seen" in out.getvalue()
    source.stop.assert_called_once()
