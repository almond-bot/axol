"""Mantis preserves head stereo and wrist cameras from settings to recording."""

from copy import deepcopy
from unittest.mock import Mock, patch

import pytest

from almond_axol.cli import collect_data
from almond_axol.lerobot.robot.config_mantis import MantisRobotConfig
from almond_axol.serve.runner import OperationRunner


def camera_spec():
    return {
        "serials": {"overhead": "101", "left_arm": "102", "right_arm": "103"},
        "mantis_serials": {"overhead": "101", "left_arm": "202", "right_arm": "203"},
    }


def collection_config(spec):
    runner = OperationRunner()
    selected = runner._camera_spec_for_profile(spec, mantis=True)
    with patch("almond_axol.serve.runner.stereo_serials", return_value={101, 201}):
        args = runner._merge_camera_args(
            {"repo_id": "test/mantis", "task": "Pick", "mantis": True},
            selected,
            streams_video=True,
        )
    cfg = runner._build_config("collect-data", args)
    collect_data._apply_mantis_profile(cfg)
    collect_data._prepare_recording_cameras(cfg)
    cfg.robot_config.apply_detected_stereo({101, 201})
    return cfg


def test_legacy_mantis_mapping_leaves_head_unassigned_without_mutating_axol():
    # A wrist-only layout saved before the Head slot existed must not pick up
    # Axol's overhead camera: that would add a camera to existing datasets.
    spec = camera_spec()
    del spec["mantis_serials"]["overhead"]
    before = deepcopy(spec)
    selected = OperationRunner._camera_spec_for_profile(spec, mantis=True)
    assert selected["serials"] == {
        "overhead": "",
        "left_arm": "202",
        "right_arm": "203",
    }
    assert spec == before
    assert (
        OperationRunner._camera_spec_for_profile(spec, mantis=False)["serials"]
        == spec["serials"]
    )


@pytest.mark.parametrize("head", ["201", ""])
def test_explicit_mantis_head_overrides_legacy_assignment(head):
    spec = camera_spec()
    spec["mantis_serials"]["overhead"] = head
    selected = OperationRunner._camera_spec_for_profile(spec, mantis=True)
    assert selected["serials"]["overhead"] == head


def test_legacy_mantis_wrists_still_inherit_axol_wrists():
    spec = camera_spec()
    spec["mantis_serials"] = {}
    selected = OperationRunner._camera_spec_for_profile(spec, mantis=True)
    assert selected["serials"] == {
        "overhead": "",
        "left_arm": "102",
        "right_arm": "103",
    }


def test_head_and_wrists_reach_dataset_and_video_relay():
    cfg = collection_config(camera_spec())
    assert isinstance(cfg.robot_config, MantisRobotConfig)
    observations = cfg.robot_config.observation_cameras()
    assert set(observations) == {
        "overhead_left",
        "overhead_right",
        "left_arm",
        "right_arm",
    }
    assert observations["overhead_left"][1] == "left"
    assert observations["overhead_right"][1] == "right"
    relay = Mock(has_sources=True)
    with (
        patch("almond_axol.video.video.webrtc_available", return_value=True),
        patch(
            "almond_axol.video.video_proc.VideoRelayProcess", return_value=relay
        ) as create_relay,
    ):
        assert collect_data._start_video_relay(cfg, "SVGA") is relay
    specs = create_relay.call_args.args[0]
    assert set(specs) == {"overhead", "left_arm", "right_arm"}
    assert specs["overhead"]["serial"] == 101
    assert specs["overhead"]["record_eyes"] == ["left", "right"]
    assert specs["overhead"]["record_suffix"] is True
    assert specs["overhead"]["record"] is True
    assert specs["left_arm"]["serial"] == 202
    assert specs["right_arm"]["serial"] == 203
    assert create_relay.call_args.kwargs == {"want_raw": True}


def test_head_only_recording_does_not_require_streaming_or_wrists():
    spec = camera_spec()
    spec["mantis_serials"] = {
        "overhead": "201",
        "left_arm": "",
        "right_arm": "",
    }
    spec["stream_resolution"] = "off"
    cfg = collection_config(spec)
    assert set(cfg.robot_config.observation_cameras()) == {
        "overhead_left",
        "overhead_right",
    }
    assert cfg.robot_config.cameras["overhead"].stream is False
    assert OperationRunner._has_recording_camera(cfg)


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (False, {"left_arm", "right_arm"}),
        ("right", {"overhead", "left_arm", "right_arm"}),
    ],
)
def test_head_recording_selection_is_independent_of_streaming(record, expected):
    spec = camera_spec()
    spec["record"] = {"overhead": record}
    spec["stream"] = {"overhead": "both"}
    cfg = collection_config(spec)
    assert set(cfg.robot_config.observation_cameras()) == expected
    head = cfg.robot_config.cameras["overhead"]
    assert head.stream is True
    assert head.streaming_eyes() == "both"
    if record == "right":
        assert cfg.robot_config.observation_cameras()["overhead"][1] == "right"
