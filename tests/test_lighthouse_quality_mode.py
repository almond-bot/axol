"""The standard Lighthouse policy keeps pre-strict behaviour; strict opts in."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore
from almond_axol.tracker import create_source
from almond_axol.tracker.base import TrackerPose
from almond_axol.tracker.bridge import StopEventControls, TrackerBridge
from almond_axol.tracker.config import (
    TrackerConfig,
    load_tracker_config,
    save_tracker_config,
)
from almond_axol.tracker.survive import SurviveSource
from almond_axol.vr.models import VRFrame, VRPose, VRPosition, VRQuaternion


def test_policy_defaults_to_standard_and_rejects_unknown_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    assert load_tracker_config(path).lighthouse_quality == "standard"
    path.write_text(json.dumps({"lighthouse_quality": "paranoid"}))
    # A typo must not silently select the strict gate (or crash the host).
    assert load_tracker_config(path).lighthouse_quality == "standard"
    with pytest.raises(ValueError, match="lighthouse_quality"):
        save_tracker_config(TrackerConfig(lighthouse_quality="paranoid"), path)
    save_tracker_config(TrackerConfig(lighthouse_quality="strict"), path)
    assert load_tracker_config(path).lighthouse_quality == "strict"


@pytest.mark.parametrize("mode", ["standard", "strict"])
def test_create_source_follows_policy(mode: str) -> None:
    source = create_source(TrackerConfig(backend="survive", lighthouse_quality=mode))
    assert source._strict is (mode == "strict")


def test_explicit_setup_transactions_stay_strict(tmp_path: Path) -> None:
    # Calibrate/validate construct SurviveSource directly.
    assert SurviveSource()._strict
    assert SurviveSource(strict=False, force_calibrate=True)._strict
    assert SurviveSource(strict=False, config_path=tmp_path / "candidate.json")._strict


@pytest.mark.parametrize("discover", [False, True])
def test_standard_runtime_uses_saved_calibration_and_pose_only_stream(
    tmp_path: Path, discover: bool
) -> None:
    saved = tmp_path / "config.json"
    saved.write_text("{}")
    source = SurviveSource(allow_frame_refinement=discover, strict=False)
    source._cli_executable = Path("/attested/survive-cli")
    calls: list[list[str]] = []

    def launch(args: list[str], **kwargs):  # type: ignore[no-untyped-def]
        calls.append(args)
        source._stop.set()
        return SimpleNamespace(stdout=[], poll=lambda: 0)

    with (
        patch("almond_axol.tracker.survive.prepare_survive_config", return_value=saved),
        patch("almond_axol.tracker.survive.subprocess.Popen", side_effect=launch),
    ):
        source._run_cli()
    args = calls[0]
    # libsurvive refines and saves the published calibration itself.
    assert args[args.index("--configfile") + 1] == str(saved)
    assert args[args.index("--record-angle") + 1] == "0"
    for flag in (
        "--disable-calibrate",
        "--globalscenesolver",
        "--mpfit-filter-individual-outliers",
    ):
        assert flag not in args
    assert source._runtime_geometry == {}


def test_standard_poses_are_not_withheld_and_report_no_quality() -> None:
    source = SurviveSource(strict=False)
    pose = TrackerPose(pos=np.zeros(3), quat=np.array([1.0, 0, 0, 0]), t=1.0)
    source._poses["LHR-LEFT"] = pose
    assert source.poses()["LHR-LEFT"].tracking
    assert source.tracking_quality() is None
    strict = SurviveSource()
    strict._poses["LHR-LEFT"] = pose
    # Strict withholds poses until the quality window is ready.
    assert not strict.poses()["LHR-LEFT"].tracking
    assert strict.tracking_quality() is not None


def test_bridge_frame_omits_quality_for_a_source_without_policing() -> None:
    now = time.perf_counter()

    class Source:
        def poses(self) -> dict[str, TrackerPose]:
            return {
                key: TrackerPose(
                    pos=np.array([x, 1.0, 0.0]),
                    quat=np.array([1.0, 0.0, 0.0, 0.0]),
                    t=now,
                )
                for key, x in (("L", -0.2), ("R", 0.2))
            }

        def tracking_quality(self) -> None:
            return None

    bridge = TrackerBridge(
        Source(),
        left="L",
        right="R",
        controls=StopEventControls(threading.Event()),
    )
    frame = bridge.compose_frame()
    assert "tracking_quality" not in frame
    assert frame["l_tracked"] and frame["r_tracked"]


def test_managed_bridge_pins_the_operation_policy() -> None:
    from almond_axol.cli import mantis_bridge

    seen: list[str] = []

    def run(config, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(config.lighthouse_quality)
        kwargs["on_ready"]()
        kwargs["controls"].quit.wait(5)

    config = TrackerConfig(backend="survive", left="L", right="R")
    with (
        patch.object(mantis_bridge, "require_mantis_tracker_readiness"),
        patch("almond_axol.tracker.load_tracker_config", return_value=config),
        patch("almond_axol.cli.tracker_bridge.run_configured_bridge", run),
    ):
        with mantis_bridge.managed_mantis_bridge(
            "lighthouse",
            left_channel="can_mantis_l",
            right_channel="can_mantis_r",
            port=8000,
            pose_source_id="test",
            lighthouse_quality="strict",
        ):
            pass
    assert seen == ["strict"]


def _untracked_frame() -> VRFrame:
    stamp = time.perf_counter()
    quat = VRQuaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    return VRFrame(
        l_ee=VRPose(position=VRPosition(x=-0.2, y=1.0, z=0.0), quaternion=quat),
        r_ee=VRPose(position=VRPosition(x=0.2, y=1.0, z=0.0), quaternion=quat),
        l_elbow=VRPosition(x=0.0, y=0.0, z=0.0),
        r_elbow=VRPosition(x=0.0, y=0.0, z=0.0),
        l_tracked=False,
        t_host=stamp,
        t=stamp * 1000,
    )


def test_invalid_frames_while_disengaged_do_not_latch_a_tracking_loss() -> None:
    core = VRTeleopCore(
        VRTeleopConfig(absolute_mode=True),
        logging.getLogger("test"),
        lambda _engaged: None,
        record_raw_tcp=True,
    )
    for _ in range(100):
        core.note_raw_tcp_frame(_untracked_frame())
    # Re-latching here made the IK loop disengage and broadcast
    # tracking=false on every iteration while the rigs were set down.
    assert not core._raw_tcp_tracking_lost
    core.left_enabled = core.right_enabled = True
    core.note_raw_tcp_frame(_untracked_frame())
    assert core._raw_tcp_tracking_lost
