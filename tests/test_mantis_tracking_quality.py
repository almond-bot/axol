from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from almond_axol.cli.collect_data import (
    CollectDataConfig,
    EpisodeQAStats,
    _QueueCollectControl,
    _require_mantis_tracking_validation,
    _resolve_lighthouse_quality,
    evaluate_episode_qa,
)
from almond_axol.mantis.quality import EpisodeTrackingQuality, quality_ready
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore
from almond_axol.tracker.base import TrackerPose
from almond_axol.tracker.bridge import StopEventControls, TrackerBridge
from almond_axol.tracker.config import TrackerConfig
from almond_axol.vr.interp import PoseInterpolator
from almond_axol.vr.models import VRFrame, VRPose, VRPosition, VRQuaternion


def quality(status="ready", faults=0, reason=None):
    return {
        "status": status,
        "faultCount": faults,
        "sourceId": "session-one",
        "reasons": [reason] if reason else [],
        "lastFault": reason,
    }


def test_transient_fault_is_latched_even_when_only_recovered_frame_is_observed():
    monitor = EpisodeTrackingQuality(required=True)
    monitor.begin(quality(faults=7))
    recovered = quality(faults=8, reason="Right rig lost optical tracking")
    verdict = monitor.update(recovered, recording=True)
    assert verdict["retakeRequired"]
    assert verdict["status"] == "error"
    assert monitor.update(quality(faults=8), recording=True)["retakeRequired"]
    assert not evaluate_episode_qa(
        EpisodeQAStats(total_frames=10000, tracking_quality_faults=monitor.faults)
    )[0]
    monitor.begin(quality(faults=8))
    assert not monitor.update(quality(faults=8), recording=True)["retakeRequired"]


def test_recovery_window_preserves_the_failed_takes_actual_cause():
    monitor = EpisodeTrackingQuality(required=True)
    monitor.begin(quality(faults=46))
    jump = "LHR-LEFT: pose discontinuity (18 mm, 52 degrees in 8 ms)"
    verdict = monitor.update(quality("error", 47, jump), recording=True)
    assert verdict["lastFault"] == jump
    recovering = quality("checking", 47, jump)
    recovering["reasons"] = ["Waiting for a stable tracking window"]
    verdict = monitor.update(recovering, recording=True)
    assert verdict["retakeRequired"]
    assert verdict["lastFault"] == jump
    assert monitor.faults == [jump]
    assert [event["reason"] for event in monitor.report()["events"]] == [jump]
    assert monitor.report()["lastTrackingQuality"] == recovering


def test_coverage_warning_during_recovery_is_not_an_additional_fault():
    monitor = EpisodeTrackingQuality(required=True)
    monitor.begin(quality())
    outage = (
        "LHR-LEFT: insufficient recent optical observations; clear tracker visibility"
    )
    current = quality("error", 1, outage)
    current["reasons"].append(
        "LHR-RIGHT: reduced optical redundancy; only one base station has "
        "sufficient recent sensor observations"
    )
    verdict = monitor.update(current, recording=True)
    assert verdict["retakeRequired"]
    assert monitor.faults == [outage]
    later_fault = "LHR-RIGHT: pose stream interrupted"
    verdict = monitor.update(quality("error", 2, later_fault), recording=True)
    assert verdict["lastFault"] == outage
    assert monitor.faults == [outage, later_fault]


def test_recovery_only_frame_still_rejects_take_without_prior_fault_evidence():
    monitor = EpisodeTrackingQuality(required=True)
    monitor.begin(quality())
    verdict = monitor.update(
        quality("checking", reason="Waiting for a stable tracking window"),
        recording=True,
    )
    assert verdict["retakeRequired"]
    assert monitor.faults


@pytest.mark.parametrize("current", [None, quality("checking"), quality("error")])
def test_missing_or_unhealthy_quality_cannot_start(current):
    with pytest.raises(ValueError, match="not ready"):
        EpisodeTrackingQuality(required=True).begin(current)


def test_reduced_redundancy_is_warning_and_not_automatic_retake():
    monitor = EpisodeTrackingQuality(required=True)
    q = quality("warning", reason="Only two of three stations recently seen")
    assert quality_ready(q)
    monitor.begin(q)
    assert not monitor.update(q, recording=True)["retakeRequired"]


def test_restarted_diagnostics_and_total_loss_reject_take():
    monitor = EpisodeTrackingQuality(required=True)
    monitor.begin(quality(faults=8))
    assert monitor.update(quality(faults=0), recording=True)["retakeRequired"]
    monitor.begin(quality())
    assert monitor.update(None, recording=True)["retakeRequired"]


def test_panel_blocks_save_and_preserves_previous_retake_reason():
    panel = _QueueCollectControl(threading.Event())
    panel.note_recording()
    q = quality("error", 1, "Left rig lost optical tracking")
    q["retakeRequired"] = True
    panel.note_tracking_quality(q)
    snap = panel.snapshot()
    assert [c["command"] for c in snap["controls"]] == ["r"]
    panel.note_ready(1)
    panel.note_tracking_quality({**quality(), "retakeRequired": False})
    assert panel.snapshot()["trackingQuality"]["lastEpisodeFault"] == q["lastFault"]
    assert panel.snapshot()["controls"][0]["command"] == "start"
    panel.note_recording()
    assert "lastEpisodeFault" not in panel.snapshot()["trackingQuality"]


def test_core_quality_arrives_independently_of_ik_and_expires():
    core = VRTeleopCore(VRTeleopConfig(), logging.getLogger(__name__), lambda _: None)
    with patch("almond_axol.teleop.core.time.perf_counter", return_value=10.0):
        core.note_tracking_quality(SimpleNamespace(tracking_quality=quality()))
        assert core.tracking_quality()["status"] == "ready"
    with patch("almond_axol.teleop.core.time.perf_counter", return_value=10.6):
        assert core.tracking_quality()["status"] == "error"


def test_interpolator_preserves_latest_fault_metadata_on_stationary_poses():
    interp = PoseInterpolator(min_delay_s=0, max_delay_s=0, smooth_window_s=0.04)
    pose = VRPose(
        position=VRPosition(x=0, y=1, z=0),
        quaternion=VRQuaternion(x=0, y=0, z=0, w=1),
    )
    previous = None
    for i in range(6):
        frame = VRFrame(
            l_ee=pose,
            r_ee=pose,
            l_elbow=pose.position,
            r_elbow=pose.position,
            t=i * 10.0,
            seq=i,
            tracking_quality=quality(faults=int(i >= 4)),
        )
        interp.push(frame, now=1 + i * 0.01)
        out = interp.sample(now=1 + i * 0.01)
        assert out.tracking_quality["faultCount"] == int(i >= 4)
        if i >= 4:
            assert out is previous
        previous = out


def test_gross_separation_is_rejected_before_absolute_pose_filters():
    class Source:
        separation = 0.5

        def poses(self):
            now = time.perf_counter()
            return {
                side: TrackerPose(np.array([x, 1, 0]), np.array([0, 0, 0, 1]), now)
                for side, x in (("L", 0), ("R", self.separation))
            }

        def tracking_quality(self):
            return quality()

    source = Source()
    bridge = TrackerBridge(
        source, left="L", right="R", controls=StopEventControls(threading.Event())
    )
    assert bridge.compose_frame()["l_tracked"]
    source.separation = 3.16
    frame = VRFrame.model_validate(bridge.compose_frame())
    assert not frame.l_tracked and not frame.r_tracked
    assert frame.tracking_quality["faultCount"] == 1
    assert frame.tracking_quality["status"] == "error"
    source.separation = 0.5
    recovered = bridge.compose_frame()["tracking_quality"]
    assert recovered["faultCount"] == 1
    assert "2 m apart" in recovered["lastFault"]


def test_startup_failure_explains_missing_optical_support():
    from almond_axol.cli.tracker_bridge import _wait_for_live_inputs

    source = SimpleNamespace(
        poses=lambda: {},
        tracking_quality=lambda: quality(
            "checking", reason="Only 2 of 3 expected base stations"
        ),
    )
    with pytest.raises(RuntimeError, match="Only 2 of 3"):
        _wait_for_live_inputs(source, "L", "R", {}, timeout_s=0)


def test_production_validation_gate_cannot_be_bypassed_by_ui_or_cli():
    cfg = CollectDataConfig(
        repo_id="test/quality",
        task="quality test",
        mantis=True,
        mantis_source="lighthouse",
    )
    cfg._lighthouse_quality = "strict"
    with patch(
        "almond_axol.tracker.lighthouse_validation.require_lighthouse_validation",
        side_effect=RuntimeError("session check expired"),
    ) as gate:
        with pytest.raises(RuntimeError, match="expired"):
            _require_mantis_tracking_validation(cfg)
        gate.assert_called_once()
        cfg.mantis_allow_uncalibrated = True
        _require_mantis_tracking_validation(cfg)
        gate.assert_called_once()


def test_standard_policy_does_not_require_validation():
    cfg = CollectDataConfig(
        repo_id="test/quality",
        task="quality test",
        mantis=True,
        mantis_source="lighthouse",
    )
    with (
        patch(
            "almond_axol.tracker.config.load_tracker_config",
            return_value=TrackerConfig(),
        ),
        patch(
            "almond_axol.tracker.lighthouse_validation.require_lighthouse_validation",
            side_effect=RuntimeError("session check expired"),
        ) as gate,
    ):
        assert _resolve_lighthouse_quality(cfg) == "standard"
        _require_mantis_tracking_validation(cfg)
    gate.assert_not_called()


def test_panel_runs_resolve_the_saved_strict_policy_without_the_cli_entry():
    # The control panel builds the config and calls the gates directly,
    # without main(); the strict policy must still apply.
    cfg = CollectDataConfig(
        repo_id="test/quality",
        task="quality test",
        mantis=True,
        mantis_source="lighthouse",
    )
    with (
        patch(
            "almond_axol.tracker.config.load_tracker_config",
            return_value=TrackerConfig(lighthouse_quality="strict"),
        ) as load,
        patch(
            "almond_axol.tracker.lighthouse_validation.require_lighthouse_validation",
            side_effect=RuntimeError("session check expired"),
        ),
    ):
        with pytest.raises(RuntimeError, match="expired"):
            _require_mantis_tracking_validation(cfg)
        assert cfg._lighthouse_quality == "strict"
        # Resolved once; later checks in the session reuse it.
        assert _resolve_lighthouse_quality(cfg) == "strict"
    load.assert_called_once()
