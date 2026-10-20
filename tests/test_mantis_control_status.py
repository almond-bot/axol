from __future__ import annotations

import threading
from unittest.mock import patch

import numpy as np

from almond_axol.tracker.base import TrackerPose
from almond_axol.tracker.bridge import StopEventControls, TrackerBridge
from almond_axol.vr.models import VRState


class _Source:
    lag = 0.0
    status = "ready"

    def poses(self):
        return {
            side: TrackerPose(
                np.array([x, 1.0, 0.0]),
                np.array([0.0, 0.0, 0.0, 1.0]),
                10.0 - (self.lag if side == "L" else 0),
                tracking=self.status == "ready",
            )
            for side, x in (("L", 0.2), ("R", -0.2))
        }

    def tracking_quality(self):
        return {
            "status": self.status,
            "faultCount": 0,
            "lastFault": None,
            "reasons": [],
        }


class _Trigger:
    value = 1.0
    stale = False

    def grip(self):
        return self.value

    def is_stale(self):
        return self.stale


def _bridge(source, **kwargs):
    return TrackerBridge(
        source,
        left="L",
        right="R",
        controls=StopEventControls(threading.Event()),
        **kwargs,
    )


def _frame(bridge):
    with patch("almond_axol.tracker.bridge.time.perf_counter", return_value=10.0):
        return bridge.compose_frame()


def test_pair_skew_is_a_visible_fault_latched_once_per_dropout():
    source = _Source()
    bridge = _bridge(source)
    assert _frame(bridge)["tracking_quality"]["faultCount"] == 0
    source.lag = 0.075
    lost = _frame(bridge)
    quality = lost["tracking_quality"]
    assert quality["status"] == "error"
    assert quality["faultCount"] == 1
    assert "Left tracker sample is 75 ms behind" in quality["lastFault"]
    assert not lost["l_tracked"]
    assert _frame(bridge)["tracking_quality"]["faultCount"] == 1

    # A native recovery wait must not hide the concrete timing failure or
    # increment it again simply because the bridge polls the same samples.
    source.status = "checking"
    assert _frame(bridge)["tracking_quality"]["faultCount"] == 1
    assert "75 ms behind" in _frame(bridge)["tracking_quality"]["lastFault"]
    source.status = "ready"
    source.lag = 0
    recovered = _frame(bridge)["tracking_quality"]
    assert recovered["status"] == "ready"
    assert recovered["faultCount"] == 1
    assert recovered["lastFault"] == quality["lastFault"]
    source.lag = 0.08
    assert _frame(bridge)["tracking_quality"]["faultCount"] == 2


def test_recovered_tracking_reports_gripper_confirmation_until_engage_ack():
    source = _Source()
    left, right = _Trigger(), _Trigger()
    bridge = _bridge(
        source,
        left_trigger=left,
        right_trigger=right,
        auto_engage=True,
        confirm_auto_engage=True,
    )
    bridge._state = VRState.RECORDING
    bridge._handle_tracking_state(True)
    frame = _frame(bridge)
    bridge._handle_lock_release(frame["lock_release_id"])
    assert frame["tracking_quality"]["controlState"] == "engaged"

    source.lag = 0.075
    lost = _frame(bridge)["tracking_quality"]
    assert lost["controlState"] == "recovering"
    assert lost["reengageRequired"]
    bridge._handle_tracking_state(False)
    source.lag = 0
    recovered = _frame(bridge)
    bridge._handle_lock_release(recovered["lock_release_id"])
    assert recovered["tracking_quality"]["status"] == "ready"
    assert recovered["tracking_quality"]["controlState"] == "awaiting_engagement"
    assert recovered["tracking_quality"]["reengageRequired"]
    assert not recovered["l_lock"]

    left.value = right.value = 0
    assert not _frame(bridge)["l_lock"]
    left.value = right.value = 1
    requested = _frame(bridge)
    assert requested["l_lock"]
    assert requested["tracking_quality"]["reengageRequired"]
    bridge._handle_tracking_state(True)
    engaged = _frame(bridge)["tracking_quality"]
    assert engaged["controlState"] == "engaged"
    assert not engaged["reengageRequired"]


def test_trigger_dropout_is_distinguished_from_healthy_optical_tracking():
    left, right = _Trigger(), _Trigger()
    bridge = _bridge(
        _Source(),
        left_trigger=left,
        right_trigger=right,
        auto_engage=True,
        confirm_auto_engage=True,
    )
    bridge._handle_tracking_state(True)
    frame = _frame(bridge)
    bridge._handle_lock_release(frame["lock_release_id"])
    left.stale = True
    quality = _frame(bridge)["tracking_quality"]
    assert quality["status"] == "ready"
    assert quality["controlState"] == "recovering"
    assert quality["triggersLive"] == {"left": False, "right": True}


def test_standalone_non_lighthouse_bridge_keeps_existing_skew_gate():
    class SourceWithoutQuality:
        def poses(self):
            source = _Source()
            source.lag = 0.075
            return source.poses()

    frame = _frame(_bridge(SourceWithoutQuality()))
    assert "tracking_quality" not in frame
    assert not frame["l_tracked"]
    assert frame["r_tracked"]
