"""Exercise production QA/save branches with fake recorder IO.

As in test_mantis_quality_end, extract the nested session branches so tests do
not start cameras, CAN or a native tracker merely to exercise save decisions.
"""

from __future__ import annotations

import ast
import inspect
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from almond_axol.cli import collect_data


def _session_ast():
    return ast.parse(inspect.getsource(collect_data._run_session)).body[0]


def _save_attempt(
    *, qa=None, rows=30, capture_error=None, discard=False, save_error=None
):
    control = collect_data._QueueCollectControl(threading.Event())
    control.note_ready(14)
    control.note_recording()
    recorder = SimpleNamespace(
        save_episode=Mock(side_effect=save_error),
        episode_count=lambda: 14,
    )
    namespace = {
        **vars(collect_data),
        "control": control,
        "recorder": recorder,
        "_discard_episode": Mock(),
        "restore_dataset_ownership": Mock(),
        "dataset_root": None,
        "cfg": SimpleNamespace(qa_gate=True),
        "recording": True,
        "mantis_mode": True,
        "rerecord": discard,
        "captured_rows": rows,
        "capture_failure": capture_error,
        "qa": qa or collect_data.EpisodeQAStats(total_frames=100),
        "episodes_recorded": 0,
        "episode_idx": 13,
    }
    session = _session_ast()
    # Include the empty-take and aggregate QA decisions which precede save.
    qa_branch = next(
        node
        for node in ast.walk(session)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "recording and mantis_mode"
    )
    outer_loop = next(
        node
        for node in ast.walk(session)
        if isinstance(node, ast.While) and qa_branch in node.body
    )
    empty_branch = outer_loop.body[outer_loop.body.index(qa_branch) - 1]
    wrap = next(
        node
        for node in session.body
        if isinstance(node, ast.FunctionDef) and node.name == "_wrap_up_episode"
    )
    # The test's fake session counter lives in this namespace, not a closure.
    for index, node in enumerate(wrap.body):
        if isinstance(node, ast.Nonlocal):
            wrap.body[index] = ast.copy_location(ast.Global(names=node.names), node)
    reject = next(
        node
        for node in session.body
        if isinstance(node, ast.FunctionDef) and node.name == "_reject_take"
    )
    code = ast.Module(body=[reject, empty_branch, qa_branch, wrap], type_ignores=[])
    exec(compile(code, "<episode-save>", "exec"), namespace)
    namespace["_wrap_up_episode"](True, namespace["rerecord"], rows, capture_error)
    control.note_returning()
    control.note_ready(14)
    control.note_tracking_quality(
        {"status": "ready", "reasons": [], "faultCount": 0, "retakeRequired": False}
    )
    return control, recorder, namespace["_discard_episode"]


def test_aggregate_qa_rejection_stays_visible_after_live_tracking_recovers():
    control, recorder, discard = _save_attempt(
        qa=collect_data.EpisodeQAStats(
            total_frames=100, stale_frames=10, max_pose_lag_s=0.625
        )
    )
    snapshot = control.snapshot()
    assert "10.0% of control samples" in snapshot["notice"]["message"]
    assert "250 ms" in snapshot["notice"]["message"]
    assert "625 ms" in snapshot["notice"]["message"]
    assert snapshot["phase"] == "ready"
    assert snapshot["episode"] == 14
    assert snapshot["episodesRecorded"] == 0
    assert snapshot["trackingQuality"]["status"] == "ready"
    assert [c["command"] for c in snapshot["controls"]] == ["start"]
    recorder.save_episode.assert_not_called()
    discard.assert_called_once()

    control.note_countdown(100)
    control.note_ready(14)  # Cancelling a countdown is not a new take.
    assert control.snapshot()["notice"]["message"] == snapshot["notice"]["message"]
    control.note_recording()
    assert "notice" not in control.snapshot()


@pytest.mark.parametrize(
    ("options", "reason"),
    [
        ({"capture_error": "Left camera disconnected"}, "Left camera disconnected"),
        ({"rows": 0}, "no dataset rows"),
        (
            {"save_error": RuntimeError("Video encoder dropped 10 frames")},
            "Video encoder dropped 10 frames",
        ),
    ],
)
def test_recoverable_save_failures_publish_reason_without_incrementing_count(
    options, reason
):
    control, _recorder, _discard = _save_attempt(**options)
    assert reason in control.snapshot()["notice"]["message"]
    assert control.snapshot()["episodesRecorded"] == 0


def test_intentional_discard_is_not_reported_as_a_save_failure():
    control, recorder, discard = _save_attempt(discard=True)
    assert "notice" not in control.snapshot()
    assert control.snapshot()["episodesRecorded"] == 0
    recorder.save_episode.assert_not_called()
    discard.assert_called_once()


def test_successful_save_increments_count_and_clears_previous_rejection():
    control, recorder, discard = _save_attempt()
    recorder.save_episode.assert_called_once()
    discard.assert_not_called()
    assert control.snapshot()["episodesRecorded"] == 1
    control.note_rejected("Earlier take failed")
    control.note_saved()
    assert "notice" not in control.snapshot()
    assert control.snapshot()["episodesRecorded"] == 2


def test_a_pose_starved_capture_failure_names_the_tracking_loss() -> None:
    stall = (
        "no retained robot-state snapshot brackets camera exposure 5071.8: "
        "the control loop stopped publishing robot state: newest snapshot is "
        "333 ms older than the exposure"
    )
    strict = collect_data.EpisodeQAStats(
        tracking_quality_faults=[
            "Lighthouse calibration invalidated for station fecdd975"
        ]
    )
    assert collect_data._tracking_capture_reason(strict, stall) == (
        "tracking failed mid-take "
        "(Lighthouse calibration invalidated for station fecdd975)"
    )
    strict.tracking_quality_faults += [
        "Right tracker sample is 94 ms behind the other side",
        "Right tracker sample is 118 ms behind the other side",
    ]
    assert collect_data._tracking_capture_reason(strict, stall) == (
        "tracking failed mid-take "
        "(Lighthouse calibration invalidated for station fecdd975; "
        "2 more in the tracking report)"
    )
    lost = collect_data.EpisodeQAStats(
        total_frames=100, untracked_frames=8, disengaged_frames=9, stale_frames=7
    )
    assert "8 untracked, 9 disengaged and 7 stale" in (
        collect_data._tracking_capture_reason(lost, stall) or ""
    )
    # A real camera or recorder fault, or a stall without any tracking
    # trouble, still reads as a capture failure.
    assert collect_data._tracking_capture_reason(lost, "ZED grab timed out") is None
    assert (
        collect_data._tracking_capture_reason(collect_data.EpisodeQAStats(), stall)
        is None
    )
