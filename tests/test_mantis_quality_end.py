"""Exercise the production episode loop's tracking fault fence without hardware.

Extract only the nested loop so the real start/tick/end ordering runs against
fake IO, without opening cameras, CAN, or the IK worker in _run_session.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import threading
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from lerobot.teleoperators.utils import TeleopEvents

from almond_axol.cli import collect_data
from almond_axol.lerobot.robot.robot_axol import _LEFT_EE_KEYS, _RIGHT_EE_KEYS
from almond_axol.mantis.relative import quat_xyzw_to_rotvec
from almond_axol.teleop.core import TCPPoseSnapshot
from almond_axol.vr.models import VRState


def _run_take(
    *, fault_during: str, dataset_root=None, observed=None, tcp_available=True
):
    quality = {
        "status": "ready",
        "faultCount": 0,
        "sourceId": "same-native-source",
        "lastFault": None,
        "reasons": [],
    }
    stopped = False
    capture_started = False
    rows: list[tuple] = []

    def fault() -> None:
        # The native source already recovered before the collector reads it.
        # Only the retained counter proves this take saw a discontinuity.
        quality.update(faultCount=1, lastFault="Left tracker pose jumped")

    async def send_action(_action) -> None:
        await asyncio.sleep(0)
        if capture_started and fault_during == "backend_exception":
            raise RuntimeError("tracker backend disconnected during the take")
        if capture_started and fault_during == "final_send":
            fault()

    def start_capture(_task) -> None:
        nonlocal capture_started, stopped
        capture_started = True
        if fault_during == "startup_stop":
            fault()
            stopped = True

    def publish(*row) -> None:
        if capture_started:
            rows.append(row)

    def finish_capture(**_kwargs) -> int:
        if fault_during == "after_fence":
            fault()
        return len(rows)

    async def blocking(function, _tick, _interval, **_kwargs):
        return function()

    robot = SimpleNamespace(
        set_control_trace_active=Mock(),
        get_joint_observation=Mock(
            side_effect=AssertionError("Mantis tick invoked FK")
        ),
        action_to_dataset=Mock(side_effect=AssertionError("Mantis action invoked FK")),
        get_gripper_observation=lambda: {},
        gripper_action_to_dataset=lambda action: dict(action),
        send_action_async=send_action,
        enable_grippers_async=AsyncMock(),
        disable_grippers_async=AsyncMock(),
    )
    pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    teleop = SimpleNamespace(
        is_resetting=False,
        is_tracking=True,
        is_engaged=lambda: True,
        tracking_quality=lambda: dict(quality),
        tracking_sides=lambda: {"left": True, "right": True},
        trigger_sides=lambda: {"left": True, "right": True},
        send_feedback=Mock(),
        send_feedback_state=Mock(),
        get_action=lambda: {},
        recording_pose_snapshot=lambda: (
            TCPPoseSnapshot.from_message(
                {"left": pose, "right": pose}, collect_data.time.perf_counter()
            )
            if tcp_available
            else None
        ),
        get_teleop_events=lambda: {
            "start_recording": not capture_started,
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.RERECORD_EPISODE: False,
        },
    )
    recorder = SimpleNamespace(
        start_episode=start_capture,
        finish_episode=finish_capture,
        publish=publish,
        poll_capture_error=lambda: None,
        save_episode=Mock(),
        discard_episode=Mock(),
    )
    control = collect_data._QueueCollectControl(threading.Event())
    control.poll_command = lambda: "s" if capture_started else None
    namespace = {
        **vars(collect_data),
        "np": np,
        "TeleopEvents": TeleopEvents,
        "VRState": VRState,
        "quat_xyzw_to_rotvec": quat_xyzw_to_rotvec,
        "_LEFT_EE_KEYS": _LEFT_EE_KEYS,
        "_RIGHT_EE_KEYS": _RIGHT_EE_KEYS,
        "cfg": SimpleNamespace(
            mantis_source="lighthouse", _lighthouse_quality="strict"
        ),
        "mantis_mode": True,
        "teleop": teleop,
        "robot": robot,
        "recorder": recorder,
        "control": control,
        "vrt_cfg": SimpleNamespace(teleop_torque_threshold=0),
        "_stopped": lambda: stopped,
        "_maybe_log_rate": lambda _: None,
        "_sync_spring_caps": lambda: None,
        "_note_ready": lambda _: None,
        "teleop_action_proc": lambda pair: pair[0],
        "robot_action_proc": lambda pair: pair[0],
        "sect": defaultdict(float),
        "max_pose_lag": {"v": 0.0},
        "max_slip": {"v": 0.0},
        "prev_t0": {"v": 0.0},
        "task": "tracking quality fence",
        "relay": None,
        "episode_idx": 0,
        "take_gc": SimpleNamespace(begin=Mock()),
        "tick_watchdog": SimpleNamespace(resume=Mock(), suspend=Mock()),
        "teleop_interval": 1 / 120,
        "run_blocking_with_control_ticks": blocking,
        "tracking_attempt": [],
        "dataset_root": dataset_root,
        "_run_on_robot_loop": asyncio.run,
        "_disable_mantis_grippers": Mock(),
        "_discard_episode": recorder.discard_episode,
    }
    session = ast.parse(inspect.getsource(collect_data._run_session)).body[0]
    loop = next(
        node
        for node in session.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_episode_loop"
    )
    exec(
        compile(ast.Module(body=[loop], type_ignores=[]), "<episode-loop>", "exec"),
        namespace,
    )
    if observed is not None:
        observed.update(namespace)
    if dataset_root is None:
        result = asyncio.run(namespace["_episode_loop"]())
    else:
        # Run the production call/exception and Stop/discard boundary too.
        # Anything which incorrectly falls through reaches a save sentinel.
        guarded_call = next(
            node
            for node in ast.walk(session)
            if isinstance(node, ast.Try)
            and any(
                isinstance(child, ast.Assign)
                and isinstance(child.value, ast.Call)
                and isinstance(child.value.func, ast.Name)
                and child.value.func.id == "_run_on_robot_loop"
                and any(
                    isinstance(arg, ast.Call)
                    and isinstance(arg.func, ast.Name)
                    and arg.func.id == "_episode_loop"
                    for arg in child.value.args
                )
                for child in node.body
            )
        )
        boundary_loop = next(
            node
            for node in ast.walk(session)
            if isinstance(node, ast.While) and guarded_call in node.body
        )
        stop_branch = boundary_loop.body[boundary_loop.body.index(guarded_call) + 1]
        boundary = ast.parse("while True:\n    recorder.save_episode()\n    break\n")
        boundary.body[0].body[:0] = [guarded_call, stop_branch]
        exec(compile(boundary, "<episode-boundary>", "exec"), namespace)
        result = tuple(
            namespace[key]
            for key in (
                "recording",
                "rerecord",
                "contact",
                "captured_rows",
                "capture_failure",
                "qa",
            )
        )
    return result, control, robot


@pytest.mark.parametrize("fault_during", ["final_send", "startup_stop"])
def test_fault_at_last_await_before_take_end_is_latched(fault_during: str) -> None:
    result, control, robot = _run_take(fault_during=fault_during)
    recording, _rerecord, _contact, _rows, capture_error, stats = result
    assert recording and capture_error is None
    assert stats.tracking_quality_faults == ["Left tracker pose jumped"]
    assert stats.tracking_quality_report["verdict"] == "retake_required"
    assert not collect_data.evaluate_episode_qa(stats)[0]
    assert control.snapshot()["trackingQuality"]["retakeRequired"]
    robot.disable_grippers_async.assert_awaited_once()


def test_fault_after_capture_fence_does_not_taint_finished_take() -> None:
    result, _control, robot = _run_take(fault_during="after_fence")
    recording, _rerecord, _contact, rows, capture_error, stats = result
    assert recording and rows > 0 and capture_error is None
    assert stats.tracking_quality_faults == []
    assert stats.tracking_quality_report["verdict"] == "passed"
    assert collect_data.evaluate_episode_qa(stats)[0]
    robot.disable_grippers_async.assert_awaited_once()


def test_missing_raw_tcp_never_records_virtual_fk_or_partial_rows() -> None:
    result, _control, robot = _run_take(fault_during="none", tcp_available=False)
    recording, _rerecord, _contact, rows, capture_error, stats = result
    assert recording and capture_error is None
    assert rows == 0
    assert stats.stale_frames > 0
    assert not collect_data.evaluate_episode_qa(stats)[0]
    robot.get_joint_observation.assert_not_called()
    robot.action_to_dataset.assert_not_called()


def test_backend_exception_preserves_failed_attempt_before_cleanup(tmp_path) -> None:
    observed = {}
    with pytest.raises(RuntimeError, match="tracker backend disconnected"):
        _run_take(
            fault_during="backend_exception",
            dataset_root=tmp_path,
            observed=observed,
        )
    files = list((tmp_path / "meta" / "tracking_quality").glob("*.json"))
    assert len(files) == 1
    report = json.loads(files[0].read_text())
    assert report["verdict"] == "retake_required"
    assert report["termination"] == "error"
    assert "tracker backend disconnected" in report["terminationReason"]
    assert report["capturedRows"] is None  # Unknown after the abrupt exception.
    assert not report["episodeQAPassed"] and not report["saveRequested"]
    observed["recorder"].save_episode.assert_not_called()
    observed["_disable_mantis_grippers"].assert_called_once()


def test_stop_preserves_failed_attempt_before_discard(tmp_path) -> None:
    observed = {}
    _run_take(fault_during="startup_stop", dataset_root=tmp_path, observed=observed)
    files = list((tmp_path / "meta" / "tracking_quality").glob("*.json"))
    assert len(files) == 1
    report = json.loads(files[0].read_text())
    assert report["verdict"] == "retake_required"
    assert report["termination"] == "stopped"
    assert "Collection stopped" in report["terminationReason"]
    assert not report["episodeQAPassed"] and not report["saveRequested"]
    observed["recorder"].save_episode.assert_not_called()
    observed["recorder"].discard_episode.assert_called_once()


def test_diagnostic_write_failure_does_not_mask_backend_exception(
    tmp_path, monkeypatch
) -> None:
    from almond_axol.utils import state_files

    def fail_write(*_args, **_kwargs):
        raise OSError("diagnostic destination is full")

    monkeypatch.setattr(state_files, "secure_atomic_write_json", fail_write)
    observed = {}
    with pytest.raises(RuntimeError, match="tracker backend disconnected"):
        _run_take(
            fault_during="backend_exception",
            dataset_root=tmp_path,
            observed=observed,
        )
    observed["recorder"].save_episode.assert_not_called()
    observed["_disable_mantis_grippers"].assert_called_once()


def test_interrupted_attempt_diagnostics_skip_unstarted_take_and_bound_reason(
    tmp_path,
) -> None:
    monitor = collect_data.EpisodeTrackingQuality(required=True)
    kwargs = {
        "episode_index": 0,
        "reason": "backend failure: " + "x" * 5000,
        "termination": "error",
        "setup_validation": None,
    }
    collect_data._preserve_interrupted_tracking_attempt([monitor], tmp_path, **kwargs)
    assert not (tmp_path / "meta").exists()
    monitor.begin({"status": "ready", "faultCount": 0})
    collect_data._preserve_interrupted_tracking_attempt([monitor], tmp_path, **kwargs)
    report = json.loads(
        next((tmp_path / "meta" / "tracking_quality").glob("*.json")).read_text()
    )
    assert len(report["terminationReason"]) == 1000
    assert report["verdict"] == "retake_required"
