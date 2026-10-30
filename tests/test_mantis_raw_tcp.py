"""Mantis capture stays tied to raw tracker samples while virtual IK stalls."""

from __future__ import annotations

import logging
import multiprocessing
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from almond_axol.cli import collect_data
from almond_axol.lerobot.teleop.config_vr import AxolVRTeleopConfig
from almond_axol.lerobot.teleop.teleop_vr import AxolVRTeleop
from almond_axol.mantis.relative import quat_xyzw_to_matrix
from almond_axol.robot.jelly import JellyConfig
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore
from almond_axol.teleop.tcp import AbsoluteTCPMapping
from almond_axol.teleop.worker import IKWorker
from almond_axol.vr.models import VRFrame, VRPose, VRPosition, VRQuaternion


def _frame(
    offset: float = 0.0, *, locks: bool = False, stamp: float | None = None
) -> VRFrame:
    stamp = time.perf_counter() if stamp is None else stamp
    quat = VRQuaternion(x=0.0, y=np.sin(offset / 2), z=0.0, w=np.cos(offset / 2))
    return VRFrame(
        l_ee=VRPose(position=VRPosition(x=-0.2, y=1.0, z=offset), quaternion=quat),
        r_ee=VRPose(position=VRPosition(x=0.2, y=1.0, z=-offset), quaternion=quat),
        l_elbow=VRPosition(x=0.0, y=0.0, z=0.0),
        r_elbow=VRPosition(x=0.0, y=0.0, z=0.0),
        l_lock=locks,
        r_lock=locks,
        t_host=stamp,
        t=stamp * 1000,
    )


def _worker(*, calibrated: bool) -> IKWorker:
    worker = object.__new__(IKWorker)
    worker._config = VRTeleopConfig(absolute_mode=True)
    worker._last_mapped_quat = {}
    half = np.sqrt(0.5)
    rotation = quat_xyzw_to_matrix(np.array([0.0, half, 0.0, half]))
    worker._tcp_transforms = (
        {
            "left": (np.array([0.01, -0.03, 0.07]), rotation),
            "right": (np.array([-0.02, 0.04, 0.05]), rotation.T),
        }
        if calibrated
        else {}
    )
    worker._solver = SimpleNamespace(
        config=SimpleNamespace(reach_soft_start=0.7),
        shoulder_positions={
            "left": np.array([0.0, 0.2, 0.0]),
            "right": np.array([0.0, -0.2, 0.0]),
        },
    )
    worker._abs_active = True
    worker._abs_base = None
    worker._abs_offset = {}
    worker._rest_fk_poses = lambda: (
        (np.array([0.3, 0.2, 0.2]), np.eye(3)),
        (np.array([0.3, -0.2, 0.2]), np.eye(3)),
    )
    # Real engage fitter, with swap rejection tested separately in its suite.
    worker._side_swap_rejection = lambda *_args: None
    anchor = _frame()
    mapped = []
    for side in ("left", "right"):
        pose = getattr(anchor, f"{side[0]}_ee")
        pos, quat = worker._apply_tcp_transform(
            side,
            np.array([pose.position.x, pose.position.y, pose.position.z]),
            np.array([0.0, 0.0, 0.0, 1.0]),
        )
        mapped.extend((pos, quat_xyzw_to_matrix(quat)))
    assert worker._engage_absolute(*mapped)
    return worker


def _core() -> VRTeleopCore:
    return VRTeleopCore(
        VRTeleopConfig(absolute_mode=True),
        logging.getLogger(__name__),
        lambda _enabled: None,
        record_raw_tcp=True,
    )


def _engage(core: VRTeleopCore, mapping: dict) -> None:
    frame = _frame(locks=True)
    core.update_engage(frame)
    core._adopt_raw_tcp_mapping(mapping, core._raw_tcp_epoch, frame)
    assert core.last_tcp_snapshot is not None


@pytest.mark.parametrize("calibrated", [False, True])
def test_exported_raw_mapping_matches_worker_mount_and_engage_transforms(
    calibrated: bool,
) -> None:
    worker = _worker(calibrated=calibrated)
    mapping = AbsoluteTCPMapping.from_message(worker.raw_tcp_mapping())
    for offset in (0.0, 0.1, -0.4, 2.0):
        frame = _frame(offset)
        actual = mapping.map_frame(frame)
        expected = []
        for side in ("left", "right"):
            pose = getattr(frame, f"{side[0]}_ee")
            pos, quat = worker._apply_tcp_transform(
                side,
                np.array([pose.position.x, pose.position.y, pose.position.z]),
                np.array(
                    [
                        pose.quaternion.x,
                        pose.quaternion.y,
                        pose.quaternion.z,
                        pose.quaternion.w,
                    ]
                ),
            )
            target = worker._absolute_target(side, pos, quat_xyzw_to_matrix(quat))
            expected.append(target)
            np.testing.assert_allclose(actual[side][:3], target[0], atol=2e-7)
            np.testing.assert_allclose(
                quat_xyzw_to_matrix(np.array(actual[side][3:])), target[1], atol=2e-7
            )
        assert tuple(actual["out_of_reach"]) == worker._out_of_reach(*expected)


def test_raw_mapping_copies_worker_calibration_values() -> None:
    message = _worker(calibrated=True).raw_tcp_mapping()
    mapping = AbsoluteTCPMapping.from_message(message)
    before = mapping.map_frame(_frame(0.2))
    message["left"][0] += 100
    message["base"][0] += 100
    assert mapping.map_frame(_frame(0.2)) == before


def test_receive_callback_tracks_motion_during_half_second_ik_stall() -> None:
    teleop = AxolVRTeleop(
        AxolVRTeleopConfig(
            vr_teleop_config=VRTeleopConfig(absolute_mode=True),
            jelly=JellyConfig(wheels=False, lift=False),
            record_raw_tcp=True,
        )
    )
    core = teleop._core
    worker = _worker(calibrated=True)
    message = worker.raw_tcp_mapping()
    mapping = AbsoluteTCPMapping.from_message(message)
    first = _frame(locks=True)
    latest = [first]
    parent, child = multiprocessing.Pipe()
    stop = threading.Event()
    solved = threading.Event()
    loop = threading.Thread(
        target=core.run_ik_loop,
        args=(parent, lambda: latest[0], stop, lambda: True, lambda _now: solved.set()),
    )
    loop.start()
    try:
        assert child.poll(2)
        initial = child.recv()
        child.send(
            (
                "q",
                np.zeros(14),
                worker.abs_base_msg,
                mapping.map_frame(initial),
                message,
            )
        )
        assert solved.wait(2)
        solved.clear()
        next_frame = _frame(0.01)
        teleop._on_vr_frame(next_frame)
        latest[0] = next_frame
        assert child.poll(2)
        pending = child.recv()
        # Keep the worker reply blocked for longer than the old 250 ms stale
        # QA limit, while real receive callbacks publish moving/rotating poses.
        deadline = time.perf_counter() + 0.55
        index = 1
        while time.perf_counter() < deadline:
            moving = _frame(0.01 * index)
            teleop._on_vr_frame(moving)
            latest[0] = moving
            sample = teleop.tcp_pose_snapshot()
            assert sample.pose_host_ts == moving.t_host
            np.testing.assert_allclose(
                sample.left, mapping.map_frame(moving)["left"], atol=1e-9
            )
            index += 1
            time.sleep(0.01)
        captured = teleop.tcp_pose_snapshot()
        child.send(
            ("q", np.ones(14), worker.abs_base_msg, mapping.map_frame(pending), message)
        )
        assert solved.wait(2)
        stop.set()
        # The late reply neither overwrites the pose nor advances its timestamp.
        assert teleop.tcp_pose_snapshot() is captured
        assert teleop.pose_capture_ts() == captured.pose_host_ts
    finally:
        stop.set()
        loop.join(2)
        parent.close()
        child.close()
    assert not loop.is_alive()


@pytest.mark.parametrize("fault", ["tracking", "quality", "reset", "block", "pause"])
def test_invalidated_anchor_cannot_be_restored_by_in_flight_reply(fault: str) -> None:
    core = _core()
    mapping = _worker(calibrated=False).raw_tcp_mapping()
    _engage(core, mapping)
    old_epoch = core._raw_tcp_epoch
    if fault == "tracking":
        core.note_raw_tcp_frame(_frame().model_copy(update={"l_tracked": False}))
    elif fault == "quality":
        core.note_raw_tcp_frame(
            _frame().model_copy(update={"tracking_quality": {"status": "error"}})
        )
    elif fault == "reset":
        core.request_reset()
    elif fault == "block":
        core.block_engage()
    else:
        core.pause_ik()
    core._adopt_raw_tcp_mapping(mapping, old_epoch, _frame())
    core.note_raw_tcp_frame(_frame(0.3))
    assert core.last_tcp_snapshot is None
    assert core.last_pose_host_ts is None


def test_new_engage_uses_new_base_and_refuses_previous_reply() -> None:
    core = _core()
    first = _worker(calibrated=False).raw_tcp_mapping()
    _engage(core, first)
    old_epoch = core._raw_tcp_epoch
    core._disengage_all()
    release = _frame()
    assert core._accept_tracking_frame(release)
    core.update_engage(release)
    squeeze = _frame(locks=True)
    assert core._accept_tracking_frame(squeeze)
    core.update_engage(squeeze)
    second = _worker(calibrated=False).raw_tcp_mapping()
    second["base"][0] += 0.4
    core._adopt_raw_tcp_mapping(first, old_epoch, squeeze)
    assert core.last_tcp_snapshot is None
    core._adopt_raw_tcp_mapping(second, core._raw_tcp_epoch, squeeze)
    moving = _frame(0.2)
    core.note_raw_tcp_frame(moving)
    expected = AbsoluteTCPMapping.from_message(second).map_frame(moving)
    np.testing.assert_allclose(core.last_tcp_snapshot.left, expected["left"])


def test_transient_fault_while_ik_waits_forces_release_even_when_frame_is_unchanged() -> (
    None
):
    core = _core()
    worker = _worker(calibrated=False)
    message = worker.raw_tcp_mapping()
    mapping = AbsoluteTCPMapping.from_message(message)
    latest = [_frame(locks=True)]
    parent, child = multiprocessing.Pipe()
    stop = threading.Event()
    solved = threading.Event()
    loop = threading.Thread(
        target=core.run_ik_loop,
        args=(parent, lambda: latest[0], stop, lambda: True, lambda _now: solved.set()),
    )
    loop.start()
    try:
        assert child.poll(2)
        initial = child.recv()
        child.send(
            (
                "q",
                np.zeros(14),
                worker.abs_base_msg,
                mapping.map_frame(initial),
                message,
            )
        )
        assert solved.wait(2)
        solved.clear()
        # Locks released, but the toggle-engaged session remains active. The
        # interpolator can return this same object for every stationary sample.
        unchanged = _frame()
        latest[0] = unchanged
        assert child.poll(2)
        pending = child.recv()
        assert pending.l_lock and pending.r_lock
        core.note_raw_tcp_frame(unchanged.model_copy(update={"r_tracked": False}))
        core.note_raw_tcp_frame(unchanged)  # recovers before the solve returns
        assert core.last_tcp_snapshot is None
        child.send(
            (
                "q",
                np.zeros(14),
                worker.abs_base_msg,
                mapping.map_frame(pending),
                message,
            )
        )
        # Identity reuse must not suppress the forced-disengage low handshake.
        assert child.poll(2)
        released = child.recv()
        assert not released.l_lock and not released.r_lock
        assert not core.teleop_enabled
        assert core.last_tcp_snapshot is None
        child.send(("q", np.zeros(14), None, mapping.map_frame(released), None))
        latest[0] = _frame(locks=True)
        assert child.poll(2)
        reengaged = child.recv()
        assert reengaged.l_lock and reengaged.r_lock
        solved.clear()
        child.send(
            (
                "q",
                np.zeros(14),
                worker.abs_base_msg,
                mapping.map_frame(reengaged),
                message,
            )
        )
        assert solved.wait(2)
        stop.set()
        assert core.last_tcp_snapshot is not None
        assert core.last_tcp_snapshot.pose_host_ts == reengaged.t_host
    finally:
        stop.set()
        loop.join(2)
        parent.close()
        child.close()
    assert not loop.is_alive()


@pytest.mark.parametrize("stamp", [None, "old", "future"])
def test_missing_or_late_capture_is_never_retimestamped(stamp: str | None) -> None:
    core = _core()
    _engage(core, _worker(calibrated=True).raw_tcp_mapping())
    original = core.last_tcp_snapshot
    value = (
        None
        if stamp is None
        else time.perf_counter() + (-0.6 if stamp == "old" else 0.1)
    )
    core.note_raw_tcp_frame(_frame(1.0).model_copy(update={"t_host": value}))
    assert core.last_tcp_snapshot is original


def test_raw_capture_is_disabled_for_normal_robot_configs_and_enabled_for_mantis() -> (
    None
):
    assert not AxolVRTeleopConfig().record_raw_tcp
    cfg = collect_data.CollectDataConfig(repo_id="test/raw_tcp", task="test")
    collect_data._apply_mantis_profile(cfg)
    assert cfg.teleop_config.record_raw_tcp
    assert cfg.teleop_config.vr_teleop_config.absolute_mode


def test_recording_accessor_cannot_fall_back_to_ik_after_raw_anchor_invalidates():
    from unittest.mock import Mock

    core = _core()
    _engage(core, _worker(calibrated=True).raw_tcp_mapping())
    teleop = AxolVRTeleop.__new__(AxolVRTeleop)
    teleop.config = SimpleNamespace(record_raw_tcp=True)
    teleop._core = core
    # The upstream pull-based adapter could otherwise resurrect a mapping
    # from the old IK calibration immediately after a receive-side fault.
    core.calibrated_pose_snapshot = Mock(
        side_effect=AssertionError("old IK calibration must not be consulted")
    )
    before = core.last_tcp_snapshot
    assert teleop.recording_pose_snapshot() is before
    core.note_raw_tcp_frame(_frame().model_copy(update={"l_tracked": False}))
    assert teleop.recording_pose_snapshot() is None
    core.calibrated_pose_snapshot.assert_not_called()
