"""Exact pause/resume commits, including a whole boundary inside one read."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest import mock

import pytest

from almond_axol.recording.record_proc import (
    InProcessRecorder,
    _CaptureGate,
    run_capture_loop,
    run_encoded_capture_loop,
)


class Camera:
    frames_are_independent = True
    capture_fps = 60
    pending = 0

    def __init__(self, before_capture=None, delayed_packet=False):
        self.before_capture = before_capture
        self.delayed_packet = delayed_packet
        self.calls = 0
        self.first_stamp = None

    def flush(self):
        pass

    def packet(self, minimum):
        self.calls += 1
        stamp = max(time.perf_counter(), minimum)
        if self.calls == 1:
            self.first_stamp = stamp
            if self.before_capture is not None:
                self.before_capture()
        if self.delayed_packet and self.calls == 2:
            # Newer than the rejected first exposure, but still before the
            # gate reopened. A monotonicity check alone cannot reject it.
            stamp = self.first_stamp + 0.000001
        return f"frame-{self.calls}".encode(), stamp, time.perf_counter()

    def read_at_or_after(self, target_perf_ts, timeout_ms):
        return self.packet(target_perf_ts)

    def read_next_au(self, timeout_ms):
        if timeout_ms == 0:
            raise TimeoutError
        return self.packet(0.0)


def start_capture(kind, *, stage="prepare", delayed_packet=False, block_commit=False):
    stop, entered, release = threading.Event(), threading.Event(), threading.Event()
    record = _CaptureGate()
    record.set()
    rows, row_times, errors = [], [], []
    counter = {"n": 0}
    source = {"intervention": False}

    def block():
        entered.set()
        assert release.wait(timeout=3), "test did not release blocked capture"

    camera = Camera(block if stage == "capture" else None, delayed_packet)
    prepared = 0

    def process(obs):
        nonlocal prepared
        prepared += 1
        if prepared == 1 and stage == "prepare":
            block()
        return obs

    def add_frame(row):
        if block_commit:
            block()
        rows.append(row)
        stop.set()

    def snapshot(stamp=None):
        flag = source["intervention"]
        return (
            {"state": int(flag)},
            {"target": int(flag)},
            stamp or time.perf_counter(),
            flag,
        )

    dataset = SimpleNamespace(features={"intervention": {}}, add_frame=add_frame)
    loop = run_capture_loop if kind == "raw" else run_encoded_capture_loop

    def run():
        with (
            mock.patch(
                "lerobot.utils.feature_utils.build_dataset_frame",
                side_effect=lambda _features, values, prefix: {
                    f"{prefix}.{key}": value for key, value in values.items()
                },
            ),
            mock.patch("lerobot.utils.visualization_utils.log_rerun_data"),
        ):
            loop(
                cameras={"eye": camera},
                read_snapshot=snapshot,
                read_snapshot_nearest=snapshot,
                dataset=dataset,
                robot_obs_proc=process,
                fps=60,
                task="fence",
                rerun_ip=None,
                stop_event=stop,
                record_event=record,
                frame_counter=counter,
                row_times=row_times,
                on_error=errors.append,
            )

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    return SimpleNamespace(
        worker=worker,
        stop=stop,
        entered=entered,
        release=release,
        record=record,
        source=source,
        counter=counter,
        rows=rows,
        row_times=row_times,
        errors=errors,
        camera=camera,
    )


def finish_capture(env):
    env.release.set()
    env.worker.join(timeout=3)
    env.stop.set()
    assert not env.worker.is_alive()
    assert not env.errors


@pytest.mark.parametrize("kind", ["raw", "encoded"])
@pytest.mark.parametrize("stage", ["capture", "prepare"])
def test_fast_pause_resume_rejects_old_acquisition_even_when_gate_was_never_seen_closed(
    kind, stage
):
    env = start_capture(kind, stage=stage)
    try:
        assert env.entered.wait(timeout=3)
        assert env.record.transition(False, env.counter) == 0
        # This models the collector publishing its first new-source snapshot
        # before reopening capture. Action and label move together.
        env.source["intervention"] = True
        assert env.record.transition(True, env.counter) == 0
        finish_capture(env)
        assert env.counter == {"n": 1}
        assert len(env.row_times) == 1
        assert env.rows[0]["observation.eye"] == b"frame-2"
        assert env.rows[0]["action.target"] == 1
        assert bool(env.rows[0]["intervention"][0])
        assert env.row_times[0] >= env.record.opened_at
    finally:
        env.release.set()
        env.stop.set()
        env.worker.join(timeout=3)


@pytest.mark.parametrize("kind", ["raw", "encoded"])
def test_pause_ack_does_not_wait_for_acquisition_and_no_row_commits_while_paused(kind):
    env = start_capture(kind)
    try:
        assert env.entered.wait(timeout=3)
        started = time.perf_counter()
        assert env.record.transition(False, env.counter) == 0
        assert time.perf_counter() - started < 0.1
        env.release.set()
        time.sleep(0.04)
        assert not env.rows
        assert env.counter == {"n": 0}
        env.source["intervention"] = True
        assert env.record.transition(True, env.counter) == 0
        finish_capture(env)
        assert len(env.rows) == 1
        assert bool(env.rows[0]["intervention"][0])
    finally:
        env.release.set()
        env.stop.set()
        env.worker.join(timeout=3)


@pytest.mark.parametrize("kind", ["raw", "encoded"])
def test_pause_waits_for_in_progress_commit_and_reports_its_exact_count(kind):
    env = start_capture(kind, stage="none", block_commit=True)
    paused = threading.Event()
    counts = []

    def pause():
        counts.append(env.record.transition(False, env.counter))
        paused.set()

    pausing = threading.Thread(target=pause, daemon=True)
    try:
        assert env.entered.wait(timeout=3)
        pausing.start()
        assert not paused.wait(timeout=0.04)
        finish_capture(env)
        assert paused.wait(timeout=3)
        assert counts == [1]
        assert env.counter == {"n": 1}
        assert len(env.row_times) == len(env.rows) == 1
        assert not env.record.is_set()
    finally:
        env.release.set()
        env.stop.set()
        env.worker.join(timeout=3)
        if pausing.ident is not None:
            pausing.join(timeout=3)


@pytest.mark.parametrize("kind", ["raw", "encoded"])
def test_resume_discards_delayed_pre_boundary_exposure_after_old_row_was_rejected(kind):
    env = start_capture(kind, delayed_packet=True)
    try:
        assert env.entered.wait(timeout=3)
        assert env.record.transition(False, env.counter) == 0
        env.source["intervention"] = True
        assert env.record.transition(True, env.counter) == 0
        finish_capture(env)
        assert env.rows[0]["observation.eye"] == b"frame-3"
        assert env.row_times[0] >= env.record.opened_at
        assert env.counter == {"n": 1}
    finally:
        env.release.set()
        env.stop.set()
        env.worker.join(timeout=3)


def test_inprocess_recorder_uses_exact_gate_and_idempotent_transitions():
    recorder = InProcessRecorder.__new__(InProcessRecorder)
    recorder._record = _CaptureGate()
    recorder._frames = {"n": 17}
    assert recorder.resume_episode() == 17
    epoch = recorder._record.epoch
    assert recorder.resume_episode() == 17
    assert recorder._record.epoch == epoch
    assert recorder.pause_episode() == 17
    assert recorder._record.epoch == epoch + 1
    assert recorder.pause_episode() == 17
    assert recorder._record.epoch == epoch + 1


def test_wedged_commit_cannot_block_a_gate_command_indefinitely():
    gate = _CaptureGate()
    entered, release = threading.Event(), threading.Event()

    def commit():
        with gate.commit_lock:
            entered.set()
            release.wait(timeout=3)

    worker = threading.Thread(target=commit, daemon=True)
    worker.start()
    try:
        assert entered.wait(timeout=3)
        with (
            mock.patch("almond_axol.recording.record_proc._CMD_TIMEOUT_S", 0.02),
            pytest.raises(RuntimeError, match="gate timeout"),
        ):
            gate.transition(False, {"n": 0})
    finally:
        release.set()
        worker.join(timeout=3)
