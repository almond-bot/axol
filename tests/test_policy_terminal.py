"""Real pseudo-terminal tests for opt-in immediate quit; no robot is constructed."""

import os
import pty
import select
import subprocess
import sys
import termios
import threading
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest

from almond_axol.cli.run_policy import _StdinPolicyControl
from almond_axol.lerobot.rollout import stdin_watcher


@contextmanager
def terminal_reader(*, immediate=True, callback=None, subtasks=0, queued=b""):
    master, slave = pty.openpty()
    saved = termios.tcgetattr(slave)
    stream = os.fdopen(os.dup(slave), "r", buffering=1)
    result = {"choice": None}
    stop, ready = threading.Event(), threading.Event()
    thread = threading.Thread(
        target=stdin_watcher,
        args=(stop, result, callback, subtasks),
        kwargs={
            "eof_choice": "abort",
            "immediate_quit": immediate,
            "ready_event": ready,
        },
    )
    try:
        if queued:
            os.write(master, queued)
        with mock.patch("sys.stdin", stream):
            thread.start()
            assert ready.wait(1), "terminal reader startup stalled"
            yield master, slave, saved, result, stop, thread
            stop.set()
            thread.join(1)
            assert not thread.is_alive()
    finally:
        stop.set()
        thread.join(1)
        stream.close()
        os.close(master)
        os.close(slave)


def assert_finished(thread, result, choice):
    thread.join(1)
    assert not thread.is_alive()
    assert result == {"choice": choice}


@pytest.mark.parametrize("keys", [b"q", b"Q", b"qqq", b"q\n"])
def test_single_quit_key_and_repeated_q_restore_terminal(keys, capsys):
    with terminal_reader() as (master, slave, saved, result, stop, thread):
        active = termios.tcgetattr(slave)
        assert not active[3] & termios.ICANON
        assert active[3] & termios.ISIG
        os.write(master, keys)
        assert_finished(thread, result, "q")
        assert termios.tcgetattr(slave) == saved
        assert not select.select([slave], [], [], 0)[0]  # Quit repeats/Enter flushed.
    assert "Quit requested; stopping episode." in capsys.readouterr().out


def test_q_queued_before_mode_switch_is_not_flushed():
    with terminal_reader(queued=b"qqq") as (_, slave, saved, result, _, thread):
        assert_finished(thread, result, "q")
        assert termios.tcgetattr(slave) == saved


def test_legacy_mode_reproduces_accumulated_q_issue_without_changing_contract():
    with terminal_reader(immediate=False) as (master, slave, saved, result, _, thread):
        for keys in (b"q", b"q", b"q\n"):
            os.write(master, keys)
            thread.join(0.08)
            assert thread.is_alive()
            assert result["choice"] is None
        os.write(master, b"q\n")
        assert_finished(thread, result, "q")
        assert termios.tcgetattr(slave) == saved


@pytest.mark.parametrize("command", [b"s", b"r"])
def test_save_and_rerecord_still_require_enter(command):
    with terminal_reader() as (master, slave, saved, result, _, thread):
        os.write(master, command)
        thread.join(0.08)
        assert thread.is_alive()
        assert result["choice"] is None
        os.write(master, b"\n")
        assert_finished(thread, result, command.decode())
        assert termios.tcgetattr(slave) == saved


def test_batched_subtasks_invalid_input_and_quit_have_no_buffer_starvation(capsys):
    received = []
    with terminal_reader(callback=received.append, subtasks=12) as (
        master,
        _,
        _,
        result,
        _,
        thread,
    ):
        os.write(master, b"1\n12\ninvalid\nq")
        assert_finished(thread, result, "q")
    assert received == [1, 12]
    output = capsys.readouterr().out
    assert "Unrecognized input" in output
    assert "Quit requested" in output


@pytest.mark.parametrize("exit_kind", ["stop", "ctrl-d", "read error"])
def test_stop_eof_and_read_failure_restore_terminal(exit_kind):
    with terminal_reader() as (master, slave, saved, result, stop, thread):
        if exit_kind == "stop":
            stop.set()
        elif exit_kind == "ctrl-d":
            os.write(master, saved[6][termios.VEOF])
        else:
            with mock.patch("os.read", side_effect=OSError("injected read failure")):
                os.write(master, b"x")
                thread.join(1)
        thread.join(1)
        assert not thread.is_alive()
        assert termios.tcgetattr(slave) == saved
        assert result["choice"] == (None if exit_kind == "stop" else "abort")
        if exit_kind == "read error":
            assert "injected read failure" in result["error"]


def test_mode_setup_failure_is_published_before_policy_workers_can_start():
    master, slave = pty.openpty()
    stream = os.fdopen(os.dup(slave), "r")
    saved = termios.tcgetattr(slave)
    control = _StdinPolicyControl(eof_choice="abort", immediate_quit=True)
    try:
        with (
            mock.patch("sys.stdin", stream),
            mock.patch("termios.tcsetattr", side_effect=OSError("mode setup failed")),
        ):
            with pytest.raises(RuntimeError, match="mode setup failed"):
                control.begin_episode()
            with pytest.raises(RuntimeError, match="mode setup failed"):
                control.end_episode()
        assert not control._thread.is_alive()
        assert termios.tcgetattr(slave) == saved
    finally:
        stream.close()
        os.close(master)
        os.close(slave)


def test_terminal_restore_error_cannot_remain_a_normal_quit():
    original_setattr = termios.tcsetattr
    with terminal_reader() as (master, slave, saved, result, _, thread):
        with mock.patch("termios.tcsetattr", side_effect=OSError("restore failed")):
            os.write(master, b"q")
            thread.join(1)
        original_setattr(slave, termios.TCSANOW, saved)
        assert not thread.is_alive()
        assert result["choice"] == "abort"
        assert "restore failed" in result["error"]


def test_fault_prompt_discards_late_enter_from_explicit_quit():
    master, slave = pty.openpty()
    stream = os.fdopen(os.dup(slave), "r")
    control = _StdinPolicyControl(eof_choice="abort", immediate_quit=True)
    try:
        with mock.patch("sys.stdin", stream):
            control.begin_episode()
            os.write(master, b"q")
            control._thread.join(1)
            assert control.poll_choice() == "q"
            control.end_episode()
            os.write(master, b"\n")  # Arrives during park, after watcher restoration.
            assert select.select([slave], [], [], 0.5)[0]
            control.discard_quit_input()
            assert not select.select([slave], [], [], 0)[0]
    finally:
        stream.close()
        os.close(master)
        os.close(slave)


def test_real_ctrl_c_remains_sigint_and_restores_terminal():
    master, slave = pty.openpty()
    saved = termios.tcgetattr(slave)
    code = """
import fcntl,os,signal,sys,termios,time
os.setsid()
fcntl.ioctl(0, termios.TIOCSCTTY, 0)
from almond_axol.cli.run_policy import _StdinPolicyControl
control=_StdinPolicyControl(eof_choice="abort", immediate_quit=True)
try:
    control.begin_episode()
    print("READY",flush=True)
    while True: time.sleep(.1)
except KeyboardInterrupt:
    print("INTERRUPTED",flush=True)
finally:
    control.end_episode()
print("RESTORED",flush=True)
"""
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", code],
        stdin=slave,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=Path(__file__).resolve().parents[1],
    )
    try:
        assert select.select([proc.stdout], [], [], 10)[0], "child did not become ready"
        assert os.read(proc.stdout.fileno(), 4096).strip() == b"READY"
        os.write(master, saved[6][termios.VINTR])
        stdout, stderr = proc.communicate(timeout=5)
        assert proc.returncode == 0, stderr.decode()
        assert b"INTERRUPTED" in stdout and b"RESTORED" in stdout
        assert termios.tcgetattr(slave) == saved
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        os.close(master)
        os.close(slave)
