"""DAgger terminal ownership and quit semantics on real pseudo-terminals."""

import os
import pty
import termios
from contextlib import contextmanager
from unittest import mock

import pytest

from almond_axol.cli.dagger_terminal import DaggerStdinControl
from almond_axol.cli.run_policy import _GATE_CONTACT, _GATE_READY


@contextmanager
def terminal():
    master, slave = pty.openpty()
    stream = os.fdopen(os.dup(slave), "r", buffering=1)
    saved = termios.tcgetattr(slave)
    control = DaggerStdinControl()
    try:
        with mock.patch("sys.stdin", stream):
            yield control, master, slave, saved
    finally:
        with mock.patch("sys.stdin", stream):
            control.close()
        stream.close()
        os.close(master)
        os.close(slave)


@pytest.mark.parametrize("keys", [b"q", b"Q", b"qqq\n", b"s\nr\n\nq"])
def test_idle_quit_is_immediate_and_ignores_episode_commands(keys):
    with terminal() as (control, master, slave, saved):
        control.begin_gate("Waiting for VR Record")
        assert not termios.tcgetattr(slave)[3] & termios.ICANON
        os.write(master, keys)
        control._thread.join(1)
        assert not control._thread.is_alive()
        assert control.poll_gate() == "quit"
        control.end_gate()
        assert control.poll_gate() == "quit"
        assert control.quit_requested
        assert not control.abort_requested
        assert termios.tcgetattr(slave) == saved
        with pytest.raises(RuntimeError, match="terminal quit"):
            control.begin_episode()


def test_quit_is_consumed_when_idle_reader_is_joined_before_poll():
    with terminal() as (control, master, _, _):
        control.begin_gate("Waiting for VR Record")
        os.write(master, b"q")
        control._thread.join(1)
        control.end_gate()
        assert control.poll_gate() == "quit"
        assert control.quit_requested


@pytest.mark.parametrize("phase", [_GATE_READY, _GATE_CONTACT])
def test_idle_eof_is_abort_and_never_parks(phase):
    with terminal() as (control, master, slave, saved):
        control.begin_gate("Waiting", phase=phase)
        os.write(master, saved[6][termios.VEOF])
        control._thread.join(1)
        control.end_gate()
        control.note_gate("Idle again", phase=_GATE_READY)
        assert control.poll_gate() == "abort"
        assert control.abort_requested
        assert not control.quit_requested
        assert termios.tcgetattr(slave) == saved


def test_contact_quit_remains_abort_after_idle_message_is_restored():
    with terminal() as (control, master, _, _):
        control.begin_gate("Clear contact", phase=_GATE_CONTACT)
        os.write(master, b"q")
        control._thread.join(1)
        control.note_gate("Waiting for VR Record", phase=_GATE_READY)
        control.end_gate()
        assert control.poll_gate() == "abort"
        assert control.abort_requested
        assert not control.quit_requested
        control.begin_gate("Idle")
        assert not control._thread.is_alive()
        assert control.poll_gate() == "abort"


def test_idle_episode_idle_handoffs_join_readers_and_restore_modes():
    with terminal() as (control, master, slave, saved):
        control.begin_gate("Idle")
        idle_reader = control._thread
        control.end_gate()
        assert not idle_reader.is_alive()
        assert control.poll_gate() is None
        assert termios.tcgetattr(slave) == saved

        control.begin_episode()
        episode_reader = control._thread
        assert episode_reader is not idle_reader
        os.write(master, b"s\n")
        episode_reader.join(1)
        assert control.poll_choice() == "s"
        control.end_episode()
        assert not episode_reader.is_alive()
        assert termios.tcgetattr(slave) == saved

        control.begin_gate("Next episode")
        assert control.poll_gate() is None
        os.write(master, b"q")
        control._thread.join(1)
        assert control.poll_gate() == "quit"


def test_active_episode_quit_needs_no_enter():
    with terminal() as (control, master, slave, saved):
        control.begin_episode()
        os.write(master, b"q")
        control._thread.join(1)
        assert control.poll_choice() == "q"
        assert control.quit_requested
        control.end_episode()
        assert termios.tcgetattr(slave) == saved


def test_blocking_contact_prompt_has_no_competing_reader():
    with terminal() as (control, _, slave, saved):
        control.begin_gate("Idle")
        reader = control._thread

        def answer(_prompt):
            assert not reader.is_alive()
            assert termios.tcgetattr(slave) == saved
            return "q"

        with mock.patch("builtins.input", side_effect=answer):
            assert not control.await_continue("Contact", phase=_GATE_CONTACT)
        assert not control.quit_requested
        assert control.abort_requested
