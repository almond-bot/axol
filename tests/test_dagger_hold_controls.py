"""Opt-in grip holds preserve local toggle behavior and fail closed on sync."""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from almond_axol.lerobot.teleop.teleop_vr_dagger import DaggerVRTeleop
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.dagger import DaggerTeleopCore


def frame(left=False, right=False, left_grip=0.7, right_grip=0.8):
    return SimpleNamespace(
        l_lock=left,
        r_lock=right,
        l_grip=left_grip,
        r_grip=right_grip,
    )


def make_core(*, hold=True):
    broadcast = mock.Mock()
    core = DaggerTeleopCore(VRTeleopConfig(), logging.getLogger(__name__), broadcast)
    core.hold_to_intervene = hold
    core.intervention_allowed.set()
    core._sync_to_robot = mock.Mock()
    return core, broadcast


@pytest.mark.parametrize("side", ["left", "right"])
def test_one_grip_syncs_before_enable_and_starts_takeover_velocity_ramp(side):
    core, broadcast = make_core()

    def sync():
        assert not core.teleop_enabled
        assert core.l_lock_raw or core.r_lock_raw

    core._sync_to_robot.side_effect = sync
    core.update_engage(frame(left=side == "left", right=side == "right"))
    assert core.left_enabled is (side == "left")
    assert core.right_enabled is (side == "right")
    core._sync_to_robot.assert_called_once_with()
    broadcast.assert_called_once_with(True)
    assert core._engage_time is not None
    assert core.smooth_left.max_vel == core.config.engage_max_vel
    assert core.smooth_right.max_vel == core.config.engage_max_vel
    assert not core.consume_freeze()
    assert core.l_grip == (0.7 if side == "left" else 0.0)
    assert core.r_grip == (0.8 if side == "right" else 0.0)


def test_grips_follow_each_arm_and_releasing_both_disables_without_toggle():
    core, broadcast = make_core()
    core.update_engage(frame(left=True))
    core.update_engage(frame(left=True, right=True))
    assert core.left_enabled and core.right_enabled
    core.update_engage(frame(right=True))
    assert not core.left_enabled and core.right_enabled
    core.update_engage(frame())
    assert not core.teleop_enabled
    assert not core.l_lock_raw and not core.r_lock_raw
    assert core._engage_time is None
    assert broadcast.call_args_list == [mock.call(True), mock.call(False)]
    core._sync_to_robot.assert_called_once_with()


def test_failed_sync_requires_explicit_release_before_retry():
    core, broadcast = make_core()
    core._sync_to_robot.side_effect = [RuntimeError("worker unavailable"), None]
    core.update_engage(frame(left=True))
    assert not core.teleop_enabled
    assert core.l_lock_raw
    assert core._hold_inhibited
    core.update_engage(frame(left=True))
    assert core._sync_to_robot.call_count == 1
    broadcast.assert_not_called()
    core.update_engage(frame())
    core.update_engage(frame(right=True))
    assert core.right_enabled
    assert core._sync_to_robot.call_count == 2
    broadcast.assert_called_once_with(True)


@pytest.mark.parametrize("forced", ["force", "stale"])
def test_forced_disengagement_preserves_raw_grips_and_inhibits_stale_reengage(forced):
    core, _broadcast = make_core()
    core.update_engage(frame(left=True))
    if forced == "force":
        core.force_disengage()
    else:
        core._last_frame_time = time.perf_counter() - 10
        core._maybe_disengage_stale(mock.Mock(), None, lambda: True)
    assert not core.teleop_enabled
    assert core.l_lock_raw and not core.r_lock_raw
    assert core._hold_inhibited
    core.update_engage(frame(left=True))
    assert not core.teleop_enabled
    assert core._sync_to_robot.call_count == 1
    core.update_engage(frame())
    core.update_engage(frame(left=True))
    assert core.teleop_enabled
    assert core._sync_to_robot.call_count == 2


def test_pressed_while_disarmed_cannot_latch_into_next_episode():
    core, _broadcast = make_core()
    core.intervention_allowed.clear()
    core.update_engage(frame(left=True))
    core.intervention_allowed.set()
    core.update_engage(frame(left=True))
    assert not core.teleop_enabled
    core._sync_to_robot.assert_not_called()
    core.update_engage(frame())
    core.update_engage(frame(left=True))
    assert core.teleop_enabled
    core._sync_to_robot.assert_called_once_with()


def test_default_local_dagger_preserves_freeze_both_grip_takeover_and_toggle_back():
    core, broadcast = make_core(hold=False)
    core.update_engage(frame(left=True))
    assert core.consume_freeze()
    assert not core.teleop_enabled
    core.update_engage(frame())
    core.update_engage(frame(left=True, right=True))
    assert core.left_enabled and core.right_enabled
    core.update_engage(frame())
    assert core.teleop_enabled  # default is latched, unlike grip-hold mode
    core.update_engage(frame(right=True))
    assert not core.teleop_enabled
    assert broadcast.call_args_list == [mock.call(True), mock.call(False)]


def test_sync_seeds_filters_and_replaces_old_playback_target():
    core = DaggerTeleopCore(VRTeleopConfig(), logging.getLogger(__name__), mock.Mock())
    core.hold_to_intervene = True
    core.intervention_allowed.set()
    left, right = (
        np.arange(8, dtype=np.float32) / 20,
        -np.arange(8, dtype=np.float32) / 20,
    )
    q = np.concatenate((left[:7], right[:7]))
    pipe = mock.Mock()
    core.attach(pipe, lambda: (left, right))
    core.q = np.full(14, 99.0)
    core._segment = (core.q.copy(), core.q.copy(), 0.0, 1.0)
    with mock.patch(
        "almond_axol.teleop.dagger.recv_with_timeout", return_value=("synced", q)
    ):
        core.update_engage(frame(left=True))
    message = pipe.send.call_args.args[0]
    assert message[0] == "sync"
    np.testing.assert_array_equal(message[1], left[:7])
    np.testing.assert_array_equal(message[2], right[:7])
    np.testing.assert_array_equal(core.q, q)
    assert core._segment is None
    np.testing.assert_array_equal(core.smooth_left.position, left[:7])
    np.testing.assert_array_equal(core.smooth_right.position, right[:7])


def test_adapter_exposes_raw_grips_separately_from_tracking_enablement():
    core, _broadcast = make_core(hold=False)
    adapter = DaggerVRTeleop.__new__(DaggerVRTeleop)
    adapter._core = core
    adapter.set_hold_to_intervene(True)
    core.update_engage(frame(right=True))
    assert adapter.grips_held_raw and adapter.teleop_engaged
    adapter.force_disengage()
    assert adapter.grips_held_raw and not adapter.teleop_engaged
    core.update_engage(frame())
    assert not adapter.grips_held_raw
