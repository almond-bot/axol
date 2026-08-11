"""Regression tests for VRTeleopCore's stale-stream and disconnect handling.

These cover the engage-state safety machinery added after field reports of
the arms jerking when re-entering VR while still engaged, and of quitting
the app leaving the arms mid-air:

- a stopped pose stream force-disengages after ``disengage_timeout`` and
  deactivates the IK worker (fresh engage snapshot on return — no jump);
- sustained silence (``exit_reset_timeout``) returns the arms to rest even
  when no disconnect is ever observed (a killed app whose TCP FIN was lost);
- the operator-gone disconnect notification returns to rest only when the
  arms are actually away from it.
"""

from __future__ import annotations

import logging
import time

from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore
from almond_axol.vr.models import VRFrame

from .conftest import make_frame

_FRAME = VRFrame.model_validate_json(make_frame(locks=False))
_ENGAGED = _FRAME.model_copy(update={"l_lock": True, "r_lock": True})


class FakeConn:
    """Stands in for the IK-worker pipe: records sends, echoes a canned reply."""

    def __init__(self) -> None:
        self.sent: list[object] = []
        self._pending: object | None = None

    def send(self, obj: object) -> None:
        self.sent.append(obj)
        self._pending = [0.0] * 4  # any array-like "solution"

    def poll(self, _timeout: float) -> bool:
        return self._pending is not None

    def recv(self) -> object:
        out, self._pending = self._pending, None
        return out


def make_core(broadcasts: list[bool], **config: float) -> tuple[VRTeleopCore, FakeConn]:
    core = VRTeleopCore(
        VRTeleopConfig(**config), logging.getLogger("core-test"), broadcasts.append
    )
    return core, FakeConn()


def engage(core: VRTeleopCore) -> None:
    core.note_frame_reset(False)
    core.update_engage(_FRAME)  # grips released
    core.update_engage(_ENGAGED)  # both-grips rising edge
    assert core.teleop_enabled


def test_stale_stream_disengages_and_unlocks_worker() -> None:
    broadcasts: list[bool] = []
    core, conn = make_core(broadcasts, disengage_timeout=0.2)
    engage(core)
    assert broadcasts == [True]

    # Fresh heartbeat: nothing happens.
    core._maybe_disengage_stale(conn, _ENGAGED, lambda: True)
    assert core.teleop_enabled and conn.sent == []

    # Stale heartbeat: disengage, broadcast, synthetic unlocked frame to the
    # IK worker so its next engaged frame re-snaps at the new controller pose.
    time.sleep(0.25)
    core._maybe_disengage_stale(conn, _ENGAGED, lambda: True)
    assert not core.teleop_enabled
    assert broadcasts == [True, False]
    assert len(conn.sent) == 1
    sent = conn.sent[0]
    assert sent.l_lock is False and sent.r_lock is False

    # Idempotent while disengaged.
    core._maybe_disengage_stale(conn, _ENGAGED, lambda: True)
    assert len(conn.sent) == 1 and broadcasts == [True, False]

    # A resumed stream re-engages normally.
    core.note_frame_reset(False)
    core.update_engage(_FRAME)
    core.update_engage(_ENGAGED)
    assert core.teleop_enabled and broadcasts == [True, False, True]


def test_disengage_timeout_zero_disables() -> None:
    broadcasts: list[bool] = []
    core, conn = make_core(broadcasts, disengage_timeout=0.0, exit_reset_timeout=0.0)
    engage(core)
    time.sleep(0.05)
    core._maybe_disengage_stale(conn, _ENGAGED, lambda: True)
    assert core.teleop_enabled and conn.sent == []


def test_sustained_silence_returns_to_rest() -> None:
    """The lost-FIN backstop: silence alone must send the arms home."""
    broadcasts: list[bool] = []
    core, conn = make_core(broadcasts, disengage_timeout=0.1, exit_reset_timeout=0.3)
    engage(core)  # engaging from rest marks the arms as away from rest

    time.sleep(0.15)
    core._maybe_disengage_stale(conn, _ENGAGED, lambda: True)
    assert not core.teleop_enabled and not core.is_resetting

    time.sleep(0.2)  # cross exit_reset_timeout
    core._maybe_disengage_stale(conn, _ENGAGED, lambda: True)
    assert core.is_resetting, "stale stream must latch a return-to-rest"


def test_silence_at_rest_never_moves_the_arms() -> None:
    broadcasts: list[bool] = []
    core, conn = make_core(broadcasts, disengage_timeout=0.1, exit_reset_timeout=0.2)
    core.note_frame_reset(False)  # heartbeat only; never engaged → at rest
    time.sleep(0.3)
    core._maybe_disengage_stale(conn, _FRAME, lambda: True)
    assert not core.is_resetting


def test_request_reset_if_away_gates_on_rest() -> None:
    broadcasts: list[bool] = []
    core, _conn = make_core(broadcasts)

    # At rest (never engaged): a disconnect must not move the robot.
    core.request_reset_if_away()
    assert not core.is_resetting

    # Away from rest (engaged at least once): the disconnect sends it home.
    engage(core)
    core.request_reset_if_away()
    assert core.is_resetting

    # Already resetting: idempotent.
    core.request_reset_if_away()
    assert core.is_resetting
