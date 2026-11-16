"""A late IK-worker reply must not be read as the answer to a later request.

The worker answers every request in order. When a plan outlasts the dispatch
loop's wait (an arm stow plans for several seconds), the reply still arrives
after the wait gave up. The loop has to drop it before it asks for anything
else, or a reset reads a frame solution and a frame reads a reset trajectory.
"""

from __future__ import annotations

import logging
import threading
import time
import unittest
from unittest import mock

import numpy as np

from almond_axol.teleop import core as core_mod
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore, _reply_tag
from almond_axol.vr.models import VRFrame, VRPose, VRPosition, VRQuaternion

_N = 16
_TIMEOUT = 0.02


def _frame() -> VRFrame:
    identity = VRQuaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    zero = VRPosition(x=0.0, y=0.0, z=0.0)
    return VRFrame(
        l_ee=VRPose(position=zero, quaternion=identity),
        r_ee=VRPose(position=zero, quaternion=identity),
        l_elbow=zero,
        r_elbow=zero,
        l_lock=False,
        r_lock=False,
    )


def _core() -> VRTeleopCore:
    core = VRTeleopCore(
        VRTeleopConfig(ik_frequency=1000.0),
        logging.getLogger("test_ik_reply_resync"),
        broadcast_tracking=lambda _enabled: None,
    )
    core.set_target(np.zeros(_N, np.float32))
    return core


class _Worker:
    """Pipe stub that answers in order, the first frame only after a delay.

    A frame's solution carries the frame's number in ``q[0]``, so a test can
    tell which request a reply answered.
    """

    def __init__(self, hold_s: float, on_first_frame=None) -> None:
        self.frames = 0
        self.resets = 0
        self.sent_while_owed = 0
        self._replies: list[tuple[float, object]] = []
        self._hold_s = hold_s
        self._on_first_frame = on_first_frame
        self.first_reply_due: float | None = None

    def send(self, msg) -> None:
        now = time.monotonic()
        if isinstance(msg, VRFrame):
            self.frames += 1
            if self.first_reply_due is not None and now < self.first_reply_due:
                self.sent_while_owed += 1
            q = np.full(_N, float(self.frames), np.float32)
            due = now
            if self.frames == 1:
                due = self.first_reply_due = now + self._hold_s
                if self._on_first_frame is not None:
                    self._on_first_frame()
            self._replies.append((due, (q, None)))
        elif isinstance(msg, tuple) and msg[0] == "reset":
            self.resets += 1
            if self.first_reply_due is not None and now < self.first_reply_due:
                self.sent_while_owed += 1
            rest = np.full(_N, -1.0, np.float32)
            self._replies.append((now, ("reset_traj", rest, [])))

    def owe(self, reply: object) -> None:
        """Queue a reply to a request that was sent outside the frame path."""
        self.first_reply_due = time.monotonic() + self._hold_s
        self._replies.append((self.first_reply_due, reply))

    def poll(self, timeout=None) -> bool:
        # The worker is single-threaded: a reply is never ready before the
        # one ahead of it.
        if self._replies and time.monotonic() >= self._replies[0][0]:
            return True
        if timeout:
            time.sleep(min(timeout, 0.001))
        return False

    def recv(self):
        return self._replies.pop(0)[1]


class LateReplyTest(unittest.TestCase):
    def setUp(self) -> None:
        patches = (
            mock.patch.object(core_mod, "_IK_RECV_TIMEOUT", _TIMEOUT),
            mock.patch.object(core_mod, "_IK_SETTLE_POLL", 0.005),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_frames_after_a_late_reply_get_their_own_solution(self) -> None:
        core = _core()
        conn = _Worker(hold_s=4 * _TIMEOUT)
        stop = threading.Event()
        applied: list[tuple[int, float]] = []
        set_target = core.set_target

        def record(q):
            applied.append((conn.frames, float(np.asarray(q)[0])))
            if len(applied) >= 5:
                stop.set()
            return set_target(q)

        core.set_target = record
        with self.assertLogs("test_ik_reply_resync", "WARNING") as logs:
            core.run_ik_loop(conn, _frame, stop, lambda: True, lambda _t: None)
        self.assertTrue(any("IK recv timeout" in line for line in logs.output))
        # Nothing was asked of the worker while it still owed the first reply.
        self.assertEqual(conn.sent_while_owed, 0)
        # The late reply (frame 1) was dropped; every applied solution is the
        # answer to the frame just sent.
        self.assertEqual(len(applied), 5)
        for sent, solved in applied:
            self.assertEqual(solved, float(sent))
        self.assertNotIn(1.0, [solved for _sent, solved in applied])
        self.assertEqual(core._ik_owed, 0)

    def test_reset_pressed_during_the_wait_adopts_its_own_plan(self) -> None:
        core = _core()
        # The operator presses reset while the worker is still planning.
        conn = _Worker(hold_s=4 * _TIMEOUT, on_first_frame=core.request_reset)
        stop = threading.Event()
        pace = core._pace

        def paced(t0, interval):
            if conn.resets:
                stop.set()
            pace(t0, interval)

        core._pace = paced
        core.run_ik_loop(conn, _frame, stop, lambda: True, lambda _t: None)
        self.assertEqual(conn.sent_while_owed, 0)
        self.assertEqual(conn.resets, 1)
        self.assertIsNone(core.reset_error)
        # The reset's own reply was adopted, not the late frame solution.
        np.testing.assert_array_equal(core.q, np.full(_N, -1.0, np.float32))

    def test_a_timeout_in_the_engage_step_holds_that_frame_back(self) -> None:
        # The DAgger sync runs inside the engage step and swallows its own
        # timeout, so the loop reaches the frame dispatch in the same pass.
        core = _core()
        conn = _Worker(hold_s=4 * _TIMEOUT)
        stop = threading.Event()
        applied: list[tuple[int, float]] = []
        set_target = core.set_target
        update_engage = core.update_engage
        synced = []

        def engage(frame):
            if not synced:
                synced.append(True)
                conn.owe(("synced", np.zeros(_N, np.float32)))
                core._note_ik_timeout(_TIMEOUT)
            update_engage(frame)

        def record(q):
            applied.append((conn.frames, float(np.asarray(q)[0])))
            if len(applied) >= 3:
                stop.set()
            return set_target(q)

        core.update_engage = engage
        core.set_target = record
        with self.assertNoLogs("test_ik_reply_resync", "ERROR"):
            core.run_ik_loop(conn, _frame, stop, lambda: True, lambda _t: None)
        self.assertEqual(conn.sent_while_owed, 0)
        for sent, solved in applied:
            self.assertEqual(solved, float(sent))

    def test_a_dead_worker_owes_nothing(self) -> None:
        core = _core()
        core._note_ik_timeout(_TIMEOUT)
        conn = _Worker(hold_s=0.0)
        self.assertTrue(core._settle_ik_replies(conn, threading.Event(), lambda: False))
        self.assertEqual(core._ik_owed, 0)

    def test_a_tagged_reply_is_not_taken_for_a_frame_solution(self) -> None:
        core = _core()

        class _Conn:
            def send(self, msg) -> None:
                pass

            def poll(self, _timeout=None) -> bool:
                return True

            def recv(self):
                stop.set()
                return ("reset_traj", np.ones(_N, np.float32), [])

        stop = threading.Event()
        with self.assertLogs("test_ik_reply_resync", "ERROR") as logs:
            core.run_ik_loop(_Conn(), _frame, stop, lambda: True, lambda _t: None)
        self.assertTrue(any("IK dispatch error" in line for line in logs.output))
        np.testing.assert_array_equal(core.q, np.zeros(_N, np.float32))
        self.assertIsNone(core.pair_status)


class ReplyTagTest(unittest.TestCase):
    def test_only_a_leading_string_is_a_tag(self) -> None:
        self.assertEqual(_reply_tag(("reset_traj", None, [])), "reset_traj")
        self.assertIsNone(_reply_tag((np.zeros(_N), None)))
        self.assertIsNone(_reply_tag(np.zeros(_N)))
        self.assertIsNone(_reply_tag(None))
        self.assertIsNone(_reply_tag(()))


if __name__ == "__main__":
    unittest.main()
