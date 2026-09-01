from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from almond_axol.cli.collect_data import EpisodeQAStats, evaluate_episode_qa
from almond_axol.tracker.base import TrackerPose
from almond_axol.tracker.bridge import StopEventControls, TrackerBridge
from almond_axol.tracker.trigger import TriggerReader
from almond_axol.vr.models import VRFrame


class _Source:
    def poses(self) -> dict[str, TrackerPose]:
        now = time.perf_counter()
        return {
            side: TrackerPose(
                pos=np.array([x, 1.0, -0.4]),
                quat=np.array([0.0, 0.0, 0.0, 1.0]),
                t=now,
                tracking=True,
            )
            for side, x in (("left", 0.2), ("right", -0.2))
        }


class _Trigger:
    def __init__(self, grip: float = 1.0) -> None:
        self.value = grip
        self.stale = False

    def grip(self) -> float:
        return self.value

    def is_stale(self) -> bool:
        return self.stale


def _bridge() -> tuple[TrackerBridge, _Trigger, _Trigger]:
    left = _Trigger()
    right = _Trigger()
    bridge = TrackerBridge(
        _Source(),
        left="left",
        right="right",
        controls=StopEventControls(threading.Event()),
        left_trigger=left,
        right_trigger=right,
        auto_engage=True,
        confirm_auto_engage=True,
    )
    return bridge, left, right


class MantisTriggerSafetyTest(unittest.TestCase):
    def test_reader_constructor_interrupt_releases_socket(self) -> None:
        bus = Mock()
        thread = Mock()
        thread.start.side_effect = KeyboardInterrupt
        thread.is_alive.return_value = False

        with (
            patch(
                "almond_axol.tracker.trigger.threading.Thread",
                return_value=thread,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            TriggerReader("can-test", bus=bus)

        bus.shutdown.assert_called_once_with()
        thread.start.assert_called_once_with()

    def test_reader_close_surfaces_lingering_thread_after_socket_shutdown(
        self,
    ) -> None:
        reader = object.__new__(TriggerReader)
        reader._stop = threading.Event()
        reader._bus = Mock()
        reader._thread = Mock()
        reader._thread.is_alive.return_value = True

        with self.assertRaisesRegex(RuntimeError, "CAN ownership is uncertain"):
            reader.close()

        self.assertTrue(reader._stop.is_set())
        reader._bus.shutdown.assert_called_once_with()
        reader._thread.join.assert_called_once_with(timeout=1.0)

    def test_reader_close_joins_even_when_socket_shutdown_fails(self) -> None:
        reader = object.__new__(TriggerReader)
        reader._stop = threading.Event()
        reader._bus = Mock()
        reader._bus.shutdown.side_effect = OSError("shutdown failed")
        reader._thread = Mock()
        reader._thread.is_alive.return_value = False

        with self.assertRaisesRegex(RuntimeError, "CAN ownership is uncertain"):
            reader.close()

        reader._thread.join.assert_called_once_with(timeout=1.0)

    def test_stale_trigger_clears_armed_alignment_confirmation(self) -> None:
        bridge, left, right = _bridge()

        bridge.compose_frame()  # both released: confirmation is armed
        left.stale = True
        left.value = right.value = 0.0
        lost = bridge.compose_frame()
        self.assertFalse(lost["l_trigger_live"])
        self.assertTrue(lost["r_trigger_live"])
        self.assertFalse(lost["l_lock"])
        # The wire-format liveness fields survive VRFrame validation and are
        # therefore available to collection QA on the server side.
        parsed = VRFrame.model_validate(lost)
        self.assertFalse(parsed.l_trigger_live)

        # Recovering while still squeezed cannot reuse the release seen before
        # the dropout. A complete new release→squeeze is mandatory.
        left.stale = False
        self.assertFalse(bridge.compose_frame()["l_lock"])
        left.value = right.value = 1.0
        self.assertFalse(bridge.compose_frame()["l_lock"])
        left.value = right.value = 0.0
        self.assertTrue(bridge.compose_frame()["l_lock"])

    def test_mid_engagement_dropout_disengages_and_requires_realign(self) -> None:
        bridge, left, right = _bridge()
        bridge.compose_frame()
        left.value = right.value = 0.0
        self.assertTrue(bridge.compose_frame()["l_lock"])
        bridge._handle_tracking_state(True)

        # Complete the managed high→low handshake after the engage ack.
        released = bridge.compose_frame()
        self.assertFalse(released["l_lock"])
        bridge._handle_lock_release(released["lock_release_id"])

        left.stale = True
        lost = bridge.compose_frame()
        self.assertFalse(lost["l_trigger_live"])
        self.assertTrue(lost["l_lock"] and lost["r_lock"])
        bridge._handle_tracking_state(False)

        # Even after CAN recovers, held squeezes cannot re-engage. The bridge
        # first proves the core consumed a low lock level, then requires a new
        # physical release→squeeze at the alignment pose.
        left.stale = False
        low = bridge.compose_frame()
        self.assertFalse(low["l_lock"])
        bridge._handle_lock_release(low["lock_release_id"])
        self.assertFalse(bridge.compose_frame()["l_lock"])
        left.value = right.value = 1.0
        self.assertFalse(bridge.compose_frame()["l_lock"])
        left.value = right.value = 0.0
        self.assertTrue(bridge.compose_frame()["l_lock"])

    def test_recording_qa_rejects_any_declared_trigger_dropout(self) -> None:
        ok, reasons = evaluate_episode_qa(
            EpisodeQAStats(total_frames=100, trigger_loss_frames=1)
        )
        self.assertFalse(ok)
        self.assertTrue(any("trigger heartbeat" in reason for reason in reasons))

    def test_standalone_bridge_keeps_triggers_optional(self) -> None:
        bridge = TrackerBridge(
            _Source(),
            left="left",
            right="right",
            controls=StopEventControls(threading.Event()),
        )
        frame = bridge.compose_frame()
        self.assertTrue(frame["l_trigger_live"])
        self.assertTrue(frame["r_trigger_live"])
        self.assertFalse(frame["l_lock"])


if __name__ == "__main__":
    unittest.main()
