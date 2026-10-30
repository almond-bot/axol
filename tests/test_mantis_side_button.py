"""Mantis side push button: decoding and episode control in the tracker bridge.

A short press starts a take when none is open (engaging first if needed)
and ends and saves it while recording. A 1 s hold ends and discards it.
"""

from __future__ import annotations

import queue
import struct
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from almond_axol.tracker.base import TrackerPose
from almond_axol.tracker.bridge import (
    _BUTTON_HOLD_S,
    SideButtonRecognizer,
    StopEventControls,
    TrackerBridge,
)
from almond_axol.tracker.trigger import (
    TRIGGER_CAN_ID,
    TriggerReader,
    encode_trigger_frame,
    parse_trigger_frame,
)
from almond_axol.vr.models import VREpisodeOutcome, VRState


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
    def __init__(self) -> None:
        self.value = 1.0
        self.pressed: bool | None = False
        self.stale = False

    def grip(self) -> float:
        return self.value

    def button(self) -> bool | None:
        return self.pressed

    def is_stale(self) -> bool:
        return self.stale


class _Clock:
    def __init__(self, start: float) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class _Bus:
    def __init__(self, payload: bytes) -> None:
        self.frames: queue.Queue[SimpleNamespace] = queue.Queue()
        self.frames.put(
            SimpleNamespace(
                arbitration_id=TRIGGER_CAN_ID, is_extended_id=False, data=payload
            )
        )

    def recv(self, timeout: float) -> SimpleNamespace | None:
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def shutdown(self) -> None:
        pass


class ButtonDecodeTest(unittest.TestCase):
    def test_seventh_byte_is_the_button(self) -> None:
        core = struct.pack("<fH", 0.25, 1234)
        self.assertIsNone(parse_trigger_frame(core).button)
        self.assertFalse(parse_trigger_frame(core + b"\x00").button)
        self.assertTrue(parse_trigger_frame(core + b"\x01").button)
        self.assertEqual(encode_trigger_frame(0.25, 1234), core)
        self.assertEqual(encode_trigger_frame(0.25, 1234, button=True), core + b"\x01")

    def test_reader_reports_button_and_none_without_one(self) -> None:
        for payload, expected in (
            (encode_trigger_frame(0.0, button=True), True),
            (encode_trigger_frame(0.0), None),
        ):
            with self.subTest(expected=expected):
                reader = TriggerReader("can-test", bus=_Bus(payload))
                try:
                    deadline = time.monotonic() + 1.0
                    while reader.grip() is None and time.monotonic() < deadline:
                        time.sleep(0.001)
                    self.assertIsNotNone(reader.grip())
                    self.assertIs(reader.button(), expected)
                finally:
                    reader.close()


class SideButtonRecognizerTest(unittest.TestCase):
    def test_short_and_long_presses(self) -> None:
        button = SideButtonRecognizer()
        # Held at startup: ignored until released.
        self.assertIsNone(button.update(True, 0.0))
        self.assertIsNone(button.update(True, 5.0))
        self.assertIsNone(button.update(False, 5.1))

        self.assertIsNone(button.update(True, 10.0))
        self.assertEqual(button.update(False, 10.2), "short")
        self.assertEqual(button.last_press_at, 10.0)

        self.assertIsNone(button.update(True, 20.0))
        self.assertEqual(button.update(True, 20.0 + _BUTTON_HOLD_S), "long")
        self.assertEqual(button.last_press_at, 20.0)
        # The release that follows a hold is not a short press.
        self.assertIsNone(button.update(True, 25.0))
        self.assertIsNone(button.update(False, 25.1))

    def test_cancel_requires_a_release(self) -> None:
        button = SideButtonRecognizer()
        button.update(False, 0.0)
        button.update(True, 1.0)
        button.cancel()
        self.assertIsNone(button.update(True, 1.1))
        self.assertIsNone(button.update(False, 1.2))
        button.update(True, 2.0)
        self.assertEqual(button.update(False, 2.1), "short")


class BridgeSideButtonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.left = _Trigger()
        self.right = _Trigger()
        self.bridge = TrackerBridge(
            _Source(),
            left="left",
            right="right",
            controls=StopEventControls(threading.Event()),
            left_trigger=self.left,
            right_trigger=self.right,
            auto_engage=True,
            confirm_auto_engage=True,
        )
        self.bridge._state = VRState.DATA_COLLECTION
        self.clock = _Clock(100.0)
        self.frame()  # see both buttons released

    def frame(self, dt: float = 0.01) -> dict:
        self.clock.now += dt
        with mock.patch("almond_axol.tracker.bridge.time.perf_counter", self.clock):
            frame = self.bridge.compose_frame()
        if "lock_release_id" in frame:
            self.bridge._handle_lock_release(frame["lock_release_id"])
        return frame

    def click(self, trigger: _Trigger) -> tuple[float, dict]:
        """Short press; return the press time and the release frame."""
        trigger.pressed = True
        self.frame()
        pressed_at = self.clock.now
        self.frame(0.1)
        trigger.pressed = False
        return pressed_at, self.frame()

    def start_take(self) -> None:
        _, requested = self.click(self.left)
        self.assertTrue(requested["l_lock"] and requested["r_lock"])
        self.assertEqual(requested["state"], VRState.DATA_COLLECTION.value)
        self.bridge._handle_tracking_state(True)
        self.assertEqual(self.frame()["state"], VRState.RECORDING.value)

    def test_press_engages_and_starts_then_press_saves(self) -> None:
        self.start_take()
        self.frame(1.0)  # task time

        pressed_at, saved = self.click(self.right)
        self.assertEqual(saved["state"], VRState.DATA_COLLECTION.value)
        self.assertEqual(saved["episode_outcome"], VREpisodeOutcome.SUCCESS.value)
        self.assertEqual(saved["episode_end_t_host"], pressed_at)
        self.assertFalse(saved["reset"])

    def test_press_while_engaged_opens_take_immediately(self) -> None:
        self.bridge._handle_tracking_state(True)
        _, opened = self.click(self.left)
        self.assertEqual(opened["state"], VRState.RECORDING.value)

    def test_buttons_held_through_a_start_do_not_end_the_take(self) -> None:
        self.bridge._handle_tracking_state(True)

        # Both rigs pressed together; the left releases first and starts.
        self.left.pressed = self.right.pressed = True
        self.frame()
        self.frame(0.1)
        self.left.pressed = False
        self.assertEqual(self.frame()["state"], VRState.RECORDING.value)
        self.right.pressed = False
        late = self.frame()
        self.assertEqual(late["state"], VRState.RECORDING.value)
        self.assertIsNone(late["episode_outcome"])

        # A button held through the trigger start gesture, even past the
        # discard hold time, is ignored once the take is open.
        self.bridge._state = VRState.DATA_COLLECTION
        self.right.pressed = True
        self.frame()
        self.left.value = self.right.value = 0.0
        self.frame()
        self.left.value = self.right.value = 1.0
        self.assertEqual(self.frame()["state"], VRState.RECORDING.value)
        held = self.frame(_BUTTON_HOLD_S + 0.1)
        self.assertEqual(held["state"], VRState.RECORDING.value)
        self.assertFalse(held["reset"])
        self.right.pressed = False
        self.assertEqual(self.frame()["state"], VRState.RECORDING.value)

    def test_hold_discards_and_its_release_does_not_restart(self) -> None:
        self.start_take()
        self.left.pressed = True
        self.frame()
        held = self.frame(0.5)
        self.assertEqual(held["state"], VRState.RECORDING.value)
        discarded = self.frame(_BUTTON_HOLD_S)
        self.assertEqual(discarded["state"], VRState.DATA_COLLECTION.value)
        self.assertTrue(discarded["reset"])
        self.assertIsNone(discarded["episode_outcome"])

        # Still held, then released: no new take or engage request.
        self.bridge._handle_tracking_state(False)
        self.frame(0.5)
        self.left.pressed = False
        for _ in range(_PULSE_SETTLE):
            released = self.frame()
        self.assertEqual(released["state"], VRState.DATA_COLLECTION.value)
        self.assertFalse(released["l_lock"] or released["r_lock"])

    def test_hold_while_idle_is_ignored(self) -> None:
        self.left.pressed = True
        self.frame()
        self.frame(_BUTTON_HOLD_S + 0.1)
        self.left.pressed = False
        idle = self.frame()
        self.assertEqual(idle["state"], VRState.DATA_COLLECTION.value)
        self.assertFalse(idle["l_lock"] or idle["r_lock"])

    def test_stale_trigger_node_cancels_a_press(self) -> None:
        self.bridge._handle_tracking_state(True)
        self.left.pressed = True
        self.frame()
        self.left.stale = True
        self.frame()
        self.left.stale = False
        self.bridge._handle_tracking_state(True)
        self.left.pressed = False
        self.assertEqual(self.frame()["state"], VRState.DATA_COLLECTION.value)

    def test_press_during_saving_cannot_start_next_take(self) -> None:
        self.bridge._handle_tracking_state(True)
        self.bridge._state = VRState.SAVING
        self.left.pressed = True
        self.frame()
        self.bridge._state = VRState.DATA_COLLECTION
        self.frame()
        self.left.pressed = False
        self.assertEqual(self.frame()["state"], VRState.DATA_COLLECTION.value)
        _, opened = self.click(self.left)
        self.assertEqual(opened["state"], VRState.RECORDING.value)

    def test_firmware_without_button_never_acts(self) -> None:
        self.bridge._handle_tracking_state(True)
        self.left.pressed = self.right.pressed = None
        for _ in range(5):
            frame = self.frame(0.5)
        self.assertEqual(frame["state"], VRState.DATA_COLLECTION.value)


# Frames for the discard's reset/outcome pulses to finish.
_PULSE_SETTLE = 12


if __name__ == "__main__":
    unittest.main()
