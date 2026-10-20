"""Camera recovery without GStreamer, a robot, or wall-clock waits."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from almond_axol.video import gst_zed


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.stopped = False
        self.stop_at = 100.0
        clock = patch.object(gst_zed.time, "perf_counter", lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        timeout = patch.object(gst_zed, "_READY_TIMEOUT_S", 0.5)
        timeout.start()
        self.addCleanup(timeout.stop)
        self.camera = gst_zed._GstPipelineBase()
        self.addCleanup(self.camera.disconnect)
        self.camera.serial = 123
        self.camera._want_raw = False
        self.camera._stop = SimpleNamespace(
            wait=self.wait, is_set=lambda: self.stopped, set=self.stop
        )
        self.channel = gst_zed._AUChannel(lambda: self.camera.alive)
        self.channel.broadcast([b"\x65initial"])
        self.queue = self.channel.subscribe()
        self.states = []
        self.deliver = True
        self.stop_on_null = False
        self.camera._gst = SimpleNamespace(
            State=SimpleNamespace(NULL="NULL", PLAYING="PLAYING"),
            StateChangeReturn=SimpleNamespace(FAILURE="FAILURE"),
        )
        self.camera._pipeline = SimpleNamespace(
            set_state=self.set_state,
            get_state=lambda _: (
                None,
                self.states[-1] if self.states else "PLAYING",
                None,
            ),
        )

    def wait(self, duration):
        self.now += duration
        if self.now >= self.stop_at:
            self.stop()
        return self.stopped

    def stop(self):
        self.stopped = True

    def set_state(self, state):
        self.states.append(state)
        if state == "NULL" and self.stop_on_null:
            self.stop()
        if state == "PLAYING" and self.deliver:
            self.channel.broadcast([b"\x65recovered"])
        return "SUCCESS"

    def test_healthy_feed_is_not_restarted(self):
        self.stop_at = 2.5
        self.camera._recover_stream((self.channel,))
        self.assertEqual(self.states, [])

    def test_stall_recovers_on_same_subscription(self):
        self.stop_at = 4.0
        self.camera._recover_stream((self.channel,))
        self.assertEqual(self.states, ["NULL", "PLAYING"])
        self.assertEqual(self.queue.get_nowait(), [b"\x65recovered"])
        self.assertGreater(self.channel.last_au, 0)

    def test_no_frames_exhausts_three_attempts(self):
        self.deliver = False
        self.camera._recover_stream((self.channel,))
        self.assertEqual(self.states, ["NULL", "PLAYING"] * 3)
        self.assertFalse(self.channel.first_au.is_set())

    def test_shutdown_during_reset_never_reopens(self):
        self.stop_on_null = True
        self.camera._recover_stream((self.channel,))
        self.assertEqual(self.states, ["NULL"])

    def test_recording_cameras_never_enable_recovery(self):
        self.camera._want_raw = True
        self.camera._start_stream_recovery((self.channel,))
        self.assertIsNone(self.camera._recovery_thread)

    def test_webrtc_source_stays_alive_during_reset(self):
        self.states.append("NULL")
        self.camera._recovery_thread = SimpleNamespace(
            is_alive=lambda: True, join=lambda **_: None
        )
        self.assertTrue(self.camera.alive)
        self.camera.disconnect()
        self.assertFalse(self.camera.alive)


if __name__ == "__main__":
    unittest.main()
