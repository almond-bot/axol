"""Recorder GC discipline: hold across the take, sweep only after the reply.

The recorder subprocess holds cyclic GC for each take (fc055ac). Unfrozen and
swept inside the command handler, each sweep traversed the whole lerobot/torch
heap and landed on the operator's path: ~0.7 s before ``started`` (the
pre-window sweep) and ~1.0 s before ``finished``. This pins the corrected
ordering: no pre-window sweep, the startup heap frozen once after ``ready``,
and the deferred sweep run after the take's reply has gone out.
"""

from __future__ import annotations

import unittest
from unittest import mock

from almond_axol.recording import record_proc


class _Events:
    """Interleaved record of GcHold calls and recorder replies."""

    def __init__(self) -> None:
        self.log: list[tuple] = []


class _FakeGcHold:
    def __init__(self, events: _Events) -> None:
        self._events = events
        self.held = False

    def begin(self, *, collect: bool = True) -> None:
        self._events.log.append(("gc.begin", collect))
        self.held = True

    def end(self) -> None:
        if self.held:
            self._events.log.append(("gc.end",))
        self.held = False


def _config() -> dict:
    return {
        "log_level": "INFO",
        "raw_meta": {
            "left": {
                "transport": "gstshm-h264",
                "socket_path": "/tmp/left.sock",
                "width": 640,
                "height": 480,
                "fps": 60,
                "pts_perf_offset_s": 0.0,
            }
        },
        "snapshot_shm_name": "snapshot-test",
        "obs_keys": [],
        "action_keys": [],
        "rerun_ip": None,
        "dataset_root": "/tmp/unused-recorder-test",
        "smooth_ee_hz": 0.0,
        "fps": 60,
    }


class RecorderGcHoldTest(unittest.TestCase):
    def _run(
        self, commands: list, *, repairs: list[dict] | None = None
    ) -> tuple[_Events, mock.Mock]:
        events = _Events()
        dataset = mock.Mock()
        dataset.num_episodes = 0

        def _save_episode() -> None:
            dataset.num_episodes += 1

        dataset.save_episode.side_effect = _save_episode
        conn = mock.Mock()
        conn.recv.side_effect = commands

        def _send(reply: tuple) -> None:
            events.log.append(("send", reply[0]))

        conn.send.side_effect = _send

        def capture_loop(**kwargs) -> None:
            kwargs["on_armed"]()
            if repairs:
                kwargs["repair_events"].extend(repairs)
            kwargs["stop_event"].wait(5.0)

        def freeze() -> int:
            events.log.append(("freeze",))
            return 1234

        with (
            mock.patch(
                "lerobot.processor.make_default_processors",
                return_value=(None, None, mock.Mock()),
            ),
            mock.patch(
                "almond_axol.video.shm_frames.EncodedAuReader",
                return_value=mock.Mock(),
            ),
            mock.patch(
                "almond_axol.video.shm_frames.SnapshotReader",
                return_value=mock.Mock(),
            ),
            mock.patch("almond_axol.utils.affinity.pin_background", return_value=True),
            mock.patch(
                "almond_axol.utils.stall_diag.GcHold",
                side_effect=lambda *a, **k: _FakeGcHold(events),
            ),
            mock.patch(
                "almond_axol.utils.stall_diag.freeze_startup_heap",
                side_effect=freeze,
            ),
            mock.patch(
                "almond_axol.utils.stall_diag.install_gc_pause_logger",
                return_value=lambda: None,
            ),
            mock.patch.object(record_proc, "install_encoded_dataset_encoder"),
            mock.patch.object(record_proc, "_open_dataset", return_value=dataset),
            mock.patch.object(
                record_proc, "_EpisodeVideoVerifier", return_value=mock.Mock()
            ),
            mock.patch.object(
                record_proc, "run_encoded_capture_loop", side_effect=capture_loop
            ),
            mock.patch.object(record_proc, "_log_capture_quality"),
            mock.patch.object(record_proc, "_cleanup_recorder_session"),
            mock.patch.object(record_proc, "_maybe_smooth_episode"),
            mock.patch.object(record_proc, "_prepare_streaming_episode"),
            mock.patch.object(record_proc, "make_episode_durable", return_value={}),
            mock.patch(
                "almond_axol.lerobot.nvenc_encoder.dropped_frames", return_value=0
            ),
        ):
            record_proc._recorder_main(conn, mock.Mock(), object(), _config())
        return events, dataset

    def test_repair_audit_line_numbers_the_episode_like_the_operator_sees_it(
        self,
    ) -> None:
        repair = {
            "camera": "overhead_left",
            "frame_index": 2292,
            "missing_frames": 1,
            "within_budget": True,
        }
        with self.assertLogs(record_proc._logger, level="WARNING") as captured:
            events, dataset = self._run(
                [("start_episode", "task"), ("save_episode",), None],
                repairs=[repair],
            )
        self.assertIn(("send", "saved"), events.log)
        self.assertEqual(dataset.num_episodes, 1)
        (line,) = [
            r.getMessage()
            for r in captured.records
            if "camera-gap repair" in r.getMessage()
        ]
        # "Saved episode 1" is what the control process announces for this
        # take (recorder.episode_count() after the save); match it.
        self.assertTrue(
            line.startswith("saved episode 1 with camera-gap repair(s): "), line
        )
        self.assertIn("overhead_left@row 2292=1 frame(s)", line)

    def test_startup_heap_is_frozen_once_after_ready(self) -> None:
        events, _ = self._run([None])
        self.assertEqual(events.log, [("send", "ready"), ("freeze",)])

    def test_take_holds_without_a_pre_sweep_and_sweeps_after_the_reply(self) -> None:
        events, _ = self._run(
            [
                ("start_episode", "task"),
                ("finish_episode", None),
                None,
            ]
        )
        self.assertEqual(
            events.log,
            [
                ("send", "ready"),
                ("freeze",),
                # Hold, no up-front sweep: the previous window's sweep already
                # ran, and a collection here sits on the record-start path.
                ("gc.begin", False),
                ("send", "started"),
                # The reply leaves first; the deferred sweep follows while the
                # control process is already returning the arms to rest.
                ("send", "finished"),
                ("gc.end",),
            ],
        )

    def test_sweep_runs_once_per_take_and_before_the_next_hold(self) -> None:
        events, _ = self._run(
            [
                ("start_episode", "a"),
                ("finish_episode", None),
                ("start_episode", "b"),
                ("finish_episode", None),
                None,
            ]
        )
        gc_events = [e for e in events.log if e[0].startswith("gc.")]
        self.assertEqual(
            gc_events,
            [("gc.begin", False), ("gc.end",), ("gc.begin", False), ("gc.end",)],
        )
        # Every sweep sits between a take's reply and the next command.
        for i, event in enumerate(events.log):
            if event == ("gc.end",):
                self.assertEqual(events.log[i - 1], ("send", "finished"))


if __name__ == "__main__":
    unittest.main()
