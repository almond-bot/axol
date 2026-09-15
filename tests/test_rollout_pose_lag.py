from __future__ import annotations

import threading
import time
import unittest
from collections.abc import Callable
from types import SimpleNamespace
from unittest import mock

import numpy as np

from almond_axol.lerobot.robot import robot_axol
from almond_axol.lerobot.robot.robot_axol import AxolRobot
from almond_axol.lerobot.rollout import ActionPublisher, RolloutCaptureThread


def _robot_with_cameras(
    cameras: dict[str, object],
    *,
    state_offset_s: float | Callable[[], float] = 0.004,
    state_nearest: Callable[[float], object] | None = None,
) -> AxolRobot:
    """Bare AxolRobot over fake cameras and a fake Rust-core telemetry history.

    ``state_nearest`` mimics ``Axol.state_nearest``: the retained feedback
    sample closest to the requested exposure time, here ``state_offset_s``
    before it, so the expected pose lag is simply that offset.
    """
    robot = object.__new__(AxolRobot)
    robot.config = SimpleNamespace(observe_torques=False)  # type: ignore[assignment]
    zeros = np.zeros(1, dtype=np.float32)

    def default_state_nearest(ts: float):  # noqa: ANN202
        offset = state_offset_s() if callable(state_offset_s) else state_offset_s
        return zeros, zeros, zeros, zeros, ts - offset

    robot._axol = SimpleNamespace(  # type: ignore[assignment]
        state_nearest=mock.Mock(side_effect=state_nearest or default_state_nearest)
    )
    robot.cameras = cameras
    robot._joint_state_from_arrays = mock.Mock(  # type: ignore[method-assign]
        return_value={"joint": 1.25}
    )
    robot._joint_state = mock.Mock(  # type: ignore[method-assign]
        side_effect=AssertionError("policy joints must come from retained telemetry")
    )
    return robot


def _camera(
    fps: int,
    frame: int,
    cap_ts: float,
    recv_ts: float,
    *,
    history: bool = False,
) -> SimpleNamespace:
    """A policy camera double publishing one frame.

    ``history=True`` gives it the shared-memory reader's history surface
    (``latest_capture_ts`` / ``read_nearest``); otherwise it is an in-process
    reader exposing only ``read_latest_with_ts`` / ``read_at_or_after``.
    """
    image = np.array([frame], dtype=np.uint8)
    camera = SimpleNamespace(
        fps=fps,
        read_latest_with_ts=mock.Mock(return_value=(image, cap_ts, recv_ts)),
        read_at_or_after=mock.Mock(
            side_effect=AssertionError("fresh frames must not wait for the future")
        ),
    )
    if history:
        camera.latest_capture_ts = mock.Mock(return_value=(cap_ts, recv_ts))
        camera.read_nearest = mock.Mock(return_value=(image, cap_ts, recv_ts))
    return camera


class AxolObservationPoseLagTest(unittest.TestCase):
    """Policy observations pair exposures with the nearest core feedback sample.

    The joint state is never read "now": it is selected from the Rust core's
    retained telemetry nearest the cameras' median exposure, and the returned
    pose lag is that residual. Frames are the ones the cameras have *already*
    delivered — the observation never waits for an exposure after "now" (that
    wait, paid per camera, ran the 2026-09-14 DAgger loop at 3-8 Hz), and it
    is served again until a newer exposure arrives (the policy ring runs
    below the control rate). A camera
    without a fresh, timestamped frame or an unbracketed exposure aborts the
    observation instead of falling back; frames merely skewed across cameras
    (a ring that dropped an exposure) are served and the skew is reported,
    never turned into a skipped tick.
    """

    def test_timing_is_returned_out_of_band_using_median_exposure(self) -> None:
        earlier = _camera(50, 1, 100.012, 100.015)
        later = _camera(50, 2, 100.027, 100.030)
        robot = _robot_with_cameras({"earlier": earlier, "later": later})

        with mock.patch(
            "almond_axol.lerobot.robot.robot_axol.time.perf_counter",
            return_value=100.05,
        ):
            observation, pose_lag = robot.get_observation_with_pose_lag()

        self.assertNotIn("pose_lag", observation)
        self.assertEqual(observation["joint"], 1.25)
        np.testing.assert_array_equal(observation["earlier"], np.array([1]))
        np.testing.assert_array_equal(observation["later"], np.array([2]))
        self.assertAlmostEqual(pose_lag, 0.004)
        earlier.read_at_or_after.assert_not_called()
        later.read_at_or_after.assert_not_called()
        # Median of the two exposures selects the telemetry sample.
        robot._axol.state_nearest.assert_called_once()
        self.assertAlmostEqual(robot._axol.state_nearest.call_args.args[0], 100.0195)

    def test_capture_timestamp_api_returns_the_median_exposure(self) -> None:
        camera = _camera(60, 3, 250.010, 250.012)
        robot = _robot_with_cameras({"wrist": camera})

        with mock.patch(
            "almond_axol.lerobot.robot.robot_axol.time.perf_counter",
            return_value=250.02,
        ):
            observation, capture_ts = robot.get_observation_with_capture_timestamp()

        np.testing.assert_array_equal(observation["wrist"], np.array([3]))
        self.assertAlmostEqual(capture_ts, 250.010)

    def test_frames_anchor_on_the_slowest_pipeline_without_waiting(self) -> None:
        # The wrist pipeline is two frames ahead of the stereo overhead's; the
        # observation anchors on the overhead's newest exposure and takes the
        # wrist's retained frame nearest it, never waiting on either.
        overhead = _camera(60, 1, 300.0170, 300.0400, history=True)
        wrist = _camera(60, 2, 300.0503, 300.0550, history=True)
        wrist.read_nearest = mock.Mock(
            return_value=(np.array([4], dtype=np.uint8), 300.0168, 300.0210)
        )
        robot = _robot_with_cameras({"overhead": overhead, "wrist": wrist})

        with mock.patch(
            "almond_axol.lerobot.robot.robot_axol.time.perf_counter",
            return_value=300.06,
        ):
            observation, capture_ts = robot.get_observation_with_capture_timestamp()

        np.testing.assert_array_equal(observation["overhead"], np.array([1]))
        np.testing.assert_array_equal(observation["wrist"], np.array([4]))
        self.assertAlmostEqual(capture_ts, 300.0169)
        for camera in (overhead, wrist):
            # The lookup tolerance is the camera-silence limit (two periods +
            # 200 ms), not the alignment limit: a dropped exposure yields the
            # neighbouring frame rather than a failed observation.
            camera.read_nearest.assert_called_once()
            self.assertAlmostEqual(camera.read_nearest.call_args.args[0], 300.0170)
            self.assertAlmostEqual(
                camera.read_nearest.call_args.kwargs["tolerance_s"], 2 / 60 + 0.2
            )
            camera.read_at_or_after.assert_not_called()
            camera.read_latest_with_ts.assert_not_called()

    def test_the_frame_set_is_served_again_until_a_newer_exposure(self) -> None:
        # The policy ring runs below the control rate (policy_fps), so most
        # ticks find the slowest pipeline still at the exposure the last set
        # was built on: they get that set back — same frames, same joints,
        # same timestamps — without copying a frame or waiting for the next.
        overhead = _camera(20, 1, 300.0170, 300.0400, history=True)
        wrist = _camera(20, 2, 300.0503, 300.0550, history=True)
        wrist.read_nearest = mock.Mock(
            return_value=(np.array([2], dtype=np.uint8), 300.0168, 300.0210)
        )
        robot = _robot_with_cameras({"overhead": overhead, "wrist": wrist})
        clock = mock.patch(
            "almond_axol.lerobot.robot.robot_axol.time.perf_counter",
            return_value=300.06,
        )
        with clock:
            first, first_ts, first_state_ts = robot._get_synchronized_observation()
        self.assertAlmostEqual(first_ts, 300.0169)

        overhead.read_nearest.reset_mock()
        wrist.read_nearest.reset_mock()
        with mock.patch(
            "almond_axol.lerobot.robot.robot_axol.time.perf_counter",
            return_value=300.0767,  # one control tick later, no new exposure
        ):
            again, again_ts, again_state_ts = robot._get_synchronized_observation()

        self.assertAlmostEqual(again_ts, first_ts)
        self.assertAlmostEqual(again_state_ts, first_state_ts)
        self.assertEqual(again, first)
        self.assertIsNot(again, first)  # a fresh dict, the caller may mutate it
        overhead.read_nearest.assert_not_called()
        wrist.read_nearest.assert_not_called()
        overhead.read_at_or_after.assert_not_called()
        robot._axol.state_nearest.assert_called_once()

        # The overhead delivers its next ring frame (50 ms on): a new set.
        overhead.latest_capture_ts = mock.Mock(return_value=(300.0670, 300.0900))
        overhead.read_nearest = mock.Mock(
            return_value=(np.array([5], dtype=np.uint8), 300.0670, 300.0900)
        )
        wrist.read_nearest = mock.Mock(
            return_value=(np.array([6], dtype=np.uint8), 300.0668, 300.0710)
        )
        with mock.patch(
            "almond_axol.lerobot.robot.robot_axol.time.perf_counter",
            return_value=300.11,
        ):
            observation, second_ts = robot.get_observation_with_capture_timestamp()

        self.assertAlmostEqual(second_ts, 300.0669)
        np.testing.assert_array_equal(observation["overhead"], np.array([5]))
        np.testing.assert_array_equal(observation["wrist"], np.array([6]))
        self.assertEqual(robot._axol.state_nearest.call_count, 2)

    def test_a_camera_gone_silent_is_not_covered_by_the_cached_set(self) -> None:
        overhead = _camera(20, 1, 300.0170, 300.0400, history=True)
        wrist = _camera(20, 2, 300.0168, 300.0210, history=True)
        robot = _robot_with_cameras({"overhead": overhead, "wrist": wrist})
        with mock.patch(
            "almond_axol.lerobot.robot.robot_axol.time.perf_counter",
            return_value=300.06,
        ):
            robot.get_observation()

        # Two ring periods plus the slack later with no newer frame from the
        # overhead: silent, and the set built from it must not be re-served.
        with (
            mock.patch(
                "almond_axol.lerobot.robot.robot_axol.time.perf_counter",
                return_value=300.0400 + 2 / 20 + 0.2 + 0.001,
            ),
            self.assertRaisesRegex(RuntimeError, "'overhead' produced no fresh frame"),
        ):
            robot.get_observation()

    def test_history_without_the_anchor_exposure_falls_back_to_newest(self) -> None:
        # The wrist ring no longer reaches the overhead's newest exposure (it
        # is more than a ring ahead, or it dropped everything near it): the
        # observation still goes out, with the wrist's newest frame, and the
        # skew is reported instead of failing the tick — a failed tick sends
        # no action, which is what made the 2026-09-14 arms step.
        overhead = _camera(60, 1, 300.0170, 300.0400, history=True)
        wrist = _camera(60, 2, 300.2000, 300.2050, history=True)
        wrist.read_nearest = mock.Mock(
            side_effect=LookupError("nearest is 150.0ms away")
        )
        robot = _robot_with_cameras({"overhead": overhead, "wrist": wrist})

        with (
            mock.patch(
                "almond_axol.lerobot.robot.robot_axol.time.perf_counter",
                return_value=300.21,
            ),
            mock.patch.object(robot_axol._logger, "warning") as warning,
        ):
            observation, capture_ts = robot.get_observation_with_capture_timestamp()

        np.testing.assert_array_equal(observation["overhead"], np.array([1]))
        np.testing.assert_array_equal(observation["wrist"], np.array([2]))
        wrist.read_latest_with_ts.assert_called_once()
        self.assertAlmostEqual(capture_ts, (300.0170 + 300.2000) / 2)
        robot._axol.state_nearest.assert_called_once()
        monitor = robot._policy_skew_monitor()
        self.assertEqual(monitor._skewed, 1)
        self.assertEqual(monitor._cameras, {"wrist": 1})
        self.assertAlmostEqual(monitor._max_skew_s, 0.183)
        # The report is rate limited: nothing on the first observation.
        warning.assert_not_called()

    def test_skewed_frames_are_served_and_reported_once_per_window(self) -> None:
        # Two periods of skew (the wrist dropped the anchor exposure and its
        # neighbour) used to raise "not synchronized"; now the frames are
        # served and one warning summarises the window.
        overhead = _camera(60, 1, 300.0170, 300.0400, history=True)
        wrist = _camera(60, 2, 300.0503, 300.0550, history=True)
        wrist.read_nearest = mock.Mock(
            return_value=(np.array([4], dtype=np.uint8), 300.0503, 300.0550)
        )
        robot = _robot_with_cameras({"overhead": overhead, "wrist": wrist})

        with (
            mock.patch(
                "almond_axol.lerobot.robot.robot_axol.time.perf_counter",
                return_value=300.06,
            ),
            mock.patch.object(robot_axol._logger, "warning") as warning,
        ):
            observation, _ts = robot.get_observation_with_capture_timestamp()
            np.testing.assert_array_equal(observation["wrist"], np.array([4]))
            warning.assert_not_called()
            # Past the report interval the next observation emits the summary
            # (the window is timed on perf_counter, so age the window rather
            # than the clock — the frames must stay fresh).
            monitor = robot._policy_skew_monitor()
            monitor._window_start -= robot_axol._POLICY_SKEW_REPORT_INTERVAL_S
            robot._last_policy_observation = None  # fresh frame set again
            robot.get_observation_with_capture_timestamp()

        warning.assert_called_once()
        message = warning.call_args.args[0]
        self.assertIn("skewed in 2 of 2 observations", message)
        self.assertIn("max 33ms apart", message)
        self.assertIn("wrist x2", message)
        # The window resets after a report.
        self.assertEqual(robot._policy_skew_monitor()._skewed, 0)

    def test_missing_fresh_frame_is_fatal_instead_of_falling_back(self) -> None:
        silent = _camera(40, 7, 199.5, 199.6)
        empty = SimpleNamespace(
            fps=40,
            read_latest_with_ts=mock.Mock(
                side_effect=RuntimeError("shared-memory camera has no frames yet.")
            ),
            read_latest=mock.Mock(return_value=np.array([9], dtype=np.uint8)),
        )
        for label, camera in (("silent", silent), ("empty", empty)):
            robot = _robot_with_cameras({"wrist": camera})
            for read in (robot.get_observation_with_pose_lag, robot.get_observation):
                with (
                    self.subTest(camera=label, api=read.__name__),
                    mock.patch(
                        "almond_axol.lerobot.robot.robot_axol.time.perf_counter",
                        return_value=200.0,
                    ),
                    self.assertRaisesRegex(RuntimeError, "produced no fresh frame"),
                ):
                    read()
            robot._axol.state_nearest.assert_not_called()
        empty.read_latest.assert_not_called()

    def test_unbracketed_or_distant_telemetry_is_fatal(self) -> None:
        camera = _camera(30, 7, 499.0, 499.0)
        robot = _robot_with_cameras({"wrist": camera}, state_nearest=lambda _ts: None)
        with (
            mock.patch(
                "almond_axol.lerobot.robot.robot_axol.time.perf_counter",
                return_value=499.01,
            ),
            self.assertRaisesRegex(RuntimeError, "no retained robot telemetry"),
        ):
            robot.get_observation_with_pose_lag()

        far = robot_axol._POLICY_STATE_ALIGNMENT_LIMIT_S * 4
        robot = _robot_with_cameras({"wrist": camera}, state_offset_s=far)
        with (
            mock.patch(
                "almond_axol.lerobot.robot.robot_axol.time.perf_counter",
                return_value=499.01,
            ),
            self.assertRaisesRegex(RuntimeError, "too far from policy camera"),
        ):
            robot.get_observation_with_pose_lag()

    def test_concurrent_calls_keep_their_own_lag(self) -> None:
        barrier = threading.Barrier(2)
        expected = {"capture-a": 0.0011, "capture-b": 0.0043}

        class Camera:
            fps = 60

            @staticmethod
            def latest_capture_ts() -> tuple[float, float]:
                now = time.perf_counter()
                return now, now

            @staticmethod
            def read_nearest(target: float, tolerance_s: float):
                del tolerance_s
                barrier.wait(timeout=2.0)
                delta = expected[threading.current_thread().name]
                return np.array([1], dtype=np.uint8), target + delta, target + delta

        robot = _robot_with_cameras(
            {"wrist": Camera()},
            state_offset_s=lambda: expected[threading.current_thread().name],
        )
        results: dict[str, float] = {}
        errors: list[BaseException] = []

        def read() -> None:
            try:
                _observation, lag = robot.get_observation_with_pose_lag()
                results[threading.current_thread().name] = lag
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        threads = [
            threading.Thread(target=read, name=name)
            for name in ("capture-a", "capture-b")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3.0)

        self.assertFalse(errors)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(results.keys(), expected.keys())
        for name, lag in results.items():
            self.assertAlmostEqual(lag, expected[name])


class _OneRowDataset:
    def __init__(self) -> None:
        self.features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["joint"],
            },
            "observation.pose_lag": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["pose_lag"],
            },
            "action": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["motor"],
            },
        }
        self.rows: list[dict[str, object]] = []
        self.capture: RolloutCaptureThread | None = None

    def add_frame(self, row: dict[str, object]) -> None:
        self.rows.append(row)
        assert self.capture is not None
        self.capture.request_stop()


class RolloutCapturePoseLagTest(unittest.TestCase):
    def test_capture_records_lag_from_the_same_observation_call(self) -> None:
        robot = SimpleNamespace(
            cameras={},
            get_observation_with_pose_lag=mock.Mock(
                return_value=({"joint": 2.5}, 0.037)
            ),
            get_observation=mock.Mock(
                side_effect=AssertionError("timed dataset used plain observation")
            ),
        )
        publisher = ActionPublisher()
        publisher.publish({"motor": 3.5})
        dataset = _OneRowDataset()
        capture = RolloutCaptureThread(
            publisher=publisher,
            robot=robot,
            dataset=dataset,
            robot_obs_proc=lambda observation: {
                **observation,
                "pose_lag": 999.0,
            },
            fps=60,
            task="test",
            rerun_ip=None,
        )
        dataset.capture = capture

        capture.run()

        self.assertEqual(len(dataset.rows), 1)
        np.testing.assert_allclose(
            dataset.rows[0]["observation.pose_lag"],
            np.array([0.037], dtype=np.float32),
        )
        robot.get_observation_with_pose_lag.assert_called_once_with()
        robot.get_observation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
