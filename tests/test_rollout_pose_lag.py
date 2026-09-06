from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from almond_axol.lerobot.robot.robot_axol import AxolRobot
from almond_axol.lerobot.rollout import ActionPublisher, RolloutCaptureThread


def _robot_with_cameras(cameras: dict[str, object]) -> AxolRobot:
    robot = object.__new__(AxolRobot)
    robot._axol = object()  # type: ignore[assignment]  # connected sentinel
    robot.cameras = cameras
    robot._joint_state = mock.Mock(return_value={"joint": 1.25})  # type: ignore[method-assign]
    return robot


class AxolObservationPoseLagTest(unittest.TestCase):
    def test_timing_is_returned_out_of_band_using_freshest_camera(self) -> None:
        earlier = SimpleNamespace(
            fps=50,
            read_at_or_after=mock.Mock(
                return_value=(np.array([1], dtype=np.uint8), 100.012, 100.015)
            ),
        )
        later = SimpleNamespace(
            fps=50,
            read_at_or_after=mock.Mock(
                return_value=(np.array([2], dtype=np.uint8), 100.027, 100.030)
            ),
        )
        robot = _robot_with_cameras({"earlier": earlier, "later": later})

        with mock.patch(
            "almond_axol.lerobot.robot.robot_axol.time.perf_counter",
            return_value=100.0,
        ):
            observation, pose_lag = robot.get_observation_with_pose_lag()

        self.assertNotIn("pose_lag", observation)
        self.assertEqual(observation["joint"], 1.25)
        np.testing.assert_array_equal(observation["earlier"], np.array([1]))
        np.testing.assert_array_equal(observation["later"], np.array([2]))
        self.assertAlmostEqual(pose_lag, 0.027)
        earlier.read_at_or_after.assert_called_once_with(100.0, timeout_ms=240)
        later.read_at_or_after.assert_called_once_with(100.0, timeout_ms=240)

    def test_timeout_fallback_uses_latest_frame_capture_timestamp(self) -> None:
        camera = SimpleNamespace(
            fps=40,
            read_at_or_after=mock.Mock(side_effect=TimeoutError("late exposure")),
            read_latest_with_ts=mock.Mock(
                return_value=(np.array([7], dtype=np.uint8), 199.875, 199.900)
            ),
            read_latest=mock.Mock(side_effect=AssertionError("lost timestamp")),
        )
        robot = _robot_with_cameras({"wrist": camera})

        with mock.patch(
            "almond_axol.lerobot.robot.robot_axol.time.perf_counter",
            return_value=200.0,
        ):
            observation, pose_lag = robot.get_observation_with_pose_lag()

        np.testing.assert_array_equal(observation["wrist"], np.array([7]))
        self.assertAlmostEqual(pose_lag, -0.125)
        camera.read_latest_with_ts.assert_called_once_with()
        camera.read_latest.assert_not_called()

    def test_timed_fallback_retains_stale_frame_rejection(self) -> None:
        camera = SimpleNamespace(
            fps=30,
            read_at_or_after=mock.Mock(side_effect=TimeoutError("late exposure")),
            read_latest_with_ts=mock.Mock(
                return_value=(np.array([7], dtype=np.uint8), 499.0, 499.0)
            ),
        )
        robot = _robot_with_cameras({"wrist": camera})

        with (
            mock.patch(
                "almond_axol.lerobot.robot.robot_axol.time.perf_counter",
                return_value=500.0,
            ),
            self.assertRaisesRegex(TimeoutError, "frame is 1000ms old"),
        ):
            robot.get_observation_with_pose_lag()

    def test_concurrent_calls_keep_their_own_lag(self) -> None:
        barrier = threading.Barrier(2)
        expected = {"capture-a": 0.011, "capture-b": 0.043}

        class Camera:
            fps = 60

            @staticmethod
            def read_at_or_after(target: float, timeout_ms: int):
                del timeout_ms
                barrier.wait(timeout=2.0)
                delta = expected[threading.current_thread().name]
                return np.array([1], dtype=np.uint8), target + delta, target + delta

        robot = _robot_with_cameras({"wrist": Camera()})
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

    def test_timed_read_refuses_timestamp_less_fallback(self) -> None:
        camera = SimpleNamespace(
            fps=30,
            read_at_or_after=mock.Mock(side_effect=TimeoutError("late exposure")),
            read_latest=mock.Mock(return_value=np.array([9], dtype=np.uint8)),
        )
        robot = _robot_with_cameras({"wrist": camera})

        with (
            mock.patch(
                "almond_axol.lerobot.robot.robot_axol.time.perf_counter",
                return_value=300.0,
            ),
            self.assertRaisesRegex(RuntimeError, "refusing to fabricate pose_lag"),
        ):
            robot.get_observation_with_pose_lag()

        camera.read_latest.assert_not_called()

    def test_plain_observation_retains_timestamp_less_camera_compatibility(
        self,
    ) -> None:
        camera = SimpleNamespace(
            fps=30,
            read_at_or_after=mock.Mock(side_effect=TimeoutError("late exposure")),
            read_latest=mock.Mock(return_value=np.array([4], dtype=np.uint8)),
        )
        robot = _robot_with_cameras({"legacy": camera})

        with mock.patch(
            "almond_axol.lerobot.robot.robot_axol.time.perf_counter",
            return_value=400.0,
        ):
            observation = robot.get_observation()

        self.assertNotIn("pose_lag", observation)
        np.testing.assert_array_equal(observation["legacy"], np.array([4]))
        camera.read_latest.assert_called_once_with()


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
