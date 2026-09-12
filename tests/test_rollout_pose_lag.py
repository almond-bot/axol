from __future__ import annotations

import threading
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


class AxolObservationPoseLagTest(unittest.TestCase):
    """Policy observations pair exposures with the nearest core feedback sample.

    The joint state is never read "now": it is selected from the Rust core's
    retained telemetry nearest the cameras' median exposure, and the returned
    pose lag is that residual. A camera without a fresh, timestamped frame or
    an unbracketed exposure aborts the observation instead of falling back.
    """

    def test_timing_is_returned_out_of_band_using_median_exposure(self) -> None:
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
        self.assertAlmostEqual(pose_lag, 0.004)
        earlier.read_at_or_after.assert_called_once_with(100.0, timeout_ms=240)
        later.read_at_or_after.assert_called_once_with(100.0, timeout_ms=240)
        # Median of the two exposures selects the telemetry sample.
        robot._axol.state_nearest.assert_called_once()
        self.assertAlmostEqual(robot._axol.state_nearest.call_args.args[0], 100.0195)

    def test_capture_timestamp_api_returns_the_median_exposure(self) -> None:
        camera = SimpleNamespace(
            fps=60,
            read_at_or_after=mock.Mock(
                return_value=(np.array([3], dtype=np.uint8), 250.010, 250.012)
            ),
        )
        robot = _robot_with_cameras({"wrist": camera})

        with mock.patch(
            "almond_axol.lerobot.robot.robot_axol.time.perf_counter",
            return_value=250.0,
        ):
            observation, capture_ts = robot.get_observation_with_capture_timestamp()

        np.testing.assert_array_equal(observation["wrist"], np.array([3]))
        self.assertAlmostEqual(capture_ts, 250.010)

    def test_missing_fresh_frame_is_fatal_instead_of_falling_back(self) -> None:
        camera = SimpleNamespace(
            fps=40,
            read_at_or_after=mock.Mock(side_effect=TimeoutError("late exposure")),
            read_latest_with_ts=mock.Mock(
                return_value=(np.array([7], dtype=np.uint8), 199.875, 199.900)
            ),
            read_latest=mock.Mock(return_value=np.array([9], dtype=np.uint8)),
        )
        robot = _robot_with_cameras({"wrist": camera})

        for read in (robot.get_observation_with_pose_lag, robot.get_observation):
            with (
                self.subTest(api=read.__name__),
                mock.patch(
                    "almond_axol.lerobot.robot.robot_axol.time.perf_counter",
                    return_value=200.0,
                ),
                self.assertRaisesRegex(RuntimeError, "produced no fresh frame"),
            ):
                read()

        camera.read_latest_with_ts.assert_not_called()
        camera.read_latest.assert_not_called()
        robot._axol.state_nearest.assert_not_called()

    def test_unbracketed_or_distant_telemetry_is_fatal(self) -> None:
        camera = SimpleNamespace(
            fps=30,
            read_at_or_after=mock.Mock(
                return_value=(np.array([7], dtype=np.uint8), 499.0, 499.0)
            ),
        )
        robot = _robot_with_cameras({"wrist": camera}, state_nearest=lambda _ts: None)
        with self.assertRaisesRegex(RuntimeError, "no retained robot telemetry"):
            robot.get_observation_with_pose_lag()

        far = robot_axol._POLICY_STATE_ALIGNMENT_LIMIT_S * 4
        robot = _robot_with_cameras({"wrist": camera}, state_offset_s=far)
        with self.assertRaisesRegex(RuntimeError, "too far from policy camera"):
            robot.get_observation_with_pose_lag()

    def test_concurrent_calls_keep_their_own_lag(self) -> None:
        barrier = threading.Barrier(2)
        expected = {"capture-a": 0.0011, "capture-b": 0.0043}

        class Camera:
            fps = 60

            @staticmethod
            def read_at_or_after(target: float, timeout_ms: int):
                del timeout_ms
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
