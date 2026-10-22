"""Mantis recording reads calibrated tracker poses at their capture time."""

from __future__ import annotations

import logging
import math
import unittest
from types import SimpleNamespace

import numpy as np

from almond_axol.cli.collect_data import _newer_pose, _row_observation
from almond_axol.lerobot.robot.robot_axol import AxolRobot
from almond_axol.mantis.relative import quat_xyzw_to_matrix
from almond_axol.teleop.absolute_pose import AbsoluteCalibration
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import TCPPoseSnapshot, VRTeleopCore, _split_calibration
from almond_axol.teleop.worker import IKWorker, _quat_xyzw_to_matrix
from almond_axol.vr.models import VRFrame, VRPose, VRPosition, VRQuaternion


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _quat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return np.array([*(axis * math.sin(angle / 2)), math.cos(angle / 2)])


def _engaged_worker() -> IKWorker:
    """An IK worker mid-session: left has a calibrated mount, right does not."""
    worker = object.__new__(IKWorker)
    worker._tcp_transforms = {
        "left": (np.array([0.0, 0.035, 0.047]), _rot_x(math.pi)),
    }
    worker._last_mapped_quat = {}
    worker._abs_base = (_rot_z(0.4), np.array([0.3, 0.9, -0.2]))
    worker._abs_offset = {
        "left": (np.zeros(3), np.eye(3)),
        "right": (np.array([0.01, -0.02, 0.05]), _rot_z(-0.3) @ _rot_x(0.2)),
    }
    worker._abs_active = True
    worker.abs_base_msg = {"pos": [0.3, 0.9, -0.2], "quat": [0.0, 0.0, 0.2, 0.98]}
    return worker


def _frame(t_ms: float | None = 1234.5, **overrides: object) -> VRFrame:
    values: dict[str, object] = {
        "l_ee": VRPose(
            position=VRPosition(x=0.21, y=1.02, z=-0.38),
            quaternion=VRQuaternion(
                **dict(zip("xyzw", _quat(np.array([0.2, 1.0, 0.1]), 0.7)))
            ),
        ),
        "r_ee": VRPose(
            position=VRPosition(x=-0.19, y=0.97, z=-0.41),
            quaternion=VRQuaternion(
                **dict(zip("xyzw", _quat(np.array([1.0, 0.1, -0.3]), -0.4)))
            ),
        ),
        "l_elbow": VRPosition(x=0.0, y=0.0, z=0.0),
        "r_elbow": VRPosition(x=0.0, y=0.0, z=0.0),
        "l_lock": True,
        "r_lock": True,
        "t": t_ms,
        "pose_source_kind": "tracker",
    }
    values.update(overrides)
    return VRFrame(**values)  # type: ignore[arg-type]


def _engaged_core(calibration: AbsoluteCalibration | None) -> VRTeleopCore:
    core = VRTeleopCore(
        VRTeleopConfig(absolute_mode=True),
        logging.getLogger(__name__),
        lambda _enabled: None,
    )
    core.left_enabled = core.right_enabled = True
    core.last_tracking = {"left": True, "right": True}
    core.abs_calibration = calibration
    return core


class AbsoluteCalibrationTest(unittest.TestCase):
    def test_matches_the_worker_mapping_of_a_raw_pose(self) -> None:
        worker = _engaged_worker()
        msg = worker.absolute_reply_base_msg()
        calibration = AbsoluteCalibration.from_message(msg["calibration"])
        self.assertIsNotNone(calibration)

        frame = _frame()
        for side, ee in (("left", frame.l_ee), ("right", frame.r_ee)):
            pos = np.array([ee.position.x, ee.position.y, ee.position.z])
            quat = np.array(
                [ee.quaternion.x, ee.quaternion.y, ee.quaternion.z, ee.quaternion.w]
            )
            # The worker's path, without its control-side pose filter.
            mapped_pos, mapped_quat = worker._apply_tcp_transform(side, pos, quat)
            expected_pos, expected_rot = worker._absolute_target(
                side,
                mapped_pos,
                _quat_xyzw_to_matrix(*mapped_quat).astype(np.float64),
            )

            pose = calibration.base_pose(side, pos, quat)

            np.testing.assert_allclose(pose[:3], expected_pos, atol=1e-5)
            np.testing.assert_allclose(
                quat_xyzw_to_matrix(np.array(pose[3:])), expected_rot, atol=1e-5
            )

    def test_round_trips_and_rejects_malformed_messages(self) -> None:
        msg = _engaged_worker().absolute_reply_base_msg()["calibration"]
        calibration = AbsoluteCalibration.from_message(msg)
        self.assertEqual(calibration.to_message(), msg)
        self.assertIsNone(calibration.tcp["right"])

        broken = {**msg, "base": {"pos": [0.0, 1.0], "rot": msg["base"]["rot"]}}
        self.assertIsNone(AbsoluteCalibration.from_message(broken))
        self.assertIsNone(AbsoluteCalibration.from_message({"base": msg["base"]}))
        self.assertIsNone(AbsoluteCalibration.from_message(None))

    def test_worker_reports_calibration_only_while_a_fit_is_active(self) -> None:
        worker = _engaged_worker()
        self.assertIn("calibration", worker.absolute_reply_base_msg())

        worker._abs_active = False
        self.assertIs(worker.absolute_reply_base_msg(), worker.abs_base_msg)

        worker._abs_active = True
        worker.abs_base_msg = {"rejected": "rigs swapped"}
        self.assertEqual(worker.absolute_reply_base_msg(), {"rejected": "rigs swapped"})

    def test_core_strips_calibration_before_the_urdf_broadcast(self) -> None:
        msg = _engaged_worker().absolute_reply_base_msg()

        base, calibration = _split_calibration(msg)

        self.assertEqual(set(base), {"pos", "quat"})
        self.assertIn("calibration", msg)  # the received message is not mutated
        self.assertIsInstance(calibration, AbsoluteCalibration)
        self.assertEqual(_split_calibration(None), (None, None))
        self.assertEqual(
            _split_calibration({"pos": [0.0] * 3}), ({"pos": [0.0] * 3}, None)
        )


class CalibratedPoseSnapshotTest(unittest.TestCase):
    def setUp(self) -> None:
        msg = _engaged_worker().absolute_reply_base_msg()["calibration"]
        self.calibration = AbsoluteCalibration.from_message(msg)

    def test_snapshot_carries_the_tracker_capture_time(self) -> None:
        core = _engaged_core(self.calibration)
        core.last_tcp_snapshot = TCPPoseSnapshot(
            pose_host_ts=1.0,
            left=(0.0,) * 7,
            right=(0.0,) * 7,
            out_of_reach=("right",),
        )
        frame = _frame(t_ms=98765.0)

        snapshot = core.calibrated_pose_snapshot(frame)

        self.assertEqual(snapshot.pose_host_ts, 98.765)
        pos = np.array(
            [frame.l_ee.position.x, frame.l_ee.position.y, frame.l_ee.position.z]
        )
        quat = np.array(
            [
                frame.l_ee.quaternion.x,
                frame.l_ee.quaternion.y,
                frame.l_ee.quaternion.z,
                frame.l_ee.quaternion.w,
            ]
        )
        self.assertEqual(snapshot.left, self.calibration.base_pose("left", pos, quat))
        # Reach is judged on the IK target; the verdict is carried over.
        self.assertEqual(snapshot.out_of_reach, ("right",))

    def test_falls_back_whenever_the_fit_or_the_frame_cannot_be_trusted(self) -> None:
        self.assertIsNone(_engaged_core(None).calibrated_pose_snapshot(_frame()))
        self.assertIsNone(
            _engaged_core(self.calibration).calibrated_pose_snapshot(None)
        )

        cases = {
            "quest frame": _frame(pose_source_kind="webxr"),
            "no capture time": _frame(t_ms=None),
            "left untracked": _frame(l_tracked=False),
            "right untracked": _frame(r_tracked=False),
        }
        for label, frame in cases.items():
            with self.subTest(label):
                core = _engaged_core(self.calibration)
                self.assertIsNone(core.calibrated_pose_snapshot(frame))

        disengaged = _engaged_core(self.calibration)
        disengaged.right_enabled = False
        self.assertIsNone(disengaged.calibrated_pose_snapshot(_frame()))

        datum_rejected = _engaged_core(self.calibration)
        datum_rejected.last_tracking = {"left": True, "right": False}
        self.assertIsNone(datum_rejected.calibrated_pose_snapshot(_frame()))


class NewerPoseTest(unittest.TestCase):
    @staticmethod
    def _snap(ts: float | None) -> TCPPoseSnapshot:
        return TCPPoseSnapshot(pose_host_ts=ts, left=(0.0,) * 7, right=(0.0,) * 7)

    def test_recorded_pose_timestamps_never_step_backwards(self) -> None:
        calibrated = self._snap(10.20)
        # Disengage: the IK snapshot is behind the calibrated tracker pose.
        self.assertFalse(_newer_pose(self._snap(10.05), calibrated))
        self.assertTrue(_newer_pose(self._snap(10.20), calibrated))
        self.assertTrue(_newer_pose(self._snap(10.30), calibrated))

    def test_first_and_untimed_samples(self) -> None:
        self.assertFalse(_newer_pose(None, self._snap(1.0)))
        self.assertTrue(_newer_pose(self._snap(1.0), None))
        self.assertTrue(_newer_pose(self._snap(1.0), self._snap(None)))
        self.assertFalse(_newer_pose(self._snap(None), self._snap(1.0)))


if __name__ == "__main__":
    unittest.main()


class _History:
    """A realtime core's timestamped telemetry with one gripper sample."""

    def __init__(self, ts: float) -> None:
        self.ts = ts
        self.calls: list[tuple[float, float]] = []

    def state_nearest(self, target: float, timeout: float = 0.1):  # type: ignore[no-untyped-def]
        self.calls.append((target, timeout))
        return (
            np.array([0.25]),
            np.array([0.75]),
            np.zeros(1),
            np.zeros(1),
            self.ts,
        )


def _robot(core: object) -> AxolRobot:
    robot = object.__new__(AxolRobot)
    robot.config = SimpleNamespace(observe_cartesian=False, observe_torques=False)
    robot._left_pos_keys = ["left_gripper.pos"]
    robot._right_pos_keys = ["right_gripper.pos"]
    robot._axol = core
    return robot


class RowObservationTest(unittest.TestCase):
    def test_gripper_state_is_taken_at_the_pose_capture(self) -> None:
        core = _History(ts=10.003)
        robot = _robot(core)

        obs = _row_observation(robot, {"left_gripper.pos": 9.0}, pose_ts=10.0)

        self.assertEqual(obs, {"left_gripper.pos": 0.25, "right_gripper.pos": 0.75})
        # Read at the pose's capture time, without waiting on the control loop.
        self.assertEqual(core.calls, [(10.0, 0.0)])

    def test_live_reading_is_kept_when_no_sample_is_close_enough(self) -> None:
        live = {"left_gripper.pos": 9.0}
        far = _robot(_History(ts=10.05))  # 50 ms from the capture
        self.assertIs(_row_observation(far, live, pose_ts=10.0), live)

        no_history = _robot(SimpleNamespace(state_nearest=lambda *_a, **_k: None))
        self.assertIs(_row_observation(no_history, live, pose_ts=10.0), live)

        no_core = _robot(object())
        self.assertIs(_row_observation(no_core, live, pose_ts=10.0), live)

        self.assertIs(_row_observation(_robot(_History(10.0)), live, None), live)
