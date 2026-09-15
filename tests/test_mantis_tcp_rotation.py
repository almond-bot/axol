"""Mantis tracker→gripper rotation fix: constants, engage guard, provenance."""

from __future__ import annotations

import json
import logging
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from almond_axol.cli.collect_data import (
    EpisodeQAStats,
    _require_mantis_resume_transform,
)
from almond_axol.mantis.calibration import (
    DESIGN_TCP_TRANSFORM_ID,
    DESIGN_TCP_TRANSFORMS,
    LEGACY_VIVE_TCP_ROTATION_QUAT,
    MEASURED_TCP_TRANSFORM_ID,
    UNCALIBRATED_TCP_TRANSFORM_ID,
    VIVE_TCP_ROTATION_QUAT,
    tcp_transform_provenance,
)
from almond_axol.mantis.relative import quat_xyzw_to_matrix
from almond_axol.recording.cartesian_frame import (
    MANTIS_TCP_TRANSFORM_KEY,
    read_cartesian_frame_marker,
    update_cartesian_frame_marker,
    write_cartesian_frame_marker,
)
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import TCPPoseSnapshot, VRTeleopCore
from almond_axol.teleop.worker import IKWorker

# Rest-pose gripper orientation in the base frame (both arms): gripper +x is
# base +y, gripper +y points backwards (base -x), gripper -z (fingers) down.
_REST_ROT = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
_REST_LEFT = (np.array([-0.007, 0.2, 0.156]), _REST_ROT)
_REST_RIGHT = (np.array([-0.007, -0.2, 0.156]), _REST_ROT)


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _engage_worker(*, calibrated: bool = True) -> IKWorker:
    worker = object.__new__(IKWorker)
    worker._config = SimpleNamespace(base_height=None)
    worker._tcp_transforms = (
        {"left": (np.zeros(3), np.eye(3)), "right": (np.zeros(3), np.eye(3))}
        if calibrated
        else {}
    )
    worker._rest_fk_poses = lambda: (_REST_LEFT, _REST_RIGHT)  # type: ignore[method-assign]
    worker._abs_base = None
    worker._abs_offset = {}
    worker._abs_active = True
    worker.abs_base_msg = None
    return worker


# World frame is y-up. Left gripper at world x=-0.2, right at x=+0.2, both
# 1 m up: the fitted base then has +x = world -z, +y = world -x, +z = world +y.
_L_POS = np.array([-0.2, 1.0, 0.0])
_R_POS = np.array([0.2, 1.0, 0.0])
_R_WB = np.column_stack([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


class VivePrimaryConstantTest(unittest.TestCase):
    def test_factory_rotation_is_ry180_for_both_families_and_sides(self) -> None:
        expected = quat_xyzw_to_matrix(np.asarray(VIVE_TCP_ROTATION_QUAT))
        np.testing.assert_allclose(expected, np.diag([-1.0, 1.0, -1.0]), atol=1e-9)
        for family in ("survive", "ultimate"):
            for side in ("left", "right"):
                self.assertEqual(
                    DESIGN_TCP_TRANSFORMS[family][side][3:],
                    list(VIVE_TCP_ROTATION_QUAT),
                    (family, side),
                )
        # The retired constant is Rx(+90°), kept only for the migration.
        legacy = quat_xyzw_to_matrix(np.asarray(LEGACY_VIVE_TCP_ROTATION_QUAT))
        np.testing.assert_allclose(legacy, _rot_x(math.pi / 2), atol=1e-6)

    def test_rest_gripper_held_like_the_robot_maps_onto_rest_fk(self) -> None:
        # Under the corrected constant an operator holding the rig exactly as
        # the robot holds its rest gripper (fingers down, tracker face up)
        # records the rest FK orientation; under the retired constant the
        # same hold recorded a ~90° pitch error.
        correct = quat_xyzw_to_matrix(np.asarray(VIVE_TCP_ROTATION_QUAT))
        legacy = quat_xyzw_to_matrix(np.asarray(LEGACY_VIVE_TCP_ROTATION_QUAT))
        tracker_in_base = _REST_ROT @ correct.T  # the physical tracker pose
        np.testing.assert_allclose(tracker_in_base @ correct, _REST_ROT, atol=1e-9)
        residual = (tracker_in_base @ legacy).T @ _REST_ROT
        angle = math.degrees(math.acos((np.trace(residual) - 1.0) / 2.0))
        self.assertGreater(angle, 85.0)


class EngageSideSwapGuardTest(unittest.TestCase):
    def test_correct_hold_engages(self) -> None:
        worker = _engage_worker()
        rot = _R_WB @ _REST_ROT
        self.assertTrue(worker._engage_absolute(_L_POS, rot, _R_POS, rot))
        assert worker._abs_base is not None
        np.testing.assert_allclose(worker._abs_base[0], _R_WB, atol=1e-9)
        assert worker.abs_base_msg is not None
        self.assertNotIn("rejected", worker.abs_base_msg)
        self.assertIn("pos", worker.abs_base_msg)

    def test_both_rigs_facing_backwards_is_rejected(self) -> None:
        worker = _engage_worker()
        # Sides swapped: the fitted base faces backwards, so relative to it
        # both grippers appear yawed 180°.
        rot = _R_WB @ _rot_z(math.pi) @ _REST_ROT
        self.assertFalse(worker._engage_absolute(_L_POS, rot, _R_POS, rot))
        self.assertIsNone(worker._abs_base)
        self.assertFalse(worker._abs_active)
        assert worker.abs_base_msg is not None
        self.assertIn("RIGHT hand", worker.abs_base_msg["rejected"])

    def test_one_twisted_wrist_is_not_a_swap(self) -> None:
        worker = _engage_worker()
        good = _R_WB @ _REST_ROT
        twisted = _R_WB @ _rot_z(math.pi) @ _REST_ROT
        self.assertTrue(worker._engage_absolute(_L_POS, twisted, _R_POS, good))

    def test_rig_pointed_vertically_is_inconclusive(self) -> None:
        worker = _engage_worker()
        # Gripper +y (the heading axis) straight up: no horizontal heading.
        vertical = _R_WB @ _rot_y(math.pi / 2) @ _REST_ROT
        heading = (_R_WB.T @ vertical)[:, 1]
        self.assertGreater(abs(heading[2]), 0.99)
        self.assertTrue(worker._engage_absolute(_L_POS, vertical, _R_POS, vertical))

    def test_uncalibrated_sides_are_never_rejected(self) -> None:
        worker = _engage_worker(calibrated=False)
        rot = _R_WB @ _rot_z(math.pi) @ _REST_ROT
        self.assertTrue(worker._engage_absolute(_L_POS, rot, _R_POS, rot))

    def test_step_clears_rejection_once_locks_drop(self) -> None:
        worker = _engage_worker()
        worker.abs_base_msg = {"rejected": "swapped"}
        frame = SimpleNamespace(l_lock=False, r_lock=False)
        q = np.zeros(3, np.float32)
        self.assertIs(worker._step_absolute(frame, q), q)
        self.assertIsNone(worker.abs_base_msg)


class CoreEngageRejectionTest(unittest.TestCase):
    def test_rejection_disengages_and_requires_release(self) -> None:
        broadcasts: list[bool] = []
        core = VRTeleopCore(
            VRTeleopConfig(absolute_mode=True),
            logging.getLogger(__name__),
            broadcasts.append,
        )
        engaged = SimpleNamespace(
            l_tracked=True,
            r_tracked=True,
            l_lock=True,
            r_lock=True,
            l_grip=0.0,
            r_grip=0.0,
            lock_release_id=None,
        )
        released = SimpleNamespace(
            **{**vars(engaged), "l_lock": False, "r_lock": False}
        )
        core.update_engage(engaged)
        self.assertTrue(core.teleop_enabled)

        with self.assertLogs(core._logger, level="ERROR") as logs:
            self.assertTrue(core._handle_engage_rejection({"rejected": "swapped"}))
            # Same reason again: no repeated log line.
            self.assertTrue(core._handle_engage_rejection({"rejected": "swapped"}))
        self.assertEqual(len(logs.output), 1)
        self.assertFalse(core.teleop_enabled)
        # Forced-disengage gate: a held-over squeeze must not re-engage.
        self.assertFalse(core._accept_tracking_frame(engaged))
        self.assertTrue(core._accept_tracking_frame(released))
        core.update_engage(released)
        self.assertTrue(core._accept_tracking_frame(engaged))

        self.assertFalse(core._handle_engage_rejection({"pos": [0, 0, 0]}))
        self.assertFalse(core._handle_engage_rejection(None))

    def test_snapshot_carries_out_of_reach_sides(self) -> None:
        plain = TCPPoseSnapshot.from_message(
            {"left": [0.0] * 7, "right": [1.0] * 7}, 1.0
        )
        self.assertEqual(plain.out_of_reach, ())
        flagged = TCPPoseSnapshot.from_message(
            {"left": [0.0] * 7, "right": [1.0] * 7, "out_of_reach": ["right"]}, 1.0
        )
        self.assertEqual(flagged.out_of_reach, ("right",))
        stats = EpisodeQAStats(total_frames=200, out_of_reach_frames=50)
        self.assertAlmostEqual(stats.out_of_reach_fraction, 0.25)


class WorkerReachFlagTest(unittest.TestCase):
    def test_out_of_reach_lists_sides_past_the_soft_clamp(self) -> None:
        worker = object.__new__(IKWorker)
        worker._solver = SimpleNamespace(
            config=SimpleNamespace(reach_soft_start=0.78),
            shoulder_positions={
                "left": np.array([0.0, 0.15, 0.4]),
                "right": np.array([0.0, -0.15, 0.4]),
            },
        )
        near = (np.array([0.5, 0.15, 0.4]), np.eye(3))
        far = (np.array([0.9, -0.15, 0.4]), np.eye(3))
        self.assertEqual(worker._out_of_reach(near, far), ("right",))
        self.assertEqual(worker._out_of_reach(near, near), ())
        msg = IKWorker._encode_tcp_msg(near, far, out_of_reach=("right",))
        self.assertEqual(msg["out_of_reach"], ["right"])
        self.assertNotIn("out_of_reach", IKWorker._encode_tcp_msg(near, near))


class ProvenanceTest(unittest.TestCase):
    def test_provenance_ids(self) -> None:
        survive = DESIGN_TCP_TRANSFORMS["survive"]
        ultimate = DESIGN_TCP_TRANSFORMS["ultimate"]
        design = tcp_transform_provenance(
            survive["left"], ultimate["right"], source="lighthouse"
        )
        self.assertEqual(design["id"], DESIGN_TCP_TRANSFORM_ID)
        self.assertEqual(design["source"], "lighthouse")
        self.assertEqual(design["left"], survive["left"])
        # q and -q are the same rotation.
        negated = [*survive["left"][:3], *(-v for v in survive["left"][3:])]
        self.assertEqual(
            tcp_transform_provenance(negated, survive["right"], source=None)["id"],
            DESIGN_TCP_TRANSFORM_ID,
        )
        measured = [0.01, 0.0355, -0.092, *VIVE_TCP_ROTATION_QUAT]
        self.assertEqual(
            tcp_transform_provenance(measured, survive["right"], source="quest")["id"],
            MEASURED_TCP_TRANSFORM_ID,
        )
        self.assertEqual(
            tcp_transform_provenance(None, survive["right"], source=None)["id"],
            UNCALIBRATED_TCP_TRANSFORM_ID,
        )

    def test_marker_round_trip_and_update(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            provenance = tcp_transform_provenance(
                DESIGN_TCP_TRANSFORMS["survive"]["left"],
                DESIGN_TCP_TRANSFORMS["survive"]["right"],
                source="lighthouse",
            )
            write_cartesian_frame_marker(root, mantis_tcp_transform=provenance)
            marker = read_cartesian_frame_marker(root)
            assert marker is not None
            self.assertEqual(marker[MANTIS_TCP_TRANSFORM_KEY], provenance)
            self.assertIn("recorded_by_axol_version", marker)

            update_cartesian_frame_marker(
                root,
                migration={"id": "m1"},
                mantis_tcp_transform={"id": "x"},
            )
            update_cartesian_frame_marker(root, migration={"id": "m2"})
            updated = read_cartesian_frame_marker(root)
            assert updated is not None
            self.assertIn("recorded_by_axol_version", updated)
            self.assertEqual([m["id"] for m in updated["migrations"]], ["m1", "m2"])
            self.assertEqual(updated[MANTIS_TCP_TRANSFORM_KEY], {"id": "x"})
            self.assertIsNone(read_cartesian_frame_marker(root / "missing"))


def _collection(left: list[float] | None, right: list[float] | None, source: str):
    return SimpleNamespace(
        mantis_source=source,
        teleop_config=SimpleNamespace(
            vr_teleop_config=SimpleNamespace(
                tcp_transform_left=left, tcp_transform_right=right
            )
        ),
    )


class ResumeGateTest(unittest.TestCase):
    def _dataset(self, marker: dict | None) -> Path:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "meta").mkdir()
        if marker is not None:
            (root / "meta" / "axol.json").write_text(json.dumps(marker))
        return root

    def test_legacy_vive_dataset_is_refused(self) -> None:
        root = self._dataset({"cartesian_pose_frame": "flu-urdf-root-v0.1.32"})
        design = DESIGN_TCP_TRANSFORMS["survive"]
        with self.assertRaisesRegex(ValueError, "mantis-tcp-rotation"):
            _require_mantis_resume_transform(
                root, _collection(design["left"], design["right"], "lighthouse")
            )

    def test_legacy_quest_dataset_only_warns(self) -> None:
        root = self._dataset({"cartesian_pose_frame": "flu-urdf-root-v0.1.32"})
        measured = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        with self.assertLogs("almond_axol.cli.collect_data", level="WARNING"):
            _require_mantis_resume_transform(
                root, _collection(measured, measured, "quest")
            )

    def test_matching_design_or_measured_transforms_resume(self) -> None:
        design = DESIGN_TCP_TRANSFORMS["survive"]
        collection = _collection(design["left"], design["right"], "lighthouse")
        root = self._dataset(
            {
                "cartesian_pose_frame": "flu-urdf-root-v0.1.32",
                MANTIS_TCP_TRANSFORM_KEY: {
                    "id": DESIGN_TCP_TRANSFORM_ID,
                    "left": None,
                    "right": None,
                    "migrated": True,
                },
            }
        )
        _require_mantis_resume_transform(root, collection)

        measured = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        root = self._dataset(
            {
                "cartesian_pose_frame": "flu-urdf-root-v0.1.32",
                MANTIS_TCP_TRANSFORM_KEY: tcp_transform_provenance(
                    measured, measured, source="quest"
                ),
            }
        )
        _require_mantis_resume_transform(root, _collection(measured, measured, "quest"))
        other = [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        with self.assertRaisesRegex(ValueError, "mix two pose conventions"):
            _require_mantis_resume_transform(
                root, _collection(other, measured, "quest")
            )

    def test_no_marker_or_uncalibrated_passes(self) -> None:
        design = DESIGN_TCP_TRANSFORMS["survive"]
        _require_mantis_resume_transform(
            self._dataset(None),
            _collection(design["left"], design["right"], "lighthouse"),
        )
        root = self._dataset(
            {
                "cartesian_pose_frame": "flu-urdf-root-v0.1.32",
                MANTIS_TCP_TRANSFORM_KEY: {"id": UNCALIBRATED_TCP_TRANSFORM_ID},
            }
        )
        _require_mantis_resume_transform(
            root, _collection(design["left"], design["right"], "lighthouse")
        )


if __name__ == "__main__":
    unittest.main()
