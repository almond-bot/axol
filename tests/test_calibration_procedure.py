"""The calibration procedure: generated poses, view measures, the fit, the checks."""

from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from almond_axol.camera_calibration.board import Board, Detection
from almond_axol.camera_calibration.optics import Lens
from almond_axol.camera_calibration.poses import (
    Coverage,
    distinct,
    look_at,
    tilt,
    tilted_views,
)


def _pose(rot: np.ndarray, t) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = t
    return out


def _lens(fisheye: bool) -> Lens:
    return Lens.from_intrinsics(
        SimpleNamespace(
            fx=456.0 if fisheye else 735.0,
            fy=456.0 if fisheye else 735.0,
            cx=960.0,
            cy=600.0,
            width=1920,
            height=1200,
            distortion_model="equidistant" if fisheye else "rational_polynomial",
            d=(0.068, 0.0047, -0.0022, -4e-5)
            if fisheye
            else (0.109, 0.063, -2e-4, -1.6e-4, -0.039, 0.117, 0.1135, -0.0526),
        )
    )


class PoseMeasuresTest(unittest.TestCase):
    def test_look_at_points_the_lens_at_the_target(self) -> None:
        pose = look_at(
            np.array([0.1, 0.2, 0.5]), np.zeros(3), np.array([1.0, 0, 0]), 30
        )
        r = pose[:3, :3]
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(r), 1.0)
        z = -pose[:3, 3] / np.linalg.norm(pose[:3, 3])
        np.testing.assert_allclose(r[:, 2], z, atol=1e-12)

    def test_tilt_and_distinct_poses(self) -> None:
        facing = _pose(np.eye(3), [0.0, 0.0, -0.3])  # camera on the board's -z side
        distance, angle, _direction = tilt(facing)
        self.assertAlmostEqual(distance, 0.3)
        self.assertAlmostEqual(angle, 0.0)
        tilted = _pose(
            Rotation.from_euler("x", 20, degrees=True).as_matrix(), [0, 0, -0.3]
        )
        self.assertAlmostEqual(tilt(tilted)[1], 20.0, places=6)
        close = _pose(np.eye(3), [0.01, 0.0, -0.3])
        far = _pose(np.eye(3), [0.06, 0.0, -0.3])
        self.assertEqual(distinct([facing, close, far]), [0, 2])

    def test_tilted_views_need_different_directions(self) -> None:
        def view(direction: float) -> np.ndarray:
            r = Rotation.from_euler("ZX", [direction, 20], degrees=True).as_matrix()
            return _pose(r, [0, 0, -0.3])

        self.assertEqual(tilted_views([view(0), view(5), view(90), view(180)]), 3)

    def test_coverage(self) -> None:
        cov = Coverage(1920, 1200)
        cov.add(np.array([[10.0, 10.0], [960.0, 600.0], [1900.0, 1190.0]]))
        self.assertEqual(cov.quadrants, 2)
        self.assertTrue(cov.centre and cov.edge)
        self.assertEqual(len(cov.cells), 3)


class PackageTest(unittest.TestCase):
    def test_ids_count_up_per_robot_and_day(self) -> None:
        from almond_axol.camera_calibration.package import next_id

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            when = datetime(2026, 12, 1, tzinfo=UTC)
            first = next_id("AXOL-0001", when, root)
            self.assertEqual(first, "AXOL-0001-20261201-01")
            (root / first).mkdir()
            (root / "AXOL-0001-20261201-07").mkdir()
            (root / "AXOL-0002-20261201-09").mkdir()
            self.assertEqual(next_id("AXOL-0001", when, root), "AXOL-0001-20261201-08")


class ProcedureTest(unittest.TestCase):
    """A recording made from exact projections: the fit finds the true
    mounts, and the checks see no error."""

    @classmethod
    def setUpClass(cls) -> None:
        from almond_axol.constants import AxolModel, urdf_path
        from almond_axol.recording.mcap.urdf_tree import UrdfTree
        from almond_axol.synthetic.episode import READY
        from almond_axol.synthetic.motion import ArmKinematics

        cls.board = Board()
        cls.tree = UrdfTree.load(urdf_path(AxolModel.MOBILE))
        cls.kin = ArmKinematics(cls.tree)
        cls.ready = READY
        cls.lenses = {
            "head_left": _lens(False),
            "head_right": _lens(False),
            "left_wrist": _lens(True),
            "right_wrist": _lens(True),
        }
        cad = {
            "head": cls.tree.fk("root", "head_camera_left_optical", {}),
            "left_wrist": cls.tree.fk("left_w2", "left_wrist_camera_optical", {}),
            "right_wrist": cls.tree.fk("right_w2", "right_wrist_camera_optical", {}),
        }

        def nudge(pose: np.ndarray, mm, deg) -> np.ndarray:
            r = Rotation.from_euler("xyz", deg, degrees=True).as_matrix()
            return pose @ _pose(r, np.asarray(mm) / 1000)

        cls.truth = {
            "head": nudge(cad["head"], (2.0, -1.0, 1.5), (0.3, -0.2, 0.1)),
            "left_wrist": nudge(cad["left_wrist"], (1.2, 0.8, -1.4), (-0.5, 0.3, 0.2)),
            "right_wrist": nudge(
                cad["right_wrist"], (-1.0, 1.2, 0.7), (0.4, 0.3, -0.4)
            ),
        }
        cls.stereo = _pose(
            Rotation.from_rotvec([0.002, 0.008, 0.001]).as_matrix(),
            [0.0498, -9e-5, 1.4e-4],
        )
        cls.rng = np.random.default_rng(3)
        cls.recording = cls._recording()

    @classmethod
    def _detect(
        cls, lens: Lens, camera: np.ndarray, board_pose: np.ndarray
    ) -> Detection | None:
        in_camera = np.linalg.inv(camera) @ board_pose
        pts = cls.board.points @ in_camera[:3, :3].T + in_camera[:3, 3]
        px = lens.project(pts)
        ok = (
            (pts[:, 2] > 0.05)
            & (px[:, 0] >= 0)
            & (px[:, 0] < 1920)
            & (px[:, 1] >= 0)
            & (px[:, 1] < 1200)
        )
        if ok.sum() < 20:
            return None
        return Detection(np.nonzero(ok)[0], px[ok], markers=20)

    @classmethod
    def _recording(cls):
        from almond_axol.camera_calibration.poses import head_board_poses, wrist_poses
        from almond_axol.camera_calibration.recording import Frame, Recording

        frames: dict[str, list] = {k: [] for k in cls.lenses}
        stamp = 0
        table_z = 0.417
        for placement, (x, y, yaw) in enumerate(
            ((0.56, 0.06, 8.0), (0.54, -0.07, -14.0)), start=1
        ):
            base = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
            r = Rotation.from_euler("z", yaw, degrees=True).as_matrix() @ base
            w, h = cls.board.size_m
            board = _pose(
                r, np.array([x, y, table_z]) - r @ np.array([w / 2, h / 2, 0.0])
            )
            for side in ("left", "right"):
                key = f"{side}_wrist"
                found = wrist_poses(
                    cls.kin,
                    side,
                    board_in_root=board,
                    board_size=cls.board.size_m,
                    points=cls.board.points,
                    lens=cls.lenses[key],
                    camera_in_gripper=cls.tree.fk(
                        f"{side}_gripper", f"{side}_wrist_camera_optical", {}
                    ),
                    table_z=table_z,
                    seed=dict(zip(cls.kin.joints[side], cls.ready[side], strict=True)),
                    count=6,
                    rng=cls.rng,
                    tries=80,
                )
                for pose in found:
                    stamp += 1
                    camera = (
                        cls.tree.fk("root", f"{side}_w2", pose.joints) @ cls.truth[key]
                    )
                    det = cls._detect(cls.lenses[key], camera, board)
                    if det is not None:
                        frames[key].append(
                            Frame(
                                key, stamp, det, dict(pose.joints), "table", placement
                            )
                        )
            stamp += 1
            for eye, camera in (
                ("head_left", cls.truth["head"]),
                ("head_right", cls.truth["head"] @ cls.stereo),
            ):
                det = cls._detect(cls.lenses[eye], camera, board)
                frames[eye].append(Frame(eye, stamp, det, {}, "table", placement))
        for board in head_board_poses(
            cls.lenses["head_left"],
            cls.truth["head"],
            board_size=cls.board.size_m,
            points=cls.board.points,
            count=8,
            rng=cls.rng,
            min_z=0.55,
        ):
            stamp += 1
            for eye, camera in (
                ("head_left", cls.truth["head"]),
                ("head_right", cls.truth["head"] @ cls.stereo),
            ):
                det = cls._detect(cls.lenses[eye], camera, board)
                if det is not None:
                    frames[eye].append(Frame(eye, stamp, det, {}, "head"))
        return Recording(
            path=Path("synthetic.mcap"),
            recording_id="r",
            sha256="0" * 64,
            metadata={},
            phases=[],
            lenses=cls.lenses,
            intrinsics={},
            frames=frames,
            stereo=cls.stereo,
        )

    def test_the_fit_finds_the_true_mounts(self) -> None:
        from almond_axol.camera_calibration.checks import change
        from almond_axol.camera_calibration.procedure import fit

        self.assertGreaterEqual(len(self.recording.frames["left_wrist"]), 8)
        mounts, _solution = fit(self.recording, self.board, self.tree)
        for key, true in self.truth.items():
            mm, deg = change(true, mounts[key])
            self.assertLess(mm, 0.05, key)
            self.assertLess(deg, 0.01, key)

    def test_the_checks_see_no_error_with_the_true_values(self) -> None:
        from almond_axol.camera_calibration.checks import (
            Rig,
            camera_errors,
            cross_camera,
            placements,
        )

        rig = Rig(self.tree, self.truth, self.stereo)
        places = placements(rig, self.recording, self.board)
        self.assertEqual(set(places), {1, 2})
        errors = camera_errors(rig, self.recording, self.board, places)
        for key, stats in errors.items():
            # Only the solvers' own tolerance.
            self.assertLess(stats["p95_px"], 0.05, key)
        cross = cross_camera(self.recording, self.board, self.stereo)
        self.assertGreater(cross["simultaneous_views"], 0)
        self.assertLess(cross["cross_p95_px"], 0.05)
        self.assertLess(cross["epipolar_p95_px"], 0.05)

    def test_a_wrong_mount_shows_in_the_errors(self) -> None:
        from almond_axol.camera_calibration.checks import Rig, camera_errors, placements

        wrong = dict(self.truth)
        wrong["head"] = self.truth["head"] @ _pose(
            Rotation.from_euler("x", 1.0, degrees=True).as_matrix(), [0, 0, 0]
        )
        rig = Rig(self.tree, wrong, self.stereo)
        errors = camera_errors(
            rig, self.recording, self.board, placements(rig, self.recording, self.board)
        )
        self.assertGreater(errors["head_left"]["p95_px"], 5.0)
        self.assertLess(errors["left_wrist"]["p95_px"], 0.05)


class JudgeTest(unittest.TestCase):
    def test_every_limit_is_judged(self) -> None:
        from almond_axol.camera_calibration.checks import ProcedureLimits, judge

        good = {
            "median_px": 0.2,
            "p95_px": 0.8,
            "within_3px": 1.0,
            "edge_p95_px": 1.0,
            "edge_corners": 50,
            "rms_px": 0.3,
        }
        views = {
            "distinct_poses": 22,
            "tilted_views": 6,
            "coverage_cells": 14,
            "quadrants": 4,
        }
        cameras = ["head_left", "left_wrist"]
        pair = {"simultaneous_views": 12, "cross_median_px": 0.4, "cross_p95_px": 1.2}
        mounts = {"head": np.eye(4)}
        outcome = judge(
            ProcedureLimits(),
            errors_b={c: good for c in cameras},
            counts_a={c: views for c in cameras},
            counts_b={c: views for c in cameras},
            cross_a=pair,
            cross_b=pair,
            mounts_a=mounts,
            mounts_b=mounts,
            cad=mounts,
            cameras=cameras,
        )
        self.assertTrue(
            outcome.passed, [c for c in outcome.checks if c.passed is False]
        )
        bad = judge(
            ProcedureLimits(),
            errors_b={c: {**good, "median_px": 0.7} for c in cameras},
            counts_a={c: views for c in cameras},
            counts_b={c: {**views, "distinct_poses": 9} for c in cameras},
            cross_a=pair,
            cross_b=pair,
            mounts_a=mounts,
            mounts_b={"head": _pose(np.eye(3), [0.004, 0, 0])},
            cad=mounts,
            cameras=cameras,
        )
        failed = {(c.name, c.camera) for c in bad.checks if c.passed is False}
        self.assertIn(("reprojection median", "head_left"), failed)
        self.assertIn(("distinct poses A / B", "left_wrist"), failed)
        self.assertIn(("repeatability: translation", "head pair"), failed)
        self.assertFalse(bad.passed)


if __name__ == "__main__":
    unittest.main()


class RobotSessionTest(unittest.TestCase):
    """The robot session's flow, with stand-ins for the hardware."""

    def test_a_recording_moves_through_its_phases(self) -> None:
        import asyncio
        import json
        from unittest import mock

        from almond_axol.camera_calibration import robot as robot_mod
        from almond_axol.camera_calibration.poses import WristPose
        from almond_axol.constants import AxolModel, urdf_path
        from almond_axol.recording.mcap.urdf_tree import UrdfTree
        from almond_axol.synthetic.episode import READY
        from almond_axol.synthetic.motion import ArmKinematics

        tree = UrdfTree.load(urdf_path(AxolModel.MOBILE))
        kin = ArmKinematics(tree)
        board = Board()
        sent: list[tuple[np.ndarray, np.ndarray]] = []

        class Arm:
            def __init__(self, q) -> None:
                self.positions = np.append(np.asarray(q, np.float32), 1.0)

        class Robot:
            left = Arm(READY["left"])
            right = Arm(READY["right"])

            async def motion_control(self, left=None, right=None) -> None:
                sent.append((left, right))

        solver = SimpleNamespace(
            num_joints=14, left_indices=np.arange(7), right_indices=np.arange(7, 14)
        )
        calls: list[str] = []
        recorder = SimpleNamespace(
            start_episode=lambda task: calls.append("start"),
            stop_capture=lambda: (calls.append("stop") or (40, None)),
            save_episode=lambda info: calls.append(("save", info)),
        )
        relay = SimpleNamespace(set_raw_enabled=lambda on: calls.append(("raw", on)))
        head_in_root = tree.fk("root", "head_camera_left_optical", {})
        on_table = _pose(np.diag([1.0, -1.0, -1.0]), [0.4, -0.1, 0.42])
        held = [
            _pose(
                Rotation.from_euler(
                    "XZ", [20 if i % 2 else 0, a], degrees=True
                ).as_matrix(),
                [0.02 * i, 0.01 * i, 0.5],
            )
            for i, a in enumerate(range(0, 360, 45))
        ]

        class Head:
            lens = _lens(False)
            last_pixels = np.array(
                [
                    [x, y]
                    for x in np.linspace(10, 1900, 8)
                    for y in np.linspace(10, 1190, 8)
                ]
            )
            calls = 0
            table = False

            def board_pose(self):
                if self.table:
                    return np.linalg.inv(head_in_root) @ on_table
                Head.calls += 1
                return held[min(Head.calls // 6, len(held) - 1)]

        head = Head()
        session = robot_mod.RobotSession(
            robot=Robot(),
            solver=solver,
            kin=kin,
            tree=tree,
            board=board,
            relay=relay,
            recorder=recorder,
            head=head,
            wrist_lens={"left": _lens(True), "right": _lens(True)},
            plan=robot_mod.RobotPlan(
                head_a=4, rate_hz=50.0, hold_s=0.02, head_timeout_s=30.0
            ),
            work=Path("."),
            say=lambda _m: None,
        )

        def linear(_solver, a, b, **_kw):
            return [a + s * (b - a) for s in np.linspace(0, 1, 5)]

        def poses(_kin, side, **kw):
            q = dict(zip(kin.joints[side], READY[side], strict=True))
            return [WristPose(q, np.eye(4), np.eye(4), np.zeros((0, 2)))]

        async def answer(_prompt: str) -> str:
            head.table = True
            return ""

        with (
            mock.patch(
                "almond_axol.teleop.trajectory.plan_collision_aware_trajectory", linear
            ),
            mock.patch.object(robot_mod, "wrist_poses", poses),
            mock.patch.object(robot_mod, "_ask", answer),
        ):
            head.table = False

            async def run() -> None:
                # The head phase asks first (Enter), then sees held poses.
                head.table = False
                original = robot_mod._ask

                async def first(prompt: str) -> str:
                    robot_mod._ask = answer
                    return ""

                robot_mod._ask = first
                try:
                    await session.record("A", 4, 1, 1)
                finally:
                    robot_mod._ask = original

            asyncio.run(run())
        self.assertEqual(calls[:2], ["start", ("raw", True)])
        self.assertIn("stop", calls)
        save = next(c for c in calls if isinstance(c, tuple) and c[0] == "save")
        phases = json.loads(save[1]["calibration_phases"])
        self.assertEqual([p["kind"] for p in phases], ["head", "table", "table"])
        self.assertEqual({p["arm"] for p in phases[1:]}, {"left", "right"})
        self.assertTrue(all(p["end_ns"] >= p["start_ns"] for p in phases))
        self.assertGreater(len(sent), 10)
        # The arms end raised, out of the head camera's way.
        last_left = sent[-1][0][:7]
        np.testing.assert_allclose(last_left, session.raised["left"], atol=1e-5)
