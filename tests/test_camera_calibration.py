"""``axol calibration.cameras``: the file, the board, the solve and the capture."""

from __future__ import annotations

import json
from datetime import UTC, datetime
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from almond_axol.camera_calibration import file as calfile
from almond_axol.camera_calibration.board import Board, BoardConfig, Detection
from almond_axol.camera_calibration.optics import Lens
from almond_axol.camera_calibration.solve import CameraSpec, Limits, View, solve


def _pose(rot: np.ndarray, t) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = t
    return out


def _look_at(position: np.ndarray, target: np.ndarray, up) -> np.ndarray:
    z = target - position
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    return _pose(np.stack([x, np.cross(z, x), z], 1), position)


def _lens(fisheye: bool) -> Lens:
    intrinsics = SimpleNamespace(
        fx=700.0 if fisheye else 1050.0,
        fy=700.0 if fisheye else 1050.0,
        cx=960.0,
        cy=600.0,
        width=1920,
        height=1200,
        distortion_model="equidistant" if fisheye else "plumb_bob",
        d=(0.02, -0.01, 0.003, -0.001) if fisheye else (-0.05, 0.02, 5e-4, -3e-4, 0.0),
    )
    return Lens.from_intrinsics(intrinsics)


def _nudge(pose: np.ndarray, mm, deg) -> np.ndarray:
    return pose @ _pose(
        Rotation.from_euler("xyz", deg, degrees=True).as_matrix(), np.asarray(mm) / 1000
    )


class Rig:
    """A synthetic robot: one wrist camera, the head camera, a few board places."""

    def __init__(self, places: int = 4, per_place: int = 6, seed: int = 1) -> None:
        self.rng = np.random.default_rng(seed)
        self.board = Board()
        cad_wrist = _pose(
            Rotation.from_euler("xyz", [180, 0, 90], degrees=True).as_matrix(),
            [0.05, 0.0, -0.08],
        )
        cad_head = _look_at(
            np.array([0.1, 0.0, 0.45]), np.array([0.6, 0.0, -0.3]), (0, 0, 1)
        )
        self.true = {
            "left_wrist": _nudge(cad_wrist, [4, -3, 6], [1.0, -0.7, 0.5]),
            "head": _nudge(cad_head, [3, 5, -4], [0.8, 0.4, -0.6]),
        }
        self.specs = {
            "left_wrist": CameraSpec(
                "left_wrist",
                "Left wrist camera",
                "left_wrist_camera_optical",
                "left_w2",
                cad_wrist,
                _lens(True),
                serial=302271563,
            ),
            "head": CameraSpec(
                "head",
                "Head camera",
                "head_camera_left_optical",
                "root",
                cad_head,
                _lens(False),
                serial=51617969,
                moves=False,
            ),
        }
        spots = [
            (0, [0.5, 0.05, -0.3]),
            (25, [0.55, -0.15, -0.3]),
            (-20, [0.4, 0.25, -0.3]),
            (50, [0.7, -0.05, -0.3]),
        ]
        self.places = {
            k: _pose(Rotation.from_euler("z", yaw, degrees=True).as_matrix(), t)
            for k, (yaw, t) in enumerate(spots[:places])
        }
        self.views: list[View] = []
        for place in self.places:
            centre = self.places[place][:3, 3] + np.array([0.12, 0.08, 0.0])
            for _ in range(per_place):
                tilt = np.radians(self.rng.uniform(10, 40))
                az = self.rng.uniform(0, 2 * np.pi)
                dist = self.rng.uniform(0.25, 0.4)
                position = centre + dist * np.array(
                    [np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), np.cos(tilt)]
                )
                cam = _look_at(position, centre, (np.cos(az + 1), np.sin(az + 1), 0))
                cam = cam @ _pose(
                    Rotation.from_euler(
                        "z", self.rng.uniform(-40, 40), degrees=True
                    ).as_matrix(),
                    [0, 0, 0],
                )
                mount = cam @ np.linalg.inv(self.true["left_wrist"])
                for key, mount_pose in (("left_wrist", mount), ("head", np.eye(4))):
                    detection = self.observe(key, mount_pose, place)
                    if detection is not None:
                        self.views.append(View(key, place, mount_pose, detection))

    def observe(self, key: str, mount: np.ndarray, place: int, noise=0.3):
        spec = self.specs[key]
        cam_board = np.linalg.inv(mount @ self.true[key]) @ self.places[place]
        pts = self.board.points @ cam_board[:3, :3].T + cam_board[:3, 3]
        px = spec.lens.project(pts)
        inside = (
            (px[:, 0] > 0)
            & (px[:, 0] < 1920)
            & (px[:, 1] > 0)
            & (px[:, 1] < 1200)
            & (pts[:, 2] > 0)
        )
        ids = np.nonzero(inside)[0]
        if len(ids) < 6:
            return None
        return Detection(ids, px[ids] + self.rng.normal(0, noise, (len(ids), 2)))

    def error(self, key: str, pose: np.ndarray) -> tuple[float, float]:
        true = self.true[key]
        mm = float(np.linalg.norm(pose[:3, 3] - true[:3, 3])) * 1000
        r = pose[:3, :3].T @ true[:3, :3]
        deg = float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))
        return mm, deg


class FileTest(unittest.TestCase):
    def _doc(self, **overrides):
        doc = calfile.document(
            robot_serial="HUB1",
            calibration_id="cam-1",
            calibrated_at="2026-11-24T18:15:00Z",
            status="pass",
            transforms=[calfile.transform_entry("root", "cam", np.eye(4))],
        )
        doc.update(overrides)
        return doc

    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sub" / "camera_calibration.json"
            pose = _pose(
                Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix(),
                [1, 2, 3],
            )
            doc = self._doc(
                transforms=[calfile.transform_entry("a", "b", pose, camera_serial=7)]
            )
            calfile.write(doc, path)
            loaded = calfile.load(path)
        self.assertEqual(loaded.calibration_id, "cam-1")
        self.assertEqual(loaded.transforms[0].camera_serial, 7)
        np.testing.assert_allclose(loaded.transforms[0].matrix(), pose, atol=1e-12)
        before = datetime(2026, 12, 1, tzinfo=UTC)
        self.assertEqual(loaded.due_at, "2026-12-24T18:15:00Z")
        self.assertIsNone(loaded.problem_for("HUB1", now=before))
        self.assertIn("not this robot", loaded.problem_for("HUB2", now=before))
        self.assertIn("unknown", loaded.problem_for(None, now=before))
        after = datetime(2026, 12, 25, tzinfo=UTC)
        self.assertIn("expired on 2026-12-24", loaded.problem_for("HUB1", now=after))
        self.assertIsNone(calfile.load(Path(directory) / "missing.json"))

    def test_broken_files(self) -> None:
        path = Path("x.json")
        cases = {
            b"{": "not valid JSON",
            json.dumps(self._doc(calibration_id="")).encode(): "no calibration_id",
            json.dumps(self._doc(status="maybe")).encode(): "no status",
            json.dumps(self._doc(due_at="soon")).encode(): "due_at that is not",
            json.dumps(self._doc(transforms=[])).encode(): "no transforms",
            json.dumps(self._doc(version=99)).encode(): "version 99",
            json.dumps(self._doc(format="other")).encode(): "not a camera calibration",
        }
        bad_q = self._doc()
        bad_q["transforms"][0]["rotation_xyzw"] = [0, 0, 0, 2]
        cases[json.dumps(bad_q).encode()] = "not a unit quaternion"
        twice = self._doc()
        twice["transforms"] *= 2
        cases[json.dumps(twice).encode()] = "more than once"
        for raw, message in cases.items():
            with self.assertRaises(calfile.CalibrationFileError) as ctx:
                calfile.parse(raw, path)
            self.assertIn(message, str(ctx.exception))

    def test_failed_calibration_is_not_used(self) -> None:
        loaded = calfile.parse(
            json.dumps(self._doc(status="fail")).encode(), Path("x.json")
        )
        self.assertIn("failed calibration", loaded.problem_for("HUB1"))

    def test_accepts_a_file_without_format_fields(self) -> None:
        doc = self._doc()
        del doc["format"], doc["version"]
        loaded = calfile.parse(json.dumps(doc).encode(), Path("x.json"))
        self.assertEqual(loaded.status, "pass")


class BoardTest(unittest.TestCase):
    def test_render_and_detect(self) -> None:
        import cv2

        board = Board()
        page = board.render(dpi=150)
        detection = board.detect(cv2.cvtColor(page, cv2.COLOR_GRAY2BGR))
        self.assertIsNotNone(detection)
        self.assertEqual(len(detection), len(board.points))
        # Corner spacing in the image is the square size at this resolution.
        order = np.argsort(detection.ids)
        px = detection.pixels[order]
        self.assertAlmostEqual(
            float(np.linalg.norm(px[1] - px[0])), 30 / 25.4 * 150, delta=1.5
        )
        self.assertEqual(detection.markers, 54)
        self.assertTrue(board.usable(detection))
        self.assertIsNone(board.detect(np.full((400, 600), 255, np.uint8)))

    def test_pages(self) -> None:
        board = Board()
        with tempfile.TemporaryDirectory() as directory:
            pdf = board.save_page(Path(directory) / "board.pdf", dpi=100)
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
            # A3 landscape at actual size.
            self.assertIn(b"MediaBox [ 0 0 1190.88 841.68 ]", pdf.read_bytes())
            png = board.save_page(Path(directory) / "board.png", dpi=100)
            self.assertTrue(png.read_bytes().startswith(b"\x89PNG"))
        with self.assertRaises(ValueError):
            Board(BoardConfig(paper="letter")).render()
        small = Board(
            BoardConfig(
                squares_x=6, squares_y=4, square_mm=40, marker_mm=30, paper="letter"
            )
        )
        self.assertEqual(small.render(dpi=50).shape[1], round(279.4 * 50 / 25.4))

    def test_usable_needs_markers_and_corners_off_one_line(self) -> None:
        from almond_axol.camera_calibration.board import Detection

        board = Board()
        row = Detection(np.arange(11), np.zeros((11, 2)), markers=8)
        self.assertFalse(board.usable(row))  # one row of corners
        block = Detection(
            np.array([0, 1, 2, 3, 11, 12, 13, 14, 22, 23, 24, 25]),
            np.zeros((12, 2)),
            markers=6,
        )
        self.assertTrue(board.usable(block))
        self.assertFalse(board.usable(Detection(block.ids, block.pixels, markers=3)))


class OpticsTest(unittest.TestCase):
    def test_board_pose_from_corners(self) -> None:
        board = Board()
        for fisheye in (False, True):
            lens = _lens(fisheye)
            true = _pose(
                Rotation.from_euler("xyz", [170, 15, -10], degrees=True).as_matrix(),
                [-0.1, -0.05, 0.4],
            )
            pts = board.points @ true[:3, :3].T + true[:3, 3]
            px = lens.project(pts)
            np.testing.assert_allclose(
                lens.normalized(px), pts[:, :2] / pts[:, 2:], atol=1e-6
            )
            pose = lens.board_pose(board.points, px)
            np.testing.assert_allclose(pose, true, atol=1e-6)
        # Behind the camera projects far off the image.
        self.assertGreater(lens.project(np.array([[0.0, 0.0, -1.0]]))[0, 0], 1e3)


class SolveTest(unittest.TestCase):
    def test_recovers_the_mounts(self) -> None:
        rig = Rig()
        solution = solve(rig.specs, rig.views, rig.board)
        self.assertTrue(
            solution.passed, [r.problems for r in solution.cameras.values()]
        )
        wrist = rig.error("left_wrist", solution.cameras["left_wrist"].transform)
        head = rig.error("head", solution.cameras["head"].transform)
        self.assertLess(wrist[0], 0.5)
        self.assertLess(wrist[1], 0.1)
        self.assertLess(head[0], 2.0)
        self.assertLess(head[1], 0.15)
        self.assertLess(solution.cameras["left_wrist"].rms_px, 0.6)

    def test_rejects_a_bad_view(self) -> None:
        rig = Rig()
        i = next(i for i, v in enumerate(rig.views) if v.camera == "left_wrist")
        view = rig.views[i]
        rig.views[i] = View(
            view.camera,
            view.placement,
            view.mount_pose,
            Detection(view.detection.ids, view.detection.pixels + 20),
        )
        solution = solve(rig.specs, rig.views, rig.board)
        self.assertEqual(solution.cameras["left_wrist"].rejected, 1)
        self.assertTrue(np.isnan(solution.view_rms[i]))
        self.assertLess(
            rig.error("left_wrist", solution.cameras["left_wrist"].transform)[0], 0.5
        )

    def test_too_few_views_and_places_fail(self) -> None:
        rig = Rig(places=2, per_place=3)
        solution = solve(rig.specs, rig.views, rig.board)
        self.assertFalse(solution.passed)
        wrist = " ".join(solution.cameras["left_wrist"].problems)
        head = " ".join(solution.cameras["head"].problems)
        self.assertIn("capture at least 10", wrist)
        self.assertIn("at least 3 places", head)

    def test_far_from_cad_fails(self) -> None:
        rig = Rig()
        limits = Limits(max_change_mm=2.0)
        solution = solve(rig.specs, rig.views, rig.board, limits)
        self.assertIn("from CAD", " ".join(solution.cameras["left_wrist"].problems))

    def test_one_turning_axis_is_not_enough(self) -> None:
        rig = Rig()
        flat = []
        for k, view in enumerate(v for v in rig.views if v.camera == "left_wrist"):
            # Every view turned about the board normal only.
            cam = _look_at(
                np.array([0.6, 0.1, 0.0]), np.array([0.6, 0.1, -0.3]), (1, 0, 0)
            )
            cam = cam @ _pose(
                Rotation.from_euler("z", 8 * k, degrees=True).as_matrix(), [0, 0, 0]
            )
            mount = cam @ np.linalg.inv(rig.true["left_wrist"])
            detection = rig.observe("left_wrist", mount, view.placement)
            if detection is not None:
                flat.append(View("left_wrist", view.placement, mount, detection))
        solution = solve(rig.specs, flat, rig.board)
        self.assertIn(
            "turned too little", " ".join(solution.cameras["left_wrist"].problems)
        )


class SessionTest(unittest.TestCase):
    """The capture rules and the report, without a robot."""

    def _session(self, directory: Path):
        from almond_axol.camera_calibration.session import (
            LiveSession,
            SessionOptions,
            _Feed,
        )
        from almond_axol.constants import AxolModel, urdf_arm_joint_names, urdf_path
        from almond_axol.recording.mcap.urdf_tree import UrdfTree

        rig = Rig()
        tree = UrdfTree.load(urdf_path(AxolModel.MOBILE))
        feeds = [_Feed(spec, camera=None) for spec in rig.specs.values()]
        options = SessionOptions(reports_dir=directory, output=directory / "cal.json")
        session = LiveSession(
            options,
            rig.specs,
            feeds,
            tree,
            {
                side: urdf_arm_joint_names(is_left=side == "left")
                for side in ("left", "right")
            },
        )
        return rig, session

    def test_rigid_mount_and_specs(self) -> None:
        from almond_axol.camera_calibration.session import camera_specs, rigid_mount
        from almond_axol.constants import AxolModel, urdf_path
        from almond_axol.recording.mcap.urdf_tree import UrdfTree

        from .test_mcap_recording import FISHEYE_CONF, HEAD_CONF

        tree = UrdfTree.load(urdf_path(AxolModel.MOBILE))
        self.assertEqual(rigid_mount(tree, "left_wrist_camera_optical"), "left_w2")
        self.assertEqual(rigid_mount(tree, "head_camera_left_optical"), "root")
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "SN51617969.conf").write_text(HEAD_CONF)
            (Path(directory) / "SN302271563.conf").write_text(FISHEYE_CONF)
            specs, problems = camera_specs(
                tree,
                {"overhead": 51617969, "left_arm": 302271563, "right_arm": 302271564},
                {51617969},
                Path(directory),
            )
        self.assertEqual(set(specs), {"head", "left_wrist"})
        self.assertIn("SN302271564", " ".join(problems))
        self.assertFalse(specs["head"].moves)
        self.assertTrue(specs["left_wrist"].lens.fisheye)
        np.testing.assert_allclose(
            specs["left_wrist"].cad, tree.fk("left_w2", "left_wrist_camera_optical", {})
        )

    def test_views_are_taken_only_when_still_and_new(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rig, session = self._session(Path(directory))
            self.assertIsNone(session.command(""))  # board placed: capturing
            self.assertTrue(session.capturing)
            feed = session.feeds["left_wrist"]
            head = session.feeds["head"]
            detection = rig.views[0].detection
            head_detection = next(v for v in rig.views if v.camera == "head").detection
            frame = np.zeros((12, 16, 3), np.uint8)

            def show(q: np.ndarray, moving: bool = False) -> None:
                now = time.perf_counter()
                session._joints["left"].clear()
                for i in range(60):
                    t = now - 0.6 + i * 0.01
                    session._joints["left"].append(
                        (t, q + (0.01 * i if moving else 0.0))
                    )
                feed.frame, feed.perf, feed.detection = frame, now - 0.15, detection
                head.frame, head.perf, head.detection = (
                    frame,
                    now - 0.15,
                    head_detection,
                )
                session.check_views()

            q = np.array([0.1, 0.2, 0.3, -0.4, 0.5, 0.2, 0.1])
            show(q, moving=True)
            self.assertEqual(len(session.views), 0)
            show(q)
            self.assertEqual([v.camera for v in session.views], ["left_wrist", "head"])
            show(q + 0.001)  # the same view
            self.assertEqual(len(session.views), 2)
            show(q + np.array([0, 0, 0, 0, 0, 0.3, 0]))  # the wrist turned
            self.assertEqual(sum(v.camera == "left_wrist" for v in session.views), 2)
            # Moving the board pauses views.
            session.command("")
            self.assertFalse(session.capturing)
            show(q + np.array([0, 0, 0, 0, 0.5, 0, 0]))
            self.assertEqual(sum(v.camera == "left_wrist" for v in session.views), 2)
            self.assertEqual(session.command("done"), "done")
            session.flush_images()
            expected = session.tree.fk(
                "root",
                "left_w2",
                dict(zip(session.arm_joints["left"], q, strict=True)),
            )
            np.testing.assert_allclose(session.views[0].mount_pose, expected)

    def test_finish_writes_a_passing_file_and_report(self) -> None:
        from almond_axol.camera_calibration.session import (
            SessionOptions,
            finish,
            load_views,
            save_views,
        )

        rig = Rig()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_views(root / "views.json", rig.specs, rig.views, BoardConfig(), {})
            specs, views, board_config, images = load_views(root / "views.json")
            options = SessionOptions(
                output=root / "camera_calibration.json", robot_serial="HUB1"
            )
            solution, written = finish(
                specs, views, Board(board_config), images, options, root / "report"
            )
            self.assertTrue(solution.passed)
            loaded = calfile.load(written)
            self.assertEqual(loaded.robot_serial, "HUB1")
            self.assertTrue(loaded.calibration_id.startswith("cam-"))
            by_child = {t.child: t for t in loaded.transforms}
            self.assertEqual(by_child["head_camera_left_optical"].parent, "root")
            self.assertEqual(by_child["left_wrist_camera_optical"].parent, "left_w2")
            self.assertEqual(
                by_child["left_wrist_camera_optical"].camera_serial, 302271563
            )
            report = json.loads((root / "report" / "report.json").read_text())
            self.assertTrue(report["passed"])

            # A failing solve writes no file.
            options = SessionOptions(
                output=root / "other.json", limits=Limits(max_rms_px=0.01)
            )
            solution, written = finish(
                specs, views, Board(board_config), images, options, root / "report2"
            )
            self.assertFalse(solution.passed)
            self.assertIsNone(written)
            self.assertFalse((root / "other.json").exists())


class CliTest(unittest.TestCase):
    def test_print_board(self) -> None:
        from almond_axol.cli import calibration_cameras

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.png"
            calibration_cameras.main(
                [
                    "--print_board",
                    str(path),
                    "--board.paper",
                    "a4",
                    "--board.squares_x",
                    "8",
                    "--board.squares_y",
                    "6",
                ]
            )
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
