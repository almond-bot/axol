"""Synthetic data: lens rays, factory files, arm motion, the episode timeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from almond_axol.synthetic.motion import (
    ArmKinematics,
    Path as TcpPath,
    Waypoint,
    pose,
    rotation_exp,
    rotation_log,
)
from almond_axol.synthetic.render import LensSpec, lens_rays

HEAD = LensSpec(
    96,
    60,
    36.8,
    36.7,
    47.3,
    30.4,
    "rational_polynomial",
    (0.109, 0.063, -0.0002, -0.00016, -0.039, 0.117, 0.1135, -0.0526),
)
WRIST = LensSpec(
    96, 60, 22.8, 22.8, 48.2, 29.9, "equidistant", (0.0685, 0.0047, -0.0022, -4e-5)
)


class LensRaysTest(unittest.TestCase):
    def _project(self, lens: LensSpec, rays: np.ndarray) -> np.ndarray:
        import cv2

        k = np.array([[lens.fx, 0, lens.cx], [0, lens.fy, lens.cy], [0, 0, 1.0]])
        d = np.asarray(lens.d)
        pts = rays.reshape(-1, 1, 3)
        if lens.model == "equidistant":
            out, _ = cv2.fisheye.projectPoints(pts, np.zeros(3), np.zeros(3), k, d)
        else:
            out, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), k, d)
        return out.reshape(-1, 2)

    def test_each_pixel_ray_projects_back_to_its_pixel(self) -> None:
        for lens in (HEAD, WRIST):
            with self.subTest(model=lens.model):
                rays = lens_rays(lens)
                np.testing.assert_allclose(
                    np.linalg.norm(rays, axis=-1), 1.0, atol=1e-9
                )
                u, v = np.meshgrid(np.arange(lens.width), np.arange(lens.height))
                # OpenCV's fisheye model only projects points in front of the
                # lens; the wide wrist lens sees past 90 degrees.
                front = rays[..., 2] > 0.05
                got = self._project(lens, rays[front])
                want = np.stack([u[front], v[front]], axis=-1)
                np.testing.assert_allclose(got, want, atol=1e-4)

    def test_the_wrist_lens_sees_past_ninety_degrees(self) -> None:
        rays = lens_rays(WRIST)
        self.assertLess(float(rays[..., 2].min()), 0.0)


class ZedFilesTest(unittest.TestCase):
    def test_files_read_as_the_camera_kit_lenses(self) -> None:
        from almond_axol.recording.mcap.zed_conf import (
            camera_intrinsics,
            load_calibration,
            stereo_transform,
        )
        from almond_axol.synthetic.zed_files import (
            HEAD_SERIAL,
            LEFT_WRIST_SERIAL,
            write_files,
        )

        with tempfile.TemporaryDirectory() as directory:
            write_files(Path(directory))
            head = load_calibration(HEAD_SERIAL, Path(directory))
            wrist = load_calibration(LEFT_WRIST_SERIAL, Path(directory))
        left = camera_intrinsics(
            head,
            eye="left",
            capture_resolution="HD1200",
            width=1920,
            height=1200,
            fisheye=False,
        )
        self.assertEqual(left.distortion_model, "rational_polynomial")
        self.assertTrue(all(abs(v) > 0 for v in left.d))
        fisheye = camera_intrinsics(
            wrist,
            eye="left",
            capture_resolution="HD1200",
            width=1920,
            height=1200,
            fisheye=True,
        )
        self.assertEqual(fisheye.distortion_model, "equidistant")
        self.assertAlmostEqual(fisheye.fx, 456, delta=5)
        stereo = stereo_transform(head, "HD1200")
        self.assertAlmostEqual(stereo[0, 3], 0.0498, delta=0.001)


class MotionTest(unittest.TestCase):
    def test_rotation_log_and_exp_invert(self) -> None:
        for v in ([0.0, 0.0, 0.0], [0.3, -0.2, 0.1], [0.0, 3.0, 0.0]):
            r = rotation_exp(np.array(v))
            np.testing.assert_allclose(rotation_exp(rotation_log(r)), r, atol=1e-9)

    def test_ik_reaches_a_table_pose(self) -> None:
        from almond_axol.constants import AxolModel, urdf_path
        from almond_axol.recording.mcap.urdf_tree import UrdfTree
        from almond_axol.synthetic.episode import READY

        kin = ArmKinematics(UrdfTree.load(urdf_path(AxolModel.MOBILE)))
        for side in ("left", "right"):
            seed = dict(zip(kin.joints[side], READY[side], strict=True))
            target = kin.tcp(side, seed)
            target[:3, 3] += (0.04, 0.0, -0.08)
            q, err = kin.solve(side, target, seed, iterations=100)
            self.assertLess(err, 1e-4, side)
            np.testing.assert_allclose(
                kin.tcp(side, q)[:3, 3], target[:3, 3], atol=2e-4
            )
            lo, hi = kin.limits[side].T
            values = np.array([q[n] for n in kin.joints[side]])
            self.assertTrue(np.all(values >= lo) and np.all(values <= hi))

    def test_path_blends_between_waypoints(self) -> None:
        a = pose((0.0, 0.0, 0.0))
        b = pose((1.0, 0.0, 0.0), rotation_exp(np.array([0.0, 0.0, 1.0])))
        path = TcpPath([Waypoint(0.0, a, 1.0), Waypoint(2.0, b, 0.0)])
        mid, grip = path.at(1.0)
        self.assertAlmostEqual(mid[0, 3], 0.5)
        self.assertAlmostEqual(grip, 0.5)
        np.testing.assert_allclose(
            rotation_log(mid[:3, :3]), [0.0, 0.0, 0.5], atol=1e-9
        )
        self.assertIs(path.at(-1.0)[0], a)
        self.assertIs(path.at(5.0)[0], b)


class TimelineTest(unittest.TestCase):
    def test_the_arm_carries_the_cube_into_the_bin(self) -> None:
        from almond_axol.synthetic.episode import (
            EpisodeSpec,
            Rig,
            SyntheticOptions,
            build_timeline,
        )

        with tempfile.TemporaryDirectory() as directory:
            rig = Rig(Path(directory), SyntheticOptions(output=Path(directory)))
        timeline = build_timeline(rig, EpisodeSpec("t", "cube_to_bin", True), 13.0)
        cube = timeline.objects["cube"]
        start, end = cube[0][:3, 3], cube[-1][:3, 3]
        self.assertGreater(np.linalg.norm(end[:2] - start[:2]), 0.15)
        np.testing.assert_allclose(end[:2], (0.59, 0.0), atol=0.03)
        # Jelly drove forward 0.15 m, then parked.
        self.assertAlmostEqual(timeline.base[-1, 0], 0.15, places=6)
        self.assertAlmostEqual(float(np.abs(timeline.base_v[-1]).max()), 0.0)
        # The gripper holds the cube open by its width while it carries it.
        held = timeline.grip["right"][(timeline.t > 7.0) & (timeline.t < 9.0)]
        self.assertGreater(float(held.min()), 0.2)
        # The calibration places the cameras a few mm from CAD.
        self.assertEqual(rig.calibration_doc["status"], "pass")
        self.assertTrue(rig.calibration_doc["due_at"])
        self.assertEqual(len(rig.calibration_doc["transforms"]), 3)


if __name__ == "__main__":
    unittest.main()
