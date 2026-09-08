"""Tests for the ZED-tracked wheel-radius calibration (diag.base-calibrate).

The fit and the frame geometry are pure functions, so they are exercised
against a synthetic cart with known radii, camera mount and wheelbase; the
camera and CAN paths are not touched here.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from almond_axol.diagnostics.base.calibrate import (
    Stroke,
    StrokePlan,
    arc_rows,
    camera_heading_axes,
    consistency_report,
    fit_calibration,
    fit_robust,
    heading_change,
    make_plan,
    read_strokes,
    wheel_scale_arg,
    write_strokes,
)
from almond_axol.robot.cart import mix, stroke_rows

TRUE_RADII = [0.0503, 0.0496, 0.0509, 0.0499]
TRUE_CAMERA = (0.31, -0.02)
TRUE_YAW = math.radians(1.5)
TRUE_LEVER = 0.42


def _rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def synth_strokes(
    plan: list[StrokePlan],
    radii: list[float] = TRUE_RADII,
    camera: tuple[float, float] = TRUE_CAMERA,
    yaw: float = TRUE_YAW,
    lever: float = TRUE_LEVER,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[Stroke]:
    """What the tool would record on a cart with these true parameters.

    The wheels turn exactly as commanded (nominal-radius mix); the body moves
    by the least-squares kinematic map of their true surface travel along a
    circular arc (constant body velocity and yaw rate), so the camera sees the
    chord; it additionally swings on its lever and reports in its own yawed
    frame. ``noise`` adds Gaussian error (m and rad) to the camera readings.
    """
    r = np.array(radii)
    strokes = []
    for p in plan:
        speeds = mix(*p.command, 20.0, 1.0)
        kx, ky, kw = stroke_rows(speeds)
        vx, vy, w = kx @ r, ky @ r, (kw @ r) / lever
        duration = p.target / (abs(w) if p.rotation else math.hypot(vx, vy))
        turns = [s * duration for s in speeds]
        kx, ky, kw = stroke_rows(turns)
        straight = np.array([kx @ r, ky @ r])
        theta = (kw @ r) / lever
        half = 0.5 * theta
        sinc = math.sin(half) / half if abs(half) > 1e-12 else 1.0
        d = _rz(half) @ straight * sinc  # chord of the arc
        d_cam_body = d + (_rz(theta) - np.eye(2)) @ np.array(camera)
        d_cam = _rz(-yaw) @ d_cam_body
        if noise and rng is not None:
            d_cam = d_cam + rng.normal(0.0, noise, 2)
            theta += rng.normal(0.0, noise)
        strokes.append(
            Stroke(p.name, turns, float(d_cam[0]), float(d_cam[1]), float(theta))
        )
    return strokes


def true_scale(radii: list[float] = TRUE_RADII) -> np.ndarray:
    scale = np.mean(radii) / np.array(radii)
    return scale / scale.mean()


class FitCalibrationTests(unittest.TestCase):
    def test_recovers_radii_camera_mount_and_lever_exactly(self) -> None:
        plan = make_plan(1.5, math.radians(90), 0.25, 0.2, 1)
        cal = fit_calibration(synth_strokes(plan))
        np.testing.assert_allclose(cal.radii_m, TRUE_RADII, rtol=1e-9)
        np.testing.assert_allclose(cal.camera_offset_m, TRUE_CAMERA, atol=1e-9)
        self.assertAlmostEqual(cal.camera_yaw_rad, TRUE_YAW, places=9)
        self.assertAlmostEqual(cal.lever_m, TRUE_LEVER, places=9)
        np.testing.assert_allclose(cal.wheel_scale, true_scale(), rtol=1e-9)
        self.assertLess(cal.rms_translation_m, 1e-12)
        self.assertFalse(cal.suspect)

    def test_known_lever_is_honoured(self) -> None:
        plan = make_plan(1.5, math.radians(90), 0.25, 0.2, 1)
        cal = fit_calibration(synth_strokes(plan), lever_m=TRUE_LEVER)
        self.assertEqual(cal.lever_m, TRUE_LEVER)
        np.testing.assert_allclose(cal.radii_m, TRUE_RADII, rtol=1e-9)

    def test_camera_yaw_is_not_mistaken_for_radius_error(self) -> None:
        # Identical wheels, camera mounted 3° off: the scales must stay at 1.
        plan = make_plan(1.5, math.radians(90), 0.25, 0.2, 1)
        strokes = synth_strokes(plan, radii=[0.05] * 4, yaw=math.radians(3.0))
        cal = fit_calibration(strokes)
        np.testing.assert_allclose(cal.wheel_scale, [1.0] * 4, atol=1e-9)
        self.assertAlmostEqual(math.degrees(cal.camera_yaw_rad), 3.0, places=6)

    def test_tracking_noise_leaves_scale_error_far_below_the_spread(self) -> None:
        rng = np.random.default_rng(1)
        plan = make_plan(1.5, math.radians(90), 0.25, 0.2, 2)
        worst = 0.0
        for _ in range(20):
            cal = fit_calibration(synth_strokes(plan, noise=0.002, rng=rng))
            worst = max(
                worst, float(np.abs(np.array(cal.wheel_scale) - true_scale()).max())
            )
        # The true spread is 2.6%; 2 mm / 0.1° of camera noise per stroke must
        # not move a scale by more than a few tenths of a percent.
        self.assertLess(worst, 0.004)

    def test_slip_shows_up_as_residual(self) -> None:
        plan = make_plan(1.5, math.radians(90), 0.25, 0.2, 1)
        strokes = synth_strokes(plan)
        strokes[2].dy_m += 0.15  # forward stroke slid 15 cm sideways
        cal = fit_calibration(strokes)
        self.assertTrue(cal.suspect)
        self.assertGreater(cal.rms_translation_m, 0.02 * cal.stroke_length_m)
        # With a single repetition the residual splits evenly between the
        # forward and back strokes, so the robust fit can only drop one of
        # the pair; a second repetition makes the culprit unambiguous.
        self.assertIn(fit_robust(strokes).dropped, ([2], [3]))
        strokes = synth_strokes(make_plan(1.5, math.radians(90), 0.25, 0.2, 2))
        strokes[2].dy_m += 0.15
        self.assertEqual(fit_robust(strokes).dropped, [2])

    def test_rejects_strokes_that_do_not_determine_the_mount(self) -> None:
        plan = [
            p for p in make_plan(1.5, math.radians(90), 0.25, 0.2, 1) if not p.rotation
        ]
        with self.assertRaises(ValueError):
            fit_calibration(synth_strokes(plan))
        with self.assertRaises(ValueError):
            fit_calibration(synth_strokes(plan[:2]))
        with self.assertRaises(ValueError):
            fit_calibration(
                synth_strokes(make_plan(1.5, 1.0, 0.25, 0.2, 1)), lever_m=0.0
            )


class ArcRowsTests(unittest.TestCase):
    def test_zero_turn_is_plain_kinematics(self) -> None:
        turns = [10.0, -10.0, 10.0, -10.0]
        self.assertEqual(arc_rows(turns, 0.0), stroke_rows(turns))

    def test_chord_is_rotated_by_half_the_turn_and_shortened(self) -> None:
        turns = [10.0, -10.0, 10.0, -10.0]
        kx, ky, kw = stroke_rows(turns)
        cx, cy, cw = arc_rows(turns, math.pi / 2)
        self.assertEqual(cw, kw)
        r = np.array([0.05] * 4)
        straight = np.array([np.dot(kx, r), np.dot(ky, r)])
        chord = np.array([np.dot(cx, r), np.dot(cy, r)])
        expected = _rz(math.pi / 4) @ straight * (math.sin(math.pi / 4) / (math.pi / 4))
        np.testing.assert_allclose(chord, expected, atol=1e-12)


class GeometryTests(unittest.TestCase):
    @staticmethod
    def _camera_rotation(yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])  # pitch down = +
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return rz @ ry @ rx

    def test_heading_axes_follow_yaw_regardless_of_pitch(self) -> None:
        for pitch in (0.0, math.radians(60), math.radians(80)):
            rot = self._camera_rotation(math.radians(30), pitch)
            fwd, left = camera_heading_axes(rot)
            np.testing.assert_allclose(
                fwd, [math.cos(math.radians(30)), math.sin(math.radians(30))], atol=1e-9
            )
            np.testing.assert_allclose(
                left,
                [-math.sin(math.radians(30)), math.cos(math.radians(30))],
                atol=1e-9,
            )

    def test_heading_axes_reject_a_rolled_over_camera(self) -> None:
        with self.assertRaises(ValueError):
            camera_heading_axes(self._camera_rotation(0.0, 0.0, math.radians(85)))

    def test_heading_change_is_yaw_about_world_z(self) -> None:
        r0 = self._camera_rotation(math.radians(10), math.radians(60))
        r1 = self._camera_rotation(math.radians(-35), math.radians(60))
        self.assertAlmostEqual(math.degrees(heading_change(r0, r1)), -45.0, places=9)


class RobustFitTests(unittest.TestCase):
    def test_slipped_stroke_is_dropped_and_radii_recovered(self) -> None:
        radii = [0.0640, 0.0645, 0.0648, 0.0645]
        strokes = synth_strokes(
            make_plan(0.7, math.radians(90), 0.25, 0.2, 3), radii=radii
        )
        # A wheel slipping on one "right" stroke: the body moved 6 cm less
        # along y and slid 6 cm in x, wheels turned as commanded.
        bad = [i for i, s in enumerate(strokes) if s.name == "right"][-1]
        strokes[bad].dx_m -= 0.06
        strokes[bad].dy_m += 0.06
        naive = fit_calibration(strokes)
        robust = fit_robust(strokes)
        self.assertEqual(robust.dropped, [bad])
        self.assertEqual(len(robust.residuals), len(strokes))
        for got, want in zip(robust.wheel_scale, true_scale(radii)):
            self.assertAlmostEqual(got, want, places=4)
        self.assertGreater(naive.radius_spread, robust.radius_spread)
        self.assertGreater(
            math.hypot(*robust.residuals[bad][:2]), 0.05
        )  # the outlier's residual is reported against the clean solution

    def test_clean_strokes_drop_nothing(self) -> None:
        strokes = synth_strokes(make_plan(0.7, math.radians(90), 0.25, 0.2, 2))
        self.assertEqual(fit_robust(strokes).dropped, [])

    def test_never_drops_below_solvable(self) -> None:
        strokes = synth_strokes(make_plan(0.7, math.radians(90), 0.25, 0.2, 1))
        strokes[2].dy_m += 0.2
        cal = fit_robust(strokes)
        # 6 strokes → at most int(0.2 * 6) = 1 drop; 5 strokes still solve
        self.assertLessEqual(len(cal.dropped), 1)


class ConsistencyTests(unittest.TestCase):
    def test_clean_strokes_are_consistent(self) -> None:
        strokes = synth_strokes(make_plan(0.7, math.radians(90), 0.25, 0.2, 3))
        lines, bad = consistency_report(strokes)
        self.assertFalse(bad)
        self.assertEqual(len(lines), 4)  # header + spin + forward/back + left/right

    def test_tracker_losing_the_forward_axis_is_flagged(self) -> None:
        strokes = synth_strokes(make_plan(0.7, math.radians(90), 0.25, 0.2, 2))
        # One forward stroke measured 28° off-axis and short, heading unchanged:
        # what a down-looking camera did on the real cart.
        fwd = next(s for s in strokes if s.name == "forward")
        fwd.dx_m, fwd.dy_m = 0.645, 0.344
        lines, bad = consistency_report(strokes)
        self.assertTrue(bad)
        self.assertTrue(
            any("forward/back" in ln and "inconsistent" in ln for ln in lines)
        )
        self.assertFalse(any("spin" in ln and "inconsistent" in ln for ln in lines))


class PlumbingTests(unittest.TestCase):
    def test_wheel_scale_arg_is_a_draccus_list(self) -> None:
        self.assertEqual(
            wheel_scale_arg([1.0, 0.98765, 1.01, 1.0]), "[1.0000,0.9877,1.0100,1.0000]"
        )

    def test_plan_repeats_the_six_strokes(self) -> None:
        plan = make_plan(2.0, math.radians(90), 0.3, 0.2, 2)
        self.assertEqual(len(plan), 12)
        self.assertEqual(
            [p.name for p in plan[:6]],
            ["rotate ccw", "rotate cw", "forward", "back", "left", "right"],
        )
        self.assertTrue(all(p.rotation == (p.name.startswith("rotate")) for p in plan))

    def test_strokes_round_trip_through_json(self) -> None:
        strokes = synth_strokes(make_plan(1.5, math.radians(90), 0.25, 0.2, 1))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "strokes.json"
            write_strokes(path, strokes, 51617969)
            self.assertEqual(json.loads(path.read_text())["zed_serial"], 51617969)
            self.assertEqual(read_strokes(path), strokes)


if __name__ == "__main__":
    unittest.main()
