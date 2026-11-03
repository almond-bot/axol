"""Save-time resampling of Mantis EE poses onto camera exposure times."""

from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from almond_axol.mantis.alignment import POSE_LAG_FEATURE, align_episode_ee_poses
from almond_axol.recording import record_proc

FPS = 30.0
NAMES = [
    "left_ee.x",
    "left_ee.y",
    "left_ee.z",
    "left_ee.rx",
    "left_ee.ry",
    "left_ee.rz",
    "left_gripper.pos",
]
FEATURES = {
    "action": {"dtype": "float32", "shape": (7,), "names": NAMES},
    "observation.state": {"dtype": "float32", "shape": (7,), "names": NAMES},
    POSE_LAG_FEATURE: {"dtype": "float32", "shape": (1,), "names": ["pose_lag"]},
}
VEL = np.array([0.6, -0.3, 0.2])  # m/s
OMEGA = np.array([0.0, 0.0, 3.0])  # rad/s (~170 deg/s)


def pose_at(t: float) -> np.ndarray:
    rv = Rotation.from_rotvec(OMEGA * t).as_rotvec()
    return np.r_[VEL * t, rv]


def make_episode(t_img, lag, gripper=None):
    """A buffered episode whose stored pose is the true pose at ``t_img - lag``."""
    t_img = np.asarray(t_img, dtype=np.float64)
    lag = np.asarray(lag, dtype=np.float64)
    if gripper is None:
        gripper = np.linspace(0.0, 1.0, t_img.size)
    rows = [
        np.r_[pose_at(t - dt), g].astype(np.float32)
        for t, dt, g in zip(t_img, lag, gripper)
    ]
    return {
        "action": [r.copy() for r in rows],
        "observation.state": [r.copy() for r in rows],
        POSE_LAG_FEATURE: [np.array([dt], dtype=np.float32) for dt in lag],
        "size": t_img.size,
    }


def jittered(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    t_img = 100.0 + np.arange(n) / FPS + rng.uniform(0.0, 1.0 / 60.0, n)
    lag = rng.uniform(-0.012, 0.012, n)
    return t_img, lag


def rot_err_deg(rv_a, rv_b) -> float:
    rel = Rotation.from_rotvec(rv_a).inv() * Rotation.from_rotvec(rv_b)
    return float(np.degrees(rel.magnitude()).max())


class AlignEpisodeTest(unittest.TestCase):
    def test_interior_rows_land_on_exposure_time(self):
        t_img, lag = jittered(90)
        ep = make_episode(t_img, lag)
        before = np.stack(ep["observation.state"]).copy()

        stats = align_episode_ee_poses(ep, FEATURES, list(t_img), FPS)

        self.assertIsNotNone(stats)
        self.assertGreaterEqual(stats.aligned, 85)
        state = np.stack(ep["observation.state"]).astype(np.float64)
        action = np.stack(ep["action"]).astype(np.float64)
        truth = np.stack([pose_at(t) for t in t_img])
        lag_after = np.array([r[0] for r in ep[POSE_LAG_FEATURE]])
        aligned = lag_after == 0.0
        self.assertEqual(int(aligned.sum()), stats.aligned)
        # Constant velocity: linear/slerp interpolation is exact up to float32.
        np.testing.assert_allclose(state[aligned, :3], truth[aligned, :3], atol=2e-6)
        self.assertLess(rot_err_deg(state[aligned, 3:6], truth[aligned, 3:6]), 1e-3)
        np.testing.assert_array_equal(state, action)
        # The uncorrected track was measurably off, so the test means something.
        self.assertGreater(np.abs(before[:, :3] - truth[:, :3]).max(), 5e-3)
        self.assertGreater(stats.max_shift_mm, 5.0)

    def test_gripper_and_unaligned_rows_untouched(self):
        t_img, lag = jittered(40, seed=1)
        ep = make_episode(t_img, lag)
        before = np.stack(ep["observation.state"]).copy()
        align_episode_ee_poses(ep, FEATURES, list(t_img), FPS)
        after = np.stack(ep["observation.state"])
        np.testing.assert_array_equal(after[:, 6], before[:, 6])
        lag_after = np.array([r[0] for r in ep[POSE_LAG_FEATURE]])
        kept = lag_after != 0.0
        np.testing.assert_array_equal(after[kept], before[kept])
        np.testing.assert_allclose(lag_after[kept], lag[kept].astype(np.float32))

    def test_first_and_last_rows_never_extrapolate(self):
        t_img = 100.0 + np.arange(10) / FPS
        lag = np.full(10, 0.010)  # every pose 10 ms before its image
        ep = make_episode(t_img, lag)
        before = np.stack(ep["observation.state"]).copy()
        stats = align_episode_ee_poses(ep, FEATURES, list(t_img), FPS)
        # The last row's exposure is after the newest pose sample.
        self.assertEqual(stats.aligned, 9)
        np.testing.assert_array_equal(ep["observation.state"][-1], before[-1])
        self.assertAlmostEqual(float(ep[POSE_LAG_FEATURE][-1][0]), 0.010, places=6)

    def test_held_pose_is_filled_from_neighbours(self):
        t_img = 100.0 + np.arange(12) / FPS
        lag = np.zeros(12)
        lag[1::2] = 0.004
        ep = make_episode(t_img, lag)
        # Row 6 reused row 5's snapshot: same pose, same pose time.
        held = 6
        ep["observation.state"][held] = ep["observation.state"][held - 1].copy()
        ep["action"][held] = ep["action"][held - 1].copy()
        t_held_pose = t_img[held - 1] - lag[held - 1]
        lag[held] = t_img[held] - t_held_pose
        ep[POSE_LAG_FEATURE][held][...] = lag[held]

        align_episode_ee_poses(ep, FEATURES, list(t_img), FPS)

        got = ep["observation.state"][held].astype(np.float64)
        np.testing.assert_allclose(got[:3], pose_at(t_img[held])[:3], atol=2e-6)
        self.assertEqual(float(ep[POSE_LAG_FEATURE][held][0]), 0.0)

    def test_rows_across_a_gap_keep_their_pose(self):
        t_img = np.r_[100.0 + np.arange(6) / FPS, 105.0 + np.arange(6) / FPS]
        lag = np.full(12, -0.005)  # poses 5 ms after the image
        ep = make_episode(t_img, lag)
        before = np.stack(ep["observation.state"]).copy()
        align_episode_ee_poses(ep, FEATURES, list(t_img), FPS)
        # Row 6 (first after the pause) would need the pre-pause sample.
        np.testing.assert_array_equal(ep["observation.state"][6], before[6])
        self.assertNotEqual(float(ep[POSE_LAG_FEATURE][6][0]), 0.0)
        self.assertEqual(float(ep[POSE_LAG_FEATURE][7][0]), 0.0)

    def test_rotation_crossing_the_rotvec_wrap(self):
        # Spin through pi about z: rotation vectors flip sign there.
        global OMEGA
        saved = OMEGA
        OMEGA = np.array([0.0, 0.0, 6.0])
        try:
            t_img = np.arange(30) / FPS + 0.4
            lag = np.where(np.arange(30) % 2, 0.008, -0.008)
            ep = make_episode(t_img, lag)
            align_episode_ee_poses(ep, FEATURES, list(t_img), FPS)
            state = np.stack(ep["observation.state"]).astype(np.float64)
            truth = np.stack([pose_at(t) for t in t_img])
            self.assertLess(rot_err_deg(state[1:-1, 3:6], truth[1:-1, 3:6]), 1e-3)
        finally:
            OMEGA = saved

    def test_no_op_without_pose_lag_or_on_mismatch(self):
        t_img, lag = jittered(20, seed=2)
        ep = make_episode(t_img, lag)
        del ep[POSE_LAG_FEATURE]
        self.assertIsNone(align_episode_ee_poses(ep, FEATURES, list(t_img), FPS))

        ep = make_episode(t_img, lag)
        before = np.stack(ep["observation.state"]).copy()
        with self.assertLogs("almond_axol.mantis.alignment", logging.WARNING):
            self.assertIsNone(
                align_episode_ee_poses(ep, FEATURES, list(t_img[:-1]), FPS)
            )
        np.testing.assert_array_equal(np.stack(ep["observation.state"]), before)


class RecorderHookTest(unittest.TestCase):
    def _dataset(self, ep):
        return SimpleNamespace(
            writer=SimpleNamespace(episode_buffer=ep),
            meta=SimpleNamespace(features=FEATURES),
        )

    def test_disabled_unless_configured(self):
        t_img, lag = jittered(20, seed=3)
        ep = make_episode(t_img, lag)
        before = np.stack(ep["observation.state"]).copy()
        record_proc._maybe_align_episode(self._dataset(ep), {"fps": FPS}, list(t_img))
        np.testing.assert_array_equal(np.stack(ep["observation.state"]), before)

    def test_enabled_aligns_and_logs(self):
        t_img, lag = jittered(20, seed=4)
        ep = make_episode(t_img, lag)
        with self.assertLogs(record_proc._logger, logging.INFO) as logs:
            record_proc._maybe_align_episode(
                self._dataset(ep),
                {"fps": FPS, "align_ee_to_exposure": True},
                list(t_img),
            )
        self.assertIn("aligned", "\n".join(logs.output))
        self.assertTrue(any(float(r[0]) == 0.0 for r in ep[POSE_LAG_FEATURE]))

    def test_failure_keeps_the_take(self):
        dataset = SimpleNamespace(writer=None, meta=None)  # raises inside
        with self.assertLogs(record_proc._logger, logging.ERROR):
            record_proc._maybe_align_episode(
                dataset, {"fps": FPS, "align_ee_to_exposure": True}, [0.0]
            )


if __name__ == "__main__":
    unittest.main()
