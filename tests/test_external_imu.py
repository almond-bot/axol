"""An operator's own IMU on the end-effector: read, matched to a run by
encoder kinematics, scored like the wrist camera's, used by the queue."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from almond_axol.tuning import external_imu as ei
from almond_axol.tuning.creep_score import flange_heights, flange_sway, score_run

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
T0 = 1_790_000_000.0
RATE = 240.0


def _queue():
    spec = importlib.util.spec_from_file_location(
        "tuning_queue", SCRIPTS / "tuning_queue.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tuning_queue"] = mod
    spec.loader.exec_module(mod)
    return mod


def _run(runs: Path, rid: str = "r1", clock_error: float = 0.6) -> np.ndarray:
    """A saved right-arm run: shoulder_2 and the elbow swaying at 1.3 / 2.1 Hz
    around a bent pose. Returns its joint angles (rad, 14 columns)."""
    t = np.arange(0, 20, 1 / RATE)
    q = np.zeros((len(t), 14))
    q[:, :7] = np.nan  # left arm not driven
    q[:, 7:] = np.radians([10, 10, -15, -60, -5, -5, 0])
    q[:, 8] += math.radians(2.0) * np.sin(2 * np.pi * 1.3 * t)
    q[:, 10] += math.radians(1.5) * np.sin(2 * np.pi * 2.1 * t + 0.4)
    d = runs / rid
    d.mkdir(parents=True)
    np.savez(d / "series.npz", t=t, target=np.nan_to_num(q), actual=q)
    meta = {
        "id": rid,
        "label": "slow_osc base r1",
        "params": {"motion": "slow_osc", "arms": "right"},
        "metrics": {"t0_wall": T0 + clock_error, "guard_trips": []},
        "startedAt": T0 + 25,
    }
    (d / "meta.json").write_text(json.dumps(meta))
    return q


def _imu_log(q: np.ndarray, start: float = -10, stop: float = 30) -> dict:
    """What an IMU on the right flange reads, on the true clock (T0 = the
    run's t = 0), at 200 Hz, tilted in its mount, with noise."""
    fs = 200.0
    t_rel = np.arange(start, stop, 1 / fs)
    t_run = np.arange(len(q)) / RATE
    qi = np.stack(
        [np.interp(t_rel, t_run, q[:, 7 + j], left=q[0, 7 + j]) for j in range(7)], 1
    )
    z = flange_heights(qi, "right")
    az = np.gradient(np.gradient(z, 1 / fs), 1 / fs)
    world = np.stack([np.zeros_like(az), np.zeros_like(az), 9.81 + az], 1)
    c, s = math.cos(0.5), math.sin(0.5)
    rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])  # mount tilt
    rng = np.random.default_rng(0)
    acc = world @ rot.T + rng.normal(0, 0.02, world.shape)
    return {"t": t_rel + T0, "acc": acc}


class ExternalImuTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.runs = Path(self._tmp.name) / "runs"
        self.q = _run(self.runs)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_matched_through_a_clock_error_and_scored_like_the_flange(self) -> None:
        out = ei.attach("r1", _imu_log(self.q), runs_dir=self.runs)
        self.assertEqual(out["side"], "right")
        self.assertAlmostEqual(out["offset_s"], -0.6, delta=0.02)
        self.assertGreater(out["corr"], 0.9)
        # A rigid IMU at the flange sees what the encoders see.
        t = np.arange(len(self.q)) / RATE
        enc = flange_sway(t, self.q, "right")
        self.assertAlmostEqual(
            out["metrics"]["low_mm"], enc["low_mm"], delta=0.15 * enc["low_mm"]
        )
        score = score_run("r1", self.runs)
        self.assertEqual(score["imu"]["source"], "external")
        self.assertIn("low_mm", score["enc"])

    def test_a_log_without_the_motion_is_refused(self) -> None:
        log = _imu_log(self.q)
        rng = np.random.default_rng(1)
        log["acc"] = np.array([0, 0, 9.81]) + rng.normal(0, 0.05, log["acc"].shape)
        with self.assertRaises(ValueError):
            ei.attach("r1", log, runs_dir=self.runs)
        with self.assertRaises(ValueError):  # the arm that didn't move
            ei.attach("r1", _imu_log(self.q), side="left", runs_dir=self.runs)
        with self.assertRaises(ValueError):  # logging stopped before the run
            ei.attach("r1", _imu_log(self.q, -60, -20), runs_dir=self.runs)
        self.assertIsNone(ei.load_attached("r1", self.runs))

    def test_csv_units_and_timestamps(self) -> None:
        log = _imu_log(self.q)
        path = Path(self._tmp.name) / "imu.csv"
        rows = ["time_ms,ax,ay,az"] + [
            f"{t * 1e3:.1f},{a[0] / 9.80665:.5f},{a[1] / 9.80665:.5f},"
            f"{a[2] / 9.80665:.5f}"
            for t, a in zip(log["t"], log["acc"])
        ]
        path.write_text("\n".join(rows))
        read = ei.load_log(path)  # ms and g, both guessed
        self.assertAlmostEqual(read["t"][0], log["t"][0], places=2)
        self.assertAlmostEqual(
            float(np.median(np.linalg.norm(read["acc"], axis=1))), 9.81, delta=0.05
        )
        self.assertEqual(ei.describe(read)["rate_hz"], 200.0)
        bad = Path(self._tmp.name) / "linear.csv"
        bad.write_text("\n".join(f"{T0 + i / 200},0.01,0.02,0.03" for i in range(100)))
        with self.assertRaises(ValueError):  # gravity removed
            ei.load_log(bad)

    def test_the_queue_decides_on_an_attached_imu(self) -> None:
        tq = _queue()
        ei.attach("r1", _imu_log(self.q), runs_dir=self.runs)
        records = [
            {
                "item": {"label": "x", "variant": "base", "round": 1, "arm": "right"},
                "ids": ["r1"],
                "trips": [],
                "scores": [{"id": "r1", "enc": {"low_mm": 1.0}}],
            }
        ]
        self.assertEqual(tq.merge_external_imu(records, self.runs), 1)
        self.assertEqual(records[0]["scores"][0]["imu"]["source"], "external")
        self.assertEqual(tq.deciding_metric(tq.summarize(records)), "imu.low_mm")


if __name__ == "__main__":
    unittest.main()
