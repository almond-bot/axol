"""Chirp identification of a joint's tracking dynamics, and inverting it
for a known trajectory — against a simulated closed loop."""

from __future__ import annotations

import argparse
import io
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.signal import lsim, lti

from almond_axol.tuning.tracking_model import (
    TrackingModel,
    chirp_motion,
    fit_tracking_model,
    frequency_response,
    invert_reference,
    load_models,
    save_model,
)

FS = 240.0
WN, ZETA, TAU = 2 * math.pi * 2.8, 0.2, 0.04


def _plant(u: np.ndarray) -> np.ndarray:
    sys = lti([WN**2], [1, 2 * ZETA * WN, WN**2])
    t = np.arange(len(u)) / FS
    _, y, _ = lsim(sys, u, t)
    d = int(TAU * FS)
    return np.concatenate([np.full(d, y[0]), y[:-d]])


def _chirp_response(carrier: float = 3.0):
    m = chirp_motion(np.zeros(14), 7, carrier_rad_s=math.radians(carrier))
    r = m.q[:, 7].astype(float)
    y = _plant(r) + math.radians(0.01) * np.random.default_rng(0).standard_normal(
        len(r)
    )
    return m, r, y


class ChirpTest(unittest.TestCase):
    def test_amplitude_is_capped_by_acceleration_and_ends_at_the_base(self) -> None:
        base = np.linspace(-0.3, 0.3, 14)
        m = chirp_motion(base, 10, f0=0.3, f1=8.0, duration=40.0)
        x = m.q[:, 10] - base[10]
        self.assertAlmostEqual(float(x[0]), 0.0, places=6)
        self.assertAlmostEqual(float(x[-1]), 0.0, places=6)
        self.assertLessEqual(np.abs(x).max(), math.radians(1.5) + 1e-6)
        # That stretch of the sweep is above ~5 Hz: 200°/s² / (2π·5.3 Hz)² ≈ 0.18°.
        n = len(x)
        tail = x[int(0.85 * n) : int(0.93 * n)]
        self.assertLess(np.abs(tail).max(), math.radians(0.25))
        # Other columns untouched.
        np.testing.assert_allclose(m.q[:, 3], base[3], atol=1e-6)
        acc = np.gradient(np.gradient(x.astype(float))) * FS * FS
        self.assertLess(np.abs(acc).max(), math.radians(260))


class IdentifyTest(unittest.TestCase):
    def test_recovers_the_simulated_loop(self) -> None:
        _, r, y = _chirp_response()
        f, h, coh = frequency_response(r, y, FS, (0.3, 8.0))
        self.assertGreater(float(np.median(coh)), 0.9)
        m = fit_tracking_model(f, h, coh)
        self.assertAlmostEqual(m.fn_hz, 2.8, delta=0.05)
        self.assertAlmostEqual(m.zeta, ZETA, delta=0.02)
        self.assertAlmostEqual(m.tau, TAU, delta=0.006)
        self.assertAlmostEqual(m.k, 1.0, delta=0.03)

    def test_no_excitation_is_refused(self) -> None:
        t = np.arange(int(30 * FS)) / FS
        r = math.radians(20) * np.sin(2 * math.pi * 0.2 * t)
        y = _plant(r) + np.random.default_rng(1).standard_normal(len(r)) * 1e-3
        f, h, coh = frequency_response(r, y, FS, (2.0, 8.0))
        with self.assertRaises(ValueError):
            fit_tracking_model(f, h, coh)

    def test_inversion_cancels_the_linear_tracking_error(self) -> None:
        _, r, y = _chirp_response()
        f, h, coh = frequency_response(r, y, FS, (0.3, 8.0))
        model = fit_tracking_model(f, h, coh)
        t = np.arange(int(30 * FS)) / FS
        ref = math.radians(20) * np.sin(2 * math.pi * 0.3 * t) + math.radians(
            2
        ) * np.sin(2 * math.pi * 1.7 * t + 1.0)
        u = invert_reference(ref, FS, model)
        s = slice(int(2 * FS), -int(2 * FS))
        before = np.std((_plant(ref) - ref)[s])
        after = np.std((_plant(u) - ref)[s])
        self.assertLess(after, 0.05 * before)
        # Ends preserved: the stream starts and stops where the motion does.
        self.assertAlmostEqual(float(u[0]), float(ref[0]), places=4)
        self.assertAlmostEqual(float(u[-1]), float(ref[-1]), places=4)

    def test_inverse_gain_is_capped(self) -> None:
        # A very lightly damped model would ask for a huge notch-inverse boost
        # far above its peak; the cap keeps the command bounded.
        model = TrackingModel(
            1.0, 2 * math.pi * 3.0, 0.02, math.inf, 0.0, 0.3, 8.0, 0.0
        )
        t = np.arange(int(20 * FS)) / FS
        ref = math.radians(1) * np.sin(2 * math.pi * 6.0 * t)
        u = invert_reference(ref, FS, model, max_gain=3.0)
        self.assertLess(np.abs(u).max(), 3.2 * np.abs(ref).max())


class StoreTest(unittest.TestCase):
    def test_models_round_trip(self) -> None:
        m = TrackingModel(0.98, 17.6, 0.21, math.inf, 0.041, 0.3, 8.0, 0.05, {"x": 1.0})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tm.json"
            save_model("right.shoulder_1", m, path)
            save_model("right.elbow", m, path)
            got = load_models(path)
        self.assertEqual(set(got), {"right.shoulder_1", "right.elbow"})
        self.assertEqual(got["right.shoulder_1"], m)


class CliTest(unittest.TestCase):
    def _saved_chirp_run(self, runs_dir: Path) -> str:
        from almond_axol.cli.tune.motion import _COLUMNS
        from almond_axol.tuning.runs import save_run

        _, r, y = _chirp_response()
        n = len(r)
        target = np.zeros((n, 14), dtype=np.float32)
        actual = np.zeros((n, 14), dtype=np.float32)
        target[:, 7], actual[:, 7] = r, y
        return save_run(
            "motion",
            {
                "t": np.arange(n) / FS,
                "target": target,
                "actual": actual,
                "torque": np.zeros_like(actual),
            },
            {},
            gains={"right.shoulder_1.kp": 450.0},
            params={"columns": _COLUMNS},
            runs_dir=runs_dir,
        )

    def test_tune_tf_fits_and_saves_and_invert_uses_it(self) -> None:
        from almond_axol.cli.tune import motion as motion_cli
        from almond_axol.cli.tune import tf as tf_cli
        from almond_axol.tuning import runs, tracking_model

        with tempfile.TemporaryDirectory() as d:
            runs_dir, store = Path(d) / "runs", Path(d) / "tm.json"
            rid = self._saved_chirp_run(runs_dir)
            with (
                mock.patch.object(
                    tf_cli, "load_run", lambda x: runs.load_run(x, runs_dir)
                ),
                mock.patch.object(
                    tf_cli,
                    "save_model",
                    lambda k, m: tracking_model.save_model(k, m, store),
                ),
                redirect_stdout(io.StringIO()) as out,
            ):
                tf_cli.run(
                    argparse.Namespace(
                        joint="right.shoulder_1",
                        runs=[rid],
                        f0=None,
                        f1=None,
                        min_coherence=0.6,
                        save=True,
                    )
                )
            self.assertIn("natural frequency 2.8", out.getvalue())
            model = tracking_model.load_models(store)["right.shoulder_1"]
            self.assertEqual(model.gains, {"right.shoulder_1.kp": 450.0})

            # --invert: the modelled moving joint changes, held and
            # unmodelled joints do not.
            t = np.arange(int(20 * FS)) / FS
            ref = np.zeros((len(t), 14))
            ref[:, 7] = math.radians(20) * np.sin(2 * math.pi * 0.3 * t)
            ref[:, 10] = math.radians(20) * np.sin(2 * math.pi * 0.3 * t)
            ref[:, 12] = math.radians(20) * np.sin(2 * math.pi * 0.3 * t)
            real_load = tracking_model.load_models
            with (
                mock.patch.object(
                    tracking_model,
                    "load_models",
                    lambda path=store: real_load(store),
                ),
                redirect_stdout(io.StringIO()),
            ):
                out_stream = motion_cli._invert_stream(
                    ref,
                    ref,
                    {12: None},
                    {("right", "shoulder_1", "kp"): 450.0},
                    FS,
                    "right",
                )
            self.assertGreater(
                np.abs(out_stream[:, 7] - ref[:, 7]).max(), math.radians(0.5)
            )
            np.testing.assert_array_equal(out_stream[:, 10], ref[:, 10])
            np.testing.assert_array_equal(out_stream[:, 12], ref[:, 12])


if __name__ == "__main__":
    unittest.main()
