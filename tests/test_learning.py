"""Iterative learning of a command correction against a simulated joint:
an impedance-like resonance with delay and a repeatable disturbance."""

from __future__ import annotations

import math
import unittest

import numpy as np

from almond_axol.tuning.learning import CommandLearner, band_limit

FS = 240.0
N = int(24 * FS)


def _plant(
    u: np.ndarray, wn_hz: float = 2.5, zeta: float = 0.15, delay_s: float = 0.03
):
    """Discrete 2nd-order position loop: y'' = wn²(u(t-d) - y) - 2ζwn·y'."""
    wn = 2 * math.pi * wn_hz
    d = int(delay_s * FS)
    dt = 1 / FS
    y = np.zeros_like(u)
    v = 0.0
    for i in range(1, len(u)):
        ui = u[max(i - 1 - d, 0)]
        acc = wn * wn * (ui - y[i - 1]) - 2 * zeta * wn * v
        v += acc * dt
        y[i] = y[i - 1] + v * dt
    return y


def _setup():
    t = np.arange(N) / FS
    ref = np.radians(20) * np.sin(2 * math.pi * 0.15 * t)
    rng = np.random.default_rng(0)
    # Repeatable in-band "friction" disturbance: 1-3 Hz, a few tenths of a degree.
    dist = band_limit(rng.standard_normal(N), FS, (1.0, 3.0))
    dist *= np.radians(0.3) / dist.std()
    return ref, dist


def _fly(ref, dist, offset):
    return _plant(ref + offset) + dist


class LearnerTest(unittest.TestCase):
    def test_repeatable_in_band_error_drops_by_80_percent(self) -> None:
        ref, dist = _setup()
        learner = CommandLearner(N, FS, columns=np.array([True]))
        totals = []
        for _ in range(8):
            y = _fly(ref, dist, learner.offset[:, 0])
            rep = learner.update(ref[:, None], y[:, None])
            totals.append(rep.total)
        self.assertLess(min(totals[3:]), 0.2 * totals[0], totals)
        # The slow tracking lag is not what it learned: the offset is in band.
        spec = np.fft.rfft(learner.best[:, 0])
        f = np.fft.rfftfreq(N, 1 / FS)
        low = np.sqrt(np.sum(np.abs(spec[(f > 0) & (f < 0.5)]) ** 2))
        self.assertLess(low, 0.02 * np.sqrt(np.sum(np.abs(spec) ** 2)))

    def test_offsets_are_clamped_and_masked_columns_stay_zero(self) -> None:
        ref, dist = _setup()
        learner = CommandLearner(
            N, FS, columns=np.array([True, False]), max_rad=math.radians(0.1)
        )
        refs = np.stack([ref, ref], 1)
        for _ in range(4):
            y = np.stack([_fly(ref, dist * 5, learner.offset[:, 0])] * 2, 1)
            learner.update(refs, y)
        self.assertLessEqual(
            np.abs(learner.offset[:, 0]).max(), math.radians(0.1) + 1e-12
        )
        self.assertEqual(np.abs(learner.offset[:, 1]).max(), 0.0)

    def test_a_worse_pass_rolls_back_to_the_best_offset(self) -> None:
        ref, dist = _setup()
        learner = CommandLearner(N, FS, columns=np.array([True]))
        y0 = _fly(ref, dist, 0 * ref)
        learner.update(ref[:, None], y0[:, None])  # baseline → first step
        # Pretend the corrected pass came out far worse.
        rep = learner.update(ref[:, None], (y0 + 3 * dist)[:, None])
        self.assertTrue(rep.rolled_back)
        self.assertEqual(rep.step, "rollback")
        np.testing.assert_array_equal(learner.offset, np.zeros_like(learner.offset))
        self.assertAlmostEqual(
            rep.gain, 0.5 * CommandLearner(N, FS, columns=np.array([True])).gain
        )

    def test_short_pass_is_refused(self) -> None:
        learner = CommandLearner(100, FS, columns=np.array([True]))
        with self.assertRaises(ValueError):
            learner.update(np.zeros((50, 1)), np.zeros((50, 1)))


class ExplainTest(unittest.TestCase):
    def test_a_learned_coulomb_and_stribeck_correction_is_recognised(self) -> None:
        from almond_axol.tuning.friction_model import coulomb_unit, stribeck_shape
        from almond_axol.tuning.learning import explain_correction

        t = np.arange(N) / FS
        q = np.radians(20) * np.sin(2 * math.pi * 0.15 * t)
        v = np.gradient(q) * FS
        a = np.gradient(v) * FS
        kp = 450.0
        # What the joint needed on top of its feedforward: 0.3 Nm Coulomb and
        # 0.2 Nm Stribeck excess, learned as a position offset (torque / kp).
        need = 0.3 * coulomb_unit(v, 100.0) + 0.2 * stribeck_shape(v, 0.1)
        learned = band_limit(need / kp, FS, (0.7, 8.0))
        coef, r2 = explain_correction(learned, kp, v, a, FS)
        self.assertGreater(r2, 0.95)
        self.assertAlmostEqual(coef["coulomb (fc)"], 0.3, delta=0.05)
        self.assertAlmostEqual(coef["stribeck excess (dfs)"], 0.2, delta=0.05)
        # A correction unrelated to the motion is not explained.
        noise = (
            band_limit(np.random.default_rng(3).standard_normal(N), FS, (1, 3)) * 1e-3
        )
        _, r2 = explain_correction(noise, kp, v, a, FS)
        self.assertLess(r2, 0.1)


class CliTest(unittest.TestCase):
    def test_learn_columns_default_to_moving_unheld_joints_of_driven_arms(self) -> None:
        import argparse

        from almond_axol.cli.tune.motion import _COLUMNS, _learn_columns

        ref = np.zeros((100, 14))
        ref[:, 7] = np.linspace(0, 0.5, 100)  # right.shoulder_1 moves
        ref[:, 10] = np.linspace(0, -0.5, 100)  # right.elbow moves
        ref[:, 12] = np.linspace(0, 0.5, 100)  # right.wrist_2 moves but is held
        ref[:, 0] = np.linspace(0, 0.5, 100)  # left arm not driven
        args = argparse.Namespace(learn_joint=[], arms="right")
        mask = _learn_columns(args, ref, {12: None})
        self.assertEqual(
            [_COLUMNS[i] for i in np.where(mask)[0]],
            ["right.shoulder_1", "right.elbow"],
        )
        args = argparse.Namespace(learn_joint=["right.elbow"], arms="right")
        mask = _learn_columns(args, ref, {})
        self.assertEqual([_COLUMNS[i] for i in np.where(mask)[0]], ["right.elbow"])
        with self.assertRaises(SystemExit):
            _learn_columns(
                argparse.Namespace(learn_joint=["elbow"], arms="right"), ref, {}
            )

    def test_correction_round_trips_through_a_saved_run(self) -> None:
        import tempfile
        from pathlib import Path
        from unittest import mock

        from almond_axol.cli.tune import motion as cli
        from almond_axol.tuning import runs

        with tempfile.TemporaryDirectory() as d:
            corr = np.full((50, 14), 0.01, dtype=np.float32)
            with mock.patch.object(runs, "TUNING_RUNS_DIR", Path(d)):
                run_id = runs.save_run(
                    "motion", {"correction": corr}, {}, runs_dir=Path(d)
                )
                with mock.patch.object(
                    cli, "load_run", lambda rid: runs.load_run(rid, Path(d))
                ):
                    np.testing.assert_allclose(cli._load_correction(run_id, 50), corr)
                    with self.assertRaises(SystemExit):
                        cli._load_correction(run_id, 60)  # another motion
                    with self.assertRaises(SystemExit):
                        cli._load_correction("nope", 50)


if __name__ == "__main__":
    unittest.main()
