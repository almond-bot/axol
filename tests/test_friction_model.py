"""The slow-profile friction calibration: the runtime-shaped fit, the sweep
windows, and the calibration-file round trip of what it saves."""

from __future__ import annotations

import csv
import io
from contextlib import redirect_stdout
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from almond_axol.robot.calibration import load_calibration, update_joint_calibration
from almond_axol.robot.config import JointConfig, FrictionParams, _calibrated_joint
from almond_axol.tuning.friction_model import (
    K_MAX,
    STRIBECK_V0,
    FrictionFit,
    fit_friction,
    matched_samples,
    read_raw_csv,
    speed_table,
)

TRUE = FrictionFit(
    fc=0.6,
    fl=0.045,
    k=K_MAX,
    fv=0.3,
    fo=0.1,
    dfs=0.4,
    ls=0.04,
    vs=0.06,
    rms=0.0,
    r2=1.0,
    n=0,
    load_span=0.0,
    speeds=(),
)
SPEEDS_DEG = (1, 2, 3, 5, 8, 15, 30)


def _gravity(q: np.ndarray) -> np.ndarray:
    return -14.0 * np.cos(q)  # |load| 14 Nm down to ~5 over the sweep


def _sweep(seed: int = 1, speeds=SPEEDS_DEG, noise: float = 0.15):
    rng = np.random.default_rng(seed)
    cols = {"speed": [], "direction": [], "q": [], "tau": []}
    for deg in speeds:
        v = math.radians(deg)
        for d in (1, -1):
            q = np.radians(np.linspace(-60, 70, 1500))
            g = _gravity(q)
            tau = (
                g
                + TRUE.fo
                + d * TRUE.halfdiff(np.full(len(q), v), g)
                + noise * rng.standard_normal(len(q))
                + 0.1 * np.sin(q * 400)  # a gear-mesh ripple the curve ignores
            )
            cols["speed"] += [v] * len(q)
            cols["direction"] += [d] * len(q)
            cols["q"] += list(q)
            cols["tau"] += list(tau)
    return {k: np.asarray(v) for k, v in cols.items()}


class RuntimeShapeTest(unittest.TestCase):
    def test_halfdiff_is_the_core_law(self) -> None:
        # (fc + fl|g|)·tanh(0.1·min(k,100)·v) + fv·v + (dfs + ls|g|)·exp(-(v/vs)²)·tanh(v/0.02)
        v, g = 0.05, -8.0
        want = (
            (0.6 + 0.045 * 8) * math.tanh(0.1 * 100 * v)
            + 0.3 * v
            + (0.4 + 0.04 * 8)
            * math.exp(-((v / 0.06) ** 2))
            * math.tanh(v / STRIBECK_V0)
        )
        self.assertAlmostEqual(
            float(TRUE.halfdiff(np.array([v]), g)[0]), want, places=12
        )
        # A k above the runtime cap behaves as the cap.
        capped = FrictionFit(**{**TRUE.as_dict(), "k": 742.0, "speeds": ()})
        self.assertAlmostEqual(
            float(capped.halfdiff(np.array([v]), g)[0]), want, places=12
        )


class FitTest(unittest.TestCase):
    def test_recovers_the_curve_from_a_slow_multi_load_sweep(self) -> None:
        d = _sweep()
        samples = matched_samples(
            d["speed"], d["direction"], d["q"], d["tau"], _gravity
        )
        fit = fit_friction(
            samples, gravity_residual=lambda s: s.average - _gravity(s.q)
        )
        self.assertAlmostEqual(fit.fc, TRUE.fc, delta=0.05)
        self.assertAlmostEqual(fit.fl, TRUE.fl, delta=0.005)
        self.assertAlmostEqual(fit.fv, TRUE.fv, delta=0.05)
        self.assertAlmostEqual(fit.dfs, TRUE.dfs, delta=0.06)
        self.assertAlmostEqual(fit.ls, TRUE.ls, delta=0.006)
        self.assertAlmostEqual(fit.vs, TRUE.vs, delta=0.01)
        self.assertAlmostEqual(fit.fo, TRUE.fo, delta=0.02)
        self.assertEqual(fit.k, K_MAX)
        for row in speed_table(fit, samples):
            self.assertAlmostEqual(row["fit_nm"], row["measured_nm"], delta=0.02)

    def test_narrow_load_pins_the_load_terms(self) -> None:
        d = _sweep()
        flat = lambda q: np.full_like(np.asarray(q, float), 6.0)  # noqa: E731
        samples = matched_samples(d["speed"], d["direction"], d["q"], d["tau"], flat)
        fit = fit_friction(samples)
        self.assertEqual((fit.fl, fit.ls), (0.0, 0.0))
        self.assertLess(fit.load_span, 3.0)

    def test_too_few_speeds_or_samples_is_refused(self) -> None:
        d = _sweep(speeds=(3, 6))
        samples = matched_samples(
            d["speed"], d["direction"], d["q"], d["tau"], _gravity
        )
        with self.assertRaisesRegex(ValueError, "sweep speeds"):
            fit_friction(samples)
        with self.assertRaisesRegex(ValueError, "too few"):
            fit_friction(samples[:5])

    def test_per_sample_load_and_csv_round_trip(self) -> None:
        d = _sweep(speeds=(2, 5, 15))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.csv"
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "joint",
                        "side",
                        "pass",
                        "v_rad_s",
                        "direction",
                        "q_rad",
                        "tau_nm",
                        "load_nm",
                    ]
                )
                for s, dn, q, t in zip(d["speed"], d["direction"], d["q"], d["tau"]):
                    w.writerow(
                        [
                            "shoulder_1",
                            "right",
                            0,
                            s,
                            "+" if dn > 0 else "-",
                            q,
                            t,
                            abs(_gravity(q)),
                        ]
                    )
            raw = read_raw_csv([path])
        self.assertEqual(str(raw["joint"]), "shoulder_1")
        a = matched_samples(
            raw["speed"],
            raw["direction"],
            raw["q"],
            raw["tau"],
            raw["load"],
            raw["group"],
        )
        b = matched_samples(d["speed"], d["direction"], d["q"], d["tau"], _gravity)
        self.assertEqual(len(a), len(b))
        np.testing.assert_allclose(
            [s.halfdiff for s in a], [s.halfdiff for s in b], atol=1e-6
        )
        np.testing.assert_allclose([s.load for s in a], [s.load for s in b], rtol=0.02)


class SlowWindowsTest(unittest.TestCase):
    def test_slow_speeds_get_a_low_and_a_high_load_window(self) -> None:
        from almond_axol.cli.tune.friction import slow_profile_seconds, slow_windows

        load = lambda q: abs(14.0 * math.cos(q))  # noqa: E731
        lo, hi = math.radians(-60), math.radians(70)
        w = slow_windows(lo, hi, math.radians(1.0), load)
        self.assertEqual(len(w), 2)
        loads = sorted(load(0.5 * (a + b)) for a, b in w)
        self.assertLess(loads[0], 7.0)
        self.assertGreater(loads[1], 13.0)
        for a, b in w:
            self.assertGreaterEqual(a, lo - 1e-9)
            self.assertLessEqual(b, hi + 1e-9)
            self.assertAlmostEqual(math.degrees(b - a), 15.0, delta=0.01)
        # Fast speeds sweep the whole range.
        self.assertEqual(slow_windows(lo, hi, math.radians(15.0), load), [(lo, hi)])
        minutes = (
            slow_profile_seconds(lo, hi, [math.radians(v) for v in SPEEDS_DEG], load)
            / 60
        )
        self.assertLess(minutes, 10.0)


class _FakeMotor:
    """A joint that tracks its program exactly; its torque is gravity at the
    sweep pose plus the TRUE friction curve at the programmed velocity."""

    def __init__(self, load) -> None:
        self.pos = 0.0
        self.load = load
        self.rng = np.random.default_rng(5)

    async def get_position(self) -> float:
        return self.pos

    async def run_experiment(
        self, kp, kd, rate_hz, samples, differentiate, feedforward
    ):
        rows = []
        for sample in samples:
            q = float(sample[0])
            v = float(sample[3]) if len(sample) > 3 else 0.0
            g = -self.load(q)
            fric = (
                math.copysign(1.0, v) * float(TRUE.halfdiff(np.array([abs(v)]), g)[0])
                if v
                else 0.0
            )
            tau = g + TRUE.fo + fric + 0.05 * self.rng.standard_normal()
            rows.append({"actual": q, "torque": tau})
            self.pos = q
        return rows


class SlowSessionTest(unittest.TestCase):
    def test_the_slow_sweep_recovers_the_curve_it_measured(self) -> None:
        import asyncio
        from unittest import mock

        from almond_axol.cli.tune import friction as cli
        from almond_axol.constants import Joint

        load = lambda q: abs(14.0 * math.cos(q))  # noqa: E731
        motor = _FakeMotor(load)
        lo, hi = math.radians(-60), math.radians(60)
        speeds = [math.radians(v) for v in SPEEDS_DEG]
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch("asyncio.sleep", mock.AsyncMock()),
        ):
            raw = Path(tmp) / "raw.csv"
            rows = asyncio.run(
                cli._identify_slow(
                    motor,
                    250.0,
                    3.5,
                    speeds,
                    lo,
                    hi,
                    load,
                    Joint.SHOULDER_1,
                    False,
                    raw,
                )
            )
            self.assertTrue(raw.exists())
            back = read_raw_csv([raw])
        self.assertEqual(len(back["q"]), len(rows["q"]))
        # Every speed made it in, the slow ones in two windows.
        groups_per_speed = {}
        for v, g in zip(rows["speed"], rows["group"]):
            groups_per_speed.setdefault(round(math.degrees(v), 1), set()).add(g)
        self.assertEqual(len(groups_per_speed[1.0]), 2)
        self.assertEqual(len(groups_per_speed[30.0]), 1)
        # Fit with the sample loads (the fake's gravity is not the URDF's).
        samples = matched_samples(
            np.asarray(rows["speed"]),
            np.asarray(rows["direction"]),
            np.asarray(rows["q"]),
            np.asarray(rows["tau"]),
            np.asarray(rows["load"]),
            np.asarray(rows["group"]),
        )
        fit = fit_friction(samples)
        for sp in SPEEDS_DEG:
            v = math.radians(sp)
            for g in (5.0, 13.0):
                self.assertAlmostEqual(
                    float(fit.halfdiff(np.array([v]), g)[0]),
                    float(TRUE.halfdiff(np.array([v]), g)[0]),
                    delta=0.04,
                )


class ReportTest(unittest.TestCase):
    def test_sweep_pose_rows_give_fo_and_the_excess_stays_slow(self) -> None:
        from almond_axol.cli.tune.friction import fit_and_report, sweep_load
        from almond_axol.constants import ARM_JOINTS, Joint
        from almond_axol.robot.gravity import GravityCompensator
        from almond_axol.tuning import sweep_safety

        joint = Joint.SHOULDER_1
        other, _, _, _ = sweep_safety(joint, False)
        load = sweep_load(joint, False, other)
        gc = GravityCompensator()
        arm = np.zeros(7, dtype=np.float32)
        for j, t in other.items():
            arm[ARM_JOINTS.index(j)] = t
        rows = {k: [] for k in ("speed", "direction", "q", "tau", "group", "load")}
        # Friction that stays high to 15°/s and drops by 30°/s — the jelly
        # shape that dragged a free vs out to 19°/s.
        curve = {1: 1.3, 2: 1.3, 3: 1.28, 5: 1.3, 8: 1.33, 15: 1.2, 30: 0.82}
        for gi, (deg, h) in enumerate(curve.items()):
            v = math.radians(deg)
            for d in (1, -1):
                for q in np.radians(np.linspace(-40, 40, 400)):
                    arm[0] = q
                    g = float(gc.gravity_arm(arm, is_left=False)[0])
                    rows["speed"].append(v)
                    rows["direction"].append("+" if d > 0 else "-")
                    rows["q"].append(q)
                    rows["tau"].append(g + 0.1 + d * h)
                    rows["group"].append(gi)
                    rows["load"].append(load(q))
        with redirect_stdout(io.StringIO()):
            fit = fit_and_report(rows, joint, False, other)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit.fo, 0.1, delta=0.01)
        self.assertLessEqual(fit.vs, 0.1 + 1e-9)
        self.assertGreater(fit.fc, 0.2)


class CalibrationTest(unittest.TestCase):
    def test_friction_fl_and_stribeck_fields_round_trip_and_apply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            update_joint_calibration(
                "right",
                "shoulder_1",
                friction=TRUE.friction_params(),
                stribeck=TRUE.stribeck_params(0.8),
                hub_serial="hub1",
                path=path,
            )
            got = load_calibration(path, expected_hub_serial="hub1")["right"][
                "shoulder_1"
            ]
            self.assertEqual(got["friction"]["fl"], 0.045)
            self.assertEqual(got["stribeck_gain"], 0.8)
            self.assertEqual(got["stribeck_vs"], 0.06)
            jc = JointConfig(
                kp=1.0,
                kd=0.1,
                friction=FrictionParams(fc=0, k=0, fv=0, fo=0),
                mass=1.0,
                com=(0.0, 0.0, 0.0),
            )
            applied = _calibrated_joint(jc, got)
            self.assertEqual(applied.friction.fl, 0.045)
            self.assertEqual(applied.stribeck_dfs, 0.4)
            self.assertEqual(applied.stribeck_load_gain, 0.04)
            # Negative or junk stribeck values are dropped on load.
            raw = json.loads(path.read_text())
            raw["right"]["shoulder_1"]["stribeck_vs"] = -1
            path.write_text(json.dumps(raw))
            got = load_calibration(path, expected_hub_serial="hub1")["right"][
                "shoulder_1"
            ]
            self.assertNotIn("stribeck_vs", got)
            with self.assertRaises(ValueError):
                update_joint_calibration(
                    "right",
                    "shoulder_1",
                    stribeck={"bogus": 1.0},
                    hub_serial="hub1",
                    path=path,
                )

    def test_a_retired_cogging_series_is_ignored_and_scrubbed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            cogging = {"period_deg": 3.66, "harmonics": [[2, 0.06, 0.0]]}
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "hub_serial": "hub1",
                        "right": {"elbow": {"kp": 200.0, "cogging": cogging}},
                    }
                )
            )
            got = load_calibration(path, expected_hub_serial="hub1")["right"]["elbow"]
            self.assertEqual(got, {"kp": 200.0})
            # The next save of that joint drops it from the file.
            update_joint_calibration(
                "right", "elbow", kd=5.0, hub_serial="hub1", path=path
            )
            entry = json.loads(path.read_text())["right"]["elbow"]
            self.assertNotIn("cogging", entry)
            self.assertEqual(entry["kp"], 200.0)


if __name__ == "__main__":
    unittest.main()
