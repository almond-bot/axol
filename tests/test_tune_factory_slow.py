"""tune.factory's slow profile: one sweep per joint feeds the runtime-law
friction fit (with load and Stribeck terms) and the gravity fit, and the
saved/uploaded entry carries all of it."""

from __future__ import annotations

import argparse
import asyncio
import io
import math
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np

from almond_axol.constants import Joint
from almond_axol.robot.config import AxolConfig
from almond_axol.tuning import sweep_safety
from almond_axol.tuning.friction_model import FrictionFit

TRUE = FrictionFit(
    fc=0.5,
    fl=0.05,
    k=100.0,
    fv=0.0,
    fo=0.0,
    dfs=0.7,
    ls=0.04,
    vs=0.08,
    rms=0.0,
    r2=1.0,
    n=0,
    load_span=0.0,
    speeds=(),
)


class SlowFactoryTest(unittest.TestCase):
    def test_entry_has_friction_load_stribeck_and_com(self) -> None:
        from almond_axol.cli.tune import factory

        joint, is_left = Joint.SHOULDER_1, False
        other, lo_d, hi_d, _ = sweep_safety(joint, is_left)
        from almond_axol.cli.tune.gravity import _model_torques, _with_com

        cfg = AxolConfig()
        com0 = np.array(cfg.right.shoulder_1.com)
        # The real link's CoM sits 10 mm off the CAD's along y (x runs along
        # shoulder_1's own axis and no sweep can see it).
        true_cfg = _with_com(cfg, is_left, joint, tuple(com0 + [0.0, 0.01, 0.0]))

        def gravity(q: float) -> float:
            return float(
                _model_torques(true_cfg, joint, is_left, np.array([q]), other)[0]
            )

        async def fake_sweep(motor, kp, kd, speeds, lo, hi, load, jt, left, raw):
            rows = {k: [] for k in ("speed", "direction", "q", "tau", "group", "load")}
            rng = np.random.default_rng(0)
            for g, v in enumerate(speeds):
                span = (lo, hi) if math.degrees(v) >= 8 else (lo, lo + 0.3 * (hi - lo))
                for d in (1, -1):
                    for q in np.linspace(*span, 600):
                        gq = gravity(q)
                        h = float(TRUE.halfdiff(np.array([v]), gq)[0])
                        rows["speed"].append(v)
                        rows["direction"].append("+" if d > 0 else "-")
                        rows["q"].append(q)
                        rows["tau"].append(gq + d * h + 0.02 * rng.standard_normal())
                        rows["group"].append(g)
                        rows["load"].append(load(q))
            return rows

        saved = {}

        def fake_save(side, jt, **kw):
            saved.update(kw, side=side, joint=jt)

        jc = AxolConfig().resolved().right.shoulder_1
        args = argparse.Namespace(profile="slow", raw_dir=None, stribeck_gain=None)
        with (
            mock.patch.object(factory, "_identify_slow", fake_sweep),
            mock.patch.object(factory, "_park_joint", mock.AsyncMock()),
            mock.patch.object(factory, "update_joint_calibration", fake_save),
            redirect_stdout(io.StringIO()),
        ):
            entry = asyncio.run(
                factory._calibrate_joint_slow(
                    {joint: None},
                    joint,
                    is_left,
                    [math.radians(v) for v in (1, 2, 3, 5, 8, 15, 30)],
                    "hub1",
                    args,
                    jc,
                    jc.kp,
                    jc.kd,
                    other,
                    lo_d,
                    hi_d,
                )
            )
        self.assertIsNotNone(entry)
        self.assertIn("com", entry)
        self.assertAlmostEqual(entry["com"][1] - com0[1], 0.01, delta=0.004)
        self.assertEqual(entry["stribeck_gain"], 0.8)  # shoulder_1's default
        self.assertGreater(entry["friction"]["fl"], 0.0)
        self.assertGreater(entry["stribeck_dfs"], 0.0)
        # The saved curve reproduces the true one where slow motion lives.
        fit = FrictionFit(
            fc=entry["friction"]["fc"],
            fl=entry["friction"]["fl"],
            k=100.0,
            fv=entry["friction"]["fv"],
            fo=0.0,
            dfs=entry["stribeck_dfs"],
            ls=entry["stribeck_load_gain"],
            vs=entry["stribeck_vs"],
            rms=0,
            r2=0,
            n=0,
            load_span=0,
            speeds=(),
        )
        for v_deg in (2, 5, 15):
            v = math.radians(v_deg)
            for g in (3.0, 10.0):
                self.assertAlmostEqual(
                    float(fit.halfdiff(np.array([v]), g)[0]),
                    float(TRUE.halfdiff(np.array([v]), g)[0]),
                    delta=0.08,
                )
        self.assertEqual(saved["stribeck"]["stribeck_gain"], 0.8)
        self.assertEqual(saved["hub_serial"], "hub1")


if __name__ == "__main__":
    unittest.main()
