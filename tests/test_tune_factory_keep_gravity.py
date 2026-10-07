"""tune.factory --keep-gravity: a robot whose gravity comp was tuned by hand in
the panel settings keeps it — the fits run against those link masses / CoMs,
and only friction, Stribeck and Fo are saved."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

from almond_axol.constants import Joint
from almond_axol.robot.config import AxolConfig
from almond_axol.serve.settings import SettingsStore
from almond_axol.tuning import sweep_safety
from almond_axol.tuning.friction_model import FrictionFit

# A customer robot's settings.json (version 1): gripperless, hand-tuned link
# masses and CoMs, plus unrelated knobs (an elbow kp) that are not gravity.
CUSTOMER_SETTINGS = {
    "advanced": {
        "axol.experiments.stiction_gain": "0.8",
        "axol.left.elbow.com": [-0.0256064, 0.04, -0.072044],
        "axol.left.elbow.kp": "180",
        "axol.left.shoulder_1.mass": "1.3",
        "axol.left.shoulder_3.mass": "3.5",
        "axol.left.wrist_3.com": [-0.0285, 0.01, -0.106],
        "axol.left.wrist_3.mass": "0.969",
        "axol.right.elbow.com": [0.0256064, 0.04, -0.072044],
        "axol.right.elbow.kp": "180",
        "axol.right.shoulder_1.mass": "1.3",
        "axol.right.shoulder_3.mass": "3.5",
        "axol.right.wrist_3.com": [0.0285, 0.01, -0.106],
        "axol.right.wrist_3.mass": "0.969",
    },
    "cameras": None,
    "values": {"robot.has_gripper": False},
    "version": 1,
}

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


class KeepGravityTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "settings.json"
        path.write_text(json.dumps(CUSTOMER_SETTINGS))
        store = SettingsStore(path, strict=True)
        patch = mock.patch("almond_axol.settings.load_store", return_value=store)
        patch.start()
        self.addCleanup(patch.stop)

    def test_reads_exactly_the_hand_tuned_links(self) -> None:
        from almond_axol.cli.tune import factory

        got = factory.settings_link_overrides(["left", "right"])
        want = {
            "left": {
                "shoulder_1": {"mass": 1.3},
                "shoulder_3": {"mass": 3.5},
                "elbow": {"com": (-0.0256064, 0.04, -0.072044)},
                "wrist_3": {"mass": 0.969, "com": (-0.0285, 0.01, -0.106)},
            },
            "right": {
                "shoulder_1": {"mass": 1.3},
                "shoulder_3": {"mass": 3.5},
                "elbow": {"com": (0.0256064, 0.04, -0.072044)},
                "wrist_3": {"mass": 0.969, "com": (0.0285, 0.01, -0.106)},
            },
        }
        self.assertEqual(set(got), set(want))
        for side in want:
            self.assertEqual(set(got[side]), set(want[side]), side)
            for joint, link in want[side].items():
                for key, value in link.items():
                    np.testing.assert_allclose(got[side][joint][key], value)
        # One arm only: the other side's links are not touched.
        self.assertEqual(set(factory.settings_link_overrides(["right"])), {"right"})

    def test_slow_profile_saves_friction_but_no_gravity(self) -> None:
        from almond_axol.cli.tune import factory
        from almond_axol.cli.tune.gravity import _model_torques

        joint, is_left = Joint.SHOULDER_1, False
        other, lo_d, hi_d, _ = sweep_safety(joint, is_left)
        cfg = AxolConfig()

        async def fake_sweep(motor, kp, kd, speeds, lo, hi, load, jt, left, raw):
            rows = {k: [] for k in ("speed", "direction", "q", "tau", "group", "load")}
            for g, v in enumerate(speeds):
                span = (lo, hi) if math.degrees(v) >= 8 else (lo, lo + 0.3 * (hi - lo))
                for d in (1, -1):
                    for q in np.linspace(*span, 300):
                        gq = float(
                            _model_torques(cfg, jt, left, np.array([q]), other)[0]
                        )
                        h = float(TRUE.halfdiff(np.array([v]), gq)[0])
                        rows["speed"].append(v)
                        rows["direction"].append("+" if d > 0 else "-")
                        rows["q"].append(q)
                        rows["tau"].append(gq + d * h)
                        rows["group"].append(g)
                        rows["load"].append(load(q))
            return rows

        saved: dict = {}
        jc = cfg.resolved().right.shoulder_1
        args = argparse.Namespace(
            profile="slow", raw_dir=None, stribeck_gain=None, keep_gravity=True
        )
        no_fit = mock.Mock(
            side_effect=AssertionError("CoM fitted under --keep-gravity")
        )
        with (
            mock.patch.object(factory, "_identify_slow", fake_sweep),
            mock.patch.object(factory, "_park_joint", mock.AsyncMock()),
            mock.patch.object(factory, "fit_com", no_fit),
            mock.patch.object(
                factory, "update_joint_calibration", lambda s, j, **kw: saved.update(kw)
            ),
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
                    250.0,
                    3.5,
                    other,
                    lo_d,
                    hi_d,
                )
            )
        no_fit.assert_not_called()
        self.assertIn("friction", entry)
        self.assertIn("stribeck_dfs", entry)
        self.assertNotIn("com", entry)
        self.assertNotIn("mass", entry)
        self.assertIsNone(saved["com"])
        self.assertIsNone(saved["mass"])
        self.assertIsNotNone(saved["friction"])


if __name__ == "__main__":
    unittest.main()
