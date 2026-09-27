"""A whole ``tune.motion`` session against a simulated arm: the pass loop,
``--learn``, ``--correction`` and ``--invert``, scoring and saving.

The robot is a fake whose right arm tracks each streamed command through a
lightly damped second-order loop with a delay, plus a repeatable
position-locked disturbance — the shape of the real arm's slow-motion error.
The clock is simulated, so a session of several passes runs in seconds.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import math
import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from almond_axol.constants import ARM_JOINTS
from almond_axol.tuning.motion import ReferenceMotion, save_motion

RATE = 240.0
WN, ZETA, DELAY = 2 * math.pi * 2.5, 0.15, 6  # 2.5 Hz, ζ 0.15, 25 ms


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def perf_counter(self) -> float:
        return self.now

    def time(self) -> float:
        return self.now

    def strftime(self, *a: object) -> str:
        return "20260927-000000"

    async def sleep(self, dt: float) -> None:
        self.now += max(float(dt), 0.0)


class _Arm:
    """Seven joints tracking their commands through the simulated loop."""

    def __init__(self, start: np.ndarray, clock: _Clock) -> None:
        self.clock = clock
        self.y = start.astype(float).copy()
        self.v = np.zeros(7)
        self.queue = [start.astype(float).copy()] * (DELAY + 1)
        self.positions = np.zeros(8, dtype=np.float32)
        self.torques = np.zeros(8, dtype=np.float32)
        self.motors = {j: SimpleNamespace(_feedback_ts=None) for j in ARM_JOINTS}
        self._publish()

    def _publish(self) -> None:
        # Position-locked disturbance: 0.25° at a 3° period — the same at
        # the same angle on every pass.
        d = math.radians(0.25) * np.sin(self.y * (2 * math.pi / math.radians(3.0)))
        self.positions[:7] = (self.y + d).astype(np.float32)

    def step(self, cmd: np.ndarray) -> None:
        self.queue.append(np.asarray(cmd[:7], dtype=float).copy())
        u = self.queue.pop(0)
        dt = 1.0 / RATE
        acc = WN * WN * (u - self.y) - 2 * ZETA * WN * self.v
        self.v += acc * dt
        self.y += self.v * dt
        self._publish()

    def torque_residuals(self) -> np.ndarray:
        return np.zeros(7)


class _FakeAxol:
    instances: list["_FakeAxol"] = []
    start = np.zeros(7)
    clock = _Clock()

    def __init__(self, config=None, record=None, loop_hz=None, **channels) -> None:
        self.left = (
            None if "left_channel" in channels else _Arm(np.zeros(7), self.clock)
        )
        self.right = _Arm(self.start, self.clock)
        _FakeAxol.instances.append(self)

    async def __aenter__(self) -> "_FakeAxol":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def motion_control(self, left=None, right=None) -> None:
        if left is not None and self.left is not None:
            self.left.step(left)
        if right is not None:
            self.right.step(right)

    def set_recording_engaged(self, on: bool) -> None:
        pass


def _plan(solver, q_from, q_to, speed, rate, min_duration):
    n = int(1.5 * rate)
    return [
        q_from + (q_to - q_from) * (0.5 - 0.5 * math.cos(math.pi * k / (n - 1)))
        for k in range(n)
    ]


def _motion(path: Path) -> ReferenceMotion:
    t = np.arange(int(12 * RATE)) / RATE
    q = np.zeros((len(t), 14), dtype=np.float32)
    ramp = np.clip(t / 2.0, 0, 1) * np.clip((t[-1] - t) / 2.0, 0, 1)
    q[:, 7] = (np.radians(10) * np.sin(2 * math.pi * 0.25 * t) * ramp).astype(
        np.float32
    )
    q[:, 10] = (
        np.radians(-30) + np.radians(8) * np.sin(2 * math.pi * 0.2 * t) * ramp
    ).astype(np.float32)
    m = ReferenceMotion(name="sim", q=q, rate=RATE, meta={})
    save_motion(m, path)
    return m


class SessionTest(unittest.TestCase):
    def _run(self, argv: list[str], tmp: Path) -> str:
        from almond_axol.cli.tune import motion as cli
        from almond_axol.tuning import runs

        motion = _motion(tmp / "sim.npz")
        _FakeAxol.start = motion.q[0, 7:].astype(float)
        _FakeAxol.clock = _Clock()
        solver = SimpleNamespace(
            num_joints=14, left_indices=list(range(7)), right_indices=list(range(7, 14))
        )
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        cli.add_parser(sub)
        args = parser.parse_args(
            [
                "tune.motion",
                "--motion",
                str(tmp / "sim.npz"),
                "--arms",
                "right",
                "--no-imu",
            ]
            + argv
        )
        clock = _FakeAxol.clock
        runs_dir = tmp / "runs"
        out = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(cli, "Axol", _FakeAxol))
            stack.enter_context(
                mock.patch(
                    "almond_axol.kinematics.solver.KinematicsSolver", lambda: solver
                )
            )
            stack.enter_context(
                mock.patch(
                    "almond_axol.teleop.trajectory.plan_collision_aware_trajectory",
                    _plan,
                )
            )
            stack.enter_context(mock.patch.object(cli, "time", clock))
            stack.enter_context(mock.patch("asyncio.sleep", clock.sleep))
            stack.enter_context(
                mock.patch.object(
                    cli, "save_run", partial(runs.save_run, runs_dir=runs_dir)
                )
            )
            stack.enter_context(
                mock.patch.object(
                    cli, "load_run", lambda rid: runs.load_run(rid, runs_dir)
                )
            )
            stack.enter_context(contextlib.redirect_stdout(out))
            asyncio.run(cli._run(args))
        self.runs_dir = runs_dir
        return out.getvalue()

    def _pass_errors(self, text: str) -> list[float]:
        return [
            float(line.split("band error ")[1].split(" mdeg")[0])
            for line in text.splitlines()
            if "learning: pass" in line and "band error" in line
        ]

    def test_learning_cuts_the_repeatable_error_and_the_correction_replays(
        self,
    ) -> None:
        from almond_axol.tuning import runs

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            text = self._run(
                ["--learn", "8", "--learn-gain", "0.9", "--label", "sim-learn"], tmp
            )
            errs = self._pass_errors(text)
            self.assertEqual(len(errs), 7, text[-2000:])
            self.assertLess(errs[-1], 0.5 * errs[0], errs)
            self.assertTrue(all(b <= a * 1.02 for a, b in zip(errs, errs[1:])), errs)
            ids = [p.name for p in self.runs_dir.iterdir()]
            self.assertEqual(len(ids), 8)

            def pass_no(rid: str) -> int:
                label = runs.load_run(rid, self.runs_dir)[0]["label"]
                return int(label.split("[")[1].split("/")[0])

            saved = sorted(ids, key=pass_no)
            metas = [runs.load_run(r, self.runs_dir) for r in saved]
            corr = [m[1].get("correction") for m in metas]
            self.assertTrue(all(c is not None for c in corr))
            self.assertEqual(float(np.abs(corr[0]).max()), 0.0)  # the baseline
            self.assertGreater(float(np.abs(corr[-1]).max()), 0.0)
            # Learned only the moving right-arm joints; wrists and left zero.
            moving = [7, 10]
            others = [i for i in range(14) if i not in moving]
            self.assertEqual(float(np.abs(corr[-1][:, others]).max()), 0.0)

            best = saved[-1]
            text = self._run(
                ["--correction", best, "--repeat", "2", "--label", "replay"], tmp
            )
            self.assertIn("correction:", text)

    def test_invert_streams_through_a_saved_model(self) -> None:
        from almond_axol.tuning import tracking_model

        model = tracking_model.TrackingModel(
            1.0, WN, ZETA, math.inf, DELAY / RATE, 0.3, 8.0, 0.0
        )
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            store = tmp / "tm.json"
            tracking_model.save_model("right.shoulder_1", model, store)
            real = tracking_model.load_models
            with mock.patch.object(
                tracking_model, "load_models", lambda path=store: real(store)
            ):
                text = self._run(["--invert", "--label", "inv"], tmp)
            self.assertIn("invert: right.shoulder_1 through its 2.50 Hz", text)
            self.assertNotIn("invert: right.elbow", text)


if __name__ == "__main__":
    unittest.main()
