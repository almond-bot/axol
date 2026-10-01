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

    applied: list = []

    async def motion_control(self, left=None, right=None) -> None:
        if left is not None and self.left is not None:
            self.left.step(left)
        if right is not None:
            extra = getattr(self.right, "extra_torque", None)
            _FakeAxol.applied.append(None if extra is None else np.array(extra))
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


class _FakeImu:
    """A live wrist IMU reading a 2 Hz vertical shake on the fake clock."""

    def __init__(self, sides, enabled=True, live=False, **_kw) -> None:
        self.sides = list(sides) if enabled else []
        self.live = live

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def poll(self, side: str) -> np.ndarray:
        t = _FakeAxol.clock.now
        az = 9.80665 + 0.3 * math.sin(2 * math.pi * 2.0 * t)
        # ... and a 2 Hz flex rotation about the vertical the fake FK's
        # rotation axis shares (5°/s).
        gz = 5.0 * math.sin(2 * math.pi * 2.0 * t)
        return np.array([[t, 0.0, 0.0, az, 0.0, 0.0, gz]])

    def run_blocks(self, t0, t1, origin=None):
        return {}, {}


def _ee_rotations(rows):
    rows = np.asarray(rows, dtype=float).reshape(-1, 14)
    a = rows[:, 7] + rows[:, 10]
    c, sn = np.cos(a), np.sin(a)
    rot = np.zeros((len(rows), 3, 3))
    rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1] = c, -sn, sn, c
    rot[:, 2, 2] = 1.0
    return np.repeat(np.eye(3)[None], len(rows), 0), rot


def _ee_positions(rows):
    rows = np.asarray(rows, dtype=float).reshape(-1, 14)
    z = 0.6 * np.sin(rows[:, 7]) + 0.3 * np.sin(rows[:, 7] + rows[:, 10])
    right = np.stack([np.zeros(len(rows)), np.zeros(len(rows)), z], 1)
    return np.zeros_like(right), right


class SessionTest(unittest.TestCase):
    def _run(self, argv: list[str], tmp: Path, imu: bool = False) -> str:
        from almond_axol.cli.tune import motion as cli
        from almond_axol.tuning import runs

        motion = _motion(tmp / "sim.npz")
        _FakeAxol.start = motion.q[0, 7:].astype(float)
        _FakeAxol.clock = _Clock()
        solver = SimpleNamespace(
            num_joints=14,
            left_indices=list(range(7)),
            right_indices=list(range(7, 14)),
            ee_positions=_ee_positions,
            ee_rotations=_ee_rotations,
        )
        _FakeAxol.applied = []
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
            ]
            + ([] if imu else ["--no-imu"])
            + argv
        )
        clock = _FakeAxol.clock
        runs_dir = tmp / "runs"
        out = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(cli, "Axol", _FakeAxol))
            if imu:
                stack.enter_context(mock.patch.object(cli, "WristImu", _FakeImu))
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

    def test_imu_damping_alternates_clamps_and_is_saved(self) -> None:
        from almond_axol.tuning import runs

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            text = self._run(
                [
                    "--imu-damp",
                    "200",
                    "--imu-damp-max",
                    "0.4",
                    "--imu-damp-alternate",
                    "--repeat",
                    "2",
                    "--label",
                    "damp",
                ],
                tmp,
                imu=True,
            )
            self.assertIn("IMU damping (right): 200 N·s/m", text)
            self.assertIn("IMU damping off this pass", text)
            self.assertIn("IMU damping ON this pass", text)
            applied = [a for a in _FakeAxol.applied if a is not None]
            self.assertTrue(applied)
            peak = np.abs(np.stack(applied)).max(axis=0)
            self.assertLessEqual(peak.max(), 0.4 + 1e-9)
            self.assertGreater(peak[0], 0.0)  # shoulder_1
            self.assertGreater(peak[3], 0.0)  # elbow
            self.assertEqual(peak[4], 0.0)  # wrist_1 untouched
            metas = [
                runs.load_run(p.name, self.runs_dir) for p in self.runs_dir.iterdir()
            ]
            by_pass = {m[0]["label"]: m for m in metas}
            on = by_pass["damp [2/2]"]
            off = by_pass["damp [1/2]"]
            self.assertEqual(on[0]["metrics"]["imu_damp"], 200.0)
            self.assertEqual(off[0]["metrics"]["imu_damp"], 0.0)
            self.assertIn("imu_damp", on[1])
            self.assertNotIn("imu_damp", off[1])
            self.assertEqual(on[1]["imu_damp"].shape[1], 9)

    def test_gyro_damping_needs_a_mount_then_clamps_and_is_saved(self) -> None:
        import json

        from almond_axol.tuning import runs

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            mount = tmp / "mount.json"
            with self.assertRaises(SystemExit) as err:
                self._run(
                    ["--gyro-damp", "5", "--gyro-mount", str(mount)], tmp, imu=True
                )
            self.assertIn("--gyro-mount-fit", str(err.exception))
            mount.write_text(json.dumps({"right": {"rotation": np.eye(3).tolist()}}))
            text = self._run(
                [
                    "--gyro-damp",
                    "5",
                    "--gyro-damp-joint",
                    "right.shoulder_1",
                    "--gyro-damp-joint",
                    "right.elbow=2",
                    "--gyro-mount",
                    str(mount),
                    "--imu-damp-max",
                    "0.3",
                    "--imu-damp-alternate",
                    "--repeat",
                    "2",
                    "--label",
                    "gyro",
                ],
                tmp,
                imu=True,
            )
            self.assertIn("gyro flex damping (right): shoulder_1 5, elbow 2", text)
            applied = [a for a in _FakeAxol.applied if a is not None]
            self.assertTrue(applied)
            peak = np.abs(np.stack(applied)).max(axis=0)
            self.assertLessEqual(peak.max(), 0.3 + 1e-9)
            self.assertGreater(peak[0], 0.0)
            self.assertEqual(peak[1], 0.0)  # shoulder_2 not damped
            metas = [
                runs.load_run(p.name, self.runs_dir) for p in self.runs_dir.iterdir()
            ]
            by_pass = {m[0]["label"]: m for m in metas}
            self.assertEqual(by_pass["gyro [2/2]"][0]["metrics"]["gyro_damp"], 5.0)
            self.assertEqual(by_pass["gyro [1/2]"][0]["metrics"]["gyro_damp"], 0.0)

    def test_torque_probe_drives_only_its_joint_within_its_amplitude(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            mount = tmp / "mount.json"
            mount.write_text(json.dumps({"right": {"rotation": np.eye(3).tolist()}}))
            with self.assertRaises(SystemExit):
                self._run(
                    ["--torque-probe", "right.elbow=2", "--gyro-mount", str(mount)],
                    tmp,
                    imu=True,
                )
            text = self._run(
                [
                    "--torque-probe",
                    "right.elbow=0.3",
                    "--gyro-mount",
                    str(mount),
                    "--label",
                    "probe",
                ],
                tmp,
                imu=True,
            )
            self.assertIn("torque probe (right): elbow 0.3 Nm", text)
            applied = np.stack([a for a in _FakeAxol.applied if a is not None])
            peak = np.abs(applied).max(axis=0)
            self.assertGreater(peak[3], 0.1)
            self.assertLessEqual(peak[3], 0.3 + 1e-9)
            self.assertEqual(float(np.delete(peak, 3).max()), 0.0)

    def test_encoder_damping_runs_without_the_imu(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            text = self._run(
                [
                    "--imu-damp",
                    "100",
                    "--imu-damp-source",
                    "encoder",
                    "--imu-damp-joint",
                    "right.shoulder_1",
                    "--imu-damp-max",
                    "0.3",
                    "--label",
                    "enc",
                ],
                Path(d),
            )
            self.assertIn("encoder damping (right): 100 N·s/m", text)
            applied = np.stack([a for a in _FakeAxol.applied if a is not None])
            peak = np.abs(applied).max(axis=0)
            self.assertGreater(peak[0], 0.0)
            self.assertLessEqual(peak[0], 0.3 + 1e-9)
            self.assertEqual(float(np.delete(peak, 0).max()), 0.0)

    def test_model_reference_damps_through_a_saved_tracking_model(self) -> None:
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
                text = self._run(
                    [
                        "--imu-damp",
                        "100",
                        "--imu-damp-ref",
                        "model",
                        "--imu-damp-joint",
                        "right.shoulder_1",
                        "--imu-damp-joint",
                        "right.elbow=0.6",
                        "--imu-damp-joint-lp",
                        "right.elbow=8",
                        "--label",
                        "model",
                    ],
                    tmp,
                    imu=True,
                )
            self.assertIn(
                "damping reference: the expected path through right.shoulder_1", text
            )
            self.assertIn("elbow ×0.6 (lp 8 Hz)", text)
            with mock.patch.object(
                tracking_model, "load_models", lambda path=store: real(store)
            ):
                text = self._run(
                    [
                        "--imu-damp",
                        "100",
                        "--imu-damp-source",
                        "encoder",
                        "--imu-damp-ref",
                        "model",
                        "--imu-damp-joint",
                        "right.shoulder_1",
                        "--label",
                        "encmodel",
                    ],
                    tmp,
                )
            self.assertIn("encoder height against the expected path", text)
            applied = [a for a in _FakeAxol.applied if a is not None]
            self.assertTrue(applied)

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
