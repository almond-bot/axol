from __future__ import annotations

import contextlib
import copy
import dataclasses
import io
import sys
import tempfile
import unittest
from unittest import mock

import scripts.mantis_ik_bench as benchmark
from almond_axol.teleop.config import VRTeleopConfig
from scripts.mantis_ik_bench import (
    PRESETS,
    _schedule_configs,
    _trajectory_seed,
    check_spec,
)


def _benchmark_result() -> dict:
    trajectories = {}
    for name in benchmark.SPEC_TRAJS:
        trajectories[name] = {
            "tip_mm": {"mean": 1.0, "p95": 2.0, "max": 3.0},
            "ori_deg": {"p95": 1.0},
            "solve_ms": {"p95": 1.0},
        }
    trajectories["hold"]["hold_drift_rad_s"] = 0.0
    return {"init_s": 0.0, "trajs": trajectories}


class MantisIkBenchmarkTest(unittest.TestCase):
    def test_reference_relaxes_only_runtime_bounds(self) -> None:
        shipping = PRESETS["mantis"]
        expected = dataclasses.replace(
            shipping,
            max_joint_delta=10.0,
            max_iterations=64,
        )
        self.assertEqual(PRESETS["oracle_mantis"], expected)

    def test_trajectory_tremor_seed_is_stable(self) -> None:
        self.assertEqual(_trajectory_seed("hold"), 53_326_568)
        self.assertEqual(_trajectory_seed("slow_wave"), 1_056_714_365)
        self.assertEqual(_trajectory_seed("reach"), 1_986_086_381)

    def test_production_run_always_schedules_fresh_reference_first(self) -> None:
        scheduled = _schedule_configs({"mantis": PRESETS["mantis"]})
        self.assertEqual(list(scheduled), ["oracle_mantis", "mantis"])

        reversed_explicit = _schedule_configs(
            {
                "mantis": PRESETS["mantis"],
                "oracle_mantis": PRESETS["oracle_mantis"],
            }
        )
        self.assertEqual(list(reversed_explicit), ["oracle_mantis", "mantis"])

    def test_benchmark_config_discards_host_tracker_transforms(self) -> None:
        transform = [0.047, 0.0, 0.046, 0.0, 0.0, 0.0, 1.0]

        def seed_host_transforms(config: VRTeleopConfig) -> None:
            config.tcp_transform_left = transform
            config.tcp_transform_right = transform

        with mock.patch.object(
            benchmark,
            "apply_mantis_teleop_profile",
            side_effect=seed_host_transforms,
        ):
            config = benchmark._benchmark_teleop_config(120.0)

        self.assertEqual(config.frequency, 120.0)
        self.assertIsNone(config.tcp_transform_left)
        self.assertIsNone(config.tcp_transform_right)

    def test_spec_requires_reference_and_complete_trajectory_suite(self) -> None:
        production = _benchmark_result()
        ok, failures = check_spec(production, None)
        self.assertFalse(ok)
        self.assertIn("reference floor is missing", failures[0])

        floor = _benchmark_result()
        del production["trajs"]["pick_place"]
        ok, failures = check_spec(production, floor)
        self.assertFalse(ok)
        self.assertIn("pick_place: production result is missing", failures)

    def test_spec_rejects_non_finite_production_and_floor_metrics(self) -> None:
        production = _benchmark_result()
        floor = _benchmark_result()
        production["trajs"]["hold"]["tip_mm"]["p95"] = float("nan")
        ok, failures = check_spec(production, floor)
        self.assertFalse(ok)
        self.assertTrue(any("non-finite" in failure for failure in failures))

        production = _benchmark_result()
        floor = copy.deepcopy(floor)
        floor["trajs"]["hold"]["tip_mm"]["max"] = float("inf")
        ok, failures = check_spec(production, floor)
        self.assertFalse(ok)
        self.assertTrue(any("non-finite" in failure for failure in failures))

    def test_main_exits_nonzero_when_production_fails_spec(self) -> None:
        reference = _benchmark_result()
        production = _benchmark_result()
        production["trajs"]["hold"]["tip_mm"]["p95"] = float("nan")
        run_order: list[str] = []

        def fake_run(name: str, *_args: object) -> dict:
            run_order.append(name)
            return reference if name == "oracle_mantis" else production

        with tempfile.TemporaryDirectory() as output_dir:
            argv = [
                "mantis_ik_bench.py",
                "--out",
                output_dir,
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(benchmark, "make_trajectories", return_value={}),
                mock.patch.object(benchmark, "run_config", side_effect=fake_run),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                benchmark.main()

        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(run_order, ["oracle_mantis", "mantis"])


if __name__ == "__main__":
    unittest.main()
