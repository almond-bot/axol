"""The dataset/policy fps defaults agree across the ops that share a policy.

A policy runs at the fps its dataset was collected at (run-policy refuses a
mismatch), so collect-data, collect-dagger and run-policy must default to the
same rate — 30, halving the Orin's dataset encode/recorder load relative to
the 60 fps capture (see CollectDataConfig.fps).
"""

from __future__ import annotations

import dataclasses
import unittest

from almond_axol.cli.collect_dagger import DaggerConfig
from almond_axol.cli.collect_data import CollectDataConfig
from almond_axol.cli.inference_server import InferenceServerConfig
from almond_axol.cli.run_policy import RunPolicyConfig


def _default(cls: type, name: str) -> object:
    return next(f.default for f in dataclasses.fields(cls) if f.name == name)


class FpsDefaultsTest(unittest.TestCase):
    def test_dataset_and_policy_ops_default_to_the_same_30_fps(self) -> None:
        self.assertEqual(_default(CollectDataConfig, "fps"), 30)
        self.assertEqual(_default(DaggerConfig, "fps"), 30)
        self.assertEqual(_default(RunPolicyConfig, "fps"), 30)
        self.assertEqual(_default(InferenceServerConfig, "fps"), 30)

    def test_teleop_motion_rate_is_independent_of_the_dataset_rate(self) -> None:
        # Lowering the dataset rate must not slow the operator's commanded
        # motion: collect-data and dagger tick teleop at their own rate.
        self.assertEqual(_default(CollectDataConfig, "teleop_hz"), 120)
        self.assertEqual(_default(DaggerConfig, "teleop_hz"), 120)


if __name__ == "__main__":
    unittest.main()
