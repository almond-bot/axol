"""ROM soak keeps the wrist_2/wrist_3 sweeps ROM_LIMIT_MARGIN inside their limits.

Commanding the exact limit parks those joints against their end of travel for
the whole waypoint pause, which overtorques the motors over a two-hour soak.
Every other joint still sweeps to its full limit.
"""

from __future__ import annotations

import asyncio
import math
import unittest
from unittest.mock import patch

import numpy as np

from almond_axol.constants import ARM_JOINTS, Joint
from almond_axol.diagnostics.rom import enable as rom
from almond_axol.robot.axol import arm_limits


class RomLimitMarginTest(unittest.TestCase):
    def _sweep_targets(self, shoulder3_mirror: bool) -> list[tuple[str, np.ndarray]]:
        targets: list[tuple[str, np.ndarray]] = []

        async def record(robot, left_q, right_q, left_t, right_t, speed, pause):
            targets.append(("left", left_t.copy()))
            targets.append(("right", right_t.copy()))
            return left_t.copy(), right_t.copy()

        with patch.object(rom, "sweep_to_target", side_effect=record):
            asyncio.run(
                rom.run_rom_cycle(
                    robot=None,
                    left_q=rom.home_pose(),
                    right_q=rom.home_pose(),
                    speed=1.0,
                    pre_pose_speed=1.0,
                    pause=0.0,
                    shoulder3_mirror=shoulder3_mirror,
                )
            )
        return targets

    def test_margin_is_five_degrees(self) -> None:
        self.assertAlmostEqual(rom.ROM_LIMIT_MARGIN, math.radians(5))

    def test_only_wrist_2_and_wrist_3_are_inset(self) -> None:
        self.assertEqual(rom.ROM_INSET_JOINTS, {Joint.WRIST_2, Joint.WRIST_3})

    def test_no_inset_sweep_extreme_comes_within_the_margin_of_a_limit(
        self,
    ) -> None:
        for mirror in (True, False):
            for arm, q in self._sweep_targets(mirror):
                for joint in rom.ROM_INSET_JOINTS:
                    low, high = arm_limits(joint, arm == "left")
                    value = float(q[rom.JOINT_INDEX[joint]])
                    if value == 0.0:
                        # Home, not a sweep extreme (the elbow's home is its
                        # lower limit; every sweep returns there, as teleop
                        # rest does).
                        continue
                    with self.subTest(mirror=mirror, arm=arm, joint=joint.value):
                        self.assertGreaterEqual(
                            value, low + rom.ROM_LIMIT_MARGIN - 1e-6
                        )
                        self.assertLessEqual(value, high - rom.ROM_LIMIT_MARGIN + 1e-6)

    def test_sweeps_reach_the_expected_extremes(self) -> None:
        targets = self._sweep_targets(True)
        for joint in ARM_JOINTS:
            margin = rom.ROM_LIMIT_MARGIN if joint in rom.ROM_INSET_JOINTS else 0.0
            reached = {round(float(q[rom.JOINT_INDEX[joint]]), 6) for _, q in targets}
            for arm in ("left", "right"):
                low, high = arm_limits(joint, arm == "left")
                expected = {round(low + margin, 6), round(high - margin, 6)}
                with self.subTest(joint=joint.value, arm=arm):
                    self.assertTrue(reached & expected)


if __name__ == "__main__":
    unittest.main()
