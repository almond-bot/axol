"""ROM soak keeps every sweep ROM_LIMIT_MARGIN inside each joint limit.

Commanding the exact limit parks a joint against its end of travel for the
whole waypoint pause, which overtorques the motors over a two-hour soak.
"""

from __future__ import annotations

import asyncio
import math
import unittest
from unittest.mock import patch

import numpy as np

from almond_axol.constants import ARM_JOINTS
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

    def test_no_sweep_extreme_comes_within_the_margin_of_a_limit(self) -> None:
        for mirror in (True, False):
            for arm, q in self._sweep_targets(mirror):
                for joint in ARM_JOINTS:
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

    def test_sweeps_still_reach_the_inset_extremes(self) -> None:
        targets = self._sweep_targets(True)
        for joint in ARM_JOINTS:
            reached = {round(float(q[rom.JOINT_INDEX[joint]]), 6) for _, q in targets}
            for arm in ("left", "right"):
                low, high = arm_limits(joint, arm == "left")
                inset = {
                    round(low + rom.ROM_LIMIT_MARGIN, 6),
                    round(high - rom.ROM_LIMIT_MARGIN, 6),
                }
                with self.subTest(joint=joint.value, arm=arm):
                    self.assertTrue(reached & inset)


if __name__ == "__main__":
    unittest.main()
