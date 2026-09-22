"""Sweep-safety clearance poses stay inside each arm's joint limits.

Right shoulder_1 was driven into its +90° hard stop by a shoulder_3 sweep on
2026-09-22: the humerus-horizontal raise was a left-arm value applied to the
mirrored right-arm frame. Every clearance target of every sweep, both arms,
must lie strictly inside the arm's limits, and the mirrored joints' targets
must mirror.
"""

from __future__ import annotations

import math
import unittest

from almond_axol.constants import ARM_JOINTS, Joint
from almond_axol.robot.axol import arm_limits
from almond_axol.tuning.runner import probe_clearance_targets, sweep_safety


class ClearancePosesTest(unittest.TestCase):
    def test_every_clearance_target_is_inside_its_arm_limits(self) -> None:
        margin = math.radians(1.0)
        for is_left in (True, False):
            for joint in ARM_JOINTS:
                targets, _lo, _hi, _notes = sweep_safety(joint, is_left)
                targets = {**probe_clearance_targets(joint, is_left), **targets}
                for j, q in targets.items():
                    lo, hi = arm_limits(j, is_left)
                    with self.subTest(
                        arm="left" if is_left else "right",
                        sweep=joint.value,
                        held=j.value,
                    ):
                        self.assertGreater(q, lo + margin)
                        self.assertLess(q, hi - margin)

    def test_shoulder_3_sweep_raises_shoulder_1_mirrored(self) -> None:
        left, *_ = sweep_safety(Joint.SHOULDER_3, True)
        right, *_ = sweep_safety(Joint.SHOULDER_3, False)
        self.assertAlmostEqual(left[Joint.SHOULDER_1], math.radians(90.0))
        self.assertAlmostEqual(right[Joint.SHOULDER_1], -math.radians(90.0))
        # And the note tells the operator the signed value the arm will take.
        _, _, _, notes = sweep_safety(Joint.SHOULDER_3, False)
        self.assertTrue(any("-90°" in n for n in notes))


if __name__ == "__main__":
    unittest.main()
