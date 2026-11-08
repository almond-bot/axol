"""Lifting box mode's spring caps doesn't freeze the arm in the core.

With a box squeezed, the spring-cap back-off holds the sent command well
short of the caller's target. Lifting the caps (box mode off) used to send
the target as is: a step the realtime core rejects on ``max_step_rad``, then
rejects every target after it, so the arm stayed frozen.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from almond_axol.constants import ARM_JOINTS, Joint
from almond_axol.robot.axol import AxolArm

N = len(ARM_JOINTS)


def _arm(max_step: float = 0.5) -> AxolArm:
    arm = object.__new__(AxolArm)
    arm._config = SimpleNamespace(max_step_rad=max_step)
    arm._spring_caps = {}
    arm._last_sent_arm = None
    arm._last_unbacked_arm = None
    arm._release_offset = None
    arm._limits_lo = np.full(N + 1, -10.0)
    arm._limits_hi = np.full(N + 1, 10.0)
    return arm


def _q(elbow: float) -> np.ndarray:
    q = np.zeros(N + 1)
    q[ARM_JOINTS.index(Joint.ELBOW)] = elbow
    return q


class SpringCapReleaseTest(unittest.TestCase):
    def _send(self, arm: AxolArm, q: np.ndarray, backed: np.ndarray | None = None):
        arm._back_off_to_spring_caps = lambda _q: backed if backed is not None else _q
        out = arm._shape_sink_command(q)
        arm._last_sent_arm = out[:N].astype(np.float64)
        return out

    def test_lifting_the_caps_keeps_the_arm_where_it_stopped(self) -> None:
        arm = _arm()
        arm._spring_caps = {Joint.SHOULDER_1: 5.0}
        self._send(arm, _q(0.8), backed=_q(0.0))
        arm.set_spring_caps(None)
        out = self._send(arm, _q(0.8))
        self.assertAlmostEqual(out[ARM_JOINTS.index(Joint.ELBOW)], 0.0)

    def test_the_offset_wears_away_as_the_target_moves(self) -> None:
        arm = _arm()
        arm._spring_caps = {Joint.SHOULDER_1: 5.0}
        self._send(arm, _q(0.8), backed=_q(0.0))
        arm.set_spring_caps(None)
        self._send(arm, _q(0.8))
        out = self._send(arm, _q(0.6))  # moved 0.2 back: offset 0.8 -> 0.6
        self.assertAlmostEqual(out[ARM_JOINTS.index(Joint.ELBOW)], 0.0)
        out = self._send(arm, _q(0.0))  # moved 0.6 more: offset gone
        self.assertAlmostEqual(out[ARM_JOINTS.index(Joint.ELBOW)], 0.0)
        self.assertIsNone(arm._release_offset)
        out = self._send(arm, _q(0.1))
        self.assertAlmostEqual(out[ARM_JOINTS.index(Joint.ELBOW)], 0.1)

    def test_a_sent_step_never_exceeds_the_cores_limit(self) -> None:
        arm = _arm(max_step=0.5)
        arm._spring_caps = {Joint.SHOULDER_1: 5.0}
        self._send(arm, _q(0.0))
        out = self._send(arm, _q(0.45), backed=_q(0.9))  # back-off jumped
        self.assertAlmostEqual(out[ARM_JOINTS.index(Joint.ELBOW)], 0.4)


if __name__ == "__main__":
    unittest.main()
