"""The friction sweeps carry no gravity feedforward, so a loaded joint sits
``g/kp`` short of its target. Every ramp and pass must start where the last
command left the target: starting at the measured position drops that spring
torque at once, and the arm falls and bounces at every turnaround."""

from __future__ import annotations

import math
import unittest
from unittest import mock

from almond_axol.cli.tune import friction
from almond_axol.constants import Joint

_SAG = math.radians(5.0)  # 21 Nm on kp 250


class _SaggingMotor:
    """Joint-frame double of a loaded joint: it sits ``_SAG`` short of its
    target, and records the target program of every experiment."""

    def __init__(self, position: float) -> None:
        self.target = position  # on its gravity-fed hold: no sag yet
        self.sag = 0.0
        self.hold: float | None = position
        self.programs: list[list[float]] = []

    async def get_position(self) -> float:
        return self.target - self.sag

    async def run_experiment(self, *, samples, **_kw) -> list[dict]:
        targets = [s[0] for s in samples]
        self.programs.append(targets)
        self.target = targets[-1]
        self.sag = _SAG  # from the first experiment on, the spring carries it
        return [{"actual": t - _SAG, "torque": 1.0} for t in targets]


async def _no_sleep(_s: float) -> None:
    return None


class SweepTurnaroundTest(unittest.IsolatedAsyncioTestCase):
    def _assert_target_never_steps(self, motor: _SaggingMotor) -> None:
        self.assertGreater(len(motor.programs), 3)
        for before, after in zip(motor.programs, motor.programs[1:]):
            # A pass ends a sample short of its end point: thousandths of a
            # degree, against the 5° the measured position is off by.
            self.assertLess(
                abs(after[0] - before[-1]),
                math.radians(0.02),
                "a move started off the last commanded target "
                f"by {math.degrees(after[0] - before[-1]):+.2f}°",
            )

    async def test_slow_profile_starts_every_move_at_the_last_target(self) -> None:
        motor = _SaggingMotor(0.0)
        with mock.patch.object(friction.asyncio, "sleep", _no_sleep):
            rows = await friction._identify_slow(
                motor,
                250.0,
                3.5,
                [math.radians(2.0), math.radians(8.0)],
                -0.6,
                0.6,
                lambda _q: 21.0,
                Joint.SHOULDER_1,
                True,
                None,
            )
        self.assertTrue(rows["q"])
        self._assert_target_never_steps(motor)

    async def test_standard_profile_starts_every_move_at_the_last_target(self) -> None:
        motor = _SaggingMotor(0.0)
        with mock.patch.object(friction.asyncio, "sleep", _no_sleep):
            await friction._identify_joint(
                motor,
                Joint.ELBOW,
                130.0,
                5.0,
                True,
                [math.radians(18.0), math.radians(36.0)],
            )
        self._assert_target_never_steps(motor)

    async def test_first_ramp_leaves_the_hold_at_the_measured_position(self) -> None:
        motor = _SaggingMotor(0.3)
        await friction._ramp_to(motor, 250.0, 3.5, 0.5, duration=0.1)
        self.assertAlmostEqual(motor.programs[0][0], 0.3, 9)
        self.assertAlmostEqual(motor.programs[0][-1], 0.5, 9)

    async def test_a_start_far_from_the_joint_is_not_trusted(self) -> None:
        # A stale target a long way off would step the joint at full stiffness.
        motor = _SaggingMotor(0.3)
        await friction._ramp_to(motor, 250.0, 3.5, 0.5, duration=0.1, start=1.4)
        self.assertAlmostEqual(motor.programs[0][0], 0.3, 9)

    def test_a_pass_too_short_to_run_leaves_the_target_where_it_was(self) -> None:
        self.assertEqual(friction._swept_to(0.20, 0.21), 0.20)
        self.assertEqual(friction._swept_to(0.20, 0.50), 0.50)


if __name__ == "__main__":
    unittest.main()
