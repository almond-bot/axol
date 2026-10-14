"""The tuners' joint frame survives a ±360° multi-turn boot wrap.

Right elbow, 2026-09-22: after the tuner's mode-switch reset the motor
re-derived its multi-turn angle a full turn off (−213.7° for a joint at
146.3°), the fixed motor→joint offset turned that into −363.7° in the joint
frame, and the homing ramp drove the motor a full turn into its hard stop at
40 Nm until the stall protection tripped. The proxy now re-derives the wrap
on every read and the ramp refuses an implausible reading before commanding.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from almond_axol.cli.tune.friction import _ramp_verified
from almond_axol.constants import Joint
from almond_axol.robot.axol import closer_end_stop
from almond_axol.tuning.joint_frame import JointFrameMotor


class _FakeMotor:
    def __init__(self, joint: Joint, motor_pos_deg: float) -> None:
        self.joint = joint
        self.position = math.radians(motor_pos_deg)
        self.commands: list[float] = []

    async def get_position(self) -> float:
        return self.position

    async def set_position_velocity(self, position: float, max_speed: float) -> None:
        self.commands.append(position)


class WrapAwareJointFrameTest(unittest.IsolatedAsyncioTestCase):
    async def test_wrapped_reading_is_corrected_and_commands_use_the_correction(
        self,
    ) -> None:
        offset = closer_end_stop(Joint.ELBOW, False)[0]
        # The motor really is at 146.3° but re-derived its angle as −213.7°.
        fake = _FakeMotor(Joint.ELBOW, -213.66)
        jm = JointFrameMotor(fake, offset, is_left=False)
        q = await jm.get_position()
        self.assertAlmostEqual(jm.wrap, math.tau, places=9)
        self.assertAlmostEqual(math.degrees(q), 146.34 + math.degrees(offset), places=1)
        self.assertLess(abs(math.degrees(q)), 10.0)  # a few degrees off rest, not −363
        # Commanding rest sends a motor-frame target next to the wrapped
        # reading, not a full turn away from it.
        await jm.set_position_velocity(0.0, 0.5)
        self.assertAlmostEqual(
            math.degrees(fake.commands[-1] - fake.position), -math.degrees(q), places=1
        )
        self.assertLess(abs(math.degrees(fake.commands[-1] - fake.position)), 10.0)

    async def test_unwrapped_reading_needs_no_correction(self) -> None:
        offset = closer_end_stop(Joint.ELBOW, False)[0]
        fake = _FakeMotor(Joint.ELBOW, 146.3)
        jm = JointFrameMotor(fake, offset, is_left=False)
        q = await jm.get_position()
        self.assertEqual(jm.wrap, 0.0)
        self.assertLess(abs(math.degrees(q)), 10.0)
        self.assertAlmostEqual(jm.frame_offset, offset)

    async def test_proxy_without_side_keeps_the_legacy_fixed_offset(self) -> None:
        fake = _FakeMotor(Joint.ELBOW, -213.66)
        jm = JointFrameMotor(fake, 0.5)
        self.assertAlmostEqual(await jm.get_position(), fake.position + 0.5)
        self.assertEqual(jm.wrap, 0.0)


class RampRefusesImplausibleReadingsTest(unittest.IsolatedAsyncioTestCase):
    async def test_ramp_reads_first_and_refuses_a_reading_outside_the_limits(
        self,
    ) -> None:
        sent: list[float] = []

        async def get_position() -> float:
            return math.radians(-363.7)

        async def set_position_velocity(position: float, max_speed: float) -> None:
            sent.append(position)

        elbow = SimpleNamespace(
            _is_left=False,
            get_position=get_position,
            set_position_velocity=set_position_velocity,
        )
        with self.assertRaisesRegex(RuntimeError, "implausible"):
            await _ramp_verified({Joint.ELBOW: elbow}, {Joint.ELBOW: 0.0})
        self.assertEqual(sent, [])


if __name__ == "__main__":
    unittest.main()
