"""The calibration tools move and hold the arm on impedance only.

A customer's left shoulder_1 swung violently in ``tune.factory`` (2026-10-07):
the tools homed one joint at a time on the motors' own 0xA4 position loops,
with every other joint reset, brake released and unheld until its turn. Now
every joint is held on impedance (calibration gains, gravity fed forward)
before anything moves, and every move is an impedance ramp.
"""

from __future__ import annotations

import asyncio
import math
import re
import unittest
from pathlib import Path
from unittest import mock

from almond_axol.cli.tune import friction
from almond_axol.constants import ARM_JOINTS, Joint
from almond_axol.motor import ControlMode
from almond_axol.robot.axol import closer_end_stop
from almond_axol.tuning.joint_frame import JointFrameMotor

_TOOLS = ("factory", "friction", "gravity", "breakaway")


class _FakeMotor:
    """Motor-frame double: impedance commands land instantly; 0xA4 fails."""

    def __init__(self, joint: Joint, motor_pos: float, log: list) -> None:
        self.joint = joint
        self.position = motor_pos
        self.mode: ControlMode | None = None
        self.log = log

    async def get_position(self) -> float:
        return self.position

    async def set_control_mode(self, mode: ControlMode) -> None:
        self.mode = mode
        self.log.append(("mode", self.joint, mode))

    async def set_impedance(self, p_des, v_des, kp, kd, t_ff) -> None:
        assert self.mode == ControlMode.IMPEDANCE, "impedance frame before the mode"
        self.log.append(("imp", self.joint, p_des, kp, kd, t_ff))
        self.position = p_des

    async def set_position_velocity(self, *_a) -> None:
        raise AssertionError("a calibration tool sent a 0xA4 position command")


def _arm(joint_pos_deg: dict[Joint, float], log: list) -> dict[Joint, JointFrameMotor]:
    motors = {}
    for j in ARM_JOINTS:
        try:
            offset = closer_end_stop(j, True)[0]
        except Exception:  # noqa: BLE001 - either-stop wrists
            offset = 0.0
        q = math.radians(joint_pos_deg.get(j, 0.0))
        motors[j] = JointFrameMotor(
            _FakeMotor(j, q - offset, log), offset, is_left=True
        )
    return motors


class ImpedanceOnlyTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        for name, value in (("_RESET_SETTLE_S", 0.0), ("_RAMP_SPEED", 20.0)):
            p = mock.patch.object(friction, name, value)
            p.start()
            self.addCleanup(p.stop)

    async def test_every_joint_is_held_where_it_is_before_anything_moves(self) -> None:
        log: list = []
        start = {Joint.SHOULDER_1: -3.0, Joint.ELBOW: 20.0, Joint.WRIST_1: 5.0}
        motors = _arm(start, log)
        await friction._enter_impedance_hold(motors)
        modes = [e for e in log if e[0] == "mode"]
        holds = [e for e in log if e[0] == "imp"]
        self.assertEqual({e[1] for e in modes}, set(ARM_JOINTS))
        self.assertTrue(all(e[2] == ControlMode.IMPEDANCE for e in modes))
        # One hold per joint, at its current pose, at the calibration gains.
        self.assertEqual(
            sorted(e[1].value for e in holds), sorted(j.value for j in ARM_JOINTS)
        )
        for _, j, p_des, kp, kd, _t in holds:
            m = motors[j]
            self.assertAlmostEqual(
                p_des + m.frame_offset, math.radians(start.get(j, 0.0)), 9
            )
            self.assertEqual((kp, kd), friction.CAL_GAINS[j])
            self.assertAlmostEqual(m.hold, math.radians(start.get(j, 0.0)), 9)

    async def test_homing_moves_on_impedance_and_keeps_every_other_joint_held(
        self,
    ) -> None:
        log: list = []
        start = {Joint.SHOULDER_1: -20.0, Joint.SHOULDER_3: 15.0, Joint.ELBOW: 40.0}
        motors = _arm(start, log)
        await friction._enter_impedance_hold(motors)
        log.clear()
        await friction._home_all(motors)
        for j, m in motors.items():
            self.assertLess(
                abs(await m.get_position() - friction.rest_target(j, True)), 1e-6, j
            )
        # Every frame was an impedance command.
        self.assertTrue(all(e[0] == "imp" for e in log))
        # While the elbow moved (it homes before shoulder_1), shoulder_1 got a
        # hold every tick, and every one was its original pose.
        s1 = motors[Joint.SHOULDER_1]
        # The elbow's own move: the ticks where its command changes (it keeps
        # being held, unchanged, through the joints that home after it).
        elbow = [(i, e[2]) for i, e in enumerate(log) if e[1] == Joint.ELBOW]
        elbow_idx = [i for (i, p), (_, prev) in zip(elbow[1:], elbow) if p != prev]
        first, last = elbow_idx[0], elbow_idx[-1]
        s1_during = [e for e in log[first : last + 1] if e[1] == Joint.SHOULDER_1]
        self.assertGreaterEqual(len(s1_during), len(elbow_idx) - 1)
        for e in s1_during:
            self.assertAlmostEqual(e[2] + s1.frame_offset, math.radians(-20.0), 9)

    async def test_homing_after_an_interrupted_sweep_holds_where_the_joint_is(
        self,
    ) -> None:
        # A sweep moves shoulder_1 off its last hold (its ramp target) and is
        # cut short; homing must not step it back to that stale target.
        log: list = []
        motors = _arm({Joint.SHOULDER_1: -8.0}, log)
        await friction._enter_impedance_hold(motors)
        s1 = motors[Joint.SHOULDER_1]
        s1.motor.position = math.radians(-2.5) - s1.frame_offset  # mid-sweep
        log.clear()
        await friction._home_all(motors)
        first_s1 = next(e for e in log if e[1] == Joint.SHOULDER_1)
        self.assertAlmostEqual(math.degrees(first_s1[2] + s1.frame_offset), -2.5, 6)

    async def test_holds_feed_the_arms_gravity_forward(self) -> None:
        log: list = []
        motors = _arm({Joint.SHOULDER_1: -90.0}, log)  # arm out: loaded shoulder
        await friction._enter_impedance_hold(motors)
        s1 = [e for e in log if e[0] == "imp" and e[1] == Joint.SHOULDER_1][-1]
        self.assertGreater(abs(s1[5]), 1.0)  # Nm of gravity at the shoulder

    def test_no_calibration_tool_sends_a_position_loop_command(self) -> None:
        root = Path(friction.__file__).parent
        for tool in _TOOLS:
            src = (root / f"{tool}.py").read_text()
            code = "\n".join(
                line for line in src.splitlines() if not line.lstrip().startswith("#")
            )
            self.assertIsNone(
                re.search(r"POSITION_VELOCITY|set_position_velocity", code),
                f"{tool}.py still drives a joint on its firmware position loop",
            )


if __name__ == "__main__":
    asyncio.run(unittest.main())
