"""ROM soak on a partial bench arm: only the selected joints need to be there.

A single-channel adapter with just a wrist assembly (wrist_2, wrist_3, gripper)
on the bus must be able to run ``diag.rom-enable --joints wrist_2,wrist_3,gripper``.
The pieces that make that work: the pre-flight presence probe and its
decision rule, an ``AxolArm`` restricted to the motors actually present, and
an ``RtAxol`` that configures/feeds only those motors (the core slots them by
motor id).
"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any
from unittest.mock import patch

import numpy as np

from almond_axol.constants import ARM_JOINTS, Joint
from almond_axol.diagnostics.rom import enable as rom
from almond_axol.motor import ControlMode, MotorError
from almond_axol.robot.axol import Axol
from almond_axol.rt import RtAxol
from almond_axol.serve.robot_link import scoped_motor_faults

WRIST_KIT = {Joint.WRIST_2, Joint.WRIST_3, Joint.GRIPPER}


# --------------------------------------------------------------------------- #
# Presence probe + decision rule                                              #
# --------------------------------------------------------------------------- #


class ResolveBusJointsTest(unittest.TestCase):
    def test_selected_subset_present_runs_with_just_those_motors(self) -> None:
        got = rom.resolve_bus_joints(set(WRIST_KIT), set(WRIST_KIT), set(Joint), "left")
        self.assertEqual(got, WRIST_KIT)

    def test_unselected_motors_that_answer_are_still_brought_up(self) -> None:
        on_bus = WRIST_KIT | {Joint.WRIST_1}
        got = rom.resolve_bus_joints(set(WRIST_KIT), on_bus, set(Joint), "left")
        self.assertEqual(got, on_bus)

    def test_missing_selected_joint_fails_by_name(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            rom.resolve_bus_joints(
                set(WRIST_KIT), {Joint.WRIST_2, Joint.GRIPPER}, set(Joint), "left"
            )
        self.assertIn("left arm's wrist_3 did not answer", str(ctx.exception))

    def test_full_selection_still_requires_the_whole_arm(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            rom.resolve_bus_joints(set(Joint), set(WRIST_KIT), set(Joint), "right")
        message = str(ctx.exception)
        for joint in set(Joint) - WRIST_KIT:
            self.assertIn(joint.value, message)

    def test_gripperless_candidates_never_demand_a_gripper(self) -> None:
        arm_only = set(ARM_JOINTS)
        got = rom.resolve_bus_joints(set(Joint), arm_only, arm_only, "left")
        self.assertEqual(got, arm_only)

    def test_empty_bus_fails(self) -> None:
        with self.assertRaises(SystemExit):
            rom.resolve_bus_joints({Joint.GRIPPER}, set(), set(Joint), "left")


class _ProbeBus:
    instances: list[_ProbeBus] = []

    def __init__(self, channel: str) -> None:
        self.channel = channel
        self.started = False
        self.closed = False
        _ProbeBus.instances.append(self)

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True


class _ProbeMotor:
    """Answers, refuses, or hangs, per joint (a hang is an absent Damiao)."""

    answering: set[Joint] = set()
    hanging: set[Joint] = set()

    def __init__(self, bus: Any, joint: Joint) -> None:
        self.joint = joint

    async def get_error_code(self) -> str:
        if self.joint in _ProbeMotor.hanging:
            await asyncio.sleep(60)
        if self.joint not in _ProbeMotor.answering:
            raise MotorError("no reply")
        return "OK"


class ProbeBusJointsTest(unittest.IsolatedAsyncioTestCase):
    async def test_probe_reports_answering_motors_and_releases_the_bus(self) -> None:
        _ProbeBus.instances = []
        _ProbeMotor.answering = set(WRIST_KIT)
        _ProbeMotor.hanging = {Joint.SHOULDER_1}
        with (
            patch.object(rom, "CanBus", _ProbeBus),
            patch.object(rom, "Motor", _ProbeMotor),
            patch.object(rom, "PROBE_TIMEOUT", 0.05),
        ):
            found = await rom.probe_bus_joints("can0", set(Joint))
        self.assertEqual(found, WRIST_KIT)
        (bus,) = _ProbeBus.instances
        self.assertEqual(bus.channel, "can0")
        self.assertTrue(bus.started and bus.closed)


# --------------------------------------------------------------------------- #
# Partial AxolArm / RtAxol                                                    #
# --------------------------------------------------------------------------- #


class _FakeBus:
    def __init__(self, channel: str) -> None:
        self._channel = channel

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass


class _FakeDriver:
    kp_max = 500.0
    kd_max = 5.0

    def __init__(self) -> None:
        self.impedance: list[tuple[float, ...]] = []
        self.position_force: list[tuple[float, ...]] = []

    def set_feedback_callback(self, _cb: Any) -> None:
        pass

    async def set_impedance(self, *args: float) -> None:
        self.impedance.append(args)

    async def set_position_force(self, *args: float) -> None:
        self.position_force.append(args)


def _partial_axol(joints: set[Joint]) -> Axol:
    with (
        patch("almond_axol.robot.axol.CanBus", _FakeBus),
        patch(
            "almond_axol.motor.motor.make_driver",
            side_effect=lambda *_a, **_k: _FakeDriver(),
        ),
    ):
        return Axol(left_channel="can0", right_channel=None, left_joints=joints)


class PartialAxolArmTest(unittest.IsolatedAsyncioTestCase):
    def test_only_present_motors_exist_and_arrays_keep_their_shape(self) -> None:
        arm = _partial_axol(set(WRIST_KIT)).left
        assert arm is not None
        self.assertEqual(set(arm.motors), WRIST_KIT)
        self.assertEqual(arm.present_joints, frozenset(WRIST_KIT))
        self.assertTrue(arm._has_gripper)
        # Absent joints have nothing to resolve or verify; present either-stop
        # wrists still detect their zero from the first reading.
        self.assertEqual(arm._unresolved_offsets, {Joint.WRIST_2, Joint.WRIST_3})
        self.assertEqual(arm._unverified_zeros, set())
        offsets = arm._joint_offsets
        for joint in set(Joint) - WRIST_KIT:
            self.assertEqual(offsets[list(Joint).index(joint)], 0.0)

        for joint in WRIST_KIT:
            arm.motors[joint]._position = 0.25
            arm.motors[joint]._torque = 0.5
        arm._unresolved_offsets.clear()
        arm._joint_offsets[:] = 0.0
        positions = arm.positions
        self.assertEqual(positions.shape, (8,))
        for joint in set(Joint) - WRIST_KIT:
            self.assertEqual(positions[list(Joint).index(joint)], 0.0)
        self.assertEqual(arm.torques.tolist()[5:7], [0.5, 0.5])

    def test_elbow_kit_keeps_its_fixed_stop_verification(self) -> None:
        arm = _partial_axol({Joint.ELBOW, Joint.WRIST_2}).left
        assert arm is not None
        self.assertFalse(arm._has_gripper)
        self.assertEqual(arm._unverified_zeros, {Joint.ELBOW})
        self.assertEqual(arm._unresolved_offsets, {Joint.WRIST_2})

    def test_gripperless_config_drops_a_requested_gripper(self) -> None:
        from almond_axol.robot.config import AxolConfig

        with (
            patch("almond_axol.robot.axol.CanBus", _FakeBus),
            patch(
                "almond_axol.motor.motor.make_driver",
                side_effect=lambda *_a, **_k: _FakeDriver(),
            ),
        ):
            axol = Axol(
                AxolConfig(has_gripper=False),
                left_channel="can0",
                right_channel=None,
                left_joints=set(WRIST_KIT),
            )
        assert axol.left is not None
        self.assertEqual(set(axol.left.motors), {Joint.WRIST_2, Joint.WRIST_3})

    async def test_motion_control_streams_all_eight_slots_to_the_core(self) -> None:
        arm = _partial_axol(set(WRIST_KIT)).left
        assert arm is not None
        arm._unresolved_offsets.clear()
        arm._joint_offsets[:] = 0.0
        shipped: list[list[tuple[float, ...]]] = []
        arm._command_sink = shipped.append
        q = np.zeros(8, dtype=np.float32)
        q[5] = 0.3
        q[7] = 1.0
        await arm.motion_control(q)
        (cmds,) = shipped
        self.assertEqual(len(cmds), 8)
        self.assertAlmostEqual(cmds[5][0], 0.3, places=5)
        # Absent joints ride along at the rest pose; the core ignores their
        # slots (no motor configured there).
        self.assertEqual(cmds[0][0], 0.0)

    async def test_classic_paths_only_command_present_motors(self) -> None:
        arm = _partial_axol(set(WRIST_KIT)).left
        assert arm is not None
        arm._unresolved_offsets.clear()
        arm._joint_offsets[:] = 0.0
        for joint, motor in arm.motors.items():
            motor._position = 0.0
            motor._torque = 0.0
            motor._feedback_ts = 1.0
            motor.mode = (
                ControlMode.POSITION_FORCE
                if joint == Joint.GRIPPER
                else ControlMode.IMPEDANCE
            )
        await arm.motion_control(np.zeros(8, dtype=np.float32))
        await arm.gravity_compensate()
        for joint in (Joint.WRIST_2, Joint.WRIST_3):
            self.assertEqual(len(arm.motors[joint]._driver.impedance), 2)
        self.assertEqual(len(arm.motors[Joint.GRIPPER]._driver.position_force), 2)


class PartialRtAxolTest(unittest.IsolatedAsyncioTestCase):
    def test_config_lists_only_present_motors(self) -> None:
        rt = RtAxol(_partial_axol(set(WRIST_KIT)))
        lines = rt._config_text().splitlines()
        joint_lines = [line for line in lines if line.startswith("joint ")]
        self.assertEqual(
            [line.split()[3:5] for line in joint_lines],
            [["wrist_2", "6"], ["wrist_3", "7"]],
        )
        self.assertIn("gripper 0 can0 8", lines)

    async def test_feedback_feed_fills_present_slots_and_ignores_the_rest(self) -> None:
        rt = RtAxol(_partial_axol(set(WRIST_KIT)))
        arm = rt.left
        assert arm is not None
        feed = rt._make_feedback_feed()
        # Slot 5 = wrist_2, slot 7 = gripper; slot 0 (shoulder_1) has no motor.
        feed(
            0,
            {0: (9.0, 0.0, 0.0, 1.0), 5: (0.4, 0.0, 0.1, 1.0), 7: (2.0, 0.0, 0.0, 1.0)},
        )
        self.assertEqual(arm.motors[Joint.WRIST_2].position, 0.4)
        self.assertEqual(arm.motors[Joint.GRIPPER].position, 2.0)
        self.assertNotIn(Joint.SHOULDER_1, arm.motors)
        # Telemetry is complete once every *present* arm joint has reported.
        with self.assertRaises(RuntimeError):
            await rt.wait_for_telemetry(timeout=0.05)
        feed(0, {6: (0.1, 0.0, 0.0, 1.0)})
        await rt.wait_for_telemetry(timeout=0.05)


# --------------------------------------------------------------------------- #
# Dashboard launch gate                                                       #
# --------------------------------------------------------------------------- #


class RomFaultScopeTest(unittest.TestCase):
    def _faults(self) -> list[dict[str, Any]]:
        def fault(joint: str, problem: str) -> dict[str, Any]:
            return {
                "arm": "left",
                "joint": joint,
                "problem": problem,
                "temperature": None,
            }

        return [
            fault("SHOULDER_1", "unreachable"),
            fault("ELBOW", "over temperature"),
            fault("WRIST_3", "unreachable"),
        ]

    def test_rom_scope_skips_absent_unselected_but_not_faulted_ones(self) -> None:
        args = {"joints": "wrist_2,gripper", "no_right": True}
        kept = scoped_motor_faults(
            self._faults(), args, unselected_joints_only_skip_absent=True
        )
        self.assertEqual([f["joint"] for f in kept], ["ELBOW"])

        args = {"joints": "wrist_2,wrist_3,gripper"}
        kept = scoped_motor_faults(
            self._faults(), args, unselected_joints_only_skip_absent=True
        )
        self.assertEqual([f["joint"] for f in kept], ["ELBOW", "WRIST_3"])

    def test_default_scope_is_unchanged(self) -> None:
        kept = scoped_motor_faults(self._faults(), {"joints": "wrist_2,gripper"})
        self.assertEqual(kept, [])
        kept = scoped_motor_faults(self._faults(), {})
        self.assertEqual(len(kept), 3)


if __name__ == "__main__":
    unittest.main()
