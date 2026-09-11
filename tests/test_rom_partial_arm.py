"""ROM soak on a partial bench arm: only the selected joints need to be there.

A single-channel adapter with just a wrist assembly (wrist_2, wrist_3, gripper)
on the bus must be able to run ``diag.rom-enable --joints wrist_2,wrist_3,gripper``.
The pieces that make that work: the pre-flight presence probe and its
decision rule, an ``AxolArm`` restricted to the motors actually present, and
an ``Axol`` that configures/feeds only those motors (the core slots them by
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
from almond_axol.robot.axol import AxolHardware
from almond_axol.rt import Axol
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
# Partial AxolArm / Axol                                                      #
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


def _partial_axol(joints: set[Joint], config: Any = None) -> AxolHardware:
    with (
        patch("almond_axol.robot.axol.CanBus", _FakeBus),
        patch(
            "almond_axol.motor.motor.make_driver",
            side_effect=lambda *_a, **_k: _FakeDriver(),
        ),
    ):
        kwargs = {} if config is None else {"config": config}
        return AxolHardware(
            left_channel="can0", right_channel=None, left_joints=joints, **kwargs
        )


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
            axol = AxolHardware(
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


class PartialArmTelemetryCaptureTest(unittest.IsolatedAsyncioTestCase):
    async def test_capture_leaves_absent_motor_cells_empty(self) -> None:
        """The ROM capture samples ``arm.motors`` per joint; absent motors
        must yield empty cells, not a KeyError that would kill the sampler
        (and, re-raised from ``logger.stop()`` in the run's ``finally``, skip
        the motor disable)."""
        import csv
        import tempfile
        from pathlib import Path

        from almond_axol.diagnostics.telemetry_log import TelemetryCsvLogger

        axol = _partial_axol(set(WRIST_KIT))
        arm = axol.left
        assert arm is not None
        arm.motors[Joint.WRIST_2]._position = 0.4
        arm.motors[Joint.WRIST_2]._torque = 0.1
        with tempfile.TemporaryDirectory() as tmp:
            logger = TelemetryCsvLogger(axol, "rom", hz=200.0, out_dir=Path(tmp))
            with patch("builtins.print"):
                logger.start()
            await asyncio.sleep(0.05)
            await logger.stop()  # re-raises any sampler exception
            with open(logger.path, newline="") as f:
                rows = list(csv.reader(f))
        header, first = rows[0], rows[1]
        self.assertEqual(len(header), 1 + 2 * len(list(Joint)))
        self.assertEqual(len(first), len(header))
        self.assertEqual(first[header.index("left:SHOULDER_1:pos")], "")
        self.assertEqual(first[header.index("left:WRIST_2:pos")], "0.4")
        self.assertEqual(first[header.index("left:WRIST_3:pos")], "")


class PartialAxolTest(unittest.IsolatedAsyncioTestCase):
    def test_config_lists_only_present_motors(self) -> None:
        rt = Axol(hardware=_partial_axol(set(WRIST_KIT)))
        lines = rt._config_text().splitlines()
        # Slot-by-motor-id is protocol generation 2; a core that predates it
        # would slot these wrists at 0 and 1 and then reject every target,
        # so the config declares the generation and such a core refuses it.
        self.assertEqual(lines[0], "proto 2")
        joint_lines = [line for line in lines if line.startswith("joint ")]
        self.assertEqual(
            [line.split()[3:5] for line in joint_lines],
            [["wrist_2", "6"], ["wrist_3", "7"]],
        )
        self.assertIn("gripper 0 can0 8", lines)

    async def test_feedback_feed_fills_present_slots_and_ignores_the_rest(self) -> None:
        rt = Axol(hardware=_partial_axol(set(WRIST_KIT)))
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
# Bench gains                                                                 #
# --------------------------------------------------------------------------- #


class BenchConfigTest(unittest.IsolatedAsyncioTestCase):
    """A partial arm is off the robot; drive it as plain soft PD.

    The production wrist gains (130/3.5 on wrist_2) plus the model
    feedforwards vibrated heavily on a wrist kit clamped to a bench. The bench
    config is the soft end of the stiffness slider with every model term off,
    so what reaches the core is kp/kd and a zero feedforward.
    """

    def test_bench_config_is_soft_pd_with_no_model_terms(self) -> None:
        from almond_axol.robot.config import AxolConfig

        cfg = rom.bench_config(AxolConfig(left_stiffness=1.0, right_stiffness=1.0))
        self.assertEqual(cfg.left_stiffness, rom.BENCH_STIFFNESS)
        self.assertEqual(cfg.right_stiffness, rom.BENCH_STIFFNESS)
        resolved = cfg.resolved()
        for arm in (resolved.left, resolved.right):
            for joint in ARM_JOINTS:
                jc = getattr(arm, joint.value)
                self.assertEqual(jc.kd_host, 0.0, joint)
                self.assertEqual(jc.j_eff, 0.0, joint)
                self.assertEqual(jc.mass, 0.0, joint)
                self.assertEqual(
                    (jc.friction.fc, jc.friction.k, jc.friction.fv, jc.friction.fo),
                    (0.0, 0.0, 0.0, 0.0),
                    joint,
                )
        # The soft endpoint of the slider (``_SOFT_GAINS``), not the tuned top.
        self.assertAlmostEqual(resolved.left.wrist_2.kp, 25.0)
        self.assertAlmostEqual(resolved.left.wrist_2.kd, 1.5)
        self.assertAlmostEqual(resolved.left.wrist_3.kp, 25.0)
        self.assertAlmostEqual(resolved.left.wrist_3.kd, 0.9)
        # The gripper is position/force controlled and untouched.
        self.assertEqual(resolved.left.gripper, AxolConfig().left.gripper)

    async def test_bench_arm_streams_soft_pd_and_zero_feedforward(self) -> None:
        from almond_axol.robot.config import AxolConfig

        axol = _partial_axol(set(WRIST_KIT), rom.bench_config(AxolConfig()))
        arm = axol.left
        assert arm is not None
        arm._unresolved_offsets.clear()
        arm._joint_offsets[:] = 0.0
        shipped: list[list[tuple[float, ...]]] = []
        arm._command_sink = shipped.append
        # Wrist_2 rotated with wrist_3 held: on the robot's gains this pose
        # carries a non-zero gravity feedforward for both wrists.
        q = np.zeros(8, dtype=np.float32)
        q[5] = 1.2
        q[7] = 1.0
        await arm.motion_control(q)
        (cmds,) = shipped
        p_des, mode, kp, kd, t_ff, kd_host, _w0, _q, j_eff = cmds[5]
        self.assertAlmostEqual(p_des, 1.2, places=5)
        self.assertEqual(mode, 1.0)
        self.assertAlmostEqual(kp, 25.0)
        self.assertAlmostEqual(kd, 1.5)
        self.assertEqual((t_ff, kd_host, j_eff), (0.0, 0.0, 0.0))
        _p, _m, kp3, kd3, t_ff3, kd_host3, _w03, _q3, j_eff3 = cmds[6]
        self.assertAlmostEqual(kp3, 25.0)
        self.assertAlmostEqual(kd3, 0.9)
        self.assertEqual((t_ff3, kd_host3, j_eff3), (0.0, 0.0, 0.0))
        # The core's friction model rides the config; the bench config zeroes it.
        rt = Axol(hardware=axol)
        for line in rt._config_text().splitlines():
            if line.startswith("joint "):
                self.assertEqual(line.split()[9:], ["0.0", "0.0", "0.0", "0.0"], line)

    def test_only_a_partial_arm_is_a_bench_run(self) -> None:
        """A full arm on the bus — even with a joint subset selected — is the
        robot and keeps the production gains; any missing motor means bench."""
        candidates = set(Joint)
        self.assertFalse(
            rom.is_bench_run({"left": candidates, "right": None}, candidates)
        )
        self.assertFalse(
            rom.is_bench_run({"left": candidates, "right": candidates}, candidates)
        )
        self.assertTrue(
            rom.is_bench_run({"left": set(WRIST_KIT), "right": None}, candidates)
        )
        self.assertTrue(
            rom.is_bench_run({"left": candidates, "right": set(WRIST_KIT)}, candidates)
        )
        # Gripperless SKU: the candidate set has no gripper, so a full
        # seven-motor arm is not partial.
        arm_only = set(ARM_JOINTS)
        self.assertFalse(rom.is_bench_run({"left": arm_only, "right": None}, arm_only))
        # A mounted arm whose gripper does not answer (unpowered, missing, or
        # not fitted on a gripper-configured robot) is still the robot: the
        # gripper says nothing about the mounting, and soft PD with no
        # gravity feedforward would let the held shoulders sag.
        self.assertFalse(
            rom.is_bench_run({"left": arm_only, "right": None}, candidates)
        )
        # Whereas a gripper that answers on a partial arm is still a bench.
        self.assertTrue(
            rom.is_bench_run(
                {"left": {Joint.WRIST_2, Joint.GRIPPER}, "right": None}, candidates
            )
        )


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
