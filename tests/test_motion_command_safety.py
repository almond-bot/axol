from __future__ import annotations

import asyncio
import math
import signal
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

import numpy as np

from almond_axol.kinematics.solver import KinematicsSolver
from almond_axol.lerobot.robot.robot_axol import AxolRobot
from almond_axol.lerobot.teleop.teleop_vr import AxolVRTeleop
from almond_axol.motor import ControlMode, Joint, MotorError
from almond_axol.motor.damiao import _float_to_uint as damiao_float_to_uint
from almond_axol.motor.myactuator import _float_to_uint as myactuator_float_to_uint
from almond_axol.robot.axol import Axol, AxolArm
from almond_axol.robot.base import (
    HardwareCleanupError,
    RobotBase,
    is_hardware_cleanup_uncertain,
)
from almond_axol.robot.cart import Cart, CartConfig
from almond_axol.robot.mantis import Mantis, MantisGripperArm
from almond_axol.teleop.teleop import VRTeleop


class _RecordingArm:
    def __init__(self) -> None:
        self.commands: list[np.ndarray] = []

    async def motion_control(self, target: np.ndarray) -> None:
        self.commands.append(target)


class MotionCommandSafetyTest(unittest.IsolatedAsyncioTestCase):
    async def test_axol_rejects_duplicate_arm_channel_before_bus_creation(self) -> None:
        with (
            patch("almond_axol.robot.axol.CanBus") as can_bus,
            self.assertRaisesRegex(ValueError, "different CAN interfaces"),
        ):
            Axol(left_channel="can-shared", right_channel="can-shared")

        can_bus.assert_not_called()

    async def test_axol_enable_waits_for_both_issued_arm_actions(self) -> None:
        sibling_started = asyncio.Event()
        release_sibling = asyncio.Event()

        async def fail_enable(
            _held: list[Joint], _cold: list[Joint], *, hold: bool
        ) -> None:
            del hold
            raise RuntimeError("left enable failed")

        async def blocked_enable(
            _held: list[Joint], _cold: list[Joint], *, hold: bool
        ) -> None:
            del hold
            sibling_started.set()
            await release_sibling.wait()

        robot = object.__new__(Axol)
        robot.connect = AsyncMock()
        robot.left = SimpleNamespace(
            _prepare_enable_state=AsyncMock(return_value=([], [])),
            _enable_from_holding_state=fail_enable,
            motors={},
        )
        robot.right = SimpleNamespace(
            _prepare_enable_state=AsyncMock(return_value=([], [])),
            _enable_from_holding_state=blocked_enable,
            motors={},
        )
        robot._shutdown_pending = False
        robot._motors_disabled = True

        task = asyncio.create_task(robot.enable())
        await sibling_started.wait()
        await asyncio.sleep(0)
        self.assertFalse(task.done())

        release_sibling.set()
        with self.assertRaisesRegex(RuntimeError, "left enable failed"):
            await task

    async def test_axol_enable_peer_failure_rolls_back_cold_motors_on_both_arms(
        self,
    ) -> None:
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()
        setup_error = RuntimeError("right arm enable failed")

        async def blocked_disable() -> None:
            cleanup_started.set()
            await release_cleanup.wait()

        held_motor = SimpleNamespace(disable=AsyncMock())
        left_cold = SimpleNamespace(disable=AsyncMock(side_effect=blocked_disable))
        right_cold = SimpleNamespace(disable=AsyncMock())
        left = SimpleNamespace(
            _prepare_enable_state=AsyncMock(
                return_value=([Joint.WRIST_2], [Joint.SHOULDER_1])
            ),
            _enable_from_holding_state=AsyncMock(),
            reset_command_state=Mock(),
            motors={
                Joint.WRIST_2: held_motor,
                Joint.SHOULDER_1: left_cold,
            },
        )
        right = SimpleNamespace(
            _prepare_enable_state=AsyncMock(return_value=([], [Joint.GRIPPER])),
            _enable_from_holding_state=AsyncMock(side_effect=setup_error),
            reset_command_state=Mock(),
            motors={Joint.GRIPPER: right_cold},
        )
        robot = object.__new__(Axol)
        robot.connect = AsyncMock()
        robot.left = left
        robot.right = right
        robot._shutdown_pending = False
        robot._motors_disabled = False

        task = asyncio.create_task(robot.enable())
        await asyncio.wait_for(cleanup_started.wait(), timeout=1.0)
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        held_motor.disable.assert_not_awaited()
        right_cold.disable.assert_awaited_once_with()

        release_cleanup.set()
        with self.assertRaisesRegex(RuntimeError, "right arm enable failed") as raised:
            await task

        self.assertIs(raised.exception, setup_error)
        self.assertFalse(is_hardware_cleanup_uncertain(raised.exception))
        left_cold.disable.assert_awaited_once_with()
        right_cold.disable.assert_awaited_once_with()
        held_motor.disable.assert_not_awaited()
        self.assertFalse(robot._shutdown_pending)
        self.assertTrue(robot._motors_disabled)
        left.reset_command_state.assert_called_once_with()
        right.reset_command_state.assert_called_once_with()

    async def test_axol_enable_failed_global_rollback_blocks_reenable(self) -> None:
        setup_error = RuntimeError("right arm enable failed")
        cleanup_error = RuntimeError("left motor torque-off timed out")
        cold_motor = SimpleNamespace(
            disable=AsyncMock(side_effect=[cleanup_error, None])
        )
        left = SimpleNamespace(
            _prepare_enable_state=AsyncMock(return_value=([], [Joint.WRIST_2])),
            _enable_from_holding_state=AsyncMock(),
            reset_command_state=Mock(),
            stop_telemetry=AsyncMock(),
            disable=AsyncMock(),
            motors={Joint.WRIST_2: cold_motor},
        )
        right = SimpleNamespace(
            _prepare_enable_state=AsyncMock(return_value=([], [])),
            _enable_from_holding_state=AsyncMock(side_effect=setup_error),
            reset_command_state=Mock(),
            stop_telemetry=AsyncMock(),
            disable=AsyncMock(),
            motors={},
        )
        robot = object.__new__(Axol)
        robot.connect = AsyncMock()
        robot.left = left
        robot.right = right
        robot._shutdown_pending = False
        robot._motors_disabled = True
        robot._left_bus = SimpleNamespace(close=AsyncMock())
        robot._right_bus = SimpleNamespace(close=AsyncMock())

        with self.assertRaisesRegex(RuntimeError, "right arm enable failed") as raised:
            await robot.enable()

        self.assertIs(raised.exception, setup_error)
        self.assertTrue(is_hardware_cleanup_uncertain(raised.exception))
        self.assertTrue(robot._shutdown_pending)
        self.assertFalse(robot._motors_disabled)
        cold_motor.disable.assert_awaited_once_with()
        left.reset_command_state.assert_called_once_with()
        right.reset_command_state.assert_not_called()

        await robot.disable()

        self.assertEqual(cold_motor.disable.await_count, 2)
        left.disable.assert_not_awaited()
        right.disable.assert_not_awaited()
        robot._left_bus.close.assert_awaited_once_with()
        robot._right_bus.close.assert_awaited_once_with()
        self.assertFalse(robot._shutdown_pending)

    async def test_axol_selective_startup_cleanup_stops_both_telemetry_loops(
        self,
    ) -> None:
        left_stop_started = asyncio.Event()
        right_stop_started = asyncio.Event()
        release_telemetry = asyncio.Event()

        async def stop_left() -> None:
            left_stop_started.set()
            await release_telemetry.wait()

        async def stop_right() -> None:
            right_stop_started.set()
            await release_telemetry.wait()

        cold_motor = SimpleNamespace(disable=AsyncMock())
        left = SimpleNamespace(
            stop_telemetry=AsyncMock(side_effect=stop_left),
            disable=AsyncMock(),
        )
        right = SimpleNamespace(
            stop_telemetry=AsyncMock(side_effect=stop_right),
            disable=AsyncMock(),
        )
        robot = object.__new__(Axol)
        robot.left = left
        robot.right = right
        robot._left_bus = SimpleNamespace(close=AsyncMock())
        robot._right_bus = SimpleNamespace(close=AsyncMock())
        robot._shutdown_pending = True
        robot._motors_disabled = False
        robot._startup_rollback_pending = [("left.wrist_2", cold_motor)]

        task = asyncio.create_task(robot.disable())
        await asyncio.wait_for(left_stop_started.wait(), timeout=1.0)
        await asyncio.wait_for(right_stop_started.wait(), timeout=1.0)
        await asyncio.sleep(0)

        cold_motor.disable.assert_not_awaited()
        robot._left_bus.close.assert_not_awaited()
        robot._right_bus.close.assert_not_awaited()

        release_telemetry.set()
        await task

        left.stop_telemetry.assert_awaited_once_with()
        right.stop_telemetry.assert_awaited_once_with()
        cold_motor.disable.assert_awaited_once_with()
        left.disable.assert_not_awaited()
        right.disable.assert_not_awaited()
        robot._left_bus.close.assert_awaited_once_with()
        robot._right_bus.close.assert_awaited_once_with()
        self.assertFalse(robot._shutdown_pending)

    async def test_axol_motion_waits_for_both_issued_arm_actions(self) -> None:
        sibling_started = asyncio.Event()
        release_sibling = asyncio.Event()

        async def fail_motion(_q: np.ndarray) -> None:
            raise RuntimeError("left motion failed")

        async def blocked_motion(_q: np.ndarray) -> None:
            sibling_started.set()
            await release_sibling.wait()

        robot = object.__new__(Axol)
        robot.left = SimpleNamespace(motion_control=fail_motion)
        robot.right = SimpleNamespace(motion_control=blocked_motion)
        target = np.zeros(len(Joint), dtype=np.float32)

        task = asyncio.create_task(robot.motion_control(left=target, right=target))
        await sibling_started.wait()
        await asyncio.sleep(0)
        self.assertFalse(task.done())

        release_sibling.set()
        with self.assertRaisesRegex(RuntimeError, "left motion failed"):
            await task

    async def test_mantis_motion_waits_for_both_issued_arm_actions(self) -> None:
        sibling_started = asyncio.Event()
        release_sibling = asyncio.Event()

        async def fail_motion(_q: np.ndarray) -> None:
            raise RuntimeError("left Mantis motion failed")

        async def blocked_motion(_q: np.ndarray) -> None:
            sibling_started.set()
            await release_sibling.wait()

        robot = object.__new__(Mantis)
        robot.left = SimpleNamespace(motion_control=fail_motion)
        robot.right = SimpleNamespace(motion_control=blocked_motion)
        target = np.zeros(len(Joint), dtype=np.float32)

        task = asyncio.create_task(robot.motion_control(left=target, right=target))
        await sibling_started.wait()
        await asyncio.sleep(0)
        self.assertFalse(task.done())

        release_sibling.set()
        with self.assertRaisesRegex(RuntimeError, "left Mantis motion failed"):
            await task

    async def test_axol_disconnect_waits_for_all_telemetry_before_bus_close(
        self,
    ) -> None:
        sibling_started = asyncio.Event()
        release_sibling = asyncio.Event()

        async def fail_stop() -> None:
            raise RuntimeError("left telemetry stop failed")

        async def blocked_stop() -> None:
            sibling_started.set()
            await release_sibling.wait()

        robot = object.__new__(Axol)
        robot._shutdown_pending = False
        robot.left = SimpleNamespace(stop_telemetry=fail_stop)
        robot.right = SimpleNamespace(stop_telemetry=blocked_stop)
        robot._left_bus = SimpleNamespace(close=AsyncMock())
        robot._right_bus = SimpleNamespace(close=AsyncMock())

        task = asyncio.create_task(robot.disconnect())
        await sibling_started.wait()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        robot._left_bus.close.assert_not_awaited()
        robot._right_bus.close.assert_not_awaited()

        release_sibling.set()
        with self.assertRaisesRegex(RuntimeError, "left telemetry stop failed"):
            await task

        robot._left_bus.close.assert_awaited_once_with()
        robot._right_bus.close.assert_awaited_once_with()

    async def test_arm_enable_waits_for_every_issued_motor_action(self) -> None:
        sibling_started = asyncio.Event()
        release_sibling = asyncio.Event()
        mode_changes: list[Joint] = []

        class FailingMotor:
            async def enable(self) -> None:
                raise RuntimeError("motor enable failed")

            async def disable(self) -> None:
                pass

            async def set_control_mode(self, _mode: object) -> None:
                mode_changes.append(Joint.SHOULDER_1)

        class BlockedMotor:
            async def enable(self) -> None:
                sibling_started.set()
                await release_sibling.wait()

            async def disable(self) -> None:
                pass

            async def set_control_mode(self, _mode: object) -> None:
                mode_changes.append(Joint.SHOULDER_2)

        arm = object.__new__(AxolArm)
        arm.motors = {
            Joint.SHOULDER_1: FailingMotor(),
            Joint.SHOULDER_2: BlockedMotor(),
        }
        arm.resolve_joint_offsets = AsyncMock()
        arm.get_holding = AsyncMock(return_value=[False, False])

        task = asyncio.create_task(arm.enable())
        await sibling_started.wait()
        await asyncio.sleep(0)
        self.assertFalse(task.done())

        release_sibling.set()
        with self.assertRaisesRegex(RuntimeError, "motor enable failed"):
            await task
        self.assertEqual(mode_changes, [])

    async def test_arm_failed_gripper_calibration_rolls_back_only_cold_motors(
        self,
    ) -> None:
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()

        async def blocked_disable() -> None:
            cleanup_started.set()
            await release_cleanup.wait()

        held_motor = SimpleNamespace(
            attach=AsyncMock(),
            disable=AsyncMock(),
        )
        cold_arm_motor = SimpleNamespace(
            enable=AsyncMock(),
            set_control_mode=AsyncMock(),
            disable=AsyncMock(side_effect=blocked_disable),
        )
        cold_gripper = SimpleNamespace(
            enable=AsyncMock(),
            set_control_mode=AsyncMock(),
            get_position=AsyncMock(return_value=1.25),
            set_impedance=AsyncMock(),
            get_torque=AsyncMock(return_value=0.1),
            disable=AsyncMock(),
        )
        arm = object.__new__(AxolArm)
        arm.motors = {
            Joint.WRIST_2: held_motor,
            Joint.SHOULDER_1: cold_arm_motor,
            Joint.GRIPPER: cold_gripper,
        }
        arm.resolve_joint_offsets = AsyncMock()
        arm.get_holding = AsyncMock(return_value=[True, False, False])
        arm._has_gripper = True
        arm._gripper_i = list(Joint).index(Joint.GRIPPER)
        arm._unverified_zeros = set()

        with (
            patch("almond_axol.robot.axol._GRIPPER_CALIB_MAX_STEPS", 3),
            patch("almond_axol.robot.axol._GRIPPER_CALIB_SETTLE", 0.0),
        ):
            task = asyncio.create_task(arm.enable())
            await asyncio.wait_for(cleanup_started.wait(), timeout=1.0)
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            held_motor.disable.assert_not_awaited()
            cold_gripper.disable.assert_awaited_once_with()

            release_cleanup.set()
            with self.assertRaisesRegex(
                MotorError, "no hard stop was detected"
            ) as raised:
                await task

        self.assertFalse(is_hardware_cleanup_uncertain(raised.exception))
        cold_arm_motor.disable.assert_awaited_once_with()
        cold_gripper.disable.assert_awaited_once_with()
        held_motor.disable.assert_not_awaited()

    async def test_arm_startup_marks_failed_cold_motor_rollback_uncertain(
        self,
    ) -> None:
        setup_error = RuntimeError("motor enable failed")
        cleanup_error = RuntimeError("motor torque-off timed out")
        motor = SimpleNamespace(
            enable=AsyncMock(side_effect=setup_error),
            disable=AsyncMock(side_effect=cleanup_error),
        )
        arm = object.__new__(AxolArm)
        arm.motors = {Joint.WRIST_2: motor}
        arm.resolve_joint_offsets = AsyncMock()
        arm.get_holding = AsyncMock(return_value=[False])
        arm._has_gripper = False

        with self.assertRaisesRegex(RuntimeError, "motor enable failed") as raised:
            await arm.enable()

        self.assertIs(raised.exception, setup_error)
        self.assertTrue(is_hardware_cleanup_uncertain(raised.exception))
        motor.disable.assert_awaited_once_with()

    async def test_ik_warmup_failure_prevents_false_ready_solver(self) -> None:
        solver = object.__new__(KinematicsSolver)
        solver._pyroki_index = np.arange(14)
        solver.config = SimpleNamespace(elbow_weight=0.0)
        solver.ik = Mock(side_effect=RuntimeError("JIT backend failed"))

        with self.assertRaisesRegex(RuntimeError, "JIT backend failed"):
            solver._warmup()

    async def test_cartesian_no_elbow_warmup_failure_is_not_cached_ready(self) -> None:
        robot = object.__new__(AxolRobot)
        robot._ik = None
        robot._ik_config = SimpleNamespace()
        solver = SimpleNamespace(
            num_joints=14,
            ik=Mock(side_effect=RuntimeError("no-elbow JIT failed")),
        )

        with (
            patch(
                "almond_axol.kinematics.solver.KinematicsSolver",
                return_value=solver,
            ),
            self.assertRaisesRegex(RuntimeError, "no-elbow JIT failed"),
        ):
            robot.prepare_cartesian_actions()

        self.assertIsNone(robot._ik)

    async def test_axol_arm_rejects_invalid_target_before_hardware_access(self) -> None:
        arm = object.__new__(AxolArm)
        arm._is_left = True
        prior = np.arange(8, dtype=float)
        arm._last_q_commanded = prior.copy()
        arm.resolve_joint_offsets = AsyncMock()

        for target in (
            np.array([0.0] * 7 + [np.nan]),
            np.array([np.inf] + [0.0] * 7),
            np.zeros(7),
            np.zeros((8, 1)),
        ):
            with self.subTest(target=target), self.assertRaises(ValueError):
                await arm.motion_control(target)

        arm.resolve_joint_offsets.assert_not_awaited()
        np.testing.assert_array_equal(arm._last_q_commanded, prior)

    async def test_mantis_arm_rejects_invalid_target_without_latching_or_sending(
        self,
    ) -> None:
        arm = object.__new__(MantisGripperArm)
        prior_virtual = np.arange(7, dtype=np.float32)
        arm._virtual_arm = prior_virtual.copy()
        arm._gripper_target = 0.25
        arm._enabled = True
        arm._send_gripper_target = AsyncMock()

        for target in (
            np.array([0.0] * 7 + [np.nan]),
            np.array([0.0] * 3 + [-np.inf] + [0.0] * 4),
            np.zeros(7),
            np.zeros((2, 4)),
        ):
            with self.subTest(target=target), self.assertRaises(ValueError):
                await arm.motion_control(target)

        arm._send_gripper_target.assert_not_awaited()
        np.testing.assert_array_equal(arm._virtual_arm, prior_virtual)
        self.assertEqual(arm._gripper_target, 0.25)

    async def test_mantis_calibration_without_torque_stop_fails_closed(self) -> None:
        motor = SimpleNamespace(
            enable=AsyncMock(),
            disable=AsyncMock(),
            set_control_mode=AsyncMock(),
            get_position=AsyncMock(return_value=1.25),
            set_impedance=AsyncMock(),
            get_torque=AsyncMock(return_value=0.1),
            set_position_force=AsyncMock(),
        )
        with patch("almond_axol.robot.mantis.Motor", return_value=motor):
            arm = MantisGripperArm(
                Mock(), SimpleNamespace(max_speed=10.0, torque_limit=0.5)
            )

        with (
            patch("almond_axol.robot.axol._GRIPPER_CALIB_MAX_STEPS", 3),
            patch("almond_axol.robot.axol._GRIPPER_CALIB_SETTLE", 0.0),
            self.assertRaisesRegex(MotorError, "no hard stop was detected"),
        ):
            await arm.enable()

        self.assertFalse(arm.is_enabled)
        self.assertFalse(arm.is_calibrated)
        self.assertFalse(arm._disable_pending)
        motor.disable.assert_awaited_once_with()
        self.assertEqual(motor.set_impedance.await_count, 3)
        self.assertEqual(motor.get_torque.await_count, 3)
        motor.set_position_force.assert_not_awaited()
        self.assertEqual(
            [args.args[0] for args in motor.set_control_mode.await_args_list],
            [ControlMode.IMPEDANCE],
        )

    async def test_dual_arm_validation_prevents_partial_send(self) -> None:
        good = np.zeros(8)
        bad = good.copy()
        bad[3] = np.nan

        for robot_type in (Axol, Mantis):
            with self.subTest(robot=robot_type.__name__):
                robot = object.__new__(robot_type)
                robot.left = _RecordingArm()
                robot.right = _RecordingArm()
                with self.assertRaises(ValueError):
                    await robot.motion_control(left=good, right=bad)
                self.assertEqual(robot.left.commands, [])
                self.assertEqual(robot.right.commands, [])

    async def test_motor_encoders_reject_nonfinite_commands(self) -> None:
        for encoder in (damiao_float_to_uint, myactuator_float_to_uint):
            with self.subTest(encoder=encoder.__module__):
                for value in (np.nan, np.inf, -np.inf):
                    with self.assertRaisesRegex(ValueError, "non-finite"):
                        encoder(float(value), -1.0, 1.0, 16)
                self.assertEqual(encoder(0.0, -1.0, 1.0, 16), 32767)

    async def test_cart_invalid_command_fails_to_stop_not_full_speed(self) -> None:
        cart = Cart(CartConfig(lift=False))
        cart._target = (0.5, 0.25, -0.5, 1)

        for command in (
            (math.nan, 0.0, 0.0, 0),
            (0.0, math.inf, 0.0, 0),
            (0.0, 0.0, -math.inf, 0),
            (0.0, 0.0, 0.0, 2),
            (0.0, 0.0, 0.0, True),
        ):
            with self.subTest(command=command), self.assertRaises(ValueError):
                cart.set_command(*command)
            self.assertEqual(cart._target, (0.0, 0.0, 0.0, 0))

        cart.set_command(2.0, -2.0, 0.25, -1)
        self.assertEqual(cart._target, (1.0, -1.0, 0.25, -1))

    async def test_cart_nonfinite_yaw_sample_disables_heading_hold(self) -> None:
        cart = Cart(CartConfig(lift=False))
        cart.feed_yaw_rate(0.25)
        self.assertIsNotNone(cart._yaw_rate)
        samples = cart._yaw_samples

        cart.feed_yaw_rate(math.nan)

        self.assertIsNone(cart._yaw_rate)
        self.assertEqual(cart._yaw_samples, samples)

    async def test_async_robot_context_marks_disable_failure_as_uncertain(self) -> None:
        robot = SimpleNamespace(
            disable=AsyncMock(side_effect=RuntimeError("disable timed out"))
        )

        with self.assertRaisesRegex(HardwareCleanupError, "ownership is uncertain"):
            await RobotBase.__aexit__(robot)

    async def test_teleop_disable_attempts_cart_and_robot_and_reports_uncertainty(
        self,
    ) -> None:
        cart_error = RuntimeError("cart disable timed out")
        teleop = object.__new__(VRTeleop)
        teleop._ik_thread = None
        teleop._parent_conn = None
        teleop._ik_process = None
        teleop._vr_thread = None
        teleop._cart = SimpleNamespace(disable=AsyncMock(side_effect=cart_error))
        teleop._robot = SimpleNamespace(disable=AsyncMock())

        with self.assertRaisesRegex(HardwareCleanupError, "cart disable failed"):
            await teleop.disable()

        teleop._cart.disable.assert_awaited_once()
        teleop._robot.disable.assert_awaited_once()

    async def test_teleop_kills_worker_and_proves_exit_before_clearing(self) -> None:
        process = Mock()
        process.is_alive.side_effect = (True, True, False)
        teleop = object.__new__(VRTeleop)
        teleop._ik_thread = None
        teleop._ik_stop = threading.Event()
        teleop._parent_conn = None
        teleop._ik_process = process
        teleop._vr_thread = None
        teleop._cart = None
        teleop._robot = SimpleNamespace(disable=AsyncMock())

        await teleop.disable()

        self.assertEqual(
            process.join.call_args_list,
            [call(timeout=3.0), call(timeout=2.0), call(timeout=2.0)],
        )
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertIsNone(teleop._ik_process)

    async def test_teleop_process_join_interrupt_still_terminates_and_proves_exit(
        self,
    ) -> None:
        process = Mock(pid=8126)
        process.join.side_effect = (KeyboardInterrupt(), None)
        process.is_alive.side_effect = (True, False)
        teleop = object.__new__(VRTeleop)
        teleop._ik_thread = None
        teleop._ik_stop = threading.Event()
        teleop._parent_conn = None
        teleop._ik_process = process
        teleop._vr_thread = None
        teleop._cart = None
        teleop._robot = SimpleNamespace(disable=AsyncMock())

        await teleop.disable()

        self.assertEqual(
            process.join.call_args_list,
            [call(timeout=3.0), call(timeout=2.0)],
        )
        process.terminate.assert_called_once_with()
        process.kill.assert_not_called()
        self.assertIsNone(teleop._ik_process)

    async def test_teleop_disable_defers_and_restores_repeated_sigint(self) -> None:
        previous_handler = object()
        teleop = object.__new__(VRTeleop)
        teleop._ik_thread = None
        teleop._ik_stop = threading.Event()
        teleop._parent_conn = None
        teleop._ik_process = None
        teleop._vr_thread = None
        teleop._cart = None
        teleop._robot = SimpleNamespace(disable=AsyncMock())

        with (
            patch(
                "almond_axol.teleop.teleop.signal.getsignal",
                return_value=previous_handler,
            ),
            patch("almond_axol.teleop.teleop.signal.signal") as install,
        ):
            await teleop.disable()

        self.assertEqual(
            install.call_args_list,
            [
                call(signal.SIGINT, signal.SIG_IGN),
                call(signal.SIGINT, previous_handler),
            ],
        )
        teleop._robot.disable.assert_awaited_once_with()

    async def test_teleop_retains_worker_reference_when_kill_is_unverified(
        self,
    ) -> None:
        process = Mock()
        process.is_alive.return_value = True
        teleop = object.__new__(VRTeleop)
        teleop._ik_thread = None
        teleop._ik_stop = threading.Event()
        teleop._parent_conn = None
        teleop._ik_process = process
        teleop._vr_thread = None
        teleop._cart = None
        teleop._robot = SimpleNamespace(disable=AsyncMock())

        with self.assertRaisesRegex(RuntimeError, "background ownership is uncertain"):
            await teleop.disable()

        self.assertIs(teleop._ik_process, process)
        teleop._robot.disable.assert_awaited_once_with()

    async def test_lerobot_teleop_surfaces_cart_failure_after_other_cleanup(
        self,
    ) -> None:
        cart_error = RuntimeError("cart disable timed out")
        vr_server = SimpleNamespace(disable=AsyncMock())
        teleop = object.__new__(AxolVRTeleop)
        teleop._ik_thread = None
        teleop._ik_stop = threading.Event()
        teleop._parent_conn = None
        teleop._ik_process = None
        teleop._cart = SimpleNamespace(disable=AsyncMock(side_effect=cart_error))
        teleop._vr_server = vr_server

        with self.assertRaisesRegex(HardwareCleanupError, "cart disable failed"):
            await teleop._disconnect_async()

        teleop._cart.disable.assert_awaited_once_with()
        vr_server.disable.assert_awaited_once_with()
        self.assertIsNone(teleop._vr_server)
        self.assertTrue(teleop._cleanup_pending)

    async def test_lerobot_teleop_retains_lingering_worker(self) -> None:
        process = Mock()
        process.is_alive.return_value = True
        teleop = object.__new__(AxolVRTeleop)
        teleop._ik_thread = None
        teleop._ik_stop = threading.Event()
        teleop._parent_conn = None
        teleop._ik_process = process
        teleop._cart = None
        teleop._vr_server = None

        with self.assertRaisesRegex(RuntimeError, "background ownership is uncertain"):
            await teleop._disconnect_async()

        self.assertIs(teleop._ik_process, process)
        self.assertTrue(teleop._cleanup_pending)

    async def test_cart_disable_attempts_every_wheel_and_retains_failed_bus(
        self,
    ) -> None:
        disable_error = RuntimeError("front-left torque-off timed out")
        motors = [
            SimpleNamespace(
                set_velocity=AsyncMock(),
                disable=AsyncMock(
                    side_effect=[disable_error, None] if index == 0 else None
                ),
            )
            for index in range(4)
        ]
        bus = SimpleNamespace(close=AsyncMock())
        lift = SimpleNamespace(close=AsyncMock())
        cart = Cart(CartConfig(lift=False))
        cart._motors = motors
        cart._bus = bus
        cart._lift = lift

        with self.assertRaisesRegex(
            HardwareCleanupError, "hardware ownership is uncertain"
        ):
            await cart.disable()

        self.assertTrue(all(motor.disable.await_count == 1 for motor in motors))
        bus.close.assert_not_awaited()
        lift.close.assert_awaited_once_with()
        self.assertIsNone(cart._lift)
        self.assertIs(cart._bus, bus)
        self.assertEqual(cart._motors, motors)
        self.assertTrue(cart._shutdown_pending)

        await cart.disable()
        self.assertTrue(all(motor.disable.await_count == 2 for motor in motors))
        bus.close.assert_awaited_once_with()
        self.assertFalse(cart._shutdown_pending)

    async def test_cart_partial_enable_failure_disables_all_and_marks_uncertain(
        self,
    ) -> None:
        setup_error = RuntimeError("second wheel enable failed")
        cleanup_error = RuntimeError("first wheel torque-off timed out")
        motors = []
        for index in range(4):
            motor = SimpleNamespace(
                _p_max=400.0,
                _write_register=AsyncMock(),
                enable=AsyncMock(side_effect=setup_error if index == 1 else None),
                set_control_mode=AsyncMock(),
                set_velocity=AsyncMock(),
                disable=AsyncMock(side_effect=cleanup_error if index == 0 else None),
            )
            motors.append(motor)
        bus = SimpleNamespace(start=AsyncMock(), close=AsyncMock())
        cart = Cart(
            CartConfig(
                channel="can-test",
                lift=False,
                yaw_hold_gain=0.0,
            )
        )

        with (
            patch("almond_axol.cli.can.setup.iface_up", return_value=True),
            patch("almond_axol.robot.cart.CanBus", return_value=bus),
            patch("almond_axol.robot.cart.make_driver", side_effect=motors),
            self.assertRaisesRegex(
                RuntimeError, "second wheel enable failed"
            ) as raised,
        ):
            await cart.enable()

        self.assertIs(raised.exception, setup_error)
        self.assertTrue(is_hardware_cleanup_uncertain(raised.exception))
        self.assertTrue(all(motor.disable.await_count == 1 for motor in motors))
        bus.close.assert_not_awaited()
        self.assertIs(cart._bus, bus)
        self.assertEqual(cart._motors, motors)
        self.assertTrue(cart._shutdown_pending)

    async def test_teleop_startup_preserves_error_and_marks_failed_cleanup(
        self,
    ) -> None:
        setup_error = ValueError("IK startup failed")
        teleop = SimpleNamespace(
            enable=AsyncMock(side_effect=setup_error),
            disable=AsyncMock(side_effect=RuntimeError("robot disable timed out")),
        )

        with self.assertRaisesRegex(ValueError, "IK startup failed") as raised:
            await VRTeleop.__aenter__(teleop)

        self.assertIs(raised.exception, setup_error)
        self.assertTrue(is_hardware_cleanup_uncertain(raised.exception))
        teleop.disable.assert_awaited_once()

    async def test_axol_shutdown_attempts_both_arms_and_retains_buses(self) -> None:
        disable_error = RuntimeError("left disable timed out")
        left = SimpleNamespace(
            stop_telemetry=AsyncMock(),
            disable=AsyncMock(side_effect=[disable_error, None]),
        )
        right = SimpleNamespace(stop_telemetry=AsyncMock(), disable=AsyncMock())
        left_bus = SimpleNamespace(close=AsyncMock())
        right_bus = SimpleNamespace(close=AsyncMock())
        robot = object.__new__(Axol)
        robot.left = left
        robot.right = right
        robot._left_bus = left_bus
        robot._right_bus = right_bus
        robot._shutdown_pending = False
        robot._motors_disabled = False

        with self.assertRaisesRegex(RuntimeError, "left disable timed out"):
            await robot.disable()

        left.disable.assert_awaited_once()
        right.disable.assert_awaited_once()
        left_bus.close.assert_not_awaited()
        right_bus.close.assert_not_awaited()
        self.assertTrue(robot._shutdown_pending)
        with self.assertRaisesRegex(MotorError, "shutdown is incomplete"):
            await robot.connect()

        await robot.disable()
        self.assertEqual(left.disable.await_count, 2)
        self.assertEqual(right.disable.await_count, 2)
        left_bus.close.assert_awaited_once()
        right_bus.close.assert_awaited_once()
        self.assertFalse(robot._shutdown_pending)

    async def test_axol_close_failure_retries_without_motor_commands(self) -> None:
        close_error = RuntimeError("left close failed")
        left = SimpleNamespace(stop_telemetry=AsyncMock(), disable=AsyncMock())
        right = SimpleNamespace(stop_telemetry=AsyncMock(), disable=AsyncMock())
        left_bus = SimpleNamespace(close=AsyncMock(side_effect=[close_error, None]))
        right_bus = SimpleNamespace(close=AsyncMock())
        robot = object.__new__(Axol)
        robot.left = left
        robot.right = right
        robot._left_bus = left_bus
        robot._right_bus = right_bus
        robot._shutdown_pending = False
        robot._motors_disabled = False

        with self.assertRaisesRegex(RuntimeError, "left close failed"):
            await robot.disable()
        await robot.disable()

        left.disable.assert_awaited_once()
        right.disable.assert_awaited_once()
        self.assertEqual(left_bus.close.await_count, 2)
        self.assertEqual(right_bus.close.await_count, 2)
        self.assertFalse(robot._shutdown_pending)

    async def test_lerobot_camera_failure_cannot_skip_hardware_shutdown(self) -> None:
        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        loop_thread.start()
        hardware = SimpleNamespace(disable=AsyncMock())
        camera = SimpleNamespace(
            is_connected=True,
            disconnect=Mock(side_effect=RuntimeError("camera close failed")),
        )
        robot = object.__new__(AxolRobot)
        robot.cameras = {"overhead": camera}
        robot._loop = loop
        robot._loop_thread = loop_thread
        robot._connect_future = None
        robot._disconnect_future = None
        robot._axol = hardware
        robot._fk = None
        robot._ik = None

        try:
            with self.assertRaisesRegex(RuntimeError, "camera close failed"):
                robot.disconnect()
        finally:
            if loop_thread.is_alive():
                loop.call_soon_threadsafe(loop.stop)
                loop_thread.join(timeout=1.0)

        hardware.disable.assert_awaited_once()
        self.assertFalse(loop_thread.is_alive())
        self.assertIsNone(robot._axol)

    async def test_lerobot_disconnect_never_overlaps_timed_out_connect(self) -> None:
        pending_connect = SimpleNamespace(
            result=Mock(side_effect=TimeoutError("connect still running")),
            done=Mock(return_value=False),
        )
        robot = object.__new__(AxolRobot)
        robot.cameras = {}
        robot._loop = SimpleNamespace()
        robot._loop_thread = None
        robot._connect_future = pending_connect
        robot._disconnect_future = None
        robot._axol = SimpleNamespace()

        with self.assertRaisesRegex(HardwareCleanupError, "connect is still running"):
            robot.disconnect()

        pending_connect.result.assert_called_once_with(timeout=10)
        self.assertIs(robot._connect_future, pending_connect)
        self.assertIsNone(robot._disconnect_future)

    async def test_mantis_shutdown_failure_keeps_buses_open_for_retry(self) -> None:
        disable_error = RuntimeError("left disable timed out")
        left = SimpleNamespace(
            force_disable=AsyncMock(side_effect=disable_error),
            disable=AsyncMock(),
        )
        right = SimpleNamespace(force_disable=AsyncMock(), disable=AsyncMock())
        left_bus = SimpleNamespace(close=AsyncMock())
        right_bus = SimpleNamespace(close=AsyncMock())
        robot = object.__new__(Mantis)
        robot.left = left
        robot.right = right
        robot._left_bus = left_bus
        robot._right_bus = right_bus
        robot._connected = True
        robot._shutdown_pending = False
        robot._lifecycle_lock = asyncio.Lock()
        robot._stop_telemetry_unlocked = AsyncMock()

        with self.assertRaisesRegex(RuntimeError, "left disable timed out"):
            await robot.disable()

        left.force_disable.assert_awaited_once()
        right.force_disable.assert_awaited_once()
        left_bus.close.assert_not_awaited()
        right_bus.close.assert_not_awaited()
        self.assertTrue(robot._connected)
        self.assertTrue(robot._shutdown_pending)
        with self.assertRaisesRegex(MotorError, "shutdown is incomplete"):
            await robot._connect_unlocked()

        # A retry uses each arm's pending bit through disable(): a side that
        # already verified torque-off is a no-op, while the failed side retries.
        await robot.disable()
        left.disable.assert_awaited_once()
        right.disable.assert_awaited_once()
        left_bus.close.assert_awaited_once()
        right_bus.close.assert_awaited_once()
        self.assertFalse(robot._connected)
        self.assertFalse(robot._shutdown_pending)

    async def test_mantis_partial_enable_failed_rollback_marks_uncertain(
        self,
    ) -> None:
        setup_error = RuntimeError("left gripper enable failed")
        cleanup_error = RuntimeError("right gripper torque-off timed out")
        left = SimpleNamespace(
            enable=AsyncMock(side_effect=setup_error),
            force_disable=AsyncMock(),
        )
        right = SimpleNamespace(
            enable=AsyncMock(),
            force_disable=AsyncMock(side_effect=cleanup_error),
        )
        robot = object.__new__(Mantis)
        robot.left = left
        robot.right = right
        robot._telemetry_settings = None
        robot._shutdown_pending = False

        with self.assertRaisesRegex(
            RuntimeError, "left gripper enable failed"
        ) as raised:
            await robot._enable_grippers_unlocked()

        self.assertIs(raised.exception, setup_error)
        self.assertTrue(is_hardware_cleanup_uncertain(raised.exception))
        self.assertTrue(robot._shutdown_pending)
        left.force_disable.assert_awaited_once_with()
        right.force_disable.assert_awaited_once_with()

    async def test_mantis_telemetry_restart_failed_rollback_marks_uncertain(
        self,
    ) -> None:
        telemetry_error = RuntimeError("telemetry restart failed")
        cleanup_error = RuntimeError("left gripper torque-off timed out")
        left = SimpleNamespace(
            enable=AsyncMock(),
            force_disable=AsyncMock(side_effect=cleanup_error),
        )
        right = SimpleNamespace(enable=AsyncMock(), force_disable=AsyncMock())
        robot = object.__new__(Mantis)
        robot.left = left
        robot.right = right
        robot._telemetry_settings = (50.0, True)
        robot._shutdown_pending = False
        robot._stop_telemetry_unlocked = AsyncMock()
        robot._start_telemetry_unlocked = AsyncMock(side_effect=telemetry_error)

        with self.assertRaisesRegex(RuntimeError, "telemetry restart failed") as raised:
            await robot._enable_grippers_unlocked()

        self.assertIs(raised.exception, telemetry_error)
        self.assertTrue(is_hardware_cleanup_uncertain(raised.exception))
        self.assertTrue(robot._shutdown_pending)
        robot._stop_telemetry_unlocked.assert_awaited_once_with()
        robot._start_telemetry_unlocked.assert_awaited_once_with(50.0, True)
        left.force_disable.assert_awaited_once_with()
        right.force_disable.assert_awaited_once_with()

    async def test_mantis_deferred_connect_disable_failure_retains_retry_path(
        self,
    ) -> None:
        disable_error = RuntimeError("left force-disable timed out")
        left = SimpleNamespace(
            force_disable=AsyncMock(side_effect=disable_error),
            disable=AsyncMock(),
        )
        right = SimpleNamespace(force_disable=AsyncMock(), disable=AsyncMock())
        left_bus = SimpleNamespace(start=AsyncMock(), close=AsyncMock())
        right_bus = SimpleNamespace(start=AsyncMock(), close=AsyncMock())
        robot = object.__new__(Mantis)
        robot.left = left
        robot.right = right
        robot._left_bus = left_bus
        robot._right_bus = right_bus
        robot._defer_gripper_enable = True
        robot._connected = False
        robot._shutdown_pending = False
        robot._lifecycle_lock = asyncio.Lock()
        robot._stop_telemetry_unlocked = AsyncMock()

        with self.assertRaisesRegex(RuntimeError, "force-disable timed out"):
            await robot._connect_unlocked()

        left_bus.close.assert_not_awaited()
        right_bus.close.assert_not_awaited()
        self.assertTrue(robot._connected)
        self.assertTrue(robot._shutdown_pending)

        await robot.disable()
        left.disable.assert_awaited_once()
        right.disable.assert_awaited_once()
        left_bus.close.assert_awaited_once()
        right_bus.close.assert_awaited_once()
        self.assertFalse(robot._connected)
        self.assertFalse(robot._shutdown_pending)

    async def test_mantis_bus_close_failure_is_surfaced_and_retryable(self) -> None:
        close_error = RuntimeError("left close failed")
        left = SimpleNamespace(force_disable=AsyncMock(), disable=AsyncMock())
        right = SimpleNamespace(force_disable=AsyncMock(), disable=AsyncMock())
        left_bus = SimpleNamespace(
            close=AsyncMock(side_effect=[close_error, None]),
        )
        right_bus = SimpleNamespace(close=AsyncMock())
        robot = object.__new__(Mantis)
        robot.left = left
        robot.right = right
        robot._left_bus = left_bus
        robot._right_bus = right_bus
        robot._connected = True
        robot._shutdown_pending = False
        robot._lifecycle_lock = asyncio.Lock()
        robot._stop_telemetry_unlocked = AsyncMock()

        with self.assertRaisesRegex(RuntimeError, "left close failed"):
            await robot.disable()
        self.assertTrue(robot._connected)
        self.assertTrue(robot._shutdown_pending)

        await robot.disable()
        left.force_disable.assert_awaited_once()
        right.force_disable.assert_awaited_once()
        left.disable.assert_awaited_once()
        right.disable.assert_awaited_once()
        self.assertEqual(left_bus.close.await_count, 2)
        self.assertEqual(right_bus.close.await_count, 2)
        self.assertFalse(robot._connected)
        self.assertFalse(robot._shutdown_pending)


if __name__ == "__main__":
    unittest.main()
