from __future__ import annotations

import asyncio
import math
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

import numpy as np

from almond_axol.kinematics.solver import KinematicsSolver
from almond_axol.motor import MotorError
from almond_axol.motor.damiao import _float_to_uint as damiao_float_to_uint
from almond_axol.motor.myactuator import _float_to_uint as myactuator_float_to_uint
from almond_axol.lerobot.robot.robot_axol import AxolRobot
from almond_axol.lerobot.teleop.teleop_vr import AxolVRTeleop
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
