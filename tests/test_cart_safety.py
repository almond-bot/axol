"""Powered-cart safety: the wheel CAN timeout is armed at enable, and the
command task is silent on CAN unless it has a live command source."""

from __future__ import annotations

import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from almond_axol.motor import ControlMode, MotorError, MotorStatus
from almond_axol.motor.config import DAMIAO_TIMEOUT_MS_PER_UNIT
from almond_axol.motor.damiao import _DM_REG_TIMEOUT
from almond_axol.robot import cart as cart_module
from almond_axol.robot.cart import WHEELS, Cart, CartConfig
from almond_axol.robot.lift import STOP, UP


def _wheel(*, timeout_ticks: int = 4000, status: MotorStatus = MotorStatus.OK):
    motor = SimpleNamespace(
        _p_max=400.0,
        _write_register=AsyncMock(),
        _read_register=AsyncMock(return_value=timeout_ticks),
        clear_errors=AsyncMock(),
        enable=AsyncMock(),
        disable=AsyncMock(),
        set_control_mode=AsyncMock(),
        set_velocity=AsyncMock(),
        set_impedance=AsyncMock(),
        get_velocity=AsyncMock(return_value=0.0),
        get_position=AsyncMock(return_value=0.0),
        last_status=None,
    )
    motor.get_error_code = AsyncMock(side_effect=lambda: motor.last_status or status)

    async def _enable() -> None:
        # A real motor's next feedback echo reports it enabled again.
        motor.last_status = MotorStatus.OK

    motor.enable = AsyncMock(side_effect=_enable)
    return motor


def _fast_config(**overrides) -> CartConfig:
    """A config the loop can be exercised against with real (short) sleeps."""
    base = dict(
        channel="can-test",
        lift=False,
        yaw_hold_gain=0.0,
        hold_kp=0.0,  # no park state machine: every linked cycle is a velocity frame
        frequency=200.0,  # 5 ms cycles
        command_timeout=0.04,
        can_timeout_ms=100.0,
        slew=100.0,  # reach the target in one cycle
        axis_snap_deg=0.0,
    )
    base.update(overrides)
    return CartConfig(**base)


class CartCanTimeoutArmingTest(unittest.IsolatedAsyncioTestCase):
    async def _enable(self, cart: Cart, motors: list) -> SimpleNamespace:
        bus = SimpleNamespace(start=AsyncMock(), close=AsyncMock())
        with (
            patch("almond_axol.cli.can.setup.iface_up", return_value=True),
            patch("almond_axol.robot.cart.CanBus", return_value=bus),
            patch("almond_axol.robot.cart.make_driver", side_effect=motors),
        ):
            await cart.enable()
        return bus

    async def test_enable_writes_and_verifies_200ms_timeout_on_every_wheel(
        self,
    ) -> None:
        motors = [_wheel() for _ in WHEELS]
        cart = Cart(_fast_config(can_timeout_ms=200.0))
        await self._enable(cart, motors)
        try:
            ticks = round(200.0 / DAMIAO_TIMEOUT_MS_PER_UNIT)
            self.assertEqual(ticks, 4000)
            for motor in motors:
                writes = [c.args for c in motor._write_register.await_args_list]
                self.assertIn((_DM_REG_TIMEOUT, ticks), writes)
                motor._read_register.assert_awaited_with(_DM_REG_TIMEOUT)
                motor.enable.assert_awaited_once()
        finally:
            await cart.disable()

    async def test_enable_refuses_a_wheel_whose_timeout_readback_differs(
        self,
    ) -> None:
        motors = [_wheel() for _ in WHEELS]
        motors[2]._read_register = AsyncMock(return_value=0)  # alarm disabled
        cart = Cart(_fast_config(can_timeout_ms=200.0))

        with self.assertRaisesRegex(MotorError, "back_left=0ms"):
            await self._enable(cart, motors)

        for motor in motors:
            motor.enable.assert_not_awaited()  # never torqued on
            motor.disable.assert_awaited_once()  # and torqued off regardless
        self.assertIsNone(cart._bus)
        self.assertEqual(cart._motors, [])

    async def test_enable_rejects_configs_that_defeat_the_safety_layer(self) -> None:
        for kwargs, message in (
            ({"can_timeout_ms": 0.0}, "cannot be disabled"),
            ({"can_timeout_ms": -5.0}, "cannot be disabled"),
            ({"can_timeout_ms": 30.0, "frequency": 50.0}, "twice the command period"),
            ({"command_timeout": 0.0}, "command_timeout"),
        ):
            with self.subTest(**kwargs):
                cart = Cart(_fast_config(**kwargs))
                with self.assertRaisesRegex(ValueError, message):
                    await cart.enable()
                self.assertIsNone(cart._bus)
                self.assertEqual(cart._motors, [])

    async def test_disable_clears_a_tripped_fault_before_torque_off(self) -> None:
        motors = [_wheel() for _ in WHEELS]
        cart = Cart(_fast_config())
        cart._motors = motors
        cart._bus = SimpleNamespace(close=AsyncMock())

        await cart.disable()

        for motor in motors:
            motor.clear_errors.assert_awaited_once()
            motor.disable.assert_awaited_once()


class CartCommandLoopSilenceTest(unittest.IsolatedAsyncioTestCase):
    """Drive the real command loop against mock wheels."""

    def _start(self, cart: Cart, motors: list) -> asyncio.Task:
        cart._motors = motors
        cart._bus = SimpleNamespace()
        cart._lift = None
        return asyncio.create_task(cart._command_loop())

    @staticmethod
    def _velocity_frames(motors: list) -> int:
        return sum(m.set_velocity.await_count for m in motors)

    async def _settle(self, cycles: float, cart: Cart) -> None:
        await asyncio.sleep(cycles / cart.config.frequency)

    async def test_no_frames_without_a_command_source(self) -> None:
        motors = [_wheel() for _ in WHEELS]
        cart = Cart(_fast_config())
        task = self._start(cart, motors)
        try:
            await self._settle(8, cart)
            self.assertEqual(self._velocity_frames(motors), 0)
            for motor in motors:
                motor.get_error_code.assert_not_awaited()
                motor.enable.assert_not_awaited()
            self.assertFalse(cart.linked)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_live_source_streams_then_stale_source_stops_once_and_goes_quiet(
        self,
    ) -> None:
        motors = [_wheel() for _ in WHEELS]
        cart = Cart(_fast_config())
        task = self._start(cart, motors)
        try:
            # Keep the source alive for a while: one frame per wheel per cycle.
            t_end = time.monotonic() + 10 / cart.config.frequency
            while time.monotonic() < t_end:
                cart.set_command(0.5, 0.0, 0.0)
                await asyncio.sleep(0.002)
            self.assertTrue(cart.linked)
            streamed = self._velocity_frames(motors)
            self.assertGreaterEqual(streamed, 5 * len(WHEELS))
            self.assertTrue(
                any(c.args[0] > 0.0 for c in motors[0].set_velocity.await_args_list)
            )

            # Source dies: exactly one zero-velocity frame per wheel, then
            # nothing at all — the motor timeout has to be what stops them.
            await asyncio.sleep(cart.config.command_timeout + 6 / cart.config.frequency)
            self.assertFalse(cart.linked)
            after_edge = self._velocity_frames(motors)
            for motor in motors:
                self.assertEqual(motor.set_velocity.await_args_list[-1].args, (0.0,))
            await self._settle(10, cart)
            self.assertEqual(self._velocity_frames(motors), after_edge)
            self.assertEqual(cart.body_cmd, (0.0, 0.0, 0.0))
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    @staticmethod
    def _hold_frames(motors: list) -> int:
        return sum(m.set_impedance.await_count for m in motors)

    async def test_parked_cart_keeps_its_hold_without_a_source(self) -> None:
        """A stationary hold is not a motion: the link deadman does not drop it."""
        motors = [_wheel() for _ in WHEELS]
        for motor in motors:
            motor.last_status = MotorStatus.OK
        cart = Cart(_fast_config(hold_kp=60.0))
        lift = SimpleNamespace(command=Mock(), suspend=Mock())
        task = self._start(cart, motors)
        cart._lift = lift
        try:
            # Live source commanding zero → the wheels park and hold.
            t_end = time.monotonic() + 8 / cart.config.frequency
            while time.monotonic() < t_end:
                cart.set_command(0.0, 0.0, 0.0)
                await asyncio.sleep(0.002)
            self.assertTrue(cart.parked)
            for motor in motors:
                motor.set_control_mode.assert_awaited_with(ControlMode.IMPEDANCE)
            held_linked = self._hold_frames(motors)
            self.assertGreater(held_linked, 0)

            # Source dies: the lift is released, but the hold keeps streaming
            # and the wheels are never given a velocity frame or re-enabled.
            await asyncio.sleep(cart.config.command_timeout + 6 / cart.config.frequency)
            self.assertFalse(cart.linked)
            self.assertTrue(cart.parked)
            lift.suspend.assert_called_once()
            held_unlinked = self._hold_frames(motors)
            self.assertGreater(held_unlinked, held_linked)
            await self._settle(6, cart)
            self.assertGreater(self._hold_frames(motors), held_unlinked)
            for motor in motors:
                motor.set_velocity.assert_not_awaited()
                motor.enable.assert_not_awaited()
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_fresh_wheels_park_immediately_without_a_source(self) -> None:
        """Startup with no headset yet: straight into the hold, no trip first."""
        motors = [_wheel() for _ in WHEELS]
        cart = Cart(_fast_config(hold_kp=60.0))
        task = self._start(cart, motors)
        try:
            await self._settle(3, cart)
            self.assertTrue(cart.parked)
            self.assertFalse(cart.linked)
            for motor in motors:
                motor.set_control_mode.assert_awaited_once_with(ControlMode.IMPEDANCE)
                motor.set_velocity.assert_not_awaited()
                self.assertGreater(motor.set_impedance.await_count, 0)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_driving_cart_losing_its_source_trips_then_reanchors(self) -> None:
        motors = [_wheel() for _ in WHEELS]
        cart = Cart(_fast_config(hold_kp=60.0))
        task = self._start(cart, motors)
        try:
            t_end = time.monotonic() + 8 / cart.config.frequency
            while time.monotonic() < t_end:
                cart.set_command(0.7, 0.0, 0.0)
                await asyncio.sleep(0.002)
            self.assertFalse(cart.parked)
            for motor in motors:
                motor.set_control_mode.assert_not_awaited()

            # Right after the stale edge: one stop, then silence while the
            # motor-side timeout does its work — no hold, no re-enable yet.
            await asyncio.sleep(cart.config.command_timeout + 4 / cart.config.frequency)
            self.assertFalse(cart.linked)
            velocity_frames = self._velocity_frames(motors)
            for motor in motors:
                self.assertEqual(motor.set_velocity.await_args_list[-1].args, (0.0,))
                motor.set_control_mode.assert_not_awaited()
                motor.enable.assert_not_awaited()
            self.assertEqual(self._hold_frames(motors), 0)
            # The wheels have now tripped and torqued off.
            for motor in motors:
                motor.last_status = MotorStatus.LOST_COMM

            # Past the trip: IMPEDANCE mode first (zero command state), then
            # enable, then the hold streams — and no velocity frame ever again.
            await asyncio.sleep(cart.config.can_timeout_ms / 1e3 + 0.05)
            self.assertTrue(cart.parked)
            for motor in motors:
                motor.set_control_mode.assert_awaited_once_with(ControlMode.IMPEDANCE)
                motor.enable.assert_awaited_once()
                self.assertGreater(motor.set_impedance.await_count, 0)
            self.assertEqual(self._velocity_frames(motors), velocity_frames)
            self.assertEqual(cart.wheel_faults, {})
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_reanchor_waits_for_rolling_wheels_to_stop(self) -> None:
        motors = [_wheel() for _ in WHEELS]
        motors[0].get_velocity = AsyncMock(return_value=2.0)  # still rolling
        cart = Cart(_fast_config(hold_kp=60.0))
        task = self._start(cart, motors)
        try:
            await asyncio.sleep(
                cart.config.can_timeout_ms / 1e3 + 2 * cart_module._PARK_RETRY_S + 0.05
            )
            self.assertFalse(cart.parked)
            for motor in motors:
                motor.enable.assert_not_awaited()
                motor.set_impedance.assert_not_awaited()
            self.assertGreater(motors[0].get_velocity.await_count, 1)  # retrying

            motors[0].get_velocity = AsyncMock(return_value=0.0)
            await asyncio.sleep(cart_module._PARK_RETRY_S + 0.05)
            self.assertTrue(cart.parked)
            for motor in motors:
                motor.enable.assert_awaited_once()
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_returning_source_reenables_tripped_wheels_before_driving(
        self,
    ) -> None:
        motors = [_wheel() for _ in WHEELS]
        cart = Cart(_fast_config())
        task = self._start(cart, motors)
        try:
            # First contact: the wheels report the trip the enable-to-first-
            # command gap (or a headset outage) left behind.
            for motor in motors:
                motor.last_status = MotorStatus.LOST_COMM

            cart.set_command(0.0, 0.5, 0.0)
            await self._settle(3, cart)

            for motor in motors:
                motor.enable.assert_awaited_once()
                motor.set_control_mode.assert_awaited_with(ControlMode.VELOCITY)
                # Recovery happens before the first velocity frame.
                self.assertGreater(motor.set_velocity.await_count, 0)
            self.assertEqual(cart.wheel_faults, {})
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_non_comm_faults_are_reported_but_not_cleared(self) -> None:
        motors = [_wheel() for _ in WHEELS]
        motors[1].last_status = MotorStatus.OVER_CURRENT
        cart = Cart(_fast_config())
        task = self._start(cart, motors)
        try:
            cart.set_command(0.0, 0.0, 0.3)
            await self._settle(3, cart)
            motors[1].enable.assert_not_awaited()
            self.assertEqual(
                cart.wheel_faults, {WHEELS[1].name: MotorStatus.OVER_CURRENT}
            )
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_stale_edge_suspends_the_lift(self) -> None:
        motors = [_wheel() for _ in WHEELS]
        cart = Cart(_fast_config())
        lift = SimpleNamespace(command=Mock(), suspend=Mock())
        task = self._start(cart, motors)
        cart._lift = lift
        try:
            cart.set_command(0.0, 0.0, 0.0, UP)
            await self._settle(2, cart)
            lift.command.assert_called_with(UP)
            lift.suspend.assert_not_called()

            await asyncio.sleep(cart.config.command_timeout + 4 / cart.config.frequency)
            lift.suspend.assert_called_once()
            calls_after_edge = lift.command.call_count
            await self._settle(4, cart)
            # Silent: not even command(STOP) is repeated to the lift.
            self.assertEqual(lift.command.call_count, calls_after_edge)
            self.assertEqual(cart.lift_dir, STOP)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task


class CartVrFrameFreshnessTest(unittest.TestCase):
    def test_same_frame_object_does_not_refresh_the_deadline(self) -> None:
        cart = Cart(CartConfig(lift=False, channel=None))
        frame = SimpleNamespace(
            reset=False,
            l_stick_x=0.0,
            l_stick_y=-1.0,
            r_stick_x=0.0,
            l_stick_click=False,
            r_stick_click=False,
        )
        with patch.object(cart_module.time, "monotonic", return_value=100.0):
            cart.apply_vr_frame(frame)
        self.assertEqual(cart._target_time, 100.0)
        self.assertEqual(cart._target[0], 1.0)

        # A poller handing the server's latest frame over again is not a
        # fresh command: the deadline stays where the real frame put it.
        with patch.object(cart_module.time, "monotonic", return_value=100.5):
            cart.apply_vr_frame(frame)
        self.assertEqual(cart._target_time, 100.0)

        fresh = SimpleNamespace(**vars(frame))
        with patch.object(cart_module.time, "monotonic", return_value=100.5):
            cart.apply_vr_frame(fresh)
        self.assertEqual(cart._target_time, 100.5)


if __name__ == "__main__":
    unittest.main()
