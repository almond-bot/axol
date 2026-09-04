from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import math
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from almond_axol.constants import CAN_CHEST, Joint
from almond_axol.diagnostics.lift import cycle
from almond_axol.robot.lift import LiftStatus


def _status(
    position: int,
    *,
    moving: bool = False,
    pos_move: bool = False,
    homed: bool = True,
    stall_fault: bool = False,
    driver_fault_mask: int = 0,
    drivers_enabled: bool = True,
    vm_present: bool = True,
    flash_interlock: bool = False,
    save_pending: bool = False,
    drift: int = 0,
) -> LiftStatus:
    return LiftStatus(
        position_permille=position,
        velocity=0,
        drift=drift,
        homed=homed,
        moving=moving,
        pos_move=pos_move,
        stall_fault=stall_fault,
        at_lower=position == 0,
        at_upper=position == 1000,
        homing=False,
        jog=False,
        driver_fault_mask=driver_fault_mask,
        drivers_enabled=drivers_enabled,
        vm_present=vm_present,
        flash_interlock=flash_interlock,
        save_pending=save_pending,
    )


class FakeLift:
    def __init__(self, status: LiftStatus, events: list[str]) -> None:
        self.status = status
        self.last_status_monotonic = time.monotonic()
        self.status_age = 0.0
        self._events = events

    def status_is_fresh(self, max_age_s: float) -> bool:
        return max_age_s >= 0

    async def set_position(
        self,
        target: int,
        speed: int,
        *,
        before_send=None,  # noqa: ANN001 - mirrors the production callback
    ) -> None:
        del target, speed
        if before_send is not None:
            await before_send()

    async def stop_motion(self) -> None:
        self._events.append("lift.stop")

    async def close(self) -> None:
        self._events.append("lift.close")


class FakeAxol:
    def __init__(
        self,
        *,
        config,  # noqa: ANN001
        left_channel: str | None,
        right_channel: str | None,
        events: list[str],
    ) -> None:
        self.config = config
        self.left_channel = left_channel
        self.right_channel = right_channel
        self._events = events
        self._positions = (
            None
            if left_channel is None
            else np.linspace(-0.4, 0.3, len(Joint), dtype=np.float32),
            None
            if right_channel is None
            else np.linspace(0.4, -0.3, len(Joint), dtype=np.float32),
        )
        self.left = (
            None
            if left_channel is None
            else SimpleNamespace(
                motors={
                    joint: SimpleNamespace(
                        is_holding=AsyncMock(return_value=True),
                        disable=AsyncMock(),
                    )
                    for joint in cycle.ARM_JOINTS
                }
            )
        )
        self.right = (
            None
            if right_channel is None
            else SimpleNamespace(
                motors={
                    joint: SimpleNamespace(
                        is_holding=AsyncMock(return_value=True),
                        disable=AsyncMock(),
                    )
                    for joint in cycle.ARM_JOINTS
                }
            )
        )

    async def enable(self) -> None:
        self._events.append("arms.enable")

    async def get_positions(self):  # noqa: ANN201
        self._events.append("arms.positions")
        return self._positions

    async def disconnect(self) -> None:
        self._events.append("arms.disconnect")


class FakeOpeningLift:
    def __init__(self, *, broadcasts: bool, interval_s: float = 0.2) -> None:
        self.status: LiftStatus | None = None
        self.last_status_monotonic: float | None = None
        self.status_timestamps: list[float] = []
        self.status_age = 0.0
        self.broadcasts = broadcasts
        self.interval_s = interval_s
        self.events: list[str] = []

    def status_is_fresh(self, max_age_s: float) -> bool:
        return max_age_s >= 0 and self.status is not None

    def _publish(self, stamp: float) -> None:
        self.last_status_monotonic = stamp
        self.status_timestamps.append(stamp)

    async def start(self, *, request_status: bool = True) -> None:
        self.events.append(f"start:{request_status}")
        self.status = _status(1000)
        self._publish(1.0)
        if self.broadcasts:
            loop = asyncio.get_running_loop()
            for index in range(1, 4):
                loop.call_later(
                    0.02 * index,
                    self._publish,
                    1.0 + self.interval_s * index,
                )

    async def set_status_period(
        self, period_ms: int, *, recover_stale: bool = True
    ) -> None:
        self.events.append(f"rate:{period_ms}:{recover_stale}")

    def enable_broadcast_recovery(self) -> None:
        self.events.append("recovery")

    async def close(self) -> None:
        self.events.append("close")


@contextlib.contextmanager
def _interrupt_context():
    yield asyncio.Event()


def _args(cycles: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        cycles=cycles,
        lift_channel=CAN_CHEST,
        speed=0,
        no_left=False,
        no_right=False,
        left_channel="can-left-test",
        right_channel="can-right-test",
    )


class LiftCycleHelpersTest(unittest.TestCase):
    def test_clearance_targets_only_change_s1_and_mirror_right(self) -> None:
        left = np.linspace(-0.7, 0.7, len(Joint), dtype=np.float32)
        right = np.linspace(0.7, -0.7, len(Joint), dtype=np.float32)
        original_left = left.copy()
        original_right = right.copy()

        target_left, target_right = cycle._clearance_targets(left, right)

        assert target_left is not None
        assert target_right is not None
        s1 = list(Joint).index(Joint.SHOULDER_1)
        self.assertAlmostEqual(float(target_left[s1]), math.pi / 2, places=6)
        self.assertAlmostEqual(float(target_right[s1]), -math.pi / 2, places=6)
        np.testing.assert_array_equal(target_left[1:], original_left[1:])
        np.testing.assert_array_equal(target_right[1:], original_right[1:])
        np.testing.assert_array_equal(left, original_left)
        np.testing.assert_array_equal(right, original_right)

    def test_rest_targets_only_return_s1_to_zero(self) -> None:
        left = np.linspace(-0.7, 0.7, len(Joint), dtype=np.float32)
        right = np.linspace(0.7, -0.7, len(Joint), dtype=np.float32)
        original_left = left.copy()
        original_right = right.copy()

        target_left, target_right = cycle._rest_targets(left, right)

        assert target_left is not None
        assert target_right is not None
        s1 = list(Joint).index(Joint.SHOULDER_1)
        self.assertEqual(float(target_left[s1]), 0.0)
        self.assertEqual(float(target_right[s1]), 0.0)
        np.testing.assert_array_equal(target_left[1:], original_left[1:])
        np.testing.assert_array_equal(target_right[1:], original_right[1:])
        np.testing.assert_array_equal(left, original_left)
        np.testing.assert_array_equal(right, original_right)

    def test_controller_validation_requires_v08_health(self) -> None:
        legacy = LiftStatus(
            1000,
            0,
            0,
            True,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
        )

        with self.assertRaisesRegex(cycle.DiagnosticFailure, "v0.8"):
            cycle._validate_controller(legacy, "preflight")
        with self.assertRaisesRegex(cycle.DiagnosticFailure, "fault mask"):
            cycle._validate_controller(_status(1000, driver_fault_mask=1), "preflight")

    def test_endpoint_requires_limit_idle_and_low_drift(self) -> None:
        self.assertTrue(cycle._at_endpoint(_status(1000), 1000))
        self.assertTrue(
            cycle._at_endpoint(replace(_status(1000), at_upper=False), 1000)
        )
        self.assertFalse(cycle._at_endpoint(_status(999), 1000))
        self.assertFalse(cycle._at_endpoint(_status(1000, moving=True), 1000))
        self.assertFalse(cycle._at_endpoint(_status(1000, drift=9), 1000))

    def test_parser_defaults_to_c_without_wheel_bus_fallback(self) -> None:
        parser = argparse.ArgumentParser()
        cycle._add_arguments(parser, cycles_required=False)

        args = parser.parse_args(["--cycles", "3"])

        self.assertEqual(args.lift_channel, "can_alm_axol_c")
        self.assertEqual(args.lift_channel, CAN_CHEST)

    def test_parser_has_no_gripper_control_surface(self) -> None:
        parser = argparse.ArgumentParser()
        cycle._add_arguments(parser, cycles_required=False)

        args = parser.parse_args(["--cycles", "1"])

        self.assertFalse(hasattr(args, "has_gripper"))
        self.assertNotIn("gripper", parser.format_help().lower())

    def test_terminal_can_prompt_for_cycle_count(self) -> None:
        with (
            patch.object(cycle.sys.stdin, "isatty", return_value=True),
            patch("builtins.input", side_effect=["zero", "4"]),
            patch("builtins.print"),
        ):
            self.assertEqual(cycle._resolve_cycles(None), 4)

    def test_cycle_count_must_be_positive(self) -> None:
        with self.assertRaisesRegex(SystemExit, "greater than zero"):
            cycle._resolve_cycles(0)

    def test_cycle_speed_rejects_values_that_cannot_finish_full_travel(self) -> None:
        self.assertEqual(cycle._speed_value("0"), 0)
        self.assertEqual(cycle._speed_value("250"), 250)
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "at least 250"):
            cycle._speed_value("249")


class LiftCycleSequenceTest(unittest.IsolatedAsyncioTestCase):
    async def test_open_lift_proves_receive_only_broadcast_before_motion(self) -> None:
        lift = FakeOpeningLift(broadcasts=True)
        with (
            patch.object(cycle, "Lift", return_value=lift) as lift_type,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            opened = await cycle._open_lift("can-test")

        self.assertIs(opened, lift)
        lift_type.assert_called_once_with("can-test")
        self.assertEqual(lift.events, ["rate:200:False", "start:False", "recovery"])

    async def test_open_lift_closes_when_broadcast_never_starts(self) -> None:
        lift = FakeOpeningLift(broadcasts=False)
        with (
            patch.object(cycle, "Lift", return_value=lift),
            patch.object(cycle, "_FIRST_STATUS_TIMEOUT_S", 0.0),
            self.assertRaisesRegex(cycle.DiagnosticFailure, "200 ms status broadcasts"),
        ):
            await cycle._open_lift("can-test")

        self.assertEqual(lift.events, ["rate:200:False", "start:False", "close"])

    async def test_open_lift_rejects_default_50ms_broadcast_cadence(self) -> None:
        lift = FakeOpeningLift(broadcasts=True, interval_s=0.05)
        with (
            patch.object(cycle, "Lift", return_value=lift),
            self.assertRaisesRegex(cycle.DiagnosticFailure, "observed 50, 50 ms"),
        ):
            await cycle._open_lift("can-test")

        self.assertEqual(lift.events[-1], "close")

    async def test_open_lift_preserves_cleanup_cancellation(self) -> None:
        lift = SimpleNamespace(
            set_status_period=AsyncMock(),
            start=AsyncMock(side_effect=OSError("startup failed")),
            close=AsyncMock(side_effect=asyncio.CancelledError),
        )
        with (
            patch.object(cycle, "Lift", return_value=lift),
            self.assertRaises(asyncio.CancelledError),
        ):
            await cycle._open_lift("can-test")

        self.assertEqual(lift.close.await_count, 2)

    async def test_clearance_monitor_detects_non_s1_joint_drift(self) -> None:
        target = np.zeros(len(Joint), dtype=np.float32)
        measured = target.copy()
        elbow_index = list(Joint).index(Joint.ELBOW)
        measured[elbow_index] = math.radians(6)
        axol = SimpleNamespace(
            left=SimpleNamespace(),
            right=None,
            get_positions=AsyncMock(return_value=(measured, None)),
        )

        with self.assertRaisesRegex(cycle.DiagnosticFailure, "left elbow"):
            await cycle._verify_arm_targets(axol, target, None, "monitor")

    async def test_arm_ramp_checks_safety_before_commanding_motion(self) -> None:
        pose = np.zeros(len(Joint), dtype=np.float32)
        axol = SimpleNamespace(motion_control=AsyncMock())
        safety_check = AsyncMock(
            side_effect=cycle.DiagnosticFailure("lift left upper endpoint")
        )

        with self.assertRaisesRegex(cycle.DiagnosticFailure, "left upper endpoint"):
            await cycle._ramp_arms(
                axol,
                pose,
                None,
                pose.copy(),
                None,
                asyncio.Event(),
                safety_check=safety_check,
            )

        safety_check.assert_awaited_once_with()
        axol.motion_control.assert_not_awaited()

    async def test_move_rechecks_interrupt_after_awaited_safety_callback(self) -> None:
        events: list[str] = []
        lift = FakeLift(_status(500), events)
        lift.set_position = AsyncMock()  # type: ignore[method-assign]
        interrupted = asyncio.Event()

        async def interrupt_during_check() -> None:
            interrupted.set()

        with self.assertRaises(cycle.Interrupted):
            await cycle._move_lift(
                lift,
                1000,
                0,
                interrupted,
                "raise",
                interrupt_during_check,
            )

        lift.set_position.assert_not_awaited()

    async def test_move_rechecks_interlock_after_driver_stop_handshake(self) -> None:
        events: list[str] = []
        lift = FakeLift(_status(500), events)
        interrupted = asyncio.Event()
        checks = 0

        async def interrupt_on_post_stop_check() -> None:
            nonlocal checks
            checks += 1
            if checks == 2:
                interrupted.set()

        with self.assertRaises(cycle.Interrupted):
            await cycle._move_lift(
                lift,
                1000,
                0,
                interrupted,
                "raise",
                interrupt_on_post_stop_check,
            )

        self.assertEqual(checks, 2)

    async def test_arm_ramp_rechecks_interrupt_after_safety_callback(self) -> None:
        pose = np.zeros(len(Joint), dtype=np.float32)
        axol = SimpleNamespace(motion_control=AsyncMock())
        interrupted = asyncio.Event()

        async def interrupt_during_check() -> None:
            interrupted.set()

        with self.assertRaises(cycle.Interrupted):
            await cycle._ramp_arms(
                axol,
                pose,
                None,
                pose.copy(),
                None,
                interrupted,
                safety_check=interrupt_during_check,
            )

        axol.motion_control.assert_not_awaited()

    async def test_arm_positions_require_exact_shape_and_finite_values(self) -> None:
        for pose, message in (
            (np.zeros(len(Joint) - 1), "shape"),
            (
                np.array([0.0] * (len(Joint) - 1) + [math.nan]),
                "finite",
            ),
        ):
            axol = SimpleNamespace(
                left=SimpleNamespace(),
                right=None,
                get_positions=AsyncMock(return_value=(pose, None)),
            )
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(cycle.DiagnosticFailure, message),
            ):
                await cycle._read_valid_arm_positions(axol, "test")

    async def test_arm_monitor_rejects_motor_that_is_not_holding(self) -> None:
        motors = {
            joint: SimpleNamespace(is_holding=AsyncMock(return_value=True))
            for joint in cycle.ARM_JOINTS
        }
        motors[Joint.ELBOW].is_holding = AsyncMock(return_value=False)
        axol = SimpleNamespace(
            left=SimpleNamespace(motors=motors),
            right=None,
        )

        with self.assertRaisesRegex(cycle.DiagnosticFailure, "elbow is not enabled"):
            await cycle._verify_arms_holding(axol, "monitor")

    async def test_arm_shutdown_requires_every_motor_to_report_disabled(self) -> None:
        motors = {
            joint: SimpleNamespace(
                disable=AsyncMock(), is_holding=AsyncMock(return_value=False)
            )
            for joint in cycle.ARM_JOINTS
        }
        holding_motor = SimpleNamespace(
            disable=AsyncMock(), is_holding=AsyncMock(return_value=True)
        )
        motors[Joint.ELBOW] = holding_motor
        axol = SimpleNamespace(
            left=SimpleNamespace(motors=motors),
            right=None,
        )

        with self.assertRaisesRegex(
            cycle.DiagnosticFailure, "left elbow still reports enabled"
        ):
            await cycle._disable_arms_verified(axol)

        motors[Joint.SHOULDER_1].disable.assert_awaited_once_with()
        holding_motor.disable.assert_awaited_once_with()

    async def test_arm_shutdown_passes_when_every_motor_reports_disabled(self) -> None:
        motors = {
            joint: SimpleNamespace(
                disable=AsyncMock(), is_holding=AsyncMock(return_value=False)
            )
            for joint in cycle.ARM_JOINTS
        }
        axol = SimpleNamespace(
            left=SimpleNamespace(motors=motors),
            right=None,
        )

        await cycle._disable_arms_verified(axol)

        for motor in motors.values():
            motor.disable.assert_awaited_once_with()
            motor.is_holding.assert_awaited_once_with()

    async def test_arm_shutdown_never_ignores_disable_exception(self) -> None:
        motors = {
            joint: SimpleNamespace(
                disable=AsyncMock(), is_holding=AsyncMock(return_value=False)
            )
            for joint in cycle.ARM_JOINTS
        }
        motors[Joint.SHOULDER_1].disable = AsyncMock(
            side_effect=OSError("disable lost")
        )
        axol = SimpleNamespace(
            left=SimpleNamespace(motors=motors),
            right=None,
        )

        with self.assertRaisesRegex(cycle.DiagnosticFailure, "disable lost"):
            await cycle._disable_arms_verified(axol)

    async def test_cleanup_retry_finishes_then_preserves_cancellation(self) -> None:
        operation = AsyncMock(side_effect=[asyncio.CancelledError(), None])

        with self.assertRaises(asyncio.CancelledError):
            await cycle._retry_cleanup(operation, label="test cleanup")

        self.assertEqual(operation.await_count, 2)

    async def test_stop_retry_finishes_then_preserves_cancellation(self) -> None:
        lift = SimpleNamespace(
            stop_motion=AsyncMock(side_effect=[asyncio.CancelledError(), None])
        )
        wait = AsyncMock(return_value=_status(1000))
        with (
            patch.object(cycle, "_wait_for_status_after", wait),
            self.assertRaises(asyncio.CancelledError),
        ):
            await cycle._stop_lift_verified(
                lift,
                context="test stop",
                require_upper=True,
            )

        self.assertEqual(lift.stop_motion.await_count, 2)
        wait.assert_awaited_once()

    async def test_cleanup_cancellation_overrides_prior_diagnostic_failure(
        self,
    ) -> None:
        events: list[str] = []
        lift = FakeLift(_status(1000), events)
        lift.status_is_fresh = lambda _max_age_s: False
        with (
            patch.object(cycle, "_open_lift", AsyncMock(return_value=lift)),
            patch.object(cycle, "interrupt_event", _interrupt_context),
            patch.object(
                cycle,
                "_stop_lift_verified",
                AsyncMock(side_effect=asyncio.CancelledError),
            ),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(asyncio.CancelledError),
        ):
            await cycle._run(_args(cycles=1))

        self.assertEqual(events[-1], "lift.close")

    async def _run_with_fakes(
        self,
        *,
        initial: LiftStatus,
        move_side_effect,  # noqa: ANN001
        cycles: int = 2,
        disconnect_side_effect=None,  # noqa: ANN001
        stdout: io.StringIO | None = None,
    ):
        events: list[str] = []
        lift = FakeLift(initial, events)
        fake_axol: FakeAxol | None = None

        def make_axol(*, config, left_channel, right_channel):  # noqa: ANN001, ANN202
            nonlocal fake_axol
            fake_axol = FakeAxol(
                config=config,
                left_channel=left_channel,
                right_channel=right_channel,
                events=events,
            )
            if disconnect_side_effect is not None:
                fake_axol.disconnect = AsyncMock(  # type: ignore[method-assign]
                    side_effect=disconnect_side_effect
                )
            return fake_axol

        async def ramp(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            events.append("arms.ramp")
            assert fake_axol is not None
            measured_left = None if args[3] is None else args[3].copy()
            measured_right = None if args[4] is None else args[4].copy()
            if events.count("arms.ramp") == 1:
                # A small, in-tolerance measured offset proves the return ramp
                # starts from live feedback rather than the old command.
                if measured_left is not None:
                    measured_left[1] += 0.01
                if measured_right is not None:
                    measured_right[1] -= 0.01
            fake_axol._positions = (measured_left, measured_right)
            safety_check = kwargs.get("safety_check")
            if safety_check is not None:
                await safety_check()

        move = AsyncMock(side_effect=move_side_effect)
        wait_after = AsyncMock(return_value=_status(1000))

        async def disable_arms(_axol) -> None:  # noqa: ANN001
            events.append("arms.disable")

        output = stdout if stdout is not None else io.StringIO()
        with (
            patch.object(cycle, "_open_lift", AsyncMock(return_value=lift)),
            patch.object(cycle, "Axol", side_effect=make_axol),
            patch.object(cycle, "interrupt_event", _interrupt_context),
            patch.object(
                cycle, "_wait_for_position_save", AsyncMock(return_value=initial)
            ),
            patch.object(cycle, "_wait_for_status_after", wait_after),
            patch.object(cycle, "_move_lift", move),
            patch.object(cycle, "_ramp_arms", side_effect=ramp) as ramp_mock,
            patch.object(cycle, "_disable_arms_verified", side_effect=disable_arms),
            patch.object(cycle, "_verify_arm_targets", AsyncMock()),
            contextlib.redirect_stdout(output),
        ):
            await cycle._run(_args(cycles))

        assert fake_axol is not None
        return events, move, ramp_mock, fake_axol

    async def test_happy_path_returns_s1_to_rest_then_disables_arms(self) -> None:
        async def successful_move(  # noqa: ANN001
            lift, target, speed, interrupted, label, safety_check
        ):
            del speed, interrupted, label
            await safety_check()
            lift.status = _status(target)
            lift.last_status_monotonic = time.monotonic()
            return lift.status

        events, move, ramp, fake_axol = await self._run_with_fakes(
            initial=_status(1000), move_side_effect=successful_move
        )

        self.assertEqual(
            [item.args[1] for item in move.await_args_list],
            [0, 1000, 0, 1000],
        )
        self.assertEqual(ramp.await_count, 2)
        self.assertEqual(fake_axol.left_channel, "can-left-test")
        self.assertEqual(fake_axol.right_channel, "can-right-test")
        self.assertFalse(fake_axol.config.has_gripper)
        start_left, start_right = fake_axol._positions
        clearance_left, clearance_right = cycle._clearance_targets(
            start_left, start_right
        )
        rest_left, rest_right = cycle._rest_targets(start_left, start_right)
        second_ramp = ramp.await_args_list[1].args
        np.testing.assert_array_equal(second_ramp[1], clearance_left)
        np.testing.assert_array_equal(second_ramp[2], clearance_right)
        np.testing.assert_array_equal(second_ramp[3], rest_left)
        np.testing.assert_array_equal(second_ramp[4], rest_right)
        self.assertTrue(callable(ramp.await_args_list[1].kwargs["safety_check"]))
        first_stop = events.index("lift.stop")
        final_ramp = len(events) - 1 - events[::-1].index("arms.ramp")
        self.assertLess(first_stop, final_ramp)
        self.assertLess(final_ramp, events.index("arms.disable"))
        self.assertEqual(events.count("arms.disable"), 1)
        final_stop = len(events) - 1 - events[::-1].index("lift.stop")
        self.assertLess(final_stop, events.index("arms.disconnect"))
        self.assertEqual(events[-1], "lift.close")

    async def test_pass_is_withheld_when_final_cleanup_cannot_be_verified(
        self,
    ) -> None:
        async def successful_move(  # noqa: ANN001
            lift, target, speed, interrupted, label, safety_check
        ):
            del speed, interrupted, label
            await safety_check()
            lift.status = _status(target)
            lift.last_status_monotonic = time.monotonic()
            return lift.status

        stdout = io.StringIO()
        with self.assertRaisesRegex(cycle.DiagnosticFailure, "closing arm CAN buses"):
            await self._run_with_fakes(
                initial=_status(1000),
                move_side_effect=successful_move,
                cycles=1,
                disconnect_side_effect=[
                    OSError("close failed once"),
                    OSError("close failed twice"),
                ],
                stdout=stdout,
            )

        self.assertNotIn("PASS", stdout.getvalue())

    async def test_return_to_rest_aborts_if_lift_leaves_upper_endpoint(self) -> None:
        async def successful_move(  # noqa: ANN001
            lift, target, speed, interrupted, label, safety_check
        ):
            del speed, interrupted, label
            await safety_check()
            lift.status = _status(target)
            lift.last_status_monotonic = time.monotonic()
            return lift.status

        events: list[str] = []
        lift = FakeLift(_status(1000), events)

        def make_axol(*, config, left_channel, right_channel):  # noqa: ANN001, ANN202
            return FakeAxol(
                config=config,
                left_channel=left_channel,
                right_channel=right_channel,
                events=events,
            )

        ramp_calls = 0

        async def ramp(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            nonlocal ramp_calls
            del args
            ramp_calls += 1
            events.append("arms.ramp")
            if ramp_calls == 2:
                lift.status = _status(999, moving=True)
                await kwargs["safety_check"]()

        stderr = io.StringIO()
        with (
            patch.object(cycle, "_open_lift", AsyncMock(return_value=lift)),
            patch.object(cycle, "Axol", side_effect=make_axol),
            patch.object(cycle, "interrupt_event", _interrupt_context),
            patch.object(
                cycle,
                "_wait_for_position_save",
                AsyncMock(return_value=lift.status),
            ),
            patch.object(
                cycle,
                "_wait_for_status_after",
                AsyncMock(return_value=_status(1000)),
            ),
            patch.object(cycle, "_move_lift", AsyncMock(side_effect=successful_move)),
            patch.object(cycle, "_ramp_arms", AsyncMock(side_effect=ramp)) as ramp_mock,
            patch.object(cycle, "_verify_arm_targets", AsyncMock()),
            contextlib.redirect_stderr(stderr),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(
                cycle.DiagnosticFailure, "left its stopped upper endpoint"
            ),
        ):
            await cycle._run(_args(cycles=1))

        self.assertEqual(ramp_mock.await_count, 2)
        self.assertNotIn("arms.disable", events)
        self.assertLess(events.index("lift.stop"), events.index("arms.disconnect"))
        self.assertIn("last commanded", stderr.getvalue())

    async def test_initial_non_upper_position_is_raised_before_cycles(self) -> None:
        async def successful_move(  # noqa: ANN001
            lift, target, speed, interrupted, label, safety_check
        ):
            del speed, interrupted, label
            await safety_check()
            lift.status = _status(target)
            lift.last_status_monotonic = time.monotonic()
            return lift.status

        events: list[str] = []
        lift = FakeLift(_status(450), events)
        move = AsyncMock(side_effect=successful_move)
        wait_after = AsyncMock(
            side_effect=[
                _status(450),
                _status(1000),
                _status(1000),
                _status(1000),
            ]
        )

        def make_axol(*, config, left_channel, right_channel):  # noqa: ANN001, ANN202
            return FakeAxol(
                config=config,
                left_channel=left_channel,
                right_channel=right_channel,
                events=events,
            )

        with (
            patch.object(cycle, "_open_lift", AsyncMock(return_value=lift)),
            patch.object(cycle, "Axol", side_effect=make_axol),
            patch.object(cycle, "interrupt_event", _interrupt_context),
            patch.object(
                cycle,
                "_wait_for_position_save",
                AsyncMock(return_value=lift.status),
            ),
            patch.object(cycle, "_wait_for_status_after", wait_after),
            patch.object(cycle, "_move_lift", move),
            patch.object(cycle, "_ramp_arms", AsyncMock()),
            patch.object(
                cycle,
                "_disable_arms_verified",
                AsyncMock(side_effect=lambda _axol: events.append("arms.disable")),
            ),
            patch.object(cycle, "_verify_arm_targets", AsyncMock()),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            await cycle._run(_args(cycles=1))

        self.assertEqual(
            [item.args[1] for item in move.await_args_list], [1000, 0, 1000]
        )

    async def test_failed_upstroke_stops_and_does_not_restore_arms(self) -> None:
        calls = 0

        async def fail_up(  # noqa: ANN001
            lift, target, speed, interrupted, label, safety_check
        ):
            nonlocal calls
            del speed, interrupted, label
            await safety_check()
            calls += 1
            if calls == 2:
                raise cycle.DiagnosticFailure("upstroke failed")
            lift.status = _status(target)
            return lift.status

        events: list[str] = []
        lift = FakeLift(_status(1000), events)

        def make_axol(*, config, left_channel, right_channel):  # noqa: ANN001, ANN202
            return FakeAxol(
                config=config,
                left_channel=left_channel,
                right_channel=right_channel,
                events=events,
            )

        ramp = AsyncMock(side_effect=lambda *args, **kwargs: events.append("arms.ramp"))
        stderr = io.StringIO()
        stdout = io.StringIO()
        with (
            patch.object(cycle, "_open_lift", AsyncMock(return_value=lift)),
            patch.object(cycle, "Axol", side_effect=make_axol),
            patch.object(cycle, "interrupt_event", _interrupt_context),
            patch.object(
                cycle,
                "_wait_for_position_save",
                AsyncMock(return_value=lift.status),
            ),
            patch.object(
                cycle,
                "_wait_for_status_after",
                AsyncMock(return_value=_status(1000)),
            ),
            patch.object(cycle, "_move_lift", AsyncMock(side_effect=fail_up)),
            patch.object(cycle, "_ramp_arms", ramp),
            patch.object(cycle, "_verify_arm_targets", AsyncMock()),
            contextlib.redirect_stderr(stderr),
            contextlib.redirect_stdout(stdout),
            self.assertRaisesRegex(cycle.DiagnosticFailure, "upstroke failed"),
        ):
            await cycle._run(_args(cycles=1))

        self.assertEqual(ramp.await_count, 1)
        self.assertNotIn("arms.disable", events)
        self.assertLess(events.index("lift.stop"), events.index("arms.disconnect"))
        self.assertIn("90 degree clearance", stderr.getvalue())

    async def test_wait_checks_arm_clearance_while_waiting_for_status(self) -> None:
        events: list[str] = []
        lift = FakeLift(_status(1000), events)
        after = lift.last_status_monotonic

        async def safety_check() -> None:
            lift.last_status_monotonic = after + 1.0

        safety_mock = AsyncMock(side_effect=safety_check)
        result = await cycle._wait_for_status_after(
            lift,
            after,
            asyncio.Event(),
            "test",
            safety_mock,
        )

        self.assertIs(result, lift.status)
        safety_mock.assert_awaited_once_with()

    async def test_position_move_requires_v08_save_pending_behavior(self) -> None:
        events: list[str] = []
        lift = FakeLift(_status(1000), events)
        old_firmware_motion = _status(
            900,
            moving=True,
            pos_move=True,
            save_pending=False,
        )
        with (
            patch.object(
                cycle,
                "_wait_for_status_after",
                AsyncMock(return_value=old_firmware_motion),
            ),
            self.assertRaisesRegex(cycle.DiagnosticFailure, "v0.8"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            await cycle._move_lift(
                lift,
                0,
                0,
                asyncio.Event(),
                "down",
            )


if __name__ == "__main__":
    unittest.main()
