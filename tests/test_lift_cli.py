from __future__ import annotations

import asyncio
import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from almond_axol.cli import lift as lift_cli
from almond_axol.cli.lift import goto, home
from almond_axol.robot.lift import LiftStatus


def _status(
    *,
    moving: bool = False,
    homed: bool = True,
    stall_fault: bool = False,
    driver_fault_mask: int | None = 0,
    drivers_enabled: bool | None = True,
    vm_present: bool | None = True,
    flash_interlock: bool | None = False,
    save_pending: bool | None = False,
) -> LiftStatus:
    return LiftStatus(
        position_permille=500,
        velocity=0,
        drift=0,
        homed=homed,
        moving=moving,
        pos_move=moving,
        stall_fault=stall_fault,
        at_lower=False,
        at_upper=False,
        homing=False,
        jog=False,
        driver_fault_mask=driver_fault_mask,
        drivers_enabled=drivers_enabled,
        vm_present=vm_present,
        flash_interlock=flash_interlock,
        save_pending=save_pending,
    )


class FakeLift:
    def __init__(self, status: LiftStatus) -> None:
        self.status = status
        self.last_status_monotonic = 1.0
        self.confirm_stop = True
        self.stop_motion = AsyncMock(side_effect=self._stop_motion)

    async def _stop_motion(self) -> None:
        if self.confirm_stop:
            asyncio.get_running_loop().call_later(0.001, self._confirm_stopped)

    def _confirm_stopped(self) -> None:
        self.status = _status(moving=False)
        self.last_status_monotonic = lift_cli.time.monotonic()

    def status_is_fresh(self, max_age_s: float) -> bool:
        return max_age_s >= 0


class LiftCliPreflightTest(unittest.TestCase):
    def test_healthy_idle_status_passes(self) -> None:
        lift = FakeLift(_status())

        result = lift_cli.require_motion_preflight(
            lift, operation="position move", require_homed=True
        )

        self.assertIs(result, lift.status)

    def test_every_controller_fault_blocks_direct_motion(self) -> None:
        cases = (
            (_status(stall_fault=True), "stall"),
            (_status(driver_fault_mask=1), "fault mask"),
            (_status(drivers_enabled=False), "disabled"),
            (_status(vm_present=False), "24 V"),
            (_status(flash_interlock=True), "interlock"),
            (_status(save_pending=True), "save"),
            (_status(moving=True), "already moving"),
            (_status(homed=False), "not homed"),
            (_status(driver_fault_mask=None), "current driver/interlock"),
        )
        for status, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(SystemExit, message),
            ):
                lift_cli.require_motion_preflight(
                    FakeLift(status),
                    operation="position move",
                    require_homed=True,
                )

    def test_stale_status_blocks_direct_motion(self) -> None:
        lift = FakeLift(_status())
        lift.status_is_fresh = lambda _max_age_s: False

        with self.assertRaisesRegex(SystemExit, "stale"):
            lift_cli.require_motion_preflight(
                lift, operation="homing", require_homed=False
            )


class LiftCliWatchTest(unittest.IsolatedAsyncioTestCase):
    async def test_stop_requires_fresh_idle_confirmation(self) -> None:
        lift = FakeLift(_status(moving=True))
        lift.confirm_stop = False

        with (
            patch.object(lift_cli, "STATUS_STALE_S", 0.0),
            self.assertRaisesRegex(lift_cli.StopNotVerified, "not confirmed"),
        ):
            await lift_cli._stop_motion_verified(lift)  # noqa: SLF001

        lift.stop_motion.assert_awaited_once_with()

    async def test_requires_fresh_post_command_status_and_stops_on_timeout(
        self,
    ) -> None:
        lift = FakeLift(_status())

        with (
            patch.object(lift_cli, "STATUS_STALE_S", 0.02),
            self.assertRaisesRegex(TimeoutError, "status stopped updating"),
        ):
            await lift_cli.watch_motion(
                lift,
                started=lambda status: status.moving,
                finished=lambda status: not status.moving,
                start_timeout_s=1.0,
                timeout_s=2.0,
                interrupted=asyncio.Event(),
                commanded_at=lift.last_status_monotonic,
            )

        lift.stop_motion.assert_awaited_once_with()

    async def test_success_uses_new_frames_and_stops_before_return(self) -> None:
        lift = FakeLift(_status())

        async def publish_statuses() -> None:
            await asyncio.sleep(0.01)
            lift.status = _status(moving=True)
            lift.last_status_monotonic = 2.0
            await asyncio.sleep(0.05)
            lift.status = _status(moving=False)
            lift.last_status_monotonic = 3.0

        publisher = asyncio.create_task(publish_statuses())
        result = await lift_cli.watch_motion(
            lift,
            started=lambda status: status.moving,
            finished=lambda status: not status.moving,
            start_timeout_s=1.0,
            timeout_s=2.0,
            interrupted=asyncio.Event(),
            commanded_at=lift.last_status_monotonic,
        )
        await publisher

        self.assertFalse(result.moving)
        lift.stop_motion.assert_awaited_once_with()

    async def test_interruption_also_stops(self) -> None:
        lift = FakeLift(_status(moving=True))
        interrupted = asyncio.Event()
        interrupted.set()

        with self.assertRaises(lift_cli.Interrupted):
            await lift_cli.watch_motion(
                lift,
                started=lambda status: status.moving,
                finished=lambda status: not status.moving,
                start_timeout_s=1.0,
                timeout_s=2.0,
                interrupted=interrupted,
                commanded_at=lift.last_status_monotonic,
            )

        lift.stop_motion.assert_awaited_once_with()

    async def test_stop_failure_is_never_hidden(self) -> None:
        lift = FakeLift(_status())
        lift.stop_motion.side_effect = OSError("STOP failed")

        with (
            patch.object(lift_cli, "STATUS_STALE_S", 0.0),
            self.assertRaisesRegex(OSError, "STOP failed"),
        ):
            await lift_cli.watch_motion(
                lift,
                started=lambda status: status.moving,
                finished=lambda status: not status.moving,
                start_timeout_s=1.0,
                timeout_s=2.0,
                interrupted=asyncio.Event(),
                commanded_at=lift.last_status_monotonic,
            )


class LiftCliCommandInterlockTest(unittest.IsolatedAsyncioTestCase):
    async def test_open_lift_closes_partial_start_after_dropped_rate(self) -> None:
        lift = SimpleNamespace(
            start=AsyncMock(side_effect=OSError("SET_RATE was not delivered")),
            close=AsyncMock(),
        )
        with (
            patch.object(lift_cli, "Lift", return_value=lift),
            self.assertRaisesRegex(SystemExit, "SET_RATE was not delivered"),
        ):
            await lift_cli.open_lift("can-test")

        lift.close.assert_awaited_once_with()

    async def test_open_lift_preserves_cleanup_cancellation(self) -> None:
        lift = SimpleNamespace(
            start=AsyncMock(side_effect=OSError("startup failed")),
            close=AsyncMock(side_effect=asyncio.CancelledError),
        )
        with (
            patch.object(lift_cli, "Lift", return_value=lift),
            self.assertRaises(asyncio.CancelledError),
        ):
            await lift_cli.open_lift("can-test")

        lift.close.assert_awaited_once_with()

    async def test_direct_commands_recheck_interrupt_after_driver_stop(self) -> None:
        for module, method, args, status in (
            (
                goto,
                "set_position",
                SimpleNamespace(channel="can-test", percent=50.0, speed=0),
                _status(),
            ),
            (
                home,
                "home",
                SimpleNamespace(channel="can-test"),
                _status(homed=False),
            ),
        ):
            with self.subTest(command=method):
                interrupted = asyncio.Event()
                lift = FakeLift(status)
                lift.close = AsyncMock()
                sent_motion = False

                async def command(*_args, before_send, **_kwargs) -> None:  # noqa: ANN003
                    nonlocal sent_motion
                    interrupted.set()
                    await before_send()
                    sent_motion = True

                setattr(lift, method, command)

                @contextlib.contextmanager
                def interrupted_context():
                    yield interrupted

                with (
                    patch.object(module, "open_lift", AsyncMock(return_value=lift)),
                    patch.object(module, "interrupt_event", interrupted_context),
                    self.assertRaisesRegex(SystemExit, "Interrupted"),
                ):
                    await module._run(args)  # noqa: SLF001

                self.assertFalse(sent_motion)
                lift.close.assert_awaited_once_with()


if __name__ == "__main__":
    unittest.main()
