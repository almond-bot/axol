"""Safety and dispatch tests for ``diag.mantis-trigger``."""

from __future__ import annotations

import unittest
from collections.abc import Callable
from io import StringIO
from typing import Any
from unittest import mock

import numpy as np

import almond_axol.cli as cli_entrypoint
from almond_axol.diagnostics.mantis import trigger as mantis_trigger
from almond_axol.robot.base import HardwareCleanupError


class _FakeReader:
    def __init__(
        self,
        channel: str,
        grip: float | None = 1.0,
        *,
        close_error: BaseException | None = None,
    ) -> None:
        self.channel = channel
        self.value = grip
        self.stale = grip is None
        self.closed = False
        self.close_error = close_error

    def grip(self) -> float | None:
        return self.value

    def is_stale(self) -> bool:
        return self.stale

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class _FakeRobot:
    def __init__(
        self,
        readers: dict[str, _FakeReader],
        *,
        on_motion: Callable[[], None] | None = None,
        enable_error: BaseException | None = None,
        disable_error: BaseException | None = None,
    ) -> None:
        self.readers = readers
        self.on_motion = on_motion
        self.enable_error = enable_error
        self.disable_error = disable_error
        self.connected = False
        self.enabled = False
        self.disable_calls = 0
        self.commands: list[tuple[np.ndarray | None, np.ndarray | None]] = []

    async def connect(self) -> None:
        self.connected = True

    async def enable(self) -> None:
        # Support implementations which use ``async with Mantis(...)``;
        # deferred enable connects the buses but does not energise the jaws.
        await self.connect()

    async def __aenter__(self) -> _FakeRobot:
        await self.enable()
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.disable()

    async def enable_grippers(self) -> None:
        # A diagnostic must not energise either jaw while a trigger is already
        # squeezed: doing so could produce an immediate, surprising close.
        for reader in self.readers.values():
            if reader.stale or reader.value is None or reader.value < 0.8:
                raise AssertionError("grippers enabled before both triggers released")
        if self.enable_error is not None:
            raise self.enable_error
        self.enabled = True

    async def motion_control(
        self,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
    ) -> None:
        self.commands.append((left, right))
        if self.on_motion is not None:
            self.on_motion()

    async def get_positions(self) -> tuple[np.ndarray, np.ndarray]:
        return np.ones(8), np.ones(8)

    async def disable(self) -> None:
        self.enabled = False
        self.disable_calls += 1
        if self.disable_error is not None:
            raise self.disable_error


class MantisTriggerTestCommandTest(unittest.IsolatedAsyncioTestCase):
    async def test_maps_fresh_trigger_grips_without_trackers_or_cameras(self) -> None:
        readers: dict[str, _FakeReader] = {}
        robot: _FakeRobot | None = None

        def reader_factory(channel: str) -> _FakeReader:
            # Exercise the exact release threshold on one side.
            value = 0.8 if channel == "left-can" else 1.0
            reader = _FakeReader(channel, value)
            readers[channel] = reader
            return reader

        def robot_factory(*args: Any, **kwargs: Any) -> _FakeRobot:
            nonlocal robot
            self.assertFalse(args)
            self.assertEqual(kwargs["left_channel"], "left-can")
            self.assertEqual(kwargs["right_channel"], "right-can")
            self.assertTrue(kwargs["defer_gripper_enable"])

            def lose_heartbeat() -> None:
                # End the otherwise-continuous command loop through its safety
                # path immediately after capturing one proportional command.
                readers["left-can"].stale = True

            robot = _FakeRobot(readers, on_motion=lose_heartbeat)
            return robot

        with self.assertRaisesRegex(RuntimeError, "trigger|heartbeat|stale"):
            await mantis_trigger._run(
                "left-can",
                "right-can",
                reader_factory=reader_factory,
                robot_factory=robot_factory,
                wait_timeout=0.1,
                poll_interval=0.0,
                status_interval=60.0,
            )

        assert robot is not None
        self.assertTrue(robot.connected)
        self.assertEqual(len(robot.commands), 1)
        left, right = robot.commands[0]
        assert left is not None and right is not None
        np.testing.assert_array_equal(left[:7], np.zeros(7))
        np.testing.assert_array_equal(right[:7], np.zeros(7))
        self.assertEqual(left.shape, (8,))
        self.assertEqual(right.shape, (8,))
        self.assertAlmostEqual(float(left[7]), 0.8)
        self.assertAlmostEqual(float(right[7]), 1.0)
        self.assertGreaterEqual(robot.disable_calls, 1)
        self.assertTrue(all(reader.closed for reader in readers.values()))

    async def test_waits_for_both_fresh_released_triggers_before_enable(self) -> None:
        readers: dict[str, _FakeReader] = {}
        robot: _FakeRobot | None = None

        def reader_factory(channel: str) -> _FakeReader:
            # A live but half-squeezed trigger is not a safe arming state.
            reader = _FakeReader(channel, 0.5)
            readers[channel] = reader
            return reader

        def robot_factory(*_args: Any, **_kwargs: Any) -> _FakeRobot:
            nonlocal robot
            robot = _FakeRobot(readers)
            return robot

        with self.assertRaisesRegex(RuntimeError, "release|released|trigger"):
            await mantis_trigger._run(
                "left-can",
                "right-can",
                reader_factory=reader_factory,
                robot_factory=robot_factory,
                wait_timeout=0.001,
                poll_interval=0.0,
                status_interval=60.0,
            )

        assert robot is not None
        self.assertFalse(robot.enabled)
        self.assertEqual(robot.commands, [])
        self.assertGreaterEqual(robot.disable_calls, 1)
        self.assertTrue(all(reader.closed for reader in readers.values()))

    async def test_never_enables_when_trigger_heartbeat_is_missing(self) -> None:
        readers: dict[str, _FakeReader] = {}
        robot: _FakeRobot | None = None

        def reader_factory(channel: str) -> _FakeReader:
            reader = _FakeReader(channel, None)
            readers[channel] = reader
            return reader

        def robot_factory(*_args: Any, **_kwargs: Any) -> _FakeRobot:
            nonlocal robot
            robot = _FakeRobot(readers)
            return robot

        with self.assertRaisesRegex(RuntimeError, "trigger|heartbeat|frame"):
            await mantis_trigger._run(
                "left-can",
                "right-can",
                reader_factory=reader_factory,
                robot_factory=robot_factory,
                wait_timeout=0.001,
                poll_interval=0.0,
                status_interval=60.0,
            )

        assert robot is not None
        self.assertFalse(robot.enabled)
        self.assertEqual(robot.commands, [])
        self.assertGreaterEqual(robot.disable_calls, 1)
        self.assertTrue(all(reader.closed for reader in readers.values()))

    async def test_enable_failure_and_partial_reader_construction_clean_up(
        self,
    ) -> None:
        with self.subTest("enable failure"):
            readers: dict[str, _FakeReader] = {}
            robot: _FakeRobot | None = None

            def reader_factory(channel: str) -> _FakeReader:
                reader = _FakeReader(channel)
                readers[channel] = reader
                return reader

            def robot_factory(*_args: Any, **_kwargs: Any) -> _FakeRobot:
                nonlocal robot
                robot = _FakeRobot(readers, enable_error=OSError("motor offline"))
                return robot

            with self.assertRaisesRegex(OSError, "motor offline"):
                await mantis_trigger._run(
                    "left-can",
                    "right-can",
                    reader_factory=reader_factory,
                    robot_factory=robot_factory,
                    wait_timeout=0.1,
                    poll_interval=0.0,
                    status_interval=60.0,
                )

            assert robot is not None
            self.assertGreaterEqual(robot.disable_calls, 1)
            self.assertTrue(all(reader.closed for reader in readers.values()))

        with self.subTest("second reader construction failure"):
            first = _FakeReader("left-can")
            calls = 0

            def broken_reader_factory(_channel: str) -> _FakeReader:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return first
                raise OSError("right trigger socket failed")

            with self.assertRaisesRegex(OSError, "right trigger socket failed"):
                await mantis_trigger._run(
                    "left-can",
                    "right-can",
                    reader_factory=broken_reader_factory,
                    robot_factory=lambda **_kwargs: self.fail(
                        "robot must not be constructed after reader setup fails"
                    ),
                    wait_timeout=0.1,
                    poll_interval=0.0,
                    status_interval=60.0,
                )

            self.assertTrue(first.closed)

    async def test_command_failure_or_interrupt_always_disables_and_closes(
        self,
    ) -> None:
        for failure in (OSError("CAN transmit failed"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                readers: dict[str, _FakeReader] = {}
                robot: _FakeRobot | None = None

                def reader_factory(channel: str) -> _FakeReader:
                    reader = _FakeReader(channel)
                    readers[channel] = reader
                    return reader

                def robot_factory(*_args: Any, **_kwargs: Any) -> _FakeRobot:
                    nonlocal robot

                    def fail_command() -> None:
                        raise failure

                    robot = _FakeRobot(readers, on_motion=fail_command)
                    return robot

                with self.assertRaises(type(failure)):
                    await mantis_trigger._run(
                        "left-can",
                        "right-can",
                        reader_factory=reader_factory,
                        robot_factory=robot_factory,
                        wait_timeout=0.1,
                        poll_interval=0.0,
                        status_interval=60.0,
                    )

                assert robot is not None
                self.assertEqual(len(robot.commands), 1)
                self.assertGreaterEqual(robot.disable_calls, 1)
                self.assertTrue(all(reader.closed for reader in readers.values()))

    async def test_disable_failure_still_closes_both_trigger_readers(self) -> None:
        readers: dict[str, _FakeReader] = {}

        def reader_factory(channel: str) -> _FakeReader:
            reader = _FakeReader(channel)
            readers[channel] = reader
            return reader

        def robot_factory(*_args: Any, **_kwargs: Any) -> _FakeRobot:
            return _FakeRobot(readers, disable_error=OSError("disable failed"))

        with self.assertRaisesRegex(
            HardwareCleanupError, "hardware ownership is uncertain"
        ) as raised:
            await mantis_trigger._run(
                "left-can",
                "right-can",
                reader_factory=reader_factory,
                robot_factory=robot_factory,
                wait_timeout=0.1,
                poll_interval=0.0,
                status_interval=60.0,
                duration=0.0,
            )

        self.assertIsInstance(raised.exception.__cause__, OSError)
        self.assertTrue(all(reader.closed for reader in readers.values()))

    async def test_reader_failure_still_disables_and_closes_other_reader(self) -> None:
        readers: dict[str, _FakeReader] = {}
        robot: _FakeRobot | None = None

        def reader_factory(channel: str) -> _FakeReader:
            reader = _FakeReader(
                channel,
                close_error=(
                    OSError("left socket stuck") if channel == "left-can" else None
                ),
            )
            readers[channel] = reader
            return reader

        def robot_factory(*_args: Any, **_kwargs: Any) -> _FakeRobot:
            nonlocal robot
            robot = _FakeRobot(readers)
            return robot

        with self.assertRaisesRegex(RuntimeError, "CAN ownership is uncertain"):
            await mantis_trigger._run(
                "left-can",
                "right-can",
                reader_factory=reader_factory,
                robot_factory=robot_factory,
                wait_timeout=0.1,
                poll_interval=0.0,
                status_interval=60.0,
                duration=0.0,
            )

        assert robot is not None
        self.assertEqual(robot.disable_calls, 1)
        self.assertTrue(all(reader.closed for reader in readers.values()))


class MantisTriggerTestDispatchTest(unittest.TestCase):
    def test_web_confirmation_emits_dashboard_marker_and_fails_closed(self) -> None:
        output = StringIO()
        with (
            mock.patch.object(mantis_trigger.sys, "stdin", StringIO("\n")),
            mock.patch("sys.stdout", output),
        ):
            mantis_trigger._confirm("left-can", "right-can", web_prompts=True)
        self.assertIn("[prompt] Jaws are clear", output.getvalue())

        with (
            mock.patch.object(mantis_trigger.sys, "stdin", StringIO("")),
            mock.patch("sys.stdout", StringIO()),
            self.assertRaisesRegex(SystemExit, "not enabled"),
        ):
            mantis_trigger._confirm("left-can", "right-can", web_prompts=True)

    def test_diag_command_is_dispatched_lazily_with_untouched_arguments(self) -> None:
        diagnostic_module = mock.Mock()
        argv = ["diag.mantis-trigger", "--yes", "--duration", "2"]

        with (
            mock.patch.object(cli_entrypoint, "load_local_env"),
            mock.patch.object(
                cli_entrypoint.importlib,
                "import_module",
                return_value=diagnostic_module,
            ) as import_module,
            mock.patch.object(cli_entrypoint.sys, "argv", ["axol", *argv]),
        ):
            cli_entrypoint.main()

        import_module.assert_called_once_with("almond_axol.diagnostics.mantis.trigger")
        diagnostic_module.main.assert_called_once_with(argv[1:])


if __name__ == "__main__":
    unittest.main()
