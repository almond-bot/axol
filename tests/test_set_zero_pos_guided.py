"""Guided zeroing: direction flags first, and no zero on a wrong-way move."""

from __future__ import annotations

import argparse
import math
import unittest
from contextlib import asynccontextmanager
from unittest import mock

from almond_axol.cli.motor import set_zero_pos
from almond_axol.constants import Joint


class _FakeMotor:
    def __init__(self, positions: list[float], flag_changes: bool = False):
        self._positions = list(positions)
        self.flag_changes = flag_changes
        self.calls: list[str] = []
        self._driver = object()  # not a Damiao: no power-cycle note

    async def get_position(self) -> float:
        self.calls.append("get_position")
        return self._positions.pop(0)

    async def set_zero_position(self) -> None:
        self.calls.append("set_zero_position")

    async def set_direction_flag(self) -> bool:
        self.calls.append("set_direction_flag")
        return self.flag_changes


class CalibrateJointTest(unittest.IsolatedAsyncioTestCase):
    """The elbow on the left arm zeroes at +150 deg, reached by + motion."""

    async def _run(self, motor: _FakeMotor, answers: list[bool]) -> bool:
        prompts: list[str] = []

        def prompt(instruction: str, web_prompts: bool) -> bool:
            prompts.append(instruction)
            return answers.pop(0)

        self.prompts = prompts
        with mock.patch.object(set_zero_pos, "_prompt", prompt):
            return await set_zero_pos._calibrate_joint(
                motor,  # type: ignore[arg-type]
                Joint.ELBOW,
                0.2,
                math.radians(150),
                +1,
                False,
            )

    async def test_right_way_zeroes(self) -> None:
        motor = _FakeMotor([0.2 + math.radians(140)])
        self.assertTrue(await self._run(motor, [True]))
        self.assertEqual(motor.calls.count("set_zero_position"), 1)

    async def test_wrong_way_asks_again_and_never_zeroes_there(self) -> None:
        motor = _FakeMotor([0.2 - math.radians(140), 0.2 + math.radians(140)])
        self.assertTrue(await self._run(motor, [True, True]))
        self.assertIn("moved the wrong way", self.prompts[1])
        # Zeroed once, after the move in the expected direction.
        self.assertEqual(
            motor.calls, ["get_position", "get_position", "set_zero_position"]
        )

    async def test_a_reversed_motor_is_skipped_not_zeroed(self) -> None:
        # The joint reaches the right stop but reads the wrong way each time.
        motor = _FakeMotor([0.2 - math.radians(140)] * 2)
        self.assertFalse(await self._run(motor, [True, True, False]))
        self.assertNotIn("set_zero_position", motor.calls)


class GuidedRunTest(unittest.IsolatedAsyncioTestCase):
    async def test_direction_flags_are_set_before_the_start_reference(self) -> None:
        motor = _FakeMotor([0.2, 0.2 + math.radians(140)], flag_changes=True)

        @asynccontextmanager
        async def fake_bus(channel: str):  # type: ignore[no-untyped-def]
            yield object()

        args = argparse.Namespace(
            l=True, r=False, channel="can_test", joints="elbow", web_prompts=False
        )
        with (
            mock.patch.object(set_zero_pos, "CanBus", fake_bus),
            mock.patch.object(set_zero_pos, "Motor", lambda bus, joint: motor),
            mock.patch.object(set_zero_pos, "_prompt", lambda *a: True),
        ):
            await set_zero_pos._run_guided(args)
        self.assertEqual(
            motor.calls,
            [
                "set_direction_flag",
                "get_position",
                "get_position",
                "set_zero_position",
            ],
        )


if __name__ == "__main__":
    unittest.main()
