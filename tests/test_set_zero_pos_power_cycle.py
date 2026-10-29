"""Guided zeroing power-cycles Axol through Jelly's safety board, where fitted,
so new Damiao zeros apply without a manual power cycle."""

from __future__ import annotations

import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from almond_axol.cli.motor import set_zero_pos
from almond_axol.constants import Joint
from almond_axol.robot.safety import SafetyError


class FakeBoard:
    """Stands in for ``SafetyBoard``."""

    present = True
    cycle_error: SafetyError | None = None
    instances: list[FakeBoard] = []

    def __init__(self, channel: str) -> None:
        self.channel = channel
        self.calls: list[str] = []
        FakeBoard.instances.append(self)

    async def start(self, *, bring_up: bool = True) -> None:
        self.calls.append("start")

    async def request_status(self):  # noqa: ANN201
        if not FakeBoard.present:
            raise SafetyError("The safety board is not answering on can-test.")
        return None

    async def power_cycle(self):  # noqa: ANN201
        self.calls.append("power_cycle")
        if FakeBoard.cycle_error is not None:
            raise FakeBoard.cycle_error

    async def close(self) -> None:
        self.calls.append("close")


class FakeMotor:
    def __init__(self, joint: Joint) -> None:
        self.joint = joint

    async def get_position(self) -> float:
        return 0.01


class ApplyDamiaoZerosTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        FakeBoard.present = True
        FakeBoard.cycle_error = None
        FakeBoard.instances = []
        self.iface_up = True
        self.prompt_answer = True
        self.prompts: list[str] = []

        def prompt(instruction: str, web_prompts: bool) -> bool:
            self.prompts.append(instruction)
            return self.prompt_answer

        for p in (
            patch.object(set_zero_pos, "SafetyBoard", FakeBoard),
            patch.object(
                set_zero_pos, "resolve_safety_channel", return_value="can-test"
            ),
            patch.object(set_zero_pos, "_prompt", side_effect=prompt),
            patch(
                "almond_axol.cli.can.setup.iface_up",
                side_effect=lambda _c: self.iface_up,
            ),
        ):
            p.start()
            self.addCleanup(p.stop)

    async def _apply(self) -> str:
        out = io.StringIO()
        motors = [FakeMotor(Joint.WRIST_2), FakeMotor(Joint.WRIST_3)]
        with contextlib.redirect_stdout(out):
            await set_zero_pos._apply_damiao_zeros(motors, web_prompts=True)  # type: ignore[arg-type]
        return out.getvalue()

    async def test_power_cycles_after_the_operator_confirms(self) -> None:
        out = await self._apply()
        self.assertEqual(len(self.prompts), 1)
        self.assertIn("support both arms", self.prompts[0])
        self.assertEqual(
            FakeBoard.instances[0].calls, ["start", "power_cycle", "close"]
        )
        self.assertIn("Damiao zeros applied", out)
        self.assertNotIn("power-cycle required", out)

    async def test_without_a_lift_bus_asks_for_a_manual_power_cycle(self) -> None:
        self.iface_up = False
        out = await self._apply()
        self.assertEqual(FakeBoard.instances, [])
        self.assertEqual(self.prompts, [])
        self.assertIn("WRIST_2, WRIST_3) — power-cycle required", out)

    async def test_without_a_safety_board_asks_for_a_manual_power_cycle(self) -> None:
        FakeBoard.present = False
        out = await self._apply()
        self.assertEqual(self.prompts, [])
        self.assertEqual(FakeBoard.instances[0].calls, ["start", "close"])
        self.assertIn("power-cycle required", out)

    async def test_declined_keeps_power_on(self) -> None:
        self.prompt_answer = False
        out = await self._apply()
        self.assertNotIn("power_cycle", FakeBoard.instances[0].calls)
        self.assertIn("close", FakeBoard.instances[0].calls)
        self.assertIn("power-cycle required", out)

    async def test_failed_cycle_says_why(self) -> None:
        FakeBoard.cycle_error = SafetyError("The E-Stop is pressed.")
        out = await self._apply()
        self.assertIn("power cycle failed: The E-Stop is pressed.", out)
        self.assertIn("power-cycle required", out)
        self.assertIn("close", FakeBoard.instances[0].calls)


class GuidedRunTest(unittest.IsolatedAsyncioTestCase):
    async def test_only_zeroed_damiao_joints_trigger_the_power_cycle(self) -> None:
        applied: list[list[Joint]] = []

        async def apply(motors, web_prompts):  # noqa: ANN001, ANN202
            applied.append([m.joint for m in motors])

        class Bus:
            def __init__(self, channel: str) -> None:
                pass

            async def __aenter__(self):  # noqa: ANN204
                return self

            async def __aexit__(self, *exc) -> None:  # noqa: ANN002
                return None

        class Driver:
            pass

        class Motor:
            def __init__(self, bus, joint: Joint) -> None:  # noqa: ANN001
                self.joint = joint
                self._driver = (
                    set_zero_pos.DamiaoMotor.__new__(set_zero_pos.DamiaoMotor)
                    if joint in (Joint.WRIST_2, Joint.WRIST_3)
                    else Driver()
                )

            async def get_position(self) -> float:
                return 0.0

        async def calibrate(motor, joint, *args):  # noqa: ANN001, ANN002, ANN202
            return joint != Joint.WRIST_3  # WRIST_3 skipped by the operator

        args = SimpleNamespace(l=True, joints="elbow,wrist_2,wrist_3", web_prompts=True)
        with (
            patch.object(set_zero_pos, "resolve_channel", return_value="can-l"),
            patch.object(set_zero_pos, "CanBus", Bus),
            patch.object(set_zero_pos, "Motor", Motor),
            patch.object(set_zero_pos, "_prompt", return_value=True),
            patch.object(set_zero_pos, "_calibrate_joint", side_effect=calibrate),
            patch.object(
                set_zero_pos, "_calibrate_joint_either_stop", side_effect=calibrate
            ),
            patch.object(set_zero_pos, "_apply_damiao_zeros", side_effect=apply),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            await set_zero_pos._run_guided(args)  # type: ignore[arg-type]
        self.assertEqual(applied, [[Joint.WRIST_2]])


if __name__ == "__main__":
    unittest.main()
