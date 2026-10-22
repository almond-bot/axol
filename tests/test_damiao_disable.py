"""Damiao disable clears the comm-timeout fault a crashed session leaves."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from almond_axol.motor.damiao import DamiaoMotor, _DamiaoStatus, _MotorFeedback
from almond_axol.motor.errors import MotorError

_CLEAR = bytes([0xFF] * 7 + [0xFB])
_DISABLE = bytes([0xFF] * 7 + [0xFD])


class _Gripper(DamiaoMotor):
    """A Damiao whose status follows the commands it receives; no bus."""

    def __init__(self, status: _DamiaoStatus, *, clears: bool = True) -> None:
        super().__init__(MagicMock(), 0x08, 0x18)
        self.status = status
        self.clears = clears
        self.sent: list[bytes] = []

    async def _raw_send(self, data: bytes, arb_id: int | None = None) -> None:  # type: ignore[override]
        self.sent.append(data)
        if data == _CLEAR and self.clears:
            self.status = _DamiaoStatus.DISABLED
        if data == _DISABLE and self.status == _DamiaoStatus.ENABLED:
            self.status = _DamiaoStatus.DISABLED

    async def _request_feedback(self, timeout=0.1, attempts=5):  # type: ignore[override]
        return _MotorFeedback(
            status=self.status,
            position=0.0,
            velocity=0.0,
            torque=0.0,
            t_mos=25.0,
            t_rotor=25.0,
        )


class DamiaoDisableTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        patcher = patch("almond_axol.motor.damiao.asyncio.sleep")
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_a_lost_comm_fault_is_cleared_into_disabled(self) -> None:
        motor = _Gripper(_DamiaoStatus.LOST_COMM)

        await motor.disable()

        self.assertEqual(motor.status, _DamiaoStatus.DISABLED)
        self.assertIn(_CLEAR, motor.sent)

    async def test_an_enabled_motor_is_disabled_without_clearing(self) -> None:
        motor = _Gripper(_DamiaoStatus.ENABLED)

        await motor.disable()

        self.assertEqual(motor.status, _DamiaoStatus.DISABLED)
        self.assertNotIn(_CLEAR, motor.sent)

    async def test_other_faults_are_not_cleared_and_still_refuse(self) -> None:
        # Over-current and the like must stay visible to the operator.
        motor = _Gripper(_DamiaoStatus.OVER_CURRENT)

        with self.assertRaisesRegex(MotorError, "did not confirm disabled"):
            await motor.disable()

        self.assertNotIn(_CLEAR, motor.sent)
        self.assertEqual(motor.status, _DamiaoStatus.OVER_CURRENT)


if __name__ == "__main__":
    unittest.main()
