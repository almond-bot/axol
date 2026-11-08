"""Damiao status codes: encoder faults are named, undocumented codes are faults."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from almond_axol.motor.damiao import DamiaoMotor, _DamiaoStatus
from almond_axol.motor.observer import JointObservation, _DamiaoDecoder
from almond_axol.motor.types import MotorStatus
from almond_axol.serve.robot_link import motor_faults


def _feedback(status_nibble: int, motor_id: int = 0x08) -> bytes:
    """A Damiao MIT feedback frame at mid-range with the given status nibble."""
    return bytes(
        [(status_nibble << 4) | motor_id, 0x7F, 0xFF, 0x7F, 0xF7, 0xFF, 24, 22]
    )


class DamiaoDriverStatusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.motor = DamiaoMotor(MagicMock(), 0x08, 0x18)

    def test_documented_codes_decode_as_before(self) -> None:
        self.motor._handle_feedback(_feedback(0x0), 0.0)
        self.assertEqual(self.motor.last_status, MotorStatus.DISABLED)
        self.motor._handle_feedback(_feedback(0xA), 0.0)
        self.assertEqual(self.motor.last_status, MotorStatus.OVER_CURRENT)

    def test_encoder_fault_codes_are_named(self) -> None:
        # A Mantis gripper reported 0x3 with its LED flashing red.
        cases = {
            0x3: MotorStatus.OUTPUT_SHAFT_CALIBRATION_ERROR,
            0x4: MotorStatus.SENSOR_ERROR,
            0x5: MotorStatus.MOTOR_ENCODER_CALIBRATION_ERROR,
        }
        for code, expected in cases.items():
            with self.subTest(code=code):
                self.motor._handle_feedback(_feedback(code), 0.0)
                self.assertEqual(self.motor.last_status, expected)

    def test_undocumented_code_is_unknown_not_disabled(self) -> None:
        with self.assertLogs("almond_axol.motor.damiao", "WARNING") as logs:
            self.motor._handle_feedback(_feedback(0x6), 0.0)
        self.assertEqual(self.motor.last_status, MotorStatus.UNKNOWN)
        self.assertNotEqual(self.motor._feedback.status, _DamiaoStatus.DISABLED)
        self.assertEqual(self.motor._feedback.status, 0x6)
        self.assertIn("0x6", logs.output[0])
        self.assertIn("0x08", logs.output[0])

    def test_undocumented_code_logs_once_until_it_changes(self) -> None:
        with self.assertLogs("almond_axol.motor.damiao", "WARNING") as logs:
            for _ in range(5):
                self.motor._handle_feedback(_feedback(0x6), 0.0)
            self.motor._handle_feedback(_feedback(0x0), 0.0)
            self.motor._handle_feedback(_feedback(0x6), 0.0)
            self.motor._handle_feedback(_feedback(0x7), 0.0)
        self.assertEqual(len(logs.output), 3)


class DamiaoObserverStatusTest(unittest.TestCase):
    def test_undocumented_code_is_unknown(self) -> None:
        obs = JointObservation()
        decoder = _DamiaoDecoder(obs, 0x08)
        decoder.on_feedback(_feedback(0x6), 1.0)
        self.assertEqual(obs.status, "UNKNOWN")
        decoder.on_feedback(_feedback(0x3), 1.5)
        self.assertEqual(obs.status, "OUTPUT_SHAFT_CALIBRATION_ERROR")
        decoder.on_feedback(_feedback(0x0), 2.0)
        self.assertEqual(obs.status, "DISABLED")


class UnknownStatusFaultTest(unittest.TestCase):
    def test_unknown_status_is_reported_as_a_fault(self) -> None:
        motors = [
            {"arm": "left", "joint": "GRIPPER", "reachable": True, "status": "UNKNOWN"},
            {
                "arm": "right",
                "joint": "GRIPPER",
                "reachable": True,
                "status": "DISABLED",
            },
        ]
        faults = motor_faults(motors, connected=True)
        self.assertEqual(len(faults), 1)
        self.assertEqual(faults[0]["arm"], "left")
        self.assertEqual(faults[0]["problem"], "unknown fault")

    def test_encoder_fault_reads_in_plain_words(self) -> None:
        motors = [
            {
                "arm": "left",
                "joint": "GRIPPER",
                "reachable": True,
                "status": "OUTPUT_SHAFT_CALIBRATION_ERROR",
            },
        ]
        faults = motor_faults(motors, connected=True)
        self.assertEqual(faults[0]["problem"], "output shaft calibration error")


if __name__ == "__main__":
    unittest.main()
