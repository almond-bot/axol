"""The Damiao wrists' temperature limit: its config and the core's line.

The realtime core runs the limit (``rust/axol-rt/src/thermal.rs``, tested by
``cargo test``); the package configures it with a ``thermal`` line.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from almond_axol.robot.config import AxolConfig, WristThermalConfig
from tests.test_rom_partial_arm import WRIST_KIT, _partial_axol, _wrap_rt


class WristThermalConfigTest(unittest.TestCase):
    def test_on_by_default_below_the_motor_trip(self) -> None:
        cfg = AxolConfig().wrist_thermal
        self.assertTrue(cfg.enabled)
        # The wrists' OT_VALUE is 100 °C and the driver cuts out at 120 °C.
        self.assertLess(cfg.coil_limit_c, 100.0)
        self.assertLess(cfg.driver_limit_c, 120.0)
        # Cooling runs under the DM-J4310's rated 3.5 Nm.
        self.assertLess(cfg.cool_torque_nm, 3.5)
        self.assertEqual(cfg.config_line(), "thermal 85.0 75.0 105.0 95.0 1.5 2.0")

    def test_off(self) -> None:
        self.assertEqual(WristThermalConfig(enabled=False).config_line(), "thermal off")

    def test_resume_must_be_below_the_limit(self) -> None:
        with self.assertRaises(ValueError):
            WristThermalConfig(coil_limit_c=80.0, coil_resume_c=80.0)
        with self.assertRaises(ValueError):
            WristThermalConfig(driver_limit_c=100.0, driver_resume_c=101.0)
        with self.assertRaises(ValueError):
            WristThermalConfig(cool_torque_nm=0.0)
        with self.assertRaises(ValueError):
            WristThermalConfig(ramp_s=float("nan"))

    def test_survives_resolving(self) -> None:
        cfg = AxolConfig(wrist_thermal=WristThermalConfig(coil_limit_c=80.0))
        self.assertEqual(cfg.resolved().wrist_thermal.coil_limit_c, 80.0)


class CoreConfigLineTest(unittest.TestCase):
    def _lines(self, axol) -> list[str]:
        return _wrap_rt(axol)._config_text().splitlines()

    def test_the_core_config_carries_the_line(self) -> None:
        lines = self._lines(_partial_axol(set(WRIST_KIT)))
        self.assertIn("thermal 85.0 75.0 105.0 95.0 1.5 2.0", lines)
        # Ahead of the joint lines, like the other directives.
        first_joint = next(
            i for i, line in enumerate(lines) if line.startswith("joint ")
        )
        self.assertLess(
            lines.index("thermal 85.0 75.0 105.0 95.0 1.5 2.0"), first_joint
        )

    def test_follows_the_robot_config(self) -> None:
        axol = _partial_axol(set(WRIST_KIT))
        assert axol.left is not None
        axol.left._config = replace(
            axol.left._config, wrist_thermal=WristThermalConfig(enabled=False)
        )
        self.assertIn("thermal off", self._lines(axol))


if __name__ == "__main__":
    unittest.main()
