"""A gripperless robot still on the stock gripper's wrist_3 mass is told so:
its own gravity comp isn't in the settings or the calibration."""

from __future__ import annotations

import contextlib
import io
import unittest
from dataclasses import replace
from unittest import mock

from almond_axol.cli.tune import factory
from almond_axol.robot.config import AxolConfig, stock_gravity_sides


def _cfg(has_gripper: bool = False, right_mass: float | None = None) -> AxolConfig:
    cfg = AxolConfig(has_gripper=has_gripper)
    if right_mass is not None:
        cfg.right.wrist_3 = replace(cfg.right.wrist_3, mass=right_mass)
    return cfg


class StockGravityTest(unittest.TestCase):
    def test_which_arms(self) -> None:
        self.assertEqual(stock_gravity_sides(_cfg(has_gripper=True)), [])
        self.assertEqual(stock_gravity_sides(_cfg()), ["left", "right"])
        self.assertEqual(stock_gravity_sides(_cfg(right_mass=0.4)), ["left"])

    def _factory(self, cfg: AxolConfig, sides: list[str], overrides=None):
        out = io.StringIO()
        with (
            mock.patch("almond_axol.settings.shared_axol_config", lambda: cfg),
            contextlib.redirect_stdout(out),
        ):
            stock = factory._warn_stock_gravity(sides, overrides or {})
        return stock, out.getvalue()

    def test_factory_banner(self) -> None:
        stock, text = self._factory(_cfg(right_mass=0.4), ["left", "right"])
        self.assertEqual(stock, ["left"])
        self.assertIn("left: no gripper, but wrist_3 still has the stock", text)
        # Only the arms being calibrated, and a --mass settles it.
        self.assertEqual(self._factory(_cfg(right_mass=0.4), ["right"])[0], [])
        stock, text = self._factory(
            _cfg(), ["left"], {"left": {"wrist_3": {"mass": 0.5}}}
        )
        self.assertEqual((stock, text), ([], ""))
        self.assertEqual(self._factory(_cfg(has_gripper=True), ["left"]), ([], ""))

    def test_tune_motion_says_it_too(self) -> None:
        from tests.test_tune_motion_config import _args

        from almond_axol.cli.tune import motion

        out = io.StringIO()
        with (
            mock.patch.object(motion, "shared_axol_config", _cfg),
            contextlib.redirect_stdout(out),
        ):
            motion._run_config(_args())
        self.assertIn("WARNING: left and right: no gripper", out.getvalue())


if __name__ == "__main__":
    unittest.main()
