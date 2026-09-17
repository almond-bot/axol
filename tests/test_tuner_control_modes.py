"""A MyActuator mode switch is a system reset, and the joint is limp for it.

`MyActuatorMotor.set_control_mode` sends 0x76 and sleeps `_MA_RESET_SETTLE_S`
(2 s) for the motor to come back; a resetting motor holds nothing. Reported
from the bench: tuning a wrist, "it just fell (not held at 0)"; tuning a
shoulder, the elbow "was not held at zero so it flopped". All three sweep
tuners switched the swept joint into IMPEDANCE *after* ramping the arm into a
deliberately gravity-loaded pose, so it hung limp for the 2 s reset plus the
1 s settle that followed.

These tests pin the ordering rather than the hardware: modes are assigned
once, before anything is posed, and no mode switch may appear between posing
the arm and sweeping it.
"""

from __future__ import annotations

import ast
import inspect
import unittest

from almond_axol.cli.tune import breakaway, friction, gravity


def _calls(fn, name: str) -> list[int]:
    """Line offsets (within ``fn``) of every call to ``name``."""
    tree = ast.parse(inspect.getsource(fn).lstrip())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        attr = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if attr == name:
            out.append(node.lineno)
    return sorted(out)


class ModeSwitchOrderingTest(unittest.TestCase):
    def test_reset_cost_is_documented_where_it_is_paid(self) -> None:
        src = inspect.getsource(friction.assign_modes)
        self.assertIn("system reset", src)
        # It must take hold of the impedance joint: that joint has just spent
        # the reset limp and nothing streams to it until someone does.
        self.assertIn("_ramp_to", src)

    def test_sweep_tuners_do_not_switch_modes_after_posing(self) -> None:
        for mod, poser in (
            (gravity, "_ramp_verified"),
            (friction, "_ramp_verified"),
            (breakaway, "ramp_others_to_zero"),
        ):
            with self.subTest(tuner=mod.__name__):
                run = mod._run
                posed = _calls(run, poser)
                switches = _calls(run, "set_control_mode")
                self.assertTrue(posed, f"{mod.__name__}: no posing call found")
                self.assertTrue(switches, f"{mod.__name__}: no mode switch found")
                # Every switch is either before the arm is posed, or in the
                # teardown after it has been brought home again.
                homed = max(_calls(run, "_home_all") or _calls(run, "_ramp_to") or [0])
                for line in switches:
                    self.assertTrue(
                        line < min(posed) or line > homed,
                        f"{mod.__name__}: set_control_mode at offset {line} lands "
                        f"between posing ({min(posed)}) and homing ({homed}) — "
                        f"that is 2 s of free fall in a loaded pose",
                    )

    def test_gravity_and_friction_assign_modes_before_homing(self) -> None:
        for mod in (gravity, friction):
            with self.subTest(tuner=mod.__name__):
                src = inspect.getsource(mod._run)
                self.assertIn("assign_modes", src)
                self.assertLess(
                    src.index("assign_modes"),
                    src.index("_home_all"),
                    "modes must be settled before the first motion",
                )

    def test_homing_keeps_its_order_for_the_impedance_joint(self) -> None:
        """Homing the swept joint separately would break the distal-to-proximal
        order, which is what makes the all-zero pose safe from any start."""
        src = inspect.getsource(friction._home_all)
        self.assertIn("impedance", src)
        self.assertIn("_HOME_ORDER", src)
        self.assertIn("_ramp_to", src)


class GravityPoseTest(unittest.TestCase):
    def test_fit_uses_the_measured_pose_not_the_commanded_one(self) -> None:
        """Holders sit on their own firmware position loop at whatever
        position_kp the motor shipped with (0.06 on these elbows, measured),
        and a loaded one sags. Fitting at the commanded pose makes the CoM
        absorb that sag silently."""
        src = inspect.getsource(gravity._run)
        self.assertIn("held[j] = await m.get_position()", src)
        self.assertIn("fit_com(q_bins, tau_meas, joint, is_left, held)", src)
        self.assertLess(src.index("held = {}"), src.index("fit_com("))
        # And a sagging holder is reported, not silently absorbed.
        self.assertIn("_HOLDER_DRIFT_WARN", src)
        self.assertIn("holders are not where they were put", src)


if __name__ == "__main__":
    unittest.main()
