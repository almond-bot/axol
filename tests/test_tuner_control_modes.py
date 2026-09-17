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
                if mod is breakaway:
                    # Still on its own teardown: a switch is either before the
                    # arm is posed or after it has been brought home again.
                    homed = max(
                        _calls(run, "_home_all") or _calls(run, "_ramp_to") or [0]
                    )
                    for line in switches:
                        self.assertTrue(
                            line < min(posed) or line > homed,
                            f"breakaway: set_control_mode at offset {line} lands "
                            f"between posing ({min(posed)}) and homing ({homed})",
                        )
                    continue
                # gravity/friction: the only switches left in _run are
                # assign_modes' (before posing); the teardown's live in
                # safe_return_to_rest, after the arm is verified at rest.
                for line in switches:
                    self.assertLess(
                        line,
                        min(posed),
                        f"{mod.__name__}: set_control_mode at offset {line} lands "
                        f"after posing ({min(posed)}) — 2 s of free fall in a "
                        f"loaded pose",
                    )
                self.assertIn("safe_return_to_rest(", inspect.getsource(run))
                td = inspect.getsource(friction.safe_return_to_rest)
                self.assertLess(
                    td.index("holders.at_rest()"), td.index("set_control_mode")
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


class HoldersUnderImpedanceTest(unittest.TestCase):
    """The non-swept joints must be held by streamed impedance, not parked.

    Bench, twice: a shoulder sweep swung the forearm toward horizontal and the
    elbow -- "held at 0" by a single 0xA4 command on the motor's stored
    position_kp of 0.06 -- flopped under ~5 Nm. Impedance carries its gains in
    every frame and is fed the gravity model; the holders class from
    tune.position-loop held these joints at -90 deg the night before.
    """

    def test_assign_modes_puts_every_joint_under_impedance_and_returns_holders(
        self,
    ) -> None:
        src = inspect.getsource(friction.assign_modes)
        self.assertIn("ImpedanceHolders(motors, impedance, is_left, config)", src)
        self.assertIn("await holders.start()", src)
        self.assertIn("return holders", src)

    def test_homing_and_clearance_ramps_go_through_the_holders(self) -> None:
        home = inspect.getsource(friction._home_all)
        self.assertIn("_ramp_verified(motors, {j: 0.0}, holders)", home)
        ramp = inspect.getsource(friction._ramp_verified)
        self.assertIn("holders.ramp_to(j, targets[j], _RAMP_SPEED)", ramp)

    def test_sweep_tuners_thread_holders_through_and_stop_them(self) -> None:
        for mod in (gravity, friction):
            with self.subTest(tuner=mod.__name__):
                src = inspect.getsource(mod._run)
                self.assertIn("holders = await assign_modes(", src)
                self.assertIn("is_left=is_left, config=resolved", src)
                self.assertIn("_ramp_verified(motors, stage, holders)", src)
                # The stream is stopped inside safe_return_to_rest, only after
                # homing and the at-rest check (tests/test_safe_teardown).
                self.assertIn(
                    "safe_return_to_rest(motors, holders, joint, kp, kd)", src
                )
                td = inspect.getsource(friction.safe_return_to_rest)
                self.assertLess(
                    td.index("holders.ramp_to("), td.index("holders.stop()")
                )

    def test_shared_module_is_the_one_position_loop_uses(self) -> None:
        from almond_axol.cli.tune import position_loop
        from almond_axol.tuning.holders import ImpedanceHolders

        self.assertIs(position_loop._Holders, ImpedanceHolders)


class HolderDriftReportTest(unittest.TestCase):
    def test_tied_drifts_do_not_compare_joint_enums(self) -> None:
        """Bench: two holders off by the same amount raised
        TypeError: '<' not supported between instances of 'Joint' and 'Joint',
        after the arm was already posed."""
        import re

        src = inspect.getsource(gravity._run)
        block = src[src.index("drift = sorted(") : src.index("reverse=True")]
        self.assertIn("key=lambda d: d[0]", block)
        # And the expression itself survives a tie.
        from almond_axol.constants import Joint

        held = {Joint.SHOULDER_1: 0.02, Joint.ELBOW: 0.02}
        targets: dict = {}
        drift = sorted(
            ((abs(held[j] - targets.get(j, 0.0)), j) for j in held),
            key=lambda d: d[0],
            reverse=True,
        )
        self.assertEqual(len(drift), 2)
        self.assertTrue(re.search(r"key=lambda d: d\[0\]", block))
