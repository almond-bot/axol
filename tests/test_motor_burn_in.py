"""motor.burn-in: ID assignment order, zero checks and the full-turn cycle."""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from almond_axol.cli.motor import burn_in
from almond_axol.cli.motor.burn_in import (
    DAMIAO,
    MYACTUATOR,
    TURN,
    CycleRunner,
    RigError,
    RigLog,
    RigMotor,
    assign_ids,
    check_zeros,
    detect,
    id_plan,
)
from almond_axol.motor.types import ControlMode, MotorStatus


class _Rig:
    """A bus of motors keyed by ID, plus the motors waiting to be connected."""

    def __init__(self, waiting: list, already: dict[int, list[str]] | None = None):
        # Each waiting motor is a type (arrives on ID 1) or an (id, type) pair.
        self.waiting = [w if isinstance(w, tuple) else (1, w) for w in waiting]
        self.on_bus: dict[int, list[str]] = {
            i: list(k) for i, k in (already or {}).items()
        }
        self.changes: list[tuple[int, int, str]] = []
        self.prompts: list[str] = []

    def prompt(self, text: str) -> bool:
        self.prompts.append(text)
        if text.startswith("Disconnect every motor"):
            self.on_bus.clear()
        if text.startswith(("Connect motor", "No new motor")) and self.waiting:
            motor_id, kind = self.waiting.pop(0)
            self.on_bus.setdefault(motor_id, []).append(kind)
        return True

    async def probe(self, _bus, motor_id: int) -> list[str]:
        return list(self.on_bus.get(motor_id, []))

    async def set_id(self, _bus, old: int, new: int, kind: str) -> None:
        self.changes.append((old, new, kind))
        self.on_bus[old].remove(kind)
        if not self.on_bus[old]:
            del self.on_bus[old]
        self.on_bus.setdefault(new, []).append(kind)


class AssignTest(unittest.IsolatedAsyncioTestCase):
    async def _assign(self, rig: _Rig, count: int = 7) -> dict[int, str]:
        return await assign_ids(
            None,  # type: ignore[arg-type]
            count=count,
            park_id=9,
            prompt=rig.prompt,
            probe_fn=rig.probe,
            set_id_fn=rig.set_id,
        )

    def test_plan_parks_the_first_motor(self) -> None:
        self.assertEqual(id_plan(7, 9), [9, 2, 3, 4, 5, 6, 7])
        self.assertEqual(id_plan(1, 9), [1])

    async def test_seven_motors_get_ids_one_to_seven(self) -> None:
        kinds = [MYACTUATOR] * 5 + [DAMIAO] * 2
        rig = _Rig(kinds)

        motors = await self._assign(rig)

        self.assertEqual(motors, dict(zip(range(1, 8), kinds)))
        self.assertEqual(
            [(o, n) for o, n, _ in rig.changes],
            [(1, 9), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (9, 1)],
        )
        self.assertEqual(sorted(rig.on_bus), list(range(1, 8)))

    async def test_asks_for_an_empty_bus_first(self) -> None:
        rig = _Rig([MYACTUATOR, DAMIAO], already={3: [MYACTUATOR]})

        motors = await self._assign(rig, count=2)

        self.assertIn("MyActuator on ID 3", rig.prompts[0])
        self.assertEqual(motors, {1: MYACTUATOR, 2: DAMIAO})

    async def test_used_motors_are_found_on_any_id(self) -> None:
        # A used MyActuator on ID 4, then a factory Damiao on ID 1 next to a
        # MyActuator already there: the two types never collide.
        rig = _Rig([(4, MYACTUATOR), (12, MYACTUATOR), (1, DAMIAO)])

        motors = await self._assign(rig, count=3)

        self.assertEqual(motors, {1: MYACTUATOR, 2: MYACTUATOR, 3: DAMIAO})
        self.assertEqual(
            [(o, n) for o, n, _ in rig.changes], [(4, 9), (12, 2), (1, 3), (9, 1)]
        )

    async def test_two_new_motors_at_once_ask_again(self) -> None:
        rig = _Rig([MYACTUATOR])
        rig.on_bus = {}
        prompts = iter([True, True, True])

        def two_then_one(text: str) -> bool:
            rig.prompts.append(text)
            if len(rig.prompts) == 1:
                rig.on_bus = {1: [MYACTUATOR], 5: [DAMIAO]}
            elif "More than one" in text:
                rig.on_bus = {1: [MYACTUATOR]}
            return next(prompts)

        rig.prompt = two_then_one  # type: ignore[method-assign]
        motors = await self._assign(rig, count=1)

        self.assertEqual(motors, {1: MYACTUATOR})
        self.assertTrue(any("More than one new motor" in p for p in rig.prompts))

    async def test_a_missing_motor_asks_again(self) -> None:
        rig = _Rig([MYACTUATOR, DAMIAO])
        connect = rig.prompt
        calls = {"n": 0}

        def flaky(text: str) -> bool:
            calls["n"] += 1
            if calls["n"] == 2:  # motor 2's first prompt: nothing plugged in
                rig.prompts.append(text)
                return True
            return connect(text)

        rig.prompt = flaky  # type: ignore[method-assign]

        motors = await self._assign(rig, count=2)

        self.assertEqual(motors, {1: MYACTUATOR, 2: DAMIAO})
        self.assertTrue(
            any("No new motor answers on ID 1 to 16" in p for p in rig.prompts)
        )

    async def test_detect_needs_every_id(self) -> None:
        rig = _Rig([], already={1: [MYACTUATOR]})
        with self.assertRaisesRegex(RigError, "No motor answers on ID 2"):
            await detect(None, [1, 2], rig.probe)  # type: ignore[arg-type]


class ZeroCheckTest(unittest.TestCase):
    def test_off_zero_motor_refuses(self) -> None:
        motors = {1: MYACTUATOR, 2: DAMIAO}
        check_zeros({1: 0.001, 2: -0.002}, motors, math.radians(2))
        with self.assertRaisesRegex(RigError, "ID 2 did not read zero"):
            check_zeros({1: 0.0, 2: math.radians(5)}, motors, math.radians(2))


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += seconds
        await asyncio.sleep(0)


class _Driver:
    """An MIT-mode motor that follows its setpoint at up to ``rate`` rad/s."""

    def __init__(self, clock: _Clock, rate: float = 10.0, temp: float = 40.0) -> None:
        self.clock = clock
        self.rate = rate
        self.temp = temp
        self.pos = 0.0
        self.target = 0.0
        self.t = 0.0
        self.enabled = False
        self.holds: list[float] = []  # each new setpoint held after a ramp
        self.status = MotorStatus.OK
        self.stuck_at: float | None = None
        self.silent = False
        self.mode: ControlMode | None = None
        self.gains: tuple[float, float] | None = None
        self._p_max = 12.5
        self._feedback = None

    def _advance(self) -> None:
        dt, self.t = self.clock.t - self.t, self.clock.t
        step = self.rate * dt
        goal = self.target
        if self.stuck_at is not None:
            goal = max(min(goal, self.stuck_at), -self.stuck_at)
        self.pos += max(-step, min(step, goal - self.pos))

    def set_feedback_callback(self, cb) -> None:  # noqa: ANN001
        self._feedback = cb

    async def enable(self) -> None:
        self.enabled = True

    async def disable(self) -> None:
        self.enabled = False

    async def set_control_mode(self, mode: ControlMode) -> None:
        self.mode = mode

    async def set_impedance(
        self, q: float, v: float, kp: float, kd: float, t_ff: float
    ) -> None:
        self._advance()
        if v == 0.0 and (not self.holds or self.holds[-1] != q):
            self.holds.append(q)
        self.target = q
        self.gains = (kp, kd)
        if not self.silent and self._feedback is not None:
            self._feedback(self.pos, 0.0, 0.0, self.clock.t)

    async def get_position(self) -> float:
        self._advance()
        return self.pos

    async def get_temperature(self) -> float:
        return self.temp

    async def get_voltage(self) -> float:
        return 24.0

    async def get_error_code(self) -> MotorStatus:
        return self.status


class CycleTest(unittest.IsolatedAsyncioTestCase):
    def _runner(
        self, drivers: dict[int, tuple[str, _Driver]], clock: _Clock, **kw
    ) -> CycleRunner:
        bus = kw.pop("bus", "")
        args = dict(
            motors=[RigMotor(i, kind, d, bus=bus) for i, (kind, d) in drivers.items()],
            speed=math.radians(90),
            tolerance=math.radians(2),
            dwell_s=0.5,
            max_temp=70.0,
            duration_s=3600.0,
            max_cycles=2,
            rate_hz=50.0,
            bus=bus,
            clock=clock,
            sleep=clock.sleep,
        )
        args.update(kw)
        return CycleRunner(**args)

    async def test_full_turn_each_way_then_off(self) -> None:
        clock = _Clock()
        ma, dm = _Driver(clock), _Driver(clock)
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "run.csv"
            runner = self._runner(
                {1: (MYACTUATOR, ma), 6: (DAMIAO, dm)}, clock, log_path=log
            )

            await runner.run()

            rows = list(csv.DictReader(log.read_text().splitlines()))
        cycle = [TURN, 0.0, -TURN, 0.0]
        self.assertEqual(ma.holds, [0.0] + cycle * 2)
        self.assertEqual(dm.holds, [0.0] + cycle * 2)
        self.assertEqual(dm.mode, ControlMode.IMPEDANCE)
        self.assertEqual(ma.gains, (40.0, 1.0))
        self.assertFalse(ma.enabled or dm.enabled)
        self.assertEqual([m.cycles for m in runner.motors], [2, 2])
        self.assertTrue(all(m.failed is None for m in runner.motors))
        self.assertEqual(len(rows), 2 * (1 + 2 * 4))
        self.assertEqual({r["result"] for r in rows}, {"ok"})
        self.assertIn("PASS", runner.summary())

    async def test_stops_at_the_time_limit(self) -> None:
        clock = _Clock()
        d = _Driver(clock)  # about 26 s per cycle at 90°/s
        runner = self._runner(
            {1: (MYACTUATOR, d)}, clock, duration_s=10.0, max_cycles=None
        )

        await runner.run()

        self.assertEqual(runner.motors[0].cycles, 1)

    async def test_a_stuck_motor_fails_and_the_rest_keep_going(self) -> None:
        clock = _Clock()
        good, stuck = _Driver(clock), _Driver(clock)
        stuck.stuck_at = math.radians(200)
        runner = self._runner({1: (MYACTUATOR, good), 2: (MYACTUATOR, stuck)}, clock)

        await runner.run()

        ok, bad = runner.motors
        self.assertIsNone(ok.failed)
        self.assertEqual(ok.cycles, 2)
        self.assertIn("did not reach +360.0°", bad.failed or "")
        self.assertFalse(stuck.enabled)
        self.assertIn("FAIL", runner.summary())

    async def test_overheating_and_faults_fail_the_motor(self) -> None:
        clock = _Clock()
        hot, faulty = _Driver(clock, temp=80.0), _Driver(clock)
        faulty.status = MotorStatus.OVER_CURRENT
        runner = self._runner({1: (MYACTUATOR, hot), 6: (DAMIAO, faulty)}, clock)

        await runner.run()

        self.assertIn("80 °C", runner.motors[0].failed or "")
        self.assertIn("over_current", runner.motors[1].failed or "")

    async def test_a_silent_motor_fails(self) -> None:
        clock = _Clock()
        quiet, ok = _Driver(clock), _Driver(clock)
        quiet.silent = True
        runner = self._runner({1: (MYACTUATOR, quiet), 2: (MYACTUATOR, ok)}, clock)

        await runner.run()

        self.assertEqual(runner.motors[0].failed, "stopped answering on the CAN bus")
        self.assertIsNone(runner.motors[1].failed)

    async def test_damiao_with_a_short_position_range_refuses(self) -> None:
        clock = _Clock()
        d = _Driver(clock)
        d._p_max = math.pi
        runner = self._runner({6: (DAMIAO, d)}, clock)

        await runner.run()

        self.assertIn("too small for a full turn", runner.motors[0].failed or "")

    async def test_two_buses_share_one_log(self) -> None:
        clock = _Clock()
        left, right = _Driver(clock), _Driver(clock)
        with tempfile.TemporaryDirectory() as tmp:
            log = RigLog(Path(tmp) / "run.csv")
            runners = [
                self._runner({1: (MYACTUATOR, left)}, clock, log=log, bus="can0"),
                self._runner({1: (DAMIAO, right)}, clock, log=log, bus="can1"),
            ]

            await asyncio.gather(*(r.run() for r in runners))
            log.close()

            rows = list(
                csv.DictReader((Path(tmp) / "run.csv").read_text().splitlines())
            )
        self.assertEqual({r["bus"] for r in rows}, {"can0", "can1"})
        self.assertEqual(len(rows), 2 * (1 + 2 * 4))
        self.assertIn("can1 ID 1 (Damiao)", runners[1].summary())


class CliTest(unittest.TestCase):
    def _parse(self, *argv: str) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        burn_in.add_parser(sub)
        return parser.parse_args(["motor.burn-in", *argv])

    def _parse_on_can0(self, *argv: str) -> argparse.Namespace:
        return self._parse("--buses", "can0", *argv)

    def test_defaults(self) -> None:
        args = self._parse_on_can0()
        self.assertEqual(args.steps, ["assign", "zero", "cycle"])
        self.assertEqual((args.count, args.park_id, args.hours), (7, 9, 8.0))

    def test_steps_run_in_order(self) -> None:
        self.assertEqual(
            self._parse_on_can0("--steps", "cycle,zero").steps, ["zero", "cycle"]
        )
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            self._parse_on_can0("--steps", "spin")

    def test_buses_are_named_interfaces(self) -> None:
        self.assertEqual(self._parse("--buses", "can0").buses, ["can0"])
        self.assertEqual(self._parse("--channel", "can0").buses, ["can0"])
        self.assertEqual(self._parse("--buses", "can0, can1").buses, ["can0", "can1"])

    def test_bad_bus_choices_refuse(self) -> None:
        for argv in ((), ("--buses", ","), ("--buses", "can0,can0")):
            with (
                self.subTest(argv),
                self.assertRaises(SystemExit),
                redirect_stderr(io.StringIO()),
            ):
                self._parse(*argv)

    def test_web_prompt_waits_for_a_line(self) -> None:
        with (
            mock.patch("sys.stdin", io.StringIO("\n")),
            redirect_stdout(io.StringIO()) as out,
        ):
            self.assertTrue(burn_in._web_prompt("Connect motor 1 of 7."))
        self.assertEqual(out.getvalue(), "[prompt] Connect motor 1 of 7.\n")
        with mock.patch("sys.stdin", io.StringIO("")), redirect_stdout(io.StringIO()):
            self.assertFalse(burn_in._web_prompt("Connect motor 2 of 7."))

    def test_park_id_must_clear_the_working_ids(self) -> None:
        with self.assertRaisesRegex(SystemExit, "park-id"):
            burn_in._validate(self._parse_on_can0("--park-id", "5"))


if __name__ == "__main__":
    unittest.main()
