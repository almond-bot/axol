"""
axol motor.burn-in

Burn-in test rig for MyActuator and Damiao motors on a test stand. Name the
CAN interface(s) the motors are on with --buses: one, such as can0 on a
Raspberry Pi with a CAN adapter, or several, which run at the same time.
Each bus holds its own set of --count motors. Three steps run in order, on
each bus:

  assign  Connect the motors one at a time. Each new motor is found on
          whatever ID it has (a factory motor is on 1). The first one moves
          to a parking ID (9) so the next new motor can use ID 1, the rest
          get IDs 2 to N, and at the end the parked motor moves back to 1.
  zero    Set each motor's zero to where it sits now. Damiao motors apply a
          new zero only after a power cycle, so the step asks for one.
  cycle   Turn every motor +360°, back to 0, -360°, back to 0, and repeat
          until --hours (default 8) or --cycles runs out. The motors run in
          MIT mode: the Pi streams a smooth setpoint with --kp and --kd. A motor that faults,
          overheats, stops answering or misses a target is switched off and
          marked failed. The others keep running.

Every move is logged to a CSV file (default ~/.almond/burn-in/<time>.csv).
The command exits with status 1 if any motor failed.

Run single steps with --steps, for example to restart the cycle step after a
stop without re-assigning IDs.

Examples:
    axol motor.burn-in --buses can0
    axol motor.burn-in --buses can_alm_axol_l,can_alm_axol_r
    axol motor.burn-in --buses can0 --steps zero,cycle
    axol motor.burn-in --buses can0 --steps cycle --cycles 1
    axol motor.burn-in --buses can0 --count 5 --hours 2 --speed 120
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
import sys
import time
from collections.abc import Awaitable, Callable, Iterable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ...motor.bus import CanBus
from ...motor.damiao import DamiaoMotor
from ...motor.errors import MotorError
from ...motor.motor import make_driver
from ...motor.myactuator import MyActuatorMotor
from ...motor.types import ControlMode, MotorStatus
from ...utils.paths import almond_path
from .. import web_prompt

STEPS = ("assign", "zero", "cycle")
MYACTUATOR = MyActuatorMotor.MOTOR_TYPE
DAMIAO = DamiaoMotor.MOTOR_TYPE
_LABEL = {MYACTUATOR: "MyActuator", DAMIAO: "Damiao"}

TURN = 2.0 * math.pi
# One cycle: a full turn each way, back to zero after each.
CYCLE_TARGETS = (TURN, 0.0, -TURN, 0.0)

# Damiao ESC_ID register: its read reply carries the motor's own ID.
_DM_REG_CAN_ID = 8
# Seconds to let a Damiao finish its flash store after an ID change.
_DM_STORE_SETTLE_S = 0.3
# Extra room on top of the nominal move time before a target counts as missed.
_MOVE_TIMEOUT_FACTOR = 1.5
_MOVE_TIMEOUT_SLACK_S = 3.0
# MIT positions wrap at ±PMAX: a full turn needs a little more than 2π.
_PMAX_MARGIN = 0.2
# IDs searched for a newly connected motor.
SCAN_MAX_ID = 16

# Marker a --web-prompts step prints before blocking on stdin. The control
# panel's Diagnostics page shows it with a Continue button (see set_zero_pos).
PROMPT_MARKER = "[prompt]"

ProbeFn = Callable[[CanBus, int], Awaitable[list[str]]]
SetIdFn = Callable[[CanBus, int, int, str], Awaitable[None]]
PromptFn = Callable[[str], bool]


class RigError(RuntimeError):
    """A step cannot go on. The message tells the operator what to do."""


def _label(kind: str) -> str:
    return _LABEL.get(kind, kind)


def _deg(rad: float) -> str:
    return f"{math.degrees(rad):+.1f}°"


def _hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _console_prompt(text: str) -> bool:
    """Block on Enter. False on Ctrl-C or end of input."""
    try:
        input(f"\n  {text}\n  Press Enter to continue, or Ctrl-C to stop: ")
        return True
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def _web_prompt(text: str) -> bool:
    """Print a prompt marker and block until the panel's Continue answers.

    The panel's Stop raises ``KeyboardInterrupt`` here, which :func:`run`
    reports as stopped.
    """
    return web_prompt.ask(f"{PROMPT_MARKER} {text}") != ""


def _bus_prompt(prompt: PromptFn, label: str) -> PromptFn:
    """``prompt`` with the bus named, when more than one bus can be meant."""
    if not label:
        return prompt
    return lambda text: prompt(f"{label}: {text}")


# ---------------------------------------------------------------------------
# Bus helpers
# ---------------------------------------------------------------------------


async def probe(bus: CanBus, motor_id: int) -> list[str]:
    """Motor types that answer on ``motor_id``. Moves nothing.

    MyActuator: a status-1 read (0x9A). Damiao: a read of its ESC_ID
    register, whose reply must carry ``motor_id`` itself, so a frame of ours
    looped back by another socket on the interface never counts.
    """
    found: list[str] = []
    ma = MyActuatorMotor(bus, motor_id, kt=1.0)
    for _ in range(3):
        try:
            await ma.get_voltage()
            found.append(MYACTUATOR)
            break
        except MotorError:
            continue
    dm = DamiaoMotor(bus, motor_id, feedback_id=0x10 + motor_id)
    try:
        value = await dm._read_register(_DM_REG_CAN_ID, timeout=0.1, attempts=3)
        if int(value) == motor_id:
            found.append(DAMIAO)
    except MotorError:
        pass
    return found


async def set_id(bus: CanBus, old_id: int, new_id: int, kind: str) -> None:
    """Move the ``kind`` motor on ``old_id`` to ``new_id`` and save it."""
    driver = make_driver(bus, old_id, kt=1.0, motor_type=kind)
    await driver.set_can_id(new_id)
    if kind == DAMIAO:
        await asyncio.sleep(_DM_STORE_SETTLE_S)


async def detect(
    bus: CanBus, ids: Iterable[int], probe_fn: ProbeFn = probe
) -> dict[int, str]:
    """``{id: type}`` for the motors on ``ids``. Every ID must answer once."""
    motors: dict[int, str] = {}
    for motor_id in ids:
        found = await probe_fn(bus, motor_id)
        if not found:
            raise RigError(
                f"No motor answers on ID {motor_id}. Check its power and CAN "
                f"cable, or run the assign step first."
            )
        if len(found) > 1:
            raise RigError(
                f"A MyActuator and a Damiao both answer on ID {motor_id}. "
                f"Disconnect one of them and run the assign step again."
            )
        motors[motor_id] = found[0]
    return motors


# ---------------------------------------------------------------------------
# Step 1: assign IDs
# ---------------------------------------------------------------------------


def id_plan(count: int, park_id: int) -> list[int]:
    """The ID each connected motor gets, in connection order.

    The first motor parks on ``park_id`` so later motors can arrive on the
    factory ID. It moves to ID 1 at the end. A single motor keeps ID 1.
    """
    if count == 1:
        return [1]
    return [park_id, *range(2, count + 1)]


async def scan(
    bus: CanBus, ids: Iterable[int], probe_fn: ProbeFn = probe
) -> list[tuple[int, str]]:
    """``(id, type)`` for every motor answering on ``ids``, probed together."""
    ids = list(ids)
    results = await asyncio.gather(*(probe_fn(bus, i) for i in ids))
    return [(i, kind) for i, found in zip(ids, results) for kind in found]


def _ids_text(found: list[tuple[int, str]]) -> str:
    return ", ".join(f"{_label(kind)} on ID {i}" for i, kind in found)


async def assign_ids(
    bus: CanBus,
    *,
    count: int,
    park_id: int,
    prompt: PromptFn,
    scan_max: int = SCAN_MAX_ID,
    probe_fn: ProbeFn = probe,
    set_id_fn: SetIdFn = set_id,
) -> dict[int, str]:
    """Give ``count`` motors IDs 1..count, connected one at a time.

    A new motor is found on whatever ID it has (a factory motor is on 1).
    Returns ``{id: type}``.
    """
    plan = id_plan(count, park_id)
    scan_ids = range(1, max(scan_max, park_id, count) + 1)

    text = "Disconnect every motor from the CAN bus."
    while True:
        found = await scan(bus, scan_ids, probe_fn)
        if not found:
            break
        if not prompt(f"{text} These still answer: {_ids_text(found)}."):
            raise RigError("Stopped by the operator.")

    assigned: dict[int, str] = {}
    for slot, new_id in enumerate(plan, start=1):
        text = f"Connect motor {slot} of {count}. Leave the motors already connected in place."
        while True:
            if not prompt(text):
                raise RigError("Stopped by the operator.")
            found = [
                (i, kind)
                for i, kind in await scan(bus, scan_ids, probe_fn)
                if assigned.get(i) != kind
            ]
            if len(found) == 1:
                break
            if not found:
                text = (
                    f"No new motor answers on ID 1 to {scan_ids[-1]}. Check its "
                    f"power and CAN cable. If it is the same type as a motor "
                    f"already connected and has the same ID, it can't be found: "
                    f"set its ID alone with axol motor.set-can-id first."
                )
            else:
                text = (
                    f"More than one new motor answers ({_ids_text(found)}). "
                    f"Connect only one new motor at a time."
                )
        # A MyActuator and a Damiao may share an ID: their frames never
        # collide, so the new one can still be moved.
        old_id, kind = found[0]
        if new_id != old_id:
            await set_id_fn(bus, old_id, new_id, kind)
            if kind not in await probe_fn(bus, new_id):
                raise RigError(
                    f"Motor {slot} ({_label(kind)}) did not answer on ID {new_id} "
                    f"after the change. Power-cycle it and run the assign step again."
                )
        assigned[new_id] = kind
        print(f"  Motor {slot}: {_label(kind)} on ID {new_id}")

    if park_id in assigned and park_id != 1:
        kind = assigned.pop(park_id)
        print(f"  Moving motor 1 from ID {park_id} to ID 1 ...")
        await set_id_fn(bus, park_id, 1, kind)
        if kind not in await probe_fn(bus, 1):
            raise RigError(
                f"Motor 1 ({_label(kind)}) did not answer on ID 1 after the "
                f"change. Power-cycle it, then run axol motor.set-can-id "
                f"--current-id {park_id} --new-id 1 --type {kind}."
            )
        assigned[1] = kind
        print(f"  Motor 1: {_label(kind)} on ID 1")
    return dict(sorted(assigned.items()))


# ---------------------------------------------------------------------------
# Step 2: zero
# ---------------------------------------------------------------------------


async def _driver_for(bus: CanBus, motor_id: int, kind: str):  # noqa: ANN202
    driver = make_driver(bus, motor_id, kt=1.0, motor_type=kind)
    if kind == DAMIAO:
        # Feedback scaling follows the motor's own position range.
        await driver._read_limits()  # type: ignore[attr-defined]
    return driver


async def zero_all(bus: CanBus, motors: dict[int, str]) -> None:
    """Save every motor's current position as its zero."""
    for motor_id, kind in motors.items():
        driver = make_driver(bus, motor_id, kt=1.0, motor_type=kind)
        if kind == DAMIAO:
            # A Damiao takes a new zero only while disabled.
            await driver.disable()
        await driver.set_zero_position()
        print(f"  ID {motor_id} ({_label(kind)}): zero saved")


async def zero_offsets(bus: CanBus, motors: dict[int, str]) -> dict[int, float]:
    """``{id: position}`` read back after zeroing."""
    out: dict[int, float] = {}
    for motor_id, kind in motors.items():
        driver = await _driver_for(bus, motor_id, kind)
        out[motor_id] = await driver.get_position()
    return out


def check_zeros(
    positions: dict[int, float], motors: dict[int, str], tolerance: float
) -> None:
    """Raise when a motor does not read zero after zeroing."""
    bad = {i: p for i, p in positions.items() if abs(p) > tolerance}
    for motor_id, pos in positions.items():
        mark = "ok" if motor_id not in bad else "NOT ZERO"
        print(f"  ID {motor_id} ({_label(motors[motor_id])}): {_deg(pos)}  {mark}")
    if bad:
        ids = ", ".join(str(i) for i in bad)
        raise RigError(
            f"ID {ids} did not read zero after zeroing. Power-cycle the motors "
            f"and run the zero step again with --steps zero,cycle."
        )


# ---------------------------------------------------------------------------
# Step 3: cycle
# ---------------------------------------------------------------------------


@dataclass
class RigMotor:
    """One motor under test and its running record."""

    motor_id: int
    kind: str
    driver: Any
    bus: str = ""  # CAN interface, shown when several buses run
    failed: str | None = None
    failed_at: float | None = None
    cycles: int = 0
    moves: int = 0
    position: float = 0.0  # rad, from the latest MIT feedback frame
    feedback_at: float = 0.0  # clock time of that frame
    setpoint: float = 0.0  # rad, last position commanded
    track: float = 0.0  # worst |position - setpoint| in the current move
    max_track: float = 0.0
    max_move_s: float = 0.0
    max_error: float = 0.0
    max_temp: float = -math.inf
    min_voltage: float = math.inf

    @property
    def name(self) -> str:
        bus = f"{self.bus} " if self.bus else ""
        return f"{bus}ID {self.motor_id} ({_label(self.kind)})"


LOG_FIELDS = (
    "time",
    "elapsed_s",
    "cycle",
    "target_deg",
    "bus",
    "motor_id",
    "type",
    "result",
    "move_s",
    "track_deg",
    "error_deg",
    "temperature_c",
    "voltage_v",
    "status",
)


class RigLog:
    """The cycle step's CSV log, shared by the runners of every bus."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=LOG_FIELDS)
        self._writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


@dataclass
class CycleRunner:
    """Drives the cycle step on MIT (impedance) commands.

    Every running motor gets a raised-cosine position and velocity setpoint
    streamed at ``rate_hz``, with fixed ``kp`` / ``kd`` and no feedforward
    torque. The motors' own position and speed loops are not used, so their
    stored firmware gains don't matter here. Positions come from the MIT
    feedback frames each command returns. Time and sleep are injectable for
    tests.
    """

    motors: list[RigMotor]
    speed: float  # rad/s, peak of the profile
    tolerance: float  # rad
    dwell_s: float
    max_temp: float
    duration_s: float
    kp: float = 40.0
    kd: float = 1.0
    rate_hz: float = 200.0
    max_cycles: int | None = None
    feedback_timeout_s: float = 0.5
    log_path: Path | None = None
    log: RigLog | None = None  # shared log; wins over log_path
    bus: str = ""
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    _start: float = 0.0
    _owns_log: bool = False
    _cycle: int = field(default=0)

    # -- lifecycle ---------------------------------------------------------

    def active(self) -> list[RigMotor]:
        return [m for m in self.motors if m.failed is None]

    async def _fail(self, motor: RigMotor, reason: str) -> None:
        motor.failed = reason
        motor.failed_at = self.clock() - self._start
        print(
            f"  ✗ {motor.name} failed at {_hms(motor.failed_at)}: {reason}. Switched it off."
        )
        try:
            await motor.driver.disable()
        except Exception:  # noqa: BLE001 - best effort; it is already marked failed
            pass

    async def _prepare(self, motor: RigMotor) -> None:
        d = motor.driver

        def on_feedback(pos: float, _vel: float, _torque: float, _ts: float) -> None:
            motor.position = pos
            motor.feedback_at = self.clock()

        d.set_feedback_callback(on_feedback)
        if motor.kind == DAMIAO:
            await d.disable()
            await d.set_control_mode(ControlMode.IMPEDANCE)
        await d.enable()
        p_max = getattr(d, "_p_max", None)
        if p_max is not None and p_max < TURN + _PMAX_MARGIN:
            fix = (
                "Set PMAX to 12.5 with axol motor.set-config"
                if motor.kind == DAMIAO
                else "Update its firmware"
            )
            raise MotorError(
                f"its MIT position range is ±{p_max:.2f} rad, too small for a "
                f"full turn. {fix}"
            )
        motor.position = await d.get_position()
        motor.feedback_at = self.clock()
        motor.setpoint = motor.position

    async def _send(self, motor: RigMotor, q: float, v: float) -> None:
        motor.setpoint = q
        try:
            await motor.driver.set_impedance(q, v, self.kp, self.kd, 0.0)
        except MotorError:
            await self._fail(motor, "did not accept a command")

    async def _tick(self, setpoints: dict[int, tuple[float, float]]) -> None:
        """Send one setpoint to each running motor and check its feedback."""
        now = self.clock()
        for m in self.active():
            if now - m.feedback_at > self.feedback_timeout_s:
                await self._fail(m, "stopped answering on the CAN bus")
                continue
            m.track = max(m.track, abs(m.position - m.setpoint))
            q, v = setpoints.get(m.motor_id, (m.setpoint, 0.0))
            await self._send(m, q, v)

    async def _stream(
        self,
        setpoints: Callable[[float], dict[int, tuple[float, float]]],
        seconds: float,
        done: Callable[[], bool] | None = None,
    ) -> None:
        """Stream ``setpoints(elapsed)`` at ``rate_hz`` for ``seconds``."""
        period = 1.0 / self.rate_hz
        t0 = self.clock()
        n = 0
        while self.active():
            elapsed = self.clock() - t0
            if elapsed >= seconds or (done is not None and done()):
                return
            await self._tick(setpoints(elapsed))
            n += 1
            await self.sleep(max(0.0, t0 + n * period - self.clock()))

    async def _health(self, motor: RigMotor) -> tuple[float, float, MotorStatus] | None:
        d = motor.driver
        try:
            temp = await d.get_temperature()
            volts = await d.get_voltage()
            status = await d.get_error_code()
        except MotorError:
            await self._fail(motor, "stopped answering on the CAN bus")
            return None
        motor.max_temp = max(motor.max_temp, temp)
        motor.min_voltage = min(motor.min_voltage, volts)
        if status != MotorStatus.OK:
            await self._fail(motor, f"reported {status.value}")
        elif temp > self.max_temp:
            await self._fail(
                motor, f"reached {temp:.0f} °C (limit {self.max_temp:.0f} °C)"
            )
        return temp, volts, status

    def _log(
        self,
        motor: RigMotor,
        target: float,
        result: str,
        move_s: float | None,
        health: tuple[float, float, MotorStatus] | None,
    ) -> None:
        if self.log is None:
            return
        temp, volts, status = health if health else (math.nan, math.nan, None)
        self.log.write(
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "elapsed_s": f"{self.clock() - self._start:.1f}",
                "cycle": self._cycle,
                "target_deg": f"{math.degrees(target):.0f}",
                "bus": motor.bus,
                "motor_id": motor.motor_id,
                "type": motor.kind,
                "result": result,
                "move_s": "" if move_s is None else f"{move_s:.2f}",
                "track_deg": f"{math.degrees(motor.track):.2f}",
                "error_deg": f"{math.degrees(motor.position - target):+.2f}",
                "temperature_c": f"{temp:.0f}",
                "voltage_v": f"{volts:.1f}",
                "status": status.value if status else "",
            }
        )

    # -- one move ----------------------------------------------------------

    async def move(self, target: float) -> None:
        """Ramp every running motor to ``target`` and wait for each to arrive."""
        movers = self.active()
        if not movers:
            return
        start = {m.motor_id: m.setpoint for m in movers}
        for m in movers:
            m.track = 0.0
        distance = max(abs(target - s) for s in start.values())
        # Raised cosine: the peak speed is pi/2 times the mean.
        duration = max(distance * math.pi / (2.0 * self.speed), 0.2)

        def ramp(elapsed: float) -> dict[int, tuple[float, float]]:
            frac = min(1.0, elapsed / duration)
            s = 0.5 - 0.5 * math.cos(math.pi * frac)
            ds = 0.5 * math.pi / duration * math.sin(math.pi * frac)
            return {
                i: (s0 + s * (target - s0), ds * (target - s0))
                for i, s0 in start.items()
            }

        t0 = self.clock()
        await self._stream(ramp, duration)

        hold = {m.motor_id: (target, 0.0) for m in movers}
        arrived: dict[int, float] = {}

        def all_arrived() -> bool:
            for m in movers:
                if (
                    m.failed is None
                    and m.motor_id not in arrived
                    and abs(m.position - target) <= self.tolerance
                ):
                    arrived[m.motor_id] = self.clock() - t0
            return all(m.failed is not None or m.motor_id in arrived for m in movers)

        timeout = duration * _MOVE_TIMEOUT_FACTOR + _MOVE_TIMEOUT_SLACK_S
        await self._stream(lambda _e: hold, timeout - duration, all_arrived)
        all_arrived()

        for m in movers:
            if m.failed is not None:
                self._log(m, target, m.failed, None, None)
                continue
            if m.motor_id not in arrived:
                reason = (
                    f"did not reach {_deg(target)} within {timeout:.0f} s "
                    f"(stopped at {_deg(m.position)})"
                )
                await self._fail(m, reason)
                self._log(m, target, reason, None, None)
                continue
            move_s = arrived[m.motor_id]
            m.moves += 1
            m.max_move_s = max(m.max_move_s, move_s)
            m.max_track = max(m.max_track, m.track)
            m.max_error = max(m.max_error, abs(m.position - target))
            health = await self._health(m)
            self._log(m, target, m.failed or "ok", move_s, health)
            # Keep the motors fed between the health reads.
            await self._tick(hold)
        await self._stream(lambda _e: hold, self.dwell_s)

    # -- whole run ---------------------------------------------------------

    def _out_of_time(self) -> bool:
        if self.max_cycles is not None and self._cycle >= self.max_cycles:
            return True
        return self.clock() - self._start >= self.duration_s

    def _progress(self) -> None:
        elapsed = self.clock() - self._start
        left = max(0.0, self.duration_s - elapsed)
        parts = []
        for m in self.motors:
            if m.failed:
                parts.append(f"{m.motor_id}:FAIL")
            else:
                t = "" if m.max_temp == -math.inf else f" {m.max_temp:.0f}°C"
                parts.append(f"{m.motor_id}:ok{t} {math.degrees(m.track):.1f}°")
        bus = f"{self.bus} bus " if self.bus else ""
        print(
            f"  {bus}cycle {self._cycle}  {_hms(elapsed)} run, {_hms(left)} left  |  "
            + "  ".join(parts),
            flush=True,
        )

    async def run(self) -> None:
        self._start = self.clock()
        if self.log is None and self.log_path is not None:
            self.log = RigLog(self.log_path)
            self._owns_log = True
        try:
            # Damiao last: once enabled it is limp until the first command.
            for m in sorted(self.motors, key=lambda m: m.kind == DAMIAO):
                try:
                    await self._prepare(m)
                    print(f"  {m.name}: enabled at {_deg(m.position)}")
                except MotorError as exc:
                    await self._fail(m, f"could not be enabled: {exc}")
            # Enabling takes seconds. Time feedback from the first command.
            for m in self.active():
                m.feedback_at = self.clock()
            print(
                f"  MIT kp {self.kp:g}, kd {self.kd:g} at {self.rate_hz:g} Hz. "
                f"Moving every motor to zero ..."
            )
            await self.move(0.0)
            while self.active() and not self._out_of_time():
                for target in CYCLE_TARGETS:
                    await self.move(target)
                self._cycle += 1
                for m in self.active():
                    m.cycles = self._cycle
                self._progress()
        finally:
            for m in self.active():
                try:
                    await m.driver.disable()
                except Exception:  # noqa: BLE001 - keep switching the rest off
                    print(f"  ! could not switch off {m.name}. Cut its power.")
            if self._owns_log and self.log is not None:
                self.log.close()

    def summary(self) -> str:
        lines = [
            "",
            f"  {'Motor':<26} {'Result':<6} {'Cycles':>6} {'Max move':>9} "
            f"{'Max track':>10} {'Max error':>10} {'Max temp':>9} {'Min volts':>10}",
        ]
        for m in self.motors:
            temp = "-" if m.max_temp == -math.inf else f"{m.max_temp:.0f} °C"
            volts = "-" if m.min_voltage == math.inf else f"{m.min_voltage:.1f} V"
            lines.append(
                f"  {m.name:<26} {'FAIL' if m.failed else 'PASS':<6} {m.cycles:>6} "
                f"{m.max_move_s:>8.1f}s {math.degrees(m.max_track):>9.2f}° "
                f"{math.degrees(m.max_error):>9.2f}° {temp:>9} {volts:>10}"
            )
        for m in self.motors:
            if m.failed:
                lines.append(
                    f"  {m.name} failed at {_hms(m.failed_at or 0)}: {m.failed}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_buses(spec: str) -> list[str]:
    buses = [b.strip() for b in spec.split(",") if b.strip()]
    if not buses:
        raise argparse.ArgumentTypeError("name at least one CAN interface")
    if len(set(buses)) != len(buses):
        raise argparse.ArgumentTypeError(f"{spec} names an interface twice")
    return buses


def _parse_steps(spec: str) -> list[str]:
    steps = [s.strip().lower() for s in spec.split(",") if s.strip()]
    unknown = [s for s in steps if s not in STEPS]
    if unknown or not steps:
        raise argparse.ArgumentTypeError(
            f"unknown step {', '.join(unknown) or '(none)'}. Use one or more of: {', '.join(STEPS)}"
        )
    return [s for s in STEPS if s in steps]


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``motor.burn-in`` subcommand."""
    p = subparsers.add_parser(
        "motor.burn-in",
        help="Assign IDs, zero and run a long full-turn test on a rig of motors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument(
        "--buses",
        "--channel",
        dest="buses",
        type=_parse_buses,
        required=True,
        metavar="IFACES",
        help="CAN interface the motors are on, such as can0. Separate several "
        "with commas to test them at the same time.",
    )
    p.add_argument(
        "--steps",
        type=_parse_steps,
        default=",".join(STEPS),
        metavar="STEPS",
        help="Comma-separated steps to run, in order: assign, zero, cycle "
        "(default: %(default)s).",
    )
    p.add_argument(
        "--count",
        type=int,
        default=7,
        help="Number of motors on each bus (default: %(default)s).",
    )
    p.add_argument(
        "--park-id",
        type=int,
        default=9,
        metavar="ID",
        help="ID the first motor uses while the others are connected (default: %(default)s).",
    )
    p.add_argument(
        "--hours",
        type=float,
        default=8.0,
        help="Length of the cycle step in hours (default: %(default)s).",
    )
    p.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Stop the cycle step after this many cycles, even if time is left.",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=90.0,
        help="Turning speed in degrees per second (default: %(default)s).",
    )
    p.add_argument(
        "--dwell",
        type=float,
        default=0.5,
        help="Seconds to wait at each target (default: %(default)s).",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=2.0,
        help="Degrees from the target that count as arrived (default: %(default)s).",
    )
    p.add_argument(
        "--kp",
        type=float,
        default=40.0,
        help="Stiffness in Nm/rad for the cycle step (default: %(default)s).",
    )
    p.add_argument(
        "--kd",
        type=float,
        default=1.0,
        help="Damping in Nm·s/rad for the cycle step, 0 to 5 (default: %(default)s).",
    )
    p.add_argument(
        "--rate",
        type=float,
        default=200.0,
        help="Commands per second to each motor in the cycle step (default: %(default)s).",
    )
    p.add_argument(
        "--max-temp",
        type=float,
        default=70.0,
        help="Temperature in °C that fails a motor (default: %(default)s).",
    )
    p.add_argument(
        "--log",
        type=Path,
        default=None,
        metavar="PATH",
        help="CSV log file (default: ~/.almond/burn-in/<date>-<time>.csv).",
    )
    p.add_argument(
        "--web-prompts",
        action="store_true",
        help="Emit '[prompt] ...' markers and block on stdin for the hands-on "
        "steps, so the web dashboard can drive them with a Continue button "
        "(set automatically by the dashboard).",
    )
    p.set_defaults(func=run)


def _validate(args: argparse.Namespace) -> None:
    if not 1 <= args.count <= 30:
        raise SystemExit("error: --count must be between 1 and 30.")
    if args.count > 1 and args.park_id <= args.count:
        raise SystemExit(f"error: --park-id must be above --count ({args.count}).")
    if args.park_id == 1:
        raise SystemExit("error: --park-id must not be 1.")
    if not (0 < args.kp <= 500 and 0 <= args.kd <= 5 and 20 <= args.rate <= 1000):
        raise SystemExit("error: use --kp 0-500, --kd 0-5 and --rate 20-1000.")
    if args.speed <= 0 or args.hours <= 0 or args.tolerance <= 0:
        raise SystemExit("error: --speed, --hours and --tolerance must be above zero.")


def run(args: argparse.Namespace) -> None:
    """Run the selected burn-in steps."""
    _validate(args)
    try:
        ok = asyncio.run(_run(args, args.buses))
    except RigError as exc:
        raise SystemExit(f"\n  {exc}") from None
    except KeyboardInterrupt:
        print("\n  Stopped. Every motor was switched off.")
        sys.exit(130)
    if not ok:
        sys.exit(1)


async def _run(args: argparse.Namespace, channels: list[str]) -> bool:
    ids = list(range(1, args.count + 1))
    prompt: PromptFn = _web_prompt if args.web_prompts else _console_prompt
    # {interface: {id: type}}, filled by the first step that finds the motors.
    found: dict[str, dict[int, str]] = {}
    # Name the bus in prompts and results only when there is more than one.
    buses = [(c if len(channels) > 1 else "", c) for c in channels]
    each = " each" if len(buses) > 1 else ""
    print(
        f"\nmotor.burn-in on {', '.join(channels)}: "
        f"{args.count} motors{each}, steps {', '.join(args.steps)}"
    )

    if "assign" in args.steps:
        for label, channel in buses:
            print(f"\n[assign] {channel}: connect the motors one at a time.")
            async with CanBus(channel) as bus:
                found[channel] = await assign_ids(
                    bus,
                    count=args.count,
                    park_id=args.park_id,
                    prompt=_bus_prompt(prompt, label),
                )

    if "zero" in args.steps:
        for label, channel in buses:
            print(f"\n[zero] {channel}: saving each motor's position as its zero.")
            async with CanBus(channel) as bus:
                if channel not in found:
                    found[channel] = await detect(bus, ids)
                await zero_all(bus, found[channel])
        if any(DAMIAO in m.values() for m in found.values()):
            if not prompt("Power-cycle the motors so the Damiao zeros take effect."):
                raise RigError("Stopped by the operator.")
        for label, channel in buses:
            print(f"\n[zero] {channel}: checking the zeros.")
            async with CanBus(channel) as bus:
                motors = found[channel]
                check_zeros(
                    await zero_offsets(bus, motors),
                    motors,
                    math.radians(args.tolerance),
                )

    if "cycle" not in args.steps:
        return True

    log_path = args.log or almond_path(
        "burn-in", datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv"
    )
    hours = f"{args.hours:g} h" + (f" or {args.cycles} cycles" if args.cycles else "")
    print(f"\n[cycle] ±360° at {args.speed:g}°/s for {hours}. Log: {log_path}")
    log = RigLog(log_path)
    runners: list[CycleRunner] = []
    try:
        async with AsyncExitStack() as stack:
            for label, channel in buses:
                bus = await stack.enter_async_context(CanBus(channel))
                if channel not in found:
                    found[channel] = await detect(bus, ids)
                rig = [
                    RigMotor(i, kind, await _driver_for(bus, i, kind), bus=label)
                    for i, kind in found[channel].items()
                ]
                runners.append(
                    CycleRunner(
                        motors=rig,
                        speed=math.radians(args.speed),
                        tolerance=math.radians(args.tolerance),
                        dwell_s=args.dwell,
                        max_temp=args.max_temp,
                        duration_s=args.hours * 3600.0,
                        kp=args.kp,
                        kd=args.kd,
                        rate_hz=args.rate,
                        max_cycles=args.cycles,
                        log=log,
                        bus=label,
                    )
                )
            # Each bus streams on its own. Wait for all of them, so no bus
            # closes while another is still driving its motors.
            results = await asyncio.gather(
                *(r.run() for r in runners), return_exceptions=True
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
    finally:
        log.close()
        for r in runners:
            print(r.summary())
        print(f"\n  Log: {log_path}")
    return all(m.failed is None for r in runners for m in r.motors)
