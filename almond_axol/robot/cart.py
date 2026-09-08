"""Powered Axol Cart: x-drive omni wheel base + telescoping lift.

The powered cart has four omni wheels mounted at 45° on the corners (an
x-drive), each driven by a Damiao motor in VELOCITY mode on a dedicated
CAN bus, plus a telescoping lift driven by the jelly_legs board on its own
chest CAN bus (see :mod:`almond_axol.robot.lift`). Wheel CAN IDs are
fixed by convention:

    id 1  front-left      id 2  front-right
    id 3  back-left       id 4  back-right

:class:`Cart` exposes a latched command interface: any thread calls
:meth:`Cart.set_command` with a normalized body velocity + lift direction,
and an internal asyncio task (started by :meth:`Cart.enable`) applies ramp
limiting, x-drive mixing, and the park/unpark state machine at
``CartConfig.frequency``:

- While the command is non-zero the wheels track it in VELOCITY mode.
- When the ramp-limited command reaches zero (and the wheels are measured
  slow), the wheels are parked: switched to MIT/impedance mode and held at
  their current positions by the motor's internal high-bandwidth position
  loop, so the base does not roll under load.
- If no fresh command arrives within ``command_timeout`` the target is
  forced to zero (streaming sources that die mid-motion cannot leave the
  base driving).

Damiao position commands/feedback are mapped into ±PMAX (12.5 rad from
factory — about two wheel turns), which drive wheels escape almost
immediately; anchoring at the reported position then means a phantom error
of several radians and instant overcurrent. Re-zeroing at park time doesn't
help either: on this firmware the 0xFE zero command only applies after a
power cycle. So at startup the PMAX register is raised (RAM only, reverts
on power-off) to keep multi-turn positions valid for a whole session, and
parking refuses (with a warning) if a wheel ever approaches the widened
limit.

Body-frame convention: +x forward, +y left, +wz counter-clockwise. The
mixing assumes each wheel's positive spin has a forward (+x) component;
if a wheel runs backwards on your cart, flip its entry in
:data:`WHEEL_SIGNS`.

Straight-line drift. Three mechanisms make an x-drive veer on a real floor,
and they need different fixes (an uneven-floor simulation of this exact
plant — wobbly-table load redistribution, tanh traction, per-wheel stiction
and radius spread, the real 50 Hz loop — ranked them):

- *Rotation* — corrected by the gyro heading hold (``CartConfig.imu``,
  ``yaw_hold_gain``). The single largest factor: without it the heading
  wanders several degrees per stroke.
- *Effective-radius mismatch* — four omni wheels never wear identically, and
  a 3% spread slides the base sideways ~2 cm per 3 m even with the heading
  held, because the wheels' speeds are mutually inconsistent and the free
  rollers absorb the difference. Unobservable from the wheels themselves
  (each tracks its command perfectly); corrected by ``CartConfig.wheel_scale``,
  measured by ``axol diag.base-calibrate`` (the cart drives six short strokes
  tracked by the overhead ZED) or fitted from tape-measured strokes with
  :func:`solve_wheel_scale`.
- *Diagonal unloading* — one diagonal pair goes light (a wobbly floor, or
  the weight shift of a launch/stop on a tall base whose mass sits off the
  wheelbase centre — measured on the cart: 6–28° of veer at the default
  ramp, under 2° at a third of it), and the loaded pair can only push along
  its shared 45° axis, so the base slides sideways while it accelerates and
  straightens once cruising. No wheel command can fix it while the pair is
  unloaded; the lever is gentler acceleration, applied automatically while a
  wheel reads light by the torque-based :class:`TractionGuard`
  (``CartConfig.traction``), or standing (``accel``/``jerk``).
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..constants import CAN_BASE, CAN_CHEST
from ..motor import CanBus, ControlMode, make_driver
from ..motor.damiao import _DM_REG_PMAX
from ..motor.driver import MotorDriver
from .base import HardwareCleanupError, mark_hardware_cleanup_uncertain
from .lift import DOWN, JOG_SPEED, STOP, UP, Lift, LiftStatus

_logger = logging.getLogger(__name__)

# The cart's wheels ride their own CAN interface, separate from the arm buses.
# ``axol can.setup`` names the cart's adapter to this and includes it in the
# @reboot bring-up alongside the arm channels.
DEFAULT_CHANNEL = CAN_BASE

# Per-wheel spin-direction calibration: flip an entry to -1 if that wheel
# drives the wrong way with everything else correct.
WHEEL_SIGNS: dict[int, float] = {1: 1.0, 2: -1.0, 3: 1.0, 4: -1.0}

# Position-mapping range (PMAX, register 21) written at startup, in rad.
# Wide enough that a session's accumulated wheel rotation stays in range
# (the factory 12.5 rad is ~2 wheel turns), narrow enough that the 16-bit
# MIT position encoding keeps sub-centidegree resolution (~12 mrad here).
_SESSION_PMAX = 400.0

# Measured wheel speed (rad/s) below which parking is allowed. Guards
# against anchoring a wheel that is still coasting (e.g. on a slope where
# the velocity loop hasn't fully braked when the command reaches zero).
_PARK_MAX_WHEEL_SPEED = 0.5

# Rate of the per-cycle heading-hold trace line (CartConfig.yaw_log). The
# hold's dynamics are ~1 s, so this resolves them without flooding a console
# the 50 Hz command loop has to keep up with.
_YAW_TRACE_HZ = 10.0

# Seconds of driving with the IMU requested but no yaw sample ever fed
# before the cart says the heading hold is dead.
_YAW_SILENT_WARN_S = 3.0


async def _gather_all_or_raise(*awaitables: Awaitable[Any]) -> None:
    """Finish every parallel hardware command before surfacing a failure."""
    results = await asyncio.gather(*awaitables, return_exceptions=True)
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        first = failures[0]
        for extra in failures[1:]:
            first.add_note(
                "additional parallel cart command failure: "
                f"{type(extra).__name__}: {extra}"
            )
        raise first


@dataclass(frozen=True)
class _Wheel:
    """One wheel's CAN ID and its x-drive mixing coefficients.

    Wheel speed = ``mx·vx + my·vy + mw·wz`` (body frame: +x forward, +y
    left, +wz CCW), with each wheel's positive drive direction chosen to
    have a forward component. The common √2/2 translation factor is folded
    into the normalization in :func:`mix`.
    """

    name: str
    motor_id: int
    mx: float
    my: float
    mw: float


WHEELS: tuple[_Wheel, ...] = (
    _Wheel("front_left", 1, +1.0, -1.0, -1.0),
    _Wheel("front_right", 2, +1.0, +1.0, +1.0),
    _Wheel("back_left", 3, +1.0, +1.0, -1.0),
    _Wheel("back_right", 4, +1.0, -1.0, +1.0),
)


def deadzone(value: float, threshold: float) -> float:
    """Zero the stick inside the deadzone and rescale the rest to [-1, 1]."""
    if abs(value) < threshold:
        return 0.0
    scaled = (abs(value) - threshold) / (1.0 - threshold)
    return scaled if value > 0 else -scaled


def mix(
    vx: float,
    vy: float,
    wz: float,
    max_speed: float,
    turn_scale: float,
    wheel_scale: tuple[float, float, float, float] | None = None,
) -> list[float]:
    """Map normalized body command ([-1, 1] each) to per-wheel rad/s.

    The raw mix can exceed 1 when translation and rotation combine, so the
    whole set is scaled down together to preserve the motion direction while
    keeping every wheel within ``max_speed``. ``wheel_scale`` (in
    :data:`WHEELS` order) then multiplies each wheel to compensate its
    effective-radius error — see ``CartConfig.wheel_scale``.
    """
    wz *= turn_scale
    raw = [
        WHEEL_SIGNS[w.motor_id] * (w.mx * vx + w.my * vy + w.mw * wz) for w in WHEELS
    ]
    scale = max(1.0, max(abs(r) for r in raw))
    speeds = [r / scale * max_speed for r in raw]
    if wheel_scale is not None:
        speeds = [s * k for s, k in zip(speeds, wheel_scale)]
    return speeds


class VectorRamp:
    """Rate- and jerk-limited ramp of the normalized (vx, vy, wz) command.

    The command is slewed as a single vector — the step's magnitude is capped
    but its direction kept — so a mostly-forward command with a small lateral
    part doesn't finish its lateral ramp first and veer before straightening.

    Two rate limits apply: ``accel`` while the step moves the command away
    from zero (its projection onto the current command is non-negative) and
    ``decel`` while it moves toward zero, so stops and speed reductions can
    be brisker than launches. A reversal uses ``decel`` down to zero, then
    ``accel`` out the other side.

    With ``jerk`` > 0 the command's rate of change is itself a state — a
    velocity vector in command space — that may only change by ``jerk`` per
    second. It is steered toward the limit speed along the remaining delta,
    capped at √(2·jerk·remaining) so it reaches zero exactly as the command
    reaches its target: an S-shaped profile with no acceleration step at
    either end of a launch or stop, including a stick released mid-launch
    (the rate swings smoothly through zero instead of flipping sign).
    ``jerk`` = 0 is the plain trapezoid.

    All quantities are in normalized command units (full stick = 1) per
    second / second².
    """

    def __init__(self, accel: float, decel: float, jerk: float, dt: float) -> None:
        if not (accel > 0.0 and decel > 0.0 and jerk >= 0.0 and dt > 0.0):
            raise ValueError("ramp needs accel > 0, decel > 0, jerk >= 0, dt > 0")
        self.accel = accel
        self.decel = decel
        self.jerk = jerk
        self.dt = dt
        self.vel = [0.0, 0.0, 0.0]  # command rate of change (normalized/s)
        self.limit = 0.0  # rate limit in force on the last step (after scaling)

    @property
    def rate(self) -> float:
        """Magnitude of the command's current rate of change (normalized/s)."""
        return math.sqrt(sum(v * v for v in self.vel))

    def step(
        self,
        cmd: list[float],
        target: tuple[float, ...],
        accel_scale: float = 1.0,
        decel_scale: float = 1.0,
    ) -> None:
        """Advance ``cmd`` one interval toward ``target`` (in place).

        ``accel_scale`` / ``decel_scale`` (in (0, 1]) ease the respective rate
        limit for this step — the traction guard's lever. The jerk limit is
        untouched, so an easing that arrives mid-ramp is itself felt as a
        smooth change of acceleration rather than a step.
        """
        deltas = [t - c for t, c in zip(target, cmd)]
        norm = math.sqrt(sum(d * d for d in deltas))
        if norm <= 0.0:
            self.vel = [0.0, 0.0, 0.0]
            self.limit = 0.0
            return
        toward_zero = sum(d * c for d, c in zip(deltas, cmd)) < 0.0
        limit = self.decel * decel_scale if toward_zero else self.accel * accel_scale
        self.limit = limit
        if self.jerk > 0.0:
            speed = min(limit, math.sqrt(2.0 * self.jerk * norm))
            want = [d / norm * speed for d in deltas]
            dv = [w - v for w, v in zip(want, self.vel)]
            dv_norm = math.sqrt(sum(x * x for x in dv))
            max_dv = self.jerk * self.dt
            if dv_norm > max_dv:
                dv = [x * max_dv / dv_norm for x in dv]
            vel = [v + x for v, x in zip(self.vel, dv)]
        else:
            vel = [d / norm * limit for d in deltas]
        step = [v * self.dt for v in vel]
        advance = sum(s * d for s, d in zip(step, deltas)) / norm
        if advance >= norm:
            # Would reach or pass the target this interval: land on it. The
            # velocity state is zeroed so a fresh target starts from rest.
            cmd[:] = list(target)
            self.vel = [0.0, 0.0, 0.0]
            return
        self.vel = vel
        for i, s in enumerate(step):
            cmd[i] += s


# The guard never eases braking below this fraction of ``decel``: a stop that
# slides a little beats one that takes twice as long, and the command-timeout
# safety path (dead operator → decay to zero) rides on decel too.
_TRACTION_DECEL_FLOOR = 0.5
# The guard judges only while the ramp is moving the command at no less than
# this fraction of its (scaled) rate limit — the heart of a launch or stop,
# where the wheels share a large net force. In the S-curve's tails the net
# force fades and individual torques cross zero, which is not a lift.
_TRACTION_MIN_RAMP_FRAC = 0.3
# Recovery time constant of the guard's scale while the ramp is idle.
_TRACTION_IDLE_RECOVER_S = 0.3


class TractionGuard:
    """Eases the command ramp while a wheel has lost the floor.

    A four-wheel x-drive on a rigid frame is statically indeterminate, and a
    tall base with its mass off the wheelbase centre transfers a good share of
    its weight front↔back whenever it accelerates or brakes along x (``m·a·h/L``
    — ~10% per 0.5 m/s² with the mass a metre up on a 0.4 m wheelbase). The
    wheel that goes light spins without pushing, and the two wheels that share
    a drive diagonal with it and each other become the only ones pushing: they
    can only push along their diagonal, so the base veers toward it until the
    lifted wheel lands. No wheel command can push through a wheel that isn't
    touching the floor; what does work is asking for less acceleration while
    it's light (measured on the cart: a 0.5/s ramp veered 6–28°, 0.15/s under
    2°, 0.08/s within noise). This guard does that automatically, so the ramp
    can stay brisk whenever the floor takes it.

    Detection is from the motors' torque feedback, which every velocity
    command's reply carries at no bus cost. A wheel carrying its share of the
    load — accelerating the base or, at cruise, rolling against its share of
    the rolling resistance — shows a torque in proportion; one in the air
    shows only its own inertia and bearing drag. So the wheel with the
    smallest ``|τ|`` is judged light when it falls under ``light`` × the mean
    ``|τ|`` of the four, provided that mean exceeds ``torque_min`` (below it,
    at a gentle cruise or a standstill, the torques say nothing and the guard
    stands down). Torque alone suffices: the same relation holds for any
    command mix, since a wheel's expected torque always scales with its load.

    A wheel has to look light for ``confirm`` consecutive cycles before the
    guard acts: every wheel's torque passes through zero when a ramp turns
    around (the rear pair goes from braking to driving as a stop completes),
    and for a cycle or two that looks exactly like a lift. A real lift lasts
    the whole ramp.

    The output is a scale on the ramp's rate limits. It drops toward
    ``floor`` within ``drop_s`` of a wheel being confirmed light (fast — the
    lift takes ~0.2 s to turn into a slide) and recovers toward 1 once all
    four carry load: over ``recover_s`` while the ramp is still moving (slow,
    so a launch hunts at most once rather than chattering), and within a few
    tenths of a second once it isn't, so an easing picked up in the last
    moments of a stop — where the unloaded end's torque fades first, which
    looks light and is harmless — doesn't soften the next launch. Braking is
    eased by the same scale but never below :data:`_TRACTION_DECEL_FLOOR`.
    """

    def __init__(
        self,
        light: float,
        floor: float,
        torque_min: float,
        dt: float,
        drop_s: float = 0.1,
        recover_s: float = 1.5,
        confirm: int = 3,
    ) -> None:
        if not (0.0 < light < 1.0):
            raise ValueError("traction light ratio must be in (0, 1)")
        if not (0.0 < floor <= 1.0):
            raise ValueError("traction floor must be in (0, 1]")
        if not (torque_min >= 0.0 and dt > 0.0 and drop_s > 0.0 and recover_s > 0.0):
            raise ValueError(
                "traction torque_min must be >= 0; dt, drop_s, recover_s > 0"
            )
        self.light = light
        self.floor = floor
        self.torque_min = torque_min
        self.dt = dt
        self.drop_s = drop_s
        self.recover_s = recover_s
        self.confirm = max(1, confirm)
        self._streak = 0  # consecutive cycles with a light wheel
        self.scale = 1.0
        self.light_wheel: int | None = None  # WHEELS index judged light this cycle
        # Last reading's mean |τ|, and its lightest wheel (index, |τ|, ratio to
        # the mean) whether or not it was judged light — for the log.
        self.mean_torque = 0.0
        self.light_index = 0
        self.light_torque = 0.0
        self.light_ratio = 1.0

    @property
    def decel_scale(self) -> float:
        return max(self.scale, _TRACTION_DECEL_FLOOR)

    def update(self, torques: Sequence[float] | None, ramping: bool = True) -> float:
        """Feed this cycle's per-wheel torques (Nm, :data:`WHEELS` order, or
        ``None`` when there is no fresh reading) and return the ramp scale.

        ``ramping`` says the command is being moved at a substantial rate
        (see :data:`_TRACTION_MIN_RAMP_FRAC`). Only then is the judgement
        made: while accelerating or braking the wheels share a large net force
        and each one's torque tracks its load, but at cruise the net force is
        just rolling resistance and the torques are dominated by the wheels
        working against each other (a heading-hold nudge, a hair of speed
        inconsistency), which says nothing about the floor — and the ramp
        scale is moot then anyway, so the guard recovers.
        """
        light: int | None = None
        if ramping and torques is not None and len(torques) == len(WHEELS):
            mags = [abs(t) for t in torques]
            mean = sum(mags) / len(mags)
            i = min(range(len(mags)), key=mags.__getitem__)
            self.mean_torque = mean
            self.light_index = i
            self.light_torque = mags[i]
            self.light_ratio = mags[i] / mean if mean > 0.0 else 1.0
            if mean >= self.torque_min and mags[i] < self.light * mean:
                light = i
        self._streak = self._streak + 1 if light is not None else 0
        if self._streak < self.confirm:
            light = None
        self.light_wheel = light
        if light is not None:
            self.scale += (self.floor - self.scale) * min(1.0, self.dt / self.drop_s)
        else:
            recover = self.recover_s if ramping else _TRACTION_IDLE_RECOVER_S
            self.scale += (1.0 - self.scale) * min(1.0, self.dt / recover)
        return self.scale


def stroke_rows(
    turns: tuple[float, float, float, float] | list[float],
) -> tuple[list[float], list[float], list[float]]:
    """Kinematic rows mapping the wheels' effective radii to a stroke's motion.

    ``turns`` is each wheel's net rotation over the stroke (rad, in
    :data:`WHEELS` order, motor convention — as reported by the drivers, before
    ``WHEEL_SIGNS``). Returns three 4-vectors ``(kx, ky, kw)`` such that, with
    ``R`` the wheels' effective radii, the body moved ``kx·R`` forward and
    ``ky·R`` left and turned ``kw·R / lever`` CCW (``lever`` the wheel's
    rotation lever arm, ``(a + b) / √2`` for a wheelbase of ``2a`` × ``2b``).

    Surface travel of wheel *i* along its drive axis is
    ``u_i = (mx·x + my·y)/√2 + mw·lever·θ`` (the mixing rows are ±1 stand-ins
    for the true ±1/√2 drive directions). That 4×3 map has orthogonal columns
    (norms² 2, 2, 4·lever²), so its least-squares inverse is the transpose
    scaled per column — which is what these rows are, with ``u_i = R_i·φ_i``.
    """
    if len(turns) != len(WHEELS):
        raise ValueError("each stroke needs one wheel rotation per wheel")
    phi = [WHEEL_SIGNS[w.motor_id] * float(t) for w, t in zip(WHEELS, turns)]
    kx = [w.mx * p / (2.0 * math.sqrt(2.0)) for w, p in zip(WHEELS, phi)]
    ky = [w.my * p / (2.0 * math.sqrt(2.0)) for w, p in zip(WHEELS, phi)]
    kw = [w.mw * p / 4.0 for w, p in zip(WHEELS, phi)]
    return kx, ky, kw


def solve_wheel_scale(
    strokes: list[tuple[tuple[float, float, float, float], float, float, float]],
    lever_m: float,
) -> tuple[float, float, float, float]:
    """Fit per-wheel speed scales from measured straight strokes.

    Each stroke is ``(wheel_turns, forward_m, left_m, heading_rad)``:
    the four wheels' net rotation (rad, in :data:`WHEELS` order, motor
    convention — i.e. as reported by the drivers, before ``WHEEL_SIGNS``),
    and the body displacement actually measured over the stroke: forward and
    leftward distance (tape measure against the start marks, body frame at
    stroke start) and net heading change (gyro, CCW positive). ``lever_m`` is
    the wheel's rotation lever arm, ``(a + b) / √2`` for a wheelbase of
    ``2a`` × ``2b``.

    Inverting the x-drive kinematics, the body displacement is a fixed linear
    map of the wheels' surface travel ``R_i·φ_i`` (``R_i`` the *effective*
    radius of wheel *i*), so each stroke gives three linear equations in the
    four ``R_i`` (see :func:`stroke_rows`); two strokes in different directions
    (forward and left) determine them. The returned scales are ``R̄ / R_i``
    normalized to a mean of 1 — multiply wheel *i*'s command by its scale and
    the wheels become mutually consistent, which is what removes the lateral
    slide (the common-mode radius only rescales ``max_speed``, so it's
    deliberately not resolved).

    The measurements can come from a tape measure and the gyro, or — without
    marking the floor — from the overhead ZED's positional tracking via
    ``axol diag.base-calibrate`` (:mod:`almond_axol.diagnostics.base.calibrate`),
    which also solves for the camera's mounting offset.

    Raises ``ValueError`` if the strokes don't determine the radii (fewer than
    two, or all in one direction).
    """
    import numpy as np

    if lever_m <= 0.0:
        raise ValueError("lever_m must be positive")
    rows: list[list[float]] = []
    rhs: list[float] = []
    for turns, fwd, left, heading in strokes:
        kx, ky, kw = stroke_rows(turns)
        rows.extend([kx, ky, kw])
        rhs.extend([fwd, left, heading * lever_m])
    a = np.array(rows)
    b = np.array(rhs)
    if len(strokes) < 2 or np.linalg.matrix_rank(a) < len(WHEELS):
        raise ValueError(
            "wheel radii are not determined by these strokes — record at least "
            "two, in different directions (e.g. forward and left)"
        )
    radii, *_ = np.linalg.lstsq(a, b, rcond=None)
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("fit produced a non-positive wheel radius — check the signs")
    scales = radii.mean() / radii
    scales /= scales.mean()
    return tuple(float(s) for s in scales)


@dataclass
class CartConfig:
    """Configuration for the powered Axol Cart.

    Attributes:
        enabled:         Whether this robot has a powered cart at all. Only
                         consulted by entry points that support both variants
                         (``axol teleop``); code constructing a :class:`Cart`
                         directly ignores it.
        channel:         SocketCAN interface for the wheel motors. ``None``
                         disables the wheels entirely (lift-only cart).
        max_speed:       Peak wheel speed (rad/s) at a full-deflection command.
        turn_scale:      Rotation weight relative to translation, in [0, 1].
        accel:           Ramp rate of the normalized body command while it
                         grows away from zero, in full-stick units per second.
                         The default takes 2 s from rest to full deflection.
                         Gentler launches also slide less sideways on uneven
                         floors (an unloaded diagonal can't take up lateral
                         force; see the module docstring).
        decel:           Ramp rate while the command shrinks toward zero
                         (stick released, speed reduced, or reversed). Faster
                         than ``accel`` so stops are brisk: the default halts
                         from full stick in 1 s (plus the jerk tail).
        jerk:            Limit on how fast the ramp rate itself changes, in
                         full-stick units per second². Turns the trapezoid
                         into an S-curve with no acceleration step at either
                         end of a launch or stop, which is what the operator
                         feels as "smooth". The default reaches ``accel`` in
                         0.25 s and adds ~0.3 s to a stop. 0 disables.
        wheel_scale:     Per-wheel command multipliers, in :data:`WHEELS`
                         order (front-left, front-right, back-left,
                         back-right), compensating each wheel's effective
                         radius. Wheels that aren't mutually consistent slide
                         the base sideways even with the heading held — a 3%
                         radius spread is ~2 cm per 3 m. Measure with
                         :func:`solve_wheel_scale` from two tape-measured
                         strokes; ``(1, 1, 1, 1)`` is uncalibrated.
        traction:        Ease the ramp while a wheel has lost the floor, judged
                         from the motors' torque feedback (see
                         :class:`TractionGuard`). A wheel that lifts under the
                         weight shift of a launch or a stop can't push, and the
                         base veers along the remaining pair's diagonal; asking
                         for less acceleration while it's light is the only
                         wheel-level remedy. On by default; it does nothing on
                         a floor and load that keep all four wheels down.
        traction_light:  A wheel whose ``|torque|`` is below this fraction of
                         the four wheels' mean counts as light. A loaded wheel
                         carries its share (~1 Nm on a launch); one in the air
                         shows only its inertia and bearing drag (~0.1 Nm).
        traction_floor:  The guard never eases ``accel`` below this fraction.
                         (Braking has its own, fixed floor of 0.5.)
        traction_torque_min: Mean ``|torque|`` (Nm) below which the wheels say
                         nothing about load (gentle cruise, standstill) and the
                         guard stands down.
        traction_log:    Log the guard: a line whenever it starts easing and a
                         per-stroke summary (min ratio, which wheel, how far the
                         ramp was eased). For tuning; off in normal operation.
        axis_snap_deg:   Translation headings within this many degrees of a
                         cardinal axis (forward/back/left/right) are snapped
                         onto that axis, absorbing off-axis thumb error during
                         stick flicks so a "straight" command drives exactly
                         straight. 0 disables. Deliberate diagonals (further
                         off-axis than this) pass through unchanged.
        imu:             Use the carrier board's BMI088 as the yaw reference
                         for the heading hold (wired by teleop; see
                         ``almond_axol.robot.gyro``). Independent of the
                         cameras — the overhead ZED keeps its gst pipeline.
                         On by default: without a yaw reference the hold is
                         inert and the heading wanders several degrees per
                         stroke. A missing gyro only logs a warning.
        yaw_hold_gain:   Heading-hold feedback gain, normalized wz per rad of
                         heading error. While translating without a commanded
                         rotation, the yaw rate fed via :meth:`Cart.feed_yaw_rate`
                         is integrated into a heading error that is steered
                         back to zero. 0 disables; a *negative* gain
                         compensates a sensor whose sign convention is
                         inverted. Idle no-op unless yaw rates are actually
                         fed (and fresh).
        yaw_hold_max:    Clamp on the heading-hold correction (normalized wz).
        yaw_log:         Trace the heading hold: a 10 Hz state line while the
                         cart translates and a per-stroke summary of the
                         heading it actually drifted (see :class:`_YawLog`).
                         For diagnosing drift; off in normal operation.
        deadzone:        Stick deadzone (fraction of full deflection) applied
                         by input frontends (VR thumbsticks, gamepad).
        hold_kp:         Position stiffness (Nm/rad) of the parked MIT hold;
                         0 disables parking (wheels just idle in velocity mode).
        hold_kd:         Damping (Nm·s/rad) of the parked MIT hold.
        frequency:       Wheel command task rate in Hz.
        command_timeout: Seconds without a fresh :meth:`Cart.set_command`
                         before the target is forced to zero (and the lift
                         stopped). Protects against a dead command source.
        lift:            Whether the telescoping lift is present (the
                         jelly_legs board on the chest CAN bus, see
                         :mod:`almond_axol.robot.lift`). The chest bus being
                         down at enable time only disables the lift with a
                         warning — the buses are independent, so a cart can
                         still drive without it.
        lift_channel:    SocketCAN interface of the chest bus carrying the
                         jelly_legs lift controller.
        lift_speed:      Lift jog speed in encoder counts/s (the firmware's
                         full speed is ~650).
    """

    enabled: bool = False
    channel: str | None = DEFAULT_CHANNEL
    max_speed: float = 20.0
    turn_scale: float = 1.0
    accel: float = 0.5
    decel: float = 1.0
    jerk: float = 2.0
    wheel_scale: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    traction: bool = True
    traction_light: float = 0.35
    traction_floor: float = 0.2
    traction_torque_min: float = 0.3
    traction_log: bool = False
    axis_snap_deg: float = 15.0
    imu: bool = True
    yaw_hold_gain: float = 2.0
    yaw_hold_max: float = 0.3
    yaw_log: bool = False
    deadzone: float = 0.15
    hold_kp: float = 60.0
    hold_kd: float = 1.5
    frequency: float = 50.0
    command_timeout: float = 0.3
    lift: bool = True
    lift_channel: str = CAN_CHEST
    lift_speed: int = JOG_SPEED

    def __post_init__(self) -> None:
        # A non-positive ramp rate is not "no ramp" — it's a command that
        # never changes, and for decel that is a cart that never stops.
        for name in ("accel", "decel"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value)):
                raise ValueError(f"cart {name} must be a finite number")
            if value <= 0.0:
                raise ValueError(f"cart {name} must be positive (got {value})")
        if not (
            isinstance(self.jerk, (int, float))
            and math.isfinite(self.jerk)
            and self.jerk >= 0.0
        ):
            raise ValueError("cart jerk must be a finite number >= 0")
        scale = tuple(self.wheel_scale)
        if len(scale) != len(WHEELS) or not all(
            isinstance(s, (int, float)) and math.isfinite(s) and 0.5 <= s <= 2.0
            for s in scale
        ):
            raise ValueError(
                "cart wheel_scale needs one finite value in [0.5, 2] per wheel "
                f"({len(WHEELS)} wheels)"
            )
        self.wheel_scale = tuple(float(s) for s in scale)
        if self.traction:
            # Same checks as the guard itself, surfaced at config time.
            TractionGuard(
                self.traction_light,
                self.traction_floor,
                self.traction_torque_min,
                1.0 / self.frequency,
            )


class _TractionLog:
    """Per-stroke trace of the traction guard (see ``CartConfig.traction_log``).

    Logs once when the guard first eases within a stroke (which wheel, its
    torque against the mean) and a summary when the stroke ends: peak mean
    torque, the lightest any wheel got, how far and how long the ramp was
    eased. Strokes where the guard never had enough torque to judge are
    summarized too — that's how ``traction_torque_min`` gets tuned.
    """

    def __init__(self) -> None:
        self._active = False
        self._reset()

    def _reset(self) -> None:
        self._t0 = time.monotonic()
        self._peak_mean = 0.0
        self._min_ratio = 1.0
        self._min_wheel: int | None = None
        self._min_scale = 1.0
        self._eased_cycles = 0
        self._judged_cycles = 0
        self._announced = False

    def update(self, guard: TractionGuard, *, driving: bool) -> None:
        if driving and not self._active:
            self._active = True
            self._reset()
        elif not driving and self._active:
            self._active = False
            self._summarize()
            return
        if not self._active:
            return
        mean = guard.mean_torque
        self._peak_mean = max(self._peak_mean, mean)
        if mean >= guard.torque_min:
            self._judged_cycles += 1
        if guard.light_wheel is not None:
            self._eased_cycles += 1
            self._min_scale = min(self._min_scale, guard.scale)
            if not self._announced:
                self._announced = True
                _logger.info(
                    "traction: %s light (%.2f of %.2f Nm mean) — easing the ramp",
                    WHEELS[guard.light_wheel].name,
                    abs(guard.light_torque),
                    mean,
                )
        if mean >= guard.torque_min and guard.light_ratio < self._min_ratio:
            self._min_ratio = guard.light_ratio
            self._min_wheel = guard.light_index

    def _summarize(self) -> None:
        dur = time.monotonic() - self._t0
        if self._judged_cycles == 0:
            _logger.info(
                "traction: stroke %.1fs — never judged (peak mean |τ| %.2f Nm < "
                "torque_min); lower traction_torque_min if wheels lifted",
                dur,
                self._peak_mean,
            )
            return
        wheel = WHEELS[self._min_wheel].name if self._min_wheel is not None else "none"
        _logger.info(
            "traction: stroke %.1fs — peak mean |τ| %.2f Nm, lightest %s at %.2f of "
            "mean, eased %d cycles to %.2f× accel",
            dur,
            self._peak_mean,
            wheel,
            self._min_ratio,
            self._eased_cycles,
            self._min_scale,
        )


class _YawLog:
    """Per-stroke trace of the heading hold (see ``CartConfig.yaw_log``).

    Fed every command cycle; emits a throttled state line while the cart is
    translating and a summary when the stroke ends.

    The number to read is the stroke's heading drift — ``yaw_err``, the
    measured rotation since the stroke began. A stroke ending near zero means
    the hold did its job and whatever drift is still visible is the lateral
    slide along the unloaded diagonal, which no wheel command can correct
    (``diagnostics/base/floor_sim.py``); a growing one means the hold isn't
    working, and the rest of the line says why — no samples, stale samples, a
    correction pinned at ``yaw_hold_max``, or a bias being integrated into the
    error while the cart never sits still long enough to learn it.
    """

    def __init__(self) -> None:
        self._t0: float | None = None  # stroke start, None between strokes
        self._next_trace = 0.0
        self._cycles = 0
        self._held = 0
        self._saturated = 0
        self._corr_sum = 0.0
        self._corr_max = 0.0
        self._age_max = 0.0
        self._err_last = 0.0  # the controller zeroes yaw_err as a stroke ends
        self._samples0 = 0

    def update(
        self,
        *,
        now: float,
        translating: bool,
        held: bool,
        rate: float | None,
        bias: float,
        err: float,
        corr: float,
        saturated: bool,
        age: float | None,
        samples: int,
    ) -> None:
        if not translating:
            if self._t0 is not None:
                self._summarize(now, bias, samples)
                self._t0 = None
            return

        if self._t0 is None:
            self._t0 = now
            self._next_trace = now
            self._cycles = self._held = self._saturated = 0
            self._corr_sum = self._corr_max = self._age_max = 0.0
            self._samples0 = samples

        self._cycles += 1
        self._held += int(held)
        self._saturated += int(saturated)
        self._corr_sum += abs(corr)
        self._corr_max = max(self._corr_max, abs(corr))
        self._err_last = err
        if age is not None:
            self._age_max = max(self._age_max, age)

        if now >= self._next_trace:
            self._next_trace = now + 1.0 / _YAW_TRACE_HZ
            _logger.info(
                "yaw t=%5.2fs rate=%s bias=%+.4f err=%+6.2fdeg corr=%+.3f%s age=%s",
                now - self._t0,
                f"{rate:+.4f}" if rate is not None else "none",
                bias,
                math.degrees(err),
                corr,
                " SAT" if saturated else "",
                f"{age * 1e3:.0f}ms" if age is not None else "none",
            )

    def _summarize(self, now: float, bias: float, samples: int) -> None:
        assert self._t0 is not None
        dt = max(now - self._t0, 1e-6)
        cycles = max(self._cycles, 1)
        _logger.info(
            "yaw stroke: %.1fs, heading drift %+.2fdeg, hold active %d%% of %d "
            "cycles (|corr| mean %.3f max %.3f, saturated %d%%), imu %.0fHz "
            "max age %.0fms, bias %+.4frad/s",
            dt,
            math.degrees(self._err_last),
            round(100 * self._held / cycles),
            self._cycles,
            self._corr_sum / cycles,
            self._corr_max,
            round(100 * self._saturated / cycles),
            (samples - self._samples0) / dt,
            self._age_max * 1e3,
            bias,
        )


class Cart:
    """Latched-command controller for the powered cart (wheels + lift).

    Typical usage::

        cart = Cart(CartConfig())
        await cart.enable()
        cart.set_command(vx=0.5, vy=0.0, wz=0.0, lift=0)   # from any thread
        ...
        await cart.disable()

    :meth:`set_command` only latches the target; the internal command task
    owns all bus/GPIO traffic. Values are normalized to [-1, 1] (body frame:
    +x forward, +y left, +wz CCW) and scaled by ``CartConfig.max_speed`` /
    ``turn_scale``; ``lift`` is +1 up / 0 stop / -1 down.
    """

    def __init__(self, config: CartConfig = CartConfig()) -> None:
        self._config = config
        self._bus: CanBus | None = None
        self._motors: list[MotorDriver] = []
        self._lift: Lift | None = None
        self._task: asyncio.Task | None = None
        self._shutdown_pending = False
        self._motors_disabled = False

        # Latched target, written from any thread (single-reference swap is
        # atomic under the GIL), consumed by the command task.
        self._target: tuple[float, float, float, int] = (0.0, 0.0, 0.0, STOP)
        self._target_time: float = 0.0

        # Latest external yaw-rate sample (rad/s CCW, monotonic timestamp),
        # written from any thread; None until a sensor feeds one. The counter
        # lets the command loop report the sensor's delivered rate, which is
        # what distinguishes a slow source from a dead one.
        self._yaw_rate: tuple[float, float] | None = None
        self._yaw_samples = 0

        # Latest torque reported by each wheel (Nm, WHEELS order), written by
        # the drivers' feedback callbacks — every command's reply carries one —
        # and read by the traction guard. None until a wheel has replied.
        self._torques: list[float | None] = [None] * len(WHEELS)

        # Introspection for status displays (updated by the command task).
        self.body_cmd: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.wheel_speeds: list[float] = [0.0] * len(WHEELS)
        self.wheel_torques: list[float] = [0.0] * len(WHEELS)
        self.traction_scale: float = 1.0
        self.yaw_correction: float = 0.0
        self.lift_dir: int = STOP
        self.parked: bool = False
        self.park_failed: bool = False
        self.send_failed: bool = False

    @property
    def config(self) -> CartConfig:
        """The configuration this cart was constructed with (read-only use)."""
        return self._config

    @property
    def has_wheels(self) -> bool:
        """True when a wheel CAN channel is configured."""
        return self._config.channel is not None

    @property
    def has_lift(self) -> bool:
        """True when the lift driver is up (chest bus opened at enable time)."""
        return self._lift is not None

    @property
    def lift_status(self) -> LiftStatus | None:
        """Latest jelly_legs status frame, or None (no lift / board silent)."""
        return self._lift.status if self._lift is not None else None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def enable(self) -> None:
        """Open the CAN bus, enable the wheel motors, init the lift, and start
        the command task."""
        if self._shutdown_pending:
            raise HardwareCleanupError(
                "cart shutdown is incomplete; retry disable() before reconnecting"
            )
        if self._task is not None or self._bus is not None or self._motors:
            raise RuntimeError("cart is already enabled")
        cfg = self._config
        if cfg.lift:
            # The chest bus is optional and independent of the wheels: a
            # missing/unpowered lift only costs the lift, never the drive.
            lift = Lift(cfg.lift_channel, cfg.lift_speed)
            self._lift = lift
            try:
                await lift.start()
            except BaseException as setup_error:
                try:
                    await lift.close()
                except BaseException as cleanup_error:
                    self._shutdown_pending = True
                    mark_hardware_cleanup_uncertain(setup_error, cleanup_error)
                    raise setup_error
                self._lift = None
                if not isinstance(setup_error, Exception):
                    raise setup_error
                _logger.warning(
                    "cart lift: could not open the chest bus %s (%s) — "
                    "the lift is disabled for this session",
                    cfg.lift_channel,
                    setup_error,
                )

        try:
            if cfg.channel is not None:
                # Seamless enable: bring the wheel interface up if it isn't yet
                # (boot normally handles this via can.setup's @reboot script,
                # but a freshly plugged adapter or a manual teardown shouldn't
                # need a separate can.enable). A missing interface raises with
                # its name.
                from ..cli.can.setup import bring_up_interfaces, iface_up

                if not iface_up(cfg.channel):
                    bring_up_interfaces([cfg.channel])
                self._bus = CanBus(cfg.channel)
                await self._bus.start()
                # Wheel IDs 1-4 collide with the arm-bus MyActuator IDs in the
                # driver-inference table, so the Damiao protocol is forced.
                self._motors = [
                    make_driver(self._bus, w.motor_id, motor_type="damiao")
                    for w in WHEELS
                ]
                self._motors_disabled = False
                self._torques = [None] * len(WHEELS)
                for i, m in enumerate(self._motors):
                    m.set_feedback_callback(self._torque_sink(i))
                # Widen the position-mapping range (RAM only) before enable()
                # reads it back, so multi-turn wheel positions stay valid for
                # the MIT park hold.
                await _gather_all_or_raise(
                    *[
                        m._write_register(_DM_REG_PMAX, _SESSION_PMAX)
                        for m in self._motors
                    ]
                )
                await _gather_all_or_raise(*[m.enable() for m in self._motors])
                for w, m in zip(WHEELS, self._motors):
                    if abs(m._p_max - _SESSION_PMAX) > 1.0:
                        _logger.warning(
                            "cart wheel %s PMAX readback %.0f != %.0f — parking "
                            "may misbehave",
                            w.name,
                            m._p_max,
                            _SESSION_PMAX,
                        )
                await _gather_all_or_raise(
                    *[m.set_control_mode(ControlMode.VELOCITY) for m in self._motors]
                )
                _logger.info("cart wheels enabled on %s", cfg.channel)
        except BaseException as setup_error:
            # A failed enable() propagates to the caller, who never calls
            # disable() — so everything opened above must be torn down here,
            # or the lift's CAN reader and jog task would keep running (and a
            # half-started wheel bus would stay open).
            try:
                await self.disable()
            except BaseException as cleanup_error:
                mark_hardware_cleanup_uncertain(setup_error, cleanup_error)
            raise

        if cfg.yaw_hold_gain != 0.0:
            _logger.info(
                "cart heading hold: gain=%.2f max=%.2f imu=%s%s",
                cfg.yaw_hold_gain,
                cfg.yaw_hold_max,
                cfg.imu,
                " (yaw_log on)" if cfg.yaw_log else "",
            )

        self._task = asyncio.create_task(self._command_loop(), name="cart-command")

    async def disable(self) -> None:
        """Stop the command task, stop and disable the wheels, release the lift."""
        failures: list[tuple[str, BaseException]] = []

        task = self._task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except BaseException as exc:
                failures.append(("command task", exc))
            if task.done():
                self._task = None
            else:
                failures.append(
                    ("command task", RuntimeError("cart command task did not stop"))
                )

        if self._motors and not self._motors_disabled:
            try:
                # Leave impedance park (if held) and command a stop before
                # disabling, mirroring the manual-drive teardown.
                if self.parked:
                    await self._unpark()
                await asyncio.gather(*[m.set_velocity(0.0) for m in self._motors])
            except Exception:  # noqa: BLE001 - best-effort stop before disable
                _logger.exception("cart wheel stop command failed")
            try:
                results = await asyncio.gather(
                    *[m.disable() for m in self._motors], return_exceptions=True
                )
            except BaseException as exc:
                failures.append(("wheel disable group", exc))
                self._motors_disabled = False
            else:
                wheel_failures = [
                    (f"{wheel.name} wheel disable", result)
                    for wheel, result in zip(WHEELS, results)
                    if isinstance(result, BaseException)
                ]
                failures.extend(wheel_failures)
                self._motors_disabled = not wheel_failures

        # Keep the wheel bus open until every torque-off is verified, so a
        # later disable() call can retry rather than losing the only control
        # path to a potentially energised motor.
        if self._bus is not None and (not self._motors or self._motors_disabled):
            try:
                await self._bus.close()
            except BaseException as exc:
                failures.append(("wheel bus close", exc))
            else:
                self._bus = None
                self._motors = []
                self._motors_disabled = False

        if self._lift is not None:
            try:
                await self._lift.close()
            except BaseException as exc:
                failures.append(("lift close", exc))
            else:
                self._lift = None

        self._shutdown_pending = bool(failures)
        if failures:
            label, first = failures[0]
            for extra_label, extra in failures[1:]:
                first.add_note(
                    f"additional cart {extra_label} cleanup failure: "
                    f"{type(extra).__name__}: {extra}"
                )
            raise HardwareCleanupError(
                f"cart {label} failed; hardware ownership is uncertain"
            ) from first
        _logger.info("cart disabled")

    # ------------------------------------------------------------------
    # Command interface (any thread)
    # ------------------------------------------------------------------

    def set_command(self, vx: float, vy: float, wz: float, lift: int = STOP) -> None:
        """Latch a normalized body-velocity + lift command.

        Args:
            vx:   Forward velocity, [-1, 1].
            vy:   Leftward velocity, [-1, 1].
            wz:   Counter-clockwise rotation, [-1, 1].
            lift: +1 raise, 0 stop, -1 lower.

        Safe to call from any thread at any rate. The command task consumes
        the latest value; if no fresh command arrives within
        ``CartConfig.command_timeout`` the target decays to a full stop.
        """

        def clamp(v: float, *, name: str) -> float:
            if isinstance(v, bool):
                raise ValueError(f"cart {name} command must be a finite number")
            try:
                value = float(v)
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"cart {name} command must be a finite number"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"cart {name} command must be a finite number")
            return max(-1.0, min(1.0, value))

        try:
            vx, vy, wz = (
                clamp(vx, name="vx"),
                clamp(vy, name="vy"),
                clamp(wz, name="wz"),
            )
            if (
                isinstance(lift, bool)
                or not isinstance(lift, int)
                or lift
                not in {
                    DOWN,
                    STOP,
                    UP,
                }
            ):
                raise ValueError("cart lift command must be -1, 0, or 1")
        except ValueError:
            # Invalid input must fail toward a stop. In particular,
            # ``min(1, NaN)`` evaluates to 1 in Python, so a naive clamp can
            # turn a malformed network/SDK value into full cart speed.
            self._target = (0.0, 0.0, 0.0, STOP)
            self._target_time = time.monotonic()
            raise

        # Snap near-cardinal translation onto the axis (see
        # CartConfig.axis_snap_deg): thumbstick flicks are rarely perfectly
        # straight, and without this the transient off-axis component steers
        # the launch direction.
        snap = math.radians(self._config.axis_snap_deg)
        if snap > 0.0 and (vx != 0.0 or vy != 0.0):
            heading = math.atan2(vy, vx)
            nearest = round(heading / (math.pi / 2)) * (math.pi / 2)
            if abs(heading - nearest) <= snap:
                mag = math.hypot(vx, vy)
                vx = mag * math.cos(nearest)
                vy = mag * math.sin(nearest)

        self._target = (vx, vy, wz, lift)
        self._target_time = time.monotonic()

    def apply_vr_frame(self, frame, resetting: bool = False) -> None:  # noqa: ANN001
        """Map a headset :class:`~almond_axol.vr.models.VRFrame` to a command.

        The single source of truth for the VR control mapping, shared by
        plain teleop (``VRTeleop``) and data collection (``AxolVRTeleop``) so
        the two flows cannot drift apart. Stick deflection is the deadman:
        the cart moves only while a stick is pushed past its deadzone (or a
        stick click holds the lift), independent of the arm engage toggle.

        Args:
            frame:     The incoming VR frame (only its stick fields are read).
            resetting: True while the arms replay a reset trajectory (or the
                       frame itself carries a reset) — forces a stop so the
                       base doesn't creep during the return to rest.

        Thread-safe (only latches the target); staleness is handled by the
        command task, so a dead frame stream times out to a full stop.
        """
        if resetting or frame.reset:
            self.set_command(0.0, 0.0, 0.0, STOP)
            return
        dz = self._config.deadzone
        # WebXR sticks: +x right, +y pulled back → body frame +x forward,
        # +y left, +wz CCW.
        vx = -deadzone(frame.l_stick_y, dz)
        vy = -deadzone(frame.l_stick_x, dz)
        wz = -deadzone(frame.r_stick_x, dz)
        if frame.r_stick_click and not frame.l_stick_click:
            lift = UP
        elif frame.l_stick_click and not frame.r_stick_click:
            lift = DOWN
        else:
            lift = STOP
        self.set_command(vx, vy, wz, lift)

    async def read_wheels(self) -> tuple[list[float], list[float]]:
        """Read every wheel's current position and velocity from the motors.

        Returns ``(positions, velocities)`` in :data:`WHEELS` order, motor
        convention (rad and rad/s as the drivers report them, before
        ``WHEEL_SIGNS``). Positions are multi-turn within the session's widened
        ±PMAX mapping (see the module docstring), so differences between two
        reads are the wheels' net rotation — the odometry the wheel-radius
        calibration (``axol diag.base-calibrate``) pairs with the camera's
        measured displacement. Safe to call while the command task is driving;
        the feedback requests interleave with its command frames.

        Raises ``RuntimeError`` if the wheels are not enabled.
        """
        if not self._motors:
            raise RuntimeError("cart wheels are not enabled")
        positions = await asyncio.gather(*[m.get_position() for m in self._motors])
        velocities = await asyncio.gather(*[m.get_velocity() for m in self._motors])
        return list(positions), list(velocities)

    def _torque_sink(self, index: int) -> Callable[[float, float], None]:
        def sink(_position: float, torque: float) -> None:
            self._torques[index] = torque

        return sink

    def feed_yaw_rate(self, rate: float) -> None:
        """Latch an external yaw-rate sample (rad/s, CCW positive from above).

        Fed by a gyro source (the board BMI088, see
        ``almond_axol.robot.gyro``) from any thread at any rate; the command
        task's heading hold consumes the latest sample. Samples older than
        the staleness window are ignored, so a dead sensor simply disables
        the hold rather than freezing a stale correction.
        """
        try:
            value = float(rate)
        except (OverflowError, TypeError, ValueError):
            value = math.nan
        if not math.isfinite(value):
            # A malformed/dead IMU must disable heading hold, never inject a
            # NaN into wheel mixing. Keep the motion target itself unchanged.
            self._yaw_rate = None
            return
        self._yaw_rate = (value, time.monotonic())
        self._yaw_samples += 1

    # ------------------------------------------------------------------
    # Command task
    # ------------------------------------------------------------------

    async def _park(self) -> list[float] | None:
        """Switch the wheels to the MIT position hold at their current positions.

        Returns the per-wheel anchor positions, or None if parking is not
        currently safe:

        - a wheel is still measurably moving (coasting past the ramped-down
          command, e.g. on a slope) — retried next cycle once it settles, or
        - a wheel reports a position too close to the widened ±PMAX mapping
          limit, where a wrapped/clamped anchor would mean a phantom position
          error at full torque (sets :attr:`park_failed`; not retried).
        """
        velocities = await asyncio.gather(*[m.get_velocity() for m in self._motors])
        if any(abs(v) > _PARK_MAX_WHEEL_SPEED for v in velocities):
            return None
        positions = await asyncio.gather(*[m.get_position() for m in self._motors])
        if any(abs(p) > 0.9 * _SESSION_PMAX for p in positions):
            self.park_failed = True
            _logger.warning(
                "cart wheel position near the ±PMAX mapping limit — parking "
                "disabled. Power-cycle the base to reset wheel positions."
            )
            return None
        await asyncio.gather(
            *[m.set_control_mode(ControlMode.IMPEDANCE) for m in self._motors]
        )
        return list(positions)

    async def _unpark(self) -> None:
        """Return parked wheels to VELOCITY mode (clears the motors' command state)."""
        await asyncio.gather(
            *[m.set_control_mode(ControlMode.VELOCITY) for m in self._motors]
        )

    async def _command_loop(self) -> None:
        """Apply ramp limiting, mixing, park/unpark, and lift edges at the
        configured rate.

        While driving the wheels track the ramp-limited command in VELOCITY
        mode. Once the command has ramped to zero (and the wheels are measured
        slow) they are parked: held at their current positions by the motor's
        internal MIT position loop with ``hold_kp``/``hold_kd``. The hold
        command is re-sent every cycle to keep the lost-comm watchdog fed.
        Holding in the motor's own loop (rather than an outer software loop
        over CAN) is what makes the wheel rigid instead of giving first and
        correcting after.
        """
        cfg = self._config
        interval = 1.0 / cfg.frequency
        ramp = VectorRamp(cfg.accel, cfg.decel, cfg.jerk, interval)
        guard = (
            TractionGuard(
                cfg.traction_light,
                cfg.traction_floor,
                cfg.traction_torque_min,
                interval,
            )
            if cfg.traction
            else None
        )
        traction_log = (
            _TractionLog() if cfg.traction_log and guard is not None else None
        )
        cmd = [0.0, 0.0, 0.0]  # ramped (vx, vy, wz), normalized [-1, 1]
        hold_pos: list[float] | None = None  # per-wheel park anchors (rad)
        yaw_err = 0.0  # integrated heading error (rad) since the stroke start
        yaw_bias = 0.0  # gyro bias estimate (rad/s), learned while stopped
        yaw_log = _YawLog() if cfg.yaw_log else None
        warned_silent = False
        t_loop0 = time.monotonic()

        while True:
            t_iter = time.perf_counter()

            vx, vy, wz, lift_dir = self._target
            # A dead command source (teleop thread gone, headset stream
            # dropped) must not leave the base driving: decay to a stop.
            if time.monotonic() - self._target_time > cfg.command_timeout:
                vx, vy, wz, lift_dir = 0.0, 0.0, 0.0, STOP

            # Traction guard: judge last cycle's torque replies (only while the
            # wheels are in velocity mode — a parked hold's torques say nothing
            # about the floor) and ease the ramp while a wheel is light.
            accel_scale = decel_scale = 1.0
            if guard is not None:
                fresh = hold_pos is None and all(t is not None for t in self._torques)
                torques = [t for t in self._torques if t is not None] if fresh else None
                ramping = (
                    ramp.limit > 0.0
                    and ramp.rate >= _TRACTION_MIN_RAMP_FRAC * ramp.limit
                )
                accel_scale = guard.update(torques, ramping)
                decel_scale = guard.decel_scale
                if torques is not None:
                    self.wheel_torques = torques
                self.traction_scale = accel_scale

            # Ramp the (vx, vy, wz) command as a single vector (direction
            # preserved), accel/decel asymmetric, jerk-limited — see
            # VectorRamp.
            ramp.step(cmd, (vx, vy, wz), accel_scale, decel_scale)

            moving = any(abs(c) >= 1e-3 for c in cmd)
            driving = moving or any(abs(t) >= 1e-3 for t in (vx, vy, wz))
            if traction_log is not None and guard is not None:
                traction_log.update(guard, driving=driving)

            # Heading hold on an external yaw reference. NB: deliberately
            # *not* torque feedback — a simulation study of this exact plant
            # showed the drift mechanisms (lateral slide on an unloaded
            # diagonal, radius-mismatch path curvature) are unobservable from
            # wheel torque, while a gyro heading hold fixes everything
            # fixable (see the removed diagnostics/base/floor_sim.py in git
            # history). (Torque does reveal the *unloading* itself, which is
            # what the traction guard above acts on — by easing, not by
            # steering.) While translating with
            # no commanded rotation, integrate the fed yaw rate into the
            # heading error since the stroke began and steer it out;
            # re-reference on stops and commanded turns (the operator is
            # choosing a new heading). The gyro bias is learned while the
            # cart is stopped so it doesn't masquerade as rotation.
            yaw_corr = 0.0
            translating = math.hypot(cmd[0], cmd[1]) > 0.1
            turning = abs(cmd[2]) > 0.05
            sample = self._yaw_rate
            rate: float | None = None
            age: float | None = None
            held = saturated = False
            if cfg.yaw_hold_gain != 0.0 and sample is not None:
                rate, ts = sample
                age = time.monotonic() - ts
                if age > 0.3:  # sensor died: drop the hold
                    yaw_err = 0.0
                elif translating and not turning:
                    yaw_err += (rate - yaw_bias) * interval
                    raw_corr = -cfg.yaw_hold_gain * yaw_err
                    yaw_corr = max(-cfg.yaw_hold_max, min(cfg.yaw_hold_max, raw_corr))
                    saturated = yaw_corr != raw_corr
                    held = True
                else:
                    yaw_err = 0.0
                    if not driving:
                        yaw_bias += 0.02 * (rate - yaw_bias)
            self.yaw_correction = yaw_corr

            now = time.monotonic()
            if yaw_log is not None:
                yaw_log.update(
                    now=now,
                    translating=translating,
                    held=held,
                    rate=rate,
                    bias=yaw_bias,
                    err=yaw_err,
                    corr=yaw_corr,
                    saturated=saturated,
                    age=age,
                    samples=self._yaw_samples,
                )

            # An IMU that was asked for but never delivers is the quiet
            # failure: an unprovisioned sampler or a stalled driver leaves the
            # hold looking configured and doing nothing. Say it once, the
            # first time it matters.
            if (
                not warned_silent
                and cfg.imu
                and cfg.yaw_hold_gain != 0.0
                and driving
                and self._yaw_rate is None
                and now - t_loop0 > _YAW_SILENT_WARN_S
            ):
                warned_silent = True
                _logger.warning(
                    "cart heading hold: no yaw samples after %.0fs of driving — "
                    "the hold is inert. Check that the board gyro opened "
                    "(almond_axol.robot.gyro) earlier in this log.",
                    _YAW_SILENT_WARN_S,
                )

            speeds = mix(
                cmd[0],
                cmd[1],
                cmd[2] + yaw_corr,
                cfg.max_speed,
                cfg.turn_scale,
                cfg.wheel_scale,
            )

            if self._lift is not None:
                self._lift.command(lift_dir)
            self.lift_dir = lift_dir if self._lift is not None else STOP

            if self._motors:
                try:
                    if driving:
                        self.park_failed = False
                        if hold_pos is not None:
                            await self._unpark()
                            hold_pos = None

                    if not driving and cfg.hold_kp > 0.0 and not self.park_failed:
                        if hold_pos is None:
                            hold_pos = await self._park()
                        if hold_pos is not None:
                            await asyncio.gather(
                                *[
                                    m.set_impedance(
                                        p, 0.0, cfg.hold_kp, cfg.hold_kd, 0.0
                                    )
                                    for m, p in zip(self._motors, hold_pos)
                                ]
                            )
                    if hold_pos is None:
                        await asyncio.gather(
                            *[m.set_velocity(s) for m, s in zip(self._motors, speeds)]
                        )
                    self.send_failed = False
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - transient bus errors: retry
                    # Transient send failures (buffer full, bus-off recovery)
                    # are surfaced via send_failed; the next cycle retries.
                    self.send_failed = True

            self.body_cmd = (cmd[0], cmd[1], cmd[2])
            self.wheel_speeds = speeds
            self.parked = hold_pos is not None

            elapsed = time.perf_counter() - t_iter
            await asyncio.sleep(max(0.0, interval - elapsed))
