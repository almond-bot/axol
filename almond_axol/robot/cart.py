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
and an internal asyncio task (started by :meth:`Cart.enable`) applies slew
limiting, x-drive mixing, and the park/unpark state machine at
``CartConfig.frequency``:

- While the command is non-zero the wheels track it in VELOCITY mode.
- When the slew-limited command reaches zero (and the wheels are measured
  slow), the wheels are parked: switched to MIT/impedance mode and held at
  their current positions by the motor's internal high-bandwidth position
  loop, so the base does not roll under load.
- If no fresh command arrives within ``command_timeout`` the command source
  is treated as dead. The lift gets one STOP and then nothing. Wheels that
  were *driving* get one zero-velocity command and then the cart goes
  **silent on CAN** for them — no keepalive, no re-send of the last (stale)
  command. That silence is deliberate: every wheel motor is armed with a
  ``can_timeout_ms`` loss-of-comms alarm at enable time (and the jelly_legs
  jog has its own 300 ms deadman), so a motor that stops hearing from the
  host torques off on its own. The motor-side timeout is the safety layer
  against a runaway; the host's job is to never feed it with anything but a
  live motion command. Once the trip has happened and the wheels have
  stopped rolling they are re-enabled straight into the park hold.
- The park hold itself is exempt from the link requirement: it is a fixed
  position anchor with no velocity in it, so a parked cart keeps holding
  whether or not a headset is attached (a cart free to roll is the hazard
  there, not a runaway). When the source returns, driving resumes from
  rest.

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
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any

from ..constants import CAN_BASE, CAN_CHEST
from ..motor import CanBus, ControlMode, MotorError, MotorStatus, make_driver
from ..motor.config import DAMIAO_TIMEOUT_MS_PER_UNIT
from ..motor.damiao import _DM_REG_PMAX, _DM_REG_TIMEOUT
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

# Minimum spacing between wheel re-enable attempts while the command source is
# live but a wheel keeps reporting a fault (dead bus, motor unpowered). Each
# attempt is several CAN round trips and blocks the command task while its
# requests time out, so it must not run every cycle.
_WHEEL_RECOVER_MIN_S = 1.0

# Spacing of attempts to anchor tripped wheels into the park hold while no
# command source is attached (each attempt polls every wheel's velocity; a
# base still rolling on a slope is retried until it settles).
_PARK_RETRY_S = 0.2

# Wheel statuses the cart clears and re-enables on its own: the two the CAN
# timeout leaves behind. Anything else (over-current, over-temperature,
# under-voltage) is a real fault that stays put until an operator looks.
_RECOVERABLE_WHEEL_STATUSES = frozenset({MotorStatus.LOST_COMM, MotorStatus.DISABLED})


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
    vx: float, vy: float, wz: float, max_speed: float, turn_scale: float
) -> list[float]:
    """Map normalized body command ([-1, 1] each) to per-wheel rad/s.

    The raw mix can exceed 1 when translation and rotation combine, so the
    whole set is scaled down together to preserve the motion direction while
    keeping every wheel within ``max_speed``.
    """
    wz *= turn_scale
    raw = [
        WHEEL_SIGNS[w.motor_id] * (w.mx * vx + w.my * vy + w.mw * wz) for w in WHEELS
    ]
    scale = max(1.0, max(abs(r) for r in raw))
    return [r / scale * max_speed for r in raw]


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
        slew:            Max change of the normalized body command per second;
                         limits accel/decel so command steps ramp the wheels.
                         The default takes 2s from rest to full deflection.
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
        frequency:       Wheel command task rate in Hz. Every cycle while the
                         command source is live sends one command frame to
                         each wheel (velocity, or the parked hold), which is
                         what keeps the wheels' ``can_timeout_ms`` alarm fed.
        command_timeout: Seconds without a fresh :meth:`Cart.set_command`
                         before the command source is considered dead. On
                         that edge the lift gets a STOP and, if the wheels
                         were driving, they get a single zero-velocity
                         command and then **nothing** on CAN — no keepalive,
                         no re-send of the stale command — so the motors' own
                         timeouts (``can_timeout_ms``; the lift's 300 ms jog
                         deadman) trip and torque off, and a wedged host can
                         never keep the base driving. Tripped wheels are then
                         re-enabled into the park hold once they have stopped.
                         A cart that was already parked keeps its hold. Must
                         comfortably exceed the source's frame period (a
                         headset streams at 72-90 Hz, with occasional 50-100
                         ms WiFi gaps).
        can_timeout_ms:  Loss-of-comms alarm (Damiao ``TIMEOUT`` register)
                         written to every wheel motor at enable time and read
                         back to verify (RAM only, re-applied each session).
                         A wheel that goes this long without a command frame
                         faults ``LOST_COMM`` and torques off by itself — the
                         safety layer that does not depend on this process
                         running. Must be > 0 and well above one command
                         period (``1000 / frequency`` ms); enable refuses
                         otherwise. When the source comes back the cart
                         clears the fault and re-enables the wheels.
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
    slew: float = 0.5
    axis_snap_deg: float = 15.0
    imu: bool = False
    yaw_hold_gain: float = 2.0
    yaw_hold_max: float = 0.3
    yaw_log: bool = False
    deadzone: float = 0.15
    hold_kp: float = 60.0
    hold_kd: float = 1.5
    frequency: float = 50.0
    command_timeout: float = 0.2
    can_timeout_ms: float = 200.0
    lift: bool = True
    lift_channel: str = CAN_CHEST
    lift_speed: int = JOG_SPEED


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

    The command source must keep calling :meth:`set_command` (at least every
    ``CartConfig.command_timeout``) for the cart to send any *motion*: with
    no live source the wheels are never given a velocity, and wheels that
    were driving when the source vanished are left to their ``can_timeout_ms``
    loss-of-comms alarm, which torques them off. The stationary park hold is
    the one thing sent without a source — it anchors the base in place and
    carries no velocity, so it is kept (or re-established once tripped wheels
    have stopped) until the source returns.
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
        # The last VR frame object mapped by apply_vr_frame. A frame that is
        # handed over twice (a poller re-reading the server's latest frame,
        # say) must not count as a fresh command.
        self._last_vr_frame: object | None = None

        # Latest external yaw-rate sample (rad/s CCW, monotonic timestamp),
        # written from any thread; None until a sensor feeds one. The counter
        # lets the command loop report the sensor's delivered rate, which is
        # what distinguishes a slow source from a dead one.
        self._yaw_rate: tuple[float, float] | None = None
        self._yaw_samples = 0

        # Introspection for status displays (updated by the command task).
        self.body_cmd: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.wheel_speeds: list[float] = [0.0] * len(WHEELS)
        self.yaw_correction: float = 0.0
        self.lift_dir: int = STOP
        self.parked: bool = False
        self.park_failed: bool = False
        self.send_failed: bool = False
        # True while the command source is fresh (commands are streaming to
        # the wheels/lift); False means the cart is silent on CAN.
        self.linked: bool = False
        # Wheels currently reporting a fault the cart could not clear.
        self.wheel_faults: dict[str, MotorStatus] = {}

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
        if cfg.channel is not None:
            # The motor-side timeout is the safety layer; refuse a config that
            # disables it or that the command stream cannot reliably feed.
            period_ms = 1000.0 / cfg.frequency
            if not math.isfinite(cfg.can_timeout_ms) or cfg.can_timeout_ms <= 0.0:
                raise ValueError(
                    "cart can_timeout_ms must be a positive number — the wheel "
                    "loss-of-comms alarm is the runaway safety layer and cannot "
                    "be disabled"
                )
            if cfg.can_timeout_ms < 2.0 * period_ms:
                raise ValueError(
                    f"cart can_timeout_ms ({cfg.can_timeout_ms:g} ms) must be at "
                    f"least twice the command period ({period_ms:g} ms at "
                    f"{cfg.frequency:g} Hz), or a single late cycle trips the wheels"
                )
        if not math.isfinite(cfg.command_timeout) or cfg.command_timeout <= 0.0:
            raise ValueError("cart command_timeout must be a positive number")
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
                # Arm the loss-of-comms alarm first (RAM only, so it is
                # re-applied every session and never depends on flash state),
                # and prove every wheel took it before any torque is applied.
                await self._arm_can_timeout()
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
                _logger.info(
                    "cart wheels enabled on %s (loss-of-comms alarm %.0f ms: the "
                    "wheels torque off on their own when the command stream stops)",
                    cfg.channel,
                    cfg.can_timeout_ms,
                )
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

    async def _arm_can_timeout(self) -> None:
        """Write the loss-of-comms alarm to every wheel and verify the readback.

        RAM-only on purpose: the value is re-asserted on every enable, so the
        safety layer never depends on what a motor happens to have in flash.
        A wheel that does not read back the requested value fails the enable
        — driving with an unverified timeout is exactly the runaway case.
        """
        ticks = round(self._config.can_timeout_ms / DAMIAO_TIMEOUT_MS_PER_UNIT)
        await _gather_all_or_raise(
            *[m._write_register(_DM_REG_TIMEOUT, ticks) for m in self._motors]
        )
        readback = await asyncio.gather(
            *[m._read_register(_DM_REG_TIMEOUT) for m in self._motors]
        )
        mismatched = [
            f"{w.name}={int(value) * DAMIAO_TIMEOUT_MS_PER_UNIT:g}ms"
            for w, value in zip(WHEELS, readback)
            if int(value) != ticks
        ]
        if mismatched:
            raise MotorError(
                f"cart wheel CAN timeout readback mismatch (wanted "
                f"{self._config.can_timeout_ms:g} ms): {', '.join(mismatched)} — "
                "refusing to drive without the loss-of-comms safety layer"
            )

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
                # A wheel whose CAN timeout tripped while the source was away
                # sits in LOST_COMM; the torque-off below is only confirmed
                # from a clean DISABLED status, so clear the fault first.
                await asyncio.gather(*[m.clear_errors() for m in self._motors])
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

        Safe to call from any thread at any rate — but it must be called
        *continuously* while the source is alive, including while it is
        commanding zero. The command task consumes the latest value and
        streams it to the wheels; if no fresh command arrives within
        ``CartConfig.command_timeout`` the source is treated as dead: driving
        wheels are stopped once and then left to their CAN timeout, parked
        wheels keep their hold (see :class:`Cart`). Re-sending an old command from a
        source that has actually lost its input is a bug: that is what keeps
        the wheels' loss-of-comms alarm from tripping.
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
        command task, so a dead frame stream times out to a stop and then to
        CAN silence. Only a *new* frame counts as a live source: call this
        from the server's on-frame callback, once per received frame. The
        same frame object handed over again (a poller re-reading the latest
        frame after the headset went quiet) is ignored rather than refreshing
        the deadline — that would be a stale command dressed up as a fresh one.
        """
        if frame is self._last_vr_frame:
            return
        self._last_vr_frame = frame
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

    async def _park(self, *, reenable: bool = False) -> list[float] | None:
        """Switch the wheels to the MIT position hold at their current positions.

        Returns the per-wheel anchor positions, or None if parking is not
        currently safe:

        - a wheel is still measurably moving (coasting past the ramped-down
          command, e.g. on a slope) — retried next cycle once it settles, or
        - a wheel reports a position too close to the widened ±PMAX mapping
          limit, where a wrapped/clamped anchor would mean a phantom position
          error at full torque (sets :attr:`park_failed`; not retried).

        With ``reenable`` the wheels are expected to be torqued off (their
        CAN timeout tripped after the command source went away) and are
        brought back for the hold: the mode is switched to IMPEDANCE *first*
        — a mode switch zeroes the motor's command state, and MIT with zero
        gains is torque-free — and only then re-enabled, so no velocity target
        can be replayed by the enable. The hold frames that follow are the
        only thing the wheels then act on.
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
        if reenable:
            await _gather_all_or_raise(*[m.enable() for m in self._motors])
        return list(positions)

    async def _send_hold(self, hold_pos: list[float]) -> None:
        """Send one cycle of the MIT park hold to every wheel."""
        cfg = self._config
        await asyncio.gather(
            *[
                m.set_impedance(p, 0.0, cfg.hold_kp, cfg.hold_kd, 0.0)
                for m, p in zip(self._motors, hold_pos)
            ]
        )

    async def _unpark(self) -> None:
        """Return parked wheels to VELOCITY mode (clears the motors' command state)."""
        await asyncio.gather(
            *[m.set_control_mode(ControlMode.VELOCITY) for m in self._motors]
        )

    async def _stop_wheels_once(self) -> None:
        """Send the single zero-velocity command that precedes CAN silence.

        Failures are logged, not retried: the point of what follows is that
        the wheels stop on their own without another frame from us.
        """
        if not self._motors:
            return
        try:
            await asyncio.gather(*[m.set_velocity(0.0) for m in self._motors])
            self.send_failed = False
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the motor timeout covers a lost stop
            self.send_failed = True
            _logger.warning(
                "cart: stop command on source loss failed to send — relying on "
                "the wheels' %.0f ms loss-of-comms alarm",
                self._config.can_timeout_ms,
            )

    async def _recover_wheels(self) -> bool:
        """Clear and re-enable wheels the CAN timeout (or a torque-off) left
        faulted, so a returning command source can drive again.

        Asks each wheel for its status and re-runs the enable sequence (clear
        errors, enable, VELOCITY mode) on those reporting ``LOST_COMM`` or
        ``DISABLED``, then reads the status back so the cached feedback
        reflects the re-enabled state. Other faults are reported in
        :attr:`wheel_faults` and left alone — an over-current or thermal trip
        is not ours to clear blindly. Returns True when every wheel is
        enabled and healthy.
        """
        faults: dict[str, MotorStatus]
        try:
            statuses = await asyncio.gather(*[m.get_error_code() for m in self._motors])
            tripped = [
                (w, m)
                for w, m, status in zip(WHEELS, self._motors, statuses)
                if status in _RECOVERABLE_WHEEL_STATUSES
            ]
            if tripped:
                _logger.info(
                    "cart: re-enabling wheels after loss-of-comms trip: %s",
                    ", ".join(w.name for w, _ in tripped),
                )
                await _gather_all_or_raise(*[m.enable() for _, m in tripped])
                await _gather_all_or_raise(
                    *[m.set_control_mode(ControlMode.VELOCITY) for _, m in tripped]
                )
                statuses = await asyncio.gather(
                    *[m.get_error_code() for m in self._motors]
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - dead bus / unpowered wheels
            faults = {w.name: MotorStatus.UNKNOWN for w in WHEELS}
            if faults != self.wheel_faults:
                _logger.warning(
                    "cart: wheels not answering or refusing to re-enable (%s) — "
                    "the base stays torqued off until they do",
                    exc,
                )
            self.wheel_faults = faults
            return False

        faults = {
            w.name: status
            for w, status in zip(WHEELS, statuses)
            if status is not MotorStatus.OK
        }
        if faults and faults != self.wheel_faults:
            _logger.warning(
                "cart: wheel fault(s) the cart will not clear on its own: %s",
                ", ".join(f"{name}={status.value}" for name, status in faults.items()),
            )
        self.wheel_faults = faults
        return not faults

    async def _command_loop(self) -> None:
        """Apply slew limiting, mixing, park/unpark, and lift edges at the
        configured rate — while, and only while, the command source is live.

        While driving the wheels track the slew-limited command in VELOCITY
        mode. Once the command has ramped to zero (and the wheels are measured
        slow) they are parked: held at their current positions by the motor's
        internal MIT position loop with ``hold_kp``/``hold_kd``. Holding in
        the motor's own loop (rather than an outer software loop over CAN) is
        what makes the wheel rigid instead of giving first and correcting
        after. Every cycle sends one frame per wheel (velocity or hold), which
        is also what feeds the wheels' loss-of-comms alarm.

        The live-source requirement applies to *motion*. When the latched
        command goes stale while driving (VELOCITY mode) the loop stops the
        wheels once, suspends the lift, and then sends nothing: the wheels'
        CAN timeout trips and torques them off. Once that has certainly
        happened and the wheels have stopped rolling they are re-enabled
        straight into the park hold (IMPEDANCE mode set before enable, so
        nothing but the hold can act). A stationary hold is not a runaway
        risk, so a parked cart keeps its hold whether or not a source is
        attached — the hold frames are the one thing streamed without a live
        source, and they carry no velocity. When a fresh command arrives the
        loop re-enables any wheel still tripped and ramps up from rest.
        """
        cfg = self._config
        interval = 1.0 / cfg.frequency
        max_delta = cfg.slew * interval
        cmd = [0.0, 0.0, 0.0]  # slewed (vx, vy, wz), normalized [-1, 1]
        hold_pos: list[float] | None = None  # per-wheel park anchors (rad)
        yaw_err = 0.0  # integrated heading error (rad) since the stroke start
        yaw_bias = 0.0  # gyro bias estimate (rad/s), learned while stopped
        yaw_log = _YawLog() if cfg.yaw_log else None
        warned_silent = False
        t_loop0 = time.monotonic()
        linked = False  # the source was fresh on the previous cycle
        next_recover = 0.0  # earliest time for the next wheel re-enable attempt
        wheels_ok = not self._motors  # every wheel enabled and healthy
        # The trip is certain this long after the last velocity frame; then the
        # wheels are torque-off and merely need to stop rolling before being
        # anchored into the park hold.
        trip_settle = cfg.can_timeout_ms / 1e3 + 2.0 * interval
        # Unlinked and not holding: when the last velocity frame went out, and
        # when to next try anchoring. Freshly enabled wheels have never been
        # given a velocity, so with no source yet they are anchored on the very
        # first cycle rather than left to trip first.
        silent_since = t_loop0 - trip_settle
        next_park = 0.0

        while True:
            t_iter = time.perf_counter()
            now = time.monotonic()

            fresh = now - self._target_time <= cfg.command_timeout
            if not fresh:
                if linked:
                    # The source died (teleop thread gone, headset stream
                    # dropped, wedged poller). The lift is released either way.
                    # Driving wheels get one stop, then silence: from here the
                    # CAN timeout is what stops them — never a re-sent stale
                    # command. Parked wheels keep their hold: it is an anchor,
                    # not a motion, and dropping it would only free the base.
                    linked = False
                    if self._lift is not None:
                        self._lift.suspend()
                    if hold_pos is None:
                        _logger.info(
                            "cart: command source silent for %.0f ms while "
                            "driving — stopping and going quiet on CAN (wheels "
                            "torque off after %.0f ms, then re-anchor)",
                            cfg.command_timeout * 1e3,
                            cfg.can_timeout_ms,
                        )
                        await self._stop_wheels_once()
                        silent_since = now
                        next_park = now + trip_settle
                        wheels_ok = False  # the silence is about to trip them
                        next_recover = 0.0
                    else:
                        _logger.info(
                            "cart: command source silent for %.0f ms while "
                            "parked — keeping the park hold",
                            cfg.command_timeout * 1e3,
                        )
                cmd = [0.0, 0.0, 0.0]
                yaw_err = 0.0
                self.linked = False
                self.body_cmd = (0.0, 0.0, 0.0)
                self.wheel_speeds = [0.0] * len(WHEELS)
                self.yaw_correction = 0.0
                self.lift_dir = STOP

                if self._motors:
                    try:
                        if hold_pos is not None and any(
                            m.last_status in _RECOVERABLE_WHEEL_STATUSES
                            for m in self._motors
                        ):
                            # The hold lapsed anyway (a stall of this loop past
                            # the CAN timeout): the wheels are torque-off, so
                            # treat them like a tripped drive and re-anchor.
                            _logger.warning(
                                "cart: park hold lapsed (wheel loss-of-comms) — "
                                "re-anchoring"
                            )
                            hold_pos = None
                            wheels_ok = False
                            silent_since = now
                            next_park = now + trip_settle
                        if hold_pos is not None:
                            await self._send_hold(hold_pos)
                            self.send_failed = False
                        elif (
                            cfg.hold_kp > 0.0
                            and not self.park_failed
                            and now >= next_park
                            and now - silent_since >= trip_settle
                        ):
                            next_park = now + _PARK_RETRY_S
                            hold_pos = await self._park(reenable=True)
                            if hold_pos is not None:
                                wheels_ok = True
                                self.wheel_faults = {}
                                await self._send_hold(hold_pos)
                                self.send_failed = False
                                _logger.info(
                                    "cart: wheels re-enabled into the park hold "
                                    "(no command source)"
                                )
                    except asyncio.CancelledError:
                        raise
                    except Exception:  # noqa: BLE001 - retried next window
                        self.send_failed = True
                        next_park = now + _PARK_RETRY_S
                self.parked = hold_pos is not None
                await asyncio.sleep(interval)
                continue

            if not linked:
                linked = True
                self.linked = True
                _logger.info("cart: command source live — resuming control")

            vx, vy, wz, lift_dir = self._target

            # A wheel the CAN timeout tripped (a source outage, or a stall of
            # this very loop) reports LOST_COMM on the feedback it echoes for
            # each command; re-enable it before commanding it, and ramp from
            # rest since it has stopped.
            if self._motors:
                if wheels_ok and any(
                    m.last_status in _RECOVERABLE_WHEEL_STATUSES for m in self._motors
                ):
                    wheels_ok = False
                    next_recover = 0.0
                    _logger.warning(
                        "cart: a wheel dropped out of enable mid-stream "
                        "(loss-of-comms or torque-off) — re-enabling"
                    )
                if not wheels_ok and now >= next_recover:
                    next_recover = now + _WHEEL_RECOVER_MIN_S
                    wheels_ok = await self._recover_wheels()
                    cmd = [0.0, 0.0, 0.0]
                    hold_pos = None

            # Slew the (vx, vy, wz) command as a single vector: cap the step's
            # magnitude but keep its direction. Ramping each axis at its own
            # fixed rate distorts direction — a mostly-forward command with a
            # small lateral part would finish the lateral ramp almost
            # instantly while forward is still climbing, veering the base
            # sideways before it straightens out.
            deltas = [t - c for t, c in zip((vx, vy, wz), cmd)]
            norm = math.sqrt(sum(d * d for d in deltas))
            if norm > max_delta:
                k = max_delta / norm
                deltas = [d * k for d in deltas]
            for i, d in enumerate(deltas):
                cmd[i] += d

            moving = any(abs(c) >= 1e-3 for c in cmd)
            driving = moving or any(abs(t) >= 1e-3 for t in (vx, vy, wz))

            # Heading hold on an external yaw reference. NB: deliberately
            # *not* torque feedback — a simulation study of this exact plant
            # showed the drift mechanisms (lateral slide on an unloaded
            # diagonal, radius-mismatch path curvature) are unobservable from
            # wheel torque, while a gyro heading hold fixes everything
            # fixable (see the removed diagnostics/base/floor_sim.py in git
            # history). While translating with
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
                cmd[0], cmd[1], cmd[2] + yaw_corr, cfg.max_speed, cfg.turn_scale
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
                            await self._send_hold(hold_pos)
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
