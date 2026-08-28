"""Jelly: x-drive omni-wheel base + telescoping lift.

Jelly has four omni wheels mounted at 45° on the corners (an
x-drive), each driven by a Damiao motor in VELOCITY mode on a dedicated
CAN bus, plus a telescoping lift driven by the jelly_legs board on its own
chest CAN bus (see :mod:`almond_axol.robot.lift`). Wheel CAN IDs are
fixed by convention:

    id 1  front-left      id 2  front-right
    id 3  back-left       id 4  back-right

:class:`Jelly` exposes a latched command interface: any thread calls
:meth:`Jelly.set_command` with a normalized body velocity + lift direction,
and an internal asyncio task (started by :meth:`Jelly.enable`) applies slew
limiting, x-drive mixing, and the park/unpark state machine at
``JellyConfig.frequency``:

- While the command is non-zero the wheels track it in VELOCITY mode.
- When the slew-limited command reaches zero (and the wheels are measured
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
if a wheel runs backwards on Jelly, flip its entry in
:data:`WHEEL_SIGNS`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time
from dataclasses import dataclass

from ..constants import CAN_BASE, CAN_CHEST
from ..motor import CanBus, ControlMode, make_driver
from ..motor.damiao import _DM_REG_PMAX
from ..motor.driver import MotorDriver
from .lift import DOWN, JOG_SPEED, STOP, UP, Lift, LiftStatus

_logger = logging.getLogger(__name__)

# Jelly's wheels ride their own CAN interface, separate from the arm buses.
# ``axol can.setup`` names Jelly's adapter to this and includes it in the
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

# Rate of the per-cycle heading-hold trace line (JellyConfig.yaw_log). The
# hold's dynamics are ~1 s, so this resolves them without flooding a console
# the 50 Hz command loop has to keep up with.
_YAW_TRACE_HZ = 10.0

# Seconds of driving with the IMU requested but no yaw sample ever fed
# before Jelly says the heading hold is dead.
_YAW_SILENT_WARN_S = 3.0


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
class JellyConfig:
    """Configuration for Jelly.

    Attributes:
        enabled:         Whether this robot has Jelly. Only
                         consulted by entry points that support both variants
                         (``axol teleop``); code constructing a :class:`Jelly`
                         directly ignores it.
        channel:         SocketCAN interface for the wheel motors. ``None``
                         disables the wheels entirely (lift-only Jelly).
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
                         rotation, the yaw rate fed via :meth:`Jelly.feed_yaw_rate`
                         is integrated into a heading error that is steered
                         back to zero. 0 disables; a *negative* gain
                         compensates a sensor whose sign convention is
                         inverted. Idle no-op unless yaw rates are actually
                         fed (and fresh).
        yaw_hold_max:    Clamp on the heading-hold correction (normalized wz).
        yaw_log:         Trace the heading hold: a 10 Hz state line while the
                         Jelly translates and a per-stroke summary of the
                         heading it actually drifted (see :class:`_YawLog`).
                         For diagnosing drift; off in normal operation.
        deadzone:        Stick deadzone (fraction of full deflection) applied
                         by input frontends (VR thumbsticks, gamepad).
        hold_kp:         Position stiffness (Nm/rad) of the parked MIT hold;
                         0 disables parking (wheels just idle in velocity mode).
        hold_kd:         Damping (Nm·s/rad) of the parked MIT hold.
        frequency:       Wheel command task rate in Hz.
        command_timeout: Seconds without a fresh :meth:`Jelly.set_command`
                         before the target is forced to zero (and the lift
                         stopped). Protects against a dead command source.
        lift:            Whether the telescoping lift is present (the
                         jelly_legs board on the chest CAN bus, see
                         :mod:`almond_axol.robot.lift`). The chest bus being
                         down at enable time only disables the lift with a
                         warning — the buses are independent, so Jelly can
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
    command_timeout: float = 0.3
    lift: bool = True
    lift_channel: str = CAN_CHEST
    lift_speed: int = JOG_SPEED


class _YawLog:
    """Per-stroke trace of the heading hold (see ``JellyConfig.yaw_log``).

    Fed every command cycle; emits a throttled state line while Jelly is
    translating and a summary when the stroke ends.

    The number to read is the stroke's heading drift — ``yaw_err``, the
    measured rotation since the stroke began. A stroke ending near zero means
    the hold did its job and whatever drift is still visible is the lateral
    slide along the unloaded diagonal, which no wheel command can correct
    (``diagnostics/base/floor_sim.py``); a growing one means the hold isn't
    working, and the rest of the line says why — no samples, stale samples, a
    correction pinned at ``yaw_hold_max``, or a bias being integrated into the
    error while Jelly never sits still long enough to learn it.
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


class Jelly:
    """Latched-command controller for Jelly (wheels + lift).

    Typical usage::

        jelly = Jelly(JellyConfig())
        await jelly.enable()
        jelly.set_command(vx=0.5, vy=0.0, wz=0.0, lift=0)   # from any thread
        ...
        await jelly.disable()

    :meth:`set_command` only latches the target; the internal command task
    owns all bus/GPIO traffic. Values are normalized to [-1, 1] (body frame:
    +x forward, +y left, +wz CCW) and scaled by ``JellyConfig.max_speed`` /
    ``turn_scale``; ``lift`` is +1 up / 0 stop / -1 down.
    """

    def __init__(self, config: JellyConfig = JellyConfig()) -> None:
        self._config = config
        self._bus: CanBus | None = None
        self._motors: list[MotorDriver] = []
        self._lift: Lift | None = None
        self._task: asyncio.Task | None = None

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

        # Introspection for status displays (updated by the command task).
        self.body_cmd: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.wheel_speeds: list[float] = [0.0] * len(WHEELS)
        self.yaw_correction: float = 0.0
        self.lift_dir: int = STOP
        self.parked: bool = False
        self.park_failed: bool = False
        self.send_failed: bool = False

    @property
    def config(self) -> JellyConfig:
        """The configuration this Jelly controller uses (read-only)."""
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
        cfg = self._config
        if cfg.lift:
            # The chest bus is optional and independent of the wheels: a
            # missing/unpowered lift only costs the lift, never the drive.
            lift = Lift(cfg.lift_channel, cfg.lift_speed)
            try:
                await lift.start()
                self._lift = lift
            except Exception as exc:  # noqa: BLE001 - lift is best-effort
                await lift.close()
                _logger.warning(
                    "Jelly lift: could not open the chest bus %s (%s) — "
                    "the lift is disabled for this session",
                    cfg.lift_channel,
                    exc,
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
                # Widen the position-mapping range (RAM only) before enable()
                # reads it back, so multi-turn wheel positions stay valid for
                # the MIT park hold.
                await asyncio.gather(
                    *[
                        m._write_register(_DM_REG_PMAX, _SESSION_PMAX)
                        for m in self._motors
                    ]
                )
                await asyncio.gather(*[m.enable() for m in self._motors])
                for w, m in zip(WHEELS, self._motors):
                    if abs(m._p_max - _SESSION_PMAX) > 1.0:
                        _logger.warning(
                            "Jelly wheel %s PMAX readback %.0f != %.0f — parking "
                            "may misbehave",
                            w.name,
                            m._p_max,
                            _SESSION_PMAX,
                        )
                await asyncio.gather(
                    *[m.set_control_mode(ControlMode.VELOCITY) for m in self._motors]
                )
                _logger.info("Jelly wheels enabled on %s", cfg.channel)
        except BaseException:
            # A failed enable() propagates to the caller, who never calls
            # disable() — so everything opened above must be torn down here,
            # or the lift's CAN reader and jog task would keep running (and a
            # half-started wheel bus would stay open).
            self._motors = []
            if self._bus is not None:
                with contextlib.suppress(Exception):
                    await self._bus.close()
                self._bus = None
            if self._lift is not None:
                with contextlib.suppress(Exception):
                    await self._lift.close()
                self._lift = None
            raise

        if cfg.yaw_hold_gain != 0.0:
            _logger.info(
                "Jelly heading hold: gain=%.2f max=%.2f imu=%s%s",
                cfg.yaw_hold_gain,
                cfg.yaw_hold_max,
                cfg.imu,
                " (yaw_log on)" if cfg.yaw_log else "",
            )

        self._task = asyncio.create_task(self._command_loop(), name="jelly-command")

    async def disable(self) -> None:
        """Stop the command task, stop and disable the wheels, release the lift."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._motors:
            try:
                # Leave impedance park (if held) and command a stop before
                # disabling, mirroring the manual-drive teardown.
                if self.parked:
                    await self._unpark()
                await asyncio.gather(*[m.set_velocity(0.0) for m in self._motors])
            except Exception:  # noqa: BLE001 - best-effort stop before disable
                pass
            try:
                await asyncio.gather(*[m.disable() for m in self._motors])
            except Exception:  # noqa: BLE001 - keep teardown going
                _logger.exception("Jelly wheel disable failed")
            self._motors = []

        if self._bus is not None:
            await self._bus.close()
            self._bus = None

        if self._lift is not None:
            await self._lift.close()
            self._lift = None
        _logger.info("Jelly disabled")

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
        ``JellyConfig.command_timeout`` the target decays to a full stop.
        """

        def clamp(v: float) -> float:
            return max(-1.0, min(1.0, float(v)))

        vx, vy, wz = clamp(vx), clamp(vy), clamp(wz)

        # Snap near-cardinal translation onto the axis (see
        # JellyConfig.axis_snap_deg): thumbstick flicks are rarely perfectly
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

        self._target = (vx, vy, wz, int(lift))
        self._target_time = time.monotonic()

    def apply_vr_frame(self, frame, resetting: bool = False) -> None:  # noqa: ANN001
        """Map a headset :class:`~almond_axol.vr.models.VRFrame` to a command.

        The single source of truth for the VR control mapping, shared by
        plain teleop (``VRTeleop``) and data collection (``AxolVRTeleop``) so
        the two flows cannot drift apart. Stick deflection is the deadman:
        Jelly moves only while a stick is pushed past its deadzone (or a
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

    def feed_yaw_rate(self, rate: float) -> None:
        """Latch an external yaw-rate sample (rad/s, CCW positive from above).

        Fed by a gyro source (the board BMI088, see
        ``almond_axol.robot.gyro``) from any thread at any rate; the command
        task's heading hold consumes the latest sample. Samples older than
        the staleness window are ignored, so a dead sensor simply disables
        the hold rather than freezing a stale correction.
        """
        self._yaw_rate = (float(rate), time.monotonic())
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
                "Jelly wheel position near the ±PMAX mapping limit — parking "
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
        """Apply slew limiting, mixing, park/unpark, and lift edges at the
        configured rate.

        While driving the wheels track the slew-limited command in VELOCITY
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
        max_delta = cfg.slew * interval
        cmd = [0.0, 0.0, 0.0]  # slewed (vx, vy, wz), normalized [-1, 1]
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
            # Jelly is stopped so it doesn't masquerade as rotation.
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
                    "Jelly heading hold: no yaw samples after %.0fs of driving — "
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
