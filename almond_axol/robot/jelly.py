"""Jelly: x-drive omni-wheel base + telescoping lift.

Jelly has four omni wheels mounted at 45° on the corners (an
x-drive), each driven by a Damiao motor in VELOCITY mode on a dedicated
CAN bus, plus a telescoping lift driven by the jelly_legs board on its own
chest CAN bus (see :mod:`almond_axol.robot.lift`). Wheel CAN IDs are
fixed by convention:

    id 1  front-left      id 2  front-right
    id 3  back-left       id 4  back-right

:class:`Jelly` exposes a latched command interface: any thread calls
:meth:`Jelly.set_command` with a normalized body velocity + lift direction.
The Rust ``axol-rt jelly`` service applies slew limiting, x-drive mixing,
heading hold, the watchdog, and park/unpark at ``JellyConfig.frequency``:

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
import logging
import math
import os
import struct
import subprocess
import time
from dataclasses import dataclass

from ..constants import CAN_BASE, CAN_CHEST
from ..rt.link import find_binary
from .lift import DOWN, JOG_SPEED, STOP, UP, Lift, LiftStatus

_logger = logging.getLogger(__name__)

# Jelly's wheels ride their own CAN interface, separate from the arm buses.
# ``axol can.setup`` names Jelly's adapter to this and includes it in the
# @reboot bring-up alongside the arm channels.
DEFAULT_CHANNEL = CAN_BASE

# Per-wheel spin-direction calibration: flip an entry to -1 if that wheel
# drives the wrong way with everything else correct.
WHEEL_SIGNS: dict[int, float] = {1: 1.0, 2: -1.0, 3: 1.0, 4: -1.0}

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
                         One vector limit across translation and rotation
                         together. The default takes 4s from rest to full
                         deflection (halved from 2s: the faster ramp lurched
                         the base, and the arms on it, at every stick flick).
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
    slew: float = 0.25
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

    :meth:`set_command` only latches the target. The Rust service owns wheel
    control and CAN; a small Python bridge forwards targets and owns the
    separate lift driver. Values are normalized to [-1, 1] (body frame: +x
    forward, +y left, +wz CCW); ``lift`` is +1 up / 0 stop / -1 down.
    """

    def __init__(self, config: JellyConfig = JellyConfig()) -> None:
        self._config = config
        self._lift: Lift | None = None
        self._task: asyncio.Task | None = None
        self._reader_task: asyncio.Task | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._socket_path = f"/tmp/axol-jelly-{os.getpid()}-{id(self):x}.sock"
        self._ready = asyncio.Event()

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
        """Start the Rust wheel controller and the optional lift."""
        try:
            await self._enable()
        except BaseException:
            # enable() callers cannot invoke disable() through an async context
            # that never entered. Finish rollback even when enable was cancelled.
            cleanup = asyncio.create_task(self.disable(), name="jelly-startup-rollback")
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except BaseException:  # noqa: BLE001 - preserve original error
                    # A second cancellation (or other interruption) must not
                    # turn shield into fire-and-forget cleanup: the event loop
                    # may close as soon as the original error reaches caller.
                    continue
            if not cleanup.cancelled():
                cleanup_exc = cleanup.exception()
                if cleanup_exc is not None:
                    _logger.error(
                        "Jelly startup rollback failed",
                        exc_info=cleanup_exc,
                    )
            else:
                _logger.error("Jelly startup rollback was unexpectedly cancelled")
            raise

    async def _enable(self) -> None:
        """Start Jelly resources; :meth:`enable` owns rollback."""
        cfg = self._config
        self._ready.clear()
        self.send_failed = False
        if cfg.lift:
            # Publish the lift before its first await so cancellation during a
            # partial start can still find and close its bus.
            lift = Lift(cfg.lift_channel, cfg.lift_speed)
            self._lift = lift
            try:
                await lift.start()
            except Exception as exc:  # noqa: BLE001 - lift is best-effort
                # Keep it published until close succeeds; if close itself
                # fails, the outer rollback gets another chance to finish.
                await lift.close()
                self._lift = None
                _logger.warning(
                    "Jelly lift: could not open the chest bus %s (%s) — "
                    "the lift is disabled for this session",
                    cfg.lift_channel,
                    exc,
                )

        if cfg.channel is not None:
            from ..cli.can.setup import bring_up_interfaces, iface_up

            if not iface_up(cfg.channel):
                bring_up_interfaces([cfg.channel])
            self._proc = subprocess.Popen(
                [
                    find_binary(),
                    "jelly",
                    "--socket",
                    self._socket_path,
                    "--iface",
                    cfg.channel,
                ]
            )
            deadline = asyncio.get_running_loop().time() + 5.0
            while True:
                try:
                    self._reader, self._writer = await asyncio.open_unix_connection(
                        self._socket_path
                    )
                    break
                except (ConnectionRefusedError, FileNotFoundError):
                    if self._proc.poll() is not None:
                        returncode = self._proc.returncode
                        raise RuntimeError(
                            f"axol-rt Jelly core exited with {returncode}"
                        ) from None
                    if asyncio.get_running_loop().time() >= deadline:
                        raise RuntimeError("timed out connecting to axol-rt Jelly core")
                    await asyncio.sleep(0.05)
            self._reader_task = asyncio.create_task(
                self._rust_reader_loop(), name="jelly-rust-reader"
            )
            values = (
                cfg.max_speed,
                cfg.turn_scale,
                cfg.slew,
                cfg.axis_snap_deg,
                cfg.yaw_hold_gain,
                cfg.yaw_hold_max,
                cfg.hold_kp,
                cfg.hold_kd,
                cfg.frequency,
                cfg.command_timeout,
            )
            self._send_rust(b"C" + struct.pack("<10d", *values))
            await self._writer.drain()
            await asyncio.wait_for(self._ready.wait(), 10.0)
            if self.send_failed:
                raise RuntimeError("axol-rt Jelly core failed during startup")
            _logger.info("Jelly Rust wheel core enabled on %s", cfg.channel)

        if cfg.yaw_hold_gain != 0.0:
            _logger.info(
                "Jelly heading hold: gain=%.2f max=%.2f imu=%s%s",
                cfg.yaw_hold_gain,
                cfg.yaw_hold_max,
                cfg.imu,
                " (yaw_log on)" if cfg.yaw_log else "",
            )

        self._task = asyncio.create_task(self._bridge_loop(), name="jelly-rust-bridge")

    async def disable(self) -> None:
        """Stop the command task, stop and disable the wheels, release the lift."""
        # Do this synchronously before any wheel/core await. Lift._run owns the
        # CAN writes and will observe the latch independently of wheel teardown.
        if self._lift is not None:
            self._lift.command(STOP)
        self.lift_dir = STOP

        if self._task is not None:
            task = self._task
            self._task = None
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001 - bridge may have lost IPC
                _logger.warning("Jelly bridge stopped with an error: %s", exc)

        if self._writer is not None:
            try:
                self._send_rust(b"Q")
                await asyncio.wait_for(self._writer.drain(), 1.0)
            except (ConnectionError, RuntimeError, TimeoutError):
                pass
            self._writer.close()
            try:
                await asyncio.wait_for(self._writer.wait_closed(), 1.0)
            except (ConnectionError, TimeoutError):
                pass
            self._writer = None
        if self._reader_task is not None:
            try:
                await asyncio.wait_for(self._reader_task, 2.0)
            except TimeoutError:
                self._reader_task.cancel()
                try:
                    await self._reader_task
                except asyncio.CancelledError:
                    pass
            except Exception as exc:  # noqa: BLE001 - keep teardown going
                _logger.warning("Jelly reader stopped with an error: %s", exc)
            self._reader_task = None
        self._reader = None
        if self._proc is not None:
            if self._proc.poll() is None:
                try:
                    await asyncio.to_thread(self._proc.wait, 3.0)
                except subprocess.TimeoutExpired:
                    self._proc.terminate()
                    try:
                        await asyncio.to_thread(self._proc.wait, 2.0)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
                        await asyncio.to_thread(self._proc.wait)
            self._proc = None
        try:
            os.unlink(self._socket_path)
        except OSError:
            pass

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
    # Rust bridge (IPC only; control timing and CAN stay in Rust)
    # ------------------------------------------------------------------

    def _send_rust(self, payload: bytes) -> None:
        if self._writer is None or self._writer.is_closing():
            raise RuntimeError("Jelly Rust core is not connected")
        self._writer.write(struct.pack("<I", len(payload)) + payload)

    async def _rust_reader_loop(self) -> None:
        assert self._reader is not None
        yaw_log = _YawLog() if self._config.yaw_log else None
        try:
            while True:
                (size,) = struct.unpack("<I", await self._reader.readexactly(4))
                payload = await self._reader.readexactly(size)
                if payload == b"R":
                    self._ready.set()
                    continue
                if payload[:1] == b"E":
                    self.send_failed = True
                    _logger.error(
                        "Jelly Rust core: %s",
                        payload[1:].decode(errors="replace"),
                    )
                    self._ready.set()
                    continue
                if payload[:1] != b"U" or len(payload) != 82:
                    _logger.warning("Jelly Rust core sent an invalid status packet")
                    continue
                *values, flags = struct.unpack("<10dB", payload[1:])
                self.body_cmd = tuple(values[:3])
                self.wheel_speeds = list(values[3:7])
                self.yaw_correction = values[7]
                self.parked = bool(flags & 1)
                self.park_failed = bool(flags & 2)
                self.send_failed = bool(flags & 4)
                if yaw_log is not None:
                    sample = self._yaw_rate
                    rate = sample[0] if sample is not None else None
                    age = time.monotonic() - sample[1] if sample is not None else None
                    translating = math.hypot(*self.body_cmd[:2]) > 0.1
                    held = (
                        translating
                        and abs(self.body_cmd[2]) <= 0.05
                        and age is not None
                        and age <= 0.3
                    )
                    yaw_log.update(
                        now=time.monotonic(),
                        translating=translating,
                        held=held,
                        rate=rate,
                        bias=values[9],
                        err=values[8],
                        corr=values[7],
                        saturated=abs(values[7]) >= self._config.yaw_hold_max,
                        age=age,
                        samples=self._yaw_samples,
                    )
        except (asyncio.IncompleteReadError, ConnectionResetError):
            if not self._ready.is_set():
                self.send_failed = True
            self._ready.set()
        except asyncio.CancelledError:
            raise

    async def _bridge_loop(self) -> None:
        interval = 1.0 / self._config.frequency
        warned_silent = False
        started = time.monotonic()
        try:
            while True:
                now = time.monotonic()
                vx, vy, wz, lift_dir = self._target
                age = max(0.0, now - self._target_time)
                if age > self._config.command_timeout:
                    lift_dir = STOP
                if self._lift is not None:
                    self._lift.command(lift_dir)
                self.lift_dir = lift_dir if self._lift is not None else STOP

                if self._writer is not None:
                    self._send_rust(b"T" + struct.pack("<4d", vx, vy, wz, age))
                    if self._yaw_rate is not None:
                        rate, ts = self._yaw_rate
                        self._send_rust(
                            b"Y" + struct.pack("<2d", rate, max(0.0, now - ts))
                        )
                    # A wedged wheel socket must not wedge the independent lift
                    # deadman past the target's own expiry. A normal drain is
                    # much faster than one command interval; the 1 ms floor
                    # only keeps wait_for's timeout strictly positive when the
                    # target is already stale.
                    drain_timeout = max(
                        1e-3,
                        min(
                            interval,
                            self._config.command_timeout - age,
                        ),
                    )
                    await asyncio.wait_for(self._writer.drain(), drain_timeout)
                if (
                    not warned_silent
                    and self._config.imu
                    and self._config.yaw_hold_gain != 0.0
                    and any(abs(v) >= 1e-3 for v in (vx, vy, wz))
                    and self._yaw_rate is None
                    and now - started > _YAW_SILENT_WARN_S
                ):
                    warned_silent = True
                    _logger.warning(
                        "Jelly heading hold has no yaw samples after %.0fs",
                        _YAW_SILENT_WARN_S,
                    )
                await asyncio.sleep(interval)
        finally:
            # Lift._run otherwise keeps retransmitting its last jog forever,
            # preventing the firmware deadman from expiring when wheel IPC dies.
            if self._lift is not None:
                self._lift.command(STOP)
            self.lift_dir = STOP
