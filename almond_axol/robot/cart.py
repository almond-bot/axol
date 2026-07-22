"""Powered Axol Cart: x-drive omni wheel base + Jiecang telescoping lift.

The powered cart has four omni wheels mounted at 45° on the corners (an
x-drive), each driven by a Damiao motor in VELOCITY mode on a dedicated
CAN bus, plus a telescoping lift (a Jiecang JCB35N2 box driving two desk
legs) commanded through handset-emulating GPIOs (see
:mod:`almond_axol.robot.lift`). CAN IDs are fixed by convention:

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
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from ..motor import CanBus, ControlMode, make_driver
from ..motor.damiao import _DM_REG_PMAX
from ..motor.driver import MotorDriver
from .lift import DOWN, STOP, UP, JiecangLift

_logger = logging.getLogger(__name__)

# The cart's wheels ride their own CAN interface, separate from the arm buses.
DEFAULT_CHANNEL = "can_alm_axol_base"

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
        deadzone:        Stick deadzone (fraction of full deflection) applied
                         by input frontends (VR thumbsticks, gamepad).
        hold_kp:         Position stiffness (Nm/rad) of the parked MIT hold;
                         0 disables parking (wheels just idle in velocity mode).
        hold_kd:         Damping (Nm·s/rad) of the parked MIT hold.
        frequency:       Wheel command task rate in Hz.
        command_timeout: Seconds without a fresh :meth:`Cart.set_command`
                         before the target is forced to zero (and the lift
                         stopped). Protects against a dead command source.
        lift:            Whether the telescoping lift is present.
        lift_chip:       gpiochip device for the lift button lines.
        lift_up_gpio:    GPIO line offset wired to lift RJ45 pin 7 (HS1, up).
        lift_down_gpio:  GPIO line offset wired to lift RJ45 pin 8 (HS0, down).
    """

    enabled: bool = False
    channel: str | None = DEFAULT_CHANNEL
    max_speed: float = 10.0
    turn_scale: float = 0.5
    slew: float = 2.0
    deadzone: float = 0.15
    hold_kp: float = 60.0
    hold_kd: float = 1.5
    frequency: float = 50.0
    command_timeout: float = 0.3
    lift: bool = True
    lift_chip: str = "/dev/gpiochip0"
    lift_up_gpio: int = 23
    lift_down_gpio: int = 24


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
        self._lift: JiecangLift | None = None
        self._task: asyncio.Task | None = None

        # Latched target, written from any thread (single-reference swap is
        # atomic under the GIL), consumed by the command task.
        self._target: tuple[float, float, float, int] = (0.0, 0.0, 0.0, STOP)
        self._target_time: float = 0.0

        # Introspection for status displays (updated by the command task).
        self.body_cmd: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.wheel_speeds: list[float] = [0.0] * len(WHEELS)
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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def enable(self) -> None:
        """Open the CAN bus, enable the wheel motors, init the lift GPIOs,
        and start the command task."""
        cfg = self._config
        if cfg.lift:
            self._lift = JiecangLift(cfg.lift_chip, cfg.lift_up_gpio, cfg.lift_down_gpio)
            _logger.info(
                "cart lift: %s up=GPIO%d down=GPIO%d",
                cfg.lift_chip,
                cfg.lift_up_gpio,
                cfg.lift_down_gpio,
            )

        if cfg.channel is not None:
            self._bus = CanBus(cfg.channel)
            await self._bus.start()
            # Wheel IDs 1-4 collide with the arm-bus MyActuator IDs in the
            # driver-inference table, so the Damiao protocol is forced.
            self._motors = [
                make_driver(self._bus, w.motor_id, motor_type="damiao")
                for w in WHEELS
            ]
            # Widen the position-mapping range (RAM only) before enable()
            # reads it back, so multi-turn wheel positions stay valid for the
            # MIT park hold.
            await asyncio.gather(
                *[m._write_register(_DM_REG_PMAX, _SESSION_PMAX) for m in self._motors]
            )
            await asyncio.gather(*[m.enable() for m in self._motors])
            for w, m in zip(WHEELS, self._motors):
                if abs(m._p_max - _SESSION_PMAX) > 1.0:
                    _logger.warning(
                        "cart wheel %s PMAX readback %.0f != %.0f — parking "
                        "may misbehave",
                        w.name,
                        m._p_max,
                        _SESSION_PMAX,
                    )
            await asyncio.gather(
                *[m.set_control_mode(ControlMode.VELOCITY) for m in self._motors]
            )
            _logger.info("cart wheels enabled on %s", cfg.channel)

        self._task = asyncio.create_task(self._command_loop(), name="cart-command")

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
                _logger.exception("cart wheel disable failed")
            self._motors = []

        if self._bus is not None:
            await self._bus.close()
            self._bus = None

        if self._lift is not None:
            self._lift.close()
            self._lift = None
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
        def clamp(v: float) -> float:
            return max(-1.0, min(1.0, float(v)))

        self._target = (clamp(vx), clamp(vy), clamp(wz), int(lift))
        self._target_time = time.monotonic()

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

        while True:
            t_iter = time.perf_counter()

            vx, vy, wz, lift_dir = self._target
            # A dead command source (teleop thread gone, headset stream
            # dropped) must not leave the base driving: decay to a stop.
            if time.monotonic() - self._target_time > cfg.command_timeout:
                vx, vy, wz, lift_dir = 0.0, 0.0, 0.0, STOP

            for i, target in enumerate((vx, vy, wz)):
                delta = target - cmd[i]
                cmd[i] += max(-max_delta, min(max_delta, delta))

            speeds = mix(cmd[0], cmd[1], cmd[2], cfg.max_speed, cfg.turn_scale)
            moving = any(abs(c) >= 1e-3 for c in cmd)
            driving = moving or any(abs(t) >= 1e-3 for t in (vx, vy, wz))

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
                            *[
                                m.set_velocity(s)
                                for m, s in zip(self._motors, speeds)
                            ]
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
