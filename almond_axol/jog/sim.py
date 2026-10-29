"""Simulated Jelly lift and base for jogging without hardware.

Both models reproduce what the real controllers do with the same command
stream, so a jog in sim feels like a jog on the robot:

- :class:`LiftSim` is the jelly_legs board: a held jog runs the legs at a
  fixed speed and a release (or a lost command) brings them to rest on a
  constant deceleration. The legs slow down into either end of travel.
- :class:`BaseSim` is the Rust ``axol-rt jelly`` core: the same axis snap,
  the same vector slew limit on the normalized command, the same x-drive mix
  and per-wheel speed cap, and the same stop when the command source goes
  quiet. Body speed follows from the wheels' radius and placement in the
  Jelly URDF.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..robot.jelly import WHEEL_SIGNS, WHEELS, JellyConfig
from ..robot.lift import JOG_SPEED

# Seconds the jelly_legs firmware takes to run the full travel at full jog
# speed (``JOG_SPEED`` counts/s): 610 mm at 40 mm/s.
LIFT_FULL_TRAVEL_S = 15.25
# Seconds the legs take to reach full jog speed from rest, and to stop from
# it. The firmware ramps both ways at the same rate.
LIFT_RAMP_S = 0.35

# Jelly wheel geometry, from axol_jelly.urdf: each omni wheel's radius and its
# distance from the base centre (the wheels sit at (±0.2158, ±0.2158) and roll
# tangentially, 45° to the body axes).
WHEEL_RADIUS_M = 0.06335
WHEEL_CENTRE_DISTANCE_M = math.hypot(0.215834, 0.215832)


@dataclass
class LiftSim:
    """Kinematic model of the telescoping lift.

    ``raised`` is the height as a fraction of travel (1 = fully raised, the
    same convention as :attr:`almond_axol.robot.jelly.Jelly.lift_raised`);
    ``velocity`` is in fractions of travel per second, positive up.
    """

    raised: float = 1.0
    velocity: float = 0.0
    jog_speed: int = JOG_SPEED

    @property
    def max_speed(self) -> float:
        """Held-jog speed in fractions of travel per second."""
        return (self.jog_speed / JOG_SPEED) / LIFT_FULL_TRAVEL_S

    @property
    def accel(self) -> float:
        """Acceleration and deceleration in fractions of travel per s²."""
        return (1.0 / LIFT_FULL_TRAVEL_S) / LIFT_RAMP_S

    @property
    def moving(self) -> bool:
        return abs(self.velocity) > 1e-6

    def step(self, direction: int, dt: float) -> None:
        """Advance ``dt`` seconds with the jog held ``direction`` (+1/0/-1)."""
        if dt <= 0.0:
            return
        accel = self.accel
        target = float(max(-1, min(1, direction))) * self.max_speed
        # Brake into the ends of travel: never faster than the speed that
        # can still stop at the end on the firmware's deceleration.
        room_up = max(0.0, 1.0 - self.raised)
        room_down = max(0.0, self.raised)
        target = min(target, math.sqrt(2.0 * accel * room_up))
        target = max(target, -math.sqrt(2.0 * accel * room_down))
        change = target - self.velocity
        limit = accel * dt
        velocity = self.velocity + max(-limit, min(limit, change))
        # Trapezoidal integration keeps the travel exact across a ramp.
        raised = self.raised + 0.5 * (self.velocity + velocity) * dt
        if raised >= 1.0 or raised <= 0.0:
            raised = min(1.0, max(0.0, raised))
            velocity = 0.0
        self.raised = raised
        self.velocity = velocity if abs(velocity) > 1e-9 else 0.0


def _snap(vx: float, vy: float, degrees: float) -> tuple[float, float]:
    """Snap a translation within ``degrees`` of an axis onto it (``jelly.rs``)."""
    if degrees > 0.0 and (vx != 0.0 or vy != 0.0):
        heading = math.atan2(vy, vx)
        quarter = math.pi / 2.0
        near = round(heading / quarter) * quarter
        if abs(heading - near) <= math.radians(degrees):
            magnitude = math.hypot(vx, vy)
            vx = magnitude * math.cos(near)
            vy = magnitude * math.sin(near)
    return vx, vy


def body_velocity(
    vx: float, vy: float, wz: float, cfg: JellyConfig
) -> tuple[float, float, float]:
    """Body-frame (m/s, m/s, rad/s) the wheels produce for a normalized command.

    The mix runs every wheel at ``max_speed`` for a unit command along one
    axis. Each wheel rolls at 45° to the body axes, so a rim speed ``rim``
    moves the base at ``√2 · rim``; for a turn the wheels roll tangentially,
    so the base spins at ``rim / distance``. The mix scales the whole command
    down together when a wheel would exceed ``max_speed``, and the body
    velocity scales with it.
    """
    w = wz * cfg.turn_scale
    raw = [WHEEL_SIGNS[h.motor_id] * (h.mx * vx + h.my * vy + h.mw * w) for h in WHEELS]
    scale = max(1.0, max(abs(r) for r in raw))
    rim = cfg.max_speed * WHEEL_RADIUS_M
    linear = rim * math.sqrt(2.0)
    angular = rim / WHEEL_CENTRE_DISTANCE_M
    return (
        vx / scale * linear,
        vy / scale * linear,
        w / scale * angular,
    )


@dataclass
class BaseSim:
    """Kinematic model of the Jelly base in the floor plane.

    ``x``, ``y`` and ``yaw`` are the base pose in the floor frame (metres,
    radians, the URDF's ``base_x`` / ``base_y`` / ``base_yaw``). ``command``
    is the slew-limited normalized command the wheels are running.
    """

    config: JellyConfig
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    command: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def moving(self) -> bool:
        return any(abs(v) >= 1e-3 for v in self.command)

    def velocity(self) -> tuple[float, float, float]:
        """Body-frame velocity (m/s, m/s, rad/s) for the current command."""
        return body_velocity(*self.command, self.config)

    def step(self, target: tuple[float, float, float] | None, dt: float) -> None:
        """Advance ``dt`` seconds toward ``target``.

        ``None`` is a lost command source: like the core, the wheels get a
        zero-velocity frame at once instead of ramping down.
        """
        if dt <= 0.0:
            return
        if target is None:
            self.command = (0.0, 0.0, 0.0)
            return
        tx, ty = _snap(target[0], target[1], self.config.axis_snap_deg)
        goal = (tx, ty, target[2])
        delta = [g - c for g, c in zip(goal, self.command)]
        norm = math.sqrt(sum(d * d for d in delta))
        limit = self.config.slew * dt
        k = limit / norm if norm > limit and norm > 0.0 else 1.0
        previous = self.velocity()
        self.command = tuple(c + d * k for c, d in zip(self.command, delta))  # type: ignore[assignment]
        current = self.velocity()
        # Integrate the average body velocity over the tick in the floor frame.
        bx, by, bw = ((a + b) / 2.0 for a, b in zip(previous, current))
        heading = self.yaw + bw * dt / 2.0
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        self.x += (bx * cos_h - by * sin_h) * dt
        self.y += (bx * sin_h + by * cos_h) * dt
        self.yaw = math.remainder(self.yaw + bw * dt, 2.0 * math.pi)
