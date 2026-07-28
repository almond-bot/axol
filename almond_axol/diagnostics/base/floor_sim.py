"""Simulation study: x-drive straight-line drift on an uneven floor.

Reproduces the veer seen on the real cart when commanding pure forward
translation: the rigid four-wheel frame on a non-planar floor is statically
indeterminate (the wobbly-table problem), so the diagonal wheel pair that
carries the load — and how much traction the light pair has — changes with
position. A lightly-loaded wheel saturates its available friction during
acceleration and cruise disturbances (local slope, rolling-resistance
asymmetry) push the base around, none of which is visible in wheel velocity
feedback because every wheel tracks its commanded speed regardless. The only
signal the real hardware has is per-wheel torque, so this sim exists to
answer: what torque-feedback controller actually holds a straight line?

Physics model (body frame: +x forward, +y left, +wz CCW):

- Rigid body (m, Iz) with four 45-degree omni wheels at the corners. Wheel i
  drives along unit vector u_i = (mx_i, my_i)/sqrt(2) using the same mixing
  geometry as ``almond_axol.robot.cart.WHEELS``.
- Floor: a sum of smooth random sinusoids, mm-scale amplitude over metre
  wavelengths. Normal loads follow the plane-fit residual of the four
  contact heights through a stiff contact spring, clamped at zero (a wheel
  can fully unload); for a symmetric wheel layout the residual is the
  diagonal pattern (+d, -d, -d, +d).
- Traction: F = mu * N * tanh(slip / s0) along u_i (rollers are free
  perpendicular to u_i); rolling resistance opposes contact motion; the
  local floor gradient pulls the body downhill.
- Motors: first-order velocity tracking (tau_m); reported torque is the
  contact load torque r * F plus Gaussian sensor noise — matching what the
  Damiao feedback frames give the cart's command loop.

The control loop replicates ``Cart._command_loop`` (50 Hz, vector slew,
``mix``) with pluggable correction laws. Run with:

    uv run python -m almond_axol.diagnostics.base.floor_sim

Findings (July 2026, 8 random floors x 2 strokes of ~5 m):

- With wheels holding commanded velocity, x-drive yaw is kinematically
  pinned even when a diagonal pair fully unloads: the pair shares a drive
  direction, so any two loaded wheels still constrain wz. Ideal-wheel
  open-loop heading error is ~0. The drift that remains is a *lateral
  slide* along the unloaded pair's axis — an axis with no traction, hence
  no observability and no control authority. No wheel-command controller
  can act on it; keeping all four wheels loaded (mechanics) can.
- The yaw-weighted torque sum is ~ Iz*dwz/dt, not a steady imbalance
  signal. A torque-fed yaw integrator (deployed on the cart for a while)
  winds up on launch transients, then freezes nonzero while the wheels
  dutifully track it — holding the base in a slow constant rotation. It
  *added* ~1.4 deg heading error on ideal wheels and reproduced the
  "veers even after launch" symptom seen on hardware; removed.
- A few percent of per-wheel effective-radius mismatch (omni roller
  reality) curves the path in a floor-dependent way. Its motion-biasing
  component produces *balanced* torques — unobservable from torque
  feedback: cancelling the observable internal-fight component
  (null-trim row) does not measurably help.
- A heading hold on an external yaw reference fixes everything fixable:
  ~7x less heading error and roughly half the lateral drift, limited only
  by the uncontrollable slide. If straight-line fidelity matters, put a
  gyro (any cheap I2C IMU) on the base and close wz on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...robot.cart import WHEEL_SIGNS, WHEELS, mix

# ---------------------------------------------------------------------------
# Physics parameters
# ---------------------------------------------------------------------------


@dataclass
class Params:
    mass: float = 45.0  # kg, robot + arms + lift
    inertia_z: float = 3.0  # kg m^2
    half_length: float = 0.24  # m, wheel x offset
    half_width: float = 0.24  # m, wheel y offset
    wheel_radius: float = 0.05  # m
    mu: float = 0.7  # tyre-floor friction
    slip_scale: float = 0.03  # m/s, tanh friction-curve scale
    contact_k: float = 2.0e5  # N/m, contact spring
    rolling_resist: float = 0.012  # fraction of N
    motor_tau: float = 0.02  # s, velocity-loop tracking constant
    torque_noise: float = 0.05  # Nm std-dev on reported torque
    radius_err: float = 0.0  # per-wheel effective-radius error scale (fraction)
    floor_amp: float = 2.5e-3  # m, bump amplitude scale
    dt: float = 1.0e-3  # s, physics step
    ctrl_hz: float = 50.0  # cart command-loop rate
    max_speed: float = 20.0  # rad/s (CartConfig default)
    turn_scale: float = 1.0  # (CartConfig default)
    slew: float = 1.0  # 1/s (CartConfig default)
    g: float = 9.81


# Wheel drive directions and yaw arms from the cart's mixing geometry.
def _wheel_geometry(p: Params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (positions[4,2], drive_dirs[4,2], yaw_arms[4])."""
    pos = np.array(
        [
            [+p.half_length, +p.half_width],  # front_left
            [+p.half_length, -p.half_width],  # front_right
            [-p.half_length, +p.half_width],  # back_left
            [-p.half_length, -p.half_width],  # back_right
        ]
    )
    dirs = np.array([[w.mx, w.my] for w in WHEELS]) / math.sqrt(2.0)
    # Contact-speed yaw arm: (omega x r) . u = wz * (x*uy - y*ux)
    arms = pos[:, 0] * dirs[:, 1] - pos[:, 1] * dirs[:, 0]
    return pos, dirs, arms


class Floor:
    """Smooth random height field h(x, y): a sum of sinusoids."""

    def __init__(self, rng: np.random.Generator, amp: float, n: int = 8):
        # Wavelengths 0.6 m .. 4 m in random directions.
        wavelength = rng.uniform(0.6, 4.0, n)
        angle = rng.uniform(0.0, 2.0 * np.pi, n)
        self.kvec = (2.0 * np.pi / wavelength[:, None]) * np.stack(
            [np.cos(angle), np.sin(angle)], axis=1
        )
        self.phase = rng.uniform(0.0, 2.0 * np.pi, n)
        self.amp = rng.uniform(0.3, 1.0, n) * amp / math.sqrt(n)

    def h(self, xy: np.ndarray) -> np.ndarray:
        """Heights for points xy[..., 2]."""
        arg = xy @ self.kvec.T + self.phase
        return np.sin(arg) @ self.amp

    def grad(self, xy: np.ndarray) -> np.ndarray:
        arg = xy @ self.kvec.T + self.phase
        return (np.cos(arg) * self.amp) @ self.kvec


# ---------------------------------------------------------------------------
# Yaw-trim controllers under study
# ---------------------------------------------------------------------------


@dataclass
class PITrim:
    """PI correction on the yaw-weighted torque sum (P=0 -> deployed integrator).

    Replicates the cart's gating: adapt only while translating without a
    commanded turn; keep the integrator across stops; reset the EMA at rest.
    """

    gain_p: float = 0.0
    gain_i: float = 0.15
    clamp: float = 0.3
    ema_alpha: float = 0.2
    trim: float = field(default=0.0, init=False)
    ema: float = field(default=0.0, init=False)

    def wz_correction(
        self, torques: np.ndarray, cmd: list[float], dt: float, psi: float
    ) -> float:
        translating = math.hypot(cmd[0], cmd[1]) > 0.1
        turning = abs(cmd[2]) > 0.05
        if translating and not turning:
            yaw_sum = float(sum(w.mw * t for w, t in zip(WHEELS, torques)))
            self.ema += self.ema_alpha * (yaw_sum - self.ema)
            self.trim -= self.gain_i * self.ema * dt
            self.trim = max(-self.clamp, min(self.clamp, self.trim))
        elif not translating:
            self.ema = 0.0
        if not translating:
            return 0.0
        correction = self.trim - self.gain_p * self.ema
        return max(-self.clamp, min(self.clamp, correction))


# Null space of the force->(Fx, Fy, Mz) map for the x-drive geometry: the
# front pair shares drive direction u=(1,1)/sqrt(2)... (FL/BR and FR/BL are
# the shared-direction pairs), and the (1, 1, -1, -1) front-vs-back force
# pattern produces zero net force and zero net moment. Torque along this
# pattern is pure internal fight — wheels working against each other —
# which is exactly what mutually-inconsistent wheel speeds (effective-radius
# mismatch) pump into the system.
_NULL_PATTERN = np.array([1.0, 1.0, -1.0, -1.0])


@dataclass
class GyroHold:
    """Heading hold from an external yaw reference (what an IMU would give).

    Latches the heading when a straight translation begins and steers the
    heading error to zero. Not implementable with torque feedback — included
    to quantify what adding a gyro to the base would buy.
    """

    gain: float = 2.0  # normalized wz per rad of heading error
    clamp: float = 0.3
    noise: float = math.radians(0.2)  # heading estimate noise (std-dev)
    psi_ref: float | None = field(default=None, init=False)
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(1234), init=False
    )

    def wz_correction(
        self, torques: np.ndarray, cmd: list[float], dt: float, psi: float
    ) -> float:
        translating = math.hypot(cmd[0], cmd[1]) > 0.1
        turning = abs(cmd[2]) > 0.05
        if not translating or turning:
            self.psi_ref = None
            return 0.0
        psi_meas = psi + self._rng.normal(0.0, self.noise)
        if self.psi_ref is None:
            self.psi_ref = psi_meas
        err = psi_meas - self.psi_ref
        return max(-self.clamp, min(self.clamp, -self.gain * err))


@dataclass
class NullTrim:
    """Per-wheel multiplicative speed trim cancelling the internal-fight torque.

    Adapts a scale factor along the null pattern so the four commanded wheel
    speeds become mutually consistent — an online relative-radius
    calibration using only the torque feedback the cart already has.
    """

    gain: float = 0.02  # trim rate per Nm of fight torque
    clamp: float = 0.05  # max +-5% speed scaling
    ema_alpha: float = 0.2
    scale: float = field(default=0.0, init=False)
    ema: float = field(default=0.0, init=False)

    def wheel_scales(
        self, torques: np.ndarray, cmd: list[float], dt: float
    ) -> np.ndarray:
        moving = math.hypot(cmd[0], cmd[1]) > 0.1 or abs(cmd[2]) > 0.05
        if moving:
            fight = float(_NULL_PATTERN @ torques) / 4.0
            self.ema += self.ema_alpha * (fight - self.ema)
            self.scale -= self.gain * self.ema * dt
            self.scale = max(-self.clamp, min(self.clamp, self.scale))
        return 1.0 + self.scale * _NULL_PATTERN


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@dataclass
class StrokeResult:
    heading_deg: float  # heading change over the stroke
    lateral_cm: float  # end deviation from the stroke's straight line
    distance_m: float


def simulate(
    p: Params,
    floor: Floor,
    controller: PITrim | GyroHold | None,
    strokes: list[tuple[float, tuple[float, float, float]]],
    rng: np.random.Generator,
    wheel_ctrl: NullTrim | None = None,
) -> list[StrokeResult]:
    """Run the command profile and score each non-zero-command stroke.

    ``strokes`` is a list of (duration_s, (vx, vy, wz)) segments applied in
    order through the replicated cart command loop.
    """
    pos_w, dirs, arms = _wheel_geometry(p)
    n_sub = round(1.0 / (p.ctrl_hz * p.dt))
    ctrl_dt = 1.0 / p.ctrl_hz
    max_delta = p.slew * ctrl_dt

    # State
    xy = np.zeros(2)  # world position
    psi = 0.0  # heading
    v_b = np.zeros(2)  # body-frame velocity
    wz = 0.0
    omega = np.zeros(4)  # wheel speeds (geometry frame, rad/s)
    omega_cmd = np.zeros(4)
    cmd = [0.0, 0.0, 0.0]
    torques_rep = np.zeros(4)  # last reported torques (geometry frame)

    signs = np.array([WHEEL_SIGNS[w.motor_id] for w in WHEELS])
    results: list[StrokeResult] = []
    weight = p.mass * p.g
    # Per-wheel effective radius: omni wheels' contact radius wanders as the
    # contact point hops between rollers, so the four enforced surface speeds
    # are never exactly consistent.
    radius = p.wheel_radius * (1.0 + rng.uniform(-p.radius_err, p.radius_err, 4))

    for duration, target in strokes:
        stroke_active = any(abs(t) > 1e-6 for t in target)
        start_xy = xy.copy()
        start_dir = np.array([math.cos(psi), math.sin(psi)])
        psi_start = psi

        for _ in range(round(duration * p.ctrl_hz)):
            # ---- control tick (replicates Cart._command_loop) ----
            deltas = [t - c for t, c in zip(target, cmd)]
            norm = math.sqrt(sum(d * d for d in deltas))
            if norm > max_delta:
                deltas = [d * max_delta / norm for d in deltas]
            for i, d in enumerate(deltas):
                cmd[i] += d

            wz_corr = (
                controller.wz_correction(torques_rep, cmd, ctrl_dt, psi)
                if controller is not None
                else 0.0
            )
            speeds = mix(cmd[0], cmd[1], cmd[2] + wz_corr, p.max_speed, p.turn_scale)
            # mix() bakes in WHEEL_SIGNS (motor frame); undo them to get
            # geometry-frame wheel speeds for the physics.
            omega_cmd = np.asarray(speeds) * signs
            if wheel_ctrl is not None:
                omega_cmd = omega_cmd * wheel_ctrl.wheel_scales(
                    torques_rep, cmd, ctrl_dt
                )

            # ---- physics sub-steps ----
            for _ in range(n_sub):
                # Wheel world positions and contact heights.
                c, s = math.cos(psi), math.sin(psi)
                rot = np.array([[c, -s], [s, c]])
                wheel_xy = xy + pos_w @ rot.T
                h = floor.h(wheel_xy)

                # Wobbly-table loads: plane-fit residual is the diagonal
                # pattern for a symmetric layout.
                d_res = (h[0] - h[1] - h[2] + h[3]) / 4.0
                imbalance = p.contact_k * d_res
                light = min(weight / 4.0, abs(imbalance))
                pattern = np.array([1.0, -1.0, -1.0, 1.0]) * np.sign(imbalance)
                loads = weight / 4.0 + pattern * light
                loads = np.clip(loads, 0.0, None)
                loads *= weight / max(loads.sum(), 1e-9)

                # Contact speeds along each wheel's drive direction.
                v_contact = dirs @ v_b + arms * wz
                surf = radius * omega
                slip = surf - v_contact
                f_trac = p.mu * loads * np.tanh(slip / p.slip_scale)
                f_roll = -p.rolling_resist * loads * np.tanh(v_contact / 0.02)
                f_wheel = f_trac + f_roll

                # Downhill gravity pull from the local floor gradient.
                g_world = -p.mass * p.g * floor.grad(xy[None, :])[0]
                g_body = rot.T @ g_world

                force_b = dirs.T @ f_wheel + g_body
                moment = float(
                    np.sum(pos_w[:, 0] * (dirs[:, 1] * f_wheel))
                    - np.sum(pos_w[:, 1] * (dirs[:, 0] * f_wheel))
                )

                # Rigid-body integration (body-frame velocity states).
                acc_b = force_b / p.mass - wz * np.array([-v_b[1], v_b[0]])
                v_b += acc_b * p.dt
                wz += moment / p.inertia_z * p.dt
                psi += wz * p.dt
                xy += rot @ v_b * p.dt

                # Motor velocity loops track their commands.
                omega += (omega_cmd - omega) / p.motor_tau * p.dt

                torques_true = radius * f_wheel

            # Reported torque: geometry frame + sensor noise, sampled at 50 Hz.
            torques_rep = torques_true + rng.normal(0.0, p.torque_noise, 4)

        if stroke_active:
            disp = xy - start_xy
            lateral = float(np.cross(start_dir, disp))
            results.append(
                StrokeResult(
                    heading_deg=math.degrees(psi - psi_start),
                    lateral_cm=lateral * 100.0,
                    distance_m=float(np.linalg.norm(disp)),
                )
            )
    return results


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------


def run_study(seeds: range, make_controllers, label: str, radius_err: float) -> str:
    p = Params(radius_err=radius_err)
    strokes = [
        (4.0, (1.0, 0.0, 0.0)),  # launch + cruise ~5 m
        (1.5, (0.0, 0.0, 0.0)),  # stop
        (4.0, (1.0, 0.0, 0.0)),  # relaunch over new floor
    ]
    heading = []
    lateral = []
    lateral2 = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        floor = Floor(rng, Params.floor_amp)
        yaw_ctrl, wheel_ctrl = make_controllers()
        res = simulate(p, floor, yaw_ctrl, strokes, rng, wheel_ctrl=wheel_ctrl)
        heading.extend(abs(r.heading_deg) for r in res)
        lateral.append(abs(res[0].lateral_cm))
        lateral2.append(abs(res[1].lateral_cm))
    return (
        f"{label:26s} |heading| {np.mean(heading):6.2f} deg"
        f"  stroke1 lat {np.mean(lateral):6.1f} cm"
        f"  stroke2 lat {np.mean(lateral2):6.1f} cm"
    )


def main() -> None:
    seeds = range(8)
    print(f"{len(list(seeds))} random floors, 2 strokes of ~5 m each")

    print("\n-- ideal wheels (uneven floor only) --")
    rows = [
        ("no correction", lambda: (None, None)),
        ("deployed yaw I=0.15", lambda: (PITrim(gain_p=0.0, gain_i=0.15), None)),
        ("null-trim", lambda: (None, NullTrim())),
    ]
    for label, make in rows:
        print(run_study(seeds, make, label, radius_err=0.0))

    print("\n-- +-1.5% effective-radius mismatch (omni roller reality) --")
    rows = [
        ("no correction", lambda: (None, None)),
        ("deployed yaw I=0.15", lambda: (PITrim(gain_p=0.0, gain_i=0.15), None)),
        ("null-trim", lambda: (None, NullTrim())),
        ("null-trim + yaw I=0.15", lambda: (PITrim(gain_i=0.15), NullTrim())),
        ("gyro heading hold", lambda: (GyroHold(), None)),
    ]
    for label, make in rows:
        print(run_study(seeds, make, label, radius_err=0.015))


if __name__ == "__main__":
    main()
