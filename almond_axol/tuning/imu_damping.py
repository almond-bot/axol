"""Damp the tool's vertical shake from the wrist IMU.

After learning had shoulder_1 and the elbow tracking to within 10 mdeg,
60% of slow_osc's 1-3 Hz wrist shake and 80% of its 3-15 Hz shake were
motion the joint encoders do not see (2026-09-28: compliance downstream of
the motor encoders — gearboxes, links, the mount). No encoder-based law can
remove it; the wrist camera's IMU is the one sensor that measures it.

:class:`VerticalVelocity` turns the camera's accelerometer into the tool's
vertical velocity in the shake band, causally: the vertical is the
accelerometer's own slow mean (gravity), the vertical acceleration is the
projection on it less ``g``, and a leaky integrator, a second-order
Butterworth high-pass and a first-order low-pass keep the band — the
deliberate motion (below ~0.5 Hz) is not "shake" and must not be dragged.

:class:`EncoderVelocity` runs the tool height the joint encoders give
(FK of the measured pose) through the *same* band chain, and
:class:`TipDamper` damps the difference — the **flex velocity**, the tool
motion downstream of the encoders. Not the IMU velocity itself: on slow_osc
(2026-09-28) the IMU's band velocity was 19.5 mm/s RMS, of which 13.4 mm/s
the encoders see too (coherence 0.75 — the joint loops' business) and
11 mm/s is the commanded motion leaking through the 1 Hz edge; damping that
would drag the deliberate motion. The first hardware trial damped the raw
IMU velocity and tripped its runaway guard on the motion alone.

The damper applies ``F = −c·v_flex`` at the tool through the arm's joints,
``τ = J_zᵀ F`` — ``J_z`` the tool height's Jacobian at the *measured* pose —
on the chosen joints, clamped, ramped in over a second, and dropped to zero
when the IMU goes quiet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

G = 9.80665


def _butter_hp2(fc: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Second-order Butterworth high-pass biquad (bilinear, prewarped)."""
    k = math.tan(math.pi * fc / fs)
    q = 1 / math.sqrt(2)
    norm = 1 / (1 + k / q + k * k)
    b = np.array([1.0, -2.0, 1.0]) * norm
    a = np.array([1.0, 2 * (k * k - 1) * norm, (1 - k / q + k * k) * norm])
    return b, a


@dataclass
class VerticalVelocity:
    """Band-passed vertical velocity (m/s, up positive) from accelerometer
    samples (m/s², gravity included, any camera orientation).

    ``hp_hz`` sets the low edge (the deliberate motion's leak is ~(f/hp)²
    down per octave below it), ``lp_hz`` the high edge; ``fs`` is the IMU's
    nominal rate the biquad is designed at (the ZED X One delivers ~200 Hz).
    """

    hp_hz: float = 1.0
    lp_hz: float = 15.0
    fs: float = 200.0
    up_tau_s: float = 2.0
    leak_hz: float = 0.2
    _up: np.ndarray | None = field(default=None, init=False)
    _t: float | None = field(default=None, init=False)
    _vi: float = field(default=0.0, init=False)
    _hx: list[float] = field(default_factory=lambda: [0.0, 0.0], init=False)
    _hy: list[float] = field(default_factory=lambda: [0.0, 0.0], init=False)
    _lp: float = field(default=0.0, init=False)
    value: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._b, self._a = _butter_hp2(self.hp_hz, self.fs)

    def reset(self) -> None:
        self._up, self._t, self._vi, self._lp, self.value = None, None, 0.0, 0.0, 0.0
        self._hx, self._hy = [0.0, 0.0], [0.0, 0.0]

    def update(
        self, t: float, acc: np.ndarray, gyro_deg_s: np.ndarray | None = None
    ) -> float:
        """One sample. With ``gyro_deg_s`` (the IMU's angular rate, camera
        frame) the gravity direction is carried through the camera's rotation
        between samples — a complementary filter — instead of only relaxing
        toward the accelerometer: slow_osc turns the wrist camera at up to
        ~110°/s, and a gravity estimate lagging that rotation lets horizontal
        acceleration leak into "vertical" below 1 Hz."""
        acc = np.asarray(acc, dtype=float)
        if self._up is None or self._t is None:
            self._up, self._t = acc.copy(), t
            return self.value
        dt = t - self._t
        self._t = t
        if not 0.0 < dt < 0.1:
            return self.value  # out of order or a gap: skip, keep state
        if gyro_deg_s is not None:
            w = np.radians(np.asarray(gyro_deg_s, dtype=float))
            # A fixed world vector seen from a frame rotating at ω turns at −ω.
            self._up = self._up - np.cross(w, self._up) * dt
        self._up += (acc - self._up) * min(1.0, dt / self.up_tau_s)
        g = float(np.linalg.norm(self._up)) or G
        a_v = float(acc @ self._up) / g - g
        self._vi += (a_v - 2 * math.pi * self.leak_hz * self._vi) * dt
        b, a = self._b, self._a
        y = (
            b[0] * self._vi
            + b[1] * self._hx[0]
            + b[2] * self._hx[1]
            - a[1] * self._hy[0]
            - a[2] * self._hy[1]
        )
        self._hx = [self._vi, self._hx[0]]
        self._hy = [y, self._hy[0]]
        alpha = 1 - math.exp(-2 * math.pi * self.lp_hz * dt)
        self._lp += (y - self._lp) * alpha
        self.value = self._lp
        return self.value


@dataclass
class EncoderVelocity:
    """The band velocity of a height signal (m, e.g. FK of the measured
    joints), phase-matched to :class:`VerticalVelocity`: its derivative
    through the same leak (as a first-order high-pass), high-pass biquad and
    low-pass. ``delay_s`` holds the height back by the IMU's own latency so
    the two line up in time."""

    hp_hz: float = 1.0
    lp_hz: float = 15.0
    fs: float = 240.0
    leak_hz: float = 0.2
    delay_s: float = 0.008
    _buf: list[tuple[float, float]] = field(default_factory=list, init=False)
    _t: float | None = field(default=None, init=False)
    _z: float = field(default=0.0, init=False)
    _hp1: float = field(default=0.0, init=False)
    _v_prev: float = field(default=0.0, init=False)
    _hx: list[float] = field(default_factory=lambda: [0.0, 0.0], init=False)
    _hy: list[float] = field(default_factory=lambda: [0.0, 0.0], init=False)
    _lp: float = field(default=0.0, init=False)
    value: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._b, self._a = _butter_hp2(self.hp_hz, self.fs)

    def reset(self) -> None:
        self._buf, self._t, self._z = [], None, 0.0
        self._hp1 = self._v_prev = self._lp = self.value = 0.0
        self._hx, self._hy = [0.0, 0.0], [0.0, 0.0]

    def update(self, t: float, z: float) -> float:
        self._buf.append((t, float(z)))
        # Release the samples older than the delay, in order.
        while len(self._buf) > 1 and self._buf[1][0] <= t - self.delay_s:
            self._step(*self._buf.pop(0))
        return self.value

    def _step(self, t: float, z: float) -> None:
        if self._t is None:
            self._t, self._z = t, z
            return
        dt = t - self._t
        if not 0.0 < dt < 0.1:
            self._t, self._z = t, z
            return
        v = (z - self._z) / dt
        self._t, self._z = t, z
        # Leak as a first-order high-pass: the IMU path integrates with a
        # leak at leak_hz, which is s/(s + ω_leak) on the true velocity.
        a = 1.0 / (1.0 + 2 * math.pi * self.leak_hz * dt)
        self._hp1 = a * (self._hp1 + v - self._v_prev)
        self._v_prev = v
        b, a2 = self._b, self._a
        x = self._hp1
        y = (
            b[0] * x
            + b[1] * self._hx[0]
            + b[2] * self._hx[1]
            - a2[1] * self._hy[0]
            - a2[2] * self._hy[1]
        )
        self._hx = [x, self._hx[0]]
        self._hy = [y, self._hy[0]]
        alpha = 1 - math.exp(-2 * math.pi * self.lp_hz * dt)
        self._lp += (y - self._lp) * alpha
        self.value = self._lp


@dataclass
class TipDamper:
    """``τ = J_zᵀ·(−c·v)`` on ``columns`` of the 7 arm joints.

    Args:
        gain: ``c``, N·s/m of vertical damping at the tool.
        columns: Arm-joint indices (0-6) that apply it.
        max_torque: Per-joint clamp (Nm).
        ramp_s: Fade-in after :meth:`start` (and after a stale gap).
        stale_s: No IMU sample for this long → torque 0.
        trip_speed: A flex velocity above this (m/s) for ``trip_s`` means the
            loop is feeding the shake, not damping it (too much gain for the
            IMU's delay — the simulation diverged by 80 N·s/m): the damper
            switches itself off for the rest of the pass (:attr:`tripped`).
            slow_osc's own flex velocity peaks near 60 mm/s undamped.
    """

    gain: float
    columns: tuple[int, ...]
    max_torque: float = 1.0
    ramp_s: float = 1.0
    stale_s: float = 0.05
    trip_speed: float = 0.08
    trip_s: float = 0.15
    estimator: VerticalVelocity = field(default_factory=VerticalVelocity)
    encoder: EncoderVelocity = field(default_factory=EncoderVelocity)
    tripped: bool = field(default=False, init=False)
    _fast_since: float | None = field(default=None, init=False)
    _last_sample: float | None = field(default=None, init=False)
    _ramp_from: float | None = field(default=None, init=False)

    @property
    def flex(self) -> float:
        """Band tool velocity the encoders do not account for (m/s)."""
        return self.estimator.value - self.encoder.value

    def start(self, now: float) -> None:
        self.estimator.reset()
        self.encoder.reset()
        self._last_sample = None
        self._ramp_from = now
        self.tripped = False
        self._fast_since = None

    def feed(self, rows: np.ndarray) -> None:
        """Live IMU rows (``t, acc xyz, gyro xyz``), oldest first."""
        for r in rows:
            self.estimator.update(float(r[0]), r[1:4], r[4:7])
            self._last_sample = float(r[0])

    def feed_height(self, t: float, z: float) -> None:
        """The tool height the encoders give (FK of the measured pose, m)."""
        self.encoder.update(t, z)

    def torque(self, now: float, jac_z: np.ndarray) -> np.ndarray:
        """Joint torques (7,) for this tick. ``jac_z[i]`` = ∂(tool height)/∂q_i
        (m/rad) at the measured pose."""
        tau = np.zeros(7)
        if abs(self.flex) > self.trip_speed:
            if self._fast_since is None:
                self._fast_since = now
            elif now - self._fast_since > self.trip_s:
                self.tripped = True
        else:
            self._fast_since = None
        if self.tripped:
            return tau
        if self._last_sample is None or now - self._last_sample > self.stale_s:
            self._ramp_from = None  # re-ramp after the gap
            return tau
        if self._ramp_from is None:
            self._ramp_from = now
        ramp = min(1.0, max(0.0, (now - self._ramp_from) / self.ramp_s))
        force = -self.gain * self.flex
        for i in self.columns:
            tau[i] = float(
                np.clip(ramp * jac_z[i] * force, -self.max_torque, self.max_torque)
            )
        return tau
