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
    #: Per-column scale on ``gain`` (default 1): on jelly shoulder_1 damps
    #: the 1-3 Hz sway and the elbow the 3-15 Hz shake, each up to its own
    #: gain before its loop phase runs out.
    weights: dict[int, float] = field(default_factory=dict)
    #: Centre (Hz) of a first-order lead-lag on the flex velocity (zero at
    #: lead_hz/2, pole at 2·lead_hz: +37° there, 4× gain above), or 0 for
    #: none. On jelly's shoulder_1 the damping's loop phase wraps near 8 Hz.
    lead_hz: float = 0.0
    #: Notch (Hz, 0 = none) on the damping force, Q :attr:`notch_q`, designed
    #: at the 240 Hz motion rate: where the loop phase has wrapped the
    #: damping drives the mode it should damp (jelly: ~11 Hz with the elbow
    #: in, a buzz the vertical shake score barely sees but the IMU's
    #: acceleration does).
    notch_hz: float = 0.0
    notch_q: float = 1.0
    tripped: bool = field(default=False, init=False)
    _nx: list[float] = field(default_factory=lambda: [0.0, 0.0], init=False)
    _ny: list[float] = field(default_factory=lambda: [0.0, 0.0], init=False)
    _lead_in: float = field(default=0.0, init=False)
    _lead_out: float = field(default=0.0, init=False)
    _lead_t: float | None = field(default=None, init=False)
    _started: float | None = field(default=None, init=False)
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
        self._started = now
        self.tripped = False
        self._lead_t = None
        self._nx, self._ny = [0.0, 0.0], [0.0, 0.0]
        self._fast_since = None

    def feed(self, rows: np.ndarray) -> None:
        """Live IMU rows (``t, acc xyz, gyro xyz``), oldest first."""
        for r in rows:
            self.estimator.update(float(r[0]), r[1:4], r[4:7])
            self._last_sample = float(r[0])

    def feed_height(self, t: float, z: float) -> None:
        """The tool height the encoders give (FK of the measured pose, m)."""
        self.encoder.update(t, z)

    def _lead(self, now: float, x: float) -> float:
        """The lead-lag stage (:attr:`lead_hz`), bilinear at this tick's dt."""
        if self.lead_hz <= 0:
            return x
        if self._lead_t is None:
            self._lead_t, self._lead_in, self._lead_out = now, x, x
            return x
        dt = now - self._lead_t
        self._lead_t = now
        if not 0.0 < dt < 0.1:
            return self._lead_out
        wz, wp = math.pi * self.lead_hz, 4 * math.pi * self.lead_hz
        # H(s) = (1 + s/wz) / (1 + s/wp), Tustin with s = (2/dt)(z-1)/(z+1).
        k = 2.0 / dt
        b0, b1 = 1 + k / wz, 1 - k / wz
        a0, a1 = 1 + k / wp, 1 - k / wp
        y = (b0 * x + b1 * self._lead_in - a1 * self._lead_out) / a0
        self._lead_in, self._lead_out = x, y
        return y

    def _notch(self, x: float) -> float:
        if self.notch_hz <= 0:
            return x
        w0 = 2 * math.pi * self.notch_hz / 240.0
        alpha = math.sin(w0) / (2 * self.notch_q)
        a0 = 1 + alpha
        b0, b1, b2 = 1 / a0, -2 * math.cos(w0) / a0, 1 / a0
        a1, a2 = -2 * math.cos(w0) / a0, (1 - alpha) / a0
        y = (
            b0 * x
            + b1 * self._nx[0]
            + b2 * self._nx[1]
            - a1 * self._ny[0]
            - a2 * self._ny[1]
        )
        self._nx = [x, self._nx[0]]
        self._ny = [y, self._ny[0]]
        return y

    def torque(self, now: float, jac_z: np.ndarray) -> np.ndarray:
        """Joint torques (7,) for this tick. ``jac_z[i]`` = ∂(tool height)/∂q_i
        (m/rad) at the measured pose."""
        tau = np.zeros(7)
        started = self._started is not None and now - self._started >= self.ramp_s
        # The guard waits out the ramp-in: the filters' start-up transient
        # (larger with a lower band edge) is not a runaway, and the torque is
        # ramped down meanwhile anyway.
        if started and abs(self.flex) > self.trip_speed:
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
        force = -self.gain * self._notch(self._lead(now, self.flex))
        for i in self.columns:
            tau[i] = float(
                np.clip(
                    ramp * self.weights.get(i, 1.0) * jac_z[i] * force,
                    -self.max_torque,
                    self.max_torque,
                )
            )
        return tau


# ---------------------------------------------------------------------------
# Gyro flex damping
# ---------------------------------------------------------------------------
#
# The accelerometer path above has to integrate and high-pass near 1 Hz to
# turn acceleration into velocity, and that filter's phase is what kept
# TipDamper out of the 1-3 Hz band. The gyro measures rotation *rate*
# directly: less the rotation rate the joint encoders imply (FK of the
# measured pose, differenced), it is the rate of the flex past the encoders,
# with only the IMU's few-ms latency. On slow_osc (2026-09-30) that causal
# signal, projected on shoulder_1's and the elbow's axes, matched its
# zero-phase version to −1° and coherence 0.96-0.97 over 1-3 Hz.


def _vee(m: np.ndarray) -> np.ndarray:
    """The rotation vector of a small rotation matrix (its skew part)."""
    return 0.5 * np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])


def fit_mount(
    t: np.ndarray,
    rotations: np.ndarray,
    imu_t: np.ndarray,
    gyro_deg_s: np.ndarray,
    band: tuple[float, float] = (0.1, 1.0),
) -> tuple[np.ndarray, float]:
    """The wrist camera's rotation against the gripper mount (cam ← mount)
    from a recorded motion, and the fit's R².

    Kabsch-aligns the gyro to the FK body rate on the slow deliberate motion
    (``band``), where the arm is rigid and both see the same rotation.
    ``rotations`` are the measured pose's world ← mount matrices at ``t``.
    """
    from .learning import band_limit

    t = np.asarray(t, dtype=float)
    fs = (len(t) - 1) / (t[-1] - t[0])
    dr = np.einsum("nji,njk->nik", rotations[:-1], rotations[1:])
    w = np.stack([_vee(m) for m in dr]) * fs
    w = np.vstack([w, w[-1:]])
    g = np.radians(np.asarray(gyro_deg_s, dtype=float))
    g = np.stack([np.interp(t, imu_t, g[:, i]) for i in range(3)], 1)
    a, b = band_limit(w, fs, band), band_limit(g, fs, band)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    mount = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    r2 = 1.0 - float(np.sum((b - a @ mount.T) ** 2)) / max(float(np.sum(b**2)), 1e-18)
    return mount, r2


@dataclass
class GyroFlexDamper:
    """``τ_j = −c_j · ω_flex · a_j`` on the chosen arm joints.

    ``ω_flex`` is the wrist gyro's rotation rate less the rate the joint
    encoders imply, in the world frame; ``a_j`` is joint ``j``'s axis at the
    measured pose, so each joint damps the flex about its own axis — the
    motion past its encoder. Sign as :class:`TipDamper`'s (a force against
    the flex velocity), which damped the 3-15 Hz shake on hardware.

    Args:
        gains: ``{arm-joint index: c}`` (N·m·s/rad).
        mount: The camera's rotation against the gripper mount (cam ←
            mount, :func:`fit_mount`).
        hp_hz: Causal first-order high-pass on the flex rate — removes gyro
            bias and a mount misfit's leak of the deliberate motion.
        lp_hz: Causal two-pole low-pass (two first-order sections). The
            gyro's flex rate on slow_osc was 1.27°/s at 15-60 Hz (noise and
            structural buzz) against 0.41°/s at 1-3 Hz; two poles at 15 Hz
            cut 30 Hz 5× for 15° of lag at 2 Hz.
        delay_s: The encoder rate is held back by this much so it lines up
            with the gyro's latency.
        trip_rate: A flex rate above this (rad/s) for ``trip_s`` switches
            the damper off for the pass (:attr:`tripped`). slow_osc's own
            peaks near 0.05 rad/s undamped.
    """

    gains: dict[int, float]
    mount: np.ndarray
    max_torque: float = 0.5
    hp_hz: float = 0.5
    lp_hz: float = 15.0
    delay_s: float = 0.008
    ramp_s: float = 1.0
    stale_s: float = 0.05
    trip_rate: float = 0.15
    trip_s: float = 0.15
    tripped: bool = field(default=False, init=False)
    flex_axis: np.ndarray = field(default_factory=lambda: np.zeros(7), init=False)
    _gyro: np.ndarray | None = field(default=None, init=False)
    _last_sample: float | None = field(default=None, init=False)
    _rot: np.ndarray | None = field(default=None, init=False)
    _rot_t: float | None = field(default=None, init=False)
    _rates: list[tuple[float, np.ndarray]] = field(default_factory=list, init=False)
    _hp: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False)
    _hp_in: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False)
    _lp: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False)
    _lp2: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False)
    _filt_t: float | None = field(default=None, init=False)
    _started: float | None = field(default=None, init=False)
    _ramp_from: float | None = field(default=None, init=False)
    _fast_since: float | None = field(default=None, init=False)

    @property
    def columns(self) -> tuple[int, ...]:
        return tuple(sorted(self.gains))

    @property
    def flex(self) -> float:
        """Largest flex rate about a damped joint's axis (rad/s)."""
        return float(np.max(np.abs(self.flex_axis[list(self.columns)])))

    def start(self, now: float) -> None:
        self.tripped = False
        self._gyro = self._last_sample = self._rot = self._rot_t = None
        self._rates = []
        self._hp, self._hp_in = np.zeros(3), np.zeros(3)
        self._lp, self._lp2 = np.zeros(3), np.zeros(3)
        self._filt_t = None
        self.flex_axis = np.zeros(7)
        self._started = self._ramp_from = now
        self._fast_since = None

    def feed(self, rows: np.ndarray) -> None:
        """Live IMU rows (``t, acc xyz, gyro xyz`` in °/s), oldest first."""
        if len(rows):
            self._gyro = np.radians(np.asarray(rows[-1][4:7], dtype=float))
            self._last_sample = float(rows[-1][0])

    def feed_pose(self, now: float, rotation: np.ndarray, axes: np.ndarray) -> None:
        """The measured pose this tick: world ← mount ``rotation`` (3, 3) and
        the arm joints' world-frame unit ``axes`` (7, 3)."""
        rotation = np.asarray(rotation, dtype=float)
        if self._rot is not None and self._rot_t is not None:
            dt = now - self._rot_t
            if 0.0 < dt < 0.1:
                self._rates.append((now, _vee(self._rot.T @ rotation) / dt))
        self._rot, self._rot_t = rotation, now
        while len(self._rates) > 1 and self._rates[1][0] <= now - self.delay_s:
            self._rates.pop(0)
        if self._gyro is None or not self._rates:
            return
        w_enc = self._rates[0][1]  # mount frame, delay_s old
        raw = self._gyro - self.mount @ w_enc  # camera frame
        if self._filt_t is None:
            self._filt_t, self._hp_in = now, raw
            return
        dt = now - self._filt_t
        self._filt_t = now
        if not 0.0 < dt < 0.1:
            return
        a = 1.0 / (1.0 + 2 * math.pi * self.hp_hz * dt)
        self._hp = a * (self._hp + raw - self._hp_in)
        self._hp_in = raw
        alpha = 1 - math.exp(-2 * math.pi * self.lp_hz * dt)
        self._lp += (self._hp - self._lp) * alpha
        self._lp2 += (self._lp - self._lp2) * alpha
        world = rotation @ (self.mount.T @ self._lp2)
        self.flex_axis = np.asarray(axes, dtype=float) @ world

    def torque(self, now: float) -> np.ndarray:
        tau = np.zeros(7)
        started = self._started is not None and now - self._started >= self.ramp_s
        if started and self.flex > self.trip_rate:
            if self._fast_since is None:
                self._fast_since = now
            elif now - self._fast_since > self.trip_s:
                self.tripped = True
        else:
            self._fast_since = None
        if self.tripped:
            return tau
        if self._last_sample is None or now - self._last_sample > self.stale_s:
            self._ramp_from = None
            return tau
        if self._ramp_from is None:
            self._ramp_from = now
        ramp = (
            min(1.0, max(0.0, (now - self._ramp_from) / self.ramp_s))
            if self.ramp_s > 0
            else 1.0
        )
        for j, c in self.gains.items():
            tau[j] = float(
                np.clip(
                    -ramp * c * self.flex_axis[j], -self.max_torque, self.max_torque
                )
            )
        return tau


@dataclass
class TorqueProbe:
    """A known torque excitation on chosen arm joints, for measuring the
    joint-torque → flex response the gyro damper closes its loop through.

    A multisine: ``components`` sines spread log-uniformly over ``band`` with
    random phases (``seed``), scaled to ``amplitude`` Nm peak per joint and
    faded in and out over ``ramp_s``. It plugs into the same pass loop as
    :class:`GyroFlexDamper` (it keeps that damper's flex estimate running,
    with zero gain, so the log carries it) and never reacts to what it
    measures.
    """

    amplitudes: dict[int, float]
    mount: np.ndarray
    band: tuple[float, float] = (0.5, 15.0)
    components: int = 40
    seed: int = 0
    ramp_s: float = 1.0
    monitor: GyroFlexDamper = field(init=False)
    tripped: bool = field(default=False, init=False)
    _started: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self._freqs = np.geomspace(self.band[0], self.band[1], self.components)
        self._phases = rng.uniform(0.0, 2 * math.pi, self.components)
        # The multisine's own crest: scale so the peak is ``amplitude``.
        t = np.arange(0.0, 60.0, 0.002)
        peak = np.abs(
            np.sin(2 * math.pi * self._freqs[:, None] * t + self._phases[:, None]).sum(
                0
            )
        ).max()
        self._scale = 1.0 / float(peak)
        self.monitor = GyroFlexDamper(
            gains={j: 0.0 for j in self.amplitudes}, mount=self.mount
        )

    @property
    def columns(self) -> tuple[int, ...]:
        return tuple(sorted(self.amplitudes))

    @property
    def flex(self) -> float:
        """The flex rate about the first probed joint's axis (rad/s, signed)."""
        return float(self.monitor.flex_axis[self.columns[0]])

    def start(self, now: float) -> None:
        self._started = now
        self.monitor.start(now)

    def feed(self, rows: np.ndarray) -> None:
        self.monitor.feed(rows)

    def feed_pose(self, now: float, rotation: np.ndarray, axes: np.ndarray) -> None:
        self.monitor.feed_pose(now, rotation, axes)

    def torque(self, now: float) -> np.ndarray:
        tau = np.zeros(7)
        if self._started is None:
            return tau
        s = now - self._started
        ramp = min(1.0, s / self.ramp_s) if self.ramp_s > 0 else 1.0
        u = (
            float(np.sin(2 * math.pi * self._freqs * s + self._phases).sum())
            * self._scale
        )
        for j, a in self.amplitudes.items():
            tau[j] = ramp * a * u
        return tau
