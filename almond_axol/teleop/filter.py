"""
Interpolation and filtering for VR teleoperation.
"""

from __future__ import annotations

import math

import numpy as np


class TrapezoidalFilter:
    """Per-joint velocity/acceleration-limited tracker.

    Tracks a moving IK target under hard velocity and acceleration limits.
    This is the stage that protects the arm's structural resonance from
    broadband jerk: whatever noise survives the upstream filters, the output
    here never accelerates faster than ``max_accel``.

    Two details make it track tightly without violating the limits:

    - **Target-velocity feedforward.** A tracker that only knows the current
      target position must lag a moving target by ``v²/(2·max_accel)`` (its
      braking rule assumes the target is where it will stop — 2.3° behind a
      1 rad/s hand at 12.6 rad/s²). The desired velocity is therefore the
      low-passed target velocity plus the braking-rule correction on the
      remaining error, which keeps the error near zero during smooth motion
      while the acceleration clamp still rejects high-frequency content.
    - **Acceleration-gated arrival.** Landing exactly on the target replaces
      the velocity with the raw ``err/dt``; that is only allowed when the
      implied velocity change fits one tick's acceleration budget.
      Unconditional snapping (a previous fix for an arrival limit-cycle)
      degenerated into a pass-through whenever the filter was caught up —
      i.e. during all of normal teleop — measured as 61 rad/s² output
      acceleration against a 12.6 limit. Without the snap, overshoot is
      bounded by ``max_accel·dt²/2`` (~0.01° per event), which is invisible.

    Args:
        max_vel:   Maximum joint velocity in rad/s — a scalar, or per-joint
                   values with the same shape as the target vector.
        max_accel: Maximum joint acceleration in rad/s².
        dt:        Control step duration in seconds (``1 / frequency``).
    """

    # Linear tracking gains. Two designs measurably fail here:
    #
    # - The trapezoidal sqrt(2·a·d) rule is a time-optimal — i.e. bang-bang —
    #   controller: with a noisy moving target it saturates the acceleration
    #   clamp on every tick, a full-amplitude ±max_accel square wave (the
    #   worst spectral shape the limit allows, and 16 Nm of oscillating
    #   torque once the j_eff feedforward multiplies it).
    # - Adding target-velocity feedforward to fix the linear tracker's lag
    #   introduces a closed-loop zero that inherently peaks (~1.3×) right at
    #   the arm's 2.5-3.2 Hz structural resonance — replaying recorded teleop
    #   showed *more* 2-4 Hz acceleration energy than the pass-through it
    #   replaced, for every feedforward cutoff and gain combination tried.
    #
    # So the tracker is a plain critically damped second-order loop:
    # position error → velocity command (kp) → acceleration (kv), with
    # kv = 2·ωn and kp = ωn/2 so ζ = 1. Monotone response — no peaking
    # anywhere, 27% attenuation at the 3 Hz resonance, high-frequency noise
    # rolled off — at the cost of ~v/kp tracking lag (≈1° at the ~0.3 rad/s
    # joint speeds of normal teleop). The sqrt rule survives only as a hard
    # velocity *ceiling* for large catch-up moves, where distance is large
    # and its noise sensitivity is irrelevant.
    _POS_TRACK_GAIN = 15.7  # 1/s = ωn/2 with ωn = 2π·5 Hz
    _VEL_TRACK_GAIN = 62.8  # 1/s = 2·ωn

    # Braking-ceiling margin: v_stop plans against this fraction of
    # max_accel, reserving headroom for the velocity-loop lag so
    # decelerations that start late still land without overshoot.
    #
    # (A jerk limit on the acceleration state was tried and measured out:
    # the linear loop's output jerk is already low — p99 186 rad/s³ on
    # recorded teleop, so limiting bought <6% resonance-band energy — while
    # the accel-state slew broke the braking guarantee, overshooting a 30°
    # step by 26°. Don't reintroduce it without a real lookahead planner.)
    _BRAKE_MARGIN = 0.8

    def __init__(
        self, max_vel: float | np.ndarray, max_accel: float, dt: float
    ) -> None:
        """Initialize the filter.

        Args:
            max_vel:   Maximum joint velocity in rad/s — scalar or per-joint
                       (same shape as the target vector).
            max_accel: Maximum joint acceleration in rad/s².
            dt:        Control step duration in seconds (``1 / frequency``).
        """
        self.max_vel = np.asarray(max_vel, dtype=np.float32)
        self.max_accel = max_accel
        self.dt = dt
        self._pos: np.ndarray | None = None
        self._vel: np.ndarray | None = None

    @property
    def position(self) -> np.ndarray | None:
        """The filter's current output position (the last commanded value).

        ``None`` until the first :meth:`update`. Safe to read cross-thread:
        updates replace the array reference atomically.
        """
        return self._pos

    def update(self, target: np.ndarray | None) -> np.ndarray | None:
        """Advance one step toward ``target``.

        Returns ``target`` unchanged on first call.  Subsequent calls move the
        internal position toward ``target`` subject to velocity and acceleration
        limits.
        """
        if target is None:
            return None
        target = np.asarray(target, dtype=np.float32)
        if self._pos is None:
            self._pos = target.copy()
            self._vel = np.zeros_like(target)
            return target.copy()

        err = target - self._pos
        dist = np.abs(err)
        adt = self.max_accel * self.dt

        # Discrete-time stopping speed: the largest speed from which
        # decelerating by the (margined) braking accel each tick lands on the
        # target without overshoot. Used only as a ceiling (see gains above).
        a_brake = self._BRAKE_MARGIN * self.max_accel
        bdt = 0.5 * a_brake * self.dt
        v_stop = -bdt + np.sqrt(bdt**2 + 2.0 * a_brake * dist)

        # For distances below ~5° the linear gain is smaller than v_stop and
        # the command stays fully linear; beyond that the sqrt rule takes
        # over for overshoot-free catch-up.
        ceiling = np.minimum(self.max_vel, v_stop)
        desired_vel = np.clip(self._POS_TRACK_GAIN * err, -ceiling, ceiling)

        # Proportional velocity tracking under the acceleration clamp (see
        # _VEL_TRACK_GAIN): noise produces small accelerations, moves saturate.
        vel_prev = self._vel
        self._vel = vel_prev + np.clip(
            self._VEL_TRACK_GAIN * (desired_vel - vel_prev) * self.dt, -adt, adt
        )

        # Land exactly on the target only when the implied velocity (err/dt)
        # is reachable within this tick's acceleration budget — snapping
        # unconditionally bypasses both limits whenever the filter is caught
        # up (see class docstring). When the snap is not allowed, integrating
        # the accel-limited velocity may overshoot a stationary target by at
        # most max_accel·dt²/2 and pull back next tick — negligible.
        step = self._vel * self.dt
        snap_vel = err / self.dt
        snap_ok = (np.abs(step) > dist) & (
            np.abs(snap_vel - vel_prev) <= adt * (1.0 + 1e-6)
        )
        self._pos = np.where(snap_ok, target, self._pos + step)
        self._vel = np.where(snap_ok, snap_vel, self._vel)

        return self._pos.copy()

    def reset(self, seed: np.ndarray | None = None) -> None:
        """Reset filter state, optionally seeding position with a known value."""
        if seed is not None:
            self._pos = np.asarray(seed, dtype=np.float32).copy()
            self._vel = np.zeros_like(self._pos)
        else:
            self._pos = None
            self._vel = None


class AlphaSmoothFilter:
    """Exponential smoothing filter for joint angle arrays (radians).

    Applies an alpha-weighted moving average. Lower alpha = more lag but
    smoother output. Higher alpha = more responsive but noisier.

    Args:
        alpha: Blend factor in (0, 1]. ``1.0`` disables smoothing.
    """

    def __init__(self, alpha: float) -> None:
        """Initialize the filter.

        Args:
            alpha: Smoothing blend factor in ``(0, 1]``. ``1.0`` disables smoothing.
        """
        self.alpha = alpha
        self._prev: np.ndarray | None = None

    def update(self, new: np.ndarray | None) -> np.ndarray | None:
        """Apply one smoothing step. Returns ``new`` unchanged on first call."""
        if new is None:
            return None
        new = np.asarray(new, dtype=np.float32)
        if self._prev is None or len(self._prev) != len(new):
            self._prev = new.copy()
            return new.copy()
        out = self.alpha * new + (1.0 - self.alpha) * self._prev
        self._prev = out
        return out

    def reset(self, seed: np.ndarray | None = None) -> None:
        """Reset filter state, optionally seeding with a known starting value."""
        self._prev = (
            np.asarray(seed, dtype=np.float32).copy() if seed is not None else None
        )


class LagCompensatedLowPass:
    """Linear 2-pole low-pass with velocity-feedforward lag compensation.

    Replaces the One Euro filter on the VR pose streams. OneEuro's
    speed-adaptive cutoff is *multiplicative* — the cutoff (driven by the
    signal's own speed envelope) modulates the signal per sample, an
    inherently nonlinear operation that sprays harmonics of the intentional
    motion into the 3-12 Hz band. Measured on a realistic two-tone hand
    trajectory (0.4 + 0.8 Hz, peak ~2 m/s) carrying 2 mm of 5 Hz tremor,
    the production OneEuro emitted **286%** of the input's 3-12 Hz energy —
    it manufactured mid-band noise from clean motion — and that band is
    exactly where the arm's structural modes live (4-7 Hz), so the noise
    came out of the IK as joint churn and out of the arm as visible jitter.
    OneEuro is designed for cursors, where mid-band artifacts are invisible;
    driving a resonant plant they are the failure mode.

    This filter is linear, so it cannot create in-band energy: two cascaded
    first-order poles at ``cutoff`` reject tremor (12 dB/oct), and the
    resulting lag is cancelled by adding ``T·v̂`` where ``T`` is the
    cascade's DC group delay (``2/ω_c``) and ``v̂`` a same-pole-filtered
    velocity estimate of the once-filtered signal.

    Defaults were set by replaying a real recorded session's raw VR stream
    (jit15, 81 s engaged) through candidates: full lag compensation passes
    102% of the 3-12 Hz band (the FF path's in-band gain undoes the poles'
    rejection), while ``lag_comp=0.7`` passes 81% at 52 mm p95 tracking
    error — matching the old OneEuro's in-band rejection (82% on real data)
    without its nonlinear artifacts, and with far less lag than its
    worst case. Pushing rejection further is a steep trade (62% costs
    ~100 mm p95): the raw stream's in-band noise grows with hand speed
    (0.6 mm RMS at rest → 8 mm at 1.2+ m/s) and sits barely 1.5 octaves
    above intentional motion, so target-side filtering is inherently
    capped — the rest of the defence is plant-side damping.

    Args:
        freq:     Nominal sampling frequency in Hz, used when ``update`` is
                  called without a timestamp.
        cutoff:   Pole frequency (Hz) of each of the two low-pass stages.
        lag_comp: Fraction of the DC group delay cancelled by the velocity
                  feedforward (1.0 = full cancellation; lower trades lag
                  for less in-band feedforward gain).
    """

    def __init__(
        self,
        freq: float,
        cutoff: float = 2.5,
        lag_comp: float = 0.7,
    ) -> None:
        self._freq = freq
        self._w = 2.0 * math.pi * cutoff
        self._t_ff = lag_comp * 2.0 / self._w
        self._y1: np.ndarray | None = None
        self._y2: np.ndarray | None = None
        self._v: np.ndarray | None = None
        self._y1_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    def update(self, x: np.ndarray, t: float | None = None) -> np.ndarray:
        """Apply one filter step. Returns ``x`` unchanged on the first call.

        Args:
            x: Sample value.
            t: Sample timestamp in seconds (any steady clock). When provided,
                the filter uses the true inter-sample spacing instead of the
                fixed construction-time ``freq`` — the samples this filter
                sees arrive at the irregular IK solve cadence, and a fixed-
                frequency assumption converts that timing jitter into value
                jitter (a late sample's larger motion reads as one fast step,
                a bunched sample as a stall). ``None`` keeps the fixed rate.
        """
        x = np.asarray(x, dtype=np.float32)
        if self._y1 is None:
            self._y1 = x.copy()
            self._y2 = x.copy()
            self._v = np.zeros_like(x)
            self._y1_prev = x.copy()
            self._t_prev = t
            return x.copy()

        dt = 1.0 / self._freq
        if t is not None and self._t_prev is not None:
            # Clamp so a stream gap or duplicate stamp can't blow up the
            # velocity estimate (2 ms .. 100 ms spacing).
            dt = min(max(t - self._t_prev, 0.002), 0.1)
        self._t_prev = t

        a = self._w * dt / (1.0 + self._w * dt)
        self._y1 += a * (x - self._y1)
        self._y2 += a * (self._y1 - self._y2)
        # Velocity of the once-filtered signal (already denoised), smoothed
        # by the same pole: the feedforward term T·v̂ cancels the cascade's
        # group delay on ramps without reintroducing raw-sample noise.
        v_raw = (self._y1 - self._y1_prev) / dt
        self._y1_prev = self._y1.copy()
        self._v += a * (v_raw - self._v)
        return (self._y2 + self._t_ff * self._v).astype(np.float32)

    def nudge(self, delta: np.ndarray) -> None:
        """Shift the value state by ``delta``, keeping the velocity estimate.

        For compensating an upstream reference-frame jump (e.g. a VR headset
        re-localization): the signal's frame moved, the signal's motion didn't.
        A ``reset`` would zero the velocity estimate and cold-start the
        filter; a nudge keeps it fully warm.
        """
        if self._y1 is not None:
            d = np.asarray(delta, dtype=np.float32)
            self._y1 = self._y1 + d
            self._y2 = self._y2 + d
            self._y1_prev = self._y1_prev + d

    def reset(self, seed: np.ndarray | None = None) -> None:
        """Reset filter state, optionally seeding with a known starting value."""
        if seed is not None:
            s = np.asarray(seed, dtype=np.float32)
            self._y1 = s.copy()
            self._y2 = s.copy()
            self._v = np.zeros_like(s)
            self._y1_prev = s.copy()
        else:
            self._y1 = None
            self._y2 = None
            self._v = None
            self._y1_prev = None
        self._t_prev = None


class ResetInterpolator:
    """Steps through a pre-computed collision-aware trajectory one waypoint per call.

    Optionally ramps gripper values (normalized [0, 1]) from a start value to
    1.0 (open) over the same number of steps as the arm trajectory.
    """

    def __init__(self) -> None:
        """Construct the interpolator with no active trajectory.

        Call :meth:`set_trajectory` before calling :meth:`step`.
        """
        self._trajectory: list[np.ndarray] | None = None
        self._traj_index: int = 0
        self._l_grip_start: float = 0.0
        self._r_grip_start: float = 0.0

    def set_trajectory(
        self,
        trajectory: list[np.ndarray],
        l_grip: float = 0.0,
        r_grip: float = 0.0,
    ) -> None:
        """Load a pre-computed trajectory and gripper start values."""
        self._trajectory = [np.array(q, dtype=np.float64) for q in trajectory]
        self._traj_index = 0
        self._l_grip_start = l_grip
        self._r_grip_start = r_grip

    def step(self) -> tuple[np.ndarray | None, float, float, bool]:
        """Advance one step.

        Returns ``(new_q_rad, l_grip, r_grip, done)`` where gripper values are
        smoothstepped from their start values to 1.0 over the trajectory length.
        """
        if self._trajectory is None or self._traj_index >= len(self._trajectory):
            self.clear()
            return None, 1.0, 1.0, True
        n = len(self._trajectory)
        alpha = (self._traj_index + 1) / n
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        l_grip = self._l_grip_start + smooth * (1.0 - self._l_grip_start)
        r_grip = self._r_grip_start + smooth * (1.0 - self._r_grip_start)
        q = self._trajectory[self._traj_index]
        self._traj_index += 1
        done = self._traj_index >= n
        if done:
            self._trajectory = None
            self._traj_index = 0
        return q, l_grip, r_grip, done

    def is_active(self) -> bool:
        """True if trajectory playback is in progress."""
        return self._trajectory is not None and self._traj_index < len(self._trajectory)

    def clear(self) -> None:
        """Cancel any active trajectory."""
        self._trajectory = None
        self._traj_index = 0
