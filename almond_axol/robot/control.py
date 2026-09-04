"""Motor control utilities: friction model, differentiator, contact watchdog.

Gravity compensation is handled separately — see
:class:`almond_axol.robot.gravity.GravityCompensator` — because the simple
per-joint ``ga·cos(q) + gb·sin(q)`` model used here previously ignored child
links and produced incorrect torques.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

from ..constants import ARM_JOINTS

# Filter poles for the control loop's differentiators, in rad/s (the update
# is a = 1/(1 + Ts·ω), a first-order pole at ω rad/s — NOT Hz).
#
# CUTOFF_FREQ (20 rad/s ≈ 3.2 Hz) filters the motor-facing paths: v_des
# (impedance velocity target, friction FF) and the a_des chain (j_eff
# inertia FF). These only need the intentional-motion band (<2 Hz), and a
# low pole keeps command-stream noise out of the Nm-scale feedforwards.
#
# VEL_CUTOFF_FREQ (80 rad/s ≈ 12.7 Hz) filters the two differentiators
# inside the *host damping* path only (see BandPass below): there the pole
# must sit well above the ~3 Hz shoulder resonance so the damping arrives
# in phase, and the band-pass that follows provides the high-frequency
# rolloff a low pole would otherwise supply.
CUTOFF_FREQ = 20.0
VEL_CUTOFF_FREQ = 80.0

# Host damping band-pass (see BandPass): centre and quality factor.
#
# Why a band-pass: kd_host exists to damp the shoulders' ~3 Hz structural
# resonance, which the motors' lagged internal velocity estimates can't
# touch. The damping torque is computed host-side at ~100 Hz with a
# one-cycle transport delay, so its phase degrades with frequency: at
# 25-35 Hz the delayed torque is fully anti-phase and *excites* whatever
# structural mode lives there. Any single low-pass pole must therefore
# trade resonance-band phase against high-frequency gain — the historical
# 20 rad/s velocity pole sat exactly ON the 3.2 Hz resonance and delivered
# only ~35% of kd_host in phase (a closed-loop sine probe measured a
# resonant gain of 2.1 with ~70 Nm·s/rad commanded), while an 80 rad/s pole
# delivered the damping but passed enough 25-35 Hz gain to shake the arm
# violently during the reset ramp. A unity-peak band-pass at the resonance
# escapes the trade: at 3.2 Hz the full chain (fast differentiator +
# band-pass + delay) delivers ~0.8 of kd_host in phase (2.3× the old
# design), while at 30 Hz its gain is ~7× *lower* than the old design's.
# Q = 0.8 keeps the passband wide enough (~2-5 Hz) to cover the
# resonance's pose dependence (ωn = √(kp/J)).
DAMP_BP_W0 = 20.0  # rad/s (≈3.2 Hz)
DAMP_BP_Q = 0.8

# How long a torque residual must stay above the watchdog threshold before a
# guarded move is judged to have hit something. Long enough to ride out
# feedforward transients and single noisy samples, short enough that the
# pull on the scene stays brief.
CONTACT_DEBOUNCE_S = 0.15


class ContactWatchdog:
    """Debounced measured-vs-gravity torque contact detector.

    Feed one per-arm torque-residual sample per control step (see
    :meth:`almond_axol.robot.axol.AxolArm.torque_residuals`); once any
    joint's residual magnitude stays above ``threshold`` for
    ``debounce_s``, :meth:`update` reports the offending joint. Used by the
    guarded return-to-rest moves to tell "the move is pushing/pulling on
    something that isn't in the plan" — a gripper still hooked on the
    scene, or an operator grabbing an arm — apart from the friction /
    model-error background.

    Args:
        threshold: Residual magnitude (motor torque units — Nm on Damiao)
            above which a joint counts as in contact. ``<= 0`` disables the
            watchdog (:meth:`update` never trips).
        debounce_s: How long the residual must stay above ``threshold``
            before tripping.
    """

    def __init__(self, threshold: float, debounce_s: float = CONTACT_DEBOUNCE_S):
        self.threshold = threshold
        self.debounce_s = debounce_s
        self._since: float | None = None

    def update(self, residuals, now: float | None = None) -> tuple[str, float] | None:
        """Feed one sample; return the tripped joint, or ``None``.

        Args:
            residuals: ``(left, right)`` arrays of per-arm-joint torque
                residuals in :data:`ARM_JOINTS` order (``None`` for an
                absent arm), as returned by ``torque_residuals()``.
            now: Sample time (``time.perf_counter()``); defaults to now.

        Returns:
            ``("left ELBOW", residual)`` for the worst joint once contact
            has been sustained for ``debounce_s``, else ``None``. The
            debounce state resets whenever every joint drops back under the
            threshold.
        """
        if self.threshold <= 0:
            return None
        if now is None:
            now = time.perf_counter()
        worst = 0.0
        worst_joint = ""
        for side, res in zip(("left", "right"), residuals):
            if res is None:
                continue
            for i, value in enumerate(res):
                mag = abs(float(value))
                if mag > worst:
                    worst = mag
                    worst_joint = f"{side} {ARM_JOINTS[i].name}"
        if worst <= self.threshold:
            self._since = None
            return None
        if self._since is None:
            self._since = now
            return None
        if now - self._since >= self.debounce_s:
            self._since = None
            return worst_joint, worst
        return None


# Steepness ceiling for the friction *feedforward* (identification keeps the
# raw fit). Fitted k values are often very steep (k≈800 saturates the Coulomb
# term within |v| < 0.025 rad/s) — physically right for stiction, but as
# feedforward it makes the ±Fc torque snap on/off within a couple of control
# cycles at every trajectory arrival and reversal. On the shoulders that
# ~1 Nm near-step rings the arm's 2–3 Hz structural mode: at ROM's
# gravity-loaded −90° waypoint the arrival ring measured 0.37 Nm RMS /
# 10 mdeg with raw k, and 0.10–0.13 Nm / ~0 mdeg with the cap (k=50 and
# k=100 performed equally; 100 keeps more low-speed friction comp). Sweep
# and teleop speeds (≳0.3 rad/s) stay in the saturated region either way,
# so steady tracking feedforward is unchanged.
FRICTION_FF_K_MAX = 100.0


def compute_friction(
    velocity: float, Fc: float, k: float, Fv: float, Fo: float
) -> float:
    """Tanh friction model: τ = Fc * tanh(0.1 * k * v) + Fv * v + Fo

    ``k`` is capped at :data:`FRICTION_FF_K_MAX` (see above) so the Coulomb
    term ramps smoothly through zero crossings instead of stepping.
    """
    return (
        Fc * math.tanh(0.1 * min(k, FRICTION_FF_K_MAX) * velocity) + Fv * velocity + Fo
    )


class BandPass:
    """N-channel state-variable band-pass (Chamberlin SVF), unity centre gain.

    Confines the host damping torque to the structural-resonance band: at
    :data:`DAMP_BP_W0` the output tracks the input with ~zero phase and unity
    gain; gain rolls off 6 dB/oct on both sides, so the delayed host loop
    can't excite fast structural modes and doesn't drag slow intentional
    motion. Sample spacing is taken from the wall clock, matching the control
    loop cadence its input velocities arrive at.

    Args:
        n:  Number of independent channels.
        w0: Centre frequency in rad/s — a scalar shared by all channels, or a
            per-channel sequence. Structural modes differ per joint (the
            shoulders ring near 3 Hz, the elbow near 7-11 Hz depending on
            pose), and a damper centred on the wrong joint's mode is rolled
            off and phase-shifted exactly where that joint needs it.
        q:  Quality factor (bandwidth = w0/q) — a scalar shared by all
            channels, or a per-channel sequence. The 0.8 default keeps the
            band an octave wide either side, which suits a pose-tracked
            centre that only estimates the mode; a joint pinned on a
            *measured* ring frequency can afford a higher q, confining the
            damping to the ring so it stops dragging the slow final
            approach (the drag shows up as a step test that never settles).
    """

    def __init__(
        self,
        n: int,
        w0: float | Sequence[float] = DAMP_BP_W0,
        q: float | Sequence[float] = DAMP_BP_Q,
    ) -> None:
        if isinstance(w0, (int, float)):
            self._w0 = [float(w0)] * n
        else:
            if len(w0) != n:
                raise ValueError(f"w0 has {len(w0)} entries for {n} channels")
            self._w0 = [float(v) for v in w0]
        if isinstance(q, (int, float)):
            self._q = [float(q)] * n
        else:
            if len(q) != n:
                raise ValueError(f"q has {len(q)} entries for {n} channels")
            self._q = [float(v) for v in q]
        self._n = n
        self._lp = [0.0] * n
        self._bp = [0.0] * n
        self._last_time: float | None = None

    def update(self, x: list[float], w0: Sequence[float] | None = None) -> list[float]:
        """Advance one step; returns the band-passed values (zeros on first call).

        Args:
            x:  Input sample per channel.
            w0: Optional per-channel centre frequencies (rad/s) for this step,
                overriding the constructor values. The Chamberlin SVF computes
                its coefficient from ``w0`` fresh every step, so a
                slowly-varying centre is well-behaved — used by
                ``AxolArm.motion_control`` to keep each shoulder's damper
                centred on its pose-dependent impedance mode ωn = √(kp/J(q)).
        """
        if w0 is not None:
            self._w0 = [float(v) for v in w0]
        now = time.perf_counter()
        if self._last_time is None:
            self._last_time = now
            return [0.0] * self._n
        ts = now - self._last_time
        self._last_time = now
        if ts <= 0:
            return [b / q for b, q in zip(self._bp, self._q)]
        out: list[float] = []
        for i in range(self._n):
            # Chamberlin SVF coefficient; the sin() form keeps the centre
            # accurate at low fs, and clamping keeps the filter stable
            # across loop stalls.
            f = 2.0 * math.sin(min(0.5 * self._w0[i] * ts, 0.7))
            self._lp[i] += f * self._bp[i]
            hp = x[i] - self._lp[i] - self._bp[i] / self._q[i]
            self._bp[i] += f * hp
            out.append(self._bp[i] / self._q[i])
        return out


class Differentiator:
    """First-order low-pass differentiator, matching C++ Differentiator::Differentiate.

    For each channel:
        a = 1 / (1 + Ts * cutoff)
        b = a * cutoff
        vel[i] = vel_prev[i] * a + b * (pos[i] - pos_prev[i])

    Args:
        n: Number of channels to differentiate.
        cutoff: Low-pass pole in rad/s (see the module constants for how to
            choose; defaults to :data:`CUTOFF_FREQ`).
    """

    def __init__(self, n: int, cutoff: float = CUTOFF_FREQ) -> None:
        """Initialize the differentiator.

        Args:
            n: Number of independent channels to differentiate simultaneously.
            cutoff: Low-pass pole in rad/s.
        """
        self._n = n
        self._cutoff = cutoff
        self._vel_prev = [0.0] * n
        self._pos_prev: list[float | None] = [None] * n
        self._last_time: float | None = None
        self._ts_prev: list[float] | None = None

    def differentiate(
        self, positions: list[float], timestamps: list[float] | None = None
    ) -> list[float]:
        """Compute low-pass-filtered velocities from a new position sample.

        Returns a list of length ``n`` in rad/s.  Returns all zeros on the
        first call.  If called with no elapsed time (``Ts <= 0``), returns
        the previous velocity estimate unchanged.

        With ``timestamps`` (one per channel, seconds — e.g. CAN frame
        receive times), each channel is differentiated against its *own*
        sample interval instead of the shared wall clock at call time. Use
        this when positions come from cached feedback frames: the variable
        delay between a frame's arrival and this call otherwise shows up as
        velocity noise proportional to speed (scheduling jitter × velocity),
        which host-side damping then amplifies into torque chatter. A
        channel whose timestamp has not advanced keeps its previous
        velocity. Do not mix timestamped and wall-clock calls on one
        instance.

        Args:
            positions:  Current joint positions in radians, length ``n``.
            timestamps: Per-channel sample times in seconds (any common
                        epoch), length ``n``, or ``None`` for wall-clock.
        """
        if timestamps is not None:
            return self._differentiate_timestamped(positions, timestamps)

        now = time.perf_counter()

        if self._last_time is None or any(p is None for p in self._pos_prev):
            self._last_time = now
            self._pos_prev = list(positions)
            return [0.0] * self._n

        Ts = now - self._last_time
        self._last_time = now

        if Ts <= 0:
            return list(self._vel_prev)

        a = 1.0 / (1.0 + Ts * self._cutoff)
        b = a * self._cutoff

        velocities: list[float] = []
        for i in range(self._n):
            vel = self._vel_prev[i] * a + b * (positions[i] - self._pos_prev[i])  # type: ignore[operator]
            self._vel_prev[i] = vel
            self._pos_prev[i] = positions[i]
            velocities.append(vel)

        return velocities

    def _differentiate_timestamped(
        self, positions: list[float], timestamps: list[float]
    ) -> list[float]:
        if self._ts_prev is None:
            self._ts_prev = list(timestamps)
            self._pos_prev = list(positions)
            return [0.0] * self._n

        velocities: list[float] = []
        for i in range(self._n):
            Ts = timestamps[i] - self._ts_prev[i]
            if Ts <= 0:
                velocities.append(self._vel_prev[i])
                continue
            a = 1.0 / (1.0 + Ts * self._cutoff)
            b = a * self._cutoff
            vel = self._vel_prev[i] * a + b * (positions[i] - self._pos_prev[i])  # type: ignore[operator]
            self._vel_prev[i] = vel
            self._pos_prev[i] = positions[i]
            self._ts_prev[i] = timestamps[i]
            velocities.append(vel)

        return velocities
