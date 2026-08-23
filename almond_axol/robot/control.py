"""Motor control utilities: friction model, differentiator, contact watchdog.

Gravity compensation is handled separately — see
:class:`almond_axol.robot.gravity.GravityCompensator` — because the simple
per-joint ``ga·cos(q) + gb·sin(q)`` model used here previously ignored child
links and produced incorrect torques.
"""

from __future__ import annotations

import math
import time

from ..constants import ARM_JOINTS

# Cutoff frequency for the velocity differentiator low-pass filter (Hz)
CUTOFF_FREQ = 20.0

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


class Differentiator:
    """First-order low-pass differentiator, matching C++ Differentiator::Differentiate.

    For each channel:
        a = 1 / (1 + Ts * CUTOFF_FREQ)
        b = a * CUTOFF_FREQ
        vel[i] = vel_prev[i] * a + b * (pos[i] - pos_prev[i])

    Args:
        n: Number of channels to differentiate.
    """

    def __init__(self, n: int) -> None:
        """Initialize the differentiator.

        Args:
            n: Number of independent channels to differentiate simultaneously.
        """
        self._n = n
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

        a = 1.0 / (1.0 + Ts * CUTOFF_FREQ)
        b = a * CUTOFF_FREQ

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
            a = 1.0 / (1.0 + Ts * CUTOFF_FREQ)
            b = a * CUTOFF_FREQ
            vel = self._vel_prev[i] * a + b * (positions[i] - self._pos_prev[i])  # type: ignore[operator]
            self._vel_prev[i] = vel
            self._pos_prev[i] = positions[i]
            self._ts_prev[i] = timestamps[i]
            velocities.append(vel)

        return velocities
