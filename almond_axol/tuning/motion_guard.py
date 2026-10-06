"""Unattended-run guard for replays: off-trajectory, oscillation, vibration.

``tune.motion`` replays run with nobody at the e-stop during overnight
tuning sweeps, where a gain under test can make a joint ring or buzz. The
contact watchdog only sees torque against the model, and the firmware-loop
deviation abort only covers ``a4`` / ``pv`` joints. :class:`MotionGuard`
watches every driven joint's tracking error (measured minus commanded, joint
frame) sample by sample and names the first joint that:

- **leaves its trajectory:** ``|error|`` above ``dev_deg`` for ``hold_s``;
- **oscillates:** the error's 1.5-15 Hz band above ``osc_deg`` RMS over
  ``window_s`` — the joint ringing at its impedance or structural modes;
- **vibrates:** the error above 15 Hz over ``vib_deg`` RMS over
  ``window_s`` — a buzz.

The bands are 2nd-order Butterworth sections. The 1.5 Hz low edge keeps the
lag of the motion itself out of the oscillation check: ``slow_osc`` swings
the elbow at up to 100°/s, and its tracking lag fills a 0.3-15 Hz band to
~4.6° RMS on a healthy run, but a 1.5-15 Hz band to only ~0.37° — against
2.0-2.4° on the IMU-damping runs that rang (2026-10-01). Limits are in
degrees; ``scale`` relaxes all three together, for the transit moves to and
from the motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Defaults, sized on the jelly robot's 490 saved replays (2026-10-06):
#: healthy runs peak at ~7° deviation (slow_osc's elbow at 100°/s; 99th
#: percentile 8.7°), 0.87° RMS at 1.5-15 Hz (99th percentile; a joint chirp,
#: which drives that band on purpose, 1.05°) and 0.10° above 15 Hz; the runs
#: that rang reached 11-14°, 2.0-2.4° and 0.11-0.16°. The vibration limit has margin over a
#: recorded slow_osc's ~0.10° at shoulder_1's 75°/s peak (the core trace
#: adds jitter): it is the backstop for a violent buzz, the oscillation
#: check catches the rings.
DEFAULT_DEV_DEG = 10.0
DEFAULT_OSC_DEG = 1.5
DEFAULT_VIB_DEG = 0.15
DEFAULT_WINDOW_S = 0.5
DEFAULT_HOLD_S = 0.05

_OSC_LO_HZ = 1.5
_SPLIT_HZ = 15.0


class _Biquad:
    """A 2nd-order Butterworth section (RBJ), vectorised over joints."""

    def __init__(self, kind: str, hz: float, rate: float, n: int) -> None:
        w0 = 2.0 * math.pi * min(hz, 0.45 * rate) / rate
        alpha = math.sin(w0) / (2.0 * math.sqrt(0.5))
        cw = math.cos(w0)
        if kind == "low":
            b = ((1 - cw) / 2, 1 - cw, (1 - cw) / 2)
        else:
            b = ((1 + cw) / 2, -(1 + cw), (1 + cw) / 2)
        a0 = 1 + alpha
        self.b = tuple(x / a0 for x in b)
        self.a = (-2 * cw / a0, (1 - alpha) / a0)
        self.z1 = np.zeros(n)
        self.z2 = np.zeros(n)

    def step(self, x: np.ndarray, ok: np.ndarray) -> np.ndarray:
        """Transposed direct form II; joints without a sample keep their state."""
        y = self.b[0] * x + self.z1
        z1 = self.b[1] * x - self.a[0] * y + self.z2
        z2 = self.b[2] * x - self.a[1] * y
        self.z1 = np.where(ok, z1, self.z1)
        self.z2 = np.where(ok, z2, self.z2)
        return np.where(ok, y, 0.0)

    def prime(self, x: np.ndarray, mask: np.ndarray) -> None:
        """Start ``mask`` joints at steady state on ``x`` (no step transient)."""
        # Steady state for a constant input x: y = H(1)·x, H(1) = 1 (low) / 0
        # (high), with z1 = y - b0·x and z2 = b2·x - a2·y.
        dc = 1.0 if self.b[1] > 0 else 0.0
        y = dc * x
        self.z1 = np.where(mask, y - self.b[0] * x, self.z1)
        self.z2 = np.where(mask, self.b[2] * x - self.a[1] * y, self.z2)


@dataclass(frozen=True)
class GuardTrip:
    """Why the guard stopped a replay."""

    joint: str
    kind: str  # "deviation" | "oscillation" | "vibration"
    value_deg: float
    limit_deg: float

    def __str__(self) -> str:
        what = {
            "deviation": "left its trajectory",
            "oscillation": "oscillating (1.5-15 Hz)",
            "vibration": "vibrating (> 15 Hz)",
        }[self.kind]
        unit = "°" if self.kind == "deviation" else "° RMS"
        return (
            f"{self.joint} {what}: {self.value_deg:.2f}{unit} "
            f"(limit {self.limit_deg:.2f}{unit})"
        )


class MotionGuard:
    """Sample-by-sample tracking guard over ``names`` (one per column)."""

    def __init__(
        self,
        names: list[str],
        rate: float,
        *,
        dev_deg: float = DEFAULT_DEV_DEG,
        osc_deg: float = DEFAULT_OSC_DEG,
        vib_deg: float = DEFAULT_VIB_DEG,
        window_s: float = DEFAULT_WINDOW_S,
        hold_s: float = DEFAULT_HOLD_S,
    ) -> None:
        if rate <= 0.0:
            raise ValueError(f"rate must be positive, got {rate}")
        self.names = list(names)
        self.rate = float(rate)
        self.dev = math.radians(dev_deg)
        self.osc = math.radians(osc_deg)
        self.vib = math.radians(vib_deg)
        self.window = max(1, int(round(window_s * rate)))
        self.hold = max(1, int(round(hold_s * rate)))
        self.reset()

    def reset(self) -> None:
        """Forget the filters (a new segment: the error restarts from its own value)."""
        n = len(self.names)
        self._seen = np.zeros(n, dtype=bool)
        self._osc_hp = _Biquad("high", _OSC_LO_HZ, self.rate, n)
        self._osc_lp = _Biquad("low", _SPLIT_HZ, self.rate, n)
        self._vib_hp = _Biquad("high", _SPLIT_HZ, self.rate, n)
        self._osc_sq = np.zeros((self.window, n))
        self._vib_sq = np.zeros((self.window, n))
        self._k = 0
        self._filled = 0
        self._over = np.zeros(n, dtype=int)

    def update(self, error: np.ndarray, scale: float = 1.0) -> GuardTrip | None:
        """Feed one sample of tracking error (rad, one per name).

        Returns the first trip, or ``None``. NaN entries (a joint with no
        feedback yet, or a missed sample) are skipped: the joint's filters
        hold, and its first sample after a gap starts them afresh rather than
        reading the gap as a jump.
        """
        e = np.asarray(error, dtype=np.float64)
        ok = np.isfinite(e)
        x = np.where(ok, e, 0.0)
        fresh = ok & ~self._seen
        if fresh.any():
            for f in (self._osc_hp, self._vib_hp):
                f.prime(x, fresh)
            self._osc_lp.prime(np.zeros_like(x), fresh)
        self._seen = ok
        band = self._osc_lp.step(self._osc_hp.step(x, ok), ok)
        high = self._vib_hp.step(x, ok)
        i = self._k % self.window
        self._osc_sq[i] = band * band
        self._vib_sq[i] = high * high
        self._k += 1
        self._filled = min(self._filled + 1, self.window)

        over = ok & (np.abs(x) > self.dev * scale)
        self._over = np.where(over, self._over + 1, 0)
        hit = np.flatnonzero(self._over >= self.hold)
        if hit.size:
            j = int(hit[0])
            return GuardTrip(
                self.names[j],
                "deviation",
                math.degrees(abs(x[j])),
                math.degrees(self.dev * scale),
            )
        if self._filled < self.window:
            return None
        osc = np.sqrt(self._osc_sq.mean(axis=0))
        vib = np.sqrt(self._vib_sq.mean(axis=0))
        for kind, rms, lim in (
            ("oscillation", osc, self.osc * scale),
            ("vibration", vib, self.vib * scale),
        ):
            hit = np.flatnonzero(rms > lim)
            if hit.size:
                j = int(hit[0])
                return GuardTrip(
                    self.names[j], kind, math.degrees(rms[j]), math.degrees(lim)
                )
        return None

    def peaks(self, error: np.ndarray) -> dict[str, np.ndarray]:
        """Offline: run a whole ``(N, n)`` error series, return each check's peak.

        For sizing the limits against saved runs — never trips, reports the
        largest deviation and windowed band RMS each joint reached (rad).
        """
        self.reset()
        err = np.asarray(error, dtype=np.float64)
        dev = np.zeros(err.shape[1])
        osc = np.zeros(err.shape[1])
        vib = np.zeros(err.shape[1])
        for row in err:
            self.update(row, scale=math.inf)
            dev = np.maximum(dev, np.where(np.isfinite(row), np.abs(row), 0.0))
            if self._filled >= self.window:
                osc = np.maximum(osc, np.sqrt(self._osc_sq.mean(axis=0)))
                vib = np.maximum(vib, np.sqrt(self._vib_sq.mean(axis=0)))
        return {"deviation": dev, "oscillation": osc, "vibration": vib}
