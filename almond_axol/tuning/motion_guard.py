"""Unattended-run guard for replays: off-trajectory, oscillation, vibration.

``tune.motion`` replays run with nobody at the e-stop during overnight
tuning sweeps, where a gain under test can make a joint ring or buzz. The
contact watchdog only sees torque against the model, and the firmware-loop
deviation abort only covers ``a4`` / ``pv`` joints. :class:`MotionGuard`
watches every driven joint's tracking error (measured minus commanded, joint
frame) sample by sample and names the first joint that:

- **leaves its trajectory:** ``|error|`` above ``dev_deg`` for ``hold_s``;
- **oscillates:** the error's 0.3-15 Hz band above ``osc_deg`` RMS over
  ``window_s`` — the joint ringing at its impedance modes (1-10 Hz);
- **vibrates:** the error above 15 Hz over ``vib_deg`` RMS over
  ``window_s`` — a buzz.

The bands are one-pole splits of the error (a 0.3 Hz high-pass, a 15 Hz
split), so slow tracking lag is never mistaken for an oscillation. Limits are
in degrees. ``scale`` relaxes all three together, for the transit moves to
and from the motion, which run faster than the reference motions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Defaults, from the saved replays on the jelly robot (2026-10-06): normal
#: slow motions keep the 0.3-15 Hz error under ~0.15° RMS and the >15 Hz
#: under ~0.02°; the deviation is lag, under 1° at teleop speeds.
DEFAULT_DEV_DEG = 6.0
DEFAULT_OSC_DEG = 0.6
DEFAULT_VIB_DEG = 0.15
DEFAULT_WINDOW_S = 0.5
DEFAULT_HOLD_S = 0.05

_SLOW_HZ = 0.3
_SPLIT_HZ = 15.0


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
            "oscillation": "oscillating (0.3-15 Hz)",
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
        dt = 1.0 / rate
        self._a_slow = 1.0 - math.exp(-2.0 * math.pi * _SLOW_HZ * dt)
        self._a_split = 1.0 - math.exp(-2.0 * math.pi * _SPLIT_HZ * dt)
        self.reset()

    def reset(self) -> None:
        """Forget the filters (a new segment: the error restarts from its own value)."""
        n = len(self.names)
        self._primed = False
        self._seen = np.zeros(n, dtype=bool)
        self._slow = np.zeros(n)
        self._mid = np.zeros(n)
        self._osc_sq = np.zeros((self.window, n))
        self._vib_sq = np.zeros((self.window, n))
        self._k = 0
        self._filled = 0
        self._over = np.zeros(n, dtype=int)

    def update(self, error: np.ndarray, scale: float = 1.0) -> GuardTrip | None:
        """Feed one sample of tracking error (rad, one per name).

        Returns the first trip, or ``None``. NaN entries (a joint with no
        feedback yet) are skipped for that sample.
        """
        e = np.asarray(error, dtype=np.float64)
        ok = np.isfinite(e)
        if not self._primed:
            if not ok.any():
                return None
            self._slow[:] = np.where(ok, e, 0.0)
            self._mid[:] = self._slow
            self._seen = ok.copy()
            self._primed = True
        # A joint's first sample, or one after a gap, primes its filters
        # instead of stepping them: a missing reading is not a jump.
        fresh = ok & ~self._seen
        self._slow[fresh] = e[fresh]
        self._mid[fresh] = e[fresh]
        self._seen |= ok
        e = np.where(ok, e, self._mid)
        self._slow += np.where(ok, self._a_slow * (e - self._slow), 0.0)
        self._mid += np.where(ok, self._a_split * (e - self._mid), 0.0)
        band = self._mid - self._slow  # 0.3-15 Hz
        high = e - self._mid  # > 15 Hz
        i = self._k % self.window
        self._osc_sq[i] = np.where(ok, band * band, 0.0)
        self._vib_sq[i] = np.where(ok, high * high, 0.0)
        self._k += 1
        self._filled = min(self._filled + 1, self.window)

        over = ok & (np.abs(e) > self.dev * scale)
        self._over = np.where(over, self._over + 1, 0)
        hit = np.flatnonzero(self._over >= self.hold)
        if hit.size:
            j = int(hit[0])
            return GuardTrip(
                self.names[j],
                "deviation",
                math.degrees(abs(e[j])),
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
