"""Iterative learning control for a repeated reference motion.

slow_osc's 1-3 Hz shake is 95-98% the same on every pass (2026-09-26: the
pass-averaged residual of three passes, encoders through FK and the wrist
IMU alike), yet only 3-9% of it is a *linear* response to the command — it
is friction and load, arriving at the same moments of the trajectory each
time. A deterministic error like that is what a learned command correction
removes and damping gains do not.

:class:`CommandLearner` keeps a per-joint position offset on the streamed
trajectory and updates it after each pass from that pass's tracking error:

- The error and the offset live in a band (default 0.7-8 Hz, zero-phase
  FFT). The deliberate motion below it — and with it the ~100 ms impedance
  lag, whose correction would be a different trajectory, not a smoother one
  — is left alone.
- Update ``U ← U + β·Ĝ⁻¹·E``, where ``Ĝ`` is the joint's command → position
  response over the band, measured from the passes themselves: each pass
  pair gives ``ΔA = Ĝ·ΔU``, and the offsets are what excite the band. With no
  estimate yet (after the baseline pass) the first step is the error advanced
  by the joint's tracking lag at a low gain (:data:`FIRST_GAIN`).
- ``Ĝ⁻¹`` is regularised (Wiener, :data:`WIENER_EPS`) so frequencies the
  offsets have not excited are not amplified.
- The offset is clamped (``max_rad``), and a pass whose band error rose
  above :data:`WORSE_RATIO` of the best so far rolls back to the best
  offset with the gain halved.

The learned offset belongs to *one* motion; it is the measurement of what
each joint needed, and the ceiling a model-based feedforward can aim for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: Default learning band (Hz).
LEARN_BAND = (0.7, 8.0)
#: Gain of the first, model-free step (the error advanced by the lag).
FIRST_GAIN = 0.3
#: Regularisation of the response inverse, relative to the largest |Ĝ|².
WIENER_EPS = 0.05
#: Half-width (Hz) of the frequency smoothing applied to the cross-spectra.
SMOOTH_HZ = 0.2
#: A pass whose band error exceeds the best so far by this ratio rolls back.
WORSE_RATIO = 1.15
#: Longest tracking lag searched for the first step (s).
MAX_LAG_S = 0.25


def band_limit(x: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    """``x`` (``(N,)`` or ``(N, C)``) with everything outside ``band`` removed,
    zero-phase. The line joining the ends comes off first so the FFT's
    wrap-around does not turn a start/end mismatch into in-band content."""
    x = np.asarray(x, dtype=float)
    squeeze = x.ndim == 1
    x2 = x[:, None] if squeeze else x
    n = len(x2)
    ramp = np.linspace(0.0, 1.0, n)[:, None]
    detrended = x2 - (x2[:1] + (x2[-1:] - x2[:1]) * ramp)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    mask = ((f >= band[0]) & (f <= band[1]))[:, None]
    out = np.fft.irfft(np.fft.rfft(detrended, axis=0) * mask, n, axis=0)
    return out[:, 0] if squeeze else out


def tracking_lag(ref: np.ndarray, actual: np.ndarray, fs: float) -> float:
    """Delay (s, 0 to :data:`MAX_LAG_S`) that best lines ``actual`` up with
    ``ref`` by velocity."""
    vr, va = np.gradient(ref), np.gradient(actual)
    best, best_k = -math.inf, 0
    for k in range(int(MAX_LAG_S * fs) + 1):
        score = float(np.dot(vr[: len(vr) - k], va[k:]))
        if score > best:
            best, best_k = score, k
    return best_k / fs


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return x
    kernel = np.ones(width) / width
    return np.convolve(x, kernel, mode="same")


@dataclass
class PassReport:
    """What one :meth:`CommandLearner.update` saw and did."""

    band_rms: np.ndarray  # (C,) band error RMS per column (rad), NaN = not learned
    total: float  # RMS over the learned columns (rad)
    rolled_back: bool
    gain: float
    step: str  # "first", "model" or "rollback"


@dataclass
class CommandLearner:
    """Per-column position offsets learned over repeated passes.

    Args:
        n: Samples per pass (one per streamed waypoint).
        fs: Waypoint rate (Hz).
        columns: Which of the columns learn (bool mask); the others keep a
            zero offset.
        band: Learning band (Hz).
        gain: β of the model-based step.
        max_rad: Clamp on the offset.
    """

    n: int
    fs: float
    columns: np.ndarray
    band: tuple[float, float] = LEARN_BAND
    gain: float = 0.5
    max_rad: float = math.radians(1.5)
    offset: np.ndarray = field(init=False)
    _history: list[tuple[np.ndarray, np.ndarray]] = field(init=False)
    _best: tuple[float, np.ndarray] | None = field(init=False, default=None)
    _lags: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.columns = np.asarray(self.columns, dtype=bool)
        self.offset = np.zeros((self.n, len(self.columns)))
        self._history = []

    def update(self, ref: np.ndarray, actual: np.ndarray) -> PassReport:
        """Take a pass flown with the current :attr:`offset` (``ref`` the
        clean reference, ``actual`` the measured positions, ``(n, C)``) and
        compute the next offset."""
        ref = np.asarray(ref, dtype=float)[: self.n]
        actual = np.asarray(actual, dtype=float)[: self.n]
        if len(ref) < self.n:
            raise ValueError(f"pass has {len(ref)} samples, expected {self.n}")
        cols = np.where(self.columns & np.all(np.isfinite(actual), axis=0))[0]
        err = np.zeros_like(ref)
        err[:, cols] = band_limit(ref[:, cols] - actual[:, cols], self.fs, self.band)
        band_rms = np.full(len(self.columns), math.nan)
        band_rms[cols] = np.sqrt(np.mean(err[:, cols] ** 2, axis=0))
        total = float(np.sqrt(np.mean(err[:, cols] ** 2))) if len(cols) else 0.0

        applied = self.offset.copy()
        # Every pass informs the response estimate — a bad one included.
        motion = np.zeros_like(actual)
        motion[:, cols] = band_limit(actual[:, cols], self.fs, self.band)
        self._history.append((applied, motion))
        if self._best is not None and total > WORSE_RATIO * self._best[0]:
            # Worse than the best pass: back to its offset, gentler steps.
            self.offset = self._best[1].copy()
            self.gain *= 0.5
            return PassReport(band_rms, total, True, self.gain, "rollback")
        if self._best is None or total < self._best[0]:
            self._best = (total, applied.copy())

        step = np.zeros_like(self.offset)
        informative = any(
            np.any(u1 - u0)
            for (u0, _), (u1, _) in zip(self._history, self._history[1:])
        )
        if not informative:
            if self._lags is None:
                self._lags = np.array(
                    [
                        tracking_lag(ref[:, c], actual[:, c], self.fs)
                        for c in range(ref.shape[1])
                    ]
                )
            for c in cols:
                k = int(round(self._lags[c] * self.fs))
                # Advance: the command must lead the error by the lag.
                step[:, c] = FIRST_GAIN * np.concatenate([err[k:, c], np.zeros(k)])
            kind = "first"
        else:
            for c in cols:
                g = self._response(c)
                spec = np.fft.rfft(err[:, c])
                inv = np.conj(g) / (
                    np.abs(g) ** 2 + WIENER_EPS * np.max(np.abs(g) ** 2)
                )
                step[:, c] = self.gain * np.fft.irfft(spec * inv, self.n)
            kind = "model"
        step[:, cols] = band_limit(step[:, cols], self.fs, self.band)
        self.offset = np.clip(self.offset + step, -self.max_rad, self.max_rad)
        self.offset[:, ~self.columns] = 0.0
        return PassReport(band_rms, total, False, self.gain, kind)

    @property
    def best(self) -> np.ndarray:
        """The offset of the lowest-error pass so far (zeros before any)."""
        return (
            self._best[1].copy()
            if self._best is not None
            else np.zeros_like(self.offset)
        )

    def _response(self, c: int) -> np.ndarray:
        """Ĝ(f) for column ``c``: pooled over every pass pair,
        ``Σ ΔA·ΔU* / Σ |ΔU|²``, smoothed over :data:`SMOOTH_HZ`."""
        f = np.fft.rfftfreq(self.n, 1.0 / self.fs)
        cross = np.zeros(len(f), dtype=complex)
        power = np.zeros(len(f))
        for (u0, a0), (u1, a1) in zip(self._history, self._history[1:]):
            du = np.fft.rfft(band_limit(u1[:, c] - u0[:, c], self.fs, self.band))
            da = np.fft.rfft(a1[:, c] - a0[:, c])
            cross += da * np.conj(du)
            power += np.abs(du) ** 2
        width = max(1, int(round(2 * SMOOTH_HZ / (f[1] - f[0])))) if len(f) > 1 else 1
        cross = _smooth(cross.real, width) + 1j * _smooth(cross.imag, width)
        power = _smooth(power, width)
        g = np.where(
            power > 1e-12 * max(power.max(), 1e-30),
            cross / np.maximum(power, 1e-30),
            0.0,
        )
        g[(f < self.band[0]) | (f > self.band[1])] = 0.0
        return g
