"""Fit a joint's position-periodic torque (cogging / gear mesh) for cancellation.

The realtime core cancels it by feedforward (``JointConfig.cogging``,
``filter::cogging`` in the core): a Fourier series in the joint angle, fitted
here from a slow constant-speed sweep — ``axol tune.friction --raw-csv``,
where every cruise sample carries the angle, the torque the motor supplied
and the sweep direction.

Only the position-periodic part is wanted, so each direction's torque is
high-passed **in the angle domain** first: subtracting its moving average over
``detrend_deg`` removes gravity, friction (constant per direction at one
speed) and anything else slow in angle, and leaves the ripple. The window is
rounded to a whole number of fundamental periods — a boxcar that long has a
null at every harmonic, so none of the ripple leaks into the trend (a 6°
window on the 1.81° harmonic leaked 8% and inflated the fit by as much). The torque the
motor supplied to hold the sweep through a bump is the torque to *add*, so the
fit is used as-is — no sign flip.

On the right shoulder_1 the ripple sits at 1.81° and 0.905° (with a smaller
3.62° term): a 3.62° fundamental with harmonics 1, 2 and 4 covers it, which
is the default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..robot.config import CoggingModel

#: The right shoulder_1's series: 3.62° fundamental, harmonics at 3.62°,
#: 1.81° and 0.905° (2026-09-23 analysis of the 2026-09-18 sweep).
DEFAULT_PERIOD_DEG = 3.62
DEFAULT_HARMONICS = (1, 2, 4)

_GRID_DEG = 0.02


@dataclass(frozen=True)
class CoggingFit:
    """A fitted series and how well it explains the sweep's ripple."""

    model: CoggingModel
    #: Share of the detrended ripple the series explains (0..1).
    r2: float
    #: Amplitude (Nm) of each harmonic, in ``model.harmonics`` order.
    amplitudes: tuple[float, ...]
    #: RMS (Nm) of the detrended ripple the fit saw.
    ripple_rms: float
    samples: int


def angle_highpass(
    q_deg: np.ndarray, tau: np.ndarray, detrend_deg: float = 6.0
) -> tuple[np.ndarray, np.ndarray]:
    """``(q, ripple)`` for one sweep direction: the torque minus its moving
    average over ``detrend_deg`` of travel, on a fine angle grid.

    Returns empty arrays when the direction spans less than three windows —
    too short to separate a ripple from the trend.
    """
    q_deg = np.asarray(q_deg, dtype=float)
    tau = np.asarray(tau, dtype=float)
    ok = np.isfinite(q_deg) & np.isfinite(tau)
    q_deg, tau = q_deg[ok], tau[ok]
    if len(q_deg) < 10:
        return np.empty(0), np.empty(0)
    order = np.argsort(q_deg)
    qs, ts = q_deg[order], tau[order]
    grid = np.arange(qs[0], qs[-1], _GRID_DEG)
    n = max(1, int(round(detrend_deg / _GRID_DEG)))
    if len(grid) < 3 * n:
        return np.empty(0), np.empty(0)
    values = np.interp(grid, qs, ts)
    trend = np.convolve(values, np.ones(n) / n, mode="same")
    keep = slice(n // 2, len(grid) - n // 2)
    return grid[keep], (values - trend)[keep]


def _whole_periods(window_deg: float, period_deg: float) -> float:
    """``window_deg`` rounded to a whole number (≥ 1) of ``period_deg``."""
    return max(1, round(window_deg / period_deg)) * period_deg


def _signs(direction: np.ndarray) -> np.ndarray:
    """``+1`` / ``-1`` per sample from ``+`` / ``-`` tokens or signed numbers."""
    d = np.asarray(direction)
    if d.dtype.kind in "OUS":
        text = d.astype(str)
        return np.where(text == "+", 1.0, np.where(text == "-", -1.0, 0.0))
    return np.sign(d.astype(float))


def _features(q_deg: np.ndarray, period_deg: float, harmonics: tuple[int, ...]):
    cols = []
    for k in harmonics:
        w = 2.0 * math.pi * k / period_deg
        cols += [np.cos(w * q_deg), np.sin(w * q_deg)]
    return np.stack(cols, axis=1)


def fit_cogging(
    q_rad: np.ndarray,
    tau_nm: np.ndarray,
    direction: np.ndarray,
    *,
    period_deg: float = DEFAULT_PERIOD_DEG,
    harmonics: tuple[int, ...] = DEFAULT_HARMONICS,
    detrend_deg: float = 6.0,
) -> CoggingFit:
    """Fit the series to a sweep's samples.

    Args:
        q_rad: Joint angle of each sample (rad, joint frame).
        tau_nm: Torque the motor supplied (Nm).
        direction: Sweep direction per sample (``+`` / ``-``, or its sign).
        period_deg: The series' fundamental period.
        harmonics: Harmonic numbers to fit.
        detrend_deg: Angle-domain high-pass window (see :func:`angle_highpass`).

    Raises:
        ValueError: When no direction spans enough travel to fit.
    """
    if period_deg <= 0.0 or not harmonics or min(harmonics) < 1:
        raise ValueError("period_deg must be > 0 and harmonics >= 1")
    detrend_deg = _whole_periods(detrend_deg, period_deg)
    q_deg = np.degrees(np.asarray(q_rad, dtype=float))
    tau_nm = np.asarray(tau_nm, dtype=float)
    sign = _signs(direction)
    qs, rs = [], []
    for s in (1.0, -1.0):
        sel = sign == s
        q, r = angle_highpass(q_deg[sel], tau_nm[sel], detrend_deg)
        qs.append(q)
        rs.append(r)
    q = np.concatenate(qs)
    r = np.concatenate(rs)
    if len(q) < 50:
        raise ValueError(
            f"too little travel to fit: each sweep direction needs > "
            f"{3 * detrend_deg:g}° of cruise samples"
        )
    x = _features(q, period_deg, tuple(harmonics))
    coef, *_ = np.linalg.lstsq(x, r, rcond=None)
    resid = r - x @ coef
    var = float(np.var(r))
    r2 = 1.0 - float(np.var(resid)) / var if var > 0 else 0.0
    model = CoggingModel(
        period_deg=float(period_deg),
        harmonics=tuple(
            (int(k), float(coef[2 * i]), float(coef[2 * i + 1]))
            for i, k in enumerate(harmonics)
        ),
    )
    return CoggingFit(
        model=model,
        r2=r2,
        amplitudes=tuple(float(math.hypot(a, b)) for _, a, b in model.harmonics),
        ripple_rms=float(np.sqrt(np.mean(r**2))),
        samples=len(q),
    )


def prediction_r(
    model: CoggingModel,
    q_rad: np.ndarray,
    tau_nm: np.ndarray,
    direction: np.ndarray,
    detrend_deg: float = 6.0,
) -> float:
    """Correlation between the model and another sweep's detrended ripple —
    the out-of-sample check (a table that only fits its own pass cancels
    nothing)."""
    detrend_deg = _whole_periods(detrend_deg, model.period_deg)
    q_deg = np.degrees(np.asarray(q_rad, dtype=float))
    tau_nm = np.asarray(tau_nm, dtype=float)
    sign = _signs(direction)
    preds, ripples = [], []
    for s in (1.0, -1.0):
        sel = sign == s
        q, r = angle_highpass(q_deg[sel], tau_nm[sel], detrend_deg)
        if len(q):
            preds.append(np.asarray(model.torque(np.radians(q))))
            ripples.append(r)
    if not preds:
        return float("nan")
    p, r = np.concatenate(preds), np.concatenate(ripples)
    if np.std(p) == 0 or np.std(r) == 0:
        return float("nan")
    return float(np.corrcoef(p, r)[0, 1])
