"""Linear tracking dynamics of a joint: chirp identification and inversion.

The Reforge-style half of motion calibration: excite one joint with a sine
sweep through the production controller, measure its command → position
frequency response, fit a small transfer function, and pre-compensate a
known trajectory with its inverse. It removes the *linear* part of the
tracking error — the lag and the closed-loop resonance. On slow_osc that is
the minority (3-9% of the 1-3 Hz shake was a linear response to the
command, 2026-09-26; the rest is friction, see
:mod:`almond_axol.tuning.friction_model`), but it is the part a friction
model does not touch, and it is the dominant error of faster motion.

- :func:`chirp_motion` builds the excitation: a logarithmic sweep around a
  pose, amplitude capped by acceleration so high frequencies stay gentle,
  optionally riding a slow triangle "carrier" so the joint keeps sliding one
  way (friction linearises; a bare sine reverses 2f times a second).
- :func:`frequency_response` estimates ``H(f)`` (H1: cross over auto
  spectrum) and its coherence from a replay.
- :func:`fit_tracking_model` fits ``H(s) = K (1 + s/ωz) ωn² / (s² + 2ζωn s
  + ωn²) · e^{−sτ}`` to it, weighted by coherence.
- :func:`invert_reference` pre-compensates a whole known trajectory in the
  frequency domain (non-causal: tune.motion knows the motion in advance),
  with the inverse's gain capped and rolled off above the identified band.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .motion import ReferenceMotion

#: The inversion's correction fades in and out over this long at each end.
_END_TAPER_S = 0.5
#: Where ``tune.tf --save`` keeps the fitted models, keyed ``side.joint``.
TRACKING_MODELS_PATH = Path.home() / ".almond" / "tracking_models.json"


def chirp_motion(
    base: np.ndarray,
    column: int,
    *,
    f0: float = 0.3,
    f1: float = 8.0,
    duration: float = 60.0,
    amp_rad: float = math.radians(1.5),
    acc_max: float = math.radians(200.0),
    carrier_rad_s: float = 0.0,
    carrier_span_rad: float = math.radians(15.0),
    rate: float = 240.0,
    fade_s: float = 2.0,
    settle_s: float = 1.5,
    name: str = "chirp",
) -> ReferenceMotion:
    """A logarithmic sine sweep on column ``column`` of pose ``base`` (14).

    Amplitude ``min(amp_rad, acc_max / (2πf)²)``: full amplitude at low
    frequency, constant acceleration above the corner — at the defaults
    1.5° up to 1.8 Hz, 0.08° at 8 Hz. ``carrier_rad_s`` > 0 adds a triangle
    wave of that speed over ±``carrier_span_rad/2`` so the joint slides one
    way for seconds at a time. Faded in and out; held still at both ends.
    """
    n_sweep = int(duration * rate)
    t = np.arange(n_sweep) / rate
    k = math.log(f1 / f0) / duration
    f = f0 * np.exp(k * t)
    phase = 2 * math.pi * f0 * (np.exp(k * t) - 1.0) / k
    amp = np.minimum(amp_rad, acc_max / (2 * math.pi * f) ** 2)
    fade = np.ones(n_sweep)
    nf = int(fade_s * rate)
    if nf > 0:
        ramp = 0.5 * (1 - np.cos(np.linspace(0, math.pi, nf)))
        fade[:nf] = ramp
        fade[-nf:] = ramp[::-1]
    x = amp * fade * np.sin(phase)
    if carrier_rad_s > 0:
        period = 2 * carrier_span_rad / carrier_rad_s
        # Triangle from 0 through ±span/2, centred on the base pose.
        tri = (
            (carrier_span_rad / 2)
            * (2 / math.pi)
            * np.arcsin(np.sin(2 * math.pi * t / period))
        )
        x = x + tri * fade
    hold = np.zeros(int(settle_s * rate))
    x = np.concatenate([hold, x, hold])
    q = np.tile(np.asarray(base, dtype=np.float32), (len(x), 1))
    q[:, column] += x.astype(np.float32)
    meta = {
        "source": "synthetic: chirp",
        "column": int(column),
        "f0": f0,
        "f1": f1,
        "duration": duration,
        "amp_deg": math.degrees(amp_rad),
        "acc_max_deg_s2": math.degrees(acc_max),
        "carrier_deg_s": math.degrees(carrier_rad_s),
    }
    return ReferenceMotion(name=name, q=q, rate=rate, meta=meta)


#: Welch segments at least: with one, coherence is identically 1.
_MIN_SEGMENTS = 8
#: Bins whose input power is below this fraction of the peak carry no
#: excitation — their "response" is the output's own noise over nothing.
_MIN_INPUT_POWER = 1e-5


def _welch_len(fs: float, f0: float, n: int) -> int:
    """Segment length: long enough for ~3 cycles of ``f0``, short enough for
    :data:`_MIN_SEGMENTS` half-overlapping segments."""
    return int(max(min(3.0 * fs / max(f0, 1e-3), 2 * n // (_MIN_SEGMENTS + 1)), fs))


def frequency_response(
    ref: np.ndarray, out: np.ndarray, fs: float, band: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(f, H, coherence)`` of ``out`` to ``ref`` within ``band`` (H1
    estimator, Welch, Hann). Slow trends — the carrier, drift — are removed
    by a zero-phase high-pass at a third of the band's low edge first."""
    from scipy.signal import butter, coherence, csd, sosfiltfilt, welch

    ref = np.asarray(ref, dtype=float)
    out = np.asarray(out, dtype=float)
    hp = butter(2, max(band[0] / 3.0, 0.02), btype="high", fs=fs, output="sos")
    r = sosfiltfilt(hp, ref - ref.mean())
    y = sosfiltfilt(hp, out - out.mean())
    nper = _welch_len(fs, band[0], len(r))
    f, prr = welch(r, fs=fs, nperseg=nper)
    _, pry = csd(r, y, fs=fs, nperseg=nper)
    _, coh = coherence(r, y, fs=fs, nperseg=nper)
    sel = (f >= band[0]) & (f <= band[1]) & (prr > _MIN_INPUT_POWER * prr.max())
    return f[sel], pry[sel] / prr[sel], coh[sel]


@dataclass(frozen=True)
class TrackingModel:
    """``H(s) = K (1 + s/ωz) ωn² / (s² + 2ζωn s + ωn²) · e^{−sτ}``."""

    k: float
    wn: float  # rad/s
    zeta: float
    wz: float  # rad/s; inf = no zero
    tau: float  # s
    f_lo: float  # identified band (Hz)
    f_hi: float
    fit_rms: float  # |H - Ĥ| RMS over the band, coherence-weighted
    gains: dict[str, float] | None = None  # the controller it was measured on

    @property
    def fn_hz(self) -> float:
        return self.wn / (2 * math.pi)

    def response(self, f: np.ndarray) -> np.ndarray:
        s = 2j * math.pi * np.asarray(f, dtype=float)
        zero = 1.0 + (s / self.wz if math.isfinite(self.wz) else 0.0)
        return (
            self.k
            * zero
            * self.wn**2
            / (s**2 + 2 * self.zeta * self.wn * s + self.wn**2)
            * np.exp(-s * self.tau)
        )

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["wz"] = None if not math.isfinite(self.wz) else self.wz
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrackingModel":
        d = dict(d)
        d["wz"] = math.inf if d.get("wz") is None else float(d["wz"])
        return cls(**d)


def fit_tracking_model(
    f: np.ndarray, h: np.ndarray, coh: np.ndarray, min_coherence: float = 0.6
) -> TrackingModel:
    """Fit the model to a measured response over the bins with coherence at
    least ``min_coherence`` (complex least squares, coherence-weighted)."""
    from scipy.optimize import least_squares

    keep = coh >= min_coherence
    if keep.sum() < 6:
        raise ValueError(
            f"only {int(keep.sum())} frequency bins with coherence ≥ "
            f"{min_coherence} — the excitation did not get through"
        )
    f, h, w = f[keep], h[keep], np.sqrt(coh[keep])
    s = 2j * math.pi * f

    def model(x: np.ndarray) -> np.ndarray:
        k, lwn, zeta, iwz, tau = x
        wn = math.exp(lwn)
        zero = 1.0 + s * iwz
        return k * zero * wn**2 / (s**2 + 2 * zeta * wn * s + wn**2) * np.exp(-s * tau)

    def resid(x: np.ndarray) -> np.ndarray:
        e = (model(x) - h) * w
        return np.concatenate([e.real, e.imag])

    # Start the natural frequency at the magnitude peak (or the band top).
    mag = np.abs(h)
    f_peak = float(f[np.argmax(mag)]) if mag.max() > 1.05 * mag[0] else float(f[-1])
    best = None
    for zeta0 in (0.15, 0.4, 0.8):
        for tau0 in (0.0, 0.03, 0.08):
            x0 = [float(mag[0]), math.log(2 * math.pi * f_peak), zeta0, 0.0, tau0]
            res = least_squares(
                resid,
                x0,
                bounds=(
                    [0.2, math.log(2 * math.pi * 0.2), 0.01, -0.2, 0.0],
                    [5.0, math.log(2 * math.pi * 40.0), 3.0, 0.2, 0.3],
                ),
            )
            if best is None or res.cost < best.cost:
                best = res
    assert best is not None
    k, lwn, zeta, iwz, tau = (float(v) for v in best.x)
    wz = 1.0 / iwz if abs(iwz) > 1e-6 else math.inf
    err = (model(best.x) - h) * w
    return TrackingModel(
        k=k,
        wn=math.exp(lwn),
        zeta=zeta,
        wz=wz,
        tau=tau,
        f_lo=float(f[0]),
        f_hi=float(f[-1]),
        fit_rms=float(np.sqrt(np.mean(np.abs(err) ** 2))),
    )


def invert_reference(
    ref: np.ndarray,
    fs: float,
    model: TrackingModel,
    *,
    max_gain: float = 3.0,
    f_max: float | None = None,
) -> np.ndarray:
    """``ref`` (``(N,)``) pre-compensated by the model's inverse.

    Frequency domain over the whole trajectory (known in advance): the
    reference, less the line joining its ends (so the FFT sees no jump),
    times ``1/H(f)`` — its magnitude capped at ``max_gain``, and faded to
    ``1`` above ``f_max`` (default the identified band's top) where the model
    is not known. The model's static gain ``K`` is divided out: a gain error
    at DC is gravity's and friction's business, not the tracker's.
    """
    ref = np.asarray(ref, dtype=float)
    n = len(ref)
    line = np.linspace(ref[0], ref[-1], n)
    x = ref - line
    # Mirror-extend so the ends meet smoothly (no wrap-around edge).
    ext = np.concatenate([x, -x[::-1]])
    spec = np.fft.rfft(ext)
    f = np.fft.rfftfreq(len(ext), 1.0 / fs)
    inv = np.ones(len(f), dtype=complex)
    # Unit static gain: K is the band's average gain, and scaling the whole
    # deliberate motion by 1/K is not the tracker's correction to make.
    h = model.response(f[1:]) / model.k
    inv[1:] = 1.0 / h
    inv[0] = 1.0
    mag = np.abs(inv)
    over = mag > max_gain
    inv[over] *= max_gain / mag[over]
    top = model.f_hi if f_max is None else f_max
    # Raised-cosine fade from the inverse to 1 over [top, 1.5·top].
    fade = np.clip((f - top) / (0.5 * top), 0.0, 1.0)
    w = 0.5 * (1 + np.cos(math.pi * fade))
    inv = w * inv + (1 - w) * 1.0
    out = np.fft.irfft(spec * inv, len(ext))[:n] + line
    # Fade the correction in and out over the first and last _END_TAPER_S so
    # the stream still starts and ends exactly on the motion (the approach
    # is planned to its first pose; a lead-corrected first sample would be a
    # step off it).
    taper = np.ones(n)
    m = min(int(_END_TAPER_S * fs), n // 4)
    if m > 0:
        ramp = 0.5 * (1 - np.cos(np.linspace(0, math.pi, m)))
        taper[:m] = ramp
        taper[-m:] = ramp[::-1]
    return ref + taper * (out - ref)


def load_models(path: Path = TRACKING_MODELS_PATH) -> dict[str, TrackingModel]:
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    return {k: TrackingModel.from_dict(v) for k, v in raw.get("joints", {}).items()}


def save_model(
    key: str, model: TrackingModel, path: Path = TRACKING_MODELS_PATH
) -> Path:
    try:
        raw = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        raw = {}
    raw.setdefault("joints", {})[key] = model.as_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return path


class TrackingFilter:
    """A :class:`TrackingModel` run causally, one sample at a time: the
    position a joint is expected to reach given what it was commanded.

    Bilinear (Tustin) discretisation of the second-order section and its
    zero at ``fs``, the delay rounded to whole samples. The DC gain is pinned
    to 1 unless ``keep_gain``: a model fitted over 0.3-6 Hz extrapolates its
    ``k`` to the deliberate motion below that, and a 7% "steady-state error"
    on a 100° stroke is not what the joint does. The state starts at rest at
    the first sample.
    """

    def __init__(self, model: TrackingModel, fs: float, keep_gain: bool = False):
        c = 2.0 * fs
        wn, z = model.wn, model.zeta
        r = c / model.wz if math.isfinite(model.wz) else 0.0
        k = model.k if keep_gain else 1.0
        # num(z): k wn² [r (z² - 1) + (z + 1)²]; den(z): c²(z-1)² +
        # 2ζwn c (z²-1) + wn²(z+1)², both over (z+1)².
        b = k * wn**2 * np.array([r + 1.0, 2.0, 1.0 - r])
        a = np.array(
            [
                c * c + 2 * z * wn * c + wn**2,
                -2 * c * c + 2 * wn**2,
                c * c - 2 * z * wn * c + wn**2,
            ]
        )
        self._b = b / a[0]
        self._a = a / a[0]
        self._delay = max(0, int(round(model.tau * fs)))
        self._buf: list[float] = []
        self._x = [0.0, 0.0]
        self._y = [0.0, 0.0]
        self._started = False

    def step(self, x: float) -> float:
        x = float(x)
        if not self._started:
            # At rest at the first sample: the steady state of the section.
            gain = float(self._b.sum() / self._a.sum())
            self._x = [x, x]
            self._y = [x * gain, x * gain]
            self._buf = [x] * self._delay
            self._started = True
        if self._delay:
            self._buf.append(x)
            x = self._buf.pop(0)
        b, a = self._b, self._a
        y = b[0] * x + b[1] * self._x[0] + b[2] * self._x[1]
        y -= a[1] * self._y[0] + a[2] * self._y[1]
        self._x = [x, self._x[0]]
        self._y = [y, self._y[0]]
        return y

    def run(self, xs: np.ndarray) -> np.ndarray:
        return np.array([self.step(x) for x in np.asarray(xs, dtype=float)])
