"""Tracking-accuracy and smoothness metrics shared by every tuning suite.

Two families:

* Log-based metrics operating on the per-sample dict logs produced by the
  hardware runners in :mod:`.runner` (``sine_metrics``, ``step_metrics``,
  ``chatter_metrics``) — these carry the candidate-ranking ``score``.
* Array-based metrics operating on plain ``(t, target, actual)`` series
  (``tracking_metrics``, ``band_rms``) — used by the reference-motion replay
  and the offline analysis suites, and reported alongside the graphs so a
  claim like "kd X tracks better" is backed by numbers.

Band convention (shared with ``axol diag.teleop-jitter``): below
``BAND_LOW`` is intentional motion, ``BAND_LOW``–``BAND_HIGH`` is the
structural band the arm responds to ("felt" jitter / vibration), above
``BAND_HIGH`` is noise the mechanics mostly filter out.
"""

from __future__ import annotations

import math

import numpy as np

BAND_LOW = 3.0
BAND_HIGH = 15.0

# Everything above BUZZ_LOW is treated as buzz: sustained content there is
# inaudible-motor territory (limit cycles, near-clamp kd instability, gear
# chatter) rather than commanded motion, which the 6 Hz-cutoff references
# and the trapezoid keep far below it.
BUZZ_LOW = 20.0


# --------------------------------------------------------------------- #
# Array-based metrics (uniform or timestamped series)                    #
# --------------------------------------------------------------------- #


def _resample_uniform(t: np.ndarray, x: np.ndarray) -> tuple[float, np.ndarray]:
    """Resample one channel onto a uniform grid at the stream's median rate."""
    dt = float(np.median(np.diff(t)))
    n = int((t[-1] - t[0]) / dt) + 1
    grid = t[0] + dt * np.arange(n)
    return 1.0 / dt, np.interp(grid, t, x)


def band_rms(t: np.ndarray, x: np.ndarray) -> dict[str, float]:
    """Spectral RMS split into the low/mid/high bands, plus the dominant peak.

    Returns ``{"rms_low", "rms_mid", "rms_high", "peak_hz", "peak_rms"}`` in
    the input's units; ``peak_hz``/``peak_rms`` are NaN when no spectral line
    stands clear (8×) of the local noise floor. ``rms_mid`` (3–15 Hz) is the
    "felt jitter" number.
    """
    nan = {
        k: math.nan for k in ("rms_low", "rms_mid", "rms_high", "peak_hz", "peak_rms")
    }
    mask = np.isfinite(x)
    if mask.sum() < 64:
        return nan
    fs, xr = _resample_uniform(np.asarray(t)[mask], np.asarray(x)[mask])
    xr = xr - xr.mean()
    win = np.hanning(len(xr))
    spec = np.fft.rfft(xr * win)
    freqs = np.fft.rfftfreq(len(xr), 1.0 / fs)
    # Power normalized so that summing bands reproduces the signal variance.
    power = (np.abs(spec) ** 2) / (win**2).sum() * 2.0 / len(xr)

    def band(lo: float, hi: float) -> float:
        m = (freqs >= lo) & (freqs < hi)
        return float(np.sqrt(power[m].sum())) if m.any() else 0.0

    out = {
        "rms_low": band(0.0, BAND_LOW),
        "rms_mid": band(BAND_LOW, BAND_HIGH),
        "rms_high": band(BAND_HIGH, fs / 2),
        "peak_hz": math.nan,
        "peak_rms": math.nan,
    }
    m = freqs >= BAND_LOW
    if m.any() and power[m].max() > 0:
        i = int(np.argmax(power[m]))
        floor = float(np.median(power[m]))
        if power[m][i] > 8.0 * max(floor, 1e-30):
            out["peak_hz"] = float(freqs[m][i])
            out["peak_rms"] = float(np.sqrt(power[m][i]))
    return out


def sustained_buzz(
    t: np.ndarray, x: np.ndarray, win_s: float = 0.5
) -> tuple[float, float]:
    """Sustained high-frequency (≥ :data:`BUZZ_LOW` Hz) content of a signal.

    Returns ``(buzz, buzz_hz)``: ``buzz`` is the *median* over short windows
    of the ≥20 Hz RMS (input units); ``buzz_hz`` is the median in-band peak
    frequency of the *loud* half of the windows.

    The median-over-windows is the audibility discriminator the whole-run
    numbers miss: a motion transient spikes one window and vanishes from the
    median, while an audible limit cycle (e.g. wrist_2 near its firmware kd
    clamp, measured buzzing at ~116 Hz through an entire replay) holds every
    window up. Whole-run band RMS dilutes that same buzz below the level of
    ordinary reversal transients, and a whole-run FFT hides its line among
    broadband leakage. The per-window peak-frequency vote survives what a
    per-bin median spectrum does not: real limit cycles *wander* in
    frequency, never landing twice in the same FFT bin, but their loud
    windows still agree to within a few Hz — a healthy joint's loud windows
    are reversal transients that spread broadband instead, and its ``buzz``
    magnitude stays at the floor, which is what to read first.
    """
    mask = np.isfinite(x)
    if mask.sum() < 64:
        return math.nan, math.nan
    fs, xr = _resample_uniform(np.asarray(t)[mask], np.asarray(x)[mask])
    n = int(win_s * fs)
    if n < 16 or len(xr) < 2 * n:
        return math.nan, math.nan
    win = np.hanning(n)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    band = freqs >= BUZZ_LOW
    if not band.any():
        return math.nan, math.nan
    rms: list[float] = []
    peaks: list[float] = []
    for k in range(0, len(xr) - n + 1, n // 2):
        seg = xr[k : k + n]
        seg = seg - seg.mean()
        power = (np.abs(np.fft.rfft(seg * win)) ** 2) / (win**2).sum() * 2.0 / n
        rms.append(float(np.sqrt(power[band].sum())))
        peaks.append(float(freqs[band][int(np.argmax(power[band]))]))
    rms_arr = np.asarray(rms)
    loud = rms_arr >= np.median(rms_arr)
    buzz = float(np.median(rms_arr))
    buzz_hz = float(np.median(np.asarray(peaks)[loud])) if loud.any() else math.nan
    return buzz, buzz_hz


def tracking_lag_ms(
    t: np.ndarray, target: np.ndarray, actual: np.ndarray, max_lag_s: float = 0.25
) -> float:
    """Command→measurement delay (ms) via cross-correlation of the derivatives.

    Correlating velocities rather than positions removes the DC offset and
    the slow trend, so the estimate locks onto the motion itself. NaN when
    the series are too short or too still to correlate.
    """
    if len(t) < 64:
        return math.nan
    fs, tgt = _resample_uniform(np.asarray(t), np.asarray(target))
    _, act = _resample_uniform(np.asarray(t), np.asarray(actual))
    n = min(len(tgt), len(act))
    dt_tgt, dt_act = np.diff(tgt[:n]), np.diff(act[:n])
    if float(np.std(dt_tgt)) < 1e-9 or float(np.std(dt_act)) < 1e-9:
        return math.nan
    max_shift = max(int(max_lag_s * fs), 1)
    dt_tgt = dt_tgt - dt_tgt.mean()
    dt_act = dt_act - dt_act.mean()
    corr = np.correlate(dt_act, dt_tgt, mode="full")
    center = len(dt_tgt) - 1
    window = corr[center : center + max_shift + 1]  # actual lags target only
    i = int(np.argmax(window))
    # Parabolic sub-sample refinement: narrowband references (e.g. a 1 Hz
    # sine) have a nearly flat correlation peak, so the raw argmax quantizes
    # to the sample grid and small disturbances bias it by a whole sample.
    lag = float(i)
    if 0 < i < len(window) - 1:
        a, b, c = window[i - 1], window[i], window[i + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-30:
            lag += 0.5 * float((a - c) / denom)
    return max(lag, 0.0) / fs * 1000.0


def tracking_metrics(
    t: np.ndarray,
    target: np.ndarray,
    actual: np.ndarray,
    torque: np.ndarray | None = None,
) -> dict[str, float]:
    """The standard tracking-accuracy + smoothness scorecard for one joint.

    Accuracy: ``rms_err`` / ``max_err`` (rad, against the commanded target)
    and ``lag_ms`` (transport + control delay, removed from ``rms_err_lagfree``
    so sluggishness and inaccuracy are reported separately).

    Smoothness: ``err_band_mid`` (3–15 Hz RMS of the error — vibration the
    operator feels), ``amplification`` (mid-band RMS of the *measured* motion
    over the commanded motion — >1 means the arm adds energy / rings,
    <1 means it filters), ``pos_ripple`` (second-difference RMS), and
    ``torque_hf`` (cycle-to-cycle torque chatter, Nm) when torque is given.
    ``peak_hz`` flags a dominant resonance line if one stands out.

    Audibility: ``buzz`` / ``buzz_hz`` (see :func:`sustained_buzz`) — the
    sustained ≥20 Hz content of the *measured* motion. This is the number
    that catches a joint audibly vibrating through a run, which the band
    and ripple metrics dilute into the whole-run averages.
    """
    t = np.asarray(t, dtype=float)
    target = np.asarray(target, dtype=float)
    actual = np.asarray(actual, dtype=float)
    err = actual - target
    out: dict[str, float] = {
        "rms_err": float(np.sqrt(np.mean(err**2))),
        "max_err": float(np.max(np.abs(err))),
        "lag_ms": tracking_lag_ms(t, target, actual),
    }

    # Lag-free error: shift the target by the measured lag and re-score.
    out["rms_err_lagfree"] = out["rms_err"]
    if math.isfinite(out["lag_ms"]) and out["lag_ms"] > 0:
        shifted = np.interp(t - out["lag_ms"] / 1000.0, t, target)
        out["rms_err_lagfree"] = float(np.sqrt(np.mean((actual - shifted) ** 2)))

    eb = band_rms(t, err)
    out["err_band_mid"] = eb["rms_mid"]
    out["peak_hz"] = eb["peak_hz"]
    tb = band_rms(t, target)
    ab = band_rms(t, actual)
    # NaN when the reference has no meaningful mid-band content (e.g. a
    # clean 1 Hz sine): the ratio would just amplify numerical dust.
    out["amplification"] = (
        ab["rms_mid"] / tb["rms_mid"] if tb["rms_mid"] > 1e-4 else math.nan
    )

    if len(actual) > 10:
        dd = np.diff(actual, n=2)
        out["pos_ripple"] = float(np.sqrt(np.mean(dd**2)))
    else:
        out["pos_ripple"] = math.nan
    out["buzz"], out["buzz_hz"] = sustained_buzz(t, actual)
    if torque is not None:
        tau = np.asarray(torque, dtype=float)
        tau = tau[np.isfinite(tau)]
        out["torque_hf"] = (
            float(np.sqrt(np.mean(np.diff(tau) ** 2))) if len(tau) > 10 else math.nan
        )
    else:
        out["torque_hf"] = math.nan
    return out


# --------------------------------------------------------------------- #
# Log-based metrics (per-sample dict logs from the hardware runners)     #
# --------------------------------------------------------------------- #


def chatter_metrics(log: list[dict]) -> dict[str, float | None]:
    """High-frequency roughness metrics — the 'vibration' the tracking-error
    score can't see.

    ``torque_hf``: RMS of cycle-to-cycle changes in the motor's reported
    torque. Smooth control action changes slowly between 100 Hz samples, so
    this isolates torque chatter (noise amplified by kd_host / j_eff / kd).

    ``pos_ripple``: RMS of the second difference of measured position — a
    high-pass that suppresses the commanded trajectory and exposes vibration.
    """
    taus = [r["torque"] for r in log if not math.isnan(r.get("torque", math.nan))]
    torque_hf = None
    if len(taus) > 10:
        d = [taus[i + 1] - taus[i] for i in range(len(taus) - 1)]
        torque_hf = math.sqrt(sum(x * x for x in d) / len(d))
    qs = [r["actual"] for r in log]
    pos_ripple = None
    if len(qs) > 10:
        dd = [qs[i + 2] - 2 * qs[i + 1] + qs[i] for i in range(len(qs) - 2)]
        pos_ripple = math.sqrt(sum(x * x for x in dd) / len(dd))
    return {"torque_hf": torque_hf, "pos_ripple": pos_ripple}


def sine_metrics(log: list[dict]) -> dict[str, float]:
    """Score a sine-tracking log (lower ``score`` is better)."""
    errors = [r["error"] for r in log]
    rms = math.sqrt(sum(e**2 for e in errors) / len(errors))
    max_err = max(abs(e) for e in errors)
    elapsed = log[-1]["t"] - log[0]["t"] if len(log) > 1 else 1.0
    actual_hz = len(log) / elapsed if elapsed > 0 else 0.0
    # Score: tracking RMS dominates, with a small penalty on the worst
    # excursion so two equal-RMS candidates prefer the one without spikes.
    return {
        "rms": rms,
        "max": max_err,
        "hz": actual_hz,
        "score": rms + 0.2 * max_err,
        **chatter_metrics(log),
    }


def ring_frequency(step_rows: list[dict]) -> float | None:
    """Dominant oscillation frequency (Hz) in the post-step error, if any.

    The ring frequency is the closed loop's ωn — it drops as the reflected
    inertia grows (ωn = √(kp/J)), so comparing it across poses directly
    measures the pose dependence that fixed gains can't absorb. Returns
    ``None`` when no spectral peak stands clear of the noise floor.
    """
    if len(step_rows) < 32:
        return None
    t = np.array([r["t"] for r in step_rows])
    if t[-1] <= t[0]:
        return None
    e = np.array([r["error"] for r in step_rows])
    # An oscillation must actually cross its settled value repeatedly; a
    # monotonic settle has spectral content too but is not a ring.
    e_rel = e - e[-len(e) // 4 :].mean()
    active = e_rel[np.abs(e_rel) > 0.05 * np.abs(e_rel).max()]
    if len(active) < 2 or int(np.sum(np.diff(np.sign(active)) != 0)) < 3:
        return None
    e = e - e.mean()
    fs = (len(t) - 1) / (t[-1] - t[0])
    spec = np.abs(np.fft.rfft(e * np.hanning(len(e))))
    freqs = np.fft.rfftfreq(len(e), 1.0 / fs)
    mask = freqs >= 0.5  # below ~0.5 Hz is the settling trend, not a ring
    if not mask.any():
        return None
    peak_i = int(np.argmax(spec[mask]))
    if spec[mask][peak_i] < 4.0 * float(np.median(spec[mask])):
        return None
    return float(freqs[mask][peak_i])


def step_metrics(log: list[dict], amp: float, hold: float) -> dict[str, float | None]:
    """Score a step-response log (lower ``score`` is better)."""
    targets = list(dict.fromkeys(r["target"] for r in log))
    step_target = targets[0]
    step_rows = [r for r in log if r["target"] == step_target]
    return_target = targets[1] if len(targets) > 1 else step_target - amp
    direction = 1 if step_target > return_target else -1
    overshoot = max(
        0.0, max(direction * (r["actual"] - step_target) for r in step_rows)
    )

    threshold = 0.05 * amp
    t_step_start = step_rows[0]["t"]
    settling_s = None
    for i, r in enumerate(step_rows):
        if abs(r["error"]) < threshold:
            future = step_rows[i : i + 10]
            if len(future) == 10 and all(abs(fr["error"]) < threshold for fr in future):
                settling_s = r["t"] - t_step_start
                break

    settled = step_rows[len(step_rows) // 2 :]
    ss_rms = (
        math.sqrt(sum(r["error"] ** 2 for r in settled) / len(settled))
        if settled
        else 0.0
    )
    elapsed = log[-1]["t"] - log[0]["t"] if len(log) > 1 else 1.0
    actual_hz = len(log) / elapsed if elapsed > 0 else 0.0
    overshoot_frac = overshoot / amp if amp > 0 else 0.0
    # Score (heuristic, lower is better): settling time in seconds, plus
    # 0.5 s of penalty per 10% overshoot, plus steady-state RMS weighted so
    # 0.01 rad ≈ 0.1 s. A candidate that never settles is charged 2× hold.
    score = (
        (settling_s if settling_s is not None else 2.0 * hold)
        + 5.0 * overshoot_frac
        + 10.0 * ss_rms
    )
    return {
        "settling_s": settling_s,
        "overshoot": overshoot,
        "overshoot_frac": overshoot_frac,
        "ss_rms": ss_rms,
        "hz": actual_hz,
        "score": score,
        "ring_hz": ring_frequency(step_rows),
        **chatter_metrics(log),
    }
