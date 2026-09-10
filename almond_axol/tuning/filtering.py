"""Deterministic filter-stack test: inject noise, verify the stack removes it.

The teleop smoothing chain (the lag-compensated pose low-pass, the IK-output
EMA, and the trapezoidal velocity/acceleration tracker — see
:mod:`almond_axol.teleop.filter`) exists to clean a dirty target stream.
This module tests exactly that, with no hardware and no VR headset — and it
respects *where* each real noise source enters the production pipeline,
which is what makes the sources testable independently:

    VR stream ──> pose low-pass ──> IK solver ──> EMA ──> trapezoid
       ▲ network noise enters here     ▲ IK noise is created here

* ``noise="network"`` — transport artifacts, injected **before** the pose
  low-pass (the whole stack gets to clean them): white-noise **jitter**
  (sensor/hand noise), **outliers** (isolated teleported samples — tracking
  glitches), and **stalls** (the stream freezes on its last sample, then
  jumps to catch up — wifi stalls, dropped frames).
* ``noise="ik"`` — solver output artifacts, injected **after** the pose
  low-pass, exactly where the solver sits in production, so only the EMA
  and the trapezoid can clean them: band-limited (3–20 Hz) per-joint
  **churn** (a restless null space) and occasional persistent **jumps** (a
  redundancy flip: one joint steps to another solution branch for a fraction
  of a second, then returns).
* ``noise="combined"`` — both, each at its own injection point: the full
  production insult.

Take a *clean* joint-space motion (a synthetic sine or a committed reference
motion), corrupt it per the selected mode, replay through the production
filter stack at the production rates, and score the output *against the
clean reference*. A good stack passes the intentional motion (low
``rms_err``, low ``lag_ms``) while rejecting the injected garbage
(``jitter_passed`` well below 1, ``peak_err`` far below the outlier
magnitude, output acceleration never above teleop's configured limit no
matter how hard the input slams).

Everything is seeded and offline, so a run is exactly reproducible: change a
filter parameter, rerun on the identical corrupted stream, and compare
scores run to run. The network and IK noise streams are seeded
independently, so the same seed gives the same network noise whether or not
IK noise is also enabled.

Stall spans deserve one caveat when reading scores: while the stream is
frozen the clean reference keeps moving, so error during a stall is
*missing information*, not filter failure. What the filter owns is the
catch-up — resuming without overshoot or an acceleration spike.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..constants import ARM_JOINTS
from ..teleop.config import VRTeleopConfig
from ..teleop.filter import AlphaSmoothFilter, LagCompensatedLowPass, TrapezoidalFilter
from .metrics import band_rms, tracking_lag_ms
from .motion import load_motion

# Column names of a 14-wide motion row: left arm then right arm (matches
# tune.motion's artifact layout so the UI charts both the same way).
_MOTION_COLUMNS = [f"left.{j.value}" for j in ARM_JOINTS] + [
    f"right.{j.value}" for j in ARM_JOINTS
]


def inject_noise(
    t: np.ndarray,
    x: np.ndarray,
    *,
    jitter_rms: float = 0.0,
    outlier_rate: float = 0.0,
    outlier_amp: float = 0.0,
    stall_rate: float = 0.0,
    stall_ms: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, int]]:
    """Corrupt a clean ``(N, J)`` signal with jitter, outliers, and stalls.

    Outliers teleport *whole samples* (every channel at once, like a glitched
    pose), stalls freeze the already-noisy stream on its last sample — the
    order matters, since a real stall freezes whatever garbage arrived last.
    Deterministic for a given ``seed``. Returns the corrupted copy and the
    injected event counts.
    """
    rng = np.random.default_rng(seed)
    noisy = np.asarray(x, dtype=float).copy()
    n = len(t)
    span = float(t[-1] - t[0])
    events = {"outliers": 0, "stalls": 0}

    if jitter_rms > 0.0:
        noisy += rng.normal(0.0, jitter_rms, noisy.shape)

    if outlier_rate > 0.0 and outlier_amp > 0.0:
        count = min(n, max(1, round(outlier_rate * span)))
        picks = rng.choice(n, size=count, replace=False)
        signs = rng.choice([-1.0, 1.0], size=count)
        noisy[picks] += (outlier_amp * signs)[:, None]
        events["outliers"] = count

    if stall_rate > 0.0 and stall_ms > 0.0:
        count = max(1, round(stall_rate * span))
        stall_s = stall_ms / 1000.0
        starts = rng.uniform(float(t[0]), float(t[-1]) - stall_s, size=count)
        for s in np.sort(starts):
            i0 = int(np.searchsorted(t, s))
            i1 = int(np.searchsorted(t, s + stall_s))
            if i0 <= 0 or i0 >= n:
                continue
            noisy[i0:i1] = noisy[i0 - 1]
        events["stalls"] = count
    return noisy, events


def inject_ik_noise(
    t: np.ndarray,
    n_channels: int,
    *,
    churn_rms: float = 0.0,
    jump_rate: float = 0.0,
    jump_amp: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, int]]:
    """Additive ``(N, J)`` noise shaped like IK solver output artifacts.

    * **churn** — band-limited (3–20 Hz) noise, independent per joint: the
      restless-null-space wobble ``diag.offline kinematics`` measures on real
      recordings (~0.1–0.3° RMS per joint on a healthy solve).
    * **jumps** — one joint steps by ``jump_amp`` and *stays there* for
      0.3–1 s before returning: a redundancy flip to another solution
      branch. Unlike a network outlier this is not a single bad sample, so
      an outlier-rejecting trick alone can't remove it.

    Seeded independently of :func:`inject_noise` (same ``seed``, different
    stream), so enabling one source never changes the other's realization.
    """
    rng = np.random.default_rng([seed, 1])
    n = len(t)
    noise = np.zeros((n, n_channels), dtype=float)
    span = float(t[-1] - t[0])
    dt = float(np.median(np.diff(t)))
    events = {"ik_jumps": 0}

    if churn_rms > 0.0 and n > 8:
        freqs = np.fft.rfftfreq(n, dt)
        mask = (freqs >= 3.0) & (freqs <= 20.0)
        for j in range(n_channels):
            spec = np.fft.rfft(rng.normal(0.0, 1.0, n))
            spec[~mask] = 0.0
            band = np.fft.irfft(spec, n)
            rms = float(np.sqrt(np.mean(band**2)))
            if rms > 0.0:
                noise[:, j] += band * (churn_rms / rms)

    if jump_rate > 0.0 and jump_amp > 0.0:
        count = max(1, round(jump_rate * span))
        for _ in range(count):
            ch = int(rng.integers(n_channels))
            start = float(rng.uniform(float(t[0]), float(t[-1])))
            dur = float(rng.uniform(0.3, 1.0))
            sign = float(rng.choice([-1.0, 1.0]))
            i0 = int(np.searchsorted(t, start))
            i1 = int(np.searchsorted(t, start + dur))
            noise[i0:i1, ch] += sign * jump_amp
        events["ik_jumps"] = count
    return noise, events


def replay_filter_stack(
    t_in: np.ndarray,
    x_in: np.ndarray,
    *,
    cutoff: float | None = None,
    config: VRTeleopConfig | None = None,
    post_lp_noise: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run an ``(N, J)`` target stream through the production smoothing chain.

    Mirrors teleop's structure and rates: the lag-compensated low-pass runs
    per input sample (the IK-dispatch cadence, with real timestamps), and the
    control loop at ``config.frequency`` zero-order-holds its newest output
    into the EMA and the trapezoidal tracker — the same stage order as
    :meth:`TeleopCore <almond_axol.teleop.core>` (EMA first, trapezoid last).
    The pose filter is applied in joint space here; it is a linear per-channel
    filter, so its rejection behaves identically regardless of the space.

    ``post_lp_noise`` (``(N, J)``, optional) is added *between* the pose
    low-pass and the EMA — the point where the IK solver sits in production
    — so solver-shaped artifacts hit only the stages that can actually see
    them there.

    Returns ``(t_out, filtered, hold_index)`` where ``t_out`` is the control-
    rate grid and ``hold_index[k]`` is the input sample the control tick ``k``
    was holding (for resampling the input onto the same grid).
    """
    cfg = config or VRTeleopConfig()
    t_in = np.asarray(t_in, dtype=float)
    x_in = np.asarray(x_in, dtype=float)
    rate_in = 1.0 / float(np.median(np.diff(t_in)))
    lp = LagCompensatedLowPass(
        rate_in, cutoff if cutoff is not None else cfg.pose_cutoff
    )
    pose = np.stack([lp.update(x, float(ts)) for ts, x in zip(t_in, x_in)])
    if post_lp_noise is not None:
        pose = pose + np.asarray(post_lp_noise, dtype=float)

    dt = 1.0 / cfg.frequency
    # ik_alpha is specified per-tick at the historical 120 Hz rate; convert
    # exactly like TeleopCore so the EMA time constant matches production.
    alpha = 1.0 - (1.0 - cfg.ik_alpha) ** (120.0 * dt)
    ema = AlphaSmoothFilter(alpha)
    trap = TrapezoidalFilter(cfg.teleop_max_vel, cfg.teleop_max_accel, dt)

    n_out = int((t_in[-1] - t_in[0]) / dt) + 1
    t_out = t_in[0] + dt * np.arange(n_out)
    hold_index = np.clip(np.searchsorted(t_in, t_out, side="right") - 1, 0, None)
    filtered = np.empty((n_out, x_in.shape[1]), dtype=float)
    for k, i in enumerate(hold_index):
        y = ema.update(pose[i])
        filtered[k] = trap.update(y)
    return t_out, filtered, hold_index


def _channel_scores(
    t: np.ndarray,
    clean: np.ndarray,
    noisy: np.ndarray,
    filtered: np.ndarray,
    dt: float,
) -> dict[str, float]:
    """Cleanup scorecard for one channel, everything vs the clean reference."""
    in_err = noisy - clean
    out_err = filtered - clean
    in_band = band_rms(t, in_err)["rms_mid"]
    out_band = band_rms(t, out_err)["rms_mid"]

    def accel_peak(x: np.ndarray) -> float:
        if len(x) < 3:
            return math.nan
        return float(np.max(np.abs(np.diff(x, n=2)))) / dt**2

    rms_err = float(np.sqrt(np.mean(out_err**2)))
    lag_ms = tracking_lag_ms(t, clean, filtered)
    # The raw error on a moving reference is dominated by the stack's group
    # delay; re-scoring against the lag-shifted reference separates
    # "sluggish" (lag_ms) from "dirty" (rms_err_lagfree, the residual the
    # injected noise actually left behind).
    rms_err_lagfree = rms_err
    if math.isfinite(lag_ms) and lag_ms > 0:
        shifted = np.interp(t - lag_ms / 1000.0, t, clean)
        rms_err_lagfree = float(np.sqrt(np.mean((filtered - shifted) ** 2)))

    return {
        "rms_err": rms_err,
        "rms_err_lagfree": rms_err_lagfree,
        "peak_err": float(np.max(np.abs(out_err))),
        "lag_ms": lag_ms,
        "input_rms": float(np.sqrt(np.mean(in_err**2))),
        "input_peak": float(np.max(np.abs(in_err))),
        # Mid-band (3-15 Hz, the felt-jitter band) error out over in: how
        # much of the injected vibration survives the stack. NaN when the
        # injected noise carries no mid-band content to measure against.
        "jitter_passed": out_band / in_band if in_band > 1e-6 else math.nan,
        "accel_peak": accel_peak(filtered),
        "accel_peak_in": accel_peak(noisy),
    }


def filter_noise_analysis(
    *,
    motion: str | None = None,
    duration: float = 10.0,
    amp: float = 0.3,
    freq: float = 0.5,
    noise: str = "combined",
    jitter_rms: float = 0.005,
    outlier_rate: float = 0.5,
    outlier_amp: float = 0.2,
    stall_rate: float = 0.5,
    stall_ms: float = 150.0,
    ik_churn: float = 0.003,
    ik_jump_rate: float = 0.2,
    ik_jump_amp: float = 0.05,
    cutoff: float | None = None,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """The full suite: clean signal → noise → filter stack → per-joint scores.

    ``motion`` selects a committed reference motion as the clean signal (all
    14 joints); ``None`` synthesizes an ``amp``·sin(2π·``freq``·t) sine for
    ``duration`` seconds at the IK cadence, as a single channel.

    ``noise`` selects the source under test and its real injection point
    (see the module docstring): ``"network"`` puts jitter/outliers/stalls in
    front of the pose low-pass, ``"ik"`` puts solver churn/jumps between the
    low-pass and the EMA, ``"combined"`` does both. The network knobs are
    ignored in ``"ik"`` mode and vice versa.

    Returns ``(series, metrics, params)`` in the standard tuning-run shape,
    with every series resampled onto the control-rate grid: ``clean`` (the
    reference), ``noisy`` (the fully corrupted, unfiltered signal — what the
    arm would be fed with no filtering at all), and ``filtered`` (what came
    out of the stack).
    """
    if noise not in ("network", "ik", "combined"):
        raise ValueError(f"unknown noise mode {noise!r} (network / ik / combined)")
    cfg = VRTeleopConfig()
    if motion is not None:
        ref = load_motion(motion)
        x = np.asarray(ref.q, dtype=float)
        t_in = np.arange(len(x)) / float(ref.rate)
        columns = list(_MOTION_COLUMNS)
        source = ref.name
    else:
        rate = float(cfg.ik_frequency)
        t_in = np.arange(int(duration * rate)) / rate
        x = (amp * np.sin(2.0 * math.pi * freq * t_in))[:, None]
        columns = ["sine"]
        source = "sine"

    with_network = noise in ("network", "combined")
    with_ik = noise in ("ik", "combined")
    noisy_in, events = inject_noise(
        t_in,
        x,
        jitter_rms=jitter_rms if with_network else 0.0,
        outlier_rate=outlier_rate if with_network else 0.0,
        outlier_amp=outlier_amp if with_network else 0.0,
        stall_rate=stall_rate if with_network else 0.0,
        stall_ms=stall_ms if with_network else 0.0,
        seed=seed,
    )
    ik_noise: np.ndarray | None = None
    ik_events = {"ik_jumps": 0}
    if with_ik:
        ik_noise, ik_events = inject_ik_noise(
            t_in,
            x.shape[1],
            churn_rms=ik_churn,
            jump_rate=ik_jump_rate,
            jump_amp=ik_jump_amp,
            seed=seed,
        )
    t_out, filtered, hold_index = replay_filter_stack(
        t_in, noisy_in, cutoff=cutoff, config=cfg, post_lp_noise=ik_noise
    )
    dt = 1.0 / cfg.frequency

    clean = np.stack([np.interp(t_out, t_in, x[:, i]) for i in range(x.shape[1])]).T
    noisy = noisy_in[hold_index]
    if ik_noise is not None:
        noisy = noisy + ik_noise[hold_index]

    per_joint: dict[str, dict[str, float]] = {}
    for i, name in enumerate(columns):
        # Parked motion channels score meaninglessly well — skip them (the
        # single synthetic channel is always scored).
        if len(columns) > 1 and float(np.ptp(clean[:, i])) < math.radians(1.0):
            continue
        per_joint[name] = _channel_scores(
            t_out, clean[:, i], noisy[:, i], filtered[:, i], dt
        )
    if not per_joint:
        raise ValueError("no channel moved more than 1° — nothing to score")

    worst = max(per_joint.items(), key=lambda kv: kv[1]["rms_err"])
    metrics: dict[str, Any] = {
        "per_joint": per_joint,
        "worst_joint": worst[0],
        "mean_rms_err": float(np.mean([m["rms_err"] for m in per_joint.values()])),
        # The headline cleanliness number: residual error after removing the
        # stack's delay — what the injected noise actually left behind.
        "mean_rms_lagfree": float(
            np.mean([m["rms_err_lagfree"] for m in per_joint.values()])
        ),
        "accel_limit": float(cfg.teleop_max_accel),
        **events,
        **ik_events,
    }
    params: dict[str, Any] = {
        "source": source,
        "noise": noise,
        "duration": float(t_in[-1] - t_in[0]),
        "amp": amp if motion is None else None,
        "freq": freq if motion is None else None,
        "jitter_rms": jitter_rms if with_network else 0.0,
        "outlier_rate": outlier_rate if with_network else 0.0,
        "outlier_amp": outlier_amp if with_network else 0.0,
        "stall_rate": stall_rate if with_network else 0.0,
        "stall_ms": stall_ms if with_network else 0.0,
        "ik_churn": ik_churn if with_ik else 0.0,
        "ik_jump_rate": ik_jump_rate if with_ik else 0.0,
        "ik_jump_amp": ik_jump_amp if with_ik else 0.0,
        "cutoff": cutoff if cutoff is not None else cfg.pose_cutoff,
        "seed": seed,
        "columns": columns,
    }
    series = {"t": t_out, "clean": clean, "noisy": noisy, "filtered": filtered}
    return series, metrics, params
