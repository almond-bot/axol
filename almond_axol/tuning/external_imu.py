"""An IMU the operator mounts on their own end-effector, scored like the
wrist camera's.

Robots without the wrist ZED cameras have no tool-sway measurement; the
joint encoders' (:func:`~almond_axol.tuning.creep_score.flange_sway`) can't
see flex past the motors. Any IMU bolted rigidly to the end-effector fills
that gap: log it to a CSV while ``tune.motion`` / ``scripts/tuning_queue.py``
runs, then attach each run's slice (``scripts/ext_imu.py``). The slice gets
the same :func:`~almond_axol.tuning.wrist_imu.shake_metrics` the ZED IMU
gets, saved next to the run as ``imu_external.json``, and the run scorer
uses it wherever the run has no wrist-camera IMU.

The log: one row per sample, ``t, ax, ay, az`` and optionally ``gx, gy,
gz`` — ``t`` in Unix seconds (or milliseconds), acceleration *including
gravity* (raw accelerometer, not "linear acceleration"), in m/s² or g.
At least 100 Hz (200+ preferred). Header row optional.

The two clocks are matched in two steps: the run's wall-clock start
(``t0_wall``, saved by ``tune.motion``) places the slice to within the
clocks' offset, then the IMU's vertical acceleration is cross-correlated
with the flange's, computed from the encoders by forward kinematics, within
``search_s`` — so an IMU logged on another NTP-synced computer works too.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .creep_score import flange_heights
from .runs import TUNING_RUNS_DIR
from .wrist_imu import shake_metrics

G = 9.80665
SIDECAR = "imu_external.json"
#: Below this the cross-correlation didn't find the motion in the IMU log.
MIN_CORR = 0.3
_ALIGN_RATE_HZ = 120.0
_ALIGN_BAND = (0.5, 8.0)
#: The share of the run the log must cover to be matched.
_MIN_COVER = 0.8


def load_log(path: str | Path, units: str = "auto") -> dict[str, np.ndarray]:
    """Read an IMU CSV: ``{"t": Unix s, "acc": (N, 3) m/s², "gyro"?}``.

    ``units`` is ``"ms2"``, ``"g"`` or ``"auto"`` (g when the median
    magnitude is near 1). Millisecond timestamps are recognised by size.
    """
    rows = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            try:
                rows.append([float(x) for x in row[:7]])
            except ValueError:
                continue  # a header or a comment
    if len(rows) < 50:
        raise ValueError(f"{path}: fewer than 50 numeric rows")
    width = min(len(r) for r in rows)
    if width < 4:
        raise ValueError(f"{path}: need columns t, ax, ay, az [, gx, gy, gz]")
    a = np.array([r[:width] for r in rows])
    t = a[:, 0]
    if np.median(t) > 1e11:  # milliseconds
        t = t / 1e3
    if np.median(t) < 1e9:
        raise ValueError(
            f"{path}: timestamps must be Unix time (s or ms), got {t[0]:g}"
        )
    order = np.argsort(t, kind="stable")
    acc = a[order, 1:4]
    mag = float(np.median(np.linalg.norm(acc, axis=1)))
    if units == "auto":
        units = "g" if 0.5 < mag < 2.0 else "ms2"
    if units == "g":
        acc = acc * G
        mag *= G
    if not 5.0 < mag < 15.0:
        raise ValueError(
            f"{path}: median |acc| {mag:.2f} m/s² — gravity must be included "
            "(raw accelerometer), and --units right"
        )
    out = {"t": t[order], "acc": acc}
    if width >= 7:
        out["gyro"] = a[order, 4:7]
    return out


def describe(log: dict[str, np.ndarray]) -> dict[str, Any]:
    """Rate, span and gravity of a log — a sanity check before a session."""
    t = log["t"]
    dt = np.diff(t)
    return {
        "samples": len(t),
        "rate_hz": round(1.0 / float(np.median(dt[dt > 0])), 1),
        "start_unix": round(float(t[0]), 3),
        "seconds": round(float(t[-1] - t[0]), 1),
        "gaps_over_50ms": int((dt > 0.05).sum()),
        "gravity_ms2": round(float(np.median(np.linalg.norm(log["acc"], axis=1))), 3),
        "gyro": "gyro" in log,
    }


def _bandpass(x: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    spec = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    spec[(f < band[0]) | (f > min(band[1], 0.45 * fs))] = 0
    return np.fft.irfft(spec, len(x))


def _vertical(acc: np.ndarray) -> np.ndarray:
    """Acceleration along the record's mean (gravity) direction."""
    up = acc.mean(axis=0)
    return acc @ (up / (np.linalg.norm(up) or 1.0))


def align(
    run_t: np.ndarray,
    actual: np.ndarray,
    side: str,
    t0_wall: float,
    log: dict[str, np.ndarray],
    search_s: float = 2.0,
) -> tuple[float, float]:
    """``(offset_s, corr)``: the IMU's Unix time of the run's ``t = 0`` is
    ``t0_wall + offset_s``, found where the IMU's vertical acceleration best
    matches the flange's by forward kinematics (correlation ``corr``)."""
    fs = _ALIGN_RATE_HZ
    cols = slice(0, 7) if side == "left" else slice(7, 14)
    q = np.asarray(actual, dtype=float)[:, cols]
    ok = np.isfinite(q).all(axis=1)
    if ok.mean() < 0.9:
        raise ValueError(f"the {side} arm wasn't driven in this run")
    grid = np.arange(run_t[ok][0], run_t[ok][-1], 1.0 / fs)
    qg = np.stack([np.interp(grid, run_t[ok], q[ok, j]) for j in range(7)], axis=1)
    z = flange_heights(qg, side)
    ref = _bandpass(np.gradient(np.gradient(z, 1 / fs), 1 / fs), fs, _ALIGN_BAND)
    t_imu = log["t"] - t0_wall
    near = (t_imu > grid[0] - search_s - 5) & (t_imu < grid[-1] + search_s + 5)
    if near.sum() < 50:
        raise ValueError(
            f"the IMU log ({t_imu[0]:+.0f}…{t_imu[-1]:+.0f} s around the run's "
            "start) has nothing at the run — was it logging, on a synced clock?"
        )
    t_imu, acc_raw = t_imu[near], log["acc"][near]
    gi = np.arange(t_imu[0], t_imu[-1], 1.0 / fs)
    if len(gi) < fs:
        raise ValueError("the IMU log is shorter than a second")
    acc = np.stack([np.interp(gi, t_imu, acc_raw[:, i]) for i in range(3)], 1)
    sig = _bandpass(_vertical(acc), fs, _ALIGN_BAND)
    n = len(ref)
    best = (0.0, -1.0)
    for lag in np.arange(-search_s, search_s + 0.5 / fs, 1.0 / fs):
        # The run's samples that the log covers at this lag.
        k0 = int(round((grid[0] + lag - gi[0]) * fs))
        lo, hi = max(0, -k0), min(n, len(sig) - k0)
        if hi - lo < _MIN_COVER * n:
            continue
        a, b = ref[lo:hi], sig[k0 + lo : k0 + hi]
        sa, sb = a.std(), b.std()
        if sa == 0 or sb == 0:
            continue
        c = float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))
        if c > best[1]:
            best = (float(lag), c)
    if best[1] == -1.0:
        raise ValueError(
            f"the IMU log ({t_imu[0]:+.1f}…{t_imu[-1]:+.1f} s around the run's "
            f"start) doesn't cover the run ({grid[0]:.1f}…{grid[-1]:.1f} s, "
            f"± {search_s:g} s) — was it logging, on a synced clock?"
        )
    return best


def attach(
    run_id: str,
    log: dict[str, np.ndarray],
    side: str | None = None,
    search_s: float | None = None,
    source: str = "",
    runs_dir: Path = TUNING_RUNS_DIR,
) -> dict[str, Any]:
    """Score ``run_id``'s slice of ``log`` and save it next to the run.

    Returns the sidecar written (``{"side", "metrics", "offset_s", "corr",
    ...}``); raises ``ValueError`` when the log doesn't cover the run or the
    motion can't be found in it (``corr`` < :data:`MIN_CORR`).
    """
    d = Path(runs_dir) / run_id
    meta = json.loads((d / "meta.json").read_text())
    z = np.load(d / "series.npz")
    t = z["t"].astype(float)
    actual = z["actual"].astype(float)
    if side is None:
        side = meta.get("side") or (meta.get("params") or {}).get("arms")
    if side not in ("left", "right"):
        tg = np.nan_to_num(z["target"].astype(float))
        side = (
            "left"
            if np.ptp(tg[:, :7], axis=0).max() > np.ptp(tg[:, 7:], axis=0).max()
            else "right"
        )
    metrics = meta.get("metrics") or {}
    t0_wall = metrics.get("t0_wall")
    if t0_wall is None:  # a run from before t0_wall: saved just after it ended
        t0_wall = float(meta["startedAt"]) - float(t[-1])
        search_s = search_s or 10.0
    search_s = search_s or 2.0
    offset, corr = align(t, actual, side, t0_wall, log, search_s)
    if corr < MIN_CORR:
        raise ValueError(
            f"{run_id}: the motion isn't in the IMU log (best correlation "
            f"{corr:.2f} at {offset:+.2f} s) — wrong arm, a loose mount, or "
            "the clocks are further apart than --search"
        )
    t_imu = log["t"] - (t0_wall + offset)
    sel = (t_imu >= t[0]) & (t_imu <= t[-1])
    m = shake_metrics(
        t_imu[sel], log["acc"][sel], log["gyro"][sel] if "gyro" in log else None
    )
    if not m:
        raise ValueError(f"{run_id}: too few IMU samples inside the run")
    out = {
        "side": side,
        "metrics": {k: v for k, v in m.items() if not math.isnan(v)},
        "offset_s": round(offset, 4),
        "corr": round(corr, 3),
        "source": source,
    }
    (d / SIDECAR).write_text(json.dumps(out, indent=1))
    return out


def load_attached(run_id: str, runs_dir: Path = TUNING_RUNS_DIR) -> dict | None:
    """The sidecar :func:`attach` wrote for ``run_id``, if any."""
    p = Path(runs_dir) / run_id / SIDECAR
    return json.loads(p.read_text()) if p.is_file() else None
