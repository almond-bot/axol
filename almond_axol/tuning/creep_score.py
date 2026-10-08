"""Score a saved ``tune.motion`` run the way the slow-motion tuning judged it.

The robot-tuning playbook (``.claude/skills/axol-robot-tuning``) compares
variants on these numbers, so they are computed in one place:

- ``band_mdeg``: every joint's tracking error in the 0.5–15 Hz band, mdeg RMS
  — rings, and joints shaken by a neighbour.
- For the one joint the motion moves (a creep motion moves one), its ripple
  on the constant-speed legs at 3 and 6 deg/s: ``rip3_rms_mdeg`` /
  ``rip3_p2p_mdeg`` (median 2 s peak-to-peak) / ``vrip3_dps`` (velocity
  ripple), and the same at 6 deg/s. The first 0.6 s of every leg is dropped.
- ``imu``: the wrist-IMU tool-shake metrics ``tune.motion`` saved, when the
  wrist camera opened (``low_mm`` is the 1–3 Hz sway an operator feels).

Ripple varies ±30% between sessions and even between back-to-back runs, so
only interleaved comparisons in one session mean anything (see
:mod:`scripts.tuning_queue`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .runs import TUNING_RUNS_DIR

JOINT_NAMES = (
    "shoulder_1",
    "shoulder_2",
    "shoulder_3",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)
#: Constant-speed legs scored (deg/s); a leg counts within 5% of the speed.
CREEP_SPEEDS = (3.0, 6.0)
_LEG_SETTLE_S = 0.6
_IMU_KEYS = ("vertical_mm", "low_mm", "high_mm", "acc_rms", "shake_mm")


def _band(x: np.ndarray, rate: float, lo: float = 0.5, hi: float = 15.0) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    hi = min(hi, 0.45 * rate)
    sos = butter(2, [lo, hi], btype="band", fs=rate, output="sos")
    return sosfiltfilt(sos, x)


def score_series(
    t: np.ndarray, target: np.ndarray, actual: np.ndarray, side: str
) -> dict[str, Any]:
    """Score one run's ``(t, target, actual)`` arrays (rad, 14 columns:
    left arm then right) for the arm ``side``."""
    t = np.asarray(t, dtype=float)
    tg = np.asarray(target, dtype=float)
    a = np.asarray(actual, dtype=float)
    dt = np.diff(t)
    rate = 1.0 / float(np.median(dt[dt > 0]))
    cols = slice(0, 7) if side == "left" else slice(7, 14)
    e = a[:, cols] - tg[:, cols]
    good = np.isfinite(e).all(axis=1)
    out: dict[str, Any] = {"side": side, "rate_hz": round(rate, 1)}
    if good.mean() < 0.9:
        out["note"] = "gappy"
    e = np.where(np.isfinite(e), e, 0.0)
    out["band_mdeg"] = {
        JOINT_NAMES[j]: round(
            1e3 * math.degrees(float(np.sqrt(np.mean(_band(e[:, j], rate) ** 2)))), 1
        )
        for j in range(7)
    }
    span = np.ptp(np.nan_to_num(tg[:, cols]), axis=0)
    j = int(np.argmax(span))
    if span[j] < math.radians(2.0):
        return out
    out["moving"] = JOINT_NAMES[j]
    v = np.gradient(np.nan_to_num(tg[:, cols.start + j]), 1.0 / rate)
    vd = np.degrees(np.abs(v))
    eb = _band(e[:, j], rate)
    vb = np.gradient(eb, 1.0 / rate)
    for spd in CREEP_SPEEDS:
        on = np.abs(vd - spd) < 0.05 * spd
        idx = np.flatnonzero(on)
        if idx.size < rate:
            continue
        keep = np.zeros_like(on)
        seg_start = idx[0]
        for k, i in enumerate(idx):
            if k == 0 or i - idx[k - 1] > 1:
                seg_start = i
            if i - seg_start > _LEG_SETTLE_S * rate:
                keep[i] = True
        sel = keep & good
        if sel.sum() < rate:
            continue
        x = eb[sel]
        w = int(2 * rate)
        p2p = [np.ptp(x[i : i + w]) for i in range(0, len(x) - w, w // 2)] or [
            np.ptp(x)
        ]
        tag = int(spd)
        out[f"rip{tag}_rms_mdeg"] = round(
            1e3 * math.degrees(float(np.sqrt(np.mean(x**2)))), 1
        )
        out[f"rip{tag}_p2p_mdeg"] = round(1e3 * math.degrees(float(np.median(p2p))), 1)
        out[f"vrip{tag}_dps"] = round(
            math.degrees(float(np.sqrt(np.mean(vb[sel] ** 2)))), 3
        )
    return out


def score_run(run_id: str, runs_dir: Path = TUNING_RUNS_DIR) -> dict[str, Any]:
    """Score a saved run by id: labels, guard trips, IMU and the ripple."""
    d = Path(runs_dir) / run_id
    meta = json.loads((d / "meta.json").read_text())
    z = np.load(d / "series.npz")
    target = z["target"].astype(float)
    side = meta.get("side")
    if side not in ("left", "right"):
        # The arm whose target moves more.
        tg = np.nan_to_num(target)
        left = np.ptp(tg[:, :7], axis=0).max()
        right = np.ptp(tg[:, 7:14], axis=0).max()
        side = "left" if left > right else "right"
    metrics = meta.get("metrics") or {}
    out: dict[str, Any] = {
        "id": run_id,
        "label": meta.get("label"),
        "motion": (meta.get("params") or {}).get("motion"),
        "trips": metrics.get("guard_trips", []),
        "completed": metrics.get("completed"),
    }
    out.update(score_series(z["t"], target, z["actual"].astype(float), side))
    imu = (metrics.get("imu") or {}).get(side)
    if imu:
        out["imu"] = {k: round(imu[k], 3) for k in _IMU_KEYS if k in imu}
    return out
