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
- ``enc``: the same sway measured from the joint encoders — the wrist
  flange's vertical position by forward kinematics, scored exactly like the
  IMU (``low_mm`` 1–3 Hz, ``high_mm`` 3–15 Hz, median 2 s peak-to-peak). For
  robots without a wrist camera, and independent of the end-effector. On
  jelly's 31 slow_osc A/B variants it called the IMU's direction in 12/12
  that moved > 5% (r = 0.74 on the change), but it can't see flex past the
  encoders and sometimes credits a change the IMU didn't (dither: −10% vs
  0%), so a decision on it alone needs a bigger margin.

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
#: The wrist IMU's bands (``wrist_imu.LOW_BAND`` / ``HIGH_BAND``) and window.
_SWAY_BANDS = {"vertical_mm": (1.0, 15.0), "low_mm": (1.0, 3.0), "high_mm": (3.0, 15.0)}
_SWAY_WINDOW_S = 2.0
#: Forward kinematics is cheap but not free; 120 Hz holds the 15 Hz band.
_FK_RATE_HZ = 120.0
_fk_model = None


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


def flange_heights(q: np.ndarray, side: str) -> np.ndarray:
    """World height (m) of the arm's wrist flange (the ``wrist_3`` body
    origin) for each row of 7 joint angles (rad)."""
    import mujoco

    from ..constants import urdf_arm_body_names, urdf_arm_joint_names
    from ..robot.gravity import _load_urdf_text

    global _fk_model
    if _fk_model is None:
        _fk_model = mujoco.MjModel.from_xml_string(_load_urdf_text())
    model = _fk_model
    data = mujoco.MjData(model)
    left = side == "left"
    qadr = [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
        for n in urdf_arm_joint_names(is_left=left)
    ]
    body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, urdf_arm_body_names(is_left=left)[-1]
    )
    z = np.empty(len(q))
    for i, row in enumerate(np.asarray(q, dtype=float)):
        data.qpos[qadr] = row
        mujoco.mj_kinematics(model, data)
        z[i] = data.xpos[body][2]
    return z


def flange_sway(t: np.ndarray, actual: np.ndarray, side: str) -> dict[str, float]:
    """The wrist IMU's sway metrics, measured by the encoders instead.

    ``t`` (s) and ``actual`` (rad, 14 columns: left arm then right) as a run
    saves them. The flange's vertical position, band-limited the way
    :func:`~almond_axol.tuning.wrist_imu.shake_metrics` band-limits the
    IMU's integrated acceleration, median peak-to-peak over 2 s windows
    (mm). Empty when the record is too short or gappy.
    """
    t = np.asarray(t, dtype=float)
    cols = slice(0, 7) if side == "left" else slice(7, 14)
    a = np.asarray(actual, dtype=float)[:, cols]
    ok = np.isfinite(t) & np.isfinite(a).all(axis=1)
    if ok.mean() < 0.9 or ok.sum() < 50:
        return {}
    t, a = t[ok], a[ok]
    if t[-1] - t[0] < 2 * _SWAY_WINDOW_S:
        return {}
    dt = float(np.median(np.diff(t)))
    fs = min(1.0 / dt, _FK_RATE_HZ) if dt > 0 else _FK_RATE_HZ
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    q = np.stack([np.interp(grid, t, a[:, j]) for j in range(7)], axis=1)
    z = flange_heights(q, side)
    spec = np.fft.rfft(z - z.mean())
    f = np.fft.rfftfreq(len(z), 1.0 / fs)
    w = max(2, int(round(_SWAY_WINDOW_S * fs)))
    edge = min(w // 2, len(z) // 4)
    starts = range(edge, len(z) - w - edge + 1, max(1, w // 2))
    out = {}
    for key, (lo, hi) in _SWAY_BANDS.items():
        x = np.fft.irfft(
            np.where((f >= lo) & (f <= min(hi, 0.45 * fs)), spec, 0), len(z)
        )
        p2p = [float(np.ptp(x[s : s + w])) for s in starts]
        if p2p:
            out[key] = round(1e3 * float(np.median(p2p)), 3)
    return out


def score_run(run_id: str, runs_dir: Path = TUNING_RUNS_DIR) -> dict[str, Any]:
    """Score a saved run by id: labels, guard trips, IMU, encoder sway and
    the ripple."""
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
    actual = z["actual"].astype(float)
    out.update(score_series(z["t"], target, actual, side))
    enc = flange_sway(z["t"], actual, side)
    if enc:
        out["enc"] = enc
    imu = (metrics.get("imu") or {}).get(side)
    if imu:
        out["imu"] = {k: round(imu[k], 3) for k in _IMU_KEYS if k in imu}
    return out
