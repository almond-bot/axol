"""Offline analysis suites over teleop flight-recorder captures.

Each suite isolates one stage of the teleop pipeline and answers one
question from a recording alone (no hardware):

* :func:`wifi_analysis` — is the *transport* jittery? VR frame inter-arrival
  statistics, gaps and bursts, separated from tracking noise.
* :func:`filtering_analysis` — what does the *filter stack* pass through?
  Band RMS per stage (raw VR pose → filtered pose → world EE target) and
  the lag each stage adds.
* :func:`kinematics_analysis` — does the *IK* add motion the hand didn't
  make? End-effector tracking error (FK of the solved joints vs the world
  target), per-joint churn, and how much mid-band jitter each joint carries
  relative to the EE target.

All three consume the ``--teleop.record`` capture
(``<prefix>_ik.npz`` / ``<prefix>_cmd.npz``) and return ``(metrics,
series, params)`` in the standard tuning-run shape, so results persist via
:func:`~almond_axol.tuning.runs.save_run` and chart in the diagnostics UI
next to the hardware suites.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .metrics import band_rms, tracking_lag_ms

# A VR frame later than this after the previous one counts as a gap
# (nominal inter-arrival is ~11 ms at the Quest's 90 Hz).
_GAP_S = 0.1

_AXES = ("x", "y", "z")
_SIDES = ("l", "r")


def load_stage(prefix: str, stage: str) -> dict[str, np.ndarray]:
    """Load one flight-recorder stage or raise with a actionable message."""
    path = Path(f"{prefix}_{stage}.npz")
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found — record a session with "
            "`axol teleop --teleop.record <prefix>` first."
        )
    return dict(np.load(path))


def _engaged_mask(ik: dict[str, np.ndarray]) -> np.ndarray:
    """Boolean mask of ticks where either arm is engaged (all-False → all)."""
    engaged = ik["engaged"].max(axis=1) > 0.5
    return engaged if engaged.any() else np.ones(len(ik["t"]), dtype=bool)


# --------------------------------------------------------------------- #
# Wifi / transport                                                       #
# --------------------------------------------------------------------- #


def wifi_analysis(prefix: str) -> tuple[dict, dict, dict]:
    """VR-frame transport statistics from the ``_ik`` capture.

    The IK stage records the raw VR pose every solve tick; the pose only
    changes when a new VR frame arrived, so change ticks recover the frame
    arrival times without any extra instrumentation. Inter-arrival spread,
    gaps, and bursts are pure network/transport effects — hand motion can't
    fake them — which is what separates "the wifi is dropping frames" from
    "the tracking is noisy".
    """
    ik = load_stage(prefix, "ik")
    t = np.asarray(ik["t"], dtype=float)

    # A tick carries a new VR frame when either controller's raw pose moved.
    # Sides are checked independently: a disengaged controller records NaN,
    # which must not veto the other side's changes.
    changed = np.zeros(len(t), dtype=bool)
    for side in _SIDES:
        raw = np.asarray(ik[f"raw_{side}"], dtype=float)
        if len(t) < 2 or not np.isfinite(raw).any():
            continue
        valid = np.isfinite(raw).all(axis=1)
        d = np.abs(np.diff(raw, axis=0)).max(axis=1)
        changed[1:] |= (d > 1e-9) & valid[1:] & valid[:-1]
    arrivals = t[changed]
    if len(arrivals) < 16:
        raise ValueError(
            "capture holds too few VR frames to analyze — was a headset "
            "connected during the recording?"
        )
    ia = np.diff(arrivals)

    tick = np.diff(t)
    metrics = {
        "frames": int(len(arrivals)),
        "duration_s": float(arrivals[-1] - arrivals[0]),
        "rate_hz": float(len(arrivals) / max(arrivals[-1] - arrivals[0], 1e-9)),
        "interarrival_median_ms": float(np.median(ia) * 1000),
        "interarrival_p95_ms": float(np.percentile(ia, 95) * 1000),
        "interarrival_p99_ms": float(np.percentile(ia, 99) * 1000),
        "interarrival_max_ms": float(ia.max() * 1000),
        "gaps": int(np.sum(ia > _GAP_S)),
        "gap_time_s": float(ia[ia > _GAP_S].sum()) if (ia > _GAP_S).any() else 0.0,
        # Bursts: frames arriving back-to-back (under half the median
        # spacing) — the network delivering queued frames late, in a clump.
        "burst_frames": int(np.sum(ia < 0.5 * np.median(ia))),
        "tick_median_ms": float(np.median(tick) * 1000) if len(tick) else math.nan,
        "tick_p99_ms": float(np.percentile(tick, 99) * 1000) if len(tick) else math.nan,
    }
    series = {"arrival_t": arrivals, "interarrival_ms": ia * 1000}
    return metrics, series, {"prefix": str(prefix), "gap_threshold_ms": _GAP_S * 1000}


def print_wifi_report(metrics: dict) -> None:
    print(f"\n{'═' * 60}")
    print("  VR transport (wifi) report")
    print(
        f"  frames: {metrics['frames']}  over {metrics['duration_s']:.1f} s "
        f"({metrics['rate_hz']:.1f} Hz)"
    )
    print(
        f"  inter-arrival: median {metrics['interarrival_median_ms']:.1f} ms  "
        f"p95 {metrics['interarrival_p95_ms']:.1f}  "
        f"p99 {metrics['interarrival_p99_ms']:.1f}  "
        f"max {metrics['interarrival_max_ms']:.0f}"
    )
    print(
        f"  gaps (>{_GAP_S * 1000:.0f} ms): {metrics['gaps']}  "
        f"({metrics['gap_time_s']:.2f} s lost)   "
        f"burst frames: {metrics['burst_frames']}"
    )
    print(
        f"  IK tick interval: median {metrics['tick_median_ms']:.1f} ms  "
        f"p99 {metrics['tick_p99_ms']:.1f} ms"
    )
    print(f"{'═' * 60}")
    print(
        "  Healthy: p99 within ~2x the median and zero gaps. Gaps/bursts\n"
        "  with a quiet tick interval = network; a noisy tick interval too\n"
        "  = the host is starving the IK process."
    )


# --------------------------------------------------------------------- #
# Filtering                                                              #
# --------------------------------------------------------------------- #


def filtering_analysis(prefix: str) -> tuple[dict, dict, dict]:
    """Per-stage band pass-through and lag of the pose filter stack.

    Compares the raw VR pose against the filtered pose axis by axis: how
    much 3-15 Hz content (hand tremor + tracking noise) each stage removes,
    what survives to the world EE target, and how much delay the filtering
    costs. The jitter that *survives* here is what the IK and the arm are
    asked to reproduce — if the mid-band pass-through is high, tune the
    filter, not the motors.
    """
    ik = load_stage(prefix, "ik")
    m = _engaged_mask(ik)
    t = np.asarray(ik["t"], dtype=float)[m]

    per_axis: dict[str, dict[str, float]] = {}
    lags: list[float] = []
    ratios: list[float] = []
    tgt_mid: list[float] = []
    for side in _SIDES:
        raw = np.asarray(ik[f"raw_{side}"], dtype=float)[m]
        filt = np.asarray(ik[f"filt_{side}"], dtype=float)[m]
        tgt = np.asarray(ik[f"tgt_{side}"], dtype=float)[m]
        for ax in range(3):
            if not np.isfinite(raw[:, ax]).any():
                continue
            rb = band_rms(t, raw[:, ax])
            fb = band_rms(t, filt[:, ax])
            tb = band_rms(t, tgt[:, ax])
            lag = tracking_lag_ms(t, raw[:, ax], filt[:, ax])
            ratio = fb["rms_mid"] / rb["rms_mid"] if rb["rms_mid"] > 1e-7 else math.nan
            per_axis[f"{side}_{_AXES[ax]}"] = {
                "raw_mid_mm": rb["rms_mid"] * 1000,
                "filt_mid_mm": fb["rms_mid"] * 1000,
                "tgt_mid_mm": tb["rms_mid"] * 1000,
                "passthrough": ratio,
                "lag_ms": lag,
            }
            if math.isfinite(ratio):
                ratios.append(ratio)
            if math.isfinite(lag):
                lags.append(lag)
            if math.isfinite(tb["rms_mid"]):
                tgt_mid.append(tb["rms_mid"] * 1000)

    if not per_axis:
        raise ValueError("capture holds no finite VR pose data")
    metrics = {
        "per_axis": per_axis,
        "mean_passthrough": float(np.mean(ratios)) if ratios else math.nan,
        "mean_lag_ms": float(np.mean(lags)) if lags else math.nan,
        "worst_lag_ms": float(np.max(lags)) if lags else math.nan,
        "worst_tgt_mid_mm": float(np.max(tgt_mid)) if tgt_mid else math.nan,
    }
    series = {
        "t": t,
        **{f"raw_{s}": np.asarray(ik[f"raw_{s}"], dtype=float)[m] for s in _SIDES},
        **{f"filt_{s}": np.asarray(ik[f"filt_{s}"], dtype=float)[m] for s in _SIDES},
        **{f"tgt_{s}": np.asarray(ik[f"tgt_{s}"], dtype=float)[m] for s in _SIDES},
    }
    return metrics, series, {"prefix": str(prefix)}


def print_filtering_report(metrics: dict) -> None:
    print(f"\n{'═' * 72}")
    print("  Pose filter stack report (3-15 Hz mid band, mm RMS)")
    print(
        f"  {'axis':<6} {'raw':>8} {'filtered':>9} {'EE target':>10} "
        f"{'pass':>6} {'lag ms':>7}"
    )
    for name, row in metrics["per_axis"].items():
        pas = f"{row['passthrough']:.2f}" if math.isfinite(row["passthrough"]) else "-"
        lag = f"{row['lag_ms']:.0f}" if math.isfinite(row["lag_ms"]) else "-"
        print(
            f"  {name:<6} {row['raw_mid_mm']:>8.2f} {row['filt_mid_mm']:>9.2f} "
            f"{row['tgt_mid_mm']:>10.2f} {pas:>6} {lag:>7}"
        )
    print(f"{'═' * 72}")
    print(
        f"  mean pass-through {metrics['mean_passthrough']:.2f} "
        f"(0 = filters everything, 1 = filters nothing), "
        f"lag {metrics['mean_lag_ms']:.0f} ms mean / "
        f"{metrics['worst_lag_ms']:.0f} ms worst.\n"
        "  Mid-band content that reaches the EE target is what the arm is\n"
        "  asked to reproduce — if it's high, tune the filter, not the motors."
    )


# --------------------------------------------------------------------- #
# Kinematics                                                             #
# --------------------------------------------------------------------- #

# FK evaluation cap: enough rows to resolve the band metrics while keeping
# the offline pass quick on the robot host.
_FK_MAX_ROWS = 6000


def kinematics_analysis(prefix: str) -> tuple[dict, dict, dict]:
    """IK behavior over a session: EE tracking, joint churn, jitter injection.

    Runs FK over the recorded solved joints and compares against the world
    EE target the solver was asked to reach (both are in the ``_ik``
    capture, so no re-solving is needed). Reports per-arm EE error, and per
    joint: path length ("churn" — a restless null space shows up as churn
    without EE motion) and mid-band RMS. Requires the kinematics stack
    (jax JIT on first call).
    """
    from ..constants import ARM_JOINTS
    from ..kinematics.solver import KinematicsSolver

    ik = load_stage(prefix, "ik")
    m = _engaged_mask(ik)
    t = np.asarray(ik["t"], dtype=float)[m]
    q = np.asarray(ik["q"], dtype=float)[m]
    if len(t) < 64:
        raise ValueError("capture holds too few engaged IK ticks to analyze")

    solver = KinematicsSolver()
    stride = max(1, len(t) // _FK_MAX_ROWS)
    idx = np.arange(0, len(t), stride)
    ee_err = {"l": np.empty(len(idx)), "r": np.empty(len(idx))}
    fk_pos = {"l": np.empty((len(idx), 3)), "r": np.empty((len(idx), 3))}
    for k, i in enumerate(idx):
        (lp, _), (rp, _) = solver.fk(q[i].astype(np.float32))
        fk_pos["l"][k] = lp
        fk_pos["r"][k] = rp
        ee_err["l"][k] = float(np.linalg.norm(lp - ik["tgt_l"][m][i]))
        ee_err["r"][k] = float(np.linalg.norm(rp - ik["tgt_r"][m][i]))

    duration = max(t[-1] - t[0], 1e-9)
    per_joint: dict[str, dict[str, float]] = {}
    names = [f"left.{j.value}" for j in ARM_JOINTS] + [
        f"right.{j.value}" for j in ARM_JOINTS
    ]
    cols = list(solver.left_indices) + list(solver.right_indices)
    for name, col in zip(names, cols):
        x = q[:, col]
        jb = band_rms(t, x)
        per_joint[name] = {
            "churn_deg_min": math.degrees(float(np.abs(np.diff(x)).sum()))
            / (duration / 60.0),
            "mid_band_deg": math.degrees(jb["rms_mid"]),
            "peak_hz": jb["peak_hz"],
        }

    engaged_sides = ik["engaged"][m].max(axis=0) > 0.5
    metrics = {
        "per_joint": per_joint,
        "ee_rms_mm": {
            s: float(np.sqrt(np.mean(ee_err[s] ** 2)) * 1000)
            for s, on in zip(_SIDES, engaged_sides)
            if on
        },
        "ee_max_mm": {
            s: float(ee_err[s].max() * 1000)
            for s, on in zip(_SIDES, engaged_sides)
            if on
        },
        "worst_churn_joint": max(
            per_joint, key=lambda k: per_joint[k]["churn_deg_min"]
        ),
        "ik_rate_hz": float((len(t) - 1) / duration),
    }
    series = {
        "t": t[idx],
        "ee_err_l": ee_err["l"],
        "ee_err_r": ee_err["r"],
        # EE pose set (world target given to the solver) vs actual (FK of
        # the solved joints), per axis — the pair the UI charts.
        "ee_tgt_l": np.asarray(ik["tgt_l"], dtype=float)[m][idx],
        "ee_tgt_r": np.asarray(ik["tgt_r"], dtype=float)[m][idx],
        "ee_fk_l": fk_pos["l"],
        "ee_fk_r": fk_pos["r"],
        "q_t": t,
        "q": q,
    }
    # Solve time (per-tick solver duration) — captures made before the field
    # existed simply don't have it.
    if "solve_ms" in ik:
        solve = np.asarray(ik["solve_ms"], dtype=float)[m].reshape(-1)
        finite = solve[np.isfinite(solve)]
        if len(finite) > 0:
            series["solve_ms"] = solve
            metrics["solve_median_ms"] = float(np.median(finite))
            metrics["solve_p95_ms"] = float(np.percentile(finite, 95))
            metrics["solve_max_ms"] = float(finite.max())
    return metrics, series, {"prefix": str(prefix), "fk_stride": int(stride)}


def print_kinematics_report(metrics: dict) -> None:
    print(f"\n{'═' * 66}")
    print("  IK / kinematics report")
    for side, label in (("l", "left"), ("r", "right")):
        if side in metrics["ee_rms_mm"]:
            print(
                f"  {label} EE tracking: {metrics['ee_rms_mm'][side]:.1f} mm RMS  "
                f"(max {metrics['ee_max_mm'][side]:.1f} mm)"
            )
    if "solve_median_ms" in metrics:
        print(
            f"  solve time: median {metrics['solve_median_ms']:.1f} ms  "
            f"p95 {metrics['solve_p95_ms']:.1f}  max {metrics['solve_max_ms']:.1f}  "
            f"(dispatch {metrics['ik_rate_hz']:.0f} Hz)"
        )
    print(f"\n  {'joint':<18} {'churn °/min':>11} {'3-15 Hz °':>10} {'peak Hz':>8}")
    for name, row in metrics["per_joint"].items():
        peak = f"{row['peak_hz']:.1f}" if math.isfinite(row["peak_hz"]) else "-"
        print(
            f"  {name:<18} {row['churn_deg_min']:>11.0f} "
            f"{row['mid_band_deg']:>10.3f} {peak:>8}"
        )
    print(f"{'═' * 66}")
    print(
        f"  worst churn: {metrics['worst_churn_joint']}. Churn without EE\n"
        "  motion = a restless null space (IK settings); mid-band content in\n"
        "  a joint but not in the EE target = jitter the IK is injecting."
    )
