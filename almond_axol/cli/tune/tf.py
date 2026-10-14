"""
axol tune.tf

Fit a joint's tracking dynamics from a replayed chirp.

``axol motion.chirp SIDE.JOINT`` writes a one-joint sine sweep;
``axol tune.motion --motion <that file>`` flies it through the production
controller (saving the run, and the wrist IMU); this command estimates the
command → position frequency response from the saved run(s) and fits

    H(s) = K (1 + s/ωz) ωn² / (s² + 2ζωn s + ωn²) · e^{−sτ}

— the closed loop's natural frequency, damping, a zero and the delay. With
``--save`` the model goes to ``~/.almond/tracking_models.json``, where
``tune.motion --invert`` pre-compensates a known motion with its inverse.

The model belongs to the gains it was measured on (the run's overrides are
stored with it): re-measure after changing kp, kd or the host damping.

Examples:
    axol motion.chirp right.shoulder_1 --carrier 3
    axol tune.motion --motion ~/.almond/motions/chirp_right_shoulder_1.npz --arms right --label chirp-s1
    axol tune.tf right.shoulder_1 20260927-101500-abc123 --save
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from ...tuning.runs import load_run
from ...tuning.tracking_model import (
    TRACKING_MODELS_PATH,
    fit_tracking_model,
    frequency_response,
    save_model,
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "tune.tf",
        help="Fit a joint's tracking transfer function from a replayed chirp.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("joint", metavar="SIDE.JOINT")
    p.add_argument("runs", nargs="+", metavar="RUN_ID", help="tune.motion chirp run(s)")
    p.add_argument(
        "--f0",
        type=float,
        default=None,
        help="Band low edge, Hz (default: the chirp's)",
    )
    p.add_argument(
        "--f1",
        type=float,
        default=None,
        help="Band high edge, Hz (default: the chirp's)",
    )
    p.add_argument(
        "--min-coherence",
        type=float,
        default=0.6,
        help="Frequency bins below this coherence are left out of the fit (default 0.6)",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help=f"Store the model in {TRACKING_MODELS_PATH} for tune.motion --invert",
    )
    p.set_defaults(func=run)


def _series(
    run_id: str, joint: str
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict]:
    loaded = load_run(run_id)
    if loaded is None:
        raise SystemExit(f"tune.tf: no tuning run {run_id!r}")
    meta, series = loaded
    cols = meta.get("params", {}).get("columns") or []
    if joint not in cols:
        raise SystemExit(f"tune.tf: run {run_id} has no column {joint!r}")
    c = cols.index(joint)
    return meta, series["t"], series["target"][:, c], series["actual"][:, c], series


def run(args: argparse.Namespace) -> None:
    fs_all, refs, outs, metas = [], [], [], []
    imu_parts = []
    for rid in args.runs:
        meta, t, ref, act, series = _series(rid, args.joint)
        fs_all.append((len(t) - 1) / (t[-1] - t[0]))
        refs.append(ref.astype(float))
        outs.append(act.astype(float))
        metas.append(meta)
        side = args.joint.split(".")[0]
        if f"imu_{side}_t" in series:
            imu_parts.append(
                (t, ref, series[f"imu_{side}_t"], series[f"imu_{side}_acc"])
            )
    fs = float(np.median(fs_all))
    f0 = args.f0 if args.f0 is not None else 0.3
    f1 = args.f1 if args.f1 is not None else 8.0
    ref = np.concatenate(refs)
    out = np.concatenate(outs)
    ok = np.isfinite(ref) & np.isfinite(out)
    f, h, coh = frequency_response(ref[ok], out[ok], fs, (f0, f1))
    print(
        f"\n{args.joint}: {len(args.runs)} run(s), {ok.sum() / fs:.0f} s at {fs:.0f} Hz"
    )
    print("   f Hz   |H|    phase°   coherence")
    for target in (0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0):
        if f0 <= target <= f1:
            i = int(np.argmin(np.abs(f - target)))
            print(
                f"  {f[i]:5.2f}  {abs(h[i]):5.2f}  {math.degrees(np.angle(h[i])):+7.1f}   {coh[i]:.2f}"
            )
    try:
        model = fit_tracking_model(f, h, coh, args.min_coherence)
    except ValueError as exc:
        raise SystemExit(f"tune.tf: {exc}") from None
    gains = metas[0].get("gains") or {}
    model = type(model)(
        **{**model.__dict__, "gains": {k: float(v) for k, v in gains.items()}}
    )
    peak = float(f[np.argmax(np.abs(h))])
    print(
        f"\n  model: natural frequency {model.fn_hz:.2f} Hz, damping ratio "
        f"{model.zeta:.3f}, delay {model.tau * 1e3:.0f} ms, static gain {model.k:.3f}"
        + (
            f", zero at {model.wz / (2 * math.pi):.2f} Hz"
            if math.isfinite(model.wz)
            else ""
        )
    )
    print(
        f"  measured peak |H| {np.abs(h).max():.2f} at {peak:.2f} Hz; fit residual "
        f"{model.fit_rms:.3f} (coherence-weighted |H| error) over "
        f"{model.f_lo:.2f}-{model.f_hi:.2f} Hz"
    )
    if model.fn_hz > 1.2 * model.f_hi:
        print(
            f"  ! the fitted resonance ({model.fn_hz:.1f} Hz) is above the "
            f"identified band (to {model.f_hi:.1f} Hz): its frequency and damping "
            "are extrapolated — only the lag below the band top is measured. "
            "Excite higher (a chirp to 8 Hz) before trusting or inverting it."
        )
    elif model.zeta < 0.3:
        print(
            f"  ! lightly damped: a command component near {model.fn_hz:.1f} Hz "
            f"is amplified up to {1 / (2 * model.zeta):.1f}×"
        )
    if imu_parts:
        # How much of the wrist's vertical motion follows the joint command,
        # per band: where the IMU coherence stays high the joint model is the
        # whole story; where it drops, something the encoder does not see.
        t, r, it, acc = imu_parts[0]
        up = acc.mean(axis=0) / np.linalg.norm(acc.mean(axis=0))
        av = np.interp(t, it, acc @ up)
        fi, hi, ci = frequency_response(r.astype(float), av, fs, (f0, f1))
        print("\n  wrist IMU (vertical accel) vs command — coherence by band:")
        for lo_, hi_ in ((f0, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, f1)):
            sel = (fi >= lo_) & (fi < hi_)
            if sel.any():
                print(f"    {lo_:.1f}-{hi_:.1f} Hz: {ci[sel].mean():.2f}")
    if args.save:
        path = save_model(args.joint, model)
        print(f"\n  saved to {path}")
