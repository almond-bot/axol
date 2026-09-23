"""Look for position-periodic torque (cogging / gear mesh) in a friction sweep
and turn it into a feedforward table.

Input is the ``--raw-csv`` of ``axol tune.friction``: every cruise sample of a
bidirectional constant-speed sweep, ``(joint, side, v_rad_s, direction,
q_rad, tau_nm)``. The forward and backward passes at one speed are averaged
on a fine angle grid, which cancels the speed-dependent friction and leaves
gravity plus anything that depends on *position*. A smooth trend (the gravity
model's residual) is removed, and the remainder is analysed in the **angle**
domain: cogging and gear mesh have a fixed period in degrees of travel and
line up across speeds, whereas stick-slip and structural ringing have a fixed
period in *time* and therefore change spatial period with speed.

Usage:
    uv run python scripts/cogging_map.py ~/fric-s1-raw.csv
    uv run python scripts/cogging_map.py ~/fric-s1-raw.csv --grid-deg 0.1 --table ~/cog-s1.json
    uv run python scripts/cogging_map.py ~/fric-s1-raw.csv --fit            # Fourier series
    uv run python scripts/cogging_map.py ~/fric-s1-raw.csv --fit --save     # → calibration

``--fit`` fits the Fourier series the realtime core cancels
(``almond_axol.tuning.cogging``: a ``--period`` fundamental, default 3.62°,
with ``--harmonics``, default 1 2 4 — the right shoulder_1's 3.62° / 1.81° /
0.905° ripple), reports each pass's fit and how well it predicts the *other*
passes (the out-of-sample test — a series that only fits its own pass cancels
nothing), and with ``--save`` writes it to this robot's calibration file as
the joint's ``cogging`` entry. It then applies on every bring-up: added to the
MIT feedforward on an impedance joint, carried by 0x73 on a firmware-loop joint
with ``firmware.tf_rated_current_a`` set.

The table (``--table``) is ``{"joint", "side", "grid_deg", "q_deg": [...],
"tau_nm": [...]}``: the periodic torque to *add* to the feedforward at each
angle so the motor cancels it. Only worth wiring in if the report shows a
peak that is consistent across speeds and well above the noise floor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def grid_average(q: np.ndarray, tau: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Mean torque per grid cell (NaN where empty)."""
    idx = np.clip(np.searchsorted(grid, q) - 1, 0, len(grid) - 2)
    sums = np.zeros(len(grid) - 1)
    counts = np.zeros(len(grid) - 1)
    np.add.at(sums, idx, tau)
    np.add.at(counts, idx, 1)
    with np.errstate(invalid="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def detrend(x: np.ndarray, y: np.ndarray, degree: int = 5) -> np.ndarray:
    """Remove a low-order polynomial (the gravity residual and any drift)."""
    ok = np.isfinite(y)
    coef = np.polyfit(x[ok], y[ok], degree)
    return y - np.polyval(coef, x)


def angle_spectrum(y: np.ndarray, grid_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Power spectrum against spatial frequency (cycles per degree)."""
    ok = np.isfinite(y)
    z = np.where(ok, y, 0.0)
    z = z - z[ok].mean()
    window = np.hanning(len(z))
    power = np.abs(np.fft.rfft(z * window)) ** 2
    freq = np.fft.rfftfreq(len(z), grid_deg)
    return freq, power


def analyse(path: Path, grid_deg: float, table: Path | None) -> None:
    # Keyed by (pass index, speed) so repeated passes at one speed stay apart —
    # pass-to-pass repeatability at the same speed is the strictest test of
    # whether a position table could cancel anything.
    by_speed: dict[tuple[int, float], dict[str, list[tuple[float, float]]]] = (
        defaultdict(lambda: {"+": [], "-": []})
    )
    joint = side = ""
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            joint, side = row["joint"], row["side"]
            key = (int(row.get("pass", 0) or 0), round(float(row["v_rad_s"]), 4))
            by_speed[key][row["direction"]].append(
                (math.degrees(float(row["q_rad"])), float(row["tau_nm"]))
            )
    if not by_speed:
        raise SystemExit("no samples in the CSV")
    all_q = np.array(
        [q for d in by_speed.values() for rows in d.values() for q, _ in rows]
    )
    lo, hi = np.floor(all_q.min()), np.ceil(all_q.max())
    grid = np.arange(lo, hi + grid_deg, grid_deg)
    centres = grid[:-1] + grid_deg / 2

    print(
        f"{side} {joint}: {len(all_q)} samples over {lo:.0f}..{hi:.0f}°, {grid_deg}° grid"
    )
    print(
        f"{'speed':>8s} {'periodic RMS':>13s} {'noise floor':>12s} {'top spatial peaks (° per cycle : Nm)':>40s}"
    )
    residuals: dict[tuple[int, float], np.ndarray] = {}
    for key in sorted(by_speed):
        _pass, v = key
        fwd = np.array(by_speed[key]["+"]) if by_speed[key]["+"] else np.empty((0, 2))
        bwd = np.array(by_speed[key]["-"]) if by_speed[key]["-"] else np.empty((0, 2))
        if len(fwd) < 50 or len(bwd) < 50:
            continue
        f_avg = grid_average(fwd[:, 0], fwd[:, 1], grid)
        b_avg = grid_average(bwd[:, 0], bwd[:, 1], grid)
        both = np.isfinite(f_avg) & np.isfinite(b_avg)
        avg = np.where(both, (f_avg + b_avg) / 2.0, np.nan)
        # Restrict to the span both directions covered.
        ok = np.flatnonzero(both)
        if len(ok) < 20:
            continue
        span = slice(ok[0], ok[-1] + 1)
        x = centres[span]
        y = avg[span]
        # Fill small gaps by interpolation so the spectrum is not spiked.
        good = np.isfinite(y)
        y = np.interp(x, x[good], y[good])
        resid = detrend(x, y)
        residuals[key] = np.interp(centres, x, resid, left=np.nan, right=np.nan)
        freq, power = angle_spectrum(resid, grid_deg)
        # Ignore anything slower than one cycle per 10° (that is the trend).
        sel = freq > 0.1
        order = np.argsort(power[sel])[::-1][:3]
        amps = (
            np.sqrt(power[sel][order] / power[sel].sum()) * resid.std() * math.sqrt(2)
        )
        peaks = ", ".join(
            f"{1 / freq[sel][i]:.2f}° : {a:.3f}" for i, a in zip(order, amps)
        )
        # Noise floor: median spectral amplitude.
        floor = (
            math.sqrt(np.median(power[sel]) / power[sel].sum())
            * resid.std()
            * math.sqrt(2)
        )
        print(
            f"{math.degrees(v):5.1f}°/s #{_pass} {resid.std():8.3f} Nm {floor:9.3f} Nm   {peaks}"
        )

    if len(residuals) >= 2:
        keys = sorted(residuals)
        stack = np.array([residuals[k] for k in keys])
        common = np.all(np.isfinite(stack), axis=0)
        if common.sum() > 20:
            c = np.corrcoef(stack[:, common])
            label = lambda k: f"{math.degrees(k[1]):.1f}°/s #{k[0]}"  # noqa: E731
            pairs = [
                (label(keys[i]), label(keys[j]), c[i, j])
                for i in range(len(keys))
                for j in range(i + 1, len(keys))
            ]
            print(
                "\ncorrelation of the position residual between passes (cogging repeats, stick-slip does not):"
            )
            for a, b, r in pairs:
                print(f"   {a} vs {b}: r = {r:+.2f}")
            mean_resid = stack[:, common].mean(axis=0)
            print(
                f"speed-averaged periodic torque: {mean_resid.std():.3f} Nm RMS, {np.ptp(mean_resid):.3f} Nm peak-to-peak"
            )
            if table is not None:
                out = {
                    "joint": joint,
                    "side": side,
                    "grid_deg": grid_deg,
                    "q_deg": [float(q) for q in centres[common]],
                    # The torque the motor supplied through each bump is what
                    # it has to be given ahead of time: the residual as-is.
                    "tau_nm": [float(t) for t in mean_resid],
                    "note": "feedforward to ADD at each joint-frame angle to cancel the measured position-periodic torque",
                }
                table.write_text(json.dumps(out, indent=1))
                print(f"table → {table}")
    print(
        "\nRead it as: a peak that sits at the same ° per cycle at every speed with r above ~0.7 "
        "between speeds is cogging or gear mesh and can be cancelled from a table; peaks that "
        "move with speed are time-domain (stick-slip, structural) and cannot."
    )


def fit_series(
    path: Path, period_deg: float, harmonics: tuple[int, ...], save: bool
) -> None:
    """Fit the Fourier series per pass and pooled; optionally save it."""
    from almond_axol.robot.calibration import update_joint_calibration
    from almond_axol.tuning.cogging import fit_cogging, prediction_r

    passes: dict[int, list[tuple[float, float, str]]] = defaultdict(list)
    joint = side = ""
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            joint, side = row["joint"], row["side"]
            passes[int(row.get("pass", 0) or 0)].append(
                (float(row["q_rad"]), float(row["tau_nm"]), row["direction"])
            )
    if not passes:
        raise SystemExit("no samples in the CSV")

    def cols(rows: list[tuple[float, float, str]]):
        q, t, d = zip(*rows)
        return np.array(q), np.array(t), np.array(d)

    names = ", ".join(f"{period_deg / k:.3g}°" for k in harmonics)
    print(f"\n{side} {joint}: Fourier fit, {period_deg}° fundamental ({names})")
    fits = {}
    for k in sorted(passes):
        try:
            fits[k] = fit_cogging(
                *cols(passes[k]), period_deg=period_deg, harmonics=harmonics
            )
        except ValueError as exc:
            print(f"   pass {k}: {exc}")
    for k, fit in fits.items():
        others = {o: prediction_r(fit.model, *cols(passes[o])) for o in fits if o != k}
        amps = ", ".join(
            f"{period_deg / h:.3g}°: {a:.3f} Nm"
            for h, a in zip(harmonics, fit.amplitudes)
        )
        pred = ", ".join(f"#{o} r={r:+.2f}" for o, r in others.items())
        print(
            f"   pass {k}: R² {fit.r2:.2f} of {fit.ripple_rms:.3f} Nm ripple — {amps}"
            + (f" | predicts {pred}" if pred else "")
        )
    pooled = fit_cogging(
        *cols([r for rows in passes.values() for r in rows]),
        period_deg=period_deg,
        harmonics=harmonics,
    )
    amps = ", ".join(
        f"{period_deg / h:.3g}°: {a:.3f} Nm"
        for h, a in zip(harmonics, pooled.amplitudes)
    )
    print(f"   pooled: R² {pooled.r2:.2f} — {amps}")
    print(
        "   Worth cancelling when every pass shows the same amplitudes and each "
        "predicts the others at r ≳ 0.4."
    )
    if save:
        if side not in ("left", "right") or not joint:
            raise SystemExit("the CSV names no side/joint — cannot save")
        out = update_joint_calibration(side, joint, cogging=pooled.model.as_dict())
        print(f"   saved {side}.{joint} cogging → {out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("csv", type=Path, help="--raw-csv output of axol tune.friction")
    p.add_argument(
        "--grid-deg", type=float, default=0.1, help="Angle grid (default: 0.1°)"
    )
    p.add_argument(
        "--table",
        type=Path,
        default=None,
        help="Write the cancellation table here (JSON)",
    )
    p.add_argument(
        "--fit",
        action="store_true",
        help="Fit the Fourier series the realtime core cancels (see the docstring)",
    )
    p.add_argument(
        "--period",
        type=float,
        default=3.62,
        help="Fundamental period of the fit, degrees (default: 3.62)",
    )
    p.add_argument(
        "--harmonics",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Harmonic numbers to fit (default: 1 2 4)",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="With --fit: write the pooled series to this robot's calibration file",
    )
    args = p.parse_args()
    analyse(args.csv, args.grid_deg, args.table)
    if args.fit or args.save:
        fit_series(args.csv, args.period, tuple(args.harmonics), args.save)


if __name__ == "__main__":
    main()
