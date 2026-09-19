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
    by_speed: dict[float, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: {"+": [], "-": []}
    )
    joint = side = ""
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            joint, side = row["joint"], row["side"]
            by_speed[round(float(row["v_rad_s"]), 4)][row["direction"]].append(
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
    residuals: dict[float, np.ndarray] = {}
    for v in sorted(by_speed):
        fwd = np.array(by_speed[v]["+"]) if by_speed[v]["+"] else np.empty((0, 2))
        bwd = np.array(by_speed[v]["-"]) if by_speed[v]["-"] else np.empty((0, 2))
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
        residuals[v] = np.interp(centres, x, resid, left=np.nan, right=np.nan)
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
            f"{math.degrees(v):7.1f}°/s {resid.std():10.3f} Nm {floor:9.3f} Nm   {peaks}"
        )

    if len(residuals) >= 2:
        speeds = sorted(residuals)
        stack = np.array([residuals[v] for v in speeds])
        common = np.all(np.isfinite(stack), axis=0)
        if common.sum() > 20:
            c = np.corrcoef(stack[:, common])
            pairs = [
                (math.degrees(speeds[i]), math.degrees(speeds[j]), c[i, j])
                for i in range(len(speeds))
                for j in range(i + 1, len(speeds))
            ]
            print(
                "\ncorrelation of the position residual between speeds (cogging repeats, stick-slip does not):"
            )
            for a, b, r in pairs:
                print(f"   {a:.1f} vs {b:.1f} °/s: r = {r:+.2f}")
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
                    "tau_nm": [float(-t) for t in mean_resid],
                    "note": "feedforward to ADD at each joint-frame angle to cancel the measured position-periodic torque",
                }
                table.write_text(json.dumps(out, indent=1))
                print(f"table → {table}")
    print(
        "\nRead it as: a peak that sits at the same ° per cycle at every speed with r above ~0.7 "
        "between speeds is cogging or gear mesh and can be cancelled from a table; peaks that "
        "move with speed are time-domain (stick-slip, structural) and cannot."
    )


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
    args = p.parse_args()
    analyse(args.csv, args.grid_deg, args.table)


if __name__ == "__main__":
    main()
