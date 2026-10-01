"""Score a tip-damping session: damped against undamped passes, per group.

Reads saved ``tune.motion`` runs (``~/.almond/diagnostics/tuning``) and groups
them by label (``"pd 120 [2/4]"`` → ``"pd 120"``). Within each group the
passes that flew damping (``metrics.imu_damp`` or ``metrics.gyro_damp`` > 0)
are compared with the ones that did not — ``--imu-damp-alternate`` puts both
in one session, so the comparison is free of the arm's drift over a day.

Per pass it prints the wrist IMU's shake score (vertical 2 s p2p, split into
1-3 and 3-15 Hz, and the band acceleration — the buzz a damper adds shows
there first) and the tool's deviation from its commanded path (RMS, 1-3 and
3-15 Hz: the IMU's gravity-tracked vertical displacement less the commanded
tool height, the machinery ``--learn-imu`` learns against).

    uv run python scripts/damp_report.py --since 20261002-090000
    uv run python scripts/damp_report.py --label "pd model"   # label prefix
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

RUNS = Path.home() / ".almond" / "diagnostics" / "tuning"


def _group(label: str) -> str:
    return re.sub(r"\s*\[\d+/\d+\]\s*$", "", label or "")


def _deviation(series: dict, side: str, solver) -> tuple[float, float]:
    """Tool deviation from the commanded path, RMS mm over 1-3 and 3-15 Hz."""
    from almond_axol.cli.tune.motion import _tool_error_as_joints

    t = np.asarray(series["t"], dtype=float)
    ref = np.asarray(series["target"], dtype=float)
    cols = np.zeros(14, dtype=bool)
    cols[7 if side == "right" else 0] = True

    def to_full(row):
        q = np.zeros(solver.num_joints, dtype=np.float32)
        q[solver.left_indices] = row[:7]
        q[solver.right_indices] = row[7:]
        return q

    out = []
    for band in ((1.0, 3.0), (3.0, 15.0)):
        _, rms = _tool_error_as_joints(
            t,
            np.asarray(series[f"imu_{side}_t"], dtype=float),
            np.asarray(series[f"imu_{side}_acc"], dtype=float),
            np.asarray(series[f"imu_{side}_gyro"], dtype=float),
            ref,
            cols,
            side,
            to_full,
            solver,
            band,
        )
        out.append(rms * 1e3)
    return out[0], out[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--since", help="Only runs with an id at or after this (YYYYMMDD-HHMMSS)"
    )
    p.add_argument("--label", help="Only groups whose label starts with this")
    p.add_argument("--runs", default=str(RUNS), help="Runs directory")
    p.add_argument(
        "--no-deviation", action="store_true", help="Skip the path deviation (faster)"
    )
    args = p.parse_args()

    from almond_axol.tuning.runs import load_run

    solver = None
    if not args.no_deviation:
        from almond_axol.kinematics.solver import KinematicsSolver

        solver = KinematicsSolver()

    groups: dict[str, list[dict]] = {}
    for d in sorted(Path(args.runs).iterdir()):
        if args.since and d.name < args.since:
            continue
        try:
            meta = json.loads((d / "meta.json").read_text())
        except (OSError, ValueError):
            continue
        if meta.get("kind") != "motion":
            continue
        g = _group(meta.get("label", ""))
        if args.label and not g.startswith(args.label):
            continue
        m = meta.get("metrics", {})
        imu = m.get("imu") or {}
        side = meta.get("params", {}).get("arms")
        side = side if side in ("left", "right") else next(iter(imu), None)
        if side is None or side not in imu:
            continue
        row = {
            "id": d.name,
            "motion": meta.get("params", {}).get("motion"),
            "damped": (m.get("imu_damp") or 0) > 0 or (m.get("gyro_damp") or 0) > 0,
            "vertical": imu[side]["vertical_mm"],
            "low": imu[side]["low_mm"],
            "high": imu[side]["high_mm"],
            "acc": imu[side]["acc_rms"],
            "dev13": float("nan"),
            "dev315": float("nan"),
        }
        if solver is not None:
            try:
                _, series = load_run(d.name, Path(args.runs))
                row["dev13"], row["dev315"] = _deviation(series, side, solver)
            except Exception as e:  # noqa: BLE001 - a broken run should not stop the report
                print(f"  ({d.name}: no deviation — {e})")
        groups.setdefault(g, []).append(row)

    head = f"{'':10s} {'vertical':>8s} {'1-3':>6s} {'3-15':>6s} {'accel':>6s} | {'dev 1-3':>7s} {'dev 3-15':>8s}"
    for g, rows in groups.items():
        print(f"\n== {g}  ({rows[0]['motion']})")
        print(head)
        for kind in (False, True):
            sel = [r for r in rows if r["damped"] == kind]
            if not sel:
                continue
            for r in sel:
                print(
                    f"{'damped' if kind else 'undamped':10s} {r['vertical']:8.2f} {r['low']:6.2f} "
                    f"{r['high']:6.2f} {r['acc']:6.3f} | {r['dev13']:7.3f} {r['dev315']:8.3f}"
                    f"   {r['id']}"
                )
        on = [r for r in rows if r["damped"]]
        off = [r for r in rows if not r["damped"]]
        if on and off:

            def mean(rs, k):
                return float(np.nanmean([r[k] for r in rs]))

            cells = []
            for k in ("vertical", "low", "high", "acc", "dev13", "dev315"):
                a, b = mean(off, k), mean(on, k)
                cells.append(f"{k} {100 * (b - a) / a:+.0f}%" if a else f"{k} -")
            print("  change: " + ", ".join(cells))


if __name__ == "__main__":
    main()
