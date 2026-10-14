"""Turn constant-speed ``tune.motion`` runs into ``tune.friction --raw-csv`` rows.

A creep motion (``s1_creep``, ``el_creep``) drives one joint back and forth at
fixed speeds with the rest of the arm held — a friction sweep flown through
the production controller. The motor's measured torque, forward against
backward over the same angle, is the joint's friction whatever feedforward
produced it, so these runs feed :func:`almond_axol.tuning.friction_model.fit_friction`
like a ``tune.friction`` sweep. The rows carry ``load_nm``, the gravity
torque at each sample's measured pose, since the held joints are not at the
``tune.friction`` sweep pose.

Usage:
    uv run python scripts/runs_to_friction_csv.py right.shoulder_1 out.csv RUN_ID [RUN_ID ...]
    uv run python scripts/runs_to_friction_csv.py right.elbow el.csv --runs-dir ~/runs 20260924-*
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from almond_axol.constants import ARM_JOINTS
from almond_axol.robot.gravity import GravityCompensator
from almond_axol.tuning.runs import TUNING_RUNS_DIR

#: Commanded speeds within this fraction of a leg's median are its cruise.
_CRUISE_TOL = 0.05
#: Seconds dropped at each end of a cruise (the speed blends settle).
_EDGE_S = 0.7


def cruise_rows(
    t: np.ndarray, target: np.ndarray, actual: np.ndarray, torque: np.ndarray, col: int
) -> list[tuple[float, str, float, float, int]]:
    """``(speed, direction, q, tau, sample index)`` for the cruise samples of
    column ``col``: commanded speed steady within ``_CRUISE_TOL``, ``_EDGE_S``
    clear of any change."""
    fs = (len(t) - 1) / (t[-1] - t[0])
    v = np.gradient(target[:, col].astype(float)) * fs
    speed = np.abs(v)
    moving = speed > math.radians(0.3)
    # Legs: runs of one direction; cruise speed = the leg's median.
    rows = []
    edge = int(_EDGE_S * fs)
    sign = np.sign(v) * moving
    start = 0
    for i in range(1, len(sign) + 1):
        if i == len(sign) or sign[i] != sign[start]:
            if sign[start] != 0 and i - start > 2 * edge + int(fs):
                seg = np.arange(start + edge, i - edge)
                med = float(np.median(speed[seg]))
                keep = seg[np.abs(speed[seg] - med) < _CRUISE_TOL * med]
                for k in keep:
                    rows.append(
                        (
                            round(med, 5),
                            "+" if sign[start] > 0 else "-",
                            float(actual[k, col]),
                            float(torque[k, col]),
                            int(k),
                        )
                    )
            start = i
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("joint", help="SIDE.JOINT, e.g. right.shoulder_1")
    p.add_argument("out", type=Path)
    p.add_argument("runs", nargs="+", help="run ids (directory names)")
    p.add_argument("--runs-dir", type=Path, default=TUNING_RUNS_DIR)
    args = p.parse_args()

    side, joint = args.joint.split(".")
    is_left = side == "left"
    gc = GravityCompensator()
    n = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "joint",
                "side",
                "pass",
                "v_rad_s",
                "direction",
                "q_rad",
                "tau_nm",
                "load_nm",
            ]
        )
        for pi, rid in enumerate(args.runs):
            d = args.runs_dir / rid
            meta = json.loads((d / "meta.json").read_text())
            cols = meta["params"]["columns"]
            col = cols.index(args.joint)
            base = cols.index(f"{side}.{ARM_JOINTS[0].value}")
            with np.load(d / "series.npz") as z:
                t, target, actual, torque = (
                    z[k] for k in ("t", "target", "actual", "torque")
                )
            rows = cruise_rows(t, target, actual, torque, col)
            for speed, direction, q, tau, k in rows:
                arm_q = actual[k, base : base + len(ARM_JOINTS)].astype(np.float32)
                load = float(
                    gc.gravity_arm(arm_q, is_left=is_left)[
                        ARM_JOINTS.index(ARM_JOINTS[col - base])
                    ]
                )
                w.writerow(
                    [
                        joint,
                        side,
                        pi,
                        f"{speed:.6f}",
                        direction,
                        f"{q:.6f}",
                        f"{tau:.6f}",
                        f"{load:.4f}",
                    ]
                )
            n += len(rows)
            print(f"  {rid} ({meta.get('label')}): {len(rows)} cruise samples")
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
