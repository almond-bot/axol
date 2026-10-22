"""Score tuning runs with an IMU you mounted on your own end-effector.

For robots without the wrist ZED cameras: any IMU bolted rigidly to the
end-effector, near the tool tip, logged to a CSV while the runs fly. Each
run's slice gets the same 1-3 Hz sway / 3-15 Hz shake metrics as the wrist
camera's IMU, and ``scripts/tuning_queue.py summary`` decides on them.

The CSV: ``t, ax, ay, az[, gx, gy, gz]`` per row, header optional.
``t`` is Unix time (``time.time()``; seconds or milliseconds) — log on the
robot computer, or on one synced to it by NTP. Acceleration must include
gravity (raw accelerometer), in m/s² or g (``--units``, guessed by
default). 100 Hz minimum, 200 Hz or more preferred.

    # before the session: is the log usable?
    uv run python scripts/ext_imu.py check ~/imu.csv
    # after it: attach every run of a queue session (or --run ID ...)
    uv run python scripts/ext_imu.py attach ~/imu.csv --session ~/tuning/s3
    uv run python scripts/tuning_queue.py summary ~/tuning/s3

Each run is matched by its wall-clock start, refined by cross-correlating
the IMU's vertical acceleration with the flange's from the encoders (± 2 s;
``--search``). A run whose motion isn't found in the log (wrong arm, loose
mount, clocks further apart) is reported and left without IMU metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _runs(args: argparse.Namespace) -> list[tuple[str, str | None]]:
    if args.run:
        return [(r, args.side) for r in args.run]
    path = Path(args.session).expanduser() / "results.jsonl"
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for rid in rec.get("ids") or [s.get("id") for s in rec.get("scores", [])]:
            if rid:
                out.append((rid, args.side or rec["item"].get("arm")))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    ck = sub.add_parser("check", help="rate, span and gravity of a log (JSON)")
    ck.add_argument("log")
    ck.add_argument("--units", choices=("auto", "ms2", "g"), default="auto")
    at = sub.add_parser("attach", help="score runs with their slice of the log")
    at.add_argument("log")
    src = at.add_mutually_exclusive_group(required=True)
    src.add_argument("--session", help="a scripts/tuning_queue.py session dir")
    src.add_argument("--run", nargs="+", help="tuning run ids")
    at.add_argument("--side", choices=("left", "right"), help="the IMU's arm")
    at.add_argument("--units", choices=("auto", "ms2", "g"), default="auto")
    at.add_argument("--search", type=float, default=None, help="± seconds")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from almond_axol.tuning import external_imu as ei

    log = ei.load_log(Path(args.log).expanduser(), args.units)
    if args.cmd == "check":
        info = ei.describe(log)
        if info["rate_hz"] < 100:
            info["warning"] = "under 100 Hz: the 3-15 Hz shake will be unreliable"
        print(json.dumps(info, indent=1))
        return
    ok = bad = 0
    for rid, side in _runs(args):
        try:
            out = ei.attach(rid, log, side, args.search, source=str(args.log))
        except (ValueError, FileNotFoundError, KeyError) as exc:
            bad += 1
            print(f"{rid}: not attached — {exc}")
            continue
        ok += 1
        m = out["metrics"]
        print(
            f"{rid}: {out['side']} offset {out['offset_s']:+.3f} s corr "
            f"{out['corr']:.2f} | 1-3 Hz {m['low_mm']:.2f} mm, 3-15 Hz "
            f"{m['high_mm']:.2f} mm, accel {m['acc_rms']:.2f} m/s²"
        )
    print(f"attached {ok}, failed {bad}")
    if bad and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
