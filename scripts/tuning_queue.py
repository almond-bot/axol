"""Guarded, interleaved A/B runs of ``tune.motion`` — and the verdict.

A tuning change is only judged against a baseline run in the same session,
interleaved round by round: slow-motion ripple drifts ±30% between sessions
and even between back-to-back runs. This tool plans that, runs it under
halt rules, and summarises it per variant.

    # 1. plan: 3 rounds of {baseline, two gain variants}, interleaved
    uv run python scripts/tuning_queue.py plan ~/tuning/s3 --arm right \\
        --motion s3_creep --rounds 3 \\
        --variant base \\
        --variant g0.4="--gain right.shoulder_3.stribeck_gain=0.4" \\
        --variant g0.8="--gain right.shoulder_3.stribeck_gain=0.8"
    # 2. run it (operator at the e-stop; Ctrl-C or `touch ~/tuning/s3/STOP` halts)
    uv run python scripts/tuning_queue.py run ~/tuning/s3
    # 3. read the result
    uv run python scripts/tuning_queue.py summary ~/tuning/s3

The verdict decides on the wrist IMU's 1-3 Hz sway (``imu.low_mm``) when
the runs have it, otherwise on the same sway measured by the encoders
(``enc.low_mm``, forward kinematics of the wrist flange) — for robots
without wrist cameras. ``plan --no-imu`` skips waiting for a camera that
isn't there. An IMU the operator mounted on the end-effector counts as the
IMU once attached (``scripts/ext_imu.py attach LOG --session DIR``).

The first variant is the baseline. A variant may also swap the whole
calibration file for its runs only (``--variant old@calib=PATH``); the file
in place is restored afterwards.

Halts (writes ``HALTED`` with the reason) on: an arm left holding after
repeated guard trips, a guard trip on a baseline run, two trips in a row,
a crash with no saved run, or a ``STOP`` file. Every run uses
``tune.motion``'s tracking guard; ``--no-guard`` and ``--arms`` in an item
are refused.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

_GUARD = ["--guard-dev-deg", "4", "--guard-osc-deg", "1"]
_CALIBRATION = Path.home() / ".almond" / "calibration.json"


# ------------------------------------------------------------------- plan


def parse_variant(spec: str) -> dict:
    """``NAME``, ``NAME=ARGS`` (extra tune.motion args) or ``NAME@calib=PATH``."""
    if "@calib=" in spec:
        name, path = spec.split("@calib=", 1)
        return {"name": name.strip(), "args": [], "calib": str(Path(path).expanduser())}
    name, _, args = spec.partition("=")
    return {"name": name.strip(), "args": shlex.split(args), "calib": None}


def plan_items(
    arm: str,
    motion: str,
    variants: list[dict],
    rounds: int,
    repeat: int = 2,
    prefix: str = "",
    no_imu: bool = False,
) -> list[dict]:
    """Interleave ``rounds`` rounds of every variant (baseline first)."""
    if arm not in ("left", "right"):
        raise ValueError(f"arm must be left or right, got {arm!r}")
    if not variants:
        raise ValueError("give at least one --variant (the first is the baseline)")
    names = [v["name"] for v in variants]
    if len(set(names)) != len(names):
        raise ValueError(f"variant names must be unique: {names}")
    tag = prefix or Path(motion).stem
    items = []
    for r in range(1, rounds + 1):
        for i, v in enumerate(variants):
            for bad in ("--arms", "--no-guard"):
                if bad in v["args"]:
                    raise ValueError(f"variant {v['name']}: {bad} is not allowed")
            item = {
                "label": f"{tag} {v['name']} r{r}",
                "variant": v["name"],
                "round": r,
                "arm": arm,
                "args": ["--motion", motion, "--repeat", str(repeat)]
                + _GUARD
                + (["--no-imu"] if no_imu else [])
                + v["args"],
                "baseline": i == 0,
            }
            if v["calib"]:
                item["calib"] = v["calib"]
            items.append(item)
    return items


def cmd_plan(args: argparse.Namespace) -> None:
    session = Path(args.session).expanduser()
    session.mkdir(parents=True, exist_ok=True)
    items = plan_items(
        args.arm,
        args.motion,
        [parse_variant(s) for s in args.variant],
        args.rounds,
        args.repeat,
        args.prefix,
        args.no_imu,
    )
    with (session / "queue.jsonl").open("a") as f:
        f.write("".join(json.dumps(i) + "\n" for i in items))
    print(f"queued {len(items)} items in {session / 'queue.jsonl'}")


# -------------------------------------------------------------------- run


def halt_reason(
    item: dict, rc: int, out: str, trips: list[str], ids: list[str], trips_in_row: int
) -> str | None:
    """Why the queue must stop after this item, or ``None`` to carry on."""
    if rc == 4 or "leaving the arm HOLDING" in out:
        return f"{item['label']}: arm left holding after repeated guard trips"
    if trips and item.get("baseline"):
        return f"{item['label']}: guard trip on a baseline run: {trips[0]}"
    if trips and trips_in_row >= 2:
        return f"{item['label']}: two guard trips in a row: {trips[0]}"
    if rc not in (0, 3) and not ids:
        tail = " | ".join(out.strip().splitlines()[-3:])
        return f"{item['label']}: rc={rc} with no saved run: {tail}"
    return None


def _done_labels(results: Path) -> set[str]:
    if not results.exists():
        return set()
    return {
        json.loads(line)["item"]["label"]
        for line in results.read_text().splitlines()
        if line.strip()
    }


def cmd_run(args: argparse.Namespace) -> None:
    from almond_axol.tuning.creep_score import score_run

    session = Path(args.session).expanduser()
    queue, results, log_path = (
        session / "queue.jsonl",
        session / "results.jsonl",
        session / "run.log",
    )
    halt_file, stop_file = session / "HALTED", session / "STOP"
    (session / "logs").mkdir(parents=True, exist_ok=True)
    lock = open(session / "run.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another runner holds this session")

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    def halt(reason: str) -> None:
        halt_file.write_text(f"{time.strftime('%F %T')} {reason}\n")
        log(f"HALT: {reason}")
        sys.exit(1)

    if halt_file.exists():
        sys.exit(
            f"session halted: {halt_file.read_text().strip()} — read it, then delete HALTED"
        )
    axol = shutil.which("axol") or "axol"
    trips_in_row = 0
    while True:
        if stop_file.exists():
            halt("STOP file")
        items = [json.loads(x) for x in queue.read_text().splitlines() if x.strip()]
        todo = [i for i in items if i["label"] not in _done_labels(results)]
        if not todo:
            log("queue done")
            return
        item = todo[0]
        if "--arms" in item["args"] or "--no-guard" in item["args"]:
            halt(f"refusing {item['label']}: --arms / --no-guard not allowed")
        cmd = [axol, "tune.motion", "--arms", item["arm"], "--label", item["label"]]
        cmd += item["args"]
        log(f"RUN {item['label']}: {' '.join(item['args'])}")
        t0 = time.time()
        saved = None
        if item.get("calib"):
            saved = session / "calibration.in-place.json"
            shutil.copy(_CALIBRATION, saved)
            shutil.copy(item["calib"], _CALIBRATION)
        try:
            p = subprocess.run(
                cmd, capture_output=True, text=True, timeout=item.get("timeout", 1800)
            )
            rc, out = p.returncode, p.stdout + p.stderr
        except KeyboardInterrupt:
            halt(f"{item['label']}: interrupted by the operator")
        finally:
            if saved is not None:
                shutil.copy(saved, _CALIBRATION)
        (
            session / "logs" / (re.sub(r"[^\w.-]", "_", item["label"]) + ".log")
        ).write_text(out)
        ids = re.findall(r"Saved tuning run (\S+)", out)
        trips = [x.strip() for x in out.splitlines() if "GUARD TRIP" in x]
        scores = []
        for rid in ids:
            try:
                scores.append(score_run(rid))
            except Exception as exc:  # noqa: BLE001 - recorded, not fatal
                scores.append({"id": rid, "error": repr(exc)})
        rec = {
            "item": item,
            "rc": rc,
            "secs": round(time.time() - t0),
            "ids": ids,
            "trips": trips,
            "scores": scores,
        }
        with results.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"DONE {item['label']} rc={rc} runs={len(ids)} trips={len(trips)}")
        trips_in_row = trips_in_row + 1 if trips else 0
        reason = halt_reason(item, rc, out, trips, ids, trips_in_row)
        if reason:
            halt(reason)
        time.sleep(3)


# ---------------------------------------------------------------- summary

_METRICS = (
    ("rip3_rms_mdeg", "ripple 3°/s mdeg RMS"),
    ("rip6_rms_mdeg", "ripple 6°/s mdeg RMS"),
    ("imu.low_mm", "IMU 1-3 Hz sway mm"),
    ("imu.vertical_mm", "IMU vertical mm"),
    ("imu.high_mm", "IMU 3-15 Hz shake mm"),
    ("imu.acc_rms", "IMU accel RMS m/s^2"),
    ("enc.low_mm", "encoder 1-3 Hz sway mm"),
    ("enc.high_mm", "encoder 3-15 Hz shake mm"),
)
#: A variant that raises one of these by more than this much has bought its
#: sway with buzz — the trade every stiffer or more damped setting made on
#: jelly — and is never called ``better``. The IMU's when the runs have it,
#: else the encoders' (which track the IMU's 3-15 Hz change at r = 0.8).
_BUZZ_COST = (("imu.high_mm", "3-15 Hz"), ("imu.acc_rms", "accel"))
_ENC_BUZZ_COST = (("enc.high_mm", "3-15 Hz (encoders)"),)
_BUZZ_LIMIT_PCT = 10.0
#: The smallest average gain called ``better``, per metric family. The
#: encoders can't see flex past the joints and credited some changes the
#: IMU didn't (dither, shoulder_2 alone: −14% vs −4…−2%); at 15% none of
#: jelly's 31 slow_osc variants the IMU didn't confirm would pass.
_MIN_GAIN_PCT = {"imu": 5.0, "enc": 15.0}
_MIN_GAIN_RIPPLE_PCT = 10.0


def _get(score: dict, key: str):
    v = score
    for part in key.split("."):
        v = v.get(part) if isinstance(v, dict) else None
    return v if isinstance(v, (int, float)) else None


def summarize(records: list[dict]) -> dict:
    """Per variant and metric: per-round means, overall mean, change vs the
    baseline, and how many rounds the variant beat the baseline."""
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    order, baseline, trips = [], None, defaultdict(int)
    for r in records:
        it = r["item"]
        var = it.get("variant", it["label"])
        if var not in order:
            order.append(var)
        if it.get("baseline"):
            baseline = var
        trips[var] += len(r["trips"])
        for s in r["scores"]:
            for key, _ in _METRICS:
                val = _get(s, key)
                if val is not None:
                    per[var][key][it.get("round", 0)].append(val)
    out = {"baseline": baseline, "variants": {}}
    for var in order:
        entry = {"guard_trips": trips[var], "metrics": {}}
        for key, desc in _METRICS:
            rounds = {rd: statistics.mean(v) for rd, v in per[var][key].items()}
            if not rounds:
                continue
            m = {
                "desc": desc,
                "per_round": {str(k): round(v, 3) for k, v in sorted(rounds.items())},
                "mean": round(statistics.mean(rounds.values()), 3),
            }
            if baseline and var != baseline and per[baseline][key]:
                base = {rd: statistics.mean(v) for rd, v in per[baseline][key].items()}
                common = sorted(set(rounds) & set(base))
                if common:
                    bm = statistics.mean(base[c] for c in common)
                    vm = statistics.mean(rounds[c] for c in common)
                    m["change_vs_baseline_pct"] = round(100 * (vm / bm - 1), 1)
                    m["rounds_better"] = (
                        f"{sum(rounds[c] < base[c] for c in common)}/{len(common)}"
                    )
            entry["metrics"][key] = m
        out["variants"][var] = entry
    return out


def _has(summary: dict, key: str) -> bool:
    return any(key in e["metrics"] for e in summary["variants"].values())


def deciding_metric(summary: dict, metric: str = "auto") -> str:
    """``auto``: the wrist IMU's sway when the runs have it, else the
    encoders' (a robot without wrist cameras)."""
    if metric != "auto":
        return metric
    return "imu.low_mm" if _has(summary, "imu.low_mm") else "enc.low_mm"


def verdict(
    summary: dict, metric: str = "auto", min_gain_pct: float | None = None
) -> dict:
    """A conservative call per variant against the baseline.

    ``better`` only when the variant beat the baseline in **every** round
    and by ``min_gain_pct`` on average (default 5% for the IMU, 15% for the
    encoder sway, 10% for joint ripple), the sway didn't rise, and the buzz
    cost (3–15 Hz shake, accel) didn't rise by more than 10%. ``trade`` when
    it won on ``metric`` but paid in buzz. ``worse`` when it lost every round
    by the margin. Anything else is ``inconclusive``: keep the default.

    The deciding metric is the 1–3 Hz sway an operator feels on
    ``slow_osc``: the wrist IMU's (``imu.low_mm``), or without a wrist
    camera the encoders' (``enc.low_mm``) — ``auto`` picks. On jelly every
    per-joint creep ripple win (−30…−57%) turned into at most −7% sway at
    the tool.
    """
    metric = deciding_metric(summary, metric)
    family = metric.split(".", 1)[0]
    if min_gain_pct is None:
        min_gain_pct = _MIN_GAIN_PCT.get(family, _MIN_GAIN_RIPPLE_PCT)
    has_imu = _has(summary, "imu.low_mm")
    sway_key = "imu.low_mm" if has_imu else "enc.low_mm"
    cost_keys = _BUZZ_COST if has_imu else _ENC_BUZZ_COST
    calls = {}
    for var, entry in summary["variants"].items():
        if var == summary["baseline"]:
            continue
        mets = entry["metrics"]
        m = mets.get(metric)
        if not m or "change_vs_baseline_pct" not in m:
            calls[var] = "no data" + (
                " (no IMU: is the wrist camera up, or your own attached with "
                "scripts/ext_imu.py? without either use the default)"
                if family == "imu"
                else ""
            )
            continue
        won, total = (int(x) for x in m["rounds_better"].split("/"))
        chg = m["change_vs_baseline_pct"]
        sway = mets.get(sway_key, {}).get("change_vs_baseline_pct")
        costs = [
            f"{name} {mets[k]['change_vs_baseline_pct']:+.0f}%"
            for k, name in cost_keys
            if mets.get(k, {}).get("change_vs_baseline_pct", 0.0) > _BUZZ_LIMIT_PCT
        ]
        if entry["guard_trips"]:
            calls[var] = "rejected: guard trips"
        elif chg <= -min_gain_pct and won == total:
            if costs:
                calls[var] = f"trade ({chg:+.0f}% but {', '.join(costs)}) — don't adopt"
            elif sway is not None and sway > 0 and metric != sway_key:
                calls[var] = (
                    f"inconclusive ({chg:+.0f}% on {metric}, but "
                    f"{'IMU' if has_imu else 'encoder'} sway {sway:+.0f}%)"
                )
            else:
                calls[var] = f"better ({chg:+.0f}%, {won}/{total} rounds)"
                if family == "enc":
                    calls[var] += (
                        " — encoders only (no wrist IMU): confirm with the "
                        "person's feel and a bus recording"
                    )
        elif chg >= min_gain_pct and won == 0:
            calls[var] = f"worse ({chg:+.0f}%)"
        else:
            calls[var] = f"inconclusive ({chg:+.0f}%, {won}/{total} rounds)"
        if sway is None and family not in ("imu", "enc"):
            calls[var] += " — no sway data: confirm on slow_osc"
    return calls


def merge_external_imu(records: list[dict], runs_dir: Path | None = None) -> int:
    """Give runs without a wrist-camera IMU the metrics of an IMU the operator
    mounted on the end-effector (``scripts/ext_imu.py attach``). Returns how
    many runs got one."""
    from almond_axol.tuning.external_imu import load_attached
    from almond_axol.tuning.runs import TUNING_RUNS_DIR

    n = 0
    for rec in records:
        for score in rec.get("scores", []):
            if "imu" in score or not score.get("id"):
                continue
            ext = load_attached(score["id"], runs_dir or TUNING_RUNS_DIR)
            if ext and ext.get("side") == rec["item"].get("arm"):
                score["imu"] = {**ext["metrics"], "source": "external"}
                n += 1
    return n


def cmd_summary(args: argparse.Namespace) -> None:
    path = Path(args.session).expanduser() / "results.jsonl"
    records = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    external = merge_external_imu(records)
    s = summarize(records)
    s["decided_on"] = deciding_metric(s, args.metric)
    if external:
        s["external_imu_runs"] = external
    s["verdict"] = verdict(s, args.metric, args.min_gain)
    print(json.dumps(s, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan", help="queue interleaved rounds of variants")
    pl.add_argument("session")
    pl.add_argument("--arm", required=True, choices=("left", "right"))
    pl.add_argument("--motion", required=True, help="motion name or .npz path")
    pl.add_argument(
        "--variant",
        action="append",
        required=True,
        help="NAME, NAME=EXTRA_TUNE_MOTION_ARGS or NAME@calib=PATH; first = baseline",
    )
    pl.add_argument("--rounds", type=int, default=3)
    pl.add_argument("--repeat", type=int, default=2, help="passes per run (default 2)")
    pl.add_argument(
        "--prefix", default="", help="label prefix (default: the motion name)"
    )
    pl.add_argument(
        "--no-imu",
        action="store_true",
        help="no wrist cameras on this robot: don't wait for one (the verdict "
        "then decides on the encoder sway)",
    )
    pl.set_defaults(func=cmd_plan)
    rn = sub.add_parser("run", help="run the queue under the halt rules")
    rn.add_argument("session")
    rn.set_defaults(func=cmd_run)
    sm = sub.add_parser("summary", help="per-variant results and a verdict (JSON)")
    sm.add_argument("session")
    sm.add_argument(
        "--metric",
        default="auto",
        help="deciding metric (default auto: imu.low_mm, the wrist IMU's 1-3 Hz "
        "sway on slow_osc, or enc.low_mm, the encoders', without a wrist camera; "
        "rip3_rms_mdeg for creep screening)",
    )
    sm.add_argument(
        "--min-gain",
        type=float,
        default=None,
        help="percent (default 5 IMU / 15 encoder sway / 10 ripple)",
    )
    sm.set_defaults(func=cmd_summary)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
