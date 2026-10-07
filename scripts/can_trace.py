"""Passive CAN recorder and decoder for the arm buses — watch any run without touching it.

``record`` opens the arm interfaces read-only (SocketCAN delivers every frame
to every listener; this never transmits), so it can run alongside
``tune.factory``, ``tune.friction``, teleop or anything else that owns the
bus. ``decode`` turns a recording into per-motor CSVs — what each joint was
commanded and what it reported — in the joint frame, and flags the windows
where a joint oscillates.

Usage:
    # terminal 1: start recording, then run the tool in terminal 2; Ctrl-C to stop
    uv run python scripts/can_trace.py record -o ~/trace.log
    # afterwards
    uv run python scripts/can_trace.py decode ~/trace.log --out ~/trace_csv

Decoded (MyActuator, ids 1-5): MIT commands (0x400+id: p_des, v_des, kp, kd,
t_ff) and feedback (0x500+id: position, velocity, torque); 0xA4 position
commands (0x140+id: target, speed cap) and their replies (0x240+id:
temperature, current, speed, single-turn angle). Damiao wrists (ids 6-7):
MIT feedback (0x10+id). MIT ranges depend on firmware: ``--v44`` lists the
motor ids on V4.4 firmware (the X8-P20 shoulders' 2026042402; default
"1,2"), every other MyActuator id uses the legacy ranges.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

_ARMS = {"left": "can_alm_axol_l", "right": "can_alm_axol_r"}
_JOINTS = [
    "shoulder_1",
    "shoulder_2",
    "shoulder_3",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]
_V_MAX, _KP_MAX, _KD_MAX = 45.0, 500.0, 5.0
_P_LEGACY, _T_LEGACY = 12.5, 24.0
_P_V44 = 12.566
_T_V44 = {1: 129.0, 2: 129.0, 3: 60.0, 4: 60.0, 5: 60.0}  # X8 shoulders, X6 others


def _u2f(x: int, lo: float, hi: float, bits: int) -> float:
    return x * (hi - lo) / ((1 << bits) - 1) + lo


# --------------------------------------------------------------------- record


def record(args: argparse.Namespace) -> None:
    import can

    ifaces = args.iface or list(_ARMS.values())
    out = open(args.output, "w", buffering=1)
    out.write("# time_s,iface,arb_id,data_hex\n")
    lock = threading.Lock()
    stop = threading.Event()
    counts: dict[str, int] = defaultdict(int)

    def listen(iface: str) -> None:
        # receive_own_messages off; nothing here ever calls send().
        bus = can.Bus(interface="socketcan", channel=iface)
        try:
            while not stop.is_set():
                msg = bus.recv(timeout=0.2)
                if msg is None:
                    continue
                line = f"{msg.timestamp:.6f},{iface},{msg.arbitration_id:x},{msg.data.hex()}\n"
                with lock:
                    out.write(line)
                    counts[iface] += 1
        finally:
            bus.shutdown()

    threads = [threading.Thread(target=listen, args=(i,), daemon=True) for i in ifaces]
    for t in threads:
        t.start()
    print(f"recording {', '.join(ifaces)} -> {args.output} (read-only; Ctrl-C to stop)")
    try:
        while True:
            time.sleep(5)
            with lock:
                print("  frames:", dict(counts), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=1)
        out.close()
        print(f"stopped: {dict(counts)} frames in {args.output}")


# --------------------------------------------------------------------- decode


def _joint_offset(side: str, motor_id: int) -> float | None:
    """motor→joint offset for the fixed-stop joints (zeroed at the nearer stop)."""
    try:
        from almond_axol.constants import Joint
        from almond_axol.robot.axol import EITHER_STOP_JOINTS, closer_end_stop
    except Exception:  # noqa: BLE001 - decode still works in the motor frame
        return None
    joint = list(Joint)[motor_id - 1]
    if joint in EITHER_STOP_JOINTS or motor_id > 7:
        return None  # wrists: detected at bring-up from the stop side; unknown here
    offset, _ = closer_end_stop(joint, side == "left")
    return offset


def decode(args: argparse.Namespace) -> None:
    v44 = {int(x) for x in args.v44.split(",") if x}
    side_of = {v: k for k, v in _ARMS.items()}
    rows: dict[tuple[str, int], list[dict]] = defaultdict(list)
    t0 = None
    with open(args.log) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            ts, iface, arb, data = line.strip().split(",")
            t, arb_id, d = float(ts), int(arb, 16), bytes.fromhex(data)
            t0 = t if t0 is None else t0
            side = side_of.get(iface, iface)
            rec = None
            mid = arb_id & 0xFF
            if 0x401 <= arb_id <= 0x405 and len(d) == 8:  # MIT command
                p_max = _P_V44 if mid in v44 else _P_LEGACY
                t_max = _T_V44[mid] if mid in v44 else _T_LEGACY
                p = _u2f((d[0] << 8) | d[1], -p_max, p_max, 16)
                v = _u2f((d[2] << 4) | (d[3] >> 4), -_V_MAX, _V_MAX, 12)
                kp = _u2f(((d[3] & 0xF) << 8) | d[4], 0, _KP_MAX, 12)
                kd = _u2f((d[5] << 4) | (d[6] >> 4), 0, _KD_MAX, 12)
                tff = _u2f(((d[6] & 0xF) << 8) | d[7], -t_max, t_max, 12)
                rec = dict(kind="mit_cmd", p=p, v=v, kp=kp, kd=kd, t=tff)
            elif 0x501 <= arb_id <= 0x505 and len(d) >= 6:  # MIT feedback
                p_max = _P_V44 if mid in v44 else _P_LEGACY
                t_max = _T_V44[mid] if mid in v44 else _T_LEGACY
                p = _u2f((d[1] << 8) | d[2], -p_max, p_max, 16)
                v = _u2f((d[3] << 4) | (d[4] >> 4), -_V_MAX, _V_MAX, 12)
                tq = _u2f(((d[4] & 0xF) << 8) | d[5], -t_max, t_max, 12)
                rec = dict(kind="mit_fb", p=p, v=v, t=tq)
            elif (
                0x141 <= arb_id <= 0x145 and len(d) == 8 and d[0] == 0xA4
            ):  # 0xA4 command
                speed = int.from_bytes(d[2:4], "little")
                target = int.from_bytes(d[4:8], "little", signed=True)
                rec = dict(
                    kind="a4_cmd", p=math.radians(target / 100.0), v=math.radians(speed)
                )
            elif 0x241 <= arb_id <= 0x245 and len(d) == 8 and d[0] in (0xA4, 0x9C):
                iq = int.from_bytes(d[2:4], "little", signed=True) * 0.01
                spd = int.from_bytes(d[4:6], "little", signed=True)
                ang = int.from_bytes(d[6:8], "little", signed=True)
                rec = dict(
                    kind="a4_fb", v=math.radians(spd), iq=iq, angle_deg=ang, temp=d[1]
                )
            elif 0x16 <= arb_id <= 0x17 and len(d) == 8:  # Damiao wrist feedback
                mid = arb_id - 0x10
                rec = dict(
                    kind="dm_fb",
                    p=_u2f((d[1] << 8) | d[2], -12.5, 12.5, 16),
                    v=_u2f((d[3] << 4) | (d[4] >> 4), -30.0, 30.0, 12),
                    t=_u2f(((d[4] & 0xF) << 8) | d[5], -10.0, 10.0, 12),
                )
            if rec is None:
                continue
            rec["t_s"] = t - t0
            rows[(side, mid)].append(rec)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{sum(len(v) for v in rows.values())} decoded frames, {len(rows)} motors")
    for (side, mid), recs in sorted(rows.items()):
        name = f"{side}_{_JOINTS[mid - 1] if 1 <= mid <= 7 else mid}"
        off = _joint_offset(side, mid)
        keys = ["t_s", "kind", "p", "v", "kp", "kd", "t", "iq", "angle_deg", "temp"]
        path = out / f"{name}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys + ["joint_deg"])
            for r in recs:
                jd = math.degrees(r["p"] + off) if off is not None and "p" in r else ""
                w.writerow([r.get(k, "") for k in keys] + [jd])
        fb = [r for r in recs if r["kind"] in ("mit_fb",)]
        cmds = [r for r in recs if r["kind"] in ("mit_cmd", "a4_cmd")]
        line = f"  {name:<18} {len(recs):6d} frames ({len(cmds)} cmd, {len(fb)} MIT fb) -> {path}"
        print(line)
        _flag_oscillation(name, fb, off)


def _flag_oscillation(name: str, fb: list[dict], off: float | None) -> None:
    """Print 1 s windows where the measured velocity swings hard (back-and-forth)."""
    if len(fb) < 50:
        return
    t = [r["t_s"] for r in fb]
    v = [r["v"] for r in fb]
    i0 = 0
    flagged = []
    for i in range(len(t)):
        while t[i] - t[i0] > 1.0:
            i0 += 1
        seg = v[i0 : i + 1]
        if len(seg) < 20:
            continue
        mean = sum(seg) / len(seg)
        rms = math.sqrt(sum((x - mean) ** 2 for x in seg) / len(seg))
        signs = sum(1 for a, b in zip(seg, seg[1:]) if (a - mean) * (b - mean) < 0)
        if rms > math.radians(10.0) and signs >= 4:
            flagged.append((t[i], math.degrees(rms), signs))
    if flagged:
        first, last = flagged[0][0], flagged[-1][0]
        worst = max(flagged, key=lambda x: x[1])
        pos = [r["p"] for r in fb if first - 1 <= r["t_s"] <= last]
        where = (
            f", joint {math.degrees(min(pos) + off):+.1f}..{math.degrees(max(pos) + off):+.1f}°"
            if off is not None and pos
            else ""
        )
        print(
            f"     ! oscillation {first:.1f}-{last:.1f} s: velocity swing up to "
            f"{worst[1]:.0f}°/s RMS{where}"
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="record the arm buses (read-only)")
    r.add_argument("-o", "--output", required=True, type=Path)
    r.add_argument(
        "--iface", action="append", help="interface (default: both arm buses)"
    )
    d = sub.add_parser("decode", help="decode a recording into per-motor CSVs")
    d.add_argument("log", type=Path)
    d.add_argument("--out", type=Path, default=None)
    d.add_argument(
        "--v44", default="1,2", help="MyActuator ids on V4.4 firmware (default 1,2)"
    )
    args = p.parse_args()
    if args.cmd == "record":
        record(args)
    else:
        args.out = args.out or args.log.with_suffix("")
        decode(args)


if __name__ == "__main__":
    sys.exit(main())
