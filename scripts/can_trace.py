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
    # or the per-joint oscillation verdict, as JSON
    uv run python scripts/can_trace.py summary ~/trace.log

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

import numpy as np

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


def _dm_mit(d: bytes) -> dict:
    return dict(
        p=_u2f((d[0] << 8) | d[1], -12.5, 12.5, 16),
        v=_u2f((d[2] << 4) | (d[3] >> 4), -30.0, 30.0, 12),
        kp=_u2f(((d[3] & 0xF) << 8) | d[4], 0, 500.0, 12),
        kd=_u2f((d[5] << 4) | (d[6] >> 4), 0, 5.0, 12),
        t=_u2f(((d[6] & 0xF) << 8) | d[7], -10.0, 10.0, 12),
    )


def parse_log(path: Path, v44: set[int]) -> dict[tuple[str, int], list[dict]]:
    """Every decodable frame of a recording, per ``(side, motor id)``."""
    side_of = {v: k for k, v in _ARMS.items()}
    rows: dict[tuple[str, int], list[dict]] = defaultdict(list)
    t0 = None
    with open(path) as f:
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
                mid = arb_id - 0x140
                speed = int.from_bytes(d[2:4], "little")
                target = int.from_bytes(d[4:8], "little", signed=True)
                rec = dict(
                    kind="a4_cmd", p=math.radians(target / 100.0), v=math.radians(speed)
                )
            elif 0x241 <= arb_id <= 0x245 and len(d) == 8 and d[0] in (0xA4, 0x9C):
                mid = arb_id - 0x240
                iq = int.from_bytes(d[2:4], "little", signed=True) * 0.01
                spd = int.from_bytes(d[4:6], "little", signed=True)
                ang = int.from_bytes(d[6:8], "little", signed=True)
                rec = dict(
                    kind="a4_fb", v=math.radians(spd), iq=iq, angle_deg=ang, temp=d[1]
                )
            elif arb_id in (0x06, 0x07) and len(d) == 8 and d[:7] != b"\xff" * 7:
                rec = dict(kind="dm_cmd", **_dm_mit(d))  # Damiao MIT command
            elif 0x16 <= arb_id <= 0x17 and len(d) == 8:  # Damiao wrist feedback
                mid = arb_id - 0x10
                p = _u2f((d[1] << 8) | d[2], -12.5, 12.5, 16)
                if abs(p) < 12.4:  # register replies share the id; not a position
                    rec = dict(
                        kind="dm_fb",
                        p=p,
                        v=_u2f((d[3] << 4) | (d[4] >> 4), -30.0, 30.0, 12),
                        t=_u2f(((d[4] & 0xF) << 8) | d[5], -10.0, 10.0, 12),
                    )
            if rec is None:
                continue
            rec["t_s"] = t - t0
            rows[(side, mid)].append(rec)
    return rows


def decode(args: argparse.Namespace) -> None:
    v44 = {int(x) for x in args.v44.split(",") if x}
    rows = parse_log(args.log, v44)
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
        fb = [r for r in recs if r["kind"] in ("mit_fb", "dm_fb")]
        cmds = [r for r in recs if r["kind"] in ("mit_cmd", "a4_cmd", "dm_cmd")]
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


#: Band the summary judges oscillation in (Hz): above deliberate operator
#: motion, below what the 240 Hz command stream can carry cleanly.
_OSC_BAND = (1.5, 15.0)
#: Band RMS (mdeg) above which a joint's command or tracking error counts as
#: oscillating; ``_STRONG_MDEG`` is visible to the eye at the hand.
_OSC_MDEG = 100.0
_STRONG_MDEG = 300.0
_GRID_HZ = 200.0


def _segments(t: np.ndarray, max_gap: float = 0.1, min_len: float = 2.0) -> list:
    """Index ranges of ``t`` without gaps longer than ``max_gap``."""
    cuts = np.flatnonzero(np.diff(t) > max_gap) + 1
    out = []
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(t)]):
        if t[b - 1] - t[a] >= min_len:
            out.append((a, b))
    return out


def joint_summary(recs: list[dict]) -> dict | None:
    """Oscillation metrics for one motor: is the *command* wiggling (input
    side: tracker, IK, a script) or the joint *ringing* around a smooth
    command (control side: gains, friction terms)?"""
    from scipy.signal import butter, sosfiltfilt

    cmd = [r for r in recs if r["kind"] in ("mit_cmd", "dm_cmd")]
    fb = [r for r in recs if r["kind"] in ("mit_fb", "dm_fb")]
    out: dict = {
        "a4_frames": sum(1 for r in recs if r["kind"] == "a4_cmd"),
        "commands": len(cmd),
        "feedback": len(fb),
    }
    if len(cmd) < 50 or len(fb) < 50:
        out["verdict"] = (
            "firmware position loop (0xA4)" if out["a4_frames"] else "too little data"
        )
        return out
    tc = np.array([r["t_s"] for r in cmd])
    pc = np.array([r["p"] for r in cmd])
    tf = np.array([r["t_s"] for r in fb])
    pf = np.array([r["p"] for r in fb])
    # Command in force at each feedback sample (zero-order hold).
    k = np.searchsorted(tc, tf, side="right") - 1
    ok = k >= 0
    tf, pf, pcmd = tf[ok], pf[ok], pc[k[ok]]
    sos = butter(2, _OSC_BAND, btype="band", fs=_GRID_HZ, output="sos")
    cmd_band, err_band, grids = [], [], []
    for a, b in _segments(tf):
        g = np.arange(tf[a], tf[b - 1], 1.0 / _GRID_HZ)
        if len(g) < 3 * _GRID_HZ:
            continue
        c = np.interp(g, tf[a:b], pcmd[a:b])
        e = np.interp(g, tf[a:b], pf[a:b]) - c
        cmd_band.append(sosfiltfilt(sos, c))
        err_band.append(sosfiltfilt(sos, e))
        grids.append(g)
    if not grids:
        out["verdict"] = "too little data"
        return out
    cb, eb, g = (np.concatenate(x) for x in (cmd_band, err_band, grids))

    def mdeg(x: np.ndarray) -> float:
        return round(1e3 * math.degrees(float(np.sqrt(np.mean(x**2)))), 1)

    def peak_hz(x: np.ndarray) -> float:
        n = len(x)
        spec = np.abs(np.fft.rfft(x * np.hanning(n)))
        f = np.fft.rfftfreq(n, 1.0 / _GRID_HZ)
        band = (f >= _OSC_BAND[0]) & (f <= _OSC_BAND[1])
        return round(float(f[band][np.argmax(spec[band])]), 2)

    # 2 s windows, 1 s apart: oscillation means repeated reversals; a single
    # jump of the command (a step, a Ctrl-C hold) crosses only once or twice.
    w, hop = int(2 * _GRID_HZ), int(_GRID_HZ)
    thr = math.radians(_OSC_MDEG / 1e3)

    def crossings(x: np.ndarray) -> int:
        level = 0.3 * float(np.sqrt(np.mean(x**2)))
        state, n = 0, 0
        for v in x:
            s_ = 1 if v > level else (-1 if v < -level else 0)
            if s_ and s_ != state:
                n += state != 0
                state = s_
        return n

    windows = []
    for i in range(0, max(1, len(eb) - w + 1), hop):
        c_, e_ = cb[i : i + w], eb[i : i + w]
        c_rms, e_rms = float(np.sqrt(np.mean(c_**2))), float(np.sqrt(np.mean(e_**2)))
        cmd_osc = c_rms >= thr and crossings(c_) >= 5
        windows.append(
            dict(
                t=float(g[i]) + 1.0,  # window centre
                c=c_rms,
                e=e_rms,
                cmd_osc=cmd_osc,
                step=c_rms >= thr and not cmd_osc,
                ring=e_rms >= thr and crossings(e_) >= 5 and not cmd_osc,
            )
        )
    steps = [x["t"] for x in windows if x["step"]]
    transient = [
        x for x in windows if x["ring"] and any(0 <= x["t"] - s_ <= 3 for s_ in steps)
    ]
    sustained = [x for x in windows if x["ring"] and x not in transient]
    jitter = [x for x in windows if x["cmd_osc"]]
    worst = max(windows, key=lambda x: x["e"])
    err = pf - pcmd
    out.update(
        cmd_band_mdeg=mdeg(cb),
        err_band_mdeg=mdeg(eb),
        cmd_peak_hz=peak_hz(cb),
        err_peak_hz=peak_hz(eb),
        worst_err_window_s=round(worst["t"], 1),
        worst_err_window_mdeg=round(1e3 * math.degrees(worst["e"]), 1),
        max_abs_err_deg=round(math.degrees(float(np.max(np.abs(err)))), 2),
        kp=sorted({round(r["kp"]) for r in cmd}),
        kd=sorted({round(r["kd"], 1) for r in cmd}),
        span_s=round(float(sum(x[-1] - x[0] for x in grids)), 1),
        windows_ringing=len(sustained),
        windows_command_jitter=len(jitter),
        command_jumps_s=[round(x, 1) for x in steps],
    )
    peak = max(max(x["c"] for x in windows), worst["e"])
    if out["a4_frames"]:
        verdict = (
            "firmware position loop (0xA4) in use — see the playbook's known issues"
        )
    elif len(sustained) >= 2:
        verdict = f"control ringing at ~{out['err_peak_hz']} Hz (joint oscillates around a smooth command)"
    elif len(jitter) >= 2:
        verdict = f"command jitter at ~{out['cmd_peak_hz']} Hz (the target itself oscillates: tracker / IK / input)"
    elif transient or steps:
        verdict = f"transient: command jump(s) at {out['command_jumps_s']} s" + (
            f", ringing ~{out['err_peak_hz']} Hz that decays" if transient else ""
        )
    else:
        verdict = "quiet"
    if verdict != "quiet" and math.degrees(peak) * 1e3 >= _STRONG_MDEG:
        verdict = "STRONG " + verdict
    out["verdict"] = verdict
    return out


def summary(args: argparse.Namespace) -> None:
    import json

    v44 = {int(x) for x in args.v44.split(",") if x}
    rows = parse_log(args.log, v44)
    joints = {}
    for (side, mid), recs in sorted(rows.items()):
        if not 1 <= mid <= 7:
            continue
        s = joint_summary(recs)
        if s is not None:
            joints[f"{side}.{_JOINTS[mid - 1]}"] = s
    flagged = {
        k: v["verdict"]
        for k, v in joints.items()
        if v.get("verdict") not in ("quiet", "too little data")
    }
    print(
        json.dumps(
            {
                "log": str(args.log),
                "band_hz": list(_OSC_BAND),
                "threshold_mdeg": _OSC_MDEG,
                "flagged": flagged,
                "joints": joints,
            },
            indent=1,
        )
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
    sm = sub.add_parser(
        "summary",
        help="per-joint oscillation verdict (JSON): command jitter vs control ringing",
    )
    sm.add_argument("log", type=Path)
    sm.add_argument(
        "--v44", default="1,2", help="MyActuator ids on V4.4 firmware (default 1,2)"
    )
    args = p.parse_args()
    if args.cmd == "record":
        record(args)
    elif args.cmd == "summary":
        summary(args)
    else:
        args.out = args.out or args.log.with_suffix("")
        decode(args)


if __name__ == "__main__":
    sys.exit(main())
