"""Write the per-joint creep motions the slow-motion tuning is judged on.

A creep moves one joint at constant 6 and 3 deg/s between two angles, every
other joint held at ``slow_osc``'s start pose — so the joint's own slow-speed
ripple (friction, Stribeck, cogging) shows without the rest of the arm
moving. The angles are the ones run on the jelly robot (right arm; the left
arm is the mirror, every joint negated). ``wrist_2`` stays in its outboard
half: with the elbow this straight, the inboard half meets the base.

    uv run python scripts/creep_motions.py --all            # every joint, both arms
                                                            # (+ slow_osc_left.npz)
    uv run python scripts/creep_motions.py --joint shoulder_3 --arm left
    uv run python scripts/creep_motions.py --list

Files land in ``~/.almond/motions`` as ``<key>_creep.npz`` (right) and
``<key>_creep_left.npz``; run one with
``axol tune.motion --arms right --motion ~/.almond/motions/s3_creep.npz``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

RATE = 240.0
OUT_DIR = Path.home() / ".almond" / "motions"
KEYS = {
    "shoulder_1": "s1",
    "shoulder_2": "s2",
    "shoulder_3": "s3",
    "elbow": "el",
    "wrist_1": "w1",
    "wrist_2": "w2",
    "wrist_3": "w3",
}
#: Right-arm creep range (near, far) in joint degrees; ``None`` means "the
#: start pose minus 1 / minus 12", for a joint whose start is off zero.
RIGHT_RANGE: dict[str, tuple[float, float] | None] = {
    "shoulder_1": (-20.0, 40.0),
    "shoulder_2": (2.0, 22.0),
    "shoulder_3": (-3.0, -35.0),
    "elbow": (-30.0, -90.0),
    "wrist_1": (-1.0, -12.0),
    "wrist_2": (-1.0, -12.0),  # outboard (negative on the right arm)
    "wrist_3": None,
}
_JOINTS = list(KEYS)
_HOLD_S = 0.5
_EDGE_S = 1.0
_MARGIN_DEG = 3.0


def start_pose() -> np.ndarray:
    """``slow_osc``'s first sample: both arms, 14 joints (rad)."""
    from almond_axol.tuning.motion import MOTIONS_DIR

    return np.load(MOTIONS_DIR / "slow_osc.npz")["q"][0].astype(float)


def leg(q0: float, q1: float, v_dps: float, ta: float = 0.3) -> np.ndarray:
    """Samples from ``q0`` to ``q1``: raised-cosine ramp to ``v``, cruise, ramp down."""
    d = q1 - q0
    if abs(d) < 1e-9:
        return np.array([q1])
    v = math.radians(v_dps)
    ta = min(ta, abs(d) / v)
    t_cruise = max(0.0, (abs(d) - v * ta) / v)
    n = max(2, int(round((2 * ta + t_cruise) * RATE)))
    t = np.arange(1, n + 1) / RATE
    vel = np.full(n, v)
    up = t < ta
    vel[up] = v * 0.5 * (1 - np.cos(np.pi * t[up] / ta))
    down = t > ta + t_cruise
    vel[down] = v * 0.5 * (1 + np.cos(np.pi * (t[down] - ta - t_cruise) / ta))
    pos = np.cumsum(vel) / RATE
    pos *= abs(d) / pos[-1]
    return q0 + math.copysign(1.0, d) * pos


def legs_for(joint: str, arm: str, pose: np.ndarray) -> list[list[float]]:
    """``[[angle_deg, speed_dps], ...]`` from the start pose and back."""
    col = _JOINTS.index(joint) + (0 if arm == "left" else 7)
    start = math.degrees(pose[col])
    rng = RIGHT_RANGE[joint]
    if rng is None:
        # Right-arm convention: from the start pose toward zero.
        right_start = math.degrees(pose[7 + _JOINTS.index(joint)])
        rng = (right_start - 1.0, right_start - 12.0)
    near, far = rng
    if arm == "left":
        near, far = -near, -far
    near, far = round(near, 3), round(far, 3)
    return [[near, 6], [far, 3], [near, 3], [far, 6], [near, 6], [round(start, 3), 6]]


def build(
    joint: str, arm: str, pose: np.ndarray | None = None
) -> tuple[np.ndarray, dict]:
    """The creep's ``q`` (N x 14, rad) and its meta, checked against the limits."""
    from almond_axol.motor import Joint
    from almond_axol.robot.axol import arm_limits

    pose = start_pose() if pose is None else pose
    is_left = arm == "left"
    col = _JOINTS.index(joint) + (0 if is_left else 7)
    legs = legs_for(joint, arm, pose)
    lo, hi = (math.degrees(x) for x in arm_limits(Joint(joint), is_left))
    for angle, _ in legs:
        if not lo + _MARGIN_DEG <= angle <= hi - _MARGIN_DEG:
            raise ValueError(
                f"{arm} {joint}: {angle:+.1f}° is within {_MARGIN_DEG}° of the "
                f"limits [{lo:+.0f}, {hi:+.0f}]°"
            )
    if joint == "wrist_2":
        outboard = 1.0 if is_left else -1.0
        if any(a * outboard < 0 for a, _ in legs[:-1]):
            raise ValueError(f"{arm} wrist_2 creep leaves its outboard half")
    cur = pose[col]
    parts = [np.full(int(_EDGE_S * RATE), cur)]
    for angle, speed in legs:
        target = math.radians(angle)
        parts.append(leg(cur, target, speed))
        parts.append(np.full(int(_HOLD_S * RATE), target))
        cur = target
    parts.append(np.full(int(_EDGE_S * RATE), cur))
    x = np.concatenate(parts)
    q = np.tile(pose, (len(x), 1))
    q[:, col] = x
    meta = {
        "source": f"synthetic: {arm} {joint} creep at 3 and 6 deg/s, every other "
        "joint at slow_osc's start pose (scripts/creep_motions.py)",
        "legs_deg_dps": legs,
    }
    return q, meta


def write(joint: str, arm: str, out_dir: Path = OUT_DIR) -> Path:
    q, meta = build(joint, arm)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{KEYS[joint]}_creep" + ("_left" if arm == "left" else "")
    path = out_dir / f"{name}.npz"
    np.savez(
        path,
        q=q.astype(np.float32),
        rate=np.float64(RATE),
        meta=np.array(json.dumps(meta)),
    )
    return path


def write_slow_osc_left(out_dir: Path = OUT_DIR) -> Path:
    """``slow_osc`` mirrored onto the left arm (every joint negated, the arms
    swapped) — the packaged one drives the right arm."""
    from almond_axol.tuning.motion import MOTIONS_DIR

    z = np.load(MOTIONS_DIR / "slow_osc.npz")
    q = z["q"].astype(float)
    ql = -np.concatenate([q[:, 7:], q[:, :7]], axis=1)
    meta = {"source": "slow_osc mirrored to the left arm (scripts/creep_motions.py)"}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "slow_osc_left.npz"
    np.savez(
        path, q=ql.astype(np.float32), rate=z["rate"], meta=np.array(json.dumps(meta))
    )
    return path


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--joint", choices=_JOINTS)
    p.add_argument("--arm", choices=("left", "right", "both"), default="both")
    p.add_argument("--all", action="store_true", help="every joint")
    p.add_argument("--list", action="store_true", help="print the planned legs only")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    args = p.parse_args()
    joints = _JOINTS if args.all or args.list else [args.joint]
    if joints == [None]:
        p.error("give --joint or --all")
    arms = ("right", "left") if args.arm == "both" else (args.arm,)
    pose = start_pose()
    if args.all and not args.list and "left" in arms:
        print(write_slow_osc_left(args.out))
    for j in joints:
        for a in arms:
            if args.list:
                print(f"{a:5s} {j:11s} {legs_for(j, a, pose)}")
                continue
            path = write(j, a, args.out)
            q = np.load(path)["q"]
            col = _JOINTS.index(j) + (0 if a == "left" else 7)
            print(
                f"{path}  {len(q) / RATE:.1f} s  {math.degrees(q[:, col].min()):+.1f}..{math.degrees(q[:, col].max()):+.1f}°"
            )


if __name__ == "__main__":
    main()
