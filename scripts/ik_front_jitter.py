"""Reproduce the elbow jitter when the gripper works in front of the base.

Runs the production IK solve loop (same per-tick re-seeding as the teleop
worker, no hardware) while sweeping the left gripper from its rest pose to a
target in front of the torso, then holding it there. For every tick it
records the solved joints and every collision pair's clearance vs its
activation margin, then reports:

  - per-joint tick-to-tick motion during the hold (a static target should
    converge to a static q; anything else is solver limit-cycling),
  - which collision pairs are inside their activation shell and how often
    they cross it (hinge chatter),
  - the same run with self_collision_weight=0 as an A/B control.

    uv run python scripts/ik_front_jitter.py
    uv run python scripts/ik_front_jitter.py --target 0.30 0.10 0.45
"""

from __future__ import annotations

import argparse
import dataclasses

import jax.numpy as jnp
import numpy as np

from almond_axol.constants import ARM_JOINTS
from almond_axol.kinematics.config import KinematicsConfig
from almond_axol.kinematics.solver import KinematicsSolver

MOVE_TICKS = 300
HOLD_TICKS = 300


def run_case(
    solver: KinematicsSolver, target: np.ndarray, label: str
) -> dict[str, np.ndarray]:
    q = np.zeros(solver.num_joints, dtype=np.float32)
    (lp0, lr0), (rp0, rr0) = solver.fk(q)
    robot, rc = solver.robot, solver.robot_coll
    margins = np.asarray(solver._collision_starts)
    ai, aj = np.asarray(rc.active_idx_i), np.asarray(rc.active_idx_j)
    pair_names = [f"{rc.link_names[i]}<->{rc.link_names[j]}" for i, j in zip(ai, aj)]

    qs: list[np.ndarray] = []
    dists: list[np.ndarray] = []
    for t in range(MOVE_TICKS + HOLD_TICKS):
        a = min(t / MOVE_TICKS, 1.0)
        a = a * a * (3.0 - 2.0 * a)  # smoothstep, like an operator's reach
        pos = (1.0 - a) * lp0 + a * target
        q = solver.ik(q, left_pose=(pos, lr0), right_pose=(rp0, rr0))
        qs.append(q.copy())
        d = np.asarray(
            rc.compute_self_collision_distance(
                robot, jnp.asarray(solver.to_pyroki_order(q))
            )
        )
        dists.append(d)

    qs_a = np.array(qs)
    dists_a = np.array(dists)

    print(f"\n=== {label}  target={target} ===")
    # Motion-phase smoothness: the commanded target is C1 (smoothstep), so the
    # solver output's second difference is the jitter the arm has to follow.
    move = qs_a[:MOVE_TICKS]
    accel = np.diff(move, n=2, axis=0)
    print("  move-phase second difference (left arm, mdeg/tick^2):")
    for i, j in enumerate(ARM_JOINTS):
        rms = float(np.degrees(np.sqrt(np.mean(accel[:, i] ** 2))) * 1e3)
        peak = float(np.degrees(np.abs(accel[:, i]).max()) * 1e3)
        peak_t = int(np.abs(accel[:, i]).argmax())
        print(f"    {j.value:<11} rms={rms:7.2f}  peak={peak:8.2f}  @tick {peak_t}")

    hold = qs_a[MOVE_TICKS:]
    deltas = np.abs(np.diff(hold, axis=0))
    print("  hold-phase per-joint tick deltas (left arm, mdeg):")
    for i, j in enumerate(ARM_JOINTS):
        rms = float(np.sqrt(np.mean(deltas[:, i] ** 2)))
        peak = float(deltas[:, i].max())
        span = float(hold[:, i].max() - hold[:, i].min())
        flag = "  <-- jitter" if peak > 2e-4 else ""
        print(
            f"    {j.value:<11} rms={np.degrees(rms) * 1e3:7.2f}  "
            f"peak={np.degrees(peak) * 1e3:7.2f}  "
            f"span={np.degrees(span) * 1e3:8.2f}{flag}"
        )

    # Where were the shell crossings, and what did the elbow joint do there?
    margins_row = margins[None, :]
    inside_t = dists_a[:MOVE_TICKS] < margins_row
    elbow_i = list(ARM_JOINTS).index(next(j for j in ARM_JOINTS if j.value == "elbow"))
    for k, name in enumerate(pair_names):
        flips = np.where(np.diff(inside_t[:, k].astype(int)) != 0)[0]
        if len(flips) == 0:
            continue
        acc_near = [
            float(np.degrees(np.abs(accel[max(f - 3, 0) : f + 4, elbow_i]).max()) * 1e3)
            for f in flips
            if f < len(accel)
        ]
        print(
            f"  {name}: move-phase shell crossings at ticks {list(flips)}; "
            f"elbow |accel| near each: {[f'{a:.1f}' for a in acc_near]} mdeg/tick^2"
        )

    print("  collision pairs active during run (inside activation margin):")
    inside = dists_a < margins[None, :]
    for k, name in enumerate(pair_names):
        n_in = int(inside[:, k].sum())
        if n_in == 0:
            continue
        crossings = int(np.sum(np.diff(inside[:, k].astype(int)) != 0))
        d_hold = dists_a[MOVE_TICKS:, k]
        print(
            f"    {name:<28} inside {n_in}/{len(dists_a)} ticks, "
            f"{crossings} shell crossings, hold clearance "
            f"{1e3 * d_hold.min():6.1f}..{1e3 * d_hold.max():6.1f} mm "
            f"(margin {1e3 * margins[k]:.1f})"
        )
    return {"q": qs_a, "d": dists_a}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--target",
        type=float,
        nargs=3,
        default=[0.30, 0.10, 0.45],
        help="left-gripper hold position, world frame (m)",
    )
    args = ap.parse_args()
    target = np.array(args.target, dtype=np.float32)

    solver = KinematicsSolver()
    run_case(solver, target, "production config")

    no_coll = KinematicsSolver(
        dataclasses.replace(KinematicsConfig(), self_collision_weight=0.0)
    )
    run_case(no_coll, target, "A/B: self_collision_weight=0")


if __name__ == "__main__":
    main()
