"""Range of motion test — simulation only.

Moves each joint sequentially through its full range of motion while all
other joints stay at home.  Both arms are driven simultaneously for
symmetric joints; asymmetric joints (SHOULDER_2, ELBOW) are exercised one
arm at a time.

Each waypoint is checked for self-collision via pyroki before execution.
If a waypoint would cause a collision it is skipped with a warning.

Joint sequence:
  1. SHOULDER_1  — both arms, max → min → home
  2. SHOULDER_2  — right (0=min, sweep to max), then left (0=max, sweep to min)
  3. SHOULDER_3  — both arms, max → min → home
  4. ELBOW       — both arms together, 120° bend → home
  5. WRIST_1     — both arms, max → min → home
  6. WRIST_2     — arms forward −45° first, then max → min → home, then arms return
  7. WRIST_3     — both arms, max → min → home
  8. GRIPPER     — both arms, open → closed → home

Run:
    python -m almond_axol.test.rom_test

Open http://localhost:8080 in a browser to view the 3-D simulation.
Press Enter in the terminal when you are ready to start motion.
"""

import asyncio
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import pyroki as pk
import yourdfpy

from ..kinematics.solver import _LEFT_JOINT_NAMES, _RIGHT_JOINT_NAMES
from ..robot.axol import (
    LIMITS,
    SHOULDER_2_LEFT_LIMITS,
    SHOULDER_2_RIGHT_LIMITS,
)
from ..robot.sim import Sim
from ..shared import URDF_PATH, Joint

_SPEED = 3.0  # rad/s — fast for simulation
_RATE_HZ = 100.0
_PAUSE = 0.5  # seconds to hold at each waypoint

_90_DEG = math.pi * 2 / 3  # elbow bend limit (~120°)
_FWD_45 = -math.pi / 4  # shoulder_1 forward pre-pose for wrist_2

_IDX: dict[Joint, int] = {j: i for i, j in enumerate(Joint)}
_N = len(list(Joint))  # 8


# ---------------------------------------------------------------------------
# Collision checker
# ---------------------------------------------------------------------------


class CollisionChecker:
    """Checks robot configurations for self-collision using pyroki.

    Args:
        margin: Safety margin in metres. A waypoint is rejected if any
                collision pair distance falls below ``-margin``.
    """

    def __init__(self, margin: float = 0.01) -> None:
        print("Loading collision model (pyroki) ...")
        urdf = yourdfpy.URDF.load(str(URDF_PATH), mesh_dir=str(URDF_PATH.parent))
        robot = pk.Robot.from_urdf(urdf)
        robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

        actuated = list(robot.joints.actuated_names)
        name_to_idx = {n: i for i, n in enumerate(actuated)}
        self._left_idx = [name_to_idx[n] for n in _LEFT_JOINT_NAMES]
        self._right_idx = [name_to_idx[n] for n in _RIGHT_JOINT_NAMES]
        self._n = robot.joints.num_actuated_joints
        self._margin = margin

        @jax.jit
        def _check(q: jax.Array) -> jax.Array:
            return robot_coll.compute_self_collision_distance(robot, q)

        # Trigger JIT compilation and record home-position baseline.
        # Connected links always overlap in the URDF mesh, so we compare
        # relative to home rather than against an absolute threshold.
        home_dists = np.asarray(_check(jnp.zeros(self._n, dtype=jnp.float32)))
        self._baseline = float(home_dists.min())
        self._check = _check
        print(f"Collision model ready. Home baseline: {self._baseline:.4f} m\n")

    def is_safe(self, q_l: np.ndarray, q_r: np.ndarray) -> tuple[bool, float]:
        """Return ``(safe, min_distance)``.

        Positive distance means shapes are separated; negative means
        penetration.  A configuration is considered safe when
        ``min_distance > -margin``.
        """
        q_full = np.zeros(self._n, dtype=np.float32)
        q_full[self._left_idx] = q_l[:7]  # ARM_JOINTS only, no gripper
        q_full[self._right_idx] = q_r[:7]
        distances = np.asarray(self._check(jnp.asarray(q_full)))
        min_dist = float(distances.min())
        # Safe if not significantly worse than the home-position baseline
        return min_dist > self._baseline - self._margin, min_dist


# ---------------------------------------------------------------------------
# Motion helpers
# ---------------------------------------------------------------------------


def _home() -> np.ndarray:
    return np.zeros(_N, dtype=np.float32)


async def _sweep(
    sim: Sim,
    checker: CollisionChecker,
    q_l: np.ndarray,
    q_r: np.ndarray,
    tgt_l: np.ndarray,
    tgt_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Check target for collision, then S-curve interpolate to it."""
    safe, min_dist = checker.is_safe(tgt_l, tgt_r)
    if not safe:
        print(
            f"    ⚠  collision detected (min dist {min_dist:.4f} m) — waypoint skipped"
        )
        return q_l, q_r

    dist = max(
        float(np.max(np.abs(tgt_l - q_l))),
        float(np.max(np.abs(tgt_r - q_r))),
    )
    dur = max(dist / _SPEED, 0.1)
    dt = 1.0 / _RATE_HZ
    t0 = time.monotonic()
    while True:
        t = time.monotonic() - t0
        a = min(t / dur, 1.0)
        s = a * a * (3.0 - 2.0 * a)
        await sim.motion_control(
            left=(q_l * (1 - s) + tgt_l * s).astype(np.float32),
            right=(q_r * (1 - s) + tgt_r * s).astype(np.float32),
        )
        if a >= 1.0:
            break
        await asyncio.sleep(dt)
    await asyncio.sleep(_PAUSE)
    return tgt_l.copy(), tgt_r.copy()


def _with_joint(q: np.ndarray, joint: Joint, val: float) -> np.ndarray:
    out = q.copy()
    out[_IDX[joint]] = val
    return out


async def _joint_rom(
    sim: Sim,
    checker: CollisionChecker,
    q_l: np.ndarray,
    q_r: np.ndarray,
    joint: Joint,
    arm: str,  # "both" | "left" | "right"
    waypoints: list[float],
    label: str,
    home_val: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep one joint through waypoints then return to home_val."""
    print(f"  {label}  {[round(w, 3) for w in waypoints]} → {round(home_val, 3)}")
    for val in waypoints:
        tgt_l = _with_joint(q_l, joint, val) if arm in ("both", "left") else q_l.copy()
        tgt_r = _with_joint(q_r, joint, val) if arm in ("both", "right") else q_r.copy()
        q_l, q_r = await _sweep(sim, checker, q_l, q_r, tgt_l, tgt_r)
    tgt_l = _with_joint(q_l, joint, home_val) if arm in ("both", "left") else q_l.copy()
    tgt_r = (
        _with_joint(q_r, joint, home_val) if arm in ("both", "right") else q_r.copy()
    )
    q_l, q_r = await _sweep(sim, checker, q_l, q_r, tgt_l, tgt_r)
    return q_l, q_r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _run() -> None:
    checker = CollisionChecker()

    sim = Sim()
    await sim.enable()
    try:
        print("=== ROM TEST — simulation ===")
        print("Viser server running at  http://localhost:8080")
        print("Open that URL in a browser, then come back here.\n")
        await asyncio.to_thread(input, "Press Enter to start the ROM sweep ...")

        q_l = _home()
        q_r = _home()

        await sim.motion_control(left=q_l, right=q_r)
        await asyncio.sleep(0.5)
        print("\nStarting ROM sweep ...\n")

        # 1. SHOULDER_1 — symmetric — max then min
        lo, hi = LIMITS[Joint.SHOULDER_1]
        q_l, q_r = await _joint_rom(
            sim,
            checker,
            q_l,
            q_r,
            Joint.SHOULDER_1,
            "both",
            [hi, lo],
            "SHOULDER_1  (both)",
        )

        # 2. SHOULDER_2 — 0 is home; right sweeps to max, left sweeps to min
        _, r_hi = SHOULDER_2_RIGHT_LIMITS
        q_l, q_r = await _joint_rom(
            sim,
            checker,
            q_l,
            q_r,
            Joint.SHOULDER_2,
            "right",
            [r_hi],
            "SHOULDER_2  (right)",
        )
        l_lo, _ = SHOULDER_2_LEFT_LIMITS
        q_l, q_r = await _joint_rom(
            sim,
            checker,
            q_l,
            q_r,
            Joint.SHOULDER_2,
            "left",
            [l_lo],
            "SHOULDER_2  (left) ",
        )

        # 3. SHOULDER_3 — arms come forward −45° first, then sweep, then return
        print(f"  SHOULDER_3 pre-pose: arms forward {round(math.degrees(_FWD_45), 1)}°")
        q_l, q_r = await _sweep(
            sim,
            checker,
            q_l,
            q_r,
            _with_joint(q_l, Joint.SHOULDER_1, _FWD_45),
            _with_joint(q_r, Joint.SHOULDER_1, _FWD_45),
        )
        lo, hi = LIMITS[Joint.SHOULDER_3]
        q_l, q_r = await _joint_rom(
            sim,
            checker,
            q_l,
            q_r,
            Joint.SHOULDER_3,
            "both",
            [hi, lo],
            "SHOULDER_3  (both)",
        )
        q_l, q_r = await _sweep(
            sim,
            checker,
            q_l,
            q_r,
            _with_joint(q_l, Joint.SHOULDER_1, 0.0),
            _with_joint(q_r, Joint.SHOULDER_1, 0.0),
        )

        # 4. ELBOW — both arms together
        print(f"  ELBOW       (both)  [{round(-_90_DEG, 3)}, {round(_90_DEG, 3)}] → 0")
        q_l, q_r = await _sweep(
            sim,
            checker,
            q_l,
            q_r,
            _with_joint(q_l, Joint.ELBOW, +_90_DEG),
            _with_joint(q_r, Joint.ELBOW, -_90_DEG),
        )
        q_l, q_r = await _sweep(
            sim,
            checker,
            q_l,
            q_r,
            _with_joint(q_l, Joint.ELBOW, 0.0),
            _with_joint(q_r, Joint.ELBOW, 0.0),
        )

        # 5-8. WRIST_1/2/3 + GRIPPER — elbows stay at 90° throughout
        print("  Pre-pose: elbows at 90°")
        q_l, q_r = await _sweep(
            sim,
            checker,
            q_l,
            q_r,
            _with_joint(q_l, Joint.ELBOW, +_90_DEG),
            _with_joint(q_r, Joint.ELBOW, -_90_DEG),
        )

        # 5. WRIST_1
        lo, hi = LIMITS[Joint.WRIST_1]
        q_l, q_r = await _joint_rom(
            sim,
            checker,
            q_l,
            q_r,
            Joint.WRIST_1,
            "both",
            [hi, lo],
            "WRIST 1     (both)",
        )

        # 6. WRIST_2
        lo, hi = LIMITS[Joint.WRIST_2]
        q_l, q_r = await _joint_rom(
            sim,
            checker,
            q_l,
            q_r,
            Joint.WRIST_2,
            "both",
            [hi, lo],
            "WRIST 2     (both)",
        )

        # 7. WRIST_3
        lo, hi = LIMITS[Joint.WRIST_3]
        q_l, q_r = await _joint_rom(
            sim,
            checker,
            q_l,
            q_r,
            Joint.WRIST_3,
            "both",
            [hi, lo],
            "WRIST 3     (both)",
        )

        # 8. GRIPPER — both arms — open (1.0) → closed (0.0)
        q_l, q_r = await _joint_rom(
            sim,
            checker,
            q_l,
            q_r,
            Joint.GRIPPER,
            "both",
            [1.0, 0.0],
            "GRIPPER     (both)",
        )

        # Return elbows to home
        q_l, q_r = await _sweep(
            sim,
            checker,
            q_l,
            q_r,
            _with_joint(q_l, Joint.ELBOW, 0.0),
            _with_joint(q_r, Joint.ELBOW, 0.0),
        )

        print("\nROM sweep complete. Robot is at home position.")
        print("Viser server still running — press Ctrl+C to exit.\n")
        await asyncio.Event().wait()
    finally:
        await sim.disable()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
