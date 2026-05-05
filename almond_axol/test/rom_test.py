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
    ELBOW_LEFT_LIMITS,
    ELBOW_RIGHT_LIMITS,
    LIMITS,
    SHOULDER_2_LEFT_LIMITS,
    SHOULDER_2_RIGHT_LIMITS,
)
from ..robot.sim import Sim
from ..shared import URDF_PATH, Joint

_SPEED = 3.0
_PRE_POSE_SPEED = 0.8
_RATE_HZ = 100.0
_PAUSE = 0.5

_WRIST_ELBOW = math.pi / 2
_FWD_25 = -25 * math.pi / 180

_IDX: dict[Joint, int] = {j: i for i, j in enumerate(Joint)}
_N = len(list(Joint))


class CollisionChecker:
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

        home_dists = np.asarray(_check(jnp.zeros(self._n, dtype=jnp.float32)))
        self._baseline = float(home_dists.min())
        self._check = _check
        print(f"Collision model ready. Home baseline: {self._baseline:.4f} m\n")

    def is_safe(self, q_l: np.ndarray, q_r: np.ndarray) -> tuple[bool, float]:
        q_full = np.zeros(self._n, dtype=np.float32)
        q_full[self._left_idx] = q_l[:7]
        q_full[self._right_idx] = q_r[:7]
        distances = np.asarray(self._check(jnp.asarray(q_full)))
        min_dist = float(distances.min())
        return min_dist > self._baseline - self._margin, min_dist


def _home() -> np.ndarray:
    return np.zeros(_N, dtype=np.float32)


async def _sweep(
    sim: Sim,
    checker: CollisionChecker,
    q_l: np.ndarray,
    q_r: np.ndarray,
    tgt_l: np.ndarray,
    tgt_r: np.ndarray,
    speed: float = _SPEED,
) -> tuple[np.ndarray, np.ndarray]:
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
    dur = max(dist / speed, 0.1)
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
    arm: str,
    waypoints: list[float],
    label: str,
    home_val: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
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

        _, r_hi = SHOULDER_2_RIGHT_LIMITS
        l_lo, _ = SHOULDER_2_LEFT_LIMITS
        print(f"  SHOULDER_2  (both)  → [{round(l_lo, 3)}, {round(r_hi, 3)}] → 0")
        q_l, q_r = await _sweep(
            sim,
            checker,
            q_l,
            q_r,
            _with_joint(q_l, Joint.SHOULDER_2, l_lo),
            _with_joint(q_r, Joint.SHOULDER_2, r_hi),
        )
        q_l, q_r = await _sweep(
            sim,
            checker,
            q_l,
            q_r,
            _with_joint(q_l, Joint.SHOULDER_2, 0.0),
            _with_joint(q_r, Joint.SHOULDER_2, 0.0),
        )

        print(f"  SHOULDER_3 pre-pose: arms forward {round(math.degrees(_FWD_25), 1)}°")
        q_l, q_r = await _sweep(
            sim,
            checker,
            q_l,
            q_r,
            _with_joint(q_l, Joint.SHOULDER_1, -_FWD_25),
            _with_joint(q_r, Joint.SHOULDER_1, _FWD_25),
            speed=_PRE_POSE_SPEED,
        )
        lo, hi = LIMITS[Joint.SHOULDER_3]
        print(f"  SHOULDER_3  (both fwd)  [{round(lo, 3)}, {round(hi, 3)}] → 0")
        q_l, q_r = await _sweep(
            sim,
            checker,
            q_l,
            q_r,
            _with_joint(q_l, Joint.SHOULDER_3, lo),
            _with_joint(q_r, Joint.SHOULDER_3, -lo),
        )
        q_l, q_r = await _sweep(
            sim,
            checker,
            q_l,
            q_r,
            _with_joint(q_l, Joint.SHOULDER_3, hi),
            _with_joint(q_r, Joint.SHOULDER_3, -hi),
        )
        q_l, q_r = await _sweep(
            sim,
            checker,
            q_l,
            q_r,
            _with_joint(q_l, Joint.SHOULDER_3, 0.0),
            _with_joint(q_r, Joint.SHOULDER_3, 0.0),
        )
        q_l, q_r = await _sweep(
            sim,
            checker,
            q_l,
            q_r,
            _with_joint(q_l, Joint.SHOULDER_1, 0.0),
            _with_joint(q_r, Joint.SHOULDER_1, 0.0),
            speed=_PRE_POSE_SPEED,
        )

        _, elbow_l_hi = ELBOW_LEFT_LIMITS
        elbow_r_lo, _ = ELBOW_RIGHT_LIMITS
        print(
            f"  ELBOW       (both)  [{round(elbow_r_lo, 3)}, {round(elbow_l_hi, 3)}] → 0"
        )
        q_l, q_r = await _sweep(
            sim,
            checker,
            q_l,
            q_r,
            _with_joint(q_l, Joint.ELBOW, elbow_l_hi),
            _with_joint(q_r, Joint.ELBOW, elbow_r_lo),
        )
        q_l, q_r = await _sweep(
            sim,
            checker,
            q_l,
            q_r,
            _with_joint(q_l, Joint.ELBOW, 0.0),
            _with_joint(q_r, Joint.ELBOW, 0.0),
        )

        print("  Pre-pose: elbows at 90°")
        q_l, q_r = await _sweep(
            sim,
            checker,
            q_l,
            q_r,
            _with_joint(q_l, Joint.ELBOW, +_WRIST_ELBOW),
            _with_joint(q_r, Joint.ELBOW, -_WRIST_ELBOW),
        )

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
