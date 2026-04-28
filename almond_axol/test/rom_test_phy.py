"""Range of motion test — PHYSICAL ROBOT.

Same joint sequence as rom_test.py but runs on real hardware via CAN.
Motors are enabled after a safety countdown; Ctrl+C disables them cleanly.

Each waypoint is checked for self-collision via pyroki before execution.
If a waypoint would cause a collision the sweep is aborted immediately and
motors are disabled.

Joint sequence:
  1. SHOULDER_1  — both arms, max → min → home
  2. SHOULDER_2  — right (0=min, sweep to max), then left (0=max, sweep to min)
  3. SHOULDER_3  — both arms, max → min → home
  4. ELBOW       — both arms together, 120° bend → home
  5. WRIST_1     — both arms, max → min → home
  6. WRIST_2     — arms forward −45° first, then max → min → home, arms return
  7. WRIST_3     — both arms, max → min → home
  8. GRIPPER     — both arms, open → closed → home

Run:
    python -m almond_axol.test.rom_test_phy
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
    ELBOW_LEFT_LIMITS,
    ELBOW_RIGHT_LIMITS,
    LIMITS,
    SHOULDER_2_LEFT_LIMITS,
    SHOULDER_2_RIGHT_LIMITS,
    Axol,
)
from ..shared import URDF_PATH, Joint

_SPEED = 1.2  # rad/s — joint speed during ROM test
_PRE_POSE_SPEED = 0.3  # rad/s — slow speed for shoulder_3 pre-pose moves
_GRIPPER_SPEED = 0.1  # rad/s — very slow for gripper pre-load
_RATE_HZ = 100.0
_PAUSE = 1.0  # seconds to hold at each waypoint
_COUNTDOWN = 5  # seconds before motors enable
_DURATION = 3600  # seconds to run (1 hour)
_CYCLE_PAUSE = 2.0  # seconds between cycles

_WRIST_ELBOW = math.pi / 2  # elbow angle held during wrist movements (90°)
_FWD_25 = -25 * math.pi / 180  # shoulder_1 forward pre-pose for shoulder_3 (25°)

# Gripper positions — motion_control expects normalized [0, 1] for the gripper,
# where 0 maps to 0 rad and 1 maps to LIMITS[Joint.GRIPPER][0] = -0.8037*τ rad.
# Convert from the hardware radian readings: norm = rad / (-0.8037 * 2 * pi)
_GRIPPER_LO = -0.8037 * 2 * math.pi  # -5.0514 rad
_GRIPPER_R_OPEN = -4.6817 / _GRIPPER_LO  # ~0.927
_GRIPPER_L_OPEN = -4.4760 / _GRIPPER_LO  # ~0.886
_GRIPPER_R_CLOSED = -3.2683 / _GRIPPER_LO  # ~0.647
_GRIPPER_L_CLOSED = -3.3915 / _GRIPPER_LO  # ~0.671

_IDX: dict[Joint, int] = {j: i for i, j in enumerate(Joint)}
_N = len(list(Joint))  # 8


# ---------------------------------------------------------------------------
# Collision checker
# ---------------------------------------------------------------------------


class CollisionChecker:
    """Checks robot configurations for self-collision using pyroki.

    Args:
        margin: Safety margin in metres. A waypoint is rejected if any
                collision pair distance falls more than ``margin`` below
                the home-position baseline.
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

        home_dists = np.asarray(_check(jnp.zeros(self._n, dtype=jnp.float32)))
        self._baseline = float(home_dists.min())
        self._check = _check
        print(f"Collision model ready. Home baseline: {self._baseline:.4f} m\n")

    def is_safe(self, q_l: np.ndarray, q_r: np.ndarray) -> tuple[bool, float]:
        """Return ``(safe, min_distance)``.

        Safe when ``min_distance > baseline - margin``.
        """
        q_full = np.zeros(self._n, dtype=np.float32)
        q_full[self._left_idx] = q_l[:7]
        q_full[self._right_idx] = q_r[:7]
        distances = np.asarray(self._check(jnp.asarray(q_full)))
        min_dist = float(distances.min())
        return min_dist > self._baseline - self._margin, min_dist


# ---------------------------------------------------------------------------
# Collision abort exception
# ---------------------------------------------------------------------------


class CollisionAbort(Exception):
    """Raised when a waypoint is rejected due to predicted self-collision.

    Carries the last known safe positions so the caller can return home.
    """

    def __init__(self, msg: str, q_l: np.ndarray, q_r: np.ndarray) -> None:
        super().__init__(msg)
        self.q_l = q_l
        self.q_r = q_r


# ---------------------------------------------------------------------------
# Motion helpers
# ---------------------------------------------------------------------------


def _home() -> np.ndarray:
    return np.zeros(_N, dtype=np.float32)


async def _countdown(secs: int) -> None:
    for i in range(secs, 0, -1):
        print(f"  Enabling in {i}s ...", end="\r", flush=True)
        await asyncio.sleep(1.0)
    print()


async def _sweep(
    axol: Axol,
    checker: CollisionChecker,
    q_l: np.ndarray,
    q_r: np.ndarray,
    tgt_l: np.ndarray,
    tgt_r: np.ndarray,
    speed: float = _SPEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Check target for collision, then S-curve interpolate to it.

    Raises CollisionAbort if the target configuration is unsafe.
    """
    safe, min_dist = checker.is_safe(tgt_l, tgt_r)
    if not safe:
        raise CollisionAbort(
            f"collision detected (min dist {min_dist:.4f} m)",
            q_l,
            q_r,
        )

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
        await axol.motion_control(
            left=(q_l * (1 - s) + tgt_l * s).astype(np.float32),
            right=(q_r * (1 - s) + tgt_r * s).astype(np.float32),
        )
        if a >= 1.0:
            break
        await asyncio.sleep(dt)
    await asyncio.sleep(_PAUSE)
    return tgt_l.copy(), tgt_r.copy()


async def _sweep_unchecked(
    axol: Axol,
    q_l: np.ndarray,
    q_r: np.ndarray,
    tgt_l: np.ndarray,
    tgt_r: np.ndarray,
    speed: float = _SPEED,
) -> None:
    """S-curve move with no collision check — only for returning to home."""
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
        await axol.motion_control(
            left=(q_l * (1 - s) + tgt_l * s).astype(np.float32),
            right=(q_r * (1 - s) + tgt_r * s).astype(np.float32),
        )
        if a >= 1.0:
            break
        await asyncio.sleep(dt)


def _with_joint(q: np.ndarray, joint: Joint, val: float) -> np.ndarray:
    out = q.copy()
    out[_IDX[joint]] = val
    return out


async def _joint_rom(
    axol: Axol,
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
        q_l, q_r = await _sweep(axol, checker, q_l, q_r, tgt_l, tgt_r)
    tgt_l = _with_joint(q_l, joint, home_val) if arm in ("both", "left") else q_l.copy()
    tgt_r = (
        _with_joint(q_r, joint, home_val) if arm in ("both", "right") else q_r.copy()
    )
    q_l, q_r = await _sweep(axol, checker, q_l, q_r, tgt_l, tgt_r)
    return q_l, q_r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _run() -> None:
    checker = CollisionChecker()

    print("=== ROM TEST — PHYSICAL ROBOT ===")
    print("Make sure the robot is in home position and the area is clear.\n")
    await asyncio.to_thread(input, "Press Enter to begin the safety countdown ...")

    await _countdown(_COUNTDOWN)

    axol = Axol()
    await axol.enable()
    print("Motors enabled.")
    await asyncio.sleep(2.0)

    def _start() -> tuple[np.ndarray, np.ndarray]:
        ql = _home()
        ql[_IDX[Joint.GRIPPER]] = _GRIPPER_L_CLOSED
        qr = _home()
        qr[_IDX[Joint.GRIPPER]] = _GRIPPER_R_CLOSED
        return ql, qr

    try:
        # Gripper pre-load commented out — weight is already taped, grippers must not move.
        # home = _home()
        # # Step 1: open both grippers simultaneously
        # print("Opening both grippers ...")
        # right_open = home.copy(); right_open[_IDX[Joint.GRIPPER]] = _GRIPPER_R_OPEN
        # left_open  = home.copy(); left_open[_IDX[Joint.GRIPPER]]  = _GRIPPER_L_OPEN
        # await _sweep_unchecked(axol, home, home, left_open, right_open, speed=_GRIPPER_SPEED)
        # print("Both grippers open. Waiting 5 s ...")
        # await asyncio.sleep(5.0)
        # # Step 2: close right gripper, then left gripper
        # print("Closing right gripper ...")
        # right_closed = right_open.copy(); right_closed[_IDX[Joint.GRIPPER]] = _GRIPPER_R_CLOSED
        # await _sweep_unchecked(axol, left_open, right_open, left_open, right_closed, speed=_GRIPPER_SPEED)
        # print("Right gripper closed. Waiting 5 s ...")
        # await asyncio.sleep(5.0)
        # print("Closing left gripper ...")
        # left_closed = left_open.copy(); left_closed[_IDX[Joint.GRIPPER]] = _GRIPPER_L_CLOSED
        # await _sweep_unchecked(axol, left_open, right_closed, left_closed, right_closed, speed=_GRIPPER_SPEED)
        # print("Left gripper closed. Waiting 5 s ...")
        # await asyncio.sleep(5.0)

        q_l, q_r = _start()
        await asyncio.sleep(2.0)

        deadline = time.monotonic() + _DURATION
        cycle = 0

        while time.monotonic() < deadline:
            cycle += 1
            remaining = deadline - time.monotonic()
            print(f"\n--- Cycle {cycle}  ({remaining / 60:.1f} min remaining) ---")

            q_l, q_r = _start()

            # 1. SHOULDER_1 — symmetric — max then min
            lo, hi = LIMITS[Joint.SHOULDER_1]
            q_l, q_r = await _joint_rom(
                axol,
                checker,
                q_l,
                q_r,
                Joint.SHOULDER_1,
                "both",
                [hi, lo],
                "SHOULDER_1  (both)",
            )

            # 2. SHOULDER_2 — both arms simultaneously
            _, r_hi = SHOULDER_2_RIGHT_LIMITS
            l_lo, _ = SHOULDER_2_LEFT_LIMITS
            print(f"  SHOULDER_2  (both)  → [{round(l_lo, 3)}, {round(r_hi, 3)}] → 0")
            q_l, q_r = await _sweep(
                axol,
                checker,
                q_l,
                q_r,
                _with_joint(q_l, Joint.SHOULDER_2, l_lo),
                _with_joint(q_r, Joint.SHOULDER_2, r_hi),
            )
            q_l, q_r = await _sweep(
                axol,
                checker,
                q_l,
                q_r,
                _with_joint(q_l, Joint.SHOULDER_2, 0.0),
                _with_joint(q_r, Joint.SHOULDER_2, 0.0),
            )

            # 3. SHOULDER_3 — arms come forward 25° first (slow), then sweep, then return (slow)
            print(
                f"  SHOULDER_3 pre-pose: arms forward {round(math.degrees(_FWD_25), 1)}°"
            )
            q_l, q_r = await _sweep(
                axol,
                checker,
                q_l,
                q_r,
                _with_joint(q_l, Joint.SHOULDER_1, _FWD_25),
                _with_joint(q_r, Joint.SHOULDER_1, _FWD_25),
                speed=_PRE_POSE_SPEED,
            )
            lo, hi = LIMITS[Joint.SHOULDER_3]
            q_l, q_r = await _joint_rom(
                axol,
                checker,
                q_l,
                q_r,
                Joint.SHOULDER_3,
                "both",
                [hi, lo],
                "SHOULDER_3  (both)",
            )
            q_l, q_r = await _sweep(
                axol,
                checker,
                q_l,
                q_r,
                _with_joint(q_l, Joint.SHOULDER_1, 0.0),
                _with_joint(q_r, Joint.SHOULDER_1, 0.0),
                speed=_PRE_POSE_SPEED,
            )

            # 4. ELBOW — both arms together, full range from axol.py limits
            _, elbow_l_hi = ELBOW_LEFT_LIMITS
            elbow_r_lo, _ = ELBOW_RIGHT_LIMITS
            print(
                f"  ELBOW       (both)  [{round(elbow_r_lo, 3)}, {round(elbow_l_hi, 3)}] → 0"
            )
            q_l, q_r = await _sweep(
                axol,
                checker,
                q_l,
                q_r,
                _with_joint(q_l, Joint.ELBOW, elbow_l_hi),
                _with_joint(q_r, Joint.ELBOW, elbow_r_lo),
            )
            q_l, q_r = await _sweep(
                axol,
                checker,
                q_l,
                q_r,
                _with_joint(q_l, Joint.ELBOW, 0.0),
                _with_joint(q_r, Joint.ELBOW, 0.0),
            )

            # 5-8. WRIST_1/2/3 + GRIPPER — elbows stay at exactly 90° throughout
            print("  Pre-pose: elbows at 90°")
            q_l, q_r = await _sweep(
                axol,
                checker,
                q_l,
                q_r,
                _with_joint(q_l, Joint.ELBOW, +_WRIST_ELBOW),
                _with_joint(q_r, Joint.ELBOW, -_WRIST_ELBOW),
            )

            # 5. WRIST_1
            lo, hi = LIMITS[Joint.WRIST_1]
            q_l, q_r = await _joint_rom(
                axol,
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
                axol,
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
                axol,
                checker,
                q_l,
                q_r,
                Joint.WRIST_3,
                "both",
                [hi, lo],
                "WRIST 3     (both)",
            )

            # Return elbows to home
            q_l, q_r = await _sweep(
                axol,
                checker,
                q_l,
                q_r,
                _with_joint(q_l, Joint.ELBOW, 0.0),
                _with_joint(q_r, Joint.ELBOW, 0.0),
            )

            print(f"\nCycle {cycle} complete.")
            if time.monotonic() < deadline:
                print(f"Waiting {_CYCLE_PAUSE}s ...")
                await asyncio.sleep(_CYCLE_PAUSE)

        print(f"\n1-hour soak complete — {cycle} cycle(s) finished.")

        # Open both grippers now that the test is done.
        print("Opening grippers ...")
        open_l = q_l.copy()
        open_l[_IDX[Joint.GRIPPER]] = _GRIPPER_L_OPEN
        open_r = q_r.copy()
        open_r[_IDX[Joint.GRIPPER]] = _GRIPPER_R_OPEN
        await _sweep_unchecked(axol, q_l, q_r, open_l, open_r, speed=_GRIPPER_SPEED)
        print("Grippers open.")

    except CollisionAbort as e:
        print(f"\n⚠  COLLISION ABORT: {e}")
        print("Returning to home position ...")
        safe_home_l, safe_home_r = _start()
        await _sweep_unchecked(axol, e.q_l, e.q_r, safe_home_l, safe_home_r)
        print("Home reached.")

    finally:
        await axol.disable()
        print("Motors disabled.")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
