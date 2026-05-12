"""
rom_test

Range of motion test for the Axol robot. Sweeps every joint through its full
range while checking each waypoint for self-collision via pyroki.

Pass --robot to select the backend:
  - sim: launches a viser visualizer at http://localhost:8080 and runs the
         sweep once. Collisions are skipped with a warning.
  - axol: enables motors after a 5 s safety countdown, opens/closes the
          grippers (torque-monitored) as a pre-check, then loops the sweep
          for one hour. Any predicted collision aborts the run and returns
          the robot home.

Run:
    python -m almond_axol.test.rom_test --robot sim
    python -m almond_axol.test.rom_test --robot axol
"""

import argparse
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
from ..robot.sim import Sim
from ..shared import URDF_PATH, Joint

CONTROL_RATE_HZ = 100.0  # Hz

SIM_SPEED = 3.0  # rad/s
SIM_PRE_POSE_SPEED = 0.8  # rad/s
SIM_WAYPOINT_PAUSE = 0.5  # seconds

AXOL_SPEED = 1.0  # rad/s
AXOL_PRE_POSE_SPEED = 0.3  # rad/s
AXOL_GRIPPER_SPEED = 0.1  # rad/s
AXOL_WAYPOINT_PAUSE = 1.0  # seconds
SAFETY_COUNTDOWN = 5  # seconds
SOAK_DURATION = 3600  # seconds
CYCLE_PAUSE = 2.0  # seconds

WRIST_TEST_ELBOW_ANGLE = math.pi / 2  # rad
SHOULDER_PRE_POSE_ANGLE = -25 * math.pi / 180  # rad

GRIPPER_TORQUE_THRESHOLD = 0.6  # Nm
GRIPPER_STEP = 0.005  # normalized [0, 1] per iteration
GRIPPER_STEP_DELAY = 0.01  # seconds between steps
GRIPPER_OSC_SPEED = 0.5  # normalized [0, 1] per second — oscillation speed

JOINT_INDEX: dict[Joint, int] = {j: i for i, j in enumerate(Joint)}
NUM_JOINTS = len(list(Joint))


class CollisionChecker:
    def __init__(self, margin: float = 0.01) -> None:  # margin in meters
        print("Loading collision model (pyroki) ...")
        urdf = yourdfpy.URDF.load(str(URDF_PATH), mesh_dir=str(URDF_PATH.parent))
        robot = pk.Robot.from_urdf(urdf)
        robot_collision = pk.collision.RobotCollision.from_urdf(urdf)

        actuated = list(robot.joints.actuated_names)
        name_to_index = {n: i for i, n in enumerate(actuated)}
        self.left_indices = [name_to_index[n] for n in _LEFT_JOINT_NAMES]
        self.right_indices = [name_to_index[n] for n in _RIGHT_JOINT_NAMES]
        self.num_actuated = robot.joints.num_actuated_joints
        self.margin = margin

        @jax.jit
        def check_collision(q: jax.Array) -> jax.Array:
            return robot_collision.compute_self_collision_distance(robot, q)

        home_distances = np.asarray(
            check_collision(jnp.zeros(self.num_actuated, dtype=jnp.float32))
        )
        self.baseline = float(home_distances.min())
        self.check_collision = check_collision
        print(f"Collision model ready. Home baseline: {self.baseline:.4f} m\n")

    def is_safe(
        self,
        left_q: np.ndarray,  # rad
        right_q: np.ndarray,  # rad
    ) -> tuple[bool, float]:
        full_q = np.zeros(self.num_actuated, dtype=np.float32)
        full_q[self.left_indices] = left_q[:7]
        full_q[self.right_indices] = right_q[:7]
        distances = np.asarray(self.check_collision(jnp.asarray(full_q)))
        min_distance = float(distances.min())
        return min_distance > self.baseline - self.margin, min_distance


class CollisionAbort(Exception):
    def __init__(
        self,
        msg: str,
        left_q: np.ndarray,  # rad
        right_q: np.ndarray,  # rad
    ) -> None:
        super().__init__(msg)
        self.left_q = left_q
        self.right_q = right_q


def home_pose() -> np.ndarray:
    return np.zeros(NUM_JOINTS, dtype=np.float32)


async def countdown(seconds: int) -> None:
    for i in range(seconds, 0, -1):
        print(f"  Enabling in {i}s ...", end="\r", flush=True)
        await asyncio.sleep(1.0)
    print()


async def oscillate_grippers_until_caught(
    axol: Axol,
    left_q: np.ndarray,  # rad
    right_q: np.ndarray,  # rad
    torque_threshold: float,  # Nm
) -> tuple[np.ndarray, np.ndarray]:
    """Oscillate both grippers open↔closed; each side holds when |torque| ≥ threshold during a close pass.

    Returns once both grippers are holding. Each gripper acts independently.
    Torque is only checked while closing — during opening the gripper is
    allowed to reach the open hard-stop without triggering a hold.
    """
    gripper_index = JOINT_INDEX[Joint.GRIPPER]
    left_motor = axol.left.motors[Joint.GRIPPER]
    right_motor = axol.right.motors[Joint.GRIPPER]

    left = left_q.copy()
    right = right_q.copy()
    left[gripper_index] = 0.0
    right[gripper_index] = 0.0

    left_direction = +1  # +1 opening, -1 closing
    right_direction = +1
    left_holding = False
    right_holding = False

    dt = GRIPPER_STEP_DELAY  # seconds per iteration

    while not (left_holding and right_holding):
        if not left_holding:
            new_pos = left[gripper_index] + left_direction * GRIPPER_OSC_SPEED * dt
            if new_pos >= 1.0:
                new_pos = 1.0
                left_direction = -1
            elif new_pos <= 0.0:
                new_pos = 0.0
                left_direction = +1
            left[gripper_index] = new_pos
        if not right_holding:
            new_pos = right[gripper_index] + right_direction * GRIPPER_OSC_SPEED * dt
            if new_pos >= 1.0:
                new_pos = 1.0
                right_direction = -1
            elif new_pos <= 0.0:
                new_pos = 0.0
                right_direction = +1
            right[gripper_index] = new_pos

        await axol.motion_control(left=left, right=right)
        await asyncio.sleep(dt)

        if not left_holding and left_direction == -1:
            left_torque = await left_motor.get_torque()
            if abs(left_torque) >= torque_threshold:
                print(
                    f"  LEFT gripper: |torque| {abs(left_torque):.3f} Nm ≥ "
                    f"{torque_threshold} Nm — holding at {left[gripper_index]:.3f}"
                )
                left_holding = True
        if not right_holding and right_direction == -1:
            right_torque = await right_motor.get_torque()
            if abs(right_torque) >= torque_threshold:
                print(
                    f"  RIGHT gripper: |torque| {abs(right_torque):.3f} Nm ≥ "
                    f"{torque_threshold} Nm — holding at {right[gripper_index]:.3f}"
                )
                right_holding = True

    return left, right


async def drive_grippers_to_torque(
    axol: Axol,
    left_q: np.ndarray,  # rad
    right_q: np.ndarray,  # rad
    left_direction: int,  # +1 to open, -1 to close, 0 to hold
    right_direction: int,  # +1 to open, -1 to close, 0 to hold
    torque_threshold: float,  # Nm
) -> tuple[np.ndarray, np.ndarray]:
    """Step each gripper toward open/close until torque crosses threshold or limit reached.

    Returns the final (left_q, right_q) with gripper positions captured at the
    point each side stopped. Callers hold this position for the rest of the
    test by re-using these joint vectors.
    """
    gripper_index = JOINT_INDEX[Joint.GRIPPER]
    left_motor = axol.left.motors[Joint.GRIPPER]
    right_motor = axol.right.motors[Joint.GRIPPER]

    left = left_q.copy()
    right = right_q.copy()
    left_done = left_direction == 0
    right_done = right_direction == 0

    while not (left_done and right_done):
        if not left_done:
            new_pos = max(
                0.0, min(1.0, left[gripper_index] + left_direction * GRIPPER_STEP)
            )
            left[gripper_index] = new_pos
            if new_pos in (0.0, 1.0):
                print(f"  LEFT gripper: position bound {new_pos:.2f} reached")
                left_done = True
        if not right_done:
            new_pos = max(
                0.0, min(1.0, right[gripper_index] + right_direction * GRIPPER_STEP)
            )
            right[gripper_index] = new_pos
            if new_pos in (0.0, 1.0):
                print(f"  RIGHT gripper: position bound {new_pos:.2f} reached")
                right_done = True

        await axol.motion_control(left=left, right=right)
        await asyncio.sleep(GRIPPER_STEP_DELAY)

        if not left_done:
            left_torque = await left_motor.get_torque()
            if abs(left_torque) >= torque_threshold:
                print(
                    f"  LEFT gripper: |torque| {abs(left_torque):.3f} Nm ≥ "
                    f"{torque_threshold} Nm — holding at {left[gripper_index]:.3f}"
                )
                left_done = True
        if not right_done:
            right_torque = await right_motor.get_torque()
            if abs(right_torque) >= torque_threshold:
                print(
                    f"  RIGHT gripper: |torque| {abs(right_torque):.3f} Nm ≥ "
                    f"{torque_threshold} Nm — holding at {right[gripper_index]:.3f}"
                )
                right_done = True

    return left, right


async def sweep_to_target(
    robot: Axol | Sim,
    checker: CollisionChecker,
    left_q: np.ndarray,  # rad
    right_q: np.ndarray,  # rad
    left_target: np.ndarray,  # rad
    right_target: np.ndarray,  # rad
    speed: float,  # rad/s
    pause: float,  # seconds
    abort_on_collision: bool,
) -> tuple[np.ndarray, np.ndarray]:
    safe, min_distance = checker.is_safe(left_target, right_target)
    if not safe:
        if abort_on_collision:
            raise CollisionAbort(
                f"collision detected (min dist {min_distance:.4f} m)",
                left_q,
                right_q,
            )
        print(
            f"    ⚠  collision detected (min dist {min_distance:.4f} m) — waypoint skipped"
        )
        return left_q, right_q

    max_joint_delta = max(
        float(np.max(np.abs(left_target - left_q))),
        float(np.max(np.abs(right_target - right_q))),
    )
    duration = max(max_joint_delta / speed, 0.1)  # seconds
    dt = 1.0 / CONTROL_RATE_HZ  # seconds
    start_time = time.monotonic()
    while True:
        elapsed = time.monotonic() - start_time
        progress = min(elapsed / duration, 1.0)
        smooth = progress * progress * (3.0 - 2.0 * progress)
        await robot.motion_control(
            left=(left_q * (1 - smooth) + left_target * smooth).astype(np.float32),
            right=(right_q * (1 - smooth) + right_target * smooth).astype(np.float32),
        )
        if progress >= 1.0:
            break
        await asyncio.sleep(dt)
    await asyncio.sleep(pause)
    return left_target.copy(), right_target.copy()


async def sweep_unchecked(
    robot: Axol | Sim,
    left_q: np.ndarray,  # rad
    right_q: np.ndarray,  # rad
    left_target: np.ndarray,  # rad
    right_target: np.ndarray,  # rad
    speed: float,  # rad/s
) -> None:
    max_joint_delta = max(
        float(np.max(np.abs(left_target - left_q))),
        float(np.max(np.abs(right_target - right_q))),
    )
    duration = max(max_joint_delta / speed, 0.1)  # seconds
    dt = 1.0 / CONTROL_RATE_HZ  # seconds
    start_time = time.monotonic()
    while True:
        elapsed = time.monotonic() - start_time
        progress = min(elapsed / duration, 1.0)
        smooth = progress * progress * (3.0 - 2.0 * progress)
        await robot.motion_control(
            left=(left_q * (1 - smooth) + left_target * smooth).astype(np.float32),
            right=(right_q * (1 - smooth) + right_target * smooth).astype(np.float32),
        )
        if progress >= 1.0:
            break
        await asyncio.sleep(dt)


def with_joint(
    q: np.ndarray,  # rad
    joint: Joint,
    value: float,  # rad
) -> np.ndarray:
    out = q.copy()
    out[JOINT_INDEX[joint]] = value
    return out


async def sweep_joint_range(
    robot: Axol | Sim,
    checker: CollisionChecker,
    left_q: np.ndarray,  # rad
    right_q: np.ndarray,  # rad
    joint: Joint,
    arm: str,
    waypoints: list[float],  # rad
    label: str,
    speed: float,  # rad/s
    pause: float,  # seconds
    abort_on_collision: bool,
    home_value: float = 0.0,  # rad
) -> tuple[np.ndarray, np.ndarray]:
    print(f"  {label}  {[round(w, 3) for w in waypoints]} → {round(home_value, 3)}")
    for value in waypoints:
        left_target = (
            with_joint(left_q, joint, value)
            if arm in ("both", "left")
            else left_q.copy()
        )
        right_target = (
            with_joint(right_q, joint, value)
            if arm in ("both", "right")
            else right_q.copy()
        )
        left_q, right_q = await sweep_to_target(
            robot,
            checker,
            left_q,
            right_q,
            left_target,
            right_target,
            speed,
            pause,
            abort_on_collision,
        )
    left_target = (
        with_joint(left_q, joint, home_value)
        if arm in ("both", "left")
        else left_q.copy()
    )
    right_target = (
        with_joint(right_q, joint, home_value)
        if arm in ("both", "right")
        else right_q.copy()
    )
    left_q, right_q = await sweep_to_target(
        robot,
        checker,
        left_q,
        right_q,
        left_target,
        right_target,
        speed,
        pause,
        abort_on_collision,
    )
    return left_q, right_q


async def run_rom_cycle(
    robot: Axol | Sim,
    checker: CollisionChecker,
    left_q: np.ndarray,  # rad
    right_q: np.ndarray,  # rad
    speed: float,  # rad/s
    pre_pose_speed: float,  # rad/s
    pause: float,  # seconds
    abort_on_collision: bool,
    shoulder3_mirror: bool,
) -> tuple[np.ndarray, np.ndarray]:
    low, high = LIMITS[Joint.SHOULDER_1]
    left_q, right_q = await sweep_joint_range(
        robot,
        checker,
        left_q,
        right_q,
        Joint.SHOULDER_1,
        "both",
        [high, low],
        "SHOULDER_1  (both)",
        speed,
        pause,
        abort_on_collision,
    )

    _, right_high = SHOULDER_2_RIGHT_LIMITS
    left_low, _ = SHOULDER_2_LEFT_LIMITS
    print(f"  SHOULDER_2  (both)  → [{round(left_low, 3)}, {round(right_high, 3)}] → 0")
    left_q, right_q = await sweep_to_target(
        robot,
        checker,
        left_q,
        right_q,
        with_joint(left_q, Joint.SHOULDER_2, left_low),
        with_joint(right_q, Joint.SHOULDER_2, right_high),
        speed,
        pause,
        abort_on_collision,
    )
    left_q, right_q = await sweep_to_target(
        robot,
        checker,
        left_q,
        right_q,
        with_joint(left_q, Joint.SHOULDER_2, 0.0),
        with_joint(right_q, Joint.SHOULDER_2, 0.0),
        speed,
        pause,
        abort_on_collision,
    )

    print(
        f"  SHOULDER_3 pre-pose: arms forward {round(math.degrees(SHOULDER_PRE_POSE_ANGLE), 1)}°"
    )
    left_q, right_q = await sweep_to_target(
        robot,
        checker,
        left_q,
        right_q,
        with_joint(left_q, Joint.SHOULDER_1, -SHOULDER_PRE_POSE_ANGLE),
        with_joint(right_q, Joint.SHOULDER_1, SHOULDER_PRE_POSE_ANGLE),
        pre_pose_speed,
        pause,
        abort_on_collision,
    )
    low, high = LIMITS[Joint.SHOULDER_3]
    print(f"  SHOULDER_3  (both fwd)  [{round(low, 3)}, {round(high, 3)}] → 0")
    shoulder3_left_low = low if shoulder3_mirror else -low
    shoulder3_left_high = high if shoulder3_mirror else -high
    left_q, right_q = await sweep_to_target(
        robot,
        checker,
        left_q,
        right_q,
        with_joint(left_q, Joint.SHOULDER_3, shoulder3_left_low),
        with_joint(right_q, Joint.SHOULDER_3, -low),
        speed,
        pause,
        abort_on_collision,
    )
    left_q, right_q = await sweep_to_target(
        robot,
        checker,
        left_q,
        right_q,
        with_joint(left_q, Joint.SHOULDER_3, shoulder3_left_high),
        with_joint(right_q, Joint.SHOULDER_3, -high),
        speed,
        pause,
        abort_on_collision,
    )
    left_q, right_q = await sweep_to_target(
        robot,
        checker,
        left_q,
        right_q,
        with_joint(left_q, Joint.SHOULDER_3, 0.0),
        with_joint(right_q, Joint.SHOULDER_3, 0.0),
        speed,
        pause,
        abort_on_collision,
    )
    left_q, right_q = await sweep_to_target(
        robot,
        checker,
        left_q,
        right_q,
        with_joint(left_q, Joint.SHOULDER_1, 0.0),
        with_joint(right_q, Joint.SHOULDER_1, 0.0),
        pre_pose_speed,
        pause,
        abort_on_collision,
    )

    _, elbow_left_high = ELBOW_LEFT_LIMITS
    elbow_right_low, _ = ELBOW_RIGHT_LIMITS
    print(
        f"  ELBOW       (both)  [{round(elbow_right_low, 3)}, {round(elbow_left_high, 3)}] → 0"
    )
    left_q, right_q = await sweep_to_target(
        robot,
        checker,
        left_q,
        right_q,
        with_joint(left_q, Joint.ELBOW, elbow_left_high),
        with_joint(right_q, Joint.ELBOW, elbow_right_low),
        speed,
        pause,
        abort_on_collision,
    )
    left_q, right_q = await sweep_to_target(
        robot,
        checker,
        left_q,
        right_q,
        with_joint(left_q, Joint.ELBOW, 0.0),
        with_joint(right_q, Joint.ELBOW, 0.0),
        speed,
        pause,
        abort_on_collision,
    )

    print("  Pre-pose: elbows at 90°")
    left_q, right_q = await sweep_to_target(
        robot,
        checker,
        left_q,
        right_q,
        with_joint(left_q, Joint.ELBOW, +WRIST_TEST_ELBOW_ANGLE),
        with_joint(right_q, Joint.ELBOW, -WRIST_TEST_ELBOW_ANGLE),
        speed,
        pause,
        abort_on_collision,
    )

    low, high = LIMITS[Joint.WRIST_1]
    left_q, right_q = await sweep_joint_range(
        robot,
        checker,
        left_q,
        right_q,
        Joint.WRIST_1,
        "both",
        [high, low],
        "WRIST 1     (both)",
        speed,
        pause,
        abort_on_collision,
    )

    low, high = LIMITS[Joint.WRIST_2]
    left_q, right_q = await sweep_joint_range(
        robot,
        checker,
        left_q,
        right_q,
        Joint.WRIST_2,
        "both",
        [high, low],
        "WRIST 2     (both)",
        speed,
        pause,
        abort_on_collision,
    )

    low, high = LIMITS[Joint.WRIST_3]
    left_q, right_q = await sweep_joint_range(
        robot,
        checker,
        left_q,
        right_q,
        Joint.WRIST_3,
        "both",
        [high, low],
        "WRIST 3     (both)",
        speed,
        pause,
        abort_on_collision,
    )

    left_q, right_q = await sweep_to_target(
        robot,
        checker,
        left_q,
        right_q,
        with_joint(left_q, Joint.ELBOW, 0.0),
        with_joint(right_q, Joint.ELBOW, 0.0),
        speed,
        pause,
        abort_on_collision,
    )

    return left_q, right_q


async def run_sim() -> None:
    checker = CollisionChecker()

    sim = Sim()
    await sim.enable()
    try:
        print("=== ROM TEST — simulation ===")
        print("Viser server running at  http://localhost:8080")
        print("Open that URL in a browser, then come back here.\n")
        await asyncio.to_thread(input, "Press Enter to start the ROM sweep ...")

        left_q = home_pose()
        right_q = home_pose()

        await sim.motion_control(left=left_q, right=right_q)
        await asyncio.sleep(0.5)
        print("\nStarting ROM sweep ...\n")

        await run_rom_cycle(
            sim,
            checker,
            left_q,
            right_q,
            speed=SIM_SPEED,
            pre_pose_speed=SIM_PRE_POSE_SPEED,
            pause=SIM_WAYPOINT_PAUSE,
            abort_on_collision=False,
            shoulder3_mirror=True,
        )

        print("\nROM sweep complete. Robot is at home position.")
        print("Viser server still running — press Ctrl+C to exit.\n")
        await asyncio.Event().wait()
    finally:
        await sim.disable()


async def run_axol() -> None:
    checker = CollisionChecker()

    print("=== ROM TEST — PHYSICAL ROBOT ===")
    print("Make sure the robot is in home position and the area is clear.\n")
    await asyncio.to_thread(input, "Press Enter to begin the safety countdown ...")

    await countdown(SAFETY_COUNTDOWN)

    axol = Axol()
    await axol.enable()
    print("Motors enabled.")
    await asyncio.sleep(2.0)

    closed_left_q: np.ndarray | None = None
    closed_right_q: np.ndarray | None = None

    def home_with_grippers_closed() -> tuple[np.ndarray, np.ndarray]:
        assert closed_left_q is not None and closed_right_q is not None
        return closed_left_q.copy(), closed_right_q.copy()

    try:
        home = home_pose()

        print("Oscillating grippers until each one catches torque ≥ threshold ...")
        left_q, right_q = await oscillate_grippers_until_caught(
            axol, home, home, GRIPPER_TORQUE_THRESHOLD
        )
        print("Both grippers held. Waiting 10 s ...")
        await asyncio.sleep(10.0)

        closed_left_q = left_q.copy()
        closed_right_q = right_q.copy()

        deadline = time.monotonic() + SOAK_DURATION
        cycle = 0

        while time.monotonic() < deadline:
            cycle += 1
            remaining = deadline - time.monotonic()
            print(f"\n--- Cycle {cycle}  ({remaining / 60:.1f} min remaining) ---")

            left_q, right_q = home_with_grippers_closed()

            left_q, right_q = await run_rom_cycle(
                axol,
                checker,
                left_q,
                right_q,
                speed=AXOL_SPEED,
                pre_pose_speed=AXOL_PRE_POSE_SPEED,
                pause=AXOL_WAYPOINT_PAUSE,
                abort_on_collision=True,
                shoulder3_mirror=False,
            )

            print(f"\nCycle {cycle} complete.")
            if time.monotonic() < deadline:
                print(f"Waiting {CYCLE_PAUSE}s ...")
                await asyncio.sleep(CYCLE_PAUSE)

        print(f"\n1-hour soak complete — {cycle} cycle(s) finished.")

        print("Opening right gripper")
        await asyncio.sleep(2.0)
        left_q, right_q = await drive_grippers_to_torque(
            axol, left_q, right_q, 0, +1, GRIPPER_TORQUE_THRESHOLD
        )

        print("Opening left gripper")
        await asyncio.sleep(2.0)
        left_q, right_q = await drive_grippers_to_torque(
            axol, left_q, right_q, +1, 0, GRIPPER_TORQUE_THRESHOLD
        )
        print("Grippers open.")

    except CollisionAbort as e:
        print(f"\n⚠  COLLISION ABORT: {e}")
        print("Returning to home position ...")
        safe_left, safe_right = home_with_grippers_closed()
        await sweep_unchecked(
            axol, e.left_q, e.right_q, safe_left, safe_right, speed=AXOL_SPEED
        )
        print("Home reached.")

    finally:
        await axol.disable()
        print("Motors disabled.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Range of motion test for the Axol robot."
    )
    parser.add_argument(
        "--robot",
        choices=["axol", "sim"],
        required=True,
        help="Robot backend: 'axol' for hardware, 'sim' for visualizer.",
    )
    args = parser.parse_args()

    if args.robot == "sim":
        asyncio.run(run_sim())
    else:
        asyncio.run(run_axol())


if __name__ == "__main__":
    main()
