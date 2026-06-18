"""
rom.enable

Range of motion test for the Axol robot. Sweeps every joint through its full
range while checking each waypoint for self-collision via pyroki.

Pass --robot to select the backend:
  - sim: launches a viser visualizer at http://localhost:8002 and runs the
         sweep once. Collisions are skipped with a warning.
  - axol: enables the motors, eases to home, then prompts to close each
          gripper onto the item and loops the sweep for two hours. When the
          soak finishes (or on Ctrl-C) the robot returns home but keeps
          holding the item with the motors left enabled — run
          ``almond_axol.test.rom.disable`` afterwards to open the grippers
          and retrieve the item. A predicted collision aborts the run, returns
          the robot home, and disables the motors.

Run:
    uv run -m almond_axol.test.rom.enable --robot sim
    uv run -m almond_axol.test.rom.enable --robot axol
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

from ...kinematics.solver import _LEFT_JOINT_NAMES, _RIGHT_JOINT_NAMES
from ...robot.axol import (
    ELBOW_LEFT_LIMITS,
    ELBOW_RIGHT_LIMITS,
    LIMITS,
    SHOULDER_1_LEFT_LIMITS,
    SHOULDER_2_LEFT_LIMITS,
    SHOULDER_2_RIGHT_LIMITS,
    Axol,
)
from ...robot.config import AxolConfig
from ...robot.sim import Sim
from ...utils.shared import URDF_PATH, Joint

CONTROL_RATE_HZ = 100.0  # Hz

SIM_SPEED = 3.0  # rad/s
SIM_PRE_POSE_SPEED = 0.8  # rad/s
SIM_WAYPOINT_PAUSE = 0.5  # seconds

AXOL_SPEED = 1.0  # rad/s
AXOL_PRE_POSE_SPEED = 0.3  # rad/s
# Return-to-home speed, matching teleop's VRTeleopConfig.reset_speed so the
# end-of-soak homing feels identical to a teleop return-to-rest.
AXOL_HOME_SPEED = 0.1 * 2 * math.pi  # rad/s
AXOL_WAYPOINT_PAUSE = 1.0  # seconds
SOAK_DURATION = 7200  # seconds (2 hours)
CYCLE_PAUSE = 2.0  # seconds

WRIST_TEST_ELBOW_ANGLE = math.pi / 2  # rad
SHOULDER_PRE_POSE_ANGLE = -25 * math.pi / 180  # rad

# The gripper is pure position control (like teleop): we command a normalized
# [0, 1] target and the POSITION_FORCE controller tracks it, capping force at
# ArmConfig.gripper.torque_limit. A closed grasp (target 0) therefore simply
# holds at this torque — so GRIPPER_TORQUE_LIMIT *is* the grasp force. The
# default config cap (0.5 Nm) is raised to this value for the test (see
# run_axol).
GRIPPER_TORQUE_LIMIT = 2.0  # Nm — POSITION_FORCE grasp force (output cap)
GRIPPER_SPEED = 1.0  # normalized [0, 1] per second — open/close speed

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


async def move_grippers(
    robot: Axol | Sim,
    left_q: np.ndarray,  # rad
    right_q: np.ndarray,  # rad
    left_grip: float,  # normalized [0, 1] — 0 closed, 1 open
    right_grip: float,  # normalized [0, 1] — 0 closed, 1 open
    speed: float,  # normalized [0, 1] per second
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothly drive each gripper to a normalized target, same as teleop.

    The gripper is pure position control: we just command the [0, 1] target and
    let the POSITION_FORCE controller track it, capping force at the gripper's
    ``torque_limit`` (so a closed grasp simply holds at that torque). This
    smoothsteps the command from its current value to the target so the motion
    is gradual; the arm joints are held at ``left_q`` / ``right_q``.
    """
    gripper_index = JOINT_INDEX[Joint.GRIPPER]
    left = left_q.copy()
    right = right_q.copy()
    l0 = float(left[gripper_index])
    r0 = float(right[gripper_index])

    max_delta = max(abs(left_grip - l0), abs(right_grip - r0))
    duration = max(max_delta / speed, 0.1)  # seconds
    dt = 1.0 / CONTROL_RATE_HZ  # seconds
    start_time = time.monotonic()
    while True:
        progress = min((time.monotonic() - start_time) / duration, 1.0)
        smooth = progress * progress * (3.0 - 2.0 * progress)
        left[gripper_index] = l0 + (left_grip - l0) * smooth
        right[gripper_index] = r0 + (right_grip - r0) * smooth
        await robot.motion_control(left=left, right=right)
        if progress >= 1.0:
            break
        await asyncio.sleep(dt)
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
    # Shoulder_1 limits are mirrored across arms: the right arm's range is the
    # negation of the left's, so sweep both arms simultaneously in mirror.
    s1_left_low, s1_left_high = SHOULDER_1_LEFT_LIMITS
    print(
        f"  SHOULDER_1  (both, mirrored)  "
        f"[{round(s1_left_high, 3)}, {round(s1_left_low, 3)}] → 0"
    )
    for value in (s1_left_high, s1_left_low, 0.0):
        left_q, right_q = await sweep_to_target(
            robot,
            checker,
            left_q,
            right_q,
            with_joint(left_q, Joint.SHOULDER_1, value),
            with_joint(right_q, Joint.SHOULDER_1, -value),
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
        print("Viser server running at  http://localhost:8002")
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


async def return_home(robot: Axol) -> None:
    """Ease the arms back to home from their current pose, keeping the grippers shut.

    Used to bring the robot to a safe home position while it stays clamped on
    the item. The grippers are left exactly where they are (still grasping) and
    the motors stay enabled; ``rom.disable`` releases the item afterwards.
    """
    gripper_i = JOINT_INDEX[Joint.GRIPPER]
    cur_left, cur_right = await robot.get_positions()
    home_left = home_pose()
    home_right = home_pose()
    home_left[gripper_i] = cur_left[gripper_i]
    home_right[gripper_i] = cur_right[gripper_i]
    print("Returning home (still holding the item) ...")
    await sweep_unchecked(
        robot, cur_left, cur_right, home_left, home_right, speed=AXOL_HOME_SPEED
    )


async def close_buses(robot: Axol) -> None:
    """Close the CAN buses without disabling the motors.

    Stops the reader loops and shuts the sockets so the process can exit
    cleanly, but sends no shutdown command — every motor keeps holding its last
    command (arms at home, grippers clamped on the item) so ``rom.disable`` can
    attach later and release it.
    """
    buses = []
    if robot.left is not None:
        buses.append(robot._left_bus)
    if robot.right is not None:
        buses.append(robot._right_bus)
    await asyncio.gather(*(bus.close() for bus in buses))


async def run_axol() -> None:
    checker = CollisionChecker()

    print("=== ROM TEST — PHYSICAL ROBOT ===")
    print("Make sure the area is clear.\n")

    config = AxolConfig(left_stiffness=1.0, right_stiffness=1.0)
    config.left.gripper.torque_limit = GRIPPER_TORQUE_LIMIT
    config.right.gripper.torque_limit = GRIPPER_TORQUE_LIMIT
    axol = Axol(config=config)
    await axol.enable()
    print("Motors enabled.")
    await asyncio.sleep(2.0)

    closed_left_q: np.ndarray | None = None
    closed_right_q: np.ndarray | None = None

    def home_with_grippers_closed() -> tuple[np.ndarray, np.ndarray]:
        assert closed_left_q is not None and closed_right_q is not None
        return closed_left_q.copy(), closed_right_q.copy()

    # When True (normal soak completion or Ctrl-C) the motors are left enabled
    # and the item stays gripped; otherwise (collision / unexpected error) the
    # motors are disabled in the finally block.
    keep_enabled = False

    try:
        home = home_pose()
        gripper_i = JOINT_INDEX[Joint.GRIPPER]

        # Ease the arms from wherever they actually are to home before anything
        # else, with the grippers open (1.0). The first motion_control would
        # otherwise command home as a single stiff (s=1) impedance setpoint and
        # snap the arms there; sweep_unchecked ramps them in with a smoothstep
        # trajectory instead.
        cur_left, cur_right = await axol.get_positions()
        ready_left = home.copy()
        ready_right = home.copy()
        ready_left[gripper_i] = 1.0
        ready_right[gripper_i] = 1.0
        print("Easing to home position (grippers open) ...")
        await sweep_unchecked(
            axol, cur_left, cur_right, ready_left, ready_right, speed=AXOL_HOME_SPEED
        )
        left_q, right_q = ready_left, ready_right

        await asyncio.to_thread(input, "Press Enter to close the RIGHT gripper ...")
        left_q, right_q = await move_grippers(
            axol, left_q, right_q, left_q[gripper_i], 0.0, GRIPPER_SPEED
        )

        await asyncio.to_thread(input, "Press Enter to close the LEFT gripper ...")
        left_q, right_q = await move_grippers(
            axol, left_q, right_q, 0.0, right_q[gripper_i], GRIPPER_SPEED
        )
        print("Both grippers closed.")

        closed_left_q = left_q.copy()
        closed_right_q = right_q.copy()

        print("Starting ROM test in 5 s ...")
        await asyncio.sleep(5.0)

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

        print(f"\n2-hour soak complete — {cycle} cycle(s) finished.")

        print("Returning to home position ...")
        home_left, home_right = home_with_grippers_closed()
        left_q, right_q = await sweep_to_target(
            axol,
            checker,
            left_q,
            right_q,
            home_left,
            home_right,
            AXOL_HOME_SPEED,
            AXOL_WAYPOINT_PAUSE,
            abort_on_collision=True,
        )

        # Leave the robot holding the item with the motors enabled. The operator
        # runs rom.disable to open the grippers and retrieve the item.
        keep_enabled = True

    except CollisionAbort as e:
        print(f"\n⚠  COLLISION ABORT: {e}")
        print("Returning to home position ...")
        safe_left, safe_right = home_with_grippers_closed()
        await sweep_unchecked(
            axol, e.left_q, e.right_q, safe_left, safe_right, speed=AXOL_SPEED
        )
        print("Home reached.")

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nInterrupted — returning home, keeping the item gripped ...")
        await return_home(axol)
        keep_enabled = True

    finally:
        if keep_enabled:
            await close_buses(axol)
            print(
                "\nMotors left enabled — robot is holding the item.\n"
                "Run `uv run -m almond_axol.test.rom.disable` to open the "
                "grippers and retrieve it."
            )
        else:
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
