"""
rom.enable

Range of motion test for the Axol robot. Sweeps every joint through its full
range while checking each waypoint for self-collision via pyroki.

Enables the motors, eases to home, then prompts to close each gripper onto the
item and loops the sweep for two hours. When the soak finishes (or on Ctrl-C)
the robot returns home but keeps holding the item with the motors left enabled
— run ``almond_axol.diagnostics.rom.disable`` afterwards to open the grippers
and retrieve the item. A predicted collision aborts the run, returns the robot
home, and disables the motors.

Select a subset of joints and/or a single arm:
  --joints    Comma-separated joints present on the CAN bus (e.g.
              wrist_1,wrist_2,wrist_3). Only these motors are enabled and
              swept; every other joint is left untouched at home. Default: all.
  --no-left / --no-right
              Skip an arm entirely. Only the remaining arm is opened, enabled,
              and swept. Cannot skip both.

The grasp-an-item clamp (hold with force, then soak while holding) only runs on
the full robot. Any subset run drops the grasp step and simply loops the
range-of-motion sweeps for the selected joints; if the gripper is one of them it
is cycled through its full open↔close range like any other joint (holding
nothing, at the default gentle torque).

Run:
    uv run -m almond_axol.diagnostics.rom.enable
    uv run -m almond_axol.diagnostics.rom.enable --no-right
    uv run -m almond_axol.diagnostics.rom.enable --joints wrist_1,wrist_2,wrist_3
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

from ...constants import ARM_JOINTS, CAN_LEFT, CAN_RIGHT, URDF_PATH, Joint
from ...kinematics.solver import _LEFT_JOINT_NAMES, _RIGHT_JOINT_NAMES
from ...motor import ControlMode
from ...robot.axol import (
    ELBOW_LEFT_LIMITS,
    ELBOW_RIGHT_LIMITS,
    LIMITS,
    SHOULDER_1_LEFT_LIMITS,
    SHOULDER_2_LEFT_LIMITS,
    SHOULDER_2_RIGHT_LIMITS,
    Axol,
    AxolArm,
)
from ...robot.config import AxolConfig

CONTROL_RATE_HZ = 100.0  # Hz

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

FULL_JOINT_SET: frozenset[Joint] = frozenset(Joint)


def parse_joints(spec: str | None) -> set[Joint]:
    """Parse a comma-separated joint spec into a set of present :class:`Joint`.

    ``None`` or empty selects every joint. Names match the joint enum values
    (e.g. ``shoulder_1``, ``elbow``, ``gripper``).
    """
    if not spec:
        return set(Joint)
    by_value = {j.value: j for j in Joint}
    selected: set[Joint] = set()
    for raw in spec.split(","):
        name = raw.strip().lower()
        if not name:
            continue
        if name not in by_value:
            valid = ", ".join(by_value)
            raise SystemExit(f"Unknown joint '{name}'. Valid joints: {valid}")
        selected.add(by_value[name])
    return selected or set(Joint)


def _joint_frame(arm: AxolArm, idx: int, joint: Joint, raw: float) -> float:
    """Convert a raw motor-frame reading to the public joint-frame value."""
    if joint == Joint.GRIPPER:
        gi = arm._gripper_i
        return (raw - arm._limits_hi[gi]) / (arm._limits_lo[gi] - arm._limits_hi[gi])
    return raw + float(arm._joint_offsets[idx])


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


class HardwareController:
    """Motion facade over :class:`Axol` that restricts commands to a joint subset.

    Presents the same ``enable`` / ``disable`` / ``get_positions`` /
    ``motion_control`` surface the ROM cycle drives on both backends, but only
    ever touches the joints in ``present`` on hardware. When ``present`` is the
    full joint set this delegates straight to :class:`Axol` so a normal run
    keeps its full feedforward stack and max-step safety check; otherwise absent
    motors are never enabled, commanded, or read (so they may be off the bus).
    """

    def __init__(self, axol: Axol, present: set[Joint]) -> None:
        self._axol = axol
        self._present = set(present)
        self._full = self._present == set(Joint)

    def _arms(self) -> list[AxolArm]:
        return [a for a in (self._axol.left, self._axol.right) if a is not None]

    async def enable(self) -> None:
        if self._full:
            await self._axol.enable()
            return
        bus_tasks = []
        if self._axol.left is not None:
            bus_tasks.append(self._axol._left_bus.start())
        if self._axol.right is not None:
            bus_tasks.append(self._axol._right_bus.start())
        await asyncio.gather(*bus_tasks)
        await asyncio.gather(*[self._enable_arm(a) for a in self._arms()])

    async def _enable_arm(self, arm: AxolArm) -> None:
        motors = [arm.motors[j] for j in self._present]
        await asyncio.gather(*[m.enable() for m in motors])
        await asyncio.gather(
            *[m.set_control_mode(ControlMode.IMPEDANCE) for m in motors]
        )
        if Joint.GRIPPER in self._present:
            await arm._calibrate_gripper()
            await arm.motors[Joint.GRIPPER].set_control_mode(ControlMode.POSITION_FORCE)

    async def disable(self) -> None:
        if self._full:
            await self._axol.disable()
            return
        tasks = []
        for arm in self._arms():
            tasks.extend(arm.motors[j].disable() for j in self._present)
        try:
            await asyncio.gather(*tasks)
        except Exception:
            pass
        finally:
            close_tasks = []
            if self._axol.left is not None:
                close_tasks.append(self._axol._left_bus.close())
            if self._axol.right is not None:
                close_tasks.append(self._axol._right_bus.close())
            await asyncio.gather(*close_tasks)

    async def get_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Current positions as (left, right); an absent arm reports home."""
        left = (
            await self._read_arm(self._axol.left)
            if self._axol.left is not None
            else home_pose()
        )
        right = (
            await self._read_arm(self._axol.right)
            if self._axol.right is not None
            else home_pose()
        )
        return left, right

    async def _read_arm(self, arm: AxolArm) -> np.ndarray:
        if self._full:
            return await arm.get_positions()
        q = home_pose()
        for i, j in enumerate(Joint):
            if j in self._present:
                q[i] = _joint_frame(arm, i, j, await arm.motors[j].get_position())
        return q

    async def motion_control(
        self,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
    ) -> None:
        if self._full:
            await self._axol.motion_control(left=left, right=right)
            return
        tasks = []
        if left is not None and self._axol.left is not None:
            tasks.append(self._command_arm(self._axol.left, left))
        if right is not None and self._axol.right is not None:
            tasks.append(self._command_arm(self._axol.right, right))
        if tasks:
            await asyncio.gather(*tasks)

    async def _command_arm(self, arm: AxolArm, q: np.ndarray) -> None:
        """Gravity-compensated impedance hold for the present arm joints, plus a
        position-force command to the gripper when it is present."""
        q = q.copy()
        arm_q = q[: len(ARM_JOINTS)].astype(np.float32)
        gravity = arm._gravity_comp.gravity_arm(arm_q, is_left=arm._is_left)
        offsets = arm._joint_offsets
        tasks = []
        for i, j in enumerate(ARM_JOINTS):
            if j not in self._present:
                continue
            gains = getattr(arm._arm_config, j.value)
            tasks.append(
                arm.motors[j].set_impedance(
                    float(q[i] - offsets[i]), 0.0, gains.kp, gains.kd, float(gravity[i])
                )
            )
        if Joint.GRIPPER in self._present:
            gi = arm._gripper_i
            gripper_pos = arm._limits_hi[gi] + float(q[gi]) * (
                arm._limits_lo[gi] - arm._limits_hi[gi]
            )
            tasks.append(
                arm.motors[Joint.GRIPPER].set_position_force(
                    gripper_pos,
                    arm._arm_config.gripper.max_speed,
                    arm._arm_config.gripper.torque_limit,
                )
            )
        await asyncio.gather(*tasks)


async def move_grippers(
    robot: Axol | HardwareController,
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
    robot: Axol | HardwareController,
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
    robot: Axol | HardwareController,
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
    robot: Axol | HardwareController,
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
    robot: Axol | HardwareController,
    checker: CollisionChecker,
    left_q: np.ndarray,  # rad
    right_q: np.ndarray,  # rad
    speed: float,  # rad/s
    pre_pose_speed: float,  # rad/s
    pause: float,  # seconds
    abort_on_collision: bool,
    shoulder3_mirror: bool,
    present: set[Joint] = FULL_JOINT_SET,
    run_left: bool = True,
    run_right: bool = True,
    sweep_gripper: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep the selected joints through their ranges on the selected arms.

    Only joints in ``present`` are moved; every other joint stays at its
    current value. ``run_left`` / ``run_right`` gate which arm each mirrored
    sweep drives. Pre-poses that reposition a helper joint (shoulder_1 before
    shoulder_3, elbow before the wrists) are skipped when that joint is absent.

    When ``sweep_gripper`` is set the gripper is cycled through its full
    open↔close range at the end (holding nothing). This is left off for a
    full-robot grasp run, where the gripper is instead clamped on the item.
    """
    arm_sel = "both" if run_left and run_right else ("left" if run_left else "right")

    async def step(
        joint: Joint,
        left_val: float,  # rad
        right_val: float,  # rad
        spd: float = speed,  # rad/s
    ) -> None:
        nonlocal left_q, right_q
        left_target = with_joint(left_q, joint, left_val) if run_left else left_q.copy()
        right_target = (
            with_joint(right_q, joint, right_val) if run_right else right_q.copy()
        )
        left_q, right_q = await sweep_to_target(
            robot,
            checker,
            left_q,
            right_q,
            left_target,
            right_target,
            spd,
            pause,
            abort_on_collision,
        )

    # Shoulder_1 limits are mirrored across arms: the right arm's range is the
    # negation of the left's, so sweep both arms simultaneously in mirror.
    if Joint.SHOULDER_1 in present:
        s1_left_low, s1_left_high = SHOULDER_1_LEFT_LIMITS
        print(
            f"  SHOULDER_1  ({arm_sel}, mirrored)  "
            f"[{round(s1_left_high, 3)}, {round(s1_left_low, 3)}] → 0"
        )
        for value in (s1_left_high, s1_left_low, 0.0):
            await step(Joint.SHOULDER_1, value, -value)

    if Joint.SHOULDER_2 in present:
        _, right_high = SHOULDER_2_RIGHT_LIMITS
        left_low, _ = SHOULDER_2_LEFT_LIMITS
        print(
            f"  SHOULDER_2  ({arm_sel})  → [{round(left_low, 3)}, {round(right_high, 3)}] → 0"
        )
        await step(Joint.SHOULDER_2, left_low, right_high)
        await step(Joint.SHOULDER_2, 0.0, 0.0)

    if Joint.SHOULDER_3 in present:
        # The forward pre-pose uses shoulder_1; skip it when shoulder_1 is
        # absent and sweep shoulder_3 in place instead.
        s3_prepose = Joint.SHOULDER_1 in present
        if s3_prepose:
            print(
                f"  SHOULDER_3 pre-pose: arms forward "
                f"{round(math.degrees(SHOULDER_PRE_POSE_ANGLE), 1)}°"
            )
            await step(
                Joint.SHOULDER_1,
                -SHOULDER_PRE_POSE_ANGLE,
                SHOULDER_PRE_POSE_ANGLE,
                pre_pose_speed,
            )
        low, high = LIMITS[Joint.SHOULDER_3]
        print(f"  SHOULDER_3  ({arm_sel} fwd)  [{round(low, 3)}, {round(high, 3)}] → 0")
        shoulder3_left_low = low if shoulder3_mirror else -low
        shoulder3_left_high = high if shoulder3_mirror else -high
        await step(Joint.SHOULDER_3, shoulder3_left_low, -low)
        await step(Joint.SHOULDER_3, shoulder3_left_high, -high)
        await step(Joint.SHOULDER_3, 0.0, 0.0)
        if s3_prepose:
            await step(Joint.SHOULDER_1, 0.0, 0.0, pre_pose_speed)

    if Joint.ELBOW in present:
        _, elbow_left_high = ELBOW_LEFT_LIMITS
        elbow_right_low, _ = ELBOW_RIGHT_LIMITS
        print(
            f"  ELBOW       ({arm_sel})  "
            f"[{round(elbow_right_low, 3)}, {round(elbow_left_high, 3)}] → 0"
        )
        await step(Joint.ELBOW, elbow_left_high, elbow_right_low)
        await step(Joint.ELBOW, 0.0, 0.0)

    # The wrists are swept with the elbows bent to 90° so they clear the body;
    # skip that pre-pose (and its return) when the elbow is off the bus.
    wrist_joints = [
        j for j in (Joint.WRIST_1, Joint.WRIST_2, Joint.WRIST_3) if j in present
    ]
    elbow_prepose = Joint.ELBOW in present and bool(wrist_joints)
    if elbow_prepose:
        print("  Pre-pose: elbows at 90°")
        await step(Joint.ELBOW, +WRIST_TEST_ELBOW_ANGLE, -WRIST_TEST_ELBOW_ANGLE)

    for wrist in wrist_joints:
        low, high = LIMITS[wrist]
        label = f"{wrist.value.replace('_', ' ').upper():<11} ({arm_sel})"
        left_q, right_q = await sweep_joint_range(
            robot,
            checker,
            left_q,
            right_q,
            wrist,
            arm_sel,
            [high, low],
            label,
            speed,
            pause,
            abort_on_collision,
        )

    if elbow_prepose:
        await step(Joint.ELBOW, 0.0, 0.0)

    # Gripper full range of motion (normalized 1 = open, 0 = closed). Only when
    # not grasping an item; the position-force controller caps speed and torque,
    # so closing on nothing simply drives gently to the closed stop.
    if sweep_gripper and Joint.GRIPPER in present:
        print(f"  GRIPPER     ({arm_sel})  open ↔ close")
        await step(Joint.GRIPPER, 0.0, 0.0)
        await step(Joint.GRIPPER, 1.0, 1.0)

    return left_q, right_q


async def return_home(robot: Axol | HardwareController) -> None:
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


async def run_axol(
    present: set[Joint] = FULL_JOINT_SET,
    no_left: bool = False,
    no_right: bool = False,
) -> None:
    checker = CollisionChecker()

    run_left = not no_left
    run_right = not no_right
    has_gripper = Joint.GRIPPER in present
    # The grasp-an-item clamp (hold with force, soak while holding) only runs on
    # the full robot. Any subset that includes the gripper instead sweeps it
    # through its full open↔close range like any other joint (holding nothing).
    grasp = present == set(Joint)
    sweep_gripper = has_gripper and not grasp

    print("=== ROM TEST — PHYSICAL ROBOT ===")
    print("Make sure the area is clear.")
    arms_desc = (
        "both arms"
        if run_left and run_right
        else ("left arm" if run_left else "right arm")
    )
    joints_desc = (
        "all joints"
        if present == set(Joint)
        else ", ".join(j.value for j in Joint if j in present)
    )
    print(f"Running {arms_desc}  |  joints: {joints_desc}\n")

    config = AxolConfig(left_stiffness=1.0, right_stiffness=1.0)
    if grasp:
        # Raised grasp force is only needed to hold the item; a bare gripper
        # ROM sweep keeps the default (gentle) torque cap.
        config.left.gripper.torque_limit = GRIPPER_TORQUE_LIMIT
        config.right.gripper.torque_limit = GRIPPER_TORQUE_LIMIT
    axol = Axol(
        config=config,
        left_channel=None if no_left else CAN_LEFT,
        right_channel=None if no_right else CAN_RIGHT,
    )
    robot = HardwareController(axol, present)
    await robot.enable()
    print("Motors enabled.")
    await asyncio.sleep(2.0)

    closed_left_q: np.ndarray | None = None
    closed_right_q: np.ndarray | None = None

    def home_with_grippers_closed() -> tuple[np.ndarray, np.ndarray]:
        assert closed_left_q is not None and closed_right_q is not None
        return closed_left_q.copy(), closed_right_q.copy()

    # Only meaningful on a full-robot grasp run, where the gripper is holding an
    # item: on a normal soak completion or Ctrl-C the motors are left enabled so
    # the grasp is kept and ``rom.disable`` can release it later. Any subset run
    # holds nothing, so the motors are always disabled in the finally block.
    keep_enabled = False

    try:
        home = home_pose()
        gripper_i = JOINT_INDEX[Joint.GRIPPER]

        # Ease the arms from wherever they actually are to home before anything
        # else, with the grippers open (1.0). The first motion_control would
        # otherwise command home as a single stiff (s=1) impedance setpoint and
        # snap the arms there; sweep_unchecked ramps them in with a smoothstep
        # trajectory instead.
        cur_left, cur_right = await robot.get_positions()
        ready_left = home.copy()
        ready_right = home.copy()
        if has_gripper:
            ready_left[gripper_i] = 1.0
            ready_right[gripper_i] = 1.0
            print("Easing to home position (grippers open) ...")
        else:
            print("Easing to home position ...")
        await sweep_unchecked(
            robot, cur_left, cur_right, ready_left, ready_right, speed=AXOL_HOME_SPEED
        )
        left_q, right_q = ready_left, ready_right

        # Grasp the item only on a full-robot run; any subset (including one
        # that contains the gripper) skips the grasp and sweeps the selected
        # joints instead (see module docstring).
        if grasp:
            if run_right:
                await asyncio.to_thread(
                    input, "Press Enter to close the RIGHT gripper ..."
                )
                left_q, right_q = await move_grippers(
                    robot, left_q, right_q, left_q[gripper_i], 0.0, GRIPPER_SPEED
                )
            if run_left:
                await asyncio.to_thread(
                    input, "Press Enter to close the LEFT gripper ..."
                )
                left_q, right_q = await move_grippers(
                    robot, left_q, right_q, 0.0, right_q[gripper_i], GRIPPER_SPEED
                )
            print("Grippers closed.")

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
                robot,
                checker,
                left_q,
                right_q,
                speed=AXOL_SPEED,
                pre_pose_speed=AXOL_PRE_POSE_SPEED,
                pause=AXOL_WAYPOINT_PAUSE,
                abort_on_collision=True,
                shoulder3_mirror=False,
                present=present,
                run_left=run_left,
                run_right=run_right,
                sweep_gripper=sweep_gripper,
            )

            print(f"\nCycle {cycle} complete.")
            if time.monotonic() < deadline:
                print(f"Waiting {CYCLE_PAUSE}s ...")
                await asyncio.sleep(CYCLE_PAUSE)

        print(f"\n2-hour soak complete — {cycle} cycle(s) finished.")

        print("Returning to home position ...")
        home_left, home_right = home_with_grippers_closed()
        left_q, right_q = await sweep_to_target(
            robot,
            checker,
            left_q,
            right_q,
            home_left,
            home_right,
            AXOL_HOME_SPEED,
            AXOL_WAYPOINT_PAUSE,
            abort_on_collision=True,
        )

        # On a full-robot grasp run, leave the robot holding the item with the
        # motors enabled; the operator runs rom.disable to open the grippers and
        # retrieve it. Otherwise there is nothing to hold and we disable.
        keep_enabled = grasp

    except CollisionAbort as e:
        print(f"\n⚠  COLLISION ABORT: {e}")
        print("Returning to home position ...")
        safe_left, safe_right = home_with_grippers_closed()
        await sweep_unchecked(
            robot, e.left_q, e.right_q, safe_left, safe_right, speed=AXOL_SPEED
        )
        print("Home reached.")

    except (KeyboardInterrupt, asyncio.CancelledError):
        if grasp:
            print("\nInterrupted — returning home, keeping the item gripped ...")
        else:
            print("\nInterrupted — returning home ...")
        await return_home(robot)
        keep_enabled = grasp

    finally:
        if keep_enabled:
            await close_buses(axol)
            print(
                "\nMotors left enabled — robot is holding the item.\n"
                "Run `uv run -m almond_axol.diagnostics.rom.disable` to open the "
                "grippers and retrieve it."
            )
        else:
            await robot.disable()
            print("Motors disabled.")


def main() -> None:
    valid_joints = [j.value for j in Joint]
    parser = argparse.ArgumentParser(
        description="Range of motion test for the Axol robot."
    )
    parser.add_argument(
        "--joints",
        default=None,
        help="Comma-separated joints present on the bus (e.g. wrist_1,wrist_2,wrist_3). "
        f"Only these are enabled and swept. Default: all. One of: {', '.join(valid_joints)}.",
    )
    parser.add_argument("--no-left", action="store_true", help="Skip the left arm.")
    parser.add_argument("--no-right", action="store_true", help="Skip the right arm.")
    args = parser.parse_args()

    if args.no_left and args.no_right:
        parser.error("Cannot skip both arms.")

    present = parse_joints(args.joints)

    asyncio.run(run_axol(present=present, no_left=args.no_left, no_right=args.no_right))


if __name__ == "__main__":
    main()
