"""
rom.enable

Range of motion test for the Axol robot. Sweeps every joint through its full
range.

Enables the motors, eases to home, then prompts to close each gripper onto the
item and loops the sweep for two hours. When the soak finishes (or on Ctrl-C)
the robot returns home but keeps holding the item with the motors left enabled
— run ``almond_axol.diagnostics.rom.disable`` afterwards to open the grippers
and retrieve the item.

The arms are driven through the Rust realtime core (``Axol``), the same
control path as teleop: the core owns the CAN buses, renders the streamed
targets at 240 Hz, and runs the host damping against fresh feedback. A soak
therefore exercises exactly the controller the robot ships with.

Select a subset of joints and/or a single arm:
  --joints    Comma-separated joints to sweep (e.g. wrist_1,wrist_2,wrist_3).
              Every motor found on the bus is brought up and held — the
              realtime core enables all of them — but only these joints move;
              every other joint holds home. Default: all.
  --no-left / --no-right
              Skip an arm entirely. Only the remaining arm is opened, enabled,
              and swept. Cannot skip both.

Before bring-up each bus is probed (read-only status reads) for which motors
answer. Every selected joint must be there; a selected joint that is missing
fails the run by name. An *unselected* joint that does not answer is treated
as physically absent — a bench setup with only a wrist assembly on the bus
can run ``--joints wrist_2,wrist_3,gripper`` — and is left out of bring-up
(it is never enabled, and it holds nothing). The skipped joints are printed
at startup; make sure they really are not attached, because a present but
unpowered motor is indistinguishable from an absent one and would swing free
while its neighbours sweep. A full-robot run (no ``--joints``) still requires
every motor.

A partial arm is, by construction, not mounted on the robot (a wrist kit
clamped to a bench), so it is driven as plain PD: the soft hand-guidable
gains (``stiffness 0``) with every model-based feedforward off — no gravity
(the model assumes the robot's mounting orientation), no friction, inertia
or host-damping terms (all tuned against the full arm's dynamics). The
production gains on a rigidly clamped wrist assembly vibrate heavily; the
point of a bench sweep is only to prove the joint actuates through its
range, which the soft PD does without exciting anything. See
:func:`bench_config`.

The grasp-an-item clamp (hold with force, then soak while holding) only runs
when every joint is selected. Any subset run drops the grasp step and simply
loops the range-of-motion sweeps for the selected joints; if the gripper is one
of them it is cycled through its full open↔close range like any other joint
(holding nothing, at the default gentle torque).

``--web-prompts`` makes each hands-on gripper step emit a ``[prompt] ...``
marker on stdout and then block on stdin, so the web dashboard can turn the
step into a Continue button (it writes a line to the process when the operator
clicks). Without it the steps use ordinary ``input()`` tty prompts. Joint
positions and torques are captured to a CSV under
``~/.almond/diagnostics/captures`` for the diagnostics dashboard;
``--no-capture`` disables that.

Run via the CLI or directly:
    axol diag.rom-enable
    uv run -m almond_axol.diagnostics.rom.enable
    uv run -m almond_axol.diagnostics.rom.enable --no-right
    uv run -m almond_axol.diagnostics.rom.enable --joints wrist_1,wrist_2,wrist_3
"""

import argparse
import asyncio
import math
import sys
import time
from collections.abc import Iterable
from dataclasses import replace

import numpy as np

from ...constants import (
    ARM_JOINTS,
    CAN_LEFT,
    CAN_MANTIS_LEFT,
    CAN_MANTIS_RIGHT,
    CAN_RIGHT,
    Joint,
)
from ...motor import CanBus, Motor, MotorError
from ...robot.axol import (
    ELBOW_LEFT_LIMITS,
    ELBOW_RIGHT_LIMITS,
    LIMITS,
    SHOULDER_1_LEFT_LIMITS,
    SHOULDER_2_LEFT_LIMITS,
    SHOULDER_2_RIGHT_LIMITS,
    AxolHardware,
)
from ...robot.config import ArmConfig, AxolConfig, FrictionParams
from ...robot.mantis import Mantis
from ...rt import Axol, RtMantis
from ..telemetry_log import TelemetryCsvLogger

CONTROL_RATE_HZ = 100.0  # Hz

# Marker prefix a --web-prompts step prints before blocking on stdin. The web
# dashboard watches the log for this and turns it into a Continue button.
PROMPT_MARKER = "[prompt]"

# Sweep speed. 1.0 rad/s excites the shoulders' ~3 Hz structural mode: the
# smoothstep acceleration of a 90° sweep at that speed has significant energy
# at the resonance, producing a visible ~1.3 Nm RMS wobble during motion and
# an arrival ring at every waypoint. Hardware A/B (mirrored shoulder_1 sweep,
# s=1.0) measured 0.6 rad/s cutting the in-motion residual ~35% and halving
# the arrival ring; host damping is already at its stability ceiling, so a
# gentler trajectory is the only remaining lever.
AXOL_SPEED = 0.6  # rad/s
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


class CoreLimp(RuntimeError):
    """The realtime core dropped the arms to gravity comp mid-run."""


def parse_joints(spec: str | None) -> set[Joint]:
    """Parse a comma-separated joint spec into a set of joints to sweep.

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


# Per-motor deadline for the pre-flight presence probe. A live motor answers a
# status read within a few milliseconds; the Damiao driver's own retries span
# ~1 s, so this bounds an absent motor to about the same.
PROBE_TIMEOUT = 1.5  # seconds


async def probe_bus_joints(channel: str, joints: Iterable[Joint]) -> set[Joint]:
    """Return the subset of ``joints`` whose motor answers on ``channel``.

    Read-only: one status read per motor (the same request the dashboard's
    idle survey uses), so it is safe on a robot in any state — including
    one still holding from a previous session. The bus is opened and
    closed around the reads so the realtime core can take the interface
    over afterwards.
    """
    bus = CanBus(channel)
    await bus.start()
    try:
        motors = {joint: Motor(bus, joint) for joint in joints}

        async def answers(joint: Joint, motor: Motor) -> Joint | None:
            try:
                await asyncio.wait_for(motor.get_error_code(), PROBE_TIMEOUT)
            except (MotorError, asyncio.TimeoutError, OSError):
                return None
            return joint

        found = await asyncio.gather(*(answers(j, m) for j, m in motors.items()))
    finally:
        await bus.close()
    return {joint for joint in found if joint is not None}


def resolve_bus_joints(
    selected: set[Joint],
    on_bus: set[Joint],
    candidates: set[Joint],
    side: str,
) -> set[Joint]:
    """Decide which of one arm's motors a run brings up, given the probe.

    ``candidates`` are the joints the arm could carry (all eight, or seven
    on the gripperless SKU); ``on_bus`` is the subset that answered. Every
    selected joint must have answered. Unselected joints that did not answer
    are dropped from the arm (a partial bench arm), and every joint that
    did answer is kept so an attached-but-unselected joint is still brought
    up and held at home rather than left unpowered. A full-robot selection
    therefore requires the whole arm.

    Raises:
        SystemExit: Naming the selected joints missing from the bus.
    """
    missing = [j for j in Joint if j in selected & candidates and j not in on_bus]
    if missing:
        names = ", ".join(j.value for j in missing)
        raise SystemExit(
            f"Cannot start: the {side} arm's {names} did not answer on the bus. "
            "Every selected joint must be powered and connected — check power "
            "and CAN wiring, or deselect the joints that are not attached."
        )
    if not on_bus:
        raise SystemExit(f"Cannot start: no motors answered on the {side} arm's bus.")
    return set(on_bus)


# Gains for a partial arm off the robot: the soft end of the stiffness
# slider — the hand-guidable gains (wrist_2 25/1.5, wrist_3 25/0.9, see
# ``_SOFT_GAINS`` in robot/config.py), damping-ratio-consistent with the tuned
# set. The production gains (wrist_2 130/3.5) vibrate heavily on a wrist kit
# clamped to a bench: firmware kd on the Damiao wrists already sits at the
# edge of a unit-dependent buzz on the robot (kd=5 buzzes at 110 Hz), and a
# rigid mount with none of the arm's compliance behind the stator moves that
# edge down. A bench sweep only has to show the joint tracks through its
# range; the soft gains do that without exciting anything.
BENCH_STIFFNESS = 0.0
_NO_FRICTION = FrictionParams(fc=0.0, k=0.0, fv=0.0, fo=0.0)


def bench_config(config: AxolConfig) -> AxolConfig:
    """Plain-PD gains for a partial arm that is not mounted on the robot.

    Every arm joint gets :data:`BENCH_STIFFNESS` (the soft, hand-guidable
    ``kp`` / ``kd``) and loses every model-based feedforward, all of which
    assume the full arm on the robot: gravity (``mass = 0`` — the model
    computes it for the robot's mounting orientation, which a bench kit is
    not in), friction (zeroed; the tanh compensator was fit on the full
    joint chain), inertia (``j_eff = 0``) and host damping (``kd_host = 0``,
    whose pose schedule reads the same inertia model). What remains is the
    core's trapezoid tracker driving the firmware PD — enough to sweep a
    joint through its range and see that it actuates. The gripper is
    unchanged (it is position/force controlled, not impedance).
    """

    def plain_pd(arm: ArmConfig) -> ArmConfig:
        return replace(
            arm,
            **{
                j.value: replace(
                    getattr(arm, j.value),
                    friction=_NO_FRICTION,
                    kd_host=0.0,
                    j_eff=0.0,
                    mass=0.0,
                )
                for j in ARM_JOINTS
            },
        )

    return replace(
        config,
        left=plain_pd(config.left),
        right=plain_pd(config.right),
        left_stiffness=BENCH_STIFFNESS,
        right_stiffness=BENCH_STIFFNESS,
    )


def is_bench_run(
    arm_joints: dict[str, set[Joint] | None], candidates: set[Joint]
) -> bool:
    """Whether any arm in the run is partial — *arm joints* missing from its bus.

    A partial arm cannot be on the robot, so it gets :func:`bench_config`.
    The rule is the bus probe, not the ``--joints`` selection: a joint
    subset swept on a fully populated arm is still the robot, and its held
    joints need the production gains and gravity feedforward.

    Only the seven arm joints count. The gripper says nothing about the
    mounting — a mounted arm whose gripper is unpowered, missing, or simply
    not fitted must keep the production gains: soft PD with no gravity
    feedforward would let its held shoulders sag.
    """
    arm_candidates = candidates & set(ARM_JOINTS)
    return any(
        joints is not None and (joints & set(ARM_JOINTS)) != arm_candidates
        for joints in arm_joints.values()
    )


def home_pose() -> np.ndarray:
    return np.zeros(NUM_JOINTS, dtype=np.float32)


async def _stream(
    robot: Axol | RtMantis,
    left_q: np.ndarray,  # rad
    right_q: np.ndarray,  # rad
) -> None:
    """Ship one target pair to the core, refusing to keep "sweeping" limp arms.

    Once the core has gone limp (loss-of-trust fault: a silent motor)
    ``Axol.motion_control`` streams gravity comp instead of tracking, so the
    arms would hang weightless while this script kept announcing sweeps. Stop
    the run instead; the operator hand-guides the arms to rest.
    """
    limp = robot.limp
    if limp is not None:
        raise CoreLimp(limp)
    await robot.motion_control(left=left_q, right=right_q)


async def hold_pose(
    robot: Axol | RtMantis,
    left_q: np.ndarray,  # rad
    right_q: np.ndarray,  # rad
    seconds: float,
) -> None:
    """Hold a pose for ``seconds`` while keeping the command stream alive.

    The core's watchdog would hold the last target on its own (tracker
    converged, damping live), but streaming through the pause keeps every
    hold identical to the moving segments — same cadence, same gravity
    refresh — and surfaces a limp core immediately rather than at the next
    sweep.
    """
    dt = 1.0 / CONTROL_RATE_HZ
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        await _stream(robot, left_q, right_q)
        await asyncio.sleep(dt)


async def _stream_hold_forever(
    robot: Axol | RtMantis,
    left_q: np.ndarray,  # rad
    right_q: np.ndarray,  # rad
) -> None:
    """Stream a static hold until cancelled (see :func:`hold_pose`)."""
    dt = 1.0 / CONTROL_RATE_HZ
    while True:
        await _stream(robot, left_q, right_q)
        await asyncio.sleep(dt)


async def move_grippers(
    robot: Axol | RtMantis,
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
        await _stream(robot, left, right)
        if progress >= 1.0:
            break
        await asyncio.sleep(dt)
    return left, right


async def sweep_to_target(
    robot: Axol | RtMantis,
    left_q: np.ndarray,  # rad
    right_q: np.ndarray,  # rad
    left_target: np.ndarray,  # rad
    right_target: np.ndarray,  # rad
    speed: float,  # rad/s
    pause: float,  # seconds
) -> tuple[np.ndarray, np.ndarray]:
    await sweep_unchecked(robot, left_q, right_q, left_target, right_target, speed)
    # Keep streaming through the waypoint pause (see hold_pose).
    await hold_pose(robot, left_target, right_target, pause)
    return left_target.copy(), right_target.copy()


async def sweep_unchecked(
    robot: Axol | RtMantis,
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
        await _stream(
            robot,
            (left_q * (1 - smooth) + left_target * smooth).astype(np.float32),
            (right_q * (1 - smooth) + right_target * smooth).astype(np.float32),
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
    robot: Axol | RtMantis,
    left_q: np.ndarray,  # rad
    right_q: np.ndarray,  # rad
    joint: Joint,
    arm: str,
    waypoints: list[float],  # rad
    label: str,
    speed: float,  # rad/s
    pause: float,  # seconds
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
            left_q,
            right_q,
            left_target,
            right_target,
            speed,
            pause,
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
        left_q,
        right_q,
        left_target,
        right_target,
        speed,
        pause,
    )
    return left_q, right_q


async def run_rom_cycle(
    robot: Axol | RtMantis,
    left_q: np.ndarray,  # rad
    right_q: np.ndarray,  # rad
    speed: float,  # rad/s
    pre_pose_speed: float,  # rad/s
    pause: float,  # seconds
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
    shoulder_3, elbow before the wrists) are skipped when that joint is not
    selected.

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
            left_q,
            right_q,
            left_target,
            right_target,
            spd,
            pause,
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
        # not selected and sweep shoulder_3 in place instead.
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
    # skip that pre-pose (and its return) when the elbow is not selected.
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
            left_q,
            right_q,
            wrist,
            arm_sel,
            [high, low],
            label,
            speed,
            pause,
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


async def return_home(robot: Axol | RtMantis) -> None:
    """Ease the arms back to home from their current pose, keeping the grippers shut.

    Used to bring the robot to a safe home position while it stays clamped on
    the item. The grippers are left exactly where they are (still grasping) and
    the motors stay enabled; ``rom.disable`` releases the item afterwards.
    """
    gripper_i = JOINT_INDEX[Joint.GRIPPER]
    cur_left, cur_right = await robot.get_positions()
    cur_left = home_pose() if cur_left is None else cur_left
    cur_right = home_pose() if cur_right is None else cur_right
    home_left = home_pose()
    home_right = home_pose()
    home_left[gripper_i] = cur_left[gripper_i]
    home_right[gripper_i] = cur_right[gripper_i]
    print("Returning home (still holding the item) ...")
    await sweep_unchecked(
        robot, cur_left, cur_right, home_left, home_right, speed=AXOL_HOME_SPEED
    )


async def _confirm(
    instruction: str,
    web_prompts: bool,
    robot: Axol | RtMantis,
    left_q: np.ndarray,  # rad
    right_q: np.ndarray,  # rad
) -> None:
    """Block until the operator confirms a hands-on step, streaming the hold.

    The operator is touching the robot during these steps (placing the item in
    a gripper), so the current pose keeps streaming while we wait (see
    hold_pose).

    ``--web-prompts``: emit a ``[prompt]`` marker the dashboard turns into a
    Continue button, then block until it writes a line to our stdin. Otherwise
    fall back to an ordinary tty ``input()`` prompt. A closed stdin (EOF, e.g.
    the run is being stopped) unblocks and returns so teardown can proceed.
    """
    hold = asyncio.ensure_future(_stream_hold_forever(robot, left_q, right_q))
    try:
        if web_prompts:
            print(f"{PROMPT_MARKER} {instruction}", flush=True)
            await asyncio.to_thread(sys.stdin.readline)
        else:
            await asyncio.to_thread(input, f"{instruction} Press Enter to continue ...")
    finally:
        hold.cancel()
        try:
            await hold
        except asyncio.CancelledError:
            pass


async def _positions(robot: Axol | RtMantis) -> tuple[np.ndarray, np.ndarray]:
    """Measured positions as (left, right); an absent arm reports home."""
    left, right = await robot.get_positions()
    return (
        home_pose() if left is None else left,
        home_pose() if right is None else right,
    )


async def run_axol(
    present: set[Joint] = FULL_JOINT_SET,
    no_left: bool = False,
    no_right: bool = False,
    web_prompts: bool = False,
    capture: bool = True,
    left_channel: str = CAN_LEFT,
    right_channel: str = CAN_RIGHT,
    target: str = "axol",
) -> None:
    run_left = not no_left
    run_right = not no_right
    has_gripper = Joint.GRIPPER in present
    # The grasp-an-item clamp (hold with force, soak while holding) only runs
    # when every joint is selected. Any subset that includes the gripper
    # instead sweeps it through its full open↔close range like any other joint
    # (holding nothing).
    grasp = present == set(Joint)
    sweep_gripper = has_gripper and not grasp

    print(f"=== ROM TEST — {target.upper()} ===")
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
    # Production control path: the Rust core owns the buses and runs the
    # 240 Hz loop; this script only streams targets (see the module docstring).
    robot: Axol | RtMantis
    axol: AxolHardware | Mantis
    if target == "mantis":
        # Grippers-only core on the Mantis buses; the seven arm joints per
        # side are virtual (they latch the streamed targets), so the sweep
        # helpers run unchanged and only the gripper physically moves.
        axol = Mantis(
            config=config,
            left_channel=None if no_left else left_channel,
            right_channel=None if no_right else right_channel,
        )
        robot = RtMantis(axol)
    else:
        # Bring up exactly the motors that are on each bus (see the module
        # docstring): every selected joint must answer the probe; an
        # unselected one that does not is a joint this bench arm simply
        # does not have.
        candidates = {j for j in Joint if config.has_gripper or j != Joint.GRIPPER}
        arm_joints: dict[str, set[Joint] | None] = {"left": None, "right": None}
        for side, channel, run in (
            ("left", left_channel, run_left),
            ("right", right_channel, run_right),
        ):
            if not run:
                continue
            on_bus = await probe_bus_joints(channel, candidates)
            arm_joints[side] = resolve_bus_joints(present, on_bus, candidates, side)
            skipped = [j.value for j in Joint if j in candidates and j not in on_bus]
            if skipped:
                print(
                    f"{side.capitalize()} arm: {', '.join(skipped)} not on the bus — "
                    "treating as absent (never enabled). Make sure these joints "
                    "really are not attached."
                )
        # A partial arm is off the robot (a bench kit): the production gains
        # and model feedforwards do not apply there — see bench_config.
        if is_bench_run(arm_joints, candidates):
            config = bench_config(config)
            print(
                "Partial arm — bench gains: soft PD "
                f"(stiffness {BENCH_STIFFNESS:g}), no gravity/friction/inertia/"
                "host-damping feedforward."
            )
        robot = Axol(
            config=config,
            left_channel=None if no_left else left_channel,
            right_channel=None if no_right else right_channel,
            left_joints=arm_joints["left"],
            right_joints=arm_joints["right"],
        )
        axol = robot.hardware
    await robot.enable()
    print("Motors enabled (realtime core armed).")

    # The logger samples the motor caches, which the core's telemetry fills.
    logger = TelemetryCsvLogger(axol, "rom") if capture else None
    if logger is not None:
        logger.start()

    # Settle for 2 s at the measured pose (Axol.enable already primed the
    # core with one gravity-compensated hold there).
    settle_left, settle_right = await _positions(robot)
    await hold_pose(robot, settle_left, settle_right, 2.0)

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
        gripper_i = JOINT_INDEX[Joint.GRIPPER]

        # Ease the arms from wherever they actually are to home before anything
        # else, with the grippers open (1.0). The first motion_control would
        # otherwise command home as a single stiff (s=1) impedance setpoint and
        # snap the arms there; sweep_unchecked ramps them in with a smoothstep
        # trajectory instead.
        home = home_pose()
        cur_left, cur_right = await _positions(robot)
        ready_left = home.copy()
        ready_right = home.copy()
        ready_left[gripper_i] = 1.0
        ready_right[gripper_i] = 1.0
        print("Easing to home position (grippers open) ...")
        await sweep_unchecked(
            robot, cur_left, cur_right, ready_left, ready_right, speed=AXOL_HOME_SPEED
        )
        left_q, right_q = ready_left, ready_right

        # Grasp the item only on a full-robot run; any subset (including one
        # that contains the gripper) skips the grasp and sweeps the selected
        # joints instead (see module docstring).
        if grasp:
            if run_right:
                await _confirm(
                    "Position the item in the RIGHT gripper, then close it.",
                    web_prompts,
                    robot,
                    left_q,
                    right_q,
                )
                left_q, right_q = await move_grippers(
                    robot, left_q, right_q, left_q[gripper_i], 0.0, GRIPPER_SPEED
                )
            if run_left:
                await _confirm(
                    "Position the item in the LEFT gripper, then close it.",
                    web_prompts,
                    robot,
                    left_q,
                    right_q,
                )
                left_q, right_q = await move_grippers(
                    robot, left_q, right_q, 0.0, right_q[gripper_i], GRIPPER_SPEED
                )
            print("Grippers closed.")

        closed_left_q = left_q.copy()
        closed_right_q = right_q.copy()

        print("Starting ROM test in 5 s ...")
        await hold_pose(robot, left_q, right_q, 5.0)

        deadline = time.monotonic() + SOAK_DURATION
        cycle = 0

        while time.monotonic() < deadline:
            cycle += 1
            remaining = deadline - time.monotonic()
            print(f"\n--- Cycle {cycle}  ({remaining / 60:.1f} min remaining) ---")

            left_q, right_q = home_with_grippers_closed()

            left_q, right_q = await run_rom_cycle(
                robot,
                left_q,
                right_q,
                speed=AXOL_SPEED,
                pre_pose_speed=AXOL_PRE_POSE_SPEED,
                pause=AXOL_WAYPOINT_PAUSE,
                shoulder3_mirror=False,
                present=present,
                run_left=run_left,
                run_right=run_right,
                sweep_gripper=sweep_gripper,
            )

            print(f"\nCycle {cycle} complete.")
            if time.monotonic() < deadline:
                print(f"Waiting {CYCLE_PAUSE}s ...")
                await hold_pose(robot, left_q, right_q, CYCLE_PAUSE)

        print(f"\n2-hour soak complete — {cycle} cycle(s) finished.")

        print("Returning to home position ...")
        home_left, home_right = home_with_grippers_closed()
        left_q, right_q = await sweep_to_target(
            robot,
            left_q,
            right_q,
            home_left,
            home_right,
            AXOL_HOME_SPEED,
            AXOL_WAYPOINT_PAUSE,
        )

        # On a full-robot grasp run, leave the robot holding the item with the
        # motors enabled; the operator runs rom.disable to open the grippers and
        # retrieve it. Otherwise there is nothing to hold and we disable.
        keep_enabled = grasp

    except (KeyboardInterrupt, asyncio.CancelledError):
        if robot.limp is not None:
            print(f"\nInterrupted — core is limp ({robot.limp}); not moving.")
        else:
            if grasp:
                print("\nInterrupted — returning home, keeping the item gripped ...")
            else:
                print("\nInterrupted — returning home ...")
            await return_home(robot)
            keep_enabled = grasp

    except CoreLimp as exc:
        print(
            f"\nRealtime core went limp ({exc}) — the arms are in gravity comp "
            "and will not track. Hand-guide them to rest; the run is over."
        )

    finally:
        if logger is not None:
            await logger.stop()
        if robot.limp is not None:
            # Leaves the arms limp (kp = 0, gravity feedforward) — never a
            # torque-off, and never a hold the operator can't move.
            await robot.disable()
            print("Arms left limp (gravity comp) — hand-guide them to rest.")
        elif keep_enabled:
            await robot.detach()
            print(
                "\nMotors left enabled — robot is holding the item.\n"
                "Run `uv run -m almond_axol.diagnostics.rom.disable` to open the "
                "grippers and retrieve it."
            )
        else:
            await robot.disable()
            print("Motors disabled.")


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    valid_joints = [j.value for j in Joint]
    parser.add_argument(
        "--joints",
        default=None,
        help="Comma-separated joints to sweep (e.g. wrist_1,wrist_2,wrist_3). "
        "Every motor found on the bus is brought up and held; only these "
        "move. Selected joints must be present; unselected joints that do "
        "not answer are treated as absent (partial bench arm). Default: all. "
        f"One of: {', '.join(valid_joints)}.",
    )
    parser.add_argument("--no-left", action="store_true", help="Skip the left arm.")
    parser.add_argument("--no-right", action="store_true", help="Skip the right arm.")
    parser.add_argument(
        "--target",
        choices=["axol", "mantis"],
        default="axol",
        help="Test the Axol arms or Mantis grippers (default: %(default)s).",
    )
    parser.add_argument(
        "--left-channel",
        default=None,
        metavar="IFACE",
        help="SocketCAN interface for the left arm, for setups without the "
        "selected hardware's default bus.",
    )
    parser.add_argument(
        "--right-channel",
        default=None,
        metavar="IFACE",
        help="SocketCAN interface for the right arm, for setups without the "
        "selected hardware's default bus.",
    )
    parser.add_argument(
        "--web-prompts",
        action="store_true",
        help="Emit '[prompt] ...' markers and block on stdin for hands-on "
        "gripper steps, so the web dashboard can drive them with a Continue "
        "button (set automatically by the dashboard).",
    )
    parser.add_argument(
        "--no-capture",
        action="store_true",
        help="Skip writing the telemetry CSV capture for the dashboard.",
    )


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``diag.rom-enable`` subcommand."""
    p = subparsers.add_parser(
        "diag.rom-enable",
        help="Range-of-motion soak test: sweep every joint for two hours.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    _add_arguments(p)
    p.set_defaults(func=run_cli)


def run_cli(args: argparse.Namespace) -> None:
    """Run the ROM soak from parsed arguments."""
    if args.no_left and args.no_right:
        raise SystemExit("Cannot skip both arms.")
    present = parse_joints(args.joints)
    if args.target == "mantis":
        if args.joints and present != {Joint.GRIPPER}:
            raise SystemExit("Mantis ROM supports only --joints gripper.")
        present = {Joint.GRIPPER}
    defaults = (
        (CAN_MANTIS_LEFT, CAN_MANTIS_RIGHT)
        if args.target == "mantis"
        else (CAN_LEFT, CAN_RIGHT)
    )
    asyncio.run(
        run_axol(
            present=present,
            no_left=args.no_left,
            no_right=args.no_right,
            web_prompts=args.web_prompts,
            capture=not args.no_capture,
            left_channel=args.left_channel or defaults[0],
            right_channel=args.right_channel or defaults[1],
            target=args.target,
        )
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Range of motion test for the Axol robot."
    )
    _add_arguments(parser)
    run_cli(parser.parse_args(argv))


if __name__ == "__main__":
    main()
