"""Hardware control classes for the Almond Axol dual-arm robot.

Provides :class:`AxolArm` (single-arm CAN bus controller) and :class:`Axol`
(dual-arm context manager that opens both buses and constructs all 16 motor drivers).
"""

from __future__ import annotations

import asyncio
import logging
import math

import numpy as np

from ..motor import (
    CanBus,
    ControlMode,
    Joint,
    Motor,
    MotorError,
    MotorGains,
    MotorStatus,
)
from ..utils.shared import ARM_JOINTS, CAN_LEFT, CAN_RIGHT
from .base import RobotBase
from .config import AxolConfig
from .control import Differentiator, compute_friction
from .gravity import GravityCompensator

_logger = logging.getLogger(__name__)

# Per-joint position limits (rad).  shoulder_1, shoulder_2, and elbow are mirrored across arms.
SHOULDER_1_LEFT_LIMITS = (math.radians(-90), math.radians(180))
SHOULDER_1_RIGHT_LIMITS = (math.radians(-180), math.radians(90))
SHOULDER_2_LEFT_LIMITS = (math.radians(-180), math.radians(20))
SHOULDER_2_RIGHT_LIMITS = (math.radians(-20), math.radians(180))
ELBOW_LEFT_LIMITS = (math.radians(0), math.radians(150))
ELBOW_RIGHT_LIMITS = (math.radians(-150), math.radians(0))

LIMITS: dict[Joint, tuple[float, float]] = {
    Joint.SHOULDER_3: (math.radians(-135), math.radians(135)),
    Joint.WRIST_1: (math.radians(-135), math.radians(135)),
    Joint.WRIST_2: (math.radians(-90), math.radians(90)),
    Joint.WRIST_3: (math.radians(-90), math.radians(90)),
    # Gripper absent: open position varies per unit, found at runtime by _calibrate_gripper().
}

# Fixed open-to-close travel of the gripper (rad).
GRIPPER_TRAVEL = math.radians(290)

# Gripper open-position calibration parameters.
_GRIPPER_TORQUE_THRESHOLD = 0.5  # Nm
_GRIPPER_CALIB_STEP = 0.005  # rad per step
_GRIPPER_CALIB_SETTLE = 0.001  # s per step
_GRIPPER_CALIB_MAX_STEPS = math.ceil(GRIPPER_TRAVEL / _GRIPPER_CALIB_STEP)

# Impedance gains used only during gripper open-stop calibration.
_GRIPPER_CALIB_KP = 50.0
_GRIPPER_CALIB_KD = 1.0


def arm_limits(joint: Joint, is_left: bool) -> tuple[float, float]:
    """Return (min, max) position limits for a joint on the given arm.

    Arm joints are in radians.  The gripper returns the normalised API range
    (0.0, 1.0) since gripper positions are always exposed as [0 = closed,
    1 = open] — the raw motor limits vary per unit and are calibrated at
    runtime by AxolArm._calibrate_gripper().
    """
    if joint == Joint.SHOULDER_1:
        return SHOULDER_1_LEFT_LIMITS if is_left else SHOULDER_1_RIGHT_LIMITS
    if joint == Joint.SHOULDER_2:
        return SHOULDER_2_LEFT_LIMITS if is_left else SHOULDER_2_RIGHT_LIMITS
    if joint == Joint.ELBOW:
        return ELBOW_LEFT_LIMITS if is_left else ELBOW_RIGHT_LIMITS
    if joint == Joint.GRIPPER:
        return (0.0, 1.0)
    return LIMITS.get(joint, (-math.inf, math.inf))


# Joints whose joint frame is kinematically mirrored across arms but whose
# limits are symmetric (equidistant from 0).  For these the calibration end
# stop on the right arm is the upper bound rather than the lower (the
# physical end stop closer to rest is at +X on the right arm and -X on the
# left).  Joints with already-asymmetric per-arm limits (s1, s2, elbow) are
# disambiguated by their range and don't need to be listed here.
_MIRRORED_SYMMETRIC_JOINTS: frozenset[Joint] = frozenset({Joint.WRIST_2, Joint.WRIST_3})


def closer_end_stop(joint: Joint, is_left: bool) -> tuple[float, int]:
    """Return ``(target_rad, expected_motion_sign)`` for the joint's calibration end stop.

    Picks the limit with the smallest absolute value, except when that limit
    is exactly 0 (it coincides with the rest position, so direction detection
    is unreliable) — in that case the *other* limit is used instead.  When
    both limits are equidistant the lower bound wins, except for joints in
    :data:`_MIRRORED_SYMMETRIC_JOINTS` on the right arm, which default to
    the upper bound to follow the kinematic mirror.

    ``expected_motion_sign`` is the sign of motion when the user starts inside
    the joint range and moves toward the chosen end stop:

    - target == lower bound  → range extends positively → motion is negative
    - target == upper bound  → range extends negatively → motion is positive

    Used by the guided ``set-zero-pos`` flow and by :class:`AxolArm` to derive
    the per-joint offset between motor frame (zero at end stop) and joint
    frame (zero at rest).  Not defined for ``Joint.GRIPPER``.
    """
    if joint == Joint.GRIPPER:
        raise ValueError("closer_end_stop is undefined for the gripper")
    lo, hi = arm_limits(joint, is_left)
    if lo == 0.0:
        return hi, +1
    if hi == 0.0:
        return lo, -1
    if abs(hi) < abs(lo):
        return hi, +1
    if abs(lo) < abs(hi):
        return lo, -1
    # Equidistant: default to lo, except mirrored joints on the right arm.
    if not is_left and joint in _MIRRORED_SYMMETRIC_JOINTS:
        return hi, +1
    return lo, -1


class AxolArm:
    """Controls one 7-DOF + gripper arm over a single CAN bus.

    Not instantiated directly — access via ``axol.left`` or ``axol.right``.
    """

    def __init__(
        self,
        bus: CanBus,
        config: AxolConfig,
        gravity_comp: GravityCompensator,
        is_left: bool = True,
    ) -> None:
        """Construct an AxolArm.

        Args:
            bus:          Shared CAN bus for this arm (one per physical interface).
            config:       Full dual-arm gains config; the correct side is selected via ``is_left``.
            gravity_comp: Shared MuJoCo-based gravity compensator (one per Axol).
            is_left:      ``True`` for the left arm, ``False`` for the right.
        """
        self._config = config
        self._arm_config = config.left if is_left else config.right
        self._gravity_comp = gravity_comp
        self._is_left = is_left
        self.motors: dict[Joint, Motor] = {joint: Motor(bus, joint) for joint in Joint}
        # q_des → v_des → a_des (commanded), and q_meas → v_meas. v_des feeds
        # the impedance-control velocity FF and the friction model; a_des
        # feeds inertia FF (``j_eff``); v_meas feeds software damping
        # (``kd_soft``) — all in :class:`JointConfig`.
        self._vel_diff = Differentiator(n=len(list(Joint)))
        self._accel_diff = Differentiator(n=len(list(Joint)))
        self._meas_vel_diff = Differentiator(n=len(list(Joint)))
        self._last_q_commanded: np.ndarray | None = None
        self._gc_hold_q: np.ndarray | None = None
        self._gc_hold_free: frozenset[Joint] | None = None

        # Clipping arrays.  Arm joints are in joint frame (0 = rest position,
        # matching the URDF and ``arm_limits``); gripper entries are in raw
        # motor radians (``arm_limits`` returns normalised [0, 1] for the
        # gripper, so the gripper bounds are seeded with raw defaults here
        # and ``_calibrate_gripper()`` overwrites them on enable).
        # Pre-calibration gripper defaults assume zero is closed — do not rely
        # on for actual motion.
        joints = list(Joint)
        self._gripper_i: int = joints.index(Joint.GRIPPER)
        self._limits_lo = np.array(
            [
                -GRIPPER_TRAVEL if j == Joint.GRIPPER else arm_limits(j, is_left)[0]
                for j in joints
            ],
            dtype=float,
        )
        self._limits_hi = np.array(
            [0.0 if j == Joint.GRIPPER else arm_limits(j, is_left)[1] for j in joints],
            dtype=float,
        )

        # Per-joint offset between motor frame and joint frame:
        #   joint_angle (rad) = motor_angle (rad) + offset
        # After end-stop calibration the motor's encoder zero coincides with
        # the closer end stop, so motor 0 → joint angle ``target_rad``.  The
        # gripper offset is 0 because the gripper uses its own [0, 1]
        # normalisation and is calibrated against torque, not an end stop.
        self._joint_offsets = np.array(
            [
                0.0 if j == Joint.GRIPPER else closer_end_stop(j, is_left)[0]
                for j in joints
            ],
            dtype=float,
        )

    # ------------------------------------------------------------------ #
    # Polling                                                              #
    # ------------------------------------------------------------------ #

    async def start_telemetry(self, hz: float, *, torque: bool = False) -> None:
        """Start background telemetry polling on all motors at the given frequency.

        Args:
            hz:     Poll frequency in Hz.
            torque: If True, also fetch and cache torque each cycle.
        """
        await asyncio.gather(
            *[m.start_telemetry(hz, torque=torque) for m in self.motors.values()]
        )

    async def stop_telemetry(self) -> None:
        """Stop the background telemetry polling loop on all motors."""
        await asyncio.gather(*[m.stop_telemetry() for m in self.motors.values()])

    @property
    def positions(self) -> np.ndarray:
        """Latest cached joint positions. Requires start_telemetry().

        Returns shape (8,) array in Joint enum order. Arm joints are in
        radians in the joint frame (0 = rest position); the gripper is
        normalized to [0, 1] (0.0 = closed, 1.0 = fully open), consistent
        with set_position_velocity and motion_control.
        """
        joints = list(Joint)
        values = [self.motors[j].position for j in joints]
        gripper_i = self._gripper_i
        values[gripper_i] = (values[gripper_i] - self._limits_hi[gripper_i]) / (
            self._limits_lo[gripper_i] - self._limits_hi[gripper_i]
        )
        arr = np.array(values, dtype=np.float32)
        return arr + self._joint_offsets.astype(np.float32)

    @property
    def torques(self) -> np.ndarray:
        """Latest cached joint torques (Nm / A). Requires start_telemetry().

        Returns shape (8,) array in Joint enum order.
        """
        return np.array([m.torque for m in self.motors.values()], dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Arm-wide commands                                                    #
    # ------------------------------------------------------------------ #

    async def _calibrate_gripper(self) -> None:
        """Find the gripper open position by stepping in the negative direction.

        Steps the gripper motor incrementally toward open until the torque
        magnitude drops to ``_GRIPPER_TORQUE_THRESHOLD`` (the open hard-stop).
        Updates ``_limits_lo[gripper_idx]`` (open) and ``_limits_hi[gripper_idx]``
        (close) which are used for normalization and clipping.

        Must be called with the gripper motor already enabled and in IMPEDANCE mode.
        """
        motor = self.motors[Joint.GRIPPER]
        gripper_i = self._gripper_i

        target = await motor.get_position()

        for _ in range(_GRIPPER_CALIB_MAX_STEPS):
            target -= _GRIPPER_CALIB_STEP
            await motor.set_impedance(
                target, 0.0, _GRIPPER_CALIB_KP, _GRIPPER_CALIB_KD, 0.0
            )
            await asyncio.sleep(_GRIPPER_CALIB_SETTLE)
            torque = await motor.get_torque()
            if abs(torque) >= _GRIPPER_TORQUE_THRESHOLD:
                open_pos = await motor.get_position()
                self._limits_lo[gripper_i] = open_pos
                self._limits_hi[gripper_i] = open_pos + GRIPPER_TRAVEL
                return

        open_pos = await motor.get_position()
        self._limits_lo[gripper_i] = open_pos
        self._limits_hi[gripper_i] = open_pos + GRIPPER_TRAVEL

    async def enable(self) -> None:
        """Enable all arm motors in IMPEDANCE mode and the gripper in POSITION_FORCE mode."""
        await asyncio.gather(*[m.enable() for m in self.motors.values()])
        await asyncio.gather(
            *[m.set_control_mode(ControlMode.IMPEDANCE) for m in self.motors.values()]
        )
        await self._calibrate_gripper()
        await self.motors[Joint.GRIPPER].set_control_mode(ControlMode.POSITION_FORCE)

    async def disable(self) -> None:
        """Disable all motors and engage brakes."""
        await asyncio.gather(*[m.disable() for m in self.motors.values()])

    async def clear_errors(self) -> None:
        """Clear latched error flags on all motors."""
        await asyncio.gather(*[m.clear_errors() for m in self.motors.values()])

    async def set_control_mode(self, mode: ControlMode) -> None:
        """Set the control mode on all motors.

        Args:
            mode: Desired control mode.
        """
        await asyncio.gather(*[m.set_control_mode(mode) for m in self.motors.values()])

    # ------------------------------------------------------------------ #
    # Getters                                                              #
    # ------------------------------------------------------------------ #

    async def get_positions(self) -> np.ndarray:
        """Return joint positions for every joint, fetched concurrently.

        Returns shape (8,) array in Joint enum order. Arm joints are in
        radians in the joint frame (0 = rest position); the gripper is
        normalized to [0, 1] (0.0 = closed, 1.0 = fully open), consistent
        with set_position_velocity and motion_control.
        """
        joints = list(Joint)
        values = list(
            await asyncio.gather(*[self.motors[j].get_position() for j in joints])
        )
        gripper_i = self._gripper_i
        values[gripper_i] = (values[gripper_i] - self._limits_hi[gripper_i]) / (
            self._limits_lo[gripper_i] - self._limits_hi[gripper_i]
        )
        arr = np.array(values, dtype=np.float32)
        return arr + self._joint_offsets.astype(np.float32)

    async def get_velocities(self) -> np.ndarray:
        """Return shaft velocity (rad/s) for every joint, fetched concurrently.

        Returns shape (8,) array in Joint enum order.
        """
        joints = list(Joint)
        values = await asyncio.gather(*[self.motors[j].get_velocity() for j in joints])
        return np.array(values, dtype=np.float32)

    async def get_torques(self) -> np.ndarray:
        """Return torque estimate for every joint, fetched concurrently.

        Damiao: Nm. MyActuator: phase current in A.
        Returns shape (8,) array in Joint enum order.
        """
        values = await asyncio.gather(*[m.get_torque() for m in self.motors.values()])
        return np.array(values, dtype=np.float32)

    async def get_temperatures(self) -> np.ndarray:
        """Return motor temperature (°C) for every joint, fetched concurrently.

        Returns shape (8,) array in Joint enum order.
        """
        values = await asyncio.gather(
            *[m.get_temperature() for m in self.motors.values()]
        )
        return np.array(values, dtype=np.float32)

    async def get_voltages(self) -> np.ndarray:
        """Return bus voltage (V) for every joint, fetched concurrently.

        Returns shape (8,) array in Joint enum order.
        """
        values = await asyncio.gather(*[m.get_voltage() for m in self.motors.values()])
        return np.array(values, dtype=np.float32)

    async def get_error_codes(self) -> list[MotorStatus]:
        """Return MotorStatus for every joint, fetched concurrently.

        Returns a list in Joint enum order.
        """
        joints = list(Joint)
        values = await asyncio.gather(
            *[self.motors[j].get_error_code() for j in joints]
        )
        return list(values)

    async def get_gains(self) -> list[MotorGains]:
        """Return PID gains for every joint, fetched concurrently.

        Returns a list in Joint enum order.
        """
        joints = list(Joint)
        values = await asyncio.gather(*[self.motors[j].get_gains() for j in joints])
        return list(values)

    # ------------------------------------------------------------------ #
    # Setters                                                              #
    # ------------------------------------------------------------------ #

    async def set_gains(self, gains: dict[Joint, MotorGains]) -> None:
        """Write PID gains to the specified motors.

        Changes are persisted to non-volatile memory.
        """
        await asyncio.gather(*[self.motors[j].set_gains(g) for j, g in gains.items()])

    async def set_zero_position(self, joints: list[Joint]) -> None:
        """Save the current shaft position as the encoder zero for the specified joints.

        The encoder zero is calibrated at the joint's mechanical end stop, not
        at the rest position; ``AxolArm`` applies a per-joint offset so the
        public API stays in joint frame (``0`` = rest).

        Args:
            joints: List of joints to zero.
        """
        await asyncio.gather(*[self.motors[j].set_zero_position() for j in joints])

    async def set_acceleration(self, accelerations: dict[Joint, float]) -> None:
        """Set the acceleration ramp per joint. Deceleration matches acceleration.

        Args:
            accelerations: Mapping of joint → acceleration ramp (rad/s²).
                           Joints not in the dict are unchanged.
        """
        await asyncio.gather(
            *[self.motors[j].set_acceleration(a) for j, a in accelerations.items()]
        )

    async def set_position_velocity(
        self, positions: np.ndarray, max_speed: float
    ) -> None:
        """Move joints to absolute positions using each motor's built-in controller.

        Positions are clipped to the arm's joint limits before being sent.
        Arm joints are in joint frame (0 = rest position); the gripper value
        is normalized: 0.0 = closed, 1.0 = fully open.

        Args:
            positions: Shape (8,) array of target positions (rad) in Joint enum order,
                       except gripper which is [0, 1].
            max_speed: Maximum speed for all joints (rad/s).
        """
        positions = positions.copy()
        gripper_i = self._gripper_i
        positions[gripper_i] = self._limits_hi[gripper_i] + positions[gripper_i] * (
            self._limits_lo[gripper_i] - self._limits_hi[gripper_i]
        )
        clipped = np.clip(positions, self._limits_lo, self._limits_hi)
        # Convert arm joints from joint frame to motor frame before sending.
        # Gripper offset is 0, so its raw motor value is unchanged.
        motor_targets = clipped - self._joint_offsets
        await asyncio.gather(
            *[
                self.motors[j].set_position_velocity(float(motor_targets[i]), max_speed)
                for i, j in enumerate(Joint)
            ]
        )

    async def set_velocity(self, velocities: np.ndarray) -> None:
        """Command target velocities using each motor's built-in speed controller.

        Args:
            velocities: Shape (8,) array of target velocities (rad/s) in Joint enum order.
        """
        await asyncio.gather(
            *[
                self.motors[j].set_velocity(float(velocities[i]))
                for i, j in enumerate(Joint)
            ]
        )

    async def motion_control(self, q: np.ndarray) -> None:
        """Send control commands to all joints concurrently.

        The 7 arm joints use IMPEDANCE control: gains (kp, kd) and friction
        parameters come from ArmConfig; feedforward torque is computed as
        gravity + friction compensation.

        The gripper uses POSITION_FORCE control: it tracks the target position
        at up to ``ArmConfig.gripper.max_speed`` (rad/s) with torque capped
        at ``ArmConfig.gripper.torque_limit`` (Nm).

        All positions are clipped to joint limits before being sent.

        Args:
            q: Shape (8,) array of desired positions in Joint enum order.
               Arm joints are in radians in the joint frame (0 = rest);
               gripper is normalized to [0, 1] (0.0 = closed, 1.0 = fully open).
        """
        q = q.copy()

        # Safety: reject commands with arm-joint deltas that exceed max_step_rad.
        # Deltas are frame-invariant (constant offset), so compute in joint frame.
        max_step = self._config.max_step_rad
        if self._last_q_commanded is not None and max_step < float("inf"):
            gripper_i = self._gripper_i
            arm_mask = np.ones(len(q), dtype=bool)
            arm_mask[gripper_i] = False
            deltas = np.abs(q[arm_mask] - self._last_q_commanded[arm_mask])
            worst_i = int(np.argmax(deltas))
            worst_delta = float(deltas[worst_i])
            if worst_delta > max_step:
                arm_joints = [j for j in Joint if j != Joint.GRIPPER]
                joint_name = (
                    arm_joints[worst_i].name
                    if worst_i < len(arm_joints)
                    else str(worst_i)
                )
                _logger.warning(
                    "motion_control: command dropped — joint %s delta %.3f rad exceeds "
                    "max_step_rad %.3f rad",
                    joint_name,
                    worst_delta,
                    max_step,
                )
                return

        gripper_i = self._gripper_i
        q[gripper_i] = self._limits_hi[gripper_i] + q[gripper_i] * (
            self._limits_lo[gripper_i] - self._limits_hi[gripper_i]
        )
        clipped = np.clip(q, self._limits_lo, self._limits_hi)

        # Velocity feedforward via differentiation of commanded positions (rad/s),
        # and acceleration feedforward via a second pass for inertia FF (rad/s²).
        # Velocities/accelerations are frame-invariant under a constant offset,
        # so we differentiate the joint-frame ``clipped`` array directly.
        velocities = self._vel_diff.differentiate(list(clipped))
        accelerations = self._accel_diff.differentiate(velocities)
        # v_meas drives software velocity damping. The position cache is
        # empty until the first set_impedance reply lands; fall back to v_des
        # so the ``kd_soft`` term collapses to 0 for those first cycles.
        try:
            v_meas = self._meas_vel_diff.differentiate(list(self.positions))
        except MotorError:
            v_meas = list(velocities)

        gripper_max_speed = self._arm_config.gripper.max_speed
        gripper_torque_limit = self._arm_config.gripper.torque_limit

        # Gravity feedforward (Nm) for the seven arm joints, computed from the
        # full URDF chain so child links contribute to each parent joint's load.
        # ``arm_q`` is in joint frame, which matches the URDF convention.
        arm_q = clipped[: len(ARM_JOINTS)].astype(np.float32)
        gravity = self._gravity_comp.gravity_arm(arm_q, is_left=self._is_left)

        # Convert arm joints to motor frame for the impedance command.  Gripper
        # offset is 0, so its raw motor value is unchanged.
        motor_targets = clipped - self._joint_offsets
        gripper_pos = float(motor_targets[gripper_i])

        def _mit_cmd(i: int, j: Joint):
            gains = getattr(self._arm_config, j.value)
            f = gains.friction
            t_ff = (
                float(gravity[i])
                + compute_friction(velocities[i], f.fc, f.k, f.fv, f.fo)
                + gains.j_eff * accelerations[i]
                + gains.kd_soft * (velocities[i] - v_meas[i])
            )
            return self.motors[j].set_impedance(
                float(motor_targets[i]),
                velocities[i],
                gains.kp,
                gains.kd,
                t_ff,
            )

        await asyncio.gather(
            *[_mit_cmd(i, j) for i, j in enumerate(Joint) if j != Joint.GRIPPER],
            self.motors[Joint.GRIPPER].set_position_force(
                gripper_pos,
                gripper_max_speed,
                gripper_torque_limit,
            ),
        )
        self._last_q_commanded = clipped

    async def gravity_compensate(
        self,
        kd: float = 0.5,
        free_joints: set[Joint] | None = None,
    ) -> None:
        """Apply one cycle of gravity compensation.

        For each joint in ``free_joints``: send ``set_impedance(p_des=current,
        v_des=0, kp=0, kd=kd, t_ff=gravity)``. Gravity is supported by the
        feedforward torque, and ``kd`` provides a small velocity-damping term so
        motion does not feel twitchy. These joints are free to be moved by hand.

        For each arm joint *not* in ``free_joints``: send ``set_impedance``
        with the joint's configured ``kp``/``kd`` from :class:`ArmConfig` to
        hold it rigidly at the position it had at the *first* call (or at the
        moment the free-joint set last changed), with gravity feedforward.
        This lets the operator isolate one joint at a time — everything else
        stays put for testing. To re-snapshot the hold position (e.g. after
        repositioning the arm), call :meth:`reset_gravity_hold` between calls.

        The gripper is always softly held at its current position regardless
        of ``free_joints``.

        Requires :meth:`start_telemetry` to be active so cached positions are
        fresh.

        Args:
            kd: Velocity damping for *free* joints (Nm·s/rad). 0 lets the arm
                coast freely (may feel underdamped); 0.5 is a good starting
                point. Tune to taste.
            free_joints: Set of arm joints to gravity-compensate. ``None`` (the
                default) frees all 7 arm joints. Joints not in this set are
                held rigidly at their initial position. ``Joint.GRIPPER`` is
                ignored if present.
        """
        free_set: frozenset[Joint] = (
            frozenset(ARM_JOINTS) if free_joints is None else frozenset(free_joints)
        )

        positions = self.positions
        arm_q = positions[: len(ARM_JOINTS)].astype(np.float32)
        gravity = self._gravity_comp.gravity_arm(arm_q, is_left=self._is_left)

        # Snapshot held positions on first call or whenever the free-joint set
        # changes; otherwise keep the original setpoint so kp can produce a
        # real restoring torque.
        if self._gc_hold_q is None or self._gc_hold_free != free_set:
            self._gc_hold_q = arm_q.copy()
            self._gc_hold_free = free_set

        gripper_i = self._gripper_i
        gripper_pos = float(positions[gripper_i])
        gripper_pos_raw = self._limits_hi[gripper_i] + gripper_pos * (
            self._limits_lo[gripper_i] - self._limits_hi[gripper_i]
        )

        # ``arm_q`` and ``_gc_hold_q`` are in joint frame; convert to motor
        # frame before sending to the impedance controller.
        arm_offsets = self._joint_offsets[: len(ARM_JOINTS)]

        tasks = []
        for i, j in enumerate(ARM_JOINTS):
            if j in free_set:
                p_des = float(arm_q[i] - arm_offsets[i])
                kp_cmd = 0.0
                kd_cmd = kd
            else:
                p_des = float(self._gc_hold_q[i] - arm_offsets[i])
                gains = getattr(self._arm_config, j.value)
                kp_cmd = gains.kp
                kd_cmd = gains.kd
            tasks.append(
                self.motors[j].set_impedance(
                    p_des,
                    0.0,
                    kp_cmd,
                    kd_cmd,
                    float(gravity[i]),
                )
            )
        # Hold the gripper softly so it does not drift open/closed.
        tasks.append(
            self.motors[Joint.GRIPPER].set_position_force(
                gripper_pos_raw,
                self._arm_config.gripper.max_speed,
                0.5,
            )
        )
        await asyncio.gather(*tasks)

    def reset_gravity_hold(self) -> None:
        """Forget the cached hold setpoint used by :meth:`gravity_compensate`.

        The next call to ``gravity_compensate`` will re-snapshot the held
        joints' positions from the current telemetry. Use this if you have
        manually repositioned the arm and want the held joints to lock in
        their new pose.
        """
        self._gc_hold_q = None
        self._gc_hold_free = None


class Axol(RobotBase):
    """Dual-arm Axol robot interface.

    Opens one CAN bus per arm and constructs all 16 motor drivers on entry.
    Use as an async context manager to ensure the buses are cleanly shut down.

        async with Axol() as axol:
            await axol.enable()
            await axol.start_telemetry(500)  # 500 Hz

            # control loop — instant, no await
            pos_l, pos_r = axol.left.positions, axol.right.positions

            await axol.motion_control(left=np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]))

    Attributes:
        left:  AxolArm for the left arm.
        right: AxolArm for the right arm.

    Args:
        config:        Dual-arm gains config. Left and right arm gains are specified
                       independently; the right arm defaults to the left with gravity
                       mirrored for shoulder_2 and elbow.
        left_channel:  SocketCAN interface name for the left arm.
        right_channel: SocketCAN interface name for the right arm.
    """

    def __init__(
        self,
        config: AxolConfig = AxolConfig(),
        left_channel: str | None = CAN_LEFT,
        right_channel: str | None = CAN_RIGHT,
    ) -> None:
        """Construct the dual-arm interface.

        CAN buses and motors are created but not started; call ``enable()``
        or use the class as an async context manager to bring up hardware.

        Args:
            config:        Per-joint gains, friction parameters, and gripper config.
            left_channel:  SocketCAN interface name for the left arm, or ``None`` to omit it.
            right_channel: SocketCAN interface name for the right arm, or ``None`` to omit it.
        """
        if left_channel is None and right_channel is None:
            raise ValueError(
                "At least one of left_channel or right_channel must be specified."
            )

        # Bake stiffness into the per-joint gains exactly once, here at the
        # single robot-construction boundary. ``resolved()`` is idempotent,
        # so this is safe even if the caller already resolved the config.
        config = config.resolved()

        self._gravity_comp = GravityCompensator(config)

        if left_channel is not None:
            self._left_bus = CanBus(left_channel)
            self.left = AxolArm(
                self._left_bus, config, self._gravity_comp, is_left=True
            )
        else:
            self.left = None

        if right_channel is not None:
            self._right_bus = CanBus(right_channel)
            self.right = AxolArm(
                self._right_bus, config, self._gravity_comp, is_left=False
            )
        else:
            self.right = None

    # ------------------------------------------------------------------ #
    # Polling                                                              #
    # ------------------------------------------------------------------ #

    async def start_telemetry(self, hz: float, *, torque: bool = False) -> None:
        """Start background telemetry polling on both arms at the given frequency.

        Args:
            hz:     Poll frequency in Hz.
            torque: If True, also fetch and cache torque each cycle.
        """
        tasks = []
        if self.left is not None:
            tasks.append(self.left.start_telemetry(hz, torque=torque))
        if self.right is not None:
            tasks.append(self.right.start_telemetry(hz, torque=torque))
        await asyncio.gather(*tasks)

    async def stop_telemetry(self) -> None:
        """Stop the background telemetry polling loop on both arms."""
        tasks = []
        if self.left is not None:
            tasks.append(self.left.stop_telemetry())
        if self.right is not None:
            tasks.append(self.right.stop_telemetry())
        await asyncio.gather(*tasks)

    # ------------------------------------------------------------------ #
    # Arm-wide commands                                                    #
    # ------------------------------------------------------------------ #

    async def enable(self) -> None:
        """Start CAN buses and enable all motors on both arms."""
        bus_tasks = []
        if self.left is not None:
            bus_tasks.append(self._left_bus.start())
        if self.right is not None:
            bus_tasks.append(self._right_bus.start())
        await asyncio.gather(*bus_tasks)

        motor_tasks = []
        if self.left is not None:
            motor_tasks.append(self.left.enable())
        if self.right is not None:
            motor_tasks.append(self.right.enable())
        await asyncio.gather(*motor_tasks)

    async def disable(self) -> None:
        """Disable all motors and close CAN buses."""
        tasks = []
        if self.left is not None:
            tasks.extend([self.left.stop_telemetry(), self.left.disable()])
        if self.right is not None:
            tasks.extend([self.right.stop_telemetry(), self.right.disable()])
        try:
            await asyncio.gather(*tasks)
        except Exception:
            pass
        finally:
            close_tasks = []
            if self.left is not None:
                close_tasks.append(self._left_bus.close())
            if self.right is not None:
                close_tasks.append(self._right_bus.close())
            await asyncio.gather(*close_tasks)

    async def clear_errors(self) -> None:
        """Clear latched error flags on both arms."""
        tasks = []
        if self.left is not None:
            tasks.append(self.left.clear_errors())
        if self.right is not None:
            tasks.append(self.right.clear_errors())
        await asyncio.gather(*tasks)

    async def set_control_mode(self, mode: ControlMode) -> None:
        """Set the control mode on all motors on both arms.

        Args:
            mode: Desired control mode.
        """
        tasks = []
        if self.left is not None:
            tasks.append(self.left.set_control_mode(mode))
        if self.right is not None:
            tasks.append(self.right.set_control_mode(mode))
        await asyncio.gather(*tasks)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    async def _gather_pair(left_coro, right_coro) -> tuple:
        """Run up to two coroutines concurrently; pass ``None`` to skip an arm."""
        coros = [c for c in (left_coro, right_coro) if c is not None]
        results = list(await asyncio.gather(*coros))
        left = results.pop(0) if left_coro is not None else None
        right = results.pop(0) if right_coro is not None else None
        return left, right

    # ------------------------------------------------------------------ #
    # Getters                                                              #
    # ------------------------------------------------------------------ #

    async def get_positions(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return joint positions (rad) for both arms as (left, right).

        Each array is shape (8,) in Joint enum order, or ``None`` if that arm is absent.
        """
        return await self._gather_pair(
            self.left.get_positions() if self.left is not None else None,
            self.right.get_positions() if self.right is not None else None,
        )

    async def get_velocities(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return shaft velocity (rad/s) for both arms as (left, right).

        Each array is shape (8,) in Joint enum order, or ``None`` if that arm is absent.
        """
        return await self._gather_pair(
            self.left.get_velocities() if self.left is not None else None,
            self.right.get_velocities() if self.right is not None else None,
        )

    async def get_torques(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return torque estimates for both arms as (left, right).

        Each array is shape (8,) in Joint enum order, or ``None`` if that arm is absent.
        """
        return await self._gather_pair(
            self.left.get_torques() if self.left is not None else None,
            self.right.get_torques() if self.right is not None else None,
        )

    async def get_temperatures(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return motor temperatures (°C) for both arms as (left, right).

        Each array is shape (8,) in Joint enum order, or ``None`` if that arm is absent.
        """
        return await self._gather_pair(
            self.left.get_temperatures() if self.left is not None else None,
            self.right.get_temperatures() if self.right is not None else None,
        )

    async def get_voltages(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return bus voltages (V) for both arms as (left, right).

        Each array is shape (8,) in Joint enum order, or ``None`` if that arm is absent.
        """
        return await self._gather_pair(
            self.left.get_voltages() if self.left is not None else None,
            self.right.get_voltages() if self.right is not None else None,
        )

    async def get_error_codes(
        self,
    ) -> tuple[list[MotorStatus] | None, list[MotorStatus] | None]:
        """Return MotorStatus for both arms as (left, right).

        Each list is in Joint enum order, or ``None`` if that arm is absent.
        """
        return await self._gather_pair(
            self.left.get_error_codes() if self.left is not None else None,
            self.right.get_error_codes() if self.right is not None else None,
        )

    async def get_gains(
        self,
    ) -> tuple[list[MotorGains] | None, list[MotorGains] | None]:
        """Return PID gains for both arms as (left, right).

        Each list is in Joint enum order, or ``None`` if that arm is absent.
        """
        return await self._gather_pair(
            self.left.get_gains() if self.left is not None else None,
            self.right.get_gains() if self.right is not None else None,
        )

    # ------------------------------------------------------------------ #
    # Setters                                                              #
    # ------------------------------------------------------------------ #

    async def set_gains(
        self,
        left: dict[Joint, MotorGains] = {},
        right: dict[Joint, MotorGains] = {},
    ) -> None:
        """Write PID gains to the specified joints on both arms."""
        tasks = []
        if left and self.left is not None:
            tasks.append(self.left.set_gains(left))
        if right and self.right is not None:
            tasks.append(self.right.set_gains(right))
        if tasks:
            await asyncio.gather(*tasks)

    async def set_zero_position(
        self,
        left: list[Joint] | None = None,
        right: list[Joint] | None = None,
    ) -> None:
        """Save the current shaft position as the encoder zero for the specified joints.

        The encoder zero is calibrated at each joint's mechanical end stop, not
        at the rest position; per-joint offsets keep the public API in joint
        frame (``0`` = rest).

        Args:
            left:  Joints on the left arm to zero. ``None`` skips the arm.
            right: Joints on the right arm to zero. ``None`` skips the arm.
        """
        tasks = []
        if left is not None and self.left is not None:
            tasks.append(self.left.set_zero_position(left))
        if right is not None and self.right is not None:
            tasks.append(self.right.set_zero_position(right))
        if tasks:
            await asyncio.gather(*tasks)

    async def set_acceleration(
        self,
        left: dict[Joint, float] = {},
        right: dict[Joint, float] = {},
    ) -> None:
        """Set per-joint acceleration ramps (rad/s²) on both arms.

        Args:
            left:  Joint → acceleration (rad/s²) for the left arm. ``None`` skips.
            right: Same for the right arm.
        """
        tasks = []
        if left and self.left is not None:
            tasks.append(self.left.set_acceleration(left))
        if right and self.right is not None:
            tasks.append(self.right.set_acceleration(right))
        if tasks:
            await asyncio.gather(*tasks)

    async def set_positions_velocity(
        self,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
        max_speed: float = 0.0,
    ) -> None:
        """Command joint positions (rad) via the motor's built-in position controller.

        Args:
            left:      Shape (8,) array of target positions (rad) in Joint enum order.
                       ``None`` skips the arm.
            right:     Same for the right arm.
            max_speed: Maximum speed (rad/s). 0.0 uses the motor's default.
        """
        tasks = []
        if left is not None and self.left is not None:
            tasks.append(self.left.set_position_velocity(left, max_speed))
        if right is not None and self.right is not None:
            tasks.append(self.right.set_position_velocity(right, max_speed))
        if tasks:
            await asyncio.gather(*tasks)

    async def set_velocity(
        self,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
    ) -> None:
        """Command target velocities (rad/s) on both arms concurrently.

        Args:
            left:  Shape (8,) array of target velocities (rad/s). ``None`` skips the arm.
            right: Same for the right arm.
        """
        tasks = []
        if left is not None and self.left is not None:
            tasks.append(self.left.set_velocity(left))
        if right is not None and self.right is not None:
            tasks.append(self.right.set_velocity(right))
        if tasks:
            await asyncio.gather(*tasks)

    async def motion_control(
        self,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
    ) -> None:
        """Send control commands to both arms concurrently.

        Arm joints use IMPEDANCE control; the gripper uses POSITION_FORCE control.
        See ``AxolArm.motion_control`` for full details.

        Args:
            left:  Shape (8,) array of target positions for the left arm
                   (arm joints in rad, gripper in [0, 1]).  ``None`` skips.
            right: Same for the right arm.
        """
        tasks = []
        if left is not None and self.left is not None:
            tasks.append(self.left.motion_control(left))
        if right is not None and self.right is not None:
            tasks.append(self.right.motion_control(right))
        if tasks:
            await asyncio.gather(*tasks)

    async def gravity_compensate(
        self,
        kd: float = 0.5,
        free_joints: set[Joint] | None = None,
    ) -> None:
        """Put both arms into gravity-compensation mode for one cycle.

        Joints in ``free_joints`` are sent ``set_impedance`` with ``kp=0``,
        ``kd=kd``, and a feedforward torque equal to the model-predicted
        gravity (free to move by hand). Joints *not* in ``free_joints`` are
        held rigidly at their current position using their configured
        ``ArmConfig`` gains, with gravity feedforward. ``free_joints=None``
        frees all 7 arm joints on each side. The grippers are softly held at
        their current positions.

        Telemetry must be active (positions are read from the cache) — call
        :meth:`start_telemetry` before driving this in a loop.

        Args:
            kd: Velocity damping coefficient for *free* joints (Nm·s/rad).
                Tune to taste; ``0.5`` is a reasonable starting point.
            free_joints: Set of arm joints to gravity-compensate. ``None`` (the
                default) frees every arm joint. Joints not in this set are
                held in place. The same filter is applied to both arms.
        """
        tasks = []
        if self.left is not None:
            tasks.append(self.left.gravity_compensate(kd, free_joints))
        if self.right is not None:
            tasks.append(self.right.gravity_compensate(kd, free_joints))
        if tasks:
            await asyncio.gather(*tasks)

    def reset_gravity_hold(self) -> None:
        """Re-snapshot the held setpoint on both arms' :meth:`gravity_compensate`."""
        if self.left is not None:
            self.left.reset_gravity_hold()
        if self.right is not None:
            self.right.reset_gravity_hold()
