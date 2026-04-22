from __future__ import annotations

import asyncio
import math

import numpy as np

from ..motor import CanBus, ControlMode, Joint, Motor, MotorGains, MotorStatus
from ..shared import CAN_LEFT, CAN_RIGHT
from .base import RobotBase
from .config import AxolConfig
from .control import Differentiator, compute_feedforward

_TAU = 2 * math.pi

# Per-joint position limits (rad).  shoulder_2 is asymmetric across arms.
SHOULDER_2_LEFT_LIMITS = (-0.25 * _TAU, 0.03 * _TAU)
SHOULDER_2_RIGHT_LIMITS = (-0.03 * _TAU, 0.25 * _TAU)
ELBOW_LEFT_LIMITS = (0.0, 0.42 * _TAU)
ELBOW_RIGHT_LIMITS = (-0.42 * _TAU, 0.0)

LIMITS: dict[Joint, tuple[float, float]] = {
    Joint.SHOULDER_1: (-0.25 * _TAU, 0.25 * _TAU),
    Joint.SHOULDER_3: (-0.25 * _TAU, 0.25 * _TAU),
    Joint.WRIST_1: (-0.25 * _TAU, 0.25 * _TAU),
    Joint.WRIST_2: (-0.25 * _TAU, 0.25 * _TAU),
    Joint.WRIST_3: (-0.25 * _TAU, 0.25 * _TAU),
    # Gripper absent: open position varies per unit, found at runtime by _calibrate_gripper().
}

# Fixed open-to-close travel of the gripper (rad).
GRIPPER_TRAVEL = 0.8037 * _TAU

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
    if joint == Joint.SHOULDER_2:
        return SHOULDER_2_LEFT_LIMITS if is_left else SHOULDER_2_RIGHT_LIMITS
    if joint == Joint.ELBOW:
        return ELBOW_LEFT_LIMITS if is_left else ELBOW_RIGHT_LIMITS
    if joint == Joint.GRIPPER:
        return (0.0, 1.0)
    return LIMITS.get(joint, (-math.inf, math.inf))


class AxolArm:
    """Controls one 7-DOF + gripper arm over a single CAN bus.

    Not instantiated directly — access via ``axol.left`` or ``axol.right``.
    """

    def __init__(
        self,
        bus: CanBus,
        config: AxolConfig,
        is_left: bool = True,
    ) -> None:
        self._config = config
        self._arm_config = config.left if is_left else config.right
        self.motors: dict[Joint, Motor] = {joint: Motor(bus, joint) for joint in Joint}
        self._differentiator = Differentiator(n=len(list(Joint)))

        # Clipping arrays in raw motor radians.  arm_limits() returns normalised [0, 1]
        # for the gripper, so the gripper entries are seeded with raw defaults here;
        # _calibrate_gripper() overwrites them on enable.
        # Pre-calibration defaults assume zero is closed — do not rely on for actual motion.
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

        Returns shape (8,) array in Joint enum order. Arm joints are in radians;
        the gripper is normalized to [0, 1] (0.0 = closed, 1.0 = fully open),
        consistent with set_position_velocity and motion_control.
        """
        joints = list(Joint)
        values = [self.motors[j].position for j in joints]
        gripper_i = self._gripper_i
        values[gripper_i] = (values[gripper_i] - self._limits_hi[gripper_i]) / (
            self._limits_lo[gripper_i] - self._limits_hi[gripper_i]
        )
        return np.array(values, dtype=np.float32)

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

        Returns shape (8,) array in Joint enum order. Arm joints are in radians;
        the gripper is normalized to [0, 1] (0.0 = closed, 1.0 = fully open),
        consistent with set_position_velocity and motion_control.
        """
        joints = list(Joint)
        values = list(
            await asyncio.gather(*[self.motors[j].get_position() for j in joints])
        )
        gripper_i = self._gripper_i
        values[gripper_i] = (values[gripper_i] - self._limits_hi[gripper_i]) / (
            self._limits_lo[gripper_i] - self._limits_hi[gripper_i]
        )
        return np.array(values, dtype=np.float32)

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
        The gripper value is normalized: 0.0 = closed, 1.0 = fully open.

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
        await asyncio.gather(
            *[
                self.motors[j].set_position_velocity(float(clipped[i]), max_speed)
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
               Arm joints are in radians; gripper is normalized to [0, 1]
               (0.0 = closed, 1.0 = fully open).
        """
        q = q.copy()
        gripper_i = self._gripper_i
        q[gripper_i] = self._limits_hi[gripper_i] + q[gripper_i] * (
            self._limits_lo[gripper_i] - self._limits_hi[gripper_i]
        )
        clipped = np.clip(q, self._limits_lo, self._limits_hi)

        # Velocity feedforward via differentiation of commanded positions (rad/s).
        velocities = self._differentiator.differentiate(list(clipped))

        gripper_i = self._gripper_i
        gripper_pos = float(clipped[gripper_i])
        gripper_max_speed = self._arm_config.gripper.max_speed
        gripper_torque_limit = self._arm_config.gripper.torque_limit

        def _mit_cmd(i: int, j: Joint):
            gains = getattr(self._arm_config, j.value)
            t_ff = compute_feedforward(
                float(clipped[i]),
                velocities[i],
                gains.ga,
                gains.gb,
                gains.fc,
                gains.k,
                gains.fv,
                gains.fo,
            )
            return self.motors[j].set_impedance(
                float(clipped[i]),
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
        if left_channel is None and right_channel is None:
            raise ValueError(
                "At least one of left_channel or right_channel must be specified."
            )

        if left_channel is not None:
            self._left_bus = CanBus(left_channel)
            self.left = AxolArm(self._left_bus, config, is_left=True)
        else:
            self.left = None

        if right_channel is not None:
            self._right_bus = CanBus(right_channel)
            self.right = AxolArm(self._right_bus, config, is_left=False)
        else:
            self.right = None

    async def __aenter__(self) -> Axol:
        await self.enable()
        return self

    async def __aexit__(self, *_) -> None:
        await self.disable()

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
        await asyncio.gather(*tasks)

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
