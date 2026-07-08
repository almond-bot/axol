"""Hardware control for the handheld UMI data-collection rig.

The UMI rig is a pair of handheld devices — a Quest controller rigidly mounted
to the same Damiao gripper the robot uses — held by a human demonstrator. Each
gripper sits alone on its own CAN bus (``can_alm_umi_l`` / ``can_alm_umi_r``)
at the production gripper CAN ID (0x08).

:class:`Umi` mirrors the :class:`~almond_axol.robot.axol.Axol` control surface
(``enable`` / ``get_positions`` / ``motion_control`` / per-side ``positions`` /
``torques``) so the LeRobot wrapper and ``collect-data`` drive it unchanged.
The seven arm joints per side are **virtual**: there is no arm, so
``motion_control`` just latches the commanded joint targets and they are read
back as the "measured" state. The gripper is real — commanded in
POSITION_FORCE mode from the trigger value and observed from motor feedback,
exactly like on the robot. Datasets recorded through this class therefore have
the same schema as robot-collected ones (state/action = 16 joint positions),
with the arm-state channel equal to the commanded IK solution.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from ..constants import ARM_JOINTS, CAN_UMI_LEFT, CAN_UMI_RIGHT
from ..motor import CanBus, ControlMode, Joint, Motor
from .axol import GRIPPER_TRAVEL, calibrate_gripper_open_stop
from .base import RobotBase
from .config import AxolConfig, PositionForceConfig

_logger = logging.getLogger(__name__)

_N_ARM = len(ARM_JOINTS)


class UmiGripperArm:
    """One handheld gripper plus a virtual 7-joint arm.

    Mirrors the parts of :class:`~almond_axol.robot.axol.AxolArm` that the
    control stack touches. Positions are shape (8,) in ``Joint`` order: the 7
    virtual arm joints echo the last ``motion_control`` command (radians, joint
    frame), the gripper is real feedback normalised to [0 = closed, 1 = open].
    """

    def __init__(self, bus: CanBus, gripper_config: PositionForceConfig) -> None:
        self._motor = Motor(bus, Joint.GRIPPER)
        self._gripper_config = gripper_config
        # Raw motor radians of the open / closed hard-stops, found on enable()
        # by the same torque-stop sweep the robot gripper uses.
        self._open_pos = 0.0
        self._closed_pos = GRIPPER_TRAVEL
        self._virtual_arm = np.zeros(_N_ARM, dtype=np.float32)

    # -- Lifecycle -----------------------------------------------------------

    async def enable(self) -> None:
        """Enable the gripper motor, calibrate its open stop, and arm POSITION_FORCE."""
        await self._motor.enable()
        await self._motor.set_control_mode(ControlMode.IMPEDANCE)
        self._open_pos = await calibrate_gripper_open_stop(self._motor)
        self._closed_pos = self._open_pos + GRIPPER_TRAVEL
        await self._motor.set_control_mode(ControlMode.POSITION_FORCE)

    async def disable(self) -> None:
        await self._motor.disable()

    async def start_telemetry(self, hz: float, *, torque: bool = False) -> None:
        await self._motor.start_telemetry(hz, torque=torque)

    async def stop_telemetry(self) -> None:
        await self._motor.stop_telemetry()

    async def wait_for_telemetry(self, timeout: float = 5.0) -> None:
        """Block until the gripper motor has reported at least one position."""
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while not self._motor.has_position:
            if loop.time() >= deadline:
                raise TimeoutError(
                    "No feedback from the UMI gripper — check power and CAN wiring"
                )
            await asyncio.sleep(0.01)

    # -- State ---------------------------------------------------------------

    def _normalize(self, raw: float) -> float:
        """Raw motor radians → [0 = closed, 1 = open]."""
        return (raw - self._closed_pos) / (self._open_pos - self._closed_pos)

    @property
    def positions(self) -> np.ndarray:
        """Latest positions, shape (8,) in Joint order (cached gripper feedback)."""
        out = np.empty(_N_ARM + 1, dtype=np.float32)
        out[:_N_ARM] = self._virtual_arm
        out[_N_ARM] = self._normalize(self._motor.position)
        return out

    @property
    def torques(self) -> np.ndarray:
        """Latest torques, shape (8,); the virtual arm joints report zero."""
        out = np.zeros(_N_ARM + 1, dtype=np.float32)
        out[_N_ARM] = self._motor.torque
        return out

    async def get_positions(self) -> np.ndarray:
        """Actively read the gripper position; virtual arm joints as stored."""
        out = np.empty(_N_ARM + 1, dtype=np.float32)
        out[:_N_ARM] = self._virtual_arm
        out[_N_ARM] = self._normalize(await self._motor.get_position())
        return out

    # -- Commands -------------------------------------------------------------

    async def motion_control(self, q: np.ndarray) -> None:
        """Latch the virtual arm targets and command the real gripper.

        Args:
            q: Shape (8,) targets in Joint order — arm joints in radians
               (stored, nothing physical to move), gripper normalised [0, 1].
        """
        self._virtual_arm = np.asarray(q[:_N_ARM], dtype=np.float32).copy()
        raw = self._closed_pos + float(q[_N_ARM]) * (self._open_pos - self._closed_pos)
        raw = float(np.clip(raw, self._open_pos, self._closed_pos))
        await self._motor.set_position_force(
            raw,
            self._gripper_config.max_speed,
            self._gripper_config.torque_limit,
        )


class Umi(RobotBase):
    """Dual handheld UMI grippers behind the ``Axol`` control surface.

    Args:
        config:        Reused for the per-side gripper POSITION_FORCE tuning
                       (``ArmConfig.gripper``); everything else is ignored.
        left_channel:  SocketCAN interface of the left gripper, or ``None`` to omit.
        right_channel: SocketCAN interface of the right gripper, or ``None`` to omit.
    """

    def __init__(
        self,
        config: AxolConfig = AxolConfig(),
        left_channel: str | None = CAN_UMI_LEFT,
        right_channel: str | None = CAN_UMI_RIGHT,
    ) -> None:
        if left_channel is None and right_channel is None:
            raise ValueError(
                "At least one of left_channel or right_channel must be specified."
            )

        self.left: UmiGripperArm | None = None
        self.right: UmiGripperArm | None = None
        self._left_bus: CanBus | None = None
        self._right_bus: CanBus | None = None
        if left_channel is not None:
            self._left_bus = CanBus(left_channel)
            self.left = UmiGripperArm(self._left_bus, config.left.gripper)
        if right_channel is not None:
            self._right_bus = CanBus(right_channel)
            self.right = UmiGripperArm(self._right_bus, config.right.gripper)

    # -- Lifecycle -------------------------------------------------------------

    async def enable(self) -> None:
        """Start CAN buses and enable + calibrate both grippers."""
        await asyncio.gather(
            *[b.start() for b in (self._left_bus, self._right_bus) if b is not None]
        )
        await asyncio.gather(
            *[a.enable() for a in (self.left, self.right) if a is not None]
        )

    async def disable(self) -> None:
        """Disable the grippers and close the CAN buses."""
        arms = [a for a in (self.left, self.right) if a is not None]
        try:
            await asyncio.gather(
                *[a.stop_telemetry() for a in arms],
                *[a.disable() for a in arms],
            )
        except Exception:
            _logger.exception("UMI gripper disable failed")
        finally:
            await asyncio.gather(
                *[b.close() for b in (self._left_bus, self._right_bus) if b is not None]
            )

    # -- Telemetry ---------------------------------------------------------------

    async def start_telemetry(self, hz: float, *, torque: bool = False) -> None:
        await asyncio.gather(
            *[
                a.start_telemetry(hz, torque=torque)
                for a in (self.left, self.right)
                if a is not None
            ]
        )

    async def stop_telemetry(self) -> None:
        await asyncio.gather(
            *[a.stop_telemetry() for a in (self.left, self.right) if a is not None]
        )

    async def wait_for_telemetry(self, timeout: float = 5.0) -> None:
        await asyncio.gather(
            *[
                a.wait_for_telemetry(timeout)
                for a in (self.left, self.right)
                if a is not None
            ]
        )

    # -- State / commands ----------------------------------------------------------

    async def get_positions(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        left = await self.left.get_positions() if self.left is not None else None
        right = await self.right.get_positions() if self.right is not None else None
        return left, right

    async def motion_control(
        self,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
    ) -> None:
        tasks = []
        if left is not None and self.left is not None:
            tasks.append(self.left.motion_control(left))
        if right is not None and self.right is not None:
            tasks.append(self.right.motion_control(right))
        if tasks:
            await asyncio.gather(*tasks)

    # -- Axol-surface stubs -----------------------------------------------------

    async def gravity_compensate(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("The UMI rig has no arm to gravity-compensate.")

    def reset_command_state(self) -> None:
        """No command history to clear — the arms are virtual."""
