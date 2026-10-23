"""The Mantis as a LeRobot Robot.

Same synchronous Robot surface as :class:`AxolRobot` — observations, actions,
cameras, event loop — but the hardware is the Mantis rig: a pair of handheld
Damiao grippers on their own CAN buses, with virtual arm joints latched from
the commanded IK targets. Like the robot it is driven through the Rust
realtime core (:class:`~almond_axol.robot.Mantis` wrapping
:class:`~almond_axol.robot.mantis.MantisHardware`): ``axol-rt`` owns the gripper
buses and streams the POSITION_FORCE commands for the duration of each take.
``collect-data`` drives it with the exact same control loop it uses for the
robot, so Mantis datasets are schema-identical to robot-collected ones.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_not_connected

from ...constants import Joint
from ...rt import Mantis
from .config_mantis import MantisRobotConfig
from .robot_axol import _LEFT_GRIPPER_KEY, _RIGHT_GRIPPER_KEY, AxolRobot

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ...kinematics.config import KinematicsConfig


class MantisRobot(AxolRobot):
    """LeRobot Robot wrapping the Mantis.

    Observation state is 16 joint positions like the robot's: the 14 arm
    values echo the commanded IK solution (there is no physical arm to
    measure), the 2 gripper values are real motor feedback. Actions are the
    same joint-position dicts teleop produces; only the grippers actuate.
    """

    config_class = MantisRobotConfig
    name = "axol_mantis"

    def __init__(
        self,
        config: MantisRobotConfig,
        *,
        ik_config: KinematicsConfig | None = None,
        defer_gripper_enable: bool = False,
    ) -> None:
        """Build a Mantis robot, optionally deferring gripper motor torque.

        The default preserves standalone/teleop behavior.  Data collection
        passes ``defer_gripper_enable=True`` so :meth:`connect` opens the CAN
        buses and cameras without actuating either gripper; episode control
        then calls :meth:`enable_grippers_async` and
        :meth:`disable_grippers_async` on the robot event loop, which arm and
        disarm the realtime core on the gripper buses per take.
        """
        self._defer_gripper_enable = defer_gripper_enable
        super().__init__(config, ik_config=ik_config)

    def _build_hardware(self) -> Mantis:
        return Mantis(
            self.config.axol_config,
            left_channel=self.config.left_channel,
            right_channel=self.config.right_channel,
            defer_gripper_enable=self._defer_gripper_enable,
            record=self._control_trace,
        )

    @check_if_not_connected
    def get_gripper_observation(self) -> RobotObservation:
        """Read real gripper feedback without converting virtual arm joints.

        Mantis collection supplies all end-effector dimensions from tracking.
        Running FK here only to overwrite its result costs control-thread CPU
        and can delay the receive thread. Keep optional torque fields exactly
        as the ordinary observation API emits them.
        """
        assert self._axol is not None
        assert self._axol.left is not None and self._axol.right is not None
        left, right = self._axol.left, self._axol.right
        return self._gripper_state_from_arrays(
            left.positions,
            right.positions,
            left.torques if self.config.observe_torques else None,
            right.torques if self.config.observe_torques else None,
        )

    def gripper_observation_at(
        self, ts: float, max_skew_s: float = 0.010
    ) -> RobotObservation | None:
        """Capture-aligned gripper feedback without virtual arm FK.

        Keep the same realtime telemetry lookup and 10 ms skew bound as
        ``joint_observation_at``. The caller retains the live reading when
        no sufficiently close history sample is available.
        """
        state_nearest = getattr(self._axol, "state_nearest", None)
        if state_nearest is None:
            return None
        state = state_nearest(ts, timeout=0.0)
        if state is None:
            return None
        left_pos, right_pos, left_trq, right_trq, state_ts = state
        if abs(state_ts - ts) > max_skew_s:
            return None
        return self._gripper_state_from_arrays(left_pos, right_pos, left_trq, right_trq)

    def _gripper_state_from_arrays(
        self,
        left_pos: np.ndarray,
        right_pos: np.ndarray,
        left_trq: np.ndarray | None,
        right_trq: np.ndarray | None,
    ) -> RobotObservation:
        grip_index = list(Joint).index(Joint.GRIPPER)
        observation = {
            _LEFT_GRIPPER_KEY: float(left_pos[grip_index]),
            _RIGHT_GRIPPER_KEY: float(right_pos[grip_index]),
        }
        if self.config.observe_torques:
            if left_trq is None or right_trq is None:
                raise RuntimeError("timestamped telemetry snapshot has no torques")
            observation.update(zip(self._left_trq_keys, map(float, left_trq)))
            observation.update(zip(self._right_trq_keys, map(float, right_trq)))
        return observation

    def gripper_action_to_dataset(self, action: RobotAction) -> RobotAction:
        """Keep commanded grips; raw tracking supplies the Cartesian targets.

        Match the existing joint-action packing's float32 conversion, without
        invoking FK for twelve end-effector values collection will replace.
        """
        return {
            key: float(np.float32(action[key]))
            for key in (_LEFT_GRIPPER_KEY, _RIGHT_GRIPPER_KEY)
        }

    async def open_grippers_async(self) -> None:
        """Open both grippers fully, then release torque (pre-record prep).

        See :meth:`~almond_axol.robot.Mantis.open_grippers`: this also
        performs the one-time hard-stop calibration, so the later
        :meth:`enable_grippers_async` at take start is immediate.
        """
        assert isinstance(self._axol, Mantis), "connect() first"
        await self._axol.open_grippers()

    async def enable_grippers_async(self) -> None:
        """Enable (calibrating on first use) both grippers and arm the core."""
        assert isinstance(self._axol, Mantis), "connect() first"
        await self._axol.enable_grippers()

    async def disable_grippers_async(self) -> None:
        """Disarm the core and disable both grippers, retaining calibration."""
        assert isinstance(self._axol, Mantis), "connect() first"
        await self._axol.disable_grippers()
