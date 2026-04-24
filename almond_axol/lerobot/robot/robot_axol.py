"""
Axol robot as a LeRobot Robot.

AxolRobot wraps the async Axol hardware driver behind LeRobot's synchronous
Robot interface. A background thread runs a dedicated asyncio event loop so
Axol's CAN telemetry keeps streaming while get_observation() and send_action()
block synchronously on the calling thread.

Typical usage::

    from almond_axol.lerobot.robot import AxolRobot, AxolRobotConfig
    from almond_axol.lerobot.camera import ZedCameraConfig

    config = AxolRobotConfig(
        id="axol_01",
        cameras={
            "overhead": ZedCameraConfig(host="192.168.1.10", port=30000),
            "left_arm": ZedCameraConfig(host="192.168.1.10", port=30002),
            "right_arm": ZedCameraConfig(host="192.168.1.10", port=30004),
        },
    )
    with AxolRobot(config) as robot:
        obs = robot.get_observation()
        robot.send_action(obs)  # hold position
"""

from __future__ import annotations

import asyncio
import logging
import threading

import numpy as np
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ...robot.axol import Axol
from ...shared import Joint
from .config_axol import AxolRobotConfig

_logger = logging.getLogger(__name__)

_JOINTS = list(Joint)
_LEFT_POS_KEYS = [f"left_{j.value}.pos" for j in _JOINTS]
_RIGHT_POS_KEYS = [f"right_{j.value}.pos" for j in _JOINTS]
_LEFT_TRQ_KEYS = [f"left_{j.value}.trq" for j in _JOINTS]
_RIGHT_TRQ_KEYS = [f"right_{j.value}.trq" for j in _JOINTS]


class AxolRobot(Robot):
    """LeRobot Robot wrapping the Axol dual-arm hardware.

    Observations include joint positions for all 16 joints (8 per arm) plus any
    configured cameras. Actions are joint positions sent via impedance control (arm joints) and position-force control (gripper).

    Args:
        config: Hardware channels, camera configs, and gain config.
    """

    config_class = AxolRobotConfig
    name = "axol"

    def __init__(self, config: AxolRobotConfig) -> None:
        super().__init__(config)
        self.config = config
        self._axol: Axol | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self.cameras = make_cameras_from_configs(config.cameras)
        self._observation_features: dict[str, type | tuple] | None = None
        self._action_features: dict[str, type | tuple] | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._axol is not None

    @property
    def is_calibrated(self) -> bool:
        return True  # Encoder zeros set via axol CLI, not managed here

    @property
    def observation_features(self) -> dict:
        if self._observation_features is None:
            features: dict[str, type | tuple] = {
                key: float for key in _LEFT_POS_KEYS + _RIGHT_POS_KEYS
            }
            if self.config.observe_torques:
                for key in _LEFT_TRQ_KEYS + _RIGHT_TRQ_KEYS:
                    features[key] = float
            for cam_name, cfg in self.config.cameras.items():
                features[cam_name] = (cfg.height, cfg.width, 3)
            self._observation_features = features
        return self._observation_features

    @property
    def action_features(self) -> dict:
        if self._action_features is None:
            self._action_features = {
                key: float for key in _LEFT_POS_KEYS + _RIGHT_POS_KEYS
            }
        return self._action_features

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Open CAN buses, enable motors, start telemetry, and connect cameras."""
        loop = asyncio.new_event_loop()
        self._loop = loop
        self._loop_thread = threading.Thread(
            target=loop.run_forever, name="axol-event-loop", daemon=True
        )
        self._loop_thread.start()

        asyncio.run_coroutine_threadsafe(self._connect_async(), loop).result(timeout=30)

        for cam in self.cameras.values():
            cam.connect()

        _logger.info("AxolRobot connected.")

    async def _connect_async(self) -> None:
        self._axol = Axol(
            self.config.axol_config,
            left_channel=self.config.left_channel,
            right_channel=self.config.right_channel,
        )
        await self._axol.enable()
        await self._axol.start_telemetry(self.config.telemetry_hz)

    def disconnect(self) -> None:
        """Disable motors, stop telemetry, close CAN buses, and disconnect cameras."""
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()

        if self._loop is not None and self._axol is not None:
            asyncio.run_coroutine_threadsafe(
                self._disconnect_async(), self._loop
            ).result(timeout=10)

        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5)

        self._loop = None
        self._loop_thread = None
        _logger.info("AxolRobot disconnected.")

    async def _disconnect_async(self) -> None:
        if self._axol is None:
            return
        await self._axol.stop_telemetry()
        await self._axol.__aexit__(None, None, None)
        self._axol = None

    # ------------------------------------------------------------------
    # Calibration / configuration (no-ops for Axol)
    # ------------------------------------------------------------------

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @property
    def positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Cached arm positions from telemetry. Call after connect().

        Returns ``(left, right)`` each shape (8,) in Joint enum order,
        with gripper normalized to [0, 1].
        """
        assert self._axol is not None
        assert self._axol.left is not None
        assert self._axol.right is not None
        return self._axol.left.positions, self._axol.right.positions

    # ------------------------------------------------------------------
    # Observation / action
    # ------------------------------------------------------------------

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Return cached joint positions and latest camera frames."""
        assert self._axol is not None
        assert self._axol.left is not None
        assert self._axol.right is not None

        left_pos = self._axol.left.positions  # np.ndarray (8,), from telemetry cache
        right_pos = self._axol.right.positions

        obs: RobotObservation = {}
        for i, key in enumerate(_LEFT_POS_KEYS):
            obs[key] = float(left_pos[i])
        for i, key in enumerate(_RIGHT_POS_KEYS):
            obs[key] = float(right_pos[i])

        if self.config.observe_torques:
            left_trq = self._axol.left.torques
            right_trq = self._axol.right.torques
            for i, key in enumerate(_LEFT_TRQ_KEYS):
                obs[key] = float(left_trq[i])
            for i, key in enumerate(_RIGHT_TRQ_KEYS):
                obs[key] = float(right_trq[i])

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.read_latest()

        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Send joint position targets via impedance control (arm joints) and position-force control (gripper).

        Args:
            action: Dict with keys matching action_features, values in radians.

        Returns:
            The action as sent (unmodified).
        """
        assert self._axol is not None and self._loop is not None

        left = np.array([action[k] for k in _LEFT_POS_KEYS], dtype=np.float32)
        right = np.array([action[k] for k in _RIGHT_POS_KEYS], dtype=np.float32)

        asyncio.run_coroutine_threadsafe(
            self._axol.motion_control(left=left, right=right), self._loop
        ).result(timeout=1.0)

        return action
