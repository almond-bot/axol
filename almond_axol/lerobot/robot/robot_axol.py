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
            "overhead": ZedCameraConfig(serial=41234567, stereo=True),
            "left_arm": ZedCameraConfig(serial=41234568),
            "right_arm": ZedCameraConfig(serial=41234569),
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
import time

import numpy as np
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ...robot.axol import Axol
from ...utils.shared import Joint
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
        self.cameras, self._stereo_cameras = self._build_cameras()
        self._observation_features: dict[str, type | tuple] | None = None
        self._action_features: dict[str, type | tuple] | None = None

    def _build_cameras(self) -> tuple[dict, list]:
        """Build the camera set, expanding any stereo camera into two eyes.

        The ``video_backend`` config selects the capture path. ``"gst"`` (or
        ``"auto"`` when the stack is installed) opens each camera through the
        GPU-resident zed-gstreamer pipeline (:mod:`almond_axol.vr.gst_zed`):
        one grab/encode on the GPU serves both the dataset (raw frames, via
        ``read_at_or_after``) and the headset view (encoded AUs, via
        ``subscribe``), at far lower latency than the SDK's host round trip.
        ``"sdk"`` (or ``"auto"`` without the stack) uses the ZED Python SDK.

        Either way a stereo camera is backed by a single object (one decode)
        whose left/right views are registered under ``<name>_left`` /
        ``<name>_right`` so the rest of the pipeline treats the two eyes as
        ordinary cameras.
        """
        if self._use_gst_cameras():
            return self._build_gst_cameras()
        return self._build_sdk_cameras()

    def _use_gst_cameras(self) -> bool:
        """Whether to open cameras via the gst pipeline for the chosen backend.

        Checks the plugin each configured camera actually needs — mono cameras
        use ``zedxonesrc`` (:func:`zed_gst_available`), stereo cameras use
        ``zedsrc`` (:func:`zed_stereo_gst_available`) — so the decision matches
        what :meth:`_build_gst_cameras` will open (and the teleop relay's
        per-camera gating). ``auto`` takes the gst path only when every camera's
        plugin is present; ``gst`` warns and falls back to the SDK if anything
        is missing; ``sdk`` always uses the SDK.
        """
        backend = getattr(self.config, "video_backend", "auto")
        if backend == "sdk":
            return False
        try:
            from ...vr.gst_zed import zed_gst_available, zed_stereo_gst_available
        except Exception:  # noqa: BLE001 - gst module import failed
            if backend == "gst":
                _logger.warning("video_backend='gst' but gst_zed is unimportable")
            return False
        cams = self.config.observation_cameras().values()
        needs_mono = any(eye is None for _, eye in cams)
        needs_stereo = any(eye is not None for _, eye in cams)
        available = (not needs_mono or zed_gst_available()) and (
            not needs_stereo or zed_stereo_gst_available()
        )
        if backend == "gst" and not available:
            _logger.warning(
                "video_backend='gst' requested but the required zed-gstreamer "
                "plugins are unavailable; run `axol gst.install` + "
                "`axol gst.build-zed`. Falling back to the SDK camera path."
            )
            return False
        return available

    def _build_sdk_cameras(self) -> tuple[dict, list]:
        obs_cams = self.config.observation_cameras()
        mono = {key: cfg for key, (cfg, eye) in obs_cams.items() if eye is None}
        cameras: dict = dict(make_cameras_from_configs(mono))

        eyes = {key: (cfg, eye) for key, (cfg, eye) in obs_cams.items() if eye}
        stereo_cameras: list = []
        if eyes:
            from ..camera.camera_zed import ZedStereoCamera

            by_cfg: dict[int, ZedStereoCamera] = {}
            for key, (cfg, eye) in eyes.items():
                cam = by_cfg.get(id(cfg))
                if cam is None:
                    cam = ZedStereoCamera(cfg)
                    by_cfg[id(cfg)] = cam
                    stereo_cameras.append(cam)
                cameras[key] = cam.left_view if eye == "left" else cam.right_view
        return cameras, stereo_cameras

    def _build_gst_cameras(self) -> tuple[dict, list]:
        """Build cameras on the gst pipeline (raw for dataset + encoded view)."""
        from ...vr.gst_zed import ZedGstCamera, ZedGstStereoCamera

        obs_cams = self.config.observation_cameras()
        cameras: dict = {}
        owned: list = []
        by_cfg: dict[int, ZedGstStereoCamera] = {}
        for key, (cfg, eye) in obs_cams.items():
            resolution = cfg.resolution_name() or "HD1200"
            fps = cfg.fps or 60
            if eye is None:
                cam = ZedGstCamera(
                    cfg.serial, resolution, fps, want_encoded=True, want_raw=True
                )
                cameras[key] = cam
                owned.append(cam)
                continue
            stereo = by_cfg.get(id(cfg))
            if stereo is None:
                stereo = ZedGstStereoCamera(
                    cfg.serial, resolution, fps, want_encoded=True, want_raw=True
                )
                by_cfg[id(cfg)] = stereo
                owned.append(stereo)
            cameras[key] = stereo.left_view if eye == "left" else stereo.right_view
        return cameras, owned

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
        if self._observation_features is not None:
            return self._observation_features

        features: dict[str, type | tuple] = {
            key: float for key in _LEFT_POS_KEYS + _RIGHT_POS_KEYS
        }
        if self.config.observe_torques:
            for key in _LEFT_TRQ_KEYS + _RIGHT_TRQ_KEYS:
                features[key] = float

        # Use the live camera dimensions (auto-detected from the camera on
        # connect) so stereo per-eye sizes are correct; cache only once every
        # camera reports a size so a pre-connect read isn't frozen in.
        complete = True
        for cam_name, cam in self.cameras.items():
            height = getattr(cam, "height", None)
            width = getattr(cam, "width", None)
            if height is None or width is None:
                complete = False
            features[cam_name] = (height, width, 3)

        if complete:
            self._observation_features = features
        return features

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
        await self._axol.start_telemetry(
            self.config.telemetry_hz, torque=self.config.observe_torques
        )
        await self._axol.wait_for_telemetry()

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
        await self._axol.disable()
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
    def get_joint_observation(self) -> RobotObservation:
        """Return cached joint positions only — no camera reads.

        Use this in the high-frequency teleop path to avoid copying large
        camera frames on every step.  Call :meth:`get_observation` only when
        a full observation (joints + cameras) is actually needed.
        """
        assert self._axol is not None
        assert self._axol.left is not None
        assert self._axol.right is not None

        left_pos = self._axol.left.positions
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

        return obs

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Return cached joint positions and timestamp-aligned camera frames.

        Cameras are sampled with :meth:`ZedCamera.read_at_or_after` against a
        shared ``time.perf_counter()`` target so every frame in the
        observation shares the same capture instant — matching the alignment
        guarantee that ``collect-data`` writes into the training dataset. If a
        camera fails to produce a qualifying frame within ``timeout_ms``, we
        fall back to ``read_latest()`` so a single stale camera doesn't stall
        inference.
        """
        assert self._axol is not None
        assert self._axol.left is not None
        assert self._axol.right is not None

        target_ts = time.perf_counter()

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
            cam_fps = getattr(cam, "fps", None) or 30
            timeout_ms = int(2 * 1000.0 / cam_fps + 200)
            try:
                frame, _cap_ts, _recv_ts = cam.read_at_or_after(  # type: ignore[attr-defined]
                    target_ts, timeout_ms=timeout_ms
                )
            except (TimeoutError, RuntimeError) as exc:
                _logger.debug(
                    "get_observation: %s read_at_or_after(%.6f) failed (%s); "
                    "falling back to read_latest().",
                    cam_key,
                    target_ts,
                    exc,
                )
                frame = cam.read_latest()
            obs[cam_key] = frame

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

    def gravity_compensate(
        self, kd: float = 0.5, free_joints: set[Joint] | None = None
    ) -> None:
        """Apply one cycle of gravity compensation on both arms.

        Submits onto the robot's background event loop (mirroring
        :meth:`send_action`) and blocks until the cycle is sent. Telemetry must
        be active, so call this only while connected; drive it in a loop at the
        desired rate to keep the arms free to be hand-guided. See
        :meth:`almond_axol.robot.axol.Axol.gravity_compensate` for the
        ``kd``/``free_joints`` semantics.
        """
        assert self._axol is not None and self._loop is not None
        asyncio.run_coroutine_threadsafe(
            self._axol.gravity_compensate(kd=kd, free_joints=free_joints), self._loop
        ).result(timeout=1.0)

    def reset_command_state(self) -> None:
        """Clear cached command history after an out-of-band move.

        Call after hand-guiding the arms (e.g. under
        :meth:`gravity_compensate`) and before resuming :meth:`send_action`, so
        the return-to-pose command is not rejected by the max-step safety
        check. See :meth:`almond_axol.robot.axol.Axol.reset_command_state`.

        Mutates plain Python state on the arm wrappers, so it runs directly
        without the event loop.
        """
        assert self._axol is not None
        self._axol.reset_command_state()
