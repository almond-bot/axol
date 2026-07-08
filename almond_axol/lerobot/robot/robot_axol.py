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
from typing import TYPE_CHECKING

import numpy as np
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ...constants import Joint
from ...robot.axol import Axol
from .config_axol import AxolRobotConfig

if TYPE_CHECKING:
    from ...kinematics.fk import AxolForwardKinematics
    from ...kinematics.solver import KinematicsSolver

_logger = logging.getLogger(__name__)

_JOINTS = list(Joint)
_LEFT_POS_KEYS = [f"left_{j.value}.pos" for j in _JOINTS]
_RIGHT_POS_KEYS = [f"right_{j.value}.pos" for j in _JOINTS]
_LEFT_TRQ_KEYS = [f"left_{j.value}.trq" for j in _JOINTS]
_RIGHT_TRQ_KEYS = [f"right_{j.value}.trq" for j in _JOINTS]

# The gripper position is observed in both joint and Cartesian modes — it is the
# last entry of each arm's position vector (Joint.GRIPPER is last in the enum).
_LEFT_GRIPPER_KEY = _LEFT_POS_KEYS[-1]
_RIGHT_GRIPPER_KEY = _RIGHT_POS_KEYS[-1]

# Cartesian observation keys (observe_cartesian): a 6-axis end-effector pose per
# arm, replacing that arm's 7 joint-angle keys. Axis order matches
# AxolForwardKinematics.ee_poses (position x/y/z then rotation vector rx/ry/rz).
_EE_AXES = ("x", "y", "z", "rx", "ry", "rz")
_LEFT_EE_KEYS = [f"left_ee.{a}" for a in _EE_AXES]
_RIGHT_EE_KEYS = [f"right_ee.{a}" for a in _EE_AXES]


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
        # Built on connect() only when observe_cartesian is set; turns cached
        # joint angles into end-effector poses for the observation.
        self._fk: AxolForwardKinematics | None = None
        # Full IK solver, built lazily the first time a Cartesian action is sent
        # (run-policy). Collect-data commands joint targets, so it never builds
        # this; only the cheap forward-kinematics helper above runs there.
        self._ik: KinematicsSolver | None = None

    def _build_cameras(self) -> tuple[dict, list]:
        """Build the camera set, expanding any stereo camera into two eyes.

        The ``video_backend`` config selects the capture path. ``"gst"`` (or
        ``"auto"`` when the stack is installed) opens each camera through the
        GPU-resident zed-gstreamer pipeline (:mod:`almond_axol.video.gst_zed`):
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
            from ...video.gst_zed import zed_gst_available, zed_stereo_gst_available
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
        from ...video.gst_zed import ZedGstCamera, ZedGstStereoCamera

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

    def set_external_cameras(self, cameras: dict) -> None:
        """Replace the camera set with externally-owned cameras.

        Used by ``collect-data`` when the ZED cameras live in the out-of-process
        video relay (:mod:`almond_axol.video.video_proc`) and are exposed to this
        process as shared-memory readers (:mod:`almond_axol.video.shm_frames`).
        Must be called before :meth:`connect`: the robot then treats them as
        ordinary cameras (``read_at_or_after`` / ``read_latest``; ``connect`` is
        a no-op on a proxy) and never opens the physical devices itself, so the
        control process stays off the camera grab/encode path entirely.
        """
        if self._axol is not None:
            raise RuntimeError("set_external_cameras must be called before connect().")
        self.cameras = cameras
        self._stereo_cameras = []
        self._observation_features = None

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

        if self.config.observe_cartesian:
            # Each arm's 7 joint angles become a 6-axis EE pose; the gripper
            # position is kept (it has no Cartesian equivalent).
            state_keys = (
                _LEFT_EE_KEYS
                + [_LEFT_GRIPPER_KEY]
                + _RIGHT_EE_KEYS
                + [_RIGHT_GRIPPER_KEY]
            )
        else:
            state_keys = _LEFT_POS_KEYS + _RIGHT_POS_KEYS

        features: dict[str, type | tuple] = {key: float for key in state_keys}
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
            if self.config.observe_cartesian:
                # Mirror the observation: command each arm by a 6-axis EE pose
                # (resolved to joints via IK in send_action) plus gripper.
                keys = (
                    _LEFT_EE_KEYS
                    + [_LEFT_GRIPPER_KEY]
                    + _RIGHT_EE_KEYS
                    + [_RIGHT_GRIPPER_KEY]
                )
            else:
                keys = _LEFT_POS_KEYS + _RIGHT_POS_KEYS
            self._action_features = {key: float for key in keys}
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

        if self.config.observe_cartesian and self._fk is None:
            from ...kinematics.fk import AxolForwardKinematics

            self._fk = AxolForwardKinematics()

        for cam in self.cameras.values():
            cam.connect()

        _logger.info("AxolRobot connected.")

    def _build_hardware(self) -> Axol:
        """Construct the hardware driver. Overridden by the UMI rig subclass."""
        return Axol(
            self.config.axol_config,
            left_channel=self.config.left_channel,
            right_channel=self.config.right_channel,
        )

    async def _connect_async(self) -> None:
        self._axol = self._build_hardware()
        await self._axol.enable()
        if self.config.telemetry_hz > 0:
            await self._axol.start_telemetry(
                self.config.telemetry_hz, torque=self.config.observe_torques
            )
            await self._axol.wait_for_telemetry()
        else:
            # No background poll loop: rely on motion_control replies (every
            # impedance/gripper command returns a feedback frame) to keep the
            # position/torque cache fresh, exactly like `axol teleop`. This
            # removes ~telemetry_hz × 16 redundant CAN transactions/sec that
            # otherwise contend with motion_control on the bus and the loop.
            #
            # Seed the cache before the first cached read: get_positions() uses
            # register reads that return values but don't populate the .position
            # cache (only feedback frames do), and command sends are
            # fire-and-forget, so issue one hold-in-place motion_control to
            # elicit feedback from every motor and wait for those frames to land.
            pos_l, pos_r = await self._axol.get_positions()
            await self._axol.motion_control(left=pos_l, right=pos_r)
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
        self._fk = None
        self._ik = None
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

    @property
    def event_loop(self) -> asyncio.AbstractEventLoop:
        """The robot's asyncio event loop (CAN telemetry + motion control).

        ``collect-data`` runs its hot control loop *on* this loop so
        :meth:`send_action_async` awaits ``motion_control`` inline —
        cooperatively interleaved with telemetry on a single thread, exactly
        like ``axol teleop``. That removes the per-step cross-thread
        ``send_action`` round trip (``run_coroutine_threadsafe(...).result()``),
        which is what otherwise caps the data-collection control rate.
        """
        assert self._loop is not None, "connect() first"
        return self._loop

    # ------------------------------------------------------------------
    # Observation / action
    # ------------------------------------------------------------------

    def _joints_to_cartesian(
        self, left_pos: np.ndarray, right_pos: np.ndarray
    ) -> dict[str, float]:
        """Map per-arm joint positions to the Cartesian state/action dict.

        Runs forward kinematics on the 7 arm joints to get each end-effector's
        6-axis pose and keeps the gripper position. Shared by the cartesian
        observation (current joints) and the recorded cartesian action
        (commanded joints), so both stay in exactly the same representation.
        """
        assert self._fk is not None
        left_ee, right_ee = self._fk.ee_poses(left_pos, right_pos)
        out: dict[str, float] = {}
        for key, val in zip(_LEFT_EE_KEYS, left_ee):
            out[key] = float(val)
        out[_LEFT_GRIPPER_KEY] = float(left_pos[-1])
        for key, val in zip(_RIGHT_EE_KEYS, right_ee):
            out[key] = float(val)
        out[_RIGHT_GRIPPER_KEY] = float(right_pos[-1])
        return out

    def action_to_dataset(self, action: RobotAction) -> RobotAction:
        """Express a joint-position action in the configured action space.

        The teleop produces joint-position targets; in cartesian mode the
        *recorded* action must match :attr:`action_features`, so the joint
        targets are mapped through forward kinematics to per-arm end-effector
        poses (+ gripper). Identity when ``observe_cartesian`` is off. This does
        not touch what is commanded to the arm — only the value stored in the
        dataset — so teleop keeps its exact joint fidelity.
        """
        if not self.config.observe_cartesian:
            return action
        left = np.array([action[k] for k in _LEFT_POS_KEYS], dtype=np.float32)
        right = np.array([action[k] for k in _RIGHT_POS_KEYS], dtype=np.float32)
        return dict(self._joints_to_cartesian(left, right))

    def _joint_state(self) -> RobotObservation:
        """Build the non-camera part of an observation from the telemetry cache.

        Emits either the 16 joint positions (default) or, when
        ``observe_cartesian`` is set, each arm's 6-axis end-effector pose plus
        gripper position. Joint torques are appended when ``observe_torques`` is
        set, independent of the position representation. Keys match
        :attr:`observation_features`.
        """
        assert self._axol is not None
        assert self._axol.left is not None
        assert self._axol.right is not None

        left_pos = self._axol.left.positions
        right_pos = self._axol.right.positions

        obs: RobotObservation = {}
        if self.config.observe_cartesian:
            obs.update(self._joints_to_cartesian(left_pos, right_pos))
        else:
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
    def get_joint_observation(self) -> RobotObservation:
        """Return cached joint state only — no camera reads.

        Use this in the high-frequency teleop path to avoid copying large
        camera frames on every step.  Call :meth:`get_observation` only when
        a full observation (joint state + cameras) is actually needed.
        """
        return self._joint_state()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Return cached joint state and timestamp-aligned camera frames.

        Cameras are sampled with :meth:`ZedCamera.read_at_or_after` against a
        shared ``time.perf_counter()`` target so every frame in the
        observation shares the same capture instant — matching the alignment
        guarantee that ``collect-data`` writes into the training dataset. If a
        camera fails to produce a qualifying frame within ``timeout_ms``, we
        fall back to ``read_latest()`` so a single stale camera doesn't stall
        inference.
        """
        target_ts = time.perf_counter()

        obs = self._joint_state()

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

    def _ensure_ik(self) -> KinematicsSolver:
        """Lazily build the IK solver used to resolve Cartesian action targets.

        Built on first use rather than on connect so the joint-action paths
        (collect-data, teleop) never pay for the solver's URDF load, collision
        model, and IK JIT warmup.
        """
        if self._ik is None:
            from ...kinematics.solver import KinematicsSolver

            _logger.info("Building IK solver for Cartesian actions...")
            self._ik = KinematicsSolver()

            # The solver warms up its *with-elbow* IK graph, but our Cartesian
            # sends pass no elbow hint — a distinct graph that would otherwise
            # JIT-compile on the first real send, blocking the event loop past
            # send_action's timeout (and clogging it for the rest of the run).
            # Compile that exact no-elbow variant here, on the caller thread,
            # with a dummy reachable target. Best-effort: warmup only compiles.
            dummy_pose = (
                np.array([0.0, 0.0, 0.3], dtype=np.float32),
                np.eye(3, dtype=np.float32),
            )
            q0 = np.zeros(self._ik.num_joints, dtype=np.float32)
            try:
                self._ik.ik(q0, left_pose=dummy_pose, right_pose=dummy_pose)
            except Exception:  # noqa: BLE001 - warmup just triggers compilation
                _logger.warning("Cartesian IK warmup failed", exc_info=True)
            _logger.info("IK solver ready for Cartesian actions.")
        return self._ik

    def prepare_cartesian_actions(self) -> None:
        """Pre-build the Cartesian-action IK solver before the control loop runs.

        ``send_action`` builds the solver lazily on the first Cartesian action,
        but its URDF load + JIT warmup (tens of seconds) would otherwise stall
        the caller's real-time control loop on that first action. Consumers that
        will stream Cartesian actions (run-policy, replay-dataset) call this once
        after :meth:`connect` — overlapping the build with the policy load /
        return-to-rest — so the first action dispatches immediately. A no-op once
        built, and never called by collect-data, which commands joint targets and
        so never needs IK.
        """
        self._ensure_ik()

    def _cartesian_action_to_targets(
        self, action: RobotAction
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve a Cartesian action to per-arm joint targets via IK.

        Each arm's 6-axis end-effector pose is solved to joint angles, seeded
        with the arm's current cached position so the solve tracks from where
        the arm actually is. The gripper passes straight through. Returns
        ``(left, right)`` 8-vectors (7 arm joints + gripper) in Joint order.
        """
        from ...kinematics.fk import pose6_to_pos_rot

        solver = self._ensure_ik()
        assert self._axol is not None
        assert self._axol.left is not None
        assert self._axol.right is not None

        left_cur = self._axol.left.positions
        right_cur = self._axol.right.positions
        q = np.zeros(solver.num_joints, dtype=np.float32)
        for i, gi in enumerate(solver.left_indices):
            q[gi] = left_cur[i]
        for i, gi in enumerate(solver.right_indices):
            q[gi] = right_cur[i]

        left_pose = pose6_to_pos_rot(np.array([action[k] for k in _LEFT_EE_KEYS]))
        right_pose = pose6_to_pos_rot(np.array([action[k] for k in _RIGHT_EE_KEYS]))
        q_out = solver.ik(q, left_pose=left_pose, right_pose=right_pose)

        left = np.empty(len(_JOINTS), dtype=np.float32)
        right = np.empty(len(_JOINTS), dtype=np.float32)
        for i, gi in enumerate(solver.left_indices):
            left[i] = q_out[gi]
        for i, gi in enumerate(solver.right_indices):
            right[i] = q_out[gi]
        left[-1] = action[_LEFT_GRIPPER_KEY]
        right[-1] = action[_RIGHT_GRIPPER_KEY]
        return left, right

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Send an action to the arms, running IK first for Cartesian actions.

        Accepts either joint-position targets or — when ``observe_cartesian``
        is on and the policy emits them — per-arm Cartesian end-effector poses
        (+ gripper), which are resolved to joint targets via inverse
        kinematics. Either way the arm joints go out via impedance control and
        the gripper via position-force control.

        Synchronous wrapper for callers not running on :attr:`event_loop`
        (e.g. ``run-policy``): it hops to the robot's loop and blocks on the
        result. The high-rate ``collect-data`` path uses
        :meth:`send_action_async` instead to avoid that cross-thread wait.

        Args:
            action: Dict with keys matching action_features, values in radians
                (joint mode) or metres + axis-angle radians (Cartesian mode).

        Returns:
            The action as sent (unmodified).
        """
        assert self._loop is not None

        # Build the IK solver here, on the caller's thread, so the one-time
        # URDF load + JIT warmup never blocks the robot's event loop (telemetry).
        # A Cartesian send then runs a (warmed) IK solve inline on the loop
        # before motion_control, so give it more headroom than a bare joint
        # command — the solve is milliseconds warmed, but loop contention or a
        # cold path shouldn't surface as a spurious timeout that aborts replay.
        is_cartesian = _LEFT_EE_KEYS[0] in action
        if is_cartesian:
            self._ensure_ik()

        asyncio.run_coroutine_threadsafe(
            self.send_action_async(action), self._loop
        ).result(timeout=5.0 if is_cartesian else 1.0)

        return action

    async def send_action_async(self, action: RobotAction) -> RobotAction:
        """Await ``motion_control`` directly on the robot's event loop.

        Must be awaited from a coroutine already running on
        :attr:`event_loop`. Unlike :meth:`send_action` this performs no thread
        hop: the control loop and CAN telemetry share one loop, so the command
        is dispatched inline (cooperatively with telemetry) with no
        cross-thread ``.result()`` block.

        A Cartesian action (per-arm EE pose, as emitted by a cartesian policy)
        is resolved to joint targets via IK; a joint action — what teleop and
        collect-data always send — is dispatched directly with no IK. The two
        are told apart by their keys, so collect-data never triggers a solve.

        Args:
            action: Dict with keys matching action_features.

        Returns:
            The action as sent (unmodified).
        """
        assert self._axol is not None

        if _LEFT_EE_KEYS[0] in action:
            left, right = self._cartesian_action_to_targets(action)
        else:
            left = np.array([action[k] for k in _LEFT_POS_KEYS], dtype=np.float32)
            right = np.array([action[k] for k in _RIGHT_POS_KEYS], dtype=np.float32)

        await self._axol.motion_control(left=left, right=right)

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
