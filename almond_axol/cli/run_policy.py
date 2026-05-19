"""
axol run-policy

Run a trained policy on the Axol robot with three ZED cameras using
LeRobot's async inference (``lerobot.async_inference``).

Architecture
============
A ``PolicyServer`` is auto-launched in a child process on localhost. The
parent process drives an ``AxolRobotClient`` (a thin subclass of LeRobot's
``RobotClient``) that streams observations to the server and consumes the
action chunks the server returns. Cameras + joints are sampled via
``ZedCamera.read_at_or_after(now)`` (see ``AxolRobot.get_observation``) so
each inference observation is global-timestamp aligned the same way the
training data is.

Episode termination
===================
Each episode runs until the operator presses a key:

    s  → end + save  (writes the rollout to ``--repo-id`` when set)
    r  → end + rerecord (discard buffer)
    q  → end + quit  (discard buffer, exit ``axol run-policy``)

``--episode-time-s`` is kept as a safety cap: when it fires the operator
gets the same ``[Enter]=save / r / q`` prompt as before.
"""

from __future__ import annotations

import argparse
import logging
import socket
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

from ..shared import ARM_JOINTS, parse_stiffness

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.types import RobotAction

    from ..lerobot.robot.robot_axol import AxolRobot

_logger = logging.getLogger(__name__)


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("run-policy", help="Run a trained policy on the robot.")
    p.add_argument(
        "--policy",
        required=True,
        help="Local path or HuggingFace repo ID of the trained policy checkpoint.",
    )
    p.add_argument(
        "--policy-type",
        required=True,
        choices=[
            "act",
            "smolvla",
            "diffusion",
            "tdmpc",
            "vqbet",
            "pi0",
            "pi05",
            "groot",
        ],
        help=(
            "Policy architecture as registered in lerobot.async_inference "
            "(must match the checkpoint at --policy)."
        ),
    )
    p.add_argument("--task", required=True, help="Natural language task description.")
    p.add_argument(
        "--episode-time-s",
        type=int,
        default=120,
        help=(
            "Safety cap on episode duration in seconds (default: 120). "
            "Episodes normally end on operator keypress; this only fires if "
            "the operator never signals s/r/q."
        ),
    )
    p.add_argument(
        "--fps",
        type=int,
        default=60,
        help=(
            "Control loop frame rate (default: 60, matching collect-data). "
            "Must match the fps the policy was trained on."
        ),
    )
    p.add_argument(
        "--repo-id",
        default=None,
        help="HuggingFace dataset repo ID to save rollouts (<user>/<dataset>). Optional.",
    )
    p.add_argument(
        "--root",
        default=None,
        help="Local dataset root path (default: HF_LEROBOT_HOME).",
    )
    p.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push rollout dataset to HuggingFace Hub when done.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Torch device for policy inference (default: cuda).",
    )
    p.add_argument(
        "--server-port",
        type=int,
        default=8765,
        help="Port for the localhost PolicyServer child process (default: 8765).",
    )
    p.add_argument(
        "--actions-per-chunk",
        type=int,
        default=50,
        help=(
            "Number of actions returned per inference call (default: 50). "
            "Capped by the policy's max action horizon."
        ),
    )
    p.add_argument(
        "--chunk-size-threshold",
        type=float,
        default=0.5,
        help=(
            "Send a fresh observation to the server when the local action queue "
            "drops to this fraction of a full chunk (default: 0.5)."
        ),
    )
    p.add_argument(
        "--aggregate-fn",
        default="weighted_average",
        choices=["weighted_average", "latest_only", "average", "conservative"],
        help="Action chunk aggregation function (default: weighted_average).",
    )
    p.add_argument(
        "--zed-host",
        default="192.168.10.1",
        help="IP address of the ZED streamer (default: 192.168.10.1).",
    )
    p.add_argument(
        "--zed-iface",
        default=None,
        metavar="IFACE",
        help=(
            "Network interface to configure for the ZED link before connecting "
            "(e.g. eth0). Assigns 192.168.10.2/24 and requires sudo. "
            "Skip if the interface is already configured."
        ),
    )
    p.add_argument(
        "--left-gripper-torque-limit",
        type=float,
        default=1.0,
        help="Max output torque (Nm) for the left gripper in POSITION_FORCE mode (default: 1.0).",
    )
    p.add_argument(
        "--right-gripper-torque-limit",
        type=float,
        default=1.0,
        help="Max output torque (Nm) for the right gripper in POSITION_FORCE mode (default: 1.0).",
    )
    stiffness_help = (
        "Compliance ↔ stiffness blend in [0, 1] for the {side} arm. "
        f"Either a single value applied to all {len(ARM_JOINTS)} joints, "
        f"or {len(ARM_JOINTS)} comma-separated values (one per joint, in "
        f"order: {', '.join(j.value for j in ARM_JOINTS)}; gripper "
        "excluded). 0 (default) is fully compliant; 1 restores the "
        "pre-tuning industrial gains. See AxolConfig.{attr}. Should match "
        "the values used at data collection time."
    )
    stiffness_metavar = "S|" + ",".join("S" for _ in ARM_JOINTS)
    p.add_argument(
        "--left-stiffness",
        type=parse_stiffness,
        default=0.0,
        metavar=stiffness_metavar,
        help=stiffness_help.format(side="left", attr="left_stiffness"),
    )
    p.add_argument(
        "--right-stiffness",
        type=parse_stiffness,
        default=0.0,
        metavar=stiffness_metavar,
        help=stiffness_help.format(side="right", attr="right_stiffness"),
    )
    p.add_argument(
        "--rerun-ip",
        default=None,
        help=(
            "IP of a Rerun viewer running on your local machine. "
            "When set, streams live visualization to that viewer. "
            "On the local machine run: rerun --connect rerun+http://<robot-ip>:<port>/proxy"
        ),
    )
    p.add_argument(
        "--rerun-port",
        type=int,
        default=9876,
        help="Port of the Rerun viewer (default: 9876). Only used when --rerun-ip is set.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level))
    _run(
        policy_path=args.policy,
        policy_type=args.policy_type,
        task=args.task,
        episode_time_s=args.episode_time_s,
        fps=args.fps,
        repo_id=args.repo_id,
        root=args.root,
        push_to_hub=args.push_to_hub,
        device=args.device,
        server_port=args.server_port,
        actions_per_chunk=args.actions_per_chunk,
        chunk_size_threshold=args.chunk_size_threshold,
        aggregate_fn=args.aggregate_fn,
        zed_host=args.zed_host,
        zed_iface=args.zed_iface,
        left_gripper_torque_limit=args.left_gripper_torque_limit,
        right_gripper_torque_limit=args.right_gripper_torque_limit,
        left_stiffness=args.left_stiffness,
        right_stiffness=args.right_stiffness,
        rerun_ip=args.rerun_ip,
        rerun_port=args.rerun_port,
    )


class _IKResetController:
    """Collision-aware return-to-rest, backed by an IK worker subprocess.

    Mirrors the reset path used by ``AxolVRTeleop`` (collect-data) but
    without the VR server. ``start()`` spawns ``run_ik_worker`` in a
    ``spawn``-mode subprocess; it imports JAX, JITs the IK solver
    (~10-20 s), and reports ``("ready", ...)``. ``wait_ready()`` blocks on
    that message. ``return_to_rest()`` then asks the worker for a
    collision-aware joint-space trajectory from the current pose to the
    rest pose and plays it back through a ``ResetInterpolator`` while
    streaming each waypoint to the impedance controller.

    Spawn it before ``client.start()`` so the IK JIT overlaps with the
    policy load — by the time we actually need a rest move, the worker is
    typically already ready.
    """

    def __init__(self) -> None:
        from ..kinematics.config import KinematicsConfig
        from ..teleop.config import VRTeleopConfig

        self._vr_cfg = VRTeleopConfig()
        self._kin_cfg = KinematicsConfig()
        self._proc: Any | None = None
        self._conn: Any | None = None
        self._q_init: Any | None = None
        self._left_indices: list[int] | None = None
        self._right_indices: list[int] | None = None
        self._ready = False

    def start(self) -> None:
        """Spawn the IK worker subprocess. Non-blocking — call wait_ready()."""
        import multiprocessing as mp

        from ..teleop.worker import run_ik_worker

        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(
            target=run_ik_worker,
            args=(child_conn, self._vr_cfg, self._kin_cfg, None, None),
            name="axol-ik-worker",
            daemon=True,
        )
        proc.start()
        child_conn.close()
        self._proc = proc
        self._conn = parent_conn

    def wait_ready(self, timeout: float = 60.0) -> None:
        """Block until the IK worker has finished JIT compilation."""
        if self._ready:
            return
        if self._conn is None:
            raise RuntimeError("IK reset controller not started")
        if not self._conn.poll(timeout):
            raise TimeoutError(
                f"IK worker did not become ready within {timeout:.1f}s "
                "(JAX JIT compilation may have stalled)."
            )
        msg = self._conn.recv()
        if not (isinstance(msg, tuple) and msg[0] == "ready"):
            raise RuntimeError(f"Unexpected IK worker handshake: {msg!r}")
        import numpy as np

        _, q_init, left_indices, right_indices, _startup_traj = msg
        self._q_init = np.asarray(q_init, dtype=np.float32)
        self._left_indices = [int(i) for i in left_indices]
        self._right_indices = [int(i) for i in right_indices]
        self._ready = True

    def return_to_rest(self, robot: "AxolRobot") -> None:
        """Plan and play a collision-aware trajectory to the rest pose."""
        import numpy as np

        from ..shared import Joint
        from ..teleop.filter import ResetInterpolator

        self.wait_ready()
        assert self._conn is not None
        assert self._q_init is not None
        assert self._left_indices is not None
        assert self._right_indices is not None

        pos_l, pos_r = robot.positions
        pos_l = np.asarray(pos_l, dtype=np.float32)
        pos_r = np.asarray(pos_r, dtype=np.float32)

        q_current = self._q_init.copy()
        for i, gi in enumerate(self._left_indices):
            q_current[gi] = float(pos_l[i])
        for i, gi in enumerate(self._right_indices):
            q_current[gi] = float(pos_r[i])

        self._conn.send(("reset", q_current))
        result = self._conn.recv()
        if not (isinstance(result, tuple) and result[0] == "reset_traj"):
            raise RuntimeError(f"Unexpected IK worker response: {result!r}")
        _, _q_rest, traj = result
        if not traj:
            _logger.warning("IK worker returned an empty reset trajectory; skipping.")
            return

        interp = ResetInterpolator()
        interp.set_trajectory(traj, float(pos_l[7]), float(pos_r[7]))

        joints = list(Joint)
        play_hz = float(self._vr_cfg.frequency)
        period = 1.0 / play_hz
        while interp.is_active():
            t0 = time.perf_counter()
            new_q, l_grip, r_grip, _done = interp.step()
            if new_q is None:
                break
            arm_left = np.asarray(new_q)[self._left_indices]
            arm_right = np.asarray(new_q)[self._right_indices]
            action: dict[str, float] = {}
            for j in joints:
                if j in ARM_JOINTS:
                    ai = ARM_JOINTS.index(j)
                    action[f"left_{j.value}.pos"] = float(arm_left[ai])
                    action[f"right_{j.value}.pos"] = float(arm_right[ai])
                else:
                    action[f"left_{j.value}.pos"] = float(l_grip)
                    action[f"right_{j.value}.pos"] = float(r_grip)
            robot.send_action(action)
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def stop(self) -> None:
        """Signal shutdown, close the pipe, and reap the subprocess."""
        if self._conn is not None:
            try:
                self._conn.send(None)
            except Exception:  # noqa: BLE001
                pass
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None
        if self._proc is not None:
            self._proc.join(timeout=3.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.kill()
            self._proc = None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


class _ActionPublisher:
    """Thread-safe single-slot publisher for the most recently executed action.

    Updated by ``AxolRobotClient.control_loop_action`` after every
    ``robot.send_action`` call, read by ``_RolloutCaptureThread`` to pair
    each dataset frame with the action that drove the robot at that tick.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: "RobotAction | None" = None
        self._first_event = threading.Event()

    def publish(self, action: "RobotAction") -> None:
        snap = dict(action)
        with self._lock:
            self._latest = snap
        self._first_event.set()

    def latest(self) -> "RobotAction | None":
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def wait_for_first(self, timeout: float) -> bool:
        return self._first_event.wait(timeout=timeout)

    def reset(self) -> None:
        with self._lock:
            self._latest = None
        self._first_event.clear()


class _RolloutCaptureThread(threading.Thread):
    """Tick at ``fps`` Hz, sample one global-timestamp-aligned observation per
    tick, pair it with the latest executed action, and append a dataset row.
    """

    def __init__(
        self,
        *,
        publisher: _ActionPublisher,
        robot: "AxolRobot",
        dataset: "LeRobotDataset",
        robot_obs_proc: Callable[[Any], Any],
        fps: int,
        task: str,
        rerun_ip: str | None,
    ) -> None:
        super().__init__(name="axol-rollout-capture", daemon=True)
        self.publisher = publisher
        self.robot = robot
        self.dataset = dataset
        self.robot_obs_proc = robot_obs_proc
        self.fps = fps
        self.task = task
        self.rerun_ip = rerun_ip
        self.stop_event = threading.Event()

    def run(self) -> None:
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame
        from lerobot.utils.visualization_utils import log_rerun_data

        if not self.publisher.wait_for_first(timeout=10.0):
            _logger.warning(
                "Rollout capture thread saw no action snapshot within 10s; exiting."
            )
            return
        if self.stop_event.is_set():
            return

        frame_interval = 1.0 / self.fps
        recording_start = time.perf_counter()
        tick = 0

        while not self.stop_event.is_set():
            target_perf_ts = recording_start + tick * frame_interval

            wait_s = target_perf_ts - time.perf_counter()
            if wait_s > 0 and self.stop_event.wait(timeout=wait_s):
                return

            try:
                obs = self.robot.get_observation()
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    "Capture tick %d: get_observation failed (%s).", tick, exc
                )
                tick += 1
                continue

            action = self.publisher.latest()
            if action is None:
                tick += 1
                continue

            obs_processed = self.robot_obs_proc(obs)
            obs_frame = build_dataset_frame(
                self.dataset.features, obs_processed, prefix=OBS_STR
            )
            act_frame = build_dataset_frame(
                self.dataset.features, action, prefix=ACTION
            )
            if self.stop_event.is_set():
                return
            self.dataset.add_frame({**obs_frame, **act_frame, "task": self.task})

            if self.rerun_ip:
                log_rerun_data(observation=obs_processed, action=action)

            tick += 1


def _stdin_watcher(
    stop_event: threading.Event,
    result: dict[str, str | None],
) -> None:
    """Watch stdin for ``s`` / ``r`` / ``q`` on its own line.

    Uses ``select.select`` so it never blocks past the stop event. Sets
    ``result["choice"]`` to the first valid keystroke received.
    """
    import select
    import sys

    while not stop_event.is_set():
        ready, _, _ = select.select([sys.stdin], [], [], 0.25)
        if not ready:
            continue
        line = sys.stdin.readline()
        if not line:
            return
        ch = line.strip().lower()
        if ch in ("s", "r", "q"):
            result["choice"] = ch
            return


def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> None:
    """Block until ``host:port`` accepts a TCP connection or ``timeout`` elapses."""
    deadline = time.perf_counter() + timeout
    last_exc: Exception | None = None
    while time.perf_counter() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as exc:
            last_exc = exc
            time.sleep(0.25)
    raise TimeoutError(
        f"PolicyServer at {host}:{port} did not become reachable within "
        f"{timeout:.1f}s (last error: {last_exc!r})."
    )


def _serve_policy_server(server_cfg_dict: dict[str, Any]) -> None:
    """Entry point for the policy-server child process.

    Lives in the parent module so it's picklable by ``mp.get_context('spawn')``
    on macOS/Linux. Re-imports lerobot inside the child to avoid sharing
    CUDA/JAX state with the parent.
    """
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server import serve

    serve(PolicyServerConfig(**server_cfg_dict))


# ----------------------------------------------------------------------
# AxolRobotClient: thin RobotClient subclass that reuses our connected robot
# ----------------------------------------------------------------------


def _build_axol_robot_client(
    *, config: Any, robot: "AxolRobot", publisher: _ActionPublisher
) -> Any:
    """Construct an ``AxolRobotClient`` against an already-connected robot.

    Defined as a helper so the heavy lerobot imports happen lazily inside
    ``_run`` and we don't pay the import cost at CLI parse time.
    """
    import threading as _threading
    from queue import Queue

    import grpc
    from lerobot.async_inference.helpers import (
        FPSTracker,
        RemotePolicyConfig,
        map_robot_keys_to_lerobot_features,
    )
    from lerobot.async_inference.robot_client import RobotClient
    from lerobot.transport import services_pb2_grpc
    from lerobot.transport.utils import grpc_channel_options

    class AxolRobotClient(RobotClient):  # type: ignore[misc, valid-type]
        """RobotClient that reuses a pre-connected AxolRobot.

        - Skips ``make_robot_from_config(...)`` and ``robot.connect()`` so we
          don't pay the camera reconnect cost between episodes.
        - Publishes each executed action to ``_ActionPublisher`` for the
          dataset capture thread.
        - Overrides ``stop()`` to tear down the gRPC channel without
          disconnecting the shared robot.
        """

        def __init__(self, config, robot, publisher):  # type: ignore[no-untyped-def]
            self.config = config
            self.robot = robot
            self._publisher = publisher

            lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

            self.server_address = config.server_address
            self.policy_config = RemotePolicyConfig(
                config.policy_type,
                config.pretrained_name_or_path,
                lerobot_features,
                config.actions_per_chunk,
                config.policy_device,
            )

            self.channel = grpc.insecure_channel(
                self.server_address,
                grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s"),
            )
            self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
            self.logger = RobotClient.logger
            self.logger.info(
                f"AxolRobotClient connecting to server at {self.server_address}"
            )

            self.shutdown_event = _threading.Event()
            self.latest_action_lock = _threading.Lock()
            self.latest_action = -1
            self.action_chunk_size = -1
            self._chunk_size_threshold = config.chunk_size_threshold
            self.action_queue = Queue()
            self.action_queue_lock = _threading.Lock()
            self.action_queue_size = []
            self.start_barrier = _threading.Barrier(2)
            self.fps_tracker = FPSTracker(target_fps=self.config.fps)
            self.must_go = _threading.Event()
            self.must_go.set()

        def reset_episode_state(self) -> None:
            """Reset queues + flags so threads can run a fresh episode.

            Recreates ``start_barrier`` (so two new threads can synchronize)
            and clears any leftover action queue entries.
            """
            with self.action_queue_lock:
                self.action_queue = Queue()
                self.action_queue_size = []
            with self.latest_action_lock:
                self.latest_action = -1
            self.action_chunk_size = -1
            self.must_go.set()
            self.fps_tracker.reset()
            self.shutdown_event.clear()
            self.start_barrier = _threading.Barrier(2)
            if self._publisher is not None:
                self._publisher.reset()

        def control_loop_action(self, verbose: bool = False):  # type: ignore[no-untyped-def]
            performed = super().control_loop_action(verbose)
            if self._publisher is not None and performed is not None:
                self._publisher.publish(performed)
            return performed

        def stop(self) -> None:  # type: ignore[override]
            self.shutdown_event.set()
            try:
                self.channel.close()
            except Exception:  # noqa: BLE001
                pass
            self.logger.debug("AxolRobotClient channel closed (robot left connected)")

    return AxolRobotClient(config, robot, publisher)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def _run(
    policy_path: str,
    policy_type: str,
    task: str,
    episode_time_s: int,
    fps: int,
    repo_id: str | None,
    root: str | None,
    push_to_hub: bool,
    device: str,
    server_port: int = 8765,
    actions_per_chunk: int = 50,
    chunk_size_threshold: float = 0.5,
    aggregate_fn: str = "weighted_average",
    zed_host: str = "192.168.10.1",
    zed_iface: str | None = None,
    left_gripper_torque_limit: float = 1.0,
    right_gripper_torque_limit: float = 1.0,
    left_stiffness: float | tuple[float, ...] = 0.0,
    right_stiffness: float | tuple[float, ...] = 0.0,
    rerun_ip: str | None = None,
    rerun_port: int = 9876,
) -> None:
    import multiprocessing as mp
    import shutil
    from pathlib import Path

    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor import make_default_processors
    from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
    from lerobot.utils.feature_utils import hw_to_dataset_features
    from lerobot.utils.utils import log_say
    from lerobot.utils.visualization_utils import init_rerun

    from ..lerobot.camera.configuration_zed import ZedCameraConfig
    from ..lerobot.robot.config_axol import AxolRobotConfig
    from ..lerobot.robot.robot_axol import AxolRobot
    from ..robot.config import AxolConfig
    from ..shared import setup_link_ip

    if zed_iface:
        setup_link_ip(zed_iface, "192.168.10.2/24")

    axol_config = AxolConfig(
        left_stiffness=left_stiffness,
        right_stiffness=right_stiffness,
    )
    axol_config.left.gripper.torque_limit = left_gripper_torque_limit
    axol_config.right.gripper.torque_limit = right_gripper_torque_limit

    # Build robot with 3 ZED cameras — resolution/FPS auto-detected from stream
    robot_config = AxolRobotConfig(
        cameras={
            "overhead": ZedCameraConfig(host=zed_host, port=30000),
            "left_arm": ZedCameraConfig(host=zed_host, port=30002),
            "right_arm": ZedCameraConfig(host=zed_host, port=30004),
        },
        axol_config=axol_config,
    )
    robot = AxolRobot(robot_config)
    _, robot_action_proc, robot_obs_proc = make_default_processors()

    # Dataset features are derived from the camera configs (width/height) and
    # joint enum, both static, so the dataset can be built before the robot
    # is connected. This lets us load the policy first (see below).
    dataset: "LeRobotDataset | None" = None
    dataset_root: Path | None = None
    if repo_id:
        dataset_root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        meta = dataset_root / "meta"
        has_info = (meta / "info.json").exists()
        if has_info:
            # A previous run-policy invocation saved rollouts here — refuse
            # to overwrite. (Unlike collect-data, run-policy doesn't resume:
            # each invocation writes a fresh batch of rollouts.)
            raise RuntimeError(
                f"Rollout dataset already exists at {dataset_root}. "
                f"Delete the directory and rerun to start fresh:\n"
                f"  rm -rf {dataset_root}"
            )
        if dataset_root.exists():
            # Empty leftover from a previous failed run — wipe so
            # LeRobotDataset.create can mkdir(exist_ok=False).
            log_say(f"Removing empty dataset directory at {dataset_root}.")
            shutil.rmtree(dataset_root)

        action_features = hw_to_dataset_features(robot.action_features, ACTION)
        obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            features={**action_features, **obs_features},
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
        )

    if rerun_ip:
        init_rerun(session_name="axol_run_policy", ip=rerun_ip, port=rerun_port)

    # --- Launch PolicyServer in a child process on localhost --------------
    # Important: we spawn the server and fully load the policy BEFORE
    # connecting cameras. The model download + CUDA load is a ~15s spike of
    # network + GPU activity; if the ZED streams are already open during
    # that window the streams can get briefly disrupted, and (without async
    # recovery) ``grab()`` blocks indefinitely waiting for the connection
    # to recover. Connecting cameras after the policy is ready avoids
    # walking into that contention window in the first place.
    server_cfg_dict = {
        "host": "127.0.0.1",
        "port": server_port,
        "fps": fps,
    }
    ctx = mp.get_context("spawn")
    server_proc = ctx.Process(
        target=_serve_policy_server,
        args=(server_cfg_dict,),
        name="axol-policy-server",
        daemon=True,
    )
    server_proc.start()
    log_say(f"Started PolicyServer on 127.0.0.1:{server_port} (pid={server_proc.pid}).")

    # Spawn the IK worker in parallel with the policy server so JAX JIT
    # compilation overlaps with policy load. The worker plans collision-aware
    # return-to-rest trajectories; we wait for the "ready" handshake on first
    # use (return_to_rest).
    reset_controller = _IKResetController()
    reset_controller.start()
    log_say("Started IK reset worker (collision-aware return-to-rest).")

    client = None
    episodes_recorded = 0
    robot_connected = False
    try:
        _wait_for_port("127.0.0.1", server_port, timeout=30.0)

        client_cfg = RobotClientConfig(
            robot=robot_config,
            policy_type=policy_type,
            pretrained_name_or_path=policy_path,
            actions_per_chunk=actions_per_chunk,
            task=task,
            server_address=f"127.0.0.1:{server_port}",
            policy_device=device,
            client_device="cpu",
            chunk_size_threshold=chunk_size_threshold,
            fps=fps,
            aggregate_fn_name=aggregate_fn,
        )
        publisher = _ActionPublisher()
        client = _build_axol_robot_client(
            config=client_cfg, robot=robot, publisher=publisher
        )

        log_say("Loading policy on server (one-time)...")
        if not client.start():
            raise RuntimeError("Failed to connect to policy server / load policy.")

        # Policy is now resident on cuda — safe to connect cameras.
        log_say("Connecting robot...")
        robot.connect()
        robot_connected = True

        log_say("Returning to rest pose.")
        reset_controller.return_to_rest(robot)
        try:
            input("Reset the scene, then press Enter to start the first episode.")
        except (EOFError, KeyboardInterrupt):
            return

        while True:
            log_say(f"Episode {episodes_recorded + 1}: starting in 1s.")
            time.sleep(1.0)

            if dataset is not None:
                dataset.clear_episode_buffer()

            client.reset_episode_state()
            publisher.reset()

            receiver_thread = threading.Thread(
                target=client.receive_actions,
                name="axol-recv-actions",
                daemon=True,
            )
            control_thread = threading.Thread(
                target=client.control_loop,
                args=(task,),
                name="axol-control-loop",
                daemon=True,
            )

            capture: _RolloutCaptureThread | None = None
            if dataset is not None:
                capture = _RolloutCaptureThread(
                    publisher=publisher,
                    robot=robot,
                    dataset=dataset,
                    robot_obs_proc=robot_obs_proc,
                    fps=fps,
                    task=task,
                    rerun_ip=rerun_ip,
                )

            stdin_stop = threading.Event()
            stdin_result: dict[str, str | None] = {"choice": None}
            stdin_thread = threading.Thread(
                target=_stdin_watcher,
                args=(stdin_stop, stdin_result),
                name="axol-stdin-watcher",
                daemon=True,
            )

            print(
                f"  Press s=save+end, r=rerecord+end, q=quit "
                f"(safety cap {episode_time_s}s).",
                flush=True,
            )

            receiver_thread.start()
            control_thread.start()
            if capture is not None:
                capture.start()
            stdin_thread.start()

            deadline = time.perf_counter() + episode_time_s
            timed_out = False
            interrupted = False
            try:
                while True:
                    if stdin_result["choice"] is not None:
                        break
                    if time.perf_counter() >= deadline:
                        timed_out = True
                        break
                    if not control_thread.is_alive() and not receiver_thread.is_alive():
                        # Both inference threads died unexpectedly; abort
                        # the episode and surface to the operator.
                        _logger.warning(
                            "Control/receiver threads exited before any "
                            "end signal; aborting episode."
                        )
                        break
                    time.sleep(0.1)
            except KeyboardInterrupt:
                interrupted = True

            # Tear down per-episode threads (server + client stay alive).
            stdin_stop.set()
            client.shutdown_event.set()
            if capture is not None:
                capture.stop_event.set()
                capture.join(timeout=5.0)
            control_thread.join(timeout=5.0)
            receiver_thread.join(timeout=5.0)
            # The stdin watcher blocks on select with a short timeout, so
            # it will wake up on its own; don't join (it may still be in
            # select if the user never typed anything).

            if interrupted:
                break

            choice = stdin_result["choice"]
            if timed_out:
                try:
                    raw = input(
                        f"Episode time cap ({episode_time_s}s) reached. "
                        "[Enter]=save, r=rerecord, q=quit: "
                    )
                except (EOFError, KeyboardInterrupt):
                    break
                raw = raw.strip().lower()
                choice = "q" if raw == "q" else ("r" if raw == "r" else "s")

            if choice == "q":
                if dataset is not None:
                    dataset.clear_episode_buffer()
                break

            if choice == "r":
                log_say("Re-recording episode.")
                if dataset is not None:
                    dataset.clear_episode_buffer()
                log_say("Returning to rest pose.")
                reset_controller.return_to_rest(robot)
                try:
                    input("Reset the scene, then press Enter to start.")
                except (EOFError, KeyboardInterrupt):
                    break
                continue

            # choice == "s"
            if dataset is not None:
                dataset.save_episode()
            episodes_recorded += 1
            log_say(f"Saved episode {episodes_recorded}.")
            log_say("Returning to rest pose.")
            reset_controller.return_to_rest(robot)
            try:
                input("Reset the scene, then press Enter to start the next episode.")
            except (EOFError, KeyboardInterrupt):
                break

    except KeyboardInterrupt:
        pass
    finally:
        log_say("Stopping.")
        if client is not None:
            try:
                client.stop()
            except Exception:  # noqa: BLE001
                pass
        if robot_connected:
            try:
                robot.disconnect()
            except Exception:  # noqa: BLE001
                pass

        try:
            reset_controller.stop()
        except Exception:  # noqa: BLE001
            pass

        # Tear down the server child process.
        if server_proc.is_alive():
            server_proc.terminate()
            server_proc.join(timeout=5.0)
            if server_proc.is_alive():
                server_proc.kill()
                server_proc.join(timeout=2.0)

        if dataset is not None:
            dataset.finalize()
            if push_to_hub and episodes_recorded > 0:
                dataset.push_to_hub()

        if (
            dataset_root is not None
            and episodes_recorded == 0
            and dataset_root.exists()
        ):
            try:
                shutil.rmtree(dataset_root)
                log_say(f"No episodes saved — removed empty dataset at {dataset_root}.")
            except OSError as exc:
                _logger.warning(
                    "Failed to remove empty dataset at %s: %s", dataset_root, exc
                )
