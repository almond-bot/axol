"""
axol run-policy

Run a trained policy on the Axol robot with three ZED cameras using
LeRobot's async inference (``lerobot.async_inference``). A ``PolicyServer``
is auto-launched in a child process on localhost and the parent drives an
``AxolRobotClient`` (a thin ``RobotClient`` subclass) that streams
observations to it and consumes the returned action chunks. Cameras and
joints are sampled via ``ZedCamera.read_at_or_after(now)`` so every
inference observation is global-timestamp aligned the same way the
training data is (see ``AxolRobot.get_observation``).

Each episode runs until the operator types ``s`` (save), ``r`` (rerecord
+ discard), or ``q`` (quit + discard) on stdin. ``--episode-time-s`` is a
safety cap that falls back to the same ``[Enter]=save / r / q`` prompt
when no key has been pressed.
"""

from __future__ import annotations

import argparse
import logging
import socket
import threading
import time
from typing import TYPE_CHECKING, Any

from ..lerobot.rollout import (
    ActionPublisher,
    IKResetController,
    RolloutCaptureThread,
    stdin_watcher,
)
from ..shared import ARM_JOINTS, parse_stiffness

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

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
        help="Policy architecture; must match the checkpoint at --policy.",
    )
    p.add_argument("--task", required=True, help="Natural language task description.")
    p.add_argument(
        "--episode-time-s",
        type=int,
        default=120,
        help=(
            "Safety cap on episode duration in seconds (default: 120). "
            "Episodes normally end on operator keypress (s/r/q)."
        ),
    )
    p.add_argument(
        "--fps",
        type=int,
        default=60,
        help=(
            "Control loop frame rate (default: 60). Must match the fps "
            "the policy was trained on."
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
        default=0.9,
        help=(
            "Trigger a fresh observation when the action queue drops to "
            "this fraction of a full chunk (default: 0.9). Higher than "
            "upstream's 0.5 because obs send runs on its own thread here."
        ),
    )
    p.add_argument(
        "--aggregate-fn",
        default="temporal_ensemble",
        choices=[
            "temporal_ensemble",
            "weighted_average",
            "latest_only",
            "average",
            "conservative",
        ],
        help=(
            "Action chunk aggregation strategy (default: temporal_ensemble, "
            "ACT Algorithm 2; gripper indices take the newest chunk). The "
            "other choices are upstream scalar blends applied uniformly."
        ),
    )
    p.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=0.01,
        metavar="K",
        help=(
            "Decay coefficient for --aggregate-fn temporal_ensemble "
            "(default: 0.01, ACT paper). wᵢ = exp(-K·i), i=0 oldest chunk; "
            "K>0 smoother, K=0 uniform, K<0 more reactive."
        ),
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

    # Translate operator-actionable hardware faults into a clean non-zero
    # exit instead of a multi-frame traceback.
    import sys

    import can

    from ..motor.errors import MotorError

    try:
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
            temporal_ensemble_coeff=args.temporal_ensemble_coeff,
            zed_host=args.zed_host,
            zed_iface=args.zed_iface,
            left_gripper_torque_limit=args.left_gripper_torque_limit,
            right_gripper_torque_limit=args.right_gripper_torque_limit,
            left_stiffness=args.left_stiffness,
            right_stiffness=args.right_stiffness,
            rerun_ip=args.rerun_ip,
            rerun_port=args.rerun_port,
        )
    except (MotorError, can.CanError) as exc:
        _logger.error("Robot hardware error: %s. Exiting.", exc)
        sys.exit(1)


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

    Lives at module scope so it's picklable by ``mp.get_context('spawn')``.
    SIGINT is ignored so Ctrl+C in the parent terminal doesn't dump a gRPC
    traceback; the parent explicitly terminates the server during cleanup.

    Args:
        server_cfg_dict: ``PolicyServerConfig`` keyword arguments.
    """
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    from lerobot.async_inference import policy_server as _ps

    # Upstream's 1-rad L2 similarity filter drops nearly every observation
    # on Axol's 16-DOF arms at 60 Hz, starving the action queue. There is
    # no config knob, so patch the module symbol before ``serve``.
    _ps.observations_similar = lambda *args, **kwargs: False

    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server import serve

    serve(PolicyServerConfig(**server_cfg_dict))


# ----------------------------------------------------------------------
# AxolRobotClient: thin RobotClient subclass that reuses our connected robot
# ----------------------------------------------------------------------


def _build_axol_robot_client(
    *,
    config: Any,
    robot: "AxolRobot",
    publisher: ActionPublisher,
    aggregate_strategy: str = "temporal_ensemble",
    temporal_ensemble_coeff: float = 0.01,
) -> Any:
    """Construct an ``AxolRobotClient`` against an already-connected robot.

    Wrapped in a helper so the lerobot imports stay lazy.

    Args:
        config: Built ``RobotClientConfig``.
        robot: Connected ``AxolRobot`` instance, reused across episodes.
        publisher: Sink for executed actions, drained by the capture thread.
        aggregate_strategy: One of the ``--aggregate-fn`` choices.
        temporal_ensemble_coeff: Decay coefficient for temporal_ensemble.
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
        """``RobotClient`` adapted to reuse a pre-connected ``AxolRobot``.

        Diverges from upstream in three places:

        - ``_aggregate_action_queues`` dispatches to a vectorized
          ``temporal_ensemble`` (ACT smoothing) when selected, otherwise
          falls through to the scalar blends. The final queue swap goes
          through ``_install_future_queue`` to avoid re-popping a
          just-executed timestep.
        - ``control_loop`` only pops actions; observation capture + send
          moves to ``_observation_loop`` so the ~60-70 ms ZED read + gRPC
          send can't stall the 60 Hz action stream.
        - ``control_loop_action`` updates ``latest_action`` atomically
          with the queue pop (upstream updates it after ``send_action``,
          which leaves a re-pop race for the aggregator).

        The constructor also skips ``make_robot_from_config`` / connect so
        re-recording doesn't pay the camera reconnect cost, publishes
        executed actions to ``ActionPublisher``, and ``stop()`` tears
        down only the gRPC channel (the robot is shared across episodes).
        No post-filter is applied: training actions are already
        EMA + trapezoidal-filtered in ``collect_data``.
        """

        # Indices of the gripper joints in the 16-element action vector.
        # ``_temporal_ensemble_aggregate`` snaps these to the newest
        # contributing chunk so bang-bang grasps aren't smeared.
        _GRIPPER_INDICES = (7, 15)

        def __init__(  # type: ignore[no-untyped-def]
            self,
            config,
            robot,
            publisher,
            aggregate_strategy,
            temporal_ensemble_coeff,
        ):
            self.config = config
            self.robot = robot
            self._publisher = publisher
            self._aggregate_strategy = aggregate_strategy
            self._temporal_ensemble_coeff = float(temporal_ensemble_coeff)
            # ``(origin, packed_actions, timestamp)`` per chunk, sorted
            # oldest-first. ``packed_actions`` is a (chunk_size, action_dim)
            # tensor so aggregation runs as one batched op.
            self._chunk_buffer: list[tuple[int, Any, float]] = []
            self._race_fix_warned: bool = False
            # Surfaces unhandled control-loop exceptions (typically CAN
            # faults) to the episode supervisor for an immediate teardown.
            self.fatal_error: BaseException | None = None

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
            # Receiver + control + observation threads sync at episode start.
            self.start_barrier = _threading.Barrier(3)
            self.fps_tracker = FPSTracker(target_fps=self.config.fps)
            self.must_go = _threading.Event()
            self.must_go.set()

        def reset_episode_state(self) -> None:
            """Reset queues + flags so threads can run a fresh episode."""
            with self.action_queue_lock:
                self.action_queue = Queue()
                self.action_queue_size = []
                self._chunk_buffer = []
            with self.latest_action_lock:
                self.latest_action = -1
            self.action_chunk_size = -1
            self.must_go.set()
            self.fps_tracker.reset()
            self.shutdown_event.clear()
            self.start_barrier = _threading.Barrier(3)
            self._race_fix_warned = False
            self.fatal_error = None
            if self._publisher is not None:
                self._publisher.reset()

        def _install_future_queue(self, future_queue) -> None:  # type: ignore[no-untyped-def]
            """Swap in ``future_queue``, dropping already-executed timesteps.

            The control thread can pop further actions between the
            aggregator reading ``latest_action`` and the queue swap. Holding
            ``action_queue_lock`` while re-filtering against the live
            ``latest_action`` prevents the post-swap queue from walking
            ``latest_action`` backwards and snapping the arm.

            Args:
                future_queue: Newly aggregated action queue to install.
            """
            with self.action_queue_lock:
                with self.latest_action_lock:
                    live_latest = self.latest_action
                filtered = Queue()
                dropped = 0
                while not future_queue.empty():
                    ta = future_queue.get_nowait()
                    if ta.get_timestep() > live_latest:
                        filtered.put(ta)
                    else:
                        dropped += 1
                self.action_queue = filtered
            if dropped and not self._race_fix_warned:
                self._race_fix_warned = True
                _logger.warning(
                    "Aggregator race fix engaged: %d timestep(s) popped "
                    "during aggregation were filtered out of the new "
                    "queue (informational; fix handled it).",
                    dropped,
                )

        def _temporal_ensemble_aggregate(self, incoming_actions):  # type: ignore[no-untyped-def]
            """Aggregate buffered chunks with ACT Algorithm 2.

            Each future timestep ``ts > latest_action`` covered by at
            least one buffered chunk gets ``commanded[ts] = Σ wᵢ ·
            chunkᵢ[ts] / Σ wᵢ`` with ``wᵢ = exp(-coeff · i)`` and
            ``i = 0`` the oldest chunk. ``_GRIPPER_INDICES`` bypass the
            average and snap to the newest contributing chunk's value.
            The rebuild is one batched op over an
            ``(n_chunks, n_ts, action_dim)`` grid, sub-ms even with
            ~20 chunks in flight.

            Args:
                incoming_actions: Latest action chunk from the policy
                    server. Empty input is a no-op.
            """
            import torch
            from lerobot.async_inference.helpers import TimedAction

            if not incoming_actions:
                return

            # Pack into a tensor sorted by ascending timestep. Upstream
            # always emits contiguous chunks via ``_time_action_chunk``;
            # fail loudly if that invariant is violated.
            sorted_incoming = sorted(incoming_actions, key=lambda a: a.get_timestep())
            new_origin = sorted_incoming[0].get_timestep()
            chunk_size = len(sorted_incoming)
            sample = sorted_incoming[0].get_action()
            new_packed = torch.empty(
                (chunk_size, sample.shape[0]),
                dtype=sample.dtype,
                device=sample.device,
            )
            for offset, ta in enumerate(sorted_incoming):
                if ta.get_timestep() != new_origin + offset:
                    raise RuntimeError(
                        "temporal_ensemble: incoming chunk timesteps are "
                        f"non-contiguous (expected {new_origin + offset}, "
                        f"got {ta.get_timestep()} at offset {offset})"
                    )
                new_packed[offset] = ta.get_action()
            new_timestamp = sorted_incoming[0].get_timestamp()

            self._chunk_buffer.append((new_origin, new_packed, new_timestamp))
            self._chunk_buffer.sort(key=lambda entry: entry[0])

            with self.latest_action_lock:
                latest_action = self.latest_action

            # Drop chunks whose entire range has already been executed.
            self._chunk_buffer = [
                entry
                for entry in self._chunk_buffer
                if entry[0] + entry[1].shape[0] - 1 > latest_action
            ]
            n_chunks = len(self._chunk_buffer)
            if n_chunks == 0:
                self._install_future_queue(Queue())
                return

            # Grid: every future timestep covered by ≥1 buffered chunk.
            grid_min_ts = latest_action + 1
            grid_max_ts = max(
                origin + packed.shape[0] - 1 for origin, packed, _ in self._chunk_buffer
            )
            if grid_min_ts > grid_max_ts:
                self._install_future_queue(Queue())
                return

            n_ts = grid_max_ts - grid_min_ts + 1
            dtype = sample.dtype
            device = sample.device
            action_dim = sample.shape[0]

            # ``mask[ci, ts]`` is 1.0 where chunk ``ci`` covers ``ts``.
            action_grid = torch.zeros(
                (n_chunks, n_ts, action_dim), dtype=dtype, device=device
            )
            mask = torch.zeros((n_chunks, n_ts), dtype=dtype, device=device)
            for ci, (origin, packed, _) in enumerate(self._chunk_buffer):
                chunk_max = origin + packed.shape[0] - 1
                lo = max(origin, grid_min_ts)
                hi = min(chunk_max, grid_max_ts)
                if lo > hi:
                    continue
                src_start = lo - origin
                src_stop = hi - origin + 1
                dst_start = lo - grid_min_ts
                dst_stop = hi - grid_min_ts + 1
                action_grid[ci, dst_start:dst_stop] = packed[src_start:src_stop]
                mask[ci, dst_start:dst_stop] = 1.0

            coeff = self._temporal_ensemble_coeff
            chunk_weights = torch.exp(
                -coeff * torch.arange(n_chunks, dtype=dtype, device=device)
            )
            weighted_mask = chunk_weights.unsqueeze(1) * mask  # (n_chunks, n_ts)
            norm = weighted_mask.sum(dim=0).clamp(min=1e-12)
            ensembled = (action_grid * weighted_mask.unsqueeze(-1)).sum(
                dim=0
            ) / norm.unsqueeze(-1)  # (n_ts, action_dim)

            # Gripper carve-out: snap to the newest contributing chunk via
            # a reverse-cumsum one-hot (cheaper than ``torch.max`` on tiny
            # CPU tensors).
            reverse_cumsum = mask.flip(0).cumsum(dim=0).flip(0)
            newest_mask = mask * (reverse_cumsum == 1).to(dtype)
            for gidx in self._GRIPPER_INDICES:
                ensembled[:, gidx] = (action_grid[:, :, gidx] * newest_mask).sum(dim=0)

            contributed_indices = (
                mask.any(dim=0).nonzero(as_tuple=False).flatten().tolist()
            )
            future_queue = Queue()
            for ti in contributed_indices:
                future_queue.put(
                    TimedAction(
                        timestamp=new_timestamp,
                        timestep=grid_min_ts + ti,
                        action=ensembled[ti].clone(),
                    )
                )
            self._install_future_queue(future_queue)

        def _aggregate_action_queues(self, incoming_actions, aggregate_fn=None):  # type: ignore[no-untyped-def]
            """Dispatch ``temporal_ensemble`` locally, else upstream scalar blends."""
            if self._aggregate_strategy == "temporal_ensemble":
                return self._temporal_ensemble_aggregate(incoming_actions)
            return super()._aggregate_action_queues(incoming_actions, aggregate_fn)

        def control_loop_action(self, verbose: bool = False):  # type: ignore[no-untyped-def]
            """Pop the next action, advance ``latest_action``, send to robot.

            ``latest_action`` is updated inside the queue lock so the
            aggregator can never see a stale value and re-insert a
            just-popped timestep — a race that fires ~0.8/s at 60 Hz with
            upstream's pop-then-update ordering.
            """
            with self.action_queue_lock:
                self.action_queue_size.append(self.action_queue.qsize())
                timed_action = self.action_queue.get_nowait()
                with self.latest_action_lock:
                    self.latest_action = timed_action.get_timestep()
                qs_after = self.action_queue.qsize()

            performed = self.robot.send_action(
                self._action_tensor_to_action_dict(timed_action.get_action())
            )

            if verbose:
                self.logger.debug(
                    f"Ts={timed_action.get_timestamp()} | "
                    f"Action #{timed_action.get_timestep()} performed | "
                    f"Queue size: {qs_after}"
                )

            if self._publisher is not None and performed is not None:
                self._publisher.publish(performed)
            return performed

        def control_loop(self, task, verbose: bool = False):  # type: ignore[no-untyped-def,override]
            """Action-only control loop; obs send is on ``_observation_loop``.

            Upstream interleaves microsecond action pops with the 60-70 ms
            obs send on one thread, collapsing 60 Hz down to ~27 Hz on
            Axol. Decoupling restores the target rate. Unhandled
            exceptions (typically CAN faults from ``send_action``) are
            captured in ``self.fatal_error`` and trigger shutdown.
            """
            self.start_barrier.wait()
            self.logger.info("Action-only control loop starting (obs send decoupled)")
            try:
                while self.running:
                    control_loop_start = time.perf_counter()
                    if self.actions_available():
                        self.control_loop_action(verbose)
                    elapsed = time.perf_counter() - control_loop_start
                    time.sleep(max(0.0, self.config.environment_dt - elapsed))
            except Exception as exc:  # noqa: BLE001
                self.logger.error(
                    f"Control loop hit an unhandled exception ({exc!r}); "
                    "signalling shutdown so the episode tears down."
                )
                self.fatal_error = exc
                self.shutdown_event.set()

        def _observation_loop(self, task, verbose: bool = False):  # type: ignore[no-untyped-def]
            """Dedicated thread: capture and send observations.

            Fires ``control_loop_observation`` once the action queue
            drops to ``chunk_size_threshold``; sleeps one tick otherwise.
            """
            self.start_barrier.wait()
            self.logger.info("Observation loop thread starting")
            while self.running:
                try:
                    if self._ready_to_send_observation():
                        self.control_loop_observation(task, verbose)
                    else:
                        time.sleep(self.config.environment_dt)
                except Exception as exc:  # noqa: BLE001
                    self.logger.error(f"Observation loop error: {exc!r}; continuing")
                    time.sleep(self.config.environment_dt)

        def stop(self) -> None:  # type: ignore[override]
            self.shutdown_event.set()
            try:
                self.channel.close()
            except Exception:  # noqa: BLE001
                pass
            self.logger.debug("AxolRobotClient channel closed (robot left connected)")

    return AxolRobotClient(
        config,
        robot,
        publisher,
        aggregate_strategy,
        temporal_ensemble_coeff,
    )


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
    chunk_size_threshold: float = 0.9,
    aggregate_fn: str = "temporal_ensemble",
    temporal_ensemble_coeff: float = 0.01,
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

    # Dataset features come from static camera configs + joint enum, so the
    # dataset can be constructed before the robot connects — letting us load
    # the policy first (see PolicyServer spawn below).
    dataset: "LeRobotDataset | None" = None
    dataset_root: Path | None = None
    resumed_dataset = False
    if repo_id:
        dataset_root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        meta = dataset_root / "meta"
        has_info = (meta / "info.json").exists()
        is_complete = (
            has_info
            and (meta / "tasks.parquet").exists()
            and (meta / "episodes").is_dir()
        )
        # Mirror collect-data's resume/refuse/wipe decision tree.
        if has_info and not is_complete:
            raise RuntimeError(
                f"Incomplete dataset found at {dataset_root} (missing "
                f"tasks.parquet or episodes/). Delete the directory and "
                f"rerun to start fresh:\n  rm -rf {dataset_root}"
            )
        if dataset_root.exists() and not is_complete:
            log_say(f"Removing empty dataset directory at {dataset_root}.")
            shutil.rmtree(dataset_root)

        if is_complete:
            log_say(f"Resuming existing dataset at {dataset_root}.")
            dataset = LeRobotDataset.resume(
                repo_id=repo_id,
                root=str(dataset_root),
                image_writer_threads=4,
                streaming_encoding=True,
                encoder_threads=4,
                vcodec="auto",
            )
            resumed_dataset = True
        else:
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
                streaming_encoding=True,
                encoder_threads=4,
                vcodec="auto",
            )

    if rerun_ip:
        init_rerun(session_name="axol_run_policy", ip=rerun_ip, port=rerun_port)

    # Spawn the policy server and load the policy BEFORE connecting cameras:
    # the model download + CUDA load is a ~15 s network + GPU spike that can
    # disrupt already-open ZED streams.
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

    # Spawn the IK worker in parallel so JAX JIT overlaps with policy load.
    reset_controller = IKResetController()
    reset_controller.start()
    log_say("Started IK reset worker (collision-aware return-to-rest).")

    client = None
    episodes_recorded = 0
    try:
        _wait_for_port("127.0.0.1", server_port, timeout=30.0)

        # ``RobotClientConfig`` requires a name from upstream's registry;
        # ``temporal_ensemble`` is handled in our override so pass a
        # placeholder that the dispatcher short-circuits.
        if aggregate_fn == "temporal_ensemble":
            log_say(
                f"Aggregation: temporal_ensemble "
                f"(coeff={temporal_ensemble_coeff:+.3f}, ACT default 0.01)."
            )
        else:
            log_say(f"Aggregation: {aggregate_fn}.")
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
            aggregate_fn_name=(
                "latest_only" if aggregate_fn == "temporal_ensemble" else aggregate_fn
            ),
        )
        publisher = ActionPublisher()
        client = _build_axol_robot_client(
            config=client_cfg,
            robot=robot,
            publisher=publisher,
            aggregate_strategy=aggregate_fn,
            temporal_ensemble_coeff=temporal_ensemble_coeff,
        )

        log_say("Loading policy on server (one-time)...")
        if not client.start():
            raise RuntimeError("Failed to connect to policy server / load policy.")

        log_say("Connecting robot...")
        robot.connect()

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
            # Decoupled from the control thread — see AxolRobotClient.control_loop.
            obs_thread = threading.Thread(
                target=client._observation_loop,
                args=(task,),
                name="axol-obs-loop",
                daemon=True,
            )

            capture: RolloutCaptureThread | None = None
            if dataset is not None:
                capture = RolloutCaptureThread(
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
                target=stdin_watcher,
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
            obs_thread.start()
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
                    if client.fatal_error is not None:
                        # Hardware fault from the control loop — drop the
                        # episode and exit the run.
                        log_say(
                            f"Fatal error in control loop: "
                            f"{client.fatal_error!r}. Aborting run without "
                            "saving the current episode."
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
            obs_thread.join(timeout=5.0)
            # ``stdin_thread`` wakes itself via ``select``; don't join.

            if interrupted:
                break
            if client.fatal_error is not None:
                if dataset is not None:
                    dataset.clear_episode_buffer()
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

        # Re-raise the control-loop fault so ``run()`` exits non-zero.
        if client is not None and client.fatal_error is not None:
            raise client.fatal_error

    except KeyboardInterrupt:
        pass
    finally:
        # Ignore SIGINT during cleanup so a second Ctrl+C can't abort
        # partway through disconnect/teardown. Restored at end of block.
        import signal

        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except (ValueError, OSError):
            pass

        log_say("Stopping.")
        if client is not None:
            try:
                client.stop()
            except Exception:  # noqa: BLE001
                pass
        # ``disconnect()`` is null-safe and idempotent; always call it so a
        # ``connect()`` that bailed mid-enable doesn't leak the asyncio
        # event-loop thread or any already-opened CAN buses.
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            pass

        try:
            reset_controller.stop()
        except Exception:  # noqa: BLE001
            pass

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

        # Auto-wipe only a freshly-created, never-written dataset. Resumed
        # datasets already have saved rollouts on disk and must be kept.
        if (
            dataset_root is not None
            and not resumed_dataset
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

        # Restore the default handler so Ctrl+C can still kill any leaked
        # non-daemon thread keeping the interpreter alive.
        try:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        except (ValueError, OSError):
            pass
