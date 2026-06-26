"""
axol run-policy

Run a trained policy on the Axol robot with its local ZED cameras using
LeRobot's async inference (``lerobot.async_inference``). By default a
``PolicyServer`` is auto-launched in a child process on localhost; pass
``--server_host`` to use a remote inference server started with
``axol inference-server`` on a more powerful machine instead (joint
positions + camera frames are streamed to it over gRPC and it returns
action chunks). Either way the parent drives an ``AxolRobotClient`` (a
thin ``RobotClient`` subclass) that streams observations to the server
and consumes the returned action chunks. Cameras and joints are sampled
via ``ZedCamera.read_at_or_after(now)`` so every inference observation is
timestamp-aligned the same way the training data is (see
``AxolRobot.get_observation``).

Each episode runs until the operator types ``s`` (save), ``r`` (rerecord
+ discard), or ``q`` (quit + discard) on stdin. ``--episode_time_s`` is a
safety cap that falls back to the same ``[Enter]=save / r / q`` prompt
when no key has been pressed.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lerobot.robots.config import RobotConfig

from ..lerobot.camera.configuration_zed import ZedCameraConfig
from ..lerobot.robot.config_axol import AxolRobotConfig
from ..lerobot.rollout import (
    ActionPublisher,
    IKResetController,
    RolloutCaptureThread,
)
from .config import AggregateFn, LogLevel, PolicyType, parse

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from ..lerobot.robot.robot_axol import AxolRobot

_logger = logging.getLogger(__name__)


def _default_robot_config() -> AxolRobotConfig:
    """Default Axol robot config for inference: local ZED cameras.

    All three slots (overhead, left_arm, right_arm) are seeded with the
    unassigned sentinel serial ``0`` so each stays reachable as a dotted
    ``--robot_config.cameras.<slot>.serial`` override (or control-panel field),
    but only the slots you assign a serial to are used — the rest are pruned by
    ``AxolRobotConfig.select_assigned_cameras`` (at least one must be assigned;
    assign the cameras the policy was trained on). draccus takes dict fields as
    one inline YAML/JSON value, so assign serials with e.g.
    ``--robot_config.cameras "{overhead: {serial: 41234567}}"``. Other fields
    are overridable too, e.g. ``--robot_config.axol_config.left_stiffness 0.8``
    (match the stiffness used at data-collection time).

    Inference captures through the ZED Python SDK (``video_backend="sdk"``):
    run-policy streams no headset video, so the GPU-resident gst pipeline's
    encoded branch would be pure waste here. Teleop and collect-data default
    to the gst path; pass ``--robot_config.video_backend gst`` to opt in.
    """
    return AxolRobotConfig(
        cameras={
            "overhead": ZedCameraConfig(serial=0),
            "left_arm": ZedCameraConfig(serial=0),
            "right_arm": ZedCameraConfig(serial=0),
        },
        video_backend="sdk",
    )


def _default_vcodec() -> str:
    """Pick a video codec that can actually open on this machine.

    LeRobot's "auto" prefers the NVIDIA hardware encoder (``h264_nvenc``)
    whenever the codec is compiled into ffmpeg, but on Jetson/Tegra (aarch64)
    there's no desktop ``libnvidia-encode`` to back it, so it fails to open and
    kills the encoder thread mid episode. Default to CPU "h264" (software
    libx264) on aarch64 and let "auto" pick the HW encoder everywhere else.
    """
    import platform

    return "h264" if platform.machine() == "aarch64" else "auto"


@dataclass
class RunPolicyConfig:
    """Config for ``axol run-policy``.

    ``robot_config`` is the full Axol robot config (cameras, per-joint
    gains); nest into it from the CLI (e.g.
    ``--robot_config.axol_config.left_stiffness 0.8``) or pass a
    whole-config file with ``--config_path``. The compliance/stiffness
    blend should match the values used at data-collection time.

    Inference runs on this machine by default (a ``PolicyServer`` child
    process on ``localhost:server_port``). To offload it, start ``axol
    inference-server`` on a GPU machine and pass its address via
    ``--server_host`` — ``policy_path`` / ``policy_type`` / ``device``
    then apply to that server (it downloads the policy itself, so the
    path must be reachable from it, e.g. a HF Hub repo id).
    """

    policy_path: str
    policy_type: PolicyType
    task: str
    robot_config: RobotConfig = field(default_factory=_default_robot_config)
    episode_time_s: int = 120
    fps: int = 60
    # Video codec for the recorded LeRobot dataset; defaults per-platform (see
    # _default_vcodec). Override with any of LeRobot's VALID_VIDEO_CODECS
    # (e.g. auto, h264, libsvtav1).
    vcodec: str = field(default_factory=_default_vcodec)
    repo_id: str | None = None
    root: str | None = None
    push_to_hub: bool = False
    device: str = "cuda"
    server_host: str | None = None
    server_port: int = 8765
    actions_per_chunk: int = 50
    chunk_size_threshold: float = 0.9
    aggregate_fn: AggregateFn = "temporal_ensemble"
    temporal_ensemble_coeff: float = 0.01
    rerun_ip: str | None = None
    rerun_port: int = 9876
    log_level: LogLevel = "INFO"


# ----------------------------------------------------------------------
# Episode control: abstracts how start/save/rerecord/quit decisions arrive.
#
# The CLI reads them from stdin (``s`` / ``r`` / ``q`` + Enter prompts); the
# web control panel pushes them through a queue from the API. ``_run`` is
# agnostic — it only calls the small surface below.
# ----------------------------------------------------------------------


class _StdinPolicyControl:
    """Terminal episode control: stdin keystrokes + Enter-to-continue prompts."""

    def __init__(self) -> None:
        self._stop: "threading.Event | None" = None
        self._result: dict[str, str | None] = {"choice": None}
        self._thread: "threading.Thread | None" = None

    def await_continue(self, message: str) -> bool:
        try:
            input(message)
            return True
        except (EOFError, KeyboardInterrupt):
            return False

    def begin_episode(self) -> None:
        from ..lerobot.rollout import stdin_watcher

        self._stop = threading.Event()
        self._result = {"choice": None}
        self._thread = threading.Thread(
            target=stdin_watcher,
            args=(self._stop, self._result),
            name="axol-stdin-watcher",
            daemon=True,
        )
        self._thread.start()

    def poll_choice(self) -> str | None:
        return self._result.get("choice")

    def resolve_timeout(self, episode_time_s: int) -> str:
        try:
            raw = input(
                f"Episode time cap ({episode_time_s}s) reached. "
                "[Enter]=save, r=rerecord, q=quit: "
            )
        except (EOFError, KeyboardInterrupt):
            return "q"
        raw = raw.strip().lower()
        return "q" if raw == "q" else ("r" if raw == "r" else "s")

    def end_episode(self) -> None:
        if self._stop is not None:
            self._stop.set()

    def close(self) -> None:
        self.end_episode()


class _QueuePolicyControl:
    """Web episode control: decisions arrive as API-pushed queue commands.

    Accepts ``s`` / ``r`` / ``q`` for the running episode and ``continue``
    (alias ``start``) to advance through the between-episode "ready" gate.
    """

    def __init__(self, stop_event: "threading.Event") -> None:
        import queue

        self._q: "queue.Queue[str]" = queue.Queue()
        self._stop = stop_event
        self._choice: str | None = None

    def push(self, command: str) -> None:
        self._q.put(command)

    def _drain(self) -> None:
        import queue

        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def await_continue(self, message: str) -> bool:
        import queue

        from lerobot.utils.utils import log_say

        log_say(message)
        while not self._stop.is_set():
            try:
                cmd = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            if cmd in ("continue", "start", "s"):
                return True
            if cmd == "q":
                return False
        return False

    def begin_episode(self) -> None:
        self._choice = None
        self._drain()

    def poll_choice(self) -> str | None:
        import queue

        if self._choice is not None:
            return self._choice
        try:
            cmd = self._q.get_nowait()
        except queue.Empty:
            return None
        if cmd in ("s", "r", "q"):
            self._choice = cmd
            return cmd
        return None

    def resolve_timeout(self, episode_time_s: int) -> str:
        import queue

        from lerobot.utils.utils import log_say

        log_say(
            f"Episode time cap ({episode_time_s}s) reached — choose save/rerecord/quit."
        )
        while not self._stop.is_set():
            try:
                cmd = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            if cmd in ("s", "r", "q"):
                return cmd
        return "q"

    def end_episode(self) -> None:
        pass

    def close(self) -> None:
        pass


def main(argv: list[str]) -> None:
    """Parse the CLI config and run the policy, exiting cleanly on hardware faults."""
    cfg = parse(RunPolicyConfig, argv)
    # force=True: importing lerobot (at module load) installs a root handler
    # and leaves the root level at WARNING, which would otherwise make this a
    # no-op and silently drop every log_say() status line.
    logging.basicConfig(level=getattr(logging, cfg.log_level), force=True)

    # Translate operator-actionable hardware faults into a clean non-zero
    # exit instead of a multi-frame traceback.
    import sys

    import can

    from ..motor.errors import MotorError

    try:
        _run(cfg)
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

    from ..lerobot.inference_patch import disable_observation_similarity_filter

    disable_observation_similarity_filter()

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
        aggregate_strategy: One of the ``--aggregate_fn`` choices.
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
          moves to ``observation_loop`` so the ~60-70 ms ZED read + gRPC
          send can't stall the 60 Hz action stream.
        - ``control_loop_action`` updates ``latest_action`` atomically
          with the queue pop (upstream updates it after ``send_action``,
          which leaves a re-pop race for the aggregator).

        Everything here is built on public ``RobotClient`` API (public
        instance attributes + public ``control_loop_observation`` /
        ``send_action`` / ``actions_available``) with one unavoidable
        exception: overriding ``_aggregate_action_queues``. LeRobot calls
        the aggregator by that private name inside the public
        ``receive_actions`` (robot_client.py), and its only public
        aggregation hook (``RobotClientConfig.aggregate_fn_name``) selects
        one of four *stateless, pairwise* blend functions — which cannot
        express our stateful, multi-chunk, recency-weighted
        ``temporal_ensemble``. Overriding the private method is therefore
        the minimal seam. ``__init__`` asserts the symbol still exists so a
        LeRobot bump that renames it fails loudly instead of silently
        disabling ensembling. (Upstreaming a public aggregator hook would
        remove this last dependency.)

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
            # We override the private RobotClient._aggregate_action_queues to
            # inject temporal_ensemble (no public hook can express it — see the
            # class docstring). If a LeRobot upgrade renames it, receive_actions
            # would call the new name and our override would silently never run,
            # disabling ensembling. Fail loudly at construction instead.
            if not hasattr(RobotClient, "_aggregate_action_queues"):
                raise RuntimeError(
                    "lerobot RobotClient no longer defines "
                    "'_aggregate_action_queues'; AxolRobotClient's "
                    "temporal_ensemble override needs review against the new "
                    "LeRobot version."
                )

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

            # Inlined from upstream RobotClient._action_tensor_to_action_dict
            # so we depend only on the public ``robot.action_features`` rather
            # than a private LeRobot method. Maps the flat action tensor to a
            # {motor_name: float} dict by position.
            action_tensor = timed_action.get_action()
            action = {
                key: action_tensor[i].item()
                for i, key in enumerate(self.robot.action_features)
            }
            performed = self.robot.send_action(action)

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
            """Action-only control loop; obs send is on ``observation_loop``.

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

        def observation_loop(self, task, verbose: bool = False):  # type: ignore[no-untyped-def]
            """Dedicated thread: capture and send observations.

            Fires ``control_loop_observation`` once the action queue
            drops to ``chunk_size_threshold``; sleeps one tick otherwise.
            """
            self.start_barrier.wait()
            self.logger.info("Observation loop thread starting")
            while self.running:
                try:
                    # Inlined from upstream RobotClient._ready_to_send_observation
                    # so we read only public state (``action_queue``,
                    # ``action_chunk_size``) and the public
                    # ``config.chunk_size_threshold`` instead of the private
                    # method/attribute. Send once the queue drains to threshold.
                    with self.action_queue_lock:
                        queue_fraction = (
                            self.action_queue.qsize() / self.action_chunk_size
                        )
                    if queue_fraction <= self.config.chunk_size_threshold:
                        self.control_loop_observation(task, verbose)
                    else:
                        time.sleep(self.config.environment_dt)
                except Exception as exc:  # noqa: BLE001
                    self.logger.error(f"Observation loop error: {exc!r}; continuing")
                    time.sleep(self.config.environment_dt)

        def stop(self) -> None:  # type: ignore[override]
            """Tear down the gRPC channel; the shared robot stays connected."""
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
    cfg: RunPolicyConfig,
    stop_event: "threading.Event | None" = None,
    control: "_StdinPolicyControl | _QueuePolicyControl | None" = None,
) -> None:
    """Drive the full run-policy session: spawn the policy server, connect the robot, and run episodes."""
    import multiprocessing as mp
    import shutil
    from pathlib import Path

    if stop_event is None:
        stop_event = threading.Event()
    if control is None:
        control = _StdinPolicyControl()

    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor import make_default_processors
    from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
    from lerobot.utils.feature_utils import hw_to_dataset_features
    from lerobot.utils.utils import log_say
    from lerobot.utils.visualization_utils import init_rerun

    from ..lerobot.robot.robot_axol import AxolRobot

    policy_path = cfg.policy_path
    policy_type = cfg.policy_type
    task = cfg.task
    episode_time_s = cfg.episode_time_s
    fps = cfg.fps
    vcodec = cfg.vcodec
    repo_id = cfg.repo_id
    root = cfg.root
    push_to_hub = cfg.push_to_hub
    device = cfg.device
    server_host = cfg.server_host
    server_port = cfg.server_port
    actions_per_chunk = cfg.actions_per_chunk
    chunk_size_threshold = cfg.chunk_size_threshold
    aggregate_fn = cfg.aggregate_fn
    temporal_ensemble_coeff = cfg.temporal_ensemble_coeff
    rerun_ip = cfg.rerun_ip
    rerun_port = cfg.rerun_port
    robot_config = cfg.robot_config

    # Keep only the camera slots the operator actually assigned a serial to
    # (overhead / left_arm / right_arm are all seeded as placeholders); at
    # least one must be set, and should be the cameras the policy was trained
    # on. Then flag any physically-stereo ZED X so it opens on the stereo grab
    # path (a mono open of a ZED X fails connect).
    if isinstance(robot_config, AxolRobotConfig):
        from ..zed import stereo_serials

        robot_config.select_assigned_cameras(minimum=1)
        robot_config.apply_detected_stereo(stereo_serials())

    robot = AxolRobot(robot_config)
    _, robot_action_proc, robot_obs_proc = make_default_processors()

    # The dataset is constructed before the robot connects — letting us load
    # the policy first (see PolicyServer spawn below). Pre-connect, camera
    # feature shapes come from the camera configs; connect() later enforces
    # that the live streams match, so the shapes baked in here stay valid.
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
                vcodec=vcodec,
            )
            resumed_dataset = True
        else:
            # Guard against auto-detect camera configs (width/height of None):
            # the robot is not connected yet, so unknown dimensions here would
            # bake invalid image shapes into the dataset metadata.
            incomplete_cams = [
                name
                for name, feat in robot.observation_features.items()
                if isinstance(feat, tuple) and None in feat
            ]
            if incomplete_cams:
                raise RuntimeError(
                    "Cannot create a dataset before the robot connects: "
                    f"camera(s) {', '.join(incomplete_cams)} have no width/"
                    "height set in their config. Set explicit dimensions in "
                    "the camera config (run-policy builds dataset features "
                    "before connecting)."
                )
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
                vcodec=vcodec,
            )

    if rerun_ip:
        init_rerun(session_name="axol_run_policy", ip=rerun_ip, port=rerun_port)

    # Local inference (default): spawn the policy server and load the policy
    # BEFORE connecting cameras — the model download + CUDA load is a ~15 s
    # network + GPU spike that can disrupt already-open camera pipelines.
    # Remote inference (--server_host): the server is already running via
    # ``axol inference-server`` on another machine; just point at it.
    server_proc = None
    if server_host is None:
        server_host = "127.0.0.1"
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
        log_say(
            f"Started PolicyServer on 127.0.0.1:{server_port} (pid={server_proc.pid})."
        )
    else:
        log_say(f"Using remote inference server at {server_host}:{server_port}.")

    # Spawn the IK worker in parallel so JAX JIT overlaps with policy load.
    reset_controller = IKResetController()
    reset_controller.start()
    log_say("Started IK reset worker (collision-aware return-to-rest).")

    client = None
    episodes_recorded = 0
    try:
        _wait_for_port(server_host, server_port, timeout=30.0)

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
            server_address=f"{server_host}:{server_port}",
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
        if not control.await_continue(
            "Reset the scene, then press Enter to start the first episode."
        ):
            return

        while True:
            if stop_event.is_set():
                break
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
                target=client.observation_loop,
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

            control.begin_episode()

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

            deadline = time.perf_counter() + episode_time_s
            timed_out = False
            interrupted = False
            try:
                while True:
                    if stop_event.is_set():
                        break
                    if control.poll_choice() is not None:
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
            control.end_episode()
            client.shutdown_event.set()
            if capture is not None:
                capture.stop_event.set()
                capture.join(timeout=5.0)
            control_thread.join(timeout=5.0)
            receiver_thread.join(timeout=5.0)
            obs_thread.join(timeout=5.0)

            if interrupted or stop_event.is_set():
                if dataset is not None:
                    dataset.clear_episode_buffer()
                break
            if client.fatal_error is not None:
                if dataset is not None:
                    dataset.clear_episode_buffer()
                break

            choice = control.poll_choice()
            if timed_out:
                choice = control.resolve_timeout(episode_time_s)

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
                if not control.await_continue(
                    "Reset the scene, then press Enter to start."
                ):
                    break
                continue

            # choice == "s"
            if dataset is not None:
                dataset.save_episode()
            episodes_recorded += 1
            log_say(f"Saved episode {episodes_recorded}.")
            log_say("Returning to rest pose.")
            reset_controller.return_to_rest(robot)
            if not control.await_continue(
                "Reset the scene, then press Enter to start the next episode."
            ):
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

        if server_proc is not None and server_proc.is_alive():
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
