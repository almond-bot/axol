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

import gc
import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from lerobot.robots.config import RobotConfig

from ..lerobot.camera.configuration_zed import ZedCameraConfig
from ..lerobot.robot.config_axol import AxolRobotConfig
from ..lerobot.rollout import (
    ActionPublisher,
    IKResetController,
    RolloutCaptureThread,
)
from ..recording import make_episode_durable, restore_dataset_ownership
from ..robot.control import ContactWatchdog
from ..teleop.config import VRTeleopConfig
from ..teleop.filter import TrapezoidalFilter
from .collect_data import check_resume_consistency
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
        # The control loop runs motion_control every step, whose command
        # replies keep the joint cache fresh — so the background telemetry
        # poll loop is redundant CAN/CPU load that contends with the 60 Hz
        # action stream on the same buses and event loop. Skipping it
        # (telemetry_hz=0) matches collect-data and `axol teleop`.
        telemetry_hz=0.0,
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
    # Horizon fade for the temporal ensemble: each chunk's weight tapers to
    # near-zero over the last ``ensemble_blend_s`` seconds' worth of its
    # prediction horizon, so the ensemble doesn't step when the oldest
    # chunk's coverage expires mid-queue. 0 disables.
    ensemble_blend_s: float = 0.2
    # Chunk alignment: each incoming chunk's instantaneous offset from the
    # currently executing trajectory (the stale-observation re-anchor
    # disagreement, which scales with inference latency) is cancelled at
    # arrival and faded back in over this many seconds. Removes the ~5 Hz
    # advance/pull-back oscillation at its source while absolute corrections
    # still land with a ~1 s time constant; shape corrections (the actual
    # policy behavior) pass through unfaded. 0 disables.
    align_fade_s: float = 1.0
    # Per-joint velocity/acceleration limits for the execution-side
    # TrapezoidalFilter that shapes every command sent to the arms. The
    # defaults are teleop's constants — the only command profile the policy
    # ever saw in training — so legitimate policy motion passes through
    # essentially untouched while chunk-boundary snaps (which violate the
    # acceleration limit ~10x regardless of how slow the inference platform
    # is) get spread over the ticks the physical arm needs anyway. This is
    # what keeps the arm smooth on any inference hardware: a late chunk
    # decelerates the arm to a hold and ramps it back out instead of
    # freeze-then-lurch. Set either to 0 to disable the filter.
    exec_max_vel: float = VRTeleopConfig.teleop_max_vel
    exec_max_accel: float = VRTeleopConfig.teleop_max_accel
    # Contact watchdog for the between-episode return-to-rest: a joint torque
    # residual (measured minus modeled gravity, Nm) sustained above this
    # drops the arms into a limp gravity-comp hold instead of pulling
    # through — free them by hand, then continue (Enter / the panel's Start)
    # to replan from wherever they were left. 0 disables the watchdog.
    reset_torque_threshold: float = 4.0
    # Contact watchdog while the *policy* drives the arms: the same
    # sustained-torque-residual trip, checked on every executed action. On a
    # trip the episode aborts (nothing is saved) and the arms drop into the
    # limp gravity-comp hold; clear them by hand, then continue to return to
    # rest and start the next attempt. 0 (the default) disables it — the
    # policy pushes on the scene on purpose, so only the return-to-rest
    # guard is always on; set a threshold (16 is the control panel's
    # suggested value) to enable.
    policy_torque_threshold: float = 0.0
    # Velocity damping (Nm·s/rad) for that contact-fallback hold; same
    # semantics as `axol gravity-comp --kd`.
    reset_gravity_comp_kd: float = 0.25
    rerun_ip: str | None = None
    rerun_port: int = 9876
    log_level: LogLevel = "INFO"


# ----------------------------------------------------------------------
# Episode control: abstracts how start/save/rerecord/quit decisions arrive.
#
# The CLI reads them from stdin (``s`` / ``r`` / ``q`` + Enter prompts); the
# web control panel pushes them through a queue from the API. ``_run`` is
# agnostic — it only calls the small surface below, and phrases its operator
# prompts without naming a key or a button, since each control renders the
# affordance its own surface has (the terminal appends "[Enter]", the panel
# draws a button).
# ----------------------------------------------------------------------

# Buttons the panel renders per phase (see EpisodeControls in the web app).
# The gates aren't here: each has a single button whose label depends on what
# continuing does — start the next episode, or end a limp contact hold — so
# they're carried as gate state instead.
_POLICY_PHASE_CONTROLS: dict[str, tuple[dict[str, Any], ...]] = {
    "recording": (
        {"command": "s", "label": "Save"},
        {"command": "r", "label": "Discard"},
    ),
    # The time cap already stopped the rollout, but the operator still owes a
    # save/discard decision — so the same buttons stay live.
    "deciding": (
        {"command": "s", "label": "Save"},
        {"command": "r", "label": "Discard"},
    ),
}

# Panel status line per phase. The gate phases ("ready" / "contact") instead
# show their live gate message, which carries the operator instruction.
_POLICY_PHASE_MESSAGES: dict[str, str] = {
    "preparing": "Preparing…",
    "recording": "Episode running — Save to keep it, Discard to re-record.",
    "deciding": "Time cap reached — Save to keep it, Discard to re-record.",
    "resetting": "Returning to rest…",
}

# The guarded return-to-rest hit something and dropped the arms into a limp
# gravity-comp hold. Shared by both controls so the terminal prompt and the
# panel's status line say the same thing.
_CONTACT_PROMPT = (
    "Contact during return to rest — the arms are limp and free to move. "
    "Clear them, then continue to replan the return from where they are."
)

# A discarded episode drops the arms into the same limp hold so the operator
# can untangle/reposition them by hand before anything moves on its own.
_DISCARD_LIMP_PROMPT = (
    "Episode discarded — the arms are limp and free to move. "
    "Reposition them and reset the scene, then return to rest."
)

# The tracking contact watchdog tripped mid-episode (the policy pushed or
# pulled on something sustained above the threshold), so the rollout was
# aborted and the arms dropped into the limp hold.
_EPISODE_CONTACT_PROMPT = (
    "Contact — torque exceeded the threshold, so the episode stopped and "
    "the arms are limp and free to move. Clear them, then return to rest."
)


class _StdinPolicyControl:
    """Terminal episode control: stdin keystrokes + Enter-to-continue prompts."""

    def __init__(self) -> None:
        self._stop: "threading.Event | None" = None
        self._result: dict[str, str | None] = {"choice": None}
        self._thread: "threading.Thread | None" = None

    def await_continue(self, message: str) -> bool:
        try:
            input(f"{message} [Enter] ")
            return True
        except (EOFError, KeyboardInterrupt):
            return False

    def await_contact_clear(self) -> bool:
        return self.await_continue(_CONTACT_PROMPT)

    def await_episode_contact_clear(self) -> bool:
        return self.await_continue(_EPISODE_CONTACT_PROMPT)

    def await_manual_reset(self) -> bool:
        return self.await_continue(_DISCARD_LIMP_PROMPT)

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

    def note_saved(self) -> None:
        # Web-control-only bookkeeping; the terminal path shows saves via log_say.
        pass

    def close(self) -> None:
        self.end_episode()


class _QueuePolicyControl:
    """Web episode control: decisions arrive as API-pushed queue commands.

    Accepts ``s`` / ``r`` / ``q`` for the running episode and ``continue``
    (alias ``start``) to advance through a gate — the between-episode "reset
    the scene" one, or the limp hold a contact during the return-to-rest
    drops the arms into.

    :meth:`snapshot` is what the control panel renders — the phase, a status
    line, and the buttons that make sense right now — so the panel follows the
    run without hardcoding this flow.
    """

    def __init__(self, stop_event: "threading.Event") -> None:
        import queue

        self._q: "queue.Queue[str]" = queue.Queue()
        self._stop = stop_event
        self._choice: str | None = None
        # Episode phase/count, read by the serve runner so the web control panel
        # on ANY connected computer can show the right controls (see snapshot()).
        self._state_lock = threading.Lock()
        self._phase = "preparing"
        self._episodes_recorded = 0
        # Instruction + button label of the gate currently open, if any.
        self._gate_message = ""
        self._gate_label = "Start episode"

    def push(self, command: str) -> None:
        self._q.put(command)

    def _set_phase(self, phase: str) -> None:
        with self._state_lock:
            self._phase = phase

    def note_saved(self) -> None:
        with self._state_lock:
            self._episodes_recorded += 1

    def snapshot(self) -> dict[str, Any]:
        """Thread-safe phase/count/message/buttons for the /api/op/status API."""
        with self._state_lock:
            phase = self._phase
            if phase in ("ready", "contact", "limp"):
                message = self._gate_message
                controls = [{"command": "start", "label": self._gate_label}]
            else:
                message = _POLICY_PHASE_MESSAGES.get(phase, "")
                controls = [dict(c) for c in _POLICY_PHASE_CONTROLS.get(phase, ())]
            return {
                "phase": phase,
                # Saves are what number an episode, so a discarded rollout is
                # re-recorded under the same number — as the log line says.
                "episode": self._episodes_recorded + 1,
                "episodesRecorded": self._episodes_recorded,
                "message": message,
                "controls": controls,
            }

    def _drain(self) -> None:
        import queue

        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def _await_gate(self, phase: str, message: str, label: str) -> bool:
        """Open a gate for the panel and block until the operator resolves it."""
        import queue

        from lerobot.utils.utils import log_say

        log_say(message)
        with self._state_lock:
            self._phase = phase
            self._gate_message = message
            self._gate_label = label
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

    def await_continue(self, message: str) -> bool:
        return self._await_gate("ready", message, "Start episode")

    def await_contact_clear(self) -> bool:
        cleared = self._await_gate("contact", _CONTACT_PROMPT, "Return to rest")
        if cleared:
            # The return replans and plays from here, so drop the contact
            # buttons rather than leave them up over a moving arm.
            self._set_phase("resetting")
        return cleared

    def await_episode_contact_clear(self) -> bool:
        """Gate for the mid-episode contact hold; resolves to a return-to-rest."""
        cleared = self._await_gate("contact", _EPISODE_CONTACT_PROMPT, "Return to rest")
        if cleared:
            self._set_phase("resetting")
        return cleared

    def await_manual_reset(self) -> bool:
        """Gate for the discard-cleanup limp hold; resolves to a return-to-rest."""
        cleared = self._await_gate("limp", _DISCARD_LIMP_PROMPT, "Return to rest")
        if cleared:
            self._set_phase("resetting")
        return cleared

    def begin_episode(self) -> None:
        self._choice = None
        self._drain()
        self._set_phase("recording")

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
        # The cap stopped recording, but the operator still owes a save/rerecord/
        # quit decision — keep the choice controls live (begin/end_episode leave
        # the phase at "recording"/"resetting", neither of which enables them).
        self._set_phase("deciding")
        while not self._stop.is_set():
            try:
                cmd = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            if cmd in ("s", "r", "q"):
                self._set_phase("resetting")
                return cmd
        return "q"

    def end_episode(self) -> None:
        # The episode ended; the loop returns to rest before the next gate.
        self._set_phase("resetting")

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
    ensemble_blend_s: float = 0.2,
    align_fade_s: float = 1.0,
    exec_max_vel: float = VRTeleopConfig.teleop_max_vel,
    exec_max_accel: float = VRTeleopConfig.teleop_max_accel,
    policy_torque_threshold: float = 0.0,
) -> Any:
    """Construct an ``AxolRobotClient`` against an already-connected robot.

    Wrapped in a helper so the lerobot imports stay lazy.

    Args:
        config: Built ``RobotClientConfig``.
        robot: Connected ``AxolRobot`` instance, reused across episodes.
        publisher: Sink for executed actions, drained by the capture thread.
        aggregate_strategy: One of the ``--aggregate_fn`` choices.
        temporal_ensemble_coeff: Decay coefficient for temporal_ensemble.
        ensemble_blend_s: Ensemble horizon-fade duration (seconds); 0
            disables (see ``RunPolicyConfig.ensemble_blend_s``).
        align_fade_s: Chunk-alignment offset fade duration (seconds); 0
            disables (see ``RunPolicyConfig.align_fade_s``).
        exec_max_vel: Execution filter joint-velocity limit (rad/s); 0
            disables the filter (see ``RunPolicyConfig.exec_max_vel``).
        exec_max_accel: Execution filter joint-acceleration limit (rad/s²);
            0 disables the filter (see ``RunPolicyConfig.exec_max_accel``).
        policy_torque_threshold: Tracking contact watchdog threshold (Nm)
            checked on every executed action; a trip sets
            ``contact_tripped`` and shuts the episode down. ``<= 0``
            disables (the default; see
            ``RunPolicyConfig.policy_torque_threshold``).
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
        No per-step post-filter is applied (training actions are already
        EMA + trapezoidal-filtered in ``collect_data``); the only smoothing
        added on top of the ensemble is chunk-boundary continuity: the
        execution-side ``TrapezoidalFilter`` (same class and constants as
        teleop, i.e. the exact command profile the training data went
        through) shapes every command sent to the arms, and the horizon
        fade in ``_temporal_ensemble_aggregate`` removes chunk-expiry
        steps in the ensembled signal itself.
        """

        def __init__(  # type: ignore[no-untyped-def]
            self,
            config,
            robot,
            publisher,
            aggregate_strategy,
            temporal_ensemble_coeff,
            ensemble_blend_s,
            align_fade_s,
            exec_max_vel,
            exec_max_accel,
            policy_torque_threshold,
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
            # Indices of the gripper entries in the flat action vector, derived
            # from the robot's action space so it tracks both the joint layout
            # (16-dim, grippers at 7/15) and the Cartesian layout (14-dim,
            # grippers at 6/13). ``_temporal_ensemble_aggregate`` snaps these to
            # the newest contributing chunk so bang-bang grasps aren't smeared.
            self._gripper_indices = tuple(
                i
                for i, key in enumerate(robot.action_features)
                if key.endswith("gripper.pos")
            )
            self._publisher = publisher
            self._aggregate_strategy = aggregate_strategy
            self._temporal_ensemble_coeff = float(temporal_ensemble_coeff)
            # Ticks over which each chunk's ensemble weight fades to near-zero
            # at the end of its prediction horizon (0 = off).
            self._blend_steps = max(0, round(float(ensemble_blend_s) * config.fps))
            # Ticks over which a chunk's arrival-time alignment offset fades
            # back to its absolute prediction (0 = alignment off).
            self._align_ticks = max(0, round(float(align_fade_s) * config.fps))
            # Execution-side command shaper: every popped action's arm-joint
            # entries pass through a TrapezoidalFilter (velocity + acceleration
            # limited target tracker — the same class and constants teleop runs
            # and the training data was recorded through), so the commands the
            # motors see are C1-smooth no matter how discontinuous the chunk
            # stream is. Grippers bypass it (bang-bang by design), and the
            # Cartesian action layout is excluded — its values are EE poses,
            # not joint radians, so rad/s limits don't apply here; the robot
            # applies the same shaping to the IK *output* instead (see
            # ``AxolRobot._cartesian_action_to_targets``), so Cartesian
            # policies get the identical guarantee post-IK.
            self._arm_indices = [
                i
                for i in range(len(robot.action_features))
                if i not in self._gripper_indices
            ]
            self._exec_filter = None
            if (
                exec_max_vel > 0.0
                and exec_max_accel > 0.0
                and not getattr(robot.config, "observe_cartesian", False)
            ):
                self._exec_filter = TrapezoidalFilter(
                    float(exec_max_vel),
                    float(exec_max_accel),
                    config.environment_dt,
                )
            # Full unfiltered target of the last popped action (numpy), kept
            # so starvation ticks can keep converging toward it.
            self._exec_last_target: Any | None = None
            # ``(origin, packed_actions, timestamp)`` per chunk, sorted
            # oldest-first. ``packed_actions`` is a (chunk_size, action_dim)
            # tensor so aggregation runs as one batched op.
            self._chunk_buffer: list[tuple[int, Any, float]] = []
            self._race_fix_warned: bool = False
            # Surfaces unhandled control-loop exceptions (typically CAN
            # faults) to the episode supervisor for an immediate teardown.
            self.fatal_error: BaseException | None = None
            # Tracking contact watchdog: checked after every executed action.
            # A trip records the offending joint here and signals shutdown so
            # the episode supervisor aborts the rollout and holds limp.
            self._contact_threshold = float(policy_torque_threshold)
            self._contact_watchdog = ContactWatchdog(self._contact_threshold)
            self.contact_tripped: "tuple[str, float] | None" = None

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
            # Unseeded reset: the filter passes its first target through
            # unchanged, which is safe because the first chunk is anchored on
            # the arm's actual observed (resting) state.
            if self._exec_filter is not None:
                self._exec_filter.reset()
            self._exec_last_target = None
            self.action_chunk_size = -1
            self.must_go.set()
            self.fps_tracker.reset()
            self.shutdown_event.clear()
            self.start_barrier = _threading.Barrier(3)
            self._race_fix_warned = False
            self.fatal_error = None
            self._contact_watchdog = ContactWatchdog(self._contact_threshold)
            self.contact_tripped = None
            if self._publisher is not None:
                self._publisher.reset()

        def _install_future_queue(self, future_queue) -> None:  # type: ignore[no-untyped-def]
            """Swap in ``future_queue``, dropping already-executed timesteps.

            The control thread can pop further actions between the
            aggregator reading ``latest_action`` and the queue swap. Holding
            ``action_queue_lock`` while re-filtering against the live
            ``latest_action`` prevents the post-swap queue from walking
            ``latest_action`` backwards and snapping the arm.

            Discontinuities in the installed queue (chunk-boundary re-anchor
            snaps) are tolerated here: the execution-side TrapezoidalFilter
            in ``control_loop_action`` shapes whatever is popped before it
            reaches the motors.

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
            ``i = 0`` the oldest chunk, times a per-timestep fade over the
            last ``_blend_steps`` of each chunk's horizon (see the taper
            comment below). Incoming chunks are already aligned to the
            executing trajectory by ``_align_incoming_chunk`` before they
            reach this method. ``_gripper_indices`` bypass the
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

            with self.latest_action_lock:
                latest_action = self.latest_action

            self._chunk_buffer.append((new_origin, new_packed, new_timestamp))
            self._chunk_buffer.sort(key=lambda entry: entry[0])

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

            # ``mask[ci, ts]`` is 1.0 where chunk ``ci`` covers ``ts`` (binary,
            # used for coverage and the gripper newest-chunk one-hot).
            # ``taper[ci, ts]`` additionally fades each chunk's ensemble
            # weight to near-zero over the last ``_blend_steps`` timesteps of
            # its horizon: without it, the timestep where the oldest chunk's
            # coverage ends loses a full contributor at once and the ensemble
            # steps by ~1/n_chunks of the inter-chunk spread — tens of mrad
            # executed mid-queue, where the install-time blend can't reach.
            # With the fade, a chunk's influence is already ~1/fade of its
            # weight when it expires, and late-horizon ACT predictions are the
            # least accurate anyway.
            action_grid = torch.zeros(
                (n_chunks, n_ts, action_dim), dtype=dtype, device=device
            )
            mask = torch.zeros((n_chunks, n_ts), dtype=dtype, device=device)
            taper = torch.zeros((n_chunks, n_ts), dtype=dtype, device=device)
            fade = max(1, self._blend_steps)
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
                # Remaining horizon per covered ts: chunk_max - ts + 1 (>= 1),
                # so the fade never reaches exactly zero and a timestep covered
                # by a single chunk still averages to that chunk's value.
                remaining = torch.arange(
                    chunk_max - lo + 1, chunk_max - hi, -1, dtype=dtype, device=device
                )
                taper[ci, dst_start:dst_stop] = (remaining / fade).clamp(max=1.0)

            coeff = self._temporal_ensemble_coeff
            chunk_weights = torch.exp(
                -coeff * torch.arange(n_chunks, dtype=dtype, device=device)
            )
            weighted_mask = (
                chunk_weights.unsqueeze(1) * mask * taper
            )  # (n_chunks, n_ts)
            norm = weighted_mask.sum(dim=0).clamp(min=1e-12)
            ensembled = (action_grid * weighted_mask.unsqueeze(-1)).sum(
                dim=0
            ) / norm.unsqueeze(-1)  # (n_ts, action_dim)

            # Gripper carve-out: snap to the newest contributing chunk via
            # a reverse-cumsum one-hot (cheaper than ``torch.max`` on tiny
            # CPU tensors).
            reverse_cumsum = mask.flip(0).cumsum(dim=0).flip(0)
            newest_mask = mask * (reverse_cumsum == 1).to(dtype)
            for gidx in self._gripper_indices:
                ensembled[:, gidx] = (action_grid[:, :, gidx] * newest_mask).sum(dim=0)

            contributed_indices = (
                mask.any(dim=0).nonzero(as_tuple=False).flatten().tolist()
            )

            # The chunk-boundary continuity blend runs in
            # ``_install_future_queue`` (inside the queue lock), so the ramp is
            # applied to exactly the entries that survive the pop-race filter.
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

        def _align_incoming_chunk(self, incoming_actions) -> None:  # type: ignore[no-untyped-def]
            """Align an incoming chunk to the currently executing trajectory.

            A chunk is predicted from an observation taken one inference
            round-trip before it lands, of an arm that physically lags its
            command — so at arrival it systematically disagrees with the
            trajectory being executed by roughly the tracking error, and
            installing it unmodified yanks the target backward by that offset
            (a 5-6 Hz sawtooth measured on the CAN bus; with the execution
            filter it survives as a bounded-acceleration wobble the arm still
            tracks). Cancel the disagreement at the source: measure the
            chunk's instantaneous offset from the executing trajectory at the
            next-to-execute timestep once, at arrival, and add it to the
            chunk's values in place — faded out linearly over
            ``_align_ticks`` so the command still converges to the policy's
            absolute intent. Corrections live in the chunk's *shape* and pass
            through unfaded; only the stale-state DC disagreement is
            smoothed. The offset magnitude scales with inference latency, so
            a slow platform cancels a proportionally bigger step — smoothness
            stays independent of the inference hardware.

            Runs at the aggregation dispatch so every strategy benefits
            (``temporal_ensemble`` and the upstream scalar blends alike), and
            it is action-space agnostic: pure vector offsets work for joint
            radians and for Cartesian pose values (positions in metres,
            axis-angle orientation — locally linear; a wrap at ±π could
            misread the offset, but poses one inference latency apart never
            span that). Grippers are exempt (bang-bang by design).

            Args:
                incoming_actions: Chunk from the policy server; mutated in
                    place (tensors are owned by the receive path).
            """
            if self._align_ticks <= 0 or not incoming_actions:
                return
            last_target = self._exec_last_target
            if last_target is None:
                return
            import torch

            sorted_in = sorted(incoming_actions, key=lambda a: a.get_timestep())
            with self.latest_action_lock:
                anchor_ts = self.latest_action + 1
            origin = sorted_in[0].get_timestep()
            idx = min(max(anchor_ts - origin, 0), len(sorted_in) - 1)
            anchor_action = sorted_in[idx].get_action()
            offset = (
                torch.from_numpy(last_target).to(anchor_action.dtype) - anchor_action
            )
            for gidx in self._gripper_indices:
                offset[gidx] = 0.0
            for ta in sorted_in:
                fade = 1.0 - (ta.get_timestep() - anchor_ts) / self._align_ticks
                fade = min(max(fade, 0.0), 1.0)
                if fade > 0.0:
                    ta.get_action().add_(offset * fade)

        def _aggregate_action_queues(self, incoming_actions, aggregate_fn=None):  # type: ignore[no-untyped-def]
            """Align the chunk, then dispatch to the configured aggregation."""
            self._align_incoming_chunk(incoming_actions)
            if self._aggregate_strategy == "temporal_ensemble":
                return self._temporal_ensemble_aggregate(incoming_actions)
            return super()._aggregate_action_queues(incoming_actions, aggregate_fn)

        def _shape_and_send(self, target_vec):  # type: ignore[no-untyped-def]
            """Run ``target_vec`` through the execution filter and send it.

            The arm-joint entries track the target under teleop's velocity +
            acceleration limits; grippers pass through raw. With the filter
            disabled the target is sent as-is.

            Args:
                target_vec: Full action vector (numpy, robot action order).

            Returns:
                The dict returned by ``robot.send_action``.
            """
            out = target_vec
            if self._exec_filter is not None:
                shaped = self._exec_filter.update(target_vec[self._arm_indices])
                out = target_vec.copy()
                out[self._arm_indices] = shaped
            # Inlined from upstream RobotClient._action_tensor_to_action_dict
            # so we depend only on the public ``robot.action_features`` rather
            # than a private LeRobot method. Maps the flat action vector to a
            # {motor_name: float} dict by position.
            action = {
                key: float(out[i]) for i, key in enumerate(self.robot.action_features)
            }
            performed = self.robot.send_action(action)
            if self._publisher is not None and performed is not None:
                self._publisher.publish(performed)
            # Tracking contact watchdog: a torque residual sustained above
            # the threshold means the policy is pushing/pulling on something
            # beyond legitimate task contact — abort the episode so the
            # supervisor can hold the arms limp instead of grinding on.
            if self.contact_tripped is None:
                tripped = self._contact_watchdog.update(self.robot.torque_residuals())
                if tripped is not None:
                    joint, residual = tripped
                    _logger.warning(
                        "policy contact: %s torque residual %.1f exceeds "
                        "%.1f — stopping the episode",
                        joint,
                        residual,
                        self._contact_threshold,
                    )
                    self.contact_tripped = tripped
                    self.shutdown_event.set()
            return performed

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

            target_vec = timed_action.get_action().numpy().astype(np.float32)
            self._exec_last_target = target_vec
            performed = self._shape_and_send(target_vec)

            if verbose:
                self.logger.debug(
                    f"Ts={timed_action.get_timestamp()} | "
                    f"Action #{timed_action.get_timestep()} performed | "
                    f"Queue size: {qs_after}"
                )
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
                    elif (
                        self._exec_filter is not None
                        and self._exec_last_target is not None
                        and self._exec_filter.position is not None
                        and not np.array_equal(
                            self._exec_filter.position,
                            self._exec_last_target[self._arm_indices],
                        )
                    ):
                        # Queue starvation (slow inference platform or a lost
                        # chunk): keep ticking the filter toward the last
                        # target so the arm decelerates smoothly to a hold
                        # instead of freezing mid-velocity, and ramps back out
                        # when the late chunk lands. Once converged this stops
                        # sending (the impedance controller holds position).
                        self._shape_and_send(self._exec_last_target)
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
            """Dedicated thread: capture and send paced observations.

            Upstream fires an observation whenever the queue is below
            ``chunk_size_threshold`` — but chunks are consumed from the
            moment they land, so with chunking latency the queue almost
            never sits above the threshold and the loop free-runs: every
            camera-capture interval it ships another multi-MB pickled
            observation whose serialization competes with the 60 Hz
            control thread for the GIL, and the server dedups most of
            them by timestep anyway. Two extra gates restore the
            intended cadence:

            - progress gate: skip while ``latest_action`` hasn't
              advanced since the last send — a re-send would carry the
              same timestep the server already predicted (this alone
              removes the episode-start burst while the first chunk is
              still being inferred);
            - pace gate: at most one send per drained-threshold's worth
              of executed actions
              (``(1 - chunk_size_threshold) × actions_per_chunk`` ticks).

            A stale-send timeout overrides the progress gate so a lost
            observation or chunk can't deadlock the episode.
            """
            self.start_barrier.wait()
            self.logger.info("Observation loop thread starting (paced)")
            min_interval = (
                (1.0 - self.config.chunk_size_threshold)
                * self.config.actions_per_chunk
                * self.config.environment_dt
            )
            resend_timeout = max(1.0, 2.0 * min_interval)
            last_sent_step = -1
            last_send_time = float("-inf")
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
                    with self.latest_action_lock:
                        # Match the timestep the observation would carry
                        # (control_loop_observation clamps the same way).
                        current_step = max(self.latest_action, 0)
                    now = time.perf_counter()
                    stale = (now - last_send_time) >= resend_timeout
                    if (
                        queue_fraction <= self.config.chunk_size_threshold
                        and (now - last_send_time) >= min_interval
                        and (current_step != last_sent_step or stale)
                    ):
                        self.control_loop_observation(task, verbose)
                        last_sent_step = current_step
                        last_send_time = time.perf_counter()
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
        ensemble_blend_s,
        align_fade_s,
        exec_max_vel,
        exec_max_accel,
        policy_torque_threshold,
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

    # Import lerobot's RobotClient module first, through the shim that undoes
    # its root-logger takeover — otherwise every CAN frame send gets debug-
    # logged to disk, which throttles the 60 Hz control loop (arm jitter).
    from ..lerobot.inference_patch import import_robot_client_preserving_logging

    import_robot_client_preserving_logging()

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

    # Finalize the camera set before the robot opens the cameras: prune the
    # unassigned placeholder slots (at least one must be set, and should be the
    # cameras the policy was trained on) and flag any physically-stereo ZED X so
    # it opens on the stereo grab path (a mono open of a ZED X fails connect).
    # Shared with collect-data via ``prepare_capture_cameras`` so a policy sees
    # the same observation set it was trained on.
    if isinstance(robot_config, AxolRobotConfig):
        from ..zed import stereo_serials

        robot_config.prepare_capture_cameras(stereo_serials(), minimum=1)
        if not robot_config.observation_cameras():
            raise ValueError(
                "run-policy has no camera with recording enabled — every assigned "
                "camera is set to stream-only (or recording is turned off). The "
                "policy needs the cameras it was trained on; enable recording for "
                "them in the Cameras dialog."
            )

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

        from lerobot.configs.video import RGBEncoderConfig

        rgb_encoder = RGBEncoderConfig(vcodec=vcodec)
        if is_complete:
            check_resume_consistency(dataset_root)
            log_say(f"Resuming existing dataset at {dataset_root}.")
            dataset = LeRobotDataset.resume(
                repo_id=repo_id,
                root=str(dataset_root),
                image_writer_threads=4,
                streaming_encoding=True,
                encoder_threads=4,
                rgb_encoder=rgb_encoder,
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
                rgb_encoder=rgb_encoder,
            )
            # LeRobot's codebase_version only identifies its dataset schema;
            # record Axol's Cartesian world frame separately so this fresh
            # dataset can never be mistaken for pre-v0.1.32 data and rotated
            # a second time by migrate-dataset.
            action_names = (action_features.get(ACTION) or {}).get("names") or []
            if any("_ee." in name for name in action_names):
                from ..recording.cartesian_frame import write_cartesian_frame_marker

                write_cartesian_frame_marker(dataset_root)

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
        # Evict a leftover PolicyServer from a crashed/previous run before
        # spawning ours, otherwise it would bind-fail and ``_wait_for_port``
        # would silently attach to the stale (wrong-policy) server.
        from ..utils.ports import reclaim_port

        reclaim_port(server_port)
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

    def _return_to_rest_guarded() -> bool:
        """Guarded return to rest; ``False`` when the operator aborted.

        Plays with the torque watchdog live; on contact the arms drop into
        a limp gravity-comp hold, and the contact gate (Enter on the
        terminal, "Return to rest" on the control panel) retries from
        wherever they were hand-guided to.
        """
        return reset_controller.return_to_rest(
            robot,
            torque_threshold=cfg.reset_torque_threshold,
            gravity_comp_kd=cfg.reset_gravity_comp_kd,
            stopped=stop_event.is_set,
            wait_retry=control.await_contact_clear,
        )

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
            ensemble_blend_s=cfg.ensemble_blend_s,
            align_fade_s=cfg.align_fade_s,
            exec_max_vel=cfg.exec_max_vel,
            exec_max_accel=cfg.exec_max_accel,
            policy_torque_threshold=cfg.policy_torque_threshold,
        )

        log_say("Loading policy on server (one-time)...")
        if not client.start():
            raise RuntimeError("Failed to connect to policy server / load policy.")

        log_say("Connecting robot...")
        robot.connect()

        # A Cartesian-action policy resolves each action to joints via IK in
        # send_action. Build that solver now, before the control loop, so its
        # one-time JIT warmup overlaps the return-to-rest + scene-reset prompt
        # below instead of stalling the first policy action.
        if getattr(robot.config, "observe_cartesian", False):
            log_say("Preparing Cartesian action solver (IK)...")
            robot.prepare_cartesian_actions()

        log_say("Returning to rest pose.")
        if not _return_to_rest_guarded():
            return
        if not control.await_continue("Reset the scene, then start the first episode."):
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

            # CPython's cyclic GC is stop-the-world: a gen-2 pass over this
            # process (multi-MB observation graphs, grpc/protobuf cycles) was
            # measured at ~516 ms and fired once per episode at the same
            # allocation-driven point (~24 s in), freezing every thread
            # including the 60 Hz control loop — the arm halts mid-motion and
            # snaps. Sweep now, while the arm is still stationary, then hold
            # automatic collection for the episode: refcounting still frees
            # the (acyclic) tensors and frames immediately, and the deferred
            # cyclic garbage is collected in the ``finally`` below. The
            # re-enable lives in a ``finally`` because run-policy also runs
            # in-process under ``axol serve``: an exception escaping the
            # episode must not leave automatic collection off for the rest
            # of the serve process's lifetime.
            gc.collect()
            gc.disable()
            timed_out = False
            interrupted = False
            try:
                receiver_thread.start()
                control_thread.start()
                obs_thread.start()
                if capture is not None:
                    capture.start()

                deadline = time.perf_counter() + episode_time_s
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
                        if client.contact_tripped is not None:
                            # Tracking contact — the client already signalled
                            # shutdown; the limp hold runs after teardown.
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
            finally:
                # Sweep the cyclic garbage deferred during the episode. The
                # duration doubles as confirmation of the mid-episode GC stall
                # diagnosis: a multi-hundred-ms sweep here is the pause that
                # previously landed inside the control loop.
                gc.enable()
                gc_t0 = time.perf_counter()
                gc.collect()
                _logger.info(
                    "End-of-episode gc.collect: %.0f ms",
                    (time.perf_counter() - gc_t0) * 1000.0,
                )

            if interrupted or stop_event.is_set():
                if dataset is not None:
                    dataset.clear_episode_buffer()
                break
            if client.fatal_error is not None:
                if dataset is not None:
                    dataset.clear_episode_buffer()
                break

            if client.contact_tripped is not None:
                # The tracking contact watchdog aborted the rollout: nothing
                # is saved, and the arms drop into the limp gravity-comp hold
                # so the operator can clear them by hand. The gate ("Return
                # to rest" on the panel, Enter on the terminal) then plans
                # the return from wherever the arms were left.
                joint, _residual = client.contact_tripped
                log_say(f"Contact on {joint} — episode aborted; arms are limp.")
                if dataset is not None:
                    dataset.clear_episode_buffer()
                if not reset_controller.hold_limp(
                    robot,
                    gravity_comp_kd=cfg.reset_gravity_comp_kd,
                    wait=control.await_episode_contact_clear,
                    stopped=stop_event.is_set,
                ):
                    break
                log_say("Returning to rest pose.")
                if not _return_to_rest_guarded():
                    break
                if not control.await_continue(
                    "Reset the scene, then start the next episode."
                ):
                    break
                continue

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
                # A discarded episode usually means the arms are somewhere they
                # shouldn't be (hooked on the scene, mid-failure): drop them
                # into a limp gravity-comp hold for hand-repositioning instead
                # of pulling straight back to rest. The gate's "Return to rest"
                # (Enter on the terminal) then plans the return from wherever
                # the arms were left.
                log_say("Arms are limp for cleanup.")
                if not reset_controller.hold_limp(
                    robot,
                    gravity_comp_kd=cfg.reset_gravity_comp_kd,
                    wait=control.await_manual_reset,
                    stopped=stop_event.is_set,
                ):
                    break
                log_say("Returning to rest pose.")
                if not _return_to_rest_guarded():
                    break
                if not control.await_continue("Start the episode again when ready."):
                    break
                continue

            # choice == "s"
            if dataset is not None:
                dataset.save_episode()
                # Flush the episode to disk so a kill can't lose it (mirrors
                # collect-data — see make_episode_durable). Best-effort: on
                # failure the episode is still saved and its remaining rows
                # reach disk at the next save or finalize (make_episode_durable
                # leaves the writers consistent either way).
                try:
                    make_episode_durable(dataset)
                except Exception:  # noqa: BLE001 - durability is best-effort
                    _logger.exception(
                        "could not fully flush the saved episode to disk; it "
                        "completes at the next save or finalize — do not kill "
                        "this process"
                    )
                # The serve unit records as root into the operator's home; hand
                # the tree back after every save so a crash never leaves a
                # root-owned dataset behind (no-op off the root service). After
                # the durable flush so the episode's freshly rotated files are
                # all on disk and covered by the chown.
                restore_dataset_ownership(dataset_root)
            episodes_recorded += 1
            control.note_saved()
            log_say(f"Saved episode {episodes_recorded}.")
            log_say("Returning to rest pose.")
            if not _return_to_rest_guarded():
                break
            if not control.await_continue(
                "Reset the scene, then start the next episode."
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
        elif dataset_root is not None:
            # Finalize wrote the last meta/stats files as root; adopt them too.
            # After the wipe above, so a discarded dataset isn't chowned on its
            # way to being deleted.
            restore_dataset_ownership(dataset_root)

        # Restore the default handler so Ctrl+C can still kill any leaked
        # non-daemon thread keeping the interpreter alive.
        try:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        except (ValueError, OSError):
            pass
