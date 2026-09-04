"""
axol collect-dagger

DAgger data collection: run a trained policy on the Axol robot while the
operator watches in VR and can intervene to guide it, and record the whole
thing (policy segments + human corrections) as training data.

The VR grip ("side") buttons drive the intervention flow during an episode:

    POLICY  --(either grip alone)-->  FROZEN   robot holds pose; capture PAUSED
    FROZEN  --(both grips)------->    TELEOP   operator drives;  capture RESUMES
    TELEOP  --(either grip alone)-->  POLICY   policy continues; capture keeps running

Nothing is recorded between the freeze and the takeover, so the dataset flows
seamlessly from the policy's motion into the correction (LeRobot frame
timestamps are index-based; the capture thread re-anchors its clock on resume).
On takeover the IK worker and smoothing filters are synced to the robot's
*measured* pose first (see :mod:`almond_axol.teleop.dagger`), so engaging never
drags the arm toward a stale target; the grippers adopt the controller
triggers immediately — pre-set them before engaging (hold the trigger down
so a held part stays gripped). When the operator hands back, the policy
backend is reset so it re-plans from the corrected pose instead of continuing
from its pre-freeze state, and the policy velocity envelope is re-anchored at
the corrected pose.

Episode boundaries are driven from VR exactly like ``collect-data``: press
record to start an episode (the policy starts driving and recording begins),
press it again to end and save it, or press it with the reset button held to
discard and re-record. ``s`` / ``r`` / ``q`` on stdin mirror the same choices
for an assistant at the terminal, the ``--episode_time_s`` safety cap saves
the episode, and typing a subtask number switches the policy's instruction
mid-episode when ``--subtasks`` is supplied. Between episodes the
collision-aware ``IKResetController`` homes the arms, and the grips work as
plain teleop (engage with both, release with one — nothing is recorded) so
the operator can reset the scene with the arms, mirroring ``collect-data``'s
pre-record phase; the VR reset button homes the arms on demand, and they
re-home when record is pressed if teleop left them away from rest, so the
policy always starts from the rest pose. Every one of those rest moves is
*guarded*: a torque-residual contact watchdog (the shared teleop-config knobs
``--teleop_config.vr_teleop_config.reset_torque_threshold`` /
``.reset_gravity_comp_kd``) stops the move on unexpected contact and drops
the arms into a limp gravity-comp hold; the VR reset button (idle-phase
homes) or the continue gate (terminal Enter / panel button) retries from
wherever the arms were hand-guided to.

Inference runs in-process, one policy call per control tick (the same
pre/post-processor + ``select_action`` pipeline as LeRobot's sync rollout
engine). Interventions are annotated the way LeRobot's own DAgger rollout
strategy annotates them: the dataset declares a per-frame bool
``intervention`` feature and every row recorded while the operator was
driving is tagged ``True`` (rows from the policy are ``False``). On top of
that, every intervention span is tracked in *dataset* time (recorded
frames / fps — the episode's own timeline, so spans stay inside the episode
boundaries despite the unrecorded frozen gaps); a package building on this
command can persist them in other formats (see :class:`_DaggerControlLoop`).
The policy backend itself is pluggable (:class:`DaggerPolicy`), so a
downstream package can drive the same session loop with a remotely-hosted
policy.

Camera and recording plumbing follows ``collect-data``'s proven
out-of-process split, with one twist this flow needs: the video relay owns
the ZED cameras and streams the headset view, but its raw branch is forced
onto the **pyshm transport** (``raw_transport: "pyshm"``) so the shared-memory
frames are readable by *this* process — the policy builds its observations
from them — as well as by the ``DatasetRecorderProcess`` subprocess that owns
the dataset (NVENC-encoding on its own cores). The frozen gap uses the
recorder's ``pause_episode``/``resume_episode`` gate (the capture clock
re-anchors on resume, so episodes play straight through the gap). Nothing
camera- or encode-related runs in the control process, which is what keeps
the policy at fps and teleop at ``--teleop_hz``. The relay is required —
there is no in-process fallback (per-frame camera Python in the control
process starves the control loops and the CAN bus).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig

from ..lerobot.camera.configuration_zed import ZedCameraConfig
from ..lerobot.robot.config_axol import AxolRobotConfig
from ..lerobot.rollout import (
    IKResetController,
    PolicyActionLimiter,
)
from ..lerobot.teleop.config_vr import AxolVRTeleopConfig
from ..recording import (
    DatasetRecorderProcess,
    RecorderCaptureError,
    RecorderDatasetSaveError,
    default_vcodec,
)
from ..utils.control_loop import run_blocking_with_sync_control_ticks
from .collect_data import (
    _existing_dataset_resolution,
    _start_video_relay,
    check_resume_consistency,
)
from .config import DatasetResolution, LogLevel, PolicyType, parse
from .run_policy import _GATE_CONTACT, _QueuePolicyControl, _StdinPolicyControl

if TYPE_CHECKING:
    from ..lerobot.robot.robot_axol import AxolRobot
    from ..lerobot.teleop.teleop_vr_dagger import DaggerVRTeleop

_logger = logging.getLogger(__name__)


# Control-loop states during an episode (see module docstring).
_STATE_POLICY = "policy"
_STATE_FROZEN = "frozen"
_STATE_TELEOP = "teleop"

# LeRobot's native DAgger annotation: a per-frame bool feature tagging rows
# recorded while the operator was driving (the same feature lerobot's own
# DAgger rollout strategy declares). Added to every dataset this command
# creates; the recorder tags each row from the control loop's published
# intervention flag.
INTERVENTION_FEATURE: dict[str, Any] = {"dtype": "bool", "shape": (1,), "names": None}


def _default_robot_config() -> AxolRobotConfig:
    """Default Axol robot config for DAgger collection: local ZED cameras.

    All three slots are seeded with the unassigned sentinel serial ``0`` (see
    ``collect-data``); assign the cameras the policy was trained on (draccus
    takes dict fields as one inline YAML/JSON value, e.g. ``--robot_config.cameras
    "{overhead: {serial: 41234567}}"``). The video relay owns the cameras (see
    the module docstring); the robot sees them as shared-memory readers.
    Stiffness and gripper torque keep the library data-collection defaults so
    corrections are collected under the same dynamics as demonstrations;
    override from the CLI to match the station (e.g.
    ``--robot_config.axol_config.left_stiffness 0.5``).

    The Rust core owns the continuous command and feedback streams. The DAgger
    loop only publishes targets and reads the core's latest measured state,
    matching ``collect-data`` without a separate Python telemetry poll.
    """
    return AxolRobotConfig(
        cameras={
            "overhead": ZedCameraConfig(serial=0),
            "left_arm": ZedCameraConfig(serial=0),
            "right_arm": ZedCameraConfig(serial=0),
        },
    )


@dataclass
class DaggerConfig:
    """Config for ``axol collect-dagger``.

    The policy side mirrors ``run-policy`` (``--policy_path`` /
    ``--policy_type`` / ``--device``, but inference is in-process and
    synchronous — one ``select_action`` per tick); the recording side mirrors
    ``collect-data`` (``--dataset_resolution`` for the relay's dataset branch,
    ``teleop_config`` for the VR server / IK / smoothing parameters).
    """

    policy_path: str
    policy_type: PolicyType
    task: str
    repo_id: str
    # Ordered per-step instructions; typing a number 1..N + Enter (or pushing
    # it from the web panel) switches the instruction sent to the policy
    # mid-episode without ending it. The dataset's task string stays --task.
    subtasks: list[str] | None = None
    # Safety cap per episode; hitting it saves the episode. DAgger episodes
    # include interventions, so the default is generous.
    episode_time_s: int = 600
    fps: int = 60
    # Velocity/acceleration envelope over the policy's arm actions (rad/s,
    # rad/s²) — see PolicyActionLimiter. Transparent for normal trained
    # motion; only engages on discontinuities (policy outliers, re-plans from
    # stale observations). Set policy_max_vel to 0 to disable. Teleop actions
    # are already enveloped by the teleop smoothing stack.
    policy_max_vel: float = 6.2832
    policy_max_accel: float = 21.9911
    # Control rate while the operator is engaged (the TELEOP state only). The
    # policy state always ticks at --fps (the policy was trained on fps-spaced
    # actions). Teleop ticks faster for smoother commanded motion (the robot's
    # velocity feedforward differentiates commanded positions, so a higher
    # command rate means smaller, gentler steps) — matches collect-data's
    # dedicated 120 Hz teleop loop. The teleop smoothing frequency is pinned
    # to this value at startup.
    teleop_hz: int = 120
    device: str = "cuda"
    # Downscale target for the relay's dataset/raw branch — also the
    # resolution the policy's observations arrive at. Mirrors collect-data.
    dataset_resolution: DatasetResolution = "SVGA"
    vcodec: str = field(default_factory=default_vcodec)
    robot_config: RobotConfig = field(default_factory=_default_robot_config)
    teleop_config: TeleoperatorConfig = field(default_factory=AxolVRTeleopConfig)
    root: str | None = None
    push_to_hub: bool = False
    rerun_ip: str | None = None
    rerun_port: int = 9876
    log_level: LogLevel = "INFO"


def main(argv: list[str]) -> None:
    """Parse the CLI config and run the session, exiting cleanly on hardware faults."""
    cfg = parse(DaggerConfig, argv)
    # force=True: importing lerobot (at module load) installs a root handler
    # and leaves the root level at WARNING, which would otherwise make this a
    # no-op and silently drop every log_say() status line.
    logging.basicConfig(level=getattr(logging, cfg.log_level), force=True)

    import sys

    import can

    from ..motor.errors import MotorError

    try:
        _run(cfg)
    except (MotorError, can.CanError) as exc:
        _logger.error("Robot hardware error: %s. Exiting.", exc)
        sys.exit(1)


# ----------------------------------------------------------------------
# Policy backend
# ----------------------------------------------------------------------


class DaggerPolicy(Protocol):
    """The policy backend the DAgger session loop drives.

    The stock backend is :class:`_LocalPolicy` (in-process LeRobot sync
    inference); a package building on ``almond-axol`` can implement this
    protocol to drive the same session with e.g. a remotely-hosted policy
    (its own ``_run`` wires the pieces together — see
    :class:`_DaggerControlLoop` and :func:`_idle_teleop_until_record`).
    """

    def connect(self, robot: "AxolRobot") -> None:
        """Load/prepare the policy. Called once, after ``robot.connect()``."""
        ...

    def reset(self) -> None:
        """Clear episode-scoped state (hidden state, obs history, buffered
        chunks). Called at episode start and on every operator hand-back, so
        the policy re-plans from the corrected pose."""
        ...

    def set_instruction(self, text: str) -> None:
        """Switch the instruction sent on subsequent ticks (subtask switch)."""
        ...

    def act(self, observation: dict[str, Any]) -> dict[str, float] | None:
        """Return the next action for ``observation`` (joint state + one
        frame per camera, synchronized at sensor exposure), or ``None`` to skip
        the tick (e.g. an unusable observation)."""
        ...

    def close(self) -> None:
        """Release the backend's resources at session end."""
        ...


class _LocalPolicy:
    """In-process LeRobot policy backend: one ``select_action`` per tick.

    Runs the same pipeline as LeRobot's sync rollout engine — build a dataset
    frame from the observation, ``prepare_observation_for_inference``,
    preprocessor, ``select_action``, postprocessor, map the action tensor
    back to named joints. Synchronous inference is the right shape for
    DAgger's dual-rate loop: the POLICY state consumes exactly one action per
    tick, and chunked/async backends would keep planning from pre-freeze
    observations across an intervention.
    """

    def __init__(
        self, policy_path: str, policy_type: str, device: str, task: str
    ) -> None:
        self._path = policy_path
        self._type = policy_type
        self._device_str = device
        self._instruction = task
        self._policy: Any = None
        self._pre: Any = None
        self._post: Any = None
        self._features: dict[str, Any] = {}
        self._action_keys: list[str] = []
        self._robot_type = ""
        self._obs_proc: Any = None
        self._device: Any = None

    def connect(self, robot: "AxolRobot") -> None:
        """Load the policy + processors and bind the robot's feature layout."""
        import torch
        from lerobot.configs.types import FeatureType
        from lerobot.policies import get_policy_class, make_pre_post_processors
        from lerobot.processor import make_default_processors
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import hw_to_dataset_features

        _logger.info("Loading policy %s (%s)...", self._path, self._type)
        policy_class = get_policy_class(self._type)
        policy = policy_class.from_pretrained(self._path)
        policy = policy.to(self._device_str)
        policy.eval()

        # Fail fast if the policy's visual inputs and the configured cameras
        # can't be reconciled (mirrors LeRobot's rollout context check).
        expected_visuals = {
            k
            for k, v in policy.config.input_features.items()
            if v.type == FeatureType.VISUAL
        }
        provided_visuals = {
            f"observation.images.{k}"
            for k, v in robot.observation_features.items()
            if isinstance(v, tuple)
        }
        if not expected_visuals <= provided_visuals:
            raise ValueError(
                "Visual feature mismatch between policy and robot cameras.\n"
                f"Policy expects: {sorted(expected_visuals)}\n"
                f"Robot provides: {sorted(provided_visuals)}"
            )

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=self._path,
            dataset_stats=None,
            preprocessor_overrides={
                "device_processor": {"device": self._device_str},
            },
        )

        action_features = hw_to_dataset_features(robot.action_features, ACTION)
        obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
        _, _, self._obs_proc = make_default_processors()
        self._features = {**action_features, **obs_features}
        self._action_keys = list(robot.action_features.keys())
        self._robot_type = robot.name
        self._policy = policy
        self._pre = preprocessor
        self._post = postprocessor
        self._device = torch.device(self._device_str)
        _logger.info("Policy loaded on %s.", self._device_str)

    def reset(self) -> None:
        """Reset the policy and pre/post-processors (episode start, hand-back)."""
        self._policy.reset()
        self._pre.reset()
        self._post.reset()

    def set_instruction(self, text: str) -> None:
        """Switch the task string sent with subsequent observations."""
        self._instruction = text

    def act(self, observation: dict[str, Any]) -> dict[str, float] | None:
        """Run the full inference pipeline and return a named action dict."""
        from contextlib import nullcontext

        import torch
        from lerobot.policies.utils import (
            make_robot_action,
            prepare_observation_for_inference,
        )
        from lerobot.utils.constants import OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame

        obs_processed = self._obs_proc(observation)
        obs_frame = build_dataset_frame(self._features, obs_processed, prefix=OBS_STR)

        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            prepared = prepare_observation_for_inference(
                obs_frame, self._device, self._instruction, self._robot_type
            )
            prepared = self._pre(prepared)
            action = self._policy.select_action(prepared)
            action = self._post(action)

        action_dict = make_robot_action(action, self._features)
        return {key: float(action_dict[key]) for key in self._action_keys}

    def close(self) -> None:
        """Drop the model reference (CUDA memory is released with the process)."""
        self._policy = None
        self._pre = None
        self._post = None


# ----------------------------------------------------------------------
# Control loop: policy / frozen / teleop
# ----------------------------------------------------------------------


class _DaggerControlLoop(threading.Thread):
    """Drive the robot from the policy or the operator, at a per-state rate.

    The POLICY and FROZEN states tick at ``fps`` (the policy was trained on
    fps-spaced actions and its backend produces one action per ``act()``
    call); the TELEOP state ticks at ``teleop_hz`` so engaged motion gets
    the same command rate (and smoothness) as ``collect-data``'s dedicated
    teleop loop. The recorder subprocess samples the dataset at ``fps`` on
    its own clock either way, so the recorded rate is unaffected.

    Each tick first polls the VR episode events (record button →
    ``vr_choice`` ``'s'``/``'r'`` ends the loop), then advances the
    intervention state machine from the teleop's engage state + freeze latch,
    then commands the robot from whichever source the state selects:

    - POLICY: sensor-exposure-aligned observation → ``policy.act`` → action
      dict, through the velocity envelope.
    - TELEOP: ``teleop.get_action()`` (the smoothed IK output, seeded at the
      robot's pose on takeover).
    - FROZEN: re-send the last commanded action so the robot holds pose and
      command replies keep the joint cache fresh.

    Every tick publishes the ``(joint_obs, action)`` snapshot to the recorder
    subprocess (a small shared-memory write), which pairs it with the relay's
    camera frames on its configured recording clock. Policy snapshots carry the
    median exposure timestamp of the images that produced the action; TELEOP
    snapshots carry their live control-tick time. The TELEOP state publishes with
    ``intervention=True``, which is how the recorder tags the dataset's
    per-frame ``intervention`` feature (LeRobot's native DAgger annotation).
    The frozen gap is gated in the recorder
    (``pause_episode``/``resume_episode``), and the intervention
    span times come from its row counts — dataset time = rows / fps, the
    episode's own timeline. The FROZEN state still publishes the held action
    so the recorder's snapshot always matches the live command when capture
    resumes.

    Surfaces unhandled exceptions via :attr:`fatal_error` so the episode
    supervisor can tear down cleanly.
    """

    def __init__(
        self,
        *,
        robot: "AxolRobot",
        policy: DaggerPolicy,
        teleop: "DaggerVRTeleop",
        recorder: DatasetRecorderProcess,
        fps: int,
        teleop_hz: int,
        limiter: "PolicyActionLimiter | None" = None,
    ) -> None:
        super().__init__(name="axol-dagger-control", daemon=True)
        self.robot = robot
        self.policy = policy
        self.teleop = teleop
        self.recorder = recorder
        self.limiter = limiter
        self.fps = fps
        self.teleop_hz = teleop_hz
        self.shutdown_event = threading.Event()
        self.fatal_error: BaseException | None = None
        # A capture-integrity rejection is scoped to this episode. The
        # supervisor joins the controller/capture thread, discards the buffer,
        # homes, and retries the same episode index. Lifecycle/IPC/control
        # failures continue to use fatal_error.
        self.capture_error: str | None = None
        # Episode outcome signalled from the VR record button: 's' (terminate)
        # or 'r' (reset+stop). Read by the supervisor.
        self.vr_choice: str | None = None
        self.state = _STATE_POLICY
        self.interventions = 0
        # Operator-takeover spans in dataset time (recorder row counts / fps —
        # the episode's own timeline; wall time diverges across every frozen
        # gap). ``open_span_start`` is the start of a span still open when the
        # loop exits (episode ended mid-intervention); the supervisor closes
        # it at the final row count after it pauses the recorder.
        self.intervention_spans: list[tuple[float, float]] = []
        self.open_span_start: float | None = None

    def _policy_tick(self) -> dict[str, float] | None:
        """One policy inference tick; returns the sent action or ``None``.

        ``None`` means the tick was skipped (observation/camera hiccup, or
        the backend declined the observation) — skip-and-retry.
        """
        try:
            obs, observation_ts = self.robot.get_observation_with_capture_timestamp()
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Observation failed (%s); skipping tick.", exc)
            return None

        action_dict = self.policy.act(obs)
        if action_dict is None or self.shutdown_event.is_set():
            return None

        if self.limiter is not None:
            action_dict = self.limiter.apply(action_dict)

        performed = self.robot.send_action(action_dict)
        # obs carries the historical joints selected at observation_ts (camera
        # arrays are ignored by the snapshot writer's fixed key list). Date the
        # inferred action at that same sensor-exposure instant: the outer tick's
        # t0 may precede the frames by a full camera period, while send time also
        # includes inference latency.
        self.recorder.publish(
            obs,
            performed if performed is not None else action_dict,
            observation_ts,
        )
        return action_dict

    def run(self) -> None:
        from lerobot.teleoperators.utils import TeleopEvents
        from lerobot.utils.utils import log_say

        policy_period = 1.0 / float(self.fps)
        teleop_period = 1.0 / float(self.teleop_hz)
        last_action: dict[str, float] | None = None
        loop_times: list[float] = []
        last_rate_log = time.perf_counter()

        # Anchor the policy velocity envelope at the robot's measured pose so
        # the episode's first action can't jump either.
        if self.limiter is not None:
            self.limiter.seed(*self.robot.positions)

        try:
            while not self.shutdown_event.is_set():
                t0 = time.perf_counter()

                capture_error = self.recorder.poll_capture_error()
                if capture_error is not None:
                    self.capture_error = str(capture_error)
                    log_say(
                        f"Camera capture failed; ending and discarding this "
                        f"episode: {capture_error}"
                    )
                    return

                # --- episode end requested from the VR record button?
                events = self.teleop.get_teleop_events()
                if events[TeleopEvents.TERMINATE_EPISODE]:
                    self.vr_choice = "s"
                    return
                if events[TeleopEvents.RERECORD_EPISODE]:
                    self.vr_choice = "r"
                    return

                # --- intervention state machine (grip buttons)
                frozen_press = self.teleop.consume_freeze()
                engaged = self.teleop.teleop_engaged
                if engaged:
                    if self.state != _STATE_TELEOP:
                        self.state = _STATE_TELEOP
                        self.interventions += 1
                        # Resume capture in the recorder (idempotent if a
                        # direct takeover skipped the freeze); the returned
                        # row count is exactly the dataset timestamp the
                        # correction resumes at.
                        rows = self.recorder.resume_episode()
                        self.open_span_start = rows / float(self.fps)
                        log_say("Operator took over — recording the correction.")
                else:
                    if self.state == _STATE_TELEOP:
                        self.state = _STATE_POLICY
                        if self.open_span_start is not None:
                            self.intervention_spans.append(
                                (
                                    self.open_span_start,
                                    self.recorder.frame_count() / float(self.fps),
                                )
                            )
                            self.open_span_start = None
                        # Reset the backend so it re-plans from the corrected
                        # pose: its pre-freeze state (obs history, buffered
                        # chunks) targets the pre-intervention trajectory, and
                        # continuing from it would yank the arm away from
                        # where the operator left it.
                        self.policy.reset()
                        # Re-anchor the policy velocity envelope at the
                        # corrected pose so the first resumed action can't
                        # jump away from where the operator left the arms.
                        if self.limiter is not None:
                            self.limiter.seed(*self.robot.positions)
                        log_say("Intervention over — policy resumes.")
                    elif self.state == _STATE_POLICY and frozen_press:
                        self.state = _STATE_FROZEN
                        self.recorder.pause_episode()
                        log_say(
                            "Frozen — recording paused. Squeeze both grips "
                            "to take over."
                        )

                # The tick period follows the state selected above: fps for
                # the policy/frozen states, teleop_hz while engaged.
                period = teleop_period if self.state == _STATE_TELEOP else policy_period

                # Shutdown can arrive while observation, policy reset, or a
                # recorder gate call is blocking. Re-check immediately before
                # any motor command so teardown cannot release one late action.
                if self.shutdown_event.is_set():
                    return

                # --- command the robot from the selected source
                if self.state == _STATE_POLICY:
                    sent = self._policy_tick()
                    if sent is None:
                        time.sleep(period)
                        continue
                    last_action = sent
                elif self.state == _STATE_TELEOP:
                    joint_obs = self.robot.get_joint_observation()
                    action = self.teleop.get_action()
                    if self.shutdown_event.is_set():
                        return
                    performed = self.robot.send_action(action)
                    # intervention=True: the recorder tags the rows this
                    # snapshot pairs with as human-driven (the dataset's
                    # per-frame ``intervention`` feature).
                    self.recorder.publish(
                        joint_obs,
                        performed if performed is not None else action,
                        t0,
                        intervention=True,
                    )
                    last_action = action
                else:  # FROZEN — hold pose, keep the command cadence alive.
                    if last_action is not None:
                        if self.shutdown_event.is_set():
                            return
                        self.robot.send_action(last_action)
                        # Keep the recorder's snapshot current with the live
                        # command: capture is gated in the recorder, but a row
                        # racing the takeover resume then pairs its frames
                        # with the action actually commanding the robot (the
                        # held action) instead of a stale pre-freeze snapshot.
                        self.recorder.publish(
                            self.robot.get_joint_observation(), last_action, t0
                        )

                # --- once-a-second rate readout (parity with collect-data)
                loop_times.append(t0)
                if t0 - last_rate_log >= 1.0 and len(loop_times) > 1:
                    span = loop_times[-1] - loop_times[0]
                    hz = (len(loop_times) - 1) / span if span > 0 else 0.0
                    _logger.info(
                        "loop: %.1f Hz  state: %s  vr: %.1f Hz  ik: %.1f Hz",
                        hz,
                        self.state,
                        self.teleop.vr_hz(),
                        self.teleop.ik_hz(),
                    )
                    loop_times.clear()
                    last_rate_log = t0

                elapsed = time.perf_counter() - t0
                if elapsed < period:
                    if self.shutdown_event.wait(timeout=period - elapsed):
                        return
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "DAgger control loop hit an unhandled exception (%r); "
                "signalling shutdown so the episode tears down.",
                exc,
            )
            self.fatal_error = exc
            self.shutdown_event.set()


def _stop_dagger_control_thread(
    control_thread: _DaggerControlLoop,
    timeout_s: float = 5.0,
) -> bool:
    """Signal and join the sole episode command producer.

    The supervisor calls this both on the normal episode boundary and from its
    outermost cleanup. That second call closes the Ctrl+C window between
    ``Thread.start()`` and the normal shutdown block: recorder/relay/robot
    teardown must never race a controller that was never told to stop.

    Returns:
        True when the thread is stopped. False means it is wedged in a backend
        call; the caller must disconnect the robot before releasing resources.
    """
    control_thread.shutdown_event.set()
    if control_thread.is_alive():
        control_thread.join(timeout=timeout_s)
    return not control_thread.is_alive()


# ----------------------------------------------------------------------
# Between-episode gate
# ----------------------------------------------------------------------


def _idle_teleop_until_record(
    teleop: "DaggerVRTeleop",
    robot: "AxolRobot",
    return_to_rest: Callable[[], bool],
    teleop_hz: int,
    control: "_StdinPolicyControl | _QueuePolicyControl",
    stop_event: threading.Event,
) -> tuple[bool, bool]:
    """Drive between-episode teleop until the operator starts the next episode.

    Mirrors collect-data's pre-record phase: the grips engage/disengage
    teleop (same both-grips / one-grip toggle as an intervention, including
    the robot-pose sync; grippers adopt the triggers) so the operator can
    reset the scene with the arms, and the VR reset button homes the arms
    (``return_to_rest`` — the supervisor's guarded ``IKResetController``
    closure, which plans from the robot's measured pose and bails into a
    limp gravity-comp hold on contact; another reset press retries).
    Nothing is recorded — no episode exists yet. While engaged, this loop
    runs the teleop tick at ``teleop_hz`` (matching the smoothing filters);
    while idle it just polls the VR events and ``control``'s gate, so the web
    panel's Start button opens an episode like the VR record button does.

    Returns ``(teleop_used, started)``. ``teleop_used`` says the arms may be
    away from rest (teleop was engaged and the operator didn't home
    afterwards) — the caller re-homes before starting the policy in that case
    (the policy expects to start from the rest pose). ``started`` is False when
    the operator quit instead: the panel's Stop or a quit at the gate. A
    KeyboardInterrupt propagates to the supervisor's handler (quit).
    """
    from lerobot.utils.utils import log_say

    period = 1.0 / float(teleop_hz)
    teleop_used = False
    while not stop_event.is_set():
        t0 = time.perf_counter()
        events = teleop.get_teleop_events()
        if events.get("start_recording"):
            return teleop_used, True
        if (decision := control.poll_gate()) is not None:
            return teleop_used, decision == "go"
        # A single-grip press means "freeze the policy" mid-episode; while
        # idle there is nothing to freeze, so just drop the latch.
        teleop.consume_freeze()
        if teleop.consume_idle_reset():
            # VR reset button: home the arms. Disarm the grips for the move
            # so an engage can't fight the reset trajectory, and drop any
            # events fired while it played. An aborted move (stop, or a
            # contact hold the operator never resolved) leaves teleop_used
            # as-is, so the pre-episode re-home still covers the arms.
            teleop.set_intervention_allowed(False)
            teleop.force_disengage()
            log_say("Returning to rest pose.")
            if return_to_rest():
                teleop_used = False  # the arms are at rest again
            teleop.set_intervention_allowed(True)
            teleop.get_teleop_events()
            continue
        if teleop.teleop_engaged:
            teleop_used = True
            action = teleop.get_action()
            robot.send_action(action)
        elapsed = time.perf_counter() - t0
        if elapsed < period:
            time.sleep(period - elapsed)
    return teleop_used, False


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def _run(
    cfg: DaggerConfig,
    stop_event: "threading.Event | None" = None,
    control: "_StdinPolicyControl | _QueuePolicyControl | None" = None,
) -> None:
    import os
    import shutil
    import socket

    from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
    from lerobot.utils.feature_utils import hw_to_dataset_features
    from lerobot.utils.utils import log_say
    from lerobot.utils.visualization_utils import init_rerun

    from ..lerobot.robot.robot_axol import AxolRobot
    from ..lerobot.teleop.teleop_vr_dagger import DaggerVRTeleop
    from ..utils import affinity
    from ..vr.models import VRState

    # Defaults keep the CLI path unchanged: a stop event nothing ever sets, and
    # episode decisions from the VR buttons + stdin.
    if stop_event is None:
        stop_event = threading.Event()
    if control is None:
        control = _StdinPolicyControl()

    task = cfg.task
    subtasks = cfg.subtasks or []
    episode_time_s = cfg.episode_time_s
    fps = cfg.fps
    teleop_hz = cfg.teleop_hz
    vcodec = cfg.vcodec
    repo_id = cfg.repo_id
    root = cfg.root
    rerun_ip = cfg.rerun_ip
    rerun_port = cfg.rerun_port

    # Guarded return-to-rest knobs, read from the shared teleop config (the
    # same fields collect-data / `axol teleop` use — see VRTeleopConfig).
    reset_torque_threshold = 4.0
    reset_gravity_comp_kd = 0.25

    # The teleop smoothing filters advance once per get_action() call with a
    # step of max_vel/frequency, and get_action() only runs in the TELEOP
    # state, which ticks at teleop_hz — so the configured frequency must
    # equal teleop_hz or engaged motion runs at the wrong speed. (The policy
    # and frozen states tick at fps and never touch the filters.)
    if isinstance(cfg.teleop_config, AxolVRTeleopConfig):
        vr_cfg = cfg.teleop_config.vr_teleop_config
        if vr_cfg.frequency != float(teleop_hz):
            _logger.info(
                "Pinning teleop frequency to the teleop control rate (%d Hz, was %s).",
                teleop_hz,
                vr_cfg.frequency,
            )
            vr_cfg.frequency = float(teleop_hz)
        reset_torque_threshold = vr_cfg.reset_torque_threshold
        reset_gravity_comp_kd = vr_cfg.reset_gravity_comp_kd

    # Finalize the camera set before the relay/robot open the cameras: prune
    # unassigned slots and flag physically-stereo ZED X units. Assign the
    # cameras the policy was trained on; the policy backend fail-fasts on a
    # visual-feature mismatch at connect time.
    if isinstance(cfg.robot_config, AxolRobotConfig):
        from ..zed import stereo_serials

        cfg.robot_config.prepare_capture_cameras(stereo_serials(), minimum=1)
        if not cfg.robot_config.observation_cameras():
            raise ValueError(
                "collect-dagger has no camera with recording enabled — every "
                "assigned camera is set to stream-only (or recording is turned "
                "off). The policy needs the cameras it was trained on; enable "
                "recording for them in the Cameras dialog."
            )

    # Resolve the dataset path and validate it up front (fail fast before we
    # power the robot); defer create/resume until after robot.connect() so
    # observation features pick up the cameras' auto-detected dimensions.
    dataset_root = Path(root) if root else HF_LEROBOT_HOME / repo_id
    meta = dataset_root / "meta"
    has_info = (meta / "info.json").exists()
    is_complete = (
        has_info and (meta / "tasks.parquet").exists() and (meta / "episodes").is_dir()
    )
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
        # A crashed session can lose buffered episodes while info.json's count
        # survives; resuming would number past the gap (see the check's doc).
        check_resume_consistency(dataset_root)

    # A resumed dataset's image resolution is fixed by its metadata; pin the
    # relay's dataset branch to it (mirrors collect-data).
    dataset_resolution = cfg.dataset_resolution
    if is_complete:
        existing = _existing_dataset_resolution(dataset_root)
        if existing is None:
            raise ValueError(
                f"Cannot resume the dataset at {dataset_root}: its recorded image "
                "resolution couldn't be read from meta/info.json or doesn't map to a "
                "ZED resolution the relay produces (SVGA/HD1080/HD1200). Start a "
                "fresh dataset, or resume one recorded by this tool."
            )
        if existing != cfg.dataset_resolution:
            _logger.warning(
                "resuming a dataset recorded at %s; recording at %s to match it "
                "(start a new dataset to record at %s).",
                existing,
                existing,
                cfg.dataset_resolution,
            )
        dataset_resolution = existing

    # Pin the control process to its dedicated cores before any threads are
    # created, so the relay / recorder / NVENC work — all pinned to the other
    # cores by their own processes — can't preempt the control loops. Restored
    # in the finally. No-op where affinity isn't available. Mirrors
    # collect-data (and is only safe because nothing camera- or encode-heavy
    # runs in this process).
    try:
        _orig_affinity = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        _orig_affinity = None
    affinity.pin_realtime()

    hostname = socket.gethostname()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as _s:
        _s.connect(("8.8.8.8", 80))
        local_ip = _s.getsockname()[0]
    print("Connect the VR app (https://axol.almond.bot) to this machine:")
    print(f"  Hostname : {hostname}.local")
    print(f"  IP       : {local_ip}")

    if rerun_ip:
        init_rerun(session_name="axol_dagger", ip=rerun_ip, port=rerun_port)

    # Start the IK reset worker first so its JAX JIT overlaps with the policy
    # load, robot connect, and the teleop's own IK worker JIT. It owns the
    # collision-aware homing between episodes (the teleop's reset path is
    # disabled in DAgger mode — see almond_axol.teleop.dagger).
    reset_controller = IKResetController()
    reset_controller.start()
    log_say("Started IK reset worker (collision-aware return-to-rest).")

    # The teleop's action keys must match the robot's: propagate the SKU's
    # gripper capability so the gripperless SKU commands/records no gripper
    # channels (mirrors collect-data).
    if isinstance(cfg.robot_config, AxolRobotConfig) and isinstance(
        cfg.teleop_config, AxolVRTeleopConfig
    ):
        cfg.teleop_config.has_gripper = cfg.robot_config.axol_config.has_gripper

    # The out-of-process video relay owns the cameras and streams the headset
    # view; its raw branch is forced onto the pyshm transport so the frames
    # are readable HERE (policy observations) as well as by the recorder
    # subprocess (dataset). Required — there is no in-process fallback (see
    # the module docstring). A failure anywhere in this setup stage tears the
    # relay and the reset worker down instead of leaking them (mirrors
    # collect-data's setup-failure cleanup) — the reset worker is already
    # running, and a started relay holds the cameras.
    relay = None
    try:
        robot = AxolRobot(cfg.robot_config)
        teleop = DaggerVRTeleop(cfg.teleop_config)
        policy = _LocalPolicy(cfg.policy_path, cfg.policy_type, cfg.device, task)

        relay = _start_video_relay(cfg, dataset_resolution, raw_transport="pyshm")
        expected = set(cfg.robot_config.observation_cameras().keys())
        if relay is None or not expected <= set(relay.raw_cameras):
            raise RuntimeError(
                "collect-dagger requires the gst video relay with readable raw "
                f"frames for {sorted(expected)} (got "
                f"{sorted(relay.raw_cameras) if relay else 'no relay'}). Install "
                "the GStreamer stack (`axol gst.install` + `axol gst.build-zed`) "
                "and check the camera serials."
            )
        robot.set_external_cameras({k: relay.raw_cameras[k] for k in expected})
    except BaseException:
        if relay is not None:
            relay.shutdown()
        try:
            reset_controller.stop()
        except Exception:  # noqa: BLE001
            pass
        raise

    episodes_recorded = 0
    episode_idx = 0
    recorder: DatasetRecorderProcess | None = None
    control_thread: _DaggerControlLoop | None = None

    def _return_to_rest_guarded(wait_retry: Callable[[], bool]) -> bool:
        """Guarded ``IKResetController`` home; ``False`` when aborted.

        Plays with the torque watchdog live; on contact the arms drop into a
        limp gravity-comp hold until ``wait_retry`` answers (``True`` =
        replan from wherever they were hand-guided to) or the run stops.
        """
        return reset_controller.return_to_rest(
            robot,
            torque_threshold=reset_torque_threshold,
            gravity_comp_kd=reset_gravity_comp_kd,
            stopped=stop_event.is_set,
            wait_retry=wait_retry,
        )

    def _measured_joint_hold_action() -> dict[str, float]:
        """Snapshot measured joints as a direct, IK-free impedance target."""
        left, right = robot.positions
        # These package-private lists are AxolRobot's canonical direct-action
        # keys (and already omit the gripper on a gripperless SKU). Using them
        # intentionally bypasses Cartesian policy action space and its IK/
        # shaper: a boundary heartbeat must hold, never continue a trajectory.
        return {
            **{
                key: float(left[index])
                for index, key in enumerate(robot._left_pos_keys)
            },
            **{
                key: float(right[index])
                for index, key in enumerate(robot._right_pos_keys)
            },
        }

    # The normal policy cadence is already much faster than the Rust target
    # watchdog. Keep at least 20 Hz of headroom at lifecycle boundaries even
    # if someone configures an unusually low dataset/policy frame rate.
    boundary_period = min(1.0 / float(fps), 1.0 / float(teleop_hz), 0.05)

    def _gate_retry() -> bool:
        """Contact-hold retry via the continue gate (terminal Enter / panel)."""
        return control.await_continue(
            "Contact during return to rest. Free the arms, then continue to retry.",
            label="Return to rest",
            phase=_GATE_CONTACT,
        )

    def _idle_gate_message() -> str:
        """The idle phase's gate instruction, for opening and restoring it."""
        return (
            f"Episode {episode_idx + 1}: reset the scene (grips teleop the "
            "arms, reset button homes them), then press record in VR to "
            "start (Ctrl+C quits)."
        )

    def _idle_reset_retry() -> bool:
        """Contact-hold retry via the VR reset button (idle-phase homes).

        The idle reset stays armed through the move, so the operator who
        requested the home ends its contact hold the same way — pressing
        reset again. The panel's gate works too; the terminal control's
        ``poll_gate`` is inert by design.

        The hold borrows the idle gate rather than opening its own, so the
        panel names the contact instead of still reading "reset the scene /
        Start episode" while the arms hang limp.
        """
        log_say(
            "Contact during return to rest. Free the arms, then press the "
            "VR reset button (or continue in the panel) to retry."
        )
        control.note_gate(
            "Contact during return to rest — the arms are limp and free to "
            "move. Clear them, then press the VR reset button, or return to "
            "rest here.",
            "Return to rest",
            phase=_GATE_CONTACT,
        )
        try:
            while not stop_event.is_set():
                if teleop.consume_idle_reset():
                    return True
                if (decision := control.poll_gate()) is not None:
                    return decision == "go"
                time.sleep(0.1)
            return False
        finally:
            control.note_gate(_idle_gate_message())

    try:
        log_say("Connecting robot...")
        robot.connect()

        # Load the policy before the teleop connect so its checkpoint
        # download / CUDA load doesn't contend with the IK worker's JIT.
        policy.connect(robot)

        # Connect the VR teleop stack: the position source lets takeovers
        # sync the IK worker to the robot's measured pose, and the current
        # positions seed the teleop filters (mirrors collect-data).
        teleop.set_position_source(lambda: robot.positions)
        pos_l, pos_r = robot.positions
        log_say("Connecting VR teleop (IK worker JIT may take ~20s)...")
        teleop.connect(q_start_left=pos_l, q_start_right=pos_r)

        # Stream the cameras to the headset via the relay's out-of-process
        # WebRTC manager.
        teleop.set_video_manager(relay)

        # The recorder subprocess owns the dataset end to end: it reads
        # the relay's shared-memory frames on its own 60 fps clock, pairs
        # them with the snapshots the control loop publishes, and encodes
        # on NVENC from its own cores — nothing dataset-related runs in
        # this process. Mirrors collect-data.
        if is_complete:
            log_say(f"Resuming existing dataset at {dataset_root}.")
            # An existing dataset's feature set is fixed; one created before
            # the per-frame intervention flag existed can't gain it on resume
            # (the recorder only tags rows when the dataset declares the
            # feature) — interventions in new episodes would go untagged.
            import json

            try:
                existing_features = json.loads((meta / "info.json").read_text()).get(
                    "features", {}
                )
            except (OSError, ValueError):
                existing_features = {}
            if "intervention" not in existing_features:
                _logger.warning(
                    "resuming a dataset without the per-frame 'intervention' "
                    "feature; new episodes won't carry LeRobot DAgger "
                    "intervention tags (start a fresh dataset to record them)."
                )
        action_features = hw_to_dataset_features(robot.action_features, ACTION)
        obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
        recorder = DatasetRecorderProcess(
            raw_cond=relay.raw_cond,
            raw_meta=relay.raw_meta,
            obs_keys=list(robot.get_joint_observation().keys()),
            action_keys=list(robot.action_features.keys()),
            config={
                "repo_id": repo_id,
                "root": root,
                "dataset_root": str(dataset_root),
                "is_complete": is_complete,
                "features": {
                    **action_features,
                    **obs_features,
                    "intervention": dict(INTERVENTION_FEATURE),
                },
                "robot_type": robot.name,
                "fps": fps,
                "vcodec": vcodec,
                "rerun_ip": rerun_ip,
                "rerun_port": rerun_port,
                "push_to_hub": cfg.push_to_hub,
                "log_level": cfg.log_level,
            },
        )
        episode_idx = recorder.episode_count()

        log_say("Returning to rest pose.")
        if not _return_to_rest_guarded(_gate_retry):
            return

        # Keep the relay's raw branch closed outside episodes: the per-frame
        # copy work is the bulk of the relay's raw-branch CPU and nothing
        # consumes frames while idle (the policy only runs inside episodes).
        # Mirrors collect-data.
        relay.set_raw_enabled(False)

        while not stop_event.is_set():
            teleop.send_feedback_state(VRState.DATA_COLLECTION)
            # Surface the (1-based) dataset episode number in the headset
            # HUD, matching `collect-data` — the operator is in VR, so
            # this is their episode counter. The panel reads the same
            # number off the control (its gate message names it too).
            teleop.send_feedback_episode(episode_idx + 1)
            control.note_episode(episode_idx + 1)
            control.begin_gate(_idle_gate_message())
            # Drop events latched during the reset/save phase so a stale
            # record press can't auto-start the episode, then arm the
            # grips + reset button for between-episode scene-reset teleop.
            teleop.get_teleop_events()
            teleop.set_intervention_allowed(True)
            teleop.set_idle_reset_armed(True)
            idle_teleop_used, started = _idle_teleop_until_record(
                teleop,
                robot,
                lambda: _return_to_rest_guarded(_idle_reset_retry),
                teleop_hz,
                control,
                stop_event,
            )
            teleop.set_idle_reset_armed(False)
            teleop.set_intervention_allowed(False)
            teleop.force_disengage()
            if not started:
                break
            if idle_teleop_used:
                # The operator moved the arms during the scene reset; the
                # policy expects to start from the rest pose.
                log_say("Returning to rest pose before the policy starts.")
                if not _return_to_rest_guarded(_gate_retry):
                    break

            # Fresh episode: drop the policy's episode-scoped state (obs
            # history / hidden state from the previous episode).
            policy.reset()
            policy.set_instruction(task)
            # Arm the recorder before opening the relay branch. Today DAgger
            # forces raw pyshm, but this ordering also preserves row-zero IDR
            # semantics if it later adopts the encoded transport. Both calls
            # are bounded IPC transactions, but together they can exceed the
            # Rust target watchdog; hold the just-measured post-rest pose while
            # they run off-thread. Policy inference deliberately has not
            # started yet, so camera setup cannot advance the trajectory.
            start_hold_action = _measured_joint_hold_action()

            def _start_capture() -> None:
                recorder.start_episode(task)
                relay.set_raw_enabled(True)

            def _hold_start_pose() -> None:
                robot.send_action(start_hold_action)

            run_blocking_with_sync_control_ticks(
                _start_capture,
                _hold_start_pose,
                boundary_period,
                drain_tick=_hold_start_pose,
            )

            if stop_event.is_set():
                # Stop may arrive from the panel while the bounded start IPC
                # drains. Never launch a policy controller after that request;
                # close the just-opened capture under the same measured hold.
                def _finish_cancelled_start() -> int:
                    try:
                        try:
                            return recorder.finish_episode()
                        except RecorderCaptureError:
                            # The session is already stopping and the rejected
                            # buffer was cleared by finish_episode.
                            return 0
                    finally:
                        relay.set_raw_enabled(False)

                run_blocking_with_sync_control_ticks(
                    _finish_cancelled_start,
                    _hold_start_pose,
                    boundary_period,
                    drain_tick=_hold_start_pose,
                )
                recorder.cancel_episode()
                break

            control_thread = _DaggerControlLoop(
                robot=robot,
                policy=policy,
                teleop=teleop,
                recorder=recorder,
                fps=fps,
                teleop_hz=teleop_hz,
                limiter=(
                    PolicyActionLimiter(cfg.policy_max_vel, cfg.policy_max_accel, fps)
                    if cfg.policy_max_vel > 0
                    and not getattr(robot.config, "observe_cartesian", False)
                    else None
                ),
            )

            def _switch_subtask(idx: int) -> None:
                """Switch the live policy instruction to subtask ``idx`` (1-based)."""
                text = subtasks[idx - 1]
                policy.set_instruction(text)
                log_say(f"Subtask {idx}: {text}")

            print(
                "  Grips: one=freeze (pause recording), both=take over, "
                "one again=policy resumes.",
                flush=True,
            )
            print(
                "  End the episode with the VR record button (save) or "
                "reset+record (discard + rerecord); s/r/q on stdin mirror "
                f"it. Time cap {episode_time_s}s saves the episode.",
                flush=True,
            )
            if subtasks:
                print(
                    f"  Type a subtask number 1-{len(subtasks)} + Enter to "
                    "switch the policy's instruction mid-episode.",
                    flush=True,
                )

            teleop.set_intervention_allowed(True)
            control_thread.start()
            control.begin_episode(_switch_subtask, len(subtasks))

            deadline = time.perf_counter() + episode_time_s
            timed_out = False
            interrupted = False
            try:
                while True:
                    if control.poll_choice() is not None:
                        break
                    if control_thread.vr_choice is not None:
                        break
                    if stop_event.is_set():
                        # Stopped from the panel: unwind like a Ctrl+C,
                        # discarding the episode in flight.
                        interrupted = True
                        break
                    if time.perf_counter() >= deadline:
                        timed_out = True
                        break
                    if control_thread.capture_error is not None:
                        break
                    if control_thread.fatal_error is not None:
                        log_say(
                            f"Fatal error in DAgger control loop: "
                            f"{control_thread.fatal_error!r}. Aborting run "
                            "without saving the current episode."
                        )
                        break
                    time.sleep(0.1)
            except KeyboardInterrupt:
                interrupted = True

            control.end_episode()
            control_stopped = _stop_dagger_control_thread(control_thread)
            teleop.set_intervention_allowed(False)
            teleop.force_disengage()
            if not control_stopped:
                # A policy call wedged past shutdown can otherwise wake later
                # and issue one more command while the supervisor homes/saves.
                # Disconnect first, then discard capture; never race a second
                # controller against this still-live thread.
                try:
                    robot.disconnect()
                except Exception:  # noqa: BLE001 - preserve the safety failure
                    _logger.exception("robot disconnect failed after control timeout")
                try:
                    recorder.finish_episode()
                    recorder.cancel_episode()
                except Exception:  # noqa: BLE001 - outer cleanup gets another try
                    _logger.exception("episode discard failed after control timeout")
                try:
                    relay.set_raw_enabled(False)
                except Exception:  # noqa: BLE001 - relay shutdown follows
                    _logger.exception("relay gate failed after control timeout")
                raise RuntimeError(
                    "DAgger control thread did not stop within 5s; robot was "
                    "disconnected and the episode discarded"
                )
            # Freeze and join capture at an exact row count before closing the
            # relay. A mere pause acknowledgement can leave one raw camera read
            # in flight; closing its valve then turns a normal episode end into
            # a capture timeout. The only controller has been joined, so there
            # is no race: snapshot and hold measured joints while the bounded
            # recorder + relay IPC runs on a worker thread. A direct joint hold
            # also bypasses Cartesian IK/shaping, so a Cartesian policy's last
            # desired pose cannot keep advancing during shutdown.
            finish_hold_action = _measured_joint_hold_action()

            def _finish_capture() -> tuple[int, str | None]:
                try:
                    try:
                        return recorder.finish_episode(), None
                    except RecorderCaptureError as exc:
                        return 0, str(exc)
                finally:
                    # A gate-close failure is session-fatal and deliberately
                    # overrides an episode-local capture rejection: continuing
                    # with the relay branch open is not a valid recovery.
                    relay.set_raw_enabled(False)

            def _hold_finish_pose() -> None:
                robot.send_action(finish_hold_action)

            final_rows, finish_capture_error = run_blocking_with_sync_control_ticks(
                _finish_capture,
                _hold_finish_pose,
                boundary_period,
                drain_tick=_hold_finish_pose,
            )
            capture_error = control_thread.capture_error or finish_capture_error
            # Close a span still open when the loop exited (the episode ended
            # mid-intervention) at the final row count — exact, since capture
            # is paused — so intervention_spans is complete for any consumer.
            if control_thread.open_span_start is not None:
                control_thread.intervention_spans.append(
                    (control_thread.open_span_start, final_rows / float(fps))
                )
                control_thread.open_span_start = None

            if interrupted:
                recorder.cancel_episode()
                break
            if control_thread.fatal_error is not None:
                recorder.cancel_episode()
                break

            if capture_error is not None:
                # finish_episode has joined capture and cleared its rejected
                # buffer. Home exactly like a normal episode boundary before
                # retrying the unchanged dataset episode index.
                recorder.cancel_episode()
                teleop.send_feedback_state(VRState.SAVING)
                log_say(
                    f"Episode discarded because camera capture failed: {capture_error}"
                )
                log_say("Returning to rest pose.")
                if not _return_to_rest_guarded(_gate_retry):
                    break
                continue

            choice = control.poll_choice() or control_thread.vr_choice
            if timed_out and choice is None:
                log_say(
                    f"Episode time cap ({episode_time_s}s) reached; saving the episode."
                )
                choice = "s"

            if choice == "q":
                recorder.cancel_episode()
                break

            teleop.send_feedback_state(VRState.SAVING)
            log_say("Returning to rest pose.")
            # An aborted home (stop / declined retry) must not discard a
            # fully-recorded episode: fall through to the save/discard
            # decision either way; the session loop then winds down on the
            # stop flag.
            _return_to_rest_guarded(_gate_retry)

            if choice == "r":
                log_say("Re-recording episode.")
                recorder.cancel_episode()
                continue

            if final_rows == 0:
                # Nothing reached the dataset (e.g. every observation
                # failed, or the episode was ended instantly) —
                # save_episode would raise on the empty buffer.
                log_say("No frames were captured this episode; discarding.")
                recorder.cancel_episode()
                continue
            log_say("Saving episode…")
            try:
                recorder.save_episode()
            except RecorderDatasetSaveError:
                # Writer indices/parquet rows may already have changed. A
                # retry in this process could compound the damage.
                raise
            except RecorderCaptureError as exc:
                # An encoder/capture integrity rejection is pre-commit and
                # already cleared its episode buffer. Other recorder RuntimeErrors
                # are IPC/lifecycle failures and must stop the session.
                log_say(f"Episode NOT saved: {exc}")
                continue
            episode_idx += 1
            episodes_recorded += 1
            control.note_saved()
            log_say(
                f"Saved episode {episodes_recorded} "
                f"({control_thread.interventions} intervention(s))."
            )

        if control_thread is not None and control_thread.fatal_error is not None:
            raise control_thread.fatal_error

    except KeyboardInterrupt:
        pass
    finally:
        import signal

        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except (ValueError, OSError):
            pass

        log_say("Stopping.")
        # Ctrl+C can land immediately after control_thread.start(), before the
        # normal episode shutdown block. Always signal and drain that sole
        # command producer before policy/robot/recorder/relay teardown.
        control_stopped = control_thread is None or _stop_dagger_control_thread(
            control_thread
        )
        if control_stopped:
            try:
                policy.close()
            except Exception:  # noqa: BLE001
                pass
        else:
            _logger.error(
                "skipping policy.close because the wedged control thread is "
                "still executing inside the policy backend"
            )
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            pass
        try:
            teleop.disconnect()
        except Exception:  # noqa: BLE001
            pass
        try:
            reset_controller.stop()
        except Exception:  # noqa: BLE001
            pass
        # Recorder owns the dataset: finalize (and empty-dataset cleanup)
        # happen in recorder.close(). Shut the relay down after it so the
        # recorder's shm readers never outlive their blocks.
        if recorder is not None:
            try:
                recorder.close()
            except Exception:  # noqa: BLE001
                _logger.exception("recorder close failed")
        # Detach our own shared-memory readers (the robot's external cameras)
        # before their writer blocks go away with the relay, so the resource
        # tracker doesn't report them as leaked at exit.
        for cam in relay.raw_cameras.values():
            try:
                cam.close()
            except Exception:  # noqa: BLE001
                pass
        relay.shutdown()

        # Restore the process's original CPU affinity (the process may run
        # other operations after this one).
        if _orig_affinity is not None:
            try:
                os.sched_setaffinity(0, _orig_affinity)
            except OSError:
                pass

        try:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        except (ValueError, OSError):
            pass
