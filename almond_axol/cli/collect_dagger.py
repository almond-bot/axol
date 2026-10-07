"""
axol collect-dagger

DAgger data collection: run a trained policy on the Axol robot while the
operator watches in VR and can intervene to guide it, and record the whole
thing (policy segments + human corrections) as training data.

``policy_type=custom`` selects the custom policy interface (v2) with nonblocking
inference, optional continuous shadow inference during interventions, and
held-grip controls. Either grip takes over that arm; the other holds. Both
released requests a fresh policy plan, holding the final operator command
until it arrives and blending after IK. No model-specific logic lives here.
The following freeze/toggle description applies to the local-policy mode.

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
the ZED cameras and streams the headset view, and beside its relay-encoded
dataset branch (which the ``DatasetRecorderProcess`` subprocess muxes, as in
``collect-data``) it publishes the same exposures into a shared-memory ring
readable by *this* process — the policy builds its observations from it
(``raw_transport: "gstshm+pyshm"``; without gst's shm plugin both fall back
to one pyshm ring, and the recorder re-encodes). The frozen gap uses the
recorder's ``pause_episode``/``resume_episode`` gate (the encoded loop
discards the gap's AUs and re-aligns row zero on resume; the raw loop
re-anchors its clock — either way episodes play straight through). Nothing
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

from ..constants import PARK_TIMEOUT_S
from ..lerobot.camera.configuration_zed import ZED_RESOLUTION_DIMS, ZedCameraConfig
from ..lerobot.robot.config_axol import AxolRobotConfig
from ..lerobot.rollout import (
    IKResetController,
    PolicyActionLimiter,
    arms_reporting,
)
from ..lerobot.teleop.config_vr import AxolVRTeleopConfig
from ..policy.plan_scheduler import PlanRuntimeConfig
from ..recording import (
    DatasetRecorderProcess,
    RecorderCaptureError,
    RecorderDatasetSaveError,
    default_vcodec,
    restore_dataset_ownership,
)
from ..robot.base import HardwareCleanupError, mark_hardware_cleanup_uncertain
from ..utils import affinity
from ..utils.control_loop import run_blocking_with_sync_control_ticks
from ..utils.logquiet import quiet_noisy_loggers
from ..utils.network import local_ip
from .collect_data import (
    _existing_dataset_resolution,
    _start_video_relay,
    check_resume_consistency,
)
from .config import DatasetResolution, LogLevel, RunPolicyType, parse
from .dagger_terminal import DaggerStdinControl
from .run_policy import (
    _GATE_CONTACT,
    _check_training_fps,
    _QueuePolicyControl,
    _StdinPolicyControl,
)

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


def _require_dagger_resume_schema(dataset_root: Path) -> None:
    """Fail closed when an existing dataset cannot store DAgger labels."""
    import json

    info_path = dataset_root / "meta" / "info.json"
    try:
        features = json.loads(info_path.read_text()).get("features", {})
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Cannot resume the DAgger dataset at {dataset_root}: "
            "meta/info.json could not be read. Repair the dataset or start a "
            "new one with a different repo_id."
        ) from exc

    intervention = features.get("intervention") if isinstance(features, dict) else None
    shape = intervention.get("shape") if isinstance(intervention, dict) else None
    valid = (
        isinstance(intervention, dict)
        and intervention.get("dtype") == "bool"
        and isinstance(shape, (list, tuple))
        and tuple(shape) == (1,)
    )
    if not valid:
        raise ValueError(
            f"Cannot resume the DAgger dataset at {dataset_root}: it does not "
            "declare the required per-frame bool[1] 'intervention' feature. "
            "Continuing would silently leave new human-correction frames "
            "unlabeled. Start a new DAgger dataset with a different repo_id, "
            "or migrate every existing frame and meta/info.json to add that "
            "feature before resuming."
        )


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

    Local policies use ``--policy_path`` / ``--policy_type`` / ``--device``
    for synchronous inference. ``policy_type=custom`` always uses the
    continuation-aware custom policy interface with a remote endpoint.
    The recording side mirrors
    ``collect-data`` (``--dataset_resolution`` for the relay's dataset branch,
    ``teleop_config`` for the VR server / IK / smoothing parameters).
    """

    policy_type: RunPolicyType
    task: str
    repo_id: str
    policy_path: str = ""
    server_host: str = "127.0.0.1"
    server_port: int = 8765
    actions_per_chunk: int = 30
    plan_config: PlanRuntimeConfig = field(default_factory=PlanRuntimeConfig)
    shadow_inference: bool = True
    handover_duration_s: float = 8 / 30
    hold_to_intervene: bool = False
    start_from_current_pose: bool = False
    # Home once at session startup even when later episodes use the pose
    # established by the operator during scene reset.
    home_on_start: bool = False
    # Wire action space and training action space are independent. In joint
    # recording mode both sources record the resolved command actually sent.
    record_joint_actions: bool = False
    # Ordered per-step instructions; typing a number 1..N + Enter (or pushing
    # it from the web panel) switches the instruction sent to the policy
    # mid-episode without ending it. The dataset's task string stays --task.
    subtasks: list[str] | None = None
    # Safety cap per episode; hitting it saves the episode. DAgger episodes
    # include interventions, so the default is generous.
    episode_time_s: int = 600
    # Dataset rate and the policy state's control rate — must equal the fps
    # the policy was trained at (collect-data's default, 30; see its note on
    # why 30 rather than 60); checked against the checkpoint's metadata at
    # start like run-policy. Teleop ticks at ``teleop_hz`` regardless.
    fps: int = 30
    # Escape hatch for that check (see RunPolicyConfig.allow_fps_mismatch):
    # a detectable mismatch is otherwise a hard error, since the policy would
    # move at the wrong speed and the dataset be written at the wrong rate.
    allow_fps_mismatch: bool = False
    # Velocity/acceleration envelope over the policy's arm actions (rad/s,
    # rad/s²) — see PolicyActionLimiter. Transparent for normal trained
    # motion; only engages on discontinuities (policy outliers, re-plans from
    # stale observations). Set policy_max_vel to 0 to disable. Teleop actions
    # are already enveloped by the teleop smoothing stack.
    policy_max_vel: float = 6.2832
    policy_max_accel: float = 21.9911
    # Rate of the policy's camera ring (frames/s), independent of --fps: the
    # relay decimates its control-process ring branch to this ahead of the VIC
    # convert, and the policy loop (still ticking at --fps) re-serves the
    # newest frame set between ring frames. Inference reads an observation a
    # few times a second, so a capture-rate ring only spent a third VIC pass
    # per camera and a 60 Hz RGB copy per source on frames nobody used — on
    # the Orin, the throughput the dataset encode branch was short of
    # (concealed frames with the headset streaming). 0 = capture rate.
    policy_fps: int = 20
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
    cfg = parse(DaggerConfig, argv, settings_op="collect-dagger")
    # force=True: importing lerobot (at module load) installs a root handler
    # and leaves the root level at WARNING, which would otherwise make this a
    # no-op and silently drop every _logger.info() status line.
    logging.basicConfig(level=getattr(logging, cfg.log_level), force=True)
    quiet_noisy_loggers()

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
        """Load/prepare the policy against the robot's declared features.

        Called once before physical ``robot.connect()`` so an incompatible
        action schema fails while actuators and cameras are still untouched.
        Backends must not require live hardware from this method.
        """
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

        # Most LeRobot policies retain only an action width. That is not a
        # sufficient deployment contract: Axol's gripperless joint and
        # Cartesian layouts are both 14-D. Recover the checkpoint's
        # authoritative ordered names and require exact equality before this
        # backend can ever produce an action for the configured robot.
        from ..lerobot.action_schema import (
            require_exact_action_schema,
            resolve_policy_action_schema,
        )

        policy_schema = resolve_policy_action_schema(
            self._path,
            policy_config=policy.config,
            processors=(preprocessor, postprocessor),
        )
        require_exact_action_schema(
            policy_schema,
            robot.action_features,
            policy_label="DAgger policy",
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

    def _policy_tick(self, t0: float) -> dict[str, float] | None:
        """One policy inference tick; returns the sent action or ``None``.

        ``None`` means the tick was skipped (observation/camera hiccup, or
        the backend declined the observation) — skip-and-retry.
        """
        try:
            obs, _observation_ts = self.robot.get_observation_with_capture_timestamp()
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Observation failed (%s); skipping tick.", exc)
            return None
        if self.shutdown_event.is_set():
            return None

        action_dict = self.policy.act(obs)
        if action_dict is None or self.shutdown_event.is_set():
            return None
        # Policy inference may block in a backend/native runtime.  A stop that
        # arrived while it was running must prevent the returned action from
        # reaching hardware or the recorder.
        if self.shutdown_event.is_set():
            return None

        if self.limiter is not None:
            action_dict = self.limiter.apply(action_dict)

        if self.shutdown_event.is_set():
            return None
        performed = self.robot.send_action(action_dict)
        if self.shutdown_event.is_set():
            return None
        # Snapshot the live joint state at this tick, dated at the tick, like
        # the teleop and frozen states (and collect-data) do: the recorder
        # pairs every 60 Hz exposure with the nearest snapshot, and those must
        # arrive at the tick rate on a monotonic timeline. The policy's
        # observation is not that snapshot any more — its ring runs below
        # --fps and the frame set (with the joints selected at its exposure)
        # is re-served between ring frames, so its timestamp repeats.
        self.recorder.publish(
            self.robot.get_joint_observation(),
            performed if performed is not None else action_dict,
            t0,
        )
        return action_dict

    def run(self) -> None:
        from lerobot.teleoperators.utils import TeleopEvents

        policy_period = 1.0 / float(self.fps)
        teleop_period = 1.0 / float(self.teleop_hz)
        last_action: dict[str, float] | None = None
        last_dataset_action: dict[str, float] | None = None
        loop_times: list[float] = []
        last_rate_log = time.perf_counter()

        try:
            # This thread paces every command (`send_action` posts
            # motion_control onto the robot's FIFO event-loop thread and waits
            # for it), so it is the other half of the control path: same
            # realtime core, SCHED_FIFO, anything it spawns reset to CFS — see
            # affinity.enter_control_thread. Inside the fault boundary so a
            # denied real-time class reaches the supervisor through
            # fatal_error rather than killing this thread on its own.
            affinity.enter_control_thread()

            # Anchor the policy velocity envelope at the robot's measured pose
            # so the episode's first action can't jump either. Keep this inside
            # the fault boundary so startup failures reach the supervisor.
            if self.limiter is not None:
                self.limiter.seed(*self.robot.positions)

            while not self.shutdown_event.is_set():
                t0 = time.perf_counter()

                capture_error = self.recorder.poll_capture_error()
                if capture_error is not None:
                    self.capture_error = str(capture_error)
                    _logger.info(
                        f"Camera capture failed; ending and discarding this "
                        f"episode: {capture_error}"
                    )
                    return

                # --- episode end requested from the VR record button?
                events = self.teleop.get_teleop_events()
                if self.shutdown_event.is_set():
                    return
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
                        _logger.info("Operator took over — recording the correction.")
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
                        _logger.info("Intervention over — policy resumes.")
                    elif self.state == _STATE_POLICY and frozen_press:
                        self.state = _STATE_FROZEN
                        self.recorder.pause_episode()
                        _logger.info(
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
                if self.shutdown_event.is_set():
                    return
                if self.state == _STATE_POLICY:
                    sent = self._policy_tick(t0)
                    if sent is None:
                        time.sleep(period)
                        continue
                    last_action = sent
                    last_dataset_action = sent
                elif self.state == _STATE_TELEOP:
                    joint_obs = self.robot.get_joint_observation()
                    action = self.teleop.get_action()
                    if self.shutdown_event.is_set():
                        return
                    performed = self.robot.send_action(action)
                    if self.shutdown_event.is_set():
                        return
                    # Teleop always commands joint targets, even when the
                    # policy/dataset action space is Cartesian. Keep those
                    # exact joints on the hardware path, but convert the
                    # recorder snapshot (of the action actually performed) to
                    # the configured dataset space so a Cartesian DAgger
                    # intervention has the same schema as the surrounding
                    # policy rows.
                    dataset_action = self.robot.action_to_dataset(
                        performed if performed is not None else action
                    )
                    if self.shutdown_event.is_set():
                        return
                    # intervention=True: the recorder tags the rows this
                    # snapshot pairs with as human-driven (the dataset's
                    # per-frame ``intervention`` feature).
                    self.recorder.publish(
                        joint_obs,
                        dataset_action,
                        t0,
                        intervention=True,
                    )
                    last_action = action
                    last_dataset_action = dataset_action
                else:  # FROZEN — hold pose, keep the command cadence alive.
                    if last_action is not None and last_dataset_action is not None:
                        if self.shutdown_event.is_set():
                            return
                        self.robot.send_action(last_action)
                        if self.shutdown_event.is_set():
                            return
                        # Keep the recorder's snapshot current with the live
                        # command: capture is gated in the recorder, but a row
                        # racing the takeover resume then pairs its frames
                        # with the action actually commanding the robot (the
                        # held action) instead of a stale pre-freeze snapshot.
                        self.recorder.publish(
                            self.robot.get_joint_observation(), last_dataset_action, t0
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


# ----------------------------------------------------------------------
# Between-episode gate
# ----------------------------------------------------------------------


def _finish_idle_terminal_gate(
    control: _StdinPolicyControl | _QueuePolicyControl,
    robot: AxolRobot,
    teleop_hz: int,
    *,
    hold: bool = True,
) -> str | None:
    """Drain the remote terminal reader without losing the live motor target.

    A quit arriving with the VR Record event must win before another owner
    starts commanding the arms. Joining the reader proves its last decision
    has been published and restores the terminal before a blocking prompt.
    """
    if not isinstance(control, DaggerStdinControl):
        return None
    if not hold:
        # A contact abort leaves the arms in gravity compensation. Do not
        # turn the terminal join into an impedance command in that state.
        control.end_gate()
        return control.poll_gate()
    left, right = robot.positions
    action = {
        **{key: float(left[i]) for i, key in enumerate(robot._left_pos_keys)},
        **{key: float(right[i]) for i, key in enumerate(robot._right_pos_keys)},
    }
    run_blocking_with_sync_control_ticks(
        control.end_gate,
        lambda: robot.send_action(action),
        min(1.0 / float(teleop_hz), 0.05),
    )
    return control.poll_gate()


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

    period = 1.0 / float(teleop_hz)
    teleop_used = False
    while not stop_event.is_set():
        t0 = time.perf_counter()
        # Terminal q/EOF wins even if a VR Record press is already latched.
        terminal_gate = isinstance(control, DaggerStdinControl)
        if terminal_gate and (decision := control.poll_gate()) is not None:
            return teleop_used, False
        events = teleop.get_teleop_events()
        if events.get("start_recording"):
            return teleop_used, True
        if not terminal_gate and (decision := control.poll_gate()) is not None:
            return teleop_used, decision == "go"
        # A single-grip press means "freeze the policy" mid-episode; while
        # idle there is nothing to freeze, so just drop the latch.
        teleop.consume_freeze()
        if teleop.consume_idle_reset():
            # Stop the idle stdin reader before reset can open a contact
            # prompt. Keep its final decision: q racing the reset press must
            # stop here, without starting another robot motion.
            if _finish_idle_terminal_gate(control, robot, teleop_hz) is not None:
                return teleop_used, False
            # VR reset button: home the arms. Disarm the grips for the move
            # so an engage can't fight the reset trajectory, and drop any
            # events fired while it played. An aborted move (stop, or a
            # contact hold the operator never resolved) leaves teleop_used
            # as-is, so the pre-episode re-home still covers the arms.
            teleop.set_intervention_allowed(False)
            teleop.force_disengage()
            _logger.info("Returning to rest pose.")
            returned_to_rest = return_to_rest()
            if returned_to_rest:
                teleop_used = False  # the arms are at rest again
            elif isinstance(control, DaggerStdinControl):
                # Contact q/EOF is an abort, never a request to park again.
                return teleop_used, False
            teleop.set_intervention_allowed(True)
            teleop.get_teleop_events()
            if isinstance(control, DaggerStdinControl):
                control.begin_gate("Ready: VR Record starts; q parks and exits.")
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


def _stop_dagger_control_worker(
    worker: _DaggerControlLoop | None,
    *,
    timeout: float = 5.0,
) -> tuple[bool, BaseException | None]:
    """Request stop and return ``True`` only after worker exit is proved."""
    if worker is None:
        return True, None
    failures: list[tuple[str, BaseException]] = []
    try:
        worker.shutdown_event.set()
    except BaseException as error:
        failures.append(("stop signal", error))
    try:
        if getattr(worker, "ident", None) is not None or worker.is_alive():
            worker.join(timeout=timeout)
    except BaseException as error:
        failures.append(("join", error))
    try:
        alive = bool(worker.is_alive())
    except BaseException as error:
        failures.append(("liveness check", error))
        alive = True

    if alive:
        error = RuntimeError(
            f"DAgger control loop did not stop within {timeout:g}s; deferring "
            "recorder mutation and robot/teleop/relay teardown until a final "
            "exit proof"
        )
    elif failures:
        error = RuntimeError(
            "DAgger control-loop cleanup encountered an error after exit was proved"
        )
    else:
        return True, None
    for label, failure in failures:
        error.add_note(
            f"additional DAgger control {label} failure: "
            f"{type(failure).__name__}: {failure}"
        )
    return not alive, error


def _cleanup_dagger_resource(
    *,
    control_stopped: bool,
    label: str,
    cleanup: Callable[[], Any],
) -> BaseException | None:
    """Clean a control-owned resource only after the loop's exit proof."""
    if not control_stopped:
        _logger.error(
            "skipping %s because DAgger control-loop exit was not proved", label
        )
        return None
    try:
        cleanup()
    except BaseException as error:
        _logger.exception("%s cleanup failed", label)
        return error
    return None


def _finish_dagger_cleanup(
    *,
    session_error: BaseException | None,
    disconnect_failure: BaseException | None,
    teleop_failure: BaseException | None,
    reset_failure: BaseException | None,
    relay_failure: BaseException | None,
    additional_failures: tuple[tuple[str, BaseException], ...] = (),
) -> None:
    """Propagate teardown failures without replacing a session's primary error."""
    failures = [
        (label, failure)
        for label, failure in (
            ("robot disconnect", disconnect_failure),
            ("teleop disconnect", teleop_failure),
            ("IK reset worker", reset_failure),
            *additional_failures,
            ("video relay", relay_failure),
        )
        if failure is not None
    ]
    if session_error is not None:
        for label, failure in failures:
            session_error.add_note(
                f"additional {label} cleanup failure: "
                f"{type(failure).__name__}: {failure}"
            )
        uncertain = (
            disconnect_failure
            or teleop_failure
            or reset_failure
            or next(
                (
                    failure
                    for _label, failure in additional_failures
                    if isinstance(failure, HardwareCleanupError)
                ),
                None,
            )
        )
        if uncertain is not None:
            mark_hardware_cleanup_uncertain(session_error, uncertain)
        return

    def add_remaining_notes(error: BaseException, selected: BaseException) -> None:
        for label, failure in failures:
            if failure is selected:
                continue
            error.add_note(
                f"additional {label} cleanup failure: "
                f"{type(failure).__name__}: {failure}"
            )

    if disconnect_failure is not None:
        error = HardwareCleanupError(
            "robot disconnect failed; hardware ownership is uncertain"
        )
        add_remaining_notes(error, disconnect_failure)
        raise error from disconnect_failure
    if reset_failure is not None:
        error = (
            reset_failure
            if isinstance(reset_failure, HardwareCleanupError)
            else HardwareCleanupError(
                "IK reset worker did not stop; background ownership is uncertain"
            )
        )
        add_remaining_notes(error, reset_failure)
        if error is reset_failure:
            raise error
        raise error from reset_failure
    hardware_failure = next(
        (
            failure
            for _label, failure in additional_failures
            if isinstance(failure, HardwareCleanupError)
        ),
        None,
    )
    if hardware_failure is not None:
        add_remaining_notes(hardware_failure, hardware_failure)
        raise hardware_failure
    if teleop_failure is not None:
        add_remaining_notes(teleop_failure, teleop_failure)
        if isinstance(teleop_failure, HardwareCleanupError):
            raise teleop_failure
        raise RuntimeError(
            "teleop disconnect failed; background ownership is uncertain"
        ) from teleop_failure
    if additional_failures:
        _label, failure = additional_failures[0]
        add_remaining_notes(failure, failure)
        raise failure
    if relay_failure is not None:
        raise relay_failure


def _recording_view(robot, joint_actions: bool):
    """Describe training actions independently of the policy's wire layout."""
    if not joint_actions:
        return robot
    from types import SimpleNamespace

    return SimpleNamespace(
        observation_features=robot.observation_features,
        action_features={
            key: float for key in (*robot._left_pos_keys, *robot._right_pos_keys)
        },
    )


def _run(
    cfg: DaggerConfig,
    stop_event: "threading.Event | None" = None,
    control: "_StdinPolicyControl | _QueuePolicyControl | None" = None,
) -> None:
    from ..lerobot.robot.config_mantis import MantisRobotConfig

    if isinstance(cfg.robot_config, MantisRobotConfig):
        raise ValueError("collect-dagger does not support Mantis hardware")

    import os
    import socket

    from lerobot.utils.constants import HF_LEROBOT_HOME
    from lerobot.utils.visualization_utils import init_rerun

    from ..lerobot.robot.robot_axol import AxolRobot
    from ..lerobot.teleop.teleop_vr_dagger import DaggerVRTeleop
    from ..vr.models import VRState

    # Defaults keep the CLI path unchanged: a stop event nothing ever sets, and
    # episode decisions from the VR buttons + stdin.
    if stop_event is None:
        stop_event = threading.Event()
    if control is None:
        control = (
            DaggerStdinControl()
            if cfg.policy_type == "custom"
            else _StdinPolicyControl()
        )

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

    # Fail fast (before any hardware, the relay, or the dataset) if --fps
    # disagrees with the fps the checkpoint was trained at — DAgger is the
    # path that loads existing checkpoints, so a 60 fps policy under the
    # 30 fps default would otherwise run and record at half speed.
    remote_policy = cfg.policy_type == "custom"
    if remote_policy:
        import math

        cfg.plan_config.validate(fps=fps, horizon=cfg.actions_per_chunk)
        if not cfg.hold_to_intervene:
            raise ValueError("Remote DAgger requires hold_to_intervene=true")
        if cfg.subtasks:
            raise ValueError("Remote policy instructions are managed on the desktop")
        if not math.isfinite(cfg.handover_duration_s) or cfg.handover_duration_s < 0:
            raise ValueError("handover_duration_s must be finite and nonnegative")
        if not isinstance(cfg.teleop_config, AxolVRTeleopConfig):
            raise ValueError("Remote DAgger requires Axol VR teleop")
    else:
        if not cfg.policy_path.strip():
            raise ValueError("Local DAgger requires policy_path")
        if cfg.hold_to_intervene or cfg.record_joint_actions:
            raise ValueError(
                "Hold controls and separate joint recording require a remote policy"
            )
        _check_training_fps(cfg)
    if episode_time_s < 0 or fps <= 0 or teleop_hz <= 0:
        raise ValueError("Invalid episode duration or control rate")

    # Resolve and validate the destination before camera enumeration, workers,
    # the relay, or the robot can start. LeRobotDataset.resume keeps the existing
    # schema and ignores the fresh feature dict supplied to the recorder, so a
    # non-DAgger dataset cannot be made label-capable implicitly on resume.
    root = str(Path(root).expanduser()) if root else None
    dataset_root = Path(root) if root else HF_LEROBOT_HOME / repo_id
    # Name the dataset for the panel's preview. getattr: a downstream
    # package may hand in its own control without this hook.
    if (note_dataset := getattr(control, "note_dataset", None)) is not None:
        note_dataset(repo_id, dataset_root)
    meta = dataset_root / "meta"
    has_info = (meta / "info.json").exists()
    is_complete = (
        has_info and (meta / "tasks.parquet").exists() and (meta / "episodes").is_dir()
    )
    if has_info and not is_complete:
        raise RuntimeError(
            f"Incomplete dataset found at {dataset_root} (missing "
            "tasks.parquet or episodes/). Move or delete that exact dataset "
            "directory, then rerun to start fresh."
        )
    if dataset_root.exists() and not is_complete:
        try:
            # Atomic and deliberately non-recursive: only a provably empty
            # directory may be cleared for LeRobotDataset.create. Never erase
            # an arbitrary user-supplied --root just because it lacks Axol
            # metadata.
            from ..utils.state_files import secure_rmdir

            secure_rmdir(dataset_root)
        except OSError as exc:
            raise RuntimeError(
                f"Refusing to create a DAgger dataset at {dataset_root}: the "
                "existing path is not an empty directory. Choose a new --root, "
                "or inspect and move/delete the existing data yourself."
            ) from exc
        _logger.info(f"Removed empty dataset directory at {dataset_root}.")
    if is_complete:
        _require_dagger_resume_schema(dataset_root)

    # Guarded return-to-rest knobs, read from the shared teleop config (the
    # same fields collect-data / `axol teleop` use — see VRTeleopConfig).
    reset_torque_threshold = 6.0
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

    # Build (but do not connect) the robot/camera wrappers, then load and
    # schema-check the local policy before starting the IK worker, camera
    # relay, teleop, or CAN hardware. A same-width positional mismatch must
    # fail while every physical actuator is still untouched.
    # Propagate the SKU before constructing DaggerVRTeleop: it caches its
    # action keys in __init__, so changing the config afterward is too late and
    # would leave a gripperless robot paired with a gripper-bearing teleop schema.
    if isinstance(cfg.robot_config, AxolRobotConfig) and isinstance(
        cfg.teleop_config, AxolVRTeleopConfig
    ):
        cfg.teleop_config.has_gripper = cfg.robot_config.axol_config.has_gripper
        cfg.teleop_config.vr_teleop_config.gripper = (
            cfg.robot_config.axol_config.gripper
        )

    robot = AxolRobot(cfg.robot_config)
    teleop = DaggerVRTeleop(cfg.teleop_config)
    teleop.set_hold_to_intervene(cfg.hold_to_intervene)
    from ..recording.datasets import (
        dataset_features_for_robot,
        require_dataset_resume_schema,
    )

    width, height = ZED_RESOLUTION_DIMS[dataset_resolution]
    recorder_features = dataset_features_for_robot(
        _recording_view(robot, cfg.record_joint_actions),
        image_shape=(height, width, 3),
        extra_features={"intervention": INTERVENTION_FEATURE},
    )
    if is_complete:
        require_dataset_resume_schema(
            dataset_root,
            recorder_features,
            fps=fps,
            # The recorder computes pose lag whenever an existing
            # Mantis-derived dataset declares it.
            allowed_extra_features=frozenset({"observation.pose_lag"}),
        )
        # A crashed session can lose buffered episodes while info.json's count
        # survives. Repair only after the full schema proves this is the
        # current run's intended dataset.
        check_resume_consistency(dataset_root)
    if remote_policy:
        from ..policy.plan_client import policy_url
        from ..policy.plan_dagger import PlanDaggerPolicy

        policy = PlanDaggerPolicy(
            policy_url(cfg.server_host, cfg.server_port),
            fps=fps,
            horizon=cfg.actions_per_chunk,
            config=cfg.plan_config,
            shadow_inference=cfg.shadow_inference,
        )
    else:
        policy = _LocalPolicy(cfg.policy_path, cfg.policy_type, cfg.device, task)
    try:
        policy.connect(robot)
    except BaseException:
        policy.close()
        raise

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
    reset_controller: IKResetController | None = None
    relay = None
    try:
        affinity.pin_realtime()

        hostname = socket.gethostname()
        host_ip = local_ip()
        print("Connect the VR app (https://axol.almond.bot) to this machine:")
        print(f"  Hostname : {hostname}.local")
        print(f"  IP       : {host_ip}")

        if rerun_ip:
            init_rerun(session_name="axol_dagger", ip=rerun_ip, port=rerun_port)

        # Start the IK reset worker once the policy schema has been proven. Its
        # JAX JIT overlaps with robot connect and the teleop's own IK worker
        # JIT. It owns collision-aware homing between episodes.
        if remote_policy:
            vr_cfg = cfg.teleop_config.vr_teleop_config
            reset_controller = IKResetController(
                rest_pose_left=vr_cfg.rest_pose_left,
                rest_pose_right=vr_cfg.rest_pose_right,
            )
        else:
            reset_controller = IKResetController()
        reset_controller.start()
        _logger.info("Started IK reset worker (collision-aware return-to-rest).")

        # The out-of-process video relay owns the cameras and streams the
        # headset view. Its dataset branch stays relay-encoded for the recorder
        # (mux only) and a second ring branch, decimated to --policy_fps,
        # gives this policy process readable frames; without gst's shm plugin
        # both fall back to pyshm.
        relay = _start_video_relay(
            cfg,
            dataset_resolution,
            raw_transport="gstshm+pyshm",
            policy_fps=cfg.policy_fps,
        )
        expected = set(cfg.robot_config.observation_cameras().keys())
        readable = set(relay.readable_raw_cameras) if relay is not None else set()
        if relay is None or not expected <= readable:
            raise RuntimeError(
                "collect-dagger requires the gst video relay with readable raw "
                f"frames for {sorted(expected)} (got "
                f"{sorted(readable) if relay else 'no relay'}). Install "
                "the GStreamer stack (`axol gst.install` + `axol gst.build-zed`) "
                "and check the camera serials."
            )
        robot.set_external_cameras({k: relay.raw_cameras[k] for k in expected})
        recorder_features = dataset_features_for_robot(
            _recording_view(robot, cfg.record_joint_actions),
            extra_features={"intervention": INTERVENTION_FEATURE},
        )
        if is_complete:
            # The shared-memory readers now expose the exact downscaled shapes;
            # verify the final contract before any CAN actuator is opened.
            require_dataset_resume_schema(
                dataset_root,
                recorder_features,
                fps=fps,
                allowed_extra_features=frozenset({"observation.pose_lag"}),
            )
    except BaseException as setup_error:
        setup_failures: list[tuple[str, BaseException]] = []
        if relay is not None:
            try:
                relay.shutdown()
            except BaseException as cleanup_error:
                _logger.exception("video relay setup cleanup failed")
                setup_failures.append(("video relay", cleanup_error))
        if reset_controller is not None:
            try:
                reset_controller.stop()
            except BaseException as cleanup_error:
                _logger.exception("IK reset setup cleanup failed")
                setup_failures.append(("IK reset worker", cleanup_error))
        try:
            policy.close()
        except BaseException as cleanup_error:
            _logger.exception("policy setup cleanup failed")
            setup_failures.append(("policy", cleanup_error))
        if _orig_affinity is not None:
            try:
                os.sched_setaffinity(0, _orig_affinity)
            except OSError as cleanup_error:
                setup_failures.append(("CPU affinity restore", cleanup_error))
        for label, cleanup_error in setup_failures:
            setup_error.add_note(
                f"additional {label} cleanup failure: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        reset_failure = next(
            (
                failure
                for label, failure in setup_failures
                if label == "IK reset worker"
            ),
            None,
        )
        if reset_failure is not None:
            mark_hardware_cleanup_uncertain(setup_error, reset_failure)
        raise

    assert reset_controller is not None and relay is not None

    episodes_recorded = 0
    episode_idx = 0
    recorder: DatasetRecorderProcess | None = None
    control_thread: _DaggerControlLoop | None = None
    control_worker_stopped = True
    # Disabling raised arms drops them under gravity, so the cleanup below
    # returns them to rest first — but only when they are somewhere else.
    arms_at_rest = True

    def _return_to_rest_guarded(
        wait_retry: Callable[[], bool] | None, *, final: bool = False
    ) -> bool:
        """Guarded ``IKResetController`` home; ``False`` when aborted.

        Plays with the torque watchdog live; on contact the arms drop into a
        limp gravity-comp hold until ``wait_retry`` answers (``True`` =
        replan from wherever they were hand-guided to) or the run stops.

        ``final=True`` is the teardown park played on the way out. The stop
        flag is already set by then, so a deadline bounds the move instead —
        well inside the caller's stop grace — and there is no operator left
        to answer a contact retry.
        """
        nonlocal arms_at_rest
        if final:
            deadline = time.perf_counter() + PARK_TIMEOUT_S
            contact = threading.Event()
            arms_at_rest = reset_controller.return_to_rest(
                robot,
                torque_threshold=reset_torque_threshold,
                gravity_comp_kd=reset_gravity_comp_kd,
                # A contact trip ends the park at once rather than holding
                # limp until the deadline: the torque-off follows either way.
                stopped=lambda: contact.is_set() or time.perf_counter() >= deadline,
                on_contact=contact.set,
            )
            return arms_at_rest
        arms_at_rest = reset_controller.return_to_rest(
            robot,
            torque_threshold=reset_torque_threshold,
            gravity_comp_kd=reset_gravity_comp_kd,
            stopped=stop_event.is_set,
            wait_retry=wait_retry,
        )
        return arms_at_rest

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
        quit_hint = (
            "q parks and exits; Ctrl+C aborts without parking"
            if isinstance(control, DaggerStdinControl)
            else "Ctrl+C quits"
        )
        return (
            f"Episode {episode_idx + 1}: reset the scene (grips teleop the "
            "arms, reset button homes them), then press record in VR to "
            f"start ({quit_hint})."
        )

    def _idle_reset_retry() -> bool:
        """Contact-hold retry via the VR reset button (idle-phase homes).

        The idle reset stays armed through the move, so the operator who
        requested the home ends its contact hold the same way — pressing
        reset again. The panel's gate works too. The remote terminal accepts
        q as an abort here; it must not start a parking move after contact.

        The hold borrows the idle gate rather than opening its own, so the
        panel names the contact instead of still reading "reset the scene /
        Start episode" while the arms hang limp.
        """
        _logger.info(
            "Contact during return to rest. Free the arms, then press the "
            "VR reset button (or continue in the panel) to retry."
        )
        announce_gate = (
            control.begin_gate
            if isinstance(control, DaggerStdinControl)
            else control.note_gate
        )
        announce_gate(
            "Contact during return to rest — the arms are limp and free to "
            "move. Clear them, then press the VR reset button, or return to "
            "rest here.",
            "Return to rest",
            phase=_GATE_CONTACT,
        )
        retry = False
        final_decision = None
        try:
            while not stop_event.is_set():
                terminal_gate = isinstance(control, DaggerStdinControl)
                if not terminal_gate and teleop.consume_idle_reset():
                    retry = True
                    break
                if (decision := control.poll_gate()) is not None:
                    retry = decision == "go"
                    break
                if terminal_gate and teleop.consume_idle_reset():
                    retry = True
                    break
                time.sleep(0.1)
        finally:
            if isinstance(control, DaggerStdinControl):
                # The controller is already holding limp on contact; leave
                # that mode untouched while restoring terminal ownership.
                control.end_gate()
                final_decision = control.poll_gate()
            control.note_gate(_idle_gate_message())
        return retry and final_decision is None

    session_error: BaseException | None = None
    normal_quit = False
    abort_requested = False
    robot_connected = False
    try:
        if remote_policy:
            # Current-pose sessions can skip return_to_rest, which normally
            # consumes this handshake. Always prove reset/parking readiness
            # before enabling motors, including quit before the first episode.
            _logger.info("Waiting for the collision-aware reset worker...")
            if not reset_controller.wait_ready(stopped=stop_event.is_set):
                return
            if cfg.robot_config.cartesian_actions:
                # The first Cartesian solve can compile for many seconds.
                # Finish that work before enabling motors, so the first
                # policy dispatch cannot stall an armed control loop.
                robot.prepare_cartesian_actions()
        _logger.info("Connecting robot...")
        robot.connect()
        robot_connected = True

        # Connect the VR teleop stack: the position source lets takeovers
        # sync the IK worker to the robot's measured pose, and the current
        # positions seed the teleop filters (mirrors collect-data).
        teleop.set_position_source(lambda: robot.positions)
        pos_l, pos_r = robot.positions
        _logger.info("Connecting VR teleop (IK worker JIT may take ~20s)...")
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
            _logger.info(f"Resuming existing dataset at {dataset_root}.")
        recorder = DatasetRecorderProcess(
            raw_cond=relay.raw_cond,
            raw_meta=relay.raw_meta,
            obs_keys=list(robot.get_joint_observation().keys()),
            action_keys=list(
                _recording_view(robot, cfg.record_joint_actions).action_features
            ),
            config={
                "repo_id": repo_id,
                "root": root,
                "dataset_root": str(dataset_root),
                "is_complete": is_complete,
                "features": recorder_features,
                "allowed_resume_features": ["observation.pose_lag"],
                "robot_type": robot.name,
                "fps": fps,
                "vcodec": vcodec,
                "rerun_ip": rerun_ip,
                "rerun_port": rerun_port,
                "push_to_hub": cfg.push_to_hub,
                "log_level": cfg.log_level,
                # The relay runs the policy ring beside the dataset branch, so
                # the mux-only recorder needs the IK core to keep 60 rows/s
                # (affinity.pin_background_and_ik).
                "share_ik_core": True,
            },
        )
        episode_idx = recorder.episode_count()

        if cfg.home_on_start or not cfg.start_from_current_pose:
            _logger.info("Returning to rest pose.")
            if not _return_to_rest_guarded(_gate_retry):
                return
        else:
            # Starting from wherever the arms are: until a home proves
            # otherwise, the teardown park must not assume they are at rest.
            arms_at_rest = False

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
            # Scene-reset teleop can leave the arms anywhere; count them as
            # off the rest pose for the teardown park until proven otherwise.
            was_at_rest = arms_at_rest
            arms_at_rest = False
            idle_teleop_used, started = _idle_teleop_until_record(
                teleop,
                robot,
                lambda: _return_to_rest_guarded(_idle_reset_retry),
                teleop_hz,
                control,
                stop_event,
            )
            # An idle VR-reset home sets ``arms_at_rest`` itself on the way.
            arms_at_rest = not idle_teleop_used and (was_at_rest or arms_at_rest)
            teleop.set_idle_reset_armed(False)
            teleop.set_intervention_allowed(False)
            teleop.force_disengage()
            final_gate_decision = _finish_idle_terminal_gate(
                control,
                robot,
                teleop_hz,
                hold=(
                    not stop_event.is_set()
                    and getattr(control, "abort_requested", False) is not True
                ),
            )
            if final_gate_decision is not None:
                started = False
                abort_requested = final_gate_decision == "abort"
            if not started:
                break
            if idle_teleop_used and not cfg.start_from_current_pose:
                # The operator moved the arms during the scene reset; the
                # policy expects to start from the rest pose.
                _logger.info("Returning to rest pose before the policy starts.")
                if not _return_to_rest_guarded(_gate_retry):
                    break

            # Fresh episode: drop the policy's episode-scoped state (obs
            # history / hidden state from the previous episode).
            if not remote_policy:
                policy.reset()
            policy.set_instruction(task)
            if isinstance(control, DaggerStdinControl):
                # Keep q live through recorder setup, before any policy
                # reset or control-worker launch. No blocking prompt owns
                # stdin after this point until end_episode joins the reader.
                control.begin_episode()
            # Arm the recorder before opening the relay branch: on the encoded
            # transport that is what makes row zero an IDR admitted at the
            # shared exposure boundary (the raw fallback is indifferent). Both calls
            # are bounded IPC transactions, but together they can exceed the
            # Rust target watchdog; hold the just-measured post-rest pose while
            # they run off-thread. Policy inference deliberately has not
            # started yet, so camera setup cannot advance the trajectory.
            start_hold_action = _measured_joint_hold_action()

            def _start_capture() -> None:
                recorder.start_episode(task)
                relay.set_raw_enabled(True)

            def _hold_start_pose(action=start_hold_action) -> None:
                robot.send_action(action)
                # The relay branch opens on an exposure boundary inside
                # _start_capture, before the control loop's first tick
                # publishes: bracket the take's first exposures with robot
                # state from here, or the recorder's history starts after
                # them and those rows are dropped (5 per take, 2026-09-15).
                recorder.publish(
                    robot.get_joint_observation(),
                    action
                    if cfg.record_joint_actions
                    else robot.action_to_dataset(action),
                    time.perf_counter(),
                )

            run_blocking_with_sync_control_ticks(
                _start_capture,
                _hold_start_pose,
                boundary_period,
                drain_tick=_hold_start_pose,
            )

            terminal_stop = isinstance(
                control, DaggerStdinControl
            ) and control.poll_choice() in ("q", "abort")
            if stop_event.is_set() or terminal_stop:
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

            if remote_policy:
                from .plan_dagger_control import PlanDaggerControlLoop

                policy.reset()
                control_thread = PlanDaggerControlLoop(
                    robot=robot,
                    policy=policy,
                    teleop=teleop,
                    recorder=recorder,
                    fps=fps,
                    teleop_hz=teleop_hz,
                    initial_action=start_hold_action,
                    handover_duration_s=cfg.handover_duration_s,
                    record_joint_actions=cfg.record_joint_actions,
                )
            else:
                control_thread = _DaggerControlLoop(
                    robot=robot,
                    policy=policy,
                    teleop=teleop,
                    recorder=recorder,
                    fps=fps,
                    teleop_hz=teleop_hz,
                    limiter=(
                        PolicyActionLimiter(
                            cfg.policy_max_vel, cfg.policy_max_accel, fps
                        )
                        if cfg.policy_max_vel > 0
                        and not getattr(robot.config, "observe_cartesian", False)
                        else None
                    ),
                )

            def _switch_subtask(idx: int) -> None:
                """Switch the live policy instruction to subtask ``idx`` (1-based)."""
                text = subtasks[idx - 1]
                policy.set_instruction(text)
                _logger.info(f"Subtask {idx}: {text}")

            print(
                "  Hold either grip to operate that arm; release both to resume policy."
                if remote_policy
                else "  Grips: one=freeze (pause recording), both=take over, "
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
            # Retain the exact worker and mark its resources owned before
            # start(), so a partial thread-start failure also reaches the
            # final liveness gate.
            control_worker_stopped = False
            # The policy is about to drive the arms off the rest pose.
            arms_at_rest = False
            control_thread.start()
            if not isinstance(control, DaggerStdinControl):
                control.begin_episode(_switch_subtask, len(subtasks))

            deadline = (
                time.perf_counter() + episode_time_s if episode_time_s else float("inf")
            )
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
                        _logger.info(
                            f"Fatal error in DAgger control loop: "
                            f"{control_thread.fatal_error!r}. Aborting run "
                            "without saving the current episode."
                        )
                        break
                    time.sleep(0.1)
            except KeyboardInterrupt:
                interrupted = True
                abort_requested = True

            control_end_error: BaseException | None = None
            try:
                control.end_episode()
            except BaseException as error:
                control_end_error = error
            control_worker_stopped, worker_stop_error = _stop_dagger_control_worker(
                control_thread
            )
            if worker_stop_error is not None:
                if control_thread.fatal_error is not None:
                    worker_stop_error.add_note(
                        "The control loop had already reported: "
                        f"{type(control_thread.fatal_error).__name__}: "
                        f"{control_thread.fatal_error}"
                    )
                if control_end_error is not None:
                    worker_stop_error.add_note(
                        "additional episode-control cleanup failure: "
                        f"{type(control_end_error).__name__}: {control_end_error}"
                    )
                if control_worker_stopped:
                    raise worker_stop_error
                # Wedged past shutdown: handled below (disconnect first).
                _logger.error("%s", worker_stop_error)
            if control_end_error is not None and control_worker_stopped:
                raise control_end_error
            teleop.set_intervention_allowed(False)
            teleop.force_disengage()
            if not control_worker_stopped:
                # A policy call wedged past shutdown can otherwise wake later
                # and issue one more command while the supervisor homes/saves.
                # Disconnect first, then discard capture; never race a second
                # controller against this still-live thread.
                try:
                    if remote_policy:
                        robot.disconnect_preserving_position()
                    else:
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

            if interrupted or (
                isinstance(control, DaggerStdinControl)
                and control.poll_choice() == "abort"
            ):
                abort_requested = True
                recorder.cancel_episode()
                break
            if control_thread.fatal_error is not None:
                recorder.cancel_episode()
                break

            choice = control.poll_choice() or control_thread.vr_choice
            if choice == "q":
                # Quit also wins over an episode-local capture rejection.
                # Retrying capture here would lose the latched terminal quit
                # instead of proceeding to the clean parking path.
                normal_quit = True
                recorder.cancel_episode()
                break

            if capture_error is not None:
                # finish_episode has joined capture and cleared its rejected
                # buffer. Home exactly like a normal episode boundary before
                # retrying the unchanged dataset episode index.
                recorder.cancel_episode()
                teleop.send_feedback_state(VRState.SAVING)
                _logger.info(
                    f"Episode discarded because camera capture failed: {capture_error}"
                )
                _logger.info("Returning to rest pose.")
                if not _return_to_rest_guarded(_gate_retry):
                    break
                continue

            if timed_out and choice is None:
                _logger.info(
                    f"Episode time cap ({episode_time_s}s) reached; saving the episode."
                )
                choice = "s"

            teleop.send_feedback_state(VRState.SAVING)
            _logger.info("Returning to rest pose.")
            # An aborted home (stop / declined retry) must not discard a
            # fully-recorded episode: fall through to the save/discard
            # decision either way; the session loop then winds down on the
            # stop flag. The outcome still reaches the teardown park, which
            # reads the ``arms_at_rest`` the call updates.
            _return_to_rest_guarded(_gate_retry)

            if choice == "r":
                _logger.info("Re-recording episode.")
                recorder.cancel_episode()
                continue

            if final_rows == 0:
                # Nothing reached the dataset (e.g. every observation
                # failed, or the episode was ended instantly) —
                # save_episode would raise on the empty buffer.
                _logger.info("No frames were captured this episode; discarding.")
                recorder.cancel_episode()
                continue
            _logger.info("Saving episode…")
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
                _logger.info(f"Episode NOT saved: {exc}")
                continue
            restore_dataset_ownership(dataset_root)
            episode_idx += 1
            episodes_recorded += 1
            control.note_saved()
            _logger.info(
                f"Saved episode {episodes_recorded} "
                f"({control_thread.interventions} intervention(s))."
            )

        if control_thread is not None and control_thread.fatal_error is not None:
            raise control_thread.fatal_error

    except KeyboardInterrupt:
        abort_requested = True
    except BaseException as error:
        session_error = error
        raise
    finally:
        import signal

        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except (ValueError, OSError):
            pass

        _logger.info("Stopping.")
        cleanup_failures: list[tuple[str, BaseException]] = []

        if not control_worker_stopped:
            control_worker_stopped, final_worker_error = _stop_dagger_control_worker(
                control_thread
            )
            if not control_worker_stopped:
                ownership_error = HardwareCleanupError(
                    "DAgger control loop remained alive after the final join; "
                    "recorder and robot/teleop/relay ownership are uncertain"
                )
                if final_worker_error is not None:
                    ownership_error.add_note(
                        "additional final DAgger worker cleanup failure: "
                        f"{type(final_worker_error).__name__}: {final_worker_error}"
                    )
                cleanup_failures.append(("DAgger control loop", ownership_error))
            elif final_worker_error is not None:
                cleanup_failures.append(("DAgger control loop", final_worker_error))

        def _cleanup(
            label: str,
            cleanup: Callable[[], Any],
            *,
            requires_control_exit: bool = True,
        ) -> BaseException | None:
            error = _cleanup_dagger_resource(
                control_stopped=control_worker_stopped or not requires_control_exit,
                label=label,
                cleanup=cleanup,
            )
            if error is not None:
                cleanup_failures.append((label, error))
            return error

        if remote_policy:
            # Stop acquisition locally before releasing hardware. Network
            # close may wait seconds for a peer; it must not starve the live
            # realtime target watchdog while no control worker owns the arms.
            _cleanup("policy pause", policy.pause)
        else:
            _cleanup("policy", policy.close)
        # Park before the torque comes off: ``disconnect()`` disables the
        # motors, and arms left raised drop under gravity.
        #
        # Local policies only: a remote policy's disconnect_plan_robot below
        # owns its own park-or-preserve decision. Skipped when the arms are
        # already at rest, when a limp hold left
        # them in the operator's hands, when the bus no longer reports a pose
        # to plan from, or when a live control worker may still be inside the
        # robot (a wedged worker also means the episode boundary already
        # disconnected). ``collect-data`` reads the same states off its teleop
        # core instead, because its rest moves are planned by the teleop IK
        # worker rather than by this out-of-band reset controller.
        #
        # Bounded by the deadline ``final=True`` installs. Every failure is
        # swallowed, so the disconnect below happens either way and a lost
        # park costs only what was lost before it existed.
        try:
            if (
                not remote_policy
                and control_worker_stopped
                and not arms_at_rest
                and not reset_controller.arms_limp
                and arms_reporting(robot)
            ):
                _logger.info("Returning to rest before disabling the arms.")
                _return_to_rest_guarded(None, final=True)
        except BaseException:
            _logger.exception("return to rest before disconnect failed")
        # Disconnect never waits on an exit proof: a wedged control thread
        # that wakes later finds no core to command. Remote-policy aborts
        # preserve motor support; only an explicit clean quit parks first.
        # Disconnect is idempotent after the episode-boundary teardown.
        disconnect_robot = robot.disconnect
        if remote_policy and robot_connected:
            from .plan_dagger_control import disconnect_plan_robot

            def disconnect_robot():
                disconnect_plan_robot(
                    robot,
                    reset_controller,
                    park=(
                        (
                            normal_quit
                            or getattr(control, "quit_requested", False) is True
                        )
                        and session_error is None
                        and not abort_requested
                        and getattr(control, "abort_requested", False) is not True
                        and not cleanup_failures
                        and control_worker_stopped
                        and not stop_event.is_set()
                        and (
                            control_thread is None or control_thread.fatal_error is None
                        )
                    ),
                    torque_threshold=reset_torque_threshold,
                    stopped=stop_event.is_set,
                )

        disconnect_failure = _cleanup(
            "robot disconnect", disconnect_robot, requires_control_exit=False
        )
        if isinstance(control, DaggerStdinControl):
            # Hardware no longer relies on a target heartbeat, so a bounded
            # reader join cannot starve the realtime controller here.
            _cleanup("terminal input", control.close, requires_control_exit=False)
        if remote_policy:
            _cleanup("policy", policy.close)
        teleop_failure = _cleanup("teleop disconnect", teleop.disconnect)
        reset_failure = _cleanup(
            "IK reset worker",
            reset_controller.stop,
            requires_control_exit=False,
        )
        # Close the relay's raw/dataset branch before any reader detaches (a
        # session error mid-episode leaves it open; see collect_data).
        _cleanup("video relay raw branch", lambda: relay.set_raw_enabled(False))
        # Recorder owns the dataset: finalize (and empty-dataset cleanup)
        # happen in recorder.close(). Shut the relay down after it so the
        # recorder's shm readers never outlive their blocks.
        if recorder is not None:
            _cleanup("recorder", recorder.close)
            _cleanup(
                "dataset ownership restore",
                lambda: restore_dataset_ownership(dataset_root),
            )
        # Detach our own shared-memory readers (the robot's external cameras)
        # before their writer blocks go away with the relay, so the resource
        # tracker doesn't report them as leaked at exit.
        for name, cam in relay.raw_cameras.items():
            _cleanup(f"raw camera {name}", cam.close)
        relay_failure = _cleanup("video relay", relay.shutdown)

        # Restore the process's original CPU affinity (the process may run
        # other operations after this one).
        if _orig_affinity is not None:
            _cleanup(
                "CPU affinity restore",
                lambda: os.sched_setaffinity(0, _orig_affinity),
                requires_control_exit=False,
            )

        try:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        except (ValueError, OSError):
            pass
        _finish_dagger_cleanup(
            session_error=session_error,
            disconnect_failure=disconnect_failure,
            teleop_failure=teleop_failure,
            reset_failure=reset_failure,
            relay_failure=relay_failure,
            additional_failures=tuple(
                (label, failure)
                for label, failure in cleanup_failures
                if all(
                    failure is not selected
                    for selected in (
                        disconnect_failure,
                        teleop_failure,
                        reset_failure,
                        relay_failure,
                    )
                )
            ),
        )
