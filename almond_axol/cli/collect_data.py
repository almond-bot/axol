"""
axol collect-data

Record teleoperation episodes with the Axol robot and its local ZED cameras.
Episode boundaries are driven by VR controller commands:
  - DATA_COLLECTION → RECORDING:              start collecting frames
  - RECORDING → DATA_COLLECTION:              stop; save episode (success)
  - RECORDING → DATA_COLLECTION + reset btn:  stop; discard episode (rerecord)

While saving, the VR headset is pushed into the SAVING state so recording
controls are blocked until save_episode() completes.

After an episode ends the arms return to rest automatically, but *guarded*:
a torque-residual watchdog compares measured torque against the gravity
model while the move plays. Unexpected contact — a gripper still hooked on
the scene, or the operator grabbing an arm — trips the watchdog: the move
stops where it is and the arms drop into a limp gravity-compensation hold
so they can be hand-guided clear (the episode saves in the background
meanwhile). Pressing reset (X) then replans the collision-aware
return-to-rest from wherever the arms were left.

Recording continues until Ctrl+C.

When run from the web control panel (``axol serve``), an injected
``_QueueCollectControl`` merges dashboard commands with the VR events each
control tick, so a session can be driven with the headset off: Start opens a
3 s countdown to recording (mirroring the in-headset one), and the Save /
Discard buttons end the episode with the same outcomes as the VR record and
reset+record presses. The control's ``snapshot()`` mirrors the headset HUD to
the panel — phase, episode number, saved count, a status line, and the buttons
valid right now — and the panel additionally shows the relay's camera streams
(it joins the VR WebSocket server as one more WebRTC client). A contact hold is
one of those phases: without it the panel sat on "Saving…" (no buttons) for as
long as the arms stayed limp, which reads as a stuck save, and a headset-off
session had no way out of the hold but Stop.

The teleop loop runs at ``--teleop_hz`` and publishes the latest
``(joint_obs, action)`` snapshot every tick. The dataset itself — frame
capture, row assembly, encoding, ``save_episode`` — is owned by a separate
recorder (see :mod:`almond_axol.recording.record_proc`): a subprocess when the video
relay is up (so the per-frame work never shares the GIL with the control loop),
or in-process as a fallback when there is no relay. Either way each recorded
frame is aligned to the joint sample by its shared ``perf_counter`` capture
timestamp.
"""

import asyncio
import logging
import math
import os
import queue
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig

from ..lerobot.camera.configuration_zed import ZedCameraConfig, resolution_for_dims
from ..lerobot.robot.config_axol import AxolRobotConfig
from ..lerobot.teleop.config_vr import AxolVRTeleopConfig
from ..recording import (
    DatasetRecorderProcess,
    InProcessRecorder,
    default_vcodec,
    restore_dataset_ownership,
)
from ..utils import affinity
from ..utils.jetson_diag import TegraStatsDiag
from ..utils.proc_diag import SystemDiag
from .config import DatasetResolution, LogLevel, parse

if TYPE_CHECKING:
    from ..lerobot.robot.robot_axol import AxolRobot

_logger = logging.getLogger(__name__)


def _default_robot_config() -> AxolRobotConfig:
    """Default Axol robot config for data collection: local ZED cameras.

    All three slots (overhead, left_arm, right_arm) are seeded with the
    unassigned sentinel serial ``0`` so each stays reachable as a dotted
    ``--robot_config.cameras.<slot>.serial`` override (or control-panel field),
    but only the slots you assign a serial to are recorded — the rest are
    pruned by ``AxolRobotConfig.select_assigned_cameras`` (at least one must be
    assigned). draccus takes dict fields as one inline YAML/JSON value, so
    assign serials with e.g. ``--robot_config.cameras "{overhead: {serial:
    41234567}}"``. Other fields are overridable too, e.g.
    ``--robot_config.axol_config.left.elbow.kp 60``.
    """
    return AxolRobotConfig(
        cameras={
            "overhead": ZedCameraConfig(serial=0),
            "left_arm": ZedCameraConfig(serial=0),
            "right_arm": ZedCameraConfig(serial=0),
        },
        # The control loop runs motion_control every step, whose command replies
        # keep the joint cache fresh — so the background telemetry poll loop is
        # redundant CAN/CPU load. Skipping it (telemetry_hz=0) matches `axol
        # teleop` and keeps the control rate from sagging when teleop engages.
        telemetry_hz=0.0,
    )


def _register_camera_video(robot: "AxolRobot", teleop: Any) -> None:
    """Register the ZED cameras as WebRTC video sources for the headset.

    Relays every camera the robot exposes (overhead — or ``overhead_left`` /
    ``overhead_right`` when stereo — plus both wrist cameras) so the headset can
    show them. Each camera is registered bare and the relay picks the right
    WebRTC track per source (see :func:`almond_axol.video.video._track_for_source`):
    a gst camera/eye already produces GPU-encoded H.264 access units (its
    ``subscribe()`` feeds a pre-encoded track — the same grab/encode serves the
    dataset), while an SDK camera is adapted to a frame-driven source that
    encodes each frame as soon as it's captured. Reads only consume the latest
    frame each camera already keeps, so the dataset capture pipeline is never
    blocked.
    """
    if not robot.cameras:
        return

    try:
        teleop.set_video_sources(dict(robot.cameras))
    except Exception as exc:
        _logger.warning("failed to enable camera video: %s", exc)


def check_resume_consistency(dataset_root: "Path") -> None:
    """Make sure a resumed dataset didn't lose episodes to a killed recorder.

    A recorder subprocess killed before flushing (its "recorder subprocess
    exited … before shutdown" error is in that session's log) loses the
    episodes still buffered in its parquet writer, while ``info.json``'s
    ``total_episodes`` — already bumped per save — survives. Resuming such a
    dataset as-is would number the next episode past the lost ones, leaving a
    permanent index gap. That gap is poison downstream: LeRobot's episode
    metadata lookups are positional (``meta.episodes[i]`` is a row position,
    not a key), so on a gapped dataset every episode after the gap silently
    resolves to a *different* episode's video span.

    Since the lost episodes' frames are unrecoverable anyway, the torn tail
    is repaired in place: the dataset is truncated back to the longest
    verified contiguous prefix of episodes and its totals rewritten to match
    (see :mod:`almond_axol.recording.repair`), so the session resumes cleanly
    right after the last intact episode. Raises only when not a single
    complete episode survives.
    """
    from ..recording.repair import ensure_resume_consistency

    ensure_resume_consistency(dataset_root)


def _existing_dataset_resolution(dataset_root: "Path") -> str | None:
    """Resolution name of an existing dataset's recorded images, or ``None``.

    On resume the dataset's image feature shape is fixed (baked into
    ``meta/info.json`` when the dataset was created), so the relay must deliver
    frames at that resolution — a differently-sized frame fails LeRobot's
    ``validate_frame`` and kills the capture thread mid-episode. The caller reads
    this to pin the relay to the existing resolution. Returns ``None`` if the
    file/shape can't be read or doesn't map to a known ZED resolution.
    """
    import json

    try:
        data = json.loads((dataset_root / "meta" / "info.json").read_text())
        features = data.get("features", {})
    except (OSError, ValueError):
        return None
    for key, spec in features.items():
        if not key.startswith("observation.images."):
            continue
        shape = spec.get("shape")
        if not shape or len(shape) != 3:
            continue
        dims = [int(x) for x in shape]
        # Stored as HWC ((H, W, 3)); tolerate a leading channel dim (CHW) too.
        h, w = (dims[1], dims[2]) if dims[0] == 3 else (dims[0], dims[1])
        try:
            return resolution_for_dims(w, h)
        except ValueError:
            return None
    return None


def _start_video_relay(
    cfg: "CollectDataConfig",
    dataset_resolution: str,
    raw_transport: str | None = None,
) -> Any | None:
    """Start the out-of-process video relay for data collection.

    The relay subprocess opens the ZED cameras on the GPU-resident gst pipeline,
    streams the headset view over WebRTC (aiortc), **and** publishes each
    camera's raw RGB frames back to this process through shared memory for the
    dataset (see :mod:`almond_axol.video.shm_frames`). This keeps the control
    process off the camera grab/encode/RTP path entirely, so the teleop and IK
    loops stay as fast as ``axol teleop`` — even while recording.

    ``dataset_resolution`` is the effective downscale target for the dataset (raw)
    branch — the configured value for a fresh dataset, or the existing dataset's
    resolution when resuming (the caller resolves this; see _run).

    ``raw_transport`` optionally forces the relay's raw-branch transport for
    every camera (see :class:`~almond_axol.video.video_proc.VideoRelayProcess`):
    ``collect-dagger`` passes ``"pyshm"`` so the raw frames are readable by the
    control process (policy observations) as well as the recorder subprocess.
    ``None`` keeps the relay's default (gst shm where available).

    Returns the :class:`VideoRelayProcess`, or ``None`` when it can't be used
    (no cameras or aiortc unavailable), in which case the caller uses the
    in-process camera path. The caller must still verify the relay exported raw
    frames for every observation camera before relying on it.
    """
    cameras = getattr(cfg.robot_config, "cameras", {})
    if not cameras:
        return None
    try:
        from ..video.video import webrtc_available
        from ..video.video_proc import VideoRelayProcess
    except Exception as exc:  # noqa: BLE001 - aiortc / gst module missing
        _logger.debug("video relay unavailable: %s", exc)
        return None
    if not webrtc_available():
        return None

    specs: dict[str, dict[str, Any]] = {}
    for name, camcfg in cameras.items():
        # Each camera opts into either branch: ``stream`` (headset) and ``record``
        # (dataset). A camera in neither is dropped — never opened by the relay.
        wants_record = bool(getattr(camcfg, "record", True))
        wants_stream = bool(getattr(camcfg, "stream", True))
        if not (wants_record or wants_stream):
            continue
        serial = int(camcfg.serial)
        spec: dict[str, Any] = {
            "serial": serial,
            "fps": camcfg.fps or 60,
            "record": wants_record,
            "stream": wants_stream,
        }
        res = camcfg.resolution_name() if hasattr(camcfg, "resolution_name") else None
        if res:
            spec["resolution"] = res
        # Downscale target for the dataset (raw) branch only; the encoded headset
        # branch keeps the full capture resolution. Clamped to capture in the relay.
        spec["dataset_resolution"] = dataset_resolution
        if raw_transport is not None:
            spec["raw_transport"] = raw_transport
        # The recorded eyes (``eyes``) must match observation_cameras() so the
        # relay's raw branch exports exactly the keys the recorder expects; the
        # streamed eyes (``stream_eyes``) drive the headset feed independently, so
        # the operator can e.g. stream both eyes for depth while recording only
        # one. Physically-stereo cameras are already flagged stereo on the config
        # (see AxolRobotConfig.apply_detected_stereo, applied in _run before this).
        # For each branch: ``"both"`` records/streams both eyes suffixed
        # (overhead_left / overhead_right); a single eye is cropped and exported
        # under the plain name, so it costs and reads like a mono camera.
        if bool(getattr(camcfg, "stereo", False)):
            record_eyes = getattr(camcfg, "eyes", "both")
            stream_eyes = (
                camcfg.streaming_eyes()
                if hasattr(camcfg, "streaming_eyes")
                else getattr(camcfg, "stream_eyes", None) or record_eyes
            )
            spec["stereo"] = True
            spec["record_eyes"] = (
                ["left", "right"] if record_eyes == "both" else [record_eyes]
            )
            spec["record_suffix"] = record_eyes == "both"
            spec["stream_eyes"] = (
                ["left", "right"] if stream_eyes == "both" else [stream_eyes]
            )
            spec["stream_suffix"] = stream_eyes == "both"
            # Both streamed eyes ship packed side-by-side in one track
            # ({name}_sbs — one decoder session on the headset); the per-eye
            # keys remain as the SDK fallback, which can't pack. Recording
            # (record_eyes) is per-eye regardless.
            spec["stream_sbs"] = stream_eyes == "both"
        else:
            spec["stereo"] = False
        specs[name] = spec

    relay = VideoRelayProcess(specs, want_raw=True)
    # Keep the relay if it can serve *either* branch: raw frames for the dataset
    # (the primary purpose for collect-data) or encoded streams for the headset.
    # A record-only setup (streaming disabled for every camera) has no encoded
    # sources but still needs the relay's raw export, so don't discard it just
    # because nothing streams — otherwise we fall back to the in-process path and
    # open every camera a second time.
    if not (relay.has_sources or relay.raw_cameras):
        relay.shutdown()
        return None
    return relay


@dataclass
class CollectDataConfig:
    """Config for ``axol collect-data``.

    ``robot_config`` and ``teleop_config`` are the full lerobot subsystem
    configs (cameras, per-joint gains, IK, VR server); nest into
    them from the CLI (e.g. ``--robot_config.axol_config.left_stiffness
    0.8``) or supply a whole-config file with ``--config_path``.
    """

    repo_id: str
    task: str
    robot_config: RobotConfig = field(default_factory=_default_robot_config)
    teleop_config: TeleoperatorConfig = field(default_factory=AxolVRTeleopConfig)
    fps: int = 60
    teleop_hz: int = 120
    # Resolution the recorded dataset video is downscaled to (on the relay's VIC,
    # before frames cross to the control process). The headset/teleop stream
    # stays at the camera's full capture resolution. Defaults to SVGA (960x600):
    # full HD1200 frames are ~9 MB each and recording three of them at 60 fps
    # saturates the Jetson CPU moving raw bytes, collapsing the control loop.
    # Clamped to the capture resolution, so it only ever downscales.
    dataset_resolution: DatasetResolution = "SVGA"
    # Video codec for the recorded LeRobot dataset; defaults per-platform (see
    # record_proc.default_vcodec). Override with any of LeRobot's
    # VALID_VIDEO_CODECS (e.g. auto, h264, libsvtav1).
    vcodec: str = field(default_factory=default_vcodec)
    # Every return-to-rest is guarded: a torque watchdog drops the arms into
    # a limp gravity-comp hold on unexpected contact (reset replans from
    # wherever they were left). The knobs live on the shared teleop config —
    # ``--teleop_config.vr_teleop_config.reset_torque_threshold`` (0 disables
    # the watchdog) and ``.reset_gravity_comp_kd`` — the same fields `axol
    # teleop` uses, so the two flows behave identically.
    root: str | None = None
    push_to_hub: bool = False
    rerun_ip: str | None = None
    rerun_port: int = 9876
    log_level: LogLevel = "INFO"


# ----------------------------------------------------------------------
# Episode control: lets the web control panel drive a session with the
# headset off.
#
# The VR flow stays authoritative when a headset is worn — the record button
# starts and ends episodes exactly as before. The panel is a parallel input:
# its commands arrive through a queue (pushed from /api/op/episode by the
# serve layer) and are merged with the VR events each control tick. The panel
# also mirrors everything the headset HUD would show — phase, episode number,
# saved count, a status line, and the buttons valid right now — through
# ``snapshot()``, which the serve layer polls.
# ----------------------------------------------------------------------

# Delay between the panel's "Start recording" click and recording actually
# starting, mirroring the in-headset record countdown so the operator has time
# to pick the controllers back up. A second click cancels.
_PANEL_START_COUNTDOWN_S = 3.0

# Buttons the panel renders per phase (see EpisodeControls in the web app):
# ``confirm`` asks for a second, confirming click — the panel's stand-in for
# the headset's double-press save/discard confirmation. ``contact`` is the
# guarded return's limp hold, whose button stands in for the VR reset press
# that ends it (same label as the equivalent gate in run-policy, which reaches
# the same state through its continue gate).
_COLLECT_PHASE_CONTROLS: dict[str, tuple[dict[str, Any], ...]] = {
    "ready": ({"command": "start", "label": "Start recording"},),
    "countdown": ({"command": "start", "label": "Cancel countdown"},),
    "recording": (
        {"command": "s", "label": "Save episode", "confirm": True},
        {"command": "r", "label": "Discard episode", "confirm": True},
    ),
    "contact": ({"command": "continue", "label": "Return to rest"},),
}


class _NullCollectControl:
    """No-op episode control for the plain CLI (VR-only sessions)."""

    def poll_command(self) -> str | None:
        return None

    def note_ready(self, episode: int) -> None:
        pass

    def note_countdown(self, deadline: float) -> None:
        pass

    def note_recording(self) -> None:
        pass

    def note_saving(self) -> None:
        pass

    def note_saved(self) -> None:
        pass

    def note_contact(self) -> None:
        pass

    def note_returning(self) -> None:
        pass


class _QueueCollectControl:
    """Web episode control for ``collect-data``: panel-driven recording.

    Mirrors the VR controller flow with dashboard buttons so a session can be
    driven with the headset off: ``start`` opens a short countdown to
    recording (``start`` again cancels), and ``s`` / ``r`` end the episode
    saving / discarding it — the same outcomes as the VR record and
    reset+record presses.

    Constructed by the serve runner as ``cls(stop_event)``; ``push`` and
    ``snapshot`` are the API surface it uses (see the ``episode_control``
    argument of :class:`~almond_axol.serve.commands.CommandDef`). The
    loop-side surface (``poll_command`` plus the ``note_*`` phase updates) is
    what :func:`_run` drives.
    """

    def __init__(self, stop_event: threading.Event) -> None:
        self._q: queue.Queue[str] = queue.Queue()
        self._stop = stop_event
        self._lock = threading.Lock()
        self._phase = "preparing"
        self._message = "Preparing…"
        self._episode: int | None = None
        self._episodes_recorded = 0
        # perf_counter deadline of a pending panel-started countdown.
        self._countdown_deadline: float | None = None

    # -- serve API surface --------------------------------------------------

    def push(self, command: str) -> None:
        self._q.put(command)

    def snapshot(self) -> dict[str, Any]:
        """Thread-safe phase/count/message/buttons for ``/api/op/status``."""
        with self._lock:
            phase = self._phase
            message = self._message
            if phase == "countdown" and self._countdown_deadline is not None:
                remaining = max(0.0, self._countdown_deadline - time.perf_counter())
                message = f"Recording starts in {math.ceil(remaining)} s…"
            snap: dict[str, Any] = {
                "phase": phase,
                "episodesRecorded": self._episodes_recorded,
                "message": message,
                "controls": [dict(c) for c in _COLLECT_PHASE_CONTROLS.get(phase, ())],
            }
            if self._episode is not None:
                snap["episode"] = self._episode
            return snap

    # -- loop-side surface --------------------------------------------------

    def poll_command(self) -> str | None:
        """Next panel command (``start``/``s``/``r``/``continue``), or ``None``.

        ``continue`` is only meaningful during a contact hold (it stands in
        for the VR reset press); the episode loop ignores it, so a stale one
        can't affect a recording.
        """
        while True:
            try:
                cmd = self._q.get_nowait()
            except queue.Empty:
                return None
            if cmd in ("start", "s", "r", "continue"):
                return cmd
            # Anything else (stray run-policy commands, typos) is ignored.

    def _set(self, phase: str, message: str) -> None:
        with self._lock:
            self._phase = phase
            self._message = message
            if phase != "countdown":
                self._countdown_deadline = None

    def note_ready(self, episode: int) -> None:
        with self._lock:
            self._episode = episode
        self._set(
            "ready",
            f"Episode {episode}: press record on the VR controller, or start "
            "recording here.",
        )

    def note_countdown(self, deadline: float) -> None:
        with self._lock:
            self._phase = "countdown"
            self._countdown_deadline = deadline

    def note_recording(self) -> None:
        self._set("recording", "Recording — save or discard the episode to end it.")

    def note_saving(self) -> None:
        self._set("saving", "Saving episode…")

    def note_saved(self) -> None:
        with self._lock:
            self._episodes_recorded += 1

    def note_contact(self) -> None:
        self._set(
            "contact",
            "Contact during return to rest — the arms are limp and free to "
            "move. Clear them, then press reset on the controller (or return "
            "to rest here) to replan from where they are.",
        )

    def note_returning(self) -> None:
        self._set("resetting", "Returning to rest…")


Control = _NullCollectControl | _QueueCollectControl


def main(argv: list[str]) -> None:
    """Parse the CLI config and run a data-collection session."""
    cfg = parse(CollectDataConfig, argv)
    # force=True: importing lerobot (at module load) installs a root handler
    # and leaves the root level at WARNING, which would otherwise make this a
    # no-op and silently drop every log_say() status line.
    logging.basicConfig(level=getattr(logging, cfg.log_level), force=True)

    # System setup (Jetson clock pinning, the GStreamer NVENC stack) is handled
    # by the host installer + its boot service, not here — see
    # `axol jetson.setup` / `axol gst.install`. This entry point just runs.

    _run(cfg)


def _run(
    cfg: CollectDataConfig,
    stop_event: "threading.Event | None" = None,
    control: "Control | None" = None,
) -> None:
    from lerobot.processor import make_default_processors
    from lerobot.teleoperators.utils import TeleopEvents
    from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
    from lerobot.utils.feature_utils import (
        hw_to_dataset_features,
    )
    from lerobot.utils.utils import log_say
    from lerobot.utils.visualization_utils import init_rerun

    from ..lerobot.robot.robot_axol import AxolRobot
    from ..lerobot.teleop.teleop_vr import AxolVRTeleop
    from ..vr.models import VRState

    # Default keeps the CLI path unchanged: episode decisions come from the VR
    # controller only. The web panel injects a _QueueCollectControl so the
    # session can also be driven (and followed) from the dashboard with the
    # headset off.
    if control is None:
        control = _NullCollectControl()

    repo_id = cfg.repo_id
    task = cfg.task
    fps = cfg.fps
    teleop_hz = cfg.teleop_hz
    vcodec = cfg.vcodec
    root = cfg.root
    push_to_hub = cfg.push_to_hub
    rerun_ip = cfg.rerun_ip
    rerun_port = cfg.rerun_port

    # Pin the control process to its dedicated cores before any threads are
    # created (the control loop, VR server, and IK dispatch threads inherit it on
    # connect), so background recording work — relay, recorder, NVENC encoders,
    # all pinned to the other cores — can't preempt the 120 Hz loop. Restored in
    # the finally so a long-lived serve process isn't left pinned. No-op where
    # affinity isn't available.
    try:
        _orig_affinity = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        _orig_affinity = None
    affinity.pin_realtime()

    # Finalize the camera set before the relay/robot open the cameras: prune the
    # unassigned placeholder slots (at least one must be set) and flag any
    # physically-stereo ZED X so the relay and the in-process fallback both open
    # it on the stereo grab path. Shared with run-policy via
    # ``prepare_capture_cameras`` so the two commands set cameras up identically.
    if isinstance(cfg.robot_config, AxolRobotConfig):
        from ..zed import stereo_serials

        cfg.robot_config.prepare_capture_cameras(stereo_serials(), minimum=1)
        if not cfg.robot_config.observation_cameras():
            raise ValueError(
                "collect-data has no camera with recording enabled — every "
                "assigned camera is set to stream-only (or recording is turned "
                "off). Enable recording for at least one camera in the Cameras "
                "dialog (or set its record_resolution / eyes)."
            )

    # The teleop's action keys must match the robot's: propagate the SKU's
    # gripper capability so the gripperless SKU records no gripper channels.
    if isinstance(cfg.robot_config, AxolRobotConfig) and isinstance(
        cfg.teleop_config, AxolVRTeleopConfig
    ):
        cfg.teleop_config.has_gripper = cfg.robot_config.axol_config.has_gripper

    robot = AxolRobot(cfg.robot_config)
    teleop = AxolVRTeleop(cfg.teleop_config)

    # Check resume eligibility before connecting (file check only)
    dataset_root = Path(root) if root else HF_LEROBOT_HOME / repo_id
    meta = dataset_root / "meta"
    has_info = (meta / "info.json").exists()
    is_complete = (
        has_info and (meta / "tasks.parquet").exists() and (meta / "episodes").is_dir()
    )
    if has_info and not is_complete:
        raise RuntimeError(
            f"Incomplete dataset found at {dataset_root} (missing tasks.parquet or episodes/). "
            f"Delete the directory and rerun to start fresh:\n"
            f"  rm -rf {dataset_root}"
        )
    if is_complete:
        check_resume_consistency(dataset_root)

    # A resumed dataset's image resolution is fixed by its existing metadata, so
    # the relay must record at it regardless of the configured dataset_resolution
    # — otherwise the downscaled frames mismatch the stored feature shape and
    # LeRobot's validate_frame kills the capture thread mid-episode. A fresh
    # dataset uses the configured resolution.
    dataset_resolution = cfg.dataset_resolution
    if is_complete:
        existing = _existing_dataset_resolution(dataset_root)
        if existing is None:
            # We can't read/map the resumed dataset's image resolution, so we
            # can't guarantee recorded frames match its stored feature shape —
            # recording would fail LeRobot's validate_frame mid-episode. Fail
            # fast with guidance instead of crashing the capture thread later.
            raise ValueError(
                f"Cannot resume the dataset at {dataset_root}: its recorded image "
                "resolution couldn't be read from meta/info.json or doesn't map to a "
                "ZED resolution the recorder produces (SVGA/HD1080/HD1200). Start a "
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

    hostname = socket.gethostname()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as _s:
        _s.connect(("8.8.8.8", 80))
        local_ip = _s.getsockname()[0]
    print("Connect the VR app (https://axol.almond.bot) to this machine:")
    print(f"  Hostname : {hostname}.local")
    print(f"  IP       : {local_ip}")

    if rerun_ip:
        init_rerun(session_name="axol_record", ip=rerun_ip, port=rerun_port)

    # Prefer the out-of-process video relay: it owns the cameras (gst grab +
    # NVENC + WebRTC) in a subprocess and ships raw frames back via shared
    # memory, so the control loops stay as fast as `axol teleop`. Only use it
    # when it exported raw frames for every observation camera; otherwise tear
    # it down and fall back to the in-process camera path (robot owns cameras).
    relay = _start_video_relay(cfg, dataset_resolution)
    expected = set(cfg.robot_config.observation_cameras().keys())
    use_relay = relay is not None and expected <= set(relay.raw_cameras)
    if use_relay:
        robot.set_external_cameras({k: relay.raw_cameras[k] for k in expected})
    elif relay is not None:
        _logger.info(
            "video relay exported raw frames for %s but the dataset needs %s; "
            "using the in-process camera path instead.",
            sorted(relay.raw_cameras),
            sorted(expected),
        )
        relay.shutdown()
        relay = None

    # On the relay's encoded (gstshm-h264) transport the dataset capture loop
    # is paced by camera frame arrival — exactly one encoded frame per dataset
    # row — so rows land at the camera rate no matter what fps was requested,
    # while ``meta/info.json`` is stamped with the requested value. A mismatch
    # therefore records a dataset whose metadata lies about its timing, and
    # every consumer (replay-dataset, training) plays it back at the wrong
    # speed. Fail fast with the rates the relay actually opened the cameras at
    # (they can fall back, e.g. to 30 fps) instead of recording bad data.
    if use_relay:
        mismatched = {
            src: int(m["fps"])
            for src, m in relay.raw_meta.items()
            if src in expected
            and m["transport"] == "gstshm-h264"
            and int(m["fps"]) != fps
        }
        if mismatched:
            relay.shutdown()
            rates = ", ".join(
                f"{src} at {v} fps" for src, v in sorted(mismatched.items())
            )
            raise ValueError(
                f"Recording fps is {fps}, but dataset frames are captured at "
                f"the camera rate ({rates}) — the episode would actually "
                f"record at the camera rate while claiming {fps} fps, so "
                f"replay and training would run at the wrong speed. Set the "
                f"recording fps to the camera rate, or raise the camera fps "
                f"to {fps}."
            )

    # Connect first — cameras auto-detect resolution and FPS on open, which
    # is then used to define the dataset observation features. With the relay
    # the robot's cameras are shared-memory proxies, so this only opens the arms.
    # If any of this setup fails, tear the relay subprocess down so it doesn't
    # leak a held camera (it is daemonic, but a long-lived parent could outlive
    # the failure).
    imu_src: Any | None = None  # board-gyro yaw source for the cart, if wired
    try:
        robot.connect()

        # The dataset lives in the recorder (subprocess or in-process), not here.
        # Its features come from the robot's joint features + the camera image
        # dims; the snapshot schema is the joint-observation keys (no images) +
        # action keys, in a fixed order shared with the recorder's SnapshotReader.
        action_features = hw_to_dataset_features(robot.action_features, ACTION)
        obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
        features = {**action_features, **obs_features}
        obs_keys = list(robot.get_joint_observation().keys())
        action_keys = list(robot.action_features.keys())

        # The VR server accepts headsets during the IK worker's JAX compile
        # inside connect(), before the video registration below runs. Declare
        # video as expected so an early headset request waits for the offer
        # instead of being told there is no video.
        if use_relay or robot.cameras:
            teleop.set_video_expected(True)

        pos_l, pos_r = robot.positions
        teleop.connect(q_start_left=pos_l, q_start_right=pos_r)

        # Stream the overhead + wrist cameras to the headset so the operator can
        # see the scene and grippers. With the relay this is the subprocess's
        # WebRTC manager (out of process); otherwise fall back to the in-process
        # relay, which reuses the frames the robot's cameras already decode.
        if use_relay:
            teleop.set_video_manager(relay)
        else:
            _register_camera_video(robot, teleop)

        # Cart heading hold: feed the carrier board's BMI088 yaw rate to the
        # cart, same as native teleop (see almond_axol.robot.gyro — nothing
        # here touches the video path). Best-effort: on failure the hold is
        # simply inert (no yaw rates arrive), which the cart logs once driving.
        if teleop.cart is not None and teleop.cart.config.imu:
            try:
                from ..robot.gyro import BoardYawRateSource

                imu_src = BoardYawRateSource(teleop.cart.feed_yaw_rate)
                imu_src.open()
            except Exception as exc:  # noqa: BLE001 - heading hold is best-effort
                _logger.warning(
                    "cart.imu: could not start the board gyro (%s); heading "
                    "hold disabled",
                    exc,
                )
    except BaseException:
        # Tear down teleop too: if a stop interrupts teleop.connect() while the
        # IK worker is still compiling JAX, its VR server thread is otherwise
        # left running and keeps holding its WebSocket port, so the next run
        # can't bind it. disconnect() is a no-op if connect() never ran.
        if imu_src is not None:
            imu_src.close()
        try:
            teleop.disconnect()
        except Exception:
            _logger.exception("teleop cleanup after failed setup failed")
        if relay is not None:
            relay.shutdown()
        raise

    teleop_action_proc, robot_action_proc, robot_obs_proc = make_default_processors()

    # The dataset capture + encode runs OUT of the control process so its
    # per-frame numpy / add_frame / save_episode work never shares the GIL with
    # the 120 Hz control loop. With the relay up, a recorder subprocess attaches
    # its own readers to the relay's shared-memory frames and owns the dataset;
    # without a relay (no gst stack) we fall back to capturing in-process.
    recorder_config = {
        "repo_id": repo_id,
        "root": root,
        "dataset_root": str(dataset_root),
        "is_complete": is_complete,
        "features": features,
        "robot_type": robot.name,
        "fps": fps,
        "vcodec": vcodec,
        "rerun_ip": rerun_ip,
        "rerun_port": rerun_port,
        "push_to_hub": push_to_hub,
        "log_level": cfg.log_level,
    }
    try:
        if is_complete:
            log_say(f"Resuming existing dataset at {dataset_root}.")
        if use_relay:
            recorder: DatasetRecorderProcess | InProcessRecorder = (
                DatasetRecorderProcess(
                    raw_cond=relay.raw_cond,
                    raw_meta=relay.raw_meta,
                    obs_keys=obs_keys,
                    action_keys=action_keys,
                    config=recorder_config,
                )
            )
        else:
            recorder = InProcessRecorder(recorder_config, robot, robot_obs_proc)
    except BaseException:
        if relay is not None:
            relay.shutdown()
        raise

    # Background perf samplers (per-second /proc CPU + memory breakdown and
    # Jetson GPU/EMC/NVENC/thermal), started unconditionally as in `axol teleop`
    # so the two flows' lines line up and a collection session can be compared
    # against the teleop baseline. Both keep their heavy system-wide tiers at
    # DEBUG, so the default INFO output stays a couple of lines per second.
    # Labels map the known subprocesses (mp spawn children all report
    # comm=python) so the IK solver, video relay, and dataset recorder are
    # legible in the breakdown.
    diag_labels: dict[int, str] = {os.getpid(): "main"}
    ik_proc = getattr(teleop, "_ik_process", None)
    if ik_proc is not None and getattr(ik_proc, "pid", None):
        diag_labels[ik_proc.pid] = "ik"
    if relay is not None and getattr(relay, "_proc", None) is not None:
        relay_pid = getattr(relay._proc, "pid", None)
        if relay_pid:
            diag_labels[relay_pid] = "relay"
    if getattr(recorder, "pid", None):
        diag_labels[recorder.pid] = "recorder"  # type: ignore[union-attr]
    diag = SystemDiag(diag_labels, _logger)
    diag.start()
    tegra = TegraStatsDiag(_logger)  # no-op off-Tegra
    tegra.start()

    # Keep the relay's raw dataset branch closed until an episode records: the
    # raw VIC convert + shared-memory copy for every camera is the bulk of the
    # relay's CPU (~2 cores), and nothing reads raw frames during the pre-record
    # teleop phase. Closing it there makes that phase as light as `axol teleop`.
    if relay is not None:
        relay.set_raw_enabled(False)

    episodes_recorded = 0
    episode_idx = recorder.episode_count()
    teleop_interval = 1.0 / teleop_hz

    # Rolling control-loop rate readout, mirroring `axol teleop`: the loop rate
    # is measured here; vr/ik come from the teleop's ~2s windows. Logged once a
    # second at INFO so collection-time perf can be compared against teleop.
    loop_times: list[float] = []
    last_rate_log = time.perf_counter()
    # Also break the per-step body into its sections so we can see where the
    # time goes (joint read, action read, robot send).
    time_sections = True
    # `proc` isolates the LeRobot action processors (which native teleop does
    # not run) from `send`, so `send` is now the pure CAN motion_control
    # round-trip — directly comparable to teleop's `send`.
    sect = {"obs": 0.0, "act": 0.0, "proc": 0.0, "send": 0.0}

    # Set when the on-loop coroutines must unwind (Ctrl+C): the hot loop now
    # runs on the robot's event loop, so a KeyboardInterrupt on the main thread
    # can't break it directly — it has to exit via this flag before teardown.
    loop_stop = threading.Event()

    def _stopped() -> bool:
        return (stop_event is not None and stop_event.is_set()) or loop_stop.is_set()

    # Worst single-iteration stall and scheduler slip within each window. `gap`
    # is the longest time between consecutive loop iterations (a starved control
    # thread shows up as gaps >> the 1/teleop_hz period); `slip` is how late the
    # loop woke past its absolute deadline. Both isolate "the thread lost the
    # CPU" from "the CAN call itself was slow" — jerk tracks the former.
    max_gap = {"v": 0.0}
    max_slip = {"v": 0.0}
    prev_t0 = {"v": 0.0}

    def _maybe_log_rate(t0: float) -> None:
        nonlocal last_rate_log, sect
        loop_times.append(t0)
        if prev_t0["v"]:
            gap = t0 - prev_t0["v"]
            if gap > max_gap["v"]:
                max_gap["v"] = gap
        prev_t0["v"] = t0
        if t0 - last_rate_log < 1.0 or len(loop_times) <= 1:
            return
        span = loop_times[-1] - loop_times[0]
        n = len(loop_times)
        loop_hz = (n - 1) / span if span > 0 else 0.0
        _logger.info(
            "loop: %.1f Hz  vr: %.1f Hz  ik: %.1f Hz",
            loop_hz,
            teleop.vr_hz(),
            teleop.ik_hz(),
        )
        # Jitter detail (maxgap/maxslip = "the thread lost the CPU") and the
        # per-section breakdown stay at DEBUG so INFO is just the rate line.
        if time_sections:
            _logger.debug(
                "loop maxgap=%.1fms maxslip=%.1fms  sections (mean ms): "
                "obs=%.2f act=%.2f proc=%.2f send=%.2f",
                1e3 * max_gap["v"],
                1e3 * max_slip["v"],
                1e3 * sect["obs"] / n,
                1e3 * sect["act"] / n,
                1e3 * sect["proc"] / n,
                1e3 * sect["send"] / n,
            )
            sect = {"obs": 0.0, "act": 0.0, "proc": 0.0, "send": 0.0}
        loop_times.clear()
        max_gap["v"] = 0.0
        max_slip["v"] = 0.0
        last_rate_log = t0

    # The hot control loop runs *on the robot's event loop* (see
    # AxolRobot.event_loop) so motion_control is awaited inline — cooperatively
    # interleaved with CAN telemetry on one thread, exactly like `axol teleop`.
    # The main thread drives the episode lifecycle (dataset writes, rest-pose
    # moves) and blocks on each coroutine until the episode (or reset) finishes.
    async def _episode_loop() -> tuple[bool, bool]:
        recording = False
        rerecord = False
        # perf_counter deadline of a panel-started record countdown, or None.
        pending_start: float | None = None

        # Absolute-deadline pacing (mirrors `axol teleop`): late wakeups are
        # corrected on the next cycle instead of stretching the command interval.
        # Regular command timing matters because motion_control derives its
        # velocity feedforward by differentiating commanded positions, so a
        # jittery interval shows up as jerk.
        deadline = time.perf_counter()
        while not _stopped():
            # A reset playing outside a recording — the startup move at
            # session start, or a reset press during the pre-record phase —
            # runs through the same guarded engine as the post-episode
            # return, so *every* rest move yields on contact. (During a
            # recording the trajectory streams through the normal path: the
            # discard flow consumes its reset press, and a headset-exit
            # reset mid-take is deliberately left as-is.)
            if not recording and teleop.is_resetting:
                await _guarded_return()
                # A contact hold during that move left the panel on the
                # "contact" phase, and nothing else re-announces this phase
                # until the next episode — the outer loop only runs
                # note_ready before the episode starts.
                control.note_ready(episode_idx + 1)
                deadline = time.perf_counter()
                prev_t0["v"] = 0.0
                continue
            deadline += teleop_interval
            t0 = time.perf_counter()
            _maybe_log_rate(t0)

            # Camera reads happen on the capture thread; the control loop only
            # ever touches joint state.
            joint_obs = robot.get_joint_observation()
            t_obs = time.perf_counter()
            teleop.send_feedback(joint_obs)
            act = teleop.get_action()
            t_act = time.perf_counter()
            act_processed = teleop_action_proc((act, joint_obs))
            robot_act = robot_action_proc((act_processed, joint_obs))
            t_proc = time.perf_counter()
            await robot.send_action_async(robot_act)
            t_send = time.perf_counter()
            sect["obs"] += t_obs - t0
            sect["act"] += t_act - t_obs
            sect["proc"] += t_proc - t_act
            sect["send"] += t_send - t_proc

            # Record the action in the configured action space: identity for
            # joint datasets, FK-to-Cartesian when observe_cartesian is set. The
            # arm is still commanded with the teleop joint targets above, so its
            # motion is unchanged — only the stored representation differs.
            recorder.publish(joint_obs, robot.action_to_dataset(act_processed), t0)

            events = teleop.get_teleop_events()
            panel_cmd = control.poll_command()

            # Panel-driven start (headset off): a Start click opens a short
            # countdown — mirroring the in-headset one, so the operator can
            # pick the controllers back up — and a second click cancels it.
            # The VR record button stays immediate (the headset app already
            # ran its own countdown before flipping to RECORDING).
            if not recording and panel_cmd == "start":
                if pending_start is None:
                    pending_start = time.perf_counter() + _PANEL_START_COUNTDOWN_S
                    control.note_countdown(pending_start)
                    log_say(
                        f"Recording starts in {_PANEL_START_COUNTDOWN_S:.0f} seconds."
                    )
                else:
                    pending_start = None
                    control.note_ready(episode_idx + 1)
                    log_say("Recording start cancelled.")
            start_requested = events.get("start_recording") or (
                pending_start is not None and time.perf_counter() >= pending_start
            )

            if start_requested and not recording:
                pending_start = None
                recording = True
                # Open the relay's raw branch so the recorder has frames, then
                # tell the recorder to start an episode.
                if relay is not None:
                    relay.set_raw_enabled(True)
                recorder.start_episode(task)
                control.note_recording()
                log_say("Recording started.")

            # The episode outcome comes from the VR record button (terminate →
            # save, reset+record → discard) or the panel's Save / Discard
            # buttons — whichever arrives first. A panel press has to move the
            # headset itself, since its own state machine never saw the press:
            # SAVING blocks its controls until the write completes, while a
            # discard writes nothing and goes straight back to DATA_COLLECTION
            # — the states the equivalent A→A / X→X presses leave it in.
            if recording and panel_cmd in ("s", "r"):
                rerecord = panel_cmd == "r"
                teleop.send_feedback_state(
                    VRState.DATA_COLLECTION if rerecord else VRState.SAVING
                )
                break
            if events[TeleopEvents.TERMINATE_EPISODE]:
                teleop.send_feedback_state(VRState.SAVING)
                break
            if events[TeleopEvents.RERECORD_EPISODE]:
                rerecord = True
                break

            await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
            slip = time.perf_counter() - deadline
            if slip > max_slip["v"]:
                max_slip["v"] = slip

        return recording, rerecord

    # Guarded return-to-rest: the sequencing (torque watchdog, gravity-comp
    # fallback, reset-press retry) lives in the shared engine
    # (VRTeleopCore.guarded_return — the same one native `axol teleop` runs),
    # bound here to this flow's robot, processors, and headset states.
    vrt_cfg = cfg.teleop_config.vr_teleop_config

    async def _guard_send_step() -> None:
        joint_obs = robot.get_joint_observation()
        act = teleop.get_action()
        await robot.send_action_async(robot_action_proc((act, joint_obs)))

    async def _guard_gravity_step() -> None:
        await robot.gravity_compensate_async(kd=vrt_cfg.reset_gravity_comp_kd)

    def _guard_on_contact() -> None:
        # The headset may still be in SAVING (which blocks the reset button)
        # when contact trips, so unblock it. The panel needs telling too: it
        # would otherwise keep showing the phase the move started from
        # ("Saving…", which has no buttons) for as long as the hold lasts.
        teleop.send_feedback_state(VRState.DATA_COLLECTION)
        control.note_contact()

    def _guard_hold_tick() -> None:
        # Recording can't start while the arms are limp: answer a start press
        # (VR record button or the panel's Start) by snapping the headset back
        # to DATA_COLLECTION. Draining the panel queue here also keeps a click
        # from leaking into the next episode's countdown.
        events = teleop.get_teleop_events()
        panel_cmd = control.poll_command()
        if panel_cmd == "continue":
            # The hold only ends on a latched reset, so a headset-off session
            # (or one whose operator has the headset off their face) would
            # otherwise have no way out of it but Stop.
            teleop.request_reset()
        elif events.get("start_recording") or panel_cmd == "start":
            teleop.send_feedback_state(VRState.DATA_COLLECTION)
            log_say("Press reset to return to rest before recording.")
        if teleop.reset_pending:
            # Latched, from either input — the hold exits on the next cycle
            # and replans, so stop offering the panel a button for it.
            control.note_returning()

    async def _guarded_return() -> None:
        """Play the pending reset guarded; await on the robot's event loop."""
        await teleop.guarded_return(
            send_step=_guard_send_step,
            gravity_step=_guard_gravity_step,
            torque_residuals=robot.torque_residuals,
            reset_command_state=robot.reset_command_state,
            get_positions=lambda: robot.positions,
            stopped=_stopped,
            announce=log_say,
            on_contact=_guard_on_contact,
            hold_tick=_guard_hold_tick,
            vr_alive=teleop.vr_alive,
        )

    async def _return_home_loop() -> None:
        """Post-episode return: request the reset, then play it guarded."""
        log_say("Returning to rest pose.")
        teleop.request_reset()
        await _guarded_return()

    def _run_on_robot_loop(coro: Any) -> Any:
        """Run ``coro`` on the robot's event loop and block until it returns.

        On Ctrl+C, signal the coroutine to unwind and wait for it to finish so
        it stops commanding the robot before teardown, then re-raise.
        """
        fut = asyncio.run_coroutine_threadsafe(coro, robot.event_loop)
        try:
            return fut.result()
        except KeyboardInterrupt:
            loop_stop.set()
            try:
                fut.result(timeout=5.0)
            except BaseException:
                fut.cancel()
            raise

    def _wrap_up_episode(recording: bool, rerecord: bool) -> None:
        """Save or discard the just-ended episode and announce the result."""
        nonlocal episodes_recorded
        if rerecord:
            log_say("Re-recording episode.")
            if recording:
                recorder.cancel_episode()
        elif recording:
            log_say("Saving episode…")
            recorder.save_episode()
            # The serve unit records as root into the operator's home; hand the
            # tree back after every save so a crash never leaves a root-owned
            # dataset behind (no-op off the root service).
            restore_dataset_ownership(dataset_root)
            episodes_recorded += 1
            control.note_saved()
            log_say(
                f"Saved episode {recorder.episode_count()} "
                f"({episodes_recorded} this session)."
            )
        else:
            log_say("Episode ended before recording started, skipping.")

    try:
        while not _stopped():
            episode_idx = recorder.episode_count()
            # Surface the (1-based) episode number in the headset HUD so the
            # operator can see which episode they're about to record. The panel
            # gets the same readout (plus the Start button) through the
            # control's snapshot.
            teleop.send_feedback_episode(episode_idx + 1)
            control.note_ready(episode_idx + 1)
            log_say(
                f"Episode {episode_idx + 1}: robot is at rest pose. Press record on the VR controller when ready."
            )

            recording, rerecord = _run_on_robot_loop(_episode_loop())

            # Recording done — close the raw branch so the rest-pose/reset and
            # next pre-record phase stay light. (The recorder stops its own
            # capture loop on save/cancel, below.)
            if relay is not None:
                relay.set_raw_enabled(False)

            if _stopped():
                if recording:
                    recorder.cancel_episode()
                break

            if recording and not rerecord:
                # Mirror the headset's SAVING state in the panel for the whole
                # rest-pose + save stretch (recording controls are blocked).
                control.note_saving()
            else:
                # Nothing to write, so the return to rest is all that's left.
                control.note_returning()

            # Return home right away, but *guarded*: the move bails into a
            # limp gravity-comp hold on contact (see _return_home_loop), so
            # a gripper still hooked on the scene means a brief tug — not a
            # sustained yank. The episode is saved/discarded on this thread
            # in parallel; on a save the headset stays in SAVING (controls
            # blocked) until the write completes.
            home_future = asyncio.run_coroutine_threadsafe(
                _return_home_loop(), robot.event_loop
            )
            try:
                _wrap_up_episode(recording, rerecord)
                home_future.result()
            except BaseException:
                # Ctrl+C or a failed save: unwind the guarded return so it
                # stops commanding the robot before teardown.
                loop_stop.set()
                try:
                    home_future.result(timeout=5.0)
                except BaseException:
                    home_future.cancel()
                raise
            # Drain VR events fired during the return, then unblock the
            # headset for the next take.
            teleop.get_teleop_events()
            teleop.send_feedback_state(VRState.DATA_COLLECTION)

    except KeyboardInterrupt:
        pass
    except Exception:
        teleop.send_feedback_error()
        raise
    finally:
        log_say("Stopping.")

        diag.stop()
        tegra.stop()

        if imu_src is not None:
            imu_src.close()
        robot.disconnect()
        teleop.disconnect()
        # Recorder owns the dataset: finalize, optional push, and empty-dataset
        # cleanup all happen in recorder.close().
        recorder.close()
        # Finalize wrote the last meta/stats files as root; adopt them too.
        restore_dataset_ownership(dataset_root)
        if relay is not None:
            relay.shutdown()

        # Restore the process's original CPU affinity (a serve process is
        # long-lived and runs other operations after this one).
        if _orig_affinity is not None:
            try:
                os.sched_setaffinity(0, _orig_affinity)
            except OSError:
                pass
