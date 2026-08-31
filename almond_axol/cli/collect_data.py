"""
axol collect-data

Record teleoperation episodes with the Axol robot and its local ZED cameras.
Episode boundaries are driven by VR controller commands:
  - DATA_COLLECTION → RECORDING:              start collecting frames
  - RECORDING → DATA_COLLECTION:              stop; save episode (success)
  - RECORDING → DATA_COLLECTION + failure:    stop; save episode (failure)
  - RECORDING → DATA_COLLECTION + reset btn:  stop; discard episode (rerecord)

and/or, when launched from the web control panel (``axol serve``), by episode
commands pushed through :class:`_QueueCollectControl` (``POST
/api/op/episode``): ``start`` begins recording after a spoken 3-second
countdown, ``s`` terminates + saves, ``r`` discards + re-records, ``q`` quits
the session. Both sources are live at once — either can start or end an
episode.

While saving, the VR headset is pushed into the SAVING state so recording
controls are blocked until save_episode() completes.

At save time each episode passes a data-quality gate (see
:func:`evaluate_episode_qa`): episodes with a mid-recording re-engage, too
many frozen-TCP/disengaged/untracked frames, or a stale Mantis trigger
heartbeat are discarded and re-recorded instead of silently saved
(``--qa_gate false`` disables the refusal; the QA summary is always logged).

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
from ..robot.control import ContactWatchdog
from ..utils import affinity
from ..utils.jetson_diag import TegraStatsDiag
from ..utils.proc_diag import SystemDiag
from .config import DatasetResolution, LogLevel, MantisSource, parse

if TYPE_CHECKING:
    from ..lerobot.robot.robot_axol import AxolRobot

_logger = logging.getLogger(__name__)


def _apply_mantis_profile(cfg: "CollectDataConfig") -> None:
    """Rewrite the parsed config for Mantis collection (``--mantis``).

    Robot side: swap a plain :class:`AxolRobotConfig` for
    :class:`MantisRobotConfig` (handheld grippers on ``can_mantis_l/r``, virtual
    arms), carrying over every field the operator may have set and only
    replacing CAN channels still at the robot-arm defaults. Teleop side: force
    the Mantis mapping/faithfulness profile —

    - ``absolute_mode``: controllers map to world-anchored absolute targets
      (the engage squeeze is the start-pose alignment act);
    - ``ik_alpha = 1.0`` and effectively-unbounded trapezoid limits: those
      filters exist to protect a physical arm and only add lag between the
      recorded action and where the hand actually was — with no arm to
      protect, the recorded joints should follow the raw IK solution.

    Applied after parsing, so it overrides these specific teleop fields even if
    set on the CLI (a warning is logged); other teleop knobs (One Euro, rest
    poses, frequency) pass through untouched.
    """
    from dataclasses import fields

    from ..constants import CAN_LEFT, CAN_MANTIS_LEFT, CAN_MANTIS_RIGHT, CAN_RIGHT
    from ..lerobot.robot.config_mantis import MantisRobotConfig

    if isinstance(cfg.robot_config, AxolRobotConfig) and not isinstance(
        cfg.robot_config, MantisRobotConfig
    ):
        kwargs = {
            f.name: getattr(cfg.robot_config, f.name)
            for f in fields(cfg.robot_config)
            if f.init
        }
        if kwargs.get("left_channel") == CAN_LEFT:
            kwargs["left_channel"] = CAN_MANTIS_LEFT
        if kwargs.get("right_channel") == CAN_RIGHT:
            kwargs["right_channel"] = CAN_MANTIS_RIGHT
        # The rig has no overhead camera; drop the placeholder slot so the
        # camera dialog / CLI surface matches the hardware (an explicitly
        # assigned serial is kept — someone may rig a scene camera).
        cameras = dict(kwargs.get("cameras") or {})
        overhead = cameras.get("overhead")
        if overhead is not None and int(getattr(overhead, "serial", 0) or 0) <= 0:
            cameras.pop("overhead")
        kwargs["cameras"] = cameras
        cfg.robot_config = MantisRobotConfig(**kwargs)

    # Mantis datasets are Cartesian: state/action are absolute base-frame EE
    # poses (+ gripper), the same schema on-robot collection produces with
    # ``observe_cartesian`` — so rig and robot episodes mix in one dataset,
    # train with ``axol mantis.train``, and deploy with ``run-policy``. The
    # rig's virtual joints are IK artifacts and are not recorded.
    if not getattr(cfg.robot_config, "observe_cartesian", False):
        _logger.info("--mantis: forcing observe_cartesian (EE-pose dataset schema).")
    cfg.robot_config.observe_cartesian = True

    if isinstance(cfg.teleop_config, AxolVRTeleopConfig):
        from ..kinematics.config import apply_mantis_kinematics_profile
        from ..teleop.config import apply_mantis_teleop_profile

        tc = cfg.teleop_config.vr_teleop_config
        if not tc.absolute_mode or tc.hold_to_engage or tc.ik_alpha != 1.0:
            _logger.info(
                "--mantis: forcing absolute_mode, hold_to_engage=false, "
                "ik_alpha=1.0, and transparent trapezoid limits on the "
                "teleop config."
            )
        apply_mantis_teleop_profile(
            tc,
            tracker_source=cfg.mantis_source,
        )
        cfg.teleop_config.vr_server_config.pose_source_kind = (
            "webxr" if cfg.mantis_source == "quest" else "tracker"
        )
        apply_mantis_kinematics_profile(cfg.teleop_config.kinematics_config)


def _prepare_mantis_collection(cfg: "CollectDataConfig") -> None:
    """Apply Mantis defaults and fail deterministic calibration preflight."""
    _apply_mantis_profile(cfg)
    _validate_mantis_calibration(cfg)


def _validate_mantis_calibration(cfg: "CollectDataConfig") -> None:
    """Reject transforms whose tracker identity/datum is not production-safe."""
    from ..mantis.calibration import (
        MANTIS_TCP_TRANSFORM_FILE,
        design_transform_for,
        load_tcp_transforms,
        parse_quest_tracker_key,
        tracker_key_for_side,
        validate_tcp_transform,
    )

    vrt = cfg.teleop_config.vr_teleop_config
    # Do not trust presence alone here: this is the last deterministic gate
    # before hardware and recording start, and callers/tests may invoke it
    # without first applying the normal Mantis profile. In particular,
    # explicit Advanced/CLI values never pass through the JSON file loader.
    for side in ("left", "right"):
        attr = f"tcp_transform_{side}"
        transform = getattr(vrt, attr)
        if transform is None:
            continue
        try:
            setattr(vrt, attr, validate_tcp_transform(transform))
        except ValueError as exc:
            raise ValueError(
                f"Mantis {side} tracker→gripper TCP transform is invalid: {exc}"
            ) from exc

    # The bring-up escape hatch permits *missing* transforms, not malformed
    # transforms that could crash the worker or create invalid rotations.
    if cfg.mantis_allow_uncalibrated:
        return

    def quest_error() -> ValueError:
        return ValueError(
            "Mantis Quest production collection requires measured left and "
            "right transforms under one profile-scoped "
            "`quest:<WebXR-profile>:grip` key. Bare `quest`, target-ray, "
            "missing, or conflicting datum metadata is unsafe because the "
            "controller-local frame differs. Add the constants to "
            f"{MANTIS_TCP_TRANSFORM_FILE}; the live WebXR profile is "
            "reported by the updated Quest client. Use "
            "--mantis_allow_uncalibrated true only for a bring-up capture "
            "that will not be used for training."
        )

    if cfg.mantis_source == "quest":
        datum = parse_quest_tracker_key(vrt.tracker_key or "")
        datum_ready = (
            datum is not None
            and datum[1] == "grip"
            and vrt.quest_controller_profile == datum[0]
            and vrt.quest_pose_space == datum[1]
        )
        if not datum_ready:
            raise quest_error()

    # A valid seven-float value is not sufficient provenance for production:
    # an Advanced/CLI value can survive a source or tracker swap. Re-resolve
    # the authoritative measured/factory value for each active tracker and
    # require the final value sent to IK to match it. This also keeps the
    # server's readiness gate and the last CLI preflight equivalent.
    saved = load_tcp_transforms()
    missing: list[tuple[str, str]] = []
    mismatched: list[tuple[str, str]] = []
    for side in ("left", "right"):
        if cfg.mantis_source == "quest":
            key = vrt.tracker_key or "quest"
        else:
            # Hardware trackers are independently bound. Do not reuse the
            # singular config.tracker_key override for both sides here; doing
            # so would let one device identity authorize the other rig.
            key, _ = tracker_key_for_side(side, source=cfg.mantis_source)

        entries = saved.get(side, {})
        # Saved hardware measurements must name the bound device. A bare
        # family key has unknown provenance; only a factory design constant is
        # allowed to cover a whole hardware family.
        device_scoped = cfg.mantis_source == "quest" or bool(key.partition(":")[2])
        authoritative = entries.get(key) if device_scoped else None
        if authoritative is None:
            authoritative = design_transform_for(side, key)
        if authoritative is None:
            missing.append((side, key))
            continue
        authoritative = validate_tcp_transform(authoritative)
        actual = getattr(vrt, f"tcp_transform_{side}")
        if actual is None:
            missing.append((side, key))
            continue
        position_matches = all(
            math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
            for a, b in zip(actual[:3], authoritative[:3], strict=True)
        )
        # Unit quaternions q and -q encode the same rotation.
        quat_dot = sum(
            a * b for a, b in zip(actual[3:], authoritative[3:], strict=True)
        )
        orientation_matches = math.isclose(
            abs(quat_dot), 1.0, rel_tol=1e-9, abs_tol=1e-9
        )
        if not position_matches or not orientation_matches:
            mismatched.append((side, key))

    if cfg.mantis_source == "quest" and (missing or mismatched):
        raise quest_error()
    if missing:
        details = ", ".join(f"{side} ({key})" for side, key in missing)
        raise ValueError(
            f"Mantis {cfg.mantis_source} collection has no verified "
            f"tracker→gripper TCP transform for {details}. Add measured "
            "constants keyed to each bound tracker in "
            f"{MANTIS_TCP_TRANSFORM_FILE}, or use "
            "--mantis_allow_uncalibrated true only for a calibration/bring-up "
            "capture that will not be used for training."
        )
    if mismatched:
        details = ", ".join(f"{side} ({key})" for side, key in mismatched)
        raise ValueError(
            "Mantis production collection refused an unproven transform "
            f"override for {details}. Remove the stale Advanced/CLI value or "
            "make it exactly match the active tracker entry in "
            f"{MANTIS_TCP_TRANSFORM_FILE}."
        )


def _prepare_recording_cameras(cfg: "CollectDataConfig") -> None:
    """Prune placeholders and require a real dataset camera, without I/O."""
    if not isinstance(cfg.robot_config, AxolRobotConfig):
        return
    cfg.robot_config.select_assigned_cameras(minimum=1)
    if not cfg.robot_config.observation_cameras():
        raise ValueError(
            "collect-data has no camera with recording enabled — every "
            "assigned camera is set to stream-only (or recording is turned "
            "off). Enable recording for at least one camera in the Cameras "
            "dialog (or set its record_resolution / eyes)."
        )


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

    ``--mantis true`` records with the Mantis instead of the robot:
    grippers on ``can_mantis_l/r``, wrist cameras only, absolute
    (world-anchored) pose mapping, dataset rows stamped with the VR pose
    capture time, and the Cartesian EE-pose schema (state/action are absolute
    base-frame poses from the tracker, same schema as on-robot
    ``observe_cartesian`` collection). See :func:`_apply_mantis_profile`.
    """

    repo_id: str
    task: str
    robot_config: RobotConfig = field(default_factory=_default_robot_config)
    teleop_config: TeleoperatorConfig = field(default_factory=AxolVRTeleopConfig)
    # Record with the Mantis handheld rig instead of the robot: its grippers
    # on can_mantis_l/r, wrist cameras only, absolute pose mapping, and the
    # Cartesian EE-pose dataset schema. The Axol arms are not involved.
    mantis: bool = False
    mantis_source: MantisSource = "lighthouse"
    """Pose source for Mantis mode. Direct Mantis runs inherit the host's
    Settings → Mantis choice when saved; otherwise Lighthouse is the default.
    A config file or explicit CLI value wins. Quest connects through WebXR;
    Lighthouse and Ultimate start the corresponding local tracker bridge."""
    mantis_allow_uncalibrated: bool = False
    """Allow a Mantis dataset when either tracker→TCP transform is missing.
    Intended only for bring-up/calibration captures: the resulting Cartesian
    TCP rows are mount-dependent and must not be mixed into training data."""
    # Mantis only: zero-phase low-pass cutoff (Hz) applied to the recorded EE
    # pose track at episode save, removing broadband tracker noise without lag
    # (intentional hand motion lives below ~10 Hz). 0 disables. Ignored for
    # on-robot collection, whose FK poses come from joint encoders.
    mantis_smooth_hz: float = 10.0
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
    # Every return-to-rest is guarded — and, opted in, the tracking phase
    # too: a torque watchdog drops the arms into a limp gravity-comp hold on
    # unexpected contact (reset replans from wherever they were left; a
    # recording episode is discarded). The knobs live on the shared teleop
    # config — ``--teleop_config.vr_teleop_config.reset_torque_threshold``
    # and ``.teleop_torque_threshold`` (the tracking watchdog; 0 disables
    # either, the tracking one defaults off) and ``.reset_gravity_comp_kd``
    # — the same fields `axol teleop` uses, so the two flows behave
    # identically.
    root: str | None = None
    push_to_hub: bool = False
    # Refuse to save episodes the quality gate marks corrupt — a mid-recording
    # re-engage (frame shift), over five percent frozen-TCP frames, or over one
    # percent disengaged
    # frames (see evaluate_episode_qa): the episode is discarded and
    # re-recorded with a loud spoken/logged explanation. Set false as an
    # escape hatch (debugging the gate, or deliberately recording unusual
    # sessions) — the per-episode QA summary is still logged, the bad episode
    # is just saved anyway. Only Mantis episodes can fail the gate; on-robot
    # episodes read poses from joint encoders and always pass.
    qa_gate: bool = True
    rerun_ip: str | None = None
    rerun_port: int = 9876
    log_level: LogLevel = "INFO"


# ----------------------------------------------------------------------
# Episode quality gate
# ----------------------------------------------------------------------

# Maximum age of the capture represented by the current Mantis action. The
# timestamp advances on equal-valued live poses without an unnecessary IK
# solve, but stops on a transport or changing-pose IK stall. Pose *stillness*
# is valid data and must never be mistaken for a stale source.
_QA_STALE_POSE_S = 0.25
# Verdict thresholds, as fractions of recorded control ticks (ticks ≈ dataset
# rows — the loop publishes one snapshot per tick). Exceeding either marks the
# episode bad; a mid-recording re-engage is always fatal (frame shift).
_QA_MAX_STALE_FRACTION = 0.05
_QA_MAX_DISENGAGED_FRACTION = 0.01
_QA_MAX_UNTRACKED_FRACTION = 0.01


@dataclass
class EpisodeQAStats:
    """Per-episode data-quality counters, collected while recording.

    Populated by ``_episode_loop`` in Mantis mode, one increment per control
    tick while an episode is recording. On-robot collection leaves everything
    at zero — its poses come from joint encoders, so the tracker failure
    modes measured here can't occur and the episode always passes the gate.
    """

    # Control ticks recorded (denominator for the fractions below).
    total_frames: int = 0
    # Ticks where the pose behind the latest action was older than
    # _QA_STALE_POSE_S: tracker transport or IK output stalled. A live hand
    # holding perfectly still continues to advance its equal-pose heartbeat.
    stale_frames: int = 0
    # Ticks recorded while teleop tracking was disengaged: the recorded pose
    # holds still while the operator's hands actually move.
    disengaged_frames: int = 0
    # Ticks where either side was held because its optical/SLAM tracking was
    # invalid. Counting sides explicitly catches the old failure mode where
    # one frozen hand passed the combined TCP-motion heuristic while the other
    # hand kept moving.
    untracked_frames: int = 0
    # Ticks where either managed Mantis trigger lacked its 100 Hz CAN
    # heartbeat. The bridge already waits 250 ms before declaring the input
    # stale, so even one such tick represents a meaningful grip-input dropout
    # and makes the recorded command stream untrustworthy.
    trigger_loss_frames: int = 0
    # Tracking re-engaged (False -> True) while recording. The engage squeeze
    # re-fits the world->base transform, silently shifting the frame of every
    # later row — the episode mixes two incompatible coordinate frames and is
    # unusable regardless of the other counters.
    reengaged_while_recording: bool = False
    # Worst pose-stream age (tick time minus pose capture time) seen while
    # recording, in seconds. Reported in the QA summary; not itself a gate.
    max_pose_lag_s: float = 0.0

    @property
    def stale_fraction(self) -> float:
        """Stale ticks as a fraction of recorded ticks (0.0 when empty)."""
        return self.stale_frames / self.total_frames if self.total_frames else 0.0

    @property
    def disengaged_fraction(self) -> float:
        """Disengaged ticks as a fraction of recorded ticks (0.0 when empty)."""
        return self.disengaged_frames / self.total_frames if self.total_frames else 0.0

    @property
    def untracked_fraction(self) -> float:
        """Invalid-per-side ticks as a fraction of recorded ticks."""
        return self.untracked_frames / self.total_frames if self.total_frames else 0.0


def evaluate_episode_qa(stats: EpisodeQAStats) -> tuple[bool, list[str]]:
    """Quality verdict for a recorded episode: ``(ok, reasons)``.

    ``ok`` is ``False`` when the episode is corrupt beyond use:

    - tracking re-engaged mid-recording (world->base re-fit shifts the frame
      of every later row — always fatal), or
    - stale-frame fraction above :data:`_QA_MAX_STALE_FRACTION` (5%), or
    - disengaged-frame fraction above :data:`_QA_MAX_DISENGAGED_FRACTION` (1%).
    - any managed-trigger liveness loss (the bridge only marks this after the
      CAN heartbeat has already been stale for its dropout threshold).

    ``reasons`` lists the human-readable failures (empty when ``ok``). A pure
    function of the counters so the verdict logic is unit-testable without
    hardware; the caller decides what the verdict means (save vs. discard —
    see ``CollectDataConfig.qa_gate``).
    """
    reasons: list[str] = []
    if stats.reengaged_while_recording:
        reasons.append(
            "tracking re-engaged mid-recording — the world-to-base transform "
            "was re-fit, so later rows are in a different frame than earlier "
            "ones"
        )
    if stats.stale_fraction > _QA_MAX_STALE_FRACTION:
        reasons.append(
            f"{100 * stats.stale_fraction:.1f}% of frames had a stale pose "
            f"heartbeat (limit {100 * _QA_MAX_STALE_FRACTION:.0f}%; tracker "
            "transport or IK stall)"
        )
    if stats.disengaged_fraction > _QA_MAX_DISENGAGED_FRACTION:
        reasons.append(
            f"{100 * stats.disengaged_fraction:.1f}% of frames were recorded "
            f"while tracking was disengaged (limit "
            f"{100 * _QA_MAX_DISENGAGED_FRACTION:.0f}%)"
        )
    if stats.untracked_fraction > _QA_MAX_UNTRACKED_FRACTION:
        reasons.append(
            f"{100 * stats.untracked_fraction:.1f}% of frames had a lost "
            f"left or right tracker (limit {100 * _QA_MAX_UNTRACKED_FRACTION:.0f}%)"
        )
    if stats.trigger_loss_frames:
        reasons.append(
            f"{stats.trigger_loss_frames} frames had a stale left or right "
            "Mantis trigger heartbeat; grip commands were held during the dropout"
        )
    return (not reasons, reasons)


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

    def note_ready(self, episode: int, message: str | None = None) -> None:
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

    def note_ready(self, episode: int, message: str | None = None) -> None:
        with self._lock:
            self._episode = episode
        self._set(
            "ready",
            message
            or f"Episode {episode}: press record on the VR controller, or "
            "start recording here.",
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
            "Contact — torque exceeded the threshold, so the arms are limp "
            "and free to move. Clear them, then press reset on the controller "
            "(or return to rest here) to replan from where they are.",
        )

    def note_returning(self) -> None:
        self._set("resetting", "Returning to rest…")


Control = _NullCollectControl | _QueueCollectControl


def main(argv: list[str]) -> None:
    """Parse the CLI config and run a data-collection session."""
    cfg = parse(CollectDataConfig, argv)
    if cfg.mantis:
        from .mantis_bridge import (
            add_quest_key_to_direct_fallback,
            load_direct_mantis_fallback,
        )

        fallback, quest_key = load_direct_mantis_fallback(collection=True)
        cfg = parse(CollectDataConfig, argv, fallback_overlay=fallback)
        if cfg.mantis_source == "quest" and quest_key is not None:
            add_quest_key_to_direct_fallback(fallback, quest_key, collection=True)
            cfg = parse(CollectDataConfig, argv, fallback_overlay=fallback)
    # force=True: importing lerobot (at module load) installs a root handler
    # and leaves the root level at WARNING, which would otherwise make this a
    # no-op and silently drop every log_say() status line.
    logging.basicConfig(level=getattr(logging, cfg.log_level), force=True)

    # System setup (Jetson clock pinning, the GStreamer NVENC stack) is handled
    # by the host installer + its boot service, not here — see
    # `axol jetson.setup` / `axol gst.install`. This entry point just runs.

    if cfg.mantis:
        # Deterministic source/transform errors must be raised before opening
        # a dongle, starting libsurvive, or waiting for live tracker inputs.
        _prepare_mantis_collection(cfg)
        _prepare_recording_cameras(cfg)
        from .mantis_bridge import managed_mantis_bridge, set_managed_pose_source_id

        pose_source_id = (
            set_managed_pose_source_id(cfg) if cfg.mantis_source != "quest" else None
        )

        with managed_mantis_bridge(
            cfg.mantis_source,
            left_channel=cfg.robot_config.left_channel,
            right_channel=cfg.robot_config.right_channel,
            port=cfg.teleop_config.vr_server_config.port,
            pose_source_id=pose_source_id,
        ):
            _run(cfg)
    else:
        _run(cfg)


def _run(
    cfg: CollectDataConfig,
    stop_event: "threading.Event | None" = None,
    control: "Control | None" = None,
) -> None:
    """Run the collection session until quit/stop.

    ``stop_event`` (optional) aborts the session from another thread (the
    serve runner's Stop). ``control`` (optional) is a
    :class:`_QueueCollectControl` carrying web-panel episode commands, merged
    with the VR headset events inside the episode loop; ``None`` (plain CLI)
    leaves the VR headset as the only episode-control source.
    """
    import numpy as np
    from lerobot.processor import make_default_processors
    from lerobot.teleoperators.utils import TeleopEvents
    from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
    from lerobot.utils.feature_utils import (
        hw_to_dataset_features,
    )
    from lerobot.utils.utils import log_say
    from lerobot.utils.visualization_utils import init_rerun

    from ..lerobot.robot.robot_axol import _LEFT_EE_KEYS, _RIGHT_EE_KEYS, AxolRobot
    from ..lerobot.robot.robot_mantis import MantisRobot
    from ..lerobot.teleop.teleop_vr import AxolVRTeleop
    from ..mantis.relative import quat_xyzw_to_rotvec
    from ..vr.models import VRState

    if cfg.mantis:
        # Serve calls _run directly, while the plain CLI calls this a second
        # time after its pre-bridge check. The preparation is idempotent.
        _prepare_mantis_collection(cfg)

    # This is pure config validation/pruning. Direct Mantis CLI runs also call
    # it before entering their managed bridge so a missing/all-stream-only
    # camera setup cannot open trackers and wait for triggers before failing.
    _prepare_recording_cameras(cfg)

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

    # Flag physically-stereo ZED X before the relay/robot opens the cameras so
    # the relay and in-process fallback both use the stereo grab path. The pure
    # assigned/recording validation above already pruned placeholder slots.
    if isinstance(cfg.robot_config, AxolRobotConfig):
        from ..zed import stereo_serials

        cfg.robot_config.apply_detected_stereo(stereo_serials())

    from ..lerobot.robot.config_mantis import MantisRobotConfig

    mantis_mode = isinstance(cfg.robot_config, MantisRobotConfig)

    # The teleop's action keys must match the robot's: propagate the SKU's
    # gripper capability so the gripperless SKU records no gripper channels.
    # The Mantis always has grippers, so only the real robot propagates.
    if (
        not mantis_mode
        and isinstance(cfg.robot_config, AxolRobotConfig)
        and isinstance(cfg.teleop_config, AxolVRTeleopConfig)
    ):
        cfg.teleop_config.has_gripper = cfg.robot_config.axol_config.has_gripper

    robot = (
        MantisRobot(cfg.robot_config, defer_gripper_enable=True)
        if mantis_mode
        else AxolRobot(cfg.robot_config)
    )
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

    from ..utils.network import local_ip

    hostname = socket.gethostname()
    host_ip = local_ip()
    if cfg.mantis and cfg.mantis_source != "quest":
        print(
            "Optional camera/episode UI (tracking comes from the local "
            f"{cfg.mantis_source} bridge):"
        )
    else:
        print("Connect the VR app (https://axol.almond.bot) to this machine:")
    print(f"  Hostname : {hostname}.local")
    print(f"  IP       : {host_ip}")

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
        if mantis_mode:
            # ``observation.pose_lag`` is the residual pose↔image skew per row
            # (camera capture time minus pose capture time), injected by the
            # capture loop for training-time latency handling. On-robot
            # sessions appended to this dataset (collect-data resume,
            # run-policy recording) fill it too.
            features["observation.pose_lag"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["pose_lag"],
            }

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
        # Tracked (VR) poses carry measurement noise the robot's encoder FK
        # doesn't — smooth only Mantis episodes (see record_proc._maybe_smooth_episode).
        "smooth_ee_hz": cfg.mantis_smooth_hz if mantis_mode else 0.0,
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
    # Mantis only: worst pose-stream age (tick time minus pose capture time) in the
    # window — transit + playout delay + filter lag, the residual misalignment
    # between what the wrist cameras saw and the pose the row records.
    max_pose_lag = {"v": 0.0}

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
        if mantis_mode:
            _logger.info(
                "loop: %.1f Hz  vr: %.1f Hz  ik: %.1f Hz  pose_lag: %.0f ms",
                loop_hz,
                teleop.vr_hz(),
                teleop.ik_hz(),
                1e3 * max_pose_lag["v"],
            )
            max_pose_lag["v"] = 0.0
        else:
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

    def _note_ready(episode: int) -> str:
        """Announce the episode using controls that exist for this source."""
        panel = isinstance(control, _QueueCollectControl)
        if mantis_mode and cfg.mantis_source != "quest":
            message = (
                f"Episode {episode}: Mantis is ready. Keep both trackers live; "
                "at the start pose release then squeeze both triggers together "
                "to align, then rapidly squeeze either trigger three times to "
                "start" + (" (or select Start recording here)." if panel else ".")
            )
        elif mantis_mode:
            message = (
                f"Episode {episode}: Mantis is ready. Hold both Quest controllers "
                "at the start pose and press both grip buttons to align; press A "
                "in Quest to start"
                + (" (or select Start recording here)." if panel else ".")
            )
        else:
            message = (
                f"Episode {episode}: robot is at rest pose. Press record on the "
                "VR controller" + (" or select Start recording here." if panel else ".")
            )
        control.note_ready(episode, message)
        return message

    # The hot control loop runs *on the robot's event loop* (see
    # AxolRobot.event_loop) so motion_control is awaited inline — cooperatively
    # interleaved with CAN telemetry on one thread, exactly like `axol teleop`.
    # The main thread drives the episode lifecycle (dataset writes, rest-pose
    # moves) and blocks on each coroutine until the episode (or reset) finishes.
    async def _episode_loop() -> tuple[bool, bool, bool, EpisodeQAStats]:
        recording = False
        rerecord = False
        # Per-episode data-quality counters (Mantis mode; see EpisodeQAStats).
        # Reset when recording actually starts so the pre-record phase never
        # pollutes the verdict.
        stats = EpisodeQAStats()
        # Latest tracked TCP poses, held across ticks so a slow IK reply
        # never reverts a row to FK-of-joints mid-stream. ``None`` until the
        # first IK solve (those early rows fall back to FK, which the worker
        # seeds from the rest pose anyway).
        last_tcp: dict[str, list[float]] | None = None
        # perf_counter deadline of a panel-started record countdown, or None.
        pending_start: float | None = None
        # Capture-health ack: ~2 s into recording, poll the recorder's row
        # count once and shout if nothing has been captured (a silently dead
        # capture thread otherwise only surfaces at save).
        recording_started_at: float | None = None
        capture_checked = False
        # Engage-edge tracking for the re-engage-while-recording flag.
        was_engaged = teleop.is_engaged()
        # Tracking-phase contact watchdog (opt-in — the threshold defaults
        # to 0 = off): the same sustained-torque trip the guarded return
        # uses, active while the operator drives the arms. A trip ends the
        # loop (third element of the return tuple True); the caller discards
        # any recording and runs the limp contact hold.
        watchdog = (
            ContactWatchdog(vrt_cfg.teleop_torque_threshold)
            if vrt_cfg.teleop_torque_threshold > 0
            else None
        )

        async def _start_recording() -> bool:
            """Enable Mantis grippers, then start capture for this take."""
            nonlocal recording, stats, recording_started_at, capture_checked
            nonlocal was_engaged
            if mantis_mode:
                tracking = teleop.tracking_sides()
                if not all(tracking.values()):
                    missing = ", ".join(
                        side for side, live in tracking.items() if not live
                    )
                    log_say(
                        "Cannot start recording: live tracking is missing for "
                        f"{missing}. Restore visibility/SLAM tracking and try again."
                    )
                    return False
                triggers = teleop.trigger_sides()
                if not all(triggers.values()):
                    missing = ", ".join(
                        side for side, live in triggers.items() if not live
                    )
                    log_say(
                        "Cannot start recording: the Mantis trigger heartbeat is "
                        f"missing for {missing}. Restore the CAN connection, "
                        "release both triggers, re-align, and try again."
                    )
                    return False
                if not teleop.is_engaged():
                    log_say(
                        "Cannot start recording: Mantis is not aligned and engaged. "
                        "Hold both rigs at the start pose and engage, then try again."
                    )
                    return False
                log_say("Preparing Mantis grippers.")
                enable_task = asyncio.create_task(robot.enable_grippers_async())
                while not enable_task.done() and not _stopped():
                    await asyncio.sleep(0.05)
                if _stopped():
                    # Calibration is the only autonomous Mantis gripper move.
                    # Stop must interrupt its sweep instead of waiting for all
                    # hard-stop steps and CAN retries to finish.
                    enable_task.cancel()
                    try:
                        await enable_task
                    except asyncio.CancelledError:
                        pass
                    finally:
                        await robot.disable_grippers_async()
                    return False
                await enable_task
                # The hard-stop calibration above can take seconds. Inputs
                # that were valid before it began may have dropped and forced
                # a disengage meanwhile; never open the recorder on that stale
                # preflight result.
                tracking = teleop.tracking_sides()
                triggers = teleop.trigger_sides()
                engaged = teleop.is_engaged()
                if (
                    not all(tracking.values())
                    or not all(triggers.values())
                    or not engaged
                ):
                    failures: list[str] = []
                    if not all(tracking.values()):
                        failures.append("tracker visibility/SLAM tracking changed")
                    if not all(triggers.values()):
                        failures.append("a trigger CAN heartbeat dropped")
                    if not engaged:
                        failures.append("Mantis disengaged")
                    await robot.disable_grippers_async()
                    _note_ready(episode_idx + 1)
                    log_say(
                        "Cannot start recording: "
                        + "; ".join(failures)
                        + " while the grippers were preparing. Restore both "
                        "inputs, release both triggers, re-align, squeeze "
                        "together, and try again."
                    )
                    return False
            if _stopped():
                if mantis_mode:
                    await robot.disable_grippers_async()
                return False
            try:
                if relay is not None:
                    relay.set_raw_enabled(True)
                recorder.start_episode(task)
            except BaseException:
                if relay is not None:
                    relay.set_raw_enabled(False)
                if mantis_mode:
                    await robot.disable_grippers_async()
                raise
            recording = True
            stats = EpisodeQAStats()
            recording_started_at = time.perf_counter()
            capture_checked = False
            was_engaged = teleop.is_engaged()
            control.note_recording()
            log_say("Recording started.")
            # Reflect the recording state on the headset HUD (no-op for the
            # VR-initiated start, where the headset already switched itself).
            teleop.send_feedback_state(VRState.RECORDING)
            return True

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
                _note_ready(episode_idx + 1)
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

            if watchdog is not None:
                tripped = watchdog.update(robot.torque_residuals())
                if tripped is not None:
                    joint, residual = tripped
                    _logger.warning(
                        "teleop contact: %s torque residual %.1f exceeds %.1f — going limp",
                        joint,
                        residual,
                        vrt_cfg.teleop_torque_threshold,
                    )
                    return recording, rerecord, True, stats

            # Record the action in the configured action space: identity for
            # joint datasets, FK-to-Cartesian when observe_cartesian is set. The
            # arm is still commanded with the teleop joint targets above, so its
            # motion is unchanged — only the stored representation differs.
            #
            # Row timestamp: on the robot the loop tick is correct (state and
            # image both describe the physical robot at t0). On the Mantis the
            # pose stream *is* the plant's ground truth, so the row is stamped
            # with the pose's capture time — the moment the hand was actually
            # there — keeping it on the same capture timeline as the camera
            # exposure timestamps.
            row_ts = t0
            act_ds = robot.action_to_dataset(act_processed)
            pose_ts: float | None = None
            if mantis_mode:
                pose_ts = teleop.pose_capture_ts()
                if pose_ts is not None:
                    row_ts = pose_ts
                    lag = t0 - pose_ts
                    if lag > max_pose_lag["v"]:
                        max_pose_lag["v"] = lag
                    if recording and lag > stats.max_pose_lag_s:
                        stats.max_pose_lag_s = lag
                # Overwrite the EE dims of both the state and the action with
                # the tracked ground-truth TCP pose: on the rig the tracker
                # (not IK-solved virtual joints roundtripped through FK) is
                # where the gripper physically is, and training must not
                # inherit filter/solver artifacts. Grippers stay as-is —
                # measured feedback in the state, commanded in the action.
                tcp = teleop.tcp_poses()
                if tcp is not None:
                    last_tcp = tcp
                if last_tcp is not None:
                    joint_obs = dict(joint_obs)
                    act_ds = dict(act_ds)
                    for keys, pose in (
                        (_LEFT_EE_KEYS, last_tcp["left"]),
                        (_RIGHT_EE_KEYS, last_tcp["right"]),
                    ):
                        pose6 = [
                            *pose[:3],
                            *quat_xyzw_to_rotvec(np.asarray(pose[3:7])),
                        ]
                        joint_obs.update(zip(keys, pose6))
                        act_ds.update(zip(keys, pose6))
            recorder.publish(joint_obs, act_ds, row_ts)

            # Per-episode QA counters (Mantis only — encoder-FK poses can't go
            # stale). Frozen TCPs and disengaged spans record a motionless
            # pose under fresh timestamps; a re-engage re-fits the world→base
            # transform and shifts the frame of every later row.
            if mantis_mode and recording:
                stats.total_frames += 1
                engaged = teleop.is_engaged()
                if not engaged:
                    stats.disengaged_frames += 1
                elif not was_engaged:
                    stats.reengaged_while_recording = True
                was_engaged = engaged
                if not all(teleop.tracking_sides().values()):
                    stats.untracked_frames += 1
                if not all(teleop.trigger_sides().values()):
                    stats.trigger_loss_frames += 1
                if pose_ts is None or t0 - pose_ts > _QA_STALE_POSE_S:
                    stats.stale_frames += 1

            # Capture-health ack: recorder.start_episode has no feedback, so
            # ~2 s in, check that rows are actually landing (one cheap
            # round-trip; a dead capture thread otherwise stays silent until
            # save).
            if (
                recording
                and not capture_checked
                and recording_started_at is not None
                and t0 - recording_started_at >= 2.0
            ):
                capture_checked = True
                try:
                    rows = recorder.frame_count()
                except Exception as exc:  # noqa: BLE001 - check is advisory
                    _logger.warning("capture health check failed: %s", exc)
                    rows = -1
                if rows == 0:
                    log_say(
                        "WARNING: recording for 2 seconds but the recorder "
                        "has captured zero rows — check the cameras and the "
                        "recorder log."
                    )

            # Merge the two episode-control sources: VR headset state
            # transitions and (when serving) web-panel queue commands.
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
                    _note_ready(episode_idx + 1)
                    log_say("Recording start cancelled.")
            start_requested = events.get("start_recording") or (
                pending_start is not None and time.perf_counter() >= pending_start
            )

            if start_requested and not recording:
                pending_start = None
                if not await _start_recording():
                    return recording, rerecord, False, stats
                # First-use calibration can take seconds. Re-anchor absolute
                # pacing so the hot loop does not try to "catch up" that gap
                # with a burst of CAN commands and recorder publications.
                deadline = time.perf_counter()
                prev_t0["v"] = 0.0
                continue

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
                if events.get(TeleopEvents.FAILURE):
                    log_say("Episode ended as failure.")
                else:
                    log_say("Episode ended successfully.")
                teleop.send_feedback_state(VRState.SAVING)
                break
            if events[TeleopEvents.RERECORD_EPISODE]:
                rerecord = True
                break

            await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
            slip = time.perf_counter() - deadline
            if slip > max_slip["v"]:
                max_slip["v"] = slip

        return recording, rerecord, False, stats

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

    async def _contact_hold_loop() -> None:
        """Tracking contact: hold limp until reset, then return to rest guarded.

        The hold leaves the operator's reset press latched, so the guarded
        return that follows plans from wherever the arms were hand-guided; on
        an orphaned/stopped hold nothing is latched and the return is skipped
        (the arms hold position where they are).
        """
        await teleop.contact_hold(
            gravity_step=_guard_gravity_step,
            reset_command_state=robot.reset_command_state,
            get_positions=lambda: robot.positions,
            stopped=_stopped,
            announce=log_say,
            on_contact=_guard_on_contact,
            hold_tick=_guard_hold_tick,
            vr_alive=teleop.vr_alive,
        )
        if teleop.is_resetting:
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

    def _disable_mantis_grippers() -> None:
        """Leave the handheld grippers torque-off between collection takes."""
        if mantis_mode:
            _run_on_robot_loop(robot.disable_grippers_async())

    def _begin_disable_mantis_grippers() -> Any | None:
        """Start torque-off without waiting on recorder/camera shutdown."""
        if not mantis_mode:
            return None
        return asyncio.run_coroutine_threadsafe(
            robot.disable_grippers_async(), robot.event_loop
        )

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
            log_say(_note_ready(episode_idx + 1))

            try:
                recording, rerecord, contact, qa = _run_on_robot_loop(_episode_loop())
            except BaseException:
                # A failure after a recording start must not leave the handheld
                # grippers powered while the rest of collection unwinds.
                try:
                    _disable_mantis_grippers()
                except Exception:
                    _logger.exception(
                        "failed to disable Mantis grippers after episode error"
                    )
                raise

            # The episode is over the moment the loop breaks. Start Mantis
            # torque-off immediately, in parallel with freezing the recorder:
            # a wedged camera read must never keep the grippers powered while
            # stop_capture waits. The recorder stop signal is still issued in
            # this same turn, so at most its already-in-flight row can finish.
            disable_future = _begin_disable_mantis_grippers()
            try:
                if recording:
                    captured_rows, capture_error = recorder.stop_capture()
                else:
                    captured_rows, capture_error = 0, None
            finally:
                if disable_future is not None:
                    disable_future.result()

            # Recording done — close the raw branch so the rest-pose/reset and
            # next pre-record phase stay light.
            if relay is not None:
                relay.set_raw_enabled(False)

            if _stopped():
                if recording:
                    recorder.cancel_episode()
                break

            if contact:
                # The tracking contact watchdog tripped: an in-flight episode
                # is unusable (the arms went limp mid-take), so discard it,
                # then run the limp hold + guarded return on the robot loop.
                if recording:
                    recorder.cancel_episode()
                    log_say("Episode discarded (contact).")
                _run_on_robot_loop(_contact_hold_loop())
                # Drain VR events fired during the hold/return, then unblock
                # the headset for the next take.
                teleop.get_teleop_events()
                teleop.send_feedback_state(VRState.DATA_COLLECTION)
                continue

            # LeRobot refuses ``save_episode`` when the capture thread did not
            # manage to append even one row. This can happen when an episode is
            # ended immediately after it starts (before the first camera tick)
            # or when capture fails at startup. Treat it as a recoverable bad
            # take: clear the empty buffer and leave the session ready to try
            # the same episode again instead of tearing down robot control.
            if recording and not rerecord:
                if capture_error is not None:
                    log_say(
                        "Episode capture failed — discarding and re-recording. "
                        + capture_error
                    )
                    rerecord = True
                elif captured_rows == 0:
                    log_say(
                        "Episode captured no dataset rows — it ended before the "
                        "first camera frame arrived. Discarding and re-recording."
                    )
                    rerecord = True

            # Episode QA: always log the one-line verdict; a bad episode is
            # refused at save and downgraded to discard + re-record unless
            # the gate is disabled (cfg.qa_gate — escape hatch).
            if recording:
                qa_ok, qa_reasons = evaluate_episode_qa(qa)
                _logger.info(
                    "episode QA: control_frames=%d captured_rows=%d stale=%d "
                    "(%.1f%%) disengaged=%d (%.1f%%) untracked=%d (%.1f%%) "
                    "trigger_loss=%d reengaged=%s "
                    "max_pose_lag=%.0fms capture_error=%s -> %s",
                    qa.total_frames,
                    captured_rows,
                    qa.stale_frames,
                    100 * qa.stale_fraction,
                    qa.disengaged_frames,
                    100 * qa.disengaged_fraction,
                    qa.untracked_frames,
                    100 * qa.untracked_fraction,
                    qa.trigger_loss_frames,
                    qa.reengaged_while_recording,
                    1e3 * qa.max_pose_lag_s,
                    capture_error or "none",
                    (
                        "OK"
                        if qa_ok and captured_rows > 0 and capture_error is None
                        else "BAD"
                    ),
                )
                if not qa_ok and not rerecord:
                    if cfg.qa_gate:
                        log_say(
                            "Episode REJECTED by the quality gate — "
                            "discarding and re-recording. " + " ".join(qa_reasons)
                        )
                        rerecord = True
                    else:
                        _logger.warning(
                            "qa_gate is off — saving a bad episode anyway: %s",
                            "; ".join(qa_reasons),
                        )

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
