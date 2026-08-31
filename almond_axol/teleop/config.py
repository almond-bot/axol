"""VRTeleopConfig dataclass with all tunable parameters for a VRTeleop session."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

_logger = logging.getLogger(__name__)


@dataclass
class VRTeleopConfig:
    """Configuration for a :class:`VRTeleop` session.

    Attributes:
        rest_pose_left: Left arm rest configuration in radians, shape (7,) in
            ARM_JOINTS order (no gripper). Used as the reset target.
        rest_pose_right: Right arm rest configuration in radians, shape (7,) in
            ARM_JOINTS order (no gripper). Used as the reset target.
        frequency: Control loop rate in Hz used by :meth:`VRTeleop.run` and
            as waypoint density for reset trajectories.
        reset_speed: Average joint velocity (rad/s) of the worst-case joint
            during a return-to-rest move. The smoothstep profile gives a
            peak joint velocity of ``1.5 * reset_speed``. Determines the
            number of trajectory waypoints based on the distance to the
            rest pose, subject to ``reset_min_duration`` below.
        reset_min_duration: Floor (seconds) on the return-to-rest trajectory
            duration. Prevents near-rest starts from snapping home in a
            handful of frames and gives every reset a consistent minimum
            feel regardless of starting pose. Defaults to ``1.5`` s.
        reset_rest_weight: Cost weight penalising deviation from the reset
            target pose during collision-aware trajectory generation.
        reset_limit_weight: Cost weight penalising joint-limit violations
            during reset trajectory generation.
        reset_collision_margin: Minimum clearance (m) enforced between
            collision bodies during reset trajectory generation.
        reset_collision_weight: Cost weight on self-collision penalty during
            reset trajectory generation.
        reset_max_iterations: Maximum solver iterations per reset waypoint.
        reset_torque_threshold: Contact watchdog for guarded return-to-rest
            moves (hardware only). If any arm joint's torque residual
            (measured minus modeled gravity, in the motor's torque units —
            Nm on Damiao) stays above this for a sustained window, the move
            is judged to have hit something — a gripper still hooked on the
            scene, or an operator grabbing an arm — and the arms drop into
            a limp gravity-comp hold until reset is pressed again, which
            replans from wherever the arms were left. Raise if normal
            returns false-trip on the friction / model-error background;
            ``0`` disables the watchdog. Defaults to ``4.0``.
        teleop_torque_threshold: Contact watchdog for the *tracking* phase
            (hardware only) — the same sustained-torque-residual trip as
            ``reset_torque_threshold``, but active while the operator is
            driving the arms (``axol teleop`` and ``collect-data``; replay
            playback shares this field). On a trip, tracking disengages and
            the arms drop into the limp gravity-comp hold; hand-guide them
            clear and press reset to return to rest and continue. ``0``
            (the default) disables it — tracking pushes against payloads
            and the scene on purpose, so only the return-to-rest guard is
            always on. Set a threshold (``16.0`` is the control panel's
            suggested value) to enable.
        reset_gravity_comp_kd: Velocity damping (Nm·s/rad) for the arm
            joints during the contact-fallback gravity-comp hold; same
            semantics as ``axol gravity-comp --kd``. Defaults to ``0.25``.
        hold_to_engage: Grip behaviour. ``False`` (default) is the toggle
            scheme: from rest, click both grips together to engage both
            arms; while engaged, a click on either grip toggles *that* arm
            between tracking and frozen (a frozen arm holds its pose and
            gripper — e.g. it keeps a grasp steady while the other arm
            works). ``True`` is the dead-man scheme: hold both grips to
            engage from rest, and each arm tracks only while its grip
            stays held — release a grip and that arm freezes where it is,
            hold on and it keeps going; re-engaging from a full release
            requires holding both again.
        engage_max_vel: Starting joint-velocity cap (rad/s) for the
            trapezoidal filter when teleop is first engaged after a rest-pose
            trajectory (startup or reset). Softens the transition from rest
            pose to the first IK target: the cap smoothsteps from this value
            up to ``teleop_max_vel`` over ``engage_duration`` seconds, so the
            arm starts gentle and opens up progressively (a hard restore used
            to release the error accumulated during the slow phase all at
            once). Defaults to ``reset_speed`` for a consistent feel.
        engage_duration: Seconds over which the velocity cap ramps from
            ``engage_max_vel`` to ``teleop_max_vel`` after the post-rest
            engage rising edge.
        teleop_max_vel: Maximum joint velocity (rad/s) enforced by the
            trapezoidal filter during normal teleoperation.  Limits how fast
            any single joint can move toward a new IK target.  Defaults to
            1.0 rev/s (~360 °/s).
        teleop_max_accel: Maximum joint acceleration (rad/s²) enforced by the
            trapezoidal filter.  Controls how quickly the commanded velocity
            ramps up or down.  Defaults to 3.5 rev/s² (~1260 °/s²), giving a
            ~0.3 s ramp from rest to full speed.
        ik_alpha: Blend factor for the exponential moving average applied to
            the IK output before the trapezoidal filter.  Range ``(0, 1]``
            where ``1.0`` disables smoothing.  Lower values kill more
            high-frequency jitter at the cost of a small fixed lag
            (``~(1-alpha)/alpha`` frames).  Defaults to ``0.3`` (~20 ms lag
            at 120 Hz), favouring smoothness over minimum latency.
        pose_min_cutoff: Minimum cutoff frequency (Hz) for the One Euro Filter
            applied to raw VR controller positions, quaternions, and elbow
            positions **before** they enter the IK solver.  This is the
            primary tremor / tracking-noise kill knob.  Lower values give
            heavier smoothing at rest (more tremor rejection) at the cost of
            slightly more lag when still.  Typical range: 0.5–3 Hz.  Defaults
            to ``0.8`` Hz, favouring smoothness over minimum latency (the
            fixed-lag smoother in the VR server's pose interpolator does the
            heavy lifting upstream).
        pose_beta: Speed coefficient for the One Euro Filter.  Raises the
            filter cutoff proportionally to the signal's instantaneous speed,
            keeping the filter transparent during fast intentional moves.
            Increase if fast moves feel sticky.  Defaults to ``2.0`` so the
            filter stays partially engaged during motion instead of opening
            fully and passing noise through.
        position_multiplier: Scale factor applied to the controller's
            **position** displacement (not orientation) when mapping hand
            motion to the end-effector target.  ``1.0`` is 1:1 motion;
            ``2.0`` moves the arm twice as far as the hand, which helps cover
            the robot's full reach when the arm is longer than the operator's.
            Applied to both the end-effector and elbow position deltas so the
            arm posture scales coherently.  Defaults to ``1.0``.
        rotation_multiplier: Scale factor applied to the controller's
            **orientation** displacement (not position) when mapping hand
            motion to the end-effector target.  The relative rotation of the
            controller since engage is converted to axis-angle and its angle
            is scaled by this factor; ``1.0`` is 1:1 motion, ``2.0`` rotates
            the end-effector twice as far as the wrist.  Defaults to ``1.0``.
        disengage_timeout: Auto-disengage when no VR pose frame has arrived
            for this many seconds while teleop is engaged — the operator left
            VR (headset doffed, session exited without pressing X/Y) or the
            link dropped.  Without it the engage toggle survives the gap, so
            the next frames after re-entering VR are tracked against the old
            engage snapshot and the arms jerk toward wherever the controllers
            now are.  After the disengage the arms hold position wherever
            they are — a lost link never moves the robot on its own — until
            the operator deliberately re-engages (a fresh engage snapshot, so
            motion resumes relative to the controllers' new pose) or presses
            reset.  The headset streams poses at 72+ Hz while presenting,
            so anything beyond a few hundred ms is a real gap, not jitter.
            ``0`` disables the timeout.  Defaults to ``0.5`` s.
        absolute_mode: Mantis mapping. Instead of re-applying
            controller *deltas* onto the robot's engage-time FK pose (normal
            teleop), the engage rising edge solves a world-anchored robot
            **base transform** — gravity-aligned, positioned/oriented so the
            rest-pose FK gripper poses coincide with the two controllers'
            current poses — and every subsequent frame maps the controller
            pose 1:1 into that fixed base frame as an absolute IK target.
            Engaging is therefore the alignment act: the operator holds both
            grippers at the agreed start pose (matching the robot's rest
            pose relative to the task scene) and squeezes both grips. The
            offset between each controller and its gripper TCP is applied from
            ``tcp_transform_left/right`` when known. Without that transform,
            the engage snapshot aligns the starting pose only; later poses are
            mount-dependent and unsuitable for production collection. Elbow
            hints are ignored in this mode — human
            elbow positions say nothing about the robot's preferred null-space
            posture. ``position_multiplier`` / ``rotation_multiplier`` do not
            apply (the mapping is 1:1 by construction). Defaults to ``False``.
        base_height: Optional fixed height (metres) of the robot base origin
            above the VR floor (the WebXR ``local-floor`` reference space) in
            ``absolute_mode``. When set, the engage-time base fit pins the
            base's vertical position to it — matching the robot's real
            mounting height so datasets stay consistent across operators of
            different heights. ``None`` (default) lets the fit take the
            vertical position from the held grippers.
        tcp_transform_left: Tracker→gripper SE(3) transform for the left rig
            as ``[x, y, z, qx, qy, qz, qw]`` (gripper frame expressed in the
            tracker's local frame) — the rig's factory design constant, or a
            per-unit override from ``~/.almond/mantis/tcp_transform.json``.
            When set, ``absolute_mode`` maps each tracked pose through it —
            recorded TCP poses become mount-independent and wrist rotations
            stop smearing into position error. ``None`` falls back to a
            start-pose-only engage alignment, which is useful for bring-up but
            not production Cartesian data.
        tcp_transform_right: Same for the right rig.
        tracker_key: Identity key of the active tracker (e.g.
            ``"quest:meta-quest-touch-plus:grip"``, ``"survive:T20"``, or
            ``"ultimate:<mac>"``) used to select the matching per-tracker
            entry from the saved calibration file when
            ``tcp_transform_left`` / ``tcp_transform_right`` are unset.
            ``None`` (default) uses the operation's explicit Mantis source when
            available; legacy callers infer it from the saved tracker config
            (file present → its backend/devices, absent → ``"quest"``).
        quest_controller_profile: Expected WebXR controller profile for a
            calibrated Quest rig (for example ``"oculus-touch-v3"``). A
            profile-scoped transform key of the form
            ``quest:<profile>:<space>`` fills this automatically. Absolute
            tracking fails closed when a connected controller reports a
            different profile or omits profile metadata.
        quest_pose_space: Expected WebXR controller pose datum: ``"grip"``
            for production Mantis capture, or ``"target-ray"`` only for
            uncalibrated compatibility bring-up. Filled from a profile-scoped
            Quest transform key alongside ``quest_controller_profile``.
        urdf_viewer_world_aligned: Whether the active pose producer's world is
            registered to the viewer headset's local-floor world. ``None``
            chooses True for Quest and False for Lighthouse/Ultimate. External
            tracker sessions keep the Quest video/HUD viewer useful but hide
            the spatial URDF overlay unless an operator explicitly asserts an
            out-of-band world registration.
    """

    rest_pose_left: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                -0.025 * 2 * math.pi,
                0.0,
                0.0,
                0.05 * 2 * math.pi,
                0.0,
                0.0,
                -0.025 * 2 * math.pi,
            ],
            dtype=np.float32,
        )
    )
    rest_pose_right: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                0.025 * 2 * math.pi,
                0.0,
                0.0,
                -0.05 * 2 * math.pi,
                0.0,
                0.0,
                0.025 * 2 * math.pi,
            ],
            dtype=np.float32,
        )
    )
    frequency: float = 120.0
    reset_speed: float = 0.1 * 2 * math.pi
    reset_min_duration: float = 1.5
    reset_rest_weight: float = 50.0
    reset_limit_weight: float = 100.0
    reset_collision_margin: float = 0.025
    reset_collision_weight: float = 100.0
    reset_max_iterations: int = 10
    reset_torque_threshold: float = 4.0
    teleop_torque_threshold: float = 0.0
    reset_gravity_comp_kd: float = 0.25
    hold_to_engage: bool = False
    engage_max_vel: float = 0.1 * 2 * math.pi
    engage_duration: float = 1.0
    teleop_max_vel: float = 1.0 * 2 * math.pi
    teleop_max_accel: float = 3.5 * 2 * math.pi
    ik_alpha: float = 0.3
    pose_min_cutoff: float = 0.8
    pose_beta: float = 2.0
    position_multiplier: float = 1.0
    rotation_multiplier: float = 1.0
    disengage_timeout: float = 0.5
    absolute_mode: bool = False
    base_height: float | None = None
    tcp_transform_left: list[float] | None = None
    tcp_transform_right: list[float] | None = None
    tracker_key: str | None = None
    quest_controller_profile: str | None = None
    quest_pose_space: str | None = None
    urdf_viewer_world_aligned: bool | None = None


def apply_mantis_teleop_profile(
    config: VRTeleopConfig, *, tracker_source: str | None = None
) -> None:
    """Force the Mantis mapping/faithfulness profile in place.

    Shared by ``collect-data --mantis`` and ``teleop --mantis`` so the two flows
    behave identically: ``absolute_mode`` (the engage squeeze is the start-pose
    alignment act), toggle-style lock semantics, and transparent smoothing —
    the EMA and trapezoid filters exist to protect a physical arm and only add
    lag between the solution and where the hand actually was, so with no arm to
    protect the joints should follow the raw IK output. Managed bridges use an
    acknowledged low→high edge when automatically freezing or re-engaging, so
    ``hold_to_engage`` must be disabled; otherwise that required low release
    would act as a dead-man disengage. The One Euro cutoff is raised for the
    same reason: its rest-tremor smoothing costs ~100 ms of lag at slow speeds,
    which on the rig is pure pose↔image misalignment (and visible slack in the
    headset's URDF overlay); a higher cutoff keeps the solution pinned to the
    hand at the price of passing through a little tremor.

    Also resolves the tracker→gripper transform per side into
    ``tcp_transform_left`` / ``tcp_transform_right``, first match wins:
    explicitly set config values; the override-file entry for the active
    tracker (``config.tracker_key``, or derived from ``tracker_source`` — see
    :func:`almond_axol.mantis.calibration.tracker_key_for_side`); the rig's
    factory design transform for the tracker family
    (:data:`~almond_axol.mantis.calibration.DESIGN_TCP_TRANSFORMS` — a design
    constant, so it applies out of the box). A legacy unkeyed override is
    considered only when no active source is declared; a Mantis run never
    applies a transform measured with an unknown tracker. A tracker with none
    of these is warned about loudly: the session
    would otherwise silently record uncalibrated (engage-snapshot) TCP poses.
    """
    config.absolute_mode = True
    config.hold_to_engage = False
    config.ik_alpha = 1.0
    config.teleop_max_vel = 1e6
    config.teleop_max_accel = 1e6
    config.engage_max_vel = 1e6
    config.pose_min_cutoff = 5.0
    if config.urdf_viewer_world_aligned is None:
        config.urdf_viewer_world_aligned = tracker_source not in (
            "lighthouse",
            "ultimate",
        )

    if tracker_source == "quest" and config.tracker_key is not None:
        from ..mantis.calibration import parse_quest_tracker_key

        quest_datum = parse_quest_tracker_key(config.tracker_key)
        if quest_datum is not None:
            profile, pose_space = quest_datum
            if config.quest_controller_profile not in (None, profile):
                raise ValueError(
                    "Quest tracker_key profile conflicts with "
                    f"quest_controller_profile: {profile!r} != "
                    f"{config.quest_controller_profile!r}"
                )
            if config.quest_pose_space not in (None, pose_space):
                raise ValueError(
                    "Quest tracker_key pose space conflicts with "
                    f"quest_pose_space: {pose_space!r} != "
                    f"{config.quest_pose_space!r}"
                )
            config.quest_controller_profile = profile
            config.quest_pose_space = pose_space

    if config.tcp_transform_left is None or config.tcp_transform_right is None:
        from ..mantis.calibration import (
            LEGACY_TRACKER_KEY,
            MANTIS_TCP_TRANSFORM_FILE,
            design_transform_for,
            load_tcp_transforms,
            parse_quest_tracker_key,
            select_quest_transform_key,
            tracker_key_for_side,
        )

        saved = load_tcp_transforms()
        if tracker_source == "quest" and config.tracker_key is None:
            config.tracker_key = select_quest_transform_key(saved)
            if config.tracker_key is not None:
                _logger.info(
                    "Mantis Quest: selected the sole profile-scoped transform %r.",
                    config.tracker_key,
                )
        if tracker_source == "quest" and config.tracker_key is not None:
            quest_datum = parse_quest_tracker_key(config.tracker_key)
            if quest_datum is not None:
                profile, pose_space = quest_datum
                if config.quest_controller_profile not in (None, profile):
                    raise ValueError(
                        "Quest transform profile conflicts with "
                        f"quest_controller_profile: {profile!r} != "
                        f"{config.quest_controller_profile!r}"
                    )
                if config.quest_pose_space not in (None, pose_space):
                    raise ValueError(
                        "Quest transform pose space conflicts with "
                        f"quest_pose_space: {pose_space!r} != "
                        f"{config.quest_pose_space!r}"
                    )
                config.quest_controller_profile = profile
                config.quest_pose_space = pose_space
        for side in ("left", "right"):
            attr = f"tcp_transform_{side}"
            if getattr(config, attr) is not None:
                continue
            key, reason = tracker_key_for_side(
                side,
                override=config.tracker_key,
                source=tracker_source,
            )
            entries = saved.get(side, {})
            design = design_transform_for(side, key)
            scoped_quest_key = parse_quest_tracker_key(key)
            if key in entries and not (
                tracker_source == "quest" and scoped_quest_key is None
            ):
                setattr(config, attr, entries[key])
                _logger.info(
                    "Mantis %s: loaded tracker→gripper calibration for tracker %r (%s).",
                    side,
                    key,
                    reason,
                )
            elif design is not None:
                setattr(config, attr, design)
                _logger.info(
                    "Mantis %s: using the rig's factory tracker→gripper transform "
                    "for tracker %r (%s) — a design constant of the standard "
                    "mount. A per-unit entry in %s overrides it.",
                    side,
                    key,
                    reason,
                    MANTIS_TCP_TRANSFORM_FILE,
                )
            elif tracker_source is None and LEGACY_TRACKER_KEY in entries:
                setattr(config, attr, entries[LEGACY_TRACKER_KEY])
                _logger.warning(
                    "Mantis %s: no transform for the active tracker %r (%s) — "
                    "falling back to a LEGACY entry measured with an unknown "
                    "tracker. If the tracker changed since, this transform "
                    "is wrong; re-key or delete the entry in %s.",
                    side,
                    key,
                    reason,
                    MANTIS_TCP_TRANSFORM_FILE,
                )
            else:
                from ..mantis.calibration import candidate_transform_for

                candidate_note = (
                    " A CAD candidate exists, but it is intentionally not "
                    "applied until the live tracker datum and orientation are "
                    "bench-verified."
                    if candidate_transform_for(side, key) is not None
                    else ""
                )
                _logger.warning(
                    "Mantis %s: NO tracker→gripper transform for the active "
                    "tracker %r (%s) — no factory design constant covers this "
                    "tracker family and %s has no entry for it. Absolute mode "
                    "will absorb the whole controller→gripper offset into the "
                    "engage snapshot, so recorded TCP poses will be "
                    "mount-dependent and wrist rotations will smear into "
                    "position error. Add a measured transform to the file "
                    "before collecting data.%s",
                    side,
                    key,
                    reason,
                    MANTIS_TCP_TRANSFORM_FILE,
                    candidate_note,
                )

    # Saved transforms were validated while loading, but explicit dotted CLI
    # / control-panel Advanced values bypass the file loader. Validate the
    # final value from every source before the IK worker can slice it into a
    # position and quaternion. Normalizing here also gives explicit values
    # the same behavior as saved calibration entries.
    from ..mantis.calibration import validate_tcp_transform

    for side in ("left", "right"):
        attr = f"tcp_transform_{side}"
        transform = getattr(config, attr)
        if transform is None:
            continue
        try:
            setattr(config, attr, validate_tcp_transform(transform))
        except ValueError as exc:
            raise ValueError(
                f"Mantis {side} tracker→gripper TCP transform is invalid: {exc}"
            ) from exc
