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
        engage_max_vel: Maximum joint velocity (rad/s) used by the
            trapezoidal filter when teleop is first engaged after a
            rest-pose trajectory (startup or reset).  Slows the transition from
            rest pose to the first IK target.  Restored to ``teleop_max_vel``
            after ``engage_duration`` seconds.  Defaults to
            ``reset_speed`` for a consistent feel.
        engage_duration: Seconds to hold ``engage_max_vel`` after the
            post-rest engage rising edge before restoring ``teleop_max_vel``.
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
            (``~(1-alpha)/alpha`` frames).  Defaults to ``0.5`` (~8 ms lag
            at 120 Hz), which removes most IK noise without a perceptible
            feel difference.
        pose_min_cutoff: Minimum cutoff frequency (Hz) for the One Euro Filter
            applied to raw VR controller positions, quaternions, and elbow
            positions **before** they enter the IK solver.  This is the
            primary tremor / tracking-noise kill knob.  Lower values give
            heavier smoothing at rest (more tremor rejection) at the cost of
            slightly more lag when still.  Typical range: 0.5–3 Hz.  Defaults
            to ``1.5`` Hz.
        pose_beta: Speed coefficient for the One Euro Filter.  Raises the
            filter cutoff proportionally to the signal's instantaneous speed,
            keeping the filter transparent during fast intentional moves.
            Increase if fast moves feel sticky.  Defaults to ``5.0``.
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
        absolute_mode: UMI handheld-rig mapping. Instead of re-applying
            controller *deltas* onto the robot's engage-time FK pose (normal
            teleop), the engage rising edge solves a world-anchored robot
            **base transform** — gravity-aligned, positioned/oriented so the
            rest-pose FK gripper poses coincide with the two controllers'
            current poses — and every subsequent frame maps the controller
            pose 1:1 into that fixed base frame as an absolute IK target.
            Engaging is therefore the alignment act: the operator holds both
            grippers at the agreed start pose (matching the robot's rest
            pose relative to the task scene) and squeezes both grips. The
            offset between each controller and its gripper TCP (mount
            geometry, URDF frame conventions) is absorbed by the same
            engage snapshot. Elbow hints are ignored in this mode — human
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
        tcp_transform_left: Calibrated tracker→gripper SE(3) transform for the
            left rig as ``[x, y, z, qx, qy, qz, qw]`` (gripper frame expressed
            in the tracker's local frame), measured by ``axol umi.calibrate``.
            When set, ``absolute_mode`` maps each tracked pose through it —
            recorded TCP poses become mount-independent and wrist rotations
            stop smearing into position error — instead of absorbing the
            whole controller→gripper offset into the engage snapshot. ``None``
            falls back to the engage-snapshot absorption.
        tcp_transform_right: Same for the right rig.
        tracker_key: Identity key of the active tracker (e.g. ``"quest"``,
            ``"survive:T20"``, ``"ultimate:<mac>"``) used to select the
            matching per-tracker entry from the saved calibration file when
            ``tcp_transform_left`` / ``tcp_transform_right`` are unset.
            ``None`` (default) derives the key from the saved tracker config
            (``~/.almond/tracker/config.json`` present → its backend/devices;
            absent → ``"quest"``). Set it explicitly when that presumption is
            wrong, e.g. teleoperating with the Quest on a machine that also
            has a Vive backend configured.
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
    engage_max_vel: float = 0.1 * 2 * math.pi
    engage_duration: float = 1.0
    teleop_max_vel: float = 1.0 * 2 * math.pi
    teleop_max_accel: float = 3.5 * 2 * math.pi
    ik_alpha: float = 0.5
    pose_min_cutoff: float = 1.5
    pose_beta: float = 5.0
    position_multiplier: float = 1.0
    rotation_multiplier: float = 1.0
    absolute_mode: bool = False
    base_height: float | None = None
    tcp_transform_left: list[float] | None = None
    tcp_transform_right: list[float] | None = None
    tracker_key: str | None = None


def apply_umi_teleop_profile(config: VRTeleopConfig) -> None:
    """Force the UMI handheld-rig mapping/faithfulness profile in place.

    Shared by ``collect-data --umi`` and ``teleop --umi`` so the two flows
    behave identically: ``absolute_mode`` (the engage squeeze is the start-pose
    alignment act), and transparent smoothing — the EMA and trapezoid filters
    exist to protect a physical arm and only add lag between the solution and
    where the hand actually was, so with no arm to protect the joints should
    follow the raw IK output. The One Euro cutoff is raised for the same
    reason: its rest-tremor smoothing costs ~100 ms of lag at slow speeds,
    which on the rig is pure pose↔image misalignment (and visible slack in the
    headset's URDF overlay); a higher cutoff keeps the solution pinned to the
    hand at the price of passing through a little tremor.

    Also loads any saved ``axol umi.calibrate`` tracker→gripper transforms
    into ``tcp_transform_left`` / ``tcp_transform_right`` (explicitly set
    values win over the file). The calibration file is keyed per tracker
    identity; the entry matching the active tracker (``config.tracker_key``,
    or derived — see :func:`almond_axol.umi.calibration.tracker_key_for_side`)
    is selected per side. A missing entry for the active tracker is warned
    about loudly: the session would otherwise silently record uncalibrated
    (engage-snapshot) TCP poses.
    """
    config.absolute_mode = True
    config.ik_alpha = 1.0
    config.teleop_max_vel = 1e6
    config.teleop_max_accel = 1e6
    config.engage_max_vel = 1e6
    config.pose_min_cutoff = 5.0

    if config.tcp_transform_left is None or config.tcp_transform_right is None:
        from ..umi.calibration import (
            LEGACY_TRACKER_KEY,
            UMI_TCP_TRANSFORM_FILE,
            load_tcp_transforms,
            tracker_key_for_side,
        )

        saved = load_tcp_transforms()
        for side in ("left", "right"):
            attr = f"tcp_transform_{side}"
            if getattr(config, attr) is not None:
                continue
            key, reason = tracker_key_for_side(side, override=config.tracker_key)
            entries = saved.get(side, {})
            if key in entries:
                setattr(config, attr, entries[key])
                _logger.info(
                    "UMI %s: loaded tracker→gripper calibration for tracker %r (%s).",
                    side,
                    key,
                    reason,
                )
            elif LEGACY_TRACKER_KEY in entries:
                setattr(config, attr, entries[LEGACY_TRACKER_KEY])
                _logger.warning(
                    "UMI %s: no calibration for the active tracker %r (%s) — "
                    "falling back to a LEGACY entry measured with an unknown "
                    "tracker. If the tracker changed since, this transform "
                    "is wrong; re-run `axol umi.calibrate`.",
                    side,
                    key,
                    reason,
                )
            else:
                _logger.warning(
                    "UMI %s: NO tracker→gripper calibration for the active "
                    "tracker %r (%s) in %s — absolute mode will absorb the "
                    "whole controller→gripper offset into the engage "
                    "snapshot, so recorded TCP poses will be mount-dependent "
                    "and wrist rotations will smear into position error. Run "
                    "`axol umi.calibrate` before collecting data.",
                    side,
                    key,
                    reason,
                    UMI_TCP_TRANSFORM_FILE,
                )
