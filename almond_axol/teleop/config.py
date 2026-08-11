"""VRTeleopConfig dataclass with all tunable parameters for a VRTeleop session."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


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
        disengage_timeout: Auto-disengage when no VR pose frame has arrived
            for this many seconds while teleop is engaged — the operator left
            VR (headset doffed, session exited without pressing X/Y) or the
            link dropped.  Without it the engage toggle survives the gap, so
            the next frames after re-entering VR are tracked against the old
            engage snapshot and the arms jerk toward wherever the controllers
            now are.  The headset streams poses at 72+ Hz while presenting,
            so anything beyond a few hundred ms is a real gap, not jitter.
            ``0`` disables the timeout.  Defaults to ``0.5`` s.
        reset_on_disconnect: Return the arms to the rest pose when the
            operator's last pose-sending connection closes while the arms are
            away from rest — quitting the VR app (e.g. via the Quest menu)
            can't reliably deliver the Y-exit reset frame.  Pausing without
            quitting (system menu, doffed headset) keeps the socket open and
            only auto-disengages; the arms hold in place.  Defaults to True.
        exit_reset_timeout: Also return the arms to rest when no VR pose
            frame has arrived for this many seconds while the arms are away
            from rest.  This is the backstop for exits whose socket close is
            delayed or lost — a killed headset app whose TCP FIN never went
            out (WiFi power-save after quitting, a crash, a link drop) —
            and it means stepping away mid-session parks the arms safely.
            Long enough that a normal Quest-menu visit doesn't trigger it.
            ``0`` disables.  Defaults to ``10`` s.
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
    disengage_timeout: float = 0.5
    reset_on_disconnect: bool = True
    exit_reset_timeout: float = 10.0
