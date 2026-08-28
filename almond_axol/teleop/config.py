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
        frequency: Control (CAN command) loop rate in Hz used by
            :meth:`VRTeleop.run` and as waypoint density for reset
            trajectories. Each cycle sends 8 impedance/position commands per
            arm bus whose replies carry all feedback — ~16 frames ≈ 2.1 ms
            of bus time — so 120 Hz runs each 1 Mbps arm bus at ~25%
            utilisation. A higher rate mainly buys the host damping loop
            phase margin (its transport delay is one cycle).
        ik_frequency: IK dispatch rate in Hz — how often VR frames are sent
            to the IK subprocess. Decoupled from ``frequency``: solves cost
            5-10 ms, so the solver can't follow the CAN rate; the control
            loop interpolates between solutions (segment playback + the
            trapezoidal output filter).
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
            trapezoidal filter.  Defaults to 3.5 rev/s² (~1260 °/s²).  Since
            the filter became a critically damped linear tracker, smoothness
            comes from the tracker itself, not this clamp: replaying recorded
            teleop showed resonance-band excitation identical at 2.0 and
            3.5 rev/s², while the lower cap saturated ~4% of moving time —
            felt as lag-then-surge "jumpiness" on fast moves (vibration
            windows correlated 5.5× with saturation). Keep it high enough
            that genuine motion never saturates; it is a safety ceiling.
        ik_alpha: Blend factor for the exponential moving average applied to
            the IK output before the trapezoidal filter.  Range ``(0, 1]``
            where ``1.0`` disables smoothing.  Lower values kill more
            high-frequency jitter at the cost of a small fixed lag
            (``~(1-alpha)/alpha`` frames).  Defaults to ``0.3`` (~20 ms lag
            at 120 Hz), favouring smoothness over minimum latency.
        pose_cutoff: Pole frequency (Hz) of the lag-compensated low-pass
            applied to raw VR controller positions, quaternions, and elbow
            positions **before** they enter the IK solver (see
            :class:`~almond_axol.teleop.filter.LagCompensatedLowPass` for why
            this replaced the One Euro filter, and for the real-session
            replay that set this value and the filter's lag-compensation
            fraction).  Lower values reject more tremor but the trade is
            steep — the raw stream's 3-12 Hz noise grows with hand speed
            and sits barely 1.5 octaves above intentional motion — so this
            stage only trims the band (~81% pass at 2.5 Hz); the resonance
            itself is handled by pose-tracked host damping on the robot
            side.
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
        record: File prefix for the teleop flight recorder (see
            :mod:`almond_axol.teleop.recorder`).  A bare name (``rec1``)
            records into ``~/.almond/recordings/``, where ``axol
            motion.build`` finds it; a path prefix (``/tmp/jit``) is used
            verbatim.  When set, every stage of the teleop pipeline — raw
            VR pose, filtered pose, IK output, smoothed command, measured
            joints — is captured to ``<prefix>_{ik,cmd,meas}.npz``. The
            measured stage runs at the native 240 Hz Rust-core rate. The
            capture covers the **latest engage→disengage segment** (last
            ~5 minutes of it): recording starts at engagement, disengaging
            writes the files, and re-engaging starts the segment over.  The
            same prefix overwrites on the next run.  For ``axol
            motion.build`` or ``axol diag.offline``.  ``None`` (the
            default) disables recording entirely.
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
    ik_frequency: float = 120.0
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
    pose_cutoff: float = 2.5
    position_multiplier: float = 1.0
    rotation_multiplier: float = 1.0
    disengage_timeout: float = 0.5
    record: str | None = None
