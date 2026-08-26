"""Per-joint and per-arm configuration dataclasses.

A single :class:`JointConfig` carries everything needed to drive one arm
joint: impedance gains (``kp``, ``kd``), the friction-compensation model
(:class:`FrictionParams`), and the inertial of the body that joint drives
(``mass`` and ``com`` — the latter expressed in the body's URDF link frame,
used by :class:`almond_axol.robot.gravity.GravityCompensator` to compute
gravity feedforward).

:class:`ArmConfig` bundles the seven per-joint configs and a
:class:`PositionForceConfig` for the gripper. :class:`AxolConfig` holds the
left and right :class:`ArmConfig` plus a few global knobs. Defaults encode
the production-tuned values; override individual fields at construction or
via :func:`dataclasses.replace`::

    from almond_axol.robot.config import AxolConfig, FrictionParams

    config = AxolConfig()
    config.left.elbow.kp = 200
    config.left.elbow.mass = 0.6
    config.left.elbow.com = (-0.025, 0.0, -0.07)
    config.left.elbow.friction = FrictionParams(fc=0.4, k=10.0, fv=0.05, fo=0.0)
    async with Axol(config=config) as axol: ...
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ..constants import ARM_JOINTS
from .calibration import load_calibration, load_factory_calibration


@dataclass
class FrictionParams:
    """tanh-Coulomb + viscous friction model.

    ``τ_friction = fc · tanh(k · v) + fv · v + fo``

    where ``v`` is the joint velocity (rad/s).

    Attributes:
        fc: Coulomb friction magnitude (Nm).
        k:  Tanh sharpness factor — larger is closer to a sign() function.
        fv: Viscous friction coefficient (Nm·s/rad).
        fo: Constant friction offset (Nm). Captures direction-independent
            biases such as imperfect gravity compensation or motor cogging.
    """

    fc: float
    k: float
    fv: float
    fo: float


@dataclass
class JointConfig:
    """Full per-joint configuration: gains + friction + driven body inertial.

    Each arm joint drives exactly one URDF body; ``mass`` and ``com`` describe
    that body in its own link frame. The gravity compensator (see
    :class:`almond_axol.robot.gravity.GravityCompensator`) reads these to
    overwrite the placeholder inertials in the bundled URDF.

    Attributes:
        kp:       Position stiffness for impedance control [0, 500].
        kd:       Velocity damping for impedance control, encoded against
                  [0, 5] on every motor family and firmware. (The MyActuator
                  V4.4 changelog claims a widened 0-50 range, but hardware
                  decodes the 12-bit field against 0-5 on all versions; the
                  historical defaults here were tuned under the wrong 0-50
                  encoding and have been divided by 10, preserving the exact
                  wire bits and therefore the tuned behavior.)
        friction: Parameters of the friction-compensation model.
        mass:     Mass of the body driven by this joint (kg). For ``wrist_3``
                  this includes the gripper assembly (fixed-jointed to
                  ``*_w2``).
        com:      Centre of mass of the same body, in the body's URDF link
                  frame (m).
        j_eff:    Effective scalar inertia (kg·m²) for acceleration
                  feedforward: ``τ = j_eff · q̈_des`` is added to ``t_ff``
                  so inertia is not driven through tracking error.
        kd_host:  Host-side velocity damping (Nm·s/rad):
                  ``τ = kd_host · (v_des − v_meas)`` is added to ``t_ff``,
                  with ``v_meas`` differentiated from measured positions at
                  the command rate. Needed on the high-inertia shoulders,
                  where the motor firmware's internal velocity estimate is
                  too filtered to damp the ~2 Hz closed-loop resonance —
                  measured on left shoulder_2 at kp=250: firmware kd=3.5
                  (tuned as "35" under the old 0-50 encoding) alone left a
                  62%-overshoot ring; kd_host=30 on top damped it
                  critically. The elbow needs its own dose for the same
                  reason, band-passed at its own mode (see ``kd_host_hz``).
                  Leave at 0 for joints whose firmware kd works (the wrists)
                  — and beware the failure mode that took shoulder_3's and
                  wrist_2's dampers away: past ~90 deg of loop phase lag the
                  "damping" torque arrives with the motion instead of
                  against it, and a damper whose band leaks onto a
                  structural mode above its centre *powers* that mode (see
                  the shoulder_3 comment below).
                  This value is the *max-inertia-pose* anchor: at runtime
                  the controller scales it by J(q)/J_ref, where J_ref is
                  the per-joint maximum reflected inertia over arm shapes
                  (URDF mass matrix), tapering it toward 0 in poses where
                  the joint's reflected inertia collapses (e.g. shoulder_1
                  with the arm raised to the side, or shoulder_3 at rest) —
                  there the mode is fast, the stale host torque arrives out
                  of phase, and un-scheduled kd_host measurably sustains
                  jitter (see ``AxolArm.motion_control``).
                  Never raise a joint's value past the highest verified
                  stable on its hardware without a step/teleop check —
                  the shipped shoulder 45s *are* that ceiling (60
                  re-sustained shoulder_2's ring): host damping runs at
                  the command rate on a one-cycle-stale velocity, so it
                  only works on modes well below that rate, and broadband
                  host damping at the old 120 Hz rate *fed* the elbow's
                  ~11 Hz mode (host-kd 39 there diverged violently; even
                  10 sustained a limit cycle on a full-size step).
        kd_host_hz: Centre frequency (Hz) of the band-pass confining this
                  joint's host damping to its resonance band (see ``BandPass``
                  in :mod:`almond_axol.robot.control`; the control math
                  works in rad/s internally — converted once at arm
                  construction). A damper centred on the wrong mode is
                  rolled off and phase-shifted exactly where the joint needs
                  it: with a fixed centre at the shoulders' 3.2 Hz *rest*
                  resonance, teleop bursts at 4.3-8.6 Hz (jit14/15 surveys)
                  saw only ~14% of the damping (-47° band-pass phase, -23°
                  differentiator, -12° loop delay). ``None`` (the default)
                  makes the centre *track the pose*: the shoulders' mode is
                  the impedance mode ωn = √(kp/J(q)) — validated against
                  hardware (2.2 Hz predicted at rest vs ~2-3 measured;
                  5.4 Hz raised to the side vs the 4.3-8.6 Hz burst band) —
                  so motion_control scales the hardware-anchored rest centre
                  (``DAMP_BP_W0``, 3.2 Hz) by √(J_rest/J(q)) each cycle.
                  Set an explicit value for joints whose ringing is
                  *structural* rather than the impedance mode (the elbow
                  rings at 6.7 Hz under load where √(kp/J) predicts 2.3 Hz —
                  transmission compliance, not reflected inertia).
        kd_host_q: Quality factor of that band-pass (bandwidth = centre/q).
                  ``None`` uses the shared default (``DAMP_BP_Q``, 0.8),
                  which keeps the band about an octave wide either side —
                  right for a pose-tracked centre that only *estimates* the
                  mode. The wide band is also the mechanism behind the
                  damping↔accuracy trade-off: centred low it reaches into
                  the <1.5 Hz intentional-motion band and drags the final
                  approach (measured on shoulder_1 as a step that never
                  enters the 5% settle band — a 0.2° RMS sub-Hz wander, not
                  a ring). When ``kd_host_hz`` is pinned on a *measured*
                  ring frequency, set q ≈ 2-3 to confine the damping to the
                  ring and release the approach.
    """

    kp: float
    kd: float
    friction: FrictionParams
    mass: float
    com: tuple[float, float, float]
    j_eff: float = 0.0
    kd_host: float = 0.0
    kd_host_hz: float | None = None
    kd_host_q: float | None = None


@dataclass
class PositionForceConfig:
    """Position-force control parameters.

    Attributes:
        torque_limit: Peak output torque (Nm).
        max_speed:    Maximum joint speed (rad/s).
    """

    torque_limit: float
    max_speed: float


# Placeholder used in :class:`ArmConfig` defaults. Real per-arm friction
# values are injected by :class:`AxolConfig` via the ``_LEFT_FRICTION`` /
# ``_RIGHT_FRICTION`` maps below.
_ZERO_FRICTION = FrictionParams(fc=0.0, k=1.0, fv=0.0, fo=0.0)


@dataclass
class ArmConfig:
    """Per-joint configuration for a single arm.

    Each ``shoulder_*`` / ``elbow`` / ``wrist_*`` field is a
    :class:`JointConfig` carrying gains, friction model, and the inertial of
    the URDF body that joint drives. ``gripper`` is a
    :class:`PositionForceConfig` (gripper mass is already lumped into
    ``wrist_3.mass``).

    Defaults encode the gains, masses, and CoMs that are common to both
    arms. **Friction defaults to zero** — the real per-arm friction values
    are supplied by :class:`AxolConfig` at construction (left and right
    motors differ enough that there is no meaningful "shared" default).
    Per-link masses come from the Onshape CAD geometry but are tuned in
    place against measured joint torques — typically lower than the CAD
    values because Onshape often over-assigns aluminum-class densities to
    parts that are hollow / 3D-printed.

    These ``kp`` / ``kd`` are the tuned **midpoint** (``s=0.5``, the
    production default) of the :attr:`AxolConfig.left_stiffness` /
    :attr:`AxolConfig.right_stiffness` blend — the slider softens them
    toward :data:`_SOFT_GAINS` below 0.5 and stiffens them toward
    :data:`_STIFF_GAINS` above.
    """

    # Gains below were identified on the reference robot (both arms, which
    # converged on the same values) with ``axol tune.pid``: kp from the knee
    # of the sine-tracking error curve, kd (+ kd_host where firmware kd
    # can't deliver, see the :class:`JointConfig` docstring) from step
    # response (<2% overshoot, minimal settling), j_eff from minimizing
    # sine RMS at fixed gains. ``axol tune.pid --save`` / ``axol
    # tune.friction --save`` store per-robot values that override these.
    shoulder_1: JointConfig = field(
        default_factory=lambda: JointConfig(
            # Step/sine-validated on hardware (2026-08, tuning workbench):
            # kp 350 / kd 5 with the host damping pinned on the measured
            # ~2 Hz impedance ring (kd_host_hz) and narrowed to it
            # (kd_host_q 1.5 — the shared 0.8 band reaches into the <1.5 Hz
            # intentional-motion band and drags the final approach). Result
            # at 80° under full gravity load: no ring, 0.15° lag-free sine
            # tracking, parked error at the stiction floor.
            kp=350.0,
            kd=5.0,
            friction=_ZERO_FRICTION,
            mass=1.8,
            com=(0.0652231, 0.0, 0.0),
            j_eff=1.27,
            kd_host=45.0,
            kd_host_hz=2.0,
            kd_host_q=1.5,
        )
    )
    shoulder_2: JointConfig = field(
        default_factory=lambda: JointConfig(
            # Step/sine-validated on hardware (2026-08, tuning workbench):
            # same kp/kd as shoulder_1 with the host damping pinned on this
            # joint's own measured ~3.5 Hz impedance ring (an octave above
            # shoulder_1's — lighter link) and narrowed to it (kd_host_q
            # 1.5). kd_host 45 is the phase-lag ceiling: 60 re-sustained the
            # ring, so the guardrail sits at the validated value. Result:
            # step ring dead in ~2 swings with 0.016° RMS parking at rest,
            # and at -80° under full gravity load 0.32° RMS sine tracking
            # (matching shoulder_1) with 0.07° droop. The ~12% amplitude
            # gain on a 1 Hz sine is the resonance's below-band tail —
            # damping can't shrink it, and the teleop trapezoid rate-limits
            # what reaches it.
            kp=350.0,
            kd=5.0,
            friction=_ZERO_FRICTION,
            mass=1.0,
            com=(0.0, 0.0115864, -0.0302711),
            j_eff=1.1,
            kd_host=45.0,
            kd_host_hz=3.5,
            kd_host_q=1.5,
        )
    )
    shoulder_3: JointConfig = field(
        default_factory=lambda: JointConfig(
            # Step-validated on hardware (2026-08, tuning workbench): kp 250 /
            # kd 3 settles a 3° step in 0.07 s with zero overshoot, no
            # detectable ring, and ~0.001° steady-state RMS — all with
            # kd_host 0 (see below). kd 2 at the same kp reintroduced
            # overshoot; kp 200 settled 6x slower.
            kp=250.0,
            kd=3.0,
            friction=_ZERO_FRICTION,
            mass=3.75,
            com=(0.0, 0.00286547, -0.164964),
            j_eff=0.25,
            # No host damping. A kd_host (12, band-passed at 4.8 Hz per the
            # reference robot's 3.9-5.9 Hz extension-jitter band, jit13)
            # actively *pumps* an ~11 Hz structural mode on current builds:
            # torque x velocity measured on two robots showed shoulder_3's
            # damper injecting energy at 11 Hz (its band-pass still passes
            # ~55% there, arriving ~120 deg late — past the 90 deg where
            # damping flips to excitation), lighting up shoulder_2/wrist_2
            # on BOTH arms via the shared mast. Even kd_host=6 re-ignited it
            # (jit19); 0 killed it (jit18). Cost: ~150 mdeg RMS of undamped
            # 3.3 Hz ring during motion. To win that back, damp phase-safely
            # (firmware kd has no host-loop delay) or notch the structural
            # mode in the host path first — do not just restore kd_host. The
            # reference robot's extension jitter, if it returns there, gets a
            # per-robot calibration override, not a shipping default.
        )
    )
    elbow: JointConfig = field(
        default_factory=lambda: JointConfig(
            # Step/sine-validated on hardware (2026-08, tuning workbench):
            # kp 200 / kd 5 under full gravity load at 80° settles a 10°
            # step in 0.07 s (0.8° overshoot, one ~2.5 Hz impedance-mode
            # swing that decays in a cycle) and tracks a 1 Hz sine at
            # 0.19° RMS. No host damping: the elbow settles so fast that
            # its intentional transient occupies the same 2-3 Hz band as
            # its ring — a band-passed kd_host cannot separate them, and
            # every magnitude/centre tried (8 @ 9.5 Hz shipped, 6-20
            # broadband, 6 @ 2.8 Hz narrow) tripled the overshoot and
            # slowed settling with no ring to kill. The historical 6.7 Hz
            # loaded structural shudder (jit16, ~5° p2p during fast
            # reversals) did not reproduce at kd 5 — loaded step and sine
            # both show <7% error energy in the 5-15 Hz band. If it
            # returns on a build, damp it there per-robot (calibration
            # override), not here.
            kp=200.0,
            kd=5.0,
            friction=_ZERO_FRICTION,
            mass=0.25,
            com=(-0.0256064, 0.0, -0.072044),
            j_eff=0.6,
        )
    )
    wrist_1: JointConfig = field(
        default_factory=lambda: JointConfig(
            # Step/sine-validated on hardware (2026-08, tuning workbench):
            # kp 250 / kd 2 settles a 10° step in 0.03 s with a single
            # 0.34° peak (3.4%, no ring — one decaying crest) and parks
            # ~10x closer than kp 180 (9 vs 97 mdeg): wrist_1's error is
            # stiction-dominated (its axis is usually near-parallel to
            # gravity, so it runs unloaded), and the stiffer spring drags
            # through the sticking band. 1 Hz / 10° sine RMS 0.57° vs
            # 0.67° at kp 180, with tracking lag down 15 → 10 ms. kd 1.7
            # rang with visible overshoot at both kp; kd ≥ 2.5 killed the
            # peak but doubled settling. No kd_host: no structural mode
            # ever showed in wrist_1's band, there is nothing to damp.
            kp=250.0,
            kd=2.0,
            friction=_ZERO_FRICTION,
            mass=0.25,
            com=(0.0, 0.0, -0.0614121),
        )
    )
    wrist_2: JointConfig = field(
        default_factory=lambda: JointConfig(
            # Step/sine-validated on hardware (2026-08, tuning workbench,
            # elbow-raised pose so the joint ran under gravity load): kp 200 /
            # kd 5 settles a 10° step in 0.07 s with a single smooth 0.33°
            # crest, no ring, 27 mdeg steady-state (vs 114 ms / 0.51° / 49
            # mdeg at the old kp 130 / kd 3.5) and tracks a 1 Hz / 10° sine
            # at 0.67° RMS at the production 240 Hz loop. kd 4 halved its
            # damping margin: a two-crest ~7 Hz ring at 0.59° reappeared for
            # only a 0.06° sine gain. kd 5 is the Damiao firmware clamp
            # (kd decodes against 0-5); the contact chatter historically
            # seen near the clamp did not reproduce — torque HF was the
            # lowest of the sweep.
            kp=200.0,
            kd=5.0,
            friction=_ZERO_FRICTION,
            mass=0.65,
            com=(0.0, 0.0285, -0.0285),
            # No host damping. A kd_host (1.5, band-passed at 8.8 Hz per the
            # reference robot's 7.8-10 Hz burst band, jit16) sits nearly on
            # top of current builds' ~11 Hz structural mode and pumps it
            # (see shoulder_3 above; wrist_2 rang hardest in that mode as the
            # lightest joint on the shaking mast). If wrist_2 ringing
            # returns, notch the structural mode in the host path before
            # restoring any kd_host.
        )
    )
    wrist_3: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=130.0,
            kd=3.0,
            friction=_ZERO_FRICTION,
            mass=0.75,
            com=(-0.0285, 0.0, -0.089453),
        )
    )
    gripper: PositionForceConfig = field(
        default_factory=lambda: PositionForceConfig(torque_limit=0.5, max_speed=10.0)
    )

    def mirror_to_right(self) -> "ArmConfig":
        """Return a copy with link CoMs mirrored across the X axis.

        Gains, friction, and mass are unchanged. ``com.x`` is sign-flipped on
        every joint, and ``com.y`` is additionally sign-flipped on
        ``wrist_2`` (because the CAD models the wrist-2 link asymmetrically
        per side rather than as a true mirror — see the URDF for details).
        """
        out = replace(
            self,
            shoulder_1=replace(self.shoulder_1, com=_flip_x(self.shoulder_1.com)),
            shoulder_2=replace(self.shoulder_2, com=_flip_x(self.shoulder_2.com)),
            shoulder_3=replace(self.shoulder_3, com=_flip_x(self.shoulder_3.com)),
            elbow=replace(self.elbow, com=_flip_x(self.elbow.com)),
            wrist_1=replace(self.wrist_1, com=_flip_x(self.wrist_1.com)),
            wrist_2=replace(self.wrist_2, com=_flip_x_y(self.wrist_2.com)),
            wrist_3=replace(self.wrist_3, com=_flip_x(self.wrist_3.com)),
        )
        return out


def _flip_x(com: tuple[float, float, float]) -> tuple[float, float, float]:
    return (-com[0], com[1], com[2])


def _flip_x_y(com: tuple[float, float, float]) -> tuple[float, float, float]:
    return (-com[0], -com[1], com[2])


@dataclass(frozen=True)
class _ArmFriction:
    """Per-joint friction values for one physical arm. Field names mirror
    :class:`ArmConfig` so values are injected by attribute (not string key).
    """

    shoulder_1: FrictionParams
    shoulder_2: FrictionParams
    shoulder_3: FrictionParams
    elbow: FrictionParams
    wrist_1: FrictionParams
    wrist_2: FrictionParams
    wrist_3: FrictionParams


# Per-joint friction values measured with ``axol tune.friction`` on the
# reference robot. The two arms share gains, masses, and (after mirroring)
# CoMs, but motor-by-motor friction differs enough to be worth identifying
# per side — and per robot: these are only the *fallback* for machines that
# have not been calibrated. Run ``axol tune.friction --save`` on each new
# Axol to write its own values to ``~/.almond/calibration.json``, which
# overrides these defaults (see :mod:`almond_axol.robot.calibration`).
#
# Shoulders were swept at 0.13–0.38 rad/s only: above ~0.5 rad/s the
# MyActuator torque telemetry on the heavy joints degrades (apparent
# friction *decreases* with speed, scatter grows to ±10 Nm), which is what
# produced the large phantom viscous terms in earlier fits — every joint is
# in fact Coulomb-dominated (fv ≈ 0) except a small real viscous drag on
# wrist_1 and right shoulder_3.
_LEFT_FRICTION = _ArmFriction(
    shoulder_1=FrictionParams(fc=0.9091, k=799.59, fv=0.0, fo=0.37),
    shoulder_2=FrictionParams(fc=1.1378, k=756.76, fv=0.0, fo=-0.3969),
    shoulder_3=FrictionParams(fc=0.3599, k=835.51, fv=0.0, fo=-0.0133),
    elbow=FrictionParams(fc=0.6023, k=855.95, fv=0.0, fo=-0.072),
    wrist_1=FrictionParams(fc=0.3765, k=88.91, fv=0.0298, fo=-0.0115),
    wrist_2=FrictionParams(fc=0.1521, k=780.31, fv=0.0, fo=-0.0152),
    wrist_3=FrictionParams(fc=0.0714, k=927.26, fv=0.0, fo=0.0042),
)

_RIGHT_FRICTION = _ArmFriction(
    shoulder_1=FrictionParams(fc=1.2972, k=742.11, fv=0.0, fo=-0.1557),
    shoulder_2=FrictionParams(fc=1.3950, k=768.06, fv=0.0, fo=0.2082),
    shoulder_3=FrictionParams(fc=0.4377, k=107.94, fv=0.0853, fo=-0.0147),
    elbow=FrictionParams(fc=0.6066, k=784.78, fv=0.0, fo=0.049),
    wrist_1=FrictionParams(fc=0.5245, k=98.58, fv=0.3062, fo=-0.0097),
    wrist_2=FrictionParams(fc=0.1092, k=899.67, fv=0.0, fo=0.0021),
    wrist_3=FrictionParams(fc=0.1172, k=204.46, fv=0.0, fo=0.0042),
)


def _calibrated_joint(jc: JointConfig, entry: dict[str, Any]) -> JointConfig:
    """Overlay one joint's calibration-file entry onto its config."""
    overrides: dict[str, Any] = {
        f: entry[f]
        for f in ("kp", "kd", "j_eff", "kd_host", "kd_host_hz", "kd_host_q")
        if f in entry
    }
    friction = entry.get("friction")
    if friction is not None:
        overrides["friction"] = FrictionParams(**friction)
    com = entry.get("com")
    if com is not None:
        # Fitted by ``axol tune.gravity --save``; already per-side (measured
        # on this arm), so it replaces the mirrored CAD value as-is.
        overrides["com"] = tuple(com)
    return replace(jc, **overrides) if overrides else jc


def _build_arm(friction: _ArmFriction, *, is_left: bool) -> ArmConfig:
    """Build an :class:`ArmConfig` for one side: shared gains + masses, with
    per-side CoMs (mirrored on the right) and per-motor friction injected.

    Values saved by ``axol tune.friction --save`` / ``axol tune.pid --save``
    to this machine's calibration file (see
    :mod:`almond_axol.robot.calibration`) are then overlaid per joint, so a
    calibrated robot uses its own measured friction and gains as the
    defaults. Explicit overrides at construction (draccus dotted flags, the
    panel's Advanced settings) apply on top of the built defaults and
    therefore still win.

    Each :class:`FrictionParams` is copied (``replace()`` with no field
    overrides) so that mutating one config's friction — e.g.
    ``config.left.shoulder_1.friction.fc = 0.5`` — does not aliasing-leak
    back into :data:`_LEFT_FRICTION` / :data:`_RIGHT_FRICTION` and corrupt
    every subsequent :class:`AxolConfig` in the process.
    """
    arm = ArmConfig() if is_left else ArmConfig().mirror_to_right()
    arm = replace(
        arm,
        shoulder_1=replace(arm.shoulder_1, friction=replace(friction.shoulder_1)),
        shoulder_2=replace(arm.shoulder_2, friction=replace(friction.shoulder_2)),
        shoulder_3=replace(arm.shoulder_3, friction=replace(friction.shoulder_3)),
        elbow=replace(arm.elbow, friction=replace(friction.elbow)),
        wrist_1=replace(arm.wrist_1, friction=replace(friction.wrist_1)),
        wrist_2=replace(arm.wrist_2, friction=replace(friction.wrist_2)),
        wrist_3=replace(arm.wrist_3, friction=replace(friction.wrist_3)),
    )
    # Override order: coded defaults ← this robot's factory calibration
    # (fetched from the cloud by ``axol calibration.pull``) ← the local
    # calibration file. A joint present in both merges field-by-field with
    # local values winning, so a locally retuned friction fit shadows the
    # factory's while the factory's com (say) still applies.
    side = "left" if is_left else "right"
    factory = load_factory_calibration()[side]
    local = load_calibration()[side]
    if not factory and not local:
        return arm
    return replace(
        arm,
        **{
            joint.value: _calibrated_joint(
                getattr(arm, joint.value),
                {**factory.get(joint.value, {}), **local.get(joint.value, {})},
            )
            for joint in ARM_JOINTS
            if joint.value in factory or joint.value in local
        },
    )


@dataclass(frozen=True)
class _ArmGains:
    """Per-joint ``(kp, kd)`` tuples for one arm. Field names mirror
    :class:`ArmConfig` so values are looked up by attribute (not string key).
    """

    shoulder_1: tuple[float, float]
    shoulder_2: tuple[float, float]
    shoulder_3: tuple[float, float]
    elbow: tuple[float, float]
    wrist_1: tuple[float, float]
    wrist_2: tuple[float, float]
    wrist_3: tuple[float, float]


# High-``kp`` "industrial robot" gains used as the ``s=1.0`` endpoint of
# :attr:`AxolConfig.left_stiffness` and :attr:`AxolConfig.right_stiffness`.
#
# ``kd`` here stays damping-ratio-consistent with the tuned compliant
# endpoint — ``kd_stiff = kd * sqrt(kp_stiff / kp)``, both taken at the
# compliant (``s=0``) values — and the
# geometric blend in :func:`_blend_joint` then holds the damping ratio at
# every ``s``, so intermediate slider positions stay as well damped as the
# tuned endpoint (verified on left wrist_3: 100/0.8 overshot 23.8% on a 10°
# step, the consistent 100/1.6 overshot 0.5%). ``kp_stiff`` is capped where
# the required ``kd`` would exceed the firmware range, which is [0, 5] on
# every motor family and firmware (Damiao wrist_2: 250 → kd 4.9; MyActuator
# elbow: 200 → kd 4.96 — MyActuator kd values here are on the true 0-5
# wire scale, see ``JointConfig.kd``).
#
# The shoulders are capped at 350 by measured torque chatter, not firmware:
# at kp=500 each shoulder injected ~1.5 Nm RMS of cycle-to-cycle torque
# noise (vs 0.96 at the kp=250 midpoint) — with both arms driven that
# excited a structural torso vibration in the full-robot ROM test — while
# settling only marginally faster than kp=350 (466 vs 544 ms on a 3° step,
# both 0% overshoot). kp=350 holds chatter near the midpoint level (1.08).
#
# wrist_2 is capped at 160 by a measured stability margin, not the damping
# ratio: with Damiao kd hardware-clamped at 5, kp=250 sat at the edge of a
# bistable ~35 Hz limit cycle (±0.55°, ±1.4 Nm sustained, triggered by load
# shifts and latching even at rest — observed on one unit whose wrist_2 had
# less mechanical margin, while its twin stayed quiet). Host-side damping
# can't reach a 35 Hz mode, so the stiff endpoint keeps the loop bandwidth
# (∝ √kp) safely below the phase-lagged region instead.
#
# wrist_1 is capped at 200 by an in-motion structural buzz: sweeping the
# wrist with the elbow at 90° (the loaded ROM pose) excites a ~27 Hz forearm
# mode whose amplitude scales with kp, not kd (kp=300: 0.24–0.27 Nm RMS on
# both arms; lowering kd made it worse, so firmware damping was already
# helping). kp=200/kd=18 cuts the buzz ~35% with identical gravity-hold
# accuracy (0.165° vs 0.159° mean error at the ±135° waypoint), so the
# extra kp bought no stiffness in practice — a gripped payload would lower
# the mode's frequency and amplify it further.
_STIFF_GAINS = _ArmGains(
    shoulder_1=(350.0, 4.14),
    shoulder_2=(350.0, 4.14),
    shoulder_3=(250.0, 2.36),
    # Ratio-consistent elbow kd at kp=200 is 4.96, just inside the
    # universal firmware clamp of 5.
    elbow=(200.0, 4.96),
    wrist_1=(200.0, 1.8),
    wrist_2=(160.0, 3.9),
    wrist_3=(250.0, 2.8),
)

# Soft "hand-guidable" gains used as the ``s=0.0`` endpoint of the blend.
# ``kp`` values are the pre-retune compliant defaults (the soft feel users
# know from earlier releases); ``kd`` is damping-ratio-consistent with the
# tuned midpoint (``kd_soft = kd · sqrt(kp_soft / kp)``, tuned values), same
# rule as :data:`_STIFF_GAINS` above.
_SOFT_GAINS = _ArmGains(
    shoulder_1=(40.0, 1.4),
    shoulder_2=(50.0, 1.57),
    shoulder_3=(45.0, 1.0),
    elbow=(40.0, 2.22),
    wrist_1=(30.0, 0.69),
    wrist_2=(25.0, 1.5),
    wrist_3=(25.0, 0.9),
)


def _blend_joint(
    jc: JointConfig,
    kp_soft: float,
    kd_soft: float,
    kp_stiff: float,
    kd_stiff: float,
    s: float,
) -> JointConfig:
    """Blend one joint's gains along the soft ↔ tuned ↔ stiff slider.

    The joint's own ``kp`` / ``kd`` (the hardware-tuned optimum) anchor the
    **midpoint** ``s=0.5`` — the production default. The lower half blends
    toward the soft hand-guidable endpoint (:data:`_SOFT_GAINS`), the upper
    half toward the industrial stiff endpoint (:data:`_STIFF_GAINS`).

    ``kp`` and ``kd`` interpolate geometrically (log-space — matches how
    stiffness is perceived); with damping-ratio-consistent endpoints the
    blend then holds the damping ratio at every ``s``. On the **soft half**
    ``kd_host`` scales down with ``√(kp(s)/kp)`` — critical damping grows
    with the square root of stiffness. On the **stiff half** it stays at
    the midpoint value: hardware step tests at the stiff endpoint showed
    host-kd 40 and the √kp-scaled 56.6 damp identically (5.4% vs 1.6%
    overshoot, equal torque chatter), and the midpoint values are the
    hardware-verified stability ceilings — scaling past them buys nothing
    and risks the out-of-phase host-damping instability on faster modes.
    ``j_eff`` stays constant on the soft half (it compensates real inertia,
    which matters most at low ``kp``) and scales linearly to 0 at ``s=1``.
    """
    if s <= 0.5:
        u = 1.0 - 2.0 * s  # 1 at the soft endpoint, 0 at the tuned midpoint
        kp_end, kd_end = kp_soft, kd_soft
        j_eff = jc.j_eff
        stiff_half = False
    else:
        u = 2.0 * s - 1.0  # 0 at the tuned midpoint, 1 at the stiff endpoint
        kp_end, kd_end = kp_stiff, kd_stiff
        j_eff = jc.j_eff * (1.0 - u)
        stiff_half = True
    kp_factor = (kp_end / jc.kp) ** u
    host_factor = 1.0 if stiff_half else math.sqrt(kp_factor)
    return replace(
        jc,
        kp=jc.kp * kp_factor,
        kd=jc.kd * (kd_end / jc.kd) ** u,
        j_eff=j_eff,
        kd_host=jc.kd_host * host_factor,
    )


def _normalize_stiffness(s: float | Sequence[float]) -> tuple[float, ...]:
    """Coerce ``s`` to a 7-tuple of per-joint blend factors in ``[0, 1]``.

    Accepts a scalar (broadcast to all 7 joints) or a sequence of length
    ``len(ARM_JOINTS)`` in :data:`almond_axol.constants.ARM_JOINTS` order.
    """
    if isinstance(s, (int, float)):
        if not 0.0 <= float(s) <= 1.0:
            raise ValueError(f"stiffness must be in [0, 1], got {s}")
        return (float(s),) * len(ARM_JOINTS)
    seq = tuple(float(x) for x in s)
    if len(seq) != len(ARM_JOINTS):
        raise ValueError(
            f"per-joint stiffness must have {len(ARM_JOINTS)} values (one "
            f"per joint, excluding the gripper), got {len(seq)}"
        )
    for i, x in enumerate(seq):
        if not 0.0 <= x <= 1.0:
            raise ValueError(
                f"stiffness[{i}] ({ARM_JOINTS[i].value}) must be in [0, 1], got {x}"
            )
    return seq


def _apply_stiffness(arm: ArmConfig, s: float | Sequence[float]) -> ArmConfig:
    """Blend each of ``arm``'s 7 joints along the soft ↔ tuned ↔ stiff slider.

    ``s`` is either a scalar or a 7-tuple in
    :data:`almond_axol.constants.ARM_JOINTS` order (see
    :func:`_normalize_stiffness`). ``0.5`` is the tuned midpoint and returns
    ``arm`` unchanged; lower values blend toward :data:`_SOFT_GAINS`, higher
    toward :data:`_STIFF_GAINS` (see :func:`_blend_joint`).
    """
    factors = _normalize_stiffness(s)
    if all(f == 0.5 for f in factors):
        return arm
    return replace(
        arm,
        **{
            j.value: _blend_joint(
                getattr(arm, j.value),
                *getattr(_SOFT_GAINS, j.value),
                *getattr(_STIFF_GAINS, j.value),
                factors[i],
            )
            for i, j in enumerate(ARM_JOINTS)
        },
    )


@dataclass
class AxolConfig:
    """Top-level configuration for both arms and grippers.

    Each arm is built from the shared :class:`ArmConfig` defaults (gains,
    masses, link CoMs) with side-specific friction values
    (:data:`_LEFT_FRICTION` / :data:`_RIGHT_FRICTION`, both
    :class:`_ArmFriction` instances) injected, and CoMs mirrored across X
    for the right arm. Pass an explicit ``left=`` / ``right=`` argument to
    bypass either default.

    Attributes:
        left:            Per-joint config for the left arm.
        right:           Per-joint config for the right arm.
        has_gripper:     Whether this robot is a gripper-equipped SKU. Set to
                         ``False`` for the gripperless SKU: the gripper motor
                         is never constructed or calibrated, gripper commands
                         (the last element of every ``(8,)`` joint array) are
                         ignored, and gripper reads report ``0.0``. The array
                         shapes of the public API are unchanged.
        max_step_rad:    Maximum allowed change in any arm joint (rad)
                         between consecutive ``motion_control`` calls.
                         Commands that exceed this are dropped and a warning
                         is logged. Set to ``float('inf')`` to disable.
        left_stiffness:  Compliance ↔ stiffness blend for the **left** arm
                         in ``[0, 1]``. Either a scalar (applied to every
                         joint) or 7 values in
                         :data:`almond_axol.constants.ARM_JOINTS` order
                         (gripper excluded). ``0.5`` (default) runs the
                         hardware-tuned per-joint gains; ``0`` softens to
                         the hand-guidable :data:`_SOFT_GAINS`; ``1``
                         stiffens to the industrial :data:`_STIFF_GAINS`.
                         ``kp`` / ``kd`` interpolate geometrically
                         (log-space) in each half; ``j_eff`` holds on the
                         soft half and scales linearly to 0 at ``s=1``. The
                         blend is baked into the ``left`` / ``right`` gains
                         by :meth:`resolved`, which is called once at the
                         robot-construction boundary (``Axol.__init__``).
                         The stiffness fields are left untouched on the
                         config itself, so a serialized :class:`AxolConfig`
                         round-trips cleanly (loading a dumped config and
                         resolving it again is idempotent).
        right_stiffness: Same, for the **right** arm.
    """

    left: ArmConfig = field(
        default_factory=lambda: _build_arm(_LEFT_FRICTION, is_left=True)
    )
    right: ArmConfig = field(
        default_factory=lambda: _build_arm(_RIGHT_FRICTION, is_left=False)
    )
    has_gripper: bool = True
    max_step_rad: float = 0.5
    left_stiffness: float | list[float] = 0.5
    right_stiffness: float | list[float] = 0.5

    def resolved(self) -> "AxolConfig":
        """Return a copy with stiffness baked into the ``left``/``right`` gains.

        Blends each arm along the soft ↔ tuned ↔ stiff slider by its
        stiffness factor (see :func:`_apply_stiffness`) and resets
        ``left_stiffness`` / ``right_stiffness`` to ``0.5`` — the tuned
        midpoint, where the blend is the identity — so the result is
        **idempotent**: calling :meth:`resolved` again is a no-op. This is
        applied once at the single robot-construction boundary
        (``Axol.__init__``) so every consumer sees consistent gains while
        the unresolved config stays safe to serialize and reload.
        """
        return replace(
            self,
            left=_apply_stiffness(self.left, self.left_stiffness),
            right=_apply_stiffness(self.right, self.right_stiffness),
            left_stiffness=0.5,
            right_stiffness=0.5,
        )
