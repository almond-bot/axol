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
    from almond_axol.robot import Axol
    from almond_axol.rt import RtAxol

    config = AxolConfig()
    config.left.elbow.kp = 200
    config.left.elbow.mass = 0.6
    config.left.elbow.com = (-0.025, 0.0, -0.07)
    config.left.elbow.friction = FrictionParams(fc=0.4, k=10.0, fv=0.05, fo=0.0)
    async with RtAxol(Axol(config=config)) as axol: ...
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
                  ring and release the approach. Production shoulder_1 uses
                  q=3 on both arms even with its pose-tracked centre: hardware
                  traces found a separate 12.5-13.6 Hz mast/forearm mode that
                  the old wide band could feed.
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

    These ``kp`` / ``kd`` are the **top** of the
    :attr:`AxolConfig.left_stiffness` / :attr:`AxolConfig.right_stiffness`
    blend (``s=1.0``, the production default) — they are hardware-tuned
    optima that are also measured ceilings, so the slider only *adds
    compliance*, softening toward :data:`_SOFT_GAINS` as ``s`` drops
    toward 0.
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
            kp=250.0,
            kd=3.5,
            friction=_ZERO_FRICTION,
            mass=1.8,
            com=(0.0652231, 0.0, 0.0),
            j_eff=1.27,
            # Pose-tracked band-pass centre (kd_host_hz None): the shoulder
            # mode is the impedance mode, moving with reflected inertia.
            kd_host=40.0,
        )
    )
    shoulder_2: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=250.0,
            kd=3.5,
            friction=_ZERO_FRICTION,
            mass=1.0,
            com=(0.0, 0.0115864, -0.0302711),
            j_eff=1.1,
            kd_host=35.0,
        )
    )
    shoulder_3: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=180.0,
            # Firmware damping is phase-safe. The synchronized Rust trace
            # caught a load-dependent 3.5 Hz mode at 0.52° RMS versus only
            # 0.13° commanded with kd=4, while the motor torque remained
            # dissipative. Use the firmware maximum to add damping without
            # reviving the delayed host-torque failure below.
            kd=5.0,
            friction=_ZERO_FRICTION,
            mass=3.75,
            com=(0.0, 0.00286547, -0.164964),
            j_eff=0.25,
            # No host damping. Earlier jit18/19 power measurements showed
            # kd_host pumping the coupled 11 Hz mast mode, and kd_host=0
            # killed it. The 240 Hz rust_damping captures reproduce the same
            # failure at 8.7 Hz: shoulder_3 and wrist_2 ring together even
            # though wrist_2 has no host damping. Narrowing shoulder_3 from
            # Q=0.8 to Q=3 did not remove it. Keep damping on the motor side;
            # do not chase the coupled wrist symptom with another host term.
        )
    )
    elbow: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=130.0,
            # The motor-side maximum is the phase-safe lever for the loaded
            # 6.7-12.5 Hz transmission mode.
            kd=5.0,
            friction=_ZERO_FRICTION,
            mass=0.25,
            com=(-0.0256064, 0.0, -0.072044),
            j_eff=0.6,
            # No host damping. The far-forward rust_damping event sits at
            # 8.7-11.3 Hz while this joint's old host band was nearly fully
            # active at 9.55 Hz. Hardware step/replay A/Bs found that term
            # increased overshoot without removing a ring; firmware kd=5
            # settled the joint without the host-loop phase risk.
        )
    )
    wrist_1: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=180.0,
            kd=1.7,
            friction=_ZERO_FRICTION,
            mass=0.25,
            com=(0.0, 0.0, -0.0614121),
        )
    )
    wrist_2: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=130.0,
            # The same trace measured 0.36° RMS at the coupled 3.5 Hz mode
            # versus 0.08° commanded, with materially weaker motor damping
            # than shoulder_3. kd=3.5 was previously replay-verified clean;
            # stop there because kd=5 produced a unit-dependent 110 Hz buzz.
            kd=3.5,
            friction=_ZERO_FRICTION,
            mass=0.65,
            com=(0.0, 0.0285, -0.0285),
        )
    )
    wrist_3: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=130.0,
            kd=2.0,
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
    # Both sides couple shoulder-1 into a ~13 Hz mast/forearm structural
    # mode. With the old wide Q=0.8 shoulder damper, its high-side leakage
    # became phase-positive at that mode: an RT control-term trace measured
    # 0.551 Nm / +0.0255 W and a whole-arm right-side shudder (47/87/99 mdeg
    # RMS at shoulder-1 / elbow / wrist-3); repeated left-side traces expose
    # the same 12.5-12.9 Hz mode. Q=3 keeps unity gain at the intended,
    # pose-tracked ~3.2 Hz shoulder mode while confining it enough to cut the
    # right-side figures to 6/10/9 mdeg in an otherwise-identical hardware
    # A/B. Apply it symmetrically: the structure and control law are shared,
    # and leaving one side at Q=0.8 merely moves the excitation risk there.
    # This shared config feeds every production control path; factory and
    # local calibration entries can still override it.
    arm = replace(
        arm,
        shoulder_1=replace(arm.shoulder_1, kd_host_q=3.0),
    )
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


# There is deliberately no "stiffer than tuned" endpoint: the tuned
# :class:`JointConfig` gains ARE the ``s=1.0`` top of the stiffness slider
# (2026-08-26; the previous scale put them at ``s=0.5`` with a higher-kp
# "industrial" endpoint above). The headroom above them was measured to be
# illusory, joint by joint:
#
# - Shoulders: kp past 350 buys torque chatter, not speed — at kp=500 each
#   shoulder injected ~1.5 Nm RMS of cycle-to-cycle torque noise (vs 0.96
#   at kp=250); with both arms driven that excited a structural torso
#   vibration in the full-robot ROM test, while settling only marginally
#   faster than 350 (466 vs 544 ms on a 3° step, both 0% overshoot).
# - wrist_2: with Damiao kd hardware-clamped at 5, kp=250 sat at the edge
#   of a bistable ~35 Hz limit cycle (±0.55°, ±1.4 Nm sustained, latching
#   even at rest on the unit with less mechanical margin). Host damping
#   can't reach a 35 Hz mode; staying at the tuned kp keeps the loop
#   bandwidth (∝ √kp) below the phase-lagged region.
# - wrist_1: kp past ~200 pumps a ~27 Hz forearm mode whose amplitude
#   scales with kp, not kd (kp=300: 0.24–0.27 Nm RMS on both arms) with
#   identical gravity-hold accuracy — no stiffness actually gained, and a
#   gripped payload would lower the mode's frequency and amplify it.
# - Universal: firmware kd decodes against [0, 5] on every motor family,
#   so damping-ratio-consistent kd for higher kp simply doesn't exist
#   (elbow at kp 200 already needs 4.96).

# Soft "hand-guidable" gains used as the ``s=0.0`` endpoint of the blend.
# ``kp`` values are the pre-retune compliant defaults (the soft feel users
# know from earlier releases); ``kd`` is damping-ratio-consistent with the
# tuned gains (``kd_soft = kd · sqrt(kp_soft / kp)``, tuned values), so the
# geometric blend in :func:`_blend_joint` holds the damping ratio at every
# ``s`` (verified on left wrist_3: 100/0.8 overshot 23.8% on a 10° step,
# the consistent 100/1.6 overshot 0.5%).
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
    s: float,
) -> JointConfig:
    """Blend one joint's gains along the soft ↔ tuned slider.

    The joint's own ``kp`` / ``kd`` (the hardware-tuned optimum, also the
    measured ceiling — see the retired-stiff-endpoint note above
    :data:`_SOFT_GAINS`) sit at ``s=1.0``, the production default; lower
    ``s`` only *adds compliance*, blending toward the soft hand-guidable
    endpoint (:data:`_SOFT_GAINS`) at ``s=0``.

    ``kp`` and ``kd`` interpolate geometrically (log-space — matches how
    stiffness is perceived); with a damping-ratio-consistent soft endpoint
    the blend then holds the damping ratio at every ``s``. ``kd_host``
    scales down with ``√(kp(s)/kp)`` — critical damping grows with the
    square root of stiffness. ``j_eff`` stays constant: it compensates
    real inertia, which matters most at low ``kp``.
    """
    u = 1.0 - s  # 1 at the soft endpoint, 0 at the tuned top
    kp_factor = (kp_soft / jc.kp) ** u
    return replace(
        jc,
        kp=jc.kp * kp_factor,
        kd=jc.kd * (kd_soft / jc.kd) ** u,
        kd_host=jc.kd_host * math.sqrt(kp_factor),
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
    """Blend each of ``arm``'s 7 joints along the soft ↔ tuned slider.

    ``s`` is either a scalar or a 7-tuple in
    :data:`almond_axol.constants.ARM_JOINTS` order (see
    :func:`_normalize_stiffness`). ``1.0`` is the tuned gains and returns
    ``arm`` unchanged; lower values blend toward :data:`_SOFT_GAINS` (see
    :func:`_blend_joint`).
    """
    factors = _normalize_stiffness(s)
    if all(f == 1.0 for f in factors):
        return arm
    return replace(
        arm,
        **{
            j.value: _blend_joint(
                getattr(arm, j.value),
                *getattr(_SOFT_GAINS, j.value),
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
        left_stiffness:  Compliance blend for the **left** arm in
                         ``[0, 1]``. Either a scalar (applied to every
                         joint) or 7 values in
                         :data:`almond_axol.constants.ARM_JOINTS` order
                         (gripper excluded). ``1.0`` (default) runs the
                         hardware-tuned per-joint gains — the stiffest the
                         hardware was measured to support (see the
                         retired-stiff-endpoint note above
                         :data:`_SOFT_GAINS`); lower values only *add
                         compliance*, down to the hand-guidable
                         :data:`_SOFT_GAINS` at ``0``. ``kp`` / ``kd``
                         interpolate geometrically (log-space); ``j_eff``
                         is unchanged by the blend. The blend is baked
                         into the ``left`` / ``right`` gains by
                         :meth:`resolved`, which is called once at the
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
    left_stiffness: float | list[float] = 1.0
    right_stiffness: float | list[float] = 1.0

    def resolved(self) -> "AxolConfig":
        """Return a copy with stiffness baked into the ``left``/``right`` gains.

        Blends each arm along the soft ↔ tuned slider by its stiffness
        factor (see :func:`_apply_stiffness`) and resets
        ``left_stiffness`` / ``right_stiffness`` to ``1.0`` — the tuned
        gains, where the blend is the identity — so the result is
        **idempotent**: calling :meth:`resolved` again is a no-op. This is
        applied once at the single robot-construction boundary
        (``Axol.__init__``) so every consumer sees consistent gains while
        the unresolved config stays safe to serialize and reload.
        """
        return replace(
            self,
            left=_apply_stiffness(self.left, self.left_stiffness),
            right=_apply_stiffness(self.right, self.right_stiffness),
            left_stiffness=1.0,
            right_stiffness=1.0,
        )
