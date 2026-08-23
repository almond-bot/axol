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
from .calibration import load_calibration


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
        kd:       Velocity damping for impedance control. The motor clamps
                  this to its firmware's range — 5 on Damiao and legacy
                  MyActuator, up to 50 on newer (V4.4+) MyActuator firmware
                  (auto-detected on ``enable()``).
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
                  measured on left shoulder_2 at kp=250: firmware kd=35
                  alone left a 62%-overshoot ring; kd_host=30 on top damped
                  it critically. The elbow needs a small dose for the same
                  reason (4 halves its step overshoot). Leave at 0 for
                  joints whose firmware kd works (wrists, shoulder_3).
                  This value is the *max-inertia-pose* anchor: at runtime
                  the controller scales it by J(q)/J_ref, where J_ref is
                  the per-joint maximum reflected inertia over arm shapes
                  (URDF mass matrix), tapering it toward 0 in poses where
                  the joint's reflected inertia collapses (e.g. shoulder_1
                  with the arm raised to the side, or shoulder_3 at rest) —
                  there the mode is fast, the stale host torque arrives out
                  of phase, and un-scheduled kd_host measurably sustains
                  jitter (see ``AxolArm.motion_control``).
        kd_host_max: Hard ceiling on *total* host-side damping, including
                  the firmware-kd spillover added when ``kd`` exceeds the
                  motor's firmware range (see ``_mit_cmd`` in
                  :mod:`almond_axol.robot.axol`). Host damping runs at the
                  ~100 Hz command rate on a one-cycle-stale velocity, so it
                  only works on modes far below that rate: the shoulders'
                  ~2.3 Hz resonance (40 samples/cycle) damps cleanly up to
                  the hardware-verified 40–45, but the elbow rings at
                  ~11 Hz (8 samples/cycle, ~45° phase lag) where host
                  damping *feeds* the oscillation — host-kd 39 there
                  diverged violently and even 10 sustained a limit cycle on
                  a full-size step. Only the motor's internal kHz loop can
                  damp such fast modes, so joints like the elbow must not
                  spill (``kd_host_max == kd_host``). Set only to values
                  verified stable on hardware; ``0`` (default) allows no
                  spillover beyond ``kd_host`` itself. The stiffness blend
                  scales this ceiling with the same ``√kp`` factor as
                  ``kd_host``, keeping the spill allowance damping-ratio-
                  consistent at every slider position.
    """

    kp: float
    kd: float
    friction: FrictionParams
    mass: float
    com: tuple[float, float, float]
    j_eff: float = 0.0
    kd_host: float = 0.0
    kd_host_max: float = 0.0


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
            kp=250.0,
            kd=42.0,
            friction=_ZERO_FRICTION,
            mass=1.8,
            com=(0.0652231, 0.0, 0.0),
            j_eff=1.27,
            kd_host=50.0,
            kd_host_max=50.0,
        )
    )
    shoulder_2: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=250.0,
            kd=42.0,
            friction=_ZERO_FRICTION,
            mass=1.0,
            com=(0.0, 0.0115864, -0.0302711),
            j_eff=1.1,
            kd_host=45.0,
            kd_host_max=45.0,
        )
    )
    shoulder_3: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=180.0,
            kd=24.0,
            friction=_ZERO_FRICTION,
            mass=3.75,
            com=(0.0, 0.00286547, -0.164964),
            j_eff=0.25,
            # If extension jitter returns (arm forward, elbow bent), a small
            # kd_host (~10-15) is the fix: the J(q)/J_ref schedule keeps it
            # near zero at rest (J_rest ≈ 3% of max, fast mode) and ramps
            # it in only where the mode slows into the host loop's band.
        )
    )
    elbow: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=130.0,
            kd=48.0,
            friction=_ZERO_FRICTION,
            mass=0.25,
            com=(-0.0256064, 0.0, -0.072044),
            j_eff=0.6,
            # Extra elbow damping must come from firmware kd (kHz loop):
            # its ~11 Hz ring is too fast for the ~100 Hz host loop —
            # host-kd 39 diverged violently there, and the damping
            # band-pass rolls off well below 11 Hz. Do not raise kd_host.
            kd_host=4.0,
            kd_host_max=4.0,
        )
    )
    wrist_1: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=180.0,
            kd=20.0,
            friction=_ZERO_FRICTION,
            mass=0.25,
            com=(0.0, 0.0, -0.0614121),
        )
    )
    wrist_2: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=130.0,
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
        f: entry[f] for f in ("kp", "kd", "j_eff", "kd_host") if f in entry
    }
    friction = entry.get("friction")
    if friction is not None:
        overrides["friction"] = FrictionParams(**friction)
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
    calibrated = load_calibration()["left" if is_left else "right"]
    if not calibrated:
        return arm
    return replace(
        arm,
        **{
            joint.value: _calibrated_joint(
                getattr(arm, joint.value), calibrated[joint.value]
            )
            for joint in ARM_JOINTS
            if joint.value in calibrated
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
# the required ``kd`` would exceed the firmware range: Damiao clamps kd at 5
# (wrist_2: 250 → kd 4.9), MyActuator V4.4 at 50 (elbow: 200 → kd 50).
# Legacy MyActuator firmware clamps kd at 5; part of the excess is
# delivered host-side instead, up to each joint's ``kd_host_max`` stability
# ceiling (see ``kd_host`` spillover in :mod:`almond_axol.robot.axol`).
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
    shoulder_1=(350.0, 49.7),
    shoulder_2=(350.0, 49.7),
    shoulder_3=(250.0, 28.3),
    # Ratio-consistent elbow kd at kp=200 would be 55.8, but MyActuator
    # V4.4 clamps kd at 50 — the stiff half runs slightly underdamped.
    elbow=(200.0, 50.0),
    wrist_1=(200.0, 18.0),
    wrist_2=(160.0, 3.9),
    wrist_3=(250.0, 2.8),
)

# Soft "hand-guidable" gains used as the ``s=0.0`` endpoint of the blend.
# ``kp`` values are the pre-retune compliant defaults (the soft feel users
# know from earlier releases); ``kd`` is damping-ratio-consistent with the
# tuned midpoint (``kd_soft = kd · sqrt(kp_soft / kp)``, tuned values), same
# rule as :data:`_STIFF_GAINS` above.
_SOFT_GAINS = _ArmGains(
    shoulder_1=(40.0, 16.8),
    shoulder_2=(50.0, 18.8),
    shoulder_3=(45.0, 12.0),
    elbow=(40.0, 26.6),
    wrist_1=(30.0, 8.2),
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
        # The spillover ceiling tracks the same scaling as kd_host, so the
        # soft half keeps its damping-ratio-consistent cap and the stiff
        # half keeps the hardware-verified midpoint ceiling.
        kd_host_max=jc.kd_host_max * host_factor,
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
