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

from dataclasses import dataclass, field, replace


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
        kd:       Velocity damping for impedance control [0, 5]. Hardware-
                  capped by the motor firmware at 5; use ``kd_soft`` to
                  augment.
        friction: Parameters of the friction-compensation model.
        mass:     Mass of the body driven by this joint (kg). For ``wrist_3``
                  this includes the gripper assembly (fixed-jointed to
                  ``*_w2``).
        com:      Centre of mass of the same body, in the body's URDF link
                  frame (m).
        j_eff:    Effective scalar inertia (kg·m²) for acceleration
                  feedforward: ``τ = j_eff · q̈_des`` is added to ``t_ff``
                  so inertia is not driven through tracking error.
        kd_soft:  Extra software velocity damping (Nm·s/rad) applied as
                  ``τ = kd_soft · (v_des − v_meas)``; mathematically
                  equivalent to raising ``kd`` past the firmware's 5 cap.
    """

    kp: float
    kd: float
    friction: FrictionParams
    mass: float
    com: tuple[float, float, float]
    j_eff: float = 0.0
    kd_soft: float = 0.0


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
    """

    shoulder_1: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=40.0,
            kd=5.0,
            friction=_ZERO_FRICTION,
            mass=2.00,
            com=(0.0652231, 0.0, 0.0),
            # Identified from step response (ζ ≈ 0.35 at kp=40, kd=5 → J ≈ 1.27).
            j_eff=1.27,
            # Doubles effective kd past the firmware's 5 cap (ζ ≈ 0.35 → 0.70).
            kd_soft=5.0,
        )
    )
    shoulder_2: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=50.0,
            kd=5.0,
            friction=_ZERO_FRICTION,
            mass=1.50,
            com=(0.0, 0.0115864, -0.0302711),
            # Identified from step response (ζ ≈ 0.36 at kp=50, kd=5 → J ≈ 0.91).
            j_eff=0.91,
            kd_soft=5.0,
        )
    )
    shoulder_3: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=45.0,
            kd=1.0,
            friction=_ZERO_FRICTION,
            mass=2.75,
            com=(0.0, 0.00286547, -0.164964),
        )
    )
    elbow: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=40.0,
            kd=3.0,
            friction=_ZERO_FRICTION,
            mass=0.80,
            com=(-0.0256064, 0.0, -0.072044),
        )
    )
    wrist_1: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=30.0,
            kd=1.0,
            friction=_ZERO_FRICTION,
            mass=0.50,
            com=(0.0, 0.0, -0.0614121),
        )
    )
    wrist_2: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=25.0,
            kd=1.0,
            friction=_ZERO_FRICTION,
            mass=0.60,
            # left_w1 CoM (right side has y sign-flipped — done by mirror_to_right).
            com=(0.0, 0.0285, -0.0285),
        )
    )
    wrist_3: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=25.0,
            kd=0.5,
            friction=_ZERO_FRICTION,
            mass=0.65,
            # left_w2 lumps wrist-3 segment with the gripper assembly (fixed
            # joint): merged CAD inertial is 1.267 kg @ (-0.0285, 0, -0.08945);
            # the mass is tuned in place.
            com=(-0.0285, 0.0, -0.089453),
        )
    )
    gripper: PositionForceConfig = field(
        default_factory=lambda: PositionForceConfig(torque_limit=1.0, max_speed=10.0)
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


# Per-joint friction values measured with ``axol tune.friction``. Each
# instance is the source of truth for one physical arm — the two arms share
# gains, masses, and (after mirroring) CoMs, but motor-by-motor friction
# differs enough to be worth identifying per side. Re-run the tuner on a
# fresh Axol to refresh these.
_LEFT_FRICTION = _ArmFriction(
    shoulder_1=FrictionParams(fc=1.0191, k=723.53, fv=3.3848, fo=0.2853),
    shoulder_2=FrictionParams(fc=1.6873, k=115.41, fv=2.7202, fo=-0.1701),
    shoulder_3=FrictionParams(fc=0.5979, k=106.56, fv=2.1515, fo=0.0242),
    elbow=FrictionParams(fc=0.6806, k=801.34, fv=0.8665, fo=-0.2496),
    wrist_1=FrictionParams(fc=0.5601, k=66.02, fv=1.2435, fo=0.0504),
    wrist_2=FrictionParams(fc=0.2658, k=180.00, fv=0.9962, fo=0.0691),
    wrist_3=FrictionParams(fc=0.1048, k=829.09, fv=0.5857, fo=0.0638),
)

_RIGHT_FRICTION = _ArmFriction(
    shoulder_1=FrictionParams(fc=1.0390, k=781.53, fv=3.5425, fo=0.2861),
    shoulder_2=FrictionParams(fc=1.6873, k=115.41, fv=2.7202, fo=0.1701),
    shoulder_3=FrictionParams(fc=0.4773, k=91.37, fv=1.8673, fo=0.0631),
    elbow=FrictionParams(fc=0.5255, k=159.25, fv=0.8480, fo=0.3607),
    wrist_1=FrictionParams(fc=0.4415, k=80.96, fv=1.3184, fo=0.0497),
    wrist_2=FrictionParams(fc=0.1880, k=813.44, fv=1.1331, fo=0.0252),
    wrist_3=FrictionParams(fc=0.1137, k=852.61, fv=0.5843, fo=0.0345),
)


def _build_arm(friction: _ArmFriction, *, is_left: bool) -> ArmConfig:
    """Build an :class:`ArmConfig` for one side: shared gains + masses, with
    per-side CoMs (mirrored on the right) and per-motor friction injected.
    """
    arm = ArmConfig() if is_left else ArmConfig().mirror_to_right()
    return replace(
        arm,
        shoulder_1=replace(arm.shoulder_1, friction=friction.shoulder_1),
        shoulder_2=replace(arm.shoulder_2, friction=friction.shoulder_2),
        shoulder_3=replace(arm.shoulder_3, friction=friction.shoulder_3),
        elbow=replace(arm.elbow, friction=friction.elbow),
        wrist_1=replace(arm.wrist_1, friction=friction.wrist_1),
        wrist_2=replace(arm.wrist_2, friction=friction.wrist_2),
        wrist_3=replace(arm.wrist_3, friction=friction.wrist_3),
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


# Pre-compliance-tuning gains — the high-``kp`` "industrial robot" defaults
# used as the ``stiffness=1.0`` endpoint of :attr:`AxolConfig.stiffness`.
_STIFF_GAINS = _ArmGains(
    shoulder_1=(500.0, 5.0),
    shoulder_2=(500.0, 5.0),
    shoulder_3=(250.0, 2.0),
    elbow=(100.0, 2.0),
    wrist_1=(150.0, 1.0),
    wrist_2=(150.0, 2.5),
    wrist_3=(100.0, 0.8),
)


def _blend_joint(
    jc: JointConfig, kp_stiff: float, kd_stiff: float, s: float
) -> JointConfig:
    """Blend one joint's gains toward the stiff endpoint by factor ``s``.

    ``kp`` and ``kd`` interpolate geometrically (log-space — matches how
    stiffness is perceived); ``j_eff`` and ``kd_soft`` scale linearly to 0
    since they only compensate for the low-``kp`` regime.
    """
    return replace(
        jc,
        kp=jc.kp * (kp_stiff / jc.kp) ** s,
        kd=jc.kd * (kd_stiff / jc.kd) ** s,
        j_eff=jc.j_eff * (1.0 - s),
        kd_soft=jc.kd_soft * (1.0 - s),
    )


def _apply_stiffness(arm: ArmConfig, s: float) -> ArmConfig:
    """Return ``arm`` with all 7 joints blended toward :data:`_STIFF_GAINS`
    by factor ``s`` ∈ ``[0, 1]``. ``s=0`` returns ``arm`` unchanged.
    """
    if s == 0.0:
        return arm
    return replace(
        arm,
        shoulder_1=_blend_joint(arm.shoulder_1, *_STIFF_GAINS.shoulder_1, s),
        shoulder_2=_blend_joint(arm.shoulder_2, *_STIFF_GAINS.shoulder_2, s),
        shoulder_3=_blend_joint(arm.shoulder_3, *_STIFF_GAINS.shoulder_3, s),
        elbow=_blend_joint(arm.elbow, *_STIFF_GAINS.elbow, s),
        wrist_1=_blend_joint(arm.wrist_1, *_STIFF_GAINS.wrist_1, s),
        wrist_2=_blend_joint(arm.wrist_2, *_STIFF_GAINS.wrist_2, s),
        wrist_3=_blend_joint(arm.wrist_3, *_STIFF_GAINS.wrist_3, s),
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
        left:         Per-joint config for the left arm.
        right:        Per-joint config for the right arm.
        max_step_rad: Maximum allowed change in any arm joint (rad) between
                      consecutive ``motion_control`` calls. Commands that
                      exceed this are dropped and a warning is logged. Set
                      to ``float('inf')`` to disable.
        stiffness:    Compliance ↔ stiffness blend in ``[0, 1]``. ``0``
                      (default) keeps the per-joint compliant gains; ``1``
                      restores the pre-tuning industrial gains in
                      :data:`_STIFF_GAINS`. ``kp`` / ``kd`` interpolate
                      geometrically (log-space); ``j_eff`` / ``kd_soft``
                      scale linearly to 0 at ``s=1``. The blend is baked
                      into ``left`` / ``right`` at construction time —
                      mutate ``stiffness`` after the fact has no effect,
                      and ``replace()`` would re-apply it (don't).
    """

    left: ArmConfig = field(
        default_factory=lambda: _build_arm(_LEFT_FRICTION, is_left=True)
    )
    right: ArmConfig = field(
        default_factory=lambda: _build_arm(_RIGHT_FRICTION, is_left=False)
    )
    max_step_rad: float = 0.5
    stiffness: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.stiffness <= 1.0:
            raise ValueError(f"stiffness must be in [0, 1], got {self.stiffness}")
        if self.stiffness > 0.0:
            self.left = _apply_stiffness(self.left, self.stiffness)
            self.right = _apply_stiffness(self.right, self.stiffness)
