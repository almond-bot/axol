"""Per-joint and per-arm configuration dataclasses.

A single :class:`JointConfig` carries everything needed to drive one arm
joint: impedance gains (``kp``, ``kd``), the friction-compensation model
(:class:`FrictionParams`), and the inertial of the body that joint drives
(``mass`` and ``com`` — the latter expressed in the body's URDF link frame,
used by :class:`almond_axol.robot.gravity.GravityCompensator` to compute
gravity feedforward).

:class:`ArmConfig` bundles the seven per-joint configs and a
:class:`PositionForceConfig` for the gripper. :class:`AxolConfig` holds the
left and right :class:`ArmConfig` plus a few global safety knobs. Defaults
encode the production-tuned values; override individual fields at
construction or via :func:`dataclasses.replace`::

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
        kd:       Velocity damping for impedance control [0, 5].
        friction: Parameters of the friction-compensation model.
        mass:     Mass of the body driven by this joint (kg). For ``wrist_3``
                  this includes the gripper assembly (fixed-jointed to
                  ``*_w2``).
        com:      Centre of mass of the same body, in the body's URDF link
                  frame (m).
    """

    kp: float
    kd: float
    friction: FrictionParams
    mass: float
    com: tuple[float, float, float]


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
            kp=500.0,
            kd=5.0,
            friction=_ZERO_FRICTION,
            mass=2.00,
            com=(0.0652231, 0.0, 0.0),
        )
    )
    shoulder_2: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=500.0,
            kd=5.0,
            friction=_ZERO_FRICTION,
            mass=1.50,
            com=(0.0, 0.0115864, -0.0302711),
        )
    )
    shoulder_3: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=250.0,
            kd=2.0,
            friction=_ZERO_FRICTION,
            mass=2.75,
            com=(0.0, 0.00286547, -0.164964),
        )
    )
    elbow: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=100.0,
            kd=2.0,
            friction=_ZERO_FRICTION,
            mass=0.80,
            com=(-0.0256064, 0.0, -0.072044),
        )
    )
    wrist_1: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=150.0,
            kd=1.0,
            friction=_ZERO_FRICTION,
            mass=0.50,
            com=(0.0, 0.0, -0.0614121),
        )
    )
    wrist_2: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=150.0,
            kd=2.5,
            friction=_ZERO_FRICTION,
            mass=0.60,
            # left_w1 CoM (right side has y sign-flipped — done by mirror_to_right).
            com=(0.0, 0.0285, -0.0285),
        )
    )
    wrist_3: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=100.0,
            kd=0.8,
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
    """

    left: ArmConfig = field(
        default_factory=lambda: _build_arm(_LEFT_FRICTION, is_left=True)
    )
    right: ArmConfig = field(
        default_factory=lambda: _build_arm(_RIGHT_FRICTION, is_left=False)
    )
    max_step_rad: float = 0.5
