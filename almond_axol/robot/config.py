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


@dataclass
class ArmConfig:
    """Per-joint configuration for a single arm.

    Each ``shoulder_*`` / ``elbow`` / ``wrist_*`` field is a
    :class:`JointConfig` carrying gains, friction model, and the inertial of
    the URDF body that joint drives. ``gripper`` is a
    :class:`PositionForceConfig` (gripper mass is already lumped into
    ``wrist_3.mass``).

    The defaults below encode the **left** arm's tuned values. Use
    :class:`AxolConfig` to get a paired left/right configuration with
    mirrored CoMs on the right side. Per-link masses come from the Onshape
    CAD geometry but are tuned in place against measured joint torques —
    typically lower than the CAD values because Onshape often over-assigns
    aluminum-class densities to parts that are hollow / 3D-printed.
    """

    shoulder_1: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=500.0,
            kd=5.0,
            friction=FrictionParams(fc=1.2588, k=892.72, fv=4.0400, fo=-0.2332),
            mass=2.00,
            com=(0.0652231, 0.0, 0.0),
        )
    )
    shoulder_2: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=500.0,
            kd=5.0,
            friction=FrictionParams(fc=1.8254, k=142.20, fv=3.6122, fo=-2.5386),
            mass=1.50,
            com=(0.0, 0.0115864, -0.0302711),
        )
    )
    shoulder_3: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=250.0,
            kd=2.0,
            friction=FrictionParams(fc=0.7140, k=89.62, fv=2.1274, fo=-0.0028),
            mass=2.75,
            com=(0.0, 0.00286547, -0.164964),
        )
    )
    elbow: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=100.0,
            kd=2.0,
            friction=FrictionParams(fc=0.9459, k=760.52, fv=1.0965, fo=-0.1867),
            mass=0.80,
            com=(-0.0256064, 0.0, -0.072044),
        )
    )
    wrist_1: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=150.0,
            kd=1.0,
            friction=FrictionParams(fc=0.5977, k=72.71, fv=1.2183, fo=0.1210),
            mass=0.50,
            com=(0.0, 0.0, -0.0614121),
        )
    )
    wrist_2: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=150.0,
            kd=2.5,
            friction=FrictionParams(fc=0.1171, k=796.16, fv=1.0274, fo=0.0833),
            mass=0.60,
            # left_w1 CoM (right side has y sign-flipped — done by mirror_to_right).
            com=(0.0, 0.0285, -0.0285),
        )
    )
    wrist_3: JointConfig = field(
        default_factory=lambda: JointConfig(
            kp=100.0,
            kd=0.8,
            friction=FrictionParams(fc=0.1311, k=175.50, fv=0.5887, fo=0.0486),
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


@dataclass
class AxolConfig:
    """Top-level configuration for both arms and grippers.

    The ``left`` and ``right`` :class:`ArmConfig` instances have identical
    gains, friction, and masses by default, but mirrored CoMs — the right
    arm is generated by :meth:`ArmConfig.mirror_to_right`. Pass an explicit
    ``right=`` argument if you want asymmetric tuning.

    Attributes:
        left:         Per-joint config for the left arm.
        right:        Per-joint config for the right arm. Defaults to the
                      left arm with CoMs mirrored across X.
        max_step_rad: Maximum allowed change in any arm joint (rad) between
                      consecutive ``motion_control`` calls. Commands that
                      exceed this are dropped and a warning is logged. Set
                      to ``float('inf')`` to disable.
    """

    left: ArmConfig = field(default_factory=ArmConfig)
    right: ArmConfig = field(default_factory=lambda: ArmConfig().mirror_to_right())
    max_step_rad: float = 0.5
