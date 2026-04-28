"""Dataclasses for per-joint impedance gains, friction parameters, and robot configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass
class JointGains:
    """Impedance control gains and friction parameters for one joint.

    Attributes:
        kp:   Position stiffness [0, 500]
        kd:   Velocity damping   [0, 5]
        fc:   Coulomb friction magnitude (Nm)
        k:    Tanh sharpness factor for friction model
        fv:   Viscous friction coefficient (Nm·s/rad)
        fo:   Constant friction offset (Nm)
        ga:   Gravity feedforward cosine coefficient: τ_grav = ga·cos(q) + gb·sin(q)
        gb:   Gravity feedforward sine coefficient
    """

    kp: float
    kd: float
    fc: float
    k: float
    fv: float
    fo: float
    ga: float
    gb: float


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
    """Per-joint impedance control gains and friction parameters for one arm.

    All joints default to zero gains. Override individual joints at construction
    or use ``dataclasses.replace()`` for partial updates::

        from dataclasses import replace

        left = replace(ArmConfig(), elbow=JointGains(kp=20.0, kd=2.0))
        config = AxolConfig(left=left)
        async with Axol(config=config) as axol: ...

    The feedforward torque sent to each motor is computed automatically from the
    friction parameters (fc, k, fv, fo) and gravity compensation; t_ff is not
    specified directly.
    """

    shoulder_1: JointGains = field(
        default_factory=lambda: JointGains(
            kp=500.0,
            kd=5.0,
            fc=1.2588,
            k=892.72,
            fv=4.0400,
            fo=-0.2332,
            ga=0.2936,
            gb=24.8329,
        )
    )
    shoulder_2: JointGains = field(
        default_factory=lambda: JointGains(
            kp=500.0,
            kd=5.0,
            fc=1.8254,
            k=142.20,
            fv=3.6122,
            fo=-2.5386,
            ga=2.1617,
            gb=24.8353,
        )
    )
    shoulder_3: JointGains = field(
        default_factory=lambda: JointGains(
            kp=250.0,
            kd=2.0,
            fc=0.7140,
            k=89.62,
            fv=2.1274,
            fo=-0.0028,
            ga=0.0484,
            gb=0.0633,
        )
    )
    elbow: JointGains = field(
        default_factory=lambda: JointGains(
            kp=100.0,
            kd=2.0,
            fc=0.9459,
            k=760.52,
            fv=1.0965,
            fo=-0.1867,
            ga=-0.1780,
            gb=5.6560,
        )
    )
    wrist_1: JointGains = field(
        default_factory=lambda: JointGains(
            kp=150.0,
            kd=1.0,
            fc=0.5977,
            k=72.71,
            fv=1.2183,
            fo=0.1210,
            ga=-0.0744,
            gb=-0.0047,
        )
    )
    wrist_2: JointGains = field(
        default_factory=lambda: JointGains(
            kp=150.0,
            kd=2.5,
            fc=0.1171,
            k=796.16,
            fv=1.0274,
            fo=0.0833,
            ga=0.0273,
            gb=0.3204,
        )
    )
    wrist_3: JointGains = field(
        default_factory=lambda: JointGains(
            kp=100.0,
            kd=0.8,
            fc=0.1311,
            k=175.50,
            fv=0.5887,
            fo=0.0486,
            ga=-0.0191,
            gb=0.6200,
        )
    )
    gripper: PositionForceConfig = field(
        default_factory=lambda: PositionForceConfig(torque_limit=1.0, max_speed=10.0)
    )

    def mirror_gravity(self) -> ArmConfig:
        """Return a copy with gb negated for shoulder_2 and elbow.

        shoulder_2 and elbow have mirrored angle ranges on the right arm
        (e.g. elbow: [0, 0.42τ] left vs [-0.42τ, 0] right).  Because
        sin(-q) = -sin(q), the gb coefficient must flip sign while ga
        (cosine term) stays the same.  All other joints have symmetric
        limits so their gravity fits are valid for both arms unchanged.
        """
        return replace(
            self,
            shoulder_2=replace(self.shoulder_2, gb=-self.shoulder_2.gb),
            elbow=replace(self.elbow, gb=-self.elbow.gb),
        )


@dataclass
class AxolConfig:
    """Top-level configuration for both arms and grippers.

    ``right`` defaults to ``left`` with gravity terms mirrored for
    shoulder_2 and elbow; override either arm independently::

        from dataclasses import replace

        config = AxolConfig(left=replace(ArmConfig(), elbow=JointGains(kp=20.0, kd=2.0)))
        async with Axol(config=config) as axol: ...

    Attributes:
        left: Per-joint gains, friction parameters, and gripper config for the left arm.
        right: Same as ``left`` but with mirrored gravity terms.
    """

    left: ArmConfig = field(default_factory=ArmConfig)
    right: ArmConfig = field(default_factory=lambda: ArmConfig().mirror_gravity())
