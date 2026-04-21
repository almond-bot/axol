from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass
class JointGains:
    """MIT impedance control gains and friction parameters for one joint.

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

    kp: float = 0.0
    kd: float = 0.0
    fc: float = 0.0
    k: float = 0.0
    fv: float = 0.0
    fo: float = 0.0
    ga: float = 0.0
    gb: float = 0.0


@dataclass
class AxolConfig:
    """Per-joint impedance control gains and friction parameters for the full arm.

    All joints default to zero gains. Override individual joints at construction
    or use ``dataclasses.replace()`` for partial updates::

        from dataclasses import replace

        config = replace(AxolConfig(), elbow=JointGains(kp=20.0, kd=2.0))
        async with Axol(config=config) as axol: ...

    The feedforward torque sent to each motor is computed automatically from the
    friction parameters (fc, k, fv, fo) and gravity compensation; t_ff is not
    specified directly.
    """

    shoulder_1: JointGains = field(
        default_factory=lambda: JointGains(
            kp=600.0,
            kd=5.0,
            fc=0.9277,
            k=689.84,
            fv=3.8339,
            fo=-0.1024,
            ga=0.2402,
            gb=23.1623,
        )
    )
    shoulder_2: JointGains = field(
        default_factory=lambda: JointGains(
            kp=600.0,
            kd=5.0,
            fc=1.5435,
            k=130.94,
            fv=2.7748,
            fo=-2.0204,
            ga=2.0630,
            gb=23.0335,
        )
    )
    shoulder_3: JointGains = field(
        default_factory=lambda: JointGains(
            kp=250.0,
            kd=2.0,
            fc=0.4545,
            k=102.73,
            fv=1.5295,
            fo=0.0917,
            ga=-0.0256,
            gb=0.0225,
        )
    )
    elbow: JointGains = field(
        default_factory=lambda: JointGains(
            kp=100.0,
            kd=2.0,
            fc=0.8210,
            k=918.07,
            fv=0.6716,
            fo=-0.0428,
            ga=-0.2960,
            gb=5.2126,
        )
    )
    wrist_1: JointGains = field(
        default_factory=lambda: JointGains(
            kp=150.0,
            kd=1.0,
            fc=0.4129,
            k=83.02,
            fv=0.9732,
            fo=0.0916,
            ga=-0.0497,
            gb=0.0149,
        )
    )
    wrist_2: JointGains = field(
        default_factory=lambda: JointGains(
            kp=150.0,
            kd=2.5,
            fc=0.1862,
            k=94.51,
            fv=0.7391,
            fo=0.0835,
            ga=0.0315,
            gb=0.3816,
        )
    )
    wrist_3: JointGains = field(
        default_factory=lambda: JointGains(
            kp=100.0,
            kd=0.8,
            fc=0.1287,
            k=842.60,
            fv=0.5523,
            fo=0.0497,
            ga=-0.0304,
            gb=0.5935,
        )
    )
    gripper: JointGains = field(
        default_factory=lambda: JointGains(
            kp=50.0,
            kd=1.0,
            fc=0.1194,
            k=129.10,
            fv=0.2604,
            fo=0.0298,
            ga=0.0002,
            gb=0.0008,
        )
    )

    def mirror_gravity(self) -> AxolConfig:
        """Return a copy with gb negated for shoulder_2 and elbow.

        shoulder_2 and elbow have mirrored angle ranges on the right arm
        (e.g. elbow: [0, 0.42τ] left vs [-0.42τ, 0] right).  Because
        sin(−q) = −sin(q), the gb coefficient must flip sign while ga
        (cosine term) stays the same.  All other joints have symmetric
        limits so their gravity fits are valid for both arms unchanged.
        """
        return replace(
            self,
            shoulder_2=replace(self.shoulder_2, gb=-self.shoulder_2.gb),
            elbow=replace(self.elbow, gb=-self.elbow.gb),
        )
