from __future__ import annotations

from dataclasses import dataclass, field


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

    shoulder_1: JointGains = field(default_factory=lambda: JointGains(kp=600.0, kd=5.0))
    shoulder_2: JointGains = field(default_factory=lambda: JointGains(kp=600.0, kd=5.0))
    shoulder_3: JointGains = field(default_factory=lambda: JointGains(kp=250.0, kd=2.0))
    elbow: JointGains = field(default_factory=lambda: JointGains(kp=100.0, kd=2.0))
    wrist_1: JointGains = field(
        default_factory=lambda: JointGains(
            kp=150.0, kd=1.0, fc=0.4052, k=1000.0, fv=1.0126, fo=0.0670
        )
    )
    wrist_2: JointGains = field(
        default_factory=lambda: JointGains(
            kp=150.0, kd=2.5, fc=0.9011, k=1000.0, fv=0.5591, fo=0.0
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
    gripper: JointGains = field(default_factory=JointGains)
