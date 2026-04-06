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
    """

    kp: float = 0.0
    kd: float = 0.0
    fc: float = 0.0
    k: float = 0.0
    fv: float = 0.0
    fo: float = 0.0


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

    shoulder_1: JointGains = field(default_factory=JointGains)
    shoulder_2: JointGains = field(default_factory=JointGains)
    shoulder_3: JointGains = field(default_factory=JointGains)
    elbow: JointGains = field(default_factory=lambda: JointGains(kp=100.0, kd=2.0))
    wrist_1: JointGains = field(default_factory=lambda: JointGains(kp=150.0, kd=1.0))
    wrist_2: JointGains = field(default_factory=lambda: JointGains(kp=100.0, kd=2.5))
    wrist_3: JointGains = field(default_factory=lambda: JointGains(kp=100.0, kd=0.8))
    gripper: JointGains = field(default_factory=JointGains)
