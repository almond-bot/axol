from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class JointGains:
    """MIT impedance control gains for one joint.

    Attributes:
        kp:   Position stiffness [0, 500]
        kd:   Velocity damping   [0, 5]
        t_ff: Feedforward torque (Nm)
    """

    kp: float = 0.0
    kd: float = 0.0
    t_ff: float = 0.0


@dataclass
class AxolConfig:
    """Per-joint impedance control gains for the full arm.

    All joints default to zero gains. Override individual joints at construction
    or use ``dataclasses.replace()`` for partial updates::

        from dataclasses import replace

        config = replace(AxolConfig(), elbow=JointGains(kp=20.0, kd=2.0))
        async with Axol(config=config) as axol: ...
    """

    shoulder_1: JointGains = field(default_factory=JointGains)
    shoulder_2: JointGains = field(default_factory=JointGains)
    shoulder_3: JointGains = field(default_factory=JointGains)
    elbow: JointGains = field(default_factory=JointGains)
    wrist_1: JointGains = field(default_factory=JointGains)
    wrist_2: JointGains = field(default_factory=JointGains)
    wrist_3: JointGains = field(default_factory=JointGains)
    gripper: JointGains = field(default_factory=JointGains)
