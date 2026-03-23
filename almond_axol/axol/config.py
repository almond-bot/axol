from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_CONFIG = Path(__file__).parent / "config.json"


@dataclass
class JointGains:
    """MIT impedance control gains for one joint.

    Attributes:
        kp:   Position stiffness [0, 500]
        kd:   Velocity damping   [0, 5]
        t_ff: Feedforward torque (Nm)
    """

    kp: float
    kd: float
    t_ff: float


@dataclass
class AxolConfig:
    """Per-joint impedance control gains for the full robot.

    Load from a JSON file with ``AxolConfig.load(path)`` or use the
    bundled defaults with ``AxolConfig.load()``.

    JSON schema — each key is a joint name, value has kp, kd, and t_ff:

        {
          "shoulder_1": {"kp": 100.0, "kd": 2.0, "t_ff": 0.0},
          ...
        }
    """

    shoulder_1: JointGains
    shoulder_2: JointGains
    shoulder_3: JointGains
    elbow: JointGains
    wrist_1: JointGains
    wrist_2: JointGains
    wrist_3: JointGains
    gripper: JointGains

    @classmethod
    def load(cls, path: str | Path | None = None) -> AxolConfig:
        """Load gains from a JSON file.

        Args:
            path: Path to the JSON config file.
                  Defaults to the bundled ``config.json``.
        """
        with open(path or _DEFAULT_CONFIG) as f:
            data = json.load(f)
        return cls(**{name: JointGains(**vals) for name, vals in data.items()})
