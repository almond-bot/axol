"""
almond_axol.axol

High-level interface for the Axol dual-arm robot.

Public API
──────────
    Axol            Dual-arm robot — async context manager
    ArmController   Single-arm controller (access via axol.left / axol.right)
    MotionCommand   Per-joint MIT impedance control command

Usage
─────
    async with Axol() as axol:
        # Both arms, all joints, fetched concurrently
        positions = await axol.get_positions()
        print(positions.left[Joint.SHOULDER_1])   # revolutions

        # Send MIT control to specific joints on both arms simultaneously
        cmd = MotionCommand(p_des=0.5, kp=100.0, kd=2.0)
        await axol.motion_control(
            left={Joint.ELBOW: cmd},
            right={Joint.ELBOW: cmd},
        )
"""

from .axol import ArmController, Axol
from .config import AxolConfig, JointGains

__all__ = [
    "Axol",
    "ArmController",
    "AxolConfig",
    "JointGains",
]
