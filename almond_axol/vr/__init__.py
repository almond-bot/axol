"""
almond_axol.vr

VR teleoperation WebSocket server for the Axol arm.

Public API
──────────
    VRServer   Secure WebSocket server — receives frames from the VR headset
    VRFrame    Teleoperation frame model
    VRPose     6-DOF controller pose (position + quaternion)

Usage
─────
    from almond_axol.vr import VRServer, VRFrame

    async with VRServer() as vr:
        frame: VRFrame | None = vr.get_frame()
"""

from .models import VRFrame, VRPose, VRPosition, VRQuaternion, VRState
from .server import VRServer

__all__ = ["VRServer", "VRFrame", "VRPose", "VRPosition", "VRQuaternion", "VRState"]
