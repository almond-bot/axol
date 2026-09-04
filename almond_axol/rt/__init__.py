"""Production realtime core: the CAN control loop runs in Rust.

Hardware control flows wrap :class:`~almond_axol.robot.axol.Axol` in
:class:`RtAxol`. Python keeps VR, IK, and MuJoCo gravity, while per-joint
targets are shipped over a Unix socket to ``axol-rt`` (see ``rust/axol-rt``),
which solely owns the buses and paces the 240 Hz control loop.
"""

from .robot import RtAxol

__all__ = ["RtAxol"]
