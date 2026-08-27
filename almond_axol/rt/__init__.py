"""Hybrid realtime-core mode: the CAN control loop runs in Rust.

``axol teleop --rt`` wraps the :class:`~almond_axol.robot.axol.Axol` robot in
:class:`RtAxol`: Python keeps everything smart (VR, IK, MuJoCo gravity, host
damping — all of ``AxolArm.motion_control``) but the fully computed per-joint
MIT tuples are shipped over a Unix socket to the ``axol-rt`` core (see
``rust/axol-rt``), which owns the buses, paces a hard 240 Hz loop, and
interpolates between the ~120 Hz targets.
"""

from .robot import RtAxol

__all__ = ["RtAxol"]
