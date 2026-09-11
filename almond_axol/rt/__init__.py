"""Production realtime core: the CAN control loop runs in Rust.

:class:`almond_axol.robot.Axol` (defined here as :class:`Axol`) wraps the
low-level :class:`~almond_axol.robot.axol.AxolHardware`, and the Mantis rig's
:class:`~almond_axol.robot.mantis.Mantis` is wrapped by :class:`RtMantis`.
Python keeps VR, IK, and MuJoCo gravity, while per-joint targets are shipped
over a Unix socket to ``axol-rt`` (see ``rust/axol-rt``), which solely owns
the buses and paces the 240 Hz control loop.

``RtAxol`` is the deprecated previous spelling of ``Axol``.
"""

from .mantis import RtMantis
from .robot import Axol, RtAxol

__all__ = ["Axol", "RtAxol", "RtMantis"]
