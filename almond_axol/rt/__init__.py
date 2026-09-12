"""Production realtime core: the CAN control loop runs in Rust.

:class:`almond_axol.robot.Axol` (defined here as :class:`Axol`) wraps the
low-level :class:`~almond_axol.robot.axol.AxolHardware`, and the Mantis rig's
:class:`almond_axol.robot.Mantis` (:class:`Mantis` here) wraps
:class:`~almond_axol.robot.mantis.MantisHardware`.
Python keeps VR, IK, and MuJoCo gravity, while per-joint targets are shipped
over a Unix socket to ``axol-rt`` (see ``rust/axol-rt``), which solely owns
the buses and paces the 240 Hz control loop.
"""

# ``almond_axol.robot`` re-exports ``Axol`` / ``Mantis`` from this package's
# submodules. Initialize it first so that, whichever package is imported
# first, each submodule below is loaded exactly once and to completion —
# ``robot/__init__`` importing ``rt.mantis`` while ``rt.mantis`` is itself
# mid-import (through ``robot.base``) would otherwise fail.
from .. import robot as _robot  # noqa: F401
from .mantis import Mantis
from .robot import Axol

__all__ = ["Axol", "Mantis"]
