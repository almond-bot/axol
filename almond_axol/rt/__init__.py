"""Production realtime core: the CAN control loop runs in Rust.

:class:`almond_axol.robot.Axol` and :class:`almond_axol.robot.Mantis` are
defined here. They present the classic robot API and own the low-level bus /
motor / model objects (:mod:`almond_axol.robot.axol`,
:mod:`almond_axol.robot.mantis`) as an implementation detail. Python keeps
VR, IK, and MuJoCo gravity, while per-joint targets are shipped over a Unix
socket to ``axol-rt`` (see ``rust/axol-rt``), which solely owns the buses and
paces the 240 Hz control loop while the robot is enabled.
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
