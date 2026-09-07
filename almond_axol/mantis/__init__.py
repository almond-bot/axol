"""Mantis calibration and data-pipeline utilities.

The Mantis is a handheld gripper with a 6-DoF tracker (Quest controller or
Vive tracker) rigidly mounted to it. Everything that ties the rig's tracked
poses to the robot lives here:

- :mod:`.calibration` — the rig's factory tracker→TCP design transforms and
  the per-unit override file (``~/.almond/mantis/tcp_transform.json``), consumed
  by the absolute-mode IK worker.
- :mod:`.relative` — quaternion→rotation-vector conversions the recorder
  uses to store tracked poses in the dataset's Cartesian layout.
- :mod:`.smoothing` — zero-phase (acausal) low-pass of the recorded EE pose
  track, applied by the dataset recorder at episode save to strip tracker
  noise from Mantis episodes.
- :mod:`.processor` — chunk-relative EE processor steps that give any
  LeRobot policy start-pose invariance; injected by ``axol
  mantis.train`` and reloaded from the checkpoint by ``axol run-policy``
  (imports torch / lerobot; import the module directly).
"""

from .calibration import (
    DESIGN_TCP_TRANSFORMS,
    MANTIS_TCP_TRANSFORM_FILE,
    load_tcp_transforms,
    validate_tcp_transform,
)
from .relative import quat_xyzw_to_matrix, quat_xyzw_to_rotvec

__all__ = [
    "DESIGN_TCP_TRANSFORMS",
    "MANTIS_TCP_TRANSFORM_FILE",
    "load_tcp_transforms",
    "validate_tcp_transform",
    "quat_xyzw_to_matrix",
    "quat_xyzw_to_rotvec",
]
