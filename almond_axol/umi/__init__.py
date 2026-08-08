"""UMI handheld-rig calibration and data-pipeline utilities.

The UMI rig is a handheld gripper with a 6-DoF tracker (Quest controller or
Vive tracker) rigidly mounted to it. Everything that ties the rig's tracked
poses to the robot lives here:

- :mod:`.handeye` — the AX=XB hand-eye solver that recovers the rigid
  tracker→TCP transform from a robot sweep (RDT2-style calibration).
- :mod:`.calibration` — persistence of the per-side tracker→TCP transforms
  (``~/.almond/umi/tcp_transform.json``), consumed by the absolute-mode IK
  worker.
- :mod:`.fk` — a lightweight forward-kinematics helper over the bundled
  Axol URDF (no IK solver / JIT warmup), used by the calibration sweep.
- :mod:`.relative` — quaternion→rotation-vector conversions the recorder
  uses to store tracked poses in the dataset's Cartesian layout.
- :mod:`.smoothing` — zero-phase (acausal) low-pass of the recorded EE pose
  track, applied by the dataset recorder at episode save to strip tracker
  noise from UMI episodes.
- :mod:`.processor` — the chunk-relative EE processor steps that give any
  LeRobot policy the UMI papers' start-pose invariance; injected by ``axol
  umi.train`` and reloaded from the checkpoint by ``axol run-policy``
  (imports torch / lerobot; import the module directly).
"""

from .calibration import (
    UMI_TCP_TRANSFORM_FILE,
    load_tcp_transforms,
    save_tcp_transforms,
)
from .handeye import HandEyeResult, solve_hand_eye
from .relative import quat_xyzw_to_matrix, quat_xyzw_to_rotvec

__all__ = [
    "UMI_TCP_TRANSFORM_FILE",
    "HandEyeResult",
    "load_tcp_transforms",
    "quat_xyzw_to_matrix",
    "quat_xyzw_to_rotvec",
    "save_tcp_transforms",
    "solve_hand_eye",
]
