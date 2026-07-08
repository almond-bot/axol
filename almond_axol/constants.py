"""Shared constants and utilities for the Almond Axol robot."""

from enum import Enum
from pathlib import Path


class Joint(Enum):
    """All motor joints on one arm, in control order.

    The seven arm joints (``SHOULDER_1`` through ``WRIST_3``) are collected in
    ``ARM_JOINTS``. ``GRIPPER`` is the eighth entry and is handled separately
    from the arm joints throughout the control stack.
    """

    SHOULDER_1 = "shoulder_1"
    SHOULDER_2 = "shoulder_2"
    SHOULDER_3 = "shoulder_3"
    ELBOW = "elbow"
    WRIST_1 = "wrist_1"
    WRIST_2 = "wrist_2"
    WRIST_3 = "wrist_3"
    GRIPPER = "gripper"


CAN_LEFT = "can_alm_axol_l"
CAN_RIGHT = "can_alm_axol_r"

# Handheld UMI data-collection rig: one dual-channel adapter, each channel
# wired to a single Damiao gripper (CAN ID 0x08, same as Joint.GRIPPER).
CAN_UMI_LEFT = "can_alm_umi_l"
CAN_UMI_RIGHT = "can_alm_umi_r"

ARM_JOINTS: list[Joint] = [j for j in Joint if j != Joint.GRIPPER]


URDF_PATH: Path = Path(__file__).resolve().parent / "kinematics" / "urdf" / "axol.urdf"


# Single source of truth for URDF joint and body names. All helpers
# (gravity comp, IK solver, simulation) compose ``f"{side}_{suffix}"`` from
# these tables via the ``urdf_*_name`` helpers below.

# ``Joint.GRIPPER`` is intentionally absent: the gripper is a fixed URDF
# joint with no actuator counterpart.
_ARM_JOINT_URDF_SUFFIX: dict[Joint, str] = {
    Joint.SHOULDER_1: "s1_0",
    Joint.SHOULDER_2: "s2_0",
    Joint.SHOULDER_3: "s3_0",
    Joint.ELBOW: "e1_0",
    Joint.WRIST_1: "e2_0",
    Joint.WRIST_2: "w1_0",
    Joint.WRIST_3: "w2_0",
}

# Body driven by each joint. ``Joint.GRIPPER`` maps to the gripper *base* link
# (``*_gripper``), which is fixed-jointed to ``*_w2`` and carries the two
# prismatic finger links as children. It is also the end-effector frame used by
# the IK solver and forward kinematics.
_BODY_URDF_SUFFIX: dict[Joint, str] = {
    Joint.SHOULDER_1: "s2",
    Joint.SHOULDER_2: "s3",
    Joint.SHOULDER_3: "e1",
    Joint.ELBOW: "e2",
    Joint.WRIST_1: "w0",
    Joint.WRIST_2: "w1",
    Joint.WRIST_3: "w2",
    Joint.GRIPPER: "gripper",
}

# Actuated (non-mimic) prismatic finger joint that drives each gripper open and
# closed in the URDF. The opposing finger mimics this joint. ``GRIPPER_URDF_OPEN``
# is its travel limit (m): 0 = fully closed, ``GRIPPER_URDF_OPEN`` = fully open.
_GRIPPER_JOINT_SUFFIX: str = "gripper_joint_a"
GRIPPER_URDF_OPEN: float = 0.058804


def urdf_joint_name(joint: Joint, *, is_left: bool) -> str:
    """URDF revolute-joint name driving ``joint`` on the given arm.

    Example::

        urdf_joint_name(Joint.SHOULDER_1, is_left=True) == "left_s1_0"

    Raises ``KeyError`` for ``Joint.GRIPPER`` (no actuator joint in the URDF).
    """
    side = "left" if is_left else "right"
    return f"{side}_{_ARM_JOINT_URDF_SUFFIX[joint]}"


def urdf_body_name(joint: Joint, *, is_left: bool) -> str:
    """URDF body driven by ``joint`` on the given arm.

    Example::

        urdf_body_name(Joint.SHOULDER_1, is_left=True) == "left_s2"
        urdf_body_name(Joint.GRIPPER,    is_left=True) == "left_gripper"
    """
    side = "left" if is_left else "right"
    return f"{side}_{_BODY_URDF_SUFFIX[joint]}"


def urdf_gripper_joint_name(*, is_left: bool) -> str:
    """URDF prismatic finger joint that actuates the gripper on the given arm.

    This is the single non-mimic finger joint (the opposing finger mimics it);
    driving it from ``0`` to :data:`GRIPPER_URDF_OPEN` moves the gripper from
    fully closed to fully open.

    Example::

        urdf_gripper_joint_name(is_left=True) == "left_gripper_joint_a"
    """
    side = "left" if is_left else "right"
    return f"{side}_{_GRIPPER_JOINT_SUFFIX}"


def urdf_arm_joint_names(*, is_left: bool) -> list[str]:
    """URDF revolute-joint names for the 7 arm joints, in :data:`ARM_JOINTS` order."""
    return [urdf_joint_name(j, is_left=is_left) for j in ARM_JOINTS]


def urdf_arm_body_names(*, is_left: bool) -> list[str]:
    """URDF bodies driven by the 7 arm joints, in :data:`ARM_JOINTS` order."""
    return [urdf_body_name(j, is_left=is_left) for j in ARM_JOINTS]
