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
# The Jelly's wheel bus (its own single-channel adapter, separate from
# the arm hub), carrying the four Damiao wheel motors at IDs 0x01-0x04.
# NB: kernel interface names are capped at 15 chars (IFNAMSIZ), so this can't
# be the more readable "can_alm_axol_base".
CAN_BASE = "can_alm_axol_b"
# The chest bus (another single-channel adapter): the jelly_legs lift
# controller — our own PCB replacing the Jiecang control box, driving the
# telescoping lift legs (see almond_axol/robot/lift.py for the protocol).
CAN_CHEST = "can_alm_axol_c"

# CAN bring-up script written by `axol can.setup`. Runs at boot and on adapter
# hotplug, and is also the sanctioned way to reset the interfaces at runtime:
# it flaps both arm-hub channels *together* (flapping one at a time can wedge
# the adapter's RX path). The CAN bus layer reuses it to purge stale TX frames
# after an e-stop (see almond_axol/motor/bus.py).
CAN_BRINGUP_SCRIPT: Path = Path.home() / ".almond" / "can" / "startup.sh"

ARM_JOINTS: list[Joint] = [j for j in Joint if j != Joint.GRIPPER]


URDF_PATH: Path = Path(__file__).resolve().parent / "kinematics" / "urdf" / "axol.urdf"


# Where the fingers close, in the gripper link frame (metres, FLU). The URDF
# chain ends at the gripper mount, which is what forward kinematics and the IK
# solver report; the gripper mesh runs 145 mm out along that link's -Z. Any
# code that cares where the *tool* is — rather than where it is bolted on —
# needs this offset, and the distance is large enough that ignoring it turns a
# straight-line move into a visible arc as the wrist reorients.
GRIPPER_TIP_OFFSET: tuple[float, float, float] = (0.0, 0.0, -0.145)


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

# Body driven by each joint. ``Joint.GRIPPER`` maps to the (fixed-jointed)
# gripper link itself; MuJoCo merges this body into ``*_w2`` at load time.
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


def urdf_arm_joint_names(*, is_left: bool) -> list[str]:
    """URDF revolute-joint names for the 7 arm joints, in :data:`ARM_JOINTS` order."""
    return [urdf_joint_name(j, is_left=is_left) for j in ARM_JOINTS]


def urdf_arm_body_names(*, is_left: bool) -> list[str]:
    """URDF bodies driven by the 7 arm joints, in :data:`ARM_JOINTS` order."""
    return [urdf_body_name(j, is_left=is_left) for j in ARM_JOINTS]
