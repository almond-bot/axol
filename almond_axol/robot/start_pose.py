"""Refuse to start a session with an arm far from its zero (home) pose.

Joint zero is the robot's natural resting pose — where a limp arm settles,
and so where every session leaves the arms once they are parked and torqued
off. Teleop, data collection, DAgger, policy rollout and dataset replay all
begin by playing a planned move from wherever the arms are to the configured
rest pose. When an arm has been left far from zero — folded over by hand
while limp, collapsed onto the table, or reading a bad joint offset — that
first move is large and unsupervised, so these flows check the measured pose
first and refuse to start if any arm joint is more than
:data:`START_POSE_TOLERANCE_DEG` from zero.

The reference is zero rather than the configured rest pose on purpose: the
rest pose is an operator setting that may sit well away from where the arms
naturally settle, so it says nothing about whether they are where they
should be at power-up.

The check runs right after the robot is enabled, before anything is
commanded: enabling only holds the measured pose, and the flow's normal
startup-failure teardown then torques the arms back off where they are. It
can't run earlier — on a merely ``connect()``-ed bus the MyActuator joints
have not been through the realtime core's prep reset yet, so their
multi-turn positions are not trustworthy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ..constants import ARM_JOINTS

START_POSE_TOLERANCE_DEG = 25.0

ArmPair = tuple[Sequence[float] | None, Sequence[float] | None]


class StartPoseError(RuntimeError):
    """An arm joint is too far from zero to start a session."""


@dataclass(frozen=True)
class JointDeviation:
    """One arm joint measured outside the start tolerance."""

    side: str
    joint: str
    position_deg: float


def start_pose_deviations(
    positions: ArmPair,
    tolerance_deg: float = START_POSE_TOLERANCE_DEG,
) -> list[JointDeviation]:
    """Arm joints more than ``tolerance_deg`` from zero, left arm first.

    ``positions`` is the measured ``(left, right)`` pair in Joint enum order
    (the gripper, if present, is ignored; ``None`` is an absent arm). A
    non-finite reading counts as out of tolerance — it can't be shown to be
    near zero.
    """
    out: list[JointDeviation] = []
    for side, q in zip(("left", "right"), positions):
        if q is None:
            continue
        for i, joint in enumerate(ARM_JOINTS):
            pos = math.degrees(float(q[i]))
            if not abs(pos) <= tolerance_deg:
                out.append(JointDeviation(side, joint.value, pos))
    return out


def check_start_pose(
    positions: ArmPair,
    tolerance_deg: float = START_POSE_TOLERANCE_DEG,
) -> None:
    """Raise :class:`StartPoseError` naming every joint outside tolerance.

    Arguments as for :func:`start_pose_deviations`.
    """
    deviations = start_pose_deviations(positions, tolerance_deg)
    if not deviations:
        return
    lines = [f"  {d.side} {d.joint}: {d.position_deg:+.1f}°" for d in deviations]
    raise StartPoseError(
        f"refusing to start: {len(deviations)} arm joint(s) more than "
        f"{tolerance_deg:g}° from zero:\n"
        + "\n".join(lines)
        + "\nMove the arms back to their natural resting pose by hand (e.g. "
        "under `axol gravity-comp`) and start again."
    )
