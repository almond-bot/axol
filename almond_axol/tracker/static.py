"""Fixed-pose tracker source for gripper-only Mantis teleop.

Reports two devices (``static-left`` / ``static-right``) parked at a
constant chest-height pose, so ``axol tracker.bridge`` can run with **no
tracker hardware at all**: the arms hold still and the only thing that
moves is the gripper, driven by the rig's CAN trigger node. That is the
whole rig minus the trackers, which is exactly what you want when
bringing up or debugging a Mantis gripper::

    axol teleop --mantis                     # one terminal
    axol tracker.bridge --backend static  # another; Enter to engage

The Mantis's seven arm joints per side are virtual (there is no arm), so a
frozen arm pose costs nothing physically. Unlike the ``synthetic``
backend — which orbits the devices through small circles to exercise the
whole bridge → IK → sim pipeline — nothing here ever moves, so a
squeeze of the trigger is the only thing that can change the commanded
state.

The poses match the placeholder the bridge streams for an unbound side,
and are restamped on every read so the bridge's staleness watchdog stays
satisfied.
"""

from __future__ import annotations

import time

import numpy as np

from .base import TrackerPose, TrackerSource

LEFT_KEY = "static-left"
RIGHT_KEY = "static-right"

# Chest height, shoulder-width apart, a little in front of the operator —
# the same placeholder the bridge uses for a side with no tracker bound.
_LEFT_POS = (0.2, 1.0, -0.4)
_RIGHT_POS = (-0.2, 1.0, -0.4)
_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


class StaticSource(TrackerSource):
    """Two virtual trackers frozen at a fixed pose.

    Args:
        separation: Lateral distance between the two device positions
            (metres); overrides the default shoulder-width spacing.
        center_y:   Height of both devices above the floor (metres).
    """

    def __init__(
        self,
        separation: float | None = None,
        center_y: float | None = None,
    ) -> None:
        left = list(_LEFT_POS)
        right = list(_RIGHT_POS)
        if separation is not None:
            left[0] = separation / 2
            right[0] = -separation / 2
        if center_y is not None:
            left[1] = right[1] = center_y
        self._positions = {
            LEFT_KEY: np.array(left, dtype=float),
            RIGHT_KEY: np.array(right, dtype=float),
        }
        self._running = False

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def poses(self) -> dict[str, TrackerPose]:
        if not self._running:
            return {}
        now = time.perf_counter()
        return {
            key: TrackerPose(
                pos=pos.copy(),
                quat=np.array(_IDENTITY_QUAT, dtype=float),
                t=now,
            )
            for key, pos in self._positions.items()
        }
