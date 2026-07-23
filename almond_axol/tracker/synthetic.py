"""Synthetic tracker source for end-to-end tests without hardware.

Generates two devices (``synthetic-left`` / ``synthetic-right``) moving
through slow, small circles at roughly chest height in the y-up WebXR
world, so the bridge → VRServer → IK → sim pipeline can be exercised in
CI or the cloud VM: ``axol tracker.bridge --backend synthetic`` against
``axol teleop --sim``.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .base import TrackerPose, TrackerSource

LEFT_KEY = "synthetic-left"
RIGHT_KEY = "synthetic-right"


class SyntheticSource(TrackerSource):
    """Two virtual trackers orbiting small circles, poses computed on read.

    Args:
        radius: Circle radius in metres.
        period: Seconds per revolution.
        center_y: Height of the circle centres above the floor (metres).
        separation: Lateral distance between the two device centres (metres).
    """

    def __init__(
        self,
        radius: float = 0.05,
        period: float = 8.0,
        center_y: float = 1.0,
        separation: float = 0.4,
    ) -> None:
        self._radius = radius
        self._period = period
        self._center_y = center_y
        self._separation = separation
        self._t0: float | None = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        self._t0 = None

    def poses(self) -> dict[str, TrackerPose]:
        if self._t0 is None:
            return {}
        now = time.perf_counter()
        phase = 2.0 * math.pi * (now - self._t0) / self._period
        out: dict[str, TrackerPose] = {}
        for key, x0 in (
            (LEFT_KEY, self._separation / 2),
            (RIGHT_KEY, -self._separation / 2),
        ):
            pos = np.array(
                [
                    x0 + self._radius * math.cos(phase),
                    self._center_y + self._radius * math.sin(phase),
                    -0.4,
                ]
            )
            out[key] = TrackerPose(pos=pos, quat=np.array([0.0, 0.0, 0.0, 1.0]), t=now)
        return out
