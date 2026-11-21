"""Follow a planned path with an omni-wheel base.

Jelly can move in any direction while turning, so the follower steers the
velocity straight at a point :attr:`DriveConfig.lookahead` metres further
along the path, and turns toward the goal heading independently. Speed is
capped by :attr:`DriveConfig.max_speed` and by a braking profile
(``v = sqrt(2·a·d)``) so the base, whose command ramps at a limited rate,
stops on the goal instead of past it.

Commands come out in metres and radians per second in the base frame
(+x forward, +y left, +wz counter-clockwise); :meth:`DriveConfig.normalize`
turns them into Jelly's ``[-1, 1]`` command.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import Pose2D, wrap_angle


@dataclass
class DriveConfig:
    """Speeds and tolerances for autonomous driving.

    Attributes:
        max_speed:    Top speed while navigating (m/s).
        max_turn:     Top turn rate while navigating (rad/s).
        accel:        Braking rate the speed profile assumes (m/s²). Keep it
                      at or below what the base can do, or it overshoots.
        lookahead:    How far ahead on the path to steer toward (m).
        goal_tolerance: Distance from the goal that counts as arrived (m).
        yaw_tolerance: Heading error that counts as arrived (deg).
        heading_gain: Turn rate per radian of heading error (1/s).
        position_gain: Speed per metre of error on the final approach (1/s).
        min_speed:    Slowest useful speed before arriving (m/s).
        base_max_linear: Jelly's speed at a full command (m/s).
        base_max_angular: Jelly's turn rate at a full command (rad/s).
    """

    max_speed: float = 0.3
    max_turn: float = 0.6
    accel: float = 0.12
    lookahead: float = 0.5
    goal_tolerance: float = 0.08
    yaw_tolerance: float = 5.0
    heading_gain: float = 1.5
    position_gain: float = 1.0
    min_speed: float = 0.04
    base_max_linear: float = 0.7
    base_max_angular: float = 1.5

    def normalize(self, vx: float, vy: float, wz: float) -> tuple[float, float, float]:
        clip = lambda v: max(-1.0, min(1.0, v))  # noqa: E731
        return (
            clip(vx / self.base_max_linear),
            clip(vy / self.base_max_linear),
            clip(wz / self.base_max_angular),
        )


@dataclass
class DriveCommand:
    vx: float
    vy: float
    wz: float
    remaining: float
    arrived: bool


class PathFollower:
    """Track one path to a final heading. Call :meth:`command` every frame."""

    def __init__(
        self,
        path: list[tuple[float, float]],
        goal_yaw: float | None,
        config: DriveConfig | None = None,
    ) -> None:
        if not path:
            raise ValueError("empty path")
        self.config = config or DriveConfig()
        self.path = np.asarray(path, float)
        if len(self.path) == 1:
            self.path = np.vstack([self.path, self.path])
        seg = np.diff(self.path, axis=0)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.cum[-1])
        self.goal = self.path[-1]
        self.goal_yaw = goal_yaw
        self.progress = 0.0  # arc length of the closest point (never goes back)

    def _project(self, p: np.ndarray) -> float:
        best_s, best_d = self.progress, math.inf
        for i, (a, length) in enumerate(zip(self.path[:-1], self.seg_len, strict=True)):
            if self.cum[i + 1] < self.progress - 0.3:
                continue
            if length < 1e-9:
                t = 0.0
            else:
                t = float(
                    np.clip(np.dot(p - a, self.path[i + 1] - a) / length**2, 0, 1)
                )
            q = a + t * (self.path[i + 1] - a)
            d = float(np.hypot(*(p - q)))
            if d < best_d - 1e-9:
                best_d, best_s = d, float(self.cum[i] + t * length)
        return max(best_s, self.progress)

    def _point_at(self, s: float) -> np.ndarray:
        s = min(max(s, 0.0), self.length)
        i = int(np.searchsorted(self.cum, s, side="right") - 1)
        i = min(i, len(self.seg_len) - 1)
        length = self.seg_len[i]
        t = 0.0 if length < 1e-9 else (s - self.cum[i]) / length
        return self.path[i] + t * (self.path[i + 1] - self.path[i])

    def command(self, pose: Pose2D, speed_limit: float | None = None) -> DriveCommand:
        cfg = self.config
        p = np.array([pose.x, pose.y])
        self.progress = self._project(p)
        to_goal = self.goal - p
        d_goal = float(np.hypot(*to_goal))
        remaining = max(self.length - self.progress, d_goal)
        yaw_err = 0.0 if self.goal_yaw is None else wrap_angle(self.goal_yaw - pose.yaw)
        tol_yaw = math.radians(cfg.yaw_tolerance)

        if d_goal <= cfg.goal_tolerance and abs(yaw_err) <= tol_yaw:
            return DriveCommand(0.0, 0.0, 0.0, 0.0, True)

        top = cfg.max_speed if speed_limit is None else min(cfg.max_speed, speed_limit)
        if d_goal <= cfg.goal_tolerance:
            v_world = np.zeros(2)
        elif remaining <= cfg.lookahead:
            # Final approach: straight at the goal, proportional speed.
            speed = min(
                top, cfg.position_gain * d_goal, math.sqrt(2 * cfg.accel * d_goal)
            )
            speed = max(speed, min(cfg.min_speed, top))
            v_world = to_goal / max(d_goal, 1e-9) * speed
        else:
            carrot = self._point_at(self.progress + cfg.lookahead)
            direction = carrot - p
            n = float(np.hypot(*direction))
            speed = min(
                top, math.sqrt(2 * cfg.accel * max(remaining - cfg.goal_tolerance, 0.0))
            )
            speed = max(speed, min(cfg.min_speed, top))
            v_world = direction / max(n, 1e-9) * speed

        # Turn toward the goal heading, gently so a turn doesn't slew the path.
        wz = 0.0
        if self.goal_yaw is not None and abs(yaw_err) > tol_yaw * 0.5:
            wz = float(np.clip(cfg.heading_gain * yaw_err, -cfg.max_turn, cfg.max_turn))
            # Brake the turn like the translation.
            wz = math.copysign(
                min(abs(wz), math.sqrt(2 * cfg.accel * 2.0 * abs(yaw_err))), wz
            )
            wz = math.copysign(max(abs(wz), 0.08), wz)

        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        vx = c * v_world[0] + s * v_world[1]
        vy = -s * v_world[0] + c * v_world[1]
        return DriveCommand(float(vx), float(vy), wz, remaining, False)
