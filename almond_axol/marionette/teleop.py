"""How an arm follows a phone: Axol's limits, smoothing and reach, over
Marionette's teleop parts.

Marionette decides nothing about motion, so every safety decision is made
here and in :mod:`almond_axol.marionette.session`:

- **Ownership.** An arm follows at most one phone: the phone that selected
  it. When two phones pick the same arm, the one that engaged it most
  recently keeps it (:func:`marionette.owners`).
- **Clutch.** A press of the phone's hold-to-move pad engages the arm: the
  phone pose ``P0`` and the gripper tip pose ``E0`` are recorded, and
  nothing moves. While engaged the tip target is ``E0`` moved by the
  phone's motion since ``P0``, turned into the torso frame by ``F``
  (:class:`marionette.Clutch`)::

      p = E0.p + s * F @ (P.p - P0.p)
      R = rot(r * s * F @ angle(P.R @ P0.R^T)) @ E0.R

  ``s`` is the phone's motion scale, which scales turns as well as moves,
  and ``r`` the rotation scale. The phone's operator frame and the torso
  frame are both +x forward, +y left, +z up, so in Axol's frame ``F`` is
  the identity. In a camera's frame (the phone's Frame choice) ``F`` sends
  the phone's forward along the camera's view and its left and up to the
  image's (:func:`marionette.camera_frame`), as the camera sat when the pad
  was pressed. A new scale or frame re-anchors without moving the target,
  and the turn is followed continuously past half a turn. The target is
  smoothed with the VR teleop's lag-compensated low-pass and then rate
  limited (:func:`marionette.teleop.follow`), so a fast hand or a tracking
  jump becomes a bounded gripper speed.
- **Gates.** The arm stops and drops its anchor on any failure: the pad
  released or its renewals stopped, a stale or degraded pose
  (:func:`gate_reason`), a new ARKit session or calibration, a tracking
  jump, Stop, or another phone taking the arm. Only a fresh press engages
  again (:class:`marionette.PressTracker`), so nothing resumes on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from marionette import Clutch, JumpDetector, Phone, PressTracker, gate
from marionette.teleop import (
    exp_rotation,
    follow,
    matrix_to_quaternion,
    quaternion_to_matrix,
    rotation_vector,
)

from ..kinematics.arm_ik import Pose
from ..teleop.filter import LagCompensatedLowPass
from .phones import DEADMAN, frame, scale


@dataclass
class TeleopLimits:
    """How fast and how carefully an arm follows a phone."""

    max_speed: float = 0.5
    """Fastest the gripper target moves (m/s)."""
    max_turn_rate: float = 3.0
    """Fastest the gripper target turns (rad/s)."""
    max_accel: float = 3.0
    """Fastest the gripper target speeds up or slows down (m/s^2)."""
    max_turn_accel: float = 20.0
    """Fastest the gripper target's turn rate changes (rad/s^2)."""
    joint_speed: float = 2.0
    """Fastest any arm joint turns (rad/s)."""
    rotation_scale: float = 1.0
    """Phone rotation to gripper rotation at motion scale 1 (1 = the same
    angle). The phone's motion scale multiplies it."""
    pose_cutoff: float = 4.0
    """Low-pass cutoff (Hz) on the gripper target."""
    pose_timeout: float = 0.25
    """A pose older than this (s) stops the arm."""
    deadman_timeout: float = 0.6
    """The pad counts as released after this long (s) without a renewal."""
    reach_tolerance_m: float = 0.02
    """The tip lagging the target by more than this is held back..."""
    reach_tolerance_rad: float = math.radians(6.0)
    """...or turned short by more than this..."""
    reach_hold_s: float = 0.3
    """...for this long, which the phone shows as the reach limit."""
    jump_m: float = 0.04
    """A pose step larger than this..."""
    jump_rad: float = 0.5
    """...or a turn larger than this..."""
    jump_speed: float = 4.0
    """...faster than this (m/s, and 5x in rad/s) is a tracking jump, not a
    hand: the arm stops instead of following it."""


def gate_reason(
    phone: Phone,
    now: float,
    limits: TeleopLimits,
    *,
    allow_synthetic: bool,
    require_calibration: bool = False,
) -> str | None:
    """Why this phone's pose can't drive an arm right now, or None.

    ``require_calibration`` (the real robot) refuses a phone that has not
    been calibrated: an uncalibrated phone's forward is wherever ARKit
    started, which may not be the robot's forward.
    """
    if (
        require_calibration
        and phone.connected
        and phone.calibration not in ("capturing", "captured")
    ):
        return "Calibrate this phone first, facing the way Axol faces."
    return gate(
        phone,
        now,
        pose_timeout=limits.pose_timeout,
        allow_synthetic=allow_synthetic,
        require_calibration=require_calibration,
    )


class ArmTeleop:
    """One arm's clutch, presses, target filter and reach state."""

    def __init__(self, side: str, limits: TeleopLimits, rate: float) -> None:
        self.side = side
        self.limits = limits
        self.owner: str | None = None  # phone that engaged this arm last
        self.frame = "axol"  # the owner's Frame choice when it engaged
        self.limited = False
        self.presses = PressTracker(DEADMAN)
        self.clutch = Clutch()
        self.jumps = JumpDetector(
            limits.jump_m, limits.jump_rad, limits.jump_speed, 5.0 * limits.jump_speed
        )
        self._lag_since: float | None = None
        self._commanded: Pose | None = None
        self._velocity = np.zeros(6)  # target velocity: m/s, then rad/s
        self._filter_p = LagCompensatedLowPass(rate, limits.pose_cutoff)
        self._filter_q = LagCompensatedLowPass(rate, limits.pose_cutoff)
        self._quat_prev: np.ndarray | None = None

    @property
    def engaged(self) -> bool:
        return self.clutch.engaged

    def engage(
        self, phone: Phone, tip: Pose, to_torso: np.ndarray | None = None
    ) -> None:
        """Anchor the phone's current pose to the tip's current pose.

        ``to_torso`` is ``F`` for the phone's Frame choice (None: Axol's
        frame), fixed until the next press.
        """
        assert phone.pose is not None
        tip = (np.asarray(tip[0], np.float64), np.asarray(tip[1], np.float64))
        self.owner = phone.id
        self.frame = frame(phone)
        self.limited = False
        self._lag_since = None
        self.clutch.engage(
            phone.pose,
            tip[0],
            tip[1],
            scale=scale(phone),
            rotation_scale=self.limits.rotation_scale,
            frame=to_torso,
        )
        self.jumps.reset(phone.pose)
        self._commanded = tip
        self._velocity = np.zeros(6)
        self._filter_p.reset(tip[0])
        quat = matrix_to_quaternion(tip[1])
        self._quat_prev = quat
        self._filter_q.reset(quat)

    def release(self) -> None:
        """Stop following; the next press re-anchors."""
        self.clutch.release()
        self.limited = False
        self._lag_since = None
        self._commanded = None

    def reframe(self, phone: Phone, to_torso: np.ndarray | None) -> None:
        """Take up the phone's new Frame choice (``F``: ``to_torso``) from
        here on, without moving the arm."""
        assert phone.pose is not None
        self.clutch.reframe(phone.pose, to_torso)
        self.frame = frame(phone)

    def target(self, phone: Phone, now: float, dt: float) -> Pose:
        """This tick's tip target: mapped, filtered, then speed and
        acceleration limited. A new motion scale applies from here on and
        never moves the arm by itself."""
        assert self._commanded is not None and phone.pose is not None
        limits = self.limits
        if scale(phone) != self.clutch.scale:
            self.clutch.rescale(phone.pose, scale(phone))
        goal = self.clutch.target(phone.pose)
        position = self._filter_p.update(goal[0], now).astype(np.float64)
        quat = matrix_to_quaternion(goal[1])
        if self._quat_prev is not None and float(np.dot(quat, self._quat_prev)) < 0:
            quat = -quat  # stay on one hemisphere so the filter doesn't flip
        self._quat_prev = quat
        rotation = quaternion_to_matrix(self._filter_q.update(quat, now))
        p_cmd, r_cmd = self._commanded
        self._velocity[:3] = follow(
            position - p_cmd, self._velocity[:3], limits.max_speed, limits.max_accel, dt
        )
        self._velocity[3:] = follow(
            rotation_vector(rotation @ r_cmd.T),
            self._velocity[3:],
            limits.max_turn_rate,
            limits.max_turn_accel,
            dt,
        )
        self._commanded = (
            p_cmd + self._velocity[:3] * dt,
            exp_rotation(self._velocity[3:] * dt) @ r_cmd,
        )
        return self._commanded

    def note_tip(self, tip: Pose, now: float) -> None:
        """Compare the reached tip with the target to flag the reach limit."""
        if not self.engaged or self._commanded is None:
            self.limited = False
            self._lag_since = None
            return
        target = self._commanded
        distance = float(np.linalg.norm(target[0] - tip[0]))
        angle = float(np.linalg.norm(rotation_vector(target[1] @ tip[1].T)))
        lagging = (
            distance > self.limits.reach_tolerance_m
            or angle > self.limits.reach_tolerance_rad
        )
        if not lagging:
            self._lag_since = None
            self.limited = False
        elif self._lag_since is None:
            self._lag_since = now
        else:
            self.limited = now - self._lag_since >= self.limits.reach_hold_s
