"""Phone-to-gripper mapping and the gates that decide when an arm follows.

Marionette decides nothing about motion, so every safety decision is made
here (see the plan in the module docstring of
:mod:`almond_axol.marionette.session`):

- **Ownership.** An arm follows at most one phone: the phone that selected
  it. When two phones pick the same arm, the one that engaged it most
  recently keeps it.
- **Clutch.** A press of the phone's hold-to-move pad engages the arm: the
  phone pose ``P0`` and the gripper tip pose ``E0`` are recorded, and
  nothing moves. While engaged the tip target is ``E0`` moved by the
  phone's motion since ``P0``, turned into the torso frame by ``F``:

      p = E0.p + s * F @ (P.p - P0.p)
      R = rot(r * s * F @ angle(P.R @ P0.R^T)) @ E0.R

  ``s`` is the phone's motion scale, which scales turns as well as moves,
  and ``r`` the rotation scale. The phone's operator frame and the torso
  frame are both +x forward, +y left, +z up, so in Axol's frame ``F`` is
  the identity. In a camera's frame (the phone's Frame choice) ``F``
  sends the phone's forward along the camera's view and its left and up
  to the image's (:func:`camera_frame`), as the camera sat when the pad
  was pressed. The turn is followed continuously past half a turn
  (:func:`unwrap_turn`), so a scaled turn never jumps. The target is
  smoothed with the VR teleop's lag-compensated low-pass and then rate
  limited, so a fast hand or a tracking jump becomes a bounded gripper
  speed.
- **Gates.** The arm stops and drops its anchor on any failure: the pad
  released or its renewals stopped, a stale or degraded pose, a new ARKit
  session or calibration, Stop, or another phone taking the arm. Only a
  fresh press engages again, so nothing resumes on its own.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ..kinematics.arm_ik import Pose, exp_rotation, rotation_vector
from ..teleop.filter import LagCompensatedLowPass
from .phones import Phone, PhonePose


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


def _quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    """A unit quaternion for a rotation matrix (any sign)."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(rotation).as_quat()


def _matrix(quat: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    q = np.asarray(quat, dtype=np.float64)
    return Rotation.from_quat(q / np.linalg.norm(q)).as_matrix()


# The phone's axes (+x forward, +y left, +z up) in a camera's ROS optical
# frame (+z along the view, +x to the image's right, +y down it): forward
# is along the view, left and up are the image's.
_PHONE_IN_OPTICAL = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def camera_frame(optical: np.ndarray) -> np.ndarray:
    """``F`` for moving in a camera's frame, from the rotation of its optical
    frame in the torso frame (:meth:`ArmKinematics.camera`)."""
    return np.asarray(optical, dtype=np.float64) @ _PHONE_IN_OPTICAL


def unwrap_turn(turn: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """The rotation vector for ``turn``'s rotation closest to ``previous``.

    A rotation vector is unique only up to whole turns about its axis, and
    past half a turn the axis flips. Scaled, that flip would jump the
    gripper; picking the vector nearest the last one follows a turn of any
    size smoothly.
    """
    angle = float(np.linalg.norm(turn))
    if angle > 1e-9:
        axis = turn / angle
    else:
        size = float(np.linalg.norm(previous))
        if size < 1e-9:
            return turn
        axis, angle = previous / size, 0.0
    whole = round((float(np.dot(previous, axis)) - angle) / (2.0 * math.pi))
    return (angle + 2.0 * math.pi * whole) * axis


def map_target(
    phone: PhonePose,
    anchor_phone: PhonePose,
    anchor_tip: Pose,
    scale: float,
    rotation_scale: float = 1.0,
    frame: np.ndarray | None = None,
    turn: np.ndarray | None = None,
) -> Pose:
    """The tip target for a phone pose, relative to the engage anchor.

    ``frame`` is ``F`` (None: Axol's frame). ``turn`` is the phone's turn
    since the anchor as a rotation vector, followed with
    :func:`unwrap_turn`; by default it is read from the two poses.
    """
    to_torso = np.eye(3) if frame is None else frame
    if turn is None:
        turn = rotation_vector(phone.rotation @ anchor_phone.rotation.T)
    position = anchor_tip[0] + scale * (
        to_torso @ (phone.position - anchor_phone.position)
    )
    rotation = exp_rotation(rotation_scale * scale * (to_torso @ turn)) @ anchor_tip[1]
    return position, rotation


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
    if not phone.connected:
        return "This phone is offline."
    if phone.calibration == "capturing":
        return "Finish or cancel calibration first."
    if require_calibration and phone.calibration != "captured":
        return "Calibrate this phone first, facing the way Axol faces."
    pose = phone.pose
    if pose is None or now - pose.received > limits.pose_timeout:
        return "No pose from this phone. Keep the app open."
    if pose.source == "synthetic" and not allow_synthetic:
        return "Simulated motion can't move the robot."
    if pose.source not in ("arkit", "synthetic"):
        return "This phone has no ARKit tracking."
    if pose.tracking == "initializing":
        return "Tracking is starting. Move the phone gently."
    if pose.tracking == "relocalizing":
        return "Tracking is finding its place. Hold still for a moment."
    if pose.tracking != "normal" and not (
        pose.source == "synthetic" and allow_synthetic
    ):
        return "Tracking is limited. Point the camera at something with texture."
    return None


def owners(
    phones: Iterable[Phone], current: dict[str, str | None]
) -> dict[str, Phone | None]:
    """Which connected phone has each arm.

    ``current`` maps each arm to the phone that engaged it most recently,
    which keeps the arm while it still selects it. Otherwise the lowest
    numbered phone selecting the arm has it. A press from another phone
    selecting the same arm takes the arm over only while it is not moving
    (the session does that).
    """
    result: dict[str, Phone | None] = {}
    connected = [p for p in phones if p.connected]
    for side in ("left", "right"):
        candidates = [p for p in connected if p.arm == side]
        keep = next((p for p in candidates if p.id == current.get(side)), None)
        if keep is None and candidates:
            keep = min(candidates, key=lambda p: p.number)
        result[side] = keep
    return result


def _follow(
    error: np.ndarray, velocity: np.ndarray, top: float, accel: float, dt: float
) -> np.ndarray:
    """The next velocity toward a target ``error`` away.

    Speeds up and slows down at ``accel`` at most, never faster than
    ``top``, and slow enough to stop at the target instead of overshooting
    it; within one tick of it, it lands there.
    """
    distance = float(np.linalg.norm(error))
    if distance < 1e-9:
        return np.zeros(3)
    direction = error / distance
    speed = min(top, math.sqrt(2.0 * accel * distance), distance / dt)
    change = direction * speed - velocity
    size = float(np.linalg.norm(change))
    step = accel * dt
    if size > step:
        change *= step / size
    new = velocity + change
    # Never step past the target.
    if float(np.dot(new, direction)) * dt > distance:
        new = direction * (distance / dt)
    return new


class ArmTeleop:
    """One arm's clutch, target filter and reach state."""

    def __init__(self, side: str, limits: TeleopLimits, rate: float) -> None:
        self.side = side
        self.limits = limits
        self._rate = rate
        self.owner: str | None = None  # phone that engaged this arm last
        self.engaged = False
        self.limited = False
        self._lag_since: float | None = None
        self._presses: dict[str, int] = {}  # phone id -> press count seen
        self._anchor: tuple[PhonePose, Pose] | None = None
        self._scale = 1.0
        self.frame = "axol"  # the owner's Frame choice when it engaged
        self._to_torso: np.ndarray | None = None  # F; None in Axol's frame
        self._turn = np.zeros(3)  # the phone's turn since the anchor
        self._last_pose: PhonePose | None = None
        self._commanded: Pose | None = None
        self._velocity = np.zeros(6)  # target velocity: m/s, then rad/s
        self._filter_p = LagCompensatedLowPass(rate, limits.pose_cutoff)
        self._filter_q = LagCompensatedLowPass(rate, limits.pose_cutoff)
        self._quat_prev: np.ndarray | None = None

    # -- clutch --------------------------------------------------------------

    def new_press(self, phone: Phone) -> bool:
        """Whether ``phone`` pressed its pad since this arm last looked.

        Every press is consumed once, whatever happens to it, so a press
        refused (Stop on, bad tracking) never engages later on its own.
        """
        seen = self._presses.get(phone.id)
        self._presses[phone.id] = phone.deadman_presses
        if seen is None:
            # First sight of this phone: a pad already held doesn't count.
            return False
        return phone.deadman_presses > seen

    def forget_presses(self, phone: Phone) -> None:
        """Mark ``phone``'s presses as seen without acting on them."""
        self._presses[phone.id] = phone.deadman_presses

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
        self.engaged = True
        self.limited = False
        self._lag_since = None
        self._anchor = (phone.pose, tip)
        self._scale = phone.scale
        self.frame = phone.frame
        self._to_torso = to_torso
        self._turn = np.zeros(3)
        self._last_pose = phone.pose
        self._commanded = tip
        self._velocity = np.zeros(6)
        self._filter_p.reset(tip[0])
        quat = _quat_xyzw(tip[1])
        self._quat_prev = quat
        self._filter_q.reset(quat)

    def release(self) -> None:
        """Stop following; the next press re-anchors."""
        self.engaged = False
        self.limited = False
        self._lag_since = None
        self._anchor = None
        self._commanded = None
        self._last_pose = None

    def anchor_valid(self, phone: Phone) -> bool:
        """The anchor still describes the same ARKit world and calibration."""
        if self._anchor is None or phone.pose is None:
            return False
        anchor = self._anchor[0]
        return (
            phone.pose.session == anchor.session
            and phone.pose.calibration == anchor.calibration
        )

    def jumped(self, phone: Phone) -> bool:
        """Whether the phone's pose jumped since the last one seen.

        ARKit corrects itself in steps now and then while still reporting
        normal tracking. A step faster than a hand moves is treated as one,
        and the arm stops rather than following it.
        """
        pose, last = phone.pose, self._last_pose
        if pose is None or last is None or pose is last:
            return False
        self._last_pose = pose
        dt = max(pose.stamp - last.stamp, 1e-3)
        distance = float(np.linalg.norm(pose.position - last.position))
        angle = float(np.linalg.norm(rotation_vector(pose.rotation @ last.rotation.T)))
        limits = self.limits
        return (distance > limits.jump_m and distance / dt > limits.jump_speed) or (
            angle > limits.jump_rad and angle / dt > 5.0 * limits.jump_speed
        )

    # -- tracking ------------------------------------------------------------

    def _goal(self, pose: PhonePose) -> Pose:
        """The unfiltered target for ``pose``, following the turn on."""
        assert self._anchor is not None
        anchor = self._anchor[0]
        self._turn = unwrap_turn(
            rotation_vector(pose.rotation @ anchor.rotation.T), self._turn
        )
        return map_target(
            pose,
            anchor,
            self._anchor[1],
            self._scale,
            self.limits.rotation_scale,
            self._to_torso,
            self._turn,
        )

    def _reanchor(self, phone: Phone) -> None:
        """Anchor afresh where the target is now, taking up the phone's
        motion scale: a new scale applies from here on and never moves the
        arm by itself."""
        assert phone.pose is not None
        here = self._goal(phone.pose)
        self._anchor = (phone.pose, here)
        self._turn = np.zeros(3)
        self._scale = phone.scale

    def reframe(self, phone: Phone, to_torso: np.ndarray | None) -> None:
        """Take up the phone's new Frame choice (``F``: ``to_torso``) from
        here on, without moving the arm."""
        self._reanchor(phone)
        self.frame = phone.frame
        self._to_torso = to_torso

    def target(self, phone: Phone, now: float, dt: float) -> Pose:
        """This tick's tip target: mapped, filtered, then speed and
        acceleration limited."""
        assert self._anchor is not None and self._commanded is not None
        assert phone.pose is not None
        limits = self.limits
        if phone.scale != self._scale:
            self._reanchor(phone)
        goal = self._goal(phone.pose)
        position = self._filter_p.update(goal[0], now).astype(np.float64)
        quat = _quat_xyzw(goal[1])
        if self._quat_prev is not None and float(np.dot(quat, self._quat_prev)) < 0:
            quat = -quat  # stay on one hemisphere so the filter doesn't flip
        self._quat_prev = quat
        rotation = _matrix(self._filter_q.update(quat, now))
        p_cmd, r_cmd = self._commanded
        self._velocity[:3] = _follow(
            position - p_cmd, self._velocity[:3], limits.max_speed, limits.max_accel, dt
        )
        self._velocity[3:] = _follow(
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
