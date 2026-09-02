"""Hardware control classes for the Almond Axol dual-arm robot.

Provides :class:`AxolArm` (single-arm CAN bus controller) and :class:`Axol`
(dual-arm context manager that opens both buses and constructs all 16 motor drivers).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np

from ..constants import ARM_JOINTS, CAN_LEFT, CAN_RIGHT
from ..motor import (
    CanBus,
    ControlMode,
    Joint,
    Motor,
    MotorError,
    MotorGains,
    MotorStatus,
)
from .base import RobotBase
from .config import AxolConfig
from .control import (
    DAMP_BP_Q,
    DAMP_BP_W0,
    VEL_CUTOFF_FREQ,
    BandPass,
    Differentiator,
    compute_friction,
)
from .gravity import GravityCompensator

_logger = logging.getLogger(__name__)

# Per-joint position limits (rad).  shoulder_1, shoulder_2, and elbow are mirrored across arms.
SHOULDER_1_LEFT_LIMITS = (math.radians(-90), math.radians(180))
SHOULDER_1_RIGHT_LIMITS = (math.radians(-180), math.radians(90))
SHOULDER_2_LEFT_LIMITS = (math.radians(-180), math.radians(20))
SHOULDER_2_RIGHT_LIMITS = (math.radians(-20), math.radians(180))
ELBOW_LEFT_LIMITS = (math.radians(0), math.radians(150))
ELBOW_RIGHT_LIMITS = (math.radians(-150), math.radians(0))

LIMITS: dict[Joint, tuple[float, float]] = {
    Joint.SHOULDER_3: (math.radians(-135), math.radians(135)),
    Joint.WRIST_1: (math.radians(-135), math.radians(135)),
    Joint.WRIST_2: (math.radians(-90), math.radians(90)),
    Joint.WRIST_3: (math.radians(-90), math.radians(90)),
    # Gripper absent: both end stops vary per unit, found at runtime by _calibrate_gripper().
}

# Nominal open-to-close travel of the gripper (rad). Only a placeholder: the
# real stroke is measured between the two hard stops by
# ``AxolArm._calibrate_gripper()`` at enable time. Used to seed the
# pre-calibration gripper range and to bound the calibration sweep.
_GRIPPER_NOMINAL_TRAVEL = math.radians(290)

# Gripper end-stop calibration parameters.
_GRIPPER_TORQUE_THRESHOLD = 0.5  # Nm — pushing this hard into a stop ends a sweep
_GRIPPER_CALIB_STEP = 0.005  # rad per step
_GRIPPER_CALIB_SETTLE = 0.001  # s per step
# Longest sweep tolerated before concluding there is no stop in that direction.
_GRIPPER_CALIB_MAX_SWEEP = 2.5 * _GRIPPER_NOMINAL_TRAVEL
_GRIPPER_CALIB_MAX_STEPS = math.ceil(_GRIPPER_CALIB_MAX_SWEEP / _GRIPPER_CALIB_STEP)
# A stop is only recognised from torque pushing *along* the sweep. Torque
# building up against the sweep means the motor is already stalled on the
# far stop (its reported torque sign disagrees with the commanded motion) —
# abort rather than wind the impedance target further into the mechanism.
_GRIPPER_CALIB_TORQUE_ABORT = 2.0  # Nm
# Two stops closer together than this are not a gripper stroke (jammed jaw,
# or a sweep that stalled on an obstruction).
_GRIPPER_MIN_TRAVEL = math.radians(30)

# Impedance gains used only during gripper end-stop calibration.
_GRIPPER_CALIB_KP = 50.0
_GRIPPER_CALIB_KD = 1.0

# The gripper end stops vary per unit and are measured at runtime by
# ``_calibrate_gripper()``. They are persisted here so a reconnecting
# ``enable()`` can restore them without re-running the sweep (which physically
# closes then opens the jaw, dropping anything a holding gripper grips).
_GRIPPER_CALIB_PATH = Path.home() / ".almond" / "gripper_calibration.json"

# Tolerance (rad) around the calibrated travel range when validating a
# persisted calibration against the gripper's current position on restore.
# A shaft position outside the range means the calibration no longer matches
# the encoder (motor re-zeroed or power-cycled) and must not be trusted.
_GRIPPER_CALIB_MARGIN = 0.35


def _save_gripper_calibration(is_left: bool, open_pos: float, close_pos: float) -> None:
    """Persist one gripper's calibrated end stops (raw motor rad).

    A write failure only costs the ability to reconnect to a holding gripper
    later, so it is logged rather than failing the enable that produced the
    calibration.
    """
    side = "left" if is_left else "right"
    data: dict = {}
    try:
        existing = json.loads(_GRIPPER_CALIB_PATH.read_text())
        if isinstance(existing, dict):
            data = existing
    except (OSError, ValueError):
        # Missing or corrupt file — overwrite it with a fresh calibration
        # rather than losing the one we just measured.
        pass
    data[side] = {"open_pos": open_pos, "close_pos": close_pos, "saved_at": time.time()}
    try:
        _GRIPPER_CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _GRIPPER_CALIB_PATH.write_text(json.dumps(data, indent=2))
    except OSError as exc:
        _logger.warning(
            "Could not persist %s gripper calibration to %s: %s",
            side,
            _GRIPPER_CALIB_PATH,
            exc,
        )


def _load_gripper_calibration(is_left: bool) -> tuple[float, float]:
    """Return the persisted ``(open_pos, close_pos)`` (raw motor rad) for one gripper.

    Raises:
        MotorError: If no calibration has been persisted for this arm, or the
            file predates the two-stop calibration (open stop only).
    """
    side = "left" if is_left else "right"
    try:
        data = json.loads(_GRIPPER_CALIB_PATH.read_text())
        entry = data[side]
        return float(entry["open_pos"]), float(entry["close_pos"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise MotorError(
            f"No persisted two-stop gripper calibration for the {side} arm in "
            f"{_GRIPPER_CALIB_PATH} — a holding gripper cannot be reconnected "
            f"to without it; empty the gripper, then disable() and enable() "
            f"to calibrate"
        ) from exc


def arm_limits(joint: Joint, is_left: bool) -> tuple[float, float]:
    """Return (min, max) position limits for a joint on the given arm.

    Arm joints are in radians.  The gripper returns the normalised API range
    (0.0, 1.0) since gripper positions are always exposed as [0 = closed,
    1 = open] — the raw motor limits vary per unit and are calibrated at
    runtime by AxolArm._calibrate_gripper().
    """
    if joint == Joint.SHOULDER_1:
        return SHOULDER_1_LEFT_LIMITS if is_left else SHOULDER_1_RIGHT_LIMITS
    if joint == Joint.SHOULDER_2:
        return SHOULDER_2_LEFT_LIMITS if is_left else SHOULDER_2_RIGHT_LIMITS
    if joint == Joint.ELBOW:
        return ELBOW_LEFT_LIMITS if is_left else ELBOW_RIGHT_LIMITS
    if joint == Joint.GRIPPER:
        return (0.0, 1.0)
    return LIMITS.get(joint, (-math.inf, math.inf))


# Joints whose joint frame is kinematically mirrored across arms but whose
# limits are symmetric (equidistant from 0).  For these the calibration end
# stop on the right arm is the upper bound rather than the lower (the
# physical end stop closer to rest is at +X on the right arm and -X on the
# left).  Joints with already-asymmetric per-arm limits (s1, s2, elbow) are
# disambiguated by their range and don't need to be listed here.
_MIRRORED_SYMMETRIC_JOINTS: frozenset[Joint] = frozenset({Joint.WRIST_2, Joint.WRIST_3})

# Joints that may be zeroed at EITHER of their two end stops.  The wrist_2 /
# wrist_3 stops are the only ones that are not laser-aligned, so per unit one
# stop can be better placed than the other; the guided zeroing flow therefore
# accepts whichever stop the operator brings the joint to.  Which stop a
# given robot was zeroed at is detected at runtime from the encoder reading
# itself (see end_stop_offset_from_position) — no per-robot file is needed,
# and robots zeroed by the old forced-side flow resolve to the offsets they
# have always had.
EITHER_STOP_JOINTS: frozenset[Joint] = frozenset({Joint.WRIST_2, Joint.WRIST_3})

# A motor-frame reading within this band of 0 means the joint is parked at
# (or pressed against) its calibration end stop — the one position where the
# two zeroing conventions cannot be told apart.  The reading at the
# calibration stop is ~0 by construction regardless of the stop's placement
# error, so only backlash/compliance widens the band; 3° is generous.
_STOP_SIDE_AMBIGUOUS_RAD = math.radians(3.0)

# Slack past the nominal end-to-end travel when sanity-checking a reading in
# end_stop_offset_from_position, covering stop placement error on both ends.
_STOP_SIDE_RANGE_SLACK_RAD = math.radians(15.0)


def end_stop_offset_from_position(joint: Joint, motor_pos: float) -> float:
    """Infer an either-stop joint's motor→joint offset from one encoder reading.

    Joints in :data:`EITHER_STOP_JOINTS` have symmetric limits ``(-L, +L)``
    and are zeroed at one of the two physical end stops.  The zero location
    fixes the sign of every reachable motor-frame reading, so a single read
    reveals which stop was used:

    - zeroed at the upper stop (``+L``): readings span ``[-2L, 0]`` → offset ``+L``
    - zeroed at the lower stop (``-L``): readings span ``[0, +2L]`` → offset ``-L``

    Args:
        joint:     Joint to resolve; must be in :data:`EITHER_STOP_JOINTS`.
        motor_pos: Current motor-frame position (rad).

    Returns:
        The offset such that ``joint_angle = motor_angle + offset``.

    Raises:
        MotorError: If the reading is too close to 0 (parked at the
            calibration stop — ambiguous), too close to the far stop (a
            single-turn boot reading past ±180° wraps by 360° and mimics the
            other side — also ambiguous), or outside the joint's physical
            travel (the encoder zero is not at either stop; re-zero).
    """
    if joint not in EITHER_STOP_JOINTS:
        raise ValueError(f"{joint} is not zeroed at a variable end stop")
    lo, hi = LIMITS[joint]
    travel = hi - lo
    if abs(motor_pos) < _STOP_SIDE_AMBIGUOUS_RAD:
        raise MotorError(
            f"Cannot tell which end stop {joint.name} was zeroed at: the "
            f"joint is at/near an end stop (motor frame "
            f"{math.degrees(motor_pos):+.1f}°). Move it away from the end "
            f"stops and retry."
        )
    if abs(motor_pos) > travel + _STOP_SIDE_RANGE_SLACK_RAD:
        raise MotorError(
            f"{joint.name} reads {math.degrees(motor_pos):+.1f}° in the motor "
            f"frame — outside its physical travel, so its encoder zero is not "
            f"at an end stop. Re-run `axol motor.set-zero-pos --guided`."
        )
    # These joints span exactly 180° of travel, and after a power cycle the
    # single-turn encoder re-reads position within ±180° of zero — so a
    # joint pressed slightly PAST the far stop wraps by 360° and reads like a
    # small excursion past the OPPOSITE stop, flipping the detected side.
    # Readings within the stop-placement slack of the far stop are therefore
    # genuinely ambiguous and must not be trusted.
    if abs(motor_pos) > travel - _STOP_SIDE_RANGE_SLACK_RAD:
        raise MotorError(
            f"Cannot tell which end stop {joint.name} was zeroed at: the "
            f"joint is at/near the far end stop (motor frame "
            f"{math.degrees(motor_pos):+.1f}°), where a reading can also be "
            f"a 360° single-turn wrap from just past the opposite stop. Move "
            f"it away from the end stops and retry."
        )
    return hi if motor_pos < 0 else lo


def fixed_stop_wrap_correction(joint: Joint, is_left: bool, motor_pos: float) -> float:
    """Return the ±2π unwrap a fixed-stop joint's reading needs (0.0 if none).

    Joints outside :data:`EITHER_STOP_JOINTS` are always zeroed at
    :func:`closer_end_stop`, so the encoder zero coincides with that stop and
    every reachable motor-frame position lies between the two stops:
    ``lo - offset`` to ``hi - offset`` (one of which is 0).

    Both motor families track multi-turn position only in RAM; the persistent
    zero lives in a single-turn absolute encoder on the output shaft.  After a
    power cycle — or a MyActuator 0x76 reset, issued by ``set_control_mode``
    and ``set_zero_position`` — the position is re-read within ±180° of the
    stored zero, so a joint parked more than 180° from its calibration stop
    comes back reading exactly 360° off.  Every fixed-stop joint's travel plus
    slack is under 360° (the widest is 270° + 2·15°), so at most one of
    ``{reading, reading ± 360°}`` lands in the expected band: that candidate
    is the true position, and the returned correction folds into the joint's
    motor→joint offset (``joint = motor + offset + correction``; commands
    apply the inverse) so the wrapped motor frame is handled transparently.

    The check is necessarily one-sided: an unset zero that happens to land
    inside the band (directly or after unwrapping) cannot be told apart from
    a valid calibration by a position read alone.

    Args:
        joint:     Joint to check; must not be in :data:`EITHER_STOP_JOINTS`
                   (nor the gripper, which has no zero to set).
        is_left:   Which arm the joint is on (limits are mirrored).
        motor_pos: Current motor-frame position (rad).

    Returns:
        The wrap correction in radians: ``0.0``, ``+2π``, or ``-2π``.

    Raises:
        MotorError: If neither the reading nor a ±360° unwrap of it falls
            inside the joint's physical travel measured from its calibration
            end stop — the zero was never set, or is stale.
    """
    if joint in EITHER_STOP_JOINTS or joint == Joint.GRIPPER:
        raise ValueError(f"{joint} is not zeroed at a fixed end stop")
    offset, _ = closer_end_stop(joint, is_left)
    lo, hi = arm_limits(joint, is_left)
    band_lo = lo - offset - _STOP_SIDE_RANGE_SLACK_RAD
    band_hi = hi - offset + _STOP_SIDE_RANGE_SLACK_RAD
    for correction in (0.0, math.tau, -math.tau):
        if band_lo <= motor_pos + correction <= band_hi:
            return correction
    raise MotorError(
        f"{joint.name} reads {math.degrees(motor_pos):+.1f}° in the motor "
        f"frame — neither it nor a ±360° single-turn wrap of it falls in "
        f"the [{math.degrees(lo - offset):+.1f}°, "
        f"{math.degrees(hi - offset):+.1f}°] band expected when its "
        f"encoder zero is at the calibration end stop, so the zero has "
        f"not been set (or is stale). "
        f"Re-run `axol motor.set-zero-pos --guided`."
    )


def closer_end_stop(joint: Joint, is_left: bool) -> tuple[float, int]:
    """Return ``(target_rad, expected_motion_sign)`` for the joint's calibration end stop.

    Picks the limit with the smallest absolute value, except when that limit
    is exactly 0 (it coincides with the rest position, so direction detection
    is unreliable) — in that case the *other* limit is used instead.  When
    both limits are equidistant the lower bound wins, except for joints in
    :data:`_MIRRORED_SYMMETRIC_JOINTS` on the right arm, which default to
    the upper bound to follow the kinematic mirror.

    ``expected_motion_sign`` is the sign of motion when the user starts inside
    the joint range and moves toward the chosen end stop:

    - target == lower bound  → range extends positively → motion is negative
    - target == upper bound  → range extends negatively → motion is positive

    Used by the guided ``set-zero-pos`` flow and by :class:`AxolArm` to derive
    the per-joint offset between motor frame (zero at end stop) and joint
    frame (zero at rest).  For joints in :data:`EITHER_STOP_JOINTS` this is
    only the historical/default side: those may be zeroed at either stop, and
    the side actually used is detected at runtime from the encoder reading
    (:func:`end_stop_offset_from_position`).  Not defined for
    ``Joint.GRIPPER``.
    """
    if joint == Joint.GRIPPER:
        raise ValueError("closer_end_stop is undefined for the gripper")
    lo, hi = arm_limits(joint, is_left)
    if lo == 0.0:
        return hi, +1
    if hi == 0.0:
        return lo, -1
    if abs(hi) < abs(lo):
        return hi, +1
    if abs(lo) < abs(hi):
        return lo, -1
    # Equidistant: default to lo, except mirrored joints on the right arm.
    if not is_left and joint in _MIRRORED_SYMMETRIC_JOINTS:
        return hi, +1
    return lo, -1


class AxolArm:
    """Controls one 7-DOF + gripper arm over a single CAN bus.

    Not instantiated directly — access via ``axol.left`` or ``axol.right``.

    On the gripperless SKU (``AxolConfig.has_gripper = False``) no gripper
    motor is constructed: gripper commands (the last element of every
    ``(8,)`` array) are ignored and gripper reads report ``0.0``.
    """

    def __init__(
        self,
        bus: CanBus,
        config: AxolConfig,
        gravity_comp: GravityCompensator,
        is_left: bool = True,
    ) -> None:
        """Construct an AxolArm.

        Args:
            bus:          Shared CAN bus for this arm (one per physical interface).
            config:       Full dual-arm gains config; the correct side is selected via ``is_left``.
            gravity_comp: Shared MuJoCo-based gravity compensator (one per Axol).
            is_left:      ``True`` for the left arm, ``False`` for the right.
        """
        self._config = config
        self._arm_config = config.left if is_left else config.right
        self._gravity_comp = gravity_comp
        self._is_left = is_left
        self._has_gripper = config.has_gripper
        # Gripperless SKU: the gripper motor is simply never constructed, so
        # every loop over ``self.motors`` skips it automatically.
        self.motors: dict[Joint, Motor] = {
            joint: Motor(bus, joint)
            for joint in Joint
            if config.has_gripper or joint != Joint.GRIPPER
        }
        # The impedance-command encodings clamp kd to the firmware range
        # silently, and there is no host-side fallback for the excess — an
        # oversized kd is a config/calibration error, so shout once here
        # rather than ship less damping than the tuner thinks they have.
        for joint, motor in self.motors.items():
            if joint == Joint.GRIPPER:
                continue
            jc = getattr(self._arm_config, joint.value)
            if jc.kd > motor.kd_max:
                _logger.warning(
                    "%s %s: configured kd=%.2f exceeds the firmware max %.1f "
                    "and will be clamped — lower kd, or add damping via "
                    "kd_host where it is phase-safe",
                    "left" if is_left else "right",
                    joint.value,
                    jc.kd,
                    motor.kd_max,
                )
        # q_des → v_des → a_des (commanded), and q_meas → v_meas. v_des feeds
        # the impedance-control velocity FF and the friction model; a_des
        # feeds inertia FF (``j_eff``); v_meas feeds host-side damping
        # (``kd_host``, needed on the shoulders where firmware kd can't damp
        # the low-frequency resonance) — all in :class:`JointConfig`.
        #
        # v_meas differentiates the positions cached from impedance feedback
        # frames against the frames' own CAN receive timestamps (see
        # ``Differentiator.differentiate``): wall-clock differentiation turns
        # scheduling jitter into velocity noise proportional to joint speed,
        # which kd_host amplifies into torque chatter (measured ±1.3 Nm at
        # 1 rad/s with kd_host=40). The motor-*reported* velocity is not
        # used: MyActuator's firmware estimate lags too much to damp the
        # shoulders' ~2.3 Hz resonance — feeding it to kd_host measured
        # violently unstable (the same lag is why firmware kd underdelivers).
        #
        # Motor-facing paths (v_des impedance target, friction FF, a_des →
        # j_eff FF) keep the slow pole. The host damping term instead gets
        # its own chain — fast differentiators on both commanded and measured
        # positions feeding a band-pass centred on each joint's own
        # structural mode (``kd_host_hz``: ~3 Hz shoulders, 9.5 Hz elbow) —
        # so kd_host arrives in phase at the mode it must damp without
        # passing the delayed, anti-phase gain that excites 25-35 Hz
        # structural modes (see the BandPass docstring in .control for the
        # measured trade).
        n_j = len(list(Joint))
        self._vel_diff = Differentiator(n=n_j)
        self._accel_diff = Differentiator(n=n_j)
        self._vel_fast_diff = Differentiator(n=n_j, cutoff=VEL_CUTOFF_FREQ)
        self._meas_vel_diff = Differentiator(n=n_j, cutoff=VEL_CUTOFF_FREQ)
        # Host-damping band-pass centres: joints with an explicit kd_host_hz
        # (structural modes — the elbow) keep it fixed; joints with None get
        # pose-tracked centres each cycle (see motion_control: the shoulders'
        # impedance mode ωn = √(kp/J(q)) moves 2.2 → 5.4 Hz between rest and
        # raised-to-the-side, and a fixed rest centre delivered only ~14% of
        # the damping in the 4.3-8.6 Hz teleop burst band). Config carries
        # Hz (the operator-facing unit); the control math wants rad/s, so
        # convert exactly once here.
        self._damp_w0 = [
            2 * math.pi * getattr(self._arm_config, j.value).kd_host_hz
            if j != Joint.GRIPPER
            and getattr(self._arm_config, j.value).kd_host_hz is not None
            else DAMP_BP_W0
            for j in Joint
        ]
        self._damp_w0_tracked = [
            j != Joint.GRIPPER and getattr(self._arm_config, j.value).kd_host_hz is None
            for j in Joint
        ]
        # Per-joint band-pass Q: the 0.8 default suits pose-tracked centres
        # (an estimate deserves a wide net); a joint pinned on a measured
        # ring can run narrower (higher q) so the damping stops reaching
        # into the <1.5 Hz intentional-motion band and dragging the final
        # approach.
        self._damp_q = [
            getattr(self._arm_config, j.value).kd_host_q
            if j != Joint.GRIPPER
            and getattr(self._arm_config, j.value).kd_host_q is not None
            else DAMP_BP_Q
            for j in Joint
        ]
        self._damp_bp = BandPass(n=n_j, w0=self._damp_w0, q=self._damp_q)
        self._last_q_commanded: np.ndarray | None = None
        self._gc_hold_q: np.ndarray | None = None
        self._gc_hold_free: frozenset[Joint] | None = None

        # Reference reflected inertia normalizing the per-cycle kd_host
        # schedule in motion_control(): per joint, the *maximum* over a
        # coarse grid of arm shapes (shoulder/elbow combinations). Host
        # damping is fully applied only near a joint's max-inertia pose,
        # where its mode is slowest and the ~100 Hz host loop is safest,
        # and tapers as J(q) drops and the mode speeds up. For every joint
        # except shoulder_3 the max is at (or equal to) the rest pose, so
        # this matches the previous rest-pose anchor exactly; shoulder_3 is
        # inverted — at rest the forearm lies along its axis (J ≈ 3% of
        # max, fast mode, host damping unstable) and the arm extended with
        # the elbow bent is its max — so its schedule must anchor there.
        j_link_rest = gravity_comp.gravity_and_inertia_arm(
            np.zeros(len(ARM_JOINTS), dtype=np.float32), is_left=is_left
        )[1].astype(np.float64)
        self._inertia_ref = j_link_rest.copy()
        for s1 in (0.0, 1.57, -1.57):
            for s2 in (0.0, -1.57):
                for s3 in (0.0, 1.57, -1.57):
                    for el in (0.0, 1.2, -1.2):
                        q = np.array([s1, s2, s3, el, 0.0, 0.0, 0.0], dtype=np.float32)
                        ine = gravity_comp.gravity_and_inertia_arm(q, is_left=is_left)[
                            1
                        ].astype(np.float64)
                        np.maximum(self._inertia_ref, ine, out=self._inertia_ref)

        # Inertia-FF pose schedule. The tuned ``j_eff`` (fit at the rest
        # pose) is the sum of two physically different terms: the reflected
        # rotor inertia (motor rotor × gear², pose-independent — dominant on
        # shoulder_3 and the elbow, whose tuned values exceed their link
        # inertia severalfold) and the link-chain inertia J_link(q), which
        # varies strongly with arm shape (shoulder_1: 1.06 kg·m² at rest →
        # 0.81 elbow bent → ~0.02 raised to the side). Feeding a constant
        # j_eff over-torques every acceleration transient away from the rest
        # pose — measured as whole-arm jitter when shoulder_1 launches or
        # stops with the arm reaching in front — so motion_control() scales
        # the FF by (J_rotor + J_link(q)) / (J_rotor + J_link(0)): exactly
        # the tuned value at the rest pose, tracking the true inertia
        # elsewhere. The rotor term is anchored to the construction-time
        # (calibrated, pre-blend) j_eff; runtime stiffness blending still
        # applies multiplicatively through ``gains.j_eff``.
        j_eff_cfg = np.array(
            [getattr(self._arm_config, j.value).j_eff for j in ARM_JOINTS],
            dtype=np.float64,
        )
        self._j_rotor = np.maximum(j_eff_cfg - j_link_rest, 0.0)
        denom = self._j_rotor + j_link_rest
        # Joints with no tuned j_eff and negligible link inertia (wrists):
        # the scale multiplies j_eff = 0 anyway, so just avoid dividing by ~0.
        self._j_ff_denom = np.where(denom > 1e-9, denom, 1.0)

        # Clipping arrays.  Arm joints are in joint frame (0 = rest position,
        # matching the URDF and ``arm_limits``); gripper entries are in raw
        # motor radians (``arm_limits`` returns normalised [0, 1] for the
        # gripper, so the gripper bounds are seeded with raw defaults here
        # and ``_calibrate_gripper()`` overwrites them on enable).
        joints = list(Joint)
        self._gripper_i: int = joints.index(Joint.GRIPPER)
        self._limits_lo = np.array(
            [0.0 if j == Joint.GRIPPER else arm_limits(j, is_left)[0] for j in joints],
            dtype=float,
        )
        self._limits_hi = np.array(
            [0.0 if j == Joint.GRIPPER else arm_limits(j, is_left)[1] for j in joints],
            dtype=float,
        )
        # Raw motor angles of the gripper's two hard stops. The jaw closes in
        # the configured ``close_direction``, so which of the two is the
        # numerically larger angle differs between the mirrored left and right
        # grippers — the [0 = closed, 1 = open] normalisation goes through
        # ``_gripper_to_raw`` / ``_gripper_from_raw`` and never assumes an
        # order. Pre-calibration placeholders assume zero is closed and a
        # nominal stroke — do not rely on for actual motion.
        self._gripper_open: float = 0.0
        self._gripper_close: float = 0.0
        self._set_gripper_range(
            open_pos=-self._arm_config.gripper.close_direction
            * _GRIPPER_NOMINAL_TRAVEL,
            close_pos=0.0,
        )

        # Per-joint offset between motor frame and joint frame:
        #   joint_angle (rad) = motor_angle (rad) + offset
        # After end-stop calibration the motor's encoder zero coincides with
        # an end stop, so motor 0 → joint angle = that stop's limit.  Most
        # joints are always zeroed at closer_end_stop(); EITHER_STOP_JOINTS
        # (wrist_2/wrist_3) may be zeroed at either stop, so their offsets
        # start as NaN and are detected from the first encoder reading by
        # resolve_joint_offsets().  Fixed-stop offsets may additionally gain
        # a ±2π term when the motor booted with a single-turn wrap (parked
        # >180° from zero at power-up/reset — see
        # fixed_stop_wrap_correction), applied during zero verification.
        # The gripper offset is 0 because the gripper uses its own [0, 1]
        # normalisation between its two torque-detected hard stops.
        self._joint_offsets = np.array(
            [
                0.0
                if j == Joint.GRIPPER
                else math.nan
                if j in EITHER_STOP_JOINTS
                else closer_end_stop(j, is_left)[0]
                for j in joints
            ],
            dtype=float,
        )
        self._unresolved_offsets: set[Joint] = set(EITHER_STOP_JOINTS)
        # Fixed-stop joints whose encoder zero has not been sanity-checked
        # yet.  resolve_joint_offsets() verifies each one's reading is
        # plausible for a zero at its calibration stop (an unset zero would
        # make every joint-frame value garbage), folds any ±360° single-turn
        # boot wrap into the joint's offset, and removes it from the set.
        self._unverified_zeros: set[Joint] = set(ARM_JOINTS) - EITHER_STOP_JOINTS
        self._offset_lock = asyncio.Lock()
        # Realtime-core hook: production motion_control and
        # gravity_compensate hand their per-joint 9-float tuples
        # (p_des motor-frame, mode, kp, kd, gravity t_ff, the pose-scheduled
        # damping coefficients kd_host/w0/q, and the pose-scaled j_eff) to
        # this callable instead of sending on the CAN bus — the Rust core
        # owns the wire and computes the velocity/friction/inertia/damping
        # terms itself each tick from its own tracker and feedback states.
        self._command_sink: (
            Callable[
                [list[tuple[float, ...]]],
                None,
            ]
            | None
        ) = None

    def _pad_gripper(self, values: list) -> list:
        """Insert a ``0.0`` placeholder in the gripper slot when absent.

        Per-motor reads iterate ``self.motors`` (7 entries on the gripperless
        SKU); this restores the public ``(8,)`` Joint-enum-order shape.
        """
        if not self._has_gripper:
            values = list(values)
            values.insert(self._gripper_i, 0.0)
        return values

    # ------------------------------------------------------------------ #
    # Joint-offset resolution                                              #
    # ------------------------------------------------------------------ #

    async def resolve_joint_offsets(self, joints: set[Joint] | None = None) -> None:
        """Detect either-stop zeros and reject joints whose zero was never set.

        Two one-shot bring-up gates, both driven by a single encoder read per
        joint:

        - :data:`EITHER_STOP_JOINTS`: derive the motor→joint offset from the
          sign of the reading (see :func:`end_stop_offset_from_position`),
          which also rejects readings outside the joint's physical travel.
        - Every other arm joint: verify the reading is plausible for an
          encoder zero at the calibration stop and detect a ±360°
          single-turn boot wrap (:func:`fixed_stop_wrap_correction`) — the
          wrap is folded into the joint's offset so positions and commands
          stay correct, while an unset zero (nothing lands in the band)
          fails bring-up.

        Idempotent and near-free once resolved/verified, so every ``AxolArm``
        entry point that uses joint-frame values calls it; diagnostics that
        enable motors directly and then read ``_joint_offsets`` must call it
        themselves.

        Args:
            joints: Restrict resolution to this subset (for flows where only
                some motors are on the bus).  ``None`` resolves all.

        Raises:
            MotorError: If a joint cannot be read, is parked at an end stop
                (ambiguous), or reads outside its physical travel — meaning
                its encoder zero has not been set (or is stale).
        """
        if not self._unresolved_offsets and not self._unverified_zeros:
            return
        async with self._offset_lock:
            pending = (
                set(self._unresolved_offsets)
                if joints is None
                else self._unresolved_offsets & set(joints)
            )
            pending_verify = (
                set(self._unverified_zeros)
                if joints is None
                else self._unverified_zeros & set(joints)
            )
            if not pending and not pending_verify:
                return
            joint_index = {j: i for i, j in enumerate(Joint)}
            side = "left" if self._is_left else "right"
            for joint in pending:
                offset, pos = await self._detect_stop_side(joint, side)
                self._joint_offsets[joint_index[joint]] = offset
                self._unresolved_offsets.discard(joint)
                _logger.info(
                    "%s %s zero detected at the %+.0f° end stop "
                    "(motor %+.3f rad → joint offset %+.3f rad)",
                    side,
                    joint.name,
                    math.degrees(offset),
                    pos,
                    offset,
                )
            for joint in pending_verify:
                correction = await self._verify_fixed_stop_zero(joint, side)
                self._joint_offsets[joint_index[joint]] = (
                    closer_end_stop(joint, self._is_left)[0] + correction
                )
                self._unverified_zeros.discard(joint)
                if correction:
                    _logger.warning(
                        "%s %s reads 360° off its zero (single-turn encoder "
                        "wrap after a power cycle/reset) — compensating with "
                        "a %+.0f° offset",
                        side,
                        joint.name,
                        math.degrees(correction),
                    )

    async def _read_motor_position(self, joint: Joint, use_cache: bool) -> float:
        """Read one joint's motor-frame position, telemetry-aware.

        ``use_cache=True`` accepts an already-cached telemetry sample;
        otherwise a direct read is made — or, while telemetry is polling
        (direct reads are rejected then), a sample is awaited.
        """
        motor = self.motors[joint]
        if use_cache and motor.has_position:
            return motor.position
        if not motor.telemetry_active:
            return await motor.get_position()
        loop = asyncio.get_event_loop()
        deadline = loop.time() + 5.0
        while not motor.has_position:
            if loop.time() >= deadline:
                raise MotorError("no telemetry sample within 5 s")
            await asyncio.sleep(0.01)
        return motor.position

    async def _detect_stop_side(self, joint: Joint, side: str) -> tuple[float, float]:
        """Read one joint's position and detect its zeroed end stop.

        Retries once on an implausible reading: detection is a one-shot gate
        at bring-up, and a single poisoned frame (e.g. a command ack decoded
        as feedback) must not abort the whole enable when a fresh read
        settles it.  Returns ``(offset, motor_pos)``.
        """
        last_exc: MotorError | None = None
        for attempt in range(2):
            if attempt:
                await asyncio.sleep(0.05)
            try:
                pos = await self._read_motor_position(joint, use_cache=attempt == 0)
            except MotorError as exc:
                raise MotorError(
                    f"Could not read {side} {joint.name} to detect its "
                    f"zeroed end stop: {exc}"
                ) from exc
            try:
                return end_stop_offset_from_position(joint, pos), pos
            except MotorError as exc:
                last_exc = exc
        raise MotorError(f"{side} arm: {last_exc}") from last_exc

    async def _verify_fixed_stop_zero(self, joint: Joint, side: str) -> float:
        """Verify one fixed-stop joint's zero; return its ±2π wrap correction.

        Rejects a joint whose reading implies an unset zero (see
        :func:`fixed_stop_wrap_correction`).  Same retry rationale as
        :meth:`_detect_stop_side`: the check is a one-shot gate at bring-up,
        and a single poisoned frame must not fail the whole enable when a
        fresh read settles it.
        """
        last_exc: MotorError | None = None
        for attempt in range(2):
            if attempt:
                await asyncio.sleep(0.05)
            try:
                pos = await self._read_motor_position(joint, use_cache=attempt == 0)
            except MotorError as exc:
                raise MotorError(
                    f"Could not read {side} {joint.name} to verify its "
                    f"encoder zero: {exc}"
                ) from exc
            try:
                return fixed_stop_wrap_correction(joint, self._is_left, pos)
            except MotorError as exc:
                last_exc = exc
        raise MotorError(f"{side} arm: {last_exc}") from last_exc

    def _require_offsets_resolved(self) -> None:
        """Raise if any arm joint's zero is still undetected or unverified."""
        pending = self._unresolved_offsets | self._unverified_zeros
        if pending:
            names = ", ".join(j.name for j in Joint if j in pending)
            raise MotorError(
                f"The encoder zero of {names} has not been detected/verified "
                f"yet — enable(), start_telemetry(), or get_positions() first "
                f"(each resolves it automatically)"
            )

    # ------------------------------------------------------------------ #
    # Polling                                                              #
    # ------------------------------------------------------------------ #

    async def start_telemetry(self, hz: float, *, torque: bool = False) -> None:
        """Start background telemetry polling on all motors at the given frequency.

        Args:
            hz:     Poll frequency in Hz.
            torque: If True, also fetch and cache torque each cycle.
        """
        # Resolve before polling starts: direct position reads are rejected
        # while telemetry runs, and the ``positions`` property needs the
        # offsets as soon as the cache fills.
        await self.resolve_joint_offsets()
        await asyncio.gather(
            *[m.start_telemetry(hz, torque=torque) for m in self.motors.values()]
        )

    async def stop_telemetry(self) -> None:
        """Stop the background telemetry polling loop on all motors."""
        await asyncio.gather(*[m.stop_telemetry() for m in self.motors.values()])

    async def wait_for_telemetry(self, timeout: float = 5.0) -> None:
        """Block until every motor has reported at least one position.

        Call after :meth:`start_telemetry` and before the first read of
        :attr:`positions`. Motors can take a while to answer their first
        poll (MyActuator motors reboot during ``set_control_mode`` and may
        still be coming back up), so a fixed sleep is not reliable.

        Args:
            timeout: Maximum time to wait (s) before raising MotorError.
        """
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while True:
            missing = [j.name for j, m in self.motors.items() if not m.has_position]
            if not missing:
                return
            if loop.time() >= deadline:
                raise MotorError(
                    f"No telemetry from {', '.join(missing)} after {timeout:.1f}s — "
                    f"check power and CAN wiring"
                )
            await asyncio.sleep(0.01)

    @property
    def positions(self) -> np.ndarray:
        """Latest cached joint positions. Requires start_telemetry().

        Returns shape (8,) array in Joint enum order. Arm joints are in
        radians in the joint frame (0 = rest position); the gripper is
        normalized to [0, 1] (0.0 = closed, 1.0 = fully open), consistent
        with set_position_velocity and motion_control. On the gripperless
        SKU the gripper element is 0.0.
        """
        self._require_offsets_resolved()
        values = self._pad_gripper([self.motors[j].position for j in self.motors])
        gripper_i = self._gripper_i
        if self._has_gripper:
            values[gripper_i] = self._gripper_from_raw(values[gripper_i])
        arr = np.array(values, dtype=np.float32)
        return arr + self._joint_offsets.astype(np.float32)

    @property
    def torques(self) -> np.ndarray:
        """Latest cached joint torques (Nm / A). Requires start_telemetry().

        Returns shape (8,) array in Joint enum order (gripper element 0.0 on
        the gripperless SKU).
        """
        values = self._pad_gripper([m.torque for m in self.motors.values()])
        return np.array(values, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Arm-wide commands                                                    #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Gripper range                                                        #
    # ------------------------------------------------------------------ #

    @property
    def gripper_travel(self) -> float:
        """Calibrated open-to-close stroke of the gripper (rad, always positive).

        The nominal placeholder until :meth:`enable` has measured the stops.
        """
        return abs(self._gripper_open - self._gripper_close)

    def _set_gripper_range(self, open_pos: float, close_pos: float) -> None:
        """Adopt a pair of gripper hard stops (raw motor rad).

        Records them for the [0, 1] normalisation and seeds the raw clipping
        bounds with the ordered pair — ``np.clip`` needs ``lo <= hi``, which
        the open/closed pair does not guarantee on a mirrored gripper.
        """
        self._gripper_open = float(open_pos)
        self._gripper_close = float(close_pos)
        gripper_i = self._gripper_i
        self._limits_lo[gripper_i] = min(open_pos, close_pos)
        self._limits_hi[gripper_i] = max(open_pos, close_pos)

    def _gripper_to_raw(self, opening: float) -> float:
        """Normalised opening (0.0 = closed, 1.0 = open) → raw motor rad."""
        return self._gripper_close + opening * (
            self._gripper_open - self._gripper_close
        )

    def _gripper_from_raw(self, raw: float) -> float:
        """Raw motor rad → normalised opening (0.0 = closed, 1.0 = open)."""
        return (raw - self._gripper_close) / (self._gripper_open - self._gripper_close)

    async def _seek_gripper_stop(self, direction: int) -> float:
        """Step the gripper in ``direction`` (±1) until it stalls on a hard stop.

        Each step nudges the impedance target ``_GRIPPER_CALIB_STEP`` further
        and reads the motor torque; the stop is reached once the torque
        pushing *along* the sweep exceeds ``_GRIPPER_TORQUE_THRESHOLD``. The
        signed test matters for the second sweep of a calibration, which
        starts pressed against the stop the first one found: that torque
        points the other way and must not end the sweep early.

        Returns:
            The measured shaft position at the stop (raw motor rad).

        Raises:
            MotorError: If torque builds up *against* the sweep (the motor
                reports pushing opposite to its commanded motion — stalled on
                the stop it should be leaving, or an inverted torque sign),
                or no stop is met within ``_GRIPPER_CALIB_MAX_SWEEP``.
        """
        motor = self.motors[Joint.GRIPPER]
        side = "left" if self._is_left else "right"
        target = await motor.get_position()

        for _ in range(_GRIPPER_CALIB_MAX_STEPS):
            target += direction * _GRIPPER_CALIB_STEP
            await motor.set_impedance(
                target, 0.0, _GRIPPER_CALIB_KP, _GRIPPER_CALIB_KD, 0.0
            )
            await asyncio.sleep(_GRIPPER_CALIB_SETTLE)
            torque = await motor.get_torque()
            along = torque * direction
            if along >= _GRIPPER_TORQUE_THRESHOLD:
                return await motor.get_position()
            if along <= -_GRIPPER_CALIB_TORQUE_ABORT:
                await self._unload_gripper_target()
                raise MotorError(
                    f"{side} gripper calibration: torque {torque:+.2f} Nm builds "
                    f"up against the sweep (direction {direction:+d}) — the motor "
                    f"reports pushing opposite to its commanded motion, so it is "
                    f"stalled on the stop the sweep should be leaving (or its "
                    f"torque sign is inverted); the jaw must be free to move"
                )

        await self._unload_gripper_target()
        raise MotorError(
            f"{side} gripper calibration: no hard stop met within "
            f"{math.degrees(_GRIPPER_CALIB_MAX_SWEEP):.0f}° sweeping in direction "
            f"{direction:+d} — is the gripper attached and the jaw free to move?"
        )

    async def _unload_gripper_target(self) -> None:
        """Re-target the calibration impedance hold at the shaft's actual position.

        Called before a failed sweep raises, so the motor is not left leaning
        on whatever it stalled against with the wound-up target.
        """
        motor = self.motors[Joint.GRIPPER]
        current = await motor.get_position()
        await motor.set_impedance(
            current, 0.0, _GRIPPER_CALIB_KP, _GRIPPER_CALIB_KD, 0.0
        )

    async def _calibrate_gripper(self) -> None:
        """Measure both gripper hard stops and leave the jaw open.

        Sweeps the jaw closed first (in the configured
        ``ArmConfig.gripper.close_direction``) until it stalls on the closed
        stop, then back the other way until it stalls on the open stop, so the
        gripper finishes open. The stroke is whatever the two stops measure —
        nothing about it is assumed. Adopts the pair for the [0, 1]
        normalisation and raw clipping (see :meth:`_set_gripper_range`) and
        persists it so a later reconnecting :meth:`enable` can restore it
        without moving the gripper.

        Must be called with the gripper motor already enabled and in IMPEDANCE mode.

        Raises:
            MotorError: If a stop is not found (see :meth:`_seek_gripper_stop`)
                or the two stops are closer together than ``_GRIPPER_MIN_TRAVEL``.
        """
        close_direction = self._arm_config.gripper.close_direction
        side = "left" if self._is_left else "right"

        close_pos = await self._seek_gripper_stop(close_direction)
        open_pos = await self._seek_gripper_stop(-close_direction)

        travel = abs(open_pos - close_pos)
        if travel < _GRIPPER_MIN_TRAVEL:
            raise MotorError(
                f"{side} gripper calibration: stops only {math.degrees(travel):.1f}° "
                f"apart (closed {close_pos:.3f} rad, open {open_pos:.3f} rad) — "
                f"the jaw is jammed or met an obstruction"
            )
        _logger.info(
            "%s gripper calibrated: closed %.3f rad, open %.3f rad (stroke %.1f°)",
            side,
            close_pos,
            open_pos,
            math.degrees(travel),
        )
        self._set_gripper_range(open_pos, close_pos)
        _save_gripper_calibration(self._is_left, open_pos, close_pos)

    async def enable(self, hold: bool = True) -> None:
        """Enable all arm motors in IMPEDANCE mode and the gripper in POSITION_FORCE mode.

        Idempotent per motor: joints that are already enabled and holding
        torque (a previous session died or disconnected while live) are
        attached to with reads only — never reset, so they keep holding their
        pose — while the rest get the full bring-up (the mode switch reboots
        MyActuator motors, which is why it must never reach a holding joint).
        A holding gripper keeps its grasp: its end-stop calibration is
        restored from the values persisted by the last full bring-up instead
        of being re-measured (the calibration sweep would close then open
        the jaw). To force a fresh bring-up of a live robot, call
        :meth:`disable` first.

        With ``hold=True`` (the default) the arm finishes actively holding
        its measured pose: one :meth:`motion_control` command is issued at
        the measured positions (configured gains + gravity feedforward — the
        arm is already there, so nothing moves). Enabled therefore *means*
        holding, and the command history is seeded so the ``max_step_rad``
        safety check applies from the very first caller command.

        Pass ``hold=False`` to leave freshly brought-up joints enabled but
        limp (no torque until the first command) — for callers that manage
        control modes themselves via :meth:`set_control_mode` and would only
        tear the hold down again. Joints kept holding from a previous
        session still hold (their previous command stays active), and the
        command history is still seeded when any exist.

        On the gripperless SKU the gripper calibration and mode switch are
        skipped (there is no gripper motor).

        Raises:
            MotorError: If a holding motor is unreachable or in an unexpected
                control mode (reconnecting targets the impedance workflow —
                a session that died in another control mode needs
                :meth:`disable` then re-enable), or if a holding gripper has
                no valid persisted calibration (re-measuring would drop
                whatever it grips — empty the gripper, then :meth:`disable`
                and re-enable), or if any arm joint's encoder reading is
                implausible for a zero at its calibration end stop — the
                zero was never set (or is stale), and the robot must not be
                brought up on garbage joint-frame values; run
                ``axol motor.set-zero-pos --guided`` first.
        """
        # Gate bring-up on plausible encoder zeros BEFORE anything is
        # actuated (position reads work on disabled motors): detect which
        # end stop the either-stop joints were zeroed at, and refuse to
        # enable a robot whose zeros were never set.
        await self.resolve_joint_offsets()

        flags = dict(zip(self.motors.keys(), await self.get_holding()))
        held = [j for j, holding in flags.items() if holding]
        cold = [j for j, holding in flags.items() if not holding]

        await asyncio.gather(
            *[
                self.motors[j].attach(
                    ControlMode.POSITION_FORCE
                    if j == Joint.GRIPPER
                    else ControlMode.IMPEDANCE
                )
                for j in held
            ],
            *[self.motors[j].enable() for j in cold],
        )
        await asyncio.gather(
            *[self.motors[j].set_control_mode(ControlMode.IMPEDANCE) for j in cold]
        )

        # The mode switch reboots MyActuator motors, which re-derive their
        # multi-turn position from the single-turn absolute encoder (within
        # ±180° of zero) — a joint parked past 180° from its calibration stop
        # comes back reading 360° off, invalidating the wrap correction
        # detected by the resolve above.  Re-verify the freshly brought-up
        # fixed-stop joints against the post-reset frame.  (Either-stop
        # joints are Damiao, whose mode switch is a register write — no
        # reboot, no re-detection needed.)
        recheck = set(cold) & (set(ARM_JOINTS) - EITHER_STOP_JOINTS)
        if recheck:
            self._unverified_zeros |= recheck
            await self.resolve_joint_offsets(recheck)

        if self._has_gripper:
            if Joint.GRIPPER in held:
                await self._restore_gripper_calibration()
            else:
                await self._calibrate_gripper()
                await self.motors[Joint.GRIPPER].set_control_mode(
                    ControlMode.POSITION_FORCE
                )

        if held or hold:
            q = (await self.get_positions()).astype(float)
        if held:
            # Seed the max-step safety check from the measured pose so the
            # first motion_control cannot yank the joints that kept holding.
            self._last_q_commanded = q
        if hold:
            # Servo on: command the measured pose once so "enabled" means
            # "holding" — without this, freshly brought-up joints stay limp
            # until the caller's first motion_control.
            await self.motion_control(q)

    async def _restore_gripper_calibration(self) -> None:
        """Restore the gripper limits persisted by the last full bring-up.

        The no-motion alternative to :meth:`_calibrate_gripper`, used when
        the idempotent :meth:`enable` finds the gripper already live and
        holding: re-measuring the stops would sweep the jaw closed and open
        and drop anything held.

        Raises:
            MotorError: If no calibration was persisted, it disagrees with the
                configured ``close_direction``, or it no longer matches the
                encoder (motor re-zeroed or power-cycled).
        """
        side = "left" if self._is_left else "right"
        open_pos, close_pos = _load_gripper_calibration(self._is_left)
        close_direction = self._arm_config.gripper.close_direction
        if (close_pos - open_pos) * close_direction <= 0:
            raise MotorError(
                f"Persisted {side} gripper calibration (open {open_pos:.2f} rad, "
                f"closed {close_pos:.2f} rad) was measured with the opposite "
                f"close_direction to the configured {close_direction:+d} — "
                f"empty the gripper, then disable() and enable() to recalibrate"
            )
        lo = min(open_pos, close_pos)
        hi = max(open_pos, close_pos)
        current = await self.motors[Joint.GRIPPER].get_position()
        if not (lo - _GRIPPER_CALIB_MARGIN <= current <= hi + _GRIPPER_CALIB_MARGIN):
            raise MotorError(
                f"Persisted {side} gripper calibration does not match the "
                f"motor: current position {current:.2f} rad is outside the "
                f"calibrated range [{lo:.2f}, {hi:.2f}] rad (motor re-zeroed or "
                f"power-cycled?) — empty the gripper, then disable() and "
                f"enable() to recalibrate"
            )
        self._set_gripper_range(open_pos, close_pos)

    async def disable(self) -> None:
        """Disable all motors and engage brakes."""
        await asyncio.gather(*[m.disable() for m in self.motors.values()])

    async def clear_errors(self) -> None:
        """Clear latched error flags on all motors."""
        await asyncio.gather(*[m.clear_errors() for m in self.motors.values()])

    async def set_control_mode(self, mode: ControlMode) -> None:
        """Set the control mode on all motors.

        WARNING: MyActuator motors reboot to switch modes (torque off ~2 s)
        — never call this while the arm is holding a load; it will fall.

        Args:
            mode: Desired control mode.
        """
        await asyncio.gather(*[m.set_control_mode(mode) for m in self.motors.values()])
        # The reboot re-derives each MyActuator's multi-turn position from
        # its single-turn absolute encoder (within ±180° of zero), so any
        # ±360° wrap correction detected earlier may be stale.  Mark the
        # fixed-stop joints (all MyActuator) for re-verification; the next
        # joint-frame entry point resolves them.
        self._unverified_zeros |= set(ARM_JOINTS) - EITHER_STOP_JOINTS

    # ------------------------------------------------------------------ #
    # Getters                                                              #
    # ------------------------------------------------------------------ #

    async def get_positions(self) -> np.ndarray:
        """Return joint positions for every joint, fetched concurrently.

        Returns shape (8,) array in Joint enum order. Arm joints are in
        radians in the joint frame (0 = rest position); the gripper is
        normalized to [0, 1] (0.0 = closed, 1.0 = fully open), consistent
        with set_position_velocity and motion_control. On the gripperless
        SKU the gripper element is 0.0.
        """
        await self.resolve_joint_offsets()
        values = self._pad_gripper(
            list(
                await asyncio.gather(
                    *[self.motors[j].get_position() for j in self.motors]
                )
            )
        )
        gripper_i = self._gripper_i
        if self._has_gripper:
            values[gripper_i] = self._gripper_from_raw(values[gripper_i])
        arr = np.array(values, dtype=np.float32)
        return arr + self._joint_offsets.astype(np.float32)

    async def get_velocities(self) -> np.ndarray:
        """Return shaft velocity (rad/s) for every joint, fetched concurrently.

        Returns shape (8,) array in Joint enum order (gripper element 0.0 on
        the gripperless SKU).
        """
        values = await asyncio.gather(
            *[self.motors[j].get_velocity() for j in self.motors]
        )
        return np.array(self._pad_gripper(list(values)), dtype=np.float32)

    async def get_torques(self) -> np.ndarray:
        """Return torque estimate for every joint, fetched concurrently.

        Damiao: Nm. MyActuator: phase current in A.
        Returns shape (8,) array in Joint enum order (gripper element 0.0 on
        the gripperless SKU).
        """
        values = await asyncio.gather(*[m.get_torque() for m in self.motors.values()])
        return np.array(self._pad_gripper(list(values)), dtype=np.float32)

    async def get_temperatures(self) -> np.ndarray:
        """Return motor temperature (°C) for every joint, fetched concurrently.

        Returns shape (8,) array in Joint enum order (gripper element 0.0 on
        the gripperless SKU).
        """
        values = await asyncio.gather(
            *[m.get_temperature() for m in self.motors.values()]
        )
        return np.array(self._pad_gripper(list(values)), dtype=np.float32)

    async def get_voltages(self) -> np.ndarray:
        """Return bus voltage (V) for every joint, fetched concurrently.

        Returns shape (8,) array in Joint enum order (gripper element 0.0 on
        the gripperless SKU).
        """
        values = await asyncio.gather(*[m.get_voltage() for m in self.motors.values()])
        return np.array(self._pad_gripper(list(values)), dtype=np.float32)

    async def get_error_codes(self) -> list[MotorStatus]:
        """Return MotorStatus for every joint, fetched concurrently.

        Returns a list in Joint enum order. On the gripperless SKU the
        gripper entry is omitted (7 entries).
        """
        values = await asyncio.gather(
            *[self.motors[j].get_error_code() for j in self.motors]
        )
        return list(values)

    async def get_holding(self) -> list[bool]:
        """Return each motor's enabled-and-holding state, fetched concurrently.

        Read-only — safe on a robot of unknown state (pairs with
        :meth:`Axol.connect` for inspecting before acting). See
        :meth:`Motor.is_holding` for what "holding" means per motor family.
        Returns a list in Joint enum order. On the gripperless SKU the
        gripper entry is omitted (7 entries).
        """
        values = await asyncio.gather(
            *[self.motors[j].is_holding() for j in self.motors]
        )
        return list(values)

    async def get_gains(self) -> list[MotorGains]:
        """Return PID gains for every joint, fetched concurrently.

        Returns a list in Joint enum order. On the gripperless SKU the
        gripper entry is omitted (7 entries).
        """
        values = await asyncio.gather(
            *[self.motors[j].get_gains() for j in self.motors]
        )
        return list(values)

    # ------------------------------------------------------------------ #
    # Setters                                                              #
    # ------------------------------------------------------------------ #

    async def set_gains(self, gains: dict[Joint, MotorGains]) -> None:
        """Write PID gains to the specified motors.

        Changes are persisted to non-volatile memory.
        """
        await asyncio.gather(*[self.motors[j].set_gains(g) for j, g in gains.items()])

    async def set_zero_position(self, joints: list[Joint]) -> None:
        """Save the current shaft position as the encoder zero for the specified joints.

        The encoder zero is calibrated at one of the joint's mechanical end
        stops, not at the rest position; ``AxolArm`` applies a per-joint
        offset so the public API stays in joint frame (``0`` = rest).

        Args:
            joints: List of joints to zero.
        """
        await asyncio.gather(*[self.motors[j].set_zero_position() for j in joints])
        # A re-zeroed joint's encoder moved under us: an either-stop joint's
        # detected offset no longer describes it (Damiao applies the new zero
        # after a power cycle, but the old detection is stale either way),
        # and a fixed-stop joint's plausibility check must run again.
        joint_index = {j: i for i, j in enumerate(Joint)}
        for j in joints:
            if j in EITHER_STOP_JOINTS:
                self._joint_offsets[joint_index[j]] = math.nan
                self._unresolved_offsets.add(j)
            elif j != Joint.GRIPPER:
                self._unverified_zeros.add(j)

    async def set_acceleration(self, accelerations: dict[Joint, float]) -> None:
        """Set the acceleration ramp per joint. Deceleration matches acceleration.

        Args:
            accelerations: Mapping of joint → acceleration ramp (rad/s²).
                           Joints not in the dict are unchanged.
        """
        await asyncio.gather(
            *[self.motors[j].set_acceleration(a) for j, a in accelerations.items()]
        )

    async def set_position_velocity(
        self, positions: np.ndarray, max_speed: float
    ) -> None:
        """Move joints to absolute positions using each motor's built-in controller.

        Positions are clipped to the arm's joint limits before being sent.
        Arm joints are in joint frame (0 = rest position); the gripper value
        is normalized: 0.0 = closed, 1.0 = fully open.

        Args:
            positions: Shape (8,) array of target positions (rad) in Joint enum order,
                       except gripper which is [0, 1] (ignored on the
                       gripperless SKU).
            max_speed: Maximum speed for all joints (rad/s).
        """
        await self.resolve_joint_offsets()
        positions = positions.copy()
        gripper_i = self._gripper_i
        if self._has_gripper:
            positions[gripper_i] = self._gripper_to_raw(float(positions[gripper_i]))
        else:
            positions[gripper_i] = 0.0
        clipped = np.clip(positions, self._limits_lo, self._limits_hi)
        # Convert arm joints from joint frame to motor frame before sending.
        # Gripper offset is 0, so its raw motor value is unchanged.
        motor_targets = clipped - self._joint_offsets
        await asyncio.gather(
            *[
                self.motors[j].set_position_velocity(float(motor_targets[i]), max_speed)
                for i, j in enumerate(Joint)
                if j in self.motors
            ]
        )

    async def set_velocity(self, velocities: np.ndarray) -> None:
        """Command target velocities using each motor's built-in speed controller.

        Args:
            velocities: Shape (8,) array of target velocities (rad/s) in Joint
                        enum order (gripper element ignored on the gripperless
                        SKU).
        """
        await asyncio.gather(
            *[
                self.motors[j].set_velocity(float(velocities[i]))
                for i, j in enumerate(Joint)
                if j in self.motors
            ]
        )

    async def motion_control(self, q: np.ndarray) -> None:
        """Send control commands to all joints concurrently.

        The 7 arm joints use IMPEDANCE control: gains (kp, kd) and friction
        parameters come from ArmConfig; feedforward torque is computed as
        gravity + friction compensation.

        The gripper uses POSITION_FORCE control: it tracks the target position
        at up to ``ArmConfig.gripper.max_speed`` (rad/s) with torque capped
        at ``ArmConfig.gripper.torque_limit`` (Nm). On the gripperless SKU
        the gripper element is ignored.

        All positions are clipped to joint limits before being sent.

        Args:
            q: Shape (8,) array of desired positions in Joint enum order.
               Arm joints are in radians in the joint frame (0 = rest);
               gripper is normalized to [0, 1] (0.0 = closed, 1.0 = fully open).
        """
        await self.resolve_joint_offsets()
        q = q.copy()

        # Safety: reject commands with arm-joint deltas that exceed max_step_rad.
        # Deltas are frame-invariant (constant offset), so compute in joint frame.
        max_step = self._config.max_step_rad
        if self._last_q_commanded is not None and max_step < float("inf"):
            gripper_i = self._gripper_i
            arm_mask = np.ones(len(q), dtype=bool)
            arm_mask[gripper_i] = False
            deltas = np.abs(q[arm_mask] - self._last_q_commanded[arm_mask])
            worst_i = int(np.argmax(deltas))
            worst_delta = float(deltas[worst_i])
            if worst_delta > max_step:
                arm_joints = [j for j in Joint if j != Joint.GRIPPER]
                joint_name = (
                    arm_joints[worst_i].name
                    if worst_i < len(arm_joints)
                    else str(worst_i)
                )
                _logger.warning(
                    "motion_control: command dropped — joint %s delta %.3f rad exceeds "
                    "max_step_rad %.3f rad",
                    joint_name,
                    worst_delta,
                    max_step,
                )
                return

        gripper_i = self._gripper_i
        if self._has_gripper:
            q[gripper_i] = self._gripper_to_raw(float(q[gripper_i]))
        else:
            q[gripper_i] = 0.0
        clipped = np.clip(q, self._limits_lo, self._limits_hi)

        # Velocity feedforward via differentiation of commanded positions (rad/s),
        # and acceleration feedforward via a second pass for inertia FF (rad/s²).
        # Velocities/accelerations are frame-invariant under a constant offset,
        # so we differentiate the joint-frame ``clipped`` array directly.
        # Classic mode only: in realtime-core mode the trajectory the wire
        # carries is rendered by the core's own tracker at 240 Hz, and the
        # velocity/friction/inertia terms are computed there by applying
        # these same low-pass derivative chains to the executed tracker
        # position (this loop's 120 Hz differentiation of the pre-tracker
        # target would be out of phase with the executed motion).
        sink_mode = self._command_sink is not None
        if not sink_mode:
            velocities = self._vel_diff.differentiate(list(clipped))
            accelerations = self._accel_diff.differentiate(velocities)

        # Gravity feedforward (Nm) for the seven arm joints, computed from the
        # full URDF chain so child links contribute to each parent joint's load.
        # ``arm_q`` is in joint frame, which matches the URDF convention.
        arm_q = clipped[: len(ARM_JOINTS)].astype(np.float32)
        gravity, inertia = self._gravity_comp.gravity_and_inertia_arm(
            arm_q, is_left=self._is_left
        )

        # kd_host schedule: host-side damping is only stable — and only
        # needed — where the joint's reflected inertia is high. High inertia
        # means a slow natural mode (ωn = √(kp/J), ~2.3 Hz for a hanging
        # shoulder) that the motor's lagged internal velocity estimate can't
        # damp but a ~100 Hz host loop can. As the pose moves mass onto a
        # joint's axis, J collapses, ωn rises toward the host loop rate, and
        # the one-cycle-stale host torque arrives out of phase — measured as
        # sustained jitter on shoulder_1 with the arm raised to the side
        # (kd_host=15 rang at 0.57° RMS; kd_host=0 was clean, firmware kd
        # alone handles the fast mode fine). Scale each joint's kd_host by
        # J(q)/J_ref (J_ref = per-joint max over arm shapes, see __init__),
        # capped at 1 so the configured values are never exceeded. Constant
        # damping *ratio* would only need √(J/J_ref), but the binding
        # constraint at low J is phase-lag stability, not ζ — the linear
        # taper reaches ~0 at the measured-unstable raised pose (J ratio
        # 0.02) where √ would still deliver 14% — and the two rules differ
        # by <25% at the moderate poses where damping matters.
        host_scale = np.clip(inertia.astype(np.float64) / self._inertia_ref, 0.0, 1.0)

        # Inertia-FF schedule (see __init__): rotor term constant, link term
        # tracking J(q), normalized to 1 at the rest pose where j_eff was
        # tuned. Unclipped above 1 on purpose — shoulder_3's link inertia
        # peaks with the arm extended (up to ~1.6× its rest anchor), exactly
        # where its acceleration FF is needed most.
        j_scale = (self._j_rotor + inertia.astype(np.float64)) / self._j_ff_denom

        # Host-damping band-pass centres for this cycle. Tracked joints (see
        # __init__) follow their pose-dependent impedance mode by scaling the
        # hardware-anchored rest centre by √(J_rest/J(q)) — j_scale is exactly
        # (J_rotor + J(q))/(J_rotor + J_rest). Clamped: below ~12 rad/s the
        # band starts dragging intentional motion, above 50 rad/s (~8 Hz) the
        # loop's phase budget is spent and more centre only points the damper
        # at modes it would excite.
        damp_w0 = list(self._damp_w0)
        for i in range(len(ARM_JOINTS)):
            if self._damp_w0_tracked[i]:
                damp_w0[i] = float(
                    np.clip(DAMP_BP_W0 / math.sqrt(max(j_scale[i], 1e-6)), 12.0, 50.0)
                )

        # Host damping input: fast-differentiated commanded and measured
        # velocities, band-passed at each joint's resonance (damp_w0 above).
        # The measured side differentiates the positions cached from
        # impedance feedback frames against the frames' own CAN receive
        # timestamps (jitter-free — see ``Differentiator.differentiate``).
        # Falls back to a zero damping input until every cache is filled by
        # the first set_impedance replies.
        #
        # In realtime-core mode the damping *torque* is not computed here at
        # all: damping is a phase race, and this loop's ~120 Hz sample plus
        # the socket transport put the counter-torque ~14 ms behind the
        # velocity it acts on — enough to push the shoulder burst band past
        # 90° of loop phase, where the damper pumps the mode instead of
        # damping it (the rt-teleop shaking of 2026-08-27). The core runs
        # the identical filter chain (see rust/axol-rt/src/filter.rs,
        # golden-tested against this module) at 240 Hz on the latest feedback,
        # applying the result within one core tick; this side only *schedules*
        # it, shipping the pose-scaled gain and band-pass centre/q per command.
        if not sink_mode:
            v_des_fast = self._vel_fast_diff.differentiate(list(clipped))
            try:
                pos_list: list[float] = []
                ts_list: list[float] = []
                for j in Joint:
                    motor = self.motors.get(j)
                    if motor is not None:
                        pos_list.append(motor.position)
                        ts_list.append(motor.feedback_ts)
                    else:
                        pos_list.append(0.0)
                        ts_list.append(0.0)
                v_meas_fast = self._meas_vel_diff.differentiate(pos_list, ts_list)
            except MotorError:
                v_meas_fast = list(v_des_fast)
            v_damp = self._damp_bp.update(
                [d - m for d, m in zip(v_des_fast, v_meas_fast)], w0=damp_w0
            )

        # Convert arm joints to motor frame for the impedance command.  Gripper
        # offset is 0, so its raw motor value is unchanged.
        motor_targets = clipped - self._joint_offsets

        if sink_mode:
            # Production realtime-core mode: ship 9-float tuples to
            # the sink — which streams them to the Rust core that owns the
            # CAN bus. mode=1 (tracked): the core's own trapezoid tracker
            # renders the trajectory toward p_des at 240 Hz and computes the
            # velocity, friction, and inertia terms from its states — t_ff
            # here carries *gravity only* (the slow, pose-shaped term this
            # side owns). The damping coefficients and the pose-scaled
            # inertia gain are the schedule the core applies each tick
            # against its own fresh feedback (see the sink_mode comment
            # above). Slot 7 carries the gripper's POSITION_FORCE command
            # (motor-frame target, speed limit, torque limit); zeros on the
            # gripperless SKU (the core has no gripper configured and
            # ignores the slot).
            sink_cmds: list[tuple[float, ...]] = []
            for i, j in enumerate(ARM_JOINTS):
                gains = getattr(self._arm_config, j.value)
                sink_cmds.append(
                    (
                        float(motor_targets[i]),
                        1.0,
                        gains.kp,
                        gains.kd,
                        float(gravity[i]),
                        float(host_scale[i]) * gains.kd_host,
                        damp_w0[i],
                        self._damp_q[i],
                        gains.j_eff * float(j_scale[i]),
                    )
                )
            if self._has_gripper:
                sink_cmds.append(
                    (
                        float(motor_targets[gripper_i]),
                        self._arm_config.gripper.max_speed,
                        self._arm_config.gripper.torque_limit,
                    )
                    + (0.0,) * 6
                )
            else:
                sink_cmds.append((0.0,) * 9)
            self._command_sink(sink_cmds)
            self._last_q_commanded = clipped
            return

        arm_cmds: list[tuple[float, float, float, float, float]] = []
        for i, j in enumerate(ARM_JOINTS):
            gains = getattr(self._arm_config, j.value)
            f = gains.friction
            # Host damping is exactly the configured kd_host, pose-scheduled.
            # A kd beyond the firmware's range is clamped by the command
            # encoding (and warned about at construction) — it is *not*
            # rerouted into host damping: the delayed host torque is only
            # phase-safe on the slow shoulder modes, so silently converting
            # excess firmware damping into it could excite the very
            # oscillation the oversized kd was meant to kill.
            t_ff = (
                float(gravity[i])
                + compute_friction(velocities[i], f.fc, f.k, f.fv, f.fo)
                + gains.j_eff * float(j_scale[i]) * accelerations[i]
                + float(host_scale[i]) * gains.kd_host * v_damp[i]
            )
            arm_cmds.append(
                (float(motor_targets[i]), velocities[i], gains.kp, gains.kd, t_ff)
            )

        tasks = [
            self.motors[j].set_impedance(*arm_cmds[i]) for i, j in enumerate(ARM_JOINTS)
        ]
        if self._has_gripper:
            tasks.append(
                self.motors[Joint.GRIPPER].set_position_force(
                    float(motor_targets[gripper_i]),
                    self._arm_config.gripper.max_speed,
                    self._arm_config.gripper.torque_limit,
                )
            )
        await asyncio.gather(*tasks)
        self._last_q_commanded = clipped

    async def gravity_compensate(
        self,
        kd: float = 0.5,
        free_joints: set[Joint] | None = None,
        gripper_target: float | None = None,
    ) -> None:
        """Apply one cycle of gravity compensation.

        For each joint in ``free_joints``: send ``set_impedance(p_des=current,
        v_des=0, kp=0, kd=kd, t_ff=gravity)``. Gravity is supported by the
        feedforward torque, and ``kd`` provides a small velocity-damping term so
        motion does not feel twitchy. These joints are free to be moved by hand.

        For each arm joint *not* in ``free_joints``: send ``set_impedance``
        with the joint's configured ``kp``/``kd`` from :class:`ArmConfig` to
        hold it rigidly at the position it had at the *first* call (or at the
        moment the free-joint set last changed), with gravity feedforward.
        This lets the operator isolate one joint at a time — everything else
        stays put for testing. To re-snapshot the hold position (e.g. after
        repositioning the arm), call :meth:`reset_gravity_hold` between calls.

        The gripper is softly held at its current position regardless of
        ``free_joints`` (skipped on the gripperless SKU), unless
        ``gripper_target`` asks for a different opening.

        Requires :meth:`start_telemetry` to be active so cached positions are
        fresh.

        Args:
            kd: Velocity damping for *free* joints (Nm·s/rad). 0 lets the arm
                coast freely (may feel underdamped); 0.5 is a good starting
                point. Tune to taste.
            free_joints: Set of arm joints to gravity-compensate. ``None`` (the
                default) frees all 7 arm joints. Joints not in this set are
                held rigidly at their initial position. ``Joint.GRIPPER`` is
                ignored if present.
            gripper_target: Normalised gripper opening to drive to
                (0.0 = closed, 1.0 = fully open) at the configured torque
                limit, for flows that operate the gripper while the arm is
                hand-guided (``axol waypoints``). ``None`` (the default) holds
                wherever the gripper currently is, softly.
        """
        await self.resolve_joint_offsets()
        free_set: frozenset[Joint] = (
            frozenset(ARM_JOINTS) if free_joints is None else frozenset(free_joints)
        )

        positions = self.positions
        arm_q = positions[: len(ARM_JOINTS)].astype(np.float32)
        gravity = self._gravity_comp.gravity_arm(arm_q, is_left=self._is_left)

        # Snapshot held positions on first call or whenever the free-joint set
        # changes; otherwise keep the original setpoint so kp can produce a
        # real restoring torque.
        if self._gc_hold_q is None or self._gc_hold_free != free_set:
            self._gc_hold_q = arm_q.copy()
            self._gc_hold_free = free_set

        # ``arm_q`` and ``_gc_hold_q`` are in joint frame; convert to motor
        # frame before sending to the impedance controller.
        arm_offsets = self._joint_offsets[: len(ARM_JOINTS)]

        arm_tuples: list[tuple[float, float, float, float, float]] = []
        for i, j in enumerate(ARM_JOINTS):
            if j in free_set:
                p_des = float(arm_q[i] - arm_offsets[i])
                kp_cmd = 0.0
                kd_cmd = kd
            else:
                p_des = float(self._gc_hold_q[i] - arm_offsets[i])
                gains = getattr(self._arm_config, j.value)
                kp_cmd = gains.kp
                kd_cmd = gains.kd
            arm_tuples.append((p_des, 0.0, kp_cmd, kd_cmd, float(gravity[i])))
        # Hold the gripper softly so it does not drift open/closed — or drive
        # it to the requested opening, at the full configured torque so it can
        # actually grasp while the arm stays hand-guidable.
        gripper_cmd: tuple[float, float, float] | None = None
        if self._has_gripper:
            gripper_i = self._gripper_i
            if gripper_target is None:
                gripper_pos = float(positions[gripper_i])
                torque = 0.5
            else:
                gripper_pos = float(np.clip(gripper_target, 0.0, 1.0))
                torque = self._arm_config.gripper.torque_limit
            gripper_pos_raw = self._gripper_to_raw(gripper_pos)
            gripper_cmd = (
                gripper_pos_raw,
                self._arm_config.gripper.max_speed,
                torque,
            )

        if self._command_sink is not None:
            # Realtime-core mode: the same tuples stream through the core
            # (which owns the bus) instead of onto the wire from here. The
            # arm tuples' second field is 0.0 = passthrough mode — p_des
            # goes to the wire as-is with v_des = 0, no tracker and no
            # friction/inertia terms (a hand-guided limp arm wants gravity
            # feedforward only), and no damping coefficients (classic
            # gravity comp runs firmware gains only).
            sink_cmds = [t + (0.0,) * 4 for t in arm_tuples]
            sink_cmds.append(
                gripper_cmd + (0.0,) * 6 if gripper_cmd is not None else (0.0,) * 9
            )
            self._command_sink(sink_cmds)
            return

        tasks = [
            self.motors[j].set_impedance(*arm_tuples[i])
            for i, j in enumerate(ARM_JOINTS)
        ]
        if gripper_cmd is not None:
            tasks.append(self.motors[Joint.GRIPPER].set_position_force(*gripper_cmd))
        await asyncio.gather(*tasks)

    def reset_gravity_hold(self) -> None:
        """Forget the cached hold setpoint used by :meth:`gravity_compensate`.

        The next call to ``gravity_compensate`` will re-snapshot the held
        joints' positions from the current telemetry. Use this if you have
        manually repositioned the arm and want the held joints to lock in
        their new pose.
        """
        self._gc_hold_q = None
        self._gc_hold_free = None

    def reset_command_state(self) -> None:
        """Forget cached command history after an out-of-band move.

        When the arm is moved without going through :meth:`motion_control`
        (e.g. hand-guided while under :meth:`gravity_compensate`), the cached
        ``_last_q_commanded`` and the velocity/accel differentiators no longer
        reflect the real pose. Left stale, the next ``motion_control`` command
        would be measured against the old setpoint: the max-step safety check
        could reject the first waypoint, and the velocity/accel feedforward
        would spike on a phantom jump.

        Calling this clears that history so the next ``motion_control`` is
        treated like the first command after :meth:`connect` — the max-step
        check is skipped and the differentiators restart from zero. Pair it
        with :meth:`reset_gravity_hold` when handing control back after a
        manual reposition.
        """
        self._last_q_commanded = None
        n = len(list(Joint))
        self._vel_diff = Differentiator(n=n)
        self._accel_diff = Differentiator(n=n)
        self._vel_fast_diff = Differentiator(n=n, cutoff=VEL_CUTOFF_FREQ)
        self._meas_vel_diff = Differentiator(n=n, cutoff=VEL_CUTOFF_FREQ)
        self._damp_bp = BandPass(n=n, w0=self._damp_w0, q=self._damp_q)

    def torque_residuals(self) -> np.ndarray:
        """Measured minus model-gravity torque per arm joint, shape (7,).

        Both terms are evaluated at the cached measured positions, so during
        slow, quasi-static moves the residual background is just friction
        plus tracking effort; unexpected contact (a gripper still hooked on
        the scene, an operator grabbing the arm) shows up as a residual well
        above that background. Units are the motor's reported torque units
        (Nm on Damiao). Entries are in :data:`ARM_JOINTS` order; the gripper
        is excluded (position-force mode — its torque tracks grasp effort,
        not contact). Requires fresh feedback (telemetry or streaming
        commands, whose replies refresh the position/torque cache).
        """
        positions = self.positions
        arm_q = positions[: len(ARM_JOINTS)].astype(np.float32)
        gravity = self._gravity_comp.gravity_arm(arm_q, is_left=self._is_left)
        tau = self.torques[: len(ARM_JOINTS)].astype(np.float32)
        return tau - gravity


class Axol(RobotBase):
    """Dual-arm Axol robot interface.

    Opens one CAN bus per arm and constructs all 16 motor drivers on entry
    (14 on the gripperless SKU, ``config.has_gripper = False``).
    Use as an async context manager to ensure the buses are cleanly shut down.

        async with Axol() as axol:
            await axol.enable()
            await axol.start_telemetry(500)  # 500 Hz

            # control loop — instant, no await
            pos_l, pos_r = axol.left.positions, axol.right.positions

            await axol.motion_control(left=np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]))

    Startup code never needs to know the robot's state: ``enable()`` is
    idempotent per motor — joints still holding from a previous session (it
    died or disconnected while live) are kept holding with reads only, and
    everything else gets the full bring-up, finishing with the arms actively
    holding their measured pose. :meth:`connect` opens the buses without
    touching motor state, for inspecting a robot of unknown state first
    (:meth:`get_holding`, :meth:`get_positions`, ...):

        axol = Axol()
        await axol.connect()      # open buses; inspect freely, nothing actuated
        await axol.enable()       # holding joints kept holding; cold joints brought up
        pos_l, pos_r = await axol.get_positions()
        # ... resume motion_control from the measured pose ...
        await axol.disconnect()   # exit, leaving torque as-is

    Pass ``enable(hold=False)`` to leave freshly brought-up joints limp, for
    flows that manage control modes themselves (:meth:`set_control_mode` —
    note it reboots MyActuator motors, so never switch modes on a loaded
    arm). A process that must not actuate anything can ``connect()``, check
    :meth:`get_holding`, and only proceed to ``enable()`` when every joint
    is already holding.

    Attributes:
        left:  AxolArm for the left arm.
        right: AxolArm for the right arm.

    Args:
        config:        Dual-arm gains config. Left and right arm gains are specified
                       independently; the right arm defaults to the left with gravity
                       mirrored for shoulder_2 and elbow.
        left_channel:  SocketCAN interface name for the left arm.
        right_channel: SocketCAN interface name for the right arm.
    """

    def __init__(
        self,
        config: AxolConfig = AxolConfig(),
        left_channel: str | None = CAN_LEFT,
        right_channel: str | None = CAN_RIGHT,
    ) -> None:
        """Construct the dual-arm interface.

        CAN buses and motors are created but not started; call ``enable()``
        or use the class as an async context manager to bring up hardware.

        Args:
            config:        Per-joint gains, friction parameters, and gripper config.
            left_channel:  SocketCAN interface name for the left arm, or ``None`` to omit it.
            right_channel: SocketCAN interface name for the right arm, or ``None`` to omit it.
        """
        if left_channel is None and right_channel is None:
            raise ValueError(
                "At least one of left_channel or right_channel must be specified."
            )

        # Bake stiffness into the per-joint gains exactly once, here at the
        # single robot-construction boundary. ``resolved()`` is idempotent,
        # so this is safe even if the caller already resolved the config.
        config = config.resolved()

        self._gravity_comp = GravityCompensator(config)

        if left_channel is not None:
            self._left_bus = CanBus(left_channel)
            self.left = AxolArm(
                self._left_bus, config, self._gravity_comp, is_left=True
            )
        else:
            self.left = None

        if right_channel is not None:
            self._right_bus = CanBus(right_channel)
            self.right = AxolArm(
                self._right_bus, config, self._gravity_comp, is_left=False
            )
        else:
            self.right = None

    # ------------------------------------------------------------------ #
    # Polling                                                              #
    # ------------------------------------------------------------------ #

    async def start_telemetry(self, hz: float, *, torque: bool = False) -> None:
        """Start background telemetry polling on both arms at the given frequency.

        Args:
            hz:     Poll frequency in Hz.
            torque: If True, also fetch and cache torque each cycle.
        """
        tasks = []
        if self.left is not None:
            tasks.append(self.left.start_telemetry(hz, torque=torque))
        if self.right is not None:
            tasks.append(self.right.start_telemetry(hz, torque=torque))
        await asyncio.gather(*tasks)

    async def stop_telemetry(self) -> None:
        """Stop the background telemetry polling loop on both arms."""
        tasks = []
        if self.left is not None:
            tasks.append(self.left.stop_telemetry())
        if self.right is not None:
            tasks.append(self.right.stop_telemetry())
        await asyncio.gather(*tasks)

    async def wait_for_telemetry(self, timeout: float = 5.0) -> None:
        """Block until every motor on both arms has reported at least one position.

        Call after :meth:`start_telemetry` and before the first read of the
        cached ``positions``.

        Args:
            timeout: Maximum time to wait (s) before raising MotorError.
        """
        tasks = []
        if self.left is not None:
            tasks.append(self.left.wait_for_telemetry(timeout))
        if self.right is not None:
            tasks.append(self.right.wait_for_telemetry(timeout))
        await asyncio.gather(*tasks)

    # ------------------------------------------------------------------ #
    # Arm-wide commands                                                    #
    # ------------------------------------------------------------------ #

    async def connect(self) -> None:
        """Open the CAN buses without touching motor state.

        Purely the transport step: after this, every read API works —
        :meth:`get_holding`, :meth:`get_positions`, :meth:`get_error_codes`,
        temperatures, … — so a process that starts with no knowledge of the
        robot's state can inspect it before deciding to act. Nothing is
        actuated and no frame that could affect torque is sent.

        The typical startup is ``connect()``, optional inspection, then
        :meth:`enable` — which is idempotent and never drops joints that are
        already holding. Calling ``connect()`` first is optional:
        :meth:`enable` opens the buses itself.
        """
        bus_tasks = []
        if self.left is not None:
            bus_tasks.append(self._left_bus.start())
        if self.right is not None:
            bus_tasks.append(self._right_bus.start())
        await asyncio.gather(*bus_tasks)

    async def enable(self, hold: bool = True) -> None:
        """Start CAN buses and bring every motor up, never dropping held joints.

        Idempotent per motor: joints still enabled and holding from a
        previous session (it died or disconnected while live) are attached to
        with reads only and keep holding their pose — including a gripper
        keeping its grasp — while cold motors get the full bring-up. Startup
        code therefore doesn't need to know the robot's state; ``enable()``
        converges holding, torque-off, and mixed robots alike. To force a
        fresh bring-up of a live robot, call :meth:`disable` first.

        With ``hold=True`` (the default) the robot finishes actively holding
        its measured pose ("enabled" means holding). Pass ``hold=False`` to
        leave freshly brought-up joints enabled but limp, for flows that
        manage control modes themselves. See :meth:`AxolArm.enable` for
        details and failure modes.
        """
        await self.connect()

        motor_tasks = []
        if self.left is not None:
            motor_tasks.append(self.left.enable(hold=hold))
        if self.right is not None:
            motor_tasks.append(self.right.enable(hold=hold))
        await asyncio.gather(*motor_tasks)

    async def disconnect(self) -> None:
        """Close the CAN buses, leaving motor torque exactly as it is.

        The counterpart to :meth:`connect` — ends this process's session
        without torquing off, so holding arms keep holding their pose and a
        later process can reconnect and :meth:`enable` again. Telemetry is
        stopped first. Use :meth:`disable` instead to torque off.
        """
        tasks = []
        if self.left is not None:
            tasks.append(self.left.stop_telemetry())
        if self.right is not None:
            tasks.append(self.right.stop_telemetry())
        try:
            await asyncio.gather(*tasks)
        finally:
            close_tasks = []
            if self.left is not None:
                close_tasks.append(self._left_bus.close())
            if self.right is not None:
                close_tasks.append(self._right_bus.close())
            await asyncio.gather(*close_tasks)

    async def disable(self) -> None:
        """Disable all motors and close CAN buses."""
        tasks = []
        if self.left is not None:
            tasks.extend([self.left.stop_telemetry(), self.left.disable()])
        if self.right is not None:
            tasks.extend([self.right.stop_telemetry(), self.right.disable()])
        try:
            await asyncio.gather(*tasks)
        except Exception:
            pass
        finally:
            close_tasks = []
            if self.left is not None:
                close_tasks.append(self._left_bus.close())
            if self.right is not None:
                close_tasks.append(self._right_bus.close())
            await asyncio.gather(*close_tasks)

    async def clear_errors(self) -> None:
        """Clear latched error flags on both arms."""
        tasks = []
        if self.left is not None:
            tasks.append(self.left.clear_errors())
        if self.right is not None:
            tasks.append(self.right.clear_errors())
        await asyncio.gather(*tasks)

    async def set_control_mode(self, mode: ControlMode) -> None:
        """Set the control mode on all motors on both arms.

        WARNING: MyActuator motors reboot to switch modes (torque off ~2 s)
        — never call this while the arms are holding a load; they will fall.
        Bring the robot to rest first (or use ``enable(hold=False)`` in
        flows that manage modes themselves).

        Args:
            mode: Desired control mode.
        """
        tasks = []
        if self.left is not None:
            tasks.append(self.left.set_control_mode(mode))
        if self.right is not None:
            tasks.append(self.right.set_control_mode(mode))
        await asyncio.gather(*tasks)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    async def _gather_pair(left_coro, right_coro) -> tuple:
        """Run up to two coroutines concurrently; pass ``None`` to skip an arm."""
        coros = [c for c in (left_coro, right_coro) if c is not None]
        results = list(await asyncio.gather(*coros))
        left = results.pop(0) if left_coro is not None else None
        right = results.pop(0) if right_coro is not None else None
        return left, right

    # ------------------------------------------------------------------ #
    # Getters                                                              #
    # ------------------------------------------------------------------ #

    async def get_positions(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return joint positions (rad) for both arms as (left, right).

        Each array is shape (8,) in Joint enum order, or ``None`` if that arm is absent.
        """
        return await self._gather_pair(
            self.left.get_positions() if self.left is not None else None,
            self.right.get_positions() if self.right is not None else None,
        )

    async def get_velocities(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return shaft velocity (rad/s) for both arms as (left, right).

        Each array is shape (8,) in Joint enum order, or ``None`` if that arm is absent.
        """
        return await self._gather_pair(
            self.left.get_velocities() if self.left is not None else None,
            self.right.get_velocities() if self.right is not None else None,
        )

    async def get_torques(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return torque estimates for both arms as (left, right).

        Each array is shape (8,) in Joint enum order, or ``None`` if that arm is absent.
        """
        return await self._gather_pair(
            self.left.get_torques() if self.left is not None else None,
            self.right.get_torques() if self.right is not None else None,
        )

    async def get_temperatures(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return motor temperatures (°C) for both arms as (left, right).

        Each array is shape (8,) in Joint enum order, or ``None`` if that arm is absent.
        """
        return await self._gather_pair(
            self.left.get_temperatures() if self.left is not None else None,
            self.right.get_temperatures() if self.right is not None else None,
        )

    async def get_voltages(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return bus voltages (V) for both arms as (left, right).

        Each array is shape (8,) in Joint enum order, or ``None`` if that arm is absent.
        """
        return await self._gather_pair(
            self.left.get_voltages() if self.left is not None else None,
            self.right.get_voltages() if self.right is not None else None,
        )

    async def get_error_codes(
        self,
    ) -> tuple[list[MotorStatus] | None, list[MotorStatus] | None]:
        """Return MotorStatus for both arms as (left, right).

        Each list is in Joint enum order, or ``None`` if that arm is absent.
        """
        return await self._gather_pair(
            self.left.get_error_codes() if self.left is not None else None,
            self.right.get_error_codes() if self.right is not None else None,
        )

    async def get_holding(self) -> tuple[list[bool] | None, list[bool] | None]:
        """Return each motor's enabled-and-holding state as (left, right).

        Read-only — safe on a robot of unknown state, e.g. right after
        :meth:`connect`, to inspect before deciding to act. Each list is in
        Joint enum order (gripper entry omitted on the gripperless SKU), or
        ``None`` if that arm is absent.
        """
        return await self._gather_pair(
            self.left.get_holding() if self.left is not None else None,
            self.right.get_holding() if self.right is not None else None,
        )

    async def get_gains(
        self,
    ) -> tuple[list[MotorGains] | None, list[MotorGains] | None]:
        """Return PID gains for both arms as (left, right).

        Each list is in Joint enum order, or ``None`` if that arm is absent.
        """
        return await self._gather_pair(
            self.left.get_gains() if self.left is not None else None,
            self.right.get_gains() if self.right is not None else None,
        )

    # ------------------------------------------------------------------ #
    # Setters                                                              #
    # ------------------------------------------------------------------ #

    async def set_gains(
        self,
        left: dict[Joint, MotorGains] = {},
        right: dict[Joint, MotorGains] = {},
    ) -> None:
        """Write PID gains to the specified joints on both arms."""
        tasks = []
        if left and self.left is not None:
            tasks.append(self.left.set_gains(left))
        if right and self.right is not None:
            tasks.append(self.right.set_gains(right))
        if tasks:
            await asyncio.gather(*tasks)

    async def set_zero_position(
        self,
        left: list[Joint] | None = None,
        right: list[Joint] | None = None,
    ) -> None:
        """Save the current shaft position as the encoder zero for the specified joints.

        The encoder zero is calibrated at each joint's mechanical end stop, not
        at the rest position; per-joint offsets keep the public API in joint
        frame (``0`` = rest).

        Args:
            left:  Joints on the left arm to zero. ``None`` skips the arm.
            right: Joints on the right arm to zero. ``None`` skips the arm.
        """
        tasks = []
        if left is not None and self.left is not None:
            tasks.append(self.left.set_zero_position(left))
        if right is not None and self.right is not None:
            tasks.append(self.right.set_zero_position(right))
        if tasks:
            await asyncio.gather(*tasks)

    async def set_acceleration(
        self,
        left: dict[Joint, float] = {},
        right: dict[Joint, float] = {},
    ) -> None:
        """Set per-joint acceleration ramps (rad/s²) on both arms.

        Args:
            left:  Joint → acceleration (rad/s²) for the left arm. ``None`` skips.
            right: Same for the right arm.
        """
        tasks = []
        if left and self.left is not None:
            tasks.append(self.left.set_acceleration(left))
        if right and self.right is not None:
            tasks.append(self.right.set_acceleration(right))
        if tasks:
            await asyncio.gather(*tasks)

    async def set_positions_velocity(
        self,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
        max_speed: float = 0.0,
    ) -> None:
        """Command joint positions (rad) via the motor's built-in position controller.

        Args:
            left:      Shape (8,) array of target positions (rad) in Joint enum order.
                       ``None`` skips the arm.
            right:     Same for the right arm.
            max_speed: Maximum speed (rad/s). 0.0 uses the motor's default.
        """
        tasks = []
        if left is not None and self.left is not None:
            tasks.append(self.left.set_position_velocity(left, max_speed))
        if right is not None and self.right is not None:
            tasks.append(self.right.set_position_velocity(right, max_speed))
        if tasks:
            await asyncio.gather(*tasks)

    async def set_velocity(
        self,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
    ) -> None:
        """Command target velocities (rad/s) on both arms concurrently.

        Args:
            left:  Shape (8,) array of target velocities (rad/s). ``None`` skips the arm.
            right: Same for the right arm.
        """
        tasks = []
        if left is not None and self.left is not None:
            tasks.append(self.left.set_velocity(left))
        if right is not None and self.right is not None:
            tasks.append(self.right.set_velocity(right))
        if tasks:
            await asyncio.gather(*tasks)

    async def motion_control(
        self,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
    ) -> None:
        """Send control commands to both arms concurrently.

        Arm joints use IMPEDANCE control; the gripper uses POSITION_FORCE control.
        See ``AxolArm.motion_control`` for full details.

        Args:
            left:  Shape (8,) array of target positions for the left arm
                   (arm joints in rad, gripper in [0, 1]).  ``None`` skips.
            right: Same for the right arm.
        """
        tasks = []
        if left is not None and self.left is not None:
            tasks.append(self.left.motion_control(left))
        if right is not None and self.right is not None:
            tasks.append(self.right.motion_control(right))
        if tasks:
            await asyncio.gather(*tasks)

    async def gravity_compensate(
        self,
        kd: float = 0.5,
        free_joints: set[Joint] | None = None,
        gripper_targets: tuple[float | None, float | None] | None = None,
    ) -> None:
        """Put both arms into gravity-compensation mode for one cycle.

        Joints in ``free_joints`` are sent ``set_impedance`` with ``kp=0``,
        ``kd=kd``, and a feedforward torque equal to the model-predicted
        gravity (free to move by hand). Joints *not* in ``free_joints`` are
        held rigidly at their current position using their configured
        ``ArmConfig`` gains, with gravity feedforward. ``free_joints=None``
        frees all 7 arm joints on each side. The grippers are softly held at
        their current positions unless ``gripper_targets`` asks otherwise.

        Telemetry must be active (positions are read from the cache) — call
        :meth:`start_telemetry` before driving this in a loop.

        Args:
            kd: Velocity damping coefficient for *free* joints (Nm·s/rad).
                Tune to taste; ``0.5`` is a reasonable starting point.
            free_joints: Set of arm joints to gravity-compensate. ``None`` (the
                default) frees every arm joint. Joints not in this set are
                held in place. The same filter is applied to both arms.
            gripper_targets: ``(left, right)`` normalised openings (0.0 =
                closed, 1.0 = open) to drive each gripper to instead of
                holding it where it is; ``None`` for either side holds that
                one. Lets a flow grasp with one arm while the other stays
                free (``axol waypoints``).
        """
        left_target, right_target = gripper_targets or (None, None)
        tasks = []
        if self.left is not None:
            tasks.append(self.left.gravity_compensate(kd, free_joints, left_target))
        if self.right is not None:
            tasks.append(self.right.gravity_compensate(kd, free_joints, right_target))
        if tasks:
            await asyncio.gather(*tasks)

    def reset_gravity_hold(self) -> None:
        """Re-snapshot the held setpoint on both arms' :meth:`gravity_compensate`."""
        if self.left is not None:
            self.left.reset_gravity_hold()
        if self.right is not None:
            self.right.reset_gravity_hold()

    def reset_command_state(self) -> None:
        """Clear cached command history on both arms after an out-of-band move.

        See :meth:`AxolArm.reset_command_state`. Call this after hand-guiding
        the arms (e.g. under :meth:`gravity_compensate`) and before resuming
        :meth:`motion_control`, so the return-to-pose command is not rejected
        by the max-step safety check.
        """
        if self.left is not None:
            self.left.reset_command_state()
        if self.right is not None:
            self.right.reset_command_state()

    def torque_residuals(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Per-arm measured-minus-gravity torques, ``(left, right)``.

        Each present arm contributes a shape ``(7,)`` array in
        :data:`ARM_JOINTS` order (``None`` for an absent arm). See
        :meth:`AxolArm.torque_residuals` for semantics; reads only the
        telemetry cache, so it costs no CAN traffic.
        """
        left = self.left.torque_residuals() if self.left is not None else None
        right = self.right.torque_residuals() if self.right is not None else None
        return left, right
