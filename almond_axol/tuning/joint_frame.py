"""Joint-frame access to raw motors for the tuning commands.

The tuners drive :class:`~almond_axol.motor.Motor` objects directly (no
:class:`AxolArm`), but all of their math — joint limits from ``arm_limits``,
sine centers, step headroom, the URDF gravity model — is in the **joint
frame** (0 = rest position). Motor encoders are zeroed against a mechanical
end stop, so a raw motor reading of 0 is an end stop, not rest::

    joint_angle (rad) = motor_angle (rad) + offset

:class:`JointFrameMotor` proxies a motor and applies that conversion on every
position-carrying call, so the tuners never see (or send) a raw motor angle.
Velocities, torques, and gains are frame-invariant and pass through.

Offsets are resolved exactly like :class:`AxolArm` does: fixed at
``closer_end_stop()`` for most joints, and detected from the current encoder
reading for the either-stop joints (wrist_2 / wrist_3, which may be zeroed
at either of their two stops — see ``end_stop_offset_from_position``).
"""

from __future__ import annotations

import math

from ..motor import ControlMode, Joint, Motor
from ..motor.damiao import DamiaoMotor
from ..motor.myactuator import (
    _MA_KD_MAX,
    _MA_KP_MAX,
    _MA_V_MAX,
    MyActuatorMotor,
)
from ..robot.axol import (
    EITHER_STOP_JOINTS,
    closer_end_stop,
    end_stop_offset_from_position,
    fixed_stop_wrap_correction,
)


class JointFrameMotor:
    """A :class:`Motor` proxy that speaks the joint frame (0 = rest).

    Construct via :func:`joint_frame_motors`, which resolves the per-joint
    motor→joint offset. Only the calls the tuners use are exposed; add
    passthroughs as needed.

    **Boot wrap.** A fixed-stop joint's multi-turn reading can come back
    exactly ±360° off after any MyActuator 0x76 reset — which every tuner
    issues through ``set_control_mode`` — because the motor re-derives it
    within ±180° of the single-turn zero, and the right elbow at rest sits
    34° from that boundary. Commanding the joint frame through a fixed
    offset against a wrapped reading sends the motor a full turn: right
    elbow, 2026-09-22, into its hard stop at 40 Nm until its stall
    protection tripped. Every :meth:`get_position` therefore re-derives the
    wrap with :func:`fixed_stop_wrap_correction` (the production bring-up's
    check) and folds it into :attr:`frame_offset`, which every command uses.
    Read before you command — :func:`~almond_axol.cli.tune.friction._ramp_verified`
    does — and a wrapped reading is corrected instead of chased.
    """

    def __init__(
        self, motor: Motor, offset: float, is_left: bool | None = None
    ) -> None:
        self.motor = motor
        self.offset = offset
        self._is_left = is_left
        #: ±2π correction the last read said the motor frame needs (0 if none).
        self.wrap = 0.0

    @property
    def joint(self) -> Joint:
        return self.motor.joint

    @property
    def frame_offset(self) -> float:
        """motor→joint offset including the current boot-wrap correction."""
        return self.offset + self.wrap

    def _refresh_wrap(self, motor_pos: float) -> None:
        if (
            self._is_left is None
            or self.joint in EITHER_STOP_JOINTS
            or self.joint == Joint.GRIPPER
        ):
            return
        self.wrap = fixed_stop_wrap_correction(self.joint, self._is_left, motor_pos)

    @property
    def position(self) -> float:
        """Latest cached position (rad, joint frame)."""
        return self.motor.position + self.frame_offset

    @property
    def torque(self) -> float:
        """Latest cached torque estimate (Nm) — frame-invariant."""
        return self.motor.torque

    @property
    def feedback_ts(self) -> float:
        """CAN receive timestamp (s) of the last cached feedback frame."""
        return self.motor.feedback_ts

    async def get_position(self) -> float:
        """Current position (rad, joint frame), re-deriving the boot wrap.

        Raises ``MotorError`` (from :func:`fixed_stop_wrap_correction`) when
        the reading fits no plausible band — the zero is unset or stale.
        """
        motor_pos = await self.motor.get_position()
        self._refresh_wrap(motor_pos)
        return motor_pos + self.frame_offset

    async def set_impedance(
        self, p_des: float, v_des: float, kp: float, kd: float, t_ff: float
    ) -> None:
        """Impedance command with ``p_des`` in the joint frame."""
        await self.motor.set_impedance(p_des - self.frame_offset, v_des, kp, kd, t_ff)

    async def set_position_velocity(self, position: float, max_speed: float) -> None:
        """Position-velocity command with ``position`` in the joint frame."""
        await self.motor.set_position_velocity(position - self.frame_offset, max_speed)

    async def set_control_mode(self, mode: ControlMode) -> None:
        await self.motor.set_control_mode(mode)

    async def disable(self) -> None:
        await self.motor.disable()

    async def run_experiment(
        self,
        *,
        kp: float,
        kd: float,
        rate_hz: float,
        samples: list[tuple[float, ...]],
        differentiate: bool,
        feedforward: tuple[float, float, float, float, float, float, float, float],
    ) -> list[dict]:
        """Execute timed impedance samples entirely in the Rust CAN core."""
        driver = self.motor._driver
        if isinstance(driver, MyActuatorMotor):
            vendor = 0
            ranges = (
                driver._p_max,
                _MA_V_MAX,
                _MA_KP_MAX,
                _MA_KD_MAX,
                driver._t_max,
            )
        elif isinstance(driver, DamiaoMotor):
            vendor = 1
            ranges = (
                driver._p_max,
                driver._v_max,
                500.0,
                5.0,
                driver._t_max,
            )
        else:  # pragma: no cover - the arm has only these two vendors
            raise TypeError(f"unsupported tuning motor {type(driver).__name__}")
        return await driver._bus.run_experiment(
            vendor=vendor,
            motor_id=driver._motor_id,
            differentiate=differentiate,
            rate_hz=rate_hz,
            offset=self.frame_offset,
            kp=kp,
            kd=kd,
            ranges=ranges,
            feedforward=feedforward,
            samples=samples,
        )


async def joint_frame_motors(
    motors: dict[Joint, Motor], is_left: bool
) -> dict[Joint, JointFrameMotor]:
    """Resolve each motor's joint-frame offset and wrap it.

    Motors must already be enabled: the either-stop joints (wrist_2 /
    wrist_3) are resolved from a live encoder reading, which raises
    ``MotorError`` when such a joint is parked at its calibration end stop
    (the one ambiguous position) — move it away and retry.
    """
    wrapped: dict[Joint, JointFrameMotor] = {}
    for j, m in motors.items():
        if j in EITHER_STOP_JOINTS:
            pos = await m.get_position()
            offset = end_stop_offset_from_position(j, pos)
            print(
                f"  {j.value}: zeroed at the {math.degrees(offset):+.0f}° end stop "
                f"(motor {pos:+.3f} rad → joint offset {offset:+.3f} rad)"
            )
        else:
            offset = closer_end_stop(j, is_left)[0]
        jm = JointFrameMotor(m, offset, is_left)
        # Read once now: derives the boot wrap (and refuses an unset zero)
        # before any tuner commands the joint.
        await jm.get_position()
        wrapped[j] = jm
    return wrapped
