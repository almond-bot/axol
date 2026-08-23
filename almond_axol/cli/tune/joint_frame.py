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

from ...motor import ControlMode, Joint, Motor
from ...robot.axol import (
    EITHER_STOP_JOINTS,
    closer_end_stop,
    end_stop_offset_from_position,
)


class JointFrameMotor:
    """A :class:`Motor` proxy that speaks the joint frame (0 = rest).

    Construct via :func:`joint_frame_motors`, which resolves the per-joint
    motor→joint offset. Only the calls the tuners use are exposed; add
    passthroughs as needed.
    """

    def __init__(self, motor: Motor, offset: float) -> None:
        self.motor = motor
        self.offset = offset

    @property
    def joint(self) -> Joint:
        return self.motor.joint

    @property
    def position(self) -> float:
        """Latest cached position (rad, joint frame)."""
        return self.motor.position + self.offset

    @property
    def torque(self) -> float:
        """Latest cached torque estimate (Nm) — frame-invariant."""
        return self.motor.torque

    @property
    def feedback_ts(self) -> float:
        """CAN receive timestamp (s) of the last cached feedback frame."""
        return self.motor.feedback_ts

    async def get_position(self) -> float:
        """Current position (rad, joint frame)."""
        return await self.motor.get_position() + self.offset

    async def set_impedance(
        self, p_des: float, v_des: float, kp: float, kd: float, t_ff: float
    ) -> None:
        """Impedance command with ``p_des`` in the joint frame."""
        await self.motor.set_impedance(p_des - self.offset, v_des, kp, kd, t_ff)

    async def set_position_velocity(self, position: float, max_speed: float) -> None:
        """Position-velocity command with ``position`` in the joint frame."""
        await self.motor.set_position_velocity(position - self.offset, max_speed)

    async def set_control_mode(self, mode: ControlMode) -> None:
        await self.motor.set_control_mode(mode)

    async def disable(self) -> None:
        await self.motor.disable()


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
        wrapped[j] = JointFrameMotor(m, offset)
    return wrapped
