"""``RtAxol`` — the Axol robot driven through the Rust realtime core.

Presents the same surface :class:`~almond_axol.teleop.VRTeleop` uses
(``enable`` / ``disable`` / ``get_positions`` / ``motion_control``), but the
CAN buses are owned by the ``axol-rt`` subprocess:

- ``enable()`` runs the split bring-up: the core resets the motors (prep),
  then Python resolves joint offsets and MyActuator decode ranges against the
  post-reset state over its own (passive) bus sockets, then the core enables
  and holds. SocketCAN broadcasts every frame to every open socket, so
  Python's ``Motor`` feedback caches keep filling from the core's own 240 Hz
  MIT stream — measured positions, velocities, and torques stay available to
  ``motion_control``'s host-damping path with no extra traffic.
- ``motion_control()`` runs the full production computation in
  ``AxolArm.motion_control`` (gravity, inertia, friction, host damping), but
  a command sink ships the resulting MIT tuples to the core instead of
  sending CAN from Python.

The guarded-return extras (``torque_residuals`` / ``gravity_compensate`` /
``reset_command_state``) are deliberately not exposed: those paths send CAN
directly, so in rt mode resets play through the plain ``motion_control``
path, exactly as they do against ``Sim``. The gripper is not driven yet.
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from ..constants import ARM_JOINTS
from ..motor.motor import _JOINT_CONFIG
from ..robot.axol import Axol, AxolArm
from .link import RtLink

_logger = logging.getLogger(__name__)

_N_ARM = len(ARM_JOINTS)


class RtAxol:
    """Axol with the control loop in the Rust realtime core."""

    def __init__(
        self,
        robot: Axol,
        loop_hz: float = 240.0,
        watchdog_ms: float = 150.0,
        abort_deg: float = 25.0,
    ) -> None:
        self._robot = robot
        self._loop_hz = loop_hz
        self._watchdog_ms = watchdog_ms
        self._abort_deg = abort_deg
        self._link = RtLink()
        self._seq = 0
        # Gripper positions captured before the core starts streaming; the
        # grippers stay disabled in rt mode, so these stay exact.
        self._gripper_norm: dict[int, float] = {}

    @property
    def left(self) -> AxolArm | None:
        return self._robot.left

    @property
    def right(self) -> AxolArm | None:
        return self._robot.right

    def _arms(self) -> list[tuple[int, AxolArm]]:
        out = []
        if self._robot.left is not None:
            out.append((0, self._robot.left))
        if self._robot.right is not None:
            out.append((1, self._robot.right))
        return out

    def _config_text(self) -> str:
        max_step = self._arms()[0][1]._config.max_step_rad
        lines = [
            f"loop_hz {self._loop_hz}",
            f"watchdog_ms {self._watchdog_ms}",
            # Corruption defense on the core side; the Python max-step gate
            # in motion_control is the real per-command limit.
            f"max_step_rad {max_step}",
            f"abort_deg {self._abort_deg}",
        ]
        for side, arm in self._arms():
            # The bus channel lives on the CanBus (same package internals).
            bus = self._robot._left_bus if side == 0 else self._robot._right_bus
            iface = bus._channel
            for j in ARM_JOINTS:
                gains = getattr(arm._arm_config, j.value)
                motor_id = _JOINT_CONFIG[j].motor_id
                lines.append(
                    f"joint {side} {iface} {j.value} {motor_id} {gains.kp} {gains.kd}"
                )
        return "\n".join(lines) + "\n"

    async def enable(self) -> None:
        """Full rt bring-up: prep (core resets) -> Python reads -> arm."""
        await self._link.start()
        await self._link.configure(self._config_text())
        # The core's prep resets the MyActuator motors (multi-turn wrap state
        # changes) — it must complete before Python resolves offsets, and
        # before Python's buses open so no pre-reset frame is ever cached.
        await self._link.prep()

        await self._robot.connect()
        for _side, arm in self._arms():
            await arm.resolve_joint_offsets()
            # Python never calls Motor.enable() in rt mode, so run the
            # MyActuator capability detection (position/torque decode ranges)
            # explicitly — otherwise the passive feedback decode would use
            # the legacy scaling on V4.4 firmware.
            for j in ARM_JOINTS:
                if _JOINT_CONFIG[j].motor_id <= 5:
                    await arm.motors[j]._driver._detect_capabilities()

        # Direct position reads while the bus is still quiet: primes the
        # Damiao caches (gripper included) and captures the gripper values
        # that stay frozen for the session.
        pos_l, pos_r = await self._robot.get_positions()
        if pos_l is not None:
            self._gripper_norm[0] = float(pos_l[7])
        if pos_r is not None:
            self._gripper_norm[1] = float(pos_r[7])

        for side, arm in self._arms():
            arm._command_sink = self._make_sink(side)

        await self._link.arm()
        await self._wait_for_caches()
        _logger.info(
            "rt: armed — axol-rt owns the bus at %.0f Hz; Python streams targets",
            self._loop_hz,
        )

    async def _wait_for_caches(self) -> None:
        """Block until the core's MIT stream has filled every arm-joint cache."""
        deadline = time.monotonic() + 2.0
        while True:
            ready = all(
                arm.motors[j].has_position
                for _side, arm in self._arms()
                for j in ARM_JOINTS
            )
            if ready:
                return
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "rt: armed, but the core's feedback stream did not fill "
                    "the position caches within 2 s"
                )
            await asyncio.sleep(0.02)

    def _make_sink(self, side: int):
        def sink(cmds: list[tuple[float, float, float, float, float]]) -> None:
            self._seq += 1
            self._link.send_target(side, self._seq, cmds)

        return sink

    async def motion_control(
        self, left: np.ndarray | None = None, right: np.ndarray | None = None
    ) -> None:
        """Production motion_control math; the sink ships the result."""
        tasks = []
        if left is not None and self._robot.left is not None:
            tasks.append(self._robot.left.motion_control(left))
        if right is not None and self._robot.right is not None:
            tasks.append(self._robot.right.motion_control(right))
        if tasks:
            await asyncio.gather(*tasks)

    async def get_positions(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Measured positions from the passively filled caches (no CAN sent)."""

        def arm_positions(side: int, arm: AxolArm | None) -> np.ndarray | None:
            if arm is None:
                return None
            values = arm.positions.copy()
            # The cached gripper value stops updating once the core owns the
            # bus (the gripper isn't in its stream); report the frozen
            # pre-arm capture instead of a decaying cache.
            if side in self._gripper_norm:
                values[7] = self._gripper_norm[side]
            return values

        return (
            arm_positions(0, self._robot.left),
            arm_positions(1, self._robot.right),
        )

    async def disable(self) -> None:
        """Disarm the core, tear the link down, then belt-and-braces disable."""
        for _side, arm in self._arms():
            arm._command_sink = None
        try:
            await self._link.disarm()
        except Exception as exc:  # noqa: BLE001 - core may already be gone
            _logger.warning("rt: disarm failed (%s); core teardown continues", exc)
        await self._link.close()
        # The core already disabled the motors on disarm/exit; repeating the
        # shutdown from Python is harmless (the bus is free again) and covers
        # a core that died mid-session. Also closes the Python buses.
        try:
            await self._robot.disable()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            _logger.exception("rt: python-side disable failed")
