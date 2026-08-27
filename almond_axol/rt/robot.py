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

- The gripper is brought up by Python before the core arms (the classic
  enable/calibrate or attach/restore flow — it needs the quiet bus), then
  driven by the core: ``motion_control``'s slot-7 tuple carries its
  POSITION_FORCE command (motor-frame target, speed limit, torque limit).
- Host damping runs *in the core*: ``motion_control`` ships the
  pose-scheduled coefficients (effective kd_host, band-pass centre, q) with
  every target and the core applies band-passed velocity damping each
  240 Hz tick against same-tick feedback. Computing the torque in Python
  put it ~14 ms behind the motion — past 90° of loop phase in the shoulder
  burst band, where a damper pumps instead of damps (the rt-teleop shaking
  of 2026-08-27; see rust/axol-rt/src/filter.rs).

Guarded return works exactly as in classic mode: ``torque_residuals`` and
``reset_command_state`` only touch the passively filled caches and local
state, and ``gravity_compensate`` streams its tuples through the same
command sink, so the contact watchdog, the limp contact hold, and the
replanned reset all run against the core.
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from ..constants import ARM_JOINTS
from ..motor import ControlMode, Joint
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
            if arm._has_gripper:
                lines.append(
                    f"gripper {side} {iface} {_JOINT_CONFIG[Joint.GRIPPER].motor_id}"
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

        # Gripper bring-up runs from Python while the bus is still quiet —
        # the exact classic flow (enable/calibrate or attach/restore) the
        # core can't do. The core then streams its POSITION_FORCE commands.
        for _side, arm in self._arms():
            await self._bring_up_gripper(arm)

        # Direct position reads before the core starts streaming: primes
        # every feedback cache (the gripper norm now uses the freshly
        # calibrated limits).
        await self._robot.get_positions()

        for side, arm in self._arms():
            arm._command_sink = self._make_sink(side)

        await self._link.arm()
        await self._wait_for_caches()
        # Prime one full hold target at the measured pose: the core's own
        # bring-up hold has no gravity feedforward (t_ff = 0) and no damping
        # coefficients, so a gravity-loaded joint would sag by ~gravity/kp —
        # and ring on firmware kd alone if disturbed — until the caller's
        # first command, which for teleop is minutes away (JAX compile).
        # One motion_control at the measured pose ships the gravity/friction
        # terms plus the pose-scheduled damping coefficients; the watchdog
        # then holds it, damping included.
        pos_l, pos_r = await self.get_positions()
        await self.motion_control(left=pos_l, right=pos_r)
        _logger.info(
            "rt: armed — axol-rt owns the bus at %.0f Hz; Python streams targets",
            self._loop_hz,
        )

    async def _bring_up_gripper(self, arm: AxolArm) -> None:
        """The classic gripper bring-up (see ``AxolArm.enable``), standalone.

        Cold gripper: enable, calibrate the open stop in IMPEDANCE mode
        (torque-seek sweep), then switch to POSITION_FORCE. A gripper still
        holding from a previous session: attach without disturbing torque and
        restore the persisted calibration (re-measuring would drop whatever
        it grips). Must run before the core arms — this sends CAN.
        """
        if not arm._has_gripper:
            return
        motor = arm.motors[Joint.GRIPPER]
        if await motor.is_holding():
            await motor.attach(ControlMode.POSITION_FORCE)
            await arm._restore_gripper_calibration()
        else:
            await motor.enable()
            await motor.set_control_mode(ControlMode.IMPEDANCE)
            await arm._calibrate_gripper()
            await motor.set_control_mode(ControlMode.POSITION_FORCE)

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
        def sink(
            cmds: list[tuple[float, float, float, float, float, float, float, float]],
        ) -> None:
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

    async def gravity_compensate(
        self,
        kd: float = 0.5,
        free_joints: set[Joint] | None = None,
        gripper_targets: tuple[float | None, float | None] | None = None,
    ) -> None:
        """One gravity-comp cycle, streamed through the core's command sink.

        Backs the guarded-return contact hold (limp arms, gravity held by
        feedforward). With the sinks installed, ``AxolArm.gravity_compensate``
        ships its tuples to the core instead of the bus — Python never
        touches the wire. Same signature as :meth:`Axol.gravity_compensate`.
        """
        await self._robot.gravity_compensate(kd, free_joints, gripper_targets)

    def torque_residuals(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Per-arm measured-minus-gravity torques from the passive caches.

        The core's own MIT stream refreshes measured torque every tick, so
        this needs no CAN traffic — same contract as ``Axol``.
        """
        return self._robot.torque_residuals()

    def reset_command_state(self) -> None:
        """Clear command history on both arms (pure Python state)."""
        self._robot.reset_command_state()

    def reset_gravity_hold(self) -> None:
        """Re-snapshot the gravity-comp hold setpoint (pure Python state)."""
        self._robot.reset_gravity_hold()

    async def get_positions(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Measured positions from the passively filled caches (no CAN sent).

        Every joint — gripper included — refreshes from the core's own
        stream once armed (the gripper's POSITION_FORCE replies land in the
        same passive caches).
        """

        def arm_positions(arm: AxolArm | None) -> np.ndarray | None:
            return arm.positions.copy() if arm is not None else None

        return (
            arm_positions(self._robot.left),
            arm_positions(self._robot.right),
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
