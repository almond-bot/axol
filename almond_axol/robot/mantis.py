"""Hardware control for the Mantis handheld data-collection rig.

The Mantis is a pair of handheld devices — a Quest controller rigidly mounted
to the same Damiao gripper the robot uses — held by a human demonstrator. Each
gripper sits alone on its own CAN bus (``can_mantis_l`` / ``can_mantis_r``)
at the production gripper CAN ID (0x08).

:class:`Mantis` mirrors the :class:`~almond_axol.robot.axol.Axol` control surface
(``enable`` / ``get_positions`` / ``motion_control`` / per-side ``positions`` /
``torques``) so the LeRobot wrapper and ``collect-data`` drive it unchanged.
The seven arm joints per side are **virtual**: there is no arm, so
``motion_control`` just latches the commanded joint targets and they are read
back as the "measured" state. The gripper is real — commanded in
POSITION_FORCE mode from the trigger value and observed from motor feedback,
exactly like on the robot. Datasets recorded through this class therefore have
the same schema as robot-collected ones (state/action = 16 joint positions),
with the arm-state channel equal to the commanded IK solution.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from ..constants import ARM_JOINTS, CAN_MANTIS_LEFT, CAN_MANTIS_RIGHT
from ..motor import CanBus, ControlMode, Joint, Motor, MotorError
from .axol import (
    GRIPPER_TRAVEL,
    _await_all_hardware_actions,
    _validated_motion_target,
    calibrate_gripper_open_stop,
)
from .base import RobotBase, mark_hardware_cleanup_uncertain
from .config import AxolConfig, PositionForceConfig

_logger = logging.getLogger(__name__)

_N_ARM = len(ARM_JOINTS)
_DEFAULT_GRIPPER_POSITION = 1.0


class MantisGripperArm:
    """One handheld gripper plus a virtual 7-joint arm.

    Mirrors the parts of :class:`~almond_axol.robot.axol.AxolArm` that the
    control stack touches. Positions are shape (8,) in ``Joint`` order: the 7
    virtual arm joints echo the last ``motion_control`` command (radians, joint
    frame), the gripper is real feedback normalised to [0 = closed, 1 = open].
    """

    def __init__(self, bus: CanBus, gripper_config: PositionForceConfig) -> None:
        self._motor = Motor(bus, Joint.GRIPPER)
        self._gripper_config = gripper_config
        # Raw motor radians of the open / closed hard-stops, found on enable()
        # by the same torque-stop sweep the robot gripper uses.
        self._open_pos = 0.0
        self._closed_pos = GRIPPER_TRAVEL
        self._virtual_arm = np.zeros(_N_ARM, dtype=np.float32)
        # Before the first calibration, raw motor radians cannot be mapped to
        # the public [closed=0, open=1] range.  Echo a known-safe virtual open
        # position until that mapping exists, just as the seven virtual arm
        # joints echo their latest target.
        self._gripper_target = _DEFAULT_GRIPPER_POSITION
        self._enabled = False
        self._disable_pending = False
        self._calibrated = False
        self._telemetry_active = False

    # -- Lifecycle -----------------------------------------------------------

    async def enable(self) -> None:
        """Enable this gripper, calibrating only on its first activation.

        Episode boundaries disable motor torque but deliberately retain the
        measured hard-stop in memory.  Re-enabling for a later episode can
        therefore return directly to POSITION_FORCE mode without sweeping the
        jaws open a second time.
        """
        if self._enabled:
            return

        # A previous disable may have failed after closing the software command
        # gate.  Make one more best-effort attempt before bringing the motor up
        # from a known state.
        if self._disable_pending:
            await self._motor.disable()
            self._disable_pending = False

        try:
            await self._motor.enable()
            self._disable_pending = True
            if not self._calibrated:
                await self._motor.set_control_mode(ControlMode.IMPEDANCE)
                self._open_pos = await calibrate_gripper_open_stop(self._motor)
                self._closed_pos = self._open_pos + GRIPPER_TRAVEL
                self._calibrated = True
            await self._motor.set_control_mode(ControlMode.POSITION_FORCE)
            await self._send_gripper_target()
            self._enabled = True
        except BaseException:
            # Once enable() reached the hardware, a later configuration or
            # calibration failure must not leave a single gripper live.
            self._enabled = False
            if self._disable_pending:
                try:
                    await self._motor.disable()
                except Exception:  # noqa: BLE001 - retain original failure
                    _logger.exception("Mantis gripper cleanup disable failed")
                else:
                    self._disable_pending = False
            raise

    async def disable(self) -> None:
        """Close the command gate immediately, then disable motor torque."""
        self._enabled = False
        if not self._disable_pending:
            return
        await self._motor.disable()
        self._disable_pending = False

    async def force_disable(self) -> None:
        """Disable and verify hardware whose state predates this process."""
        self._enabled = False
        self._disable_pending = True
        await self._motor.disable()
        self._disable_pending = False

    async def start_telemetry(self, hz: float, *, torque: bool = False) -> None:
        await self._motor.start_telemetry(hz, torque=torque)
        self._telemetry_active = True

    async def stop_telemetry(self) -> None:
        await self._motor.stop_telemetry()
        self._telemetry_active = False

    async def wait_for_telemetry(self, timeout: float = 5.0) -> None:
        """Block until the gripper motor has reported at least one position."""
        # With polling disabled, command replies normally seed the cache.  A
        # deferred gripper intentionally receives no command until an episode
        # starts, so there is nothing to wait for during Robot.connect().
        if not self._telemetry_active and not self._enabled:
            return
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while not self._motor.has_position:
            if loop.time() >= deadline:
                raise TimeoutError(
                    "No feedback from the Mantis gripper — check power and CAN wiring"
                )
            await asyncio.sleep(0.01)

    # -- State ---------------------------------------------------------------

    def _normalize(self, raw: float) -> float:
        """Raw motor radians → [0 = closed, 1 = open]."""
        return (raw - self._closed_pos) / (self._open_pos - self._closed_pos)

    def _normalized_position(self) -> float:
        """Return cached feedback, or a safe virtual value before calibration."""
        if not self._calibrated or not self._motor.has_position:
            return self._gripper_target
        return float(np.clip(self._normalize(self._motor.position), 0.0, 1.0))

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def positions(self) -> np.ndarray:
        """Latest positions, shape (8,) in Joint order (cached gripper feedback)."""
        out = np.empty(_N_ARM + 1, dtype=np.float32)
        out[:_N_ARM] = self._virtual_arm
        out[_N_ARM] = self._normalized_position()
        return out

    @property
    def torques(self) -> np.ndarray:
        """Latest torques, shape (8,); the virtual arm joints report zero."""
        out = np.zeros(_N_ARM + 1, dtype=np.float32)
        try:
            out[_N_ARM] = self._motor.torque
        except MotorError:
            # Disabled/deferred motors may not have emitted torque feedback
            # yet.  Zero is the only meaningful placeholder for that state.
            pass
        return out

    async def get_positions(self) -> np.ndarray:
        """Actively read the gripper position; virtual arm joints as stored."""
        out = np.empty(_N_ARM + 1, dtype=np.float32)
        out[:_N_ARM] = self._virtual_arm
        if self._telemetry_active:
            out[_N_ARM] = self._normalized_position()
        else:
            raw = await self._motor.get_position()
            out[_N_ARM] = (
                float(np.clip(self._normalize(raw), 0.0, 1.0))
                if self._calibrated
                else self._gripper_target
            )
        return out

    # -- Commands -------------------------------------------------------------

    async def motion_control(self, q: np.ndarray) -> None:
        """Latch the virtual arm targets and command the real gripper.

        Args:
            q: Shape (8,) targets in Joint order — arm joints in radians
               (stored, nothing physical to move), gripper normalised [0, 1].
        """
        q = _validated_motion_target(q, label="Mantis gripper")
        self._virtual_arm = np.asarray(q[:_N_ARM], dtype=np.float32).copy()
        self._gripper_target = float(np.clip(q[_N_ARM], 0.0, 1.0))
        if not self._enabled:
            return
        await self._send_gripper_target()

    async def _send_gripper_target(self) -> None:
        raw = self._closed_pos + self._gripper_target * (
            self._open_pos - self._closed_pos
        )
        raw = float(np.clip(raw, self._open_pos, self._closed_pos))
        await self._motor.set_position_force(
            raw,
            self._gripper_config.max_speed,
            self._gripper_config.torque_limit,
        )


class Mantis(RobotBase):
    """The Mantis rig's dual handheld grippers behind the ``Axol`` control surface.

    Args:
        config:        Reused for the per-side gripper POSITION_FORCE tuning
                       (``ArmConfig.gripper``); everything else is ignored.
        left_channel:  SocketCAN interface of the left gripper, or ``None`` to omit.
        right_channel: SocketCAN interface of the right gripper, or ``None`` to omit.
    """

    def __init__(
        self,
        config: AxolConfig = AxolConfig(),
        left_channel: str | None = CAN_MANTIS_LEFT,
        right_channel: str | None = CAN_MANTIS_RIGHT,
        *,
        defer_gripper_enable: bool = False,
    ) -> None:
        left_channel = (
            str(left_channel).strip() if left_channel is not None else None
        ) or None
        right_channel = (
            str(right_channel).strip() if right_channel is not None else None
        ) or None
        if left_channel is None and right_channel is None:
            raise ValueError(
                "At least one of left_channel or right_channel must be specified."
            )
        if (
            left_channel is not None
            and right_channel is not None
            and left_channel == right_channel
        ):
            raise ValueError(
                "left_channel and right_channel must name different CAN "
                "interfaces; both Mantis grippers reuse the same motor ID"
            )

        self.left: MantisGripperArm | None = None
        self.right: MantisGripperArm | None = None
        self._defer_gripper_enable = defer_gripper_enable
        self._connected = False
        # True from the first teardown attempt until every gripper disable and
        # bus close has succeeded.  It prevents a failed shutdown from being
        # mistaken for an idle rig and lets disable() retry only unfinished
        # work while the buses needed for a motor-disable retry remain open.
        self._shutdown_pending = False
        self._telemetry_settings: tuple[float, bool] | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._left_bus: CanBus | None = None
        self._right_bus: CanBus | None = None
        if left_channel is not None:
            self._left_bus = CanBus(left_channel)
            self.left = MantisGripperArm(self._left_bus, config.left.gripper)
        if right_channel is not None:
            self._right_bus = CanBus(right_channel)
            self.right = MantisGripperArm(self._right_bus, config.right.gripper)

    # -- Lifecycle -------------------------------------------------------------

    async def enable(self) -> None:
        """Start CAN buses and, by default, immediately enable both grippers.

        ``defer_gripper_enable=True`` changes only the second half: buses are
        connected for reads and telemetry, while the motors remain disabled
        until :meth:`enable_grippers` is called at recording start.
        """
        async with self._lifecycle_lock:
            await self._connect_unlocked()
            if not self._defer_gripper_enable:
                await self._enable_grippers_unlocked()

    async def connect(self) -> None:
        """Start both CAN buses; deferred mode also verifies torque-off."""
        async with self._lifecycle_lock:
            await self._connect_unlocked()

    async def _connect_unlocked(self) -> None:
        if self._shutdown_pending:
            raise MotorError(
                "Mantis shutdown is incomplete; retry disable() before reconnecting"
            )
        if self._connected:
            return
        buses = [b for b in (self._left_bus, self._right_bus) if b is not None]
        results = await asyncio.gather(
            *[b.start() for b in buses], return_exceptions=True
        )
        failures = [r for r in results if isinstance(r, BaseException)]
        if failures:
            await asyncio.gather(*[b.close() for b in buses], return_exceptions=True)
            raise failures[0]

        if self._defer_gripper_enable:
            # A previous process may have died with a motor enabled. Merely
            # skipping enable() would then leave it holding throughout the
            # pre-record phase, so establish and verify torque-off before this
            # deferred connection is considered ready.
            arms = [a for a in (self.left, self.right) if a is not None]
            disabled = await asyncio.gather(
                *[a.force_disable() for a in arms], return_exceptions=True
            )
            failures = [r for r in disabled if isinstance(r, BaseException)]
            if failures:
                # A timed-out force-disable leaves torque state unknown. Keep
                # every bus open and mark this as an incomplete shutdown so
                # disable() can retry only the arms whose pending bit remains
                # set. Closing here would destroy the sole command path while a
                # gripper might still be holding torque.
                self._connected = True
                self._shutdown_pending = True
                raise failures[0]
        self._connected = True

    async def enable_grippers(self) -> None:
        """Enable both grippers as one operation, cleaning up partial failure."""
        async with self._lifecycle_lock:
            await self._connect_unlocked()
            await self._enable_grippers_unlocked()

    async def _enable_grippers_unlocked(self) -> None:
        arms = [a for a in (self.left, self.right) if a is not None]
        telemetry = self._telemetry_settings
        if telemetry is not None:
            await self._stop_telemetry_unlocked()

        results = await asyncio.gather(
            *[a.enable() for a in arms], return_exceptions=True
        )
        failures = [r for r in results if isinstance(r, BaseException)]
        if failures:
            primary_error = failures[0]
            for additional in failures[1:]:
                primary_error.add_note(
                    "Additional Mantis gripper enable failure: "
                    f"{type(additional).__name__}: {additional}"
                )
            await self._rollback_gripper_enable(
                arms, primary_error, context="partial enable"
            )
            if telemetry is not None:
                try:
                    await self._start_telemetry_unlocked(*telemetry)
                except BaseException as telemetry_error:
                    _logger.exception(
                        "Could not restore Mantis telemetry after enable failure"
                    )
                    primary_error.add_note(
                        "Mantis telemetry restoration after enable failure also "
                        f"failed: {type(telemetry_error).__name__}: "
                        f"{telemetry_error}"
                    )
            raise primary_error

        if telemetry is not None:
            try:
                await self._start_telemetry_unlocked(*telemetry)
            except BaseException as telemetry_error:
                await self._rollback_gripper_enable(
                    arms, telemetry_error, context="telemetry restart"
                )
                raise

    async def _rollback_gripper_enable(
        self,
        arms: list[MantisGripperArm],
        primary_error: BaseException,
        *,
        context: str,
    ) -> None:
        """Force torque-off after group enable and retain uncertain ownership."""
        cleanup = await asyncio.gather(
            *(arm.force_disable() for arm in arms), return_exceptions=True
        )
        failures = [result for result in cleanup if isinstance(result, BaseException)]
        if not failures:
            return

        self._shutdown_pending = True
        primary_error.add_note(
            f"Mantis {context} rollback did not confirm every gripper disabled"
        )
        mark_hardware_cleanup_uncertain(primary_error, failures[0])
        for additional in failures[1:]:
            primary_error.add_note(
                "Additional Mantis gripper rollback failure: "
                f"{type(additional).__name__}: {additional}"
            )

    async def disable_grippers(self) -> None:
        """Force-disable both grippers while keeping buses and calibration."""
        async with self._lifecycle_lock:
            arms = [a for a in (self.left, self.right) if a is not None]
            results = await asyncio.gather(
                *[a.force_disable() for a in arms], return_exceptions=True
            )
            failures = [r for r in results if isinstance(r, BaseException)]
            if failures:
                raise failures[0]

    async def disable(self) -> None:
        """Disable both grippers and close both buses, or report uncertainty.

        Every side is attempted even when its peer fails.  The connected flag
        is cleared only after every hardware disable and bus close succeeds;
        otherwise callers must treat ownership as retained and may retry this
        method instead of opening a second connection to uncertain hardware.
        """
        async with self._lifecycle_lock:
            arms = [a for a in (self.left, self.right) if a is not None]
            buses = [b for b in (self._left_bus, self._right_bus) if b is not None]
            await self._stop_telemetry_unlocked()
            first_attempt = not self._shutdown_pending
            self._shutdown_pending = True
            disable_results = await asyncio.gather(
                *[a.force_disable() if first_attempt else a.disable() for a in arms],
                return_exceptions=True,
            )
            failures = [
                result
                for result in disable_results
                if isinstance(result, BaseException)
            ]
            if failures:
                for failure in failures:
                    _logger.error("Mantis gripper disable failed: %s", failure)
                if len(failures) > 1:
                    failures[0].add_note(
                        f"{len(failures) - 1} additional Mantis gripper disable "
                        "failure(s) were logged"
                    )
                raise failures[0]

            # Keep both buses open until both motors have positively disabled:
            # closing the only command path after a timeout would make a retry
            # impossible while a gripper might still be holding torque.
            close_results = await asyncio.gather(
                *[b.close() for b in buses], return_exceptions=True
            )
            failures = [
                result for result in close_results if isinstance(result, BaseException)
            ]
            if failures:
                for failure in failures:
                    _logger.error("Mantis CAN close failed: %s", failure)
                if len(failures) > 1:
                    failures[0].add_note(
                        f"{len(failures) - 1} additional Mantis CAN close "
                        "failure(s) were logged"
                    )
                raise failures[0]
            self._connected = False
            self._shutdown_pending = False

    # -- Telemetry ---------------------------------------------------------------

    async def start_telemetry(self, hz: float, *, torque: bool = False) -> None:
        async with self._lifecycle_lock:
            await self._start_telemetry_unlocked(hz, torque)

    async def _start_telemetry_unlocked(self, hz: float, torque: bool) -> None:
        arms = [a for a in (self.left, self.right) if a is not None]
        results = await asyncio.gather(
            *[a.start_telemetry(hz, torque=torque) for a in arms],
            return_exceptions=True,
        )
        failures = [r for r in results if isinstance(r, BaseException)]
        if failures:
            await asyncio.gather(
                *[a.stop_telemetry() for a in arms], return_exceptions=True
            )
            self._telemetry_settings = None
            raise failures[0]
        self._telemetry_settings = (hz, torque)

    async def stop_telemetry(self) -> None:
        async with self._lifecycle_lock:
            await self._stop_telemetry_unlocked()

    async def _stop_telemetry_unlocked(self) -> None:
        await asyncio.gather(
            *[a.stop_telemetry() for a in (self.left, self.right) if a is not None],
            return_exceptions=True,
        )
        self._telemetry_settings = None

    async def wait_for_telemetry(self, timeout: float = 5.0) -> None:
        await asyncio.gather(
            *[
                a.wait_for_telemetry(timeout)
                for a in (self.left, self.right)
                if a is not None
            ]
        )

    # -- State / commands ----------------------------------------------------------

    async def get_positions(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        left = await self.left.get_positions() if self.left is not None else None
        right = await self.right.get_positions() if self.right is not None else None
        return left, right

    async def motion_control(
        self,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
    ) -> None:
        targets: list[tuple[MantisGripperArm, np.ndarray]] = []
        if left is not None and self.left is not None:
            targets.append(
                (self.left, _validated_motion_target(left, label="left Mantis"))
            )
        if right is not None and self.right is not None:
            targets.append(
                (self.right, _validated_motion_target(right, label="right Mantis"))
            )
        if targets:
            await _await_all_hardware_actions(
                *(arm.motion_control(q) for arm, q in targets)
            )

    # -- Axol-surface stubs -----------------------------------------------------

    async def gravity_compensate(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("The Mantis has no arm to gravity-compensate.")

    def reset_command_state(self) -> None:
        """No command history to clear — the arms are virtual."""

    def torque_residuals(self) -> tuple[None, None]:
        """Return no arm contact signal; Mantis has only virtual arm joints."""
        return None, None
