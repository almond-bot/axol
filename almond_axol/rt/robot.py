"""``RtAxol`` — the Axol robot driven through the Rust realtime core.

Presents the same surface :class:`~almond_axol.teleop.VRTeleop` uses
(``enable`` / ``disable`` / ``get_positions`` / ``motion_control``), but the
CAN buses are owned by the ``axol-rt`` subprocess:

- ``enable()`` runs the split bring-up: the core resets the motors (prep),
  then Python resolves joint offsets and MyActuator decode ranges through a
  Rust maintenance proxy. That proxy exits before the realtime core enables
  and holds, making the core the sole CAN owner while armed. ``Motor`` caches
  fill from the core's per-tick telemetry packets — ~480 packet decodes/s
  replacing ~7,700 Python frame dispatches/s on this CPU-starved Jetson.
- ``motion_control()`` runs the slow model math from
  ``AxolArm.motion_control`` (limits, gravity, the pose *scheduling* of the
  fast terms) and a command sink ships per-joint tuples to the core instead
  of sending CAN from Python.

- The gripper is brought up by Python before the core arms (the classic
  enable/calibrate or attach/restore flow — it needs the quiet bus), then
  driven by the core: ``motion_control``'s slot-7 tuple carries its
  POSITION_FORCE command (motor-frame target, speed limit, torque limit).
- The *fast* physics all run in the core, per 240 Hz tick, from its own
  trajectory and feedback states: a golden-ported trapezoid tracker chases
  the latest target (replacing linear interpolation), the classic 20 rad/s
  command-derivative chain computes smooth friction/inertia feedforwards
  from that executed trajectory (friction params ride the config; the
  pose-scaled ``j_eff`` rides each target), and band-passed velocity damping
  applies the streamed pose-scheduled coefficients against the latest
  feedback within one core tick.
  Computing the damping torque in Python put it ~14 ms behind the motion —
  past 90° of loop phase in the shoulder burst band, where a damper pumps
  instead of damps (the rt-teleop shaking of 2026-08-27; see
  rust/axol-rt/src/filter.rs). Python's own trapezoid (with the engage
  velocity ramp and the output guard) still shapes the 120 Hz target
  stream; the core's tracker re-renders it at wire rate with 1.5x headroom
  on the limits.

Guarded return works exactly as in classic mode: ``torque_residuals`` and
``reset_command_state`` only touch the telemetry-filled caches and local
state, and ``gravity_compensate`` streams its tuples through the same
command sink, so the contact watchdog, the limp contact hold, and the
replanned reset all run against the core.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time

import numpy as np

from ..constants import ARM_JOINTS
from ..motor import ControlMode, Joint
from ..motor.bus import CanBus
from ..motor.motor import _JOINT_CONFIG
from ..robot.axol import Axol, AxolArm
from .link import FeedbackSlot, RtLink

_logger = logging.getLogger(__name__)

_N_ARM = len(ARM_JOINTS)


class RtAxol:
    """Axol with the control loop in the Rust realtime core."""

    # The core's tracker limits get headroom over the Python shaper's caps:
    # the in-core trapezoid exists to render a smooth 240 Hz trajectory and
    # bound corruption, not to be the binding constraint — Python's
    # trapezoid (engage ramps included) already enforces the real teleop
    # limits, so a core tracker at exactly those limits would ride its
    # ceiling during full-speed moves and add avoidable lag.
    _TRACKER_HEADROOM = 1.5

    def __init__(
        self,
        robot: Axol,
        loop_hz: float = 240.0,
        watchdog_ms: float = 150.0,
        abort_deg: float = 25.0,
        max_vel: float = 2.0 * math.pi,
        max_accel: float = 7.0 * math.pi,
        record: str | None = None,
    ) -> None:
        """Wrap ``robot`` for the realtime core.

        Args:
            max_vel: Teleop joint-velocity cap (rad/s) — the core's tracker
                runs at ``_TRACKER_HEADROOM`` times this. Defaults match
                ``VRTeleopConfig.teleop_max_vel``.
            max_accel: Teleop joint-acceleration cap (rad/s²), same
                treatment.
            record: Teleop flight-recorder prefix. When set, measured
                position/torque is captured from the core's feedback packets
                at its native ``loop_hz`` instead of the Python target rate.
        """
        self._robot = robot
        self._loop_hz = loop_hz
        self._watchdog_ms = watchdog_ms
        self._abort_deg = abort_deg
        self._max_vel = max_vel
        self._max_accel = max_accel
        self._seq = 0
        # Telemetry packets received per side since arm.
        self._fb_packets = [0, 0]
        # Pair independently arriving left/right core feedback packets into
        # one 16-DOF flight-recorder row at the native 240 Hz rate. Import the
        # recorder lazily so low-level RT users do not initialize teleop.
        self._rec = None
        self._record_prefix: str | None = None
        if record:
            from ..teleop.recorder import make as make_recorder, resolve_prefix

            self._record_prefix = resolve_prefix(record)
            self._rec = make_recorder(self._record_prefix, "meas", {"qm": 16, "tq": 16})
        self._link = RtLink(trace_prefix=self._record_prefix)
        self._record_qm = np.full(16, np.nan, dtype=np.float32)
        self._record_tq = np.full(16, np.nan, dtype=np.float32)
        self._record_sides: set[int] = set()
        self._record_side_ts: dict[int, float] = {}
        self._recording_engaged = False

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
        trk_vel = self._TRACKER_HEADROOM * self._max_vel
        trk_acc = self._TRACKER_HEADROOM * self._max_accel
        for side, arm in self._arms():
            # The bus channel lives on the CanBus (same package internals).
            bus = self._robot._left_bus if side == 0 else self._robot._right_bus
            iface = bus._channel
            for j in ARM_JOINTS:
                gains = getattr(arm._arm_config, j.value)
                f = gains.friction
                motor_id = _JOINT_CONFIG[j].motor_id
                lines.append(
                    f"joint {side} {iface} {j.value} {motor_id} "
                    f"{gains.kp} {gains.kd} {trk_vel} {trk_acc} "
                    f"{f.fc} {f.k} {f.fv} {f.fo}"
                )
            if arm._has_gripper:
                lines.append(
                    f"gripper {side} {iface} {_JOINT_CONFIG[Joint.GRIPPER].motor_id}"
                )
        return "\n".join(lines) + "\n"

    async def enable(self) -> None:
        """Full rt bring-up: prep (core resets) -> Python reads -> arm."""
        try:
            await self._enable()
        except BaseException:
            # A failed/cancelled __aenter__ has no __aexit__. Run teardown in
            # its own shielded task so every resource acquired by _enable is
            # rolled back before the original failure reaches the caller.
            cleanup = asyncio.create_task(self.disable(), name="rt-startup-rollback")
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except BaseException:  # noqa: BLE001 - preserve original cause
                    # Keep joining after repeated cancellation/interruption.
                    # Otherwise caller can close the loop while this safety
                    # cleanup is merely an unreferenced background task.
                    continue
            if not cleanup.cancelled():
                cleanup_exc = cleanup.exception()
                if cleanup_exc is not None:
                    _logger.error(
                        "rt: startup rollback failed",
                        exc_info=cleanup_exc,
                    )
            else:
                _logger.error("rt: startup rollback was unexpectedly cancelled")
            raise

    async def _enable(self) -> None:
        """Bring up the realtime core; :meth:`enable` owns rollback."""
        self._fb_packets = [0, 0]
        await self._link.start()
        await self._link.configure(self._config_text())
        # The core's prep resets the MyActuator motors (multi-turn wrap state
        # changes) — it must complete before Python resolves offsets, and
        # before Python's buses open so no pre-reset frame is ever cached.
        await self._link.prep()

        await self._robot.connect()
        for _side, arm in self._arms():
            await arm.resolve_joint_offsets()
            # Python never calls Motor.enable() in production control, so run the
            # MyActuator capability detection (position/torque decode ranges)
            # and undervoltage provisioning explicitly. Otherwise passive
            # feedback would use legacy scaling on V4.4 firmware and a fresh
            # motor could retain the factory voltage threshold.
            for j in ARM_JOINTS:
                if _JOINT_CONFIG[j].motor_id <= 5:
                    driver = arm.motors[j]._driver
                    await driver._detect_capabilities()
                    await driver._apply_low_voltage_threshold()

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
        self._link.on_feedback = self._make_feedback_feed()

        # Hand the interfaces over completely: the maintenance proxy exits
        # before the realtime bus threads open their SocketCAN sockets.
        await asyncio.gather(*(bus.close() for bus in self._buses()))
        await self._link.arm()
        await self._wait_for_caches()
        # Prime one full hold target at the measured pose: the core's own
        # bring-up hold has no gravity feedforward (t_ff = 0) and no damping
        # coefficients, so a gravity-loaded joint would sag by ~gravity/kp —
        # and ring on firmware kd alone if disturbed — until the caller's
        # first command, which for teleop is minutes away (JAX compile).
        # One motion_control at the measured pose ships gravity plus the
        # pose-scheduled fast-term coefficients; the watchdog then holds it,
        # damping included.
        pos_l, pos_r = await self.get_positions()
        await self.motion_control(left=pos_l, right=pos_r)
        _logger.info(
            "rt: armed — axol-rt owns the bus at %.0f Hz; Python streams targets",
            self._loop_hz,
        )

    async def __aenter__(self) -> RtAxol:
        """Enter the async context, arming the core via :meth:`enable`."""
        await self.enable()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Exit the async context, tearing down via :meth:`disable`."""
        await self.disable()

    async def start_telemetry(self, hz: float, *, torque: bool = False) -> None:
        """No-op: the core already streams full telemetry every tick.

        Positions, velocities, and torques for every slot arrive in the
        per-tick ``F`` packets regardless of ``hz`` / ``torque`` — a poll
        loop would need the bus, which the core owns. Kept so classic
        flows (gravity-comp, waypoints, tune.motion, the LeRobot robot)
        run unchanged against ``RtAxol``.
        """
        _logger.debug(
            "rt: start_telemetry(%s) ignored — core streams at %.0f Hz",
            hz,
            self._loop_hz,
        )

    async def stop_telemetry(self) -> None:
        """No-op counterpart of :meth:`start_telemetry`."""

    async def wait_for_telemetry(self, timeout: float = 5.0) -> None:
        """Block until the core's telemetry stream is flowing for every arm.

        Same contract as :meth:`Axol.wait_for_telemetry`; ``enable`` already
        waited once, so after a successful bring-up this returns immediately.
        """
        deadline = time.monotonic() + timeout
        arms = self._arms()

        def ready() -> bool:
            return all(
                self._fb_packets[side] > 0
                and all(arm.motors[joint].has_position for joint in ARM_JOINTS)
                for side, arm in arms
            )

        while not ready():
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"rt: incomplete arm telemetry from the core after {timeout:.1f} s"
                )
            await asyncio.sleep(0.02)

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

    def _buses(self) -> list[CanBus]:
        out = []
        for side, _arm in self._arms():
            bus = self._robot._left_bus if side == 0 else self._robot._right_bus
            if bus is not None:
                out.append(bus)
        return out

    def _make_feedback_feed(self):
        """Build the telemetry handler that fills the Motor caches.

        Writes the same four fields the passive listener path caches
        (position, velocity, torque, receive timestamp), in the same motor
        frame — the core's decode is a bit-for-bit port of the drivers' —
        so every downstream consumer (``arm.positions``, ``torque_residuals``,
        the recorder) is source-agnostic.
        """
        arms = dict(self._arms())
        joints = list(ARM_JOINTS)
        expected_sides = set(arms)

        def feed(side: int, slots: dict[int, FeedbackSlot]) -> None:
            arm = arms.get(side)
            if arm is None:
                return
            self._fb_packets[side] += 1
            for i, (pos, vel, tau, ts) in slots.items():
                if i < _N_ARM:
                    motor = arm.motors[joints[i]]
                elif arm._has_gripper:
                    motor = arm.motors[Joint.GRIPPER]
                else:
                    continue
                motor._position = pos
                motor._velocity = vel
                motor._torque = tau
                motor._feedback_ts = ts
                if self._rec is not None and self._recording_engaged:
                    self._record_qm[side * 8 + i] = pos
                    self._record_tq[side * 8 + i] = tau
            if self._rec is not None and self._recording_engaged:
                self._record_sides.add(side)
                if slots:
                    self._record_side_ts[side] = max(
                        value[3] for value in slots.values()
                    )
                if self._record_sides >= expected_sides:
                    # The core packets carry motor-frame positions. Standard
                    # `_meas.npz` files carry joint-frame positions (zero at
                    # rest, normalized gripper), so convert through the same
                    # AxolArm properties the classic recorder uses. Constant
                    # motor offsets do not affect vibration spectra, but raw
                    # values make pose attribution and replay incorrect.
                    for record_side, record_arm in arms.items():
                        base = record_side * 8
                        self._record_qm[base : base + 8] = record_arm.positions
                        self._record_tq[base : base + 8] = record_arm.torques
                    # FeedbackSlot timestamps use time.time() (reconstructed
                    # from the core's per-frame age). Convert their mean to
                    # the recorder's monotonic epoch at the instant of use;
                    # this preserves the real 240 Hz sample grid even when
                    # Python receives several socket packets in a burst.
                    wall_ts = sum(
                        self._record_side_ts[s] for s in expected_sides
                    ) / len(expected_sides)
                    mono_ts = wall_ts + (time.monotonic() - time.time())
                    self._rec.record(
                        timestamp=mono_ts,
                        qm=self._record_qm,
                        tq=self._record_tq,
                    )
                    self._record_sides.clear()
                    self._record_side_ts.clear()

        return feed

    @property
    def records_measurements_at_control_rate(self) -> bool:
        """Whether this wrapper owns the standard ``_meas`` recording."""
        return self._rec is not None

    def set_recording_engaged(self, engaged: bool) -> None:
        """Gate measured and Rust-internal traces to the VR engaged segment."""
        if engaged == self._recording_engaged:
            return
        if self._rec is None:
            return
        if engaged:
            self._link.set_recording_engaged(engaged)
            self._recording_engaged = True
            self._rec.set_engaged(True)
            return

        # Clear local state before touching a faulted core: RtLink._send may
        # reject the trace packet, but teardown and recorder finalization must
        # still proceed. Disengagement is deliberately best-effort.
        self._recording_engaged = False
        self._record_sides.clear()
        self._record_side_ts.clear()
        try:
            self._link.set_recording_engaged(False)
        except Exception as exc:  # noqa: BLE001 - core may already be faulted
            _logger.warning("rt: could not disengage the core trace (%s)", exc)
        try:
            self._rec.set_engaged(False)
        except Exception:  # noqa: BLE001 - do not abort hardware teardown
            _logger.exception("rt: could not disengage the measurement trace")

    async def _wait_for_caches(self) -> None:
        """Block until the core's telemetry stream is flowing for every arm.

        The caches were already primed by the pre-arm direct reads; this
        confirms the core's own feedback path (MIT replies -> `F` packets)
        is live before the caller starts streaming against it.
        """
        await self.wait_for_telemetry(timeout=2.0)

    def _make_sink(self, side: int):
        def sink(cmds: list[tuple[float, ...]]) -> None:
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
        """Per-arm measured-minus-gravity torques from the telemetry caches.

        The core's telemetry refreshes measured torque every tick, so this
        needs no CAN traffic — same contract as ``Axol``.
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
        """Measured positions from the telemetry-filled caches (no CAN sent).

        Every joint — gripper included — refreshes from the core's per-tick
        telemetry packets once armed (the gripper's POSITION_FORCE replies
        land in the same slot stream).
        """

        def arm_positions(arm: AxolArm | None) -> np.ndarray | None:
            return arm.positions.copy() if arm is not None else None

        return (
            arm_positions(self._robot.left),
            arm_positions(self._robot.right),
        )

    async def disable(self) -> None:
        """Disarm the core, tear the link down, then belt-and-braces disable."""
        if self._rec is not None:
            self.set_recording_engaged(False)
        for _side, arm in self._arms():
            arm._command_sink = None
        self._link.on_feedback = None
        try:
            await self._link.disarm()
        except Exception as exc:  # noqa: BLE001 - core may already be gone
            _logger.warning("rt: disarm failed (%s); core teardown continues", exc)
        try:
            await self._link.close()
        except Exception:  # noqa: BLE001 - continue with maintenance disable
            _logger.exception("rt: core link teardown failed")
        # Reopen Rust maintenance proxies only after proving that the core's
        # bus-owning process has exited.
        buses = self._buses()
        core_process = self._link._proc
        core_stopped = core_process is None or core_process.poll() is not None
        if core_stopped:
            results = await asyncio.gather(
                *(bus.start() for bus in buses), return_exceptions=True
            )
        else:
            # Never contend for SocketCAN with a core whose teardown failed.
            # Its own watchdog/exit guard remains the only safe motor owner.
            _logger.error(
                "rt: core process is still running; refusing to reopen "
                "maintenance proxies"
            )
            results = [RuntimeError("realtime core still owns the bus")] * len(buses)
        for bus, result in zip(buses, results):
            if isinstance(result, BaseException):
                _logger.warning(
                    "rt: could not reopen maintenance proxy on %s (%s)",
                    bus._channel,
                    result,
                )
        # The core already disabled the motors on disarm/exit; repeating the
        # shutdown from Python is harmless (the bus is free again) and covers
        # a core that died mid-session. Also closes the Python buses.
        try:
            await self._robot.disable()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            _logger.exception("rt: python-side disable failed")
        if self._rec is not None:
            # Join any disengage writer before teardown returns.
            try:
                self._rec.dump()
            except Exception:  # noqa: BLE001 - continue trace finalization
                _logger.exception("rt: could not dump the measurement trace")
        if self._record_prefix is not None and core_stopped:
            # Rust deliberately writes its high-volume trace outside Python
            # while armed. The bus is down now, so compacting cannot perturb
            # control timing and the operator gets one coherent recording.
            from ..teleop.recorder import compact_rt_trace

            try:
                await asyncio.to_thread(compact_rt_trace, self._record_prefix)
            except Exception:  # noqa: BLE001 - retain raw CSVs for recovery
                _logger.exception("rt: could not compact the control trace")
        elif self._record_prefix is not None:
            _logger.warning(
                "rt: retaining raw control trace because the core is still running"
            )
