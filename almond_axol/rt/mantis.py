"""Mantis grippers driven through the Rust realtime core.

:class:`Mantis` (re-exported as :class:`almond_axol.robot.Mantis`) wraps
:class:`~almond_axol.robot.mantis.MantisHardware` the way
:class:`~almond_axol.robot.Axol` wraps ``AxolHardware``: Python keeps the
tracker/IK/collection
logic and the gripper's *maintenance* flows, while every per-tick
POSITION_FORCE command and every feedback frame goes through ``axol-rt``,
which solely owns the two gripper buses and paces the loop with hard,
GIL-free timing. No Python control loop touches the wire while a take runs.

Lifecycle
---------

The Mantis rig has no arm to hold up, so unlike the robot its grippers are
torqued *per take*: ``collect-data`` enables them when a recording starts
and disables them when it ends, with the buses idle in between. The core
follows the same cadence — one ``axol-rt`` process is armed on the gripper
buses for the duration of a take and disarmed (motors disabled by the core)
at its end:

* :meth:`enable` / :meth:`connect` — open the Python maintenance proxies
  (:class:`~almond_axol.motor.bus.CanBus`, itself a Rust proxy) and, in
  deferred mode, verify torque-off; no core yet.
* :meth:`enable_grippers` — bring the grippers up from Python on the quiet
  bus (enable, one-time open-stop calibration, POSITION_FORCE), then hand
  the interfaces to the core: close the proxies, ``arm``, wait for the core's
  feedback stream, and re-send the latched targets through the core.
* :meth:`motion_control` — latches the virtual arm and, while armed, ships
  the gripper tuple to the core (slot 7 of the 8-slot target).
* :meth:`disable_grippers` — ``disarm`` (the core disables the motors),
  stop the core, reopen the proxies, and repeat the disable from Python so
  torque-off is *verified* — a Mantis gripper is always safe to release,
  so unlike ``Axol`` a core fault never leaves it holding.
* :meth:`disable` — session end: disarm if armed, then close the buses.

Between takes the gripper caches are refreshed by the explicit reads in
:meth:`get_positions` (the classic ``MantisHardware`` path); while armed, the
core's ``F`` packets fill them at ``loop_hz`` and reads return the cache.
"""

from __future__ import annotations

import asyncio
import bisect
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Self

import numpy as np

from ..constants import ARM_JOINTS
from ..motor import Joint
from ..motor.motor import _JOINT_CONFIG
from ..robot.base import RobotBase, mark_hardware_cleanup_uncertain
from ..robot.config import AxolConfig
from ..robot.mantis import MantisGripperArm, MantisHardware
from ..settings import SHARED
from .link import FeedbackSlot, RtLink, config_header

_logger = logging.getLogger(__name__)

_N_ARM = len(ARM_JOINTS)
_GRIPPER_SLOT = _N_ARM
# The gripper is the only motor on each bus, so the Python max-step gate
# does not exist; give the core a generous corruption-defense bound
# (radians of raw target change per tick) — a full gripper travel per tick
# is still far above any real trigger motion.
_MAX_STEP_RAD = 10.0


class Mantis(RobotBase):
    """The Mantis rig behind the ``Axol`` control surface, core-driven per take.

    Two gripper motors (one per SocketCAN bus) and no arm joints; the
    ``motion_control`` surface accepts full 8-slot targets and only slot 7
    (the gripper) reaches the hardware::

        async with Mantis() as mantis:
            await mantis.motion_control(left=q, right=q)

    Args:
        config:        Per-side gripper POSITION_FORCE tuning
                       (``ArmConfig.gripper``); everything else is ignored.
        left_channel:  SocketCAN interface of the left gripper, or ``None``.
        right_channel: SocketCAN interface of the right gripper, or ``None``.
        defer_gripper_enable: Leave the motors torque-off and the core
                     unstarted until :meth:`enable_grippers` (data collection).

    The keyword-only core options rarely need changing:

    Args:
        loop_hz:     Core loop rate (the gripper's POSITION_FORCE command
                     and feedback cadence).
        watchdog_ms: Core watchdog — with no fresh target for this long it
                     holds the last one (a stale trigger never opens the jaws).
        record:      Flight-recorder prefix; enables the core's internal
                     per-take trace (``<prefix>_take<N>_rt-<side>.csv``).
    """

    def __init__(
        self,
        config: AxolConfig | None = None,
        left_channel: str | None = SHARED,
        right_channel: str | None = SHARED,
        *,
        defer_gripper_enable: bool = False,
        loop_hz: float = 240.0,
        watchdog_ms: float = 150.0,
        record: str | None = None,
    ) -> None:
        self._init_core(
            MantisHardware(
                config=config,
                left_channel=left_channel,
                right_channel=right_channel,
                defer_gripper_enable=defer_gripper_enable,
            ),
            loop_hz=loop_hz,
            watchdog_ms=watchdog_ms,
            record=record,
        )

    @classmethod
    def _wrap(
        cls,
        hardware: MantisHardware,
        *,
        loop_hz: float = 240.0,
        watchdog_ms: float = 150.0,
        record: str | None = None,
    ) -> Self:
        """Build the rig around an already-constructed low-level object.

        Internal: lets tests substitute a hand-built
        :class:`~almond_axol.robot.mantis.MantisHardware` (fake buses) for
        the one :meth:`__init__` would construct.
        """
        self = cls.__new__(cls)
        self._init_core(
            hardware, loop_hz=loop_hz, watchdog_ms=watchdog_ms, record=record
        )
        return self

    def _init_core(
        self,
        hardware: MantisHardware,
        *,
        loop_hz: float,
        watchdog_ms: float,
        record: str | None,
    ) -> None:
        self._robot = hardware
        self._loop_hz = loop_hz
        self._watchdog_ms = watchdog_ms
        self._link: RtLink | None = None
        self._seq = 0
        self._takes = 0
        self._fb_packets = [0, 0]
        self._record_prefix: str | None = None
        if record:
            from ..teleop.recorder import resolve_prefix

            self._record_prefix = resolve_prefix(record)
        self._recording_engaged = False
        # Python telemetry settings paused for the armed stretch.
        self._paused_telemetry: tuple[float, bool] | None = None
        self._lifecycle_lock = asyncio.Lock()
        # Timestamped state history for capture-aligned observations, same
        # shape and clock mapping as Axol.state_nearest.
        self._state_history: deque[
            tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = deque(maxlen=512)
        self._state_cond = threading.Condition()
        self._state_sides: set[int] = set()
        self._state_side_ts: dict[int, float] = {}

    # -- Surface shared with Axol ---------------------------------------------

    @property
    def left(self) -> MantisGripperArm | None:
        return self._robot.left

    @property
    def right(self) -> MantisGripperArm | None:
        return self._robot.right

    @property
    def armed(self) -> bool:
        """True while ``axol-rt`` owns the gripper buses."""
        return self._link is not None

    @property
    def fault(self) -> str | None:
        """The armed core's latched ``fault: ...``, or ``None``."""
        return self._link.fault if self._link is not None else None

    @property
    def limp(self) -> str | None:
        """The armed core's latched ``limp: ...``, or ``None``.

        Only arm joints go limp; a gripper-only core never does. Exposed for
        callers that poll :attr:`~almond_axol.robot.Axol.limp` generically.
        """
        return self._link.limp if self._link is not None else None

    def _arms(self) -> list[tuple[int, MantisGripperArm]]:
        out: list[tuple[int, MantisGripperArm]] = []
        if self._robot.left is not None:
            out.append((0, self._robot.left))
        if self._robot.right is not None:
            out.append((1, self._robot.right))
        return out

    def _config_text(self) -> str:
        lines = [
            *config_header(),
            f"loop_hz {self._loop_hz}",
            f"watchdog_ms {self._watchdog_ms}",
            f"max_step_rad {_MAX_STEP_RAD}",
        ]
        motor_id = _JOINT_CONFIG[Joint.GRIPPER].motor_id
        for side, arm in self._arms():
            lines.append(f"gripper {side} {arm.channel} {motor_id}")
        return "\n".join(lines) + "\n"

    # -- Lifecycle --------------------------------------------------------------

    async def enable(self) -> None:
        """Open the buses; unless deferred, also arm the grippers now.

        Mirrors :meth:`MantisHardware.enable`: ``defer_gripper_enable=True`` (data
        collection) leaves the motors torque-off and the core unstarted
        until :meth:`enable_grippers`.
        """
        async with self._lifecycle_lock:
            await self._robot.connect()
            if not self._robot._defer_gripper_enable:
                await self._arm_unlocked()

    async def connect(self) -> None:
        """Open the buses (deferred mode also verifies torque-off); no core."""
        async with self._lifecycle_lock:
            await self._robot.connect()

    async def enable_grippers(self) -> None:
        """Bring both grippers up and hand their buses to the realtime core.

        Idempotent while armed. The first activation runs the open-stop
        calibration sweep (Python, quiet bus); later takes return directly
        to POSITION_FORCE. Any failure force-disables both grippers before
        it propagates, with the core stopped and the proxies reopened.
        """
        async with self._lifecycle_lock:
            await self._robot.connect()
            await self._arm_unlocked()

    async def _arm_unlocked(self) -> None:
        if self._link is not None:
            return
        self._takes += 1
        self._fb_packets = [0, 0]
        self._seq = 0
        trace_prefix = (
            f"{self._record_prefix}_take{self._takes}"
            if self._record_prefix is not None
            else None
        )
        link = RtLink(trace_prefix=trace_prefix)
        buses_handed_over = False
        try:
            # Python polling must not share the bus with the core; remember
            # the settings so the between-takes state is restored on disarm.
            self._paused_telemetry = self._robot._telemetry_settings
            await self._robot.stop_telemetry()

            await link.start()
            await link.configure(self._config_text())
            # Nothing to reset on a gripper-only bus (prep never touches the
            # gripper) — this just drains the interfaces before Python's
            # bring-up frames.
            await link.prep()

            # Classic bring-up on the quiet bus: enable, calibrate once,
            # POSITION_FORCE, first target. The core's own bring-up then
            # only verifies the mode register and reads the position.
            await self._robot.enable_grippers()
            # Prime the caches with direct reads while Python still owns the
            # bus (the gripper norm now uses the calibrated limits).
            await self._robot.get_positions()

            for side, arm in self._arms():
                arm._command_sink = self._make_sink(link, side)
            link.on_feedback = self._make_feedback_feed()

            # Hand the interfaces over completely: the maintenance proxies
            # exit before the core's bus threads open their sockets.
            await asyncio.gather(*(bus.close() for bus in self._robot.buses))
            buses_handed_over = True
            await link.arm()
            self._link = link
            # The core leaves the gripper uncommanded (and streams no
            # feedback) until its first target: re-send the latched targets
            # — the ones Python just commanded — through the core so its
            # watchdog holds them from the first tick, then confirm the
            # feedback path is live.
            for _side, arm in self._arms():
                await arm._send_gripper_target()
            await self._wait_for_caches()
            _logger.info(
                "rt: Mantis armed — axol-rt owns the gripper buses at %.0f Hz",
                self._loop_hz,
            )
        except BaseException as enable_error:
            self._link = None
            for _side, arm in self._arms():
                arm._command_sink = None
            link.on_feedback = None
            try:
                await self._stop_core(link, buses_handed_over, enable_error)
            except BaseException as cleanup_error:
                if cleanup_error is not enable_error:
                    mark_hardware_cleanup_uncertain(enable_error, cleanup_error)
            raise

    async def disable_grippers(self) -> None:
        """Torque both grippers off, retaining buses and calibration.

        While armed: disarm the core (it disables the motors), stop it,
        reopen the proxies and verify the disable from Python. Not armed:
        the classic force-disable.
        """
        async with self._lifecycle_lock:
            await self._disarm_unlocked()

    async def _disarm_unlocked(self) -> None:
        link = self._link
        if link is None:
            if self._robot.is_connected:
                await self._robot.disable_grippers()
            return
        if self._recording_engaged:
            self.set_recording_engaged(False)
        self._link = None
        for _side, arm in self._arms():
            arm._command_sink = None
        link.on_feedback = None
        await self._stop_core(link, True, None)

    async def _stop_core(
        self,
        link: RtLink,
        buses_handed_over: bool,
        cause: BaseException | None,
    ) -> None:
        """Disarm and stop ``link``, then verify torque-off from Python.

        Unlike the robot, a gripper left holding is never the safe choice
        here — there is no arm to fall — so after a fault, a failed disarm,
        or a startup failure the Python side always repeats the disable
        once the bus is free. If the core process refuses to exit the
        proxies are *not* reopened (never contend for a bus the core may
        still own) and the failure is reported as hardware-uncertain.
        """
        fault = link.fault
        if fault is not None:
            _logger.warning("rt: Mantis core reported %s", fault)
        if buses_handed_over:
            try:
                await link.disarm()
            except Exception as exc:  # noqa: BLE001 - core may already be gone
                _logger.warning(
                    "rt: Mantis disarm failed (%s); stopping the core and "
                    "disabling from Python",
                    exc,
                )
        try:
            await link.close()
        except Exception:  # noqa: BLE001 - continue with the maintenance disable
            _logger.exception("rt: Mantis core link teardown failed")

        core_process = link._proc
        if core_process is not None and core_process.poll() is None:
            error = RuntimeError(
                "realtime core process is still running; refusing to reopen "
                "the Mantis maintenance proxies"
            )
            if cause is not None:
                mark_hardware_cleanup_uncertain(cause, error)
                return
            raise error

        if buses_handed_over:
            results = await asyncio.gather(
                *(bus.start() for bus in self._robot.buses), return_exceptions=True
            )
            failures = [r for r in results if isinstance(r, BaseException)]
            if failures:
                for bus, result in zip(self._robot.buses, results):
                    if isinstance(result, BaseException):
                        _logger.error(
                            "rt: could not reopen the Mantis proxy on %s (%s)",
                            bus.channel,
                            result,
                        )
                if cause is not None:
                    mark_hardware_cleanup_uncertain(cause, failures[0])
                    return
                raise failures[0]

        # The core disabled the motors on a healthy disarm; repeating it here
        # verifies torque-off and covers every other path (fault, crash,
        # startup failure before the hand-over).
        try:
            await self._robot.disable_grippers()
        except BaseException as disable_error:
            if cause is not None:
                mark_hardware_cleanup_uncertain(cause, disable_error)
                return
            raise

        paused = self._paused_telemetry
        self._paused_telemetry = None
        if paused is not None:
            hz, torque = paused
            try:
                await self._robot.start_telemetry(hz, torque=torque)
            except Exception:  # noqa: BLE001 - torque is already off
                _logger.exception("rt: could not restore Mantis telemetry polling")

    async def open_grippers(self) -> None:
        """Open both grippers fully through the core, then release torque.

        The pre-record move (see :meth:`MantisHardware.open_grippers`): arm, drive
        the jaws to the calibrated open stop, confirm from feedback, disarm.
        A failure — or a cancellation mid-move (a Stop during the pre-record
        prep) — still torques both grippers off before it propagates.
        """
        async with self._lifecycle_lock:
            await self._robot.connect()
            arms = [arm for _side, arm in self._arms()]
            for arm in arms:
                arm.preset_open_target()
            try:
                await self._arm_unlocked()
                await asyncio.gather(*(arm.open_fully() for arm in arms))
            except BaseException as open_error:
                try:
                    await self._disarm_unlocked()
                except BaseException as cleanup_error:
                    mark_hardware_cleanup_uncertain(open_error, cleanup_error)
                raise
            await self._disarm_unlocked()

    async def disconnect(self) -> None:
        """Close the buses; on a Mantis this is :meth:`disable`.

        ``Axol.disconnect()`` leaves the arms holding for a later process.
        A Mantis gripper is disabled at every take end and has no holding
        state worth preserving, so the grippers are torqued off here too.
        """
        await self.disable()

    async def disable(self) -> None:
        """Session end: disarm if armed, torque off, close the buses."""
        async with self._lifecycle_lock:
            try:
                await self._disarm_unlocked()
            except BaseException as disarm_error:
                try:
                    await self._robot.disable()
                except BaseException as close_error:
                    mark_hardware_cleanup_uncertain(disarm_error, close_error)
                raise
            await self._robot.disable()

    # -- Telemetry --------------------------------------------------------------

    async def start_telemetry(self, hz: float, *, torque: bool = False) -> None:
        """Python polling between takes; a no-op while the core streams."""
        if self._link is not None:
            _logger.debug("rt: start_telemetry ignored — Mantis core streams")
            return
        await self._robot.start_telemetry(hz, torque=torque)

    async def stop_telemetry(self) -> None:
        if self._link is not None:
            return
        await self._robot.stop_telemetry()

    async def wait_for_telemetry(self, timeout: float = 5.0) -> None:
        """Wait for gripper feedback: the core's stream while armed, else classic."""
        if self._link is None:
            await self._robot.wait_for_telemetry(timeout)
            return
        deadline = time.monotonic() + timeout
        arms = self._arms()

        def ready() -> bool:
            return all(
                self._fb_packets[side] > 0 and arm.motor.has_position
                for side, arm in arms
            )

        while not ready():
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"rt: no Mantis gripper telemetry from the core after {timeout:.1f} s"
                )
            await asyncio.sleep(0.02)

    async def _wait_for_caches(self) -> None:
        await self.wait_for_telemetry(timeout=2.0)

    def _make_sink(
        self, link: RtLink, side: int
    ) -> Callable[[list[tuple[float, ...]]], None]:
        def sink(cmds: list[tuple[float, ...]]) -> None:
            self._seq += 1
            link.send_target(side, self._seq, cmds)

        return sink

    def _make_feedback_feed(self) -> Callable[[int, dict[int, FeedbackSlot]], None]:
        """Fill the gripper Motor caches from the core's ``F`` packets.

        Same four fields the driver's passive path caches, in motor frame,
        so ``arm.positions`` / ``arm.torques`` and every consumer above them
        are source-agnostic. A complete left+right pair also appends one
        snapshot to the state history for :meth:`state_nearest`.
        """
        arms = dict(self._arms())
        expected_sides = set(arms)

        def feed(side: int, slots: dict[int, FeedbackSlot]) -> None:
            arm = arms.get(side)
            if arm is None:
                return
            self._fb_packets[side] += 1
            gripper = slots.get(_GRIPPER_SLOT)
            if gripper is not None:
                pos, vel, tau, ts = gripper
                motor = arm.motor
                motor._position = pos
                motor._velocity = vel
                motor._torque = tau
                motor._feedback_ts = ts
                self._state_side_ts[side] = ts
            self._state_sides.add(side)
            if self._state_sides < expected_sides:
                return
            try:
                wall_ts = sum(self._state_side_ts[s] for s in expected_sides) / len(
                    expected_sides
                )
                perf_ts = time.perf_counter() - (time.time() - wall_ts)
                left = arms.get(0)
                right = arms.get(1)
                empty = np.zeros(_N_ARM + 1, dtype=np.float32)
                snapshot = (
                    perf_ts,
                    left.positions.copy() if left is not None else empty.copy(),
                    right.positions.copy() if right is not None else empty.copy(),
                    left.torques.copy() if left is not None else empty.copy(),
                    right.torques.copy() if right is not None else empty.copy(),
                )
            except (KeyError, RuntimeError, TypeError, ValueError):
                snapshot = None
            if snapshot is not None:
                with self._state_cond:
                    if (
                        not self._state_history
                        or snapshot[0] > self._state_history[-1][0]
                    ):
                        self._state_history.append(snapshot)
                        self._state_cond.notify_all()
            self._state_sides.clear()
            self._state_side_ts.clear()

        return feed

    def state_nearest(
        self, target_perf_ts: float, timeout: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None:
        """Return the telemetry snapshot nearest a camera exposure timestamp.

        Same contract as :meth:`~almond_axol.robot.Axol.state_nearest`; only populated while
        the core is armed.
        """
        deadline = time.perf_counter() + timeout
        with self._state_cond:
            while True:
                if self._state_history:
                    oldest_ts = self._state_history[0][0]
                    newest_ts = self._state_history[-1][0]
                    if target_perf_ts < oldest_ts:
                        return None
                    if target_perf_ts <= newest_ts:
                        history = list(self._state_history)
                        timestamps = [entry[0] for entry in history]
                        upper = bisect.bisect_left(timestamps, target_perf_ts)
                        if upper == 0:
                            chosen = history[0]
                        elif upper == len(history):
                            chosen = history[-1]
                        else:
                            before = history[upper - 1]
                            after = history[upper]
                            chosen = (
                                after
                                if after[0] - target_perf_ts
                                <= target_perf_ts - before[0]
                                else before
                            )
                        ts, left_pos, right_pos, left_trq, right_trq = chosen
                        return (
                            left_pos.copy(),
                            right_pos.copy(),
                            left_trq.copy(),
                            right_trq.copy(),
                            ts,
                        )
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self._state_cond.wait(remaining)

    @property
    def records_measurements_at_control_rate(self) -> bool:
        """The Mantis has no ``_meas`` recorder; only the core's own trace."""
        return False

    def set_recording_engaged(self, engaged: bool) -> None:
        """Gate the core's internal trace to the recording segment."""
        if engaged == self._recording_engaged:
            return
        self._recording_engaged = engaged
        if self._link is None or self._record_prefix is None:
            return
        try:
            self._link.set_recording_engaged(engaged)
        except Exception as exc:  # noqa: BLE001 - core may already be faulted
            _logger.warning("rt: could not gate the Mantis core trace (%s)", exc)

    # -- State / commands ---------------------------------------------------------

    async def get_positions(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Cached (core-fed) while armed; a direct proxy read between takes."""
        return await self._robot.get_positions()

    async def motion_control(
        self, left: np.ndarray | None = None, right: np.ndarray | None = None
    ) -> None:
        """Latch the virtual arms; while armed, stream the gripper targets.

        Not armed (between takes) the grippers are torque-off and the
        classic driver only latches — nothing is sent.
        """
        await self._robot.motion_control(left=left, right=right)

    async def gravity_compensate(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("The Mantis has no arm to gravity-compensate.")

    def torque_residuals(self) -> tuple[None, None]:
        return None, None

    def reset_command_state(self) -> None:
        """No command history to clear — the arms are virtual."""

    def reset_gravity_hold(self) -> None:
        """No gravity hold on a Mantis."""
