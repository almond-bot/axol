"""
Telescoping lift on the powered Axol Cart — jelly_legs CAN driver.

The lift legs are driven by our own PCB (firmware: ``jelly_legs`` in the
circuits-py repo, ``designs/jelly_legs/firmware``), which replaced the
Jiecang JCB35N2 control box. It sits alone on the chest CAN bus
(:data:`~almond_axol.constants.CAN_CHEST`, named by ``axol can.setup``).

Protocol (classic CAN 2.0, 11-bit IDs, 1 Mbps, little-endian; the firmware
README is the spec):

- The board listens on **0x420** (byte 0 = opcode) and answers/broadcasts
  status on **0x421**.
- Positions on the wire are permille of homed travel (0 = fully lowered,
  1000 = fully raised, 0xFFFF = not homed); speeds are encoder counts/s
  (~650 = full speed).
- ``JOG`` (opcode 0x06) is hold-to-move with a **300 ms deadman**: it must
  be re-sent while held, and a dead host stops the legs. This maps exactly
  onto :meth:`Lift.command`'s +1/0/-1 latch — the driver task re-sends the
  jog every 100 ms while a direction is held.
- The board stays silent until it has received at least one frame, and the
  CANable gs_usb adapters starve their own TX path while continuously
  receiving the 50 ms status broadcast. By default the driver therefore turns
  the broadcast off (``SET_RATE 0``) and polls with ``GET_STATUS`` instead. A
  slower receive-only broadcast can be selected for diagnostics with
  ``status_period_ms``.

A leg stalling mid-move sets the ``stall fault`` status flag and aborts the
move. Any new motion command clears the fault and retries, so a host must
not blindly re-send: :class:`Lift` latches the fault against the held
direction and refuses to jog again until the operator releases to STOP.
"""

from __future__ import annotations

import asyncio
import logging
import operator
import struct
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..constants import CAN_CHEST
from ..motor import CanBus

_logger = logging.getLogger(__name__)

UP = 1
STOP = 0
DOWN = -1

# Arbitration IDs (clear of the Damiao ranges, though the buses are separate).
_ID_CMD = 0x420
_ID_STATUS = 0x421

# Opcodes (command frame byte 0).
_OP_SET_POS = 0x01
_OP_STOP = 0x02
_OP_HOME = 0x03
_OP_GET_STATUS = 0x04
_OP_SET_RATE = 0x05
_OP_JOG = 0x06

# Default jog speed in encoder counts/s (650 ≈ the firmware's full speed).
JOG_SPEED = 650

# Jog re-send cadence; must stay well inside the firmware's 300 ms deadman.
_JOG_RESEND_S = 0.1
# Status poll cadence (broadcast is off — see the module docstring).
_STATUS_POLL_S = 0.2
# Retry receive-only mode after this many missing broadcast intervals.
_BROADCAST_STALE_FRAMES = 3
# One warning if the board never answers (chest unpowered / unplugged).
_SILENT_WARN_S = 2.0


@dataclass(frozen=True)
class LiftStatus:
    """One decoded jelly_legs status frame."""

    position_permille: int | None  # 0-1000, None until the legs are homed
    velocity: int  # counts/s, + = up
    drift: int  # leg1 - leg2, counts
    homed: bool
    moving: bool
    pos_move: bool
    stall_fault: bool
    at_lower: bool
    at_upper: bool
    homing: bool
    jog: bool
    # Firmware v0.4+ appends driver health in bytes 6-7; v0.8 defines bit 3 of
    # byte 7 as save-pending. Keep these optional so a six-byte legacy status
    # remains decodable. A v0.8 diagnostic additionally verifies save-pending
    # behavior during motion because an older eight-byte frame looks the same
    # while idle.
    driver_fault_mask: int | None = None
    drivers_enabled: bool | None = None
    vm_present: bool | None = None
    flash_interlock: bool | None = None
    save_pending: bool | None = None

    @property
    def height_percent(self) -> float | None:
        """Height as percent of homed travel (0 = lowered), None until homed."""
        if self.position_permille is None:
            return None
        return self.position_permille / 10.0


def _decode_status(data: bytes) -> LiftStatus:
    pos, vel, flags, drift = struct.unpack("<HhBb", data[:6])
    # Treat driver health as one versioned extension: a short/legacy frame must
    # not look like a healthy controller merely because its missing bits would
    # otherwise decode as zero.
    driver_fault_mask = data[6] if len(data) >= 8 else None
    driver_state = data[7] if len(data) >= 8 else None
    return LiftStatus(
        position_permille=None if pos == 0xFFFF else pos,
        velocity=vel,
        drift=drift,
        homed=bool(flags & 0x01),
        moving=bool(flags & 0x02),
        pos_move=bool(flags & 0x04),
        stall_fault=bool(flags & 0x08),
        at_lower=bool(flags & 0x10),
        at_upper=bool(flags & 0x20),
        homing=bool(flags & 0x40),
        jog=bool(flags & 0x80),
        driver_fault_mask=driver_fault_mask,
        drivers_enabled=(
            bool(driver_state & 0x01) if driver_state is not None else None
        ),
        vm_present=(bool(driver_state & 0x02) if driver_state is not None else None),
        flash_interlock=(
            bool(driver_state & 0x04) if driver_state is not None else None
        ),
        save_pending=(bool(driver_state & 0x08) if driver_state is not None else None),
    )


class Lift:
    """Hold-to-move lift commands over the chest CAN bus.

    Typical usage (from :class:`almond_axol.robot.cart.Cart`)::

        lift = Lift()
        await lift.start()
        lift.command(UP)      # from any thread, at any rate
        ...
        await lift.close()

    :meth:`command` only latches the direction; the internal task owns all
    bus traffic — jog re-sends inside the firmware deadman while a direction
    is held, an immediate stop on release, and the status poll feeding
    :attr:`status` / :attr:`height_percent`.
    """

    def __init__(
        self,
        channel: str = CAN_CHEST,
        jog_speed: int = JOG_SPEED,
        status_period_ms: int = 0,
    ) -> None:
        """Create a lift driver.

        Args:
            channel: SocketCAN interface carrying the jelly_legs controller.
            jog_speed: Held-jog speed in encoder counts/s.
            status_period_ms: Firmware status broadcast period in milliseconds.
                Zero (the default) disables broadcasts and explicitly polls at
                5 Hz, preserving the interactive/cart behavior. Diagnostics
                may use 200 ms to receive status without transmitting while
                the motors are moving.
        """
        self._channel = channel
        self._jog_speed = self._validate_jog_speed(jog_speed)
        self._status_period_ms = self._validate_status_period(status_period_ms)
        self._bus: CanBus | None = None
        self._task: asyncio.Task | None = None
        self._send_lock = asyncio.Lock()
        self._direction = STOP
        # STOP is the one canonical firmware abort: unlike JOG 0 it also
        # cancels HOME and SET_POS.  ``command`` is synchronous, so it queues
        # this flag for the bus-owning task when a held control is released.
        self._stop_requested = False
        self._one_shot_active = False
        # Last jog direction actually sent by the task; a release queues the
        # canonical STOP opcode (which cancels jog, HOME, and SET_POS alike).
        self._last_jog_sent = STOP
        # Direction latched at a stall fault: jogging that way stays refused
        # until the operator releases to STOP (see the module docstring).
        self._stall_dir = STOP
        self._stall_logged = False
        self._status: LiftStatus | None = None
        self._last_status_monotonic: float | None = None
        self._status_timestamps: deque[float] = deque(maxlen=64)
        # A diagnostic can temporarily suppress solicited recovery traffic
        # while proving that firmware broadcasts really arrive on their own.
        self._recover_stale_broadcasts = True

    @staticmethod
    def _validate_jog_speed(speed: int) -> int:
        try:
            value = operator.index(speed)
        except TypeError:
            raise ValueError(
                "jog_speed must be an integer between 0 and 32767"
            ) from None
        if isinstance(speed, bool) or not 0 <= value <= 0x7FFF:
            raise ValueError("jog_speed must be between 0 and 32767")
        return int(value)

    @staticmethod
    def _validate_status_period(period_ms: int) -> int:
        period_ms = int(period_ms)
        if not 0 <= period_ms <= 0xFFFF:
            raise ValueError("status_period_ms must be between 0 and 65535")
        return period_ms

    @property
    def status(self) -> LiftStatus | None:
        """The latest status frame, or None before the board first answers."""
        return self._status

    @property
    def last_status_monotonic(self) -> float | None:
        """Monotonic receive time of the latest status frame, if any."""
        return self._last_status_monotonic

    @property
    def status_timestamps(self) -> tuple[float, ...]:
        """Recent status receive times, oldest first, for cadence validation."""
        return tuple(self._status_timestamps)

    @property
    def status_age(self) -> float | None:
        """Seconds since the latest status frame, or None before any reply."""
        if self._last_status_monotonic is None:
            return None
        return max(0.0, time.monotonic() - self._last_status_monotonic)

    def status_is_fresh(self, max_age_s: float) -> bool:
        """Whether a status frame arrived within ``max_age_s`` seconds."""
        if max_age_s < 0:
            raise ValueError("max_age_s must be non-negative")
        age = self.status_age
        return age is not None and age <= max_age_s

    @property
    def status_period_ms(self) -> int:
        """Configured firmware status period; zero means explicit polling."""
        return self._status_period_ms

    @property
    def height_percent(self) -> float | None:
        """Height as percent of homed travel, or None (not homed / no reply)."""
        return self._status.height_percent if self._status is not None else None

    async def start(self, *, request_status: bool = True) -> None:
        """Open the chest bus and start the jog/status task.

        Brings the interface up if it isn't yet (mirroring the cart's wheel
        bus); a missing interface raises ``RuntimeError`` naming it. Set
        ``request_status=False`` only when a configured periodic stream will
        establish readiness without a solicited bootstrap response.
        """
        from ..cli.can.setup import bring_up_interfaces, iface_up

        # A Lift instance may be restarted; never let a previous connection's
        # status satisfy a new connection's readiness/freshness checks.
        self._status = None
        self._last_status_monotonic = None
        self._status_timestamps.clear()
        self._direction = STOP
        self._last_jog_sent = STOP
        self._stop_requested = False
        self._one_shot_active = False
        if not iface_up(self._channel):
            bring_up_interfaces([self._channel])
        self._bus = CanBus(self._channel)
        try:
            self._bus._add_listener(self._on_message)
            await self._bus.start()
            # Quiesce the firmware's default 50 ms broadcast before doing any
            # request/response traffic: the CANable adapter's TX path can starve
            # while that stream is arriving. Request one immediate status, then
            # opt into the caller's slower receive-only broadcast when configured.
            await self._send_required(_OP_SET_RATE, struct.pack("<H", 0))
            if request_status:
                await self._send(_OP_GET_STATUS)
            if self._status_period_ms:
                await self._send_required(
                    _OP_SET_RATE, struct.pack("<H", self._status_period_ms)
                )
        except BaseException as start_error:
            # No motion opcode has been issued yet, so closing this partially
            # opened transport is safe and prevents a failed setup from leaking
            # the SocketCAN reader into a long-lived serve process.
            try:
                await self._bus.close()
            except BaseException as close_error:
                start_error.add_note(
                    "partial lift startup also failed to close its CAN bus: "
                    f"{type(close_error).__name__}: {close_error}"
                )
            else:
                self._bus = None
            raise
        self._task = asyncio.create_task(self._run(), name="lift-command")
        _logger.info("lift: jelly_legs driver on %s", self._channel)

    async def close(self) -> None:
        """Stop all motion, quiet broadcasts, and close the bus.

        Safety-critical cleanup errors are never swallowed.  If STOP or
        SET_RATE fails, the bus remains attached so a caller can retry
        ``close()``; a failed bus close is likewise retryable.
        """
        task_error: BaseException | None = None
        external_cancel: asyncio.CancelledError | None = None
        if self._task is not None:
            self._task.cancel()
            try:
                (result,) = await asyncio.gather(
                    self._task,
                    return_exceptions=True,
                )
            except asyncio.CancelledError as exc:
                # Expected child cancellation is returned by gather. Reaching
                # this branch means the caller canceled close() itself.
                external_cancel = exc
                (result,) = await asyncio.gather(
                    self._task,
                    return_exceptions=True,
                )
                if isinstance(result, BaseException) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    task_error = result
            else:
                if isinstance(result, BaseException) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    task_error = result
            self._task = None
        if self._bus is not None:
            cleanup_errors: list[BaseException] = []
            try:
                await self.stop_motion()
            except BaseException as exc:  # keep the live bus for a retry
                if isinstance(exc, asyncio.CancelledError):
                    external_cancel = external_cancel or exc
                cleanup_errors.append(exc)
            try:
                # A diagnostic may have enabled receive-only broadcasts. Leave
                # the shared CAN bus quiet for the next owner even if stopping
                # the legs above failed.
                await self._send_required(_OP_SET_RATE, struct.pack("<H", 0))
            except BaseException as exc:  # keep the live bus for a retry
                if isinstance(exc, asyncio.CancelledError):
                    external_cancel = external_cancel or exc
                cleanup_errors.append(exc)
            if cleanup_errors:
                error: BaseException = external_cancel or cleanup_errors[0]
                for extra in cleanup_errors:
                    if extra is error:
                        continue
                    error.add_note(
                        f"additional lift cleanup failure: "
                        f"{type(extra).__name__}: {extra}"
                    )
                if task_error is not None:
                    error.add_note(
                        f"lift command task also failed: "
                        f"{type(task_error).__name__}: {task_error}"
                    )
                raise error
            try:
                await self._bus.close()
            except BaseException as close_error:
                if isinstance(close_error, asyncio.CancelledError):
                    external_cancel = external_cancel or close_error
                if external_cancel is not None and close_error is not external_cancel:
                    external_cancel.add_note(
                        "lift CAN close also failed: "
                        f"{type(close_error).__name__}: {close_error}"
                    )
                if task_error is not None:
                    (external_cancel or close_error).add_note(
                        f"lift command task also failed: "
                        f"{type(task_error).__name__}: {task_error}"
                    )
                raise external_cancel or close_error
            self._bus = None
        self._direction = STOP
        self._stop_requested = False
        self._one_shot_active = False
        if external_cancel is not None:
            if task_error is not None:
                external_cancel.add_note(
                    "lift command task also failed: "
                    f"{type(task_error).__name__}: {task_error}"
                )
            raise external_cancel
        if task_error is not None:
            raise task_error

    async def home(
        self,
        *,
        before_send: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Start the firmware's two-ended homing sequence (takes ~1-2 min).

        Both legs drive down to the bottom stop, then up to the top stop;
        on success the firmware rebases the counters, sets soft limits, and
        saves both to flash — so homing normally happens once ever, not per
        boot. Watch :attr:`status` (``homing`` while running, then ``homed``)
        for completion; any abort rolls back to the previous calibration.
        """
        # Never let HOME implicitly supersede a jog or another one-shot move.
        await self.stop_motion()
        if before_send is not None:
            await before_send()
        self._one_shot_active = True
        try:
            await self._send_required(_OP_HOME)
        except BaseException:
            self._one_shot_active = False
            raise

    async def set_position(
        self,
        permille: int,
        vmax: int = 0,
        *,
        before_send: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Start an absolute move of both legs (requires a homed lift).

        Args:
            permille: Target height, 0 (fully lowered) to 1000 (fully raised).
            vmax:     Speed cap in encoder counts/s; 0 = full speed (~650).

        One-shot and not deadman-guarded (unlike jogging): the move runs to
        completion on its own. The firmware refuses it while unhomed — check
        ``status.homed`` first. Watch ``status.moving`` / ``status.pos_move``
        for completion and ``status.stall_fault`` for an aborted move.
        """
        try:
            permille_value = operator.index(permille)
        except TypeError:
            raise ValueError("permille must be an integer between 0 and 1000") from None
        try:
            vmax_value = operator.index(vmax)
        except TypeError:
            raise ValueError("vmax must be an integer between 0 and 65535") from None
        if isinstance(permille, bool):
            raise ValueError("permille must be an integer between 0 and 1000")
        if isinstance(vmax, bool):
            raise ValueError("vmax must be an integer between 0 and 65535")
        permille = int(permille_value)
        vmax = int(vmax_value)
        if not 0 <= permille <= 1000:
            raise ValueError("permille must be between 0 and 1000")
        if not 0 <= vmax <= 0xFFFF:
            raise ValueError("vmax must be between 0 and 65535")
        # Validate first, then canonically cancel any prior firmware mode.
        # Invalid input must never mutate the live command latch.
        await self.stop_motion()
        # STOP is an awaited wire operation. A caller with a safety interlock
        # must be able to re-check it after that gap and immediately before the
        # motion opcode is issued.
        if before_send is not None:
            await before_send()
        self._one_shot_active = True
        try:
            await self._send_required(_OP_SET_POS, struct.pack("<HH", permille, vmax))
        except BaseException:
            self._one_shot_active = False
            raise

    async def stop_motion(self) -> None:
        """Controlled stop of any motion, including homing and position moves."""
        self._direction = self._last_jog_sent = STOP
        self._stop_requested = False
        try:
            await self._send_required(_OP_STOP)
        except BaseException:
            # Surface the failed direct request, but leave an attached driver's
            # background task armed to retry the canonical STOP.
            self._stop_requested = True
            raise
        self._one_shot_active = False

    async def set_status_period(
        self,
        period_ms: int,
        *,
        recover_stale: bool = True,
    ) -> None:
        """Select firmware broadcasts, or zero to return to explicit polling.

        The setting is retained if this instance is closed and restarted.
        When connected, it is applied immediately; otherwise it takes effect
        on the next :meth:`start`.
        """
        self._status_period_ms = self._validate_status_period(period_ms)
        self._recover_stale_broadcasts = bool(recover_stale)
        if self._bus is not None:
            await self._send_required(
                _OP_SET_RATE, struct.pack("<H", self._status_period_ms)
            )

    def enable_broadcast_recovery(self) -> None:
        """Resume SET_RATE/GET_STATUS recovery after broadcast-only proof."""
        self._recover_stale_broadcasts = True

    def command(self, direction: int) -> None:
        """Latch the commanded direction. +1 = up, 0 = stop, -1 = down.

        Safe to call from any thread at any rate (a latch, like
        ``Cart.set_command``); the driver task consumes the latest value.
        Should the caller die mid-hold, the firmware's 300 ms jog deadman
        stops the legs on its own.
        """
        try:
            value = operator.index(direction)
        except TypeError:
            raise ValueError("direction must be one of DOWN, STOP, or UP") from None
        if isinstance(direction, bool) or value not in (DOWN, STOP, UP):
            raise ValueError("direction must be one of DOWN, STOP, or UP")
        direction = int(value)
        if direction == STOP:
            self._stall_dir = STOP  # release re-arms a stalled direction
            self._stall_logged = False
            if (
                self._direction != STOP
                or self._last_jog_sent != STOP
                or self._one_shot_active
            ):
                self._stop_requested = True
        elif self._one_shot_active:
            # Never let a jog implicitly supersede HOME/SET_POS.  Queue the
            # canonical abort first; the jog begins on the following tick.
            self._stop_requested = True
        self._direction = direction

    def _on_message(self, msg) -> None:  # noqa: ANN001 - can.Message, typed lazily
        if msg.arbitration_id == _ID_STATUS and len(msg.data) >= 6:
            self._status = _decode_status(bytes(msg.data))
            received_at = time.monotonic()
            self._last_status_monotonic = received_at
            self._status_timestamps.append(received_at)

    async def _send(self, op: int, payload: bytes = b"") -> bool:
        assert self._bus is not None
        async with self._send_lock:
            return await self._bus._send(_ID_CMD, bytes([op]) + payload)

    async def _send_required(self, op: int, payload: bytes = b"") -> None:
        """Send a one-shot command and fail if CanBus deliberately dropped it."""
        delivered = await self._send(op, payload) if payload else await self._send(op)
        if delivered is False:
            raise OSError(
                f"lift CAN command 0x{op:02x} was not delivered on "
                f"{self._channel} (interface lost, stalled, or unavailable)"
            )

    async def _run(self) -> None:
        """Re-send jogs, poll in quiet mode, and recover stale broadcasts."""
        next_poll = 0.0
        started = asyncio.get_running_loop().time()
        warned_silent = False
        prev_stall = False
        while True:
            now = asyncio.get_running_loop().time()

            direction = self._direction
            # A stall aborts the move firmware-side; re-sending the jog would
            # clear the fault and grind at the obstruction, so hold off until
            # the operator releases (command(STOP) re-arms). Latched on the
            # flag's rising edge: the firmware keeps the flag up until its
            # next accepted motion command, so a deliberate release-and-retry
            # still sees it set and must be allowed through (the retry's jog
            # is what clears it).
            status = self._status
            stall = status is not None and status.stall_fault
            if direction != STOP and stall and not prev_stall:
                self._stall_dir = direction
            prev_stall = stall
            if direction != STOP and direction == self._stall_dir:
                if not self._stall_logged:
                    self._stall_logged = True
                    _logger.warning(
                        "lift: leg stall — jog refused until the control is released"
                    )
                direction = STOP

            if self._stop_requested:
                delivered = await self._send(_OP_STOP)
                if delivered is not False:
                    self._stop_requested = False
                    self._one_shot_active = False
                    self._last_jog_sent = STOP
                direction = STOP
            elif direction != STOP:
                await self._send(
                    _OP_JOG, struct.pack("<h", direction * self._jog_speed)
                )
            elif self._last_jog_sent != STOP:
                # Defensive fallback for a direction changed outside command().
                # STOP is canonical because it cancels every firmware mode.
                delivered = await self._send(_OP_STOP)
                if delivered is False:
                    self._stop_requested = True
            self._last_jog_sent = direction

            broadcast_stale = False
            if self._status_period_ms:
                age = self.status_age
                broadcast_stale = age is None or age > max(
                    _STATUS_POLL_S * 2,
                    _BROADCAST_STALE_FRAMES * self._status_period_ms / 1000,
                )
            should_poll = self._status_period_ms == 0 or (
                broadcast_stale and self._recover_stale_broadcasts
            )
            if should_poll and now >= next_poll:
                next_poll = now + _STATUS_POLL_S
                if broadcast_stale:
                    # CanBus deliberately drops sends while its interface is
                    # lost/stalled. If SET_RATE was one of those drops, do not
                    # stop monitoring forever: retry the requested rate and
                    # solicit one status until periodic frames resume.
                    await self._send(
                        _OP_SET_RATE, struct.pack("<H", self._status_period_ms)
                    )
                await self._send(_OP_GET_STATUS)

            if (
                not warned_silent
                and self._status is None
                and now - started > _SILENT_WARN_S
            ):
                warned_silent = True
                _logger.warning(
                    "lift: no status from the jelly_legs board on %s after "
                    "%.0fs — is the chest powered? (Monitoring continues.)",
                    self._channel,
                    _SILENT_WARN_S,
                )

            await asyncio.sleep(_JOG_RESEND_S)
