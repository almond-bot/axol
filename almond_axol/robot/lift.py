"""
Telescoping lift on Jelly — jelly_legs CAN driver.

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
  receiving the 50 ms status broadcast — so on connect the driver turns the
  broadcast off (``SET_RATE 0``) and polls with ``GET_STATUS`` instead.

A leg stalling mid-move sets the ``stall fault`` status flag and aborts the
move. Any new motion command clears the fault and retries, so a host must
not blindly re-send: :class:`Lift` latches the fault against the held
direction and refuses to jog again until the operator releases to STOP.
"""

from __future__ import annotations

import asyncio
import logging
import struct
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

    @property
    def height_percent(self) -> float | None:
        """Height as percent of homed travel (0 = lowered), None until homed."""
        if self.position_permille is None:
            return None
        return self.position_permille / 10.0


def _decode_status(data: bytes) -> LiftStatus:
    pos, vel, flags, drift = struct.unpack("<HhBb", data[:6])
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
    )


class Lift:
    """Hold-to-move lift commands over the chest CAN bus.

    Typical usage (from :class:`almond_axol.robot.jelly.Jelly`)::

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

    def __init__(self, channel: str = CAN_CHEST, jog_speed: int = JOG_SPEED) -> None:
        self._channel = channel
        self._jog_speed = int(jog_speed)
        self._bus: CanBus | None = None
        self._task: asyncio.Task | None = None
        self._direction = STOP
        # Last jog direction actually sent by the task; a change to STOP
        # emits one JOG 0 (crisp stop instead of the deadman coast). Motion
        # commands (home, set_position) reset it so the release transition
        # can't emit a JOG 0 that would cancel the just-started motion.
        self._last_jog_sent = STOP
        # Direction latched at a stall fault: jogging that way stays refused
        # until the operator releases to STOP (see the module docstring).
        self._stall_dir = STOP
        self._stall_logged = False
        self._status: LiftStatus | None = None

    @property
    def status(self) -> LiftStatus | None:
        """The latest status frame, or None before the board first answers."""
        return self._status

    @property
    def height_percent(self) -> float | None:
        """Height as percent of homed travel, or None (not homed / no reply)."""
        return self._status.height_percent if self._status is not None else None

    async def start(self) -> None:
        """Open the chest bus and start the jog/status task.

        Brings the interface up if it isn't yet (mirroring Jelly's wheel
        bus); a missing interface raises ``RuntimeError`` naming it.
        """
        from ..cli.can.setup import bring_up_interfaces, iface_up

        if not iface_up(self._channel):
            bring_up_interfaces([self._channel])
        self._bus = CanBus(self._channel)
        self._bus._add_listener(self._on_message)
        await self._bus.start()
        # Turn the status broadcast off and request one status: the CANable
        # adapter's TX path starves under the 50 ms broadcast, and the board
        # stays silent until it has received at least one frame.
        await self._send(_OP_SET_RATE, struct.pack("<H", 0))
        await self._send(_OP_GET_STATUS)
        self._task = asyncio.create_task(self._run(), name="lift-command")
        _logger.info("lift: jelly_legs driver on %s", self._channel)

    async def close(self) -> None:
        """Stop the task and the legs, and close the bus."""
        if self._task is not None:
            task = self._task
            self._task = None
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001 - still close the bus below
                _logger.warning("lift: command task stopped with an error: %s", exc)
        try:
            if self._bus is not None:
                try:
                    await self._send(_OP_JOG, struct.pack("<h", 0))
                except Exception:  # noqa: BLE001 - best-effort stop before close
                    pass
                await self._bus.close()
                self._bus = None
        finally:
            self._direction = STOP

    async def home(self) -> None:
        """Start the firmware's two-ended homing sequence (takes ~1-2 min).

        Both legs drive down to the bottom stop, then up to the top stop;
        on success the firmware rebases the counters, sets soft limits, and
        saves both to flash — so homing normally happens once ever, not per
        boot. Watch :attr:`status` (``homing`` while running, then ``homed``)
        for completion; any abort rolls back to the previous calibration.
        """
        self._direction = self._last_jog_sent = STOP
        await self._send(_OP_HOME)

    async def set_position(self, permille: int, vmax: int = 0) -> None:
        """Start an absolute move of both legs (requires a homed lift).

        Args:
            permille: Target height, 0 (fully lowered) to 1000 (fully raised).
            vmax:     Speed cap in encoder counts/s; 0 = full speed (~650).

        One-shot and not deadman-guarded (unlike jogging): the move runs to
        completion on its own. The firmware refuses it while unhomed — check
        ``status.homed`` first. Watch ``status.moving`` / ``status.pos_move``
        for completion and ``status.stall_fault`` for an aborted move.
        """
        self._direction = self._last_jog_sent = STOP
        permille = max(0, min(1000, int(permille)))
        await self._send(_OP_SET_POS, struct.pack("<HH", permille, int(vmax)))

    async def stop_motion(self) -> None:
        """Controlled stop of any motion, including homing and position moves."""
        self._direction = self._last_jog_sent = STOP
        await self._send(_OP_STOP)

    def command(self, direction: int) -> None:
        """Latch the commanded direction. +1 = up, 0 = stop, -1 = down.

        Safe to call from any thread at any rate (a latch, like
        ``Jelly.set_command``); the driver task consumes the latest value.
        Should the caller die mid-hold, the firmware's 300 ms jog deadman
        stops the legs on its own.
        """
        direction = int(direction)
        if direction == STOP:
            self._stall_dir = STOP  # release re-arms a stalled direction
            self._stall_logged = False
        self._direction = direction

    def _on_message(self, msg) -> None:  # noqa: ANN001 - can.Message, typed lazily
        if msg.arbitration_id == _ID_STATUS and len(msg.data) >= 6:
            self._status = _decode_status(bytes(msg.data))

    async def _send(self, op: int, payload: bytes = b"") -> None:
        assert self._bus is not None
        await self._bus._send(_ID_CMD, bytes([op]) + payload)

    async def _run(self) -> None:
        """Re-send the held jog inside the deadman and poll status."""
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

            if direction != STOP:
                await self._send(
                    _OP_JOG, struct.pack("<h", direction * self._jog_speed)
                )
            elif self._last_jog_sent != STOP:
                # Crisp stop on release instead of the 300 ms deadman coast.
                await self._send(_OP_JOG, struct.pack("<h", 0))
            self._last_jog_sent = direction

            if now >= next_poll:
                next_poll = now + _STATUS_POLL_S
                await self._send(_OP_GET_STATUS)

            if (
                not warned_silent
                and self._status is None
                and now - started > _SILENT_WARN_S
            ):
                warned_silent = True
                _logger.warning(
                    "lift: no status from the jelly_legs board on %s after "
                    "%.0fs — is the chest powered? (Polling continues.)",
                    self._channel,
                    _SILENT_WARN_S,
                )

            await asyncio.sleep(_JOG_RESEND_S)
