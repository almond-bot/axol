"""
Jelly's safety board — software E-stop over CAN.

Some Jelly builds carry the Almond Safety board (firmware: ``almond_safety``
in the circuits-py repo, ``designs/almond_safety/firmware``) in place of the
hand-wired E-stop relay harness. The E-stop latch stays in hardware: the
E-Stop's NC contact and the green on-switch seal in a relay (K1) that holds
the main contactor closed. The board adds a second relay (K2, the software
enable) in series with that latch, and a remote start (a PhotoMOS across the
green on-switch)::

    contactor on = E-Stop released AND K1 latched AND K2 (software enable)

- :meth:`SafetyBoard.stop` is the software E-stop. It drops K2, which opens
  the contactor at once and unlatches K1, exactly like pressing the E-Stop:
  power stays off until a start.
- :meth:`SafetyBoard.remote_start` acts like the green on-switch: the board
  re-enables K2 and pulses the PhotoMOS until the contactor closes. It is
  refused while the E-Stop is pressed, and the firmware can never close the
  contactor against a pressed E-Stop.
- The green on-switch always clears a software stop. A person at the robot
  can restart it whatever the host wants.
- K2 is held in only by a toggling GPIO, so a firmware crash, watchdog reset
  or reflash is a stop too.

The board shares the CAN bus of the jelly_legs lift controller (its own
chest bus, or the wheel bus the lift shares with the Damiao motors), so
:func:`resolve_safety_channel` follows the same wiring as
:func:`~almond_axol.robot.lift.resolve_lift_channel`. Its IDs (0x440-0x442)
are clear of the lift's (0x420-0x424) and of every Damiao range.

Protocol (classic CAN 2.0, 11-bit IDs, 1 Mbps, little-endian; the firmware
README is the spec):

- Commands go to **0x440** (byte 0 = opcode). Every command except
  ``GET_VOLTAGES`` is answered with a status frame on **0x441**; voltages come
  back on **0x442**.
- The board is silent until it hears a command, then broadcasts status every
  50 ms. The CANable adapters on Jelly's buses starve their own TX path while
  continuously receiving a stream like that (see :mod:`almond_axol.robot.lift`),
  so the driver polls with ``SET_RATE 0`` rather than ``GET_STATUS``: the board
  answers it with a status frame like any command, and a board that rebooted
  (back to its 50 ms default) is quieted again by the next poll.
- Every command also feeds the optional heartbeat timeout
  (:meth:`SafetyBoard.set_heartbeat`). When it lapses the board performs a
  software stop (``stop_cause == "heartbeat"``).
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass

from ..motor import CanBus
from .lift import resolve_lift_channel

_logger = logging.getLogger(__name__)

# Arbitration IDs (see the module docstring).
_ID_CMD = 0x440
_ID_STATUS = 0x441
_ID_VOLTAGES = 0x442

# Opcodes (command frame byte 0).
_OP_SW_STOP = 0x01
_OP_SW_ENABLE = 0x02
_OP_REMOTE_START = 0x03
_OP_GET_STATUS = 0x04
_OP_SET_RATE = 0x05
_OP_HEARTBEAT = 0x06
_OP_SET_HEARTBEAT = 0x07
_OP_GET_VOLTAGES = 0x08

# Status byte 2: why the software enable is off.
STOP_CAUSES: dict[int, str] = {
    0: "none",
    1: "can",
    2: "heartbeat",
    4: "console",
    5: "update",  # firmware update running, or a new image on trial
}

# Status poll cadence once the board has answered, and while it has not (a
# Jelly without a safety board: one unanswered frame a second is plenty).
_POLL_S = 0.2
_SEARCH_POLL_S = 1.0
# A board that answered counts as present while its last status is this fresh.
_FRESH_S = 1.0
# Re-send cadence of a command (or status request) while waiting for the
# status that confirms it.
_RESEND_S = 0.05
# A stop must be confirmed (software enable off and the contactor open)
# within this. K2 drops 10-20 ms after the pump stops; the rest is margin for
# a lost frame or two.
STOP_TIMEOUT_S = 0.5
# The remote-start pulse runs until the output has held for 200 ms (through
# the contactor's pull-in), 1 s at most; firmware before 0.3 stopped at 300 ms.
START_TIMEOUT_S = 2.0


def resolve_safety_channel(channel: str | None = None) -> str:
    """The SocketCAN interface carrying the safety board.

    The board sits on the lift controller's bus, so this resolves exactly
    like :func:`~almond_axol.robot.lift.resolve_lift_channel`: an explicit
    ``channel`` unchanged, else the chest bus when it exists, else the wheel
    bus.
    """
    return resolve_lift_channel(channel)


class SafetyError(RuntimeError):
    """A safety-board command was refused, or its result was never confirmed."""


@dataclass(frozen=True)
class SafetyStatus:
    """One decoded safety-board status frame (``0x441``).

    Field names follow the firmware README's status bits, spelled out.
    """

    estop_ok: bool  # E-Stop released and its loop fed (NC closed)
    latched: bool  # K1 sealed in
    output_on: bool  # contactor coil energized: power is on
    sw_enable: bool  # software enable on (K2 pumped)
    remote_start_active: bool  # a remote-start pulse is running
    heartbeat_on: bool  # heartbeat timeout armed
    supply_ok: bool  # 24 V rail at or above 18 V
    k2_stuck: bool  # output on with the software enable off
    k2_no_close: bool  # latched and enabled, but no output
    k1_pole_b: bool  # output on with K1 not latched
    remote_start_failed: bool  # sticky: the last remote start did not latch
    heartbeat_lost: bool  # sticky: the heartbeat timeout tripped
    stop_cause: str  # "none", "can", "heartbeat", "console", "update", or the raw code
    firmware: tuple[int, int]  # (major, minor)
    supply_volts: float
    output_volts: float
    # A firmware update is running or a new image is on trial: the board
    # keeps the software enable off until it is confirmed (firmware 0.3+).
    update_locked: bool = False
    # Sticky: the 24 V supply dipped below 18 V since the last start, which
    # can cut power by itself (firmware 0.4+).
    supply_dip: bool = False

    @property
    def estop_pressed(self) -> bool:
        """The E-Stop loop is open: pressed, unplugged, or its fuse blown."""
        return not self.estop_ok

    @property
    def software_stopped(self) -> bool:
        """A software stop is in force (cleared by a start or SW_ENABLE)."""
        return not self.sw_enable

    @property
    def relay_fault(self) -> bool:
        """A relay reads back inconsistently: the board needs service."""
        return self.k2_stuck or self.k2_no_close or self.k1_pole_b

    @property
    def firmware_version(self) -> str:
        return f"{self.firmware[0]}.{self.firmware[1]}"


def decode_status(data: bytes) -> SafetyStatus:
    """Decode one safety-board status frame (``0x441`` payload, 8 bytes)."""
    if len(data) < 8:
        raise ValueError(f"safety status frame needs 8 bytes, got {len(data)}")
    state, faults, cause, version = data[0], data[1], data[2], data[3]
    supply_mv, output_mv = struct.unpack_from("<HH", data, 4)
    return SafetyStatus(
        estop_ok=bool(state & 0x01),
        latched=bool(state & 0x02),
        output_on=bool(state & 0x04),
        sw_enable=bool(state & 0x08),
        remote_start_active=bool(state & 0x20),
        heartbeat_on=bool(state & 0x40),
        supply_ok=bool(state & 0x80),
        k2_stuck=bool(faults & 0x01),
        k2_no_close=bool(faults & 0x02),
        k1_pole_b=bool(faults & 0x04),
        remote_start_failed=bool(faults & 0x40),
        heartbeat_lost=bool(faults & 0x80),
        stop_cause=STOP_CAUSES.get(cause, str(cause)),
        firmware=(version >> 4, version & 0x0F),
        supply_volts=supply_mv / 1000.0,
        output_volts=output_mv / 1000.0,
        update_locked=bool(state & 0x10),
        supply_dip=bool(faults & 0x08),
    )


@dataclass(frozen=True)
class SafetyVoltages:
    """One decoded voltages frame (``0x442``): each chain node, in volts."""

    supply: float  # 24 V rail
    estop_ok: float  # after the E-Stop NC contact
    latch: float  # K1 coil
    output: float  # contactor coil


def decode_voltages(data: bytes) -> SafetyVoltages:
    """Decode one safety-board voltages frame (``0x442`` payload, 8 bytes)."""
    if len(data) < 8:
        raise ValueError(f"safety voltages frame needs 8 bytes, got {len(data)}")
    supply, estop_ok, latch, output = (
        v / 1000.0 for v in struct.unpack("<4H", data[:8])
    )
    return SafetyVoltages(supply, estop_ok, latch, output)


def explain_start_failure(status: SafetyStatus) -> str:
    """Why a remote start left the power off, in operator terms."""
    if status.update_locked:
        return (
            "The safety board is updating its firmware. Wait for the update to "
            "finish, then try again."
        )
    if status.estop_pressed:
        return "The E-Stop is pressed. Release it, then try again."
    if status.relay_fault:
        return "The safety board reports a relay fault. Contact Almond support."
    if not status.supply_ok:
        return (
            f"The battery reads {status.supply_volts:.1f} V, too low to switch "
            "power on. Charge Jelly, then try again."
        )
    return (
        "Power did not come on. Press the green on-switch on Jelly's power "
        "panel, or try again."
    )


class SafetyBoard:
    """Status, software stop and remote start for Jelly's safety board.

    Typical usage::

        board = SafetyBoard()
        await board.start()
        status = await board.stop()          # software E-stop, confirmed
        status = await board.remote_start()  # power back on
        await board.close()

    While started, a background task polls the board's status (5 Hz once it
    answers, 1 Hz until then) into :attr:`status`. Closing the driver sends
    nothing: the board keeps whatever state it is in, so a host that goes
    away neither stops nor starts the robot. (An armed heartbeat keeps
    running, and stops the robot when it lapses: see :meth:`set_heartbeat`.)
    """

    def __init__(self, channel: str | None = None) -> None:
        """Create a driver; ``channel`` None follows the lift's bus."""
        self._channel = resolve_safety_channel(channel)
        self._bus: CanBus | None = None
        self._task: asyncio.Task | None = None
        self._send_lock = asyncio.Lock()
        self._status: SafetyStatus | None = None
        self._status_monotonic: float | None = None
        self._status_seq = 0
        self._voltages: SafetyVoltages | None = None
        self._voltages_seq = 0
        self._frame_event = asyncio.Event()
        self._failure: BaseException | None = None

    @property
    def channel(self) -> str:
        """The SocketCAN interface this driver talks to."""
        return self._channel

    @property
    def status(self) -> SafetyStatus | None:
        """The latest status frame, or None before the board first answers."""
        return self._status

    @property
    def status_age(self) -> float | None:
        """Seconds since the latest status frame, or None before any."""
        if self._status_monotonic is None:
            return None
        return max(0.0, time.monotonic() - self._status_monotonic)

    def status_is_fresh(self, max_age_s: float = _FRESH_S) -> bool:
        """Whether a status frame arrived within ``max_age_s`` seconds."""
        age = self.status_age
        return age is not None and age <= max_age_s

    @property
    def running(self) -> bool:
        """True while the bus is open and the poll task is alive."""
        return self._task is not None and not self._task.done()

    @property
    def failure(self) -> BaseException | None:
        """Why the poll task stopped (the bus went away), if it did."""
        return self._failure

    async def start(self, *, bring_up: bool = True) -> None:
        """Open the bus and start polling.

        ``bring_up`` brings the interface up when it is down, like the lift
        driver; a missing interface raises naming it.
        """
        if bring_up:
            from ..cli.can.setup import bring_up_interfaces, iface_up

            if not iface_up(self._channel):
                bring_up_interfaces([self._channel])
        self._status = None
        self._status_monotonic = None
        self._failure = None
        # Never commands a motor, so a stall it might see is not this
        # channel's owner's to clear or report.
        self._bus = CanBus(self._channel, publishes_stall=False)
        self._bus._add_listener(self._on_message)
        try:
            await self._bus.start()
            await self._send(_OP_SET_RATE, struct.pack("<H", 0))
        except BaseException:
            bus, self._bus = self._bus, None
            await bus.close()
            raise
        self._task = asyncio.create_task(self._run(), name="safety-board-poll")
        _logger.debug("safety board driver on %s", self._channel)

    async def close(self) -> None:
        """Stop polling and close the bus. Sends nothing to the board."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        bus, self._bus = self._bus, None
        if bus is not None:
            await bus.close()

    async def stop(self, timeout: float = STOP_TIMEOUT_S) -> SafetyStatus:
        """Software E-stop: cut power now, like pressing the E-Stop.

        Re-sends ``SW_STOP`` until a status frame shows the software enable
        off and the contactor open. Raises :class:`SafetyError` when that is
        not confirmed within ``timeout`` (no answer, or a relay that did not
        release). Power stays off until :meth:`remote_start` or the green
        on-switch.
        """
        status = await self._wait_for(
            lambda s: not s.sw_enable and not s.output_on,
            op=_OP_SW_STOP,
            resend=True,
            timeout=timeout,
        )
        if status is not None:
            return status
        latest = self._status if self.status_is_fresh() else None
        if latest is not None and not latest.sw_enable:
            raise SafetyError(
                "The safety board took the stop, but power is still on. "
                "Press the E-Stop."
            )
        raise SafetyError(
            "The safety board did not confirm the stop. Press the E-Stop."
        )

    async def remote_start(self, timeout: float = START_TIMEOUT_S) -> SafetyStatus:
        """Switch power back on, like pressing the green on-switch.

        Clears a software stop and closes the latch. Returns at once when
        power is already on. Raises :class:`SafetyError` with the reason when
        the board refuses (E-Stop pressed) or power does not come on.
        """
        status = await self.request_status()
        if status.output_on:
            return status
        if status.estop_pressed or status.latched or status.update_locked:
            # The board refuses a remote start in all three cases. Latched
            # with no output is a relay fault (or the green switch held down).
            raise SafetyError(explain_start_failure(status))
        await self._send(_OP_REMOTE_START)
        status = await self._wait_for(
            lambda s: s.output_on and not s.remote_start_active,
            op=_OP_GET_STATUS,
            resend=True,
            timeout=timeout,
        )
        if status is not None:
            return status
        latest = self._status if self.status_is_fresh() else None
        if latest is None:
            raise SafetyError("The safety board stopped answering.")
        raise SafetyError(explain_start_failure(latest))

    async def enable(self) -> SafetyStatus:
        """Clear a software stop without switching power on.

        Power still needs a start (the green on-switch, or
        :meth:`remote_start`, which also clears the stop on its own).
        """
        status = await self._wait_for(
            lambda s: s.sw_enable, op=_OP_SW_ENABLE, resend=True, timeout=STOP_TIMEOUT_S
        )
        if status is None:
            raise SafetyError("The safety board did not confirm the enable.")
        return status

    async def set_heartbeat(self, timeout_ms: int) -> None:
        """Arm (or with 0, disarm) the board's host heartbeat.

        While armed, the board performs a software stop if it hears no
        command for ``timeout_ms``. The poll task keeps feeding it while
        this driver runs, so it trips when this process dies or loses the
        bus. Any host's commands feed it, and the green on-switch restarts
        the robot after a trip. It stays armed after :meth:`close`.
        """
        if not 0 <= int(timeout_ms) <= 0xFFFF:
            raise ValueError("timeout_ms must be between 0 and 65535")
        await self._send(_OP_SET_HEARTBEAT, struct.pack("<H", int(timeout_ms)))

    async def request_status(self, timeout: float = STOP_TIMEOUT_S) -> SafetyStatus:
        """A status frame received after this call; raises if none arrives."""
        status = await self._wait_for(
            lambda s: True, op=_OP_GET_STATUS, resend=True, timeout=timeout
        )
        if status is None:
            raise SafetyError(f"The safety board is not answering on {self._channel}.")
        return status

    async def read_voltages(self, timeout: float = STOP_TIMEOUT_S) -> SafetyVoltages:
        """Each chain node's voltage, for wiring checks."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        seq0 = self._voltages_seq
        while True:
            await self._send(_OP_GET_VOLTAGES)
            if await self._wait_frame(
                lambda: self._voltages_seq > seq0,
                min(_RESEND_S * 2, deadline - loop.time()),
            ):
                assert self._voltages is not None
                return self._voltages
            if loop.time() >= deadline:
                raise SafetyError(
                    f"The safety board is not answering on {self._channel}."
                )

    def _on_message(self, msg) -> None:  # noqa: ANN001 - can.Message, typed lazily
        if msg.arbitration_id == _ID_STATUS and len(msg.data) >= 8:
            self._status = decode_status(bytes(msg.data))
            self._status_monotonic = time.monotonic()
            self._status_seq += 1
            self._frame_event.set()
        elif msg.arbitration_id == _ID_VOLTAGES and len(msg.data) >= 8:
            self._voltages = decode_voltages(bytes(msg.data))
            self._voltages_seq += 1
            self._frame_event.set()

    async def _send(self, op: int, payload: bytes = b"") -> None:
        if self._bus is None:
            raise SafetyError("The safety board driver is not started.")
        async with self._send_lock:
            await self._bus._send(_ID_CMD, bytes([op]) + payload)

    async def _wait_frame(self, done: Callable[[], bool], timeout: float) -> bool:
        """Wait up to ``timeout`` for ``done()`` to hold after a received frame."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout)
        while True:
            # Clear before checking, so a frame landing in between still
            # wakes the wait below.
            self._frame_event.clear()
            if done():
                return True
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            try:
                await asyncio.wait_for(self._frame_event.wait(), remaining)
            except TimeoutError:
                return done()

    async def _wait_for(
        self,
        accept: Callable[[SafetyStatus], bool],
        *,
        op: int,
        resend: bool,
        timeout: float,
    ) -> SafetyStatus | None:
        """Send ``op`` and wait for a later status frame that ``accept`` takes.

        With ``resend`` the command goes out again every 50 ms until then, so
        one lost frame costs 50 ms rather than the whole timeout. Only frames
        received after the first send count. Returns None on timeout.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        seq0 = self._status_seq

        def accepted() -> bool:
            status = self._status
            return self._status_seq > seq0 and status is not None and accept(status)

        await self._send(op)
        while True:
            remaining = deadline - loop.time()
            if await self._wait_frame(accepted, min(_RESEND_S, remaining)):
                return self._status
            if loop.time() >= deadline:
                return None
            if resend:
                await self._send(op)

    async def _run(self) -> None:
        """Poll status (which also feeds an armed heartbeat) until cancelled."""
        try:
            while True:
                await self._send(_OP_SET_RATE, struct.pack("<H", 0))
                await asyncio.sleep(
                    _POLL_S if self.status_is_fresh(_FRESH_S * 2) else _SEARCH_POLL_S
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported through .failure
            self._failure = exc
            _logger.debug("safety board poll on %s stopped: %s", self._channel, exc)


async def read_safety_status(
    channel: str | None = None, *, timeout: float = 1.0
) -> SafetyStatus | None:
    """One-shot status read; None when no safety board answers.

    Opens the board's bus, asks for one status frame, and closes. Never
    changes the board's state (apart from quieting its broadcast).
    """
    board = SafetyBoard(channel)
    await board.start()
    try:
        return await board.request_status(timeout)
    except SafetyError:
        return None
    finally:
        await board.close()
