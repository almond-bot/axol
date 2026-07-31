"""MyActuator firmware update over CAN.

The RMD bootloader is absent from MyActuator's published protocol (V4.4); the
sequence here was recovered from the vendor setup software. It is a YMODEM-1K
transfer tunnelled over three CAN IDs derived from the motor's ID:

    trigger  ((id & 0x1F) << 6) | 0x28   host  -> motor, reboot into bootloader
    data     ((id & 0x1F) << 6) | 0x3E   host  -> motor, YMODEM byte stream
    reply    ((id & 0x1F) << 6) | 0x14   motor -> host, YMODEM control bytes

Note these are unrelated to the 0x140-series IDs the normal protocol uses, so a
flash in progress does not collide with regular motor traffic — but nothing
else may talk to the target motor while it is in the bootloader.

A YMODEM packet is ``header, block, ~block, payload, crc_hi, crc_lo`` with a
CRC-16/XMODEM over the payload only. It is split across CAN frames 8 bytes at a
time; DLC is the only framing, so a packet simply spans ceil(n / 8) frames.
The handshake is stock YMODEM: the bootloader sends ``C`` to open the transfer,
block 0 carries ``filename\\0size``, and every block is answered with ACK.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

import can

from .bus import CanBus
from .errors import MotorError

_logger = logging.getLogger(__name__)

# Sub-function codes OR-ed into (motor_id << 6) to form each channel's 11-bit ID.
_FW_TRIGGER_SUB = 0x28
_FW_DATA_SUB = 0x3E
_FW_REPLY_SUB = 0x14
_FW_ID_MASK = 0x1F  # the bootloader addresses motors with 5 bits of ID

# Sent on the trigger ID to make a running motor jump into its bootloader.
_FW_ENTER_BOOTLOADER = bytes([0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7])

# YMODEM control bytes.
_SOH = 0x01  # header for a 128-byte payload
_STX = 0x02  # header for a 1024-byte payload
_EOT = 0x04
_ACK = 0x06
_NAK = 0x15
_CANCEL = 0x18  # two in a row aborts the transfer
_CRC_REQ = 0x43  # "C" — receiver asks for a CRC-mode transfer
_PAD = 0x1A  # fills the tail of a short final block

_SHORT_BLOCK = 128
_LONG_BLOCK = 1024

# The vendor tool waits 7 s for every handshake byte; a flash erase before the
# first "C" is the slowest step and stays well inside that.
_FW_REPLY_TIMEOUT_S = 7.0

# Blocks rejected with NAK are retransmitted this many times before giving up.
_FW_BLOCK_RETRIES = 5


def _make_crc16_table() -> tuple[int, ...]:
    table = []
    for i in range(256):
        crc = i << 8
        for _ in range(8):
            crc = (
                ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
            )
        table.append(crc)
    return tuple(table)


_CRC16_TABLE = _make_crc16_table()


def _crc16(data: bytes) -> int:
    """CRC-16/XMODEM (poly 0x1021, zero seed) over ``data``."""
    crc = 0
    for byte in data:
        crc = ((crc << 8) ^ _CRC16_TABLE[((crc >> 8) ^ byte) & 0xFF]) & 0xFFFF
    return crc


class _TransferNak(Exception):
    """The bootloader NAK-ed a block; the caller should retransmit it."""


class FirmwareUpdater:
    """Flashes a firmware image to one MyActuator motor over CAN.

    The motor must be powered, idle, and the only thing being addressed on the
    bus for the duration — the bootloader does not answer normal 0x140-series
    commands, and a half-finished transfer leaves the motor in the bootloader
    (recoverable by re-running the flash, since the trigger frame is not needed
    once it is already there).

        async with CanBus("can_alm_axol_l") as bus:
            await FirmwareUpdater(bus, 0x01).flash(image, name="RMD-X6.bin")
    """

    def __init__(self, bus: CanBus, motor_id: int) -> None:
        """Bind an updater to one motor.

        Args:
            bus:      Shared CAN bus, already started.
            motor_id: Target motor's CAN ID. Only the low 5 bits address the
                      bootloader, so IDs above 0x1F cannot be flashed.
        """
        if not 0 < motor_id <= _FW_ID_MASK:
            raise ValueError(
                f"motor_id must be in [0x01, {_FW_ID_MASK:#04x}] to reach the "
                f"bootloader, got {motor_id:#04x}"
            )
        self._bus = bus
        self._motor_id = motor_id
        base = (motor_id & _FW_ID_MASK) << 6
        self._trigger_id = base | _FW_TRIGGER_SUB
        self._data_id = base | _FW_DATA_SUB
        self._reply_id = base | _FW_REPLY_SUB
        self._replies: asyncio.Queue[bytes] = asyncio.Queue()
        bus._add_listener(self._on_message)

    # ------------------------------------------------------------------ #
    # Reply channel                                                        #
    # ------------------------------------------------------------------ #

    def _on_message(self, msg: can.Message) -> None:
        if msg.arbitration_id == self._reply_id:
            self._replies.put_nowait(bytes(msg.data[: msg.dlc]))

    def _drain(self) -> None:
        """Discard replies buffered before the current step."""
        while not self._replies.empty():
            self._replies.get_nowait()

    async def _await_reply(self, expected: bytes, timeout: float, step: str) -> None:
        """Block until the bootloader answers with exactly ``expected``.

        Frames that are neither the expected reply nor a protocol control byte
        are ignored — a retransmitted ACK from the previous step can still be
        in flight. Raises :class:`_TransferNak` on NAK so the caller can resend.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        cancels = 0
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise MotorError(
                    f"MyActuator motor {self._motor_id:#04x}: no response to {step} "
                    f"within {timeout:g}s (expected {expected.hex()})"
                )
            try:
                reply = await asyncio.wait_for(self._replies.get(), remaining)
            except asyncio.TimeoutError:
                continue
            if reply == expected:
                return
            if reply[:1] == bytes([_NAK]):
                raise _TransferNak()
            if reply[:1] == bytes([_CANCEL]) or reply == bytes([_CANCEL, _CANCEL]):
                cancels += 2 if len(reply) > 1 else 1
                if cancels >= 2:
                    raise MotorError(
                        f"MyActuator motor {self._motor_id:#04x}: bootloader "
                        f"aborted the transfer during {step}"
                    )
                continue
            _logger.debug(
                "motor %#04x: ignoring %s during %s",
                self._motor_id,
                reply.hex(),
                step,
            )

    # ------------------------------------------------------------------ #
    # Transmit                                                             #
    # ------------------------------------------------------------------ #

    async def _send_stream(self, payload: bytes, frame_delay: float) -> None:
        """Push raw bytes down the data channel, 8 per CAN frame."""
        for off in range(0, len(payload), 8):
            await self._bus._send(self._data_id, payload[off : off + 8])
            if frame_delay:
                await asyncio.sleep(frame_delay)

    @staticmethod
    def _packet(header: int, block: int, payload: bytes) -> bytes:
        crc = _crc16(payload)
        return (
            bytes([header, block & 0xFF, ~block & 0xFF])
            + payload
            + bytes([crc >> 8, crc & 0xFF])
        )

    async def _send_block(
        self, block: int, payload: bytes, frame_delay: float, step: str
    ) -> None:
        """Send one YMODEM block and wait for its ACK, retrying on NAK."""
        header = _SOH if len(payload) == _SHORT_BLOCK else _STX
        packet = self._packet(header, block, payload)
        for attempt in range(1, _FW_BLOCK_RETRIES + 1):
            self._drain()
            await self._send_stream(packet, frame_delay)
            try:
                await self._await_reply(bytes([_ACK]), _FW_REPLY_TIMEOUT_S, step)
                return
            except _TransferNak:
                if attempt == _FW_BLOCK_RETRIES:
                    raise MotorError(
                        f"MyActuator motor {self._motor_id:#04x}: bootloader "
                        f"rejected {step} after {_FW_BLOCK_RETRIES} attempts"
                    )
                _logger.warning(
                    "motor %#04x: NAK on %s, retrying (%d/%d)",
                    self._motor_id,
                    step,
                    attempt,
                    _FW_BLOCK_RETRIES,
                )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def enter_bootloader(self) -> None:
        """Send the magic frame that reboots the motor into its bootloader.

        Unacknowledged by design — the motor resets immediately. A motor that is
        already in the bootloader ignores it.
        """
        await self._bus._send(self._trigger_id, _FW_ENTER_BOOTLOADER)

    async def flash(
        self,
        firmware: bytes,
        *,
        name: str = "firmware.bin",
        frame_delay: float = 0.0,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Write ``firmware`` to the motor and wait for the bootloader to accept it.

        Args:
            firmware:    Raw image, exactly as it would be flashed by the vendor
                         tool (a ``.bin``, not Intel hex or an archive).
            name:        Filename advertised in YMODEM block 0. The bootloader
                         keys off the size rather than the name, but the vendor
                         tool sends the real basename so this mirrors it.
            frame_delay: Seconds to pause between consecutive CAN frames. ``0``
                         matches the vendor tool; raise it if the adapter's TX
                         queue overruns on long blocks.
            on_progress: Called as ``(bytes_sent, total_bytes)`` after each
                         acknowledged block.

        Raises:
            MotorError: The bootloader did not answer, NAK-ed a block too many
                        times, or aborted the transfer.
        """
        if not firmware:
            raise ValueError("firmware image is empty")

        total = len(firmware)
        self._drain()

        await self.enter_bootloader()
        await self._await_reply(
            bytes([_CRC_REQ]), _FW_REPLY_TIMEOUT_S, "bootloader handshake"
        )

        header = name.encode("ascii", errors="replace") + b"\x00" + str(total).encode()
        if len(header) > _SHORT_BLOCK:
            raise ValueError(f"firmware name {name!r} is too long for YMODEM block 0")
        await self._send_header_block(header, frame_delay)

        sent = 0
        block = 1
        while sent < total:
            chunk = firmware[sent : sent + _LONG_BLOCK]
            # The vendor tool reads 1 KiB at a time and drops to a 128-byte
            # block only when the remainder fits, padding either with 0x1A.
            size = _SHORT_BLOCK if len(chunk) <= _SHORT_BLOCK else _LONG_BLOCK
            payload = chunk[:size].ljust(size, bytes([_PAD]))
            await self._send_block(
                block, payload, frame_delay, f"block {block} ({sent}/{total} B)"
            )
            sent += len(chunk)
            block = (block + 1) & 0xFF
            if on_progress is not None:
                on_progress(min(sent, total), total)

        self._drain()
        await self._bus._send(self._data_id, bytes([_EOT]))
        await self._await_reply(bytes([_ACK]), _FW_REPLY_TIMEOUT_S, "end of transfer")

    async def _send_header_block(self, header: bytes, frame_delay: float) -> None:
        """Send YMODEM block 0 and consume the ACK + "C" that follows it."""
        payload = header.ljust(_SHORT_BLOCK, b"\x00")
        packet = self._packet(_SOH, 0, payload)
        for attempt in range(1, _FW_BLOCK_RETRIES + 1):
            self._drain()
            await self._send_stream(packet, frame_delay)
            try:
                # Block 0 is answered with ACK and a fresh "C" in one frame.
                await self._await_reply(
                    bytes([_ACK, _CRC_REQ]), _FW_REPLY_TIMEOUT_S, "header block"
                )
                return
            except _TransferNak:
                if attempt == _FW_BLOCK_RETRIES:
                    raise MotorError(
                        f"MyActuator motor {self._motor_id:#04x}: bootloader "
                        f"rejected the header block"
                    )
