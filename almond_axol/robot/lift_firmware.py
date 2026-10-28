"""
Jelly Legs lift firmware update over CAN.

The jelly_legs board (firmware v0.9+, installed once over USB with its A/B
partition table) can replace its own firmware over CAN; the firmware README
in the circuits-py repo (``designs/jelly_legs/firmware``, "Firmware update
over CAN") is the spec, and its ``legs_update.py`` the reference host tool.
This module is that tool on top of :class:`~almond_axol.motor.CanBus`.

The flow is fail-safe by construction:

- The image streams into the partition the board is *not* running from, in
  256-byte chunks (a header frame plus 37 seven-byte data frames on
  **0x424**, each chunk acked on **0x423**). Its first flash sector is held
  back until the whole image has arrived, its CRC32 matches, and it carries
  the Jelly Legs identity block — so a half-sent or foreign image can never
  boot.
- The board then reboots into the new image *on trial* with motion locked.
  :meth:`LiftFirmwareUpdater.flash` confirms it once it answers with the
  expected build ID. An image that is never confirmed (crashes, hangs, loses
  power, or this host disappears) reverts to the previous one within 15 s.
- The saved lift position and homing live outside both partitions and
  survive the update.

The Almond Safety board on the same bus runs the same protocol on its own
IDs (0x440 / 0x443 / 0x444). Everything that differs between the two
boards is a :class:`BoardSpec`; :mod:`almond_axol.robot.safety_firmware`
holds the safety board's.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import struct
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..motor import CanBus

_OP_SET_RATE = 0x05
_OP_FW_INFO = 0x09
_OP_BEGIN = 0x0A
_OP_FINISH = 0x0B
_OP_ABORT = 0x0C
_OP_CONFIRM = 0x0D
_REPLY_CHUNK = 0x10

CHUNK_BYTES = 256
_FRAME_BYTES = 7
_HEADER_INDEX = 0xFF

STATE_TRIAL = 0x01
STATE_PARTITIONED = 0x02
STATE_PARTITION_B = 0x04
STATE_UPDATING = 0x08

_STATUS_OK = 0
_STATUS_BUSY = 1
_STATUS_FLASH = 6

_IDENT_SIZE = 40  # fw_ident_t in firmware.c (16-byte magic, version, date)


@dataclass(frozen=True)
class BoardSpec:
    """What differs between boards that speak the CAN firmware-update protocol."""

    name: str  # file name prefix for stored uploads, e.g. "jelly_legs"
    board: str  # in messages: "the lift board"
    file_kind: str  # in messages: "Jelly lift firmware file"
    ident_magic: bytes  # the firmware identity block's 16-byte magic
    min_version: str  # first firmware with the updater
    id_cmd: int
    id_reply: int
    id_data: int
    busy: str  # what status 1 (busy) means on this board
    trial_lock: str  # what is locked while a new image is on trial

    def status_name(self, code: int) -> str:
        names = (
            "ok",
            self.busy,
            "no partition table",
            "bad size",
            "bad state",
            "chunk out of sequence",
            "flash verify failed",
            "image CRC mismatch",
            f"not a {self.file_kind}",
            "no image on trial",
            "bootrom explicit_buy failed",
        )
        return names[code] if code < len(names) else f"status {code}"


LEGS = BoardSpec(
    name="jelly_legs",
    board="the lift board",
    file_kind="Jelly lift firmware file",
    ident_magic=b"jelly_legs-fwid\0",
    min_version="0.9",
    id_cmd=0x420,
    id_reply=0x423,
    id_data=0x424,
    busy="busy (moving, homing, or a position save pending - which needs 24 V)",
    trial_lock="motion locked",
)

IDENT_MAGIC = LEGS.ident_magic
# One A/B partition (partition_table.json): the largest image that fits.
MAX_IMAGE_BYTES = 1016 * 1024

_CHUNK_ACK_TIMEOUT_S = 1.0  # covers a sector erase on the first page
_CHUNK_RETRIES = 20
_FINISH_TIMEOUT_S = 3.0  # CRC over the image + first-sector write
_REBOOT_TIMEOUT_S = 10.0
_BUSY_WAIT_S = 10.0


# Where the control panel keeps uploaded images for `lift.update --firmware`.
UPLOAD_DIR = Path.home() / ".almond" / "lift-firmware"
_UPLOADS_KEPT = 5


class FirmwareUpdateError(Exception):
    """The update could not be completed; the message says why and what next."""


def status_name(code: int) -> str:
    return LEGS.status_name(code)


@dataclass(frozen=True)
class FirmwareImage:
    """A board's ``firmware.bin`` and the identity block it carries."""

    data: bytes
    version: str
    built: str
    build_id: int  # CRC32 of the identity block, as FW_INFO reports it
    crc: int  # CRC32 of the whole image, checked by UPDATE_FINISH
    spec: BoardSpec = LEGS

    @classmethod
    def parse(cls, data: bytes, spec: BoardSpec = LEGS) -> FirmwareImage:
        """Validate a candidate image; raises :class:`FirmwareUpdateError`.

        Only an image carrying ``spec``'s identity block passes, so one
        board's firmware is never sent to another.
        """
        if not data:
            raise FirmwareUpdateError("the firmware file is empty")
        if len(data) > MAX_IMAGE_BYTES:
            raise FirmwareUpdateError(
                f"the firmware file is {len(data)} bytes; a firmware "
                f"partition holds at most {MAX_IMAGE_BYTES}"
            )
        if data.startswith(b"UF2\n") or data[:4] == b"\x7fELF":
            raise FirmwareUpdateError(
                "this is a .uf2/.elf file; upload the build's firmware.bin"
            )
        magic = spec.ident_magic
        for off in range(0, len(data) - _IDENT_SIZE + 1, 4):
            if data[off : off + len(magic)] == magic:
                ident = data[off : off + _IDENT_SIZE]
                return cls(
                    data=data,
                    version=f"{ident[16]}.{ident[17]}",
                    built=ident[18:].rstrip(b"\0").decode(errors="replace"),
                    build_id=zlib.crc32(ident),
                    crc=zlib.crc32(data),
                    spec=spec,
                )
        raise FirmwareUpdateError(
            f"This is not a {spec.file_kind}. Upload the board's firmware.bin."
        )

    def describe(self) -> str:
        return (
            f"firmware {self.version} ({self.built}), build "
            f"0x{self.build_id:08x}, {len(self.data)} bytes"
        )


def store_upload(image: FirmwareImage, directory: Path | None = None) -> Path:
    """Save a validated upload under :data:`UPLOAD_DIR`; returns its path.

    Named by content hash, so re-uploading a build reuses its file. Only the
    newest few uploads are kept.
    """
    directory = UPLOAD_DIR if directory is None else directory
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(image.data).hexdigest()[:16]
    prefix = image.spec.name
    path = directory / f"{prefix}-{image.version}-{digest}.bin"
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(image.data)
    os.replace(tmp, path)  # also bumps the mtime of a re-uploaded build
    uploads = sorted(
        directory.glob(f"{prefix}-*.bin"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in uploads[_UPLOADS_KEPT:]:
        old.unlink(missing_ok=True)
    return path


@dataclass(frozen=True)
class FirmwareInfo:
    """One FW_INFO reply: what the board is running and its update state."""

    state: int
    version: str
    build_id: int

    @property
    def partitioned(self) -> bool:
        return bool(self.state & STATE_PARTITIONED)

    @property
    def trial(self) -> bool:
        return bool(self.state & STATE_TRIAL)

    @property
    def updating(self) -> bool:
        return bool(self.state & STATE_UPDATING)

    def describe(self) -> str:
        if self.partitioned:
            where = "partition B" if self.state & STATE_PARTITION_B else "partition A"
        else:
            where = "no partition table (not updatable over CAN)"
        flags = []
        if self.trial:
            flags.append("ON TRIAL (unconfirmed)")
        if self.updating:
            flags.append("update session active")
        extra = ", " + ", ".join(flags) if flags else ""
        return f"firmware {self.version}, build 0x{self.build_id:08x}, {where}{extra}"


class BoardFirmwareUpdater:
    """Request/response firmware-update client on an open :class:`CanBus`.

    Subclasses set :attr:`spec`. Construct one *before* the bus' ``start()``
    (the listener sees every frame on the bus and keeps only this board's
    update replies)::

        bus = CanBus(channel)
        updater = LiftFirmwareUpdater(bus)
        await bus.start()
        await updater.flash(FirmwareImage.parse(data))
    """

    spec: BoardSpec = LEGS

    def __init__(self, bus: CanBus) -> None:
        self._bus = bus
        self._replies: asyncio.Queue[bytes] = asyncio.Queue()
        bus._add_listener(self._on_message)

    def _on_message(self, msg) -> None:  # noqa: ANN001 - can.Message
        if msg.arbitration_id == self.spec.id_reply and len(msg.data) >= 6:
            self._replies.put_nowait(bytes(msg.data).ljust(8, b"\x00"))

    async def _before_begin(self, log: Callable[[str], None]) -> None:
        """Get the board ready to take an update (subclass hook)."""

    async def _while_busy(self) -> None:
        """Called between UPDATE_BEGIN attempts the board refused as busy."""

    async def _send(self, can_id: int, data: bytes) -> None:
        if await self._bus._send(can_id, data) is False:
            raise FirmwareUpdateError(
                f"CAN frame 0x{can_id:03x} was not delivered (interface lost)"
            )

    async def _cmd(self, op: int, payload: bytes = b"") -> None:
        await self._send(self.spec.id_cmd, bytes([op]) + payload)

    def _drain(self) -> None:
        while not self._replies.empty():
            self._replies.get_nowait()

    async def _wait_reply(
        self,
        op: int,
        timeout: float,
        match: Callable[[bytes], bool] | None = None,
    ) -> bytes | None:
        """The next update reply of type ``op`` within ``timeout``, else None."""
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            try:
                data = await asyncio.wait_for(self._replies.get(), left)
            except TimeoutError:
                return None
            if data[0] == op and (match is None or match(data)):
                return data

    async def _request(
        self, op: int, payload: bytes = b"", timeout: float = 0.5, retries: int = 3
    ) -> bytes | None:
        for _ in range(retries):
            self._drain()
            await self._cmd(op, payload)
            reply = await self._wait_reply(op, timeout)
            if reply is not None:
                return reply
        return None

    async def quiet_status(self) -> None:
        """Turn the status broadcast off (CANable TX starvation, see Lift)."""
        await self._cmd(_OP_SET_RATE, struct.pack("<H", 0))

    async def info(self, timeout: float = 0.3, retries: int = 3) -> FirmwareInfo | None:
        """FW_INFO, or None (board offline, or firmware without the updater)."""
        r = await self._request(_OP_FW_INFO, timeout=timeout, retries=retries)
        if r is None:
            return None
        return FirmwareInfo(
            state=r[1],
            version=f"{r[2]}.{r[3]}",
            build_id=struct.unpack_from("<I", r, 4)[0],
        )

    async def abort(self) -> bool:
        """End an update session; True when the board acknowledged it."""
        return await self._request(_OP_ABORT) is not None

    async def confirm(
        self,
        expected_build: int | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        """Confirm the image on trial so it stays installed."""
        r = await self._request(_OP_CONFIRM, timeout=2.0, retries=1)
        if r is None:
            # The reply may be what got lost; ask what the board is running.
            info = await self.info()
            if (
                info is not None
                and not info.trial
                and expected_build in (None, info.build_id)
            ):
                return
            raise FirmwareUpdateError(
                "no reply to UPDATE_CONFIRM, and the board is not confirmed"
            )
        if r[1] == _STATUS_FLASH:
            # Confirmed, but the old image is still in flash. The new one stays
            # installed: if the old image ever boots first it hands over.
            log(
                "warning: confirmed, but the previous firmware image could not "
                "be erased (flash problem?); the board retries on every boot"
            )
            return
        if r[1] != _STATUS_OK:
            raise FirmwareUpdateError(f"confirm failed: {self.spec.status_name(r[1])}")

    async def _send_chunk(self, offset: int, chunk: bytes) -> None:
        header = struct.pack("<BIH", _HEADER_INDEX, offset, len(chunk))

        def is_ack(d: bytes) -> bool:
            return struct.unpack_from("<I", d, 2)[0] == offset

        for _ in range(_CHUNK_RETRIES):
            self._drain()
            await self._send(self.spec.id_data, header)
            for i in range(0, len(chunk), _FRAME_BYTES):
                await self._send(
                    self.spec.id_data,
                    bytes([i // _FRAME_BYTES]) + chunk[i : i + _FRAME_BYTES],
                )
            r = await self._wait_reply(_REPLY_CHUNK, _CHUNK_ACK_TIMEOUT_S, is_ack)
            if r is None:
                continue  # lost frame or ack: re-send the whole chunk
            if r[1] != _STATUS_OK:
                raise FirmwareUpdateError(
                    f"chunk at {offset}: {self.spec.status_name(r[1])}"
                )
            return
        raise FirmwareUpdateError(
            f"chunk at {offset}: no ack after {_CHUNK_RETRIES} tries"
        )

    async def _wait_for_reboot(self) -> FirmwareInfo:
        await asyncio.sleep(0.5)
        deadline = time.monotonic() + _REBOOT_TIMEOUT_S
        while time.monotonic() < deadline:
            # The rebooted firmware starts its 50 ms broadcast once it hears
            # a host frame; keep it off.
            await self.quiet_status()
            info = await self.info(timeout=0.2, retries=1)
            if info is not None and not info.updating:
                return info
        raise FirmwareUpdateError(
            "the board did not answer after rebooting; if the new image is "
            "broken it reverts to the previous one within 15 s"
        )

    async def flash(
        self,
        image: FirmwareImage,
        *,
        force: bool = False,
        confirm: bool = True,
        log: Callable[[str], None] = print,
        progress: Callable[[int, int], None] | None = None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> FirmwareInfo | None:
        """Install ``image``; returns the board's final FW_INFO.

        Returns None without touching the board when it already runs this
        build (unless ``force``). ``cancelled`` is polled between chunks; when
        it turns true the session is aborted on the board (nothing was
        installed) and :class:`asyncio.CancelledError` is raised.
        """
        spec = self.spec
        if image.spec != spec:
            # Never stream one board's firmware at another on the shared bus.
            raise FirmwareUpdateError(
                f"this is a {image.spec.file_kind}, not a {spec.file_kind}"
            )
        await self.quiet_status()
        info = await self.info()
        if info is None:
            raise FirmwareUpdateError(
                f"no FW_INFO reply: {spec.board} is offline, or runs firmware "
                f"older than {spec.min_version} (install {spec.min_version} over "
                "USB once first)"
            )
        log(f"board: {info.describe()}")
        if not info.partitioned:
            raise FirmwareUpdateError(
                "the board was not booted from the A/B partition table; do the "
                "one-time USB install of the partition table first"
            )
        if info.trial:
            raise FirmwareUpdateError(
                "the board is running an unconfirmed image; wait 15 s for it "
                "to revert (or power-cycle it), then retry"
            )
        if info.build_id == image.build_id and not force:
            log("the board already runs this build - nothing to do")
            return None

        await self._before_begin(log)
        deadline = time.monotonic() + _BUSY_WAIT_S
        while True:
            r = await self._request(_OP_BEGIN, struct.pack("<I", len(image.data)), 1.0)
            if r is None:
                raise FirmwareUpdateError("no reply to UPDATE_BEGIN")
            if r[1] == _STATUS_BUSY and time.monotonic() < deadline:
                await asyncio.sleep(0.5)  # wait out a move or a position save
                await self._while_busy()
                continue
            if r[1] != _STATUS_OK:
                raise FirmwareUpdateError(f"update refused: {spec.status_name(r[1])}")
            break
        log(f"target partition: {struct.unpack_from('<I', r, 2)[0] // 1024} KB")

        size = len(image.data)
        started = time.monotonic()
        try:
            for offset in range(0, size, CHUNK_BYTES):
                if cancelled():
                    raise asyncio.CancelledError
                await self._send_chunk(
                    offset, image.data[offset : offset + CHUNK_BYTES]
                )
                if progress is not None:
                    progress(min(offset + CHUNK_BYTES, size), size)
            log(f"sent {size} bytes in {time.monotonic() - started:.1f} s")
            r = await self._request(
                _OP_FINISH,
                struct.pack("<I", image.crc),
                timeout=_FINISH_TIMEOUT_S,
                retries=1,
            )
        except BaseException:
            try:
                await self._cmd(_OP_ABORT)
            except Exception:  # noqa: BLE001 - the session also times out (10 s)
                pass
            raise
        if r is not None and r[1] != _STATUS_OK:
            raise FirmwareUpdateError(f"image rejected: {spec.status_name(r[1])}")
        if r is None:
            # The board may have accepted it and rebooted before the reply
            # got out; the build ID after the reboot tells.
            log("no reply to UPDATE_FINISH; checking whether the board rebooted")
        else:
            log("verified on the board; rebooting into the new image")

        info = await self._wait_for_reboot()
        log(f"board: {info.describe()}")
        if info.build_id != image.build_id:
            raise FirmwareUpdateError(
                "the board came back on a different build - the bootrom "
                "rejected the new image and kept the previous one"
            )
        if not info.trial:
            raise FirmwareUpdateError("the new image is not on trial (unexpected)")
        if not confirm:
            log(
                f"left on trial ({spec.trial_lock}); it reverts unless confirmed "
                "within 15 s of booting"
            )
            return info
        await self.confirm(image.build_id, log)
        final = await self.info()
        log(f"confirmed: {final.describe() if final else 'no FW_INFO reply'}")
        return final


class LiftFirmwareUpdater(BoardFirmwareUpdater):
    """Firmware updates for the jelly_legs lift board."""

    spec = LEGS
