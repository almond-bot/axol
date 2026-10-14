"""Jelly Legs lift firmware update: image checks, the CAN protocol against a
simulated board, the CLI, and the control panel's upload endpoint."""

from __future__ import annotations

import argparse
import asyncio
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from almond_axol.robot import lift_firmware
from almond_axol.robot.lift_firmware import (
    IDENT_MAGIC,
    FirmwareImage,
    FirmwareUpdateError,
    LiftFirmwareUpdater,
)
from almond_axol.serve.commands import COMMANDS
from tests.test_serve_session_reservation import _Manager, _Runner, _test_app


def _image(version=(0, 9), built=b"Oct 10 2026 12:00:00", size=3000) -> bytes:
    ident = IDENT_MAGIC + bytes(version) + built.ljust(22, b"\0")
    assert len(ident) == 40
    body = bytes(range(256)) * (size // 256 + 1)
    return body[:100] + ident + body[: size - 140]


class FakeBoard:
    """The firmware's update state machine, driven through CanBus' API."""

    def __init__(self, build_id: int = 0x1111, partitioned: bool = True) -> None:
        self.listeners = []
        self.build_id = build_id
        self.version = (0, 9)
        self.state = 0x02 if partitioned else 0x00
        self.image = bytearray()
        self.size = 0
        self.chunk: tuple[int, int, bytearray] | None = None
        self.sent_ops: list[int] = []
        self.drop_acks = 0  # lose this many chunk acks (forces re-sends)
        self.busy_replies = 0
        self.confirm_status = 0

    def _add_listener(self, listener) -> None:  # noqa: ANN001
        self.listeners.append(listener)

    def _reply(self, data: bytes) -> None:
        msg = SimpleNamespace(arbitration_id=0x423, data=data.ljust(8, b"\0"))
        for listener in self.listeners:
            listener(msg)

    async def _send(self, can_id: int, data: bytes) -> bool:
        if can_id == 0x420:
            self._command(data)
        elif can_id == 0x424:
            self._data(data)
        return True

    def _command(self, d: bytes) -> None:
        op = d[0]
        self.sent_ops.append(op)
        if op == 0x09:
            self._reply(
                bytes([0x09, self.state, *self.version])
                + struct.pack("<I", self.build_id)
            )
        elif op == 0x0A:
            if self.busy_replies:
                self.busy_replies -= 1
                self._reply(bytes([0x0A, 1]) + struct.pack("<I", 0))
                return
            self.size = struct.unpack_from("<I", d, 1)[0]
            self.image = bytearray(self.size)
            self.state |= 0x08
            self._reply(bytes([0x0A, 0]) + struct.pack("<I", 1016 * 1024))
        elif op == 0x0B:
            crc = struct.unpack_from("<I", d, 1)[0]
            if zlib.crc32(bytes(self.image)) != crc:
                self._reply(bytes([0x0B, 7]))
                return
            new = FirmwareImage.parse(bytes(self.image))
            self._reply(bytes([0x0B, 0]))
            # Reboot into the new image, on trial.
            self.build_id = new.build_id
            self.state = (self.state & ~0x08) | 0x01
        elif op == 0x0C:
            self.state &= ~0x08
            self._reply(bytes([0x0C, 0]))
        elif op == 0x0D:
            ok = self.state & 0x01
            self.state &= ~0x01
            self._reply(bytes([0x0D, self.confirm_status if ok else 9]))

    def _data(self, d: bytes) -> None:
        if d[0] == 0xFF:
            offset, length = struct.unpack_from("<IH", d, 1)
            self.chunk = (offset, length, bytearray(length))
            return
        assert self.chunk is not None
        offset, length, buf = self.chunk
        start = d[0] * 7
        part = d[1:][: max(0, length - start)]
        buf[start : start + len(part)] = part
        if start + 7 >= length:
            self.image[offset : offset + length] = buf
            if self.drop_acks:
                self.drop_acks -= 1
                return
            self._reply(bytes([0x10, 0]) + struct.pack("<I", offset))


class FirmwareImageTest(unittest.TestCase):
    def test_parses_identity_block(self) -> None:
        data = _image()
        image = FirmwareImage.parse(data)
        self.assertEqual(image.version, "0.9")
        self.assertEqual(image.built, "Oct 10 2026 12:00:00")
        self.assertEqual(image.build_id, zlib.crc32(data[100:140]))
        self.assertEqual(image.crc, zlib.crc32(data))

    def test_rejects_wrong_files(self) -> None:
        cases = (
            (b"", "empty"),
            (b"\0" * 4096, "identity block"),
            (b"UF2\n" + _image(), "uf2"),
            (b"\x7fELF" + _image(), "elf"),
            (_image(size=lift_firmware.MAX_IMAGE_BYTES + 1), "at most"),
        )
        for data, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(FirmwareUpdateError, message),
            ):
                FirmwareImage.parse(data)

    def test_store_upload_dedupes_and_prunes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            first = lift_firmware.store_upload(FirmwareImage.parse(_image()), directory)
            again = lift_firmware.store_upload(FirmwareImage.parse(_image()), directory)
            self.assertEqual(first, again)
            self.assertEqual(first.read_bytes(), _image())
            for n in range(8):
                lift_firmware.store_upload(
                    FirmwareImage.parse(_image(built=f"build {n}".encode())), directory
                )
            self.assertEqual(len(list(directory.glob("jelly_legs-*.bin"))), 5)


class LiftFirmwareUpdaterTest(unittest.IsolatedAsyncioTestCase):
    async def test_flashes_verifies_and_confirms(self) -> None:
        board = FakeBoard()
        image = FirmwareImage.parse(_image())
        updater = LiftFirmwareUpdater(board)  # type: ignore[arg-type]
        progress: list[int] = []

        with patch.object(lift_firmware.asyncio, "sleep", _no_sleep):
            info = await updater.flash(
                image, log=lambda _m: None, progress=lambda d, _t: progress.append(d)
            )

        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual(info.build_id, image.build_id)
        self.assertFalse(info.trial)
        self.assertEqual(bytes(board.image), image.data)
        self.assertEqual(progress[-1], len(image.data))
        self.assertIn(0x0D, board.sent_ops)

    async def test_confirm_with_old_image_left_is_a_warning(self) -> None:
        board = FakeBoard()
        board.confirm_status = 6  # bought, previous image not erased
        image = FirmwareImage.parse(_image())
        log: list[str] = []
        with patch.object(lift_firmware.asyncio, "sleep", _no_sleep):
            info = await LiftFirmwareUpdater(board).flash(image, log=log.append)  # type: ignore[arg-type]
        assert info is not None
        self.assertEqual(info.build_id, image.build_id)
        self.assertTrue(any("could not be erased" in line for line in log))

    async def test_resends_a_chunk_whose_ack_was_lost(self) -> None:
        board = FakeBoard()
        board.drop_acks = 2
        image = FirmwareImage.parse(_image())
        updater = LiftFirmwareUpdater(board)  # type: ignore[arg-type]
        with (
            patch.object(lift_firmware, "_CHUNK_ACK_TIMEOUT_S", 0.01),
            patch.object(lift_firmware.asyncio, "sleep", _no_sleep),
        ):
            await updater.flash(image, log=lambda _m: None)
        self.assertEqual(bytes(board.image), image.data)

    async def test_waits_out_a_busy_board(self) -> None:
        board = FakeBoard()
        board.busy_replies = 2
        updater = LiftFirmwareUpdater(board)  # type: ignore[arg-type]
        with patch.object(lift_firmware.asyncio, "sleep", _no_sleep):
            info = await updater.flash(
                FirmwareImage.parse(_image()), log=lambda _m: None
            )
        self.assertIsNotNone(info)

    async def test_same_build_is_skipped_unless_forced(self) -> None:
        image = FirmwareImage.parse(_image())
        board = FakeBoard(build_id=image.build_id)
        updater = LiftFirmwareUpdater(board)  # type: ignore[arg-type]
        self.assertIsNone(await updater.flash(image, log=lambda _m: None))
        self.assertNotIn(0x0A, board.sent_ops)

        with patch.object(lift_firmware.asyncio, "sleep", _no_sleep):
            info = await updater.flash(image, force=True, log=lambda _m: None)
        self.assertIsNotNone(info)

    async def test_refuses_boards_that_cannot_update(self) -> None:
        image = FirmwareImage.parse(_image())
        cases = (
            (FakeBoard(partitioned=False), "partition table"),
            (_trial_board(), "unconfirmed"),
        )
        for board, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(FirmwareUpdateError, message),
            ):
                await LiftFirmwareUpdater(board).flash(image, log=lambda _m: None)  # type: ignore[arg-type]

    async def test_old_firmware_without_fw_info_is_reported(self) -> None:
        board = FakeBoard()
        board._command = lambda d: board.sent_ops.append(d[0])  # never replies
        updater = LiftFirmwareUpdater(board)  # type: ignore[arg-type]
        with (
            patch.object(lift_firmware.LiftFirmwareUpdater, "_wait_reply", _no_reply),
            self.assertRaisesRegex(FirmwareUpdateError, "older than 0.9"),
        ):
            await updater.flash(FirmwareImage.parse(_image()), log=lambda _m: None)

    async def test_cancel_aborts_the_session(self) -> None:
        board = FakeBoard()
        updater = LiftFirmwareUpdater(board)  # type: ignore[arg-type]
        chunks = 0

        def progress(_done: int, _total: int) -> None:
            nonlocal chunks
            chunks += 1

        with self.assertRaises(asyncio.CancelledError):
            await updater.flash(
                FirmwareImage.parse(_image()),
                log=lambda _m: None,
                progress=progress,
                cancelled=lambda: chunks >= 2,
            )
        self.assertEqual(board.sent_ops[-1], 0x0C)
        self.assertEqual(board.build_id, 0x1111)


class LiftUpdateCliTest(unittest.TestCase):
    def _parse(self, argv: list[str]) -> argparse.Namespace:
        from almond_axol.cli.lift import update

        parser = argparse.ArgumentParser()
        update.add_parser(parser.add_subparsers())
        return parser.parse_args(["lift.update", *argv])

    def test_firmware_is_optional(self) -> None:
        args = self._parse([])
        self.assertIsNone(args.firmware)
        self.assertFalse(args.force)
        self.assertEqual(self._parse(["--firmware", "fw.bin"]).firmware, Path("fw.bin"))

    def test_registered_as_axol_helper(self) -> None:
        cmd = COMMANDS["lift.update"]
        self.assertEqual(cmd.section, "helper")
        self.assertEqual(cmd.category, "Diagnostics")


class LiftFirmwareUploadApiTest(unittest.IsolatedAsyncioTestCase):
    async def _post(self, body: bytes, directory: Path) -> httpx.Response:
        app = _test_app(_Manager(), _Runner())
        transport = httpx.ASGITransport(app=app)
        with patch.object(lift_firmware, "UPLOAD_DIR", directory):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                return await client.post("/api/lift/firmware", content=body)

    async def test_stores_a_valid_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            response = await self._post(_image(), Path(tmp))
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["version"], "0.9")
            self.assertEqual(body["size"], len(_image()))
            self.assertEqual(Path(body["path"]).read_bytes(), _image())
            self.assertEqual(Path(body["path"]).parent, Path(tmp))

    async def test_rejects_other_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            response = await self._post(b"\0" * 1024, Path(tmp))
            self.assertEqual(response.status_code, 400)
            self.assertIn("identity block", response.json()["error"])
            self.assertEqual(list(Path(tmp).iterdir()), [])

    async def test_rejects_oversized_uploads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            body = b"\0" * (lift_firmware.MAX_IMAGE_BYTES + 1)
            response = await self._post(body, Path(tmp))
            self.assertEqual(response.status_code, 413)


def _trial_board() -> FakeBoard:
    board = FakeBoard()
    board.state |= 0x01
    return board


async def _no_sleep(_s: float) -> None:
    return None


async def _no_reply(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
    return None


if __name__ == "__main__":
    unittest.main()
