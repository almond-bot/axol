"""Almond Safety board firmware update: the shared CAN update client on the
safety board's IDs, side by side with the jelly_legs lift board on one bus,
plus the CLI and the control panel's upload endpoint."""

from __future__ import annotations

import argparse
import dataclasses
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from almond_axol.robot import lift_firmware, safety_firmware
from almond_axol.robot.lift_firmware import (
    FirmwareImage,
    FirmwareUpdateError,
    LiftFirmwareUpdater,
)
from almond_axol.robot.safety import decode_status, explain_start_failure
from almond_axol.robot.safety_firmware import (
    SAFETY,
    SafetyFirmwareUpdater,
    parse_image,
)
from almond_axol.serve.commands import COMMANDS
from tests.test_lift_firmware import FakeBoard, _image, _no_sleep
from tests.test_serve_session_reservation import _Manager, _Runner, _test_app


def _safety_image(built: bytes = b"Oct 28 2026 22:19:11", size: int = 3000) -> bytes:
    ident = SAFETY.ident_magic + bytes((0, 3)) + built.ljust(22, b"\0")
    assert len(ident) == 40
    body = bytes(range(256)) * (size // 256 + 1)
    return body[:100] + ident + body[: size - 140]


class FakeSafetyBoard(FakeBoard):
    """The safety board: the legs' update state machine on 0x440 / 0x443 /
    0x444, which also refuses UPDATE_BEGIN while its output is on."""

    ID_CMD, ID_REPLY, ID_DATA = 0x440, 0x443, 0x444
    SPEC = SAFETY

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.version = (0, 3)
        self.output_on = True
        self.ignore_stops = 0  # lose this many SW_STOP frames

    def _command(self, d: bytes) -> None:
        if d[0] == 0x01:  # SW_STOP
            self.sent_ops.append(0x01)
            if self.ignore_stops:
                self.ignore_stops -= 1
            else:
                self.output_on = False
            return
        if d[0] == 0x0A and self.output_on:
            self.sent_ops.append(0x0A)
            self._reply(bytes([0x0A, 1]) + bytes(4))  # busy: power is on
            return
        if d[0] in (0x04, 0x05):  # status traffic: answered on 0x441, not here
            return
        super()._command(d)


class SharedBus:
    """One CAN bus carrying several fake boards: every frame reaches every
    board, and every reply reaches every listener."""

    def __init__(self, *boards: FakeBoard) -> None:
        self.boards = boards

    def _add_listener(self, listener) -> None:  # noqa: ANN001
        for board in self.boards:
            board._add_listener(listener)

    async def _send(self, can_id: int, data: bytes) -> bool:
        for board in self.boards:
            await board._send(can_id, data)
        return True


class SafetyImageTest(unittest.TestCase):
    def test_parses_a_safety_image(self) -> None:
        image = parse_image(_safety_image())
        self.assertEqual(image.version, "0.3")
        self.assertIs(image.spec, SAFETY)

    def test_each_board_rejects_the_others_firmware(self) -> None:
        with self.assertRaisesRegex(FirmwareUpdateError, "safety board firmware"):
            parse_image(_image())
        with self.assertRaisesRegex(FirmwareUpdateError, "Jelly lift firmware"):
            FirmwareImage.parse(_safety_image())

    def test_store_upload_names_files_by_board(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = safety_firmware.store_upload(parse_image(_safety_image()), Path(tmp))
            self.assertTrue(path.name.startswith("almond_safety-0.3-"))


class SafetyFirmwareUpdaterTest(unittest.IsolatedAsyncioTestCase):
    async def test_cuts_power_then_flashes_and_confirms(self) -> None:
        board = FakeSafetyBoard()
        image = parse_image(_safety_image())
        log: list[str] = []
        with patch.object(lift_firmware.asyncio, "sleep", _no_sleep):
            info = await SafetyFirmwareUpdater(board).flash(image, log=log.append)  # type: ignore[arg-type]
        assert info is not None
        self.assertEqual(info.build_id, image.build_id)
        self.assertFalse(info.trial)
        self.assertEqual(bytes(board.image), image.data)
        self.assertLess(board.sent_ops.index(0x01), board.sent_ops.index(0x0A))
        self.assertTrue(any("cutting power" in line for line in log))

    async def test_resends_the_stop_while_power_stays_on(self) -> None:
        board = FakeSafetyBoard()
        board.ignore_stops = 2
        with patch.object(lift_firmware.asyncio, "sleep", _no_sleep):
            info = await SafetyFirmwareUpdater(board).flash(  # type: ignore[arg-type]
                parse_image(_safety_image()), log=lambda _m: None
            )
        self.assertIsNotNone(info)
        self.assertEqual(board.sent_ops.count(0x01), 3)

    async def test_refuses_a_lift_image_before_sending_anything(self) -> None:
        board = FakeSafetyBoard()
        with self.assertRaisesRegex(FirmwareUpdateError, "not a safety board"):
            await SafetyFirmwareUpdater(board).flash(  # type: ignore[arg-type]
                FirmwareImage.parse(_image()), log=lambda _m: None
            )
        self.assertEqual(board.sent_ops, [])

    async def test_board_rejects_a_lift_image_that_slips_through(self) -> None:
        board = FakeSafetyBoard()
        forged = dataclasses.replace(FirmwareImage.parse(_image()), spec=SAFETY)
        with (
            patch.object(lift_firmware.asyncio, "sleep", _no_sleep),
            self.assertRaisesRegex(FirmwareUpdateError, "not a safety board"),
        ):
            await SafetyFirmwareUpdater(board).flash(forged, log=lambda _m: None)  # type: ignore[arg-type]
        self.assertEqual(board.build_id, 0x1111)

    async def test_old_firmware_is_reported_with_the_safety_version(self) -> None:
        board = FakeSafetyBoard()
        board._command = lambda d: None  # firmware 0.2 ignores FW_INFO
        with (
            patch.object(SafetyFirmwareUpdater, "_wait_reply", _no_reply),
            self.assertRaisesRegex(FirmwareUpdateError, "safety board.*older than 0.3"),
        ):
            await SafetyFirmwareUpdater(board).flash(  # type: ignore[arg-type]
                parse_image(_safety_image()), log=lambda _m: None
            )


class SharedBusTest(unittest.IsolatedAsyncioTestCase):
    """The lift board and the safety board on one bus, as on Jelly."""

    async def test_updating_one_board_never_touches_the_other(self) -> None:
        legs, safety = FakeBoard(build_id=0x1111), FakeSafetyBoard(build_id=0x2222)
        bus = SharedBus(legs, safety)
        safety_updater = SafetyFirmwareUpdater(bus)  # type: ignore[arg-type]
        lift_updater = LiftFirmwareUpdater(bus)  # type: ignore[arg-type]
        safety_image = parse_image(_safety_image())
        lift_image = FirmwareImage.parse(_image())

        with patch.object(lift_firmware.asyncio, "sleep", _no_sleep):
            await safety_updater.flash(safety_image, log=lambda _m: None)
            self.assertEqual(legs.sent_ops, [])
            self.assertEqual(legs.build_id, 0x1111)
            self.assertEqual(bytes(legs.image), b"")

            safety_ops = list(safety.sent_ops)
            await lift_updater.flash(lift_image, log=lambda _m: None)
            self.assertEqual(safety.sent_ops, safety_ops)
            self.assertEqual(safety.build_id, safety_image.build_id)

        self.assertEqual(legs.build_id, lift_image.build_id)
        self.assertEqual(bytes(safety.image), safety_image.data)
        self.assertEqual(bytes(legs.image), lift_image.data)


class SafetyStatusUpdateLockTest(unittest.TestCase):
    def test_decodes_the_update_lock(self) -> None:
        status = decode_status(bytes([0x91, 0, 5, 0x03, 0x20, 0x67, 0, 0]))
        self.assertTrue(status.update_locked)
        self.assertEqual(status.stop_cause, "update")
        self.assertIn("updating its firmware", explain_start_failure(status))

    def test_decodes_a_supply_dip(self) -> None:
        status = decode_status(bytes([0x89, 0x08, 0, 0x04, 0x20, 0x67, 0, 0]))
        self.assertTrue(status.supply_dip)
        self.assertEqual(status.firmware_version, "0.4")
        self.assertFalse(status.relay_fault)

    def test_older_firmware_reads_unlocked(self) -> None:
        status = decode_status(bytes([0x8F, 0, 0, 0x02, 0x20, 0x67, 0x90, 0x65]))
        self.assertFalse(status.update_locked)


class SafetyUpdateCliTest(unittest.TestCase):
    def _parse(self, argv: list[str]) -> argparse.Namespace:
        from almond_axol.cli import safety_update

        parser = argparse.ArgumentParser()
        safety_update.add_parser(parser.add_subparsers())
        return parser.parse_args(["safety.update", *argv])

    def test_firmware_is_optional(self) -> None:
        self.assertIsNone(self._parse([]).firmware)
        self.assertEqual(self._parse(["--firmware", "fw.bin"]).firmware, Path("fw.bin"))

    def test_registered_as_axol_helper(self) -> None:
        cmd = COMMANDS["safety.update"]
        self.assertEqual(cmd.section, "helper")
        self.assertEqual(cmd.category, "Diagnostics")


class SafetyFirmwareUploadApiTest(unittest.IsolatedAsyncioTestCase):
    async def _post(self, body: bytes, directory: Path) -> httpx.Response:
        app = _test_app(_Manager(), _Runner())
        transport = httpx.ASGITransport(app=app)
        with patch.object(safety_firmware, "UPLOAD_DIR", directory):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                return await client.post("/api/safety/firmware", content=body)

    async def test_stores_a_valid_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            response = await self._post(_safety_image(), Path(tmp))
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["version"], "0.3")
            self.assertEqual(Path(body["path"]).parent, Path(tmp))

    async def test_rejects_the_lift_firmware(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            response = await self._post(_image(), Path(tmp))
            self.assertEqual(response.status_code, 400)
            self.assertIn("safety board firmware", response.json()["error"])
            self.assertEqual(list(Path(tmp).iterdir()), [])


async def _no_reply(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
    return None


if __name__ == "__main__":
    unittest.main()
