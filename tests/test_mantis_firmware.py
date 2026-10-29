"""Mantis trigger firmware update: the shared CAN update client on the Mantis
board's IDs, the two-sided CLI, and the control panel's upload endpoint."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from almond_axol.cli import mantis_update
from almond_axol.robot import lift_firmware, mantis_firmware
from almond_axol.robot.lift_firmware import FirmwareImage, FirmwareUpdateError
from almond_axol.robot.mantis_firmware import (
    MANTIS,
    MantisFirmwareUpdater,
    parse_image,
)
from almond_axol.serve.commands import COMMANDS
from tests.test_lift_firmware import FakeBoard, _image, _no_sleep
from tests.test_serve_session_reservation import _Manager, _Runner, _test_app


def _mantis_image(built: bytes = b"Oct 29 2026 17:51:29", size: int = 3000) -> bytes:
    ident = MANTIS.ident_magic + bytes((0, 2)) + built.ljust(22, b"\0")
    assert len(ident) == 40
    body = bytes(range(256)) * (size // 256 + 1)
    return body[:100] + ident + body[: size - 140]


class FakeMantisBoard(FakeBoard):
    """The trigger board: the legs' update state machine on 0x460 / 0x463 /
    0x464. It has no status broadcast, so it ignores SET_RATE."""

    ID_CMD, ID_REPLY, ID_DATA = 0x460, 0x463, 0x464
    SPEC = MANTIS

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.version = (0, 2)
        self.sent_ids: set[int] = set()

    async def _send(self, can_id: int, data: bytes) -> bool:
        self.sent_ids.add(can_id)
        return await super()._send(can_id, data)

    def _command(self, d: bytes) -> None:
        if d[0] == 0x05:  # no such opcode on this board
            self.sent_ops.append(0x05)
            return
        super()._command(d)


class MantisImageTest(unittest.TestCase):
    def test_parses_a_mantis_image(self) -> None:
        image = parse_image(_mantis_image())
        self.assertEqual(image.version, "0.2")
        self.assertEqual(image.built, "Oct 29 2026 17:51:29")
        self.assertIs(image.spec, MANTIS)

    def test_magic_is_the_firmwares_zero_padded_field(self) -> None:
        self.assertEqual(MANTIS.ident_magic, b"mantis-fwid\0\0\0\0\0")

    def test_mantis_and_lift_reject_each_others_firmware(self) -> None:
        with self.assertRaisesRegex(FirmwareUpdateError, "Mantis firmware"):
            parse_image(_image())
        with self.assertRaisesRegex(FirmwareUpdateError, "Jelly lift firmware"):
            FirmwareImage.parse(_mantis_image())

    def test_store_upload_names_files_by_board(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = mantis_firmware.store_upload(parse_image(_mantis_image()), Path(tmp))
            self.assertTrue(path.name.startswith("mantis-0.2-"))


class MantisFirmwareUpdaterTest(unittest.IsolatedAsyncioTestCase):
    async def test_flashes_and_confirms_on_its_own_ids(self) -> None:
        board = FakeMantisBoard()
        image = parse_image(_mantis_image())
        with patch.object(lift_firmware.asyncio, "sleep", _no_sleep):
            info = await MantisFirmwareUpdater(board).flash(  # type: ignore[arg-type]
                image, log=lambda _m: None
            )
        assert info is not None
        self.assertEqual(info.build_id, image.build_id)
        self.assertFalse(info.trial)
        self.assertEqual(bytes(board.image), image.data)
        self.assertEqual(board.sent_ids, {0x460, 0x464})
        self.assertNotIn(0x05, board.sent_ops)  # no status broadcast to quiet
        self.assertIn(0x0D, board.sent_ops)

    async def test_refuses_a_lift_image_before_sending_anything(self) -> None:
        board = FakeMantisBoard()
        with self.assertRaisesRegex(FirmwareUpdateError, "not a Mantis"):
            await MantisFirmwareUpdater(board).flash(  # type: ignore[arg-type]
                FirmwareImage.parse(_image()), log=lambda _m: None
            )
        self.assertEqual(board.sent_ops, [])

    async def test_board_rejects_a_lift_image_that_slips_through(self) -> None:
        board = FakeMantisBoard()
        forged = dataclasses.replace(FirmwareImage.parse(_image()), spec=MANTIS)
        with (
            patch.object(lift_firmware.asyncio, "sleep", _no_sleep),
            self.assertRaisesRegex(FirmwareUpdateError, "not a Mantis"),
        ):
            await MantisFirmwareUpdater(board).flash(forged, log=lambda _m: None)  # type: ignore[arg-type]
        self.assertEqual(board.build_id, 0x1111)

    async def test_old_firmware_is_reported_with_the_mantis_version(self) -> None:
        board = FakeMantisBoard()
        board._command = lambda d: None  # firmware 0.1 has no CAN receiver
        with (
            patch.object(MantisFirmwareUpdater, "_wait_reply", _no_reply),
            self.assertRaisesRegex(FirmwareUpdateError, "Mantis rig.*older than 0.2"),
        ):
            await MantisFirmwareUpdater(board).flash(  # type: ignore[arg-type]
                parse_image(_mantis_image()), log=lambda _m: None
            )


class _FakeBus:
    """What :class:`~almond_axol.motor.CanBus` offers the updater, wired to a
    fake board; ``boards`` maps each interface to its board (None: no reply)."""

    boards: dict[str, FakeMantisBoard | None] = {}
    opened: list[str] = []

    def __init__(self, channel: str) -> None:
        self.channel = channel
        self.board = self.boards[channel]
        self.listeners = []

    def _add_listener(self, listener) -> None:  # noqa: ANN001
        self.listeners.append(listener)
        if self.board is not None:
            self.board._add_listener(listener)

    async def _send(self, can_id: int, data: bytes) -> bool:
        if self.board is not None:
            await self.board._send(can_id, data)
        return True

    async def start(self) -> None:
        self.opened.append(self.channel)

    async def close(self) -> None:
        pass


class MantisUpdateCliTest(unittest.IsolatedAsyncioTestCase):
    def _parse(self, argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        mantis_update.add_parser(parser.add_subparsers())
        return parser.parse_args(["mantis.update", *argv])

    def test_defaults(self) -> None:
        args = self._parse([])
        self.assertIsNone(args.firmware)
        self.assertEqual(args.side, "both")
        self.assertIsNone(args.channel)
        self.assertEqual(self._parse(["--firmware", "fw.bin"]).firmware, Path("fw.bin"))

    def test_registered_as_mantis_helper(self) -> None:
        cmd = COMMANDS["mantis.update"]
        self.assertEqual(cmd.section, "helper")
        self.assertEqual(cmd.category, "Diagnostics")
        self.assertEqual(cmd.hardware_profiles, ("mantis",))

    def test_targets_follow_the_mantis_settings(self) -> None:
        with patch(
            "almond_axol.settings.shared_mantis_can_channels",
            return_value=("can_l", "can_r"),
        ):
            self.assertEqual(
                mantis_update.resolve_targets("both", None),
                [("left rig", "can_l"), ("right rig", "can_r")],
            )
            self.assertEqual(
                mantis_update.resolve_targets("right", None),
                [("right rig", "can_r")],
            )
        self.assertEqual(
            mantis_update.resolve_targets("both", "can_x"), [("rig", "can_x")]
        )

    def test_a_side_without_a_channel(self) -> None:
        with patch(
            "almond_axol.settings.shared_mantis_can_channels",
            return_value=("can_l", None),
        ):
            with contextlib.redirect_stdout(None):
                targets = mantis_update.resolve_targets("both", None)
            self.assertEqual(targets, [("left rig", "can_l")])
            with self.assertRaisesRegex(SystemExit, "right rig has no CAN"):
                mantis_update.resolve_targets("right", None)

    async def _run(self, argv: list[str], iface_up=lambda _ch: True) -> str:  # noqa: ANN001
        args = self._parse(argv)
        out = []
        with (
            patch.object(mantis_update, "CanBus", _FakeBus),
            patch("almond_axol.cli.can.setup.iface_up", iface_up),
            patch(
                "almond_axol.settings.shared_mantis_can_channels",
                return_value=("can_l", "can_r"),
            ),
            patch.object(lift_firmware.asyncio, "sleep", _no_sleep),
            patch("builtins.print", lambda *a, **_k: out.append(" ".join(map(str, a)))),
        ):
            await mantis_update._run(args)
        return "\n".join(out)

    async def test_updates_both_sides_in_turn(self) -> None:
        left, right = FakeMantisBoard(), FakeMantisBoard(build_id=0x2222)
        _FakeBus.boards = {"can_l": left, "can_r": right}
        _FakeBus.opened = []
        image = parse_image(_mantis_image())
        with tempfile.TemporaryDirectory() as tmp:
            fw = Path(tmp) / "firmware.bin"
            fw.write_bytes(image.data)
            out = await self._run(["--firmware", str(fw)])
        self.assertEqual(_FakeBus.opened, ["can_l", "can_r"])
        self.assertEqual(left.build_id, image.build_id)
        self.assertEqual(right.build_id, image.build_id)
        self.assertIn("Left rig: Mantis firmware updated to 0.2.", out)
        self.assertIn("Right rig: Mantis firmware updated to 0.2.", out)

    async def test_one_silent_side_still_updates_the_other_then_fails(self) -> None:
        right = FakeMantisBoard()
        _FakeBus.boards = {"can_l": None, "can_r": right}
        _FakeBus.opened = []
        image = parse_image(_mantis_image())
        with (
            patch.object(MantisFirmwareUpdater, "info", _info_or_none),
            tempfile.TemporaryDirectory() as tmp,
        ):
            fw = Path(tmp) / "firmware.bin"
            fw.write_bytes(image.data)
            with self.assertRaisesRegex(SystemExit, "failed on the left rig"):
                await self._run(["--firmware", str(fw)])
        self.assertEqual(right.build_id, image.build_id)

    async def test_a_missing_interface_fails_only_that_rig(self) -> None:
        right = FakeMantisBoard()
        _FakeBus.boards = {"can_l": None, "can_r": right}
        _FakeBus.opened = []

        def bring_up(channels: list[str]) -> None:
            raise RuntimeError(f"CAN interface not found: {channels[0]}")

        with (
            patch("almond_axol.cli.can.setup.bring_up_interfaces", bring_up),
            self.assertRaisesRegex(SystemExit, "failed on the left rig"),
        ):
            await self._run([], iface_up=lambda ch: ch != "can_l")
        self.assertEqual(_FakeBus.opened, ["can_r"])

    async def test_reports_without_a_file(self) -> None:
        _FakeBus.boards = {"can_l": FakeMantisBoard(), "can_r": None}
        _FakeBus.opened = []
        out = await self._run(["--side", "left"])
        self.assertIn("left rig (can_l): firmware 0.2, build 0x00001111", out)
        self.assertEqual(_FakeBus.opened, ["can_l"])


class MantisFirmwareUploadApiTest(unittest.IsolatedAsyncioTestCase):
    async def _post(self, body: bytes, directory: Path) -> httpx.Response:
        app = _test_app(_Manager(), _Runner())
        transport = httpx.ASGITransport(app=app)
        with patch.object(mantis_firmware, "UPLOAD_DIR", directory):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                return await client.post("/api/mantis/firmware", content=body)

    async def test_stores_a_valid_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            response = await self._post(_mantis_image(), Path(tmp))
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["version"], "0.2")
            self.assertEqual(Path(body["path"]).parent, Path(tmp))

    async def test_rejects_the_lift_firmware(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            response = await self._post(_image(), Path(tmp))
            self.assertEqual(response.status_code, 400)
            self.assertIn("Mantis firmware", response.json()["error"])
            self.assertEqual(list(Path(tmp).iterdir()), [])


async def _info_or_none(self, timeout: float = 0.3, retries: int = 3):  # noqa: ANN001, ANN202
    """FW_INFO without the retry waits; None when the bus has no board."""
    if self._bus.board is None:
        return None
    return await lift_firmware.BoardFirmwareUpdater.info(self, timeout, retries)


async def _no_reply(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
    return None


if __name__ == "__main__":
    unittest.main()
