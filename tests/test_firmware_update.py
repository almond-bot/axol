"""MyActuator firmware flashing against a simulated bootloader."""

from __future__ import annotations

import unittest

import can

from almond_axol.motor.errors import MotorError
from almond_axol.motor.firmware import FirmwareUpdater, _crc16

_ID = 0x03
_TRIGGER = (_ID << 6) | 0x28
_DATA = (_ID << 6) | 0x3E
_REPLY = (_ID << 6) | 0x14


class _Bootloader:
    """Answers like the RMD bootloader: "C" on the trigger, then one reply per packet.

    ``replies`` overrides what the Nth data packet (0 = block 0) is answered
    with; each entry is a list of reply frames.
    """

    def __init__(self, replies: dict[int, list[bytes]] | None = None) -> None:
        self.replies = replies or {}
        self.frames: list[tuple[int, bytes]] = []
        self.packets: list[bytes] = []
        self._pending = b""
        self._listener = None

    def _add_listener(self, callback) -> None:  # type: ignore[no-untyped-def]
        self._listener = callback

    def _reply(self, *frames: bytes) -> None:
        assert self._listener is not None
        for data in frames:
            self._listener(can.Message(arbitration_id=_REPLY, data=data))

    async def _send(self, arbitration_id: int, data: bytes) -> None:
        self.frames.append((arbitration_id, bytes(data)))
        if arbitration_id == _TRIGGER:
            self._reply(bytes([0x43]))
            return
        assert arbitration_id == _DATA
        if not self._pending and data == bytes([0x04]):  # EOT
            self._reply(bytes([0x06]))
            return
        self._pending += bytes(data)
        size = {0x01: 128, 0x02: 1024}[self._pending[0]] + 5
        if len(self._pending) < size:
            return
        assert len(self._pending) == size, "a frame spilled into the next packet"
        packet, self._pending = self._pending, b""
        n = len(self.packets)
        self.packets.append(packet)
        default = [bytes([0x06, 0x43])] if n == 0 else [bytes([0x06])]
        self._reply(*self.replies.get(n, default))


def _flash(boot: _Bootloader, image: bytes, name: str = "RMD-X6.bin") -> None:
    import asyncio

    asyncio.run(FirmwareUpdater(boot, _ID).flash(image, name=name))  # type: ignore[arg-type]


class FirmwareFlashTest(unittest.TestCase):
    def test_transfer_matches_the_vendor_tool(self) -> None:
        # 1340 B: a full 1024 B block, then a 316 B tail (> 128) padded into
        # a second 1024 B block.
        image = bytes(range(256)) * 5 + b"\xaa" * 60
        boot = _Bootloader()
        _flash(boot, image)

        self.assertEqual(boot.frames[0], (_TRIGGER, bytes(range(0xF0, 0xF8))))
        self.assertEqual(boot.frames[-1], (_DATA, bytes([0x04])))
        # Every data frame carries at most 8 bytes.
        self.assertTrue(all(len(d) <= 8 for i, d in boot.frames if i == _DATA))

        header = boot.packets[0]
        self.assertEqual(header[:3], bytes([0x01, 0x00, 0xFF]))
        self.assertEqual(header[3:131].rstrip(b"\x00"), b"RMD-X6.bin\x001340")
        self.assertEqual(int.from_bytes(header[131:133], "big"), _crc16(header[3:131]))

        blocks = boot.packets[1:]
        self.assertEqual([p[:3] for p in blocks], [b"\x02\x01\xfe", b"\x02\x02\xfd"])
        payload = b"".join(p[3:-2] for p in blocks)
        self.assertEqual(payload, image.ljust(2048, b"\x1a"))
        for p in blocks:
            self.assertEqual(int.from_bytes(p[-2:], "big"), _crc16(p[3:-2]))

    def test_short_tail_goes_in_a_padded_128_byte_block(self) -> None:
        boot = _Bootloader()
        _flash(boot, b"\x11" * (1024 + 100))
        self.assertEqual([p[0] for p in boot.packets[1:]], [0x02, 0x01])
        self.assertEqual(boot.packets[2][3:-2], b"\x11" * 100 + b"\x1a" * 28)

    def test_bootloader_failure_replies_abort_with_their_meaning(self) -> None:
        cases = [
            ([bytes([0x15])], "checksum error"),
            ([bytes([0xF1])], "firmware version mismatch"),
            ([bytes([0x00, 0x00])], "failed to receive the first packet"),
            ([bytes([0x18, 0x18])], "frame header error"),
            ([bytes([0x18]), bytes([0x18])], "failed to write flash"),
        ]
        for frames, message in cases:
            with self.subTest(message=message):
                boot = _Bootloader({1: frames})
                with self.assertRaisesRegex(MotorError, message):
                    _flash(boot, b"\x22" * 2048)
                # Never retransmitted: block 1 went out once, block 2 never.
                self.assertEqual(len(boot.packets), 2)

    def test_single_cancel_between_other_frames_is_not_a_failure(self) -> None:
        boot = _Bootloader(
            {1: [bytes([0x18]), bytes([0x43]), bytes([0x18]), bytes([0x06])]}
        )
        _flash(boot, b"\x33" * 1024)
        self.assertEqual(len(boot.packets), 2)

    def test_name_longer_than_the_vendor_limit_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "120 bytes"):
            _flash(_Bootloader(), b"\x44" * 16, name="a" * 121 + ".bin")
        _flash(_Bootloader(), b"\x44" * 16, name="a" * 116 + ".bin")  # exactly 120


if __name__ == "__main__":
    unittest.main()
