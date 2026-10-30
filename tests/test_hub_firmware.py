"""Keeping the Axol arm hub on the bundled firmware.

The hub's firmware decides whether a motor command parked behind a dead bus
(an e-stop) can still reach the motors when they are powered back on, so every
bring-up reflashes a hub running an older build. Covered here: the bundled
image matches the version the code expects, hub discovery through sysfs, the
update policy (when to flash, when to only warn, when to refuse the bring-up),
and that the check runs before anything opens the buses.
"""

from __future__ import annotations

import errno
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from almond_axol.cli.can import setup as can_setup
from almond_axol.constants import CAN_LEFT, CAN_RIGHT
from almond_axol.hub import firmware as hub_fw

_UF2_MAGIC = (0x0A324655, 0x9E5D5157)
_UF2_MAGIC_END = 0x0AB16F30
_RP2350_ARM_S = 0xE48BFF59


def _uf2_payload(path: Path) -> bytes:
    data = path.read_bytes()
    assert len(data) % 512 == 0, "UF2 is a sequence of 512-byte blocks"
    payload = bytearray()
    families = set()
    for off in range(0, len(data), 512):
        block = data[off : off + 512]
        m0, m1, flags, _addr, size, _no, _total, family = struct.unpack_from(
            "<8I", block
        )
        assert (m0, m1) == _UF2_MAGIC, f"bad UF2 magic at block {off // 512}"
        assert struct.unpack_from("<I", block, 508)[0] == _UF2_MAGIC_END
        if flags & 0x2000:  # family id present
            families.add(family)
        payload += block[32 : 32 + size]
    assert _RP2350_ARM_S in families, f"not an RP2350 image: {families}"
    return bytes(payload)


class BundledImageTest(unittest.TestCase):
    def test_image_reports_the_bundled_version(self) -> None:
        # The device descriptor baked into the image: an image copied in
        # without updating FIRMWARE_VERSION (or the reverse) would make every
        # bring-up reflash forever, or never.
        payload = _uf2_payload(hub_fw.FIRMWARE_PATH)
        descriptor = struct.pack(
            "<HHH", hub_fw.HUB_VID, hub_fw.HUB_PID, hub_fw.FIRMWARE_VERSION
        )
        self.assertIn(descriptor, payload)


class _FakeSysfs:
    """A sysfs tree with one USB device per port and netdevs pointing at them."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def usb(
        self,
        port: str,
        *,
        vid: int = hub_fw.HUB_VID,
        pid: int = hub_fw.HUB_PID,
        manufacturer: str = "Almond",
        product: str = "Axol CAN Bus",
        version: int = 0x0100,
        serial: str = "A2418ACCB47A8F12",
    ) -> Path:
        device = self.root / "devices" / "usb1" / port
        (device / f"{port}:1.0").mkdir(parents=True, exist_ok=True)
        for name, value in {
            "idVendor": f"{vid:04x}",
            "idProduct": f"{pid:04x}",
            "manufacturer": manufacturer,
            "product": product,
            "bcdDevice": f"{version:04x}",
            "serial": serial,
            "busnum": "1",
            "devnum": "3",
        }.items():
            (device / name).write_text(value + "\n")
        link = self.root / "bus" / "usb" / "devices" / port
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(device)
        return device

    def netdev(self, channel: str, port: str) -> None:
        net = self.root / "class" / "net" / channel
        net.mkdir(parents=True, exist_ok=True)
        (net / "device").symlink_to(
            self.root / "devices" / "usb1" / port / f"{port}:1.0"
        )


class HubDiscoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sysfs = _FakeSysfs(Path(self.enterContext(tempfile.TemporaryDirectory())))

    def test_both_arm_channels_resolve_to_one_hub(self) -> None:
        self.sysfs.usb("1-3", version=0x0100)
        self.sysfs.netdev(CAN_LEFT, "1-3")
        self.sysfs.netdev(CAN_RIGHT, "1-3")
        hubs = hub_fw.hubs_for_channels([CAN_LEFT, CAN_RIGHT], sysfs=self.sysfs.root)
        self.assertEqual(
            hubs, [hub_fw.Hub(port="1-3", serial="A2418ACCB47A8F12", version=0x0100)]
        )
        self.assertTrue(hubs[0].outdated)

    def test_other_candlelight_adapters_and_missing_channels_are_skipped(self) -> None:
        # Same 1d50:606f identity, someone else's firmware: never flash it.
        self.sysfs.usb(
            "1-4", manufacturer="bytewerk", product="candleLight USB to CAN adapter"
        )
        self.sysfs.netdev("can0", "1-4")
        self.assertEqual(
            hub_fw.hubs_for_channels(["can0", "can_missing"], sysfs=self.sysfs.root), []
        )

    def test_attached_hubs_include_one_without_interfaces(self) -> None:
        # A Mantis hub whose CAN interfaces never came up: no netdev at all.
        self.sysfs.usb("1-3", version=0x0201)
        self.sysfs.usb(
            "1-4", manufacturer="bytewerk", product="candleLight USB to CAN adapter"
        )
        self.sysfs.usb("1-5", vid=hub_fw.BOOT_VID, pid=hub_fw.BOOT_PID)
        (self.sysfs.root / "bus" / "usb" / "devices" / "1-3:1.0").mkdir()
        self.assertEqual(
            hub_fw.attached_hubs(sysfs=self.sysfs.root),
            [hub_fw.Hub(port="1-3", serial="A2418ACCB47A8F12", version=0x0201)],
        )
        self.assertEqual(hub_fw.bootloader_ports(sysfs=self.sysfs.root), ["1-5"])

    def test_version_format(self) -> None:
        self.assertEqual(hub_fw.format_version(0x0200), "2.00")
        self.assertEqual(hub_fw.format_version(0x0100), "1.00")


class EnsureFirmwareTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sysfs = _FakeSysfs(Path(self.enterContext(tempfile.TemporaryDirectory())))
        self.enterContext(patch.object(hub_fw, "_warned", set()))
        self.enterContext(patch.dict(os.environ, {"AXOL_HUB_FLASH": "1"}))
        self.enterContext(patch.object(hub_fw, "_wait_for_channels"))
        self.enterContext(patch.object(hub_fw.os, "geteuid", return_value=0))

    def _hub(self, version: int) -> None:
        self.sysfs.usb("1-3", version=version)
        self.sysfs.netdev(CAN_LEFT, "1-3")

    def test_a_current_hub_is_left_alone(self) -> None:
        self._hub(hub_fw.FIRMWARE_VERSION)
        with patch.object(hub_fw, "flash", side_effect=AssertionError("flashed")):
            self.assertEqual(
                hub_fw.ensure_hub_firmware([CAN_LEFT], sysfs=self.sysfs.root), []
            )

    def test_an_outdated_hub_is_flashed_and_the_interfaces_awaited(self) -> None:
        self._hub(0x0100)
        with patch.object(
            hub_fw, "flash", return_value=hub_fw.FIRMWARE_VERSION
        ) as flash:
            flashed = hub_fw.ensure_hub_firmware([CAN_LEFT], sysfs=self.sysfs.root)
        flash.assert_called_once_with("1-3", sysfs=self.sysfs.root)
        self.assertEqual([h.port for h in flashed], ["1-3"])
        hub_fw._wait_for_channels.assert_called_once_with([CAN_LEFT])

    def test_a_hub_that_cannot_be_flashed_keeps_running_and_is_not_retried(
        self,
    ) -> None:
        # Still enumerated on the old firmware: the buses work, so the
        # bring-up continues — but loudly, and only one attempt per process.
        self._hub(0x0100)
        error = hub_fw.HubFirmwareError("firmware too old to update over USB")
        with (
            patch.object(hub_fw, "flash", side_effect=error) as flash,
            self.assertLogs(hub_fw._logger, "ERROR") as logs,
        ):
            self.assertEqual(
                hub_fw.ensure_hub_firmware([CAN_LEFT], sysfs=self.sysfs.root), []
            )
            self.assertEqual(
                hub_fw.ensure_hub_firmware([CAN_LEFT], sysfs=self.sysfs.root), []
            )
        flash.assert_called_once()
        self.assertIn("too old", "\n".join(logs.output))
        # A failed attempt may still have rebooted the hub: its interfaces
        # are awaited all the same (once — the retry is skipped).
        hub_fw._wait_for_channels.assert_called_once_with([CAN_LEFT])

    def test_a_hub_left_in_its_bootloader_fails_the_bring_up(self) -> None:
        self._hub(0x0100)

        def stuck(port: str, **_kw) -> int:
            self.sysfs.usb(port, vid=hub_fw.BOOT_VID, pid=hub_fw.BOOT_PID)
            raise hub_fw.HubFirmwareError("timed out waiting for the hub to restart")

        with (
            patch.object(hub_fw, "flash", side_effect=stuck),
            self.assertRaises(hub_fw.HubFirmwareError),
        ):
            hub_fw.ensure_hub_firmware([CAN_LEFT], sysfs=self.sysfs.root)

    def test_a_flash_that_did_not_take_is_reported_not_retried(self) -> None:
        # The hub came back working, on the old image: carry on, but loudly.
        self._hub(0x0100)
        with (
            patch.object(hub_fw, "flash", return_value=0x0100) as flash,
            self.assertLogs(hub_fw._logger, "ERROR"),
        ):
            self.assertEqual(
                hub_fw.ensure_hub_firmware([CAN_LEFT], sysfs=self.sysfs.root), []
            )
            hub_fw.ensure_hub_firmware([CAN_LEFT], sysfs=self.sysfs.root)
        flash.assert_called_once()
        # It rebooted, so its netdevs were recreated: wait for them.
        hub_fw._wait_for_channels.assert_called_once_with([CAN_LEFT])

    def test_a_sudo_flash_reports_the_real_failure(self) -> None:
        # Not root: the flash runs through sudo, and its own error (here the
        # BOOTSEL switch instructions) must reach the log, not a generic
        # "could not escalate".
        self._hub(0x0100)
        failed = subprocess.CompletedProcess(
            [],
            1,
            stdout="Flashing Axol hub firmware 2.00 at USB 1-3...\n",
            stderr="ERROR: firmware too old to update over USB. Flip the BOOTSEL\n",
        )
        with (
            patch.object(hub_fw.os, "geteuid", return_value=1000),
            patch("almond_axol.utils.sudo.run_root", return_value=failed),
            self.assertLogs(hub_fw._logger, "ERROR") as logs,
        ):
            hub_fw.ensure_hub_firmware([CAN_LEFT], sysfs=self.sysfs.root)
        output = "\n".join(logs.output)
        self.assertIn("Flip the BOOTSEL", output)
        self.assertNotIn("ERROR: firmware", output)

    def test_opt_out(self) -> None:
        self._hub(0x0100)
        with (
            patch.dict(os.environ, {"AXOL_HUB_FLASH": "0"}),
            patch.object(hub_fw, "flash", side_effect=AssertionError("flashed")),
        ):
            self.assertEqual(
                hub_fw.ensure_hub_firmware([CAN_LEFT], sysfs=self.sysfs.root), []
            )


class FlashTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sysfs = _FakeSysfs(Path(self.enterContext(tempfile.TemporaryDirectory())))
        self.enterContext(patch.object(hub_fw.os, "geteuid", return_value=0))

    def test_firmware_without_the_reboot_request_asks_for_a_manual_flash(self) -> None:
        # Builds before 2026-08-17 stall the vendor request.
        self.sysfs.usb("1-3", version=0x0100)
        with (
            patch.object(
                hub_fw, "_control", side_effect=OSError(errno.EPIPE, "Broken pipe")
            ),
            self.assertRaises(hub_fw.HubFirmwareError) as raised,
        ):
            hub_fw.flash("1-3", sysfs=self.sysfs.root)
        self.assertIn("BOOTSEL", str(raised.exception))

    def test_a_disk_that_vanishes_mid_copy_is_left_to_the_restart_check(
        self,
    ) -> None:
        # The bootrom reboots once the image lands, so the copy's tail I/O
        # can fail even though the flash took; flash() decides by waiting
        # for the hub, never by the copy's errors.
        mount = Path(self.enterContext(tempfile.TemporaryDirectory()))
        image = mount / "src.uf2"
        image.write_bytes(b"\0" * 512)
        mounted = subprocess.CompletedProcess([], 0, stdout=f"{mount}\n", stderr="")
        with (
            patch.object(hub_fw.subprocess, "run", return_value=mounted),
            patch.object(hub_fw.os, "fsync", side_effect=OSError(errno.EIO, "I/O")),
        ):
            hub_fw._copy_uf2("/dev/sda1", image)

    def _run_mount(self, mount_results: list[int]):
        """A subprocess.run stand-in: findmnt finds nothing, mount returns in turn."""
        calls: list[str] = []

        def run(cmd, **_k):
            calls.append(cmd[0])
            if cmd[0] == "mount":
                code = mount_results.pop(0) if mount_results else 32
                return subprocess.CompletedProcess(cmd, code, "", "mount: bad fs")
            return subprocess.CompletedProcess(
                cmd, 1 if cmd[0] == "findmnt" else 0, "", ""
            )

        return run, calls

    def test_a_failed_mount_is_a_firmware_error(self) -> None:
        image = Path(self.enterContext(tempfile.TemporaryDirectory())) / "a.uf2"
        image.write_bytes(b"")
        run, calls = self._run_mount([])
        clock = iter(range(0, 100, 2))
        with (
            patch.object(hub_fw.subprocess, "run", side_effect=run),
            patch.object(hub_fw.time, "monotonic", side_effect=lambda: next(clock)),
            patch.object(hub_fw.time, "sleep"),
            self.assertRaises(hub_fw.HubFirmwareError) as raised,
        ):
            hub_fw._copy_uf2("/dev/sda1", image)
        self.assertIn("bad fs", str(raised.exception))
        self.assertGreater(calls.count("mount"), 1)

    def test_a_mount_that_fails_as_the_disk_appears_is_retried(self) -> None:
        # The /dev node can trail the sysfs entry the disk was found by.
        image = Path(self.enterContext(tempfile.TemporaryDirectory())) / "a.uf2"
        image.write_bytes(b"\0" * 512)
        run, calls = self._run_mount([32, 0])
        with (
            patch.object(hub_fw.subprocess, "run", side_effect=run),
            patch.object(hub_fw.time, "sleep"),
            patch("builtins.open", side_effect=OSError(errno.ENOENT, "gone")),
        ):
            hub_fw._copy_uf2("/dev/sda1", image)
        self.assertEqual(calls.count("mount"), 2)
        self.assertIn("umount", calls)

    def test_refuses_without_root(self) -> None:
        with (
            patch.object(hub_fw.os, "geteuid", return_value=1000),
            self.assertRaises(hub_fw.HubFirmwareError),
        ):
            hub_fw.flash("1-3", sysfs=self.sysfs.root)

    def test_flashes_through_the_bootloader_disk(self) -> None:
        device = self.sysfs.usb("1-3", version=0x0100)
        copied: list[str] = []

        def reboot_to_bootloader(*_a, **_k) -> bytes:
            self.sysfs.usb("1-3", vid=hub_fw.BOOT_VID, pid=hub_fw.BOOT_PID)
            block = (
                device
                / "1-3:1.0"
                / "host0"
                / "target0:0:0"
                / "0:0:0:0"
                / "block"
                / "sda"
            )
            (block / "sda1").mkdir(parents=True)
            return b""

        def copy(disk: str, image: Path) -> None:
            copied.append(disk)
            self.sysfs.usb(
                "1-3",
                vid=hub_fw.HUB_VID,
                pid=hub_fw.HUB_PID,
                version=hub_fw.FIRMWARE_VERSION,
            )

        with (
            patch.object(hub_fw, "_control", side_effect=reboot_to_bootloader),
            patch.object(hub_fw, "_copy_uf2", side_effect=copy),
        ):
            version = hub_fw.flash("1-3", sysfs=self.sysfs.root)
        self.assertEqual(copied, ["/dev/sda1"])
        self.assertEqual(version, hub_fw.FIRMWARE_VERSION)

    def _bootloader_block(self, device: Path) -> Path:
        block = device / "1-3:1.0" / "host0" / "target0:0:0" / "0:0:0:0" / "block"
        (block / "sda").mkdir(parents=True)
        return block / "sda"

    def test_waits_for_the_partition_table_before_picking_the_disk(self) -> None:
        # The disk appears in sysfs before the kernel has read its partition
        # table; mounting the bare disk then fails and strands the hub in its
        # bootloader (seen on a Raspberry Pi 5).
        device = self.sysfs.usb("1-3", vid=hub_fw.BOOT_VID, pid=hub_fw.BOOT_PID)
        disk = self._bootloader_block(device)
        self.assertIsNone(hub_fw._bootloader_disk("1-3", self.sysfs.root))
        (disk / "sda1").mkdir()
        self.assertEqual(hub_fw._bootloader_disk("1-3", self.sysfs.root), "/dev/sda1")

    def test_a_disk_without_partitions_is_used_after_the_wait(self) -> None:
        device = self.sysfs.usb("1-3", vid=hub_fw.BOOT_VID, pid=hub_fw.BOOT_PID)
        self._bootloader_block(device)
        copied: list[str] = []

        def copy(disk: str, image: Path) -> None:
            copied.append(disk)
            self.sysfs.usb(
                "1-3",
                vid=hub_fw.HUB_VID,
                pid=hub_fw.HUB_PID,
                version=hub_fw.FIRMWARE_VERSION,
            )

        def no_partition(predicate, timeout_s, what):
            if "disk" in what:
                self.assertIsNone(predicate())
                raise hub_fw.HubFirmwareError(f"timed out waiting for {what}")
            return predicate()

        with (
            patch.object(hub_fw, "_wait_for", side_effect=no_partition),
            patch.object(hub_fw, "_copy_uf2", side_effect=copy),
        ):
            hub_fw.flash("1-3", sysfs=self.sysfs.root)
        self.assertEqual(copied, ["/dev/sda"])


class BringUpOrderTest(unittest.IsolatedAsyncioTestCase):
    """The firmware check runs in the bring-up window: before the purge and the buses."""

    def _axol(self):
        from almond_axol.robot.axol import AxolHardware

        opened: list[str] = []

        class _Bus:
            def __init__(self, channel: str) -> None:
                self.channel = channel

            async def start(self) -> None:
                opened.append(self.channel)

            def _add_listener(self, _listener) -> None:
                pass

        with (
            patch("almond_axol.robot.axol.CanBus", side_effect=_Bus),
            patch(
                "almond_axol.motor.motor.make_driver",
                side_effect=lambda *_a, **_k: SimpleNamespace(
                    kp_max=500.0,
                    kd_max=5.0,
                    set_feedback_callback=lambda _cb: None,
                ),
            ),
        ):
            axol = AxolHardware(left_channel=CAN_LEFT, right_channel=CAN_RIGHT)
        return axol, opened

    async def test_firmware_then_purge_then_buses(self) -> None:
        axol, opened = self._axol()
        events: list[str] = []

        def fake_ensure(channels: list[str]) -> list:
            self.assertEqual(opened, [])
            events.append(f"firmware:{','.join(channels)}")
            return []

        def fake_purge(channels: list[str]) -> list[str]:
            events.append("purge")
            return []

        with (
            patch("almond_axol.hub.ensure_hub_firmware", fake_ensure),
            patch.object(can_setup, "purge_stale_tx", fake_purge),
        ):
            await axol.connect()
        self.assertEqual(events, [f"firmware:{CAN_LEFT},{CAN_RIGHT}", "purge"])
        self.assertEqual(sorted(opened), sorted([CAN_LEFT, CAN_RIGHT]))

    async def test_a_failed_flash_refuses_the_connection(self) -> None:
        from almond_axol.motor import MotorError

        axol, opened = self._axol()
        with (
            patch(
                "almond_axol.hub.ensure_hub_firmware",
                side_effect=hub_fw.HubFirmwareError("hub did not restart"),
            ),
            patch.object(can_setup, "purge_stale_tx", side_effect=AssertionError),
            self.assertRaises(MotorError),
        ):
            await axol.connect()
        self.assertEqual(opened, [])


if __name__ == "__main__":
    unittest.main()
