"""``axol hub.info`` / ``axol hub.flash``: what docs/cli/hub-*.mdx promise.

The hub lookup and the flash itself are mocked (they need sysfs, usbfs and
root); covered here is the command-line behaviour: parsing, which hubs get
flashed by default, a hub in its bootloader, ``--port`` / ``--force``, and the
exit status.
"""

from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

from almond_axol.cli import build_parser
from almond_axol.cli import hub as hub_cli
from almond_axol.hub import firmware as hub_fw

_CURRENT = hub_fw.Hub(port="1-3", serial="AAA", version=hub_fw.FIRMWARE_VERSION)
_OLD = hub_fw.Hub(port="1-4", serial="BBB", version=0x0100)


class HubCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bootloaders = self.enterContext(
            patch.object(hub_fw, "bootloader_ports", return_value=[])
        )
        self.flash = self.enterContext(
            patch.object(hub_fw, "flash", return_value=hub_fw.FIRMWARE_VERSION)
        )
        self.euid = self.enterContext(
            patch.object(hub_cli.os, "geteuid", return_value=0)
        )

    def _run(self, *argv: str) -> str:
        args = build_parser().parse_args(list(argv))
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            args.func(args)
        return out.getvalue()

    def _hubs(self, *hubs: hub_fw.Hub):
        return patch.object(hub_fw, "attached_hubs", return_value=list(hubs))

    def _stdin(self, *, tty: bool, answer: str = ""):
        stdin = self.enterContext(patch.object(hub_cli.sys, "stdin"))
        stdin.isatty.return_value = tty
        return patch("builtins.input", return_value=answer)

    def test_parser_flags(self) -> None:
        args = build_parser().parse_args(["hub.flash", "--port", "1-3", "--force"])
        self.assertEqual((args.port, args.force), ("1-3", True))
        self.assertIs(args.func, hub_cli.run_flash)
        self.assertIs(build_parser().parse_args(["hub.info"]).func, hub_cli.run_info)

    def test_info_reports_version_and_state(self) -> None:
        with (
            self._hubs(_CURRENT, _OLD),
            patch.object(hub_fw, "reset_info", return_value=None),
        ):
            out = self._run("hub.info")
        self.assertIn("hub AAA (USB 1-3)", out)
        self.assertIn("(current;", out)
        self.assertIn("1.00 (outdated;", out)
        self.assertIn("last reset unavailable", out)

    def test_info_without_a_hub_exits_nonzero(self) -> None:
        with self._hubs(), self.assertRaises(SystemExit) as cm:
            self._run("hub.info")
        self.assertEqual(cm.exception.code, 1)

    def test_info_points_a_bootloader_at_hub_flash(self) -> None:
        self.bootloaders.return_value = ["1-5"]
        with self._hubs(), self.assertRaises(SystemExit) as cm:
            self._run("hub.info")
        self.assertEqual(cm.exception.code, 1)
        with (
            self._hubs(_CURRENT),
            patch.object(hub_fw, "reset_info", return_value=None),
        ):
            out = self._run("hub.info")
        self.assertIn("bootloader at USB 1-5", out)
        self.assertIn("`axol hub.flash --port 1-5`", out)

    def test_flash_without_any_hub_exits_nonzero(self) -> None:
        with self._hubs(), self.assertRaises(SystemExit) as cm:
            self._run("hub.flash")
        self.assertEqual(cm.exception.code, 1)
        self.flash.assert_not_called()

    def test_flash_takes_a_confirmed_bootloader(self) -> None:
        # The Mantis hub that never enumerated, replugged with its BOOTSEL switch on.
        self.bootloaders.return_value = ["1-5"]
        with self._hubs(), self._stdin(tty=True, answer="y"):
            self._run("hub.flash")
        self.flash.assert_called_once_with("1-5")

    def test_flash_leaves_an_unconfirmed_bootloader_alone(self) -> None:
        self.bootloaders.return_value = ["1-5"]
        with self._hubs(_OLD), self._stdin(tty=True, answer="n"):
            self._run("hub.flash")
        self.flash.assert_called_once_with("1-4")
        self.flash.reset_mock()
        with self._hubs(), self._stdin(tty=False):
            out = self._run("hub.flash")
        self.flash.assert_not_called()
        self.assertIn("`axol hub.flash --port 1-5`", out)

    def test_flash_defaults_to_outdated_hubs_only(self) -> None:
        with self._hubs(_CURRENT, _OLD):
            self._run("hub.flash")
        self.flash.assert_called_once_with("1-4")

    def test_flash_with_nothing_outdated_does_nothing(self) -> None:
        with self._hubs(_CURRENT):
            out = self._run("hub.flash")
        self.flash.assert_not_called()
        self.assertIn("nothing to do", out)

    def test_force_reflashes_a_current_hub(self) -> None:
        with self._hubs(_CURRENT):
            self._run("hub.flash", "--force")
        self.flash.assert_called_once_with("1-3")

    def test_port_flashes_that_hub_without_a_lookup(self) -> None:
        with patch.object(hub_fw, "attached_hubs") as lookup:
            self._run("hub.flash", "--port", "2-1")
        lookup.assert_not_called()
        self.flash.assert_called_once_with("2-1")

    def test_flash_as_a_normal_user_escalates_per_port(self) -> None:
        # The hubs are chosen (and any question asked) unprivileged; only
        # the flash itself runs through sudo.
        self.euid.return_value = 1000
        with (
            self._hubs(_CURRENT, _OLD),
            patch.object(
                hub_fw, "flash_via_sudo", return_value=hub_fw.FIRMWARE_VERSION
            ) as via_sudo,
        ):
            out = self._run("hub.flash")
        via_sudo.assert_called_once_with("1-4")
        self.flash.assert_not_called()
        self.assertIn("Done", out)

    def test_flash_failure_exits_nonzero(self) -> None:
        self.flash.side_effect = hub_fw.HubFirmwareError("flip the BOOTSEL switch")
        with self.assertRaises(SystemExit) as cm:
            self._run("hub.flash", "--port", "2-1")
        self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
