"""``axol hub.info`` / ``axol hub.flash``: what docs/cli/hub-*.mdx promise.

The hub lookup and the flash itself are mocked (they need sysfs, usbfs and
root); covered here is the command-line behaviour: parsing, which hubs get
flashed by default, ``--port`` / ``--force``, and the exit status.
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
        self.enterContext(
            patch.object(hub_cli, "_arm_channels", return_value=["a", "b"])
        )
        self.flash = self.enterContext(
            patch.object(hub_fw, "flash", return_value=hub_fw.FIRMWARE_VERSION)
        )

    def _run(self, *argv: str) -> str:
        args = build_parser().parse_args(list(argv))
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            args.func(args)
        return out.getvalue()

    def _hubs(self, *hubs: hub_fw.Hub):
        return patch.object(hub_fw, "hubs_for_channels", return_value=list(hubs))

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
        with patch.object(hub_fw, "hubs_for_channels") as lookup:
            self._run("hub.flash", "--port", "2-1")
        lookup.assert_not_called()
        self.flash.assert_called_once_with("2-1")

    def test_flash_failure_exits_nonzero(self) -> None:
        self.flash.side_effect = hub_fw.HubFirmwareError("hold BOOTSEL")
        with self.assertRaises(SystemExit) as cm:
            self._run("hub.flash", "--port", "2-1")
        self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
