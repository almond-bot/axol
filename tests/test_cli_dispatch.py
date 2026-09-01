from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from almond_axol import cli


class OptionalDependencyDispatchTest(unittest.TestCase):
    def test_missing_lerobot_has_an_actionable_cli_error(self) -> None:
        missing = ModuleNotFoundError("No module named 'lerobot'", name="lerobot")

        with (
            patch.object(cli.importlib, "import_module", side_effect=missing),
            self.assertRaisesRegex(
                SystemExit,
                r"axol mantis\.train requires.*almond-axol\[lerobot\]",
            ),
        ):
            cli._dispatch_draccus("mantis.train", ["--help"])

    def test_unrelated_missing_module_is_not_hidden(self) -> None:
        missing = ModuleNotFoundError("No module named 'pyzed'", name="pyzed")
        module = Mock()
        module.main.side_effect = missing

        with (
            patch.object(cli.importlib, "import_module", return_value=module),
            self.assertRaises(ModuleNotFoundError) as raised,
        ):
            cli._dispatch_draccus("collect-data", [])

        self.assertIs(raised.exception, missing)


if __name__ == "__main__":
    unittest.main()
