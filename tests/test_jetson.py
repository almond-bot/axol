from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from almond_axol.utils import jetson


class JetsonPowerModeTest(unittest.TestCase):
    def _preferred_mode(self, config: str) -> tuple[str, str]:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "nvpmodel.conf"
            config_path.write_text(config)
            with patch.object(jetson, "_NVPMODEL_CONFIG", config_path):
                return jetson._preferred_max_power_mode()  # noqa: SLF001

    def test_prefers_maxn_super_using_its_configured_id(self) -> None:
        self.assertEqual(
            self._preferred_mode(
                """
                < POWER_MODEL ID=0 NAME=15W >
                < POWER_MODEL ID=1 NAME=25W >
                < POWER_MODEL ID=2 NAME=MAXN_SUPER >
                """
            ),
            ("2", "MAXN SUPER"),
        )

    def test_uses_configured_maxn_when_super_is_unavailable(self) -> None:
        self.assertEqual(
            self._preferred_mode(
                """
                < POWER_MODEL ID=0 NAME=MODE_15W >
                < POWER_MODEL NAME=MAXN ID=3 >
                """
            ),
            ("3", "MAXN"),
        )

    def test_falls_back_to_mode_zero_when_names_are_unavailable(self) -> None:
        self.assertEqual(
            self._preferred_mode("< POWER_MODEL ID=1 NAME=MODE_15W >"),
            ("0", "MAXN"),
        )

    def test_sets_maxn_super_instead_of_mode_zero(self) -> None:
        escalator = Mock()
        escalator.run.return_value = (True, "")
        with (
            patch.object(jetson, "_is_jetson", return_value=True),
            patch.object(jetson.shutil, "which", return_value="/usr/sbin/nvpmodel"),
            patch.object(
                jetson,
                "_preferred_max_power_mode",
                return_value=("2", "MAXN SUPER"),
            ),
            patch.object(jetson, "_query_power_mode", side_effect=["1", "2"]),
        ):
            jetson._set_max_power_mode(escalator)  # noqa: SLF001

        escalator.run.assert_called_once_with(
            ["/usr/sbin/nvpmodel", "-m", "2"], input_text="n\n"
        )
        escalator.write.assert_not_called()

    def test_persists_selected_super_mode_when_switch_needs_reboot(self) -> None:
        escalator = Mock()
        escalator.run.return_value = (True, "")
        escalator.write.return_value = (True, "")
        with (
            patch.object(jetson, "_is_jetson", return_value=True),
            patch.object(jetson.shutil, "which", return_value="/usr/sbin/nvpmodel"),
            patch.object(
                jetson,
                "_preferred_max_power_mode",
                return_value=("2", "MAXN SUPER"),
            ),
            patch.object(jetson, "_query_power_mode", side_effect=["1", "1"]),
        ):
            jetson._set_max_power_mode(escalator)  # noqa: SLF001

        self.assertEqual(
            escalator.method_calls,
            [
                call.run(["/usr/sbin/nvpmodel", "-m", "2"], input_text="n\n"),
                call.write(jetson._NVPMODEL_STATUS, "pmode:0002"),  # noqa: SLF001
            ],
        )


if __name__ == "__main__":
    unittest.main()
