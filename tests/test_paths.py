from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from almond_axol.utils.paths import almond_home, almond_path


class AlmondHomeTest(unittest.TestCase):
    def test_defaults_to_dot_almond_in_callers_home(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(Path, "home", return_value=Path("/home/operator")),
        ):
            self.assertEqual(almond_home(), Path("/home/operator/.almond"))
            self.assertEqual(
                almond_path("tracker", "config.json"),
                Path("/home/operator/.almond/tracker/config.json"),
            )

    def test_environment_override_wins(self) -> None:
        with (
            patch.dict(os.environ, {"ALMOND_HOME": "/srv/axol state"}, clear=True),
            patch.object(Path, "home", return_value=Path("/root")),
        ):
            self.assertEqual(almond_home(), Path("/srv/axol state"))
            self.assertEqual(
                almond_path("mantis", "tcp_transform.json"),
                Path("/srv/axol state/mantis/tcp_transform.json"),
            )

    def test_empty_environment_override_preserves_legacy_default(self) -> None:
        with (
            patch.dict(os.environ, {"ALMOND_HOME": ""}, clear=True),
            patch.object(Path, "home", return_value=Path("/home/operator")),
        ):
            self.assertEqual(almond_home(), Path("/home/operator/.almond"))

    def test_state_consumers_share_environment_override(self) -> None:
        code = """
from almond_axol.constants import CAN_BRINGUP_SCRIPT
from almond_axol.mantis.calibration import MANTIS_TCP_TRANSFORM_FILE
from almond_axol.serve.settings import SETTINGS_PATH
from almond_axol.tracker.config import TRACKER_CONFIG_FILE
from almond_axol.tracker.ultimate import ULTIMATE_WIFI_CONFIG_FILE
from almond_axol.utils.certs import CERTFILE
print(CAN_BRINGUP_SCRIPT)
print(MANTIS_TCP_TRANSFORM_FILE)
print(SETTINGS_PATH)
print(TRACKER_CONFIG_FILE)
print(ULTIMATE_WIFI_CONFIG_FILE)
print(CERTFILE)
"""
        env = os.environ.copy()
        env["ALMOND_HOME"] = "/srv/operator state"
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "/srv/operator state/can/startup.sh",
                "/srv/operator state/mantis/tcp_transform.json",
                "/srv/operator state/settings.json",
                "/srv/operator state/tracker/config.json",
                "/srv/operator state/tracker/ultimate_wifi.json",
                "/srv/operator state/vr/certs/cert.pem",
            ],
        )


if __name__ == "__main__":
    unittest.main()
