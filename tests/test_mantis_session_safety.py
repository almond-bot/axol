from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from almond_axol.cli import mantis_session


class MantisSessionSafetyTest(unittest.TestCase):
    def test_retired_service_is_stopped_before_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_service = root / "old.service"
            new_service = root / "new.service"
            old_service.write_text("legacy")
            with (
                patch.object(mantis_session, "_PRE_MANTIS_SERVICE_PATH", old_service),
                patch.object(mantis_session, "_SERVICE_PATH", new_service),
                patch.object(mantis_session.shutil, "which", return_value="/opt/axol"),
                patch.object(mantis_session, "run_root") as run_root,
                patch.object(
                    mantis_session.subprocess,
                    "run",
                    return_value=type("Result", (), {"stdout": "operator\n"})(),
                ),
            ):
                mantis_session._install()

        calls = run_root.call_args_list
        self.assertEqual(
            calls[0].args[0],
            ["systemctl", "stop", mantis_session._PRE_MANTIS_SERVICE_NAME],
        )
        self.assertEqual(calls[0].kwargs, {"check": True})
        self.assertEqual(
            calls[1].args[0],
            ["systemctl", "disable", mantis_session._PRE_MANTIS_SERVICE_NAME],
        )
        self.assertEqual(calls[1].kwargs, {"check": True})
        self.assertEqual(calls[2].args[0], ["rm", "-f", str(old_service)])

    def test_retired_service_shutdown_failure_keeps_definition(self) -> None:
        for failed_action in ("stop", "disable"):
            with (
                self.subTest(failed_action=failed_action),
                tempfile.TemporaryDirectory() as directory,
            ):
                old_service = Path(directory) / "old.service"
                old_service.write_text("legacy")

                def run_root(command: list[str], **_kwargs: object) -> object:
                    if command == [
                        "systemctl",
                        failed_action,
                        mantis_session._PRE_MANTIS_SERVICE_NAME,
                    ]:
                        raise RuntimeError(f"{failed_action} denied")
                    return object()

                with (
                    patch.object(
                        mantis_session, "_PRE_MANTIS_SERVICE_PATH", old_service
                    ),
                    patch.object(
                        mantis_session.shutil, "which", return_value="/opt/axol"
                    ),
                    patch.object(mantis_session, "run_root", side_effect=run_root),
                    patch.object(
                        mantis_session.subprocess,
                        "run",
                        return_value=type("Result", (), {"stdout": "operator\n"})(),
                    ),
                    self.assertRaisesRegex(RuntimeError, f"{failed_action} denied"),
                ):
                    mantis_session._install()

                self.assertTrue(old_service.exists())


if __name__ == "__main__":
    unittest.main()
