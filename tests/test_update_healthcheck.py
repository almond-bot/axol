from __future__ import annotations

import re
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from almond_axol.cli import update_healthcheck


class UpdateHealthcheckTest(unittest.TestCase):
    def test_cold_candidate_dwell_fits_inside_the_service_start_timeout(self) -> None:
        # A first boot imports torch/lerobot and the camera stack before it can
        # answer /api/health; 45s was not enough on a Jetson.
        self.assertGreaterEqual(update_healthcheck._HEALTH_TIMEOUT_S, 120.0)  # noqa: SLF001

        installer = (
            Path(__file__).resolve().parents[1] / "web" / "app" / "public" / "install"
        ).read_text()
        unit_start = installer.index("Description=Axol control panel (axol serve)")
        unit_end = installer.index("\nEOF\n", unit_start)
        unit = installer[unit_start:unit_end]
        self.assertIn("ExecStartPost=${BIN_DIR}/axol update-healthcheck", unit)
        match = re.search(r"^TimeoutStartSec=(\d+)$", unit, re.MULTILINE)
        self.assertIsNotNone(match, "the unit must not rely on systemd's 90s default")
        assert match is not None
        # ExecStartPost is part of the start job; leave headroom for ExecStartPre.
        self.assertGreater(int(match.group(1)), update_healthcheck._HEALTH_TIMEOUT_S)  # noqa: SLF001

    def test_ordinary_start_without_candidate_state_is_a_noop(self) -> None:
        read = Mock(return_value=None)
        with (
            patch.object(update_healthcheck.os, "geteuid", return_value=0),
            patch.object(update_healthcheck, "_safe_state_dir"),
            patch.object(update_healthcheck, "_read_transaction_file", read),
            patch.object(update_healthcheck, "_health_payload") as health,
            patch.object(update_healthcheck, "_remove_durable") as remove,
        ):
            update_healthcheck.verify_candidate()

        read.assert_called_once_with(
            update_healthcheck._VERIFYING_MARKER,
            required=False,  # noqa: SLF001
        )
        health.assert_not_called()
        remove.assert_not_called()

    def test_exact_version_and_stable_pid_commit_axol_attempt_only(self) -> None:
        remove = Mock()
        with (
            patch.object(update_healthcheck.os, "geteuid", return_value=0),
            patch.object(update_healthcheck, "_safe_state_dir"),
            patch.object(
                update_healthcheck,
                "_read_transaction_file",
                side_effect=("0.1.37", "0.1.37"),
            ),
            patch.object(
                update_healthcheck.importlib.metadata,
                "version",
                return_value="0.1.37",
            ),
            patch.object(update_healthcheck, "_main_pid", return_value=42),
            patch.object(
                update_healthcheck,
                "_health_payload",
                return_value={
                    "ready": True,
                    "version": "0.1.37",
                    "pid": 42,
                },
            ),
            patch.object(
                update_healthcheck.time,
                "monotonic",
                side_effect=(0.0, 0.0, 0.0, 0.1, 0.1),
            ),
            patch.object(update_healthcheck.time, "sleep"),
            patch.object(update_healthcheck, "_HEALTH_DWELL_S", 0.0),
            patch.object(update_healthcheck, "_remove_durable", remove),
        ):
            update_healthcheck.verify_candidate()

        self.assertEqual(
            remove.call_args_list,
            [
                call(update_healthcheck._VERIFYING_MARKER),  # noqa: SLF001
            ],
        )

    def test_wrong_candidate_version_never_clears_guard(self) -> None:
        with (
            patch.object(update_healthcheck.os, "geteuid", return_value=0),
            patch.object(update_healthcheck, "_safe_state_dir"),
            patch.object(
                update_healthcheck,
                "_read_transaction_file",
                side_effect=("0.1.38", "0.1.38"),
            ),
            patch.object(
                update_healthcheck.importlib.metadata,
                "version",
                return_value="0.1.37",
            ),
            patch.object(update_healthcheck, "_health_payload") as health,
            patch.object(update_healthcheck, "_remove_durable") as remove,
        ):
            with self.assertRaisesRegex(
                update_healthcheck.UpdateHealthError, "version mismatch"
            ):
                update_healthcheck.verify_candidate()

        health.assert_not_called()
        remove.assert_not_called()

    def test_endpoint_pid_must_match_systemd_main_pid(self) -> None:
        with (
            patch.object(update_healthcheck.os, "geteuid", return_value=0),
            patch.object(update_healthcheck, "_safe_state_dir"),
            patch.object(
                update_healthcheck,
                "_read_transaction_file",
                side_effect=("0.1.37", "0.1.37"),
            ),
            patch.object(
                update_healthcheck.importlib.metadata,
                "version",
                return_value="0.1.37",
            ),
            patch.object(update_healthcheck, "_main_pid", return_value=42),
            patch.object(
                update_healthcheck,
                "_health_payload",
                return_value={
                    "ready": True,
                    "version": "0.1.37",
                    "pid": 41,
                },
            ),
            patch.object(
                update_healthcheck.time,
                "monotonic",
                side_effect=(0.0, 1.0),
            ),
            patch.object(update_healthcheck.time, "sleep"),
            patch.object(update_healthcheck, "_HEALTH_TIMEOUT_S", 0.5),
            patch.object(update_healthcheck, "_remove_durable") as remove,
        ):
            with self.assertRaisesRegex(
                update_healthcheck.UpdateHealthError, "did not remain healthy"
            ):
                update_healthcheck.verify_candidate()

        remove.assert_not_called()


if __name__ == "__main__":
    unittest.main()
