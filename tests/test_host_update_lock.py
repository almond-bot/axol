from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock, patch

from almond_axol.utils import host_update_lock


class HostUpdateLockTest(unittest.TestCase):
    def test_state_directory_is_created_with_operator_traversal_only(self) -> None:
        state_dir = Mock()
        state_dir.stat.return_value = Mock(
            st_mode=0o040751,
            st_uid=0,
            st_gid=0,
        )
        with (
            patch.object(host_update_lock, "STATE_DIR", state_dir),
            patch.object(host_update_lock.os, "chown") as chown,
            patch.object(host_update_lock.os, "chmod") as chmod,
        ):
            host_update_lock._validate_directory()  # noqa: SLF001

        state_dir.mkdir.assert_called_once_with(
            mode=0o751,
            parents=False,
        )
        chown.assert_called_once_with(state_dir, 0, 0)
        chmod.assert_called_once_with(state_dir, 0o751)

    def test_separate_open_file_descriptions_cannot_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "update.lock"
            with (
                patch.object(host_update_lock, "LOCK_PATH", lock),
                patch.object(host_update_lock.os, "geteuid", return_value=0),
                patch.object(host_update_lock.os, "fchown"),
                patch.object(host_update_lock.os, "fchmod"),
                patch.object(host_update_lock, "_validate_directory"),
                patch.object(host_update_lock, "_validate_fd"),
                host_update_lock.host_update_lock(),
                self.assertRaisesRegex(
                    host_update_lock.HostUpdateLockError, "transaction is active"
                ),
            ):
                with host_update_lock.host_update_lock():
                    pass

    def test_validated_inherited_descriptor_is_reentrant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "update.lock"
            fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                with (
                    patch.object(host_update_lock, "LOCK_PATH", lock),
                    patch.object(host_update_lock.os, "geteuid", return_value=0),
                    patch.object(host_update_lock, "_validate_directory"),
                    patch.object(host_update_lock, "_validate_fd"),
                    patch.dict(
                        os.environ,
                        {host_update_lock.LOCK_ENV: str(fd)},
                        clear=False,
                    ),
                    host_update_lock.host_update_lock(),
                ):
                    # The context borrows rather than closes the installer's FD.
                    os.fstat(fd)
            finally:
                os.close(fd)

    def test_non_root_call_fails_before_touching_state(self) -> None:
        with (
            patch.object(host_update_lock.os, "geteuid", return_value=1000),
            patch.object(host_update_lock, "_validate_directory") as validate,
            self.assertRaisesRegex(
                host_update_lock.HostUpdateLockError, "requires root"
            ),
        ):
            with host_update_lock.host_update_lock():
                pass
        validate.assert_not_called()

    def test_holder_owns_the_lock_only_until_its_stdin_closes(self) -> None:
        events: list[str] = []

        @contextlib.contextmanager
        def fake_lock() -> Iterator[None]:
            events.append("locked")
            yield
            events.append("released")

        stdin = Mock()
        stdin.buffer.read.side_effect = lambda: events.append("waited") or b""
        stdout = io.StringIO()
        with (
            patch.object(host_update_lock, "host_update_lock", fake_lock),
            patch.object(host_update_lock.sys, "stdin", stdin),
            patch.object(host_update_lock.sys, "stdout", stdout),
        ):
            code = host_update_lock.hold_until_stdin_closes()

        self.assertEqual(code, 0)
        self.assertEqual(events, ["locked", "waited", "released"])
        self.assertEqual(stdout.getvalue(), f"{host_update_lock.HOLDER_READY}\n")

    def test_holder_reports_lock_errors_without_a_ready_line(self) -> None:
        @contextlib.contextmanager
        def busy_lock() -> Iterator[None]:
            raise host_update_lock.HostUpdateLockError("transaction is active")
            yield  # pragma: no cover - never reached

        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch.object(host_update_lock, "host_update_lock", busy_lock),
            patch.object(host_update_lock.sys, "stdout", stdout),
            patch.object(host_update_lock.sys, "stderr", stderr),
        ):
            code = host_update_lock.hold_until_stdin_closes()

        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("transaction is active", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
