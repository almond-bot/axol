from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from almond_axol.utils import shared_install


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


class SharedInstallTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name).resolve()
        # The layout a self-update leaves under serve's 027 umask.
        self.tools = root / "tools"
        self.python = root / "python"
        self.venv = self.tools / "almond-axol"
        self.base = self.python / "cpython-3.13"
        (self.venv / "bin").mkdir(parents=True)
        (self.base / "bin").mkdir(parents=True)
        self.script = self.venv / "bin" / "axol"
        self.module = self.venv / "pkg.py"
        self.interpreter = self.base / "bin" / "python3.13"
        for path, mode in (
            (self.script, 0o750),
            (self.module, 0o640),
            (self.interpreter, 0o750),
        ):
            path.write_text("")
            path.chmod(mode)
        (self.venv / "bin" / "python").symlink_to(self.interpreter)
        for path in (self.venv / "bin", self.venv, self.base / "bin", self.base):
            path.chmod(0o750)
        self.unrelated = root / "unrelated"
        self.unrelated.write_text("")
        self.unrelated.chmod(0o600)

    def _share(self, env: dict[str, str], *, euid: int = 0) -> None:
        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(shared_install.os, "geteuid", return_value=euid),
            patch.object(shared_install.sys, "prefix", str(self.venv)),
            patch.object(shared_install.sys, "base_prefix", str(self.base)),
        ):
            shared_install.share()

    def _env(self) -> dict[str, str]:
        return {
            "UV_TOOL_DIR": str(self.tools),
            "UV_PYTHON_INSTALL_DIR": str(self.python),
        }

    def test_root_only_venv_is_reopened_to_every_user(self) -> None:
        self._share(self._env())

        self.assertEqual(_mode(self.venv), 0o755)
        self.assertEqual(_mode(self.venv / "bin"), 0o755)
        self.assertEqual(_mode(self.script), 0o755)
        # a+rX: data files get read, never execute.
        self.assertEqual(_mode(self.module), 0o644)
        self.assertEqual(_mode(self.base), 0o755)
        self.assertEqual(_mode(self.interpreter), 0o755)
        self.assertEqual(_mode(self.unrelated), 0o600)

    def test_non_root_run_changes_nothing(self) -> None:
        self._share(self._env(), euid=1000)

        self.assertEqual(_mode(self.venv), 0o750)
        self.assertEqual(_mode(self.module), 0o640)

    def test_tree_that_does_not_hold_this_install_is_left_alone(self) -> None:
        other = self.tools.parent / "other"
        other.mkdir(mode=0o750)
        self._share({"UV_TOOL_DIR": str(other), "UV_PYTHON_INSTALL_DIR": "relative"})

        self.assertEqual(_mode(other), 0o750)
        self.assertEqual(_mode(self.venv), 0o750)
        self.assertEqual(_mode(self.base), 0o750)


if __name__ == "__main__":
    unittest.main()
