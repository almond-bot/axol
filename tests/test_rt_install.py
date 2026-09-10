from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from almond_axol.rt import install


class _FakeDist:
    def __init__(self, direct_url: dict | None) -> None:
        self._direct_url = direct_url

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        return None if self._direct_url is None else json.dumps(self._direct_url)


def _checkout(with_crate: bool = True) -> Path:
    root = Path(tempfile.mkdtemp())
    if with_crate:
        crate = root / "rust" / "axol-rt"
        crate.mkdir(parents=True)
        (crate / "Cargo.toml").write_text('[package]\nname = "axol-rt"\n')
    return root


class SourceRefTest(unittest.TestCase):
    def setUp(self) -> None:
        p = patch.dict(os.environ, {}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        os.environ.pop(install._SOURCE_ENV, None)

    def _with_dist(self, direct_url: dict | None):
        return patch.object(install, "distribution", return_value=_FakeDist(direct_url))

    def test_git_install_uses_pinned_commit(self) -> None:
        # Exactly what uv writes for `uv tool install git+...@chemical-speak`.
        data = {
            "url": "https://github.com/almond-bot/axol",
            "vcs_info": {
                "vcs": "git",
                "commit_id": "5344412bf67b7b9ad77b357645aa8fbdc34d047a",
                "requested_revision": "chemical-speak",
            },
        }
        with self._with_dist(data):
            self.assertEqual(
                install._source_ref(),
                (
                    "https://github.com/almond-bot/axol",
                    "5344412bf67b7b9ad77b357645aa8fbdc34d047a",
                ),
            )

    def test_git_prefix_is_stripped(self) -> None:
        data = {
            "url": "git+https://github.com/almond-bot/axol",
            "vcs_info": {"vcs": "git", "commit_id": "abc123"},
        }
        with self._with_dist(data):
            self.assertEqual(
                install._source_ref(), ("https://github.com/almond-bot/axol", "abc123")
            )

    def test_index_install_uses_release_tag(self) -> None:
        with (
            self._with_dist(None),
            patch.object(install, "version", return_value="0.1.36"),
        ):
            self.assertEqual(install._source_ref(), (install._REPO_URL, "v0.1.36"))

    def test_directory_install_builds_that_checkout(self) -> None:
        # `uv tool install /path/to/axol` (non-editable): the package is in the
        # tool env, so _repo_crate misses, but direct_url.json names the tree.
        root = _checkout()
        data = {"url": root.as_uri(), "dir_info": {}}
        with self._with_dist(data):
            self.assertEqual(install._source_ref(), root / "rust" / "axol-rt")

    def test_directory_install_without_crate_is_unresolved(self) -> None:
        root = _checkout(with_crate=False)
        data = {"url": root.as_uri(), "dir_info": {}}
        with self._with_dist(data):
            self.assertIsNone(install._source_ref())

    def test_archive_install_is_unresolved(self) -> None:
        data = {
            "url": "file:///tmp/almond_axol-0.1.35-py3-none-any.whl",
            "archive_info": {"hash": "sha256=deadbeef"},
        }
        with self._with_dist(data):
            self.assertIsNone(install._source_ref())

    def test_missing_distribution_is_unresolved(self) -> None:
        with patch.object(
            install, "distribution", side_effect=install.PackageNotFoundError
        ):
            self.assertIsNone(install._source_ref())

    def test_override_path_to_checkout(self) -> None:
        root = _checkout()
        os.environ[install._SOURCE_ENV] = str(root)
        with self._with_dist(None):
            self.assertEqual(install._source_ref(), root / "rust" / "axol-rt")

    def test_override_path_to_crate_itself(self) -> None:
        crate = _checkout() / "rust" / "axol-rt"
        os.environ[install._SOURCE_ENV] = str(crate)
        with self._with_dist(None):
            self.assertEqual(install._source_ref(), crate)

    def test_override_path_without_crate_errors(self) -> None:
        os.environ[install._SOURCE_ENV] = str(_checkout(with_crate=False))
        with self._with_dist(None), self.assertRaises(RuntimeError):
            install._source_ref()

    def test_override_url_at_ref_beats_metadata(self) -> None:
        os.environ[install._SOURCE_ENV] = (
            "https://github.com/almond-bot/axol@chemical-speak"
        )
        data = {
            "url": "https://github.com/almond-bot/axol",
            "vcs_info": {"vcs": "git", "commit_id": "abc123"},
        }
        with self._with_dist(data):
            self.assertEqual(
                install._source_ref(),
                ("https://github.com/almond-bot/axol", "chemical-speak"),
            )

    def test_override_ssh_url_splits_on_last_at(self) -> None:
        os.environ[install._SOURCE_ENV] = "git@github.com:almond-bot/axol.git@v0.2.0"
        with self._with_dist(None):
            self.assertEqual(
                install._source_ref(), ("git@github.com:almond-bot/axol.git", "v0.2.0")
            )

    def test_override_without_ref_errors(self) -> None:
        os.environ[install._SOURCE_ENV] = "https://github.com/almond-bot/axol"
        with self._with_dist(None), self.assertRaises(RuntimeError):
            install._source_ref()


class RunTest(unittest.TestCase):
    """``run`` dispatch for tool installs (no dev checkout on disk)."""

    def setUp(self) -> None:
        p = patch.dict(os.environ, {}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        os.environ.pop(install._SOURCE_ENV, None)
        self.dest_dir = Path(tempfile.mkdtemp())
        os.environ["UV_TOOL_BIN_DIR"] = str(self.dest_dir)
        for name in ("_repo_crate",):
            p = patch.object(install, name, return_value=None)
            p.start()
            self.addCleanup(p.stop)
        for name in ("_grant_realtime",):
            p = patch.object(install, name)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(install, "_ensure_toolchain", return_value="cargo")
        p.start()
        self.addCleanup(p.stop)
        p = patch.object(install, "_SRC_CACHE", Path(tempfile.mkdtemp()) / "src")
        p.start()
        self.addCleanup(p.stop)

    def _fake_build(self, crate: Path, cargo: str) -> Path:
        out = crate / "target" / "release"
        out.mkdir(parents=True, exist_ok=True)
        binary = out / "axol-rt"
        binary.write_text("#!/bin/sh\n")
        return binary

    def test_unresolved_source_names_the_override(self) -> None:
        with (
            patch.object(install, "_source_ref", return_value=None),
            self.assertRaises(RuntimeError) as ctx,
        ):
            install.run()
        self.assertIn(install._SOURCE_ENV, str(ctx.exception))

    def test_local_crate_builds_in_place_and_installs(self) -> None:
        crate = _checkout() / "rust" / "axol-rt"
        with (
            patch.object(install, "_source_ref", return_value=crate),
            patch.object(install, "_build", side_effect=self._fake_build) as build,
            patch.object(install, "_fetch_crate") as fetch,
        ):
            install.run()
        build.assert_called_once_with(crate, "cargo")
        fetch.assert_not_called()
        dest = self.dest_dir / "axol-rt"
        self.assertTrue(dest.is_file())
        self.assertTrue(os.access(dest, os.X_OK))
        self.assertEqual(install._stamp_for(dest).read_text().strip(), f"path:{crate}")

    def test_git_source_fetches_then_skips_when_stamped(self) -> None:
        crate = _checkout() / "rust" / "axol-rt"
        with (
            patch.object(
                install, "_source_ref", return_value=(install._REPO_URL, "v0.2.0")
            ),
            patch.object(
                install, "_fetch_crate", return_value=(crate, "abc123")
            ) as fetch,
            patch.object(install, "_build", side_effect=self._fake_build) as build,
        ):
            install.run()
            fetch.assert_called_once_with(install._REPO_URL, "v0.2.0")
            build.assert_called_once()

            install.run()  # same ref, binary present: no refetch / rebuild
            fetch.assert_called_once()
            build.assert_called_once()

    def test_override_always_rebuilds(self) -> None:
        crate = _checkout() / "rust" / "axol-rt"
        os.environ[install._SOURCE_ENV] = f"{install._REPO_URL}@chemical-speak"
        with (
            patch.object(
                install, "_fetch_crate", return_value=(crate, "abc123")
            ) as fetch,
            patch.object(install, "_build", side_effect=self._fake_build) as build,
        ):
            install.run()
            install.run()
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(build.call_count, 2)
        fetch.assert_called_with(install._REPO_URL, "chemical-speak")


if __name__ == "__main__":
    unittest.main()
