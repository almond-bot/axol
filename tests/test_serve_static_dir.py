"""Where ``axol serve`` finds the web UI, and how release wheels ship it."""

from __future__ import annotations

import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from almond_axol.cli import serve

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGED_UI = "almond_axol/serve/ui"


class FindStaticDirTest(unittest.TestCase):
    def _bundle(self, root: Path, name: str, *, built: bool) -> Path:
        path = root / name
        path.mkdir()
        if built:
            (path / "index.html").write_text("<!doctype html>")
        return path

    def test_candidates_are_checkout_build_then_packaged_ui(self) -> None:
        checkout, packaged = serve._static_dir_candidates()
        self.assertEqual(checkout, REPO_ROOT / "web" / "app" / "dist")
        self.assertEqual(packaged, REPO_ROOT / PACKAGED_UI)

    def test_prefers_checkout_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dist = self._bundle(root, "dist", built=True)
            ui = self._bundle(root, "ui", built=True)
            with mock.patch.object(
                serve, "_static_dir_candidates", return_value=(dist, ui)
            ):
                self.assertEqual(serve._find_static_dir(), dist)

    def test_falls_back_to_packaged_ui(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dist = self._bundle(root, "dist", built=False)
            ui = self._bundle(root, "ui", built=True)
            with mock.patch.object(
                serve, "_static_dir_candidates", return_value=(dist, ui)
            ):
                self.assertEqual(serve._find_static_dir(), ui)

    def test_none_without_a_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = (root / "dist", self._bundle(root, "ui", built=False))
            with mock.patch.object(
                serve, "_static_dir_candidates", return_value=missing
            ):
                self.assertIsNone(serve._find_static_dir())


class PackagedUiConfigTest(unittest.TestCase):
    """The packaged UI path must agree across serve.py, hatch and the release job."""

    def test_hatch_includes_the_gitignored_ui(self) -> None:
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        artifacts = pyproject["tool"]["hatch"]["build"]["artifacts"]
        self.assertIn(f"/{PACKAGED_UI}/", artifacts)
        self.assertIn(f"/{PACKAGED_UI}/", (REPO_ROOT / ".gitignore").read_text())

    def test_release_workflow_stages_the_ui(self) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "publish.yml").read_text()
        self.assertIn(f"cp -R web/app/dist {PACKAGED_UI}", workflow)
        self.assertIn(f"{PACKAGED_UI}/index.html", workflow)


if __name__ == "__main__":
    unittest.main()
