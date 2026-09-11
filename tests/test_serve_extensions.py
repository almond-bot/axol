"""Downstream app extensions and advertised pages (``almond_axol.serve.extensions``)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient

from almond_axol.serve import (
    PanelPage,
    create_app,
    register_app_extension,
    register_page,
)
from almond_axol.serve.extensions import (
    app_extensions,
    clear_extensions,
    panel_pages,
)


def _add_page_route(app: FastAPI) -> None:
    @app.get("/my-page", response_model=None)
    def my_page() -> HTMLResponse:
        return HTMLResponse("EXTENSION PAGE")

    @app.get("/api/my-page/status")
    def my_status() -> dict[str, bool]:
        return {"ok": True}


class PanelPageTest(unittest.TestCase):
    def test_validates_path(self) -> None:
        PanelPage("Datasets", "/datasets")
        PanelPage("Nested", "/tools/datasets?tab=recent")
        for bad in ("datasets", "//evil.example", "/api/datasets", "http://x/y", ""):
            with self.subTest(path=bad):
                with self.assertRaises(ValueError):
                    PanelPage("x", bad)
        with self.assertRaises(ValueError):
            PanelPage("  ", "/ok")

    def test_to_dict_omits_empty_description(self) -> None:
        self.assertEqual(
            PanelPage("Datasets", "/datasets").to_dict(),
            {"label": "Datasets", "path": "/datasets"},
        )
        self.assertEqual(
            PanelPage("Datasets", "/datasets", "Browse recorded datasets").to_dict(),
            {
                "label": "Datasets",
                "path": "/datasets",
                "description": "Browse recorded datasets",
            },
        )


class RegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        clear_extensions()
        self.addCleanup(clear_extensions)

    def test_registration_is_idempotent_and_ordered(self) -> None:
        def a(app: FastAPI) -> None: ...

        def b(app: FastAPI) -> None: ...

        register_app_extension(a)
        register_app_extension(b)
        register_app_extension(a)
        self.assertEqual(app_extensions(), (a, b))

        register_page(PanelPage("One", "/one"))
        register_page(PanelPage("Two", "/two"))
        register_page(PanelPage("One again", "/one"))
        self.assertEqual([page.label for page in panel_pages()], ["One again", "Two"])


class CreateAppExtensionTest(unittest.TestCase):
    def setUp(self) -> None:
        clear_extensions()
        self.addCleanup(clear_extensions)
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.static = Path(self._temporary.name) / "static"
        self.static.mkdir()
        (self.static / "index.html").write_text("SPA INDEX", encoding="utf-8")

    def test_extension_routes_win_over_the_spa_catch_all(self) -> None:
        register_app_extension(_add_page_route)
        register_page(PanelPage("My page", "/my-page", "Served by the extension"))
        app = create_app(self.static)
        with TestClient(app) as client:
            page = client.get("/my-page")
            status = client.get("/api/my-page/status")
            spa = client.get("/control")
            info = client.get("/api/info")

        self.assertEqual(page.status_code, 200)
        self.assertEqual(page.text, "EXTENSION PAGE")
        self.assertEqual(status.json(), {"ok": True})
        self.assertEqual(spa.text, "SPA INDEX")
        self.assertEqual(
            info.json()["pages"],
            [
                {
                    "label": "My page",
                    "path": "/my-page",
                    "description": "Served by the extension",
                }
            ],
        )

    def test_no_extensions_means_no_pages(self) -> None:
        app = create_app()
        with TestClient(app) as client:
            info = client.get("/api/info")
            missing = client.get("/my-page")
        self.assertEqual(info.json()["pages"], [])
        self.assertEqual(missing.status_code, 404)


if __name__ == "__main__":
    unittest.main()
