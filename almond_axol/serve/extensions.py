"""Downstream extension points for ``axol serve`` beyond the command catalog.

:func:`~almond_axol.serve.commands.register` lets a package built on
``almond-axol`` add operations to the panel. Two more hooks cover what a
command can't express:

- :func:`register_app_extension` — a callable that receives the FastAPI app
  from :func:`~almond_axol.serve.create_app` after every built-in route is
  registered and *before* the web bundle's SPA catch-all is mounted, so the
  routes it adds (extra ``/api/...`` endpoints, a server-rendered page) win
  over the catch-all like the built-ins do. Without this a downstream package
  had to wrap ``create_app`` and reorder ``app.router.routes`` by hand.
- :func:`register_page` — a :class:`PanelPage` the web panel links to from
  its navigation. The panel is often served from another origin than the
  backend it drives (axol.almond.bot against a station on the LAN), so a page
  a backend serves itself is not reachable by a relative link in the bundle;
  the backend advertises it through ``GET /api/info`` (``pages``) and the
  panel builds the link against its configured server base.

Both registries are process-global like the command catalog and idempotent
(re-registering the same path / callable replaces it), so a downstream
``register_*`` step can run before every ``create_app``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

AppExtension = Callable[[Any], None]

# An absolute path with an optional query / fragment (RFC 3986 pchar set).
_PATH_RE = re.compile(r"^/[A-Za-z0-9._~%!$&'()*+,;=:@/?#-]*$")


@dataclass(frozen=True)
class PanelPage:
    """A backend-served page the web panel links to.

    ``path`` is absolute on the backend origin (``/finetune``); ``label`` is the
    navigation text. ``description`` is optional hover text.
    """

    label: str
    path: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("PanelPage.label must not be empty")
        if not _PATH_RE.fullmatch(self.path) or self.path.startswith("//"):
            raise ValueError(
                f"PanelPage.path must be an absolute path on the backend "
                f"(e.g. '/finetune'), got {self.path!r}"
            )
        if self.path.startswith("/api/"):
            raise ValueError("PanelPage.path must not live under /api/")

    def to_dict(self) -> dict[str, str]:
        data = {"label": self.label, "path": self.path}
        if self.description:
            data["description"] = self.description
        return data


_APP_EXTENSIONS: dict[int, AppExtension] = {}
_PAGES: dict[str, PanelPage] = {}


def register_app_extension(extension: AppExtension) -> None:
    """Run ``extension(app)`` inside every subsequent ``create_app`` call.

    Extensions run in registration order after the built-in routes and
    before the SPA catch-all (when a bundle is served). Registering the same
    callable twice keeps a single entry.
    """
    _APP_EXTENSIONS[id(extension)] = extension


def app_extensions() -> tuple[AppExtension, ...]:
    return tuple(_APP_EXTENSIONS.values())


def register_page(page: PanelPage) -> None:
    """Advertise a backend-served page in ``/api/info`` for the panel's nav.

    Re-registering a path replaces its entry, so a label change applies on
    the next app.
    """
    _PAGES[page.path] = page


def panel_pages() -> tuple[PanelPage, ...]:
    """Registered pages in registration order."""
    return tuple(_PAGES.values())


def clear_extensions() -> None:
    """Forget every registered extension and page (tests)."""
    _APP_EXTENSIONS.clear()
    _PAGES.clear()
