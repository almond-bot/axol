"""Minimal ``.env`` loader for the ``axol`` CLI.

Loads ``.env`` then ``.env.local`` (the latter overriding the former) found in
the current directory or any ancestor, so secrets like the TURN credentials
(see :mod:`almond_axol.vr.ice`) are present in ``os.environ`` for every command
without a manual ``source``. Real environment variables always win — a value
already set in the environment is never overwritten by a file.

Intentionally dependency-free and forgiving: a malformed file is ignored rather
than breaking the CLI. No variable expansion or interpolation is performed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Loaded in order; later files override earlier ones (but never the real env).
_ENV_FILES = (".env", ".env.local")


def _parse(path: Path) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines; tolerate ``export``, blanks, ``#`` comments."""
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, _, val = line.partition("=")
        key = key.strip()
        if not key:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        out[key] = val
    return out


def _find_upwards(filename: str, start: Path) -> Path | None:
    """Nearest ``filename`` at ``start`` or an ancestor, or ``None``."""
    for parent in (start, *start.parents):
        candidate = parent / filename
        if candidate.is_file():
            return candidate
    return None


def load_local_env(start: Path | None = None) -> None:
    """Populate ``os.environ`` from ``.env`` / ``.env.local`` (best effort)."""
    try:
        start = (start or Path.cwd()).resolve()
        merged: dict[str, str] = {}
        loaded: list[Path] = []
        for name in _ENV_FILES:
            path = _find_upwards(name, start)
            if path is not None:
                merged.update(_parse(path))
                loaded.append(path)

        applied = 0
        for key, val in merged.items():
            if key not in os.environ:
                os.environ[key] = val
                applied += 1

        if applied:
            files = ", ".join(str(p) for p in loaded)
            print(f"axol: loaded {applied} env var(s) from {files}", file=sys.stderr)
    except Exception:  # noqa: BLE001 - env loading must never break the CLI
        pass
