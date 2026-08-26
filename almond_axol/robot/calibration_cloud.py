"""Factory calibration in Supabase, keyed by the Axol hub adapter serial.

Every robot is friction- and gravity-calibrated at the factory (``axol
tune.factory``), and the fitted values are uploaded here so any machine the
robot is later plugged into can fetch them (``axol calibration.pull``) —
the hub adapter travels with the arms, so its USB serial is the robot's
identity even after the compute host is reflashed.

Expected table (Supabase SQL editor)::

    create table axol_calibrations (
      hub_serial text primary key,
      calibration jsonb not null,
      updated_at timestamptz not null default now()
    );

The ``calibration`` document has the same shape as
``~/.almond/calibration.json`` (see :mod:`.calibration`): per-side, per-joint
``friction`` / ``com`` entries.

Credentials come from ``AXOL_SUPABASE_URL`` and ``AXOL_SUPABASE_KEY`` — set
in the environment or a ``.env`` / ``.env.local`` (the CLI loads those into
the environment at startup; see :mod:`almond_axol.utils.dotenv`). Uploading
needs a key with write access to the table (the factory's); fetching works
with the read-only anon key. When the variables are absent the cloud half
simply reports itself unavailable — ``tune.factory`` still calibrates and
saves locally.

Uses only stdlib HTTP (PostgREST is plain JSON over HTTPS) so the SDK gains
no new dependency.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..utils.dotenv import load_local_env

_TABLE = "axol_calibrations"
_ENV_URL = "AXOL_SUPABASE_URL"
_ENV_KEY = "AXOL_SUPABASE_KEY"
_TIMEOUT_S = 15.0


def supabase_credentials() -> tuple[str, str] | None:
    """The (url, key) pair when configured, else None.

    Re-runs the ``.env`` load defensively for non-CLI entry points (it is
    idempotent and never overrides real environment variables).
    """
    load_local_env()
    url = os.environ.get(_ENV_URL)
    key = os.environ.get(_ENV_KEY)
    if not url or not key:
        return None
    return url.rstrip("/"), key


def _request(
    creds: tuple[str, str],
    method: str,
    path: str,
    body: dict[str, Any] | list[Any] | None = None,
    prefer: str | None = None,
) -> Any:
    url, key = creds
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    req = urllib.request.Request(
        f"{url}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"Supabase {method} {path} failed ({exc.code}): {detail}")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Supabase unreachable: {exc.reason}")
    return json.loads(raw) if raw else None


def push_calibration(
    creds: tuple[str, str], hub_serial: str, calibration: dict[str, Any]
) -> None:
    """Upsert one robot's calibration document (keyed by hub serial)."""
    _request(
        creds,
        "POST",
        f"/rest/v1/{_TABLE}",
        body=[{"hub_serial": hub_serial, "calibration": calibration}],
        prefer="resolution=merge-duplicates",
    )


def fetch_calibration(creds: tuple[str, str], hub_serial: str) -> dict[str, Any] | None:
    """This robot's calibration document, or None when it has none stored."""
    quoted = urllib.parse.quote(hub_serial, safe="")
    rows = _request(
        creds,
        "GET",
        f"/rest/v1/{_TABLE}?hub_serial=eq.{quoted}&select=calibration,updated_at",
    )
    if not rows:
        return None
    doc = rows[0].get("calibration")
    return doc if isinstance(doc, dict) else None
