"""Factory calibration in Supabase Storage, keyed by the hub adapter serial.

Every robot is friction- and gravity-calibrated at the factory (``axol
tune.factory``), and the fitted values are uploaded here so any machine the
robot is later plugged into can fetch them — the hub adapter travels with
the arms, so its USB serial is the robot's identity even after the compute
host is reflashed.

The documents live in a **public** Storage bucket, one object per robot::

    axol-calibrations/<hub_serial>.json

Public objects are served over plain HTTPS with no API key, so *fetching
needs no credentials at all* — ``axol calibration.pull`` (and the automatic
pull at the end of ``axol can.setup``) work on any machine that knows the
project URL, which is baked in below. **Uploading** (the factory) needs a
key with write access to the bucket::

    AXOL_SUPABASE_URL=https://<project>.supabase.co   # optional if baked in
    AXOL_SUPABASE_KEY=<service key with storage write>

set in the environment or a ``.env`` / ``.env.local`` (the CLI loads those
at startup; see :mod:`almond_axol.utils.dotenv`). Without the key the cloud
half reports itself unavailable and ``tune.factory`` still calibrates and
saves locally.

One-time Supabase setup: create a Storage bucket named
``axol-calibrations`` and mark it **public** (no table, no RLS policies
needed — write access comes from the service key).

The document has the same shape as ``~/.almond/calibration.json`` (see
:mod:`.calibration`): per-side, per-joint ``friction`` / ``com`` entries.

Uses only stdlib HTTP so the SDK gains no new dependency.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..utils.dotenv import load_local_env

_BUCKET = "axol-calibrations"
_ENV_URL = "AXOL_SUPABASE_URL"
_ENV_KEY = "AXOL_SUPABASE_KEY"
_TIMEOUT_S = 15.0

# The project URL is public by design (it's in every fetch URL), so it is
# baked in — fetching then needs zero setup on user machines. Fill in once
# the Supabase project exists; the AXOL_SUPABASE_URL environment variable
# overrides it either way.
PUBLIC_SUPABASE_URL = ""


def supabase_url() -> str | None:
    """The project URL: environment/.env override, else the baked-in one."""
    load_local_env()
    url = os.environ.get(_ENV_URL) or PUBLIC_SUPABASE_URL
    return url.rstrip("/") if url else None


def supabase_credentials() -> tuple[str, str] | None:
    """The (url, write key) pair for uploads when configured, else None.

    Re-runs the ``.env`` load defensively for non-CLI entry points (it is
    idempotent and never overrides real environment variables).
    """
    url = supabase_url()
    key = os.environ.get(_ENV_KEY)
    if not url or not key:
        return None
    return url, key


def _object_path(hub_serial: str) -> str:
    return f"{_BUCKET}/{urllib.parse.quote(hub_serial, safe='')}.json"


def push_calibration(
    creds: tuple[str, str], hub_serial: str, calibration: dict[str, Any]
) -> None:
    """Upsert one robot's calibration object (keyed by hub serial)."""
    url, key = creds
    req = urllib.request.Request(
        f"{url}/storage/v1/object/{_object_path(hub_serial)}",
        data=json.dumps(calibration, indent=2, sort_keys=True).encode(),
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "x-upsert": "true",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S):
            pass
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"Supabase upload failed ({exc.code}): {detail}")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Supabase unreachable: {exc.reason}")


def fetch_calibration(hub_serial: str) -> dict[str, Any] | None:
    """This robot's calibration document, or None when it has none stored.

    Keyless: reads the public bucket URL directly. Raises ``RuntimeError``
    when no project URL is configured or the fetch fails for a reason other
    than the object not existing.
    """
    url = supabase_url()
    if url is None:
        raise RuntimeError(
            "No Supabase project URL configured — set AXOL_SUPABASE_URL "
            "(environment or .env)."
        )
    req = urllib.request.Request(
        f"{url}/storage/v1/object/public/{_object_path(hub_serial)}"
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        # Storage answers a missing object with 400 or 404 depending on
        # version; both mean "nothing stored for this robot".
        if exc.code in (400, 404):
            return None
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"Supabase fetch failed ({exc.code}): {detail}")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Supabase unreachable: {exc.reason}")
    doc = json.loads(raw)
    return doc if isinstance(doc, dict) else None
