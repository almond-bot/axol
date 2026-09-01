"""Internal systemd readiness check for a newly installed Axol release.

The hosted installer starts an updated service while the durable update marker
still blocks every hardware launch in the API.  A one-shot token allows exactly
that candidate start.  systemd runs this command as ``ExecStartPost``; only
after the expected release has answered repeatedly from the service's stable
main PID does it remove the marker and commit the update.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import ssl
import stat
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_PACKAGE = "almond-axol"
_SERVICE = "axol.service"
_SYSTEMCTL = "/usr/bin/systemctl"
_STATE_DIR = Path("/var/lib/almond-axol")
_UPDATE_MARKER = _STATE_DIR / "update-incomplete"
_VERIFYING_MARKER = _STATE_DIR / "update-verifying"
_HEALTH_URL = "https://127.0.0.1:8001/api/health"
_VERSION_RE = re.compile(r"[0-9]+(?:\.[0-9]+)*")
_HEALTH_TIMEOUT_S = 45.0
_HEALTH_DWELL_S = 2.0
_POLL_S = 0.25


class UpdateHealthError(RuntimeError):
    """The candidate service could not be safely committed."""


def _safe_state_dir() -> None:
    try:
        metadata = _STATE_DIR.stat(follow_symlinks=False)
    except OSError as exc:
        raise UpdateHealthError(
            "cannot inspect the Axol update state directory"
        ) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_gid != 0
        or metadata.st_mode & 0o022
    ):
        raise UpdateHealthError("the Axol update state directory is unsafe")


def _read_transaction_file(path: Path, *, required: bool) -> str | None:
    """Read one root-only transaction file without following a symlink."""
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        if required:
            raise UpdateHealthError(f"required update state is missing: {path.name}")
        return None
    except OSError as exc:
        raise UpdateHealthError(f"cannot open update state: {path.name}") from exc
    try:
        metadata = os.fstat(fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_gid != 0
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise UpdateHealthError(f"update state is unsafe: {path.name}")
        payload = os.read(fd, 257)
        if len(payload) > 256:
            raise UpdateHealthError(f"update state is malformed: {path.name}")
    finally:
        os.close(fd)
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise UpdateHealthError(f"update state is malformed: {path.name}") from exc
    prefix = "target-version="
    if not text.endswith("\n") or not text.startswith(prefix):
        raise UpdateHealthError(f"update state is malformed: {path.name}")
    version = text[len(prefix) : -1]
    if _VERSION_RE.fullmatch(version) is None:
        raise UpdateHealthError(f"update state is malformed: {path.name}")
    return version


def _main_pid() -> int | None:
    try:
        result = subprocess.run(
            [_SYSTEMCTL, "show", "--property=MainPID", "--value", _SERVICE],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        pid = int(result.stdout.strip())
    except ValueError:
        return None
    return pid if pid > 0 else None


def _health_payload() -> dict[str, Any] | None:
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    request = urllib.request.Request(
        _HEALTH_URL,
        headers={"User-Agent": "almond-axol-update-healthcheck"},
    )
    try:
        with urllib.request.urlopen(  # noqa: S310 - fixed loopback URL
            request,
            timeout=2,
            context=context,
        ) as response:
            if getattr(response, "status", 200) != 200:
                return None
            raw = response.read(16_385)
    except (OSError, ValueError, urllib.error.URLError):
        return None
    if len(raw) > 16_384:
        return None
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _remove_durable(path: Path) -> None:
    try:
        path.unlink()
        directory_fd = os.open(_STATE_DIR, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise UpdateHealthError(f"cannot commit removal of {path.name}") from exc


def verify_candidate() -> None:
    """Verify one token-authorized Axol start within the shared transaction."""
    if os.geteuid() != 0:
        raise UpdateHealthError("update health verification requires root")
    _safe_state_dir()
    target = _read_transaction_file(_VERIFYING_MARKER, required=False)
    if target is None:
        # Ordinary service start, outside an update transaction.
        return
    marker_target = _read_transaction_file(_UPDATE_MARKER, required=True)
    if marker_target != target:
        raise UpdateHealthError("candidate and update marker versions do not match")
    try:
        installed = importlib.metadata.version(_PACKAGE)
    except importlib.metadata.PackageNotFoundError as exc:
        raise UpdateHealthError("cannot identify the installed Axol release") from exc
    if installed != target:
        raise UpdateHealthError(
            f"candidate version mismatch (expected {target}, running {installed})"
        )

    deadline = time.monotonic() + _HEALTH_TIMEOUT_S
    stable_pid: int | None = None
    healthy_since: float | None = None
    while time.monotonic() < deadline:
        now = time.monotonic()
        pid = _main_pid()
        payload = _health_payload()
        healthy = (
            pid is not None
            and payload is not None
            and payload.get("ready") is True
            and payload.get("version") == target
            and payload.get("pid") == pid
        )
        if healthy:
            if pid != stable_pid:
                stable_pid = pid
                healthy_since = now
            elif healthy_since is not None and now - healthy_since >= _HEALTH_DWELL_S:
                break
        else:
            stable_pid = None
            healthy_since = None
        time.sleep(_POLL_S)
    else:
        raise UpdateHealthError(
            f"Axol {target} did not remain healthy on its stable service PID"
        )

    # Commit only the Axol half of the transaction. The installer retains the
    # durable update marker until any required Mantis helper has also survived
    # its startup dwell. With this attempt marker gone, a crash at any later
    # boundary leaves both services fail-closed on their permanent guards.
    _remove_durable(_VERIFYING_MARKER)


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser(
        "update-healthcheck",
        help="Internal systemd readiness check for an installed release.",
    )
    parser.set_defaults(func=run)


def run(_args: object = None) -> None:
    try:
        verify_candidate()
    except UpdateHealthError as exc:
        raise SystemExit(str(exc)) from exc
