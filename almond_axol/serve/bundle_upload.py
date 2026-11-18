"""Offline update bundles uploaded through the control panel.

The panel's **Install from file** sends an ``axol-offline-v<version>-<platform>.tar``
(see :mod:`almond_axol.offline_bundle`) in chunks, so a dropped Wi-Fi link
resumes where it stopped instead of starting over. Once the file is complete,
it is checked in a worker thread: the Almond signature, the platform, the
version against the installed one, and every file's sha256 as it is extracted.
The verified bundle waits in :func:`updates_dir` until the operator presses
**Install update**, which hands it to :meth:`SelfUpdater.start`.

Upload lifecycle, reported as ``bundle`` in ``/api/update/status``:

``idle`` -> ``uploading`` -> ``checking`` -> ``ready`` (or ``error``)

One bundle at a time: a new upload replaces any other upload or staged bundle.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import shutil
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from .. import offline_bundle
from .update import parse_version

_logger = logging.getLogger(__name__)

# Root (the installed service) keeps uploads with the other privileged update
# state; a manual serve keeps them in the operator's home.
_ROOT_DIR = Path("/var/lib/almond-axol/updates")
_USER_DIR = Path.home() / ".almond" / "updates"
_PARTIAL = "upload.tar.part"
_STAGED = "bundle"
# Room for the .tar, the extracted copy and the new tool environment.
_FREE_SPACE_MARGIN = 3 << 30
# A bundle is under 1 GB; refuse anything far beyond that outright.
MAX_BUNDLE_BYTES = 8 << 30
# Largest single PUT the panel sends (it sends 32 MiB).
MAX_CHUNK_BYTES = 64 << 20
_WRITE_BUFFER = 4 << 20


def updates_dir() -> Path:
    return _ROOT_DIR if os.geteuid() == 0 else _USER_DIR


def _gb(size: int) -> str:
    return f"{size / 1e9:.1f} GB"


class UploadError(Exception):
    """A request the store refuses; ``status`` is the HTTP code to answer with."""

    def __init__(self, message: str, status: int = 400, code: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code


class BundleStore:
    """The one in-progress upload or verified bundle on this host."""

    def __init__(
        self,
        installed_version: Callable[[], str | None],
        directory: Path | None = None,
    ) -> None:
        self._installed_version = installed_version
        self._dir = directory or updates_dir()
        self._lock = asyncio.Lock()
        self._state = "idle"
        self._upload_id: str | None = None
        self._name: str | None = None
        self._size = 0
        self._received = 0
        self._error: str | None = None
        self._error_code: str | None = None
        self._bundle: offline_bundle.Bundle | None = None
        self._check_task: asyncio.Task[None] | None = None
        self._recover()

    # -- paths ----------------------------------------------------------------

    @property
    def _partial(self) -> Path:
        return self._dir / _PARTIAL

    @property
    def _staged(self) -> Path:
        return self._dir / _STAGED

    def _recover(self) -> None:
        """Pick up a verified bundle left from before a restart.

        A bundle whose version is now installed was the update that just
        landed, so it is removed. An unfinished upload is dropped: without
        its name and size it can't be resumed safely.
        """
        try:
            self._partial.unlink(missing_ok=True)
        except OSError:
            pass
        if not (self._staged / offline_bundle.MANIFEST).is_file():
            return
        try:
            bundle = offline_bundle.load_signed(self._staged)
        except Exception as exc:  # noqa: BLE001 - a stale or foreign leftover
            _logger.info("bundle upload: discarding staged bundle (%s)", exc)
            shutil.rmtree(self._staged, ignore_errors=True)
            return
        if bundle.version == self._installed_version():
            shutil.rmtree(self._staged, ignore_errors=True)
            return
        self._bundle = bundle
        self._state = "ready"

    # -- status ---------------------------------------------------------------

    def status(self) -> dict[str, object]:
        return {
            "state": self._state,
            "uploadId": self._upload_id if self._state == "uploading" else None,
            "name": self._name,
            "size": self._size,
            "received": self._received,
            "version": self._bundle.version if self._bundle else None,
            "error": self._error,
            "errorCode": self._error_code,
        }

    def ready_bundle(self) -> offline_bundle.Bundle | None:
        """The verified bundle, re-checking its signature before an install."""
        if self._state != "ready" or self._bundle is None:
            return None
        try:
            return offline_bundle.load_signed(self._bundle.root)
        except offline_bundle.BundleError as exc:
            self._set_error(str(exc))
            return None

    # -- lifecycle --------------------------------------------------------------

    def _set_error(self, message: str, code: str | None = None) -> None:
        _logger.warning("bundle upload: %s", message)
        self._state = "error"
        self._error = message
        self._error_code = code

    def _clear_files(self) -> None:
        self._partial.unlink(missing_ok=True)
        shutil.rmtree(self._staged, ignore_errors=True)
        shutil.rmtree(
            self._staged.with_name(self._staged.name + ".tmp"), ignore_errors=True
        )

    def _reset(self) -> None:
        self._state = "idle"
        self._upload_id = None
        self._name = None
        self._size = 0
        self._received = 0
        self._error = None
        self._error_code = None
        self._bundle = None

    def _refuse_while_checking(self) -> None:
        if self._state == "checking":
            raise UploadError("The Host Computer is checking a file. Wait for it.", 409)

    async def begin(self, name: str, size: int) -> dict[str, object]:
        """Start an upload, or resume the unfinished one with the same name and size."""
        async with self._lock:
            self._refuse_while_checking()
            if not isinstance(size, int) or size <= 0:
                raise UploadError("The file is empty.")
            if size > MAX_BUNDLE_BYTES:
                raise UploadError(
                    "This file is too large to be an Axol offline bundle.", 413
                )
            if (
                self._state == "uploading"
                and self._name == name
                and self._size == size
                and self._partial.is_file()
            ):
                self._received = self._partial.stat().st_size
                return self.status()
            self._clear_files()
            self._reset()
            self._dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            free = shutil.disk_usage(self._dir).free
            needed = 2 * size + _FREE_SPACE_MARGIN
            if free < needed:
                raise UploadError(
                    f"The Host Computer needs {_gb(needed - free)} more free space. "
                    "Delete old datasets, then try again.",
                    507,
                    "space",
                )
            self._partial.touch(mode=0o600)
            self._state = "uploading"
            self._upload_id = secrets.token_hex(8)
            self._name = name
            self._size = size
            return self.status()

    async def append(
        self, upload_id: str, offset: int, chunks: AsyncIterator[bytes]
    ) -> dict[str, object]:
        """Write one chunk of the upload, which must start at ``offset``."""
        async with self._lock:
            if self._state != "uploading" or upload_id != self._upload_id:
                raise UploadError("This upload is no longer active. Start again.", 404)
            if offset != self._received:
                # The panel resumes from the offset in this answer.
                raise UploadError(
                    "The upload is out of step. Resuming.", 409, code="offset"
                )
            written = 0
            buffer = bytearray()
            with self._partial.open("ab") as out:
                async for part in chunks:
                    written += len(part)
                    if written > MAX_CHUNK_BYTES or (
                        self._received + written > self._size
                    ):
                        out.truncate(self._received)
                        raise UploadError("The upload is larger than the file.", 413)
                    buffer += part
                    if len(buffer) >= _WRITE_BUFFER:
                        await asyncio.to_thread(out.write, bytes(buffer))
                        buffer.clear()
                if buffer:
                    await asyncio.to_thread(out.write, bytes(buffer))
            self._received += written
            return self.status()

    async def finish(
        self, upload_id: str, allow_older: bool = False
    ) -> dict[str, object]:
        """Check the complete upload in the background."""
        async with self._lock:
            if self._state != "uploading" or upload_id != self._upload_id:
                raise UploadError("This upload is no longer active. Start again.", 404)
            if self._received != self._size:
                raise UploadError("The upload is not complete yet.", 409)
            self._state = "checking"
            self._check_task = asyncio.create_task(self._check(allow_older))
            return self.status()

    async def recheck(self, allow_older: bool) -> dict[str, object]:
        """Check the staged bundle again, e.g. to accept an older version."""
        async with self._lock:
            self._refuse_while_checking()
            if not (self._staged / offline_bundle.MANIFEST).is_file():
                raise UploadError("No file to install. Upload it again.", 404)
            self._state = "checking"
            self._check_task = asyncio.create_task(
                self._check(allow_older, extracted=True)
            )
            return self.status()

    async def discard(self) -> dict[str, object]:
        async with self._lock:
            self._refuse_while_checking()
            await asyncio.to_thread(self._clear_files)
            self._reset()
            return self.status()

    def discard_installed(self) -> None:
        """Remove the bundle once the updater has installed it (sync, idempotent)."""
        self._clear_files()
        self._reset()

    async def _check(self, allow_older: bool, extracted: bool = False) -> None:
        try:
            if extracted:
                bundle = await asyncio.to_thread(
                    offline_bundle.load_signed, self._staged
                )
            else:
                bundle = await asyncio.to_thread(
                    offline_bundle.extract, self._partial, self._staged
                )
                self._partial.unlink(missing_ok=True)
            self._bundle = bundle
            self._check_version(bundle, allow_older)
        except UploadError as exc:
            self._set_error(str(exc), exc.code)
            return
        except offline_bundle.BundleError as exc:
            await asyncio.to_thread(self._clear_files)
            self._set_error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - surface to the panel
            await asyncio.to_thread(self._clear_files)
            self._set_error(f"Could not check the file: {exc}")
            return
        self._state = "ready"
        self._error = None
        self._error_code = None

    def _check_version(self, bundle: offline_bundle.Bundle, allow_older: bool) -> None:
        installed = self._installed_version()
        new = parse_version(bundle.version)
        current = parse_version(installed) if installed else None
        if new is None or current is None:
            return
        if new == current:
            raise UploadError(
                f"Axol v{bundle.version} is already installed.", code="installed"
            )
        if new < current and not allow_older:
            raise UploadError(
                f"This file has Axol v{bundle.version}, which is older than the "
                f"installed v{installed}.",
                code="older",
            )
