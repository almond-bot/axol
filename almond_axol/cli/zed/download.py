"""Small, fail-closed download helper for Stereolabs artifacts."""

from __future__ import annotations

import os
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

_CHUNK_SIZE = 1024 * 1024


def atomic_https_download(
    url: str,
    destination: Path,
    *,
    max_bytes: int,
    validate: Callable[[Path], None],
) -> None:
    """Download, validate, and atomically publish an HTTPS artifact.

    The temporary file is created exclusively in the destination directory, so
    a predictable ``.part`` symlink can never redirect a privileged write.  A
    completed file becomes visible only after the caller's validation passes.
    """
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise RuntimeError("refusing a non-HTTPS or credential-bearing download URL")

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".part",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "almond-axol-artifact-installer"},
            )
            # Both call sites construct or pin the URL and validate the final
            # HTTPS scheme below.  Bandit's generic URL-open warning does not
            # account for those restrictions.
            with urllib.request.urlopen(request, timeout=60) as response:  # nosec B310
                final_url = urlsplit(response.geturl())
                if (
                    final_url.scheme != "https"
                    or not final_url.hostname
                    or final_url.username is not None
                    or final_url.password is not None
                ):
                    raise RuntimeError("download redirected away from HTTPS")
                declared_length = response.headers.get("Content-Length")
                if declared_length is not None and int(declared_length) > max_bytes:
                    raise RuntimeError("download exceeds the permitted size")

                downloaded = 0
                while chunk := response.read(_CHUNK_SIZE):
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise RuntimeError("download exceeds the permitted size")
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())

        validate(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
