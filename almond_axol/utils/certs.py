"""Shared self-signed TLS certificate utilities.

Used by both ``axol serve`` (the control-panel API) and the VR WebSocket server
so a single certificate — and a single browser cert acceptance — covers both.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .paths import almond_path
from .state_files import (
    privileged_service_active,
    secure_atomic_copy_file,
    secure_atomic_write_bytes,
)

# Shared cert location. Kept under ``vr/`` even though ``axol serve`` now uses it
# too: renaming would force every existing install to regenerate (and re-accept)
# its certificate, so the legacy path stays for backward compatibility.
CERT_DIR = str(almond_path("vr", "certs"))
CERTFILE = os.path.join(CERT_DIR, "cert.pem")
KEYFILE = os.path.join(CERT_DIR, "key.pem")

# A tiny page served at ``/__accept`` on both the VR (:8000) and control (:8001)
# servers. The web UI opens it in a script-spawned popup so the user can approve
# the self-signed certificate in a single top-level navigation; the page then
# closes itself, and the opener retries the (now-trusted-for-the-session)
# connection. This only streamlines the browser's self-signed override — it does
# not replace it; the override is per-origin (scheme+host+port) and session-scoped.
ACCEPT_PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Axol — certificate accepted</title>
</head>
<body style="margin:0;height:100vh;display:flex;align-items:center;justify-content:center;\
background:#121212;color:#eaeaea;font-family:system-ui,-apple-system,sans-serif">
<div style="text-align:center">
<p style="font-size:1.1rem;margin:0 0 .4rem">Certificate accepted.</p>
<p style="opacity:.55;margin:0">You can close this window and return to Axol.</p>
</div>
<script>setTimeout(function(){try{window.close()}catch(e){}},700)</script>
</body>
</html>"""


@dataclass
class PreparedTLSFiles:
    """Stable certificate paths ready for a TLS library to reopen.

    The privileged service cannot safely validate a path in operator-owned state
    and then give that same path to uvicorn: the entry can be replaced between
    those operations.  Hosted-service callers therefore receive copies in a
    private temporary directory.  Direct SDK/CLI callers retain the historical
    path behavior, including support for root-managed certificate symlinks.
    """

    certfile: str
    keyfile: str
    generated: bool = False
    _temporary: tempfile.TemporaryDirectory[str] | None = field(
        default=None, repr=False
    )

    def close(self) -> None:
        temporary = self._temporary
        self._temporary = None
        if temporary is not None:
            temporary.cleanup()

    def __enter__(self) -> PreparedTLSFiles:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def create_self_signed_cert(certfile: str, keyfile: str) -> None:
    """Create a self-signed certificate and private key using openssl.

    Overwrites existing files. Creates parent directories if needed. The
    certificate is valid for 365 days with CN=localhost and a
    ``subjectAltName`` covering ``localhost`` / ``127.0.0.1`` — so the
    Quest-over-USB ``wss://localhost:8000`` pose tunnel (``adb reverse``)
    matches the cert hostname. It is still self-signed, so the browser accepts
    it once per origin via the in-app "Authorize USB certificate" flow — same
    as the LAN-IP host cert.
    """
    # OpenSSL cannot accept already-open output descriptors. Generate into a
    # root-private random directory, then publish through the no-follow atomic
    # writer; never let it open predictable names in operator-owned state.
    # Root must not honor an operator-controlled TMPDIR for OpenSSL output.
    # `/tmp`'s sticky bit protects this mode-0700 directory from other users;
    # non-root generation also needs no caller-visible temporary location.
    with tempfile.TemporaryDirectory(prefix="axol-cert-", dir="/tmp") as temporary_dir:
        temporary = Path(temporary_dir)
        temporary_cert = temporary / "cert.pem"
        temporary_key = temporary / "key.pem"
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-keyout",
                str(temporary_key),
                "-out",
                str(temporary_cert),
                "-days",
                "365",
                "-nodes",
                "-subj",
                "/CN=localhost",
                "-addext",
                "subjectAltName=DNS:localhost,IP:127.0.0.1",
            ],
            check=True,
            capture_output=True,
        )
        secure_atomic_write_bytes(certfile, temporary_cert.read_bytes(), mode=0o644)
        secure_atomic_write_bytes(keyfile, temporary_key.read_bytes(), mode=0o600)


def prepare_tls_files(certfile: str, keyfile: str) -> PreparedTLSFiles:
    """Return TLS files that remain stable until :meth:`close` is called.

    In the hosted root service, both source files are opened
    descriptor-relatively with ``O_NOFOLLOW`` and copied into a root-owned
    ``/tmp`` directory before their paths are passed to uvicorn.  This rejects
    symlinks/special files and pins the exact bytes uvicorn will later reopen,
    closing the check/use race in an operator-writable ``ALMOND_HOME``.
    ``/tmp`` is explicit so an operator-controlled ``TMPDIR`` cannot choose
    the privileged staging parent.

    If either default file is genuinely absent, the pair is generated with
    :func:`create_self_signed_cert` and then snapshotted.  An unsafe existing
    entry fails before generation, so it is never silently replaced.
    """
    default_cert = os.path.abspath(os.path.expanduser(CERTFILE))
    default_key = os.path.abspath(os.path.expanduser(KEYFILE))
    uses_shared_default = (
        os.path.abspath(os.path.expanduser(certfile)) == default_cert
        or os.path.abspath(os.path.expanduser(keyfile)) == default_key
    )
    requires_snapshot = privileged_service_active() or (
        os.geteuid() == 0 and uses_shared_default
    )
    if not requires_snapshot:
        generated = False
        if not os.path.isfile(certfile) or not os.path.isfile(keyfile):
            create_self_signed_cert(certfile, keyfile)
            generated = True
        return PreparedTLSFiles(certfile, keyfile, generated=generated)

    temporary = tempfile.TemporaryDirectory(prefix="axol-tls-", dir="/tmp")
    snapshot_dir = Path(temporary.name)
    snapshot_cert = snapshot_dir / "cert.pem"
    snapshot_key = snapshot_dir / "key.pem"
    sources = (
        (certfile, snapshot_cert, 0o644),
        (keyfile, snapshot_key, 0o600),
    )
    generated = False
    try:
        missing = False
        for source, destination, mode in sources:
            try:
                secure_atomic_copy_file(source, destination, mode=mode)
            except FileNotFoundError:
                missing = True

        if missing:
            # All existing entries were already proven safe above.  The
            # no-follow atomic writer also rechecks them during publication.
            create_self_signed_cert(certfile, keyfile)
            generated = True
            for source, destination, mode in sources:
                secure_atomic_copy_file(source, destination, mode=mode)
    except BaseException:
        temporary.cleanup()
        raise

    return PreparedTLSFiles(
        str(snapshot_cert),
        str(snapshot_key),
        generated=generated,
        _temporary=temporary,
    )
