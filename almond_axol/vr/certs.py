"""Self-signed TLS certificate generation for WSS."""

from __future__ import annotations

import os
import subprocess


def create_self_signed_cert(certfile: str, keyfile: str) -> None:
    """Create a self-signed certificate and private key using openssl.

    Overwrites existing files. Creates parent directories if needed.
    The certificate is valid for 365 days with CN=localhost.
    """
    cert_dir = os.path.dirname(certfile)
    if cert_dir:
        os.makedirs(cert_dir, exist_ok=True)

    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            keyfile,
            "-out",
            certfile,
            "-days",
            "365",
            "-nodes",
            "-subj",
            "/CN=localhost",
        ],
        check=True,
        capture_output=True,
    )
