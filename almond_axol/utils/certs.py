"""Shared self-signed TLS certificate utilities.

Used by both ``axol serve`` (the control-panel API) and the VR WebSocket server
so a single certificate — and a single browser cert acceptance — covers both.
"""

from __future__ import annotations

import os
import subprocess

# Shared cert location. Kept under ``vr/`` even though ``axol serve`` now uses it
# too: renaming would force every existing install to regenerate (and re-accept)
# its certificate, so the legacy path stays for backward compatibility.
CERT_DIR = os.path.join(os.path.expanduser("~"), ".almond", "vr", "certs")
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
            "-addext",
            "subjectAltName=DNS:localhost,IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )
