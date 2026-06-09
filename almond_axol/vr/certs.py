"""Self-signed TLS certificate generation for WSS."""

from __future__ import annotations

import os
import subprocess

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
