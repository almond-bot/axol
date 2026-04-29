"""VRServerConfig dataclass for the VR WebSocket server."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VRServerConfig:
    """Configuration for the VR WebSocket server.

    Attributes:
        port:     Port to listen on.
        certfile: Path to TLS certificate. None uses the auto-generated cert in ~/.almond/vr/certs/.
        keyfile:  Path to TLS private key. None uses the auto-generated key in ~/.almond/vr/certs/.
    """

    port: int = 8000
    certfile: str | None = None
    keyfile: str | None = None
