"""VRServerConfig dataclass for the VR WebSocket server."""

from __future__ import annotations

from dataclasses import dataclass

from ..utils.ports import VR_PORT


@dataclass
class VRServerConfig:
    """Configuration for the VR WebSocket server.

    Attributes:
        port:     Port to listen on. Defaults to the shared ``VR_PORT`` so the
                  Quest-over-USB ``adb reverse`` tunnel always targets the same
                  port the server binds.
        certfile: Path to TLS certificate. None uses the auto-generated cert in ~/.almond/vr/certs/.
        keyfile:  Path to TLS private key. None uses the auto-generated key in ~/.almond/vr/certs/.
    """

    port: int = VR_PORT
    certfile: str | None = None
    keyfile: str | None = None
