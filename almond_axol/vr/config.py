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
        usb_fallback_timeout_s:
                  How long the wired USB pose stream may go silent before the
                  server falls back to frames arriving over WiFi. The headset
                  sends every frame over both links, so this is purely how
                  quickly a USB drop fails over — small enough to be unnoticed,
                  large enough not to flap on a single dropped USB frame.
    """

    port: int = VR_PORT
    certfile: str | None = None
    keyfile: str | None = None
    usb_fallback_timeout_s: float = 0.25
