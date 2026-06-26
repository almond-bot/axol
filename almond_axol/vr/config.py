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
        interp_enabled: Reconstruct a smooth pose stream from jittery/batched
            arrivals via an adaptive playout buffer (see
            :class:`almond_axol.vr.interp.PoseInterpolator`). When False the
            consumer sees the raw latest-wins frame.
        interp_min_delay_s: Floor on the adaptive playout delay (seconds).
        interp_max_delay_s: Cap on the adaptive playout delay (seconds); bounds
            the teleop latency added in exchange for smoothness.
    """

    port: int = 8000
    certfile: str | None = None
    keyfile: str | None = None
    interp_enabled: bool = True
    interp_min_delay_s: float = 0.0
    interp_max_delay_s: float = 0.1
