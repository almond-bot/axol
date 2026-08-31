"""
axol inference-server

Serve policy inference for ``axol run-policy --server_host <this machine>``.

Runs LeRobot's async-inference ``PolicyServer`` in the foreground on a more
powerful machine (e.g. a desktop with a discrete GPU) on the same network as
the robot. The robot streams joint positions + camera frames to it over gRPC
and receives action chunks back; the policy itself (``--policy_path`` /
``--policy_type`` / ``--device``) is selected by the *client*, so one server
can serve different policies across sessions without restarting.

This service has no transport authentication or encryption. Its non-loopback
mode is for an isolated, trusted robot network protected by a host firewall;
it must not be exposed to shared Wi-Fi or the public internet.

    axol inference-server                              # loopback only
    axol inference-server --host 192.168.1.99          # explicit LAN interface

Then, on the robot:

    axol run-policy --server_host <server-ip> ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import LogLevel, parse

_logger = logging.getLogger(__name__)


@dataclass
class InferenceServerConfig:
    """Config for ``axol inference-server``.

    Args:
        host:      Interface to bind the gRPC server to. The safe default is
                   loopback; remote inference requires an explicit LAN IP.
        port:      gRPC port (must match run-policy's ``--server_port``).
        fps:       Action chunk rate; must match run-policy's ``--fps``.
        log_level: Python logging level.
    """

    host: str = "127.0.0.1"
    port: int = 8765
    fps: int = 60
    log_level: LogLevel = "INFO"


def main(argv: list[str]) -> None:
    """Parse the CLI config and serve policy inference until Ctrl+C."""
    cfg = parse(InferenceServerConfig, argv)
    logging.basicConfig(level=getattr(logging, cfg.log_level), force=True)

    from ..lerobot.inference_patch import (
        disable_observation_similarity_filter,
        enable_action_schema_handshake,
    )

    disable_observation_similarity_filter()
    enable_action_schema_handshake()

    # Register the Mantis relative-EE processor steps so checkpoints trained with
    # `axol mantis.train` deserialize their processor pipelines here.
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server import serve

    from ..mantis import processor as _mantis_processor  # noqa: F401
    from ..utils.ports import reclaim_port

    # The gRPC port is fixed (it must match run-policy's ``--server_port``), so
    # evict a leftover server from a crashed/previous run rather than failing to
    # bind. lerobot owns the socket once ``serve`` takes over.
    reclaim_port(cfg.port)

    _logger.info(
        "Serving policy inference on %s:%d (Ctrl+C to stop).", cfg.host, cfg.port
    )
    if cfg.host not in {"127.0.0.1", "::1", "localhost"}:
        _logger.warning(
            "Inference gRPC is plaintext and unauthenticated. Any reachable peer "
            "can request supported checkpoint loads or impersonate the action "
            "server. Allow only the intended robot IP through a host firewall on "
            "an isolated trusted network; never expose this port to shared Wi-Fi "
            "or the internet."
        )
    try:
        serve(PolicyServerConfig(host=cfg.host, port=cfg.port, fps=cfg.fps))
    except KeyboardInterrupt:
        _logger.info("Inference server stopped.")
