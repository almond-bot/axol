"""
axol inference-server

Serve policy inference for ``axol run-policy --server_host <this machine>``.

Runs LeRobot's async-inference ``PolicyServer`` in the foreground on a more
powerful machine (e.g. a desktop with a discrete GPU) on the same network as
the robot. The robot streams joint positions + camera frames to it over gRPC
and receives action chunks back; the policy itself (``--policy_path`` /
``--policy_type`` / ``--device``) is selected by the *client*, so one server
can serve different policies across sessions without restarting.

    axol inference-server                 # listen on 0.0.0.0:8765
    axol inference-server --port 9000

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
        host:      Interface to bind the gRPC server to. The default
                   (0.0.0.0) accepts connections from the whole network.
        port:      gRPC port (must match run-policy's ``--server_port``).
        fps:       Action chunk rate; must match run-policy's ``--fps``.
        log_level: Python logging level.
    """

    host: str = "0.0.0.0"
    port: int = 8765
    fps: int = 60
    log_level: LogLevel = "INFO"


def main(argv: list[str]) -> None:
    """Parse the CLI config and serve policy inference until Ctrl+C."""
    cfg = parse(InferenceServerConfig, argv)
    logging.basicConfig(level=getattr(logging, cfg.log_level), force=True)

    from ..lerobot.inference_patch import disable_observation_similarity_filter

    disable_observation_similarity_filter()

    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server import serve

    _logger.info(
        "Serving policy inference on %s:%d (Ctrl+C to stop).", cfg.host, cfg.port
    )
    try:
        serve(PolicyServerConfig(host=cfg.host, port=cfg.port, fps=cfg.fps))
    except KeyboardInterrupt:
        _logger.info("Inference server stopped.")
