"""
axol policy.check

Exercise a custom policy endpoint without a robot: play run-policy's side of a
session against it on a virtual clock (same scheduler, simulated latency,
mid-grey camera frames, an arm that tracks its commands perfectly), then report
whether it keeps up and how smooth the executed trajectory is.

    python my_policy.py &                  # almond_axol.policy.serve(...)
    axol policy.check                      # ws://127.0.0.1:8765
    axol policy.check --server_host 192.168.1.99 --delay_steps 4
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import LogLevel, parse

_logger = logging.getLogger(__name__)


@dataclass
class PolicyCheckConfig:
    """Config for ``axol policy.check``.

    Args:
        server_host: Endpoint address (or a ``ws://`` / ``wss://`` URL).
        server_port: Endpoint port.
        ticks: Control ticks to simulate.
        delay_steps: Simulated reply latency, in ticks.
        cartesian: Send the Cartesian layout (``observe_cartesian``) instead
            of joint positions.
        cameras: Comma-separated camera names to send.
        camera_height: Frame height.
        camera_width: Frame width.
        fps: Control rate.
        actions_per_chunk: Horizon (most rows used from one reply).
        request_interval: Rows dispatched between requests.
        max_adoption_offset_steps: Latest row a reply may start executing at.
        log_level: Python logging level.
    """

    server_host: str = "127.0.0.1"
    server_port: int = 8765
    ticks: int = 300
    delay_steps: int = 2
    cartesian: bool = False
    cameras: str = "overhead"
    camera_height: int = 360
    camera_width: int = 640
    fps: int = 30
    actions_per_chunk: int = 50
    request_interval: int = 5
    max_adoption_offset_steps: int = 6
    log_level: LogLevel = "WARNING"


def main(argv: list[str]) -> None:
    """Parse the CLI config, run the check, and print its report."""
    import sys

    from ..policy import PolicyRemoteError, check_policy, default_spec, policy_url

    cfg = parse(PolicyCheckConfig, argv)
    logging.basicConfig(level=getattr(logging, cfg.log_level), force=True)
    spec = default_spec(
        cartesian=cfg.cartesian,
        cameras=tuple(name.strip() for name in cfg.cameras.split(",") if name.strip()),
        image_shape=(cfg.camera_height, cfg.camera_width),
        fps=cfg.fps,
        actions_per_chunk=cfg.actions_per_chunk,
        request_interval=cfg.request_interval,
        max_adoption_offset_steps=cfg.max_adoption_offset_steps,
    )
    url = policy_url(cfg.server_host, cfg.server_port)
    print(f"Checking {url} …", flush=True)
    try:
        report = check_policy(
            url, spec=spec, ticks=cfg.ticks, delay_steps=cfg.delay_steps
        )
    except PolicyRemoteError as exc:
        print(f"The endpoint refused or failed a request: {exc}", file=sys.stderr)
        sys.exit(1)
    except (OSError, TimeoutError) as exc:
        print(f"Could not talk to {url}: {exc}", file=sys.stderr)
        sys.exit(1)
    print(report.summary())
    if report.recoveries:
        sys.exit(2)
