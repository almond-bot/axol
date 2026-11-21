"""``axol nav.*`` commands: map a space, save waypoints, drive to them."""

from __future__ import annotations

import argparse

from ...lidar import LidarConfig
from ...lidar.config import pose_matrix

# The simulated robot's lidar: on a mast at the front of the base.
SIM_MOUNT = (0.2, 0.0, 0.9, 0.0, 0.0, 0.0)


def add_sim_argument(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--sim",
        action="store_true",
        help="Use a simulated robot in a simulated office (no hardware needed).",
    )


def add_port_argument(p: argparse.ArgumentParser, default: int = 8010) -> None:
    p.add_argument(
        "--port", type=int, default=default, help=f"Map view port (default {default})."
    )


def lidar_setup(sim: bool) -> tuple[LidarConfig, object]:
    """The lidar config and its ``T_base_lidar`` (simulated or from setup)."""
    if sim:
        cfg = LidarConfig()
        return cfg, pose_matrix(
            *SIM_MOUNT[:3], *(v * 3.141592653589793 / 180 for v in SIM_MOUNT[3:])
        )
    cfg = LidarConfig.load()
    if not cfg.lidar_ip or not cfg.host_ip:
        raise SystemExit("The lidar isn't set up. Run `axol lidar.setup` first.")
    return cfg, cfg.mount.matrix()
