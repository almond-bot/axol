"""Shared constants and utilities for the Almond Axol robot."""

import logging
import subprocess
import sys
from enum import Enum
from pathlib import Path

_logger = logging.getLogger(__name__)


def setup_link_ip(iface: str, address: str) -> None:
    """Assign a static IP to an Ethernet interface (requires sudo).

    Args:
        iface:   Network interface name (e.g. "eth0").
        address: Address with prefix length (e.g. "192.168.10.1/24").
    """
    _logger.info("Configuring %s with %s ...", iface, address)
    cmds = [
        ["sudo", "ip", "link", "set", iface, "up"],
        ["sudo", "ip", "addr", "flush", "dev", iface],
        ["sudo", "ip", "addr", "add", address, "dev", iface],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"error: {' '.join(cmd)}\n{result.stderr.strip()}", file=sys.stderr)
            raise SystemExit(1)
    _logger.info("Interface %s ready.", iface)


class Joint(Enum):
    SHOULDER_1 = "shoulder_1"
    SHOULDER_2 = "shoulder_2"
    SHOULDER_3 = "shoulder_3"
    ELBOW = "elbow"
    WRIST_1 = "wrist_1"
    WRIST_2 = "wrist_2"
    WRIST_3 = "wrist_3"
    GRIPPER = "gripper"


CAN_LEFT = "can_alm_axol_l"
CAN_RIGHT = "can_alm_axol_r"

ARM_JOINTS: list[Joint] = [j for j in Joint if j != Joint.GRIPPER]

URDF_PATH: Path = Path(__file__).resolve().parent / "kinematics" / "urdf" / "axol.urdf"
