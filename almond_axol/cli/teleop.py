"""
axol teleop [--sim]

Run a VR teleoperation session. Drives the real Axol robot by default;
pass ``--sim`` to use the browser visualizer instead. Every robot config
field is reachable from the CLI (draccus-style) or from a JSON/YAML file:

    axol teleop                                       # real robot
    axol teleop --sim                                 # browser visualizer
    axol teleop --axol.left_stiffness 0.8
    axol teleop --axol.left.elbow.kp 60 --axol.right.gripper.torque_limit 0.7
    axol teleop --left_channel null                   # disable the left arm
    axol teleop --config_path my_teleop.json          # whole-config file
"""

import asyncio
import logging
import socket

from .config import TeleopCmdConfig, parse


def _get_local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


def _normalize_sim_flag(argv: list[str]) -> list[str]:
    """Let ``--sim`` be passed as a bare flag.

    draccus parses bool fields as value-taking arguments (``--sim true``),
    so rewrite a standalone ``--sim`` (one that's followed by another flag
    or nothing) into ``--sim true``. An explicit ``--sim true`` / ``--sim
    false`` / ``--sim=...`` is left untouched.
    """
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--sim":
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            if nxt is None or nxt.startswith("-"):
                out.extend(("--sim", "true"))
                i += 1
                continue
        out.append(tok)
        i += 1
    return out


def main(argv: list[str]) -> None:
    """Parse the CLI config and run a VR teleop session."""
    cfg = parse(TeleopCmdConfig, _normalize_sim_flag(argv))
    # force=True: a dependency imported before this point may install a root
    # handler (leaving the level at WARNING), which would make this a no-op
    # and silently drop log_say() / INFO status lines.
    logging.basicConfig(level=getattr(logging, cfg.log_level), force=True)

    hostname = socket.gethostname()
    local_ip = _get_local_ip()
    print("Connect the VR app (https://axol.almond.bot) to this machine:")
    print(f"  Hostname : {hostname}.local")
    print(f"  IP       : {local_ip}")

    asyncio.run(_run(cfg))


async def _run(cfg: TeleopCmdConfig) -> None:
    from ..robot import Axol, Sim
    from ..teleop import VRTeleop

    if cfg.sim:
        robot = Sim()
    else:
        robot = Axol(
            config=cfg.axol,
            left_channel=cfg.left_channel,
            right_channel=cfg.right_channel,
        )
    async with VRTeleop(robot) as teleop:
        await teleop.run()
