"""
axol lidar.setup

Find the Livox lidar on a wired port, give that port a static address, and
save the result to ``~/.almond/lidar.json`` for every lidar and navigation
command.

The port gets one ``/32`` address and a ``/32`` route to the lidar, so the
lidar's subnet (``192.168.1.x`` from the factory) never clashes with the
robot's own network. The port also keeps DHCP on, so a router plugged into it
still gives the robot an address and internet. Needs ``sudo`` for the network
changes.

Usage:
    axol lidar.setup
    axol lidar.setup --iface end0
    axol lidar.setup --mount 0.2 0 0.45 0 0 0     # x y z (m), roll pitch yaw (deg)
"""

from __future__ import annotations

import argparse
import subprocess

from ...lidar import LidarConfig, LivoxClient
from ...lidar import network as net
from . import format_stats, format_status, measure


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "lidar.setup",
        help="Find the lidar, set up its network port and save its address.",
    )
    p.add_argument(
        "--iface",
        default=None,
        help="Wired port the lidar is plugged into (default: search every wired port).",
    )
    p.add_argument(
        "--host-ip",
        default=None,
        help="Address to give this computer on the lidar's port "
        "(default: the address the lidar already sends to).",
    )
    p.add_argument(
        "--mount",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        default=None,
        help="Where the lidar sits on the robot: metres from the centre of "
        "the base at floor level (+x forward, +y left, +z up), then degrees.",
    )
    p.add_argument(
        "--measure-mount",
        action="store_true",
        help="Measure the lidar's height above the floor and its tilt (keep the "
        "robot still on a level floor). X, Y and YAW come from --mount.",
    )
    p.add_argument(
        "--no-network",
        action="store_true",
        help="Don't change the network. Only detect the lidar and save its address.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Configure the port even if it carries this computer's default route.",
    )
    p.set_defaults(func=run)


def _carries_default_route(iface: str) -> bool:
    proc = subprocess.run(
        ["ip", "-4", "route", "show", "default", "dev", iface],
        capture_output=True,
        text=True,
    )
    return bool(proc.stdout.strip())


def _find(cfg: LidarConfig, iface: str | None) -> net.SniffResult:
    # Fast path: the saved address still answers.
    if cfg.lidar_ip and cfg.host_ip and (iface in (None, cfg.iface)):
        if cfg.iface and net.host_has_address(cfg.iface, cfg.host_ip):
            try:
                with LivoxClient(cfg.lidar_ip, cfg.host_ip, timeout=0.5) as c:
                    c.status()
                return net.SniffResult(cfg.iface, cfg.lidar_ip, cfg.host_ip)
            except (TimeoutError, OSError):
                pass
    _require_sudo()
    candidates = [iface] if iface else net.wired_interfaces()
    if not candidates:
        raise SystemExit("No wired Ethernet port found on this computer.")
    for name in candidates:
        if not net.link_up(name):
            print(f"  {name}: no cable or no power on the other end")
            continue
        print(f"  {name}: listening for the lidar...")
        found = net.sniff_lidar_as_root(name, timeout=4.0)
        if found is not None:
            return found
        print(f"  {name}: no lidar traffic")
    raise SystemExit(
        "Lidar not found. Check that it has power (9-27 V) and that its "
        "Ethernet cable is plugged into this computer, then run "
        "`axol lidar.setup` again."
    )


def _require_sudo() -> None:
    from ...utils.sudo import prime_sudo

    if not prime_sudo():
        raise SystemExit(
            "axol lidar.setup needs sudo to listen on the network port and "
            "configure it. Run it from a terminal where you can enter your password."
        )


def _measure(cfg: LidarConfig) -> None:
    from ...lidar.mount import measure_mount
    from ...lidar.stream import LidarStream

    print("Measuring the mount (keep the robot still)...")
    with LidarStream(cfg) as stream:
        frames = [f for f in (stream.get(2.0) for _ in range(20)) if f is not None]
        imu = stream.imu_recent(400)
    try:
        m = measure_mount(frames, imu, yaw_deg=cfg.mount.yaw)
    except ValueError as exc:
        raise SystemExit(f"Couldn't measure the mount: {exc}") from None
    cfg.mount.z, cfg.mount.roll, cfg.mount.pitch = (
        round(m.height, 3),
        round(m.roll, 2),
        round(m.pitch, 2),
    )
    print(
        f"  Height {m.height:.2f} m above the floor, roll {m.roll:.1f}°, pitch {m.pitch:.1f}°. "
        f"Set X, Y and YAW with --mount if the lidar isn't at the centre of the base facing forward."
    )


def run(args: argparse.Namespace) -> None:
    cfg = LidarConfig.load()
    print("Looking for the lidar...")
    found = _find(cfg, args.iface)
    host_ip = args.host_ip or net.choose_host_ip(found.lidar_ip, found.wanted_host_ip)
    speed = net.link_speed(found.iface)
    print(
        f"Found a lidar at {found.lidar_ip} on {found.iface}"
        + (f" ({speed} Mb/s)" if speed else "")
        + "."
    )

    if not args.no_network:
        if _carries_default_route(found.iface) and not args.force:
            raise SystemExit(
                f"{found.iface} carries this computer's internet connection. "
                "Plug the lidar into a different port, or pass --force."
            )
        _require_sudo()
        print(f"Giving {found.iface} the address {host_ip} for the lidar...")
        net.apply_profile(found.iface, host_ip, found.lidar_ip)
        net.apply_sysctl()

    cfg.iface = found.iface
    cfg.lidar_ip = found.lidar_ip
    cfg.host_ip = host_ip
    if args.mount is not None:
        x, y, z, roll, pitch, yaw = args.mount
        cfg.mount.x, cfg.mount.y, cfg.mount.z = x, y, z
        cfg.mount.roll, cfg.mount.pitch, cfg.mount.yaw = roll, pitch, yaw

    try:
        with LivoxClient(cfg.lidar_ip, cfg.host_ip) as client:
            status = client.status()
    except (TimeoutError, OSError) as exc:
        raise SystemExit(
            f"The lidar at {cfg.lidar_ip} doesn't answer from {cfg.host_ip} ({exc}). "
            "Run `axol lidar.setup` again without --no-network."
        ) from None
    cfg.serial = status.serial
    if args.measure_mount:
        _measure(cfg)
    path = cfg.save()
    print(format_status(status))
    print("Checking the data stream...")
    print(format_stats(measure(cfg, 2.0)))
    print(f"Saved to {path}.")
