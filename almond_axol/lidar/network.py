"""Find the lidar on a wired port and give that port a static address.

A Livox lidar ships with a fixed address (``192.168.1.1XX``, from its
serial) and sends to a fixed host address (``192.168.1.5`` by default). That
subnet is also the most common home and office LAN, so a plain
``192.168.1.0/24`` address on the lidar's port would fight the robot's Wi-Fi
for the whole subnet. Instead, the profile written here gives the port one
``/32`` address and one ``/32`` route to the lidar: the robot reaches the lidar
over the cable and everything else on its network as before.

The profile is the only one NetworkManager has for that port, so it applies to
any cable plugged in there, not only the lidar's. It therefore also runs DHCP
and never gives up on it (``ipv4.dhcp-timeout infinity``): with the lidar alone
the static address comes up at once and DHCP keeps asking in the background,
and on a router the port gets a lease, a default route and DNS. Without that,
a robot whose Wi-Fi isn't in range is unreachable over Ethernet (no lease, no
default route, so no Tailscale either).

Finding a lidar on a port with no IPv4 address can't use UDP (the kernel
drops the replies), so :func:`sniff_lidar` reads raw Ethernet frames, which
needs root. The CLI runs it through ``sudo`` (``python -m
almond_axol.lidar.network sniff IFACE``).
"""

from __future__ import annotations

import ipaddress
import json
import socket
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import livox

PROFILE_NAME = "axol-lidar"
SYSCTL_PATH = Path("/etc/sysctl.d/60-axol-lidar.conf")
SYSCTL_TEXT = (
    "# Written by `axol lidar.setup`: room for the lidar's point stream.\n"
    "net.core.rmem_max = 16777216\n"
)
_SYS_NET = Path("/sys/class/net")
_ETH_P_ALL = 0x0003
_ETH_P_IP = 0x0800
_ETH_P_ARP = 0x0806
_LIVOX_PORTS = range(livox.DISCOVERY_PORT, livox.IMU_PORT + 200)


def wired_interfaces() -> list[str]:
    """Physical Ethernet ports, by name (wireless, CAN, bridges and VPNs excluded)."""
    out = []
    for dev in sorted(_SYS_NET.iterdir()) if _SYS_NET.exists() else []:
        try:
            if (dev / "type").read_text().strip() != "1":  # ARPHRD_ETHER
                continue
        except OSError:
            continue
        if (
            (dev / "wireless").exists()
            or (dev / "phy80211").exists()
            or (dev / "bridge").exists()
            or not (dev / "device").exists()
        ):
            continue
        # USB gadget ports (l4tbr0's usb0/usb1 on a Jetson) are not real wired ports.
        driver = dev / "device" / "driver"
        if driver.exists() and driver.resolve().name in {"rndis_host", "cdc_ether"}:
            if dev.name.startswith("usb"):
                continue
        out.append(dev.name)
    return out


def link_up(iface: str) -> bool:
    try:
        return (_SYS_NET / iface / "carrier").read_text().strip() == "1"
    except OSError:
        return False


def link_speed(iface: str) -> int | None:
    try:
        return int((_SYS_NET / iface / "speed").read_text().strip())
    except (OSError, ValueError):
        return None


@dataclass
class SniffResult:
    """What a lidar revealed by talking on the wire."""

    iface: str
    lidar_ip: str
    # The address the lidar is trying to send to (from its ARP requests).
    wanted_host_ip: str = ""
    lidar_mac: str = ""


def _parse_frame(frame: bytes) -> tuple[str, str, str, str] | None:
    """(kind, src_ip, dst_ip_or_target, src_mac) for IPv4/UDP-from-Livox and ARP."""
    if len(frame) < 42:
        return None
    (ethertype,) = struct.unpack_from("!H", frame, 12)
    mac = frame[6:12].hex(":")
    if ethertype == _ETH_P_ARP:
        (op,) = struct.unpack_from("!H", frame, 20)
        if op != 1:
            return None
        return (
            "arp",
            socket.inet_ntoa(frame[28:32]),
            socket.inet_ntoa(frame[38:42]),
            mac,
        )
    if ethertype != _ETH_P_IP or frame[23] != 17:
        return None
    ihl = (frame[14] & 0x0F) * 4
    if len(frame) < 14 + ihl + 8:
        return None
    (src_port,) = struct.unpack_from("!H", frame, 14 + ihl)
    if src_port not in _LIVOX_PORTS:
        return None
    return "udp", socket.inet_ntoa(frame[26:30]), socket.inet_ntoa(frame[30:34]), mac


def sniff_lidar(iface: str, timeout: float = 5.0) -> SniffResult | None:
    """Listen on ``iface`` for a Livox lidar's traffic (needs root).

    A powered Mid-360 broadcasts a status push about once a second, and ARPs
    for the host it's configured to stream to. Both name its address.
    """
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(_ETH_P_ALL))
    try:
        sock.bind((iface, 0))
        sock.settimeout(0.25)
        result: SniffResult | None = None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                frame = sock.recv(65535)
            except TimeoutError:
                continue
            parsed = _parse_frame(frame)
            if parsed is None:
                continue
            kind, src, dst, mac = parsed
            if kind == "udp":
                if result is None:
                    result = SniffResult(iface, src, lidar_mac=mac)
                if dst != "255.255.255.255" and not result.wanted_host_ip:
                    result.wanted_host_ip = dst
            elif kind == "arp" and result is not None and src == result.lidar_ip:
                result.wanted_host_ip = dst
            if result is not None and result.wanted_host_ip:
                break
        return result
    finally:
        sock.close()


def sniff_lidar_as_root(iface: str, timeout: float = 5.0) -> SniffResult | None:
    """Run :func:`sniff_lidar` through ``sudo`` when this process isn't root."""
    import os

    if os.geteuid() == 0:
        return sniff_lidar(iface, timeout)
    from ..utils.sudo import run_root

    # Import this same copy of the package (an installed one or a checkout):
    # -I ignores PYTHONPATH, and sudo would drop it anyway.
    root = str(Path(__file__).resolve().parents[2])
    code = (
        "import sys; sys.path.insert(0, sys.argv.pop(1)); "
        "from almond_axol.lidar.network import _main; "
        "raise SystemExit(_main(sys.argv[1:]))"
    )
    proc = run_root(
        [sys.executable, "-I", "-c", code, root, "sniff", iface, str(timeout)]
    )
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        raise RuntimeError(
            f"Couldn't listen on {iface}: {detail[-1] if detail else proc.returncode}"
        )
    data = json.loads(proc.stdout or "null")
    return SniffResult(**data) if data else None


def choose_host_ip(lidar_ip: str, wanted: str = "") -> str:
    """The address to give this computer on the lidar's port.

    Prefer the one the lidar already streams to, so other Livox tools keep
    working; otherwise ``.5`` on the lidar's subnet (``.50`` if the lidar
    itself is ``.5``).
    """
    lidar = ipaddress.IPv4Address(lidar_ip)
    if wanted and wanted != "255.255.255.255" and wanted != lidar_ip:
        return wanted
    net = ipaddress.IPv4Network(f"{lidar}/24", strict=False)
    for last in (5, 50, 55):
        candidate = net.network_address + last
        if candidate != lidar:
            return str(candidate)
    raise AssertionError("unreachable")


def nmcli_profile_args(iface: str, host_ip: str, lidar_ip: str) -> list[str]:
    """``nmcli connection`` property arguments for the lidar port's profile."""
    return [
        "connection.interface-name",
        iface,
        "connection.autoconnect",
        "yes",
        "connection.autoconnect-priority",
        "100",
        # DHCP alongside the static address, so a router on this port works
        # too. Explicit values, so re-running setup also fixes an older
        # static-only profile.
        "ipv4.method",
        "auto",
        "ipv4.addresses",
        f"{host_ip}/32",
        "ipv4.routes",
        f"{lidar_ip}/32",
        "ipv4.dhcp-timeout",
        "infinity",
        "ipv4.may-fail",
        "yes",
        "ipv4.never-default",
        "no",
        "ipv4.ignore-auto-dns",
        "no",
        "ipv6.method",
        "link-local",
    ]


def apply_profile(iface: str, host_ip: str, lidar_ip: str) -> None:
    """Create or update the ``axol-lidar`` NetworkManager profile and bring it up."""
    from ..utils.sudo import run_root

    exists = (
        subprocess.run(
            ["nmcli", "-t", "-f", "NAME", "connection", "show", PROFILE_NAME],
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )
    props = nmcli_profile_args(iface, host_ip, lidar_ip)
    if exists:
        run_root(["nmcli", "connection", "modify", PROFILE_NAME, *props], check=True)
    else:
        run_root(
            [
                "nmcli",
                "connection",
                "add",
                "type",
                "ethernet",
                "con-name",
                PROFILE_NAME,
                *props,
            ],
            check=True,
        )
    run_root(["nmcli", "connection", "up", PROFILE_NAME], check=True)


def apply_sysctl() -> None:
    from ..utils.sudo import run_root

    run_root(["tee", str(SYSCTL_PATH)], input_text=SYSCTL_TEXT, check=True)
    run_root(["sysctl", "-q", "-p", str(SYSCTL_PATH)], check=True)


def host_has_address(iface: str, ip: str) -> bool:
    proc = subprocess.run(
        ["ip", "-4", "-o", "addr", "show", "dev", iface],
        capture_output=True,
        text=True,
    )
    return f" {ip}/" in proc.stdout


def _main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "sniff":
        timeout = float(argv[2]) if len(argv) > 2 else 5.0
        result = sniff_lidar(argv[1], timeout)
        print(json.dumps(asdict(result) if result else None))
        return 0
    print("usage: python -m almond_axol.lidar.network sniff IFACE [TIMEOUT]")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
