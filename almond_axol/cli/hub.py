"""
axol hub.info / axol hub.flash

Inspect and update the Axol arm hub (the RP2350 USB-to-CAN bridge for both
arms). Bring-up already keeps the hub on the firmware this package ships
(``almond_axol.hub.ensure_hub_firmware``); these are for checking it by hand,
for the one manual flash a hub too old to reboot itself needs, and for
recovering a hub left in its bootloader.

    axol hub.info
    sudo axol hub.flash                 # every outdated hub on the arm buses
    sudo axol hub.flash --port 1-3      # one hub by USB port, e.g. in BOOTSEL
    sudo axol hub.flash --force         # reflash even if already current
"""

from __future__ import annotations

import argparse
import sys
import time

from ..hub import firmware as hub_fw


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``hub.info`` and ``hub.flash`` subcommands."""
    info = subparsers.add_parser(
        "hub.info", help="Show the Axol hub's firmware version and last reset."
    )
    info.set_defaults(func=run_info)

    flash = subparsers.add_parser(
        "hub.flash", help="Flash the bundled firmware onto the Axol hub (root)."
    )
    flash.add_argument(
        "--port",
        default=None,
        help="USB port path of the hub (e.g. 1-3, from hub.info); also flashes "
        "a hub sitting in its RP2350 bootloader. Default: the hub(s) behind "
        "the configured arm channels.",
    )
    flash.add_argument(
        "--force",
        action="store_true",
        help="Flash even when the hub already runs the bundled version.",
    )
    flash.set_defaults(func=run_flash)


def _arm_channels() -> list[str]:
    from ..settings import load_store

    return [ch for ch in load_store().can_channels() if ch]


_CHIP_RESET_BITS = {
    0x10000000: "WATCHDOG_RESET_RSM",
    0x08000000: "HZD_SYS_RESET_REQ",
    0x04000000: "GLITCH_DETECT",
    0x02000000: "SWCORE_PD",
    0x01000000: "WATCHDOG_RESET_SWCORE",
    0x00800000: "WATCHDOG_RESET_POWMAN",
    0x00400000: "WATCHDOG_RESET_POWMAN_ASYNC",
    0x00200000: "RESCUE",
    0x00080000: "DP_RESET_REQ",
    0x00040000: "RUN_LOW",
    0x00020000: "BOR",
    0x00010000: "POR",
}


def run_info(args: argparse.Namespace | None = None) -> None:
    """Print each arm hub's firmware version (and, as root, its last reset)."""
    hubs = hub_fw.hubs_for_channels(_arm_channels())
    if not hubs:
        print("No Axol hub found behind the configured arm channels.")
        raise SystemExit(1)
    bundled = hub_fw.format_version(hub_fw.FIRMWARE_VERSION)
    for hub in hubs:
        state = "outdated" if hub.outdated else "current"
        print(f"hub {hub.serial} (USB {hub.port})")
        print(
            f"  firmware   {hub_fw.format_version(hub.version)} "
            f"({state}; bundled {bundled})"
        )
        info = hub_fw.reset_info(hub)
        if info is None:
            print(
                "  last reset unavailable (needs root and firmware >= 1.00 of 2026-09-25)"
            )
            continue
        names = [n for bit, n in _CHIP_RESET_BITS.items() if info["chip_reset"] & bit]
        booted = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(time.time() - info["uptime_us"] / 1e6),
        )
        print(f"  last reset {'|'.join(names) or 'none'} (0x{info['chip_reset']:08x})")
        print(f"  uptime     {info['uptime_us'] / 1e6:.1f} s (booted ~{booted})")


def run_flash(args: argparse.Namespace) -> None:
    """Flash the bundled image onto one hub (``--port``) or every outdated arm hub."""
    if args.port:
        ports = [args.port]
    else:
        hubs = hub_fw.hubs_for_channels(_arm_channels())
        if not hubs:
            print(
                "No Axol hub found behind the configured arm channels. A hub in "
                "its bootloader has no CAN interfaces: pass --port."
            )
            raise SystemExit(1)
        ports = [hub.port for hub in hubs if hub.outdated or args.force]
        if not ports:
            print(
                "Axol hub already runs firmware "
                f"{hub_fw.format_version(hub_fw.FIRMWARE_VERSION)}; nothing to do "
                "(--force to reflash)."
            )
            return
    for port in ports:
        print(
            f"Flashing Axol hub firmware {hub_fw.format_version(hub_fw.FIRMWARE_VERSION)} "
            f"at USB {port}..."
        )
        try:
            version = hub_fw.flash(port)
        except hub_fw.HubFirmwareError as exc:
            # stderr: ensure_hub_firmware reports this line when it runs the
            # flash through sudo.
            print(f"ERROR: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
        print(f"  Done — the hub runs firmware {hub_fw.format_version(version)}.")
