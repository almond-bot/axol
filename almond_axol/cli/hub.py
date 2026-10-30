"""
axol hub.info / axol hub.flash

Inspect and update the Axol hubs: the RP2350 USB-to-CAN bridge for both arms,
and the Mantis hub, which is the same board on the same firmware. Arm
bring-up already keeps the arm hub on the firmware this package ships
(``almond_axol.hub.ensure_hub_firmware``); these are for checking a hub by
hand, for the one manual flash a hub too old to reboot itself needs, and for
recovering a hub left in its bootloader.

Both commands find hubs on USB rather than through CAN interfaces, so they
also see a hub whose interfaces never came up.

Flashing needs root: run as a normal user, ``hub.flash`` picks the hubs and
asks its questions first, then runs each flash through ``sudo`` (one password
prompt), the same root step the automatic bring-up flash uses.

    axol hub.info
    axol hub.flash                 # every outdated hub, and any hub in its bootloader
    axol hub.flash --port 1-3      # one hub by USB port
    axol hub.flash --force         # reflash even if already current
"""

from __future__ import annotations

import argparse
import os
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
        "hub.flash", help="Flash the bundled firmware onto the Axol hub (sudo)."
    )
    flash.add_argument(
        "--port",
        default=None,
        help="USB port path of the hub (e.g. 1-3, from hub.info); also flashes "
        "a hub sitting in its RP2350 bootloader without asking. Default: every "
        "outdated hub on USB, plus any RP2350 bootloader you confirm.",
    )
    flash.add_argument(
        "--force",
        action="store_true",
        help="Flash even when the hub already runs the bundled version.",
    )
    flash.set_defaults(func=run_flash)


_NO_HUB = (
    "No Axol hub found on USB. If the hub's LED is on but it does not show up, "
    "flip the BOOTSEL switch next to its USB port, replug the hub, then run "
    "`axol hub.flash`."
)


def _bootloader_line(port: str) -> str:
    return (
        f"RP2350 bootloader at USB {port}: a hub there is running no firmware; "
        f"flash it with `axol hub.flash --port {port}`"
    )


def _confirm_bootloader(port: str) -> bool:
    """Ask before flashing a bootrom: it does not say which RP2350 board it is."""
    if not sys.stdin.isatty():
        print(f"{_bootloader_line(port)} (not flashed without a confirmation)")
        return False
    answer = input(
        f"An RP2350 is in its bootloader at USB {port}. Flash the Axol hub "
        "firmware onto it? Only if it is the hub [y/N] "
    )
    return answer.strip().lower() in ("y", "yes")


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
    """Print each hub's firmware version (and, as root, its last reset)."""
    hubs = hub_fw.attached_hubs()
    for port in hub_fw.bootloader_ports():
        print(_bootloader_line(port))
    if not hubs:
        print(_NO_HUB)
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
    """Flash the bundled image onto one hub (``--port``) or every hub that needs it."""
    if args.port:
        ports = [args.port]
    else:
        hubs = hub_fw.attached_hubs()
        bootloaders = hub_fw.bootloader_ports()
        if not hubs and not bootloaders:
            print(_NO_HUB)
            raise SystemExit(1)
        ports = [hub.port for hub in hubs if hub.outdated or args.force]
        ports += [port for port in bootloaders if _confirm_bootloader(port)]
        if not ports and bootloaders:
            return
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
            if os.geteuid() == 0:
                version = hub_fw.flash(port)
            else:
                version = hub_fw.flash_via_sudo(port)
        except hub_fw.HubFirmwareError as exc:
            # stderr: ensure_hub_firmware reports this line when it runs the
            # flash through sudo.
            print(f"ERROR: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
        print(f"  Done — the hub runs firmware {hub_fw.format_version(version)}.")
