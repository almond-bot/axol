"""
axol can.driver

Builds and installs the ``gs_usb`` kernel module for the Almond Axol Hub CAN
adapter on kernels that do not ship it (NVIDIA L4T/tegra kernels on Jetson /
ZED Box hardware are built without any USB-CAN drivers).

The vendored source in ``gs_usb/`` is the upstream stable v5.15.148 driver
with two backports the Axol Hub needs — see ``gs_usb/README.md``. The module
is compiled against the running kernel's headers, installed under
``/lib/modules/$(uname -r)/updates/``, and registered in
``/etc/modules-load.d/`` so it loads on every boot. On kernels whose ``gs_usb``
already works (any stock desktop kernel) this whole command is a no-op.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from ...utils.sudo import run_root

_SRC_DIR = Path(__file__).parent / "gs_usb"
_BUILD_DIR = Path.home() / ".almond" / "can" / "gs_usb-build"
_MODULES_LOAD_FILE = Path("/etc/modules-load.d/gs_usb.conf")


def _find_modinfo() -> str | None:
    """Locate ``modinfo``, including the sbin dirs minimal PATHs often omit."""
    found = shutil.which("modinfo")
    if found:
        return found
    for candidate in ("/usr/sbin/modinfo", "/sbin/modinfo"):
        if Path(candidate).exists():
            return candidate
    return None


def is_driver_available() -> bool:
    """True when the running kernel can already load ``gs_usb``."""
    modinfo = _find_modinfo()
    if modinfo is None:
        raise RuntimeError(
            "`modinfo` not found. Install kmod first (`sudo apt install kmod`)."
        )
    return subprocess.run([modinfo, "gs_usb"], capture_output=True).returncode == 0


def _build() -> Path:
    """Compile gs_usb.ko against the running kernel. Returns the .ko path."""
    kver = os.uname().release
    kdir = Path("/lib/modules") / kver / "build"
    if not kdir.exists():
        raise RuntimeError(
            f"Kernel headers not found at {kdir}. Install them first "
            "(on Jetson/L4T: `sudo apt install nvidia-l4t-kernel-headers`)."
        )
    for tool in ("make", "gcc"):
        if shutil.which(tool) is None:
            raise RuntimeError(
                f"`{tool}` not found. Install build tools first "
                "(`sudo apt install build-essential`)."
            )

    print(f"Building gs_usb for kernel {kver}...")
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("gs_usb.c", "Makefile"):
        shutil.copy(_SRC_DIR / name, _BUILD_DIR / name)

    proc = subprocess.run(
        ["make", "-C", str(_BUILD_DIR)], capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gs_usb build failed:\n{proc.stdout}\n{proc.stderr}")
    print("  Done.")
    return _BUILD_DIR / "gs_usb.ko"


def _install(ko: Path) -> None:
    """Install the module, register it for boot, and load it (requires sudo)."""
    kver = os.uname().release
    dest = Path("/lib/modules") / kver / "updates" / "gs_usb.ko"

    print(f"Installing {dest} (requires sudo)...")
    run_root(["install", "-D", "-m", "644", str(ko), str(dest)], check=True)
    run_root(["depmod", "-a"], check=True)
    run_root(["tee", str(_MODULES_LOAD_FILE)], input_text="gs_usb\n", check=True)
    run_root(["modprobe", "gs_usb"], check=True)
    print("  Done.")


def ensure_driver() -> bool:
    """Build and install gs_usb when the running kernel lacks it.

    Returns True when the driver was installed, False when it was already
    available. Idempotent; safe to call from ``can.setup`` on every machine.
    """
    if is_driver_available():
        return False
    print("Kernel does not ship the gs_usb driver — building it from source.")
    ko = _build()
    _install(ko)
    return True


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``can.driver`` subcommand."""
    subparsers.add_parser(
        "can.driver",
        help="Build and install the gs_usb kernel driver if the kernel lacks it.",
    ).set_defaults(func=run)


def run(_args: object = None) -> None:
    """Ensure the gs_usb driver is available, building it when needed."""
    try:
        installed = ensure_driver()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if installed:
        print()
        print("gs_usb driver installed and loaded.")
        print(f"  It will load automatically on boot via {_MODULES_LOAD_FILE}.")
        print(
            "  Replug the Axol Hub (or it may already have enumerated) and "
            "run `axol can.setup`."
        )
    else:
        print("gs_usb driver already available — nothing to do.")
