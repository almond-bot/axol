"""
axol zed.driver

Upgrades the ZED Box Duo's factory-flashed GMSL capture driver
(``stereolabs-zedbox-duo``) to the pinned known-good release. ZED Box Duo
units ship with a buggy driver version, so this replaces it with the pinned
release from Stereolabs' download server.

Gated hard on the target hardware: it only acts when the
``stereolabs-zedbox-duo`` package is already installed (i.e. the host is a
factory-flashed ZED Box Duo) *and* the running L4T release matches the one the
pinned .deb was built for — so it is a quiet no-op on any other machine and
can never downgrade a newer driver (``dpkg --compare-versions`` guards that).

The new driver is a kernel module + device-tree update, so it only takes
effect after a reboot. This command NEVER reboots the box itself — it runs
from ``axol provision`` (over the operator's SSH session during install, and
from the running ``axol serve`` process after a self-update), where an
in-place reboot would drop the session or kill the robot mid-use. It prints a
reboot-required notice instead.
"""

from __future__ import annotations

import re
import subprocess
import sys
import urllib.request
from pathlib import Path

from ...utils.sudo import run_root

_PACKAGE = "stereolabs-zedbox-duo"
_TARGET_VERSION = "1.4.2"
_DEB_URL = (
    "https://download.stereolabs.com/drivers/zedx/1.4.2/R36.4/"
    "stereolabs-zedbox-duo_1.4.2-LI-MAX96712-ZEDBOX-L4T36.4.0_arm64.deb"
)
# The pinned .deb is built against L4T 36.4.x (JetPack 6.x on the ZED Box
# Duo). Installing it on any other L4T would leave the cameras dead, so the
# upgrade is skipped (with a warning) when the running release differs —
# bump _DEB_URL/_TARGET_VERSION together with this when moving to a new L4T.
_L4T_RELEASE = "36"
_L4T_REVISION_MAJOR = "4"
_L4T_RELEASE_FILE = Path("/etc/nv_tegra_release")
# Persistent cache (like zed.install's wheel cache) so a failed install can be
# recovered by re-running — or by the manual `dpkg -i` the failure prints —
# without re-downloading.
_CACHE_DIR = Path.home() / ".almond" / "drivers"


def _installed_version() -> str | None:
    """Installed ``stereolabs-zedbox-duo`` dpkg version, or None when absent."""
    try:
        proc = subprocess.run(
            ["dpkg-query", "-W", "-f", "${Version}", _PACKAGE],
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    version = proc.stdout.strip()
    return version if proc.returncode == 0 and version else None


def _l4t_matches() -> bool:
    """True when the running L4T release matches the pinned .deb's target."""
    try:
        first_line = _L4T_RELEASE_FILE.read_text().splitlines()[0]
    except (OSError, IndexError):
        return False
    # e.g. "# R36 (release), REVISION: 4.0, GCID: ..."
    match = re.search(r"R(\d+)\s*\(release\),\s*REVISION:\s*(\d+)", first_line)
    if match is None:
        return False
    return match.group(1) == _L4T_RELEASE and match.group(2) == _L4T_REVISION_MAJOR


def _is_older(installed: str) -> bool:
    """True when ``installed`` is strictly older than the pinned target."""
    return (
        subprocess.run(
            ["dpkg", "--compare-versions", installed, "lt", _TARGET_VERSION],
            capture_output=True,
        ).returncode
        == 0
    )


def _download_deb() -> Path:
    """Download the pinned .deb into the cache and verify it is a valid archive.

    Everything that can fail without root — the download and the archive check
    — happens here, *before* the factory package is removed, so a network error
    or truncated download can never leave the box with no driver installed.
    The cached copy also makes a re-run (or the printed manual recovery
    command) work offline.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    deb = _CACHE_DIR / _DEB_URL.rsplit("/", 1)[-1]
    if not deb.exists():
        print(f"Downloading {_DEB_URL}")
        # Download to a temp name and rename so an interrupted download can
        # never be mistaken for a complete cached .deb on the next run.
        partial = deb.with_suffix(".part")
        urllib.request.urlretrieve(_DEB_URL, partial)
        partial.rename(deb)
    else:
        print(f"Already downloaded: {deb}")
    proc = subprocess.run(
        ["dpkg-deb", "--info", str(deb)], capture_output=True, text=True
    )
    if proc.returncode != 0:
        deb.unlink()  # corrupt — drop it so the next run re-downloads
        raise RuntimeError(
            f"downloaded .deb failed verification: {(proc.stderr or '').strip()}"
        )
    return deb


def _upgrade() -> None:
    """Replace the installed factory package with the pinned .deb (needs root)."""
    deb = _download_deb()
    # Remove the factory package first (per Stereolabs' upgrade procedure)
    # rather than upgrading in place; best-effort since a half-removed
    # package still gets replaced by the install below.
    print(f"Removing the factory {_PACKAGE} package (requires sudo)...")
    run_root(["dpkg", "-r", _PACKAGE])
    print(f"Installing {deb.name}...")
    try:
        run_root(["dpkg", "-i", str(deb)], check=True)
    except RuntimeError:
        # The factory package is already removed at this point, so don't fail
        # silently: tell the operator exactly how to finish the install by
        # hand (the verified .deb stays in the cache).
        print(
            f"ERROR: installing {deb.name} failed after the factory package "
            f"was removed — the box has NO camera driver until this is fixed. "
            f"Recover with: sudo dpkg -i {deb}",
            file=sys.stderr,
        )
        raise


def ensure_driver() -> bool:
    """Upgrade the ZED Box Duo camera driver when the factory one is outdated.

    Returns True when the driver was upgraded (a reboot is then required for
    it to load), False when there was nothing to do. Idempotent and self-gating
    (a no-op on anything that isn't a ZED Box Duo on the pinned L4T), so it is
    safe to run from ``axol provision`` on every host.
    """
    installed = _installed_version()
    if installed is None:
        # Not a factory-flashed ZED Box Duo (or dpkg-less host) — nothing to do.
        return False
    if not _is_older(installed):
        print(f"{_PACKAGE} {installed} already >= {_TARGET_VERSION}.")
        return False
    if not _l4t_matches():
        print(
            f"WARNING: {_PACKAGE} {installed} is outdated, but the pinned "
            f"{_TARGET_VERSION} driver targets L4T {_L4T_RELEASE}."
            f"{_L4T_REVISION_MAJOR} and this host runs a different release — "
            "skipping. Update the pin in almond_axol/cli/zed/driver.py.",
            file=sys.stderr,
        )
        return False
    print(
        f"{_PACKAGE} {installed} is the known-bad factory driver — "
        f"upgrading to {_TARGET_VERSION}."
    )
    _upgrade()
    print()
    print(
        f"REBOOT REQUIRED: {_PACKAGE} {_TARGET_VERSION} is installed but the "
        "new kernel driver only loads at boot. Reboot when convenient "
        "(sudo reboot)."
    )
    return True


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``zed.driver`` subcommand."""
    subparsers.add_parser(
        "zed.driver",
        help=(
            "Upgrade the ZED Box Duo camera driver (stereolabs-zedbox-duo) "
            "to the pinned release."
        ),
    ).set_defaults(func=run)


def run(_args: object = None) -> None:
    """Ensure the pinned ZED Box Duo camera driver is installed."""
    try:
        upgraded = ensure_driver()
    except Exception as exc:  # noqa: BLE001 - network/dpkg failures land here
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    if not upgraded and _installed_version() is None:
        print(
            f"{_PACKAGE} is not installed — not a factory-flashed ZED Box Duo; "
            "nothing to do."
        )
