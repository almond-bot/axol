"""
axol zed.install

Downloads and installs the pyzed wheel matching the installed ZED SDK version.
pyzed is not on PyPI, so this command handles the install directly.
"""

from __future__ import annotations

import platform
import re
import subprocess
import sys
import urllib.request
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

_ZED_INCLUDE = Path("/usr/local/zed/include")
_CACHE_DIR = Path.home() / ".almond" / "wheels"
_BASE_URL = "https://download.stereolabs.com/zedsdk"


def _pyzed_installed(major: str, minor: str) -> bool:
    """True when pyzed for this SDK ``major.minor`` is already installed.

    Read from the package metadata of the interpreter running this CLI (the uv
    tool env). ``uv tool upgrade`` rebuilds that env and drops pyzed (not a PyPI
    dependency), so this correctly returns False right after an upgrade and lets
    ``axol provision`` skip a redundant reinstall otherwise.
    """
    try:
        installed = _pkg_version("pyzed")
    except PackageNotFoundError:
        return False
    return installed.split(".")[:2] == [major, minor]


def _sdk_version() -> tuple[str, str]:
    for header in (
        _ZED_INCLUDE / "sl" / "Camera.hpp",
        _ZED_INCLUDE / "sl_zed" / "defines.hpp",
    ):
        if not header.exists():
            continue
        text = header.read_text()
        major = re.search(r"ZED_SDK_MAJOR_VERSION\s+(\d+)", text)
        minor = re.search(r"ZED_SDK_MINOR_VERSION\s+(\d+)", text)
        if major and minor:
            return major.group(1), minor.group(1)
    print(
        "ERROR: ZED SDK not found at /usr/local/zed\n"
        "Install it from https://www.stereolabs.com/developers/release",
        file=sys.stderr,
    )
    sys.exit(1)


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``zed.install`` subcommand."""
    parser = subparsers.add_parser(
        "zed.install",
        help="Download the pyzed wheel for the installed ZED SDK version.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reinstall pyzed even when the matching version is already present.",
    )
    parser.set_defaults(func=run)


def run(_args: object = None) -> None:
    """Download and install the pyzed wheel for the installed ZED SDK."""
    if not _ZED_INCLUDE.exists():
        print(
            "ERROR: ZED SDK not found at /usr/local/zed\n"
            "Install it from https://www.stereolabs.com/developers/release",
            file=sys.stderr,
        )
        sys.exit(1)

    major, minor = _sdk_version()

    if not getattr(_args, "force", False) and _pyzed_installed(major, minor):
        print(f"pyzed {major}.{minor} already installed.")
        return
    py = f"{sys.version_info.major}{sys.version_info.minor}"
    arch = platform.machine().lower()
    sdk_ver = f"{major}.{minor}"

    whl_name = f"pyzed-{sdk_ver}-cp{py}-cp{py}-linux_{arch}.whl"
    url = f"{_BASE_URL}/{sdk_ver}/whl/linux_{arch}/{whl_name}"
    dest = _CACHE_DIR / whl_name

    if dest.exists():
        print(f"Already downloaded: {dest}")
    else:
        print(
            f"ZED SDK {sdk_ver}  Python {sys.version_info.major}.{sys.version_info.minor}  {arch}"
        )
        print(f"Downloading {url}")
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        print(f"Saved to {dest}")

    print(f"Installing {whl_name}...")
    # Pin the target to the interpreter running this CLI: a bare `uv pip
    # install` resolves the environment from VIRTUAL_ENV/cwd, which is wrong
    # when axol is installed as a uv tool.
    subprocess.check_call(
        ["uv", "pip", "install", "--python", sys.executable, str(dest)]
    )
    print("Done. pyzed is installed.")
