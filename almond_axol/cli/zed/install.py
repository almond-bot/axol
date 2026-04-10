"""
almond-axol zed.install

Downloads the pyzed wheel matching the installed ZED SDK version into vendor/,
so that 'uv sync --extra zed' can find and install it.
"""

from __future__ import annotations

import platform
import re
import sys
import urllib.request
from pathlib import Path

_ZED_INCLUDE = Path("/usr/local/zed/include")
_VENDOR_DIR = Path(__file__).parent.parent.parent.parent / "vendor"
_BASE_URL = "https://download.stereolabs.com/zedsdk"


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
    subparsers.add_parser(
        "zed.install",
        help="Download the pyzed wheel for the installed ZED SDK version.",
    ).set_defaults(func=run)


def run(_args: object = None) -> None:
    if not _ZED_INCLUDE.exists():
        print(
            "ERROR: ZED SDK not found at /usr/local/zed\n"
            "Install it from https://www.stereolabs.com/developers/release",
            file=sys.stderr,
        )
        sys.exit(1)

    major, minor = _sdk_version()
    py = f"{sys.version_info.major}{sys.version_info.minor}"
    arch = platform.machine().lower()
    sdk_ver = f"{major}.{minor}"

    whl_name = f"pyzed-{sdk_ver}-cp{py}-cp{py}-linux_{arch}.whl"
    url = f"{_BASE_URL}/{sdk_ver}/whl/linux_{arch}/{whl_name}"
    dest = _VENDOR_DIR / whl_name

    if dest.exists():
        print(f"Already downloaded: {dest}")
    else:
        print(
            f"ZED SDK {sdk_ver}  Python {sys.version_info.major}.{sys.version_info.minor}  {arch}"
        )
        print(f"Downloading {url}")
        _VENDOR_DIR.mkdir(exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        print(f"Saved to {dest}")

    print("\nRun 'uv sync --extra zed' to install.")
