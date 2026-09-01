"""
axol zed.install

Downloads and installs the pyzed wheel matching the installed ZED SDK version.
pyzed is not on PyPI, so this command handles the install directly.
"""

from __future__ import annotations

import email.parser
import platform
import re
import stat
import subprocess
import sys
import zipfile
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path, PurePosixPath

from .download import atomic_https_download

_ZED_INCLUDE = Path("/usr/local/zed/include")
_CACHE_DIR = Path.home() / ".almond" / "wheels"
_BASE_URL = "https://download.stereolabs.com/zedsdk"
_WHEEL_MAX_BYTES = 256 * 1024 * 1024
_SUPPORTED_ARCHITECTURES = {
    "aarch64": "aarch64",
    "arm64": "aarch64",
    "amd64": "x86_64",
    "x86_64": "x86_64",
}
_ALLOWED_REQUIREMENTS = {"cython", "numpy", "opencv-python", "pyopengl"}


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


def _normalise_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    if match is None:
        raise RuntimeError("pyzed wheel contains an invalid dependency declaration")
    return _normalise_distribution(match.group(1))


def _validate_wheel(
    wheel: Path,
    *,
    sdk_version: str,
    python_tag: str,
    architecture: str,
) -> None:
    """Validate wheel identity, target tags, dependencies, and archive paths."""
    if wheel.is_symlink() or not wheel.is_file():
        raise RuntimeError("downloaded pyzed wheel is not a regular file")

    expected_tag = f"cp{python_tag}-cp{python_tag}-linux_{architecture}"
    try:
        with zipfile.ZipFile(wheel) as archive:
            infos = archive.infolist()
            if not infos or archive.testzip() is not None:
                raise RuntimeError("pyzed wheel is empty or corrupt")

            names: set[str] = set()
            folded_names: set[str] = set()
            for info in infos:
                name = info.filename
                path = PurePosixPath(name)
                mode = info.external_attr >> 16
                if (
                    not name
                    or "\\" in name
                    or path.is_absolute()
                    or ".." in path.parts
                    or name in names
                    or name.casefold() in folded_names
                    or stat.S_ISLNK(mode)
                ):
                    raise RuntimeError("pyzed wheel contains an unsafe archive path")
                names.add(name)
                folded_names.add(name.casefold())

            metadata_names = [
                name for name in names if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                raise RuntimeError("pyzed wheel must contain exactly one METADATA file")
            metadata_name = metadata_names[0]
            dist_info = metadata_name.removesuffix("/METADATA")
            expected_dist_info = f"pyzed-{sdk_version}.dist-info"
            if dist_info != expected_dist_info:
                raise RuntimeError(
                    f"pyzed wheel metadata directory is {dist_info!r}, "
                    f"expected {expected_dist_info!r}"
                )

            metadata = email.parser.BytesParser().parsebytes(
                archive.read(metadata_name)
            )
            if _normalise_distribution(metadata.get("Name", "")) != "pyzed":
                raise RuntimeError("downloaded wheel distribution is not pyzed")
            if metadata.get("Version") != sdk_version:
                raise RuntimeError(
                    f"pyzed wheel version is {metadata.get('Version')!r}, "
                    f"expected {sdk_version!r}"
                )
            dependencies = {
                _requirement_name(value)
                for value in metadata.get_all("Requires-Dist", [])
            }
            unexpected = sorted(dependencies - _ALLOWED_REQUIREMENTS)
            if unexpected:
                raise RuntimeError(
                    "pyzed wheel declares unexpected dependencies: "
                    + ", ".join(unexpected)
                )

            wheel_metadata_name = f"{dist_info}/WHEEL"
            if wheel_metadata_name not in names:
                raise RuntimeError("pyzed wheel has no WHEEL metadata")
            wheel_metadata = email.parser.BytesParser().parsebytes(
                archive.read(wheel_metadata_name)
            )
            if wheel_metadata.get("Root-Is-Purelib", "").lower() != "false":
                raise RuntimeError("pyzed wheel unexpectedly claims to be pure Python")
            if wheel_metadata.get_all("Tag", []) != [expected_tag]:
                raise RuntimeError(
                    "pyzed wheel target tag does not match this interpreter and host"
                )

            if not any(
                name.startswith("pyzed/") and name.endswith(".so") for name in names
            ):
                raise RuntimeError("pyzed wheel does not contain its native extension")
    except zipfile.BadZipFile as exc:
        raise RuntimeError("downloaded pyzed wheel is not a valid ZIP archive") from exc


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
    reported_arch = platform.machine().lower()
    try:
        arch = _SUPPORTED_ARCHITECTURES[reported_arch]
    except KeyError:
        print(
            f"ERROR: pyzed wheels are not available for architecture {reported_arch!r}",
            file=sys.stderr,
        )
        sys.exit(1)
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
        atomic_https_download(
            url,
            dest,
            max_bytes=_WHEEL_MAX_BYTES,
            validate=lambda path: _validate_wheel(
                path,
                sdk_version=sdk_ver,
                python_tag=py,
                architecture=arch,
            ),
        )
        print(f"Saved to {dest}")

    try:
        _validate_wheel(
            dest,
            sdk_version=sdk_ver,
            python_tag=py,
            architecture=arch,
        )
    except Exception:
        dest.unlink(missing_ok=True)
        raise

    print(f"Installing {whl_name}...")
    # Pin the target to the interpreter running this CLI: a bare `uv pip
    # install` resolves the environment from VIRTUAL_ENV/cwd, which is wrong
    # when axol is installed as a uv tool.
    subprocess.check_call(
        ["uv", "pip", "install", "--python", sys.executable, str(dest)]
    )
    print("Done. pyzed is installed.")
