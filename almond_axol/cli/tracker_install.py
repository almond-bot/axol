"""Install the pinned libsurvive Lighthouse tracking runtime.

``survive-cli`` is a native application rather than a Python package, so it
cannot be carried by the normal ``uv tool`` install.  This command is shared by
``axol provision`` and the control panel's Mantis setup UI.  It is intentionally
idempotent: an already-installed pinned build is a fast no-op.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
from pathlib import Path

from ..tracker.survive import is_available
from ..utils.sudo import prime_sudo, run_root

_logger = logging.getLogger(__name__)

_REPO_URL = "https://github.com/collabora/libsurvive.git"
_PINNED_REF = "f1e6eddb669320f2a30760f4b42936bdb4306da0"
# Bump this whenever build options change so existing installations are rebuilt.
_BUILD_REVISION = "libusb-v1"
_UDEV_RULE = Path("useful_files/81-vive.rules")
_STAMP = ".axol-build-stamp"
_MACHINE_STAMP = Path("/var/lib/almond/libsurvive-build-stamp")
_INSTALLED_UDEV_RULE = Path("/etc/udev/rules.d/81-vive.rules")

_APT_BUILD_DEPS = (
    "build-essential",
    "cmake",
    "git",
    "pkg-config",
    "zlib1g-dev",
    "libusb-1.0-0-dev",
    "libhidapi-dev",
    "libudev-dev",
    "libeigen3-dev",
)


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``tracker.install`` subcommand."""
    subparsers.add_parser(
        "tracker.install",
        help="Install the pinned libsurvive runtime for Lighthouse trackers.",
    ).set_defaults(func=run)


def _src_dir() -> Path:
    if os.geteuid() == 0:
        return Path("/opt/almond/libsurvive")
    return Path.home() / ".almond" / "libsurvive"


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 1800) -> bool:
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 - missing command / timeout
        _logger.warning("command failed (%s): %s", " ".join(cmd), exc)
        return False
    if result.returncode != 0:
        _logger.warning(
            "command failed (%s): %s",
            " ".join(cmd),
            (result.stderr or result.stdout or "").strip()[-1200:],
        )
        return False
    return True


def _install_build_deps() -> bool:
    if shutil.which("apt-get") is None:
        _logger.warning(
            "apt-get not found; cannot install libsurvive build dependencies"
        )
        return False
    if not prime_sudo():
        _logger.warning(
            "libsurvive installation needs root; run as root or install: %s",
            " ".join(_APT_BUILD_DEPS),
        )
        return False
    update = run_root(["apt-get", "update"])
    install = run_root(["apt-get", "install", "-y", *_APT_BUILD_DEPS])
    return update.returncode == 0 and install.returncode == 0


def _sync_source(src: Path) -> bool:
    git = shutil.which("git")
    if git is None:
        _logger.warning("git not found; cannot fetch libsurvive")
        return False
    if not (src / ".git").exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.rmtree(src)
        if not _run([git, "clone", _REPO_URL, str(src)]):
            return False
    if not _run([git, "fetch", "--depth", "1", "origin", _PINNED_REF], cwd=src):
        if not _run([git, "fetch", "origin"], cwd=src):
            return False
    return (
        _run([git, "checkout", "--quiet", _PINNED_REF], cwd=src)
        and _run([git, "reset", "--hard", _PINNED_REF], cwd=src)
        and _run([git, "clean", "-fdq"], cwd=src)
    )


def _build_and_install(src: Path) -> bool:
    cmake = shutil.which("cmake")
    if cmake is None:
        _logger.warning("cmake not found; cannot build libsurvive")
        return False
    build = src / "build"
    build.mkdir(parents=True, exist_ok=True)
    jobs = str(min(4, os.cpu_count() or 2))
    if not _run(
        [
            cmake,
            "-S",
            str(src),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_INSTALL_PREFIX=/usr/local",
            "-DBUILD_APPLICATIONS=ON",
            # libsurvive's HIDAPI path creates a placeholder tracker object for
            # every Watchman dongle immediately. Its pairing loop only sends a
            # pairing request to dongles without such an object, making
            # `survive-cli --pair-device 1` ineffective. The native Linux
            # libusb path supports both normal tracking and dongle pairing.
            "-DUSE_HIDAPI=OFF",
            "-DDOWNLOAD_EIGEN=OFF",
            "-DUSE_EIGEN=ON",
            "-DUSE_OPENBLAS=OFF",
        ]
    ):
        return False
    # Build every enabled install target: CMake's install manifest includes the
    # small companion applications as well as survive-cli.
    if not _run([cmake, "--build", str(build), "-j", jobs]):
        return False
    if run_root([cmake, "--install", str(build)]).returncode != 0:
        return False
    ldconfig = shutil.which("ldconfig")
    if ldconfig is not None and run_root([ldconfig]).returncode != 0:
        return False
    return True


def _install_udev_rule(src: Path) -> bool:
    rule = src / _UDEV_RULE
    if not rule.exists():
        _logger.warning("libsurvive udev rule is missing: %s", rule)
        return False
    if (
        run_root(
            [
                "install",
                "-D",
                "-m",
                "0644",
                str(rule),
                str(_INSTALLED_UDEV_RULE),
            ]
        ).returncode
        != 0
    ):
        return False
    udevadm = shutil.which("udevadm")
    if udevadm is not None:
        if run_root([udevadm, "control", "--reload-rules"]).returncode != 0:
            return False
        if run_root([udevadm, "trigger"]).returncode != 0:
            return False
    return True


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_machine_stamp(src: Path) -> bool:
    """Publish one root-readable proof shared by serve and operator CLIs."""
    rule = src / _UDEV_RULE
    try:
        digest = _file_digest(rule)
        local_stamp = src / _STAMP
        local_stamp.write_text(f"{_PINNED_REF}\n{_BUILD_REVISION}\n")
        manifest = src / ".axol-machine-install-manifest"
        manifest.write_text(f"{_PINNED_REF}\n{_BUILD_REVISION}\n{digest}\n")
    except OSError as exc:
        _logger.warning("could not prepare libsurvive install manifest: %s", exc)
        return False
    result = run_root(
        ["install", "-D", "-m", "0644", str(manifest), str(_MACHINE_STAMP)]
    )
    return result.returncode == 0


def lighthouse_readiness(
    *,
    src: Path | None = None,
    installed_udev_rule: Path = _INSTALLED_UDEV_RULE,
    manifest_path: Path | None = None,
) -> dict[str, object]:
    """Inspect the supported Lighthouse runtime without opening USB devices.

    Merely finding a ``survive-cli`` executable is not enough: Axol relies on
    the pinned libusb build (the HIDAPI build cannot pair Watchman dongles) and
    on its matching udev rule.  The installer-owned stamp records both the
    upstream commit and Axol's build recipe revision.
    """
    source = src if src is not None else _src_dir()
    source_rule = source / _UDEV_RULE
    expected_stamp = f"{_PINNED_REF}\n{_BUILD_REVISION}"

    # An explicit source without a manifest is the hermetic/test form: verify
    # that source's local two-line stamp and rule. Normal callers use the
    # canonical machine manifest, so root serve, provisioning, and an operator
    # CLI all inspect the same installation regardless of their build cache.
    source_scoped = src is not None and manifest_path is None
    stamp_path = source / _STAMP if source_scoped else (manifest_path or _MACHINE_STAMP)

    try:
        actual_stamp = stamp_path.read_text().strip()
    except OSError:
        actual_stamp = ""
    actual_stamp_lines = actual_stamp.splitlines()
    stamp_valid = actual_stamp_lines[:2] == expected_stamp.splitlines()

    try:
        if source_scoped:
            rule_valid = (
                source_rule.is_file()
                and installed_udev_rule.is_file()
                and source_rule.read_bytes() == installed_udev_rule.read_bytes()
            )
        else:
            recorded_digest = (
                actual_stamp_lines[2] if len(actual_stamp_lines) > 2 else ""
            )
            rule_valid = bool(
                recorded_digest
                and installed_udev_rule.is_file()
                and _file_digest(installed_udev_rule) == recorded_digest
            )
    except OSError:
        rule_valid = False

    available = is_available()
    pairing_cli = shutil.which("survive-cli") is not None
    issues: list[str] = []
    if not available:
        issues.append("survive-cli/pysurvive is unavailable")
    if not pairing_cli:
        issues.append("survive-cli is unavailable, so Watchman pairing cannot run")
    if not stamp_valid:
        issues.append("the pinned libsurvive build stamp is missing or stale")
    if not rule_valid:
        issues.append("the pinned Vive USB udev rule is missing or stale")
    return {
        "installed": available and pairing_cli and stamp_valid and rule_valid,
        "available": available,
        "pairingCli": pairing_cli,
        "pinnedBuild": stamp_valid,
        "udevReady": rule_valid,
        "pinnedRef": _PINNED_REF,
        "buildRevision": _BUILD_REVISION,
        "installedRef": actual_stamp_lines[0] if actual_stamp_lines else None,
        "installedBuildRevision": (
            actual_stamp_lines[1] if len(actual_stamp_lines) > 1 else None
        ),
        "stampPath": str(stamp_path),
        "udevRulePath": str(installed_udev_rule),
        "issues": issues,
    }


def ensure_installed() -> bool:
    """Install libsurvive when needed; return whether a usable backend exists."""
    src = _src_dir()
    readiness = lighthouse_readiness()
    if readiness["installed"]:
        print("Lighthouse tracking support is already installed.", flush=True)
        return True

    # One-time upgrade from the former UID-scoped proof. Official root
    # provisioning used /opt while an operator CLI used ~/.almond, so neither
    # could trust the other's stamp. If this caller's old source and installed
    # rule still match exactly, publish the canonical manifest without a rebuild.
    legacy = lighthouse_readiness(src=src)
    if legacy["installed"]:
        print("Publishing the machine-wide Lighthouse install manifest…", flush=True)
        if _install_machine_stamp(src) and lighthouse_readiness()["installed"]:
            print("Lighthouse tracking support installed.", flush=True)
            return True

    # A removed/drifted permissions rule does not require a native rebuild.
    if readiness["available"] and readiness["pinnedBuild"]:
        print("Installing Vive USB permissions…", flush=True)
        if _install_udev_rule(src) and lighthouse_readiness()["installed"]:
            print("Lighthouse tracking support installed.", flush=True)
            return True

    print("Installing Lighthouse tracking build dependencies…", flush=True)
    if not _install_build_deps():
        return False
    print(f"Fetching pinned libsurvive {_PINNED_REF[:12]}…", flush=True)
    if not _sync_source(src):
        return False
    print("Building libsurvive (this can take a few minutes)…", flush=True)
    if not _build_and_install(src):
        return False
    print("Installing Vive USB permissions…", flush=True)
    if not _install_udev_rule(src):
        return False
    if not _install_machine_stamp(src):
        return False
    final = lighthouse_readiness()
    if not final["installed"]:
        _logger.warning(
            "Lighthouse installation did not pass final readiness: %s",
            "; ".join(str(issue) for issue in final["issues"]),
        )
        return False
    print("Lighthouse tracking support installed.", flush=True)
    return True


def run(_args: object = None) -> None:
    """Install libsurvive, failing clearly for CLI/UI callers on any error."""
    if not ensure_installed():
        raise SystemExit(
            "Lighthouse tracking support could not be installed; see the log above."
        )
