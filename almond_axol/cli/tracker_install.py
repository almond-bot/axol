"""Install the pinned libsurvive Lighthouse tracking runtime.

``survive-cli`` is a native application rather than a Python package, so it
cannot be carried by the normal ``uv tool`` install.  This command is shared by
``axol provision`` and the control panel's Mantis setup UI.  It is intentionally
idempotent: an already-installed pinned build is a fast no-op.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import stat
import subprocess
from pathlib import Path

from ..utils.state_files import secure_atomic_write_json, secure_atomic_write_text
from ..utils.sudo import prime_sudo, run_root

_logger = logging.getLogger(__name__)

_REPO_URL = "https://github.com/collabora/libsurvive.git"
_PINNED_REF = "f1e6eddb669320f2a30760f4b42936bdb4306da0"
# Bump this whenever build options change so existing installations are rebuilt.
_BUILD_REVISION = "libusb-v2-runtime-attestation"
_UDEV_RULE = Path("useful_files/81-vive.rules")
_STAMP = ".axol-build-stamp"
_MACHINE_STAMP = Path("/var/lib/almond/libsurvive-build-stamp")
_INSTALLED_UDEV_RULE = Path("/etc/udev/rules.d/81-vive.rules")
_INSTALL_PREFIX = Path("/usr/local")
_SURVIVE_CLI = _INSTALL_PREFIX / "bin" / "survive-cli"
_INSTALL_MANIFEST = Path("build/install_manifest.txt")
_MANIFEST_SCHEMA = 1

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


def _runtime_install_paths(src: Path) -> tuple[Path, ...]:
    """Return the executable and shared objects produced by this CMake build."""
    try:
        installed = (src / _INSTALL_MANIFEST).read_text().splitlines()
    except OSError as exc:
        raise RuntimeError("libsurvive's CMake install manifest is missing") from exc

    runtime: set[Path] = set()
    library_root = _INSTALL_PREFIX / "lib"
    for raw_path in installed:
        path = Path(raw_path.strip())
        if path == _SURVIVE_CLI:
            runtime.add(path)
            continue
        try:
            relative = path.relative_to(library_root)
        except ValueError:
            continue
        # libsurvive discovers its driver/poser plugins dynamically. Capture
        # every shared object installed by the same build, including SONAME
        # symlinks and cnkalman/mpfit dependencies linked into libsurvive.
        if relative.name.endswith(".so") or ".so." in relative.name:
            runtime.add(path)
    if _SURVIVE_CLI not in runtime:
        raise RuntimeError("the build did not install /usr/local/bin/survive-cli")
    return tuple(sorted(runtime, key=str))


def _safe_root_ancestry(path: Path, *, root: Path = _INSTALL_PREFIX) -> bool:
    """Whether every directory from ``root`` through ``path`` is root-sealed."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    current = root
    for part in ("", *relative.parts):
        if part:
            current /= part
        try:
            metadata = current.lstat()
        except OSError:
            return False
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_gid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            return False
    return True


def _safe_root_regular_file(path: Path) -> bool:
    """Whether a file and its absolute directory chain are root-controlled."""
    try:
        metadata = path.lstat()
    except OSError:
        return False
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != 0
        or metadata.st_gid != 0
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        return False
    for parent in path.parents:
        try:
            parent_metadata = parent.lstat()
        except OSError:
            return False
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or stat.S_ISLNK(parent_metadata.st_mode)
            or parent_metadata.st_uid != 0
            or parent_metadata.st_gid != 0
            or stat.S_IMODE(parent_metadata.st_mode) & 0o022
        ):
            return False
    return True


def _runtime_artifact_record(
    path: Path, *, require_root_control: bool
) -> dict[str, str] | None:
    """Describe one installed runtime artifact after validating its ownership."""
    if not path.is_absolute():
        return None
    try:
        logical_metadata = path.lstat()
        link_target = (
            os.readlink(path) if stat.S_ISLNK(logical_metadata.st_mode) else ""
        )
        resolved = path.resolve(strict=True)
        resolved_metadata = resolved.stat()
    except OSError:
        return None
    if not stat.S_ISREG(resolved_metadata.st_mode):
        return None
    if path == _SURVIVE_CLI and (
        stat.S_ISLNK(logical_metadata.st_mode)
        or not stat.S_IMODE(resolved_metadata.st_mode) & 0o111
    ):
        return None
    if require_root_control:
        if (
            logical_metadata.st_uid != 0
            or logical_metadata.st_gid != 0
            or resolved_metadata.st_uid != 0
            or resolved_metadata.st_gid != 0
            or stat.S_IMODE(resolved_metadata.st_mode) & 0o022
            or not _safe_root_ancestry(path.parent)
            or not _safe_root_ancestry(resolved.parent)
        ):
            return None
    try:
        digest = _file_digest(resolved)
    except OSError:
        return None
    return {
        "path": str(path),
        "resolvedPath": str(resolved),
        "linkTarget": link_target,
        "sha256": digest,
    }


def _runtime_artifact_records(
    paths: tuple[Path, ...], *, require_root_control: bool
) -> list[dict[str, str]] | None:
    records: list[dict[str, str]] = []
    for path in paths:
        record = _runtime_artifact_record(
            path, require_root_control=require_root_control
        )
        if record is None:
            return None
        records.append(record)
    return records


def _install_machine_stamp(src: Path) -> bool:
    """Publish a root-readable proof of the exact native runtime artifacts."""
    rule = src / _UDEV_RULE
    try:
        digest = _file_digest(rule)
        runtime_paths = _runtime_install_paths(src)
        artifacts = _runtime_artifact_records(runtime_paths, require_root_control=True)
        if artifacts is None:
            raise RuntimeError(
                "installed libsurvive artifacts are not root-owned and sealed"
            )
        local_stamp = src / _STAMP
        secure_atomic_write_text(
            local_stamp,
            f"{_PINNED_REF}\n{_BUILD_REVISION}\n",
            mode=0o644,
        )
        manifest = src / ".axol-machine-install-manifest"
        secure_atomic_write_json(
            manifest,
            {
                "schema": _MANIFEST_SCHEMA,
                "pinnedRef": _PINNED_REF,
                "buildRevision": _BUILD_REVISION,
                "udevRuleSha256": digest,
                "surviveCliPath": str(_SURVIVE_CLI),
                "runtimeArtifacts": artifacts,
            },
            mode=0o644,
        )
    except (OSError, RuntimeError) as exc:
        _logger.warning("could not prepare libsurvive install manifest: %s", exc)
        return False
    result = run_root(
        ["install", "-D", "-m", "0644", str(manifest), str(_MACHINE_STAMP)]
    )
    return result.returncode == 0


def _verified_runtime_artifacts(
    manifest: object, *, require_root_control: bool
) -> tuple[bool, Path | None]:
    if not isinstance(manifest, dict):
        return False, None
    raw_records = manifest.get("runtimeArtifacts")
    cli_value = manifest.get("surviveCliPath")
    if not isinstance(raw_records, list) or not raw_records:
        return False, None
    if not isinstance(cli_value, str):
        return False, None
    cli_path = Path(cli_value)
    if not cli_path.is_absolute():
        return False, None
    if require_root_control and cli_path != _SURVIVE_CLI:
        return False, None

    recorded: list[dict[str, str]] = []
    paths: list[Path] = []
    seen: set[str] = set()
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            return False, None
        if set(raw_record) != {"path", "resolvedPath", "linkTarget", "sha256"}:
            return False, None
        if not all(isinstance(value, str) for value in raw_record.values()):
            return False, None
        path_value = raw_record["path"]
        if path_value in seen:
            return False, None
        seen.add(path_value)
        path = Path(path_value)
        if not path.is_absolute():
            return False, None
        recorded.append(raw_record)
        paths.append(path)
    if str(cli_path) not in seen:
        return False, None
    actual = _runtime_artifact_records(
        tuple(paths), require_root_control=require_root_control
    )
    return actual == recorded, cli_path if actual == recorded else None


def lighthouse_readiness(
    *,
    src: Path | None = None,
    installed_udev_rule: Path = _INSTALLED_UDEV_RULE,
    manifest_path: Path | None = None,
) -> dict[str, object]:
    """Inspect and hash-verify the supported Lighthouse native runtime.

    ``PATH`` and importable Python modules are deliberately irrelevant. The
    root installer records the exact executable, shared library, and plugin
    files produced by the pinned build; every launch revalidates those files.
    ``src`` remains accepted for API compatibility but cannot substitute a
    per-user build-cache stamp for the machine-wide artifact proof.
    """
    del src
    stamp_path = manifest_path or _MACHINE_STAMP
    require_root_control = manifest_path is None
    manifest_trusted = not require_root_control or _safe_root_regular_file(stamp_path)
    try:
        manifest: object = (
            json.loads(stamp_path.read_text()) if manifest_trusted else None
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        manifest = None

    stamp_valid = bool(
        isinstance(manifest, dict)
        and manifest.get("schema") == _MANIFEST_SCHEMA
        and manifest.get("pinnedRef") == _PINNED_REF
        and manifest.get("buildRevision") == _BUILD_REVISION
    )
    runtime_valid, cli_path = _verified_runtime_artifacts(
        manifest, require_root_control=require_root_control
    )
    recorded_digest = (
        manifest.get("udevRuleSha256") if isinstance(manifest, dict) else None
    )
    try:
        rule_valid = bool(
            isinstance(recorded_digest, str)
            and len(recorded_digest) == 64
            and installed_udev_rule.is_file()
            and (
                not require_root_control or _safe_root_regular_file(installed_udev_rule)
            )
            and _file_digest(installed_udev_rule) == recorded_digest
        )
    except OSError:
        rule_valid = False

    available = stamp_valid and runtime_valid
    pairing_cli = cli_path is not None
    issues: list[str] = []
    if not stamp_valid:
        issues.append("the pinned libsurvive build manifest is missing or stale")
    if not runtime_valid:
        issues.append("the pinned libsurvive runtime artifacts are missing or changed")
    if not pairing_cli:
        issues.append("the attested survive-cli is unavailable for tracking or pairing")
    if not rule_valid:
        issues.append("the pinned Vive USB udev rule is missing or stale")
    installed = available and pairing_cli and rule_valid
    return {
        "installed": installed,
        "available": available,
        "pairingCli": pairing_cli,
        "pinnedBuild": stamp_valid,
        "runtimeArtifacts": runtime_valid,
        "surviveCliPath": str(cli_path) if cli_path is not None else None,
        "udevReady": rule_valid,
        "pinnedRef": _PINNED_REF,
        "buildRevision": _BUILD_REVISION,
        "installedRef": (
            manifest.get("pinnedRef") if isinstance(manifest, dict) else None
        ),
        "installedBuildRevision": (
            manifest.get("buildRevision") if isinstance(manifest, dict) else None
        ),
        "stampPath": str(stamp_path),
        "udevRulePath": str(installed_udev_rule),
        "issues": issues,
    }


def verified_survive_cli() -> Path | None:
    """Return the exact attested tracking executable, never a PATH candidate."""
    readiness = lighthouse_readiness()
    value = readiness.get("surviveCliPath")
    return Path(value) if readiness["installed"] and isinstance(value, str) else None


def ensure_installed() -> bool:
    """Install libsurvive when needed; return whether a usable backend exists."""
    src = _src_dir()
    readiness = lighthouse_readiness()
    if readiness["installed"]:
        print("Lighthouse tracking support is already installed.", flush=True)
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
