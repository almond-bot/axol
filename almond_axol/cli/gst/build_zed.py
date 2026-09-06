"""
axol gst.build-zed

Build and install the **patched** Stereolabs zed-gstreamer plugins
(``zedxonesrc`` / ``zedsrc``) on a Jetson.

Two reasons this exists rather than relying on a stock plugin install:

1. A fresh Jetson has no zed-gstreamer plugins at all, so the GPU-resident
   camera path (:mod:`almond_axol.video.gst_zed`) silently falls back to the
   slower ZED SDK grab. Building them here makes the fast path available.
2. The **stock** ``zedxonesrc`` / ``zedsrc`` stamp each buffer's PTS with a
   host-side software clock sampled right after ``grab()`` returns -- i.e.
   frame *receive* time, which lags the true sensor exposure by the camera
   delivery latency. Our patch (``patches/zed-gstreamer-sensor-timestamp.patch``)
   instead stamps the PTS at the true sensor-exposure instant
   (``TIME_REFERENCE::IMAGE``), so a frame's ``capture_perf_ts`` lines up with
   the joint sample on the same exposure clock as the SDK ``ZedCamera`` path.
   Without it, collected datasets pair each image with proprioception that is
   ~delivery-latency too new, and that offset differs from inference (which
   uses the SDK), i.e. a train/inference mismatch.

We pin upstream to the exact commit the patch was generated against so the
unified diff always applies cleanly. Idempotence is based on a root-owned
manifest of the exact plugin paths and bytes GStreamer resolves, not merely a
source-tree stamp. The manifest also records the ZED SDK version the plugins
were compiled against, and readiness additionally requires GStreamer to
*load* each plugin: the plugins link ``libsl_zed.so`` directly, so upgrading
the SDK in place leaves byte-identical plugins that fail to load (``undefined
symbol: sl::CameraOne::isOpened...``) while the registry still lists their
elements. Either signal (SDK version drift, or a load failure) triggers a
clean rebuild against the installed SDK. The command remains best-effort on
machines without the ZED SDK / Jetson toolchain (callers then fall back to the
SDK ``ZedCamera``). The hosted installer (``web/app/public/install``) and the
``axol serve`` self-updater run it via ``axol provision``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import stat
import subprocess
from pathlib import Path

from ...utils.jetson import _is_jetson
from ...utils.state_files import secure_atomic_write_text
from ...utils.sudo import prime_sudo, run_root

_logger = logging.getLogger(__name__)

# Upstream repo + the exact commit the vendored patch was generated against.
# Bump both together (regenerate the patch) when picking up upstream changes.
_REPO_URL = "https://github.com/stereolabs/zed-gstreamer.git"
_PINNED_REF = "4a0a3a3d896b54f9cb23f284b5b44e52b5e1a288"

_PATCH = Path(__file__).parent / "patches" / "zed-gstreamer-sensor-timestamp.patch"

# ZED SDK install (find_package(ZED) + the headers the plugins compile against).
_ZED_SDK = Path("/usr/local/zed")
# Where the SDK publishes its version macros (5.x, then the legacy 4.x layout).
_ZED_SDK_VERSION_HEADERS = (
    _ZED_SDK / "include" / "sl" / "Camera.hpp",
    _ZED_SDK / "include" / "sl_zed" / "defines.hpp",
)
_ZED_SDK_VERSION_RE = {
    part: re.compile(rf"ZED_SDK_{part}_VERSION\s+(\d+)")
    for part in ("MAJOR", "MINOR", "PATCH")
}

# This is the authority for an installed patched build.  The source-tree stamp
# is only a convenience: an operator-owned checkout cannot attest to the bytes
# GStreamer will actually load.  Keep the manifest in machine state so an
# unprivileged caller cannot bless a stock or subsequently replaced plugin.
_MACHINE_MANIFEST = Path("/var/lib/almond-axol/zed-gstreamer-manifest.json")
# Schema 2 added ``zedSdk`` (the SDK version the plugins were built against).
# A schema-1 manifest never matches the recomputed payload, so hosts upgraded
# from that release rebuild once and record their SDK version going forward.
_MANIFEST_SCHEMA = 2
_ZED_ELEMENTS = ("zedxonesrc", "zedsrc")
_GST_FILENAME_RE = re.compile(r"^\s*Filename\s+(.+?)\s*$", re.MULTILINE)

# apt build deps. OpenCV / RTSP server are optional (their plugins are skipped
# at configure time). The ZED SDK's own zed-config.cmake unconditionally calls
# ``find_package(BLAS REQUIRED)`` and links the unversioned libusb library, so
# their *development* packages are required even when the corresponding runtime
# libraries already happen to be installed. NVENC + the Jetson multimedia
# headers ship with the L4T BSP.
_APT_BUILD_DEPS = (
    "build-essential",
    "cmake",
    "git",
    "pkg-config",
    "libglib2.0-dev",
    "libgstreamer1.0-dev",
    "libgstreamer-plugins-base1.0-dev",
    "libblas-dev",
    "libusb-1.0-0-dev",
)


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``gst.build-zed`` subcommand."""
    subparsers.add_parser(
        "gst.build-zed",
        help="Build + install the patched zed-gstreamer plugins (sensor-accurate PTS).",
    ).set_defaults(func=run)


def _src_dir() -> Path:
    """Where to clone/build. Root (installer) uses /opt; a user uses ~/.almond."""
    if os.geteuid() == 0:
        return Path("/opt/almond/zed-gstreamer")
    return Path.home() / ".almond" / "zed-gstreamer"


def _desired_stamp() -> str:
    """Pinned ref + patch digest; changes whenever either is bumped."""
    patch_sha = hashlib.sha256(_PATCH.read_bytes()).hexdigest()
    return f"{_PINNED_REF}\n{patch_sha}\n"


def _zed_sdk_version() -> str | None:
    """Installed ZED SDK version (``major.minor.patch``) from its headers.

    The plugins compile against these headers and link ``libsl_zed.so``
    directly, so this is the ABI they were built for. ``None`` when the SDK
    layout is unrecognised; the manifest then records the SDK as unknown and a
    later readable version is treated as a change.
    """
    for header in _ZED_SDK_VERSION_HEADERS:
        try:
            text = header.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        parts = [_ZED_SDK_VERSION_RE[key].search(text) for key in ("MAJOR", "MINOR")]
        if not all(parts):
            continue
        version = ".".join(match.group(1) for match in parts if match)
        patch = _ZED_SDK_VERSION_RE["PATCH"].search(text)
        if patch:
            version += f".{patch.group(1)}"
        return version
    return None


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 1800) -> bool:
    """Run a command, logging on failure; returns True on exit code 0."""
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 - command missing / timed out
        _logger.warning("command failed (%s): %s", " ".join(cmd), exc)
        return False
    if result.returncode != 0:
        _logger.warning(
            "command failed (%s): %s",
            " ".join(cmd),
            (result.stderr or result.stdout or "").strip()[-800:],
        )
        return False
    return True


def _element_installed(name: str) -> bool:
    """True when gst-inspect can find ``name`` (i.e. the plugin is installed)."""
    inspect = shutil.which("gst-inspect-1.0")
    if inspect is None:
        return False
    return _run([inspect, name], timeout=60)


def _root_controlled_canonical_file(
    path: Path, *, allow_canonical_alias: bool
) -> Path | None:
    """Return a canonical immutable root-owned file, or fail closed.

    ``gst-inspect`` is external input here. Resolving and checking the entire
    ancestry prevents a root invocation from hashing an attacker-controlled
    pathname. Plugin filenames may legitimately contain system symlinks, but
    the manifest itself may not: its fixed machine-state name must be the real
    file rather than an alias.
    """
    if not path.is_absolute():
        return None
    try:
        canonical = path.resolve(strict=True)
        if not allow_canonical_alias and canonical != path:
            return None
        file_stat = canonical.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_uid != 0
            or stat.S_IMODE(file_stat.st_mode) & 0o022
        ):
            return None
        for parent in canonical.parents:
            parent_stat = parent.stat(follow_symlinks=False)
            if (
                not stat.S_ISDIR(parent_stat.st_mode)
                or parent_stat.st_uid != 0
                or stat.S_IMODE(parent_stat.st_mode) & 0o022
            ):
                return None
    except (OSError, RuntimeError):
        return None
    return canonical


def _gst_inspect(target: str) -> subprocess.CompletedProcess[str] | None:
    """Run ``gst-inspect-1.0 <target>`` (an element name or a plugin path)."""
    inspect = shutil.which("gst-inspect-1.0")
    if inspect is None:
        return None
    try:
        return subprocess.run(
            [inspect, target],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "LC_ALL": "C"},
        )
    except Exception as exc:  # noqa: BLE001 - command missing / timed out
        _logger.warning("could not run gst-inspect on %s: %s", target, exc)
        return None


def _plugin_loads(artifact: Path) -> bool:
    """True when GStreamer can actually load the plugin shared object.

    The registry cache keeps listing an element for as long as its plugin file
    is unchanged, and ``gst-inspect-1.0 <element>`` answers from that cache. So
    a plugin whose *dependencies* changed underneath it (the ZED SDK upgraded
    in place, leaving ``undefined symbol`` relocations against the new
    ``libsl_zed.so``) still looks installed until a pipeline tries to
    instantiate it. Inspecting the file itself forces ``dlopen`` and fails
    with a non-zero exit on every GStreamer release.
    """
    result = _gst_inspect(str(artifact))
    if result is None:
        return False
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        _logger.warning(
            "GStreamer cannot load the ZED plugin %s (it is stale relative to the "
            "installed ZED SDK or its libraries): %s",
            artifact,
            detail[-600:] or f"exit {result.returncode}",
        )
        return False
    return True


def _inspect_element_artifact(name: str) -> Path | None:
    """Resolve the root-controlled, *loadable* shared object behind an element."""
    result = _gst_inspect(name)
    if result is None or result.returncode != 0:
        return None
    match = _GST_FILENAME_RE.search(result.stdout)
    if match is None:
        _logger.warning("gst-inspect did not report a plugin filename for %s", name)
        return None
    artifact = _root_controlled_canonical_file(
        Path(match.group(1)), allow_canonical_alias=True
    )
    if artifact is None:
        _logger.warning("%s resolved to an unsafe or unknown plugin artifact", name)
        return None
    if not _plugin_loads(artifact):
        return None
    return artifact


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_plugin_artifacts() -> dict[str, dict[str, str]] | None:
    """Resolve and hash both ZED elements exactly as GStreamer sees them."""
    artifacts: dict[str, dict[str, str]] = {}
    for element in _ZED_ELEMENTS:
        path = _inspect_element_artifact(element)
        if path is None:
            return None
        try:
            digest = _file_sha256(path)
        except OSError as exc:
            _logger.warning(
                "could not hash %s plugin artifact %s: %s", element, path, exc
            )
            return None
        artifacts[element] = {"path": str(path), "sha256": digest}
    return artifacts


def _manifest_payload(artifacts: dict[str, dict[str, str]]) -> str:
    return (
        json.dumps(
            {
                "schema": _MANIFEST_SCHEMA,
                "pinnedRef": _PINNED_REF,
                "patchSha256": hashlib.sha256(_PATCH.read_bytes()).hexdigest(),
                "zedSdk": _zed_sdk_version(),
                "plugins": artifacts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _installed_plugins_ready(manifest_path: Path = _MACHINE_MANIFEST) -> bool:
    """Recompute installed paths/digests and compare with the root manifest.

    Ready means: the root manifest exists, every ZED element resolves to a
    root-controlled plugin that GStreamer can load, the plugin bytes are the
    ones this build published, and they were built against the ZED SDK that is
    installed now. Anything else is logged and triggers a rebuild.
    """
    safe_manifest = _root_controlled_canonical_file(
        manifest_path, allow_canonical_alias=False
    )
    if safe_manifest is None:
        return False
    try:
        saved = json.loads(safe_manifest.read_text(encoding="utf-8"))
        current = _collect_plugin_artifacts()
        if current is None:
            return False
        expected = json.loads(_manifest_payload(current))
    except (OSError, ValueError, TypeError):
        return False
    if saved == expected:
        return True
    if isinstance(saved, dict):
        built_against = saved.get("zedSdk")
        installed = expected["zedSdk"]
        if built_against != installed:
            _logger.info(
                "zed-gstreamer plugins were built against ZED SDK %s but SDK %s "
                "is installed; rebuilding them against the installed SDK",
                built_against or "unknown",
                installed or "unknown",
            )
    return False


def _installed_paths_from_build(src: Path) -> set[Path] | None:
    """Canonical paths written by the just-completed CMake install."""
    install_manifest = src / "build" / "install_manifest.txt"
    try:
        lines = install_manifest.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    installed: set[Path] = set()
    for line in lines:
        if not line:
            continue
        path = Path(line)
        if not path.is_absolute():
            return None
        try:
            installed.add(path.resolve(strict=True))
        except (OSError, RuntimeError):
            return None
    return installed


def _artifacts_came_from_build(src: Path, artifacts: dict[str, dict[str, str]]) -> bool:
    installed = _installed_paths_from_build(src)
    if installed is None:
        return False
    return all(Path(details["path"]) in installed for details in artifacts.values())


def _publish_machine_manifest(payload: str) -> bool:
    """Atomically publish the manifest from a root-private machine directory."""
    parent = _MACHINE_MANIFEST.parent
    temporary = parent / (f".{_MACHINE_MANIFEST.name}.{secrets.token_hex(16)}.stage")
    if run_root(["mkdir", "-p", "-m", "0755", str(parent)]).returncode != 0:
        return False
    try:
        canonical_parent = parent.resolve(strict=True)
        parent_stat = canonical_parent.stat(follow_symlinks=False)
        if (
            canonical_parent != parent
            or parent_stat.st_uid != 0
            or not stat.S_ISDIR(parent_stat.st_mode)
            or stat.S_IMODE(parent_stat.st_mode) & 0o022
        ):
            _logger.warning("refusing unsafe ZED manifest directory %s", parent)
            return False
    except (OSError, RuntimeError):
        return False
    try:
        if run_root(["tee", str(temporary)], input_text=payload).returncode != 0:
            return False
        if run_root(["chmod", "0644", str(temporary)]).returncode != 0:
            return False
        if run_root(["chown", "root:root", str(temporary)]).returncode != 0:
            return False
        return (
            run_root(
                ["mv", "-f", "-T", str(temporary), str(_MACHINE_MANIFEST)]
            ).returncode
            == 0
        )
    finally:
        # The staging name is random and lives below the fixed machine-state
        # directory; after a successful rename this is simply a no-op.
        run_root(["rm", "-f", "--", str(temporary)])


def _apt_install_build_deps() -> bool:
    """Install Debian build prerequisites, returning whether setup may proceed.

    On non-Debian hosts there is no package manager to drive, so let CMake
    inspect whatever the operator installed manually. When apt is available,
    however, don't continue into an opaque configure failure after escalation
    or package installation failed.
    """
    if shutil.which("apt-get") is None:
        _logger.info("apt-get not found; assuming build deps are present")
        return True
    if not prime_sudo():
        _logger.warning(
            "zed-gstreamer build deps need root; run as root or: "
            "sudo apt-get install -y %s",
            " ".join(_APT_BUILD_DEPS),
        )
        return False
    # An update failure need not block an install from an already-populated apt
    # cache. The install result itself is authoritative.
    update = run_root(["apt-get", "update"])
    if update.returncode != 0:
        _logger.warning("apt-get update failed; trying the existing package cache")
    installed = run_root(["apt-get", "install", "-y", *_APT_BUILD_DEPS])
    if installed.returncode != 0:
        _logger.warning(
            "could not install zed-gstreamer build dependencies; run: "
            "sudo apt-get install -y %s",
            " ".join(_APT_BUILD_DEPS),
        )
        return False
    return True


def _sync_source(src: Path) -> bool:
    """Clone (or update) the repo and hard-reset to the pinned ref, clean tree."""
    git = shutil.which("git")
    if git is None:
        _logger.warning("git not found; cannot fetch zed-gstreamer source")
        return False

    if not (src / ".git").exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.rmtree(src)
        if not _run([git, "clone", _REPO_URL, str(src)]):
            return False

    # Pin to the exact ref and discard any prior patched tree so the unified
    # diff always applies against pristine upstream.
    ok = _run([git, "fetch", "--depth", "1", "origin", _PINNED_REF], cwd=src)
    if not ok:
        # Shallow clones may not have the ref; deepen via a full fetch.
        _run([git, "fetch", "origin"], cwd=src)
    return (
        _run([git, "checkout", "--quiet", _PINNED_REF], cwd=src)
        and _run([git, "reset", "--hard", _PINNED_REF], cwd=src)
        and _run([git, "clean", "-fdq"], cwd=src)
    )


def _apply_patch(src: Path) -> bool:
    git = shutil.which("git")
    if git is None or not _PATCH.exists():
        _logger.warning("cannot apply patch (git=%s, patch=%s)", git, _PATCH)
        return False
    return _run([git, "apply", str(_PATCH)], cwd=src)


def _build_and_install(src: Path) -> bool:
    cmake = shutil.which("cmake")
    if cmake is None:
        _logger.warning("cmake not found; cannot build zed-gstreamer")
        return False
    build = src / "build"
    # Upstream's .gitignore covers ``*build*``, so ``git clean`` in _sync_source
    # leaves a previous build tree behind. Start from scratch: after a ZED SDK
    # upgrade the cached configure results and objects describe the old SDK,
    # and the whole point of the rebuild is to link against the new one.
    shutil.rmtree(build, ignore_errors=True)
    build.mkdir(parents=True, exist_ok=True)
    configured = _run(
        [cmake, "-DCMAKE_BUILD_TYPE=Release", "-S", str(src), "-B", str(build)]
    )
    if not configured:
        return False
    # Upstream explicitly documents that this project does not support a
    # parallel build. Keep one job even on large Jetsons to avoid intermittent
    # generated-target races.
    if not _run([cmake, "--build", str(build), "-j", "1"]):
        return False
    # Install writes into the system GStreamer plugin dir (root-owned).
    return run_root([cmake, "--install", str(build)]).returncode == 0


def run(_args: object = None) -> None:
    """Build + install the patched zed-gstreamer plugins (idempotent)."""
    if not _is_jetson():
        print(
            "Not an NVIDIA Jetson (L4T); skipping the Jetson-only "
            "zed-gstreamer GPU path. Camera capture will use the ZED SDK "
            "fallback on this host."
        )
        return
    if not _ZED_SDK.exists():
        print(
            "No ZED SDK at /usr/local/zed; skipping zed-gstreamer build "
            "(the camera path will use the ZED SDK fallback). Install the SDK "
            "and re-run 'axol gst.build-zed'."
        )
        return

    src = _src_dir()
    stamp_file = src / ".axol-build-stamp"
    desired = _desired_stamp()

    sdk_version = _zed_sdk_version()
    if _installed_plugins_ready():
        print(
            "Patched zed-gstreamer plugins already installed (pinned ref + patch, "
            f"built against ZED SDK {sdk_version or 'unknown'})."
        )
        return
    print(
        "Building the patched zed-gstreamer plugins against ZED SDK "
        f"{sdk_version or 'unknown'}..."
    )

    print("Installing zed-gstreamer build dependencies (apt)...")
    if not _apt_install_build_deps():
        raise SystemExit(
            "Could not install zed-gstreamer build dependencies. "
            "Install the packages listed above and re-run "
            "'axol gst.build-zed'."
        )

    print(f"Fetching zed-gstreamer @ {_PINNED_REF[:12]} into {src}...")
    if not _sync_source(src):
        raise SystemExit("Could not fetch zed-gstreamer source; retry the build.")

    print("Applying the sensor-exposure-timestamp patch...")
    if not _apply_patch(src):
        raise SystemExit(
            "The zed-gstreamer timestamp patch did not apply; check upstream drift."
        )

    print("Building + installing the patched plugins (this can take a few minutes)...")
    if not _build_and_install(src):
        raise SystemExit(
            "zed-gstreamer build/install failed; see the log above and re-run "
            "'axol gst.build-zed'."
        )

    artifacts = _collect_plugin_artifacts()
    if artifacts is None:
        raise SystemExit(
            "zedxonesrc and zedsrc must both be visible to gst-inspect and "
            "loadable after install; check the GStreamer plugin path and the "
            "log above."
        )
    if not _artifacts_came_from_build(src, artifacts):
        raise SystemExit(
            "GStreamer resolved a ZED element to an unknown plugin artifact "
            "instead of one installed by this patched build."
        )
    if not _publish_machine_manifest(_manifest_payload(artifacts)):
        raise SystemExit("Could not publish the root-owned ZED plugin manifest.")
    if not _installed_plugins_ready():
        raise SystemExit(
            "Installed ZED plugin bytes did not match the published manifest."
        )

    # Keep the legacy source stamp for operator visibility, but only after the
    # root-owned installed-byte proof has been written and verified.
    try:
        secure_atomic_write_text(stamp_file, desired, mode=0o644)
    except OSError:
        pass
    print("Patched zed-gstreamer plugins installed (sensor-accurate timestamps).")
