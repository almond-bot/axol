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
unified diff always applies cleanly. Idempotent (a stamp file skips a rebuild
when the pinned ref + patch are already installed) and best-effort: a no-op on
machines without the ZED SDK / Jetson toolchain (callers then fall back to the
SDK ``ZedCamera``). The hosted installer (``web/app/public/install``) runs it
once after ``axol gst.install``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
from pathlib import Path

from ...utils.sudo import prime_sudo, run_root

_logger = logging.getLogger(__name__)

# Upstream repo + the exact commit the vendored patch was generated against.
# Bump both together (regenerate the patch) when picking up upstream changes.
_REPO_URL = "https://github.com/stereolabs/zed-gstreamer.git"
_PINNED_REF = "4a0a3a3d896b54f9cb23f284b5b44e52b5e1a288"

_PATCH = Path(__file__).parent / "patches" / "zed-gstreamer-sensor-timestamp.patch"

# ZED SDK install (find_package(ZED) + the headers the plugins compile against).
_ZED_SDK = Path("/usr/local/zed")

# apt build deps. OpenCV / RTSP server are optional (their plugins are skipped
# at configure time); GStreamer + GLib dev packages are what the zedxonesrc /
# zedsrc targets actually need. NVENC + the Jetson multimedia headers ship with
# the L4T BSP.
_APT_BUILD_DEPS = (
    "build-essential",
    "cmake",
    "git",
    "pkg-config",
    "libglib2.0-dev",
    "libgstreamer1.0-dev",
    "libgstreamer-plugins-base1.0-dev",
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


def _apt_install_build_deps() -> None:
    if shutil.which("apt-get") is None:
        _logger.info("apt-get not found; assuming build deps are present")
        return
    if not prime_sudo():
        _logger.warning(
            "zed-gstreamer build deps need root; run as root or: "
            "sudo apt-get install -y %s",
            " ".join(_APT_BUILD_DEPS),
        )
        return
    run_root(["apt-get", "update"])
    run_root(["apt-get", "install", "-y", *_APT_BUILD_DEPS])


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
    build.mkdir(parents=True, exist_ok=True)
    jobs = str(min(4, (os.cpu_count() or 2)))
    configured = _run(
        [cmake, "-DCMAKE_BUILD_TYPE=Release", "-S", str(src), "-B", str(build)]
    )
    if not configured:
        return False
    if not _run([cmake, "--build", str(build), "-j", jobs]):
        return False
    # Install writes into the system GStreamer plugin dir (root-owned).
    return run_root([cmake, "--install", str(build)]).returncode == 0


def run(_args: object = None) -> None:
    """Build + install the patched zed-gstreamer plugins (idempotent)."""
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

    already = (
        stamp_file.exists()
        and stamp_file.read_text() == desired
        and _element_installed("zedxonesrc")
    )
    if already:
        print("Patched zed-gstreamer plugins already installed (pinned ref + patch).")
        return

    print("Installing zed-gstreamer build dependencies (apt)...")
    _apt_install_build_deps()

    print(f"Fetching zed-gstreamer @ {_PINNED_REF[:12]} into {src}...")
    if not _sync_source(src):
        print("WARNING: could not fetch zed-gstreamer source; skipping build.")
        return

    print("Applying the sensor-exposure-timestamp patch...")
    if not _apply_patch(src):
        print("WARNING: patch did not apply (upstream drift?); skipping build.")
        return

    print("Building + installing the patched plugins (this can take a few minutes)...")
    if not _build_and_install(src):
        print(
            "WARNING: zed-gstreamer build/install failed. The GPU camera path "
            "may be unavailable or use host-receive timestamps; see the log "
            "above and re-run 'axol gst.build-zed'."
        )
        return

    if _element_installed("zedxonesrc"):
        try:
            stamp_file.write_text(desired)
        except OSError:
            pass
        print("Patched zed-gstreamer plugins installed (sensor-accurate timestamps).")
    else:
        print(
            "WARNING: zedxonesrc is still not visible to gst-inspect after "
            "install; check the GStreamer plugin path."
        )
