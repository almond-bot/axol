"""
axol gst.install

Install the host-side GStreamer support the unified ZED camera pipeline needs
(``teleop --cameras`` / ``collect-data``): the GStreamer introspection typelibs
and **PyGObject**, so the in-process pipeline (:mod:`almond_axol.video.gst_zed`)
can pull encoded + raw frames from an ``appsink``. The NVENC encoder
(``nvv4l2h264enc`` / ``nvvidconv``) ships with the Jetson L4T BSP, and the
patched zed-gstreamer source elements (``zedxonesrc`` / ``zedsrc``) are built
separately by ``axol gst.build-zed`` — this command only verifies those are
present.

PyGObject is not a usable PyPI wheel — it builds against the system
gobject-introspection and loads the system typelibs — so it can't be a normal
dependency. This command apt-installs the system packages and then builds
PyGObject into the interpreter running the CLI (the uv tool environment),
mirroring ``axol zed.install``. PyGObject is only used here to read GStreamer
``appsink`` buffers (frames + PTS); WebRTC transport stays on aiortc, so this
does not reintroduce the ``webrtcbin``/libnice ICE path.

The hosted installer (``web/app/public/install``) runs it once; it is not run
at teleop/serve startup. Best-effort: a no-op on machines without
``apt-get``/NVENC (callers then fall back to the SDK ``ZedCamera``).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys

from ...utils.sudo import prime_sudo

_logger = logging.getLogger(__name__)

# PyGObject 3.52 dropped the legacy girepository-1.0 ABI for girepository-2.0
# (GLib >= 2.80, i.e. Ubuntu 24.04+). The Jetson runs jammy (GLib 2.72), whose
# gobject-introspection is 1.x, so an unpinned install picks a version that
# cannot build here. Pin to the last 3.5x that targets girepository-1.0.
_PYGOBJECT_SPEC = "pygobject>=3.50,<3.52"

# GStreamer elements gst.install is itself responsible for verifying: the NVENC
# encode path (nvvidconv / nvv4l2h264enc), which ships with the Jetson L4T BSP.
# These gate gst.install's success alongside the gi/appsink import.
_REQUIRED_ELEMENTS = ("nvvidconv", "nvv4l2h264enc")

# The patched ZED source element ``axol gst.build-zed`` builds + installs
# (sensor-accurate PTS). gst.install only *reports* whether it's present — its
# absence is expected before build-zed runs, so it's a note, not a failure
# (build-zed, which the installer/provision runs right after, installs it).
_ZED_SOURCE_MARKER = "zedxonesrc"

# System packages: GStreamer introspection typelibs (Gst + GstApp/appsink) and
# the build deps PyGObject needs to compile against gobject-introspection.
_APT_PACKAGES = (
    "gstreamer1.0-tools",
    "gstreamer1.0-plugins-bad",
    "gir1.2-gstreamer-1.0",
    "gir1.2-gst-plugins-base-1.0",
    "libgirepository1.0-dev",
    "gobject-introspection",
    "libcairo2-dev",
    "pkg-config",
)


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``gst.install`` subcommand."""
    subparsers.add_parser(
        "gst.install",
        help="Install GStreamer + PyGObject support for the ZED camera pipeline.",
    ).set_defaults(func=run)


def _run(cmd: list[str]) -> bool:
    """Run a command, logging on failure; returns True on exit code 0."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except Exception as exc:  # noqa: BLE001 - command missing / timed out
        _logger.warning("command failed (%s): %s", " ".join(cmd), exc)
        return False
    if result.returncode != 0:
        _logger.warning(
            "command failed (%s): %s",
            " ".join(cmd),
            (result.stderr or result.stdout or "").strip()[:500],
        )
        return False
    return True


def _gi_elements_ok(elements: tuple[str, ...]) -> bool:
    """True when a clean subprocess can import gi and find every element given.

    Checked in a subprocess so the result reflects a PyGObject that was just
    installed (importing it in this process would hit a stale import state).
    """
    from ...video.gst_zed import _set_typelib_path

    _set_typelib_path()  # ensure the child inherits the typelib search path
    els = ",".join(repr(e) for e in elements)
    code = (
        "import gi;"
        "gi.require_version('Gst','1.0');"
        "gi.require_version('GstApp','1.0');"
        "from gi.repository import Gst, GstApp;"  # GstApp registers AppSink
        "Gst.init(None);"
        "import sys;"
        f"els=({els},);"
        "sys.exit(0 if all(Gst.ElementFactory.find(e) for e in els) else 1)"
    )
    try:
        return (
            subprocess.run(
                [sys.executable, "-c", code], capture_output=True, timeout=60
            ).returncode
            == 0
        )
    except Exception:  # noqa: BLE001 - interpreter/import failure
        return False


def _gst_ok() -> bool:
    """True when gst.install's own deliverables work: PyGObject can import gi +
    GstApp (the appsink reader) and the Jetson NVENC elements are present.

    Deliberately excludes the zedxonesrc / zedsrc source elements: those are
    ``axol gst.build-zed``'s responsibility and aren't built until the step
    *after* this one, so gating on them here would misreport a clean install as
    failed. Their presence is reported separately by :func:`_note_zed_sources`.
    """
    return _gi_elements_ok(_REQUIRED_ELEMENTS)


def _note_zed_sources() -> None:
    """Report (note only) whether the patched ZED source elements are present.

    ``axol gst.build-zed`` builds + installs zedxonesrc / zedsrc, and the
    installer/provision runs it right after gst.install, so their absence here
    is expected on a first install — informational, never a warning.
    """
    if _gi_elements_ok((_ZED_SOURCE_MARKER,)):
        print("Patched ZED source elements (zedxonesrc / zedsrc) present.")
    else:
        print(
            "Note: the ZED source elements (zedxonesrc / zedsrc) aren't built "
            "yet — 'axol gst.build-zed' builds + installs them (the installer "
            "runs it right after this step)."
        )


def _apt_install() -> bool:
    apt = shutil.which("apt-get")
    if apt is None:
        _logger.info("apt-get not found; skipping system GStreamer install")
        return False
    if os.geteuid() == 0:
        prefix: list[str] = []
    else:
        # Prime sudo once (a tty prompt when run interactively); the hosted
        # installer runs this as root, so escalation is a no-op there.
        prime_sudo()
        if subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode != 0:
            _logger.warning(
                "GStreamer system packages need root; run as root or install "
                "manually: sudo apt-get install -y %s",
                " ".join(_APT_PACKAGES),
            )
            return False
        prefix = ["sudo", "-n"]
    _run([*prefix, apt, "update"])
    return _run([*prefix, apt, "install", "-y", *_APT_PACKAGES])


def _pip_install_pygobject() -> bool:
    # Target the interpreter running this CLI (the uv tool env), not whatever
    # VIRTUAL_ENV/cwd resolves to — same trick as ``axol zed.install``.
    uv = shutil.which("uv")
    pip_args = ["pip", "install", "--python", sys.executable, _PYGOBJECT_SPEC]
    if uv is not None and _run([uv, *pip_args]):
        return True
    return _run([sys.executable, "-m", "pip", "install", _PYGOBJECT_SPEC])


def run(_args: object = None) -> None:
    """Install the GStreamer framework + introspection typelibs and PyGObject.

    NVENC (``nvv4l2h264enc`` / ``nvvidconv``) ships with the Jetson L4T BSP and
    the ``zedxonesrc`` / ``zedsrc`` source elements come from
    ``axol gst.build-zed``; this command only verifies those are present.
    """
    if _gst_ok():
        print("GStreamer appsink + NVENC stack already available.")
        _note_zed_sources()
        return

    print("Installing GStreamer system packages (apt)...")
    _apt_install()
    print(f"Installing PyGObject ({_PYGOBJECT_SPEC}) into the axol environment...")
    _pip_install_pygobject()

    if _gst_ok():
        print("GStreamer appsink + NVENC stack installed.")
        _note_zed_sources()
    else:
        print(
            "WARNING: the GStreamer appsink + NVENC stack is still unavailable. "
            "Ensure PyGObject and the Jetson NVENC elements (nvvidconv / "
            "nvv4l2h264enc) are installed. Camera video will fall back to the "
            "ZED SDK path (higher latency) until then.",
            file=sys.stderr,
        )
