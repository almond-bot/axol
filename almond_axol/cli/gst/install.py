"""
axol gst.install

Install the GStreamer-native WebRTC stack that relays the ZED cameras to the
headset (``teleop --cameras`` / ``collect-data``): the system GStreamer plugins
(``webrtcbin`` from gst-plugins-bad, the ``libnice`` ICE plugin, NVENC
introspection) plus the PyGObject bindings.

PyGObject is not a usable PyPI wheel — it builds against the system
gobject-introspection and loads the system typelibs — so it can't be a normal
dependency. This command apt-installs the system packages and then builds
PyGObject into the interpreter running the CLI (the uv tool environment),
mirroring ``axol zed.install``. The hosted installer
(``web/app/public/install``) runs it once; it is not run at teleop/serve
startup. Best-effort: a no-op on machines without ``apt-get``/NVENC.
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

# System packages: GStreamer webrtc + ICE + NVENC introspection, plus the build
# deps PyGObject needs to compile against gobject-introspection.
_APT_PACKAGES = (
    "gir1.2-gstreamer-1.0",
    "gir1.2-gst-plugins-bad-1.0",
    "gstreamer1.0-plugins-bad",
    "gstreamer1.0-nice",
    "libgirepository1.0-dev",
    "gobject-introspection",
    "libcairo2-dev",
    "pkg-config",
)


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``gst.install`` subcommand."""
    subparsers.add_parser(
        "gst.install",
        help="Install the GStreamer WebRTC stack (webrtcbin + libnice + PyGObject).",
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


def _gi_ok() -> bool:
    """True when a clean subprocess can import gi and find webrtcbin + NVENC.

    Checked in a subprocess so the result reflects a PyGObject that was just
    installed (importing it in this process would hit a stale import state).
    """
    from ...vr.gst_webrtc import _set_typelib_path

    _set_typelib_path()  # ensure the child inherits the typelib search path
    code = (
        "import gi;"
        "gi.require_version('Gst','1.0');"
        "gi.require_version('GstWebRTC','1.0');"
        "gi.require_version('GstSdp','1.0');"
        "from gi.repository import Gst; Gst.init(None);"
        "import sys;"
        "els=('webrtcbin','nvv4l2h264enc','rtph264pay');"
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
    """Install the GStreamer WebRTC system packages and PyGObject bindings."""
    if _gi_ok():
        print("GStreamer WebRTC stack already available.")
        return

    print("Installing GStreamer WebRTC system packages (apt)...")
    _apt_install()
    print(f"Installing PyGObject ({_PYGOBJECT_SPEC}) into the axol environment...")
    _pip_install_pygobject()

    if _gi_ok():
        print("GStreamer WebRTC stack installed.")
    else:
        print(
            "WARNING: GStreamer WebRTC stack is still unavailable. Ensure "
            "webrtcbin (gstreamer1.0-plugins-bad), gstreamer1.0-nice, the "
            "Jetson NVENC plugin, and PyGObject are installed.",
            file=sys.stderr,
        )
