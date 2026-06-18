"""Best-effort bootstrap for the GStreamer-native WebRTC bindings.

The video relay drives ``webrtcbin`` through PyGObject (``gi``), which the
project doesn't ship by default. The system ``python3-gi`` is built against
the system interpreter and won't import under the project's Python 3.13
``uv`` venv, so PyGObject is installed **into the venv** (``uv pip install
pygobject``) while the GStreamer plugins + GObject-introspection typelibs come
from ``apt``. The venv's PyGObject loads the system typelibs via
``GI_TYPELIB_PATH``.

:func:`ensure_gst_webrtc` is best-effort and never raises: on a machine
without ``apt``/root (or anything that isn't a Jetson) it logs and returns
``False``, and the caller leaves headset video disabled.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

_logger = logging.getLogger(__name__)

# System packages: GStreamer webrtc + ICE + NVENC introspection, plus the
# build deps PyGObject needs to compile against gobject-introspection.
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

# Where apt drops the GObject-introspection typelibs; the venv PyGObject reads
# these via GI_TYPELIB_PATH.
_TYPELIB_DIRS = (
    "/usr/lib/girepository-1.0",
    f"/usr/lib/{os.uname().machine}-linux-gnu/girepository-1.0",
    "/usr/lib/aarch64-linux-gnu/girepository-1.0",
)


def _set_typelib_path() -> None:
    """Prepend the system typelib dirs to ``GI_TYPELIB_PATH`` for this process."""
    existing = os.environ.get("GI_TYPELIB_PATH", "")
    dirs = [d for d in _TYPELIB_DIRS if Path(d).is_dir()]
    if not dirs:
        return
    parts = dirs + ([existing] if existing else [])
    os.environ["GI_TYPELIB_PATH"] = os.pathsep.join(parts)


def _have_gst_webrtc() -> bool:
    """True when ``gi`` imports and ``webrtcbin`` + NVENC are registered."""
    try:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstWebRTC", "1.0")
        gi.require_version("GstSdp", "1.0")
        from gi.repository import Gst

        Gst.init(None)
    except Exception:  # noqa: BLE001 - bindings missing/unusable
        return False
    return all(
        Gst.ElementFactory.find(el) is not None
        for el in ("webrtcbin", "nvv4l2h264enc", "rtph264pay")
    )


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


def _apt_install() -> bool:
    apt = shutil.which("apt-get")
    if apt is None:
        _logger.info("apt-get not found; skipping system GStreamer install")
        return False
    prefix = [] if os.geteuid() == 0 else ["sudo", "-n"]
    _run([*prefix, apt, "update"])
    return _run([*prefix, apt, "install", "-y", *_APT_PACKAGES])


def _pip_install_pygobject() -> bool:
    uv = shutil.which("uv")
    if uv is not None and _run([uv, "pip", "install", "pygobject"]):
        return True
    return _run([sys.executable, "-m", "pip", "install", "pygobject"])


def ensure_gst_webrtc() -> bool:
    """Ensure PyGObject + ``webrtcbin`` + NVENC are importable; install if not.

    Best-effort and idempotent: returns ``True`` once the bindings work, or
    ``False`` (with a logged warning) when they can't be installed. Safe to
    call at every CLI startup.
    """
    _set_typelib_path()
    if _have_gst_webrtc():
        return True

    _logger.info("gstreamer WebRTC bindings missing; attempting install")
    _apt_install()
    _pip_install_pygobject()
    _set_typelib_path()

    if _have_gst_webrtc():
        _logger.info("gstreamer WebRTC bindings installed")
        return True
    _logger.warning(
        "gstreamer WebRTC bindings unavailable after install attempt; "
        "headset video will be disabled. Ensure webrtcbin "
        "(gstreamer1.0-plugins-bad), gstreamer1.0-nice, NVENC, and PyGObject "
        "are installed."
    )
    return False
