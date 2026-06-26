"""
axol provision

The single idempotent provisioning path for the pieces ``uv tool install`` /
``uv tool upgrade`` can't manage on their own:

* ``adb``           — Android Debug Bridge + the Oculus udev rule, for
                      streaming Quest controller poses over a USB
                      ``adb reverse`` tunnel (see :mod:`almond_axol.utils.adb`).
* ``zed.install``   — the pyzed bindings (not on PyPI; needs the ZED SDK).
* ``gst.install``   — the GStreamer + PyGObject ``appsink`` stack (PyGObject
                      builds against the system gobject-introspection and is
                      dropped on every ``uv tool upgrade``).
* ``gst.build-zed`` — the patched zedxonesrc/zedsrc plugins (sensor-accurate
                      PTS so collected images line up with joint samples).

Both the hosted installer (``web/app/public/install``) and the ``axol serve``
self-updater (:mod:`almond_axol.serve.update`) run *this* command, so the set
of steps lives in exactly one place and can't drift between them. Every step is
idempotent and best-effort (each self-gates on the ZED SDK / apt / NVENC and
no-ops when unavailable), so ``axol provision`` is safe to run on any host and
re-run anytime.

It does NOT pin Jetson clocks — that's ``axol jetson.setup``, a per-boot runtime
tweak owned by the systemd ``ExecStartPre``, not an install step.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from ..utils import adb
from .gst import build_zed as gst_build_zed
from .gst import install as gst_install
from .zed import install as zed_install

_logger = logging.getLogger(__name__)

# pyzed + the patched zed-gstreamer plugins need the ZED SDK headers; gating
# here keeps the no-SDK case quiet (zed.install otherwise hard-exits).
_ZED_SDK = Path("/usr/local/zed")


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``provision`` subcommand."""
    subparsers.add_parser(
        "provision",
        help=(
            "Install/refresh the non-PyPI + system pieces "
            "(pyzed, GStreamer/PyGObject, patched zed-gstreamer plugins)."
        ),
    ).set_defaults(func=run)


def _step(label: str, fn: Callable[[], None]) -> None:
    """Run one provisioning step; log and continue on failure (best-effort)."""
    try:
        fn()
    except SystemExit as exc:  # a step (e.g. zed.install) may hard-exit on failure
        if exc.code not in (0, None):
            _logger.warning("provision: %s failed (exit %s)", label, exc.code)
    except Exception as exc:  # noqa: BLE001 - never let one step abort the rest
        _logger.warning("provision: %s failed: %s", label, exc)


def run(_args: object = None) -> None:
    """Run every provisioning step in order; each self-gates and is idempotent."""
    # adb + the Oculus udev rule (which hands the headset to the `dialout`
    # group operators already have, so adb needs no extra group or re-login)
    # and adds the operator to that group — for streaming Quest controller
    # poses over a USB `adb reverse` tunnel (avoids WiFi latency). Self-gates
    # on apt-get.
    _step("adb (Quest-over-USB)", adb.install)
    have_sdk = _ZED_SDK.exists()
    if have_sdk:
        _step("pyzed (zed.install)", zed_install.run)
    else:
        print("No ZED SDK at /usr/local/zed; skipping pyzed + zed-gstreamer build.")
    _step("GStreamer + PyGObject (gst.install)", gst_install.run)
    if have_sdk:
        _step("patched zed-gstreamer plugins (gst.build-zed)", gst_build_zed.run)
