"""
axol provision

The single idempotent provisioning path for the pieces ``uv tool install`` /
``uv tool upgrade`` can't manage on their own:

* ``adb``           — Android Debug Bridge + the Oculus udev rule, for
                      streaming Quest controller poses over a USB
                      ``adb reverse`` tunnel (see :mod:`almond_axol.utils.adb`).
* ``zed.driver``    — replaces the ZED Box Duo's known-bad factory GMSL
                      capture driver with the pinned release (takes effect on
                      the next reboot; never reboots itself).
* ``zed.install``   — the pyzed bindings (not on PyPI; needs the ZED SDK).
* ``gst.install``   — the GStreamer + PyGObject ``appsink`` stack (PyGObject
                      builds against the system gobject-introspection and is
                      dropped on every ``uv tool upgrade``).
* ``gst.build-zed`` — the patched zedxonesrc/zedsrc plugins (sensor-accurate
                      PTS so collected images line up with joint samples).
* ``gyro.install``  — group access to the carrier board's BMI088 sampling
                      timer, Jelly heading hold's yaw reference (see
                      :mod:`almond_axol.robot.gyro`).
* ``rt.install``    — the ``axol-rt`` realtime core binary (Rust toolchain
                      via rustup if needed; sources fetched at the installed
                      package's ref for tool installs), required by hardware
                      control (see :mod:`almond_axol.rt`).
* rtprio grant      — a ``limits.d`` drop-in letting the operator's login run
                      the camera relay's capture chain ``SCHED_FIFO`` from a
                      manual ``axol serve`` (the systemd unit already has
                      ``LimitRTPRIO``); without it the relay silently runs
                      CFS and drops exposures under recording load (see
                      :mod:`almond_axol.utils.rtprio`).

Both the hosted installer (``web/app/public/install``) and the ``axol serve``
self-updater (:mod:`almond_axol.serve.update`) run *this* command, so the set
of steps lives in exactly one place and can't drift between them. Plain
``axol provision`` keeps every step idempotent and best-effort (each self-gates
on the ZED SDK / apt / NVENC), so it is safe to run on any host. The hosted
installer and post-upgrade path add ``--require-rt``: optional hardware remains
best-effort, but a failed required control-core install makes the command fail.

It does NOT pin Jetson clocks or steer the CAN adapters' interrupt — that's
``axol jetson.setup``, a per-boot runtime tweak owned by the systemd
``ExecStartPre``, not an install step.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from ..robot import gyro
from ..rt import install as rt_install
from ..utils import adb, rtprio
from .gst import build_zed as gst_build_zed
from .gst import install as gst_install
from .zed import driver as zed_driver
from .zed import install as zed_install

_logger = logging.getLogger(__name__)

# pyzed + the patched zed-gstreamer plugins need the ZED SDK headers; gating
# here keeps the no-SDK case quiet (zed.install otherwise hard-exits).
_ZED_SDK = Path("/usr/local/zed")


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``provision`` subcommand."""
    parser = subparsers.add_parser(
        "provision",
        help=(
            "Install/refresh the non-PyPI + system pieces "
            "(cameras, adb, board access, the operator's real-time scheduling "
            "grant, and the axol-rt control core)."
        ),
    )
    parser.add_argument(
        "--require-rt",
        action="store_true",
        help="exit non-zero if the required axol-rt core cannot be installed",
    )
    parser.set_defaults(func=run)


def _step(label: str, fn: Callable[[], object]) -> bool:
    """Run one step and report success, logging failures without stopping."""
    try:
        fn()
    except SystemExit as exc:  # a step (e.g. zed.install) may hard-exit on failure
        if exc.code not in (0, None):
            _logger.warning("provision: %s failed (exit %s)", label, exc.code)
            return False
    except Exception as exc:  # noqa: BLE001 - never let one step abort the rest
        _logger.warning("provision: %s failed: %s", label, exc)
        return False
    return True


def run(_args: object = None) -> None:
    """Run every provisioning step in order; each self-gates and is idempotent."""
    # Surface each step's INFO outcome (what was granted/installed, or already
    # in place) so a run at a customer site is verifiable from its output alone;
    # force=True in case an imported dependency already installed a handler.
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    # adb + the Oculus udev rule (which hands the headset to the `dialout`
    # group operators already have, so adb needs no extra group or re-login)
    # and adds the operator to that group — for streaming Quest controller
    # poses over a USB `adb reverse` tunnel (avoids WiFi latency). Self-gates
    # on apt-get.
    _step("adb (Quest-over-USB)", adb.install)
    # ZED Box Duo units ship with a known-bad factory GMSL capture driver;
    # replace it with the pinned release. Self-gates on the factory package
    # being present (ensure_driver, not run: a *quiet* no-op everywhere else)
    # and never reboots — the new kernel driver loads on the next reboot, so
    # it just prints a notice.
    _step("ZED Box camera driver (zed.driver)", zed_driver.ensure_driver)
    # Group access to the board IMU's sampling timer, so teleop can start the
    # Jelly's yaw reference without root. Self-gates on the driver's presence.
    _step("board IMU (gyro.install)", gyro.install)
    # Persistent rtprio allowance for the operator's login, so a manual
    # `axol serve` can run the camera relay's capture chain SCHED_FIFO like
    # the systemd unit does (LimitRTPRIO). Applies at the next login.
    _step("rtprio grant (utils.rtprio)", rtprio.install)
    have_sdk = _ZED_SDK.exists()
    if have_sdk:
        _step("pyzed (zed.install)", zed_install.run)
    else:
        print("No ZED SDK at /usr/local/zed; skipping pyzed + zed-gstreamer build.")
    _step("GStreamer + PyGObject (gst.install)", gst_install.run)
    if have_sdk:
        _step("patched zed-gstreamer plugins (gst.build-zed)", gst_build_zed.run)
    # The required axol-rt hardware control core: rustup toolchain if needed,
    # then build from the in-repo crate (dev checkout) or from the sources
    # at the installed package's exact ref (tool installs). Self-gates on
    # network/toolchain availability like every other step.
    rt_ok = _step("axol-rt realtime core (rt.install)", rt_install.run)
    if getattr(_args, "require_rt", False) and not rt_ok:
        raise SystemExit(1)
