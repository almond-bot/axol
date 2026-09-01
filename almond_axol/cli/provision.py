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
                      timer, the cart heading hold's yaw reference (see
                      :mod:`almond_axol.robot.gyro`).
* ``tracker.install`` — pinned libsurvive + Vive USB permissions for Mantis
                        Lighthouse tracking.

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
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from ..robot import gyro
from ..utils import adb
from ..utils.host_update_lock import HostUpdateLockError, host_update_lock
from ..utils.sudo import prime_sudo, run_root
from . import tracker_install
from .gst import build_zed as gst_build_zed
from .gst import install as gst_install
from .zed import driver as zed_driver
from .zed import install as zed_install

_logger = logging.getLogger(__name__)

# pyzed + the patched zed-gstreamer plugins need the ZED SDK headers; gating
# here keeps the no-SDK case quiet (zed.install otherwise hard-exits).
_ZED_SDK = Path("/usr/local/zed")

# Older releases generated these scripts below the operator-writable
# ``~/.almond`` tree, then installed root cron/systemd references to them.  Do
# not copy the scripts into the new privileged location: their bytes may have
# been edited after setup.  Provisioning removes only exact references to the
# known generated filenames; the next ``axol can.setup`` regenerates trusted
# root-owned copies under /etc/almond-axol/can.
_PRE_MANTIS_NAME = "u" + "mi"
_LEGACY_CAN_SCRIPT_NAMES = frozenset(
    {
        "startup.sh",
        "startup_mantis.sh",
        f"startup_{_PRE_MANTIS_NAME}.sh",
        "rp1-usb-quirk.sh",
    }
)
_LEGACY_CAN_UNIT_FILES = (
    Path("/etc/systemd/system/axol-can-up.service"),
    Path("/etc/systemd/system/axol-can-mantis-up.service"),
    Path(f"/etc/systemd/system/axol-can-{_PRE_MANTIS_NAME}-up.service"),
    Path("/etc/systemd/system/axol-rp1-usb-quirk.service"),
)


def _is_legacy_operator_can_script(value: str) -> bool:
    """Whether ``value`` is one exact historical ``~/.almond/can`` script."""
    path = Path(value)
    return bool(
        path.is_absolute()
        and path.name in _LEGACY_CAN_SCRIPT_NAMES
        and path.parent.name == "can"
        and path.parent.parent.name == ".almond"
    )


def _neutralize_legacy_can_root_execution() -> bool:
    """Remove root execution references to operator-writable CAN scripts.

    Returns ``True`` only when at least one reference was removed. Unrelated
    root cron lines and systemd units are preserved byte-for-byte.
    """
    replacement_crontab: str | None = None

    if shutil.which("crontab") is not None:
        # Force a stable diagnostic so the normal "root has no crontab" case
        # is distinguishable from a real inspection failure on localized
        # hosts.  Unknown failures remain fatal.
        current = run_root(["env", "LC_ALL=C", "crontab", "-l"])
        if current.returncode == 0:
            lines = (current.stdout or "").splitlines()
            kept: list[str] = []
            for line in lines:
                prefix = "@reboot "
                candidate = line[len(prefix) :] if line.startswith(prefix) else ""
                if candidate and _is_legacy_operator_can_script(candidate):
                    continue
                kept.append(line)
            if len(kept) != len(lines):
                replacement_crontab = "\n".join(kept)
                if kept:
                    replacement_crontab += "\n"
        elif "no crontab" not in (current.stderr or "").lower():
            detail = (current.stderr or "").strip() or f"exit {current.returncode}"
            raise RuntimeError(f"could not inspect root crontab: {detail}")

    unsafe_units: list[Path] = []
    exec_prefix = "ExecStart=/bin/bash "
    for unit_file in _LEGACY_CAN_UNIT_FILES:
        try:
            lines = unit_file.read_text().splitlines()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError(f"could not inspect {unit_file}: {exc}") from exc
        unsafe = any(
            line.startswith(exec_prefix)
            and _is_legacy_operator_can_script(line[len(exec_prefix) :])
            for line in lines
        )
        if not unsafe:
            continue
        unsafe_units.append(unit_file)

    # Plan first, then prove every unsafe unit is stopped and disabled before
    # removing any scheduler reference.  In particular, never delete a unit
    # file while an attacker-modified legacy script may still be running: once
    # the definition is gone a failed stop is harder to retry or diagnose.
    for unit_file in unsafe_units:
        run_root(["systemctl", "stop", unit_file.name], check=True)
        run_root(["systemctl", "disable", unit_file.name], check=True)

    for unit_file in unsafe_units:
        run_root(["rm", "-f", str(unit_file)], check=True)
    if unsafe_units:
        run_root(["systemctl", "daemon-reload"], check=True)

    if replacement_crontab is not None:
        run_root(
            ["crontab", "-"],
            input_text=replacement_crontab,
            check=True,
        )

    scrubbed = bool(unsafe_units) or replacement_crontab is not None
    if scrubbed:
        print(
            "WARNING: Removed legacy root cron/systemd references to "
            "operator-writable CAN scripts. Run `sudo axol can.setup` with "
            "the adapters attached to restore boot/hotplug CAN bring-up from "
            "root-owned /etc/almond-axol/can scripts."
        )
    return scrubbed


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``provision`` subcommand."""
    subparsers.add_parser(
        "provision",
        help=(
            "Install/refresh the non-PyPI + system pieces "
            "(Lighthouse tracking, pyzed, GStreamer, camera plugins)."
        ),
    ).set_defaults(func=run)


def _step(label: str, fn: Callable[[], object]) -> bool:
    """Run one step and report failure without preventing later repairs."""
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


def _root_command() -> list[str]:
    """The same interpreter and CLI arguments, re-invoked as root."""
    return [sys.executable, "-m", "almond_axol", *sys.argv[1:]]


def _reexec_as_root() -> None:
    """Escalate the whole command so it can own the host update lock.

    The lock lives in a root-only state directory, and the hosted installer /
    managed ``axol serve`` already run as root. A developer or operator running
    this from a source checkout gets the usual one-time ``sudo`` prompt instead
    of a traceback, and the child keeps this exact interpreter so it provisions
    the same environment.
    """
    command = _root_command()
    if not prime_sudo():
        raise SystemExit(
            "Axol provisioning requires root. Rerun as:\n  sudo " + shlex.join(command)
        )
    result = subprocess.run(["sudo", *command], check=False)
    raise SystemExit(result.returncode)


def run(_args: object = None) -> None:
    """Run every provisioning step in order; each self-gates and is idempotent."""
    if os.geteuid() != 0:
        _reexec_as_root()
        return
    try:
        with host_update_lock():
            _run_locked()
    except HostUpdateLockError as exc:
        raise SystemExit(f"Axol provisioning could not start: {exc}") from exc


def _run_locked() -> None:
    """Provision while the caller owns the host-wide mutation lock."""
    # Security migration, not a best-effort dependency: if inspection or
    # removal fails, abort provisioning rather than silently leaving a root
    # scheduler pointed at an operator-writable executable.
    _neutralize_legacy_can_root_execution()

    failed: list[str] = []

    def step(label: str, fn: Callable[[], object]) -> None:
        if not _step(label, fn):
            failed.append(label)

    # adb + the Oculus udev rule (which hands the headset to the `dialout`
    # group operators already have, so adb needs no extra group or re-login)
    # and adds the operator to that group — for streaming Quest controller
    # poses over a USB `adb reverse` tunnel (avoids WiFi latency). Self-gates
    # on apt-get.
    step("adb (Quest-over-USB)", adb.install)
    step("Lighthouse tracking (tracker.install)", tracker_install.run)
    # ZED Box Duo units ship with a known-bad factory GMSL capture driver;
    # replace it with the pinned release. Self-gates on the factory package
    # being present (ensure_driver, not run: a *quiet* no-op everywhere else)
    # and never reboots — the new kernel driver loads on the next reboot, so
    # it just prints a notice.
    step("ZED Box camera driver (zed.driver)", zed_driver.ensure_driver)
    # Group access to the board IMU's sampling timer, so teleop can start the
    # cart's yaw reference without root. Self-gates on the driver's presence.
    step("board IMU (gyro.install)", gyro.install)
    have_sdk = _ZED_SDK.exists()
    if have_sdk:
        step("pyzed (zed.install)", zed_install.run)
    else:
        print("No ZED SDK at /usr/local/zed; skipping pyzed + zed-gstreamer build.")
    step("GStreamer + PyGObject (gst.install)", gst_install.run)
    if have_sdk:
        step("patched zed-gstreamer plugins (gst.build-zed)", gst_build_zed.run)

    if failed:
        raise SystemExit(
            "Provisioning failed for: "
            + ", ".join(failed)
            + ". See the log above, repair the host, and retry."
        )
