"""
axol provision

The single idempotent provisioning path for the pieces ``uv tool install`` /
``uv tool upgrade`` can't manage on their own:

* ``adb``           — Android Debug Bridge + the Oculus udev rule, for
                      streaming Quest controller poses over a USB
                      ``adb reverse`` tunnel (see :mod:`almond_axol.utils.adb`).
* ``zed.driver``    — replaces a ZED Box's (Duo or Mini) outdated factory
                      GMSL capture driver with the release pinned for the
                      ZED SDK (takes effect on the next reboot; never reboots
                      itself).
* ``zed.install``   — the pyzed bindings (not on PyPI; needs the ZED SDK).
* calibration cache — group-shares ``/usr/local/zed/settings`` so calibration
                      files cached by the root service stay readable from the
                      operator's own ``axol serve`` / ``axol teleop`` and vice
                      versa (see :mod:`almond_axol.zed.calibration`).
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
* CAN purge grant   — a ``sudoers.d`` drop-in letting a manual ``axol serve``
                      flap the CAN interfaces without a password when motor
                      power dies, so the e-stop's queued position commands
                      cannot replay on the next bring-up (see
                      :mod:`almond_axol.utils.can_purge`).
* rtprio grant      — a ``limits.d`` drop-in letting the operator's login run
                      the camera relay's capture chain ``SCHED_FIFO`` from a
                      manual ``axol serve`` (the systemd unit already has
                      ``LimitRTPRIO``); without it the relay silently runs
                      CFS and drops exposures under recording load (see
                      :mod:`almond_axol.utils.rtprio`).
* ``tracker.install`` — pinned libsurvive + Vive USB permissions for Mantis
                        Lighthouse tracking.
* host tuning       — the per-boot runtime tuning for the host it runs on
                      (:func:`tune_host`). On a Jetson (Orin NX, AGX Orin,
                      Thor): the max ``nvpmodel`` mode, engine + CPU clock
                      pins, the CAN interrupt steered onto a CAN core, and the
                      Argus daemon made ``SCHED_FIFO`` on the camera cores
                      (see :mod:`almond_axol.utils.jetson`). Nothing to tune on
                      a Raspberry Pi 5 or a workstation; it still logs the
                      core layout the loops will pin to.

Both the hosted installer (``web/app/public/install``) and the ``axol serve``
self-updater (:mod:`almond_axol.serve.update`) run *this* command, so the set
of steps lives in exactly one place and can't drift between them. Plain
``axol provision`` keeps every step idempotent (each self-gates on the ZED SDK /
apt / NVENC), so it is safe to run on any host; a step that self-gates is not a
failure, but a step that fails to repair the host is reported and makes the
command exit non-zero once every other step has had its chance. The hosted
installer and post-upgrade path pass ``--require-rt`` (accepted for
compatibility: the required control-core install already fails the command).

That tuning resets on every reboot, so ``axol provision --boot`` — the
tuning step alone, no installs and no update lock — is the systemd unit's
``ExecStartPre`` and re-applies it at each boot. An operator never runs a
second command: ``axol provision`` leaves the host fully set up now, and the
unit keeps it that way. ``axol jetson.setup`` is kept as an alias of
``--boot`` for units written by older installers.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

from ..robot import gyro
from ..rt import install as rt_install
from ..utils import adb, affinity, can_purge, jetson, rtprio
from ..utils.host_update_lock import (
    HOLDER_READY,
    HostUpdateLockError,
    host_update_lock,
)
from ..utils.state_files import privileged_service_active
from ..utils.sudo import prime_sudo, run_root
from ..zed import calibration as zed_calibration
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
            "operator-writable CAN scripts. Run `axol can.setup` with "
            "the adapters attached to restore boot/hotplug CAN bring-up from "
            "root-owned /etc/almond-axol/can scripts."
        )
    return scrubbed


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``provision`` subcommand."""
    parser = subparsers.add_parser(
        "provision",
        help=(
            "Install/refresh the non-PyPI + system pieces "
            "(cameras, adb, Lighthouse tracking, board access, the operator's "
            "real-time scheduling grant, and the axol-rt control core), then "
            "apply this host's real-time tuning (Jetson clocks, CAN interrupt)."
        ),
    )
    parser.add_argument(
        "--boot",
        action="store_true",
        help=(
            "apply only the per-boot host tuning (Jetson clocks, CAN interrupt, "
            "camera daemon scheduling) — the systemd unit's ExecStartPre"
        ),
    )
    parser.add_argument(
        "--require-rt",
        action="store_true",
        help=(
            "exit non-zero if the required axol-rt core cannot be installed "
            "(accepted for compatibility; every failed step already does)"
        ),
    )
    parser.set_defaults(func=run)


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


_HOLDER_COMMAND = [sys.executable, "-m", "almond_axol.utils.host_update_lock"]


@contextlib.contextmanager
def _sudo_held_update_lock() -> Iterator[None]:
    """Own the root-only host update lock from an operator's terminal run.

    The hosted installer and the managed ``axol serve`` are root and take the
    lock directly. From a source checkout the steps must keep running as the
    operator — their ``uv``, download caches, and venv — so only a small
    holder process escalates. It prints a ready line once it owns the lock and
    releases it when this process closes the pipe (or exits).
    """
    if not prime_sudo():
        raise HostUpdateLockError(
            "the host update lock needs sudo; rerun from a terminal that can "
            "authorize it, or as root"
        )
    holder = subprocess.Popen(
        ["sudo", "-n", *_HOLDER_COMMAND],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdin is not None and holder.stdout is not None
    try:
        ready = holder.stdout.readline().strip()
        if ready != HOLDER_READY:
            raise HostUpdateLockError("the lock holder did not start")
        yield
    finally:
        holder.stdin.close()
        holder.wait()


def tune_host(*, interactive: bool = False) -> None:
    """Apply the runtime tuning this host needs; it resets on every reboot.

    Decided by what the host exposes, never by a hard-coded board list: the
    Jetson steps self-gate on the L4T release file / Tegra devfreq nodes, the
    clock pins on whichever engine nodes exist (Orin's ``*.nvenc``/``*.gpu``,
    Thor's ``gpu-gpc-*``/``gpu-nvd-*``), and the interrupt / daemon placement
    on the online core count (:func:`affinity.core_groups`). The layout is
    logged *after* the power-mode step, which can online cores.
    """
    model = jetson.host_model() or "unknown board"
    if jetson._is_jetson():
        _logger.info("host: %s (Jetson) — applying the real-time tuning", model)
        jetson.pin_realtime_clocks(interactive=interactive)
    else:
        _logger.info("host: %s — no Jetson tuning to apply", model)
    _logger.info("core layout: %s", affinity.describe_layout())


def run(args: object = None) -> None:
    """Run every provisioning step in order; each self-gates and is idempotent."""
    # Surface each step's INFO outcome (what was granted/installed, or already
    # in place) so a run at a customer site is verifiable from its output alone;
    # force=True in case an imported dependency already installed a handler.
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    if getattr(args, "boot", False):
        # Per-boot: nothing to install, nothing that rewrites the tool env, so
        # no update lock (a boot-time ExecStartPre must never wait on one).
        tune_host(interactive=sys.stdin.isatty())
        return
    lock = host_update_lock if os.geteuid() == 0 else _sudo_held_update_lock
    try:
        with lock():
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
    # ZED Box units (Duo and Mini) ship with whatever GMSL capture driver was
    # current at flash time, and the ZED SDK needs a matching one; replace an
    # outdated driver with the pinned release. Self-gates on a stereolabs-zed*
    # package being present (ensure_driver, not run: a *quiet* no-op everywhere
    # else) and never reboots — the new kernel driver loads on the next reboot,
    # so it just prints a notice.
    step("ZED Box camera driver (zed.driver)", zed_driver.ensure_driver)
    # Group access to the board IMU's sampling timer, so teleop can start the
    # Jelly's yaw reference without root. Self-gates on the driver's presence.
    step("board IMU (gyro.install)", gyro.install)
    # Persistent rtprio allowance for the operator's login, so a manual
    # `axol serve` can run the camera relay's capture chain SCHED_FIFO like
    # the systemd unit does (LimitRTPRIO). Applies at the next login.
    step("rtprio grant (utils.rtprio)", rtprio.install)
    # Passwordless escalation for the one privileged thing a running session
    # must do on its own: flapping a CAN interface whose TX queue stalled
    # because motor power died. Without it a manual `axol serve` cannot purge,
    # and the e-stop's queued commands replay on the next enable.
    step("CAN e-stop purge grant (utils.can_purge)", can_purge.install)
    have_sdk = _ZED_SDK.exists()
    if have_sdk:
        step("pyzed (zed.install)", zed_install.run)
        # The SDK caches each camera's calibration under /usr/local/zed/settings
        # as whichever account opens it first. The hosted service is root, so
        # without this the operator's own `axol serve` / `axol teleop` can't
        # read (or refresh) those files and every camera open fails with
        # CALIBRATION FILE NOT AVAILABLE. Group-share the cache and make the
        # directory setgid so files created later inherit the group too.
        step("ZED calibration cache sharing", zed_calibration.share_calibration_files)
    else:
        print("No ZED SDK at /usr/local/zed; skipping pyzed + zed-gstreamer build.")
    step("GStreamer + PyGObject (gst.install)", gst_install.run)
    if have_sdk:
        step("patched zed-gstreamer plugins (gst.build-zed)", gst_build_zed.run)
    # The required axol-rt hardware control core: rustup toolchain if needed,
    # then build from the in-repo crate (dev checkout) or from the sources
    # at the installed package's exact ref (tool installs). Like every other
    # step it is reported rather than aborting the run, and any failure makes
    # the command exit non-zero below.
    step("axol-rt realtime core (rt.install)", rt_install.run)
    # Last, so the Argus daemon and CAN interfaces the earlier steps may have
    # (re)installed exist, and so this run leaves the host tuned now rather
    # than at its next boot. The same step is the unit's per-boot hook.
    #
    # Not from inside `axol serve` (the self-updater's post-upgrade pass and
    # its startup heal): those can overlap a robot session, and moving the
    # camera daemon or switching the power mode under a live session is what
    # the boot hook exists to avoid. They need not tune anyway — the update
    # ends in a service restart and the heal follows one, and axol.service's
    # ExecStartPre tunes the host before serve starts every time.
    if privileged_service_active():
        _logger.info(
            "host tuning: left to axol.service's ExecStartPre (runs before "
            "serve starts, including on the restart an update ends with)"
        )
    else:
        step(
            "host tuning (Jetson clocks, CAN interrupt, camera daemon)",
            lambda: tune_host(interactive=sys.stdin.isatty()),
        )

    if failed:
        raise SystemExit(
            "Provisioning failed for: "
            + ", ".join(failed)
            + ". See the log above, repair the host, and retry."
        )
