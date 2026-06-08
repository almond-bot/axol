"""
axol zed.sync-clocks

Run a PTP (Precision Time Protocol) daemon over the direct ethernet link
between the Jetson sender and the upper-computer receiver, holding both
machines' ``CLOCK_REALTIME`` to sub-millisecond agreement. Required for the
ZED-frame ``TIME_REFERENCE.IMAGE`` timestamps consumed by ``collect-data``.

Both processes run in the foreground for the duration of a collection
session, one per machine:

    axol zed.sync-clocks --role master --iface eth0   # upper computer
    axol zed.sync-clocks --role slave  --iface eth0   # Jetson

``ptp4l``, ``phc2sys`` and the apt-get auto-install fallback are escalated
via ``sudo`` so the ``axol`` invocation itself does not need to be root.
On non-apt systems install ``linuxptp`` and ``ethtool`` manually first.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from ...sudo import (
    SUDO_BAD_PASSWORD_CODE,
    SUDO_BAD_PASSWORD_MARKER,
    SUDO_REQUIRED_CODE,
    SUDO_REQUIRED_MARKER,
    password_from_env,
    run_root,
)

# The PTP daemons we manage. Named so teardown can kill them by name as a
# safety net (they run as root, so a non-root signal can't reach them).
_DAEMON_NAMES = ("ptp4l", "phc2sys")

_logger = logging.getLogger(__name__)

_OFFSET_RE = re.compile(
    r"master offset\s+(?P<offset>-?\d+)\b.*?freq\s+(?P<freq>-?\d+)",
    re.IGNORECASE,
)

# Executable -> apt package, used by ``_ensure_executable`` to auto-install
# missing dependencies on Debian/Ubuntu.
_APT_PACKAGES = {
    "ptp4l": "linuxptp",
    "phc2sys": "linuxptp",
    "ethtool": "ethtool",
}


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``zed.sync-clocks`` subcommand."""
    p = subparsers.add_parser(
        "zed.sync-clocks",
        help=(
            "Synchronize sender and receiver CLOCK_REALTIME via PTP over the "
            "direct ethernet link (required for accurate ZED frame timestamps)."
        ),
    )
    p.add_argument(
        "--role",
        required=True,
        choices=["master", "slave"],
        help=(
            "PTP role for this machine. The upper computer (long-lived "
            "receiver) should be `master`; the Jetson sender should be `slave`."
        ),
    )
    p.add_argument(
        "--iface",
        required=True,
        metavar="IFACE",
        help="Network interface carrying the direct link (e.g. eth0).",
    )
    p.add_argument(
        "--transport",
        default="l2",
        choices=["l2", "udpv4"],
        help=(
            "PTP transport. `l2` (raw ethernet, default) is lower latency; "
            "`udpv4` is useful if a switch in between filters PTP ethertype."
        ),
    )
    p.add_argument(
        "--timestamping",
        default="auto",
        choices=["auto", "hardware", "software"],
        help=(
            "Force a timestamping mode. `auto` (default) probes "
            "`ethtool -T <iface>` and prefers hardware if available."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Run the PTP clock-sync daemons for this machine's role."""
    logging.basicConfig(level=getattr(logging, args.log_level))
    # The PTP daemons need root. A password may be supplied out-of-band via the
    # environment (the web control panel forwards the one the operator typed) so
    # it never lands in argv / process listings.
    sudo_password = password_from_env()
    try:
        _run(
            role=args.role,
            iface=args.iface,
            transport=args.transport,
            timestamping=args.timestamping,
            sudo_password=sudo_password,
        )
    except KeyboardInterrupt:
        pass


def _run(
    *,
    role: str,
    iface: str,
    transport: str,
    timestamping: str,
    sudo_password: str | None = None,
) -> None:
    _ensure_executable("ptp4l", required=True)
    _ensure_executable("phc2sys", required=True)

    if not Path(f"/sys/class/net/{iface}").exists():
        raise SystemExit(
            f"error: interface {iface!r} not found in /sys/class/net. "
            f"Plug in the cable or check `ip link show`."
        )

    _check_sudo(sudo_password)
    # A prior `axol serve` that was killed (or a disconnect that couldn't reach
    # the root daemons) can leave ptp4l/phc2sys orphaned and still disciplining
    # the clock. Clear any strays so we don't stack duplicates fighting over
    # CLOCK_REALTIME.
    _reap_stale_daemons(sudo_password)

    timestamping_mode = _resolve_timestamping(iface, timestamping)
    _logger.info(
        "Starting PTP role=%s iface=%s transport=%s timestamping=%s",
        role,
        iface,
        transport,
        timestamping_mode,
    )

    ptp4l_cmd = _with_sudo(
        _build_ptp4l_cmd(
            iface=iface,
            role=role,
            transport=transport,
            timestamping=timestamping_mode,
        ),
        sudo_password,
    )
    phc2sys_cmd = _with_sudo(
        _build_phc2sys_cmd(
            iface=iface,
            role=role,
            timestamping=timestamping_mode,
        ),
        sudo_password,
    )

    _logger.info("ptp4l:   %s", " ".join(_redact_sudo(ptp4l_cmd)))
    _logger.info("phc2sys: %s", " ".join(_redact_sudo(phc2sys_cmd)))

    ptp4l_proc = _spawn_daemon(ptp4l_cmd, sudo_password)
    phc2sys_proc = _spawn_daemon(phc2sys_cmd, sudo_password)

    procs: list[tuple[str, subprocess.Popen[str]]] = [
        ("ptp4l", ptp4l_proc),
        ("phc2sys", phc2sys_proc),
    ]
    stop_event = threading.Event()

    def _handle_signal(signum: int, _frame: object) -> None:
        _logger.info("Received signal %d; shutting down PTP processes.", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    threads: list[threading.Thread] = []
    for name, proc in procs:
        t = threading.Thread(
            target=_stream_subprocess,
            args=(name, proc, stop_event),
            name=f"axol-{name}-stream",
            daemon=True,
        )
        t.start()
        threads.append(t)

    try:
        while not stop_event.is_set():
            for name, proc in procs:
                if proc.poll() is not None:
                    _logger.error(
                        "%s exited unexpectedly with code %d; tearing down.",
                        name,
                        proc.returncode,
                    )
                    stop_event.set()
                    break
            stop_event.wait(timeout=0.5)
    finally:
        _terminate_procs(procs, sudo_password)
        for t in threads:
            t.join(timeout=2.0)


def _with_sudo(cmd: list[str], password: str | None = None) -> list[str]:
    """Prepend ``sudo`` unless already root.

    With a ``password`` we read it from stdin (``-S``); otherwise we require
    passwordless sudo (``-n``) so the daemon fails fast instead of blocking on a
    tty prompt that never comes (the web control panel runs without one).
    """
    if os.geteuid() == 0:
        return cmd
    if password:
        return ["sudo", "-S", "-p", "", *cmd]
    return ["sudo", "-n", *cmd]


def _redact_sudo(cmd: list[str]) -> list[str]:
    """Drop the ``-S`` flag from a logged command (cosmetic; no secret in argv)."""
    return [a for a in cmd if a != "-S"]


def _check_sudo(password: str | None) -> None:
    """Verify privilege escalation before launching the long-lived daemons.

    The sentinel markers + exit codes (shared with the web orchestrator) let it
    tell a *missing* password apart from a *wrong* one and react, instead of
    just watching the daemons die.
    """
    if os.geteuid() == 0:
        return
    if password:
        result = subprocess.run(
            ["sudo", "-S", "-p", "", "-v"],
            input=f"{password}\n",
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            print(SUDO_BAD_PASSWORD_MARKER, flush=True)
            raise SystemExit(SUDO_BAD_PASSWORD_CODE)
        return
    if subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode != 0:
        print(SUDO_REQUIRED_MARKER, flush=True)
        raise SystemExit(SUDO_REQUIRED_CODE)


def _spawn_daemon(cmd: list[str], password: str | None) -> subprocess.Popen[str]:
    """Popen a PTP daemon, feeding the sudo password on stdin when supplied."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE if password else None,
        text=True,
        bufsize=1,
    )
    if password and proc.stdin is not None:
        try:
            proc.stdin.write(f"{password}\n")
            proc.stdin.flush()
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
    return proc


def _ensure_executable(name: str, *, required: bool) -> bool:
    """Return True if ``name`` is on PATH, installing via apt if missing.

    Auto-install runs only on Debian/Ubuntu (where ``apt-get`` exists). If
    ``required`` is True and the executable still isn't available, raises
    ``SystemExit`` with a manual-install hint; otherwise returns False so
    the caller can degrade gracefully.
    """
    if shutil.which(name) is not None:
        return True

    pkg = _APT_PACKAGES.get(name)
    if pkg is not None and shutil.which("apt-get") is not None:
        _logger.info(
            "`%s` not found on PATH — installing the `%s` apt package ...",
            name,
            pkg,
        )
        cmd = _with_sudo(["apt-get", "install", "-y", "--no-install-recommends", pkg])
        env = {**os.environ, "DEBIAN_FRONTEND": "noninteractive"}
        try:
            subprocess.run(cmd, check=True, env=env)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            _logger.warning("Auto-install of `%s` failed: %s", pkg, exc)
        else:
            if shutil.which(name) is not None:
                _logger.info("Installed `%s` (provides `%s`).", pkg, name)
                return True

    if required:
        hint = (
            f"sudo apt install {pkg}"
            if pkg is not None
            else f"install `{name}` and rerun"
        )
        raise SystemExit(
            f"error: `{name}` not found on PATH and auto-install was "
            f"unavailable or failed. Try manually: {hint}."
        )
    return False


def _resolve_timestamping(iface: str, mode: str) -> str:
    if mode != "auto":
        return mode

    hw_supported = _probe_hardware_timestamping(iface)
    if hw_supported:
        _logger.info(
            "ethtool reports hardware timestamping on %s — using hardware.", iface
        )
        return "hardware"
    _logger.warning(
        "ethtool shows no PHC / hardware timestamping on %s. "
        "Falling back to software timestamping; expect ~10-100us extra jitter. "
        "(Pass --timestamping software explicitly to silence this warning.)",
        iface,
    )
    return "software"


def _probe_hardware_timestamping(iface: str) -> bool:
    if not _ensure_executable("ethtool", required=False):
        _logger.warning(
            "ethtool not installed and auto-install failed; cannot probe "
            "hardware timestamping for %s.",
            iface,
        )
        return False
    try:
        result = subprocess.run(
            ["ethtool", "-T", iface],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        _logger.warning("ethtool -T %s failed: %s", iface, exc)
        return False

    if result.returncode != 0:
        _logger.warning(
            "ethtool -T %s returned %d: %s",
            iface,
            result.returncode,
            result.stderr.strip(),
        )
        return False

    text = result.stdout
    has_phc = bool(re.search(r"PTP Hardware Clock:\s*(\d+)", text))
    has_hw_tx = "hardware-transmit" in text
    has_hw_rx = "hardware-receive" in text
    has_hw_raw = "hardware-raw-clock" in text
    return has_phc and has_hw_tx and has_hw_rx and has_hw_raw


def _build_ptp4l_cmd(
    *, iface: str, role: str, transport: str, timestamping: str
) -> list[str]:
    cmd = ["ptp4l", "-i", iface, "-m"]
    if transport == "l2":
        cmd.append("-2")
    if timestamping == "hardware":
        cmd.append("-H")
    else:
        cmd.append("-S")
    if role == "slave":
        cmd.append("-s")
    return cmd


def _build_phc2sys_cmd(*, iface: str, role: str, timestamping: str) -> list[str]:
    if timestamping != "hardware":
        # No PHC available — pin CLOCK_REALTIME to itself so ptp4l's own
        # SO_TIMESTAMPING path is the only thing disciplining the kernel clock.
        return [
            "phc2sys",
            "-c",
            "CLOCK_REALTIME",
            "-s",
            "CLOCK_REALTIME",
            "-O",
            "0",
            "-w",
        ]

    if role == "slave":
        return ["phc2sys", "-s", iface, "-c", "CLOCK_REALTIME", "-O", "0", "-w", "-m"]
    return ["phc2sys", "-s", "CLOCK_REALTIME", "-c", iface, "-O", "0", "-w", "-m"]


def _stream_subprocess(
    name: str, proc: subprocess.Popen[str], stop_event: threading.Event
) -> None:
    last_report = 0.0
    last_offset: int | None = None
    last_freq: int | None = None

    if proc.stdout is None:
        return
    for line in proc.stdout:
        if stop_event.is_set():
            break
        line = line.rstrip()
        if not line:
            continue
        sys.stdout.write(f"[{name}] {line}\n")
        sys.stdout.flush()

        m = _OFFSET_RE.search(line)
        if m is not None:
            try:
                last_offset = int(m.group("offset"))
                last_freq = int(m.group("freq"))
            except ValueError:
                pass

        now = time.monotonic()
        if last_offset is not None and now - last_report > 5.0:
            _logger.info(
                "[%s] latest master offset = %+d ns, freq adj = %+d ppb",
                name,
                last_offset,
                last_freq if last_freq is not None else 0,
            )
            last_report = now


def _root_kill(cmd: list[str], password: str | None) -> bool:
    """Best-effort privileged ``kill``/``pkill``. Returns True if it matched.

    The PTP daemons run as root (via ``sudo``), so a plain ``proc.terminate()``
    from this non-root wrapper hits ``EPERM`` and leaves them orphaned. Routing
    the kill back through ``sudo`` (cached creds from ``_check_sudo``, or the
    forwarded password) is the only thing that actually stops them.
    """
    try:
        result = run_root(cmd, password=password)
    except Exception:  # noqa: BLE001 - teardown is best-effort
        return False
    return result.returncode == 0


def _reap_stale_daemons(password: str | None) -> None:
    """Kill leftover ptp4l/phc2sys from a previous run before starting fresh."""
    for name in _DAEMON_NAMES:
        if _root_kill(["pkill", "-TERM", "-x", name], password):
            _logger.info("Reaped a stray %s from a previous run.", name)
    time.sleep(0.5)
    for name in _DAEMON_NAMES:
        _root_kill(["pkill", "-KILL", "-x", name], password)


def _terminate_procs(
    procs: list[tuple[str, subprocess.Popen[str]]], password: str | None
) -> None:
    # SIGTERM via root first: sudo relays it to the daemon (or hits the daemon
    # directly if sudo exec-replaced itself).
    for name, proc in procs:
        if proc.poll() is None:
            _logger.info("Terminating %s (pid %d).", name, proc.pid)
            _root_kill(["kill", "-TERM", str(proc.pid)], password)
    deadline = time.monotonic() + 3.0
    for name, proc in procs:
        timeout = max(0.0, deadline - time.monotonic())
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _logger.warning("%s did not exit cleanly; killing.", name)
            # Kill the daemon by name too: a SIGKILL to the sudo pid isn't
            # relayed and would orphan the root daemon.
            _root_kill(["pkill", "-KILL", "-x", name], password)
            _root_kill(["kill", "-KILL", str(proc.pid)], password)
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
