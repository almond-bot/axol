"""
axol zed.sync-clocks

Run a PTP (Precision Time Protocol) daemon over the direct ethernet link
between the Jetson sender and the upper-computer receiver so both machines'
``CLOCK_REALTIME`` clocks stay aligned to sub-millisecond accuracy.

The ZED SDK stamps every frame on the *sender's* ``CLOCK_REALTIME`` via
``TIME_REFERENCE.IMAGE``. The receiver later converts that wall-clock stamp
into its own monotonic ``perf_counter`` so the dataset row's "frame time"
reflects the moment of capture rather than the moment of decode. None of
that works unless both machines agree on what time it is — which is what
this command exists to deliver.

Typical operator workflow (one terminal per machine):

    axol zed.sync-clocks --role master --iface eth0   # upper computer
    axol zed.sync-clocks --role slave  --iface eth0   # Jetson

Both processes stay running in the foreground for the duration of any
collection session.

The privileged subprocesses (`ptp4l`, `phc2sys`, and the apt-get auto-install
fallback) are escalated inline via `sudo`, so the `axol` invocation itself
does not need to be run as root. Sudo will prompt for a password on first
escalation and reuse cached credentials for the rest of the session. On
non-apt systems install `linuxptp` (for ptp4l/phc2sys) and `ethtool`
manually before running this command.
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

_logger = logging.getLogger(__name__)

_OFFSET_RE = re.compile(
    r"master offset\s+(?P<offset>-?\d+)\b.*?freq\s+(?P<freq>-?\d+)",
    re.IGNORECASE,
)

# Map executable name -> apt package that provides it. Used by
# `_ensure_executable` to auto-install missing dependencies on Debian/Ubuntu.
_APT_PACKAGES = {
    "ptp4l": "linuxptp",
    "phc2sys": "linuxptp",
    "ethtool": "ethtool",
}


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
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
    logging.basicConfig(level=getattr(logging, args.log_level))
    try:
        _run(
            role=args.role,
            iface=args.iface,
            transport=args.transport,
            timestamping=args.timestamping,
        )
    except KeyboardInterrupt:
        pass


def _run(*, role: str, iface: str, transport: str, timestamping: str) -> None:
    _ensure_executable("ptp4l", required=True)
    _ensure_executable("phc2sys", required=True)

    if not Path(f"/sys/class/net/{iface}").exists():
        raise SystemExit(
            f"error: interface {iface!r} not found in /sys/class/net. "
            f"Plug in the cable or check `ip link show`."
        )

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
        )
    )
    phc2sys_cmd = _with_sudo(
        _build_phc2sys_cmd(
            iface=iface,
            role=role,
            timestamping=timestamping_mode,
        )
    )

    _logger.info("ptp4l:   %s", " ".join(ptp4l_cmd))
    _logger.info("phc2sys: %s", " ".join(phc2sys_cmd))

    ptp4l_proc = subprocess.Popen(
        ptp4l_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    phc2sys_proc = subprocess.Popen(
        phc2sys_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

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
        _terminate_procs(procs)
        for t in threads:
            t.join(timeout=2.0)


def _with_sudo(cmd: list[str]) -> list[str]:
    """Prepend `sudo` when the current process is not already root.

    Sudo caches credentials per-tty for ~5 minutes by default, so back-to-back
    privileged subprocesses (ptp4l + phc2sys) only prompt for a password once.
    """
    if os.geteuid() == 0:
        return cmd
    return ["sudo", *cmd]


def _ensure_executable(name: str, *, required: bool) -> bool:
    """Return True if ``name`` is on PATH, installing it via apt if missing.

    Auto-install only runs on systems where ``apt-get`` is available
    (Debian/Ubuntu). When ``required`` is True and the executable cannot be
    made available, raises ``SystemExit`` with a manual-install hint;
    otherwise returns False and the caller is expected to degrade gracefully.
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
        # No PHC available — phc2sys runs in "free" mode pinning CLOCK_REALTIME
        # to itself, which is a no-op but lets ptp4l (in software mode) still
        # discipline the kernel clock through its own SO_TIMESTAMPING path.
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
        # Receiver follows the PHC that ptp4l is disciplining.
        return ["phc2sys", "-s", iface, "-c", "CLOCK_REALTIME", "-O", "0", "-w", "-m"]
    # Master pushes its system clock onto the PHC so ptp4l serves it out.
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


def _terminate_procs(procs: list[tuple[str, subprocess.Popen[str]]]) -> None:
    for name, proc in procs:
        if proc.poll() is None:
            _logger.info("Terminating %s (pid %d).", name, proc.pid)
            try:
                proc.terminate()
            except OSError:
                pass
    deadline = time.monotonic() + 3.0
    for name, proc in procs:
        timeout = max(0.0, deadline - time.monotonic())
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _logger.warning("%s did not exit cleanly; killing.", name)
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
