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

    sudo axol zed.sync-clocks --role master --iface eth0   # upper computer
    sudo axol zed.sync-clocks --role slave  --iface eth0   # Jetson

Both processes stay running in the foreground for the duration of any
collection session.
"""

from __future__ import annotations

import argparse
import logging
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
    _require_executable("ptp4l")
    _require_executable("phc2sys")

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

    ptp4l_cmd = _build_ptp4l_cmd(
        iface=iface,
        role=role,
        transport=transport,
        timestamping=timestamping_mode,
    )
    phc2sys_cmd = _build_phc2sys_cmd(
        iface=iface,
        role=role,
        timestamping=timestamping_mode,
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


def _require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(
            f"error: `{name}` not found on PATH. Install the `linuxptp` package "
            f"(e.g. `sudo apt install linuxptp`) and rerun."
        )


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
    if shutil.which("ethtool") is None:
        _logger.warning(
            "ethtool not installed; cannot probe hardware timestamping for %s.",
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
