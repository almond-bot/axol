"""Pair a Lighthouse tracker with an HTC Watchman dongle."""

from __future__ import annotations

import selectors
import shutil
import subprocess
import time

_DEFAULT_TIMEOUT_S = 90.0
_PAIRING_START_TIMEOUT_S = 15.0


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``tracker.pair`` subcommand."""
    parser = subparsers.add_parser(
        "tracker.pair",
        help="Pair a Lighthouse tracker with its HTC Watchman dongle.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=_DEFAULT_TIMEOUT_S,
        help=f"Seconds to wait for pairing (default: {_DEFAULT_TIMEOUT_S:.0f}).",
    )
    parser.set_defaults(func=run)


def _stop_process(proc: subprocess.Popen[str]) -> None:
    """Stop survive-cli and its process group without leaving USB handles open."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()


def run(args) -> None:  # type: ignore[no-untyped-def]
    """Open libsurvive pairing mode and exit once a new tracker connects."""
    if args.timeout <= 0:
        raise SystemExit("--timeout must be greater than zero")

    executable = shutil.which("survive-cli")
    if executable is None:
        raise SystemExit(
            "Lighthouse tracking support is not installed; run `axol provision`."
        )

    print(
        "Unplug the tracker's USB cable, turn it on, then hold its button until "
        "the LED blinks blue.",
        flush=True,
    )
    print(
        "For predictable pairing, connect only the Watchman dongle intended for "
        "this tracker.",
        flush=True,
    )

    proc = subprocess.Popen(
        [executable, "--pair-device", "1", "--v", "100"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    started = time.monotonic()
    deadline = started + args.timeout
    saw_attempt = False
    seen_dongles: set[str] = set()

    try:
        while time.monotonic() < deadline:
            for key, _ in selector.select(timeout=0.25):
                line = key.fileobj.readline().strip()
                if not line:
                    continue

                if "Opening 28de:28de:00" in line:
                    # The dongle serial is parenthesized in libsurvive's log.
                    serial = line.split("(", 1)[-1].split(")", 1)[0]
                    if serial not in seen_dongles:
                        seen_dongles.add(serial)
                        print(f"Found Watchman dongle {serial}.", flush=True)
                elif "Pairing attempt..." in line:
                    if not saw_attempt:
                        print(
                            "Dongle pairing mode is active; waiting for the "
                            "blinking-blue tracker...",
                            flush=True,
                        )
                    saw_attempt = True
                elif saw_attempt and "Adding tracked object" in line:
                    print(
                        "Tracker paired successfully. The LED should be green.",
                        flush=True,
                    )
                    return
                elif "Error:" in line or "Warning:" in line:
                    print(line, flush=True)

            returncode = proc.poll()
            if returncode is not None:
                raise SystemExit(
                    f"survive-cli exited before pairing completed (code {returncode})."
                )
            if (
                not saw_attempt
                and time.monotonic() - started >= _PAIRING_START_TIMEOUT_S
            ):
                raise SystemExit(
                    "The dongle did not enter pairing mode. Run `axol provision` "
                    "to install the libusb build, reconnect the dongle, and try again."
                )

        raise SystemExit(
            "Pairing timed out. Confirm the tracker is blinking blue, reconnect its "
            "dongle, and try again."
        )
    finally:
        selector.close()
        _stop_process(proc)
