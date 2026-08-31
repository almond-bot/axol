"""
axol mantis.session

Quest browser/bootstrap helper for a dedicated Mantis host.

Ensures ``axol serve`` is available and watches USB for a Quest headset. When
one appears it sets up ``adb reverse`` tunnels for
the VR (8000) and serve (8001) ports and launches the headset browser at the
locally-served VR page with auto-connect query params — the only human steps
left are putting the headset on and pulling the trigger once to enter AR
(a browser-enforced user gesture; it cannot be scripted).

It deliberately does **not** pre-launch teleop: doing so monopolizes the VR
port and prevents the control panel from starting data collection. Select the
Quest source and launch teleop/collection from the panel after the browser
opens.

``axol mantis.session --install`` writes and enables a systemd helper on boot.
First-time note: the headset browser must accept the self-signed
certificates once (the page opens automatically; approve the interstitials);
after that, sessions are hands-free.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from ..utils.sudo import run_root

_VR_PORT = 8000
_SERVE_PORT = 8001
_BROWSER_URL = f"https://localhost:{_SERVE_PORT}/vr?host=localhost&autoconnect=1"
_SERVICE_PATH = Path("/etc/systemd/system/axol-mantis.service")
_PRE_MANTIS_SERVICE_NAME = f"axol-{'u' + 'mi'}.service"
_PRE_MANTIS_SERVICE_PATH = Path("/etc/systemd/system") / _PRE_MANTIS_SERVICE_NAME


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``mantis.session`` subcommand."""
    parser = subparsers.add_parser(
        "mantis.session",
        help="Run the zero-touch Mantis session (or --install it as a boot service).",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install + enable the axol-mantis systemd service (runs this on boot).",
    )
    parser.set_defaults(func=run)


def _adb() -> str | None:
    return shutil.which("adb")


def _quest_serial(adb: str) -> str | None:
    """Serial of the first authorized adb device, or None."""
    out = subprocess.run(
        [adb, "devices"], capture_output=True, text=True, timeout=10
    ).stdout
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) == 2 and parts[1] == "device":
            return parts[0]
    return None


def _bootstrap_headset(adb: str, serial: str) -> bool:
    """Reverse the ports and open the VR page in the headset browser."""
    for port in (_VR_PORT, _SERVE_PORT):
        reverse = subprocess.run(
            [adb, "-s", serial, "reverse", f"tcp:{port}", f"tcp:{port}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if reverse.returncode != 0:
            detail = (reverse.stderr or reverse.stdout).strip()
            print(f"headset {serial}: adb reverse for port {port} failed: {detail}")
            return False
    r = subprocess.run(
        [
            adb,
            "-s",
            serial,
            "shell",
            "am",
            "start",
            "-a",
            "android.intent.action.VIEW",
            "-d",
            _BROWSER_URL,
            "com.oculus.browser",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    ok = r.returncode == 0 and "Error" not in (r.stdout + r.stderr)
    print(
        f"headset {serial}: tunnels up, browser {'launched' if ok else 'launch FAILED'}"
    )
    return ok


def _spawn(name: str, args: list[str]) -> subprocess.Popen:
    print(f"starting {name}: {' '.join(args)}")
    return subprocess.Popen(args)


def _serve_is_up() -> bool:
    """Whether a local control-panel server already owns its HTTPS port."""
    try:
        with socket.create_connection(("127.0.0.1", _SERVE_PORT), timeout=0.5):
            return True
    except OSError:
        return False


def _session() -> None:
    """Keep the control panel available and bootstrap any Quest that appears."""
    axol = shutil.which("axol") or sys.argv[0]
    adb = _adb()
    if adb is None:
        print(
            "WARNING: adb not found — headset auto-launch disabled "
            "(install android-tools-adb)."
        )

    serve = None if _serve_is_up() else _spawn("serve", [axol, "serve"])
    bootstrapped: set[str] = set()
    try:
        while True:
            if serve is not None and serve.poll() is not None:
                print(f"serve exited ({serve.returncode}); restarting in 5s")
                time.sleep(5)
                serve = _spawn("serve", [axol, "serve"])
            if adb is not None:
                serial = _quest_serial(adb)
                if serial and serial not in bootstrapped:
                    if _bootstrap_headset(adb, serial):
                        bootstrapped.add(serial)
                elif serial is None:
                    # Replug re-bootstraps (tunnels die with the connection).
                    bootstrapped.clear()
            time.sleep(3)
    finally:
        if serve is not None:
            serve.terminate()
            try:
                serve.wait(timeout=10)
            except subprocess.TimeoutExpired:
                serve.kill()


def _install() -> None:
    """Write + enable the systemd unit that runs this session on boot."""
    axol = shutil.which("axol")
    if not axol:
        raise SystemExit("Cannot resolve the `axol` executable to bake into the unit.")
    repo_root = Path(__file__).resolve().parents[2]
    user = subprocess.run(["whoami"], capture_output=True, text=True).stdout.strip()
    unit = f"""[Unit]
Description=Almond Mantis Quest USB/browser bootstrap
After=network-online.target axol.service

[Service]
Type=simple
User={user}
WorkingDirectory={repo_root}
ExecStart={axol} mantis.session
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    if _PRE_MANTIS_SERVICE_PATH.exists():
        print("Migrating the retired Mantis session service name...")
        # Do not install the replacement while the retired service might still
        # own adb, the browser bootstrap, or axol serve. Require a verified stop
        # and disable before deleting its definition.
        run_root(["systemctl", "stop", _PRE_MANTIS_SERVICE_NAME], check=True)
        run_root(["systemctl", "disable", _PRE_MANTIS_SERVICE_NAME], check=True)
        run_root(["rm", "-f", str(_PRE_MANTIS_SERVICE_PATH)], check=True)
    print(f"Installing {_SERVICE_PATH} (requires sudo)...")
    run_root(["tee", str(_SERVICE_PATH)], input_text=unit, check=True)
    run_root(["systemctl", "daemon-reload"], check=True)
    run_root(["systemctl", "enable", "--now", "axol-mantis.service"], check=True)
    print("Done — this machine now bootstraps an attached Quest on boot.")
    print("  status : systemctl status axol-mantis")
    print("  logs   : journalctl -fu axol-mantis")
    print("  remove : sudo systemctl disable --now axol-mantis")


def run(args: object = None) -> None:
    """Run the Mantis session, or install it as a boot service with --install."""
    if getattr(args, "install", False):
        _install()
    else:
        _session()
