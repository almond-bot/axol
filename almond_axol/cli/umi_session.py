"""
axol umi.session

Zero-touch UMI data-collection session for a dedicated ("UMI") Jetson.

Runs the whole rig stack unattended: starts ``axol serve`` (control panel +
web app) and ``axol teleop --umi`` as supervised children, then watches USB
for a Quest headset. When one appears it sets up ``adb reverse`` tunnels for
the VR (8000) and serve (8001) ports and launches the headset browser at the
locally-served VR page with auto-connect query params — the only human steps
left are putting the headset on and pulling the trigger once to enter AR
(a browser-enforced user gesture; it cannot be scripted).

``axol umi.session --install`` writes and enables a systemd service so the
session starts on every boot, turning the machine into a dedicated UMI
Jetson. First-time note: the headset browser must accept the self-signed
certificates once (the page opens automatically; approve the interstitials);
after that, sessions are hands-free.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

from ..utils.sudo import run_root

_VR_PORT = 8000
_SERVE_PORT = 8001
_BROWSER_URL = f"https://localhost:{_SERVE_PORT}/vr?host=localhost&autoconnect=1"
_SERVICE_PATH = Path("/etc/systemd/system/axol-umi.service")


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``umi.session`` subcommand."""
    parser = subparsers.add_parser(
        "umi.session",
        help="Run the zero-touch UMI rig session (or --install it as a boot service).",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install + enable the axol-umi systemd service (runs this on boot).",
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
        subprocess.run(
            [adb, "-s", serial, "reverse", f"tcp:{port}", f"tcp:{port}"],
            capture_output=True,
            timeout=10,
        )
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


def _session() -> None:
    """Supervise serve + teleop --umi and bootstrap any Quest that appears."""
    axol = shutil.which("axol") or sys.argv[0]
    adb = _adb()
    if adb is None:
        print(
            "WARNING: adb not found — headset auto-launch disabled "
            "(install android-tools-adb)."
        )

    procs = {
        "serve": _spawn("serve", [axol, "serve"]),
        "teleop": _spawn("teleop --umi", [axol, "teleop", "--umi"]),
    }
    bootstrapped: set[str] = set()
    try:
        while True:
            for name, proc in list(procs.items()):
                if proc.poll() is not None:
                    print(f"{name} exited ({proc.returncode}); restarting in 5s")
                    time.sleep(5)
                    args = (
                        [axol, "serve"]
                        if name == "serve"
                        else [axol, "teleop", "--umi"]
                    )
                    procs[name] = _spawn(name, args)
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
        for proc in procs.values():
            proc.terminate()
        for proc in procs.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def _install() -> None:
    """Write + enable the systemd unit that runs this session on boot."""
    axol = shutil.which("axol")
    if not axol:
        raise SystemExit("Cannot resolve the `axol` executable to bake into the unit.")
    repo_root = Path(__file__).resolve().parents[2]
    user = subprocess.run(["whoami"], capture_output=True, text=True).stdout.strip()
    unit = f"""[Unit]
Description=Almond UMI rig session (serve + teleop --umi + Quest bootstrap)
After=network-online.target

[Service]
Type=simple
User={user}
WorkingDirectory={repo_root}
ExecStart={axol} umi.session
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    print(f"Installing {_SERVICE_PATH} (requires sudo)...")
    run_root(["tee", str(_SERVICE_PATH)], input_text=unit, check=True)
    run_root(["systemctl", "daemon-reload"], check=True)
    run_root(["systemctl", "enable", "--now", "axol-umi.service"], check=True)
    print("Done — this machine now runs the UMI session on boot.")
    print("  status : systemctl status axol-umi")
    print("  logs   : journalctl -fu axol-umi")
    print("  remove : sudo systemctl disable --now axol-umi")


def run(args: object = None) -> None:
    """Run the UMI session, or install it as a boot service with --install."""
    if getattr(args, "install", False):
        _install()
    else:
        _session()
