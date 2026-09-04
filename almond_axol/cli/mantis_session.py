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

import os
import pwd
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
_MANAGED_SERVE_SERVICE = "axol.service"
_SYSTEMD_RUNTIME = Path("/run/systemd/system")
_UPDATE_GUARD_MARKER = Path("/var/lib/almond-axol/update-incomplete")
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


def _managed_serve_is_installed() -> bool:
    """Whether systemd owns ``axol serve`` on this host.

    A helper fallback must never race the managed service's startup/restart or
    bypass its durable interrupted-update condition. Treat an inspection
    failure on a systemd boot as managed (fail closed); non-systemd development
    hosts can still use the standalone fallback.
    """
    systemctl = shutil.which("systemctl")
    if systemctl is None:
        return _SYSTEMD_RUNTIME.is_dir()
    try:
        result = subprocess.run(
            [
                systemctl,
                "show",
                "--property=LoadState",
                "--value",
                _MANAGED_SERVE_SERVICE,
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return _SYSTEMD_RUNTIME.is_dir()
    if result.returncode != 0:
        return _SYSTEMD_RUNTIME.is_dir()
    load_state = result.stdout.strip()
    if load_state == "not-found":
        return False
    if not load_state:
        return _SYSTEMD_RUNTIME.is_dir()
    return True


def _update_is_incomplete() -> bool:
    """Fail closed when the durable update marker exists or cannot be read."""
    try:
        os.lstat(_UPDATE_GUARD_MARKER)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return True


def _fallback_serve_block_reason() -> str | None:
    if _update_is_incomplete():
        return "an Axol update has not completed verification"
    if _managed_serve_is_installed():
        return f"{_MANAGED_SERVE_SERVICE} owns the control-panel server"
    return None


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def _session() -> None:
    """Keep the control panel available and bootstrap any Quest that appears."""
    axol = shutil.which("axol") or sys.argv[0]
    adb = _adb()
    if adb is None:
        print(
            "WARNING: adb not found — headset auto-launch disabled "
            "(install android-tools-adb)."
        )

    serve = None
    block_reason = _fallback_serve_block_reason()
    if block_reason is not None:
        print(f"serve fallback disabled: {block_reason}")
    elif not _serve_is_up():
        serve = _spawn("serve", [axol, "serve"])
    bootstrapped: set[str] = set()
    try:
        while True:
            next_block_reason = _fallback_serve_block_reason()
            if next_block_reason != block_reason:
                block_reason = next_block_reason
                if block_reason is not None:
                    print(f"serve fallback disabled: {block_reason}")
                else:
                    print("serve fallback is available")
            if serve is not None and block_reason is not None:
                _stop_process(serve)
                serve = None
            if serve is not None and serve.poll() is not None:
                print(f"serve exited ({serve.returncode}); restarting in 5s")
                time.sleep(5)
                if _fallback_serve_block_reason() is None:
                    serve = _spawn("serve", [axol, "serve"])
                else:
                    serve = None
            elif serve is None and block_reason is None and not _serve_is_up():
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
            _stop_process(serve)


def _operator_user() -> str:
    """Resolve the non-root operator that invoked the service installer."""
    if os.geteuid() == 0:
        raw_uid = os.environ.get("SUDO_UID", "")
        if not raw_uid.isascii() or not raw_uid.isdecimal():
            raise SystemExit(
                "Run `axol mantis.session --install` as the non-root operator; "
                "it will request sudo only for system changes."
            )
        uid = int(raw_uid)
    else:
        uid = os.geteuid()
    if uid == 0:
        raise SystemExit("The Mantis session service must not run as root.")
    try:
        account = pwd.getpwuid(uid)
    except (KeyError, OverflowError) as exc:
        raise SystemExit(f"No local account exists for operator uid {uid}.") from exc
    sudo_user = os.environ.get("SUDO_USER") if os.geteuid() == 0 else None
    if sudo_user and sudo_user != account.pw_name:
        raise SystemExit("SUDO_USER does not match SUDO_UID; refusing to install.")
    return account.pw_name


def _install() -> None:
    """Write + enable the systemd unit that runs this session on boot."""
    axol = shutil.which("axol")
    if not axol:
        raise SystemExit("Cannot resolve the `axol` executable to bake into the unit.")
    repo_root = Path(__file__).resolve().parents[2]
    user = _operator_user()
    # The condition must be *triggering* (``|``): the self-updater's drop-in
    # adds ``ConditionPathExists=|<one-shot token>`` to admit exactly one start
    # while the marker is armed, and systemd ANDs any non-triggering condition
    # with that OR group, which would make the token useless.
    unit = f"""[Unit]
Description=Almond Mantis Quest USB/browser bootstrap
Wants={_MANAGED_SERVE_SERVICE}
After=network-online.target {_MANAGED_SERVE_SERVICE}
ConditionPathExists=|!{_UPDATE_GUARD_MARKER}

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
