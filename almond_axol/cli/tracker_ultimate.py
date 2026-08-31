"""Install and inspect the Linux runtime for VIVE Ultimate Trackers.

The runtime is deliberately an explicit install rather than part of broad
``axol provision``: pyvut drives a reverse-engineered HID protocol and the
trackers must first be paired and mapped with VIVE Streaming Hub on Windows.
"""

from __future__ import annotations

import ctypes.util
import grp
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import sysconfig
import threading
import time
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

from ..tracker.config import TRACKER_CONFIG_FILE
from ..tracker.ultimate import (
    ULTIMATE_DONGLE_PID,
    ULTIMATE_DONGLE_VID,
    ULTIMATE_WIFI_CONFIG_FILE,
    ultimate_dongle_present,
    ultimate_wifi_config_error,
)
from ..utils.sudo import prime_sudo, run_root

_logger = logging.getLogger(__name__)

_PYVUT_REPO = "https://github.com/nijkah/pyvut.git"
_PYVUT_REF = "fcfcd33f4c1f16b0d84f5f741dc1319abdc7942a"
_PYVUT_SPEC = f"git+{_PYVUT_REPO}@{_PYVUT_REF}"
# Exact bytes of pyvut's package-local wifi_info.json at the pinned revision.
# Comparing its digest identifies the public placeholder without embedding or
# printing any credential value in Axol.
_PYVUT_DEFAULT_WIFI_SHA256 = (
    "fd64dd89b6dd61d06e91b1a5c913aa7fcae5ac2654903eb3f7e6dac8aeee2b67"
)

_APT_PACKAGES = ("libhidapi-hidraw0", "libhidapi-libusb0")
_UDEV_RULE_PATH = Path("/etc/udev/rules.d/70-axol-vive-ultimate.rules")
_UDEV_RULE = (
    "# Almond Axol: VIVE Ultimate Tracker wireless dongle\n"
    'SUBSYSTEM=="hidraw", KERNEL=="hidraw*", '
    'ATTRS{idVendor}=="0bb4", ATTRS{idProduct}=="0350", '
    'MODE="0660", GROUP="dialout", TAG+="uaccess"\n'
)
_UDEV_SEARCH_DIRS = (Path("/etc/udev/rules.d"), Path("/lib/udev/rules.d"))
_RUNTIME_PROBE_CACHE_TTL_S = 15.0
_runtime_probe_cache_lock = threading.Lock()
_runtime_probe_cache_at = 0.0
_runtime_probe_cache: dict[str, object] | None = None

# Run imports/enumeration in a fresh interpreter.  Besides making the result
# accurate immediately after an install, this avoids leaving a half-imported
# ``hid`` or ``pyvut`` module in the CLI process after a failed probe.
_PYTHON_PROBE = f"""
import hashlib
import json
import os
from pathlib import Path

result = {{
    "hid_ok": False,
    "hid_device_api": False,
    "hid_enumerate_api": False,
    "hid_error": "",
    "interfaces": [],
    "pyvut_ok": False,
    "pyvut_error": "",
    "api_compatible": False,
    "log_suppression_api": False,
    "packaged_wifi": "unavailable",
    "packaged_wifi_path": "",
}}
try:
    import hid
    result["hid_ok"] = True
    result["hid_device_api"] = callable(getattr(hid, "Device", None))
    result["hid_enumerate_api"] = callable(getattr(hid, "enumerate", None))
    try:
        for device in hid.enumerate({ULTIMATE_DONGLE_VID}, {ULTIMATE_DONGLE_PID}):
            raw_path = device.get("path")
            path = os.fsdecode(raw_path) if raw_path else ""
            result["interfaces"].append({{
                "interface": device.get("interface_number"),
                "path": path,
                "accessible": bool(path and os.access(path, os.R_OK | os.W_OK)),
            }})
    except Exception as exc:
        result["hid_error"] = f"{{type(exc).__name__}}: {{exc}}"
except Exception as exc:
    result["hid_error"] = f"{{type(exc).__name__}}: {{exc}}"

try:
    import pyvut
    from pyvut.tracker_core import set_tracker_core_verbose
    UltimateTrackerAPI = pyvut.UltimateTrackerAPI
    result["pyvut_ok"] = True
    result["log_suppression_api"] = callable(set_tracker_core_verbose)
    package_file = Path(pyvut.__file__).with_name("wifi_info.json")
    result["packaged_wifi_path"] = str(package_file)
    if package_file.is_file():
        digest = hashlib.sha256(package_file.read_bytes()).hexdigest()
        result["packaged_wifi"] = (
            "placeholder"
            if digest == "{_PYVUT_DEFAULT_WIFI_SHA256}"
            else "customized"
        )
    else:
        result["packaged_wifi"] = "missing"
    callback = callable(getattr(UltimateTrackerAPI, "add_pose_callback", None))
    explicit_lifecycle = (
        callable(getattr(UltimateTrackerAPI, "start", None))
        and callable(getattr(UltimateTrackerAPI, "stop", None))
    )
    context_lifecycle = (
        callable(getattr(UltimateTrackerAPI, "__enter__", None))
        and callable(getattr(UltimateTrackerAPI, "__exit__", None))
    )
    result["api_compatible"] = bool(
        callback
        and (explicit_lifecycle or context_lifecycle)
        and result["hid_device_api"]
        and result["hid_enumerate_api"]
        and result["log_suppression_api"]
    )
except Exception as exc:
    result["pyvut_error"] = f"{{type(exc).__name__}}: {{exc}}"

print(json.dumps(result, sort_keys=True))
"""


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register explicit Ultimate install and read-only check commands."""
    subparsers.add_parser(
        "tracker.ultimate.install",
        help="Install the pinned pyvut Linux runtime and Ultimate USB rules.",
    ).set_defaults(func=run_install)
    subparsers.add_parser(
        "tracker.ultimate.check",
        help="Check Ultimate dongle, runtime, permissions, and saved bindings.",
    ).set_defaults(func=run_check)


def _run(
    cmd: list[str], *, timeout: float = 900.0
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        _logger.warning("command failed (%s): %s", " ".join(cmd), exc)
        return None


def _python_probe() -> dict[str, object]:
    proc = _run([sys.executable, "-c", _PYTHON_PROBE], timeout=30.0)
    if proc is None or proc.returncode != 0:
        detail = ""
        if proc is not None:
            detail = (proc.stderr or proc.stdout).strip()
        return {
            "hid_ok": False,
            "hid_device_api": False,
            "hid_enumerate_api": False,
            "hid_error": detail or "probe interpreter failed",
            "interfaces": [],
            "pyvut_ok": False,
            "pyvut_error": detail or "probe interpreter failed",
            "api_compatible": False,
            "log_suppression_api": False,
            "packaged_wifi": "unavailable",
            "packaged_wifi_path": "",
        }
    try:
        # A dependency may print a diagnostic before our JSON; consume the
        # final non-empty line rather than treating harmless output as failure.
        line = next(line for line in reversed(proc.stdout.splitlines()) if line)
        value = json.loads(line)
    except (StopIteration, json.JSONDecodeError) as exc:
        return {
            "hid_ok": False,
            "hid_device_api": False,
            "hid_enumerate_api": False,
            "hid_error": f"invalid probe result: {exc}",
            "interfaces": [],
            "pyvut_ok": False,
            "pyvut_error": f"invalid probe result: {exc}",
            "api_compatible": False,
            "log_suppression_api": False,
            "packaged_wifi": "unavailable",
            "packaged_wifi_path": "",
        }
    return value if isinstance(value, dict) else {}


def _cached_python_probe(
    *, max_age_s: float = _RUNTIME_PROBE_CACHE_TTL_S
) -> dict[str, object]:
    """Cache the isolated import/HID-enumeration probe used by UI polling."""
    global _runtime_probe_cache_at, _runtime_probe_cache
    now = time.monotonic()
    with _runtime_probe_cache_lock:
        if _runtime_probe_cache is not None and now - _runtime_probe_cache_at < max(
            0.0, max_age_s
        ):
            # The result contains only JSON values, so this round-trip is a
            # cheap defensive copy compared with launching another interpreter.
            return json.loads(json.dumps(_runtime_probe_cache))
        result = _python_probe()
        _runtime_probe_cache = json.loads(json.dumps(result))
        _runtime_probe_cache_at = now
        return result


def _clear_runtime_probe_cache() -> None:
    global _runtime_probe_cache_at, _runtime_probe_cache
    with _runtime_probe_cache_lock:
        _runtime_probe_cache = None
        _runtime_probe_cache_at = 0.0


def _installed_pyvut() -> tuple[str | None, str | None]:
    """Return installed version and VCS commit without importing pyvut."""
    try:
        dist = distribution("pyvut")
    except PackageNotFoundError:
        return None, None
    commit = None
    try:
        direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
        vcs_info = direct_url.get("vcs_info", {})
        if isinstance(vcs_info, dict):
            value = vcs_info.get("commit_id")
            commit = value if isinstance(value, str) else None
    except (AttributeError, json.JSONDecodeError):
        pass
    return dist.version, commit


def _package_installed(name: str) -> bool | None:
    dpkg = shutil.which("dpkg-query")
    if dpkg is None:
        return None
    proc = _run([dpkg, "-W", "-f=${db:Status-Status}", name], timeout=15.0)
    return bool(
        proc is not None and proc.returncode == 0 and proc.stdout == "installed"
    )


def _missing_system_packages() -> list[str]:
    status = {package: _package_installed(package) for package in _APT_PACKAGES}
    if any(value is not None for value in status.values()):
        return [package for package, installed in status.items() if not installed]

    # Non-Debian fallback: identify the shared libraries rather than assuming
    # they are absent just because dpkg is unavailable.
    libraries = {
        "libhidapi-hidraw0": "hidapi-hidraw",
        "libhidapi-libusb0": "hidapi-libusb",
    }
    return [
        package
        for package, library in libraries.items()
        if ctypes.util.find_library(library) is None
    ]


def _install_system_packages() -> bool:
    missing = _missing_system_packages()
    if not missing:
        print("Linux hidapi libraries are already installed.", flush=True)
        return True
    apt = shutil.which("apt-get")
    if apt is None:
        print(
            "Cannot install missing HID libraries automatically; install: "
            + " ".join(missing),
            file=sys.stderr,
        )
        return False
    try:
        have_root = prime_sudo()
    except OSError:
        have_root = False
    if not have_root:
        install_command = "sudo apt-get install -y " + " ".join(missing)
        print(
            f"Installing Ultimate HID libraries needs root. Run: {install_command}",
            file=sys.stderr,
        )
        return False

    print("Installing Linux hidapi libraries…", flush=True)
    # Refreshing package indexes is best-effort; apt may already have a usable
    # cache.  Installing only missing packages avoids upgrading working ones.
    run_root([apt, "update"])
    proc = run_root([apt, "install", "-y", *missing])
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        print(
            "HID library installation failed" + (f": {detail[-1]}" if detail else "."),
            file=sys.stderr,
        )
        return False
    return True


def _pip_install_pyvut() -> bool:
    if shutil.which("git") is None:
        print("git is required to install the pinned pyvut revision.", file=sys.stderr)
        return False

    purelib = Path(sysconfig.get_path("purelib"))
    writable_environment = os.access(purelib, os.W_OK)
    if not writable_environment:
        try:
            have_root = prime_sudo()
        except OSError:
            have_root = False
        if not have_root:
            print(
                "The Axol Python environment is not writable. Run `sudo axol "
                "tracker.ultimate.install`, or rerun from a terminal that can "
                "authorize sudo.",
                file=sys.stderr,
            )
            return False

    def install(cmd: list[str]) -> subprocess.CompletedProcess[str] | None:
        if writable_environment:
            return _run(cmd)
        return run_root(cmd)

    print(f"Installing pinned pyvut {_PYVUT_REF[:12]}…", flush=True)
    uv = shutil.which("uv")
    if uv is not None:
        proc = install(
            [
                uv,
                "pip",
                "install",
                "--python",
                sys.executable,
                "--reinstall-package",
                "pyvut",
                _PYVUT_SPEC,
            ]
        )
        if proc is not None and proc.returncode == 0:
            return True
        if proc is not None:
            detail = (proc.stderr or proc.stdout).strip().splitlines()
            if detail:
                _logger.warning("uv could not install pyvut: %s", detail[-1])

    # Some system Python installs do not carry pip; keep this as a fallback
    # for conventional venvs while preferring uv-tool's explicit interpreter.
    hid_proc = install([sys.executable, "-m", "pip", "install", "hid>=1.0.5"])
    pyvut_proc = install(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            _PYVUT_SPEC,
        ]
    )
    if all(
        proc is not None and proc.returncode == 0 for proc in (hid_proc, pyvut_proc)
    ):
        return True
    for proc in (hid_proc, pyvut_proc):
        if proc is not None and proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip().splitlines()
            if detail:
                _logger.warning("pip could not install pyvut: %s", detail[-1])
    return False


def _matching_udev_rules() -> list[Path]:
    matches: list[Path] = []
    for directory in _UDEV_SEARCH_DIRS:
        try:
            candidates = directory.glob("*.rules")
        except OSError:
            continue
        for candidate in candidates:
            try:
                contents = candidate.read_text(errors="replace")
            except OSError:
                continue
            active_lines = [
                "".join(line.split("#", 1)[0].lower().split())
                for line in contents.splitlines()
            ]
            if any(_udev_line_grants_access(line) for line in active_lines):
                matches.append(candidate)
    return matches


def _udev_line_grants_access(line: str) -> bool:
    """Recognize a complete Ultimate hidraw permission rule, fail closed."""
    line = "".join(line.split("#", 1)[0].lower().split())
    if not all(token in line for token in ("hidraw", "0bb4", "0350")):
        return False
    uaccess = any(
        token in line for token in ('tag+="uaccess"', 'tag="uaccess"', 'tag:="uaccess"')
    )
    world_read_write = any(token in line for token in ('mode="0666"', 'mode:="0666"'))
    dialout_read_write = any(
        token in line for token in ('mode="0660"', 'mode:="0660"')
    ) and any(token in line for token in ('group="dialout"', 'group:="dialout"'))
    return uaccess or world_read_write or dialout_read_write


def _operator_has_dialout() -> bool:
    if os.geteuid() == 0:
        return True
    try:
        dialout_gid = grp.getgrnam("dialout").gr_gid
    except KeyError:
        return False
    return dialout_gid == os.getgid() or dialout_gid in os.getgroups()


def _install_udev_rule() -> bool:
    matches = _matching_udev_rules()
    if matches:
        print(
            "Ultimate dongle USB permissions already configured by "
            + ", ".join(str(path) for path in matches)
            + ".",
            flush=True,
        )
        if not _operator_has_dialout():
            print(
                "Note: this login is not in `dialout`; TAG+=uaccess may cover a "
                "local desktop session, but headless use needs dialout membership.",
                file=sys.stderr,
            )
        return True
    if _UDEV_RULE_PATH.exists():
        print(
            f"Preserving existing {_UDEV_RULE_PATH}; it does not contain an "
            "Ultimate 0bb4:0350 rule. Configure permissions there manually.",
            file=sys.stderr,
        )
        return False
    try:
        have_root = prime_sudo()
    except OSError:
        have_root = False
    if not have_root:
        print(
            "Installing the Ultimate udev rule needs root; rerun this command "
            "from a terminal with sudo access.",
            file=sys.stderr,
        )
        return False

    print(f"Installing Ultimate USB permissions at {_UDEV_RULE_PATH}…", flush=True)
    proc = run_root(["tee", str(_UDEV_RULE_PATH)], input_text=_UDEV_RULE)
    if proc.returncode != 0:
        print(f"Could not write {_UDEV_RULE_PATH}.", file=sys.stderr)
        return False
    udevadm = shutil.which("udevadm")
    if udevadm is None:
        print("udevadm is unavailable; reconnect the dongle after reboot.")
        return True
    reload_proc = run_root([udevadm, "control", "--reload-rules"])
    trigger_proc = run_root([udevadm, "trigger", "--subsystem-match=hidraw"])
    if reload_proc.returncode != 0 or trigger_proc.returncode != 0:
        print(
            "The udev rule was written but could not be applied immediately; "
            "reboot or reconnect the dongle.",
            file=sys.stderr,
        )
    if not _operator_has_dialout():
        print(
            "The udev rule uses group `dialout`; add the operator with `sudo "
            "usermod -aG dialout <user>` and log in again for headless use.",
            file=sys.stderr,
        )
    return True


def _read_bindings() -> tuple[str | None, str | None, str | None]:
    """Read Ultimate bindings without invoking config migration/writes."""
    try:
        value = json.loads(TRACKER_CONFIG_FILE.read_text())
    except FileNotFoundError:
        return None, None, "tracker config has not been created"
    except (OSError, json.JSONDecodeError) as exc:
        return None, None, f"cannot read {TRACKER_CONFIG_FILE}: {exc}"
    if not isinstance(value, dict):
        return None, None, "tracker config is not a JSON object"

    binding = None
    bindings = value.get("bindings")
    if isinstance(bindings, dict):
        candidate = bindings.get("ultimate")
        if isinstance(candidate, dict):
            binding = candidate
    if binding is None and value.get("backend") == "ultimate":
        binding = value
    if binding is None:
        return None, None, "no saved Ultimate binding"
    left = binding.get("left")
    right = binding.get("right")
    return (
        left if isinstance(left, str) and left else None,
        right if isinstance(right, str) and right else None,
        None,
    )


def _read_pose_conventions() -> tuple[object, object]:
    """Return configured Ultimate quaternion order/up axis without migration."""
    try:
        value = json.loads(TRACKER_CONFIG_FILE.read_text())
    except FileNotFoundError:
        return "wxyz", "z"
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(value, dict):
        return None, None
    return (
        value.get("ultimate_quat_order", "wxyz"),
        value.get("ultimate_up_axis", "z"),
    )


def is_ultimate_tracker_key(value: object) -> bool:
    """Whether ``value`` has pyvut's stable, non-zero-padded MAC shape."""
    # Upstream's mac_str does not zero-pad octets, so accept one or two hex
    # digits per field while still rejecting backend keys from other systems.
    if not isinstance(value, str):
        return False
    fields = value.split(":")
    if len(fields) != 6 or any(not 1 <= len(field) <= 2 for field in fields):
        return False
    try:
        return all(0 <= int(field, 16) <= 0xFF for field in fields)
    except ValueError:
        return False


def _packaged_wifi_status(probe: dict[str, object]) -> str:
    status = probe.get("packaged_wifi")
    if status in {"placeholder", "customized", "missing"}:
        return str(status)

    # Importing pyvut can fail solely because Python HID is absent.  Inspect
    # distribution metadata as a fallback so a repin never unknowingly
    # overwrites a package-local config that contains operator credentials.
    try:
        dist = distribution("pyvut")
    except PackageNotFoundError:
        return "unavailable"
    for entry in dist.files or ():
        if not str(entry).replace("\\", "/").endswith("pyvut/wifi_info.json"):
            continue
        package_file = Path(dist.locate_file(entry))
        try:
            digest = hashlib.sha256(package_file.read_bytes()).hexdigest()
        except OSError:
            return "missing"
        return "placeholder" if digest == _PYVUT_DEFAULT_WIFI_SHA256 else "customized"
    return "missing"


def _wifi_config_status(probe: dict[str, object]) -> tuple[str, str]:
    """Describe shared-map Wi-Fi readiness without returning credential data."""
    if ULTIMATE_WIFI_CONFIG_FILE.exists():
        error = ultimate_wifi_config_error(ULTIMATE_WIFI_CONFIG_FILE)
        if error is not None:
            return "FAIL", f"{ULTIMATE_WIFI_CONFIG_FILE}: {error}"
        try:
            exposed_bits = ULTIMATE_WIFI_CONFIG_FILE.stat().st_mode & 0o077
        except OSError as exc:
            return "FAIL", f"cannot inspect {ULTIMATE_WIFI_CONFIG_FILE}: {exc}"
        detail = (
            f"Axol config {ULTIMATE_WIFI_CONFIG_FILE} is valid "
            "(credential values redacted)"
        )
        if exposed_bits:
            return "WARN", detail + "; use mode 0600 to protect the password"
        return "OK", detail

    packaged = _packaged_wifi_status(probe)
    if packaged == "placeholder":
        return (
            "WARN",
            "Axol config is missing; pyvut's packaged public placeholder is active",
        )
    if packaged == "customized":
        return (
            "WARN",
            "Axol config is missing; a customized package-local config is active "
            "(credential values redacted, upgrade-fragile)",
        )
    if packaged == "missing":
        return "FAIL", "both the Axol and pyvut package-local configs are missing"
    return "FAIL", "Wi-Fi config cannot be checked until pyvut is importable"


def ultimate_runtime_readiness(*, cached: bool = True) -> dict[str, object]:
    """Return the non-invasive Ultimate checks shared by the CLI and UI.

    Import/API validation and ``hid.enumerate`` run in a child interpreter;
    this function never constructs ``UltimateTrackerAPI`` (whose constructor
    opens the dongle and can change RF/pairing state).  The UI path caches that
    comparatively expensive child probe while keeping sysfs, Wi-Fi, rules, and
    operator-group checks live.
    """
    probe = _cached_python_probe() if cached else _python_probe()
    sysfs_present = ultimate_dongle_present()
    missing_packages = _missing_system_packages()
    python_hid = bool(
        probe.get("hid_ok")
        and probe.get("hid_device_api")
        and probe.get("hid_enumerate_api")
        and not probe.get("hid_error")
    )
    api_compatible = bool(probe.get("api_compatible"))
    version, commit = _installed_pyvut()
    pinned = api_compatible and commit == _PYVUT_REF
    log_suppression = bool(probe.get("log_suppression_api"))

    wifi_level, wifi_detail = _wifi_config_status(probe)
    if wifi_level == "OK":
        wifi_status = "valid"
    elif (
        ULTIMATE_WIFI_CONFIG_FILE.exists()
        and ultimate_wifi_config_error(ULTIMATE_WIFI_CONFIG_FILE) is None
    ):
        wifi_status = "permissions-warning"
    elif wifi_level == "WARN":
        wifi_status = "missing"
    else:
        wifi_status = "invalid"

    raw_interfaces = probe.get("interfaces")
    interfaces = [
        item
        for item in (raw_interfaces if isinstance(raw_interfaces, list) else [])
        if isinstance(item, dict)
    ]
    interface_zero = [item for item in interfaces if item.get("interface") == 0]
    if any(bool(item.get("accessible")) for item in interface_zero):
        endpoint_status = "accessible"
    elif interface_zero:
        endpoint_status = "permission-denied"
    elif sysfs_present is True:
        endpoint_status = "missing"
    else:
        endpoint_status = "unavailable"

    udev_rules = _matching_udev_rules()
    udev_ready = bool(udev_rules)
    durable_operator_access = _operator_has_dialout()
    operator_access = durable_operator_access or endpoint_status == "accessible"
    runtime_installed = bool(
        not missing_packages
        and python_hid
        and api_compatible
        and pinned
        and log_suppression
        and udev_ready
    )

    issues: list[str] = []
    if missing_packages:
        issues.append("missing HID libraries: " + ", ".join(missing_packages))
    if not python_hid:
        issues.append(str(probe.get("hid_error") or "Python HID API is incompatible"))
    if not api_compatible:
        issues.append(
            str(probe.get("pyvut_error") or "pyvut API shape is incompatible")
        )
    elif not pinned:
        installed_ref = commit[:12] if commit else "unknown"
        issues.append(f"pyvut revision {installed_ref} is not pinned {_PYVUT_REF[:12]}")
    if not log_suppression:
        issues.append("pyvut cannot suppress credential-bearing verbose logs")
    if not udev_ready:
        issues.append("no persistent Ultimate dongle udev permission rule was found")
    if not operator_access:
        issues.append("this operator lacks durable dialout access")
    if sysfs_present is True and endpoint_status != "accessible":
        issues.append(f"Ultimate HID interface 0 is {endpoint_status}")

    return {
        "installed": runtime_installed,
        "nativeDependencies": not missing_packages,
        "missingNativeDependencies": missing_packages,
        "pythonHid": python_hid,
        "apiCompatible": api_compatible,
        "pinnedPyvut": pinned,
        "pinnedRef": _PYVUT_REF,
        "installedRef": commit,
        "pyvutVersion": version,
        "logSuppression": log_suppression,
        "udevReady": udev_ready,
        "udevRules": [str(path) for path in udev_rules],
        "operatorAccess": operator_access,
        "durableOperatorAccess": durable_operator_access,
        "dongleConnected": sysfs_present is True,
        "dongleStatus": (
            "connected"
            if sysfs_present is True
            else "disconnected"
            if sysfs_present is False
            else "unknown"
        ),
        "endpointStatus": endpoint_status,
        "wifiConfig": wifi_status,
        "wifiDetail": wifi_detail,
        "interfaces": interfaces,
        "hidError": str(probe.get("hid_error") or ""),
        "pyvutError": str(probe.get("pyvut_error") or ""),
        "issues": issues,
    }


def _print_wifi_action(probe: dict[str, object]) -> tuple[str, str]:
    level, detail = _wifi_config_status(probe)
    _result(level, "shared-map Wi-Fi", detail)
    if not ULTIMATE_WIFI_CONFIG_FILE.exists():
        print(
            f"Operator action: create {ULTIMATE_WIFI_CONFIG_FILE} with JSON keys "
            "ssid, pass, country, and freq, then set mode 0600. The installer "
            "does not invent, copy, or print credential values.",
            flush=True,
        )
    return level, detail


def _result(level: str, label: str, detail: str) -> None:
    print(f"{level:<4} {label:<18} {detail}", flush=True)


def run_install(_args: object = None) -> None:
    """Install the pin without syncing or uninstalling unrelated packages."""
    initial = _python_probe()
    version, commit = _installed_pyvut()
    # Native libraries are an independent prerequisite. A pinned Python
    # package can remain importable through a stale/partial host install, so
    # never skip this repair merely because pyvut itself is already pinned.
    native_ok = _install_system_packages()
    if bool(initial.get("api_compatible")) and commit == _PYVUT_REF:
        print(
            f"Pinned pyvut {version or 'unknown version'} ({commit[:12]}) is "
            "already installed.",
            flush=True,
        )
        python_ok = True
    else:
        durable_wifi_ok = (
            ULTIMATE_WIFI_CONFIG_FILE.exists()
            and ultimate_wifi_config_error(ULTIMATE_WIFI_CONFIG_FILE) is None
        )
        if _packaged_wifi_status(initial) == "customized" and not durable_wifi_ok:
            print(
                "Refusing to repin pyvut because its package-local wifi_info.json "
                "is customized and would be overwritten. Move those settings to "
                f"{ULTIMATE_WIFI_CONFIG_FILE} with mode 0600, then rerun. "
                "No credential values were printed.",
                file=sys.stderr,
            )
            python_ok = False
        else:
            python_ok = _pip_install_pyvut()

    udev_ok = _install_udev_rule()
    final = _python_probe()
    _clear_runtime_probe_cache()
    _, commit = _installed_pyvut()
    _print_wifi_action(final)
    final_readiness = ultimate_runtime_readiness(cached=False)
    if (
        native_ok
        and python_ok
        and bool(final.get("api_compatible"))
        and commit == _PYVUT_REF
        and udev_ok
        and final_readiness["installed"]
    ):
        print("VIVE Ultimate Linux runtime installed.", flush=True)
        print(
            "Next: pair the trackers and create their map in VIVE Streaming Hub "
            "on Windows, reconnect the dongle here, then run `axol "
            "tracker.identify --backend ultimate` followed by "
            "`axol tracker.ultimate.check`.",
            flush=True,
        )
        return

    issues = final_readiness.get("issues")
    detail = str(
        final.get("pyvut_error")
        or final.get("hid_error")
        or ("; ".join(str(issue) for issue in issues or []))
    )
    raise SystemExit(
        "VIVE Ultimate Linux runtime installation is incomplete"
        + (f": {detail}" if detail else ". See the messages above.")
    )


def run_check(_args: object = None) -> None:
    """Report setup readiness without opening HID or changing tracker state."""
    failures = 0
    print("VIVE Ultimate Tracker Linux setup check", flush=True)

    readiness = ultimate_runtime_readiness(cached=False)
    dongle_status = readiness["dongleStatus"]
    if dongle_status == "connected":
        _result("OK", "USB dongle", "0bb4:0350 is present")
    elif dongle_status == "disconnected":
        _result("FAIL", "USB dongle", "0bb4:0350 is not present")
        failures += 1
    else:
        _result("FAIL", "USB dongle", "Linux USB sysfs is unavailable")
        failures += 1

    missing = readiness["missingNativeDependencies"]
    assert isinstance(missing, list)
    if missing:
        _result(
            "FAIL",
            "HID libraries",
            "missing " + ", ".join(str(item) for item in missing),
        )
        failures += 1
    else:
        _result("OK", "HID libraries", "hidraw and libusb runtimes are installed")

    if readiness["pythonHid"]:
        _result("OK", "Python HID", "compatible `hid` API is importable")
    else:
        detail = str(readiness["hidError"] or "`hid.Device` is unavailable")
        _result("FAIL", "Python HID", detail)
        failures += 1

    version = readiness["pyvutVersion"]
    commit = readiness["installedRef"]
    if readiness["apiCompatible"]:
        detail = f"pyvut {version or 'unknown version'} API is available"
        if readiness["pinnedPyvut"]:
            detail += f" at pinned {str(readiness['pinnedRef'])[:12]}"
            _result("OK", "pyvut", detail)
        else:
            if commit:
                detail += f" from {str(commit)[:12]}"
            _result("FAIL", "pyvut", detail + "; run the installer to pin it")
            failures += 1
    else:
        detail = str(readiness["pyvutError"] or "API shape is incompatible")
        _result("FAIL", "pyvut", detail)
        failures += 1

    if readiness["logSuppression"]:
        _result(
            "OK",
            "credential logs",
            "pyvut verbose password output is suppressed before polling",
        )
    else:
        _result(
            "FAIL",
            "credential logs",
            "pyvut cannot suppress verbose password-bearing ACK output",
        )
        failures += 1

    wifi_status = readiness["wifiConfig"]
    wifi_level = "OK" if wifi_status == "valid" else "FAIL"
    _result(wifi_level, "shared-map Wi-Fi", str(readiness["wifiDetail"]))
    if not ULTIMATE_WIFI_CONFIG_FILE.exists():
        print(
            f"Operator action: create {ULTIMATE_WIFI_CONFIG_FILE} with JSON keys "
            "ssid, pass, country, and freq, then set mode 0600. The installer "
            "does not invent, copy, or print credential values.",
            flush=True,
        )
    if wifi_status != "valid":
        failures += 1

    raw_interfaces = readiness["interfaces"]
    interfaces = raw_interfaces if isinstance(raw_interfaces, list) else []
    interface_zero = [item for item in interfaces if item.get("interface") == 0]
    paths = ", ".join(str(item.get("path") or "unknown") for item in interface_zero)
    endpoint_status = readiness["endpointStatus"]
    if endpoint_status == "accessible":
        _result("OK", "HID endpoint", f"interface 0 is accessible ({paths})")
    elif endpoint_status == "permission-denied":
        _result("FAIL", "HID endpoint", f"interface 0 is not accessible ({paths})")
        failures += 1
    elif endpoint_status == "missing":
        _result("FAIL", "HID endpoint", "dongle present but HID interface 0 is absent")
        failures += 1
    else:
        _result("FAIL", "HID endpoint", "cannot inspect until the dongle is connected")
        failures += 1

    rules = readiness["udevRules"]
    assert isinstance(rules, list)
    if rules:
        _result("OK", "udev permissions", ", ".join(str(path) for path in rules))
    else:
        _result("FAIL", "udev permissions", "no persistent 0bb4:0350 rule found")
        failures += 1
    if readiness["operatorAccess"]:
        access_detail = (
            "current process has durable dialout access"
            if readiness["durableOperatorAccess"]
            else "current process can access the connected HID endpoint via uaccess"
        )
        _result("OK", "operator access", access_detail)
    else:
        _result(
            "FAIL",
            "operator access",
            "not in dialout; headless non-root HID access may fail",
        )
        failures += 1

    left, right, binding_error = _read_bindings()
    if (
        left
        and right
        and left != right
        and is_ultimate_tracker_key(left)
        and is_ultimate_tracker_key(right)
    ):
        _result("OK", "bindings", f"left={left}, right={right}")
    else:
        if left == right and left is not None:
            binding_error = "left and right are bound to the same tracker"
        elif left and not is_ultimate_tracker_key(left):
            binding_error = f"left binding is not a pyvut MAC: {left}"
        elif right and not is_ultimate_tracker_key(right):
            binding_error = f"right binding is not a pyvut MAC: {right}"
        elif binding_error is None:
            binding_error = f"left={left or 'unset'}, right={right or 'unset'}"
        _result("FAIL", "bindings", binding_error)
        failures += 1

    quat_order, up_axis = _read_pose_conventions()
    conventions = f"quaternion={quat_order!r}, up-axis={up_axis!r}"
    if quat_order == "wxyz" and up_axis == "z":
        _result("OK", "pose convention", conventions)
    elif quat_order in {"wxyz", "xyzw"} and up_axis in {"z", "y"}:
        _result(
            "WARN",
            "pose convention",
            conventions + "; differs from the pinned pyvut wxyz/z defaults",
        )
    else:
        _result("FAIL", "pose convention", conventions + "; invalid setting")
        failures += 1

    _result(
        "WARN",
        "map/live poses",
        "not opened by this non-invasive check; confirm with tracker.identify",
    )
    print(
        "The check intentionally does not construct pyvut: upstream opens the "
        "dongle and changes RF/pairing state in its constructor.",
        flush=True,
    )
    if failures:
        print(
            f"Ultimate setup check failed ({failures} issue(s)).",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)
    print(
        "Static Ultimate setup checks passed; verify the Windows map with live poses."
    )
