"""
axol can.driver

Builds and installs the ``gs_usb`` kernel module for the Almond Axol Hub CAN
adapter on kernels that do not ship it (NVIDIA L4T/tegra kernels on Jetson /
ZED Box hardware are built without any USB-CAN drivers).

The vendored source in ``gs_usb/`` is the upstream stable v5.15.148 driver
with two backports the Axol Hub needs — see ``gs_usb/README.md``. The module
is compiled against the running kernel's headers, installed under
``/lib/modules/$(uname -r)/updates/``, and registered in
``/etc/modules-load.d/`` so it loads on every boot. On kernels whose native
``gs_usb`` already has the required IDs and hub fixes, this whole command is a
no-op.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from ...utils.sudo import run_root

_SRC_DIR = Path(__file__).parent / "gs_usb"
_BUILD_DIR = Path.home() / ".almond" / "can" / "gs_usb-build"
_MODULES_ROOT = Path("/lib/modules")
_MODULES_LOAD_FILE = Path("/etc/modules-load.d/gs_usb.conf")
_LOADED_MODULE = Path("/sys/module/gs_usb")
_SIGNATURE_ENFORCEMENT = Path("/sys/module/module/parameters/sig_enforce")
_LOCKDOWN_STATE = Path("/sys/kernel/security/lockdown")
_MIN_NATIVE_HUB_KERNEL = (6, 13)
_VENDORED_MODULE_VERSION = "almond-5.15.148-hub2"

# USB IDs the installed driver must claim (modinfo alias fragments). Both the
# Axol and Mantis hubs use the dual-channel 1d50:606f board. A CANable 2.0 is
# still supported as an optional single-channel wheel/chest adapter, but only
# require its less-common ID when one is actually attached. Otherwise we would
# replace a Secure Boot-signed distro module for hardware the host does not use.
_HUB_ALIAS = "v1D50p606F"
_CANDLELIGHT_ALIAS = "v1209p2323"
_CANABLE2_ALIAS = "v16D0p117E"
_OPTIONAL_USB_ALIASES = {("16d0", "117e"): _CANABLE2_ALIAS}

# Before the version marker above existed, Almond's vendored module could only
# be distinguished from the upstream 5.15 module by this exact alias set and
# its canonical install location. That fingerprint is used solely to preserve
# an already-loaded, trusted/signed legacy module on signature-enforcing hosts;
# unsigned legacy installs are rebuilt once and acquire the explicit marker.
_LEGACY_VENDORED_ALIASES = (_HUB_ALIAS, _CANDLELIGHT_ALIAS, _CANABLE2_ALIAS)


class _DriverIdentityError(RuntimeError):
    """The active module cannot be proven to match modprobe's selection."""


def _find_modinfo() -> str | None:
    """Locate ``modinfo``, including the sbin dirs minimal PATHs often omit."""
    found = shutil.which("modinfo")
    if found:
        return found
    for candidate in ("/usr/sbin/modinfo", "/sbin/modinfo"):
        if Path(candidate).exists():
            return candidate
    return None


def _find_modprobe() -> str | None:
    """Locate ``modprobe``, including the sbin dirs minimal PATHs omit."""
    found = shutil.which("modprobe")
    if found:
        return found
    for candidate in ("/usr/sbin/modprobe", "/sbin/modprobe"):
        if Path(candidate).exists():
            return candidate
    return None


def is_driver_available() -> bool:
    """True when the running kernel has a resolvable ``gs_usb`` module."""
    modinfo = _find_modinfo()
    if modinfo is None:
        raise RuntimeError(
            "`modinfo` not found. Install kmod first (`sudo apt install kmod`)."
        )
    return subprocess.run([modinfo, "gs_usb"], capture_output=True).returncode == 0


def _signature_enforced() -> bool:
    """Whether the running kernel rejects unsigned modules."""
    try:
        if _SIGNATURE_ENFORCEMENT.read_text().strip().lower() in {"1", "y"}:
            return True
    except OSError:
        pass
    try:
        lockdown = _LOCKDOWN_STATE.read_text().strip().lower()
    except OSError:
        return False
    # The active mode is bracketed, e.g. ``none [integrity] confidentiality``.
    # Both enforcing modes reject unsigned modules even when sig_enforce itself
    # is unavailable or reports false.
    return "[integrity]" in lockdown or "[confidentiality]" in lockdown


def _module_field(field: str, module: str = "gs_usb") -> str:
    """One field reported by modinfo for a module name or module file."""
    modinfo = _find_modinfo()
    if modinfo is None:
        return ""
    result = subprocess.run(
        [modinfo, "-F", field, module], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _module_signer(module: str = "gs_usb") -> str:
    """Signer reported by modinfo for a module name or module file."""
    return _module_field("signer", module)


def _unsigned_driver_is_blocked() -> bool:
    """True when the selected module is unsigned and the kernel rejects that."""
    if not _signature_enforced():
        return False
    if _selected_driver_path() == "(builtin)":
        return False
    return not _module_signer()


def _selected_driver_path() -> str:
    """Resolved module path for diagnostics, or ``gs_usb`` if unavailable."""
    modinfo = _find_modinfo()
    if modinfo is None:
        return "gs_usb"
    result = subprocess.run(
        [modinfo, "-F", "filename", "gs_usb"], capture_output=True, text=True
    )
    return result.stdout.strip() or "gs_usb"


def _selected_driver_srcversion() -> str:
    """Build identity of the module selected by modprobe, if available."""
    return _module_field("srcversion")


def _vendored_driver_path() -> Path:
    """Canonical destination used by Almond's out-of-tree module install."""
    return _MODULES_ROOT / os.uname().release / "updates" / "gs_usb.ko"


def _resolved_path(path: str | Path) -> Path | None:
    """Resolve a module path, including weak-updates symlinks, if possible."""
    try:
        return Path(path).resolve()
    except OSError:
        return None


def _selected_driver_is_vendored() -> bool:
    """Whether modprobe selects Almond's canonical managed override path."""
    selected = _resolved_path(_selected_driver_path())
    destination = _resolved_path(_vendored_driver_path())
    return selected is not None and destination is not None and selected == destination


def _selected_driver_is_native(filename: str | None = None) -> bool:
    """Whether modprobe selects this kernel's in-tree module.

    ``weak-updates`` and the ``/lib`` -> ``/usr/lib`` layout used by several
    distributions are symlink based, so both sides must be resolved before the
    containment check.
    """
    filename = filename or _selected_driver_path()
    if filename == "(builtin)":
        return True
    selected = _resolved_path(filename)
    native_root = _resolved_path(_MODULES_ROOT / os.uname().release / "kernel")
    if selected is None or native_root is None:
        return False
    try:
        selected.relative_to(native_root)
    except ValueError:
        return False
    return True


def _module_info(module: str = "gs_usb") -> str:
    """Full modinfo output, or an empty string when it cannot be read."""
    modinfo = _find_modinfo()
    if modinfo is None:
        return ""
    result = subprocess.run([modinfo, module], capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else ""


def _loaded_module_field(field: str) -> str | None:
    """One sysfs field from the active module; ``None`` when unavailable."""
    try:
        return (_LOADED_MODULE / field).read_text().strip()
    except OSError:
        return None


def _selected_driver_is_legacy_vendored() -> bool:
    """Recognize Almond's pre-marker module without trusting arbitrary kmods.

    This narrow escape hatch exists for a module the operator previously
    signed and enrolled for Secure Boot. Replacing that known-good module with
    a newly built unsigned copy would make ``can.setup`` fail forever. The
    exact install path, matching selected/loaded source identity, out-of-tree
    metadata, empty version, trusted signer, signature enforcement, live
    module, and Almond-specific three-ID alias fingerprint must all agree. On
    ordinary unsigned hosts the old module is rebuilt once instead.
    """
    if not _selected_driver_is_vendored():
        return False
    taint = _loaded_module_field("taint")
    if (
        not _LOADED_MODULE.exists()
        or taint is None
        or "O" not in taint
        or not _signature_enforced()
        or not _module_signer()
    ):
        return False
    if _module_field("version") or _module_field("intree").strip().upper() == "Y":
        return False
    selected_srcversion = _selected_driver_srcversion()
    loaded_srcversion = _loaded_module_field("srcversion")
    if (
        not selected_srcversion
        or not loaded_srcversion
        or selected_srcversion != loaded_srcversion
    ):
        return False
    info = _module_info()
    return all(alias in info for alias in _LEGACY_VENDORED_ALIASES)


def _loaded_matches_selected_without_srcversion() -> bool:
    """Conservative identity fallback for kernels without module srcversions.

    Kernel module taint distinguishes in-tree from externally built code. For
    an in-tree selection, an active non-out-of-tree module is the only possible
    ``gs_usb`` supplied by that running kernel. For external code, require the
    active and selected Almond version markers to match, or the tightly scoped
    signed-legacy fingerprint above. Missing taint metadata remains an error.
    """
    taint = _loaded_module_field("taint")
    if taint is None:
        return False
    selected_path = _selected_driver_path()
    if _selected_driver_is_native(selected_path):
        return "O" not in taint
    if "O" not in taint:
        return False

    selected_version = _module_field("version")
    loaded_version = _loaded_module_field("version")
    if (
        selected_version == _VENDORED_MODULE_VERSION
        and loaded_version == selected_version
    ):
        return True
    return _selected_driver_is_legacy_vendored()


def _load_available_driver() -> None:
    """Load the selected module once so metadata alone cannot mask a bad file."""
    if _LOADED_MODULE.exists():
        if _selected_driver_path() == "(builtin)":
            return
        loaded_srcversion = _loaded_module_field("srcversion") or ""
        selected_srcversion = _selected_driver_srcversion()
        if loaded_srcversion and selected_srcversion:
            if loaded_srcversion == selected_srcversion:
                return
            raise _DriverIdentityError(
                "The active gs_usb module differs from the module selected on disk "
                f"({_selected_driver_path()}). Remove the stale override or reboot "
                "into the intended driver, then retry."
            )
        if _loaded_matches_selected_without_srcversion():
            return
        raise _DriverIdentityError(
            "The active gs_usb module and the module selected on disk do not "
            "expose enough build identity to prove they match "
            f"({_selected_driver_path()}). Unplug the CAN adapters, run "
            "`sudo modprobe -r gs_usb && sudo modprobe gs_usb`, then retry "
            "(or reboot after removing any stale override)."
        )
    loaded = run_root(["modprobe", "gs_usb"])
    if loaded.returncode == 0 and _LOADED_MODULE.exists():
        return
    error = (loaded.stderr or "").strip().splitlines()
    detail = error[-1] if error else f"exit code {loaded.returncode}"
    raise RuntimeError(
        f"The selected gs_usb module could not be loaded: "
        f"{_selected_driver_path()} ({detail})."
    )


def _attached_usb_ids() -> set[tuple[str, str]]:
    """Lowercase ``(vendor, product)`` IDs for attached USB devices."""
    ids: set[tuple[str, str]] = set()
    for vendor_file in Path("/sys/bus/usb/devices").glob("*/idVendor"):
        try:
            vendor = vendor_file.read_text().strip().lower()
            product = (vendor_file.parent / "idProduct").read_text().strip().lower()
        except OSError:
            continue
        ids.add((vendor, product))
    return ids


def _required_aliases() -> set[str]:
    """Module aliases required for the hub plus attached optional adapters."""
    attached = _attached_usb_ids()
    return {_HUB_ALIAS} | {
        alias for usb_id, alias in _OPTIONAL_USB_ALIASES.items() if usb_id in attached
    }


def _driver_supports_required_ids() -> bool:
    """True when the available ``gs_usb`` claims every USB ID setup requires."""
    out = _module_info()
    return all(alias in out for alias in _required_aliases())


def _kernel_release_at_least(minimum: tuple[int, int]) -> bool:
    """Whether the running kernel's leading major.minor meets ``minimum``."""
    parts = os.uname().release.split(".", 2)
    try:
        current = (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return False
    return current >= minimum


def _module_references_symbol(symbol: str, module: str | None = None) -> bool:
    """Whether kmod metadata proves that a module imports ``symbol``.

    Maintained distro kernels backported gs_usb's endpoint-discovery fix to
    releases such as 6.6 and 6.12. ``modprobe --show-modversions`` understands
    compressed distro modules and exposes their imported symbol table, so an
    exact ``usb_find_common_endpoints`` entry is positive evidence of that
    backport. Missing kmod support or module-version metadata remains unknown
    (False), never an optimistic compatibility guess.
    """
    modprobe = _find_modprobe()
    module = module or _selected_driver_path()
    if modprobe is None or module in {"", "(builtin)", "gs_usb"}:
        return False
    result = subprocess.run(
        [modprobe, "--show-modversions", module], capture_output=True, text=True
    )
    if result.returncode != 0:
        return False
    return any(
        fields and fields[-1] == symbol
        for line in result.stdout.splitlines()
        if (fields := line.split())
    )


def _selected_driver_has_hub_fixes() -> bool:
    """Whether the selected driver has the two backports the Axol Hub needs.

    Both fixes are guaranteed upstream in Linux 6.13. Maintained 6.2+ distro
    kernels may carry the endpoint-discovery backport too; accept those only
    when kmod's imported-symbol metadata proves it. (6.2 is also the first
    mainline release with per-channel ``dev_id``.) External modules need
    Almond's explicit version marker, except for the narrowly fingerprinted
    signed legacy migration above. An unverifiable older distro backport is
    rebuilt rather than being trusted merely because it lives under
    ``updates/``.
    """
    filename = _selected_driver_path()
    if filename == "(builtin)":
        return _kernel_release_at_least(_MIN_NATIVE_HUB_KERNEL)
    if _selected_driver_is_native(filename):
        return (
            _kernel_release_at_least(_MIN_NATIVE_HUB_KERNEL)
            or _kernel_release_at_least((6, 2))
            and _module_references_symbol("usb_find_common_endpoints", filename)
        )
    if _module_field("version") == _VENDORED_MODULE_VERSION:
        return True
    return _selected_driver_is_legacy_vendored()


def _build() -> Path:
    """Compile gs_usb.ko against the running kernel. Returns the .ko path."""
    kver = os.uname().release
    kdir = _MODULES_ROOT / kver / "build"
    if not kdir.exists():
        raise RuntimeError(
            f"Kernel headers not found at {kdir}. Install them first "
            "(on Jetson/L4T: `sudo apt install nvidia-l4t-kernel-headers`)."
        )
    for tool in ("make", "gcc"):
        if shutil.which(tool) is None:
            raise RuntimeError(
                f"`{tool}` not found. Install build tools first "
                "(`sudo apt install build-essential`)."
            )

    print(f"Building gs_usb for kernel {kver}...")
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("gs_usb.c", "Makefile"):
        shutil.copy(_SRC_DIR / name, _BUILD_DIR / name)

    proc = subprocess.run(
        ["make", "-C", str(_BUILD_DIR)], capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gs_usb build failed:\n{proc.stdout}\n{proc.stderr}")
    print("  Done.")
    return _BUILD_DIR / "gs_usb.ko"


def _install(ko: Path) -> None:
    """Install the module, register it for boot, and load it (requires sudo)."""
    dest = _vendored_driver_path()
    backup = _BUILD_DIR / f"gs_usb-installed-{os.getpid()}.ko"
    config_backup = _BUILD_DIR / f"gs_usb-conf-{os.getpid()}.backup"
    had_override = dest.exists()
    had_config = _MODULES_LOAD_FILE.exists()
    if had_override or had_config:
        _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if had_override:
        shutil.copy2(dest, backup)
    if had_config:
        shutil.copy2(_MODULES_LOAD_FILE, config_backup)
    was_loaded = _LOADED_MODULE.exists()

    print(f"Installing {dest} (requires sudo)...")
    new_loaded = False
    active_after_unload = False
    try:
        run_root(["install", "-D", "-m", "644", str(ko), str(dest)], check=True)
        run_root(["depmod", "-a"], check=True)
        selected = _selected_driver_path()
        try:
            selected_is_dest = Path(selected).resolve() == dest.resolve()
        except OSError:
            selected_is_dest = False
        if not selected_is_dest:
            raise RuntimeError(
                "depmod did not select the newly installed gs_usb module "
                f"({dest}); modprobe still resolves to {selected}."
            )

        # Reload so the freshly-installed module (and its device table) takes
        # effect now — a bare modprobe is a no-op when an older gs_usb is
        # already loaded. Verify both transitions so an untested override can
        # never be left to shadow a working distro module on the next boot.
        unloaded = run_root(["modprobe", "-r", "gs_usb"])
        if _LOADED_MODULE.exists():
            # A failed unload of the previously-active module needs no second
            # unload during rollback. A successful unload (or a module that
            # appeared when none was initially loaded) means something raced
            # us and must be removed before restoring the old file.
            active_after_unload = unloaded.returncode == 0 or not was_loaded
            error = (unloaded.stderr or "").strip().splitlines()
            detail = error[-1] if error else "the module remained loaded"
            raise RuntimeError(
                f"Could not unload the active gs_usb driver ({detail}). Stop "
                "programs using the CAN interfaces, unplug the adapter, and retry."
            )

        loaded = run_root(["modprobe", "gs_usb"])
        new_loaded = _LOADED_MODULE.exists()
        if loaded.returncode != 0 or not new_loaded:
            error = (loaded.stderr or "").strip().splitlines()
            detail = error[-1] if error else f"exit code {loaded.returncode}"
            raise RuntimeError(
                f"The newly installed gs_usb module could not be loaded ({detail})"
            )
        built_srcversion = _module_field("srcversion", str(ko))
        try:
            loaded_srcversion = (_LOADED_MODULE / "srcversion").read_text().strip()
        except OSError:
            loaded_srcversion = ""
        if (
            built_srcversion
            and loaded_srcversion
            and built_srcversion != loaded_srcversion
        ):
            raise RuntimeError(
                "The gs_usb module loaded after installation does not match the "
                "module that was built."
            )
        if not (built_srcversion and loaded_srcversion):
            loaded_version = _loaded_module_field("version")
            if loaded_version != _VENDORED_MODULE_VERSION:
                raise RuntimeError(
                    "The loaded gs_usb module exposes neither the built module's "
                    "source identity nor Almond's version marker."
                )
        run_root(["tee", str(_MODULES_LOAD_FILE)], input_text="gs_usb\n", check=True)
    except RuntimeError as exc:
        recovery_errors: list[str] = []

        def recover(cmd: list[str]) -> None:
            try:
                run_root(cmd, check=True)
            except RuntimeError as recovery_exc:
                recovery_errors.append(str(recovery_exc))

        # Restore the selected file before unloading a bad replacement. If
        # hotplug races and autoloads the module during recovery, it can now
        # only select the restored driver.
        if had_override:
            recover(["install", "-D", "-m", "644", str(backup), str(dest)])
        else:
            recover(["rm", "-f", str(dest)])
        recover(["depmod", "-a"])
        if (new_loaded or active_after_unload) and _LOADED_MODULE.exists():
            recover(["modprobe", "-r", "gs_usb"])
        if was_loaded and not _LOADED_MODULE.exists():
            recover(["modprobe", "gs_usb"])
            if not _LOADED_MODULE.exists():
                recovery_errors.append("the previous gs_usb module is not loaded")
        elif not was_loaded and _LOADED_MODULE.exists():
            recovery_errors.append("gs_usb remained loaded")

        if had_config:
            recover(
                [
                    "install",
                    "-D",
                    "-m",
                    "644",
                    str(config_backup),
                    str(_MODULES_LOAD_FILE),
                ]
            )
        else:
            recover(["rm", "-f", str(_MODULES_LOAD_FILE)])

        if not recovery_errors:
            if had_override:
                backup.unlink(missing_ok=True)
            if had_config:
                config_backup.unlink(missing_ok=True)
            raise RuntimeError(
                f"{exc} The previous gs_usb driver state was restored."
            ) from exc

        backups = [str(backup)] if had_override else []
        if had_config:
            backups.append(str(config_backup))
        backup_note = f" Backups: {', '.join(backups)}." if backups else ""
        raise RuntimeError(
            f"{exc} Automatic rollback was incomplete: "
            f"{'; '.join(recovery_errors)}.{backup_note}"
        ) from exc

    if had_override:
        backup.unlink(missing_ok=True)
    if had_config:
        config_backup.unlink(missing_ok=True)
    print("  Done.")


def ensure_driver() -> bool:
    """Build and install gs_usb when the kernel's is missing or outdated.

    Rebuilds when no ``gs_usb`` is loadable *or* when the available one
    doesn't claim the USB IDs required by the hub and attached supported
    adapters. Returns True when the driver was (re)installed, False when it
    was already good. Idempotent; safe to call from ``can.setup`` on every
    machine.
    """
    available = is_driver_available()
    if available and _unsigned_driver_is_blocked():
        path = _selected_driver_path()
        raise RuntimeError(
            f"The selected gs_usb module is unsigned, but this kernel enforces "
            f"module signatures: {path}. Remove or sign that override, run "
            "`sudo depmod -a`, and retry."
        )
    if available:
        rebuild_for_identity = False
        try:
            _load_available_driver()
        except _DriverIdentityError:
            # Old Jetson installs predate MODULE_VERSION and often expose no
            # srcversion in sysfs.  The active module is therefore impossible
            # to compare with the selected file, even when both came from
            # Almond.  The canonical override is ours to replace, so migrate
            # it in place on hosts that permit unsigned modules.  Keep the
            # strict failure for distro/arbitrary paths and Secure Boot hosts.
            if _signature_enforced() or not _selected_driver_is_vendored():
                raise
            rebuild_for_identity = True

        if not rebuild_for_identity:
            if _driver_supports_required_ids() and _selected_driver_has_hub_fixes():
                return False
            if _selected_driver_path() == "(builtin)":
                raise RuntimeError(
                    "This kernel's built-in gs_usb driver lacks a required USB ID "
                    "or Axol Hub fix and cannot be replaced by a module. Use a "
                    "kernel whose driver supports the attached adapters, or rebuild "
                    "it with the Axol Hub fixes and CONFIG_CAN_GS_USB=m."
                )
            print(
                "Installed gs_usb driver lacks a required USB ID or Axol Hub "
                "backport — rebuilding it."
            )
        else:
            print(
                "Active gs_usb cannot be matched to Almond's managed driver "
                "override — rebuilding it."
            )
    else:
        print("Kernel does not ship the gs_usb driver — building it from source.")
    ko = _build()
    if _signature_enforced() and not _module_signer(str(ko)):
        raise RuntimeError(
            "This kernel enforces module signatures, but the required gs_usb "
            "backport built unsigned. Sign it with a trusted/enrolled key or "
            "use a kernel whose native gs_usb supports the attached adapters. "
            "The active driver was not changed."
        )
    _install(ko)
    return True


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``can.driver`` subcommand."""
    subparsers.add_parser(
        "can.driver",
        help="Ensure a compatible gs_usb kernel driver is selected and loaded.",
    ).set_defaults(func=run)


def run(_args: object = None) -> None:
    """Ensure the gs_usb driver is available, building it when needed."""
    try:
        installed = ensure_driver()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if installed:
        print()
        print("gs_usb driver installed and loaded.")
        print(f"  It will load automatically on boot via {_MODULES_LOAD_FILE}.")
        print(
            "  Replug the Axol Hub (or it may already have enumerated) and "
            "run `axol can.setup`."
        )
    else:
        print("gs_usb driver already available — nothing to do.")
