"""
axol zed.driver

Upgrades a ZED Box's factory-flashed GMSL capture driver to the pinned
known-good release for its carrier board. Stereolabs ships one driver package
per carrier (``stereolabs-zedbox-duo`` for the ZED Box Duo, ``stereolabs-
zedbox-mini`` for the ZED Box Mini, ...), and units leave the factory with
whichever version was current when they were flashed. The ZED SDK is tightly
coupled to that driver: SDK 5.3+ needs driver >= 1.4.2, and Stereolabs pairs
SDK 5.4.1 with driver 1.4.3. Running a newer SDK on an older ``ZEDX_Daemon``
is not a benign mismatch -- the daemon restarts ``nvargus-daemon`` underneath
a live capture, which kills the video relay mid-session.

Gated hard on the target hardware: it only acts on a host where one of the
pinned ``stereolabs-zed*`` packages is already installed (i.e. a factory-
flashed ZED Box) *and* the running L4T release matches the one the pinned
.deb was built for -- so it is a quiet no-op on any other machine and can
never downgrade a newer driver (``dpkg --compare-versions`` guards that). A
``stereolabs-zed*`` package we have no pin for is reported loudly rather than
ignored, because "no pin" means the SDK/driver pairing on that box is
unmanaged.

The new driver is a kernel module + device-tree update, so it only takes
effect after a reboot. This command NEVER reboots the box itself -- it runs
from ``axol provision`` (over the operator's SSH session during install, and
from the running ``axol serve`` process after a self-update), where an
in-place reboot would drop the session or kill the robot mid-use. It prints a
reboot-required notice instead.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ...utils.state_files import (
    secure_atomic_copy_file,
    secure_ensure_directory,
    secure_unlink,
)
from ...utils.sudo import run_root
from .download import atomic_https_download


@dataclass(frozen=True)
class _Variant:
    """One carrier board's pinned GMSL driver package."""

    package: str
    carrier: str
    target_version: str
    deb_version: str
    url: str
    # Stereolabs does not publish a detached signature or authoritative
    # checksum for these fixed downloads.  Each value is the SHA-256 of the
    # artifact reviewed when its pin was added: it provides immutability from
    # that point forward, but is not independent proof of the first artifact's
    # provenance.  A mismatch must fail closed until a replacement has been
    # reviewed and the value updated.
    sha256: str

    @property
    def deb_name(self) -> str:
        return self.url.rsplit("/", 1)[-1]


_DRIVER_BASE_URL = "https://download.stereolabs.com/drivers/zedx"
# Every pin below is the 1.4.3 release Stereolabs pairs with ZED SDK 5.4.1,
# built for L4T 36.4 (JetPack 6.x). Bump a variant's four fields together.
_VARIANTS: tuple[_Variant, ...] = (
    _Variant(
        package="stereolabs-zedbox-duo",
        carrier="ZED Box Duo",
        target_version="1.4.3",
        deb_version="1.4.3-LI-MAX96712-ZEDBOX-L4T36.4.0",
        url=(
            f"{_DRIVER_BASE_URL}/1.4.3/R36.4/"
            "stereolabs-zedbox-duo_1.4.3-LI-MAX96712-ZEDBOX-L4T36.4.0_arm64.deb"
        ),
        sha256="54eb75f4f3d8dc5e562a0b3bd0d373b5bb1931f1994be3c4d346d535070e2c6b",
    ),
    _Variant(
        package="stereolabs-zedbox-mini",
        carrier="ZED Box Mini",
        target_version="1.4.3",
        deb_version="1.4.3-SL-MAX9296-ZEDBOX-MINI-L4T36.4.0",
        url=(
            f"{_DRIVER_BASE_URL}/1.4.3/R36.4/"
            "stereolabs-zedbox-mini_1.4.3-SL-MAX9296-ZEDBOX-MINI-L4T36.4.0_arm64.deb"
        ),
        sha256="5d6751be41375cd081766b5c5f4201d58e2d2b64a610d493cd5bb282e235bf09",
    ),
)
_VARIANTS_BY_PACKAGE = {variant.package: variant for variant in _VARIANTS}
_VARIANTS_BY_DEB_NAME = {variant.deb_name: variant for variant in _VARIANTS}
# dpkg glob for every Stereolabs GMSL driver package, pinned or not.
_MANAGED_PACKAGE_GLOB = "stereolabs-zed*"
_DEB_ARCHITECTURE = "arm64"
_DEB_MAX_BYTES = 16 * 1024 * 1024
# The pinned .debs are built against L4T 36.4.x (JetPack 6.x). Installing them
# on any other L4T would leave the cameras dead, so the upgrade is skipped
# (with a warning) when the running release differs — bump the variant table
# together with this when moving to a new L4T.
_L4T_RELEASE = "36"
_L4T_REVISION_MAJOR = "4"
_L4T_RELEASE_FILE = Path("/etc/nv_tegra_release")
# Persistent cache (like zed.install's wheel cache) so a failed install can be
# recovered by re-running — or by the manual `dpkg -i` the failure prints —
# without re-downloading.
_CACHE_DIR = Path.home() / ".almond" / "drivers"
_ROOT_CACHE_DIR = Path("/var/cache/almond-axol/drivers")


def _installed_driver_packages() -> dict[str, str]:
    """Installed ``stereolabs-zed*`` packages as ``{package: version}``.

    Only packages in dpkg's *installed* state count: a removed package whose
    configuration lingers (``rc``) still answers ``dpkg-query`` with its old
    version but no longer provides a driver, so it must not be treated as a
    factory install to upgrade.
    """
    try:
        proc = subprocess.run(
            [
                "dpkg-query",
                "-W",
                "-f",
                "${Package} ${db:Status-Status} ${Version}\\n",
                _MANAGED_PACKAGE_GLOB,
            ],
            capture_output=True,
            text=True,
        )
    except OSError:
        return {}
    if proc.returncode != 0:
        # "no packages found matching" — not a ZED Box (or a dpkg-less host).
        return {}
    installed: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] == "installed":
            installed[parts[0]] = parts[2]
    return installed


def _l4t_matches() -> bool:
    """True when the running L4T release matches the pinned .debs' target."""
    try:
        first_line = _L4T_RELEASE_FILE.read_text().splitlines()[0]
    except (OSError, IndexError):
        return False
    # e.g. "# R36 (release), REVISION: 4.0, GCID: ..."
    match = re.search(r"R(\d+)\s*\(release\),\s*REVISION:\s*(\d+)", first_line)
    if match is None:
        return False
    return match.group(1) == _L4T_RELEASE and match.group(2) == _L4T_REVISION_MAJOR


def _is_older(installed: str, target: str) -> bool:
    """True when ``installed`` is strictly older than ``target``."""
    return (
        subprocess.run(
            ["dpkg", "--compare-versions", installed, "lt", target],
            capture_output=True,
        ).returncode
        == 0
    )


def _deb_field(deb: Path, field: str) -> str:
    proc = subprocess.run(
        ["dpkg-deb", "--field", str(deb), field],
        capture_output=True,
        text=True,
    )
    value = proc.stdout.strip()
    if proc.returncode != 0 or not value:
        detail = (proc.stderr or "").strip()
        raise RuntimeError(f"downloaded .deb has no valid {field}: {detail}")
    return value


def _validate_deb(deb: Path, variant: _Variant) -> None:
    """Verify the pinned bytes and the package identity they encode."""
    if deb.is_symlink() or not deb.is_file():
        raise RuntimeError("downloaded .deb is not a regular file")

    with deb.open("rb") as artifact:
        digest = hashlib.file_digest(artifact, "sha256").hexdigest()
    if digest != variant.sha256:
        raise RuntimeError(
            f"{variant.carrier} driver SHA-256 does not match the reviewed "
            "artifact; refusing to install changed vendor bytes. Review the "
            f"replacement and update the {variant.package} pin in "
            "almond_axol/cli/zed/driver.py"
        )

    expected_fields = {
        "Package": variant.package,
        "Version": variant.deb_version,
        "Architecture": _DEB_ARCHITECTURE,
    }
    for field, expected in expected_fields.items():
        actual = _deb_field(deb, field)
        if actual != expected:
            raise RuntimeError(
                f"downloaded .deb {field} is {actual!r}, expected {expected!r}"
            )


def _variant_for_artifact(deb: Path) -> _Variant:
    """The pinned variant a cached artifact name belongs to (fail closed)."""
    variant = _VARIANTS_BY_DEB_NAME.get(deb.name)
    if variant is None:
        raise RuntimeError(f"unexpected ZED driver artifact name: {deb.name!r}")
    return variant


def _download_deb(variant: _Variant) -> Path:
    """Download the pinned .deb into the cache and verify it is a valid archive.

    Everything that can fail without root — the download and the archive check
    — happens here, *before* the factory package is removed, so a network error
    or truncated download can never leave the box with no driver installed.
    The cached copy also makes a re-run (or the printed manual recovery
    command) work offline.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    deb = _CACHE_DIR / variant.deb_name
    if not deb.exists():
        print(f"Downloading {variant.url}")
        atomic_https_download(
            variant.url,
            deb,
            max_bytes=_DEB_MAX_BYTES,
            validate=lambda path: _validate_deb(path, variant),
        )
    else:
        print(f"Already downloaded: {deb}")
    try:
        _validate_deb(deb, variant)
    except Exception:
        # Never retain unreviewed bytes under the trusted cache name.  A
        # changed upstream artifact will fail the same pin on the next run and
        # therefore cannot be silently accepted by retrying.
        deb.unlink(missing_ok=True)
        raise
    return deb


def _stage_deb_as_root(deb: Path) -> Path:
    """No-follow copy into a private root cache and validate the pinned fd bytes."""
    if os.geteuid() != 0:
        raise PermissionError("reviewed ZED driver staging must run as root")
    # The artifact name selects the pin it is validated against, so only a
    # reviewed artifact name can ever be staged at the trusted path.
    variant = _variant_for_artifact(deb)

    staged = _ROOT_CACHE_DIR / variant.deb_name
    secure_ensure_directory(_ROOT_CACHE_DIR, mode=0o700)
    try:
        # The source is opened descriptor-relative with O_NOFOLLOW and copied
        # from that pinned descriptor. A symlink swap cannot disclose a root-
        # readable file into the cache, and an in-place mutation makes the
        # post-copy reviewed digest fail.
        secure_atomic_copy_file(deb, staged, mode=0o600)
        _validate_deb(staged, variant)
    except Exception:
        # Never leave mismatched or partially trusted bytes readable at the
        # predictable recovery path.
        try:
            secure_unlink(staged, missing_ok=True)
        except OSError:
            pass
        raise
    return staged


def _stage_deb_for_root(deb: Path) -> Path:
    """Copy into a root-owned cache, then validate that immutable copy.

    Interactive invocations may download into an operator-owned home.  Root
    must not consume that path directly: the operator could swap it between
    validation and ``dpkg -i``. Interactive use invokes this module's narrow
    root helper; the hosted root service calls the same implementation
    directly. The fixed cache is root-only and validation happens after copy.
    """
    staged = _ROOT_CACHE_DIR / deb.name
    if os.geteuid() == 0:
        return _stage_deb_as_root(deb)
    run_root(
        [
            sys.executable,
            "-m",
            "almond_axol.cli.zed.driver",
            "--stage-reviewed-deb",
            str(deb),
        ],
        check=True,
    )
    return staged


def _upgrade(variant: _Variant) -> None:
    """Replace the installed factory package with the pinned .deb (needs root)."""
    deb = _stage_deb_for_root(_download_deb(variant))
    # Remove the factory package first (per Stereolabs' upgrade procedure)
    # rather than upgrading in place; best-effort since a half-removed
    # package still gets replaced by the install below.
    print(f"Removing the factory {variant.package} package (requires sudo)...")
    run_root(["dpkg", "-r", variant.package])
    print(f"Installing {deb.name}...")
    try:
        run_root(["dpkg", "-i", str(deb)], check=True)
    except RuntimeError:
        # The factory package is already removed at this point, so don't fail
        # silently: tell the operator exactly how to finish the install by
        # hand (the verified .deb stays in the cache).
        print(
            f"ERROR: installing {deb.name} failed after the factory package "
            f"was removed — the box has NO camera driver until this is fixed. "
            f"Recover with: sudo dpkg -i {deb}",
            file=sys.stderr,
        )
        raise


def _ensure_variant(variant: _Variant, installed: str) -> bool:
    """Upgrade one installed variant when it is older than its pin."""
    if not _is_older(installed, variant.target_version):
        print(f"{variant.package} {installed} already >= {variant.target_version}.")
        return False
    if not _l4t_matches():
        print(
            f"WARNING: {variant.package} {installed} is outdated, but the pinned "
            f"{variant.target_version} driver targets L4T {_L4T_RELEASE}."
            f"{_L4T_REVISION_MAJOR} and this host runs a different release — "
            "skipping. Update the pin in almond_axol/cli/zed/driver.py.",
            file=sys.stderr,
        )
        return False
    print(
        f"{variant.package} {installed} is an outdated {variant.carrier} "
        f"camera driver — upgrading to {variant.target_version}."
    )
    _upgrade(variant)
    print()
    print(
        f"REBOOT REQUIRED: {variant.package} {variant.target_version} is "
        "installed but the new kernel driver only loads at boot. Reboot when "
        "convenient (sudo reboot)."
    )
    return True


def ensure_driver() -> bool:
    """Upgrade the ZED Box camera driver when the installed one is outdated.

    Covers every pinned carrier variant (ZED Box Duo, ZED Box Mini). Returns
    True when a driver was upgraded (a reboot is then required for it to
    load), False when there was nothing to do. Idempotent and self-gating (a
    no-op on anything that isn't a ZED Box on the pinned L4T), so it is safe
    to run from ``axol provision`` on every host. An installed Stereolabs
    driver package without a pin is reported on stderr: that box's SDK/driver
    pairing is unmanaged, which is exactly the state that lets a newer SDK
    fight an older ``ZEDX_Daemon``.
    """
    installed = _installed_driver_packages()
    if not installed:
        # Not a factory-flashed ZED Box (or dpkg-less host) — nothing to do.
        return False
    upgraded = False
    for package, version in sorted(installed.items()):
        variant = _VARIANTS_BY_PACKAGE.get(package)
        if variant is None:
            print(
                f"WARNING: {package} {version} is installed but axol has no "
                "pinned driver for this ZED carrier, so it cannot verify the "
                "driver matches the ZED SDK. Add a pin in "
                "almond_axol/cli/zed/driver.py or upgrade it by hand from "
                "https://www.stereolabs.com/developers/drivers",
                file=sys.stderr,
            )
            continue
        if _ensure_variant(variant, version):
            upgraded = True
    return upgraded


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``zed.driver`` subcommand."""
    subparsers.add_parser(
        "zed.driver",
        help=(
            "Upgrade the ZED Box camera driver (stereolabs-zedbox-duo / "
            "stereolabs-zedbox-mini) to the pinned release."
        ),
    ).set_defaults(func=run)


def run(_args: object = None) -> None:
    """Ensure the pinned ZED Box camera driver is installed."""
    try:
        upgraded = ensure_driver()
    except Exception as exc:  # noqa: BLE001 - network/dpkg failures land here
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    if not upgraded and not _installed_driver_packages():
        print(
            "No stereolabs-zed* driver package is installed — not a "
            "factory-flashed ZED Box; nothing to do."
        )


def _module_main() -> None:
    """Entry point for the narrow interactive-sudo staging helper."""
    if len(sys.argv) == 3 and sys.argv[1] == "--stage-reviewed-deb":
        try:
            _stage_deb_as_root(Path(sys.argv[2]))
        except Exception as exc:  # noqa: BLE001 - concise sudo helper failure
            print(f"ERROR: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        return
    raise SystemExit("this module entry point only supports --stage-reviewed-deb PATH")


if __name__ == "__main__":
    _module_main()
