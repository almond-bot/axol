"""
axol rt.install

Build and install the ``axol-rt`` realtime core binary (see
``rust/axol-rt/``), the required hardware control backend. Runs as a step of
``axol provision``, and standalone:

    axol rt.install

Three concerns, each idempotent and self-gating:

* **Toolchain** — if no working ``cargo`` is available (PATH or
  ``~/.cargo/bin``), install a minimal stable Rust toolchain via rustup (no
  sudo; lands in ``~/.cargo``). Needs a C linker (``cc``) for the final link
  — present on any Jetson with build-essential.
* **Sources** — a dev checkout builds in-repo (``rust/axol-rt`` next to this
  package). Tool installs have no sources on disk, so the crate is fetched
  into ``~/.almond/axol-rt-src`` at the exact ref matching the installed
  package: the PEP 610 git commit for ``uv tool install git+...``, or the
  ``v<version>`` release tag for PyPI installs. A tool install made from a
  local checkout (``uv tool install /path/to/axol``) builds that checkout's
  crate. ``AXOL_RT_SOURCE`` (a checkout path, or ``<git url>@<ref>``)
  overrides all of the above.
* **Binary** — dev checkouts leave it in ``target/release/`` (where
  :func:`almond_axol.rt.link.find_binary` already looks, and where a manual
  ``cargo build`` stays authoritative). Tool installs copy it into
  ``UV_TOOL_BIN_DIR`` when set, or ``~/.local/bin`` otherwise, and stamp the
  built ref per destination so a re-run of ``axol provision`` at the same
  version skips the build.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

_logger = logging.getLogger(__name__)

_PACKAGE = "almond-axol"
# Explicit source override for tool installs whose origin can't be inferred
# (or is wrong): a path to an axol checkout / the crate, or ``<git url>@<ref>``.
_SOURCE_ENV = "AXOL_RT_SOURCE"
# Keep in sync with almond_axol.serve.update._REPO_URL (not imported: pulling
# in the serve package would drag the whole FastAPI stack into provision).
_REPO_URL = "https://github.com/almond-bot/axol"

_SRC_CACHE = Path.home() / ".almond" / "axol-rt-src"
# Keep the legacy stamp for the default destination so existing provision
# wrappers can still verify it. Non-default destinations receive their own
# stamp below, so switching UV_TOOL_BIN_DIR can never bless a stale binary
# left elsewhere.
_DEFAULT_INSTALL_DIR = Path.home() / ".local" / "bin"
_STAMP = _SRC_CACHE / ".installed-ref"

# A cold rustup install + first build both finish well inside this on a
# Jetson; a hung tool or network fetch should not wedge provision forever.
_CARGO_PROBE_TIMEOUT = 10.0
_RUSTUP_TIMEOUT = 600.0
_BUILD_TIMEOUT = 900.0
_FETCH_TIMEOUT = 120.0


def _install_dir() -> Path:
    """Destination shared with uv's tool launchers, or the user-local default."""
    configured = os.environ.get("UV_TOOL_BIN_DIR")
    if configured:
        return Path(configured)
    return _DEFAULT_INSTALL_DIR


def _stamp_for(dest: Path) -> Path:
    """Ref stamp bound to ``dest`` (legacy name for the default destination)."""
    if dest.parent == _DEFAULT_INSTALL_DIR:
        return _STAMP
    digest = hashlib.sha256(os.fsencode(str(dest.absolute()))).hexdigest()[:16]
    return _SRC_CACHE / f".installed-ref-{digest}"


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``rt.install`` subcommand."""
    subparsers.add_parser(
        "rt.install",
        help=(
            "Build + install the axol-rt realtime core binary "
            "(Rust toolchain via rustup if needed)."
        ),
    ).set_defaults(func=run)


def _cargo_works(cargo: str) -> bool:
    """Whether a candidate can select a toolchain and run Cargo."""
    try:
        proc = subprocess.run(
            [cargo, "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_CARGO_PROBE_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def _find_cargo() -> str | None:
    """A working home-local Cargo, or a working ``cargo`` on PATH."""
    default = Path.home() / ".cargo" / "bin" / "cargo"
    # An elevated caller can retain /home/<user>/.cargo/bin on PATH while
    # changing HOME to /root. That rustup shim exists but cannot select a
    # toolchain from /root/.rustup, so existence alone is not enough.
    candidates = [str(default), shutil.which("cargo")]
    seen: set[str] = set()
    for candidate in candidates:
        if candidate is None or candidate in seen:
            continue
        seen.add(candidate)
        if _cargo_works(candidate):
            return candidate
    return None


def _ensure_toolchain() -> str:
    """Return a cargo path, installing a minimal stable rustup if needed."""
    cargo = _find_cargo()
    if cargo:
        return cargo
    if shutil.which("curl") is None:
        raise RuntimeError("no cargo and no curl to bootstrap rustup with")
    print("Installing the Rust toolchain (rustup, minimal profile) ...")
    subprocess.run(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
        "sh -s -- -y --profile minimal --default-toolchain stable",
        shell=True,
        check=True,
        timeout=_RUSTUP_TIMEOUT,
    )
    cargo = _find_cargo()
    if cargo is None:
        raise RuntimeError("rustup finished but no working cargo appeared")
    return cargo


def _repo_crate() -> Path | None:
    """The in-repo crate for dev checkouts; ``None`` for tool installs."""
    crate = Path(__file__).resolve().parents[2] / "rust" / "axol-rt"
    return crate if (crate / "Cargo.toml").exists() else None


def _local_crate(root: Path) -> Path | None:
    """The crate under a checkout ``root`` (or ``root`` itself, if it is the crate)."""
    for crate in (root / "rust" / "axol-rt", root):
        if (crate / "Cargo.toml").exists():
            return crate
    return None


def _source_override() -> Path | tuple[str, str] | None:
    """Parse ``AXOL_RT_SOURCE``: a local checkout/crate path, or ``<git url>@<ref>``."""
    raw = os.environ.get(_SOURCE_ENV, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.exists():
        crate = _local_crate(path)
        if crate is None:
            raise RuntimeError(
                f"{_SOURCE_ENV}={raw!r} has no rust/axol-rt crate "
                "(point it at a checkout of the axol repo or at the crate itself)"
            )
        return crate
    url, sep, ref = raw.rpartition("@")
    if not sep or not url or not ref or "/" in ref:
        raise RuntimeError(
            f"{_SOURCE_ENV}={raw!r} is neither an existing path nor "
            "'<git url>@<branch|tag|commit>'"
        )
    return url, ref


def _source_ref() -> Path | tuple[str, str] | None:
    """Where the axol-rt sources for this (tool) install come from.

    Returns a local crate ``Path`` to build in place, ``(git url, ref)`` to
    fetch and build, or ``None`` when nothing can be inferred.

    * :data:`_SOURCE_ENV` wins when set (see :func:`_source_override`) —
      the escape hatch for anything below that guesses wrong.
    * Git installs (``uv tool install git+...@<branch>``) record their
      repository URL and pinned commit in PEP 610 ``direct_url.json``.
    * Local-path installs (``uv tool install /path/to/checkout``) record the
      directory instead; its in-tree crate is used while it is still there.
    * Index (PyPI) installs carry no ``direct_url.json`` at all, so the
      ``v<version>`` release tag is the ref that produced the published
      artifact (the release workflow tags and publishes together).
    * Wheel/sdist file installs can't be traced back to sources.
    """
    override = _source_override()
    if override is not None:
        return override
    try:
        dist = distribution(_PACKAGE)
    except PackageNotFoundError:
        return None
    raw = dist.read_text("direct_url.json")
    if not raw:
        return _REPO_URL, f"v{version(_PACKAGE)}"
    try:
        data = json.loads(raw)
    except ValueError:
        data = {}
    url = data.get("url") or ""
    if url.startswith("git+"):
        url = url[len("git+") :]
    commit = (data.get("vcs_info") or {}).get("commit_id")
    if commit and url:
        return url, commit
    if "dir_info" in data and url.startswith("file:"):
        # A non-editable install from a checkout: the package lives in the
        # tool env (so _repo_crate misses), but the checkout it came from
        # still carries the crate. Editable installs never get here.
        root = Path(url2pathname(urlparse(url).path))
        crate = _local_crate(root)
        if crate is not None:
            return crate
        _logger.warning(
            "installed from %s, but no rust/axol-rt crate is there any more", root
        )
    return None


def _no_source_error() -> RuntimeError:
    return RuntimeError(
        "cannot resolve the axol-rt source ref for this install: it was not "
        "installed from git, a PyPI release, or a checkout that still exists. "
        f"Set {_SOURCE_ENV} to a checkout of the axol repo or to "
        f"'<git url>@<branch|tag|commit>' (e.g. {_REPO_URL}@main) and re-run."
    )


def _fetch_crate(url: str, ref: str) -> tuple[Path, str]:
    """Shallow-fetch ``ref`` from ``url`` into the source cache.

    ``git fetch --depth 1 <url> <ref>`` accepts branches, tags and bare commit
    ids (GitHub serves arbitrary reachable commits), and the cached repository
    makes re-provisioning at a new ref an incremental fetch. Returns the crate
    path and the commit that ``ref`` resolved to.
    """
    _SRC_CACHE.mkdir(parents=True, exist_ok=True)
    if not (_SRC_CACHE / ".git").exists():
        subprocess.run(
            ["git", "init", "--quiet", str(_SRC_CACHE)],
            check=True,
            timeout=_FETCH_TIMEOUT,
        )
    git = ["git", "-C", str(_SRC_CACHE)]
    try:
        subprocess.run(
            [*git, "fetch", "--quiet", "--depth", "1", url, ref],
            check=True,
            timeout=_FETCH_TIMEOUT,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"could not fetch {ref} from {url} (does that branch/tag/commit "
            f"exist there?). Set {_SOURCE_ENV} to the sources to build from "
            "and re-run."
        ) from exc
    subprocess.run(
        [*git, "checkout", "--quiet", "--force", "FETCH_HEAD"],
        check=True,
        timeout=_FETCH_TIMEOUT,
    )
    commit = subprocess.run(
        [*git, "rev-parse", "FETCH_HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=_FETCH_TIMEOUT,
    ).stdout.strip()
    crate = _SRC_CACHE / "rust" / "axol-rt"
    if not (crate / "Cargo.toml").exists():
        raise RuntimeError(
            f"{url}@{ref} has no rust/axol-rt crate (a release that predates "
            f"the realtime core?). Set {_SOURCE_ENV} to a ref that has it, "
            f"e.g. {_REPO_URL}@main, and re-run."
        )
    return crate, commit


def _build(crate: Path, cargo: str) -> Path:
    """``cargo build --release``; returns the built binary path."""
    print(f"Building axol-rt in {crate} ...")
    env = dict(os.environ)
    # rustup's shims live next to cargo; make sure the build sees them even
    # when ~/.cargo/bin isn't on PATH yet (fresh rustup, same shell).
    env["PATH"] = f"{Path(cargo).parent}{os.pathsep}{env.get('PATH', '')}"
    subprocess.run(
        [cargo, "build", "--release"],
        cwd=crate,
        env=env,
        check=True,
        timeout=_BUILD_TIMEOUT,
    )
    binary = crate / "target" / "release" / "axol-rt"
    if not binary.exists():
        raise RuntimeError("build succeeded but the binary is missing")
    return binary


def _install(binary: Path, ref: str, dest: Path) -> Path:
    """Copy the binary onto PATH (atomic replace) and stamp the ref."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    shutil.copy2(binary, tmp)
    tmp.chmod(0o755)
    tmp.replace(dest)
    stamp = _stamp_for(dest)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(ref + "\n")
    return dest


def _grant_realtime(binary: Path) -> None:
    """Give only the Rust core permission to enter its bounded RT policy.

    A dedicated CPU is not sufficient under Linux's normal scheduler: a
    SCHED_OTHER thread can still be suspended beyond the entire 4.17 ms motor
    period. File-scoped ``CAP_SYS_NICE`` lets ``axol-rt`` raise its own two CAN
    threads without running the Python/UI/camera process as root, and
    ``CAP_IPC_LOCK`` lets it ``mlockall`` its memory past the unprivileged
    ``RLIMIT_MEMLOCK`` (8 MiB on most distros — less than its thread stacks)
    so a page reclaimed under the recorder's I/O pressure can never fault a
    CAN thread mid-tick. Without the second capability the core still runs,
    unlocked, and says so.
    """
    if sys.platform != "linux":
        return
    setcap = shutil.which("setcap")
    if setcap is None and Path("/usr/sbin/setcap").exists():
        setcap = "/usr/sbin/setcap"
    if setcap is None:
        raise RuntimeError("setcap is required to grant axol-rt real-time scheduling")

    cmd = [setcap, "cap_sys_nice,cap_ipc_lock=ep", str(binary)]
    if os.geteuid() == 0:
        subprocess.run(cmd, check=True)
    elif sys.stdin is not None and sys.stdin.isatty():
        # Interactive development install: ask once through the shared sudo
        # helper. Headless provisioning is normally the root system service;
        # if it is not, fail loudly below instead of hanging for a password.
        from ..utils.sudo import prime_sudo, run_root

        if not prime_sudo():
            raise RuntimeError("sudo is required to grant CAP_SYS_NICE to axol-rt")
        run_root(cmd, check=True)
    else:
        raise RuntimeError(
            "axol-rt needs CAP_SYS_NICE; run `axol rt.install` interactively"
        )
    print(f"axol-rt realtime scheduling enabled: {binary}")


def run(_args: object = None) -> None:
    """Build + install axol-rt; idempotent, prints what it did."""
    crate = _repo_crate()
    if crate is not None:
        # Dev checkout: build in place — find_binary's repo fallback picks
        # it up, and a manual `cargo build` there stays the source of truth
        # (no PATH copy that could shadow a newer in-repo build).
        binary = _build(crate, _ensure_toolchain())
        _grant_realtime(binary)
        print(f"axol-rt built: {binary}")
        return

    source = _source_ref()
    if source is None:
        raise _no_source_error()
    dest = _install_dir() / "axol-rt"
    if isinstance(source, Path):
        # Sources on disk but the package isn't (tool install from a checkout,
        # or AXOL_RT_SOURCE=<path>): build there, install onto PATH. Always
        # rebuild — a working tree has no ref to stamp, and cargo is
        # incremental anyway.
        binary = _build(source, _ensure_toolchain())
        dest = _install(binary, f"path:{source}", dest)
        _grant_realtime(dest)
        print(f"axol-rt installed: {dest} (built from {source})")
        return

    url, ref = source
    stamp = _stamp_for(dest)
    # Only an inferred (pinned commit / release tag) ref can be trusted to
    # name the same sources as last time; an explicit override may be a
    # branch, so it always rebuilds.
    if (
        _SOURCE_ENV not in os.environ
        and dest.is_file()
        and os.access(dest, os.X_OK)
        and stamp.exists()
        and stamp.read_text().strip() == ref
    ):
        _grant_realtime(dest)
        print(f"axol-rt already installed at {dest} (ref {ref}).")
        return
    cargo = _ensure_toolchain()
    print(f"Fetching axol-rt sources ({url} @ {ref}) ...")
    crate, commit = _fetch_crate(url, ref)
    binary = _build(crate, cargo)
    dest = _install(binary, ref, dest)
    _grant_realtime(dest)
    print(f"axol-rt installed: {dest} (ref {ref}, commit {commit[:12]})")
