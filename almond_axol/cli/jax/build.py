"""
axol jax.build

Build ``jaxlib`` + the JAX CUDA plugin from source for *this device's* GPU and
install them into the axol environment.

The prebuilt PyPI ``jax[cudaNN]`` wheels only ship GPU kernels for datacenter /
desktop GPUs, so on a Jetson (integrated Orin GPU, compute capability ``sm_87``)
every kernel launch fails with ``cudaErrorNoKernelImageForDevice: no kernel
image is available for execution on the device``. The only supported fix is a
source build of ``jaxlib`` / ``jax-cuda-plugin`` / ``jax-cuda-pjrt`` with
``--cuda_compute_capabilities`` set to the device's real capability (see JAX
issues #22723 / #32166).

This command autodetects the compute capability (Tegra SoC → ``sm_XX``, or
``nvidia-smi`` for discrete GPUs) and the CUDA *major* the driver supports, then
builds against the exact CUDA + cuDNN that JAX itself pins for that major (read
from the source ``.bazelrc``; the hermetic build downloads them). Building
against the device's older local toolkit fails — newer JAX uses cuSolver APIs
absent from older CUDA. It checks out the JAX source at the tag matching the
installed ``jax`` version, finds a usable clang, runs ``build/build.py``, and
reinstalls the three built wheels into the interpreter running this CLI.

To avoid rebuilding on every board, prebuilt wheels are published as GitHub
release assets on the axol repo under a tag that encodes the JAX version, CUDA
major, and compute capability (e.g. ``jax-wheels-v0.9.2-cuda12-sm87``). This
command **downloads that release first** and only builds from source when no
matching prebuilt wheels exist. A maintainer builds once and uploads with
``axol jax.build --publish`` (needs an authenticated ``gh``); thereafter every
other Jetson of the same class just downloads.

It is idempotent: if GPU JAX already runs a kernel on the device it no-ops, so
``axol provision`` (which calls it on Tegra via :mod:`.install`) only fetches /
builds once and then skips on subsequent upgrades.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import urllib.error
import urllib.request
import zipfile
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

from ...utils.sudo import prime_sudo, run_root

_logger = logging.getLogger(__name__)

_CUDA_HOME = Path("/usr/local/cuda")
# Where the JAX source is cloned + built (reused across runs to avoid re-cloning).
_SRC_DIR = Path.home() / ".almond" / "jax-src"
# Where downloaded prebuilt wheels are cached (shared with zed.install's cache).
_WHEEL_CACHE = Path.home() / ".almond" / "wheels"

# Prebuilt wheels are published as GitHub release assets on the axol repo, so a
# Jetson can download them instead of doing the ~hour-long source build. The
# release tag encodes everything the wheel filename doesn't — JAX version, CUDA
# major, and compute capability — so different boards/versions never collide
# (the Python ABI + arch stay in the wheel filename). Maintainers populate a tag
# once with ``axol jax.build --publish``; every other board downloads it.
_WHEEL_REPO = "almond-bot/axol"

# Tegra SoC family → CUDA compute capability. The integrated GPU's capability is
# fixed per SoC, and nvidia-smi can't report it on Jetson, so map from the SoC
# id in /proc/device-tree/compatible (e.g. "nvidia,tegra234").
_TEGRA_SM = {
    "tegra234": "sm_87",  # Orin (AGX Orin, Orin NX, Orin Nano)
    "tegra194": "sm_72",  # Xavier (AGX Xavier, Xavier NX)
    "tegra186": "sm_62",  # TX2
    "tegra210": "sm_53",  # Nano / TX1
}

# Minimum clang major JAX's build needs; Jetson's default gcc can't compile
# jaxlib cleanly, and older clang chokes too.
_MIN_CLANG_MAJOR = 16
# Version to install via apt.llvm.org when no usable clang is present.
_LLVM_VERSION = 18

_APT_BUILD_DEPS = ("libstdc++-12-dev",)


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``jax.build`` subcommand."""
    parser = subparsers.add_parser(
        "jax.build",
        help=(
            "Build jaxlib + the JAX CUDA plugin from source for this device's "
            "GPU (e.g. a Jetson) and install them into the axol environment."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when GPU JAX already runs on the device.",
    )
    parser.add_argument(
        "--compute",
        default=None,
        metavar="sm_XX",
        help="Override the CUDA compute capability (e.g. sm_87 for Orin).",
    )
    parser.add_argument(
        "--jax-version",
        default=None,
        metavar="X.Y.Z",
        help="JAX source tag to build (default: the installed jax version).",
    )
    parser.add_argument(
        "--cuda-version",
        default=None,
        metavar="X.Y.Z",
        help="CUDA version to build against (default: JAX's pinned version).",
    )
    parser.add_argument(
        "--cudnn-version",
        default=None,
        metavar="X.Y.Z",
        help="cuDNN version to build against (default: JAX's pinned version).",
    )
    parser.add_argument(
        "--clang-path",
        default=None,
        help="Path to clang (default: autodetected / installed via apt.llvm.org).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Bazel --jobs (lower this if the build runs out of memory).",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip the prebuilt-wheel download and build from source.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help=(
            "After building, upload the wheels to the GitHub release for this "
            "device class (needs an authenticated `gh`)."
        ),
    )
    parser.set_defaults(func=run)


# --------------------------------------------------------------------------- #
# Detection helpers
# --------------------------------------------------------------------------- #


def _tegra_soc() -> str | None:
    """Tegra SoC id (e.g. ``tegra234``) from the device tree, or ``None``."""
    compatible = Path("/proc/device-tree/compatible")
    if not compatible.exists():
        return None
    try:
        # NUL-separated list of "vendor,model" strings.
        text = compatible.read_bytes().decode("ascii", "replace")
    except OSError:
        return None
    match = re.search(r"nvidia,(tegra\d+)", text)
    return match.group(1) if match else None


def is_tegra() -> bool:
    """True on an NVIDIA Jetson / Tegra board (integrated GPU)."""
    return _tegra_soc() is not None


def _compute_capability(override: str | None) -> str | None:
    """CUDA compute capability as ``sm_XX`` for this device."""
    if override:
        return override
    soc = _tegra_soc()
    if soc is not None:
        sm = _TEGRA_SM.get(soc)
        if sm is not None:
            return sm
        # Unmapped / newer Tegra: fall through to nvidia-smi below rather than
        # giving up (some boards do report compute_cap); else run() errors with
        # a clear "pass --compute" message.
    # Discrete GPU (or unmapped Tegra): nvidia-smi reports e.g. "8.7".
    smi = shutil.which("nvidia-smi")
    if smi is not None:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
        except Exception as exc:  # noqa: BLE001 - no GPU / query unsupported
            _logger.info("nvidia-smi compute_cap query failed: %s", exc)
            out = ""
        match = re.search(r"(\d+)\.(\d+)", out)
        if match:
            return f"sm_{match.group(1)}{match.group(2)}"
    return None


def _cuda_version(override: str | None) -> str | None:
    """CUDA toolkit version ``X.Y.Z`` (from ``/usr/local/cuda/version.json``)."""
    if override:
        return override
    version_json = _CUDA_HOME / "version.json"
    if version_json.exists():
        try:
            data = json.loads(version_json.read_text())
            return str(data["cuda"]["version"])
        except (ValueError, KeyError, OSError) as exc:
            _logger.info("could not parse %s: %s", version_json, exc)
    return None


def _cuda_major(override: str | None) -> str | None:
    """Major CUDA version this device targets (for the release tag / plugin name).

    This is *not* the version the wheels are built against — the hermetic build
    downloads its own CUDA (see :func:`_jax_bazelrc_cuda`). It only selects the
    major (12 vs 13), which is driver-bound: e.g. a Jetson whose driver maxes at
    CUDA 12.6 must stay on ``cuda12``, never ``cuda13``.
    """
    if override:
        return override.split(".")[0]
    detected = _cuda_version(None)
    if detected:
        return detected.split(".")[0]
    smi = shutil.which("nvidia-smi")
    if smi is not None:
        try:
            out = subprocess.run(
                [smi], capture_output=True, text=True, timeout=30
            ).stdout
        except Exception:  # noqa: BLE001 - no GPU / driver mismatch
            out = ""
        match = re.search(r"CUDA Version:\s*(\d+)\.", out)
        if match:
            return match.group(1)
    return None


def _jax_bazelrc_cuda(src: Path, cuda_major: str) -> tuple[str, str] | None:
    """JAX's own recommended ``(cuda, cudnn)`` for a CUDA major, from .bazelrc.

    JAX's ``.bazelrc`` pins the exact hermetic CUDA + cuDNN it's designed to
    build against per major (e.g. ``cuda_v12`` → CUDA 12.9.1 / cuDNN 9.8.0 for
    0.9.2). Building against the device's older local toolkit instead fails —
    newer JAX uses cuSolver APIs (e.g. ``cusolverDnXgeev``) absent from older
    CUDA. Reading it from source keeps the pairing correct across JAX versions.
    """
    bazelrc = src / ".bazelrc"
    if not bazelrc.exists():
        return None
    text = bazelrc.read_text()
    cuda = re.search(
        rf'cuda_v{cuda_major}\s+--repo_env=HERMETIC_CUDA_VERSION="([^"]+)"', text
    )
    cudnn = re.search(
        rf'cuda_v{cuda_major}\s+--repo_env=HERMETIC_CUDNN_VERSION="([^"]+)"', text
    )
    if cuda and cudnn:
        return cuda.group(1), cudnn.group(1)
    return None


def _installed_jax_version() -> str | None:
    """Installed pure-Python ``jax`` version (the build tag to match)."""
    try:
        return _pkg_version("jax")
    except PackageNotFoundError:
        return None


def _find_clang(override: str | None) -> str | None:
    """Path to a clang new enough for the JAX build, or ``None``."""
    candidates = [override] if override else []
    candidates += [
        f"/usr/lib/llvm-{_LLVM_VERSION}/bin/clang",
        f"clang-{_LLVM_VERSION}",
        "clang-17",
        "clang-16",
        "clang",
    ]
    for cand in candidates:
        if cand is None:
            continue
        path = shutil.which(cand) or (cand if Path(cand).exists() else None)
        if path is None:
            continue
        try:
            out = subprocess.run(
                [path, "--version"], capture_output=True, text=True, timeout=15
            ).stdout
        except Exception:  # noqa: BLE001 - not executable
            continue
        match = re.search(r"clang version (\d+)", out)
        if match and int(match.group(1)) >= _MIN_CLANG_MAJOR:
            return path
    return None


def _cuda_lib_dir() -> str:
    """Directory holding the system CUDA shared libs (for LD_LIBRARY_PATH)."""
    return str(_CUDA_HOME / "lib64")


def _build_env() -> dict[str, str]:
    """Environment for the build/runtime with the system CUDA libs on the path."""
    env = dict(os.environ)
    lib = _cuda_lib_dir()
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{lib}:{existing}" if existing else lib
    return env


def _gpu_jax_ok() -> bool:
    """True when the installed JAX runs a real kernel on the GPU (idempotency).

    Checked in a subprocess so it reflects the currently-installed wheels and a
    clean backend init. A broken PyPI wheel (no kernel image for this device)
    fails here, which is what triggers the build. Runs with the *plain* env
    (no injected ``LD_LIBRARY_PATH``) so it mirrors how ``axol`` really launches
    JAX — the CUDA libs are found via the plugin's patched RPATH, not env vars.
    """
    code = (
        "import jax, jax.numpy as jnp;"
        "assert jax.default_backend() == 'gpu', jax.default_backend();"
        "jnp.any(jnp.arange(4)).block_until_ready();"
        "print('ok')"
    )
    try:
        return (
            subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                timeout=180,
            ).returncode
            == 0
        )
    except Exception:  # noqa: BLE001 - interpreter/backend failure
        return False


# --------------------------------------------------------------------------- #
# Build steps
# --------------------------------------------------------------------------- #


def _libstdcxx_ok() -> bool:
    """True when the libstdc++ dev headers clang needs are already installed."""
    return any(Path(f"/usr/include/c++/{v}").exists() for v in (12, 13, 14))


def _ensure_build_deps(clang_override: str | None) -> str:
    """Ensure clang + libstdc++ are present; return the clang path to use.

    Only touches apt (and thus sudo) when something is actually missing, so a
    re-run on an already-provisioned box needs no root and never blocks on a
    password prompt (important for a detached/background build).
    """
    clang = _find_clang(clang_override)
    if clang is not None and _libstdcxx_ok():
        return clang

    apt = shutil.which("apt-get")
    if apt is not None and not prime_sudo():
        _logger.info("apt available but no root; skipping build-dep install")
        apt = None
    if apt is not None:
        run_root([apt, "install", "-y", *_APT_BUILD_DEPS])
        if clang is None:
            _install_clang_via_llvm(apt)
            clang = _find_clang(clang_override)
    if clang is None:
        raise RuntimeError(
            f"clang >= {_MIN_CLANG_MAJOR} is required to build jaxlib and none "
            "was found. Install it (e.g. via https://apt.llvm.org/llvm.sh) and "
            "re-run with --clang-path, or install clang-"
            f"{_LLVM_VERSION} manually."
        )
    return clang


def _install_clang_via_llvm(apt: str) -> None:
    """Install clang from apt.llvm.org (the Jetson's jammy clang is too old)."""
    script = subprocess.run(
        ["curl", "-fsSL", "https://apt.llvm.org/llvm.sh"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if script.returncode != 0:
        _logger.warning("could not fetch llvm.sh; install clang manually")
        return
    tmp = _SRC_DIR.parent / "llvm.sh"
    _SRC_DIR.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(script.stdout)
    run_root(["bash", str(tmp), str(_LLVM_VERSION)])


def _checkout_source(jax_version: str) -> Path:
    """Clone (or update) the JAX source at the ``jax-vX.Y.Z`` tag."""
    tag = f"jax-v{jax_version}"
    if (_SRC_DIR / ".git").exists():
        subprocess.check_call(
            ["git", "-C", str(_SRC_DIR), "fetch", "--depth", "1", "origin", "tag", tag]
        )
        subprocess.check_call(["git", "-C", str(_SRC_DIR), "checkout", tag])
    else:
        _SRC_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                tag,
                "https://github.com/jax-ml/jax",
                str(_SRC_DIR),
            ]
        )
    return _SRC_DIR


def _run_build(
    src: Path, *, compute: str, cuda: str, cudnn: str, clang: str, jobs: int
) -> None:
    """Invoke ``build/build.py`` to produce the three CUDA wheels."""
    # The source tree (and its dist/) is reused across runs, so clear any wheels
    # from an earlier JAX version / CUDA major first — otherwise _built_wheels
    # would glob them in and install a conflicting mix.
    dist = src / "dist"
    for stale in dist.glob("*.whl"):
        stale.unlink()
    cmd = [
        sys.executable,
        str(src / "build" / "build.py"),
        "build",
        "--wheels=jaxlib,jax-cuda-plugin,jax-cuda-pjrt",
        f"--python_version={sys.version_info.major}.{sys.version_info.minor}",
        f"--cuda_version={cuda}",
        f"--cudnn_version={cudnn}",
        f"--cuda_compute_capabilities={compute}",
        f"--clang_path={clang}",
        f"--bazel_options=--jobs={jobs}",
    ]
    print("Building (this can take ~45-90 min):\n  " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(src), env=_build_env())


def _built_wheels(src: Path) -> list[str]:
    """The three CUDA wheels ``build/build.py`` dropped in ``dist/``."""
    dist = src / "dist"
    wheels: list[str] = []
    for pattern in ("jaxlib-*.whl", "jax_cuda*_plugin-*.whl", "jax_cuda*_pjrt-*.whl"):
        wheels += [str(p) for p in dist.glob(pattern)]
    if not wheels:
        raise RuntimeError(f"no built wheels found in {dist}")
    return wheels


def _wheel_dist_name(wheel: str) -> str:
    """Normalized distribution name from a wheel filename (for --reinstall)."""
    return Path(wheel).name.split("-")[0].replace("_", "-")


def _install_wheels(wheels: list[str]) -> None:
    """Install the CUDA wheels into the axol env, replacing any PyPI build.

    The broken PyPI ``jaxlib`` is the *same version* as ours, so a plain install
    would skip it — force a reinstall of just the three compiled packages while
    still pulling their runtime deps (the ``nvidia-*-cu12`` libs) from PyPI.
    """
    uv = shutil.which("uv")
    if uv is not None:
        reinstall: list[str] = []
        for wheel in wheels:
            reinstall += ["--reinstall-package", _wheel_dist_name(wheel)]
        cmd = [uv, "pip", "install", "--python", sys.executable, *reinstall, *wheels]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall", *wheels]
    print("Installing wheels:\n  " + "\n  ".join(wheels))
    subprocess.check_call(cmd)
    _install_cuda_runtime_deps(wheels)
    _patch_plugin_rpath()


def _plugin_wheel(wheels: list[str]) -> str | None:
    """The ``jax_cudaNN_plugin`` wheel among ``wheels`` (carries the CUDA deps)."""
    for wheel in wheels:
        if re.match(r"jax_cuda\d+_plugin-", Path(wheel).name):
            return wheel
    return None


def _with_cuda_requirements(plugin_wheel: str) -> list[str]:
    """The plugin's ``[with-cuda]`` nvidia requirement specifiers, from METADATA.

    The plugin declares the ``nvidia-*-cu12`` runtime libs it needs only under
    its ``with-cuda`` extra, which installing a raw ``.whl`` file can't pull.
    Parse them out so we can install the exact set/versions the build expects.
    """
    with zipfile.ZipFile(plugin_wheel) as zf:
        meta_name = next(
            (n for n in zf.namelist() if n.endswith(".dist-info/METADATA")), None
        )
        if meta_name is None:
            return []
        text = zf.read(meta_name).decode("utf-8", "replace")
    reqs: list[str] = []
    for line in text.splitlines():
        if not line.startswith("Requires-Dist:"):
            continue
        body = line[len("Requires-Dist:") :].strip()
        if 'extra == "with-cuda"' not in body:
            continue
        req = body.split(";", 1)[0].strip()  # drop the environment marker
        if req:
            reqs.append(req)
    return reqs


def _install_cuda_runtime_deps(wheels: list[str]) -> None:
    """Install the ``nvidia-*-cu12`` CUDA runtime libs the plugin needs.

    Without these, the plugin's patched RPATH points at ``nvidia/*/lib`` dirs
    that don't exist and GPU init fails (e.g. "cuSPARSE library was not found").
    This makes a source-built / prebuilt install self-contained on a fresh box.
    """
    plugin = _plugin_wheel(wheels)
    if plugin is None:
        return
    reqs = _with_cuda_requirements(plugin)
    if not reqs:
        return
    uv = shutil.which("uv")
    if uv is not None:
        cmd = [uv, "pip", "install", "--python", sys.executable, *reqs]
    else:
        cmd = [sys.executable, "-m", "pip", "install", *reqs]
    print("Installing CUDA runtime libs (nvidia-*-cu12):\n  " + " ".join(reqs))
    subprocess.check_call(cmd)


def _site_packages() -> Path:
    """site-packages of the interpreter running this CLI (the install target)."""
    return Path(sysconfig.get_paths()["purelib"])


def _ensure_patchelf() -> str | None:
    """Path to a ``patchelf`` binary, installing it into the env if needed."""
    exe = shutil.which("patchelf") or str(Path(sys.executable).parent / "patchelf")
    if Path(exe).exists():
        return exe
    uv = shutil.which("uv")
    try:
        if uv is not None:
            subprocess.check_call(
                [uv, "pip", "install", "--python", sys.executable, "patchelf"]
            )
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "patchelf"])
    except Exception as exc:  # noqa: BLE001 - offline / no build backend
        _logger.warning("could not install patchelf: %s", exc)
        return None
    exe = str(Path(sys.executable).parent / "patchelf")
    return exe if Path(exe).exists() else shutil.which("patchelf")


def _patch_plugin_rpath() -> None:
    """Point the CUDA plugin at the pip ``nvidia-*`` libs via its RPATH.

    Our source build only bakes Bazel-internal RPATHs, so the plugin can't find
    the ``nvidia-*-cu12`` CUDA libraries the way the official PyPI wheels do
    (which carry ``$ORIGIN/../../nvidia/<comp>/lib``). Worse, the partial set the
    build *does* emit omits transitive deps (e.g. ``libcusparse`` needs
    ``libnvjitlink``), so loading fails with a misleading "library not found".

    Append the full set of installed ``nvidia/<comp>/lib`` dirs (as
    ``$ORIGIN``-relative entries, matching the official wheels) so GPU JAX runs
    with no ``LD_LIBRARY_PATH``. Done per-device at install time, so it works for
    both freshly-built and downloaded-prebuilt wheels regardless of venv path.
    """
    site = _site_packages()
    # Patch whichever xla_cudaNN plugin actually got installed (glob rather than
    # recomputing the major, which could differ from the wheels under --cuda-
    # version and leave the real plugin unpatched).
    so_files = [
        so
        for plugin_dir in (site / "jax_plugins").glob("xla_cuda*")
        for so in plugin_dir.glob("*.so")
    ]
    nvidia = site / "nvidia"
    if not so_files or not nvidia.is_dir():
        return
    patchelf = _ensure_patchelf()
    if patchelf is None:
        _logger.warning(
            "patchelf unavailable; GPU JAX may need LD_LIBRARY_PATH=%s plus the "
            "nvidia-*-cu12 lib dirs at runtime.",
            _cuda_lib_dir(),
        )
        return
    # plugin_dir is site-packages/jax_plugins/xla_cudaNN → two levels below
    # site-packages, so the nvidia libs are at $ORIGIN/../../nvidia/<comp>/lib.
    extra = [
        f"$ORIGIN/../../nvidia/{lib.parent.name}/lib"
        for lib in sorted(nvidia.glob("*/lib"))
        if lib.is_dir()
    ]
    extra.append(_cuda_lib_dir())
    for so in so_files:
        current = subprocess.run(
            [patchelf, "--print-rpath", str(so)],
            capture_output=True,
            text=True,
        ).stdout.strip()
        parts = [p for p in current.split(":") if p]
        parts += [e for e in extra if e not in parts]
        subprocess.check_call([patchelf, "--set-rpath", ":".join(parts), str(so)])
    print(f"Patched CUDA plugin RPATH ({len(so_files)} .so) to the pip nvidia libs.")


# --------------------------------------------------------------------------- #
# Prebuilt-wheel hosting (GitHub releases)
# --------------------------------------------------------------------------- #


def _python_tag() -> str:
    """CPython ABI tag for this interpreter, e.g. ``cp313``."""
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _hosted_release_tag(jax_version: str, cuda_major: str, compute: str) -> str:
    """Release tag encoding everything the wheel filename can't.

    e.g. ``jax-wheels-v0.9.2-cuda12-sm87``. The Python ABI + arch stay in the
    wheel filename, so this fully identifies a build for a device class.
    """
    sm = compute.replace("sm_", "sm")
    return f"jax-wheels-v{jax_version}-cuda{cuda_major}-{sm}"


def _gh_get_json(url: str) -> dict | None:  # type: ignore[type-arg]
    """GET a GitHub API URL and parse JSON; ``None`` on 404 / any error."""
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "almond-axol",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            _logger.info("GitHub API %s -> HTTP %s", url, exc.code)
        return None
    except Exception as exc:  # noqa: BLE001 - offline / bad JSON
        _logger.info("GitHub API %s failed: %s", url, exc)
        return None


def _prebuilt_wheel_urls(tag: str, cuda_major: str, py_tag: str) -> list[str] | None:
    """Asset download URLs for the three wheels in ``tag``, or ``None``.

    Requires all three (jaxlib, cuda plugin, cuda pjrt) to be present; a partial
    release is treated as absent so the caller falls back to building.
    """
    data = _gh_get_json(
        f"https://api.github.com/repos/{_WHEEL_REPO}/releases/tags/{tag}"
    )
    if data is None:
        return None
    patterns = {
        "jaxlib": rf"^jaxlib-.*-{py_tag}-{py_tag}-.*_aarch64\.whl$",
        "plugin": rf"^jax_cuda{cuda_major}_plugin-.*-{py_tag}-{py_tag}-.*_aarch64\.whl$",
        # pjrt is a pure `py3-none` loader, so it carries no CPython ABI tag.
        "pjrt": rf"^jax_cuda{cuda_major}_pjrt-.*_aarch64\.whl$",
    }
    found: dict[str, str] = {}
    for asset in data.get("assets", []):
        name = asset.get("name", "")
        for key, pat in patterns.items():
            if key not in found and re.match(pat, name):
                found[key] = asset["browser_download_url"]
    if len(found) != len(patterns):
        return None
    return list(found.values())


def _download(url: str, dest: Path) -> None:
    """Stream a release asset to ``dest``."""
    req = urllib.request.Request(url, headers={"User-Agent": "almond-axol"})
    with urllib.request.urlopen(req, timeout=300) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)


def _try_install_prebuilt(jax_version: str, cuda_major: str, compute: str) -> bool:
    """Download + install prebuilt wheels for this device class; False if none."""
    tag = _hosted_release_tag(jax_version, cuda_major, compute)
    urls = _prebuilt_wheel_urls(tag, cuda_major, _python_tag())
    if not urls:
        print(f"No prebuilt wheels published at {_WHEEL_REPO} ({tag}).")
        return False
    print(f"Found prebuilt wheels ({tag}); downloading.")
    _WHEEL_CACHE.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for url in urls:
        dest = _WHEEL_CACHE / url.rsplit("/", 1)[-1]
        if not dest.exists():
            _download(url, dest)
        paths.append(str(dest))
    _install_wheels(paths)
    return True


def _publish_wheels(src: Path, jax_version: str, cuda_major: str, compute: str) -> None:
    """Upload the freshly built wheels to the GitHub release for this class."""
    gh = shutil.which("gh")
    if gh is None:
        raise RuntimeError(
            "`gh` (GitHub CLI) not found; install + `gh auth login` to --publish."
        )
    tag = _hosted_release_tag(jax_version, cuda_major, compute)
    wheels = _built_wheels(src)
    # Create the release if it doesn't exist yet, then upload (clobbering any
    # earlier asset of the same name so a rebuild overwrites cleanly).
    if (
        subprocess.run(
            [gh, "release", "view", tag, "-R", _WHEEL_REPO], capture_output=True
        ).returncode
        != 0
    ):
        subprocess.check_call(
            [
                gh,
                "release",
                "create",
                tag,
                "-R",
                _WHEEL_REPO,
                "--title",
                f"JAX {jax_version} CUDA {cuda_major} wheels for {compute}",
                "--notes",
                (
                    f"Prebuilt jaxlib + CUDA plugin for {compute} "
                    f"(CUDA {cuda_major}, {_python_tag()}, linux_aarch64). "
                    "Built by `axol jax.build --publish`."
                ),
            ]
        )
    subprocess.check_call(
        [gh, "release", "upload", tag, *wheels, "-R", _WHEEL_REPO, "--clobber"]
    )
    print(f"Published {len(wheels)} wheels to {_WHEEL_REPO} ({tag}).")


def run(_args: object = None) -> None:
    """Install device-matched JAX CUDA wheels: download prebuilt, else build."""
    force = getattr(_args, "force", False)
    publish = getattr(_args, "publish", False)
    no_download = getattr(_args, "no_download", False)

    # --publish always builds (to produce fresh wheels to upload), even if this
    # box already has working GPU JAX.
    if not force and not publish and _gpu_jax_ok():
        print("GPU JAX already runs on this device; nothing to do.")
        return

    compute = _compute_capability(getattr(_args, "compute", None))
    cuda_major = _cuda_major(getattr(_args, "cuda_version", None))
    jax_version = getattr(_args, "jax_version", None) or _installed_jax_version()

    missing = [
        name
        for name, val in (
            ("compute capability", compute),
            ("CUDA major", cuda_major),
            ("jax version", jax_version),
        )
        if not val
    ]
    if missing:
        raise RuntimeError(
            "could not detect: "
            + ", ".join(missing)
            + ". Pass them explicitly (--compute / --cuda-version / --jax-version)."
        )

    assert compute and cuda_major and jax_version  # narrowed by the check above

    # Download-first: skip the ~hour-long build when a maintainer already
    # published wheels for this device class. (--publish forces a fresh build.)
    if (
        not no_download
        and not publish
        and _try_install_prebuilt(jax_version, cuda_major, compute)
    ):
        if _gpu_jax_ok():
            print("Done. GPU JAX now runs on this device (prebuilt).")
            return
        print(
            "Prebuilt wheels installed but a GPU op still failed; "
            "building from source instead.",
            file=sys.stderr,
        )

    clang = _ensure_build_deps(getattr(_args, "clang_path", None))
    src = _checkout_source(jax_version)

    # Build against the CUDA + cuDNN JAX itself pins for this major (the
    # hermetic build downloads them), NOT the device's local toolkit — newer
    # JAX needs cuSolver APIs missing from older CUDA. Overridable via flags.
    cuda = getattr(_args, "cuda_version", None)
    cudnn = getattr(_args, "cudnn_version", None)
    if not cuda or not cudnn:
        recommended = _jax_bazelrc_cuda(src, cuda_major)
        if recommended is None:
            raise RuntimeError(
                f"could not read JAX's recommended CUDA {cuda_major} versions "
                "from .bazelrc; pass --cuda-version and --cudnn-version."
            )
        cuda = cuda or recommended[0]
        cudnn = cudnn or recommended[1]

    print(
        f"Building JAX {jax_version} for {compute} "
        f"(CUDA {cuda}, cuDNN {cudnn}, Python "
        f"{sys.version_info.major}.{sys.version_info.minor})."
    )
    _run_build(
        src,
        compute=compute,
        cuda=cuda,
        cudnn=cudnn,
        clang=clang,
        jobs=getattr(_args, "jobs", 4),
    )
    _install_wheels(_built_wheels(src))
    if publish:
        _publish_wheels(src, jax_version, cuda_major, compute)

    if _gpu_jax_ok():
        print("Done. GPU JAX now runs on this device.")
    else:
        print(
            "WARNING: built + installed the wheels, but a GPU JAX op still "
            "failed. If it can't load a CUDA lib, ensure the nvidia-*-cu12 pip "
            f"packages are installed and the plugin RPATH points at them (or set "
            f"LD_LIBRARY_PATH to {_cuda_lib_dir()} plus the nvidia lib dirs).",
            file=sys.stderr,
        )
