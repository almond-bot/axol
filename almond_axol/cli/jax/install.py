"""
axol jax.install

Install the JAX CUDA plugin that matches *this device's* CUDA version.

The pip ``jax[cudaNN]`` extras each bundle their own CUDA runtime (via the
``nvidia-*`` wheels), so the only thing the host has to provide is a driver new
enough for that CUDA major. Pinning a single extra in ``pyproject.toml`` (e.g.
``jax[cuda13]``) therefore installs CUDA 13 everywhere, which fails to import on
a box whose driver only supports CUDA 12 (the JetPack 6 / L4T Jetsons we run on
report CUDA 12.x). Rather than pin, this step detects the CUDA major the machine
actually supports and installs the matching ``jax[cudaNN]`` variant.

Detection order (first hit wins):

* ``nvidia-smi``            — the driver's *max* supported CUDA (best signal for
                             the pip-bundled runtime, which only needs a
                             compatible driver).
* ``/usr/local/cuda``       — the installed CUDA toolkit (``version.json`` /
                             ``version.txt``), the fallback on Jetsons where
                             ``nvidia-smi`` may be absent.
* ``nvcc --version``        — last-resort toolkit probe.

On a Jetson / Tegra board the PyPI ``jax[cudaNN]`` wheels don't work at all: they
ship no GPU kernels for the integrated GPU (``sm_87`` etc.), so kernel launches
die with ``cudaErrorNoKernelImageForDevice``. There we delegate to
:mod:`.build`, which builds ``jaxlib`` + the CUDA plugin from source for the
device's compute capability.

Run as a ``axol provision`` step (and thus by the hosted installer + the
``axol serve`` self-updater after every ``uv tool upgrade``, which prunes the
non-PyPI plugin from the tool env). Best-effort and idempotent: a no-op on hosts
with no NVIDIA GPU/driver (JAX then runs on CPU, which is usually faster for the
IK solver anyway) and a no-op when the matching plugin is already installed (or,
on Tegra, when GPU JAX already runs).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

_logger = logging.getLogger(__name__)

# JAX core is pinned here to match the ``jax`` dependency in pyproject.toml, so
# adding the CUDA extra can't drag the core package to a different version.
_JAX_SPEC = ">=0.9.2"

# CUDA majors JAX ships a pip extra for. A detected major is clamped into this
# range: below the minimum there's no usable wheel (skip -> CPU), above the
# maximum we install the newest extra (a newer driver is forward-compatible with
# the bundled runtime).
_MIN_CUDA_MAJOR = 12
_MAX_CUDA_MAJOR = 13

_CUDA_HOME = Path("/usr/local/cuda")


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``jax.install`` subcommand."""
    parser = subparsers.add_parser(
        "jax.install",
        help="Install the JAX CUDA plugin matching this device's CUDA version.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reinstall even when a matching JAX CUDA plugin is already present.",
    )
    parser.add_argument(
        "--cuda-major",
        type=int,
        default=None,
        metavar="N",
        help="Override CUDA-major detection (e.g. 12 or 13).",
    )
    parser.set_defaults(func=run)


def _cuda_major_from_nvidia_smi() -> int | None:
    """Driver-reported max CUDA major from ``nvidia-smi`` (``None`` if absent)."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        out = subprocess.run([smi], capture_output=True, text=True, timeout=30).stdout
    except Exception as exc:  # noqa: BLE001 - no GPU / driver mismatch / timeout
        _logger.info("nvidia-smi failed: %s", exc)
        return None
    match = re.search(r"CUDA Version:\s*(\d+)\.\d+", out)
    return int(match.group(1)) if match else None


def _cuda_major_from_toolkit() -> int | None:
    """CUDA major of the installed toolkit under ``/usr/local/cuda``."""
    version_json = _CUDA_HOME / "version.json"
    if version_json.exists():
        try:
            data = json.loads(version_json.read_text())
            release = data["cuda"]["version"]
            return int(str(release).split(".")[0])
        except (ValueError, KeyError, OSError) as exc:
            _logger.info("could not parse %s: %s", version_json, exc)
    version_txt = _CUDA_HOME / "version.txt"
    if version_txt.exists():
        try:
            match = re.search(r"CUDA Version\s+(\d+)\.", version_txt.read_text())
            if match:
                return int(match.group(1))
        except OSError as exc:
            _logger.info("could not read %s: %s", version_txt, exc)
    return None


def _cuda_major_from_nvcc() -> int | None:
    """CUDA major reported by ``nvcc --version`` (``None`` if nvcc is absent)."""
    nvcc = shutil.which("nvcc") or str(_CUDA_HOME / "bin" / "nvcc")
    if not shutil.which(nvcc) and not Path(nvcc).exists():
        return None
    try:
        out = subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception as exc:  # noqa: BLE001 - nvcc missing / timeout
        _logger.info("nvcc failed: %s", exc)
        return None
    match = re.search(r"release\s+(\d+)\.", out)
    return int(match.group(1)) if match else None


def _detect_cuda_major() -> int | None:
    """Best CUDA major for this host, or ``None`` when no CUDA is detected."""
    for probe in (
        _cuda_major_from_nvidia_smi,
        _cuda_major_from_toolkit,
        _cuda_major_from_nvcc,
    ):
        major = probe()
        if major is not None:
            return major
    return None


def _extra_for_major(major: int) -> int | None:
    """JAX pip extra major for a detected CUDA major, or ``None`` if too old.

    Below the minimum there's no usable wheel, so skip the GPU install (CPU).
    Above the maximum, install the newest extra — a newer driver is forward-
    compatible with the bundled CUDA runtime.
    """
    if major < _MIN_CUDA_MAJOR:
        return None
    return min(major, _MAX_CUDA_MAJOR)


def _plugin_installed(major: int) -> bool:
    """True when ``jax-cudaNN-plugin`` for ``major`` is already installed.

    Read from the metadata of the interpreter running this CLI (the uv tool
    env). ``uv tool upgrade`` rebuilds that env and drops the plugin (it's not a
    core PyPI dependency), so this correctly returns False right after an upgrade
    and lets ``axol provision`` re-install it, while skipping a redundant
    reinstall otherwise.
    """
    try:
        _pkg_version(f"jax-cuda{major}-plugin")
    except PackageNotFoundError:
        return False
    return True


def run(_args: object = None) -> None:
    """Detect the host's CUDA version and install the matching JAX CUDA plugin."""
    from . import build as jax_build

    # Jetson / Tegra: the PyPI CUDA wheels have no kernel image for the
    # integrated GPU, so build from source for the device's compute capability
    # instead. jax.build is itself idempotent (no-ops once GPU JAX works), so
    # provision only pays the build cost once.
    if jax_build.is_tegra():
        print(
            "Jetson/Tegra detected: PyPI JAX CUDA wheels have no kernel image "
            "for the integrated GPU; building from source instead."
        )
        jax_build.run(_args)
        return

    override = getattr(_args, "cuda_major", None)
    detected = override if override is not None else _detect_cuda_major()
    if detected is None:
        print(
            "No NVIDIA CUDA detected (no nvidia-smi / CUDA toolkit); "
            "skipping JAX GPU install. JAX will run on CPU."
        )
        return

    major = _extra_for_major(detected)
    if major is None:
        print(
            f"Detected CUDA {detected}, but JAX needs CUDA >= {_MIN_CUDA_MAJOR}; "
            "skipping JAX GPU install. JAX will run on CPU."
        )
        return
    if major != detected:
        print(
            f"Detected CUDA {detected}; installing the nearest supported JAX "
            f"extra (cuda{major})."
        )

    force = getattr(_args, "force", False)
    if not force and _plugin_installed(major):
        print(f"jax[cuda{major}] already installed.")
        return

    spec = f"jax[cuda{major}]{_JAX_SPEC}"
    print(f"Detected CUDA {detected}; installing {spec} into the axol environment...")
    # Target the interpreter running this CLI (the uv tool env), not whatever
    # VIRTUAL_ENV/cwd resolves to — same trick as ``axol zed.install``.
    uv = shutil.which("uv")
    if uv is not None:
        cmd = [uv, "pip", "install", "--python", sys.executable, spec]
    else:
        cmd = [sys.executable, "-m", "pip", "install", spec]
    subprocess.check_call(cmd)
    print(f"Done. JAX CUDA {major} plugin is installed.")
