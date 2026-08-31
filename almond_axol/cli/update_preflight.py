"""Fail-closed checks before a hosted ``uv tool install --force`` update.

The hosted installer and the in-process self-updater both use this module.  A
force install rebuilds the complete Axol environment, so anything installed
from outside Axol's PyPI dependency graph must either be carried into that
transaction explicitly or protected from replacement here.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution

from .tracker_ultimate import ultimate_runtime_update_requirement

_EXPECTED_PYPI_TORCH_VERSION = "2.10.0"
_LEROBOT_PLUGIN_DISTRIBUTION = "lerobot_robot_axol"
_INDEX_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")
_UPDATE_PREFLIGHT_BLOCKED = 20

# Keep the probe in a child interpreter. Importing torch is comparatively
# expensive and can load native libraries; update status polling must never do
# that work. This probe runs only for an explicit installer/self-update on
# aarch64 when an existing torch distribution is present.
_TORCH_PROBE = """
import json

try:
    import torch
    print(json.dumps({
        "version": str(getattr(torch, "__version__", "")),
        "cuda_build": bool(getattr(getattr(torch, "version", None), "cuda", None)),
    }))
except BaseException:
    raise SystemExit(1)
"""


def _is_aarch64() -> bool:
    """Return whether this interpreter is running on a 64-bit ARM host."""
    return platform.machine().lower() in {"aarch64", "arm64"}


def _torch_replacement_error() -> str | None:
    """Protect an existing aarch64 CUDA/custom torch from a PyPI CPU wheel.

    PyPI's default aarch64 ``torch==2.10.0`` wheel is CPU-only.  The hosted
    Axol install deliberately pins that version to avoid PyTorch 2.11's CUDA 13
    dependency surface, but a force rebuild must not silently replace an
    operator-managed JetPack/CUDA build.  We cannot reconstruct a trusted wheel
    source from installed files, so an existing nonstandard build blocks before
    any mutation and tells the operator to manage that deployment explicitly.
    """
    if not _is_aarch64():
        return None

    try:
        torch_dist = distribution("torch")
    except PackageNotFoundError:
        return None

    # A different/local version or direct wheel/VCS source is already outside
    # the exact PyPI contract. Do not resolve it away even if importing torch is
    # currently broken (the source may be a deliberate JetPack build).
    if torch_dist.version != _EXPECTED_PYPI_TORCH_VERSION or torch_dist.read_text(
        "direct_url.json"
    ):
        return _custom_torch_block_message()

    try:
        completed = subprocess.run(
            [sys.executable, "-c", _TORCH_PROBE],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return _custom_torch_block_message()
    if completed.returncode != 0:
        return _custom_torch_block_message()
    try:
        probe = json.loads(completed.stdout)
    except (TypeError, ValueError):
        return _custom_torch_block_message()
    if not isinstance(probe, dict):
        return _custom_torch_block_message()

    runtime_version = probe.get("version")
    cuda_build = probe.get("cuda_build")
    if (
        not isinstance(runtime_version, str)
        or not isinstance(cuda_build, bool)
        or cuda_build
        or runtime_version
        not in {
            _EXPECTED_PYPI_TORCH_VERSION,
            f"{_EXPECTED_PYPI_TORCH_VERSION}+cpu",
        }
    ):
        return _custom_torch_block_message()
    return None


def _custom_torch_block_message() -> str:
    # Fixed text only: never include subprocess output, paths, index URLs, or
    # credentials from package metadata in an API/UI-visible error.
    return (
        "Axol update blocked: this aarch64 environment has a CUDA-enabled or "
        "custom PyTorch build. The hosted force install would replace it with "
        "PyPI's CPU-only torch==2.10.0. Keep the current install; this updater "
        "cannot reconstruct its wheel source. Use a manual update with an "
        "explicit restoration plan for the same JetPack-compatible torch and "
        "torchvision builds, or intentionally switch to CPU Torch before retrying."
    )


def _lerobot_plugin_update_requirement() -> tuple[str | None, str | None]:
    """Preserve an explicitly installed index plugin across a force rebuild.

    PEP 610 leaves index installs without ``direct_url.json``. A direct/VCS
    plugin may contain unpublished code and its source can include credentials,
    so never guess or echo that source: block with fixed text instead.
    """
    try:
        plugin_dist = distribution(_LEROBOT_PLUGIN_DISTRIBUTION)
    except PackageNotFoundError:
        return None, None
    except Exception:  # noqa: BLE001 - keep metadata failures secret-safe
        return None, _custom_plugin_block_message()

    try:
        version = plugin_dist.version
        direct_url = plugin_dist.read_text("direct_url.json")
    except (OSError, TypeError, ValueError):
        return None, _custom_plugin_block_message()
    if not isinstance(version, str) or not _INDEX_VERSION_RE.fullmatch(version):
        return None, _custom_plugin_block_message()
    if direct_url is not None:
        return None, _custom_plugin_block_message()
    return f"{_LEROBOT_PLUGIN_DISTRIBUTION}=={version}", None


def _custom_plugin_block_message() -> str:
    return (
        "Axol update blocked: this environment has a direct or customized "
        "lerobot_robot_axol plugin install. The hosted force install cannot "
        "reconstruct that source safely. Reinstall Axol manually with an explicit "
        "plan for the same plugin source, or replace it with a published PyPI "
        "plugin release before retrying."
    )


def release_update_requirements() -> tuple[list[str], str | None]:
    """Return allowlisted extra requirements to preserve, or a safe error."""
    try:
        torch_error = _torch_replacement_error()
    except Exception:  # noqa: BLE001 - never surface package/index metadata
        return [], _inspection_block_message()
    if torch_error is not None:
        return [], torch_error

    requirements: list[str] = []
    try:
        ultimate_requirement, ultimate_error = ultimate_runtime_update_requirement()
    except Exception:  # noqa: BLE001 - Wi-Fi/package metadata may be sensitive
        return [], _inspection_block_message()
    if ultimate_error is not None:
        return [], ultimate_error
    if ultimate_requirement is not None:
        requirements.append(ultimate_requirement)

    try:
        plugin_requirement, plugin_error = _lerobot_plugin_update_requirement()
    except Exception:  # noqa: BLE001 - package metadata may contain credentials
        return [], _inspection_block_message()
    if plugin_error is not None:
        return [], plugin_error
    if plugin_requirement is not None:
        requirements.append(plugin_requirement)
    return requirements, None


def _inspection_block_message() -> str:
    return (
        "Axol update blocked: the existing environment could not be inspected "
        "safely. No changes were made. Repair its package metadata or perform a "
        "manual update with an explicit dependency-restoration plan."
    )


def run(_args: object = None) -> None:
    """Emit the machine-readable hosted-update contract for the shell installer.

    Success with empty stdout means no extra requirement is needed. Each
    output line is one allowlisted requirement: the explicit Ultimate VCS pin
    and/or the exact published LeRobot plugin version. A safe blocking error is
    written to stderr with a distinct exit status.
    """
    requirements, error = release_update_requirements()
    if error is not None:
        print(error, file=sys.stderr)
        raise SystemExit(_UPDATE_PREFLIGHT_BLOCKED)
    for requirement in requirements:
        print(requirement)


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser(
        "update-preflight",
        help="Check whether a hosted force update can preserve the current runtime.",
    )
    parser.set_defaults(func=run)
