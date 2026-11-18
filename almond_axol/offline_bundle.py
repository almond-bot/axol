"""Offline update bundles: one release, everything it installs, no internet.

Some sites keep their robots on a network with no internet access. The hosted
installer and the ``axol serve`` updater both download from PyPI and GitHub, so
for those sites the release workflow (``.github/workflows/publish.yml``)
attaches an offline bundle to every GitHub release::

    axol-offline-v<version>-<platform>.tar
      axol-offline-v<version>-<platform>/
        manifest.json   what the bundle holds (see :class:`Bundle`)
        SHA256SUMS      ``sha256sum -c`` input over every other file
        bundle.env      version/platform/python as KEY=VALUE, for the installer
        install         the hosted installer (run it from here: bundle mode)
        bin/uv          uv, for a host that has none yet
        wheels/         the PyPI release wheel + every locked dependency
        pygobject/      PyGObject + pycairo sdists and their build backends
        rt/axol-rt      the prebuilt realtime core
        sources/        git bundles of the pinned zed-gstreamer / libsurvive

``wheels/`` holds the exact ``almond-axol`` wheel published to PyPI, which also
carries the built control panel and VR app (``almond_axol/serve/ui``), so an
offline host serves the UI itself.

The installer verifies the files, then reinstalls from ``wheels/`` with uv's
``--offline --no-index --find-links``, never a path to the wheel itself: a
find-links install records no PEP 610 ``direct_url.json``, so the host still
reads as a PyPI release install (the updater stays enabled, ``rt.install``
derives the ``v<version>`` ref). It then runs ``axol provision`` with
:data:`ENV` pointing here, and each step that would download something takes
it from the bundle instead (:func:`active`).

Only the standard library is imported here: ``axol provision`` steps and the
CI build script (``scripts/build_offline_bundle.py``) both load this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

# Set by the installer for `axol provision`: the extracted bundle's directory.
ENV = "AXOL_OFFLINE_BUNDLE"
# Bump when the layout changes in a way an older reader would misread.
FORMAT_VERSION = 1
PACKAGE = "almond-axol"

MANIFEST = "manifest.json"
SHA256SUMS = "SHA256SUMS"
BUNDLE_ENV = "bundle.env"
INSTALLER = "install"
UV_BINARY = "bin/uv"
WHEELS_DIR = "wheels"
PYGOBJECT_DIR = "pygobject"
RT_BINARY = "rt/axol-rt"
SOURCES_DIR = "sources"

# Top-level directory names inside a bundle; a manifest path outside these is
# rejected so a crafted manifest cannot point a step anywhere else.
_TOP_LEVEL = frozenset(
    {
        INSTALLER,
        BUNDLE_ENV,
        "bin",
        WHEELS_DIR,
        PYGOBJECT_DIR,
        "rt",
        SOURCES_DIR,
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^\d+(?:\.\d+)*$")
_HASH_CHUNK = 1 << 20


class BundleError(RuntimeError):
    """A bundle is missing, malformed, damaged or for another host."""


def host_platform() -> str:
    """This host's bundle platform, e.g. ``"linux-aarch64"``."""
    machine = platform.machine().lower()
    machine = {"arm64": "aarch64", "amd64": "x86_64"}.get(machine, machine)
    return f"{sys.platform}-{machine}"


def bundle_name(version: str, platform_tag: str) -> str:
    """Directory (and, plus ``.tar``, file) name of a bundle."""
    return f"axol-offline-v{version}-{platform_tag}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relpath(raw: object) -> str:
    """A manifest file path, or :class:`BundleError` if it could escape."""
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise BundleError(f"bad file path in {MANIFEST}: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise BundleError(f"bad file path in {MANIFEST}: {raw!r}")
    if path.parts[0] not in _TOP_LEVEL:
        raise BundleError(f"unexpected file in {MANIFEST}: {raw!r}")
    return path.as_posix()


@dataclass(frozen=True)
class Source:
    """A pinned third-party git repository, shipped as a full git bundle."""

    path: str
    ref: str


@dataclass(frozen=True)
class Bundle:
    """A parsed ``manifest.json``.

    ``files`` maps every shipped file (relative path) to its sha256; ``rt_ref``
    is the ref the prebuilt ``axol-rt`` was built from (``v<version>``), and
    ``sources`` names each pinned git bundle with the commit it holds.
    """

    root: Path
    version: str
    platform: str
    python: str
    extras: str
    commit: str | None = None
    rt_ref: str | None = None
    sources: dict[str, Source] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)

    @property
    def requirement(self) -> str:
        """The requirement the installer resolves against ``wheels/``."""
        return f"{PACKAGE}[{self.extras}]=={self.version}"

    @property
    def wheels(self) -> Path:
        return self.root / WHEELS_DIR

    @property
    def pygobject(self) -> Path:
        return self.root / PYGOBJECT_DIR

    def verify_file(self, relpath: str) -> Path:
        """``relpath``'s on-disk path once its hash matches the manifest."""
        expected = self.files.get(relpath)
        if expected is None:
            raise BundleError(f"{relpath} is not part of this bundle")
        path = self.root / relpath
        if not path.is_file() or path.is_symlink():
            raise BundleError(f"{relpath} is missing from {self.root}")
        if sha256_file(path) != expected:
            raise BundleError(
                f"{relpath} is damaged (checksum mismatch). Copy the bundle "
                "again from the release page."
            )
        return path

    def verify(self) -> None:
        """Check every file in the manifest against its sha256."""
        for relpath in sorted(self.files):
            self.verify_file(relpath)

    def check_host(self) -> None:
        """Refuse a bundle built for another platform than this host."""
        here = host_platform()
        if self.platform != here:
            raise BundleError(
                f"this bundle is for {self.platform}, but this computer is "
                f"{here}. Download {bundle_name(self.version, here)}.tar from "
                "the release page."
            )

    def rt_binary(self, ref: str) -> Path | None:
        """The verified prebuilt ``axol-rt`` when it was built from ``ref``."""
        if self.rt_ref != ref or RT_BINARY not in self.files:
            return None
        return self.verify_file(RT_BINARY)

    def git_source(self, name: str, ref: str) -> Path | None:
        """The verified git bundle for ``name`` when it holds ``ref``.

        ``None`` when this bundle carries no such source or a different commit
        (the caller then falls back to its network remote).
        """
        source = self.sources.get(name)
        if source is None or source.ref != ref:
            return None
        return self.verify_file(source.path)


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise BundleError(f"{MANIFEST} has no {key!r}")
    return value


def parse_manifest(root: Path, data: object) -> Bundle:
    """Validate a decoded ``manifest.json`` found in ``root``."""
    if not isinstance(data, dict):
        raise BundleError(f"{MANIFEST} is not a JSON object")
    fmt = data.get("format")
    if fmt != FORMAT_VERSION:
        raise BundleError(
            f"this bundle uses format {fmt!r}; this version of Axol reads "
            f"format {FORMAT_VERSION}. Use the install script inside the bundle."
        )
    if data.get("package") != PACKAGE:
        raise BundleError(f"{MANIFEST} is not an {PACKAGE} bundle")
    version = _require_str(data, "version")
    if not _VERSION_RE.match(version):
        raise BundleError(f"bad version in {MANIFEST}: {version!r}")
    raw_files = data.get("files")
    if not isinstance(raw_files, dict) or not raw_files:
        raise BundleError(f"{MANIFEST} lists no files")
    files: dict[str, str] = {}
    for raw_path, digest in raw_files.items():
        relpath = _safe_relpath(raw_path)
        if not isinstance(digest, str) or not _SHA256_RE.match(digest):
            raise BundleError(f"bad sha256 for {relpath} in {MANIFEST}")
        files[relpath] = digest
    rt = data.get("rt") or {}
    rt_ref = rt.get("ref") if isinstance(rt, dict) else None
    sources: dict[str, Source] = {}
    raw_sources = data.get("sources") or {}
    if not isinstance(raw_sources, dict):
        raise BundleError(f"bad 'sources' in {MANIFEST}")
    for name, entry in raw_sources.items():
        if not isinstance(entry, dict):
            raise BundleError(f"bad source {name!r} in {MANIFEST}")
        path = _safe_relpath(entry.get("path"))
        ref = entry.get("ref")
        if path not in files or not isinstance(ref, str) or not ref:
            raise BundleError(f"bad source {name!r} in {MANIFEST}")
        sources[str(name)] = Source(path=path, ref=ref)
    commit = data.get("commit")
    return Bundle(
        root=root,
        version=version,
        platform=_require_str(data, "platform"),
        python=_require_str(data, "python"),
        extras=_require_str(data, "extras"),
        commit=commit if isinstance(commit, str) and commit else None,
        rt_ref=rt_ref if isinstance(rt_ref, str) and rt_ref else None,
        sources=sources,
        files=files,
    )


def load(root: Path) -> Bundle:
    """Read and validate the bundle extracted at ``root`` (hashes unchecked)."""
    root = Path(root).resolve()
    manifest = root / MANIFEST
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BundleError(f"no {MANIFEST} in {root}; is this an Axol bundle?") from exc
    except (OSError, ValueError) as exc:
        raise BundleError(f"cannot read {manifest}: {exc}") from exc
    return parse_manifest(root, data)


_active: tuple[str, Bundle] | None = None


def active() -> Bundle | None:
    """The bundle named by :data:`ENV`, or ``None`` on a normal (online) run.

    Raises :class:`BundleError` when the variable is set but the bundle is
    unusable: a step that silently fell back to the network would fail later
    with a far less useful message.
    """
    global _active
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        return None
    if _active is not None and _active[0] == raw:
        return _active[1]
    bundle = load(Path(raw))
    _active = (raw, bundle)
    return bundle


def git_remote(name: str, ref: str) -> str | None:
    """Where to clone/fetch the pinned ``name`` at ``ref`` from, offline.

    The path of the active bundle's git bundle when it holds ``ref``; ``None``
    on an online run or when the bundle doesn't carry that commit (the caller
    then uses its network remote). A git bundle works as a remote for both
    ``git clone`` and ``git fetch <bundle> <commit>``.
    """
    bundle = active()
    if bundle is None:
        return None
    path = bundle.git_source(name, ref)
    return None if path is None else str(path)
