"""Offline update bundles: one release, everything it installs, no internet.

Some sites keep their robots on a network with no internet access. The hosted
installer and the ``axol serve`` updater both download from PyPI and GitHub, so
for those sites the release workflow (``.github/workflows/publish.yml``)
attaches an offline bundle to every GitHub release::

    axol-offline-v<version>-<platform>.tar
      axol-offline-v<version>-<platform>/
        manifest.json   what the bundle holds (see :class:`Bundle`)
        manifest.sig    Almond's ed25519 signature of manifest.json
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

A bundle can also be uploaded through the control panel (**Install from
file**). ``axol serve`` has no login and runs as root, so before it installs
anything it checks ``manifest.sig`` against :data:`TRUSTED_KEYS` (the keys
built into the *running* version) and extracts only the files the signed
manifest lists, each against its sha256 (:func:`extract`). The release
workflow signs with a key held in GitHub (``scripts/bundle_signing_key.py``).
To rotate it, ship the new public key in :data:`TRUSTED_KEYS` in one release,
then sign with it from the next.

Only the standard library is imported at module level (``cryptography`` is
imported where a signature is made or checked): ``axol provision`` steps and the
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
SIGNATURE = "manifest.sig"
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
# Public ed25519 keys whose signatures an uploaded bundle may carry, by key ID
# (:func:`key_id`): raw 32-byte keys, base64. The private half lives only in
# the release workflow's ``offline-bundle`` GitHub environment.
TRUSTED_KEYS: dict[str, str] = {
    # Created with scripts/bundle_signing_key.py.
    "1349f54a9f032f1d": "bdJOepEYQLF8P3Juk/M/MJ9ZS9pCNsOI7WvaymAkR3A=",
}
# A bundle's top-level directory, as bundle_name() makes it.
_TOP_DIR_RE = re.compile(r"^axol-offline-v\d+(?:\.\d+)*-[a-z0-9]+-[a-z0-9_]+$")
_MAX_MANIFEST_BYTES = 16 << 20
_MAX_SIGNATURE_BYTES = 64 << 10


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


# -- signatures ---------------------------------------------------------------


def key_id(public_key_b64: str) -> str:
    """A short, stable name for a public key: its sha256, first 16 hex digits."""
    import base64

    return hashlib.sha256(base64.b64decode(public_key_b64)).hexdigest()[:16]


def sign_manifest(manifest: bytes, private_key_b64: str) -> bytes:
    """``manifest.sig`` for ``manifest`` (the release workflow's signing step)."""
    import base64

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.from_private_bytes(base64.b64decode(private_key_b64))
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    payload = {
        "keyId": key_id(base64.b64encode(public).decode()),
        "signature": base64.b64encode(private.sign(manifest)).decode(),
    }
    return (json.dumps(payload, indent=2) + "\n").encode()


def verify_signature(
    manifest: bytes, signature: bytes, keys: dict[str, str] | None = None
) -> str:
    """Check ``manifest.sig`` against a trusted key; returns the key ID."""
    import base64
    import binascii

    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    trusted = TRUSTED_KEYS if keys is None else keys
    not_signed = BundleError(
        "This file is not signed by Almond. Download the bundle again from the "
        "release page."
    )
    try:
        payload = json.loads(signature)
        kid = payload["keyId"]
        raw = base64.b64decode(payload["signature"], validate=True)
    except (ValueError, KeyError, TypeError, binascii.Error) as exc:
        raise not_signed from exc
    public = trusted.get(kid) if isinstance(kid, str) else None
    if public is None:
        raise BundleError(
            "This bundle is signed with a key this version of Axol does not "
            "know. Update Axol with the install script inside the bundle, or "
            "install a release that is closer to the one installed."
        )
    try:
        Ed25519PublicKey.from_public_bytes(base64.b64decode(public)).verify(
            raw, manifest
        )
    except (InvalidSignature, ValueError) as exc:
        raise not_signed from exc
    return kid


# -- extracting an uploaded bundle ---------------------------------------------


def _read_member(tar: Any, member: Any, limit: int) -> bytes:
    if member.size > limit:
        raise BundleError(f"{member.name} is too large")
    handle = tar.extractfile(member)
    if handle is None:
        raise BundleError(f"cannot read {member.name}")
    return handle.read()


def extract(tar_path: Path, dest: Path, keys: dict[str, str] | None = None) -> Bundle:
    """Verify an uploaded bundle ``.tar`` and extract it to ``dest``.

    Nothing is written before the signature, the manifest and the platform
    check out. Only regular files the signed manifest lists are extracted,
    each hashed as it is written and compared with the manifest; links,
    devices and any path outside the bundle are refused. ``dest`` is replaced
    only once every file is in place, so a failure leaves no partial bundle.
    """
    import shutil
    import tarfile

    dest = Path(dest)
    staging = dest.with_name(dest.name + ".tmp")
    try:
        tar = tarfile.open(tar_path, "r:")
    except (tarfile.TarError, OSError) as exc:
        raise BundleError(
            "This file is not an Axol offline bundle. Choose the "
            "axol-offline-v<version>-<platform>.tar file from the release page."
        ) from exc
    with tar:
        try:
            members = tar.getmembers()
        except tarfile.TarError as exc:
            raise BundleError(
                "The file is damaged. Download it again from the release page."
            ) from exc
        tops = {m.name.split("/", 1)[0] for m in members}
        if len(tops) != 1 or not _TOP_DIR_RE.match(next(iter(tops))):
            raise BundleError(
                "This file is not an Axol offline bundle. Choose the "
                "axol-offline-v<version>-<platform>.tar file from the release page."
            )
        top = next(iter(tops))
        files: dict[str, Any] = {}
        for member in members:
            if member.name == top:
                if not member.isdir():
                    raise BundleError(f"unexpected entry {member.name}")
                continue
            rel = member.name[len(top) + 1 :]
            if member.isdir():
                continue
            if not member.isfile():
                raise BundleError(f"the bundle contains a link or device: {rel}")
            if rel in (MANIFEST, SIGNATURE, SHA256SUMS):
                files[rel] = member
                continue
            files[_safe_relpath(rel)] = member
        if SIGNATURE not in files:
            raise BundleError(
                "This file is not signed by Almond. Download the bundle again "
                "from the release page."
            )
        if MANIFEST not in files:
            raise BundleError("The bundle has no manifest.json.")
        manifest = _read_member(tar, files[MANIFEST], _MAX_MANIFEST_BYTES)
        signature = _read_member(tar, files[SIGNATURE], _MAX_SIGNATURE_BYTES)
        verify_signature(manifest, signature, keys)
        try:
            data = json.loads(manifest)
        except ValueError as exc:
            raise BundleError(f"cannot read {MANIFEST}: {exc}") from exc
        bundle = parse_manifest(dest, data)
        bundle.check_host()
        if top != bundle_name(bundle.version, bundle.platform):
            raise BundleError(f"{top} does not match its manifest")
        listed = set(bundle.files)
        present = set(files) - {MANIFEST, SIGNATURE, SHA256SUMS}
        if present - listed:
            raise BundleError(
                f"the bundle has files its manifest does not list: "
                f"{', '.join(sorted(present - listed)[:3])}"
            )
        if listed - present:
            raise BundleError(
                f"the bundle is missing {', '.join(sorted(listed - present)[:3])}"
            )

        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(mode=0o700, parents=True)
        try:
            for rel in sorted(listed):
                member = files[rel]
                target = staging / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise BundleError(f"cannot read {rel}")
                digest = hashlib.sha256()
                with source, target.open("xb") as out:
                    for chunk in iter(lambda: source.read(_HASH_CHUNK), b""):
                        digest.update(chunk)
                        out.write(chunk)
                if digest.hexdigest() != bundle.files[rel]:
                    raise BundleError(
                        f"{rel} is damaged. Download the bundle again from the "
                        "release page."
                    )
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
            (staging / MANIFEST).write_bytes(manifest)
            (staging / SIGNATURE).write_bytes(signature)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    shutil.rmtree(dest, ignore_errors=True)
    staging.rename(dest)
    return load(dest)


def load_signed(root: Path, keys: dict[str, str] | None = None) -> Bundle:
    """:func:`load`, after checking ``manifest.sig`` (for a staged upload)."""
    root = Path(root)
    try:
        manifest = (root / MANIFEST).read_bytes()
        signature = (root / SIGNATURE).read_bytes()
    except OSError as exc:
        raise BundleError(f"cannot read the bundle in {root}: {exc}") from exc
    verify_signature(manifest, signature, keys)
    return parse_manifest(root.resolve(), json.loads(manifest))
