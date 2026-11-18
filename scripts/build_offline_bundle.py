"""Build the offline update bundle for one release (see almond_axol/offline_bundle.py).

Runs in the release workflow (``.github/workflows/publish.yml``) on a runner of
the bundle's platform, under the Python version installs use, because pip
picks every dependency wheel for the interpreter running it::

    python scripts/build_offline_bundle.py \\
        --wheel dist/almond_axol-0.2.47-py3-none-any.whl \\
        --rt-binary rust/axol-rt/target/release/axol-rt \\
        --uv "$(command -v uv)" --out "$RUNNER_TEMP/bundle"

``--wheel`` must be the wheel published to PyPI (it carries the built web UI,
which a plain ``uv build`` of a checkout does not). The interpreter needs pip;
``uv`` and ``git`` must be on PATH. Writes ``<out>/<name>.tar`` and
``<out>/<name>.tar.sha256``.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from almond_axol import offline_bundle as ob  # noqa: E402

# The wheel must ship the control panel and VR app (publish.yml stages them).
UI_MARKER = "almond_axol/serve/ui/index.html"
# PyGObject's build backends: uv builds the bundle's PyGObject and pycairo
# sdists offline on the robot, in an isolated environment resolved from the
# same directory. ninja and patchelf are wheels so the build needs neither on
# the host. The release's offline smoke test builds PyGObject from exactly
# this set with the network off, so a missing backend fails the release.
PYGOBJECT_BUILD_BACKENDS = ("meson-python", "meson", "ninja", "patchelf")
# The release workflow's secret: the base64 ed25519 private key that signs
# manifest.json (see scripts/bundle_signing_key.py).
SIGNING_KEY_ENV = "AXOL_BUNDLE_SIGNING_KEY"
# The third-party repositories `axol provision` builds at a pinned commit:
# (name used by offline_bundle.git_remote, module that pins it).
PINNED_SOURCES = (
    ("libsurvive", "almond_axol/cli/tracker_install.py"),
    ("zed-gstreamer", "almond_axol/cli/gst/build_zed.py"),
)


def module_constant(relpath: str, name: str) -> str:
    """A module-level string constant, read without importing the module."""
    tree = ast.parse((REPO / relpath).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise SystemExit(f"{relpath} defines no string constant {name}")


def run(cmd: list[str], **kwargs: object) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)  # type: ignore[call-overload]


def check_wheel(wheel: Path) -> str:
    """The wheel's version, after checking it is the release wheel with the UI."""
    parts = wheel.name.split("-")
    if len(parts) != 5 or parts[0] != "almond_axol" or not wheel.suffix == ".whl":
        raise SystemExit(f"{wheel.name} is not an almond-axol wheel")
    version = parts[1]
    with zipfile.ZipFile(wheel) as archive:
        if UI_MARKER not in archive.namelist():
            raise SystemExit(
                f"{wheel.name} has no {UI_MARKER}: pass the wheel the release "
                "job built (publish.yml stages the web UI before `uv build`)"
            )
    return version


def export_requirements(extras: str, dest: Path) -> None:
    """Every locked dependency of the release, with hashes, from uv.lock."""
    cmd = [
        "uv",
        "export",
        "--frozen",
        "--no-dev",
        "--no-emit-project",
        "--format",
        "requirements-txt",
        "--output-file",
        str(dest),
    ]
    for extra in extras.split(","):
        cmd += ["--extra", extra]
    run(cmd, cwd=REPO)


def collect_wheels(requirements: Path, dest: Path) -> None:
    """Download (or build, for sdist-only packages) every locked dependency."""
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--require-hashes",
            "--requirement",
            str(requirements),
            "--wheel-dir",
            str(dest),
        ]
    )


def collect_pygobject(spec: str, dest: Path) -> None:
    """PyGObject + pycairo as sdists (built on the robot) and their backends."""
    pip = [sys.executable, "-m", "pip", "download", "--dest", str(dest)]
    # Only these two as sources: ``:all:`` would also make pip compile the
    # build backends it fetches to read their metadata.
    sources = "pygobject,pycairo"
    run([*pip, "--no-deps", "--no-binary", sources, spec, "pycairo>=1.16"])
    run([*pip, "--only-binary", ":all:", *PYGOBJECT_BUILD_BACKENDS])


def bundle_git_source(url: str, ref: str, dest: Path) -> None:
    """A full-history git bundle of ``url`` at ``ref``.

    Full history, because git cannot clone a bundle made from a shallow fetch.
    """
    with tempfile.TemporaryDirectory() as tmp:
        git = ["git", "-C", tmp]
        run(["git", "init", "--quiet", tmp])
        run([*git, "fetch", "--quiet", url, ref])
        run([*git, "update-ref", "refs/heads/pinned", ref])
        run([*git, "symbolic-ref", "HEAD", "refs/heads/pinned"])
        run([*git, "bundle", "create", str(dest), "HEAD", "refs/heads/pinned"])


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def install_file(src: Path, dest: Path, *, executable: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    dest.chmod(0o755 if executable else 0o644)


def write_metadata(
    stage: Path,
    *,
    version: str,
    platform_tag: str,
    python: str,
    extras: str,
    sources: dict[str, dict[str, str]],
    signing_key: str | None,
    trusted_keys: dict[str, str] | None = None,
) -> None:
    """manifest.json, manifest.sig, bundle.env and SHA256SUMS over everything staged.

    ``signing_key`` is the base64 ed25519 private key; ``None`` builds an
    unsigned bundle (installable over SSH only). A signature must verify
    against ``trusted_keys`` (default: this release's own trusted keys).
    """
    (stage / ob.BUNDLE_ENV).write_text(
        "".join(
            f"{key}={value}\n"
            for key, value in (
                ("AXOL_BUNDLE_FORMAT", ob.FORMAT_VERSION),
                ("AXOL_BUNDLE_VERSION", version),
                ("AXOL_BUNDLE_PLATFORM", platform_tag),
                ("AXOL_BUNDLE_PYTHON", python),
                ("AXOL_BUNDLE_EXTRAS", extras),
            )
        )
    )
    files = {
        path.relative_to(stage).as_posix(): ob.sha256_file(path)
        for path in sorted(stage.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "format": ob.FORMAT_VERSION,
        "package": ob.PACKAGE,
        "version": version,
        "commit": git_commit(),
        "platform": platform_tag,
        "python": python,
        "extras": extras,
        "rt": {"path": ob.RT_BINARY, "ref": f"v{version}"},
        "sources": sources,
        "files": files,
    }
    # Validate with the reader the robot runs, so a malformed manifest fails
    # the release instead of an install.
    ob.parse_manifest(stage, manifest)
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    (stage / ob.MANIFEST).write_bytes(manifest_bytes)
    sums = {**files, ob.MANIFEST: ob.sha256_file(stage / ob.MANIFEST)}
    if signing_key is not None:
        signature = ob.sign_manifest(manifest_bytes, signing_key)
        # A key the robots don't trust would make every upload fail; catch it
        # here instead.
        ob.verify_signature(manifest_bytes, signature, trusted_keys)
        (stage / ob.SIGNATURE).write_bytes(signature)
        sums[ob.SIGNATURE] = ob.sha256_file(stage / ob.SIGNATURE)
    (stage / ob.SHA256SUMS).write_text(
        "".join(f"{digest}  {path}\n" for path, digest in sorted(sums.items()))
    )


def write_tar(stage: Path, tar_path: Path, mtime: int) -> None:
    """A reproducible, uncompressed tar of ``stage`` (wheels are compressed)."""
    top = stage.name
    with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as tar:

        def add(path: Path) -> None:
            info = tar.gettarinfo(str(path), arcname=f"{top}/{path.relative_to(stage)}")
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mtime = mtime
            if path.is_dir():
                info.mode = 0o755
                tar.addfile(info)
            else:
                info.mode = 0o755 if os.access(path, os.X_OK) else 0o644
                with path.open("rb") as handle:
                    tar.addfile(info, handle)

        root = tarfile.TarInfo(top)
        root.type, root.mode, root.mtime = tarfile.DIRTYPE, 0o755, mtime
        root.uname = root.gname = "root"
        tar.addfile(root)
        for path in sorted(stage.rglob("*")):
            if path.is_symlink():
                raise SystemExit(f"refusing to bundle a symlink: {path}")
            add(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--rt-binary", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--unsigned",
        action="store_true",
        help=f"build without {SIGNING_KEY_ENV} (the control panel refuses it)",
    )
    args = parser.parse_args(argv)
    signing_key = os.environ.get(SIGNING_KEY_ENV, "").strip() or None
    if signing_key is None and not args.unsigned:
        raise SystemExit(
            f"{SIGNING_KEY_ENV} is not set; the control panel only installs "
            "signed bundles (pass --unsigned for a local test build)"
        )

    extras = module_constant("almond_axol/serve/update.py", "_EXTRAS")
    python = module_constant("almond_axol/serve/update.py", "_PYTHON_VERSION")
    running = f"{sys.version_info.major}.{sys.version_info.minor}"
    if running != python:
        raise SystemExit(
            f"run this under Python {python} (installs use it), not {running}"
        )
    platform_tag = ob.host_platform()
    version = check_wheel(args.wheel)

    name = ob.bundle_name(version, platform_tag)
    args.out.mkdir(parents=True, exist_ok=True)
    stage = args.out / name
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()

    with tempfile.TemporaryDirectory() as tmp:
        requirements = Path(tmp) / "requirements.txt"
        export_requirements(extras, requirements)
        collect_wheels(requirements, stage / ob.WHEELS_DIR)
    install_file(args.wheel, stage / ob.WHEELS_DIR / args.wheel.name)
    collect_pygobject(
        module_constant("almond_axol/cli/gst/install.py", "_PYGOBJECT_SPEC"),
        stage / ob.PYGOBJECT_DIR,
    )
    install_file(args.rt_binary, stage / ob.RT_BINARY, executable=True)
    install_file(args.uv, stage / ob.UV_BINARY, executable=True)
    install_file(
        REPO / "web" / "app" / "public" / "install",
        stage / ob.INSTALLER,
        executable=True,
    )
    sources: dict[str, dict[str, str]] = {}
    for source_name, module in PINNED_SOURCES:
        relpath = f"{ob.SOURCES_DIR}/{source_name}.bundle"
        (stage / ob.SOURCES_DIR).mkdir(exist_ok=True)
        ref = module_constant(module, "_PINNED_REF")
        bundle_git_source(module_constant(module, "_REPO_URL"), ref, stage / relpath)
        sources[source_name] = {"path": relpath, "ref": ref}

    write_metadata(
        stage,
        version=version,
        platform_tag=platform_tag,
        python=python,
        extras=extras,
        sources=sources,
        signing_key=signing_key,
    )
    mtime = int(os.environ.get("SOURCE_DATE_EPOCH", "0") or 0)
    tar_path = args.out / f"{name}.tar"
    write_tar(stage, tar_path, mtime)
    (args.out / f"{name}.tar.sha256").write_text(
        f"{ob.sha256_file(tar_path)}  {tar_path.name}\n"
    )
    size_mb = tar_path.stat().st_size / 1e6
    print(f"Wrote {tar_path} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
