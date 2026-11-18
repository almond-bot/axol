"""Install from file: signed offline bundles, safe extraction, the upload store
and the updater's offline install."""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from almond_axol import offline_bundle as ob
from almond_axol.serve import bundle_upload, update

REPO = Path(__file__).resolve().parents[1]


def _load_build_script():
    spec = importlib.util.spec_from_file_location(
        "build_offline_bundle_for_upload_tests",
        REPO / "scripts" / "build_offline_bundle.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build_script()


def _keypair() -> tuple[str, dict[str, str]]:
    private = Ed25519PrivateKey.generate()
    seed = private.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public = base64.b64encode(
        private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
    ).decode()
    return base64.b64encode(seed).decode(), {ob.key_id(public): public}


@pytest.fixture
def keys(monkeypatch) -> tuple[str, dict[str, str]]:
    private, trusted = _keypair()
    monkeypatch.setattr(ob, "TRUSTED_KEYS", trusted)
    return private, trusted


def _stage(
    root: Path,
    private: str | None,
    trusted: dict[str, str] | None,
    *,
    version: str = "1.2.3",
    platform_tag: str | None = None,
) -> Path:
    stage = root / ob.bundle_name(version, platform_tag or ob.host_platform())
    (stage / ob.WHEELS_DIR).mkdir(parents=True)
    (stage / ob.WHEELS_DIR / f"almond_axol-{version}-py3-none-any.whl").write_bytes(
        b"w"
    )
    rt = stage / ob.RT_BINARY
    rt.parent.mkdir()
    rt.write_text("#!/bin/sh\n")
    rt.chmod(0o755)
    build.write_metadata(
        stage,
        version=version,
        platform_tag=platform_tag or ob.host_platform(),
        python="3.13",
        extras="lerobot,sim",
        sources={},
        signing_key=private,
        trusted_keys=trusted,
    )
    return stage


def _bundle_tar(tmp_path: Path, private: str | None, trusted, **kwargs) -> Path:
    stage = _stage(tmp_path / "stage", private, trusted, **kwargs)
    tar_path = tmp_path / f"{stage.name}.tar"
    build.write_tar(stage, tar_path, mtime=0)
    return tar_path


def _with_member(tar_path: Path, name: str, data: bytes, **info) -> Path:
    """A copy of ``tar_path`` with one more member appended."""
    out = tar_path.with_name("crafted.tar")
    with tarfile.open(tar_path) as src, tarfile.open(out, "w") as dst:
        for member in src.getmembers():
            dst.addfile(member, src.extractfile(member) if member.isfile() else None)
        member = tarfile.TarInfo(name)
        member.size = len(data)
        for key, value in info.items():
            setattr(member, key, value)
        dst.addfile(member, io.BytesIO(data) if member.isfile() else None)
    return out


# -- signatures ---------------------------------------------------------------


def test_signature_round_trip_and_tampering(keys) -> None:
    private, trusted = keys
    manifest = b'{"a": 1}\n'
    signature = ob.sign_manifest(manifest, private)
    assert ob.verify_signature(manifest, signature) in trusted
    with pytest.raises(ob.BundleError, match="not signed by Almond"):
        ob.verify_signature(b'{"a": 2}\n', signature)
    with pytest.raises(ob.BundleError, match="not signed by Almond"):
        ob.verify_signature(manifest, b"garbage")


def test_unknown_key_is_refused(keys) -> None:
    other, _ = _keypair()
    signature = ob.sign_manifest(b"m", other)
    with pytest.raises(ob.BundleError, match="key this version of Axol does not know"):
        ob.verify_signature(b"m", signature)


def test_build_refuses_a_key_the_release_does_not_trust(tmp_path: Path) -> None:
    private, _ = _keypair()
    _, trusted = _keypair()
    with pytest.raises(ob.BundleError):
        _stage(tmp_path, private, trusted)


def test_signature_is_in_sha256sums(tmp_path: Path, keys) -> None:
    stage = _stage(tmp_path, *keys)
    sums = (stage / ob.SHA256SUMS).read_text()
    assert f"  {ob.SIGNATURE}\n" in sums
    assert f"  {ob.MANIFEST}\n" in sums


# -- extraction ---------------------------------------------------------------


def test_extracts_a_signed_bundle(tmp_path: Path, keys) -> None:
    tar_path = _bundle_tar(tmp_path, *keys)
    bundle = ob.extract(tar_path, tmp_path / "out")
    assert bundle.version == "1.2.3"
    assert bundle.root == (tmp_path / "out").resolve()
    bundle.verify()
    assert (tmp_path / "out" / ob.RT_BINARY).stat().st_mode & 0o111
    assert ob.load_signed(tmp_path / "out").version == "1.2.3"


def test_unsigned_bundle_is_refused(tmp_path: Path, keys) -> None:
    tar_path = _bundle_tar(tmp_path, None, None)
    with pytest.raises(ob.BundleError, match="not signed by Almond"):
        ob.extract(tar_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_bundle_for_another_platform_is_refused(tmp_path: Path, keys) -> None:
    tar_path = _bundle_tar(tmp_path, *keys, platform_tag="linux-riscv64")
    with pytest.raises(ob.BundleError, match="linux-riscv64"):
        ob.extract(tar_path, tmp_path / "out")


def test_not_a_tar_is_refused(tmp_path: Path, keys) -> None:
    junk = tmp_path / "photo.tar"
    junk.write_bytes(b"\x89PNG not a tar at all" * 100)
    with pytest.raises(ob.BundleError, match="not an Axol offline bundle"):
        ob.extract(junk, tmp_path / "out")


@pytest.mark.parametrize(
    ("suffix", "info"),
    [
        ("/../../escape", {}),
        ("/wheels/evil.whl", {}),
        ("/wheels/link", {"type": tarfile.SYMTYPE, "linkname": "/etc/passwd"}),
        ("/wheels/hard", {"type": tarfile.LNKTYPE, "linkname": "x"}),
    ],
)
def test_crafted_members_are_refused(tmp_path: Path, keys, suffix: str, info) -> None:
    tar_path = _bundle_tar(tmp_path, *keys)
    top = tar_path.name.removesuffix(".tar")
    crafted = _with_member(tar_path, top + suffix, b"x", **info)
    with pytest.raises(ob.BundleError):
        ob.extract(crafted, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "escape").exists()


def test_member_outside_the_top_directory_is_refused(tmp_path: Path, keys) -> None:
    crafted = _with_member(_bundle_tar(tmp_path, *keys), "other/file", b"x")
    with pytest.raises(ob.BundleError, match="not an Axol offline bundle"):
        ob.extract(crafted, tmp_path / "out")


def test_damaged_file_is_refused_and_nothing_is_left(tmp_path: Path, keys) -> None:
    stage = _stage(tmp_path / "stage", *keys)
    # Change a file after the manifest was signed.
    (stage / ob.RT_BINARY).write_text("#!/bin/sh\necho changed\n")
    tar_path = tmp_path / "b.tar"
    build.write_tar(stage, tar_path, mtime=0)
    with pytest.raises(ob.BundleError, match="damaged"):
        ob.extract(tar_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "out.tmp").exists()


# -- the upload store ---------------------------------------------------------


async def _chunks(data: bytes, size: int = 7):
    for start in range(0, len(data), size):
        yield data[start : start + size]


def _store(tmp_path: Path, installed: str = "1.0.0") -> bundle_upload.BundleStore:
    return bundle_upload.BundleStore(lambda: installed, tmp_path / "updates")


async def _upload(store, data: bytes, name: str = "b.tar") -> dict:
    status = await store.begin(name, len(data))
    status = await store.append(status["uploadId"], 0, _chunks(data))
    await store.finish(status["uploadId"])
    await store._check_task
    return store.status()


def test_upload_check_and_ready(tmp_path: Path, keys) -> None:
    data = _bundle_tar(tmp_path, *keys).read_bytes()

    async def exercise() -> None:
        store = _store(tmp_path)
        status = await _upload(store, data)
        assert status["state"] == "ready", status
        assert status["version"] == "1.2.3"
        assert not (tmp_path / "updates" / bundle_upload._PARTIAL).exists()
        bundle = store.ready_bundle()
        assert bundle is not None and bundle.version == "1.2.3"
        store.discard_installed()
        assert store.status()["state"] == "idle"
        assert not (tmp_path / "updates" / bundle_upload._STAGED).exists()

    asyncio.run(exercise())


def test_upload_resumes_and_rejects_out_of_step_chunks(tmp_path: Path, keys) -> None:
    data = _bundle_tar(tmp_path, *keys).read_bytes()

    async def exercise() -> None:
        store = _store(tmp_path)
        first = await store.begin("b.tar", len(data))
        upload_id = first["uploadId"]
        half = len(data) // 2
        await store.append(upload_id, 0, _chunks(data[:half]))
        with pytest.raises(bundle_upload.UploadError) as err:
            await store.append(upload_id, 0, _chunks(data[half:]))
        assert err.value.status == 409 and err.value.code == "offset"
        # The panel reconnects: same file, same upload, resumes at `half`.
        again = await store.begin("b.tar", len(data))
        assert again["uploadId"] == upload_id and again["received"] == half
        await store.append(upload_id, half, _chunks(data[half:]))
        await store.finish(upload_id)
        await store._check_task
        assert store.status()["state"] == "ready"

    asyncio.run(exercise())


def test_upload_larger_than_announced_is_refused(tmp_path: Path) -> None:
    async def exercise() -> None:
        store = _store(tmp_path)
        status = await store.begin("b.tar", 10)
        with pytest.raises(bundle_upload.UploadError) as err:
            await store.append(status["uploadId"], 0, _chunks(b"x" * 11))
        assert err.value.status == 413
        assert (tmp_path / "updates" / bundle_upload._PARTIAL).stat().st_size == 0

    asyncio.run(exercise())


def test_not_enough_space_names_how_much(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        bundle_upload.shutil,
        "disk_usage",
        lambda _path: type("U", (), {"free": 1 << 30})(),
    )

    async def exercise() -> None:
        with pytest.raises(bundle_upload.UploadError) as err:
            await _store(tmp_path).begin("b.tar", 800_000_000)
        assert err.value.status == 507 and err.value.code == "space"
        assert "more free space" in str(err.value)

    asyncio.run(exercise())


def test_unsigned_upload_ends_in_an_error_and_is_removed(tmp_path: Path, keys) -> None:
    data = _bundle_tar(tmp_path, None, None).read_bytes()

    async def exercise() -> None:
        store = _store(tmp_path)
        status = await _upload(store, data)
        assert status["state"] == "error"
        assert "not signed by Almond" in status["error"]
        assert store.ready_bundle() is None
        assert not any((tmp_path / "updates").iterdir())

    asyncio.run(exercise())


def test_older_version_needs_confirming(tmp_path: Path, keys) -> None:
    data = _bundle_tar(tmp_path, *keys).read_bytes()

    async def exercise() -> None:
        store = _store(tmp_path, installed="2.0.0")
        status = await _upload(store, data)
        assert status["state"] == "error" and status["errorCode"] == "older"
        assert "older than the installed v2.0.0" in status["error"]
        await store.recheck(allow_older=True)
        await store._check_task
        assert store.status()["state"] == "ready"

    asyncio.run(exercise())


def test_same_version_is_refused(tmp_path: Path, keys) -> None:
    data = _bundle_tar(tmp_path, *keys).read_bytes()

    async def exercise() -> None:
        status = await _upload(_store(tmp_path, installed="1.2.3"), data)
        assert status["errorCode"] == "installed"

    asyncio.run(exercise())


def test_restart_keeps_a_ready_bundle_and_drops_an_installed_one(
    tmp_path: Path, keys
) -> None:
    data = _bundle_tar(tmp_path, *keys).read_bytes()

    async def upload() -> None:
        await _upload(_store(tmp_path), data)

    asyncio.run(upload())
    assert _store(tmp_path).status()["state"] == "ready"
    # After the update landed, the running version is the bundle's.
    assert _store(tmp_path, installed="1.2.3").status()["state"] == "idle"
    assert not (tmp_path / "updates" / bundle_upload._STAGED).exists()


# -- the updater's offline install ----------------------------------------------


class _Proc:
    pid = 4242

    def __init__(self, **spawn) -> None:
        self.returncode = 0
        sink = spawn.get("stdout")
        if hasattr(sink, "write"):
            sink.write(b"ok\n")

    async def wait(self):
        return 0

    async def communicate(self):
        return b"", b""


def _updater(monkeypatch, tmp_path: Path) -> update.SelfUpdater:
    monkeypatch.setattr(update, "_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(update, "_LOG_FALLBACK_DIR", tmp_path / "fallback")
    monkeypatch.setattr(update, "installed_origin", lambda: None)
    monkeypatch.setattr(update, "installed_channel", lambda: "release")
    monkeypatch.setattr(update, "installed_from_index", lambda: True)
    monkeypatch.setattr(update, "installed_version", lambda: "1.0.0")
    monkeypatch.setattr(update, "installed_commit", lambda: None)
    monkeypatch.setattr(update.shutil, "which", lambda name: f"/bin/{name}")
    return update.SelfUpdater(lambda: True)


def test_updater_installs_the_bundle_offline(monkeypatch, tmp_path: Path, keys) -> None:
    bundle = ob.extract(_bundle_tar(tmp_path, *keys), tmp_path / "staged")

    async def exercise() -> None:
        updater = _updater(monkeypatch, tmp_path)
        calls: list[tuple[tuple[str, ...], dict | None]] = []

        async def spawn(*args, **kwargs):
            calls.append((args, kwargs.get("env")))
            return _Proc(**kwargs)

        monkeypatch.setattr(update.asyncio, "create_subprocess_exec", spawn)
        exits: list[int] = []
        monkeypatch.setattr(update.os, "_exit", exits.append)
        cleaned: list[bool] = []
        started, reason = updater.start(
            bundle=bundle, after_install=lambda: cleaned.append(True)
        )
        assert started, reason
        status = await updater.status()
        assert status["source"] == "bundle" and status["targetVersion"] == "1.2.3"
        await updater._update_task

        # (status() also starts a background `git ls-remote`; ignore it.)
        steps = [call for call in calls if call[0][0] != "git"]
        (install, install_env), (provision, provision_env) = steps
        assert install[-1] == "almond-axol[lerobot,sim]==1.2.3"
        # Find-links from the bundle, with no index: never a path to the wheel.
        assert install[-5:-1] == (
            "--offline",
            "--no-index",
            "--find-links",
            str(bundle.wheels),
        )
        assert not any(arg.endswith(".whl") for arg in install)
        assert provision[1:] == ("provision", "--no-reboot", "--require-rt")
        for env in (install_env, provision_env):
            assert env is not None and env[ob.ENV] == str(bundle.root)
        assert cleaned == [True]
        assert exits == [0]

    asyncio.run(exercise())


def test_updater_refuses_a_bundle_while_busy_or_mismatched(
    monkeypatch, tmp_path: Path, keys
) -> None:
    bundle = ob.extract(_bundle_tar(tmp_path, *keys), tmp_path / "staged")
    updater = _updater(monkeypatch, tmp_path)
    updater._is_idle = lambda: False
    assert updater.start(bundle=bundle) == (
        False,
        "server is busy; stop the running operation first",
    )
    other = ob.Bundle(**{**bundle.__dict__, "extras": "sim"})
    updater._is_idle = lambda: True
    assert updater.start(bundle=other)[0] is False
    assert updater.start("main", bundle=bundle)[0] is False


def test_network_update_reports_no_bundle(monkeypatch, tmp_path: Path) -> None:
    async def exercise() -> None:
        status = await _updater(monkeypatch, tmp_path).status()
        assert status["source"] is None and status["targetVersion"] is None

    asyncio.run(exercise())


def test_trusted_keys_are_well_formed() -> None:
    for kid, public in ob.TRUSTED_KEYS.items():
        assert len(base64.b64decode(public)) == 32
        assert ob.key_id(public) == kid
    json.dumps(ob.TRUSTED_KEYS)
