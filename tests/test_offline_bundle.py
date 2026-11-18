"""Offline update bundles: the manifest, the build script and each offline step."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from almond_axol import offline_bundle as ob
from almond_axol.cli import tracker_install
from almond_axol.cli.gst import build_zed
from almond_axol.cli.gst import install as gst_install
from almond_axol.rt import install as rt_install

REPO = Path(__file__).resolve().parents[1]
INSTALLER = (REPO / "web" / "app" / "public" / "install").read_text()


def _load_build_script():
    spec = importlib.util.spec_from_file_location(
        "build_offline_bundle", REPO / "scripts" / "build_offline_bundle.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build_script()


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.com",
        },
    ).stdout.strip()


class _BundleCase(unittest.TestCase):
    """A small, real bundle staged with the build script's own writers."""

    version = "1.2.3"

    def setUp(self) -> None:
        env = patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(ob.ENV, None)
        cache = patch.object(ob, "_active", None)
        cache.start()
        self.addCleanup(cache.stop)
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.stage = self.tmp / ob.bundle_name(self.version, ob.host_platform())
        self.stage.mkdir()

    def stage_bundle(self, sources: dict[str, dict[str, str]] | None = None) -> None:
        (self.stage / ob.WHEELS_DIR).mkdir(exist_ok=True)
        (self.stage / ob.WHEELS_DIR / "dep-1.0-py3-none-any.whl").write_bytes(b"w")
        (self.stage / ob.PYGOBJECT_DIR).mkdir(exist_ok=True)
        (self.stage / ob.PYGOBJECT_DIR / "pygobject-3.50.0.tar.gz").write_bytes(b"s")
        rt = self.stage / ob.RT_BINARY
        rt.parent.mkdir(exist_ok=True)
        rt.write_text("#!/bin/sh\necho axol-rt\n")
        rt.chmod(0o755)
        (self.stage / ob.INSTALLER).write_text("#!/usr/bin/env bash\n")
        build.write_metadata(
            self.stage,
            version=self.version,
            platform_tag=ob.host_platform(),
            python="3.13",
            extras="lerobot,sim",
            sources=sources or {},
            signing_key=None,
        )

    def activate(self) -> ob.Bundle:
        os.environ[ob.ENV] = str(self.stage)
        bundle = ob.active()
        assert bundle is not None
        return bundle


class ManifestTest(_BundleCase):
    def test_written_manifest_loads_and_verifies(self) -> None:
        self.stage_bundle()
        bundle = ob.load(self.stage)
        self.assertEqual(bundle.version, self.version)
        self.assertEqual(bundle.requirement, "almond-axol[lerobot,sim]==1.2.3")
        self.assertEqual(bundle.rt_ref, "v1.2.3")
        self.assertIn(ob.BUNDLE_ENV, bundle.files)
        bundle.verify()
        bundle.check_host()

    def test_sha256sums_checks_out_with_coreutils(self) -> None:
        if shutil.which("sha256sum") is None:
            self.skipTest("no sha256sum")
        self.stage_bundle()
        subprocess.run(
            ["sha256sum", "--quiet", "--strict", "-c", ob.SHA256SUMS],
            cwd=self.stage,
            check=True,
        )
        listed = {
            line.split("  ", 1)[1]
            for line in (self.stage / ob.SHA256SUMS).read_text().splitlines()
        }
        self.assertIn(ob.MANIFEST, listed)

    def test_bundle_env_is_what_the_installer_reads(self) -> None:
        self.stage_bundle()
        env = (self.stage / ob.BUNDLE_ENV).read_text()
        for key in (
            "AXOL_BUNDLE_FORMAT",
            "AXOL_BUNDLE_VERSION",
            "AXOL_BUNDLE_PLATFORM",
            "AXOL_BUNDLE_PYTHON",
            "AXOL_BUNDLE_EXTRAS",
        ):
            self.assertIn(f"\n{key}=", f"\n{env}")
            self.assertIn(f"bundle_value {key}", INSTALLER)
        self.assertIn(f"AXOL_BUNDLE_FORMAT={ob.FORMAT_VERSION}\n", env)
        self.assertIn(
            f'[ "$(bundle_value AXOL_BUNDLE_FORMAT)" = "{ob.FORMAT_VERSION}" ]',
            INSTALLER,
        )

    def test_damaged_file_is_refused(self) -> None:
        self.stage_bundle()
        (self.stage / ob.RT_BINARY).write_text("tampered")
        bundle = ob.load(self.stage)
        with self.assertRaisesRegex(ob.BundleError, "damaged"):
            bundle.verify()
        with self.assertRaisesRegex(ob.BundleError, "damaged"):
            bundle.rt_binary("v1.2.3")

    def test_other_platform_is_refused_with_the_right_download(self) -> None:
        self.stage_bundle()
        bundle = ob.load(self.stage)
        other = ob.Bundle(**{**bundle.__dict__, "platform": "linux-riscv64"})
        with self.assertRaisesRegex(
            ob.BundleError, ob.bundle_name("1.2.3", ob.host_platform())
        ):
            other.check_host()

    def _manifest(self, **overrides: object) -> dict[str, object]:
        data: dict[str, object] = {
            "format": ob.FORMAT_VERSION,
            "package": ob.PACKAGE,
            "version": "1.2.3",
            "platform": "linux-aarch64",
            "python": "3.13",
            "extras": "lerobot,sim",
            "files": {"wheels/a.whl": "0" * 64},
        }
        data.update(overrides)
        return data

    def test_paths_that_escape_the_bundle_are_rejected(self) -> None:
        for bad in ("../etc/passwd", "/etc/passwd", "wheels/../../x", "etc/x", "a\\b"):
            with self.subTest(path=bad), self.assertRaises(ob.BundleError):
                ob.parse_manifest(self.stage, self._manifest(files={bad: "0" * 64}))

    def test_malformed_manifests_are_rejected(self) -> None:
        for overrides in (
            {"format": 99},
            {"package": "other"},
            {"version": "1.2.3; rm -rf /"},
            {"files": {}},
            {"files": {"wheels/a.whl": "nothex"}},
            {"sources": {"libsurvive": {"path": "sources/x.bundle", "ref": "abc"}}},
        ):
            with self.subTest(**{k: str(v) for k, v in overrides.items()}):
                with self.assertRaises(ob.BundleError):
                    ob.parse_manifest(self.stage, self._manifest(**overrides))

    def test_active_is_none_online_and_loud_when_broken(self) -> None:
        self.assertIsNone(ob.active())
        os.environ[ob.ENV] = str(self.tmp / "missing")
        with self.assertRaisesRegex(ob.BundleError, "no manifest.json"):
            ob.active()


class BuildScriptTest(_BundleCase):
    def test_tar_is_rooted_owned_by_root_and_keeps_modes(self) -> None:
        self.stage_bundle()
        tar_path = self.tmp / "b.tar"
        build.write_tar(self.stage, tar_path, mtime=1_700_000_000)
        with tarfile.open(tar_path) as tar:
            members = tar.getmembers()
        top = self.stage.name
        self.assertTrue(
            all(m.name == top or m.name.startswith(f"{top}/") for m in members)
        )
        self.assertTrue(all(m.uid == 0 and m.gid == 0 for m in members))
        self.assertTrue(all(m.mtime == 1_700_000_000 for m in members))
        modes = {m.name.removeprefix(f"{top}/"): m.mode for m in members}
        self.assertEqual(modes[ob.RT_BINARY], 0o755)
        self.assertEqual(modes[ob.MANIFEST], 0o644)
        # Reproducible: the same stage gives byte-identical archives.
        again = self.tmp / "c.tar"
        build.write_tar(self.stage, again, mtime=1_700_000_000)
        self.assertEqual(ob.sha256_file(tar_path), ob.sha256_file(again))

    def test_wheel_without_the_web_ui_is_refused(self) -> None:
        wheel = self.tmp / "almond_axol-1.2.3-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("almond_axol/__init__.py", "")
        with self.assertRaisesRegex(SystemExit, "web UI"):
            build.check_wheel(wheel)
        with zipfile.ZipFile(wheel, "a") as archive:
            archive.writestr(build.UI_MARKER, "<!doctype html>")
        self.assertEqual(build.check_wheel(wheel), "1.2.3")

    def test_reads_the_pins_the_provision_steps_use(self) -> None:
        modules = {"libsurvive": tracker_install, "zed-gstreamer": build_zed}
        for name, relpath in build.PINNED_SOURCES:
            with self.subTest(source=name):
                module = modules[name]
                self.assertEqual(
                    build.module_constant(relpath, "_PINNED_REF"), module._PINNED_REF
                )
                self.assertEqual(
                    build.module_constant(relpath, "_REPO_URL"), module._REPO_URL
                )
                source = Path(module.__file__).read_text()
                self.assertIn(
                    f'offline_bundle.git_remote("{name}", _PINNED_REF)', source
                )

    def test_install_settings_match_the_installer_and_updater(self) -> None:
        from almond_axol.serve import update

        extras = build.module_constant("almond_axol/serve/update.py", "_EXTRAS")
        python = build.module_constant("almond_axol/serve/update.py", "_PYTHON_VERSION")
        self.assertEqual((extras, python), (update._EXTRAS, update._PYTHON_VERSION))
        self.assertIn(f'EXTRAS="{extras}"', INSTALLER)
        self.assertIn(f'PYTHON_VERSION="{python}"', INSTALLER)
        self.assertEqual(
            build.module_constant("almond_axol/cli/gst/install.py", "_PYGOBJECT_SPEC"),
            gst_install._PYGOBJECT_SPEC,
        )


class RtInstallOfflineTest(_BundleCase):
    def setUp(self) -> None:
        super().setUp()
        os.environ.pop(rt_install._SOURCE_ENV, None)
        self.dest_dir = self.tmp / "bin"
        os.environ["UV_TOOL_BIN_DIR"] = str(self.dest_dir)
        for name, value in (
            ("_repo_crate", None),
            ("_source_ref", (rt_install._REPO_URL, "v1.2.3")),
        ):
            p = patch.object(rt_install, name, return_value=value)
            p.start()
            self.addCleanup(p.stop)
        for name in ("_grant_realtime",):
            p = patch.object(rt_install, name)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(rt_install, "_SRC_CACHE", self.tmp / "src")
        p.start()
        self.addCleanup(p.stop)

    def test_installs_the_prebuilt_binary_without_building(self) -> None:
        self.stage_bundle()
        self.activate()
        with (
            patch.object(rt_install, "_ensure_toolchain") as toolchain,
            patch.object(rt_install, "_fetch_crate") as fetch,
        ):
            rt_install.run()
        toolchain.assert_not_called()
        fetch.assert_not_called()
        dest = self.dest_dir / "axol-rt"
        self.assertEqual(dest.read_bytes(), (self.stage / ob.RT_BINARY).read_bytes())
        self.assertTrue(os.access(dest, os.X_OK))
        self.assertEqual(rt_install._stamp_for(dest).read_text().strip(), "v1.2.3")

    def test_bundle_for_another_release_builds_from_source(self) -> None:
        self.stage_bundle()
        self.activate()
        with (
            patch.object(
                rt_install, "_source_ref", return_value=(rt_install._REPO_URL, "v9.9.9")
            ),
            patch.object(
                rt_install, "_ensure_toolchain", side_effect=RuntimeError("offline")
            ),
            self.assertRaisesRegex(RuntimeError, "offline"),
        ):
            rt_install.run()

    def test_explicit_source_override_still_builds(self) -> None:
        self.stage_bundle()
        self.activate()
        os.environ[rt_install._SOURCE_ENV] = f"{rt_install._REPO_URL}@main"
        self.assertIsNone(rt_install._bundled_binary("v1.2.3"))


class PyGObjectOfflineTest(_BundleCase):
    def test_installs_from_the_bundle_with_no_index(self) -> None:
        self.stage_bundle()
        self.activate()
        with (
            patch.object(gst_install.shutil, "which", return_value="/usr/bin/uv"),
            patch.object(gst_install, "_run", return_value=True) as run,
        ):
            self.assertTrue(gst_install._pip_install_pygobject())
        (cmd,) = run.call_args.args
        self.assertEqual(cmd[:3], ["/usr/bin/uv", "pip", "install"])
        for flag in ("--offline", "--no-index"):
            self.assertIn(flag, cmd)
        self.assertEqual(
            cmd[cmd.index("--find-links") + 1], str(self.stage / ob.PYGOBJECT_DIR)
        )
        self.assertEqual(cmd[cmd.index("--python") + 1], sys.executable)
        self.assertEqual(cmd[-1], gst_install._PYGOBJECT_SPEC)

    def test_online_install_is_unchanged(self) -> None:
        with (
            patch.object(gst_install.shutil, "which", return_value="/usr/bin/uv"),
            patch.object(gst_install, "_run", return_value=True) as run,
        ):
            self.assertTrue(gst_install._pip_install_pygobject())
        (cmd,) = run.call_args.args
        self.assertNotIn("--offline", cmd)


@unittest.skipIf(shutil.which("git") is None, "needs git")
class GitSourceOfflineTest(_BundleCase):
    """A real git bundle stands in for each pinned GitHub repository."""

    def setUp(self) -> None:
        super().setUp()
        upstream = self.tmp / "upstream"
        upstream.mkdir()
        _git("init", "--quiet", str(upstream))
        (upstream / "README").write_text("pinned\n")
        _git("add", "README", cwd=upstream)
        _git("commit", "--quiet", "-m", "pinned", cwd=upstream)
        self.ref = _git("rev-parse", "HEAD", cwd=upstream)
        (upstream / "README").write_text("newer\n")
        _git("commit", "--quiet", "-am", "newer", cwd=upstream)
        sources = {}
        (self.stage / ob.SOURCES_DIR).mkdir()
        for name in ("libsurvive", "zed-gstreamer"):
            relpath = f"{ob.SOURCES_DIR}/{name}.bundle"
            build.bundle_git_source(str(upstream), self.ref, self.stage / relpath)
            sources[name] = {"path": relpath, "ref": self.ref}
        self.stage_bundle(sources)
        self.activate()

    def _check_checkout(self, src: Path, repo_url: str) -> None:
        self.assertEqual(_git("rev-parse", "HEAD", cwd=src), self.ref)
        self.assertEqual((src / "README").read_text(), "pinned\n")
        self.assertEqual(_git("remote", "get-url", "origin", cwd=src), repo_url)

    def test_libsurvive_clones_from_the_bundle(self) -> None:
        src = self.tmp / "libsurvive"
        with (
            patch.object(tracker_install, "_PINNED_REF", self.ref),
            patch.object(
                tracker_install, "_REPO_URL", "https://invalid.example/libsurvive.git"
            ),
            patch.object(tracker_install, "_apply_patches", return_value=True),
        ):
            self.assertTrue(tracker_install._sync_source(src))
        self._check_checkout(src, "https://invalid.example/libsurvive.git")

    def test_zed_gstreamer_clones_from_the_bundle(self) -> None:
        src = self.tmp / "zed-gstreamer"
        with (
            patch.object(build_zed, "_PINNED_REF", self.ref),
            patch.object(
                build_zed, "_REPO_URL", "https://invalid.example/zed-gstreamer.git"
            ),
        ):
            self.assertTrue(build_zed._sync_source(src))
        self._check_checkout(src, "https://invalid.example/zed-gstreamer.git")

    def test_existing_clone_fetches_a_new_pin_from_the_bundle(self) -> None:
        src = self.tmp / "zed-gstreamer"
        _git("init", "--quiet", str(src))
        _git(
            "remote",
            "add",
            "origin",
            "https://invalid.example/zed-gstreamer.git",
            cwd=src,
        )
        with patch.object(build_zed, "_PINNED_REF", self.ref):
            self.assertTrue(build_zed._sync_source(src))
        self.assertEqual(_git("rev-parse", "HEAD", cwd=src), self.ref)

    def test_bundle_with_another_pin_falls_back_to_the_network_remote(self) -> None:
        self.assertIsNone(ob.git_remote("libsurvive", "0" * 40))
        self.assertIsNone(ob.git_remote("unknown", self.ref))


class InstallerBundleModeTest(unittest.TestCase):
    def test_script_parses(self) -> None:
        subprocess.run(
            ["bash", "-n", str(REPO / "web" / "app" / "public" / "install")], check=True
        )

    def test_installs_from_bundle_wheels_with_find_links(self) -> None:
        self.assertIn(
            'UV_SOURCE_ARGS=(--offline --no-index --find-links "${BUNDLE}/wheels")',
            INSTALLER,
        )
        self.assertIn('${UV_SOURCE_ARGS[@]+"${UV_SOURCE_ARGS[@]}"}', INSTALLER)

    def test_checks_the_bundle_before_installing(self) -> None:
        verify = INSTALLER.index("sha256sum --quiet --strict -c SHA256SUMS")
        self.assertLess(verify, INSTALLER.index('"${UV}" tool install'))

    def test_provision_takes_downloads_from_the_bundle(self) -> None:
        export = INSTALLER.index(f'export {ob.ENV}="${{BUNDLE}}"')
        provision = INSTALLER.index(
            '"${BIN_DIR}/axol" provision --require-rt --no-reboot'
        )
        self.assertLess(export, provision)

    def test_online_install_needs_no_bundle(self) -> None:
        # Piped from curl there is no script file, so only --bundle selects it.
        self.assertIn('SELF="${BASH_SOURCE[0]:-}"', INSTALLER)
        self.assertIn(
            '[ -f "${SELF}" ] && [ -f "$(dirname "${SELF}")/manifest.json" ]', INSTALLER
        )


class ManifestFileTest(_BundleCase):
    def test_manifest_json_round_trips(self) -> None:
        self.stage_bundle()
        data = json.loads((self.stage / ob.MANIFEST).read_text())
        self.assertEqual(data["format"], ob.FORMAT_VERSION)
        self.assertEqual(
            ob.parse_manifest(self.stage, data).files, ob.load(self.stage).files
        )


if __name__ == "__main__":
    unittest.main()
