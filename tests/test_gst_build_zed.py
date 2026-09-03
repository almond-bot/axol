from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, call, patch

from almond_axol.cli.gst import build_zed, install


class ZedGstreamerBuildDependenciesTest(unittest.TestCase):
    def test_build_is_noop_off_jetson(self) -> None:
        sync_source = Mock()
        apt_install = Mock()
        output = StringIO()
        with (
            patch.object(build_zed, "_is_jetson", return_value=False),
            patch.object(build_zed, "_apt_install_build_deps", apt_install),
            patch.object(build_zed, "_sync_source", sync_source),  # noqa: SLF001
            redirect_stdout(output),
        ):
            build_zed.run()

        apt_install.assert_not_called()
        sync_source.assert_not_called()
        self.assertIn("Not an NVIDIA Jetson", output.getvalue())

    def test_gst_install_is_noop_off_jetson(self) -> None:
        gst_ok = Mock()
        apt_install = Mock()
        pip_install = Mock()
        output = StringIO()
        with (
            patch.object(install, "_is_jetson", return_value=False),
            patch.object(install, "_gst_ok", gst_ok),  # noqa: SLF001
            patch.object(install, "_apt_install", apt_install),  # noqa: SLF001
            patch.object(install, "_pip_install_pygobject", pip_install),  # noqa: SLF001
            redirect_stdout(output),
        ):
            install.run()

        gst_ok.assert_not_called()
        apt_install.assert_not_called()
        pip_install.assert_not_called()
        self.assertIn("Not an NVIDIA Jetson", output.getvalue())

    def test_gst_install_fails_when_attempted_stack_is_still_unready(self) -> None:
        with (
            patch.object(install, "_is_jetson", return_value=True),
            patch.object(install, "_gst_ok", side_effect=(False, False)),  # noqa: SLF001
            patch.object(install, "_apt_install", return_value=False),  # noqa: SLF001
            patch.object(install, "_pip_install_pygobject", return_value=False),  # noqa: SLF001
            self.assertRaisesRegex(SystemExit, "still unavailable"),
        ):
            install.run()

    def test_declares_zed_sdk_link_dependencies(self) -> None:
        # zed-config.cmake requires BLAS and links the unversioned libusb
        # library. Runtime-only packages do not provide those linker files.
        self.assertIn("libblas-dev", build_zed._APT_BUILD_DEPS)  # noqa: SLF001
        self.assertIn("libusb-1.0-0-dev", build_zed._APT_BUILD_DEPS)  # noqa: SLF001

    def test_apt_install_uses_all_declared_dependencies(self) -> None:
        succeeded = subprocess.CompletedProcess([], 0, "", "")
        run_root = Mock(return_value=succeeded)
        with (
            patch.object(build_zed.shutil, "which", return_value="/usr/bin/apt-get"),
            patch.object(build_zed, "prime_sudo", return_value=True),
            patch.object(build_zed, "run_root", run_root),
        ):
            result = build_zed._apt_install_build_deps()  # noqa: SLF001

        self.assertTrue(result)
        self.assertEqual(
            run_root.call_args_list,
            [
                call(["apt-get", "update"]),
                call(
                    [
                        "apt-get",
                        "install",
                        "-y",
                        *build_zed._APT_BUILD_DEPS,  # noqa: SLF001
                    ]
                ),
            ],
        )

    def test_apt_install_failure_stops_before_fetch_and_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sdk = Path(directory) / "zed"
            sdk.mkdir()
            src = Path(directory) / "source"
            sync_source = Mock()
            output = StringIO()
            with (
                patch.object(build_zed, "_is_jetson", return_value=True),
                patch.object(build_zed, "_ZED_SDK", sdk),  # noqa: SLF001
                patch.object(build_zed, "_src_dir", return_value=src),  # noqa: SLF001
                patch.object(build_zed, "_installed_plugins_ready", return_value=False),  # noqa: SLF001
                patch.object(build_zed, "_apt_install_build_deps", return_value=False),
                patch.object(build_zed, "_sync_source", sync_source),  # noqa: SLF001
                redirect_stdout(output),
                self.assertRaisesRegex(SystemExit, "Could not install"),
            ):
                build_zed.run()

        sync_source.assert_not_called()

    def test_build_is_forced_to_one_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            commands: list[list[str]] = []

            def record(command: list[str], **_kwargs: object) -> bool:
                commands.append(command)
                return True

            installed = subprocess.CompletedProcess([], 0, "", "")
            with (
                patch.object(build_zed.shutil, "which", return_value="/usr/bin/cmake"),
                patch.object(build_zed, "_run", side_effect=record),  # noqa: SLF001
                patch.object(build_zed, "run_root", return_value=installed),
            ):
                self.assertTrue(build_zed._build_and_install(source))  # noqa: SLF001

        self.assertEqual(
            commands[1],
            ["/usr/bin/cmake", "--build", str(source / "build"), "-j", "1"],
        )

    def test_rebuild_discards_the_previous_build_tree(self) -> None:
        # Upstream ignores ``*build*`` so ``git clean`` keeps it; a stale CMake
        # cache would otherwise describe the SDK the old plugins linked.
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            stale = source / "build" / "CMakeCache.txt"
            stale.parent.mkdir()
            stale.write_text("ZED_DIR=/usr/local/zed (5.0.5)\n", encoding="utf-8")
            installed = subprocess.CompletedProcess([], 0, "", "")
            with (
                patch.object(build_zed.shutil, "which", return_value="/usr/bin/cmake"),
                patch.object(build_zed, "_run", return_value=True),  # noqa: SLF001
                patch.object(build_zed, "run_root", return_value=installed),
            ):
                self.assertTrue(build_zed._build_and_install(source))  # noqa: SLF001

            self.assertTrue((source / "build").is_dir())
            self.assertFalse(stale.exists())


class ZedGstreamerInstalledIntegrityTest(unittest.TestCase):
    @staticmethod
    def _write_artifacts(root: Path) -> dict[str, Path]:
        mono = root / "libgstzedxonesrc.so"
        stereo = root / "libgstzedsrc.so"
        mono.write_bytes(b"patched mono")
        stereo.write_bytes(b"patched stereo")
        return {"zedxonesrc": mono, "zedsrc": stereo}

    @staticmethod
    def _artifact_records(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
        return {
            element: {
                "path": str(path),
                "sha256": build_zed._file_sha256(path),  # noqa: SLF001
            }
            for element, path in paths.items()
        }

    def test_gst_inspect_filename_is_canonicalized_and_root_checked(self) -> None:
        inspected = subprocess.CompletedProcess(
            [],
            0,
            "Plugin Details:\n  Filename                 /plugin/link.so\n",
            "",
        )
        with (
            patch.object(
                build_zed.shutil, "which", return_value="/usr/bin/gst-inspect-1.0"
            ),
            patch.object(build_zed.subprocess, "run", return_value=inspected),
            patch.object(
                build_zed,
                "_root_controlled_canonical_file",
                return_value=Path("/usr/lib/gstreamer/libgstzed.so"),
            ) as root_check,
        ):
            result = build_zed._inspect_element_artifact("zedsrc")  # noqa: SLF001

        self.assertEqual(result, Path("/usr/lib/gstreamer/libgstzed.so"))
        root_check.assert_called_once_with(
            Path("/plugin/link.so"), allow_canonical_alias=True
        )

    def test_artifact_must_actually_load_not_just_be_registered(self) -> None:
        # The registry cache keeps listing an element whose plugin file is
        # unchanged even after the ZED SDK it links was upgraded underneath it;
        # ``gst-inspect <element>`` then still exits 0 and prints the filename.
        registered = subprocess.CompletedProcess(
            [],
            0,
            "Plugin Details:\n  Filename                 /plugin/libgstzedxonesrc.so\n",
            "",
        )
        unloadable = subprocess.CompletedProcess(
            [],
            255,
            "",
            "Could not load plugin file: Opening module failed: "
            "/plugin/libgstzedxonesrc.so: undefined symbol: "
            "_ZN2sl9CameraOne8isOpenedEv\n",
        )
        artifact = Path("/plugin/libgstzedxonesrc.so")
        with (
            patch.object(
                build_zed.shutil, "which", return_value="/usr/bin/gst-inspect-1.0"
            ),
            patch.object(
                build_zed.subprocess, "run", side_effect=(registered, unloadable)
            ) as run,
            patch.object(
                build_zed, "_root_controlled_canonical_file", return_value=artifact
            ),
            self.assertLogs(build_zed._logger, level="WARNING") as logs,  # noqa: SLF001
        ):
            result = build_zed._inspect_element_artifact("zedxonesrc")  # noqa: SLF001

        self.assertIsNone(result)
        # The second gst-inspect targets the plugin *file*, which forces dlopen.
        self.assertEqual(run.call_args_list[1].args[0][1], str(artifact))
        self.assertIn("_ZN2sl9CameraOne8isOpenedEv", "\n".join(logs.output))

    def test_loadable_artifact_is_inspected_as_a_file_too(self) -> None:
        registered = subprocess.CompletedProcess(
            [], 0, "Plugin Details:\n  Filename  /plugin/libgstzedsrc.so\n", ""
        )
        loadable = subprocess.CompletedProcess([], 0, "Plugin Details:\n", "")
        artifact = Path("/plugin/libgstzedsrc.so")
        with (
            patch.object(
                build_zed.shutil, "which", return_value="/usr/bin/gst-inspect-1.0"
            ),
            patch.object(
                build_zed.subprocess, "run", side_effect=(registered, loadable)
            ) as run,
            patch.object(
                build_zed, "_root_controlled_canonical_file", return_value=artifact
            ),
        ):
            result = build_zed._inspect_element_artifact("zedsrc")  # noqa: SLF001

        self.assertEqual(result, artifact)
        self.assertEqual(
            [c.args[0][1] for c in run.call_args_list], ["zedsrc", str(artifact)]
        )

    def test_sdk_version_is_read_from_the_installed_headers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            header = Path(directory) / "Camera.hpp"
            header.write_text(
                "#define ZED_SDK_MAJOR_VERSION 5\n"
                "#define ZED_SDK_MINOR_VERSION 4\n"
                "#define ZED_SDK_PATCH_VERSION 1\n",
                encoding="utf-8",
            )
            missing = Path(directory) / "absent.hpp"
            with patch.object(build_zed, "_ZED_SDK_VERSION_HEADERS", (missing, header)):
                self.assertEqual(build_zed._zed_sdk_version(), "5.4.1")  # noqa: SLF001
            with patch.object(build_zed, "_ZED_SDK_VERSION_HEADERS", (missing,)):
                self.assertIsNone(build_zed._zed_sdk_version())  # noqa: SLF001

    def test_operator_controlled_artifact_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "libgstzed.so"
            artifact.write_bytes(b"untrusted")
            self.assertIsNone(  # noqa: SLF001
                build_zed._root_controlled_canonical_file(
                    artifact, allow_canonical_alias=True
                )
            )

    def test_readiness_requires_both_current_paths_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._write_artifacts(root)
            manifest = root / "manifest.json"
            manifest.write_text(
                build_zed._manifest_payload(self._artifact_records(paths)),  # noqa: SLF001
                encoding="utf-8",
            )

            with (
                patch.object(
                    build_zed,
                    "_root_controlled_canonical_file",
                    return_value=manifest,
                ),
                patch.object(
                    build_zed,
                    "_inspect_element_artifact",
                    side_effect=lambda element: paths[element],
                ),
            ):
                self.assertTrue(build_zed._installed_plugins_ready(manifest))  # noqa: SLF001

                paths["zedsrc"].write_bytes(b"overwritten stock plugin")
                self.assertFalse(build_zed._installed_plugins_ready(manifest))  # noqa: SLF001

    def test_readiness_rejects_plugins_built_against_another_sdk(self) -> None:
        # Byte-identical plugins are stale the moment the ZED SDK they link is
        # upgraded in place (the customer failure: SDK 5.0.5 -> 5.4.1 left
        # ``undefined symbol`` plugins that the build still called installed).
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._write_artifacts(root)
            manifest = root / "manifest.json"
            with patch.object(build_zed, "_zed_sdk_version", return_value="5.0.5"):
                manifest.write_text(
                    build_zed._manifest_payload(self._artifact_records(paths)),  # noqa: SLF001
                    encoding="utf-8",
                )
            self.assertIn('"zedSdk": "5.0.5"', manifest.read_text(encoding="utf-8"))

            with (
                patch.object(
                    build_zed,
                    "_root_controlled_canonical_file",
                    return_value=manifest,
                ),
                patch.object(
                    build_zed,
                    "_inspect_element_artifact",
                    side_effect=lambda element: paths[element],
                ),
            ):
                with patch.object(build_zed, "_zed_sdk_version", return_value="5.0.5"):
                    self.assertTrue(build_zed._installed_plugins_ready(manifest))  # noqa: SLF001
                with (
                    patch.object(build_zed, "_zed_sdk_version", return_value="5.4.1"),
                    self.assertLogs(build_zed._logger, level="INFO") as logs,  # noqa: SLF001
                ):
                    self.assertFalse(build_zed._installed_plugins_ready(manifest))  # noqa: SLF001
                self.assertIn("built against ZED SDK 5.0.5", "\n".join(logs.output))
                self.assertIn("SDK 5.4.1 is installed", "\n".join(logs.output))

    def test_legacy_manifest_without_sdk_version_rebuilds_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._write_artifacts(root)
            manifest = root / "manifest.json"
            legacy = json.loads(
                build_zed._manifest_payload(self._artifact_records(paths))  # noqa: SLF001
            )
            legacy["schema"] = 1
            del legacy["zedSdk"]
            manifest.write_text(json.dumps(legacy), encoding="utf-8")

            with (
                patch.object(
                    build_zed,
                    "_root_controlled_canonical_file",
                    return_value=manifest,
                ),
                patch.object(
                    build_zed,
                    "_inspect_element_artifact",
                    side_effect=lambda element: paths[element],
                ),
            ):
                self.assertFalse(build_zed._installed_plugins_ready(manifest))  # noqa: SLF001

    def test_readiness_rejects_missing_or_unknown_element(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._write_artifacts(root)
            records = self._artifact_records(paths)
            manifest = root / "manifest.json"
            manifest.write_text(
                build_zed._manifest_payload(records),  # noqa: SLF001
                encoding="utf-8",
            )
            unknown = root / "libgststockzedsrc.so"
            unknown.write_bytes(paths["zedsrc"].read_bytes())

            with patch.object(
                build_zed,
                "_root_controlled_canonical_file",
                return_value=manifest,
            ):
                with patch.object(
                    build_zed,
                    "_inspect_element_artifact",
                    side_effect=(paths["zedxonesrc"], None),
                ):
                    self.assertFalse(build_zed._installed_plugins_ready(manifest))  # noqa: SLF001
                with patch.object(
                    build_zed,
                    "_inspect_element_artifact",
                    side_effect=(paths["zedxonesrc"], unknown),
                ):
                    self.assertFalse(build_zed._installed_plugins_ready(manifest))  # noqa: SLF001

    def test_post_install_artifacts_must_be_in_cmake_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "build").mkdir(parents=True)
            paths = self._write_artifacts(root)
            records = self._artifact_records(paths)
            install_manifest = source / "build" / "install_manifest.txt"
            install_manifest.write_text(
                f"{paths['zedxonesrc']}\n{paths['zedsrc']}\n",
                encoding="utf-8",
            )
            self.assertTrue(  # noqa: SLF001
                build_zed._artifacts_came_from_build(source, records)
            )

            install_manifest.write_text(f"{paths['zedxonesrc']}\n", encoding="utf-8")
            self.assertFalse(  # noqa: SLF001
                build_zed._artifacts_came_from_build(source, records)
            )

    def test_partial_install_never_writes_source_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdk = root / "zed"
            sdk.mkdir()
            source = root / "source"
            publish = Mock()
            with (
                patch.object(build_zed, "_is_jetson", return_value=True),
                patch.object(build_zed, "_ZED_SDK", sdk),  # noqa: SLF001
                patch.object(build_zed, "_src_dir", return_value=source),  # noqa: SLF001
                patch.object(build_zed, "_installed_plugins_ready", return_value=False),  # noqa: SLF001
                patch.object(build_zed, "_apt_install_build_deps", return_value=True),
                patch.object(build_zed, "_sync_source", return_value=True),  # noqa: SLF001
                patch.object(build_zed, "_apply_patch", return_value=True),  # noqa: SLF001
                patch.object(build_zed, "_build_and_install", return_value=True),  # noqa: SLF001
                patch.object(build_zed, "_collect_plugin_artifacts", return_value=None),  # noqa: SLF001
                patch.object(build_zed, "_publish_machine_manifest", publish),  # noqa: SLF001
                self.assertRaisesRegex(SystemExit, "must both be visible"),
            ):
                build_zed.run()

            publish.assert_not_called()
            self.assertFalse((source / ".axol-build-stamp").exists())

    def test_manifest_publish_failure_never_writes_source_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdk = root / "zed"
            sdk.mkdir()
            source = root / "source"
            paths = self._write_artifacts(root)
            artifacts = self._artifact_records(paths)
            stamp_write = Mock()
            with (
                patch.object(build_zed, "_is_jetson", return_value=True),
                patch.object(build_zed, "_ZED_SDK", sdk),  # noqa: SLF001
                patch.object(build_zed, "_src_dir", return_value=source),  # noqa: SLF001
                patch.object(build_zed, "_installed_plugins_ready", return_value=False),  # noqa: SLF001
                patch.object(build_zed, "_apt_install_build_deps", return_value=True),
                patch.object(build_zed, "_sync_source", return_value=True),  # noqa: SLF001
                patch.object(build_zed, "_apply_patch", return_value=True),  # noqa: SLF001
                patch.object(build_zed, "_build_and_install", return_value=True),  # noqa: SLF001
                patch.object(
                    build_zed, "_collect_plugin_artifacts", return_value=artifacts
                ),  # noqa: SLF001
                patch.object(
                    build_zed, "_artifacts_came_from_build", return_value=True
                ),  # noqa: SLF001
                patch.object(
                    build_zed, "_publish_machine_manifest", return_value=False
                ),  # noqa: SLF001
                patch.object(build_zed, "secure_atomic_write_text", stamp_write),
                self.assertRaisesRegex(SystemExit, "root-owned ZED plugin manifest"),
            ):
                build_zed.run()

            stamp_write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
