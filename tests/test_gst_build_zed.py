from __future__ import annotations

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
                patch.object(build_zed, "_element_installed", return_value=False),  # noqa: SLF001
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


if __name__ == "__main__":
    unittest.main()
