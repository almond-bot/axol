from __future__ import annotations

import hashlib
import io
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from almond_axol.cli.zed import download, driver, install
from almond_axol.utils import state_files


class _Response:
    def __init__(
        self,
        body: bytes,
        *,
        url: str = "https://downloads.example/artifact",
        declared_length: str | None = None,
    ) -> None:
        self._body = io.BytesIO(body)
        self._url = url
        self.headers = (
            {} if declared_length is None else {"Content-Length": declared_length}
        )

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def geturl(self) -> str:
        return self._url


def _wheel(
    path: Path,
    *,
    version: str = "5.0",
    tag: str = "cp313-cp313-linux_aarch64",
    requirement: str = "numpy<3,>=2",
    unsafe_name: str | None = None,
) -> None:
    dist_info = f"pyzed-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("pyzed/sl.cpython-313-aarch64-linux-gnu.so", b"native")
        archive.writestr(
            f"{dist_info}/METADATA",
            "\n".join(
                [
                    "Metadata-Version: 2.4",
                    "Name: pyzed",
                    f"Version: {version}",
                    f"Requires-Dist: {requirement}",
                    "",
                ]
            ),
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            "\n".join(
                [
                    "Wheel-Version: 1.0",
                    "Root-Is-Purelib: false",
                    f"Tag: {tag}",
                    "",
                ]
            ),
        )
        archive.writestr(f"{dist_info}/RECORD", "")
        if unsafe_name is not None:
            archive.writestr(unsafe_name, b"unsafe")


class AtomicDownloadTests(unittest.TestCase):
    def test_exclusive_temp_does_not_follow_predictable_part_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "artifact.bin"
            victim = root / "victim"
            victim.write_bytes(b"keep")
            destination.with_suffix(".part").symlink_to(victim)
            validate = Mock(
                side_effect=lambda path: self.assertEqual(path.read_bytes(), b"ok")
            )

            with patch.object(
                download.urllib.request,
                "urlopen",
                return_value=_Response(b"ok"),
            ):
                download.atomic_https_download(
                    "https://vendor.example/artifact",
                    destination,
                    max_bytes=16,
                    validate=validate,
                )

            self.assertEqual(destination.read_bytes(), b"ok")
            self.assertEqual(victim.read_bytes(), b"keep")
            validate.assert_called_once()
            self.assertNotEqual(
                validate.call_args.args[0], destination.with_suffix(".part")
            )

    def test_non_https_redirect_fails_without_publishing_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "artifact.bin"
            with (
                patch.object(
                    download.urllib.request,
                    "urlopen",
                    return_value=_Response(b"bytes", url="http://mirror/artifact"),
                ),
                self.assertRaisesRegex(RuntimeError, "redirected away from HTTPS"),
            ):
                download.atomic_https_download(
                    "https://vendor.example/artifact",
                    destination,
                    max_bytes=16,
                    validate=Mock(),
                )

            self.assertFalse(destination.exists())
            self.assertEqual(list(destination.parent.glob("*.part")), [])

    def test_oversized_response_fails_before_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "artifact.bin"
            validate = Mock()
            with (
                patch.object(
                    download.urllib.request,
                    "urlopen",
                    return_value=_Response(b"012345", declared_length="6"),
                ),
                self.assertRaisesRegex(RuntimeError, "permitted size"),
            ):
                download.atomic_https_download(
                    "https://vendor.example/artifact",
                    destination,
                    max_bytes=5,
                    validate=validate,
                )

            validate.assert_not_called()
            self.assertFalse(destination.exists())


class DriverArtifactTests(unittest.TestCase):
    def test_exact_digest_and_package_metadata_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "driver.deb"
            artifact.write_bytes(b"reviewed artifact")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            fields = {
                "Package": driver._PACKAGE,  # noqa: SLF001
                "Version": driver._DEB_VERSION,  # noqa: SLF001
                "Architecture": driver._DEB_ARCHITECTURE,  # noqa: SLF001
            }
            with (
                patch.object(driver, "_DEB_SHA256", digest),
                patch.object(
                    driver,
                    "_deb_field",
                    side_effect=lambda _path, field: fields[field],
                ) as deb_field,
            ):
                driver._validate_deb(artifact)  # noqa: SLF001

            self.assertEqual(deb_field.call_count, 3)

    def test_changed_vendor_bytes_fail_before_metadata_and_are_not_cached(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            artifact = cache / driver._DEB_URL.rsplit("/", 1)[-1]  # noqa: SLF001
            artifact.write_bytes(b"changed upstream")
            with (
                patch.object(driver, "_CACHE_DIR", cache),
                patch.object(driver, "_deb_field") as deb_field,
                self.assertRaisesRegex(RuntimeError, "reviewed artifact"),
            ):
                driver._download_deb()  # noqa: SLF001

            deb_field.assert_not_called()
            self.assertFalse(artifact.exists())

    def test_operator_cache_is_copied_and_revalidated_before_root_consumes_it(
        self,
    ) -> None:
        artifact_name = driver._DEB_URL.rsplit("/", 1)[-1]  # noqa: SLF001
        cached = Path("/operator/.almond/drivers") / artifact_name
        run_root = Mock()
        validate = Mock()
        with (
            patch.object(driver.os, "geteuid", return_value=1000),
            patch.object(driver, "run_root", run_root),
            patch.object(driver, "_validate_deb", validate),
        ):
            staged = driver._stage_deb_for_root(cached)  # noqa: SLF001

        self.assertEqual(staged, driver._ROOT_CACHE_DIR / cached.name)  # noqa: SLF001
        self.assertEqual(run_root.call_count, 1)
        self.assertEqual(run_root.call_args_list[0].kwargs, {"check": True})
        command = run_root.call_args.args[0]
        self.assertEqual(
            command[1:4],
            ["-m", "almond_axol.cli.zed.driver", "--stage-reviewed-deb"],
        )
        self.assertEqual(command[-1], str(cached))
        validate.assert_not_called()

    def test_root_stage_never_follows_operator_symlink_or_leaves_exfiltration(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_name = driver._DEB_URL.rsplit("/", 1)[-1]  # noqa: SLF001
            operator_cache = root / "operator"
            operator_cache.mkdir()
            victim = root / "root-secret"
            victim.write_bytes(b"root-only bytes")
            source = operator_cache / artifact_name
            source.symlink_to(victim)
            root_cache = root / "root-cache"

            with (
                patch.object(driver, "_ROOT_CACHE_DIR", root_cache),
                patch.object(driver.os, "geteuid", return_value=0),
                patch.object(state_files.os, "fchown"),
                self.assertRaises(OSError),
            ):
                driver._stage_deb_as_root(source)  # noqa: SLF001

            self.assertEqual(victim.read_bytes(), b"root-only bytes")
            self.assertEqual(stat.S_IMODE(root_cache.stat().st_mode), 0o700)
            self.assertFalse((root_cache / artifact_name).exists())

    def test_failed_root_stage_validation_removes_private_cached_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_name = driver._DEB_URL.rsplit("/", 1)[-1]  # noqa: SLF001
            source = root / artifact_name
            source.write_bytes(b"changed bytes")
            root_cache = root / "root-cache"

            with (
                patch.object(driver, "_ROOT_CACHE_DIR", root_cache),
                patch.object(driver.os, "geteuid", return_value=0),
                patch.object(state_files.os, "fchown"),
                patch.object(
                    driver,
                    "_validate_deb",
                    side_effect=RuntimeError("bad digest"),
                ),
                self.assertRaisesRegex(RuntimeError, "bad digest"),
            ):
                driver._stage_deb_as_root(source)  # noqa: SLF001

            self.assertEqual(stat.S_IMODE(root_cache.stat().st_mode), 0o700)
            self.assertFalse((root_cache / artifact_name).exists())

    def test_failed_root_stage_stops_before_factory_driver_removal(self) -> None:
        with (
            patch.object(driver, "_download_deb", return_value=Path("driver.deb")),
            patch.object(
                driver,
                "_stage_deb_for_root",
                side_effect=RuntimeError("invalid staged bytes"),
            ),
            patch.object(driver, "run_root") as run_root,
            self.assertRaisesRegex(RuntimeError, "invalid staged bytes"),
        ):
            driver._upgrade()  # noqa: SLF001

        run_root.assert_not_called()


class PyzedWheelTests(unittest.TestCase):
    def test_valid_vendor_wheel_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "pyzed.whl"
            _wheel(wheel)
            install._validate_wheel(  # noqa: SLF001
                wheel,
                sdk_version="5.0",
                python_tag="313",
                architecture="aarch64",
            )

    def test_archive_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "pyzed.whl"
            _wheel(wheel, unsafe_name="../root-owned")
            with self.assertRaisesRegex(RuntimeError, "unsafe archive path"):
                install._validate_wheel(  # noqa: SLF001
                    wheel,
                    sdk_version="5.0",
                    python_tag="313",
                    architecture="aarch64",
                )

    def test_unexpected_dependency_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "pyzed.whl"
            _wheel(wheel, requirement="surprise-package>=1")
            with self.assertRaisesRegex(RuntimeError, "unexpected dependencies"):
                install._validate_wheel(  # noqa: SLF001
                    wheel,
                    sdk_version="5.0",
                    python_tag="313",
                    architecture="aarch64",
                )

    def test_wrong_version_or_target_tag_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "pyzed.whl"
            _wheel(wheel, version="4.2", tag="cp312-cp312-linux_x86_64")
            with self.assertRaisesRegex(RuntimeError, "metadata directory"):
                install._validate_wheel(  # noqa: SLF001
                    wheel,
                    sdk_version="5.0",
                    python_tag="313",
                    architecture="aarch64",
                )

    def test_invalid_cached_wheel_is_removed_before_uv_install(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / "pyzed-5.0-cp313-cp313-linux_aarch64.whl"
            wheel.write_bytes(b"not a wheel")
            check_call = Mock()
            with (
                patch.object(install, "_ZED_INCLUDE", root),
                patch.object(install, "_CACHE_DIR", root),
                patch.object(install, "_sdk_version", return_value=("5", "0")),
                patch.object(install, "_pyzed_installed", return_value=False),
                patch.object(install.platform, "machine", return_value="aarch64"),
                patch.object(install.subprocess, "check_call", check_call),
                patch.object(
                    install.sys, "version_info", SimpleNamespace(major=3, minor=13)
                ),
                self.assertRaisesRegex(RuntimeError, "valid ZIP archive"),
            ):
                install.run(SimpleNamespace(force=True))

            self.assertFalse(wheel.exists())
            check_call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
