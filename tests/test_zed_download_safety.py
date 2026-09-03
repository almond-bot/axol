from __future__ import annotations

import dataclasses
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


_DUO = driver._VARIANTS_BY_PACKAGE["stereolabs-zedbox-duo"]  # noqa: SLF001
_MINI = driver._VARIANTS_BY_PACKAGE["stereolabs-zedbox-mini"]  # noqa: SLF001


class DriverVariantTests(unittest.TestCase):
    """Every ZED carrier we ship must have its GMSL driver kept in step."""

    def test_pins_cover_duo_and_mini_consistently(self) -> None:
        self.assertEqual(
            {v.package for v in driver._VARIANTS},  # noqa: SLF001
            {"stereolabs-zedbox-duo", "stereolabs-zedbox-mini"},
        )
        for variant in driver._VARIANTS:  # noqa: SLF001
            with self.subTest(variant.package):
                self.assertRegex(variant.sha256, r"^[0-9a-f]{64}$")
                self.assertTrue(variant.deb_version.startswith(variant.target_version))
                self.assertEqual(
                    variant.deb_name,
                    f"{variant.package}_{variant.deb_version}_"
                    f"{driver._DEB_ARCHITECTURE}.deb",  # noqa: SLF001
                )
                self.assertIn(
                    f"L4T{driver._L4T_RELEASE}.{driver._L4T_REVISION_MAJOR}.",  # noqa: SLF001
                    variant.deb_version,
                )
                self.assertIn(
                    f"/R{driver._L4T_RELEASE}.{driver._L4T_REVISION_MAJOR}/",  # noqa: SLF001
                    variant.url,
                )

    @staticmethod
    def _dpkg_query(lines: str) -> Mock:
        return Mock(return_value=SimpleNamespace(returncode=0, stdout=lines))

    def test_only_installed_stereolabs_packages_count(self) -> None:
        listing = (
            "stereolabs-zedbox-mini installed 1.4.0-SL-MAX9296-ZEDBOX-MINI-L4T36.4.0\n"
            "stereolabs-zedbox-duo config-files 1.3.0-LI-MAX96712-ZEDBOX-L4T36.4.0\n"
        )
        with patch.object(driver.subprocess, "run", self._dpkg_query(listing)) as run:
            installed = driver._installed_driver_packages()  # noqa: SLF001

        self.assertEqual(
            installed,
            {"stereolabs-zedbox-mini": "1.4.0-SL-MAX9296-ZEDBOX-MINI-L4T36.4.0"},
        )
        self.assertEqual(run.call_args.args[0][-1], "stereolabs-zed*")

    def test_outdated_mini_driver_is_upgraded(self) -> None:
        # The customer case: a ZED Box Mini flashed with driver 1.4.0 running
        # SDK 5.4.1. The Duo-only check used to return False silently here.
        upgrade = Mock()
        with (
            patch.object(
                driver,
                "_installed_driver_packages",
                return_value={
                    "stereolabs-zedbox-mini": "1.4.0-SL-MAX9296-ZEDBOX-MINI-L4T36.4.0"
                },
            ),
            patch.object(driver, "_is_older", return_value=True),
            patch.object(driver, "_l4t_matches", return_value=True),
            patch.object(driver, "_upgrade", upgrade),
            patch.object(driver.sys, "stdout") as stdout,
        ):
            self.assertTrue(driver.ensure_driver())

        upgrade.assert_called_once_with(_MINI)
        printed = "".join(c.args[0] for c in stdout.write.call_args_list)
        self.assertIn("REBOOT REQUIRED: stereolabs-zedbox-mini 1.4.3", printed)

    def test_current_duo_driver_is_left_alone(self) -> None:
        upgrade = Mock()
        with (
            patch.object(
                driver,
                "_installed_driver_packages",
                return_value={"stereolabs-zedbox-duo": _DUO.deb_version},
            ),
            patch.object(driver, "_is_older", return_value=False),
            patch.object(driver, "_upgrade", upgrade),
            patch.object(driver.sys, "stdout"),
        ):
            self.assertFalse(driver.ensure_driver())

        upgrade.assert_not_called()

    def test_unpinned_stereolabs_driver_is_reported_not_ignored(self) -> None:
        upgrade = Mock()
        with (
            patch.object(
                driver,
                "_installed_driver_packages",
                return_value={"stereolabs-zedlink-quad": "1.2.0-L4T36.4.0"},
            ),
            patch.object(driver, "_upgrade", upgrade),
            patch.object(driver.sys, "stderr") as stderr,
        ):
            self.assertFalse(driver.ensure_driver())

        upgrade.assert_not_called()
        warned = "".join(c.args[0] for c in stderr.write.call_args_list)
        self.assertIn("stereolabs-zedlink-quad 1.2.0-L4T36.4.0", warned)
        self.assertIn("no pinned driver", warned)

    def test_not_a_zed_box_is_a_quiet_noop(self) -> None:
        with (
            patch.object(driver, "_installed_driver_packages", return_value={}),
            patch.object(driver, "_upgrade") as upgrade,
        ):
            self.assertFalse(driver.ensure_driver())
        upgrade.assert_not_called()


class DriverArtifactTests(unittest.TestCase):
    def test_exact_digest_and_package_metadata_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "driver.deb"
            artifact.write_bytes(b"reviewed artifact")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            variant = dataclasses.replace(_MINI, sha256=digest)
            fields = {
                "Package": variant.package,
                "Version": variant.deb_version,
                "Architecture": driver._DEB_ARCHITECTURE,  # noqa: SLF001
            }
            with patch.object(
                driver,
                "_deb_field",
                side_effect=lambda _path, field: fields[field],
            ) as deb_field:
                driver._validate_deb(artifact, variant)  # noqa: SLF001

            self.assertEqual(deb_field.call_count, 3)

    def test_artifact_is_validated_against_its_own_variant_pin(self) -> None:
        # Mini bytes must never be accepted under the Duo pin (or vice versa).
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / _MINI.deb_name
            artifact.write_bytes(b"mini artifact")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            mini = dataclasses.replace(_MINI, sha256=digest)
            duo = dataclasses.replace(_DUO, sha256=digest)
            fields = {
                "Package": _MINI.package,
                "Version": _MINI.deb_version,
                "Architecture": driver._DEB_ARCHITECTURE,  # noqa: SLF001
            }
            with patch.object(
                driver, "_deb_field", side_effect=lambda _path, field: fields[field]
            ):
                driver._validate_deb(artifact, mini)  # noqa: SLF001
                with self.assertRaisesRegex(RuntimeError, "Package is"):
                    driver._validate_deb(artifact, duo)  # noqa: SLF001

    def test_root_stage_rejects_unknown_artifact_names(self) -> None:
        with (
            patch.object(driver.os, "geteuid", return_value=0),
            patch.object(driver, "secure_ensure_directory") as ensure_dir,
            self.assertRaisesRegex(RuntimeError, "unexpected ZED driver artifact"),
        ):
            driver._stage_deb_as_root(Path("/operator/.almond/drivers/evil.deb"))  # noqa: SLF001
        ensure_dir.assert_not_called()

    def test_changed_vendor_bytes_fail_before_metadata_and_are_not_cached(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            artifact = cache / _DUO.deb_name
            artifact.write_bytes(b"changed upstream")
            with (
                patch.object(driver, "_CACHE_DIR", cache),
                patch.object(driver, "_deb_field") as deb_field,
                self.assertRaisesRegex(RuntimeError, "reviewed artifact"),
            ):
                driver._download_deb(_DUO)  # noqa: SLF001

            deb_field.assert_not_called()
            self.assertFalse(artifact.exists())

    def test_operator_cache_is_copied_and_revalidated_before_root_consumes_it(
        self,
    ) -> None:
        artifact_name = _DUO.deb_name
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
            artifact_name = _MINI.deb_name
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
            artifact_name = _DUO.deb_name
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
            driver._upgrade(_MINI)  # noqa: SLF001

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
