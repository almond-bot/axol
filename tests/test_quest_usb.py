from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from almond_axol.utils import adb, certs


def _completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["adb"], 0, stdout, "")


class AdbReverseTunnelTest(unittest.TestCase):
    def test_both_tunnel_ports_are_forwarded_on_connect(self) -> None:
        calls: list[list[str]] = []

        def run(args: list[str], timeout: float = 10.0):
            calls.append(args)
            if args == ["devices"]:
                return _completed("List of devices attached\n1WMHH\tdevice\n")
            return _completed("")

        with (
            patch.object(adb, "_adb", return_value="/usr/bin/adb"),
            patch.object(adb, "_run", side_effect=run),
        ):
            adb.connect()

        for port in (adb.VR_PORT, adb.CONTROL_PORT):
            self.assertIn(["reverse", "--remove", f"tcp:{port}"], calls)
            self.assertIn(["reverse", f"tcp:{port}", f"tcp:{port}"], calls)

    def test_a_reverse_list_missing_the_control_port_is_not_ready(self) -> None:
        vr_only = f"tcp:{adb.VR_PORT} tcp:{adb.VR_PORT}\n"
        both = vr_only + f"tcp:{adb.CONTROL_PORT} tcp:{adb.CONTROL_PORT}\n"

        for listing, expected in ((vr_only, False), (both, True)):
            with self.subTest(listing=listing):

                def run(args: list[str], timeout: float = 10.0, listing=listing):
                    if args == ["devices"]:
                        return _completed("List of devices attached\n1WMHH\tdevice\n")
                    return _completed(listing)

                with (
                    patch.object(adb, "_adb", return_value="/usr/bin/adb"),
                    patch.object(adb, "_run", side_effect=run),
                ):
                    status = adb.status()

                self.assertEqual(status.reverse_active, expected)
                self.assertEqual(status.ready, expected)


@unittest.skipIf(shutil.which("openssl") is None, "openssl is required")
class CertificateRotationTest(unittest.TestCase):
    def _certificate_directory(self, directory: str) -> Path:
        """Resolve the temporary directory so no path component is a symlink.

        The certificate writer opens every component with ``O_NOFOLLOW``, and
        macOS puts the temporary directory behind the ``/var`` symlink.
        """
        return Path(directory).resolve()

    def _legacy_certificate(self, directory: Path) -> tuple[str, str]:
        """Write the pre-SAN certificate an older install still carries."""
        certfile = directory / "cert.pem"
        keyfile = directory / "key.pem"
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-keyout",
                str(keyfile),
                "-out",
                str(certfile),
                "-days",
                "365",
                "-nodes",
                "-subj",
                "/CN=localhost",
            ],
            check=True,
            capture_output=True,
        )
        return str(certfile), str(keyfile)

    def test_only_a_certificate_without_a_san_is_reported_stale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._certificate_directory(directory)
            legacy_cert, _ = self._legacy_certificate(root)
            current_cert = str(root / "current-cert.pem")
            certs.create_self_signed_cert(current_cert, str(root / "current-key.pem"))

            self.assertTrue(certs.certificate_lacks_san(legacy_cert))
            self.assertFalse(certs.certificate_lacks_san(current_cert))

    def test_an_unreadable_certificate_is_left_alone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            garbage = Path(directory) / "cert.pem"
            garbage.write_text("not a certificate")

            self.assertFalse(certs.certificate_lacks_san(str(garbage)))

    def test_preparing_tls_files_rotates_a_certificate_without_a_san(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._certificate_directory(directory)
            certfile, keyfile = self._legacy_certificate(root)
            legacy_bytes = Path(certfile).read_bytes()

            with patch.object(certs, "privileged_service_active", return_value=False):
                prepared = certs.prepare_tls_files(certfile, keyfile)

            with prepared:
                self.assertTrue(prepared.generated)
                self.assertNotEqual(Path(certfile).read_bytes(), legacy_bytes)
                self.assertFalse(certs.certificate_lacks_san(certfile))

    def test_preparing_tls_files_keeps_a_certificate_that_has_a_san(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._certificate_directory(directory)
            certfile = str(root / "cert.pem")
            keyfile = str(root / "key.pem")
            certs.create_self_signed_cert(certfile, keyfile)
            current_bytes = Path(certfile).read_bytes()

            with patch.object(certs, "privileged_service_active", return_value=False):
                prepared = certs.prepare_tls_files(certfile, keyfile)

            with prepared:
                self.assertFalse(prepared.generated)
                self.assertEqual(Path(certfile).read_bytes(), current_bytes)


if __name__ == "__main__":
    unittest.main()
