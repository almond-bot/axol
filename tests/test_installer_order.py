"""The hosted installer must leave a working service even when provisioning fails."""

from __future__ import annotations

import unittest
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[1] / "web" / "app" / "public" / "install"
).read_text()


def _at(needle: str) -> int:
    index = _SCRIPT.find(needle)
    assert index >= 0, needle
    return index


class InstallerOrderTest(unittest.TestCase):
    def test_unit_is_written_and_enabled_before_provisioning(self) -> None:
        # A provision failure (e.g. a changed vendor driver) used to abort
        # before the unit existed: no service, root-env manual serve, and no
        # panel to take the fix through.
        provision = _at('"${BIN_DIR}/axol" provision --require-rt --no-reboot')
        self.assertLess(_at('cat > "${SERVICE_FILE}"'), provision)
        self.assertLess(_at('systemctl enable "${SERVICE_NAME}"'), provision)
        self.assertLess(_at("Environment=HF_LEROBOT_HOME"), provision)

    def test_service_starts_before_a_provision_failure_aborts(self) -> None:
        provision = _at('"${BIN_DIR}/axol" provision --require-rt --no-reboot')
        restart = _at('systemctl restart "${SERVICE_NAME}"')
        self.assertLess(provision, restart)
        self.assertIn("|| PROVISION_OK=0", _SCRIPT)
        self.assertLess(restart, _at('[ "${PROVISION_OK}" -eq 1 ]'))

    def test_dataset_tree_stays_usable_by_the_user(self) -> None:
        # The root service's umask 027 made calibration/robots/axol root:root
        # 0750; a default ACL keeps whatever it creates user-writable.
        self.assertIn('mkdir -p "${LEROBOT_HOME}/calibration"', _SCRIPT)
        self.assertIn(
            'setfacl -R -m "u:${DATASET_USER}:rwX" -m "d:u:${DATASET_USER}:rwX"',
            _SCRIPT,
        )


if __name__ == "__main__":
    unittest.main()
