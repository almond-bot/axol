from __future__ import annotations

import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path


class ReleaseMetadataTest(unittest.TestCase):
    def test_sdk_uses_the_next_release_version_consistently(self) -> None:
        root = Path(__file__).resolve().parents[1]
        project = tomllib.loads((root / "pyproject.toml").read_text())
        lock = tomllib.loads((root / "uv.lock").read_text())
        locked_project = next(
            package for package in lock["package"] if package["name"] == "almond-axol"
        )

        self.assertEqual(project["project"]["version"], "0.1.36")
        self.assertEqual(locked_project["version"], "0.1.36")

    def test_release_builds_pin_the_validated_backend(self) -> None:
        root = Path(__file__).resolve().parents[1]
        project_files = (
            root / "pyproject.toml",
            root / "plugins" / "lerobot_robot_axol" / "pyproject.toml",
        )

        for project_file in project_files:
            with self.subTest(project_file=project_file):
                project = tomllib.loads(project_file.read_text())
                self.assertEqual(
                    project["build-system"]["requires"],
                    ["hatchling==1.32.0"],
                )

    def test_lerobot_extra_pins_the_validated_torch_pair(self) -> None:
        project_file = Path(__file__).resolve().parents[1] / "pyproject.toml"
        project = tomllib.loads(project_file.read_text())
        requirements = project["project"]["optional-dependencies"]["lerobot"]

        self.assertIn(
            "lerobot[dataset,viz,async,training,diffusion]==0.6.1",
            requirements,
        )
        self.assertIn("torch==2.10.0", requirements)
        self.assertIn("torchvision==0.25.0", requirements)

    def test_release_paths_pin_the_validated_uv_version(self) -> None:
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github" / "workflows" / "publish.yml").read_text()
        installer = (root / "web" / "app" / "public" / "install").read_text()

        self.assertEqual(workflow.count('version: "0.12.3"'), 3)
        self.assertIn('UV_VERSION="0.12.3"', installer)
        self.assertIn("https://astral.sh/uv/${UV_VERSION}/install.sh", installer)

    def test_plugin_caps_the_sdk_api_line_and_has_a_publishable_version(self) -> None:
        root = Path(__file__).resolve().parents[1]
        plugin = tomllib.loads(
            (root / "plugins" / "lerobot_robot_axol" / "pyproject.toml").read_text()
        )["project"]

        self.assertEqual(plugin["version"], "0.1.1")
        self.assertEqual(
            plugin["dependencies"],
            ["almond-axol[lerobot]>=0.1.24,<0.2"],
        )

    def test_installed_service_uses_an_immutable_dataset_boundary(self) -> None:
        installer = (
            Path(__file__).resolve().parents[1] / "web" / "app" / "public" / "install"
        ).read_text()

        self.assertIn('SERVICE_DATASET_ROOT="/var/lib/almond-axol/datasets"', installer)
        self.assertIn('install -d -o root -g "${DATASET_GROUP}" -m 2750', installer)
        self.assertIn("Environment=AXOL_PRIVILEGED_SERVICE=1", installer)
        self.assertIn("AXOL_SERVICE_DATASET_ROOT", installer)
        self.assertIn("AXOL_OPERATOR_UID", installer)
        self.assertIn("AXOL_OPERATOR_GID", installer)
        self.assertIn("UMask=0027", installer)
        parent_seal = "install -d -o root -g root -m 0750 /var/lib/almond-axol"
        child_check = '[ ! -L "${SERVICE_DATASET_ROOT}" ]'
        child_install = 'install -d -o root -g "${DATASET_GROUP}" -m 2750'
        self.assertLess(installer.index(parent_seal), installer.index(child_check))
        self.assertLess(installer.index(child_check), installer.index(child_install))
        self.assertIn("AXOL_ACK_LEGACY_DATASETS", installer)
        self.assertIn("Existing datasets were found", installer)
        self.assertIn("-mindepth 3 -maxdepth 4", installer)
        self.assertIn(
            "sudo env AXOL_ACK_LEGACY_DATASETS=1 bash",
            installer,
        )
        self.assertNotIn(
            'chown -hR "${OPERATOR_USER}:${OPERATOR_GROUP}" "${ALMOND_STATE_HOME}"',
            installer,
        )

    def test_legacy_dataset_gate_finds_one_and_two_component_repo_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {
                root / "dataset" / "meta" / "info.json",
                root / "owner" / "dataset" / "meta" / "info.json",
            }
            for info in expected:
                info.parent.mkdir(parents=True)
                info.write_text("{}")

            result = subprocess.run(
                [
                    "find",
                    "-P",
                    str(root),
                    "-mindepth",
                    "3",
                    "-maxdepth",
                    "4",
                    "-type",
                    "f",
                    "-path",
                    "*/meta/info.json",
                    "-print",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertEqual({Path(line) for line in result.stdout.splitlines()}, expected)


if __name__ == "__main__":
    unittest.main()
