from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import tomllib


class ReleaseMetadataTest(unittest.TestCase):
    def test_sdk_uses_the_next_release_version_consistently(self) -> None:
        root = Path(__file__).resolve().parents[1]
        project = tomllib.loads((root / "pyproject.toml").read_text())
        lock = tomllib.loads((root / "uv.lock").read_text())
        locked_project = next(
            package for package in lock["package"] if package["name"] == "almond-axol"
        )

        self.assertEqual(project["project"]["version"], "0.1.37")
        self.assertEqual(locked_project["version"], "0.1.37")

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
            "lerobot[viz,async,diffusion,av-dep,accelerate-dep]==0.6.1",
            requirements,
        )
        self.assertNotIn("torchcodec", " ".join(requirements).lower())
        for requirement in (
            "datasets>=4.8,<5",
            "jsonlines>=4,<5",
            "pandas>=2,<3",
            "pyarrow>=21,<30",
            "wandb>=0.24,<0.28",
        ):
            self.assertIn(requirement, requirements)
        self.assertIn("torch==2.10.0", requirements)
        self.assertIn("torchvision==0.25.0", requirements)

    def test_release_paths_pin_the_validated_uv_version(self) -> None:
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github" / "workflows" / "publish.yml").read_text()
        installer = (root / "web" / "app" / "public" / "install").read_text()

        self.assertEqual(workflow.count('version: "0.12.3"'), 3)
        self.assertIn('UV_VERSION="0.12.3"', installer)
        self.assertIn(
            'UV_X86_64_LINUX_SHA256="600cf9a742aca00d292673b16b5acffaa7b8c269a364ad0c2e79498dcb1fe101"',
            installer,
        )
        self.assertIn(
            'UV_AARCH64_LINUX_SHA256="bb66cb52e7b1823aed1183630d8d8e5c958840d584a4c55ec10a4cfc168dcca2"',
            installer,
        )
        self.assertIn(
            "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-${UV_TARGET}.tar.gz",
            installer,
        )
        self.assertIn("sha256sum --check --status", installer)
        self.assertNotIn("astral.sh/uv/${UV_VERSION}/install.sh", installer)

    def test_release_wheel_embeds_the_reviewed_host_installer(self) -> None:
        root = Path(__file__).resolve().parents[1]
        project = tomllib.loads((root / "pyproject.toml").read_text())
        wheel = project["tool"]["hatch"]["build"]["targets"]["wheel"]
        sdist = project["tool"]["hatch"]["build"]["targets"]["sdist"]

        self.assertEqual(
            wheel["force-include"],
            {"web/app/public/install": "almond_axol/_installer.sh"},
        )
        self.assertIn("/web/app/public/install", sdist["include"])

    def test_web_validation_disables_package_manager_caching(self) -> None:
        workflow = (
            Path(__file__).resolve().parents[1]
            / ".github"
            / "workflows"
            / "publish.yml"
        ).read_text()

        self.assertIn(
            "actions/setup-node@820762786026740c76f36085b0efc47a31fe5020 # v7.0.0",
            workflow,
        )
        self.assertIn("package-manager-cache: false", workflow)

    def test_releases_use_the_namespace_invisible_to_the_legacy_updater(self) -> None:
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github" / "workflows" / "publish.yml").read_text()
        release_guide = (root / ".github" / "RELEASING.md").read_text()
        updater = (root / "almond_axol" / "serve" / "update.py").read_text()
        installer = (root / "web" / "app" / "public" / "install").read_text()

        self.assertIn('expected_tag="release-v$(uv version --short)"', workflow)
        self.assertIn("rulesets?includes_parents=true", workflow)
        self.assertIn('any(.type == "creation")', workflow)
        self.assertIn(".conditions.ref_name.exclude | length == 0", workflow)
        self.assertIn("secrets.RULESET_AUDIT_TOKEN", workflow)
        self.assertIn('if [ -z "${GH_TOKEN}" ]', workflow)
        self.assertIn('has("bypass_actors")', workflow)
        self.assertIn('.bypass_actors | type == "array"', workflow)
        self.assertIn(".bypass_actors | length == 0", workflow)
        self.assertNotIn("GH_TOKEN: ${{ github.token }}", workflow)
        self.assertIn('"refs/tags/release-v*"', updater)
        self.assertIn('"refs/tags/release-v*"', installer)
        self.assertNotIn('"refs/tags/v*"', installer)
        self.assertIn('is_hardened_release_tag "${LATEST_TAG}"', installer)
        self.assertIn("Never create or\npush another `vX.Y.Z` tag", release_guide)
        self.assertIn("Versions `v0.1.0` through `v0.1.2`", release_guide)
        self.assertIn("RULESET_AUDIT_TOKEN", release_guide)
        self.assertIn("Administration/rulesets", release_guide)

    def test_release_commit_must_be_contained_in_fetched_origin_main(self) -> None:
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github" / "workflows" / "publish.yml").read_text()
        release_guide = (root / ".github" / "RELEASING.md").read_text()

        provenance = workflow.index("  validate-release-provenance:")
        python_validation = workflow.index("  validate-python:")
        gate = workflow[provenance:python_validation]
        self.assertIn("fetch-depth: 0", gate)
        self.assertIn("+refs/heads/main:refs/remotes/origin/main", gate)
        self.assertIn("refs/tags/${RELEASE_TAG}^{commit}", gate)
        self.assertIn("git merge-base --is-ancestor", gate)
        self.assertIn("refs/remotes/origin/main", gate)
        self.assertEqual(
            workflow.count(
                "needs: [validate-release-provenance, validate-python, validate-web]"
            ),
            2,
        )
        self.assertIn("requires that commit to be an ancestor", release_guide)
        self.assertIn("## Mandatory release checklist", release_guide)
        for required_evidence in (
            "generated wheels",
            "ARM64",
            "CAN discovery",
            "tracker inputs",
            "ZED capture",
            "Drill an update",
            "roll the canary back",
        ):
            with self.subTest(required_evidence=required_evidence):
                self.assertIn(required_evidence, release_guide)

    def test_customer_notifications_wait_for_both_package_publishes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github" / "workflows" / "publish.yml").read_text()

        self.assertFalse(
            (root / ".github" / "workflows" / "release-notify.yml").exists()
        )
        notify_job = workflow.index("  notify-release:")
        self.assertIn(
            "needs: [publish-sdk, publish-plugin]",
            workflow[notify_job:],
        )
        self.assertIn("if: github.event_name == 'release'", workflow[notify_job:])
        self.assertEqual(workflow.count("toJSON(secrets)"), 1)
        self.assertNotIn("secrets.SLACK_WEBHOOK_URLS", workflow)
        self.assertIn('startswith("SLACK_WEBHOOK_URL_")', workflow)
        self.assertIn("No SLACK_WEBHOOK_URL_* secrets are set", workflow)
        self.assertIn(
            "SLACK_WEBHOOK_URL_*",
            (root / ".github" / "RELEASING.md").read_text(),
        )

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
        parent_seal = "install -d -o root -g root -m 0751 /var/lib/almond-axol"
        child_check = '[ ! -L "${SERVICE_DATASET_ROOT}" ]'
        child_install = 'install -d -o root -g "${DATASET_GROUP}" -m 2750'
        self.assertLess(installer.index(parent_seal), installer.index(child_check))
        self.assertLess(installer.index(child_check), installer.index(child_install))
        self.assertNotIn(
            'install -d -o root -g "${DATASET_GROUP}" -m 0750 /var/lib/almond-axol',
            installer,
        )
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
