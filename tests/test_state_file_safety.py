from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from almond_axol.cli import collect_dagger, collect_data, replay_dataset, run_policy
from almond_axol.diagnostics.zed import cable
from almond_axol.serve.settings import SettingsStore
from almond_axol.serve.telemetry import DiagnosticsRunStore, TelemetryHub
from almond_axol.utils import certs, state_files


class SecureStateFileTest(unittest.TestCase):
    def test_predictable_temp_and_final_symlinks_never_touch_their_victims(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "settings.json"
            predictable = root / "settings.json.tmp"
            predictable_victim = root / "predictable-victim"
            final_victim = root / "final-victim"
            predictable_victim.write_text("keep predictable")
            final_victim.write_text("keep final")
            predictable.symlink_to(predictable_victim)
            destination.symlink_to(final_victim)

            with self.assertRaisesRegex(
                state_files.UnsafeStatePathError, "non-regular"
            ):
                state_files.secure_atomic_write_text(destination, "new")

            self.assertEqual(predictable_victim.read_text(), "keep predictable")
            self.assertEqual(final_victim.read_text(), "keep final")
            self.assertTrue(predictable.is_symlink())
            self.assertTrue(destination.is_symlink())
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                [
                    "final-victim",
                    "predictable-victim",
                    "settings.json",
                    "settings.json.tmp",
                ],
            )

    def test_parent_symlink_and_hard_link_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside"
            outside.mkdir()
            linked_parent = root / "linked-parent"
            linked_parent.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(OSError):
                state_files.secure_atomic_write_text(
                    linked_parent / "state.json", "unsafe"
                )
            self.assertFalse((outside / "state.json").exists())

            victim = root / "victim"
            victim.write_text("keep")
            hard_link = root / "state.json"
            os.link(victim, hard_link)
            with self.assertRaisesRegex(
                state_files.UnsafeStatePathError, "hard-linked"
            ):
                state_files.secure_atomic_write_text(hard_link, "unsafe")
            self.assertEqual(victim.read_text(), "keep")
            self.assertEqual(hard_link.read_text(), "keep")

    def test_atomic_write_preserves_owner_mode_and_existing_shared_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = root / "shared"
            shared.mkdir(mode=0o775)
            shared.chmod(0o2775)
            before_directory = shared.stat()
            destination = shared / "nested" / "state.json"

            state_files.secure_atomic_write_json(destination, {"ok": True})

            after_directory = shared.stat()
            nested = destination.parent.stat()
            self.assertEqual(
                stat.S_IMODE(after_directory.st_mode),
                stat.S_IMODE(before_directory.st_mode),
            )
            self.assertEqual(
                (after_directory.st_uid, after_directory.st_gid),
                (before_directory.st_uid, before_directory.st_gid),
            )
            self.assertEqual(
                (nested.st_uid, nested.st_gid), (root.stat().st_uid, root.stat().st_gid)
            )
            self.assertEqual(stat.S_IMODE(nested.st_mode), 0o2770)
            self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
            self.assertEqual(json.loads(destination.read_text()), {"ok": True})

    def test_file_and_directory_are_fsynced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "state.json"
            real_fsync = os.fsync
            fsync = Mock(side_effect=real_fsync)
            with patch.object(state_files.os, "fsync", fsync):
                state_files.secure_atomic_write_text(destination, "durable")
            self.assertGreaterEqual(fsync.call_count, 2)

    def test_failed_created_directory_initialization_closes_child_fd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "new" / "state.json"
            before_fds = len(list(Path("/proc/self/fd").iterdir()))
            with (
                patch.object(
                    state_files.os,
                    "fchmod",
                    side_effect=OSError("simulated directory mode failure"),
                ),
                self.assertRaisesRegex(OSError, "simulated directory mode failure"),
            ):
                state_files.secure_atomic_write_text(destination, "never published")

            self.assertFalse(destination.exists())
            self.assertEqual(len(list(Path("/proc/self/fd").iterdir())), before_fds)

    def test_substituted_staging_payload_is_never_published(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "state.json"
            real_stat = os.stat

            def mismatched_payload(path: object, *args: object, **kwargs: object):
                result = real_stat(path, *args, **kwargs)
                if path == "payload" and kwargs.get("follow_symlinks") is False:
                    values = list(result)
                    values[1] += 1  # st_ino
                    return os.stat_result(values)
                return result

            with (
                patch.object(state_files.os, "stat", side_effect=mismatched_payload),
                self.assertRaisesRegex(
                    state_files.UnsafeStatePathError, "staging file was substituted"
                ),
            ):
                state_files.secure_atomic_write_text(destination, "untrusted")

            self.assertFalse(destination.exists())
            self.assertEqual(list(destination.parent.glob("*.stage")), [])

    def test_exclusive_create_failure_closes_fd_and_removes_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "capture.csv"
            before_fds = len(list(Path("/proc/self/fd").iterdir()))
            with (
                patch.object(
                    state_files.os,
                    "fsync",
                    side_effect=OSError("simulated directory fsync failure"),
                ),
                self.assertRaisesRegex(OSError, "simulated directory fsync failure"),
            ):
                state_files.secure_open_new_text(destination)

            self.assertFalse(destination.exists())
            self.assertEqual(len(list(Path("/proc/self/fd").iterdir())), before_fds)

    def test_recursive_chown_stays_on_pinned_tree_after_ancestor_swap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent"
            tree = parent / "dataset"
            tree.mkdir(parents=True)
            (tree / "data").write_text("dataset")
            outside = root / "outside"
            outside.mkdir()
            victim = outside / "data"
            victim.write_text("outside")
            victim_before = victim.stat()
            detached = root / "detached-parent"
            real_open_directory = state_files._open_directory  # noqa: SLF001

            def open_then_swap(path: str | Path):
                target, descriptor = real_open_directory(path)
                parent.rename(detached)
                parent.symlink_to(outside, target_is_directory=True)
                return target, descriptor

            with patch.object(
                state_files, "_open_directory", side_effect=open_then_swap
            ):
                state_files.secure_chown_tree(
                    tree,
                    os.geteuid(),
                    os.getegid(),
                )

            self.assertEqual((detached / "dataset" / "data").read_text(), "dataset")
            self.assertEqual(victim.read_text(), "outside")
            victim_after = victim.stat()
            self.assertEqual(
                (victim_after.st_uid, victim_after.st_gid),
                (victim_before.st_uid, victim_before.st_gid),
            )

    def test_dataset_tree_normalizes_to_group_read_only_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            nested = root / "meta"
            nested.mkdir(parents=True)
            payload = nested / "info.json"
            payload.write_text("{}")
            root.chmod(0o777)
            nested.chmod(0o777)
            payload.chmod(0o600)

            state_files.secure_chown_tree(
                root,
                os.geteuid(),
                os.getegid(),
                directory_mode=0o2750,
                file_mode=0o640,
            )

            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o2750)
            self.assertEqual(stat.S_IMODE(nested.stat().st_mode), 0o2750)
            self.assertEqual(stat.S_IMODE(payload.stat().st_mode), 0o640)

    def test_secure_read_rejects_final_symlink_and_hard_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            victim = root / "victim"
            victim.write_text("secret")
            symlink = root / "symlink"
            hard_link = root / "hard-link"
            symlink.symlink_to(victim)
            os.link(victim, hard_link)

            with self.assertRaises(OSError):
                state_files.secure_read_text(symlink)
            with self.assertRaisesRegex(
                state_files.UnsafeStatePathError, "unsafe state file"
            ):
                state_files.secure_read_text(hard_link)

    def test_privileged_path_allowlist_rejects_escape_and_existing_symlink(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "datasets"
            root.mkdir()
            self.assertEqual(
                state_files.require_path_beneath(
                    root / "owner" / "dataset", root, label="dataset"
                ),
                root / "owner" / "dataset",
            )
            with self.assertRaisesRegex(OSError, "must stay below"):
                state_files.require_path_beneath(
                    root.parent / "escape", root, label="dataset"
                )
            outside = root.parent / "outside"
            outside.mkdir()
            (root / "linked").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(OSError, "contains a symlink"):
                state_files.require_path_beneath(
                    root / "linked" / "dataset", root, label="dataset"
                )


class StateWriterIntegrationTest(unittest.TestCase):
    def test_every_hosted_dataset_operation_uses_repo_specific_root(self) -> None:
        boundary = Path("/var/lib/almond-axol/datasets")
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / "settings.json")
            store.update(values={"recording.root": "/home/operator/legacy"})
            with (
                patch.object(
                    state_files, "privileged_service_active", return_value=True
                ),
                patch.object(
                    state_files,
                    "validated_service_dataset_root",
                    return_value=boundary,
                ),
            ):
                for operation in (
                    "collect-data",
                    "collect-dagger",
                    "run-policy",
                    "replay-dataset",
                ):
                    with self.subTest(operation=operation):
                        merged = store.merged_args(
                            operation,
                            {"repo_id": "owner/dataset", "root": "/etc"},
                        )
                        self.assertEqual(
                            merged["root"],
                            str(boundary / "owner" / "dataset"),
                        )

                no_recording = store.merged_args(
                    "run-policy",
                    {"policy_path": "model", "root": "/etc"},
                )
                self.assertNotIn("root", no_recording)

    def test_unconfigured_service_blocks_every_dataset_operation_early(
        self,
    ) -> None:
        configs = (
            (collect_data._run, object()),  # noqa: SLF001
            (collect_dagger._run, object()),  # noqa: SLF001
            (
                run_policy._run,  # noqa: SLF001
                type(
                    "PolicyConfig", (), {"robot_config": object(), "repo_id": "x/y"}
                )(),
            ),
            (replay_dataset._run, object()),  # noqa: SLF001
        )
        with (
            patch.dict(
                os.environ,
                {"AXOL_PRIVILEGED_SERVICE": "1"},
                clear=True,
            ),
            patch.object(state_files.os, "geteuid", return_value=0),
        ):
            self.assertTrue(state_files.privileged_service_active())
            for operation, config in configs:
                with (
                    self.subTest(operation=operation.__module__),
                    self.assertRaisesRegex(OSError, "not configured"),
                ):
                    operation(config)

    def test_service_dataset_confinement_uses_only_root_controlled_boundary(
        self,
    ) -> None:
        boundary = Path("/var/lib/almond-axol/datasets")
        with (
            patch.dict(
                os.environ,
                {
                    "AXOL_PRIVILEGED_SERVICE": "1",
                    "AXOL_SERVICE_DATASET_ROOT": str(boundary),
                },
                clear=True,
            ),
            patch.object(state_files.os, "geteuid", return_value=0),
            patch.object(
                state_files,
                "_require_root_controlled_directory",
                return_value=boundary,
            ),
        ):
            self.assertEqual(
                state_files.confine_service_dataset_path(
                    boundary / "owner" / "dataset",
                    label="dataset",
                ),
                boundary / "owner" / "dataset",
            )
            with self.assertRaisesRegex(OSError, "must stay below"):
                state_files.confine_service_dataset_path(
                    "/etc",
                    label="dataset",
                )

    def test_hosted_repo_id_maps_to_one_specific_dataset_directory(self) -> None:
        boundary = Path("/var/lib/almond-axol/datasets")
        with patch.object(
            state_files,
            "validated_service_dataset_root",
            return_value=boundary,
        ):
            self.assertEqual(
                state_files.service_dataset_path_for_repo_id("owner/dataset-1"),
                boundary / "owner" / "dataset-1",
            )
            self.assertEqual(
                state_files.service_dataset_path_for_repo_id("dataset-1"),
                boundary / "dataset-1",
            )
            for unsafe in (
                "/etc",
                "../etc",
                "owner/../etc",
                "owner/name/extra",
                "owner\\name",
                " owner/name",
                "hub",
            ):
                with (
                    self.subTest(repo_id=unsafe),
                    self.assertRaisesRegex(OSError, "hosted dataset repo_id"),
                ):
                    state_files.service_dataset_path_for_repo_id(unsafe)

    def test_hosted_zed_cable_output_is_rejected_before_camera_access(self) -> None:
        with (
            patch.object(state_files, "privileged_service_active", return_value=True),
            patch.object(cable, "restart_zed_daemon") as restart,
            self.assertRaisesRegex(cable.CableTestError, "--output is disabled"),
        ):
            cable.run("/etc/cron.d/axol.png")

        restart.assert_not_called()

    def test_settings_store_refuses_preplanted_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            victim = root / "victim.json"
            victim.write_text('{"keep": true}')
            settings = root / "settings.json"
            settings.symlink_to(victim)
            store = SettingsStore(settings)

            with self.assertRaises(state_files.UnsafeStatePathError):
                store.update(values={})

            self.assertEqual(victim.read_text(), '{"keep": true}')
            self.assertTrue(settings.is_symlink())

    def test_cert_generation_stages_before_rejecting_state_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            victim = root / "victim"
            victim.write_text("keep")
            cert = root / "cert.pem"
            key = root / "key.pem"
            cert.symlink_to(victim)

            def fake_openssl(command: list[str], **_kwargs: object) -> Mock:
                Path(command[command.index("-out") + 1]).write_bytes(b"certificate")
                Path(command[command.index("-keyout") + 1]).write_bytes(b"private-key")
                return Mock(returncode=0)

            with (
                patch.object(certs.subprocess, "run", side_effect=fake_openssl),
                self.assertRaises(state_files.UnsafeStatePathError),
            ):
                certs.create_self_signed_cert(str(cert), str(key))

            self.assertEqual(victim.read_text(), "keep")
            self.assertFalse(key.exists())

    def test_diagnostics_clear_never_unlinks_forged_external_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            runs.mkdir()
            victim = root / "external.csv"
            victim.write_text("keep")
            run_id = "012345abcdef"
            (runs / f"{run_id}.meta.json").write_text(
                json.dumps({"telemetryCsv": str(victim)})
            )
            (runs / f"{run_id}.data.json").write_text("{}")
            store = DiagnosticsRunStore(TelemetryHub(), runs)

            self.assertEqual(store.clear(), 1)

            self.assertEqual(victim.read_text(), "keep")
            self.assertEqual(list(runs.iterdir()), [])

    def test_diagnostics_load_rejects_path_shaped_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = DiagnosticsRunStore(TelemetryHub(), Path(directory))
            self.assertIsNone(store.load("../../etc/passwd"))


if __name__ == "__main__":
    unittest.main()
