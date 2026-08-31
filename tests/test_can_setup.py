from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from almond_axol.cli.can import driver, setup


class CanSetupAssignmentTest(unittest.TestCase):
    @staticmethod
    def _configured_axol_only(
        profile: setup._Profile = setup._AXOL_PROFILE,
    ) -> str | None:
        return "SERIAL" if profile is setup._AXOL_PROFILE else None

    def test_setup_does_not_forget_an_unplugged_configured_hub(self) -> None:
        with (
            patch.object(setup, "_detect_serials", return_value=[]),
            patch.object(
                setup, "_configured_serial", side_effect=self._configured_axol_only
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(setup._find_dual_serials(), ("SERIAL", None))

    def test_axol_pin_is_not_preserved_when_serial_now_enumerates_single(self) -> None:
        with (
            patch.object(setup, "_detect_serials", return_value=[]),
            patch.object(
                setup, "_configured_serial", side_effect=self._configured_axol_only
            ),
            patch.object(
                setup,
                "_scan_adapters",
                return_value={"SERIAL": {"dev_ids": {0}}},
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(setup._find_dual_serials(), (None, None))

    def test_other_product_does_not_displace_an_unplugged_hub(self) -> None:
        with (
            patch.object(setup, "_detect_serials", return_value=["MANTIS"]),
            patch.object(
                setup, "_configured_serial", side_effect=self._configured_axol_only
            ),
            patch.object(setup, "_identify_dual_adapter", return_value="mantis"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(setup._find_dual_serials(), ("SERIAL", "MANTIS"))

    def test_skipped_silent_adapter_does_not_forget_an_unplugged_hub(self) -> None:
        with (
            patch.object(setup, "_detect_serials", return_value=["NEW-SILENT"]),
            patch.object(
                setup, "_configured_serial", side_effect=self._configured_axol_only
            ),
            patch.object(setup, "_identify_dual_adapter", return_value=None),
            patch("builtins.input", return_value=""),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(
                setup._find_dual_serials(),
                ("SERIAL", None),
            )

    def test_plain_setup_offers_an_attached_silent_hub(self) -> None:
        with (
            patch.object(setup, "_detect_serials", return_value=["SERIAL"]),
            patch.object(
                setup, "_configured_serial", side_effect=self._configured_axol_only
            ),
            patch.object(setup, "_identify_dual_adapter", return_value=None),
            patch("builtins.input", return_value="m"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(setup._find_dual_serials(), (None, "SERIAL"))

    def test_plain_setup_enter_preserves_an_attached_silent_hub(self) -> None:
        output = io.StringIO()
        with (
            patch.object(setup, "_detect_serials", return_value=["SERIAL"]),
            patch.object(
                setup, "_configured_serial", side_effect=self._configured_axol_only
            ),
            patch.object(setup, "_identify_dual_adapter", return_value=None),
            patch("builtins.input", return_value="") as prompt,
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(setup._find_dual_serials(), ("SERIAL", None))
        prompt.assert_called_once()
        self.assertIn("[Enter] keeps axol", prompt.call_args.args[0])

    def test_plain_setup_reclassifies_axol_pin_when_mantis_answers(self) -> None:
        with (
            patch.object(setup, "_detect_serials", return_value=["SERIAL"]),
            patch.object(
                setup, "_configured_serial", side_effect=self._configured_axol_only
            ),
            patch.object(
                setup, "_identify_dual_adapter", return_value="mantis"
            ) as identify,
            patch("builtins.input") as prompt,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(setup._find_dual_serials(), (None, "SERIAL"))
        identify.assert_called_once_with("SERIAL", reset=True)
        prompt.assert_not_called()

    def test_dual_identity_recovers_rx_before_deciding_hub_is_silent(self) -> None:
        channels = [setup.CAN_LEFT, setup.CAN_RIGHT]
        with (
            patch.object(setup, "_ifaces_for_serial", return_value=channels),
            patch.object(setup, "iface_up", return_value=True),
            patch.object(setup, "bring_up_interfaces") as bring_up,
            patch.object(
                setup,
                "_probe_mantis_trigger",
                side_effect=[False, False, True],
            ),
            patch.object(setup, "_probe_axol_shoulder", return_value=False),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(setup._identify_dual_adapter("SERIAL"), "mantis")
        self.assertEqual(
            bring_up.call_args_list,
            [
                unittest.mock.call(channels, force_cycle=False),
                unittest.mock.call(channels, force_cycle=True),
            ],
        )

    def test_dual_identity_does_not_cycle_after_a_live_first_probe(self) -> None:
        channels = [setup.CAN_LEFT, setup.CAN_RIGHT]
        with (
            patch.object(setup, "_ifaces_for_serial", return_value=channels),
            patch.object(setup, "iface_up", return_value=True),
            patch.object(setup, "bring_up_interfaces") as bring_up,
            patch.object(setup, "_probe_mantis_trigger", return_value=True),
            patch.object(setup, "_probe_axol_shoulder", return_value=False),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(setup._identify_dual_adapter("SERIAL"), "mantis")
        bring_up.assert_called_once_with(channels, force_cycle=False)

    def test_dual_identity_recovery_is_bounded_when_hardware_is_off(self) -> None:
        channels = [setup.CAN_LEFT, setup.CAN_RIGHT]
        with (
            patch.object(setup, "_ifaces_for_serial", return_value=channels),
            patch.object(setup, "iface_up", return_value=True),
            patch.object(setup, "bring_up_interfaces") as bring_up,
            patch.object(setup, "_probe_mantis_trigger", return_value=False),
            patch.object(setup, "_probe_axol_shoulder", return_value=False),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertIsNone(setup._identify_dual_adapter("SERIAL"))
        self.assertEqual(bring_up.call_count, 3)
        self.assertEqual(
            [call.kwargs["force_cycle"] for call in bring_up.call_args_list],
            [False, True, True],
        )

    def test_paired_recovery_takes_both_channels_down_before_either_up(self) -> None:
        channels = [setup.CAN_LEFT, setup.CAN_RIGHT]
        with (
            patch.object(setup.Path, "exists", return_value=True),
            patch.object(setup, "run_root") as run_root,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            setup.bring_up_interfaces(channels, force_cycle=True)

        commands = [call.args[0] for call in run_root.call_args_list]
        down = [["ip", "link", "set", channel, "down"] for channel in channels]
        up = [["ip", "link", "set", channel, "up"] for channel in channels]
        self.assertEqual(commands[:2], down)
        self.assertEqual(commands[-2:], up)
        self.assertTrue(
            all(
                commands.index(down_command) < commands.index(up[0])
                for down_command in down
            )
        )

    def test_axol_rx_retry_does_not_reset_wheel_and_lift_buses_again(self) -> None:
        with (
            patch.object(setup.Path, "exists", return_value=True),
            patch.object(setup, "run_root") as run_root,
            patch.object(
                setup,
                "rx_alive_per_arm",
                side_effect=[(False, False), (True, True)],
            ),
            patch.object(setup, "bring_up_interfaces") as recover_pair,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            setup.bring_up_can(setup._AXOL_PROFILE)

        run_root.assert_called_once_with(
            ["bash", str(setup._AXOL_PROFILE.cron_script)], check=True
        )
        recover_pair.assert_called_once_with(
            [setup.CAN_LEFT, setup.CAN_RIGHT], force_cycle=True
        )

    def test_explicit_single_bus_identification_resets_before_probing(self) -> None:
        with (
            patch.object(setup, "_iface_for_serial", return_value=setup.CAN_BASE),
            patch.object(setup, "iface_up", return_value=True),
            patch.object(setup, "bring_up_interfaces") as bring_up,
            patch.object(setup, "_send_once"),
            patch.object(setup.time, "sleep"),
            patch.object(setup, "_probe_wheels", return_value=True),
            patch.object(setup, "_probe_chest", return_value=False),
        ):
            self.assertEqual(setup._identify_adapter("WHEELS", reset=True), "wheels")
        bring_up.assert_called_once_with([setup.CAN_BASE], force_cycle=True)

    def test_unknown_silent_single_restores_its_initial_down_state(self) -> None:
        with (
            patch.object(setup, "_iface_for_serial", return_value="can7"),
            patch.object(setup, "iface_up", return_value=False),
            patch.object(setup, "bring_up_interfaces") as bring_up,
            patch.object(setup, "_send_once"),
            patch.object(setup.time, "sleep"),
            patch.object(setup, "_probe_wheels", return_value=False),
            patch.object(setup, "_probe_chest", return_value=False),
            patch.object(setup, "run_root") as run_root,
        ):
            self.assertIsNone(setup._identify_adapter("UNKNOWN", recover_silence=False))
        bring_up.assert_called_once_with(["can7"], force_cycle=True)
        run_root.assert_called_once_with(
            ["ip", "link", "set", "can7", "down"], check=False
        )

    def test_unknown_single_uses_non_disruptive_discovery_probe(self) -> None:
        with (
            patch.object(setup, "_configured_named_serial", return_value=None),
            patch.object(setup, "_mantis_claimed_serials", return_value=set()),
            patch.object(setup, "_detect_single_serials", return_value=["NEW"]),
            patch.object(setup, "_scan_adapters", return_value={"NEW": {}}),
            patch.object(setup, "_identify_adapter", return_value="wheels") as identify,
            patch("builtins.input") as prompt,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(setup._find_single_serials(None), ("NEW", None))
        identify.assert_called_once_with("NEW", recover_silence=False)
        prompt.assert_not_called()

    def test_single_bus_stale_wheel_pin_corrects_to_cart_lift(self) -> None:
        def configured(name: str) -> str | None:
            return "SERIAL" if name == setup.CAN_BASE else None

        with (
            patch.object(setup, "_configured_named_serial", side_effect=configured),
            patch.object(setup, "_mantis_claimed_serials", return_value=set()),
            patch.object(setup, "_detect_single_serials", return_value=["SERIAL"]),
            patch.object(setup, "_identify_adapter", return_value="chest") as identify,
            patch("builtins.input") as prompt,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(setup._find_single_serials(None), (None, "SERIAL"))
        identify.assert_called_once_with("SERIAL", reset=True)
        prompt.assert_not_called()

    def test_single_bus_swapped_pins_are_both_corrected(self) -> None:
        configured = {setup.CAN_BASE: "OLD-WHEEL", setup.CAN_CHEST: "OLD-CHEST"}
        identified = {"OLD-WHEEL": "chest", "OLD-CHEST": "wheels"}
        with (
            patch.object(
                setup,
                "_configured_named_serial",
                side_effect=lambda name: configured.get(name),
            ),
            patch.object(setup, "_mantis_claimed_serials", return_value=set()),
            patch.object(
                setup,
                "_detect_single_serials",
                return_value=["OLD-WHEEL", "OLD-CHEST"],
            ),
            patch.object(
                setup,
                "_identify_adapter",
                side_effect=lambda serial, **_kwargs: identified[serial],
            ) as identify,
            patch("builtins.input") as prompt,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(
                setup._find_single_serials(None), ("OLD-CHEST", "OLD-WHEEL")
            )
        self.assertEqual(identify.call_count, 2)
        self.assertTrue(
            all(call.kwargs == {"reset": True} for call in identify.call_args_list)
        )
        prompt.assert_not_called()

    def test_single_bus_enter_preserves_attached_silent_pin(self) -> None:
        def configured(name: str) -> str | None:
            return "SERIAL" if name == setup.CAN_BASE else None

        with (
            patch.object(setup, "_configured_named_serial", side_effect=configured),
            patch.object(setup, "_mantis_claimed_serials", return_value=set()),
            patch.object(setup, "_detect_single_serials", return_value=["SERIAL"]),
            patch.object(setup, "_identify_adapter", return_value=None),
            patch("builtins.input", return_value="") as prompt,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(setup._find_single_serials(None), ("SERIAL", None))
        self.assertIn("[Enter] keeps wheels", prompt.call_args.args[0])

    def test_single_bus_unplugged_pins_are_preserved(self) -> None:
        configured = {setup.CAN_BASE: "WHEELS", setup.CAN_CHEST: "LIFT"}
        with (
            patch.object(
                setup,
                "_configured_named_serial",
                side_effect=lambda name: configured.get(name),
            ),
            patch.object(setup, "_mantis_claimed_serials", return_value=set()),
            patch.object(setup, "_detect_single_serials", return_value=[]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(setup._find_single_serials(None), ("WHEELS", "LIFT"))

    def test_single_pin_is_not_preserved_when_serial_is_selected_as_hub(self) -> None:
        def configured(name: str) -> str | None:
            return "SERIAL" if name == setup.CAN_BASE else None

        with (
            patch.object(setup, "_configured_named_serial", side_effect=configured),
            patch.object(setup, "_mantis_claimed_serials", return_value=set()),
            patch.object(setup, "_detect_single_serials", return_value=[]),
            patch.object(
                setup,
                "_scan_adapters",
                return_value={"SERIAL": {"dev_ids": {0, 1}}},
            ),
        ):
            self.assertEqual(setup._find_single_serials("SERIAL"), (None, None))

    def test_plain_run_migrates_recovered_mantis_out_of_axol_profile(self) -> None:
        with (
            patch.object(setup.driver, "ensure_driver", return_value=False),
            patch.object(
                setup, "_configured_serial", side_effect=self._configured_axol_only
            ),
            patch.object(setup, "_find_dual_serials", return_value=(None, "SERIAL")),
            patch.object(setup, "_find_single_serials", return_value=(None, None)),
            patch.object(setup, "_write_udev_rules") as write_rules,
            patch.object(setup, "_configure_mantis") as configure_mantis,
            patch.object(setup, "_apply_setup") as apply_setup,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            setup.run(SimpleNamespace())
        write_rules.assert_called_once_with(None)
        configure_mantis.assert_called_once_with("SERIAL")
        apply_setup.assert_not_called()

    def test_control_panel_can_recover_a_mantis_stale_pinned_as_axol(self) -> None:
        with (
            patch.object(setup.driver, "ensure_driver", return_value=False),
            patch.object(
                setup, "_configured_serial", side_effect=self._configured_axol_only
            ),
            patch.object(setup, "_detect_serials", return_value=["SERIAL"]),
            patch.object(setup, "_identify_dual_adapter", return_value="mantis"),
            patch.object(setup, "_configured_named_serial", return_value=None),
            patch.object(setup, "_write_udev_rules") as write_rules,
            patch.object(setup, "_configure_mantis") as configure_mantis,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            setup.ensure_mantis_setup()
        write_rules.assert_called_once_with(None, None, None)
        configure_mantis.assert_called_once_with("SERIAL")

    def test_probe_rejects_nonstandard_can_frame_types(self) -> None:
        self.assertEqual(setup._standard_data_can_id(0x009), 0x009)
        for frame_flag in (0x80000000, 0x40000000, 0x20000000):
            with self.subTest(frame_flag=frame_flag):
                self.assertIsNone(setup._standard_data_can_id(frame_flag | 0x009))

    def test_swapped_hub_names_use_collision_free_temporary_stage(self) -> None:
        records = [
            (setup.CAN_LEFT, "MANTIS", 0),
            (setup.CAN_RIGHT, "MANTIS", 1),
            (setup.CAN_MANTIS_LEFT, "AXOL", 0),
            (setup.CAN_MANTIS_RIGHT, "AXOL", 1),
        ]
        temporary, final = setup._interface_rename_plan(
            records,
            {
                ("AXOL", 0): setup.CAN_LEFT,
                ("AXOL", 1): setup.CAN_RIGHT,
            },
        )

        # Both the Axol sources and the stale Mantis occupants move before any
        # final name is claimed, so neither direct rename can collide.
        self.assertEqual({source for source, _ in temporary}, {r[0] for r in records})
        self.assertEqual({name for _, name in final}, {setup.CAN_LEFT, setup.CAN_RIGHT})
        self.assertTrue(all(name.startswith("can_tmp") for _, name in temporary))
        self.assertTrue(all(len(name) <= 15 for _, name in temporary))
        self.assertEqual(len({name for _, name in temporary}), len(temporary))

    def test_cli_formats_driver_failure_without_traceback(self) -> None:
        stderr = io.StringIO()
        with (
            patch.object(
                setup.driver, "ensure_driver", side_effect=RuntimeError("test failure")
            ),
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            setup.run(SimpleNamespace(reassign=False))
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(stderr.getvalue(), "ERROR: test failure\n")


class CanDriverIdentityTest(unittest.TestCase):
    def test_lockdown_also_counts_as_signature_enforcement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sig_enforce = root / "sig_enforce"
            lockdown = root / "lockdown"
            sig_enforce.write_text("N\n")
            lockdown.write_text("none [integrity] confidentiality\n")
            with (
                patch.object(driver, "_SIGNATURE_ENFORCEMENT", sig_enforce),
                patch.object(driver, "_LOCKDOWN_STATE", lockdown),
            ):
                self.assertTrue(driver._signature_enforced())
            lockdown.write_text("[none] integrity confidentiality\n")
            with (
                patch.object(driver, "_SIGNATURE_ENFORCEMENT", sig_enforce),
                patch.object(driver, "_LOCKDOWN_STATE", lockdown),
            ):
                self.assertFalse(driver._signature_enforced())

    def test_optional_canable_alias_is_required_only_while_attached(self) -> None:
        with patch.object(driver, "_attached_usb_ids", return_value=set()):
            self.assertEqual(driver._required_aliases(), {driver._HUB_ALIAS})
        with patch.object(driver, "_attached_usb_ids", return_value={("16d0", "117e")}):
            self.assertEqual(
                driver._required_aliases(),
                {driver._HUB_ALIAS, driver._CANABLE2_ALIAS},
            )

    def test_weak_updates_symlink_resolves_to_native_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            modules = Path(directory) / "modules"
            native = modules / "6.13.0" / "kernel" / "drivers" / "gs_usb.ko"
            weak = modules / "6.13.0" / "weak-updates" / "gs_usb.ko"
            native.parent.mkdir(parents=True)
            weak.parent.mkdir(parents=True)
            native.write_bytes(b"native")
            weak.symlink_to(native)
            with (
                patch.object(driver, "_MODULES_ROOT", modules),
                patch.object(
                    driver.os,
                    "uname",
                    return_value=SimpleNamespace(release="6.13.0"),
                ),
                patch.object(driver, "_selected_driver_path", return_value=str(weak)),
            ):
                self.assertTrue(driver._selected_driver_is_native())
                self.assertTrue(driver._selected_driver_has_hub_fixes())

    def test_native_hub_fix_boundary_is_linux_6_13(self) -> None:
        with (
            patch.object(driver, "_selected_driver_path", return_value="(builtin)"),
            patch.object(
                driver.os,
                "uname",
                return_value=SimpleNamespace(release="6.12.99-custom"),
            ),
        ):
            self.assertFalse(driver._selected_driver_has_hub_fixes())

        with (
            patch.object(driver, "_selected_driver_path", return_value="(builtin)"),
            patch.object(
                driver.os,
                "uname",
                return_value=SimpleNamespace(release="6.13.0"),
            ),
        ):
            self.assertTrue(driver._selected_driver_has_hub_fixes())

    def test_signed_native_stable_backport_is_kept_under_secure_boot(self) -> None:
        native = "/lib/modules/6.8.0/kernel/drivers/net/can/usb/gs_usb.ko.zst"
        with (
            patch.object(driver, "is_driver_available", return_value=True),
            patch.object(driver, "_signature_enforced", return_value=True),
            patch.object(driver, "_module_signer", return_value="distro kernel key"),
            patch.object(driver, "_selected_driver_path", return_value=native),
            patch.object(driver, "_load_available_driver"),
            patch.object(driver, "_driver_supports_required_ids", return_value=True),
            patch.object(driver, "_selected_driver_is_native", return_value=True),
            patch.object(
                driver.os,
                "uname",
                return_value=SimpleNamespace(release="6.8.0-60-generic"),
            ),
            patch.object(driver, "_module_references_symbol", return_value=True),
            patch.object(driver, "_build") as build,
        ):
            self.assertFalse(driver.ensure_driver())
        build.assert_not_called()

    def test_native_stable_backport_capability_requires_exact_symbol(self) -> None:
        similar = SimpleNamespace(
            returncode=0, stdout="0x12345678 usb_find_common_endpoints_extra\n"
        )
        exact = SimpleNamespace(
            returncode=0, stdout="0x87654321 usb_find_common_endpoints\n"
        )
        with (
            patch.object(driver, "_find_modprobe", return_value="/sbin/modprobe"),
            patch.object(driver.subprocess, "run", side_effect=(similar, exact)) as run,
        ):
            self.assertFalse(
                driver._module_references_symbol(
                    "usb_find_common_endpoints", "/modules/gs_usb.ko.zst"
                )
            )
            self.assertTrue(
                driver._module_references_symbol(
                    "usb_find_common_endpoints", "/modules/gs_usb.ko.zst"
                )
            )
        self.assertEqual(run.call_count, 2)
        run.assert_called_with(
            [
                "/sbin/modprobe",
                "--show-modversions",
                "/modules/gs_usb.ko.zst",
            ],
            capture_output=True,
            text=True,
        )

    def test_no_srcversion_uses_native_taint_identity_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modules = root / "modules"
            native = modules / "6.14.0" / "kernel" / "drivers" / "gs_usb.ko"
            loaded = root / "loaded"
            native.parent.mkdir(parents=True)
            native.write_bytes(b"native")
            loaded.mkdir()
            (loaded / "taint").write_text("")
            common = (
                patch.object(driver, "_MODULES_ROOT", modules),
                patch.object(driver, "_LOADED_MODULE", loaded),
                patch.object(
                    driver.os,
                    "uname",
                    return_value=SimpleNamespace(release="6.14.0"),
                ),
                patch.object(driver, "_selected_driver_path", return_value=str(native)),
                patch.object(driver, "_selected_driver_srcversion", return_value=""),
                patch.object(driver, "run_root"),
            )
            with (
                common[0],
                common[1],
                common[2],
                common[3],
                common[4],
                common[5] as root_run,
            ):
                driver._load_available_driver()
                root_run.assert_not_called()

            (loaded / "taint").write_text("O")
            with common[0], common[1], common[2], common[3], common[4], common[5]:
                with self.assertRaisesRegex(RuntimeError, "enough build identity"):
                    driver._load_available_driver()

    def test_signed_legacy_vendored_fingerprint_is_narrow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modules = root / "modules"
            loaded = root / "loaded"
            destination = modules / "5.15.0-tegra" / "updates" / "gs_usb.ko"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"legacy")
            loaded.mkdir()
            (loaded / "taint").write_text("O")
            (loaded / "srcversion").write_text("LEGACY_SOURCE")
            aliases = "\n".join(driver._LEGACY_VENDORED_ALIASES)

            def module_field(field: str, _module: str = "gs_usb") -> str:
                return {
                    "version": "",
                    "intree": "N",
                    "srcversion": "LEGACY_SOURCE",
                }.get(field, "")

            patches = (
                patch.object(driver, "_MODULES_ROOT", modules),
                patch.object(driver, "_LOADED_MODULE", loaded),
                patch.object(
                    driver.os,
                    "uname",
                    return_value=SimpleNamespace(release="5.15.0-tegra"),
                ),
                patch.object(
                    driver, "_selected_driver_path", return_value=str(destination)
                ),
                patch.object(driver, "_signature_enforced", return_value=True),
                patch.object(driver, "_module_signer", return_value="enrolled key"),
                patch.object(driver, "_module_field", side_effect=module_field),
                patch.object(driver, "_module_info", return_value=aliases),
            )
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patches[5],
                patches[6],
                patches[7],
            ):
                self.assertTrue(driver._selected_driver_is_legacy_vendored())
                driver._load_available_driver()
                self.assertTrue(driver._selected_driver_has_hub_fixes())

            (loaded / "srcversion").write_text("DIFFERENT_SOURCE")
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patches[5],
                patches[6],
                patches[7],
            ):
                self.assertFalse(driver._selected_driver_is_legacy_vendored())
            (loaded / "srcversion").write_text("LEGACY_SOURCE")

            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patch.object(driver, "_module_signer", return_value=""),
                patches[6],
                patches[7],
            ):
                self.assertFalse(driver._selected_driver_is_legacy_vendored())

            incomplete_aliases = "\n".join(driver._LEGACY_VENDORED_ALIASES[:-1])
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patches[5],
                patches[6],
                patch.object(driver, "_module_info", return_value=incomplete_aliases),
            ):
                self.assertFalse(driver._selected_driver_is_legacy_vendored())


if __name__ == "__main__":
    unittest.main()
