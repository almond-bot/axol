from __future__ import annotations

import asyncio
import contextlib
import io
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx

from almond_axol.cli.can import setup
from almond_axol.serve import app as app_module
from almond_axol.serve.manager import Session
from almond_axol.serve.settings import SettingsStore
from tests.test_serve_session_reservation import (
    _Manager,
    _Robot,
    _Runner,
    _test_app,
)


def _adapter(
    serial: str,
    identity: str,
    names: tuple[str, ...],
    dev_ids: tuple[int, ...],
    *,
    vid: str = setup._VID,
    pid: str = setup._PID,
) -> dict:
    del serial
    return {
        "vid": vid,
        "pid": pid,
        "dev_ids": set(dev_ids),
        "interfaces": list(zip(names, dev_ids, strict=True)),
        "physical_identities": [
            (name, dev_id, identity)
            for name, dev_id in zip(names, dev_ids, strict=True)
        ],
    }


def _state(
    *,
    profiles: set[str] | frozenset[str] = frozenset(),
    candidates: tuple[tuple[str, str], ...] = (),
    validation: tuple[tuple[str, ...], ...] = (),
) -> setup.AttachedHubState:
    return setup.AttachedHubState(
        configured_profiles=frozenset(profiles),
        candidate_identities=candidates,
        validation_identity=validation,
    )


class AttachedHubStateTest(unittest.TestCase):
    def _snapshot(
        self,
        *,
        devices: tuple[tuple[str, str], ...],
        adapters: dict[str, dict],
        dual: dict[str, str | None] | None = None,
        wheels: str | None = None,
        chest: str | None = None,
    ) -> setup.AttachedHubState:
        claims = dual or {"axol": None, "mantis": None}
        singles = {setup.CAN_BASE: wheels, setup.CAN_CHEST: chest}
        with (
            patch.object(
                setup, "_attached_supported_usb_devices", return_value=devices
            ),
            patch.object(setup, "_configured_profile_usb_serials", return_value=claims),
            patch.object(
                setup,
                "_configured_named_serial",
                side_effect=lambda name: singles[name],
            ),
            patch.object(setup, "_scan_adapters", return_value=adapters),
        ):
            return setup.attached_hub_state()

    def test_wheel_and_chest_only_are_not_unassigned_hubs(self) -> None:
        for role in ("wheel", "chest"):
            with self.subTest(role=role):
                state = self._snapshot(
                    devices=(("1-1", "SINGLE"),),
                    adapters={"SINGLE": _adapter("SINGLE", "1-1", ("can0",), (0,))},
                    wheels="SINGLE" if role == "wheel" else None,
                    chest="SINGLE" if role == "chest" else None,
                )
                self.assertEqual(state.candidate_count, 0)
                self.assertEqual(state.validation_identity, ())

    def test_axol_plus_wheel_has_one_trusted_profile_and_no_candidate(self) -> None:
        state = self._snapshot(
            devices=(("1-1", "AXOL"), ("1-2", "WHEEL")),
            adapters={
                "AXOL": _adapter(
                    "AXOL", "1-1", (setup.CAN_LEFT, setup.CAN_RIGHT), (0, 1)
                ),
                "WHEEL": _adapter("WHEEL", "1-2", (setup.CAN_BASE,), (0,)),
            },
            dual={"axol": "AXOL", "mantis": None},
            wheels="WHEEL",
        )
        self.assertEqual(state.configured_profiles, frozenset({"axol"}))
        self.assertEqual(state.candidate_count, 0)
        self.assertTrue(state.validation_identity)

    def test_cross_role_or_duplicate_physical_claim_is_never_trusted(self) -> None:
        overlap = self._snapshot(
            devices=(("1-1", "SHARED"),),
            adapters={
                "SHARED": _adapter(
                    "SHARED", "1-1", (setup.CAN_LEFT, setup.CAN_RIGHT), (0, 1)
                )
            },
            dual={"axol": "SHARED", "mantis": None},
            wheels="SHARED",
        )
        self.assertFalse(overlap.configured_profiles)
        self.assertEqual(overlap.candidate_count, 1)

        duplicate = self._snapshot(
            devices=(("1-1", "DUP"), ("1-2", "DUP")),
            adapters={
                "DUP": _adapter("DUP", "1-1", (setup.CAN_LEFT, setup.CAN_RIGHT), (0, 1))
            },
            dual={"axol": "DUP", "mantis": None},
        )
        self.assertFalse(duplicate.configured_profiles)
        self.assertEqual(duplicate.candidate_count, 2)

    def test_profile_names_must_belong_to_the_claimed_serial(self) -> None:
        state = self._snapshot(
            devices=(("1-1", "AXOL"),),
            adapters={
                "AXOL": _adapter("AXOL", "1-1", ("can2", "can3"), (0, 1)),
                "STALE": _adapter(
                    "STALE",
                    "9-9",
                    (setup.CAN_LEFT, setup.CAN_RIGHT),
                    (0, 1),
                    vid="1209",
                    pid="2323",
                ),
            },
            dual={"axol": "AXOL", "mantis": None},
        )
        self.assertFalse(state.configured_profiles)
        self.assertEqual(state.candidate_identities, (("1-1", "AXOL"),))

    def test_predriver_bootstrap_rejects_names_occupied_by_another_serial(
        self,
    ) -> None:
        state = self._snapshot(
            devices=(("1-1", "AXOL"),),
            adapters={
                "STALE": _adapter(
                    "STALE",
                    "9-9",
                    (setup.CAN_LEFT, setup.CAN_RIGHT),
                    (0, 1),
                    vid="1209",
                    pid="2323",
                )
            },
            dual={"axol": "AXOL", "mantis": None},
        )
        self.assertFalse(state.configured_profiles)
        self.assertEqual(state.candidate_identities, (("1-1", "AXOL"),))

    def test_generic_non_hub_single_channel_adapter_remains_detectable(self) -> None:
        generic = _adapter("GENERIC", "2-1", ("can7",), (0,), vid="1209", pid="2323")
        with (
            patch.object(setup, "_attached_supported_usb_devices", return_value=()),
            patch.object(setup, "_scan_adapters", return_value={"GENERIC": generic}),
        ):
            self.assertEqual(setup._detect_single_serials(set()), ["GENERIC"])


class HeadlessHubResolverTest(unittest.TestCase):
    def _run(
        self,
        observations: dict[str, setup.DualHubIdentity | None],
        *,
        strict_axol: str | None = None,
        strict_mantis: str | None = None,
        live_axol: str | None = None,
        live_mantis: str | None = None,
        wheels: str | None = None,
        chest: str | None = None,
        candidates_before: tuple[str, ...] | None = None,
        candidates_after: tuple[str, ...] = (),
        second_topology: object | None = None,
        legacy_mantis: bool = False,
        raw_names: frozenset[str] = frozenset(),
    ) -> tuple[setup.HeadlessHubSetupResult, dict[str, Mock]]:
        serials = tuple(observations)
        devices = tuple(
            (f"1-{index + 1}", serial) for index, serial in enumerate(serials)
        )

        def interface_names(serial: str, index: int) -> tuple[str, str]:
            if serial not in raw_names and serial == strict_axol:
                return setup.CAN_LEFT, setup.CAN_RIGHT
            if serial not in raw_names and serial == strict_mantis:
                return setup.CAN_MANTIS_LEFT, setup.CAN_MANTIS_RIGHT
            return f"can{index * 2}", f"can{index * 2 + 1}"

        records = tuple(
            (
                serial,
                setup._VID,
                setup._PID,
                (
                    (interface_names(serial, index)[0], 0, identity),
                    (interface_names(serial, index)[1], 1, identity),
                ),
            )
            for index, (identity, serial) in enumerate(devices)
        )
        topology = (devices, records)
        before_serials = candidates_before if candidates_before is not None else serials
        states = [
            _state(
                candidates=tuple(
                    next(device for device in devices if device[1] == serial)
                    for serial in before_serials
                )
            ),
            _state(
                candidates=tuple(
                    next(device for device in devices if device[1] == serial)
                    for serial in candidates_after
                )
            ),
        ]
        strict = {"axol": strict_axol, "mantis": strict_mantis}
        live = {
            "axol": live_axol if live_axol is not None else strict_axol,
            "mantis": live_mantis if live_mantis is not None else strict_mantis,
        }
        singles = {setup.CAN_BASE: wheels, setup.CAN_CHEST: chest}

        def persisted_rule(path: object, _left: str, _right: str) -> str | None:
            if path == setup._MANTIS_PROFILE.rules_file:
                return None if legacy_mantis else strict_mantis
            if path == setup._PRE_MANTIS_RULES_FILE:
                return strict_mantis if legacy_mantis else None
            return strict_axol

        operations = {
            "apply": Mock(),
            "mantis": Mock(),
            "rules": Mock(),
            "reload": Mock(),
            "rename": Mock(),
            "remove_legacy": Mock(),
        }
        topology_sequence = [topology, second_topology or topology]
        with (
            patch.object(
                setup, "_attached_supported_usb_devices", return_value=devices
            ),
            patch.object(setup.driver, "ensure_driver", return_value=False),
            patch.object(setup, "_wait_for_dual_channel_serial", return_value=True),
            patch.object(
                setup, "_headless_topology_snapshot", side_effect=topology_sequence
            ),
            patch.object(setup, "_configured_profile_usb_serials", return_value=strict),
            patch.object(
                setup,
                "_configured_serial",
                side_effect=lambda profile=setup._AXOL_PROFILE: live[
                    "mantis" if profile is setup._MANTIS_PROFILE else "axol"
                ],
            ),
            patch.object(
                setup,
                "_configured_named_serial",
                side_effect=lambda name: singles[name],
            ),
            patch.object(
                setup, "_dual_channel_rule_serial", side_effect=persisted_rule
            ),
            patch.object(setup, "attached_hub_state", side_effect=states),
            patch.object(
                setup,
                "_identify_dual_adapter",
                side_effect=lambda serial, **_kwargs: observations[serial],
            ),
            patch.object(setup, "_apply_setup", operations["apply"]),
            patch.object(setup, "_configure_mantis", operations["mantis"]),
            patch.object(setup, "_write_udev_rules", operations["rules"]),
            patch.object(setup, "_reload_udev", operations["reload"]),
            patch.object(setup, "_rename_interfaces", operations["rename"]),
            patch.object(
                setup, "_remove_pre_mantis_config", operations["remove_legacy"]
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = setup._setup_detected_hubs_locked()
        return result, operations

    def test_fresh_axol_and_mantis_are_assigned_only_by_positive_identity(self) -> None:
        result, calls = self._run({"A": "axol", "M": "mantis"})
        self.assertEqual(result.status, "configured")
        calls["apply"].assert_called_once_with("A", None, None)
        calls["mantis"].assert_called_once_with("M")

    def test_silence_conflict_and_duplicate_fresh_role_never_guess(self) -> None:
        result, calls = self._run({"S": "silent"}, candidates_after=("S",))
        self.assertEqual(result.status, "unidentified")
        calls["apply"].assert_not_called()
        calls["mantis"].assert_not_called()

        with self.assertRaisesRegex(RuntimeError, "both Axol and Mantis"):
            self._run({"C": "conflict"}, candidates_after=("C",))

        duplicate, calls = self._run(
            {"A": "axol", "B": "axol"}, candidates_after=("A", "B")
        )
        self.assertEqual(duplicate.status, "unidentified")
        calls["apply"].assert_not_called()

    def test_exact_axol_mantis_swap_rewrites_both_targets(self) -> None:
        result, calls = self._run(
            {"A": "mantis", "B": "axol"},
            strict_axol="A",
            strict_mantis="B",
        )
        self.assertEqual(result.status, "configured")
        calls["apply"].assert_called_once_with("B", None, None)
        calls["mantis"].assert_called_once_with("A")

    def test_attached_silent_incumbent_is_not_replaced_by_same_role(self) -> None:
        result, calls = self._run(
            {"A": "silent", "C": "axol"},
            strict_axol="A",
            candidates_before=("C",),
            candidates_after=("C",),
        )
        self.assertEqual(result.status, "unidentified")
        calls["apply"].assert_not_called()

    def test_positive_opposite_clears_stale_claim_even_if_incumbent_retains_role(
        self,
    ) -> None:
        result, calls = self._run(
            {"A": "mantis", "B": "silent"},
            strict_axol="A",
            strict_mantis="B",
            candidates_after=("A",),
        )
        self.assertEqual(result.status, "unidentified")
        calls["rules"].assert_called_once_with(None, None, None)
        calls["rename"].assert_called_once_with(None, None, None)
        calls["mantis"].assert_not_called()

    def test_same_serial_dual_claim_reapplies_surviving_profile(self) -> None:
        _, mantis = self._run({"A": "mantis"}, strict_axol="A", strict_mantis="A")
        mantis["rules"].assert_called_once_with(None, None, None)
        mantis["rename"].assert_called_once_with(None, None, None)
        mantis["mantis"].assert_called_once_with("A")

        _, axol = self._run({"A": "axol"}, strict_axol="A", strict_mantis="A")
        axol["apply"].assert_called_once_with("A", None, None)
        axol["rules"].assert_called_once_with(None, profile=setup._MANTIS_PROFILE)
        axol["rename"].assert_called_once_with(None, profile=setup._MANTIS_PROFILE)

    def test_selected_mantis_reapplies_after_stale_auxiliary_claim_is_cleared(
        self,
    ) -> None:
        _, calls = self._run({"M": "mantis"}, strict_mantis="M", chest="M")
        calls["rules"].assert_called_once_with(None, None, None)
        calls["rename"].assert_called_once_with(None, None, None)
        calls["mantis"].assert_called_once_with("M")

    def test_strict_rule_wins_over_transient_live_managed_name(self) -> None:
        _, calls = self._run(
            {"LIVE": "mantis"},
            strict_axol="SAVED",
            live_axol="LIVE",
        )
        calls["apply"].assert_not_called()
        calls["mantis"].assert_called_once_with("LIVE")

    def test_raw_named_strict_profile_is_idempotently_repaired(self) -> None:
        _, calls = self._run(
            {"A": "axol"},
            strict_axol="A",
            candidates_before=("A",),
            raw_names=frozenset({"A"}),
        )
        calls["apply"].assert_called_once_with("A", None, None)

    def test_legacy_mantis_is_migrated_even_when_serial_is_unchanged(self) -> None:
        _, calls = self._run(
            {"M": "mantis"},
            strict_mantis="M",
            candidates_before=(),
            legacy_mantis=True,
        )
        calls["mantis"].assert_called_once_with("M")

    def test_topology_change_aborts_before_any_write(self) -> None:
        changed = ((("9-9", "A"),), ())
        with self.assertRaisesRegex(RuntimeError, "topology changed"):
            self._run({"A": "axol"}, second_topology=changed)

    def test_fresh_silent_ensure_setup_never_defaults_to_axol(self) -> None:
        apply = Mock()
        with (
            patch.object(setup.driver, "ensure_driver", return_value=False),
            patch.object(setup, "_attached_configured_hub_serials", return_value={}),
            patch.object(setup, "_configured_serial", return_value=None),
            patch.object(setup, "_mantis_claimed_serials", return_value=set()),
            patch.object(setup, "_detect_serials", return_value=["SILENT"]),
            patch.object(setup, "_identify_dual_adapter", return_value="silent"),
            patch.object(setup, "_configured_named_serial", return_value=None),
            patch.object(setup, "_apply_setup", apply),
            self.assertRaisesRegex(RuntimeError, "Axol not identified"),
        ):
            setup._ensure_setup_locked(
                hub_serial=None, wheels_serial=None, chest_serial=None
            )
        apply.assert_not_called()

    def test_unsafe_serial_is_rejected_before_root_publication(self) -> None:
        publish = Mock()
        with (
            patch.object(setup, "_publish_privileged_text", publish),
            self.assertRaisesRegex(ValueError, "safe ASCII"),
        ):
            setup._write_udev_rules('bad"\\\nserial', None, None)
        publish.assert_not_called()

    def test_duplicate_physical_serial_is_rejected_before_probe_or_write(
        self,
    ) -> None:
        devices = (("1-1", "DUP"), ("1-2", "DUP"))
        topology = (
            devices,
            (
                (
                    "DUP",
                    setup._VID,
                    setup._PID,
                    (("can0", 0, "1-1"), ("can1", 1, "1-1")),
                ),
            ),
        )
        identify = Mock()
        apply_setup = Mock()
        with (
            patch.object(
                setup, "_attached_supported_usb_devices", return_value=devices
            ),
            patch.object(setup.driver, "ensure_driver", return_value=False),
            patch.object(setup, "_wait_for_dual_channel_serial", return_value=True),
            patch.object(setup, "_headless_topology_snapshot", return_value=topology),
            patch.object(
                setup,
                "_configured_profile_usb_serials",
                return_value={"axol": None, "mantis": None},
            ),
            patch.object(setup, "_configured_serial", return_value=None),
            patch.object(setup, "_configured_named_serial", return_value=None),
            patch.object(setup, "_identify_dual_adapter", identify),
            patch.object(setup, "_apply_setup", apply_setup),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "Duplicate CAN adapter serials"),
        ):
            setup._setup_detected_hubs_locked()
        identify.assert_not_called()
        apply_setup.assert_not_called()

    def test_malformed_persisted_dual_topology_is_rejected_before_probe(
        self,
    ) -> None:
        devices = (("1-1", "AXOL"),)
        topology = (
            devices,
            (("AXOL", setup._VID, setup._PID, (("can0", 0, "1-1"),)),),
        )
        identify = Mock()
        with (
            patch.object(
                setup, "_attached_supported_usb_devices", return_value=devices
            ),
            patch.object(setup.driver, "ensure_driver", return_value=False),
            patch.object(setup, "_wait_for_dual_channel_serial", return_value=False),
            patch.object(setup, "_headless_topology_snapshot", return_value=topology),
            patch.object(
                setup,
                "_configured_profile_usb_serials",
                return_value={"axol": "AXOL", "mantis": None},
            ),
            patch.object(
                setup,
                "_configured_serial",
                side_effect=lambda profile=setup._AXOL_PROFILE: (
                    "AXOL" if profile is setup._AXOL_PROFILE else None
                ),
            ),
            patch.object(setup, "_configured_named_serial", return_value=None),
            patch.object(setup, "_identify_dual_adapter", identify),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "incomplete or ambiguous"),
        ):
            setup._setup_detected_hubs_locked()
        identify.assert_not_called()


class CanDiscoveryCacheTest(unittest.TestCase):
    def test_generation_tracks_candidate_and_validation_epoch(self) -> None:
        cache = app_module._CanDiscoveryCache()
        first = _state(
            candidates=(("usb-1", "SECRET-A"),),
            validation=(("usb", "usb-1", "SECRET-A"),),
        )
        cache.observe(first)
        self.assertEqual(
            cache.payload(),
            {
                "status": "needed",
                "candidateCount": 1,
                "generation": 1,
            },
        )

        cache.finish(first, status="unidentified", message="power hardware")
        cache.observe(first)
        self.assertEqual(cache.payload()["status"], "unidentified")
        self.assertEqual(cache.payload()["generation"], 1)

        changed_validation = _state(
            candidates=(("usb-1", "SECRET-A"),),
            validation=(("usb", "usb-1@dev2", "SECRET-A"),),
        )
        cache.observe(changed_validation)
        self.assertEqual(cache.payload()["status"], "needed")
        self.assertEqual(cache.payload()["generation"], 2)

        changed_candidate = _state(
            candidates=(("usb-2", "SECRET-B"),),
            validation=changed_validation.validation_identity,
        )
        cache.observe(changed_candidate)
        self.assertEqual(cache.payload()["generation"], 3)
        self.assertNotIn("SECRET", repr(cache.payload()))


class CanDiscoveryApiTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _transport(
        *,
        robot: _Robot | None = None,
        runner: _Runner | None = None,
        manager: _Manager | None = None,
        settings: SettingsStore | None = None,
    ) -> tuple[object, _Robot, _Runner, _Manager, httpx.ASGITransport]:
        robot = robot or _Robot()
        runner = runner or _Runner()
        manager = manager or _Manager()
        app = _test_app(manager, runner, robot, settings=settings)
        return app, robot, runner, manager, httpx.ASGITransport(app=app)

    async def test_inventory_exposes_only_count_generation_and_root_needs_validation(
        self,
    ) -> None:
        attached = _state(
            profiles={"axol"},
            validation=(("usb", "usb-1", "PRIVATE-SERIAL"),),
        )
        _app, _robot, _runner, _manager, transport = self._transport()
        with (
            patch.object(app_module.os, "geteuid", return_value=0),
            patch.object(app_module, "_list_can_interfaces", return_value=[]),
            patch.object(app_module, "_attached_hub_state", return_value=attached),
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.get("/api/can/interfaces")

        self.assertEqual(response.status_code, 200)
        discovery = response.json()["discovery"]
        self.assertEqual(discovery["status"], "needed")
        self.assertEqual(discovery["candidateCount"], 0)
        self.assertGreater(discovery["generation"], 0)
        self.assertNotIn("PRIVATE-SERIAL", response.text)

    async def test_nonroot_trusted_profile_is_ready_but_discovery_is_forbidden(
        self,
    ) -> None:
        attached = _state(
            profiles={"axol"},
            validation=(("usb", "usb-1", "SERIAL"),),
        )
        _app, robot, _runner, _manager, transport = self._transport()
        discover = Mock()
        with (
            patch.object(app_module.os, "geteuid", return_value=1000),
            patch.object(app_module, "_list_can_interfaces", return_value=[]),
            patch.object(app_module, "_attached_hub_state", return_value=attached),
            patch.object(setup, "setup_detected_hubs", discover),
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                inventory = await client.get("/api/can/interfaces")
                forbidden = await client.post("/api/can/discover")

        self.assertEqual(inventory.json()["discovery"]["status"], "ready")
        self.assertEqual(forbidden.status_code, 403)
        self.assertEqual(robot.disconnects, 0)
        discover.assert_not_called()

    async def test_nonroot_profile_with_unresolved_candidate_stays_needed(
        self,
    ) -> None:
        attached = _state(
            profiles={"axol"},
            candidates=(("usb-2", "UNRESOLVED"),),
            validation=(
                ("usb", "usb-1", "AXOL"),
                ("usb", "usb-2", "UNRESOLVED"),
            ),
        )
        _app, robot, _runner, _manager, transport = self._transport()
        discover = Mock()
        with (
            patch.object(app_module.os, "geteuid", return_value=1000),
            patch.object(app_module, "_list_can_interfaces", return_value=[]),
            patch.object(app_module, "_attached_hub_state", return_value=attached),
            patch.object(setup, "setup_detected_hubs", discover),
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                inventory = await client.get("/api/can/interfaces")
                forbidden = await client.post("/api/can/discover")

        self.assertEqual(inventory.json()["discovery"]["status"], "needed")
        self.assertEqual(inventory.json()["discovery"]["candidateCount"], 1)
        self.assertEqual(forbidden.status_code, 403)
        self.assertEqual(robot.disconnects, 0)
        discover.assert_not_called()

    async def test_success_is_single_epoch_and_delayed_second_post_is_noop(
        self,
    ) -> None:
        initial = _state(
            candidates=(("usb-1", "SERIAL"),),
            validation=(("usb", "usb-1", "SERIAL"),),
        )
        final = _state(
            profiles={"axol"},
            validation=(
                ("usb", "usb-1", "SERIAL"),
                ("claim", "axol", "SERIAL"),
            ),
        )
        current = {"state": initial}
        _app, robot, _runner, _manager, transport = self._transport()

        def discover_hubs() -> setup.HeadlessHubSetupResult:
            current["state"] = final
            return setup.HeadlessHubSetupResult("configured", 1)

        discover = Mock(side_effect=discover_hubs)
        with (
            patch.object(app_module.os, "geteuid", return_value=0),
            patch.object(app_module, "_list_can_interfaces", return_value=[]),
            patch.object(
                app_module,
                "_attached_hub_state",
                side_effect=lambda: current["state"],
            ),
            patch.object(setup, "setup_detected_hubs", discover),
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                first = await client.post("/api/can/discover")
                robot.connect()
                second = await client.post("/api/can/discover")

        self.assertEqual(first.json()["discovery"]["status"], "configured")
        self.assertEqual(second.json()["discovery"]["status"], "configured")
        self.assertEqual(discover.call_count, 1)
        self.assertEqual(robot.disconnects, 1)
        self.assertEqual(robot.connects, 1)

    async def test_cancelled_request_leaves_singleflight_worker_running(self) -> None:
        initial = _state(
            candidates=(("usb-1", "SERIAL"),),
            validation=(("usb", "usb-1", "SERIAL"),),
        )
        final = _state(
            profiles={"mantis"},
            validation=(("claim", "mantis", "SERIAL"),),
        )
        current = {"state": initial}
        entered = threading.Event()
        release = threading.Event()
        _app, robot, _runner, _manager, transport = self._transport()

        def discover_hubs() -> setup.HeadlessHubSetupResult:
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test discovery gate timed out")
            current["state"] = final
            return setup.HeadlessHubSetupResult("configured", 1)

        discover = Mock(side_effect=discover_hubs)
        with (
            patch.object(app_module.os, "geteuid", return_value=0),
            patch.object(app_module, "_list_can_interfaces", return_value=[]),
            patch.object(
                app_module,
                "_attached_hub_state",
                side_effect=lambda: current["state"],
            ),
            patch.object(setup, "setup_detected_hubs", discover),
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                request = asyncio.create_task(client.post("/api/can/discover"))
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                request.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await request

                running = await client.get("/api/can/interfaces")
                self.assertEqual(running.json()["discovery"]["status"], "running")
                joined = asyncio.create_task(client.post("/api/can/discover"))
                release.set()
                completed = await joined

        self.assertEqual(completed.json()["discovery"]["status"], "configured")
        self.assertEqual(discover.call_count, 1)
        self.assertEqual(robot.disconnects, 1)

    async def test_busy_operation_returns_authoritative_retryable_inventory(
        self,
    ) -> None:
        attached = _state(
            candidates=(("usb-1", "SERIAL"),),
            validation=(("usb", "usb-1", "SERIAL"),),
        )
        runner = _Runner(running=True)
        _app, robot, _runner, _manager, transport = self._transport(runner=runner)
        discover = Mock()
        with (
            patch.object(app_module.os, "geteuid", return_value=0),
            patch.object(app_module, "_list_can_interfaces", return_value=[]),
            patch.object(app_module, "_attached_hub_state", return_value=attached),
            patch.object(setup, "setup_detected_hubs", discover),
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post("/api/can/discover")

        self.assertEqual(response.status_code, 409)
        self.assertTrue(response.json()["retryable"])
        self.assertEqual(response.json()["discovery"]["status"], "needed")
        self.assertEqual(robot.disconnects, 0)
        discover.assert_not_called()

    async def test_diagnostic_and_maintenance_ownership_block_discovery(
        self,
    ) -> None:
        attached = _state(
            candidates=(("usb-1", "SERIAL"),),
            validation=(("usb", "usb-1", "SERIAL"),),
        )
        diagnostic = Session("motor.info", {})
        diagnostic.status = "running"
        _app, robot, _runner, _manager, transport = self._transport(
            manager=_Manager([diagnostic])
        )
        discover = Mock()
        with (
            patch.object(app_module.os, "geteuid", return_value=0),
            patch.object(app_module, "_list_can_interfaces", return_value=[]),
            patch.object(app_module, "_attached_hub_state", return_value=attached),
            patch.object(setup, "setup_detected_hubs", discover),
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                diagnostic_block = await client.post("/api/can/discover")

        self.assertEqual(diagnostic_block.status_code, 409)
        self.assertTrue(diagnostic_block.json()["retryable"])
        self.assertEqual(robot.disconnects, 0)
        discover.assert_not_called()

        _app, robot, _runner, _manager, transport = self._transport()
        with (
            patch.object(app_module.os, "geteuid", return_value=0),
            patch.object(app_module, "_list_can_interfaces", return_value=[]),
            patch.object(app_module, "_attached_hub_state", return_value=attached),
            patch.object(setup, "setup_detected_hubs", discover),
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                maintenance_started = await client.post("/api/update/start")
                maintenance_block = await client.post("/api/can/discover")

        self.assertEqual(maintenance_started.status_code, 200)
        self.assertEqual(maintenance_block.status_code, 409)
        self.assertIn("maintenance", maintenance_block.json()["error"])
        self.assertEqual(robot.disconnects, 0)
        discover.assert_not_called()

    async def test_disconnect_must_be_proven_before_setup(self) -> None:
        attached = _state(
            candidates=(("usb-1", "SERIAL"),),
            validation=(("usb", "usb-1", "SERIAL"),),
        )
        robot = _Robot()
        robot.disconnect = Mock(return_value=robot.status())  # type: ignore[method-assign]
        _app, _robot, _runner, _manager, transport = self._transport(robot=robot)
        discover = Mock()
        with (
            patch.object(app_module.os, "geteuid", return_value=0),
            patch.object(app_module, "_list_can_interfaces", return_value=[]),
            patch.object(app_module, "_attached_hub_state", return_value=attached),
            patch.object(setup, "setup_detected_hubs", discover),
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post("/api/can/discover")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["discovery"]["status"], "error")
        discover.assert_not_called()

    async def test_error_link_is_cleaned_up_before_discovery(self) -> None:
        attached = _state(
            profiles={"axol"},
            validation=(("usb", "usb-1", "SERIAL"),),
        )
        robot = _Robot(state="error")
        _app, _robot, _runner, _manager, transport = self._transport(robot=robot)
        discover = Mock(return_value=setup.HeadlessHubSetupResult("ready", 0))
        with (
            patch.object(app_module.os, "geteuid", return_value=0),
            patch.object(app_module, "_list_can_interfaces", return_value=[]),
            patch.object(app_module, "_attached_hub_state", return_value=attached),
            patch.object(setup, "setup_detected_hubs", discover),
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post("/api/can/discover")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["discovery"]["status"], "ready")
        self.assertEqual(robot.disconnects, 1)
        discover.assert_called_once_with()

    async def test_managed_connect_waits_for_root_validation(self) -> None:
        attached = _state(
            profiles={"axol"},
            validation=(("usb", "usb-1", "SERIAL"),),
        )
        with tempfile.TemporaryDirectory() as directory:
            settings = SettingsStore(Path(directory) / "settings.json")
            robot = _Robot(channels=settings.can_channels(), profile="axol")
            _app, _robot, _runner, _manager, transport = self._transport(
                robot=robot, settings=settings
            )
            discover = Mock(return_value=setup.HeadlessHubSetupResult("ready", 0))
            with (
                patch.object(app_module.os, "geteuid", return_value=0),
                patch.object(app_module, "_list_can_interfaces", return_value=[]),
                patch.object(app_module, "_attached_hub_state", return_value=attached),
                patch.object(setup, "setup_detected_hubs", discover),
            ):
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    blocked = await client.post("/api/robot/connect")
                    validation = await client.post("/api/can/discover")
                    connected = await client.post("/api/robot/connect")

        self.assertEqual(blocked.status_code, 409)
        self.assertEqual(validation.json()["discovery"]["status"], "ready")
        self.assertEqual(connected.status_code, 200)
        self.assertTrue(connected.json()["connected"])
        self.assertEqual(discover.call_count, 1)
        self.assertEqual(robot.disconnects, 1)
        self.assertEqual(robot.connects, 1)

    async def test_shutdown_waits_for_discovery_worker(self) -> None:
        initial = _state(
            candidates=(("usb-1", "SERIAL"),),
            validation=(("usb", "usb-1", "SERIAL"),),
        )
        final = _state(validation=(("claim", "axol", "SERIAL"),))
        current = {"state": initial}
        entered = threading.Event()
        release = threading.Event()
        app, _robot, _runner, _manager, transport = self._transport()

        def discover_hubs() -> setup.HeadlessHubSetupResult:
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test discovery gate timed out")
            current["state"] = final
            return setup.HeadlessHubSetupResult("configured", 1)

        async def run_shutdown_handlers() -> None:
            for handler in app.router.on_shutdown:  # type: ignore[attr-defined]
                await handler()

        with (
            patch.object(app_module.os, "geteuid", return_value=0),
            patch.object(app_module, "_list_can_interfaces", return_value=[]),
            patch.object(
                app_module,
                "_attached_hub_state",
                side_effect=lambda: current["state"],
            ),
            patch.object(setup, "setup_detected_hubs", side_effect=discover_hubs),
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                request = asyncio.create_task(client.post("/api/can/discover"))
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                shutdown = asyncio.create_task(run_shutdown_handlers())
                await asyncio.sleep(0)
                self.assertFalse(shutdown.done())
                release.set()
                response, _ = await asyncio.gather(request, shutdown)

        self.assertEqual(response.json()["discovery"]["status"], "configured")

    async def test_running_inventory_cannot_overwrite_finished_cache(self) -> None:
        initial = _state(
            candidates=(("usb-1", "SERIAL"),),
            validation=(("usb", "usb-1", "SERIAL"),),
        )
        final = _state(
            profiles={"axol"},
            validation=(("claim", "axol", "SERIAL"),),
        )
        setup_entered = threading.Event()
        release_setup = threading.Event()
        stale_scan_entered = threading.Event()
        release_stale_scan = threading.Event()
        state_calls = 0
        state_lock = threading.Lock()

        def attached_state() -> setup.AttachedHubState:
            nonlocal state_calls
            with state_lock:
                state_calls += 1
                call = state_calls
            if call == 1:
                return initial
            if call == 2:
                stale_scan_entered.set()
                if not release_stale_scan.wait(timeout=5):
                    raise RuntimeError("test stale scan gate timed out")
                return initial
            return final

        def discover_hubs() -> setup.HeadlessHubSetupResult:
            setup_entered.set()
            if not release_setup.wait(timeout=5):
                raise RuntimeError("test setup gate timed out")
            return setup.HeadlessHubSetupResult("configured", 1)

        _app, robot, _runner, _manager, transport = self._transport()
        discover = Mock(side_effect=discover_hubs)
        with (
            patch.object(app_module.os, "geteuid", return_value=0),
            patch.object(app_module, "_list_can_interfaces", return_value=[]),
            patch.object(app_module, "_attached_hub_state", side_effect=attached_state),
            patch.object(setup, "setup_detected_hubs", discover),
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                discovery = asyncio.create_task(client.post("/api/can/discover"))
                self.assertTrue(await asyncio.to_thread(setup_entered.wait, 2))
                stale_get = asyncio.create_task(client.get("/api/can/interfaces"))
                self.assertTrue(await asyncio.to_thread(stale_scan_entered.wait, 2))
                release_setup.set()
                finished = await discovery
                release_stale_scan.set()
                stale = await stale_get
                repeated = await client.post("/api/can/discover")

        self.assertEqual(finished.json()["discovery"]["status"], "configured")
        self.assertEqual(stale.json()["discovery"]["status"], "configured")
        self.assertEqual(repeated.json()["discovery"]["status"], "configured")
        self.assertEqual(discover.call_count, 1)
        self.assertEqual(robot.disconnects, 1)


if __name__ == "__main__":
    unittest.main()
