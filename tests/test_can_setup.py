from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from almond_axol.cli.can import setup


class FindSingleSerialsTest(unittest.TestCase):
    def run_find(
        self,
        *,
        configured_wheels: str | None,
        configured_chest: str | None,
        detected: dict[str, str | None],
        answers: list[str] | None = None,
    ) -> tuple[tuple[str | None, str | None], Mock, Mock, str]:
        configured = {
            setup._CAN_B: configured_wheels,
            setup._CAN_C: configured_chest,
        }
        identify = Mock(side_effect=lambda serial: detected[serial])
        find = Mock(return_value=list(reversed(detected)))
        output = io.StringIO()
        with (
            patch.object(
                setup,
                "_configured_named_serial",
                side_effect=lambda name: configured[name],
            ),
            patch.object(setup, "_detect_single_serials", find),
            patch.object(setup, "_identify_adapter", identify),
            patch("builtins.input", side_effect=answers or []),
            redirect_stdout(output),
        ):
            result = setup._find_single_serials("hub")
        return result, find, identify, output.getvalue()

    def test_reprobes_configured_adapters(self) -> None:
        result, find, identify, output = self.run_find(
            configured_wheels="wheel",
            configured_chest="chest",
            detected={"wheel": "wheels", "chest": "chest"},
        )

        self.assertEqual(result, ("wheel", "chest"))
        find.assert_called_once_with({"hub"})
        self.assertEqual(identify.call_args_list, [call("chest"), call("wheel")])
        self.assertIn("Damiao wheel motors answered", output)
        self.assertIn("jelly_legs board answered", output)

    def test_live_responses_correct_a_wheel_chest_swap(self) -> None:
        result, _, _, _ = self.run_find(
            configured_wheels="adapter-a",
            configured_chest="adapter-b",
            detected={"adapter-a": "chest", "adapter-b": "wheels"},
        )

        self.assertEqual(result, ("adapter-b", "adapter-a"))

    def test_live_roles_override_one_duplicate_stale_pin(self) -> None:
        result, _, _, _ = self.run_find(
            configured_wheels="stale",
            configured_chest="stale",
            detected={
                "live-wheel": "wheels",
                "live-chest": "chest",
            },
        )

        self.assertEqual(result, ("live-wheel", "live-chest"))

    def test_one_live_role_resolves_duplicate_pin_by_elimination(self) -> None:
        result, _, _, _ = self.run_find(
            configured_wheels="stale",
            configured_chest="stale",
            detected={"live-wheel": "wheels"},
        )

        self.assertEqual(result, ("live-wheel", "stale"))

    def test_unresolved_duplicate_stale_pin_is_rejected(self) -> None:
        error = io.StringIO()
        with redirect_stderr(error), self.assertRaises(SystemExit):
            self.run_find(
                configured_wheels="stale",
                configured_chest="stale",
                detected={},
            )
        self.assertIn("pinned as both", error.getvalue())

    def test_silent_configured_adapters_remain_unverified_fallbacks(self) -> None:
        result, _, _, output = self.run_find(
            configured_wheels="wheel",
            configured_chest="chest",
            detected={"wheel": None, "chest": None},
        )

        self.assertEqual(result, ("wheel", "chest"))
        self.assertEqual(output.count("unverified"), 2)

    def test_positive_response_replaces_an_unplugged_pin(self) -> None:
        result, _, _, _ = self.run_find(
            configured_wheels="old-wheel",
            configured_chest=None,
            detected={"new-wheel": "wheels"},
        )

        self.assertEqual(result, ("new-wheel", None))

    def test_unplugged_configured_adapters_keep_their_assignments(self) -> None:
        result, _, identify, output = self.run_find(
            configured_wheels="wheel",
            configured_chest="chest",
            detected={},
        )

        self.assertEqual(result, ("wheel", "chest"))
        identify.assert_not_called()
        self.assertEqual(output.count("is not attached"), 2)

    def test_opposite_response_clears_a_stale_role(self) -> None:
        result, _, _, _ = self.run_find(
            configured_wheels="adapter",
            configured_chest=None,
            detected={"adapter": "chest"},
        )

        self.assertEqual(result, (None, "adapter"))

    def test_duplicate_responses_prefer_the_verified_existing_pin(self) -> None:
        result, _, _, output = self.run_find(
            configured_wheels="wheel-b",
            configured_chest=None,
            detected={"wheel-a": "wheels", "wheel-b": "wheels"},
        )

        self.assertEqual(result, ("wheel-b", None))
        self.assertIn("wheel-a: also identified as the wheels bus", output)

    def test_operator_can_replace_an_unverified_pin(self) -> None:
        result, _, _, output = self.run_find(
            configured_wheels="old-wheel",
            configured_chest=None,
            detected={"unknown": None},
            answers=["w"],
        )

        self.assertEqual(result, ("unknown", None))
        self.assertIn("Replacing unverified configured adapter old-wheel", output)


class RenameInterfacesTest(unittest.TestCase):
    def run_rename(
        self,
        identities: dict[str, tuple[str, int]],
        *,
        wheels: str | None,
        chest: str | None,
    ) -> tuple[dict[str, tuple[str, int]], list[list[str]]]:
        with tempfile.TemporaryDirectory() as directory:
            net_dir = Path(directory)
            current = identities.copy()
            for name in current:
                (net_dir / name).touch()

            def make_path(value: str) -> Path:
                if value == "/sys/class/net":
                    return net_dir
                return Path(value)

            def udev_info(command: list[str], **_kwargs: object) -> SimpleNamespace:
                iface = Path(command[-1]).name
                serial, dev_id = current[iface]
                return SimpleNamespace(
                    stdout=(
                        f'  ATTRS{{serial}}=="{serial}"\n'
                        f'  ATTR{{dev_id}}=="0x{dev_id:x}"\n'
                    )
                )

            commands: list[list[str]] = []

            def run_root(command: list[str], **_kwargs: object) -> SimpleNamespace:
                commands.append(command)
                if len(command) > 4 and command[4] == "name":
                    old_name, new_name = command[3], command[5]
                    if new_name in current:
                        raise RuntimeError(f"interface {new_name} already exists")
                    current[new_name] = current.pop(old_name)
                    (net_dir / old_name).rename(net_dir / new_name)
                return SimpleNamespace(stdout="")

            with (
                patch.object(setup, "Path", side_effect=make_path),
                patch.object(setup.subprocess, "run", side_effect=udev_info),
                patch.object(setup, "run_root", side_effect=run_root),
                redirect_stdout(io.StringIO()),
            ):
                setup._rename_interfaces(None, wheels, chest)
            return current, commands

    def test_stages_a_wheel_chest_swap_before_assigning_final_names(self) -> None:
        current, commands = self.run_rename(
            {
                setup._CAN_B: ("chest", 0),
                setup._CAN_C: ("wheel", 0),
            },
            wheels="wheel",
            chest="chest",
        )

        self.assertEqual(current[setup._CAN_B], ("wheel", 0))
        self.assertEqual(current[setup._CAN_C], ("chest", 0))
        renames = [command for command in commands if command[4] == "name"]
        self.assertTrue(
            all(command[5].startswith("can_ax_tmp") for command in renames[:2])
        )
        self.assertEqual(
            {command[5] for command in renames[2:]}, {setup._CAN_B, setup._CAN_C}
        )

    def test_moves_a_stale_target_occupant_out_of_the_way(self) -> None:
        current, _ = self.run_rename(
            {
                setup._CAN_B: ("old-wheel", 0),
                "can0": ("new-wheel", 0),
            },
            wheels="new-wheel",
            chest=None,
        )

        self.assertEqual(current[setup._CAN_B], ("new-wheel", 0))
        self.assertIn(("old-wheel", 0), current.values())

    def test_evicts_a_stale_occupant_when_replacement_is_absent(self) -> None:
        current, _ = self.run_rename(
            {setup._CAN_B: ("old-wheel", 0)},
            wheels="new-wheel",
            chest=None,
        )

        self.assertNotIn(setup._CAN_B, current)
        self.assertIn(("old-wheel", 0), current.values())

    def test_evicts_a_stale_occupant_when_role_is_removed(self) -> None:
        current, _ = self.run_rename(
            {
                setup._CAN_B: ("old-wheel", 0),
                setup._CAN_C: ("chest", 0),
            },
            wheels=None,
            chest="chest",
        )

        self.assertNotIn(setup._CAN_B, current)
        self.assertEqual(current[setup._CAN_C], ("chest", 0))

    def test_rejects_one_adapter_assigned_to_two_roles(self) -> None:
        with (
            self.assertRaisesRegex(RuntimeError, "arm hub and wheel bus"),
            redirect_stdout(io.StringIO()),
        ):
            setup._rename_interfaces("same", "same", None)


if __name__ == "__main__":
    unittest.main()
