"""collect-data's per-episode task picker and its live-setting wiring."""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from almond_axol.recording.dataset_browser import MAX_TASK_LENGTH, read_task_names
from almond_axol.recording.task_picker import TASK_SETTING_KEY, TaskPicker
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore
from almond_axol.teleop.live import LiveSettingDef, LiveSettings


def _live(published: list | None = None) -> LiveSettings:
    core = VRTeleopCore(
        VRTeleopConfig(),
        logging.getLogger("test"),
        broadcast_tracking=lambda _enabled: None,
    )
    sink = published.append if published is not None else (lambda _s: None)
    return LiveSettings(core, object(), sink)


class TaskPickerTest(unittest.TestCase):
    def test_choices_start_with_task_and_drop_blanks_and_duplicates(self) -> None:
        picker = TaskPicker(
            "  Pick up the cup ",
            ["Open the drawer", "", "Pick up the cup", "  ", "Open the drawer", 3],
        )
        self.assertEqual(picker.current, "Pick up the cup")
        self.assertEqual(picker.choices, ["Pick up the cup", "Open the drawer"])

    def test_definition_lists_the_choices(self) -> None:
        picker = TaskPicker("A", ["B"])
        d = picker.definition()
        self.assertEqual(d.key, TASK_SETTING_KEY)
        self.assertEqual(d.type, "select")
        self.assertEqual(d.options, ("A", "B"))

    def test_apply_selects_a_listed_task(self) -> None:
        picker = TaskPicker("A", ["B"])
        picker.apply("B")
        self.assertEqual(picker.current, "B")
        self.assertEqual(picker.choices, ["A", "B"])

    def test_apply_adds_a_new_task(self) -> None:
        picker = TaskPicker("A")
        picker.apply("  Wipe the table  ")
        self.assertEqual(picker.current, "Wipe the table")
        self.assertEqual(picker.choices, ["A", "Wipe the table"])

    def test_apply_refuses_empty_long_or_non_text(self) -> None:
        picker = TaskPicker("A")
        for bad in ("", "   ", None, 5, "x" * (MAX_TASK_LENGTH + 1)):
            with self.assertRaises(ValueError, msg=repr(bad)):
                picker.apply(bad)
        self.assertEqual(picker.choices, ["A"])

    def test_empty_initial_task_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            TaskPicker("  ")

    def test_task_is_locked_during_a_take(self) -> None:
        picker = TaskPicker("A", ["B"])
        self.assertEqual(picker.begin_take(), "A")
        with self.assertRaises(ValueError):
            picker.apply("B")
        with self.assertRaises(ValueError):
            picker.apply("C")
        # Re-sending the current task is harmless.
        picker.apply("A")
        self.assertEqual(picker.choices, ["A", "B"])
        picker.end_take()
        picker.apply("B")
        self.assertEqual(picker.begin_take(), "B")


class LiveSettingsSessionSettingTest(unittest.TestCase):
    def test_session_setting_is_published_after_the_built_in_ones(self) -> None:
        published: list = []
        live = _live(published)
        live.add_session_setting(TaskPicker("A", ["B"]))
        snapshot = published[-1]
        keys = [d["key"] for d in snapshot["schema"]]
        self.assertEqual(keys[-1], TASK_SETTING_KEY)
        task_def = snapshot["schema"][-1]
        self.assertEqual(task_def["options"], ["A", "B"])
        self.assertNotIn("source", task_def)
        self.assertEqual(snapshot["values"][TASK_SETTING_KEY], "A")

    def test_apply_routes_to_the_session_setting_and_announces(self) -> None:
        published: list = []
        live = _live(published)
        live.add_session_setting(TaskPicker("A"))
        live.apply(TASK_SETTING_KEY, "New task")
        snapshot = published[-1]
        self.assertEqual(snapshot["values"][TASK_SETTING_KEY], "New task")
        self.assertEqual(snapshot["schema"][-1]["options"], ["A", "New task"])

    def test_refused_value_still_announces_the_current_one(self) -> None:
        published: list = []
        live = _live(published)
        picker = TaskPicker("A", ["B"])
        live.add_session_setting(picker)
        picker.begin_take()
        before = len(published)
        with self.assertRaises(ValueError):
            live.apply(TASK_SETTING_KEY, "B")
        self.assertEqual(len(published), before + 1)
        self.assertEqual(published[-1]["values"][TASK_SETTING_KEY], "A")

    def test_key_clash_is_refused(self) -> None:
        live = _live()
        live.add_session_setting(TaskPicker("A"))
        with self.assertRaises(ValueError):
            live.add_session_setting(TaskPicker("B"))

        class _Clash:
            def definition(self) -> LiveSettingDef:
                return LiveSettingDef(key="haptics", label="x", type="boolean", help="")

            def value(self) -> bool:
                return True

            def apply(self, value: object) -> None:
                pass

        with self.assertRaises(ValueError):
            live.add_session_setting(_Clash())


class ReadTaskNamesTest(unittest.TestCase):
    def test_missing_tasks_file_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(read_task_names(Path(tmp)), [])

    def test_tasks_come_back_in_index_order(self) -> None:
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta").mkdir()
            tasks = pd.DataFrame(
                {"task_index": [1, 0]},
                index=pd.Index(["Second", "First"], name="task"),
            )
            tasks.to_parquet(root / "meta" / "tasks.parquet")
            self.assertEqual(read_task_names(root), ["First", "Second"])


if __name__ == "__main__":
    unittest.main()
