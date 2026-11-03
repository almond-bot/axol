"""The task picker of a ``collect-data`` session.

The operator chooses each episode's task from the headset HUD or the control
panel. It is published as the live setting ``task`` (see
:mod:`almond_axol.teleop.live`): a ``select`` whose options are the session's
tasks and whose value is the task the next take records.

- The choices are ``--task``, then ``--tasks``, then the tasks already in a
  resumed dataset, without duplicates.
- Picking a listed task selects it. Any other non-empty text (the panel's
  **Add task**) is appended to the choices and selected, so the headset can
  pick it too.
- The task is read when a take starts (:meth:`TaskPicker.begin_take`) and
  can't change until the take ends (:meth:`TaskPicker.end_take`): every row
  of an episode carries the same task. A saved episode's task can still be
  renamed on the panel's Datasets page.

Thread-safe: ``apply`` runs on the VR server's event-loop thread, the take
hooks on the collection loop.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Any

from ..teleop.live import LiveSettingDef
from .dataset_browser import MAX_TASK_LENGTH

TASK_SETTING_KEY = "task"


def _clean(task: Any) -> str:
    if not isinstance(task, str):
        raise ValueError(f"task must be text, not {task!r}")
    text = task.strip()
    if not text:
        raise ValueError("task is empty")
    if len(text) > MAX_TASK_LENGTH:
        raise ValueError(f"task is longer than {MAX_TASK_LENGTH} characters")
    return text


class TaskPicker:
    """The task each take records, chosen live (a :class:`SessionSetting`).

    Args:
        initial: The task selected at startup (``--task``).
        choices: Further tasks to offer (``--tasks``, then a resumed
            dataset's tasks). Blank entries and duplicates are dropped.
    """

    def __init__(self, initial: str, choices: Iterable[str] = ()) -> None:
        self._lock = threading.Lock()
        self._current = _clean(initial)
        self._choices: list[str] = [self._current]
        for task in choices:
            try:
                text = _clean(task)
            except ValueError:
                continue
            if text not in self._choices:
                self._choices.append(text)
        self._in_take = False

    @property
    def current(self) -> str:
        with self._lock:
            return self._current

    @property
    def choices(self) -> list[str]:
        with self._lock:
            return list(self._choices)

    # -- SessionSetting -----------------------------------------------------

    def definition(self) -> LiveSettingDef:
        with self._lock:
            options = tuple(self._choices)
        return LiveSettingDef(
            key=TASK_SETTING_KEY,
            label="Task",
            type="select",
            options=options,
            help="The task the next episode records.",
        )

    def value(self) -> str:
        return self.current

    def apply(self, value: Any) -> None:
        text = _clean(value)
        with self._lock:
            if self._in_take and text != self._current:
                raise ValueError("the task can't change during an episode")
            if text not in self._choices:
                self._choices.append(text)
            self._current = text

    # -- Collection loop ----------------------------------------------------

    def begin_take(self) -> str:
        """Lock the selection for a take and return its task."""
        with self._lock:
            self._in_take = True
            return self._current

    def end_take(self) -> None:
        """Unlock the selection once the take's rows are frozen."""
        with self._lock:
            self._in_take = False


__all__ = ["TASK_SETTING_KEY", "TaskPicker"]
