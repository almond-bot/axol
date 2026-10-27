"""Startup progress an operation reports to the control panel.

An operation says which steps it takes to get ready (:func:`plan`), marks each
one as it begins (:func:`step`) and says when it is ready for the operator
(:func:`ready`). Under ``axol serve`` the operation runner starts a tracker for
each run (:func:`begin`) and the session serves its :meth:`Progress.to_dict`,
which the panel turns into a label and a percentage. From the CLI no tracker
is running and every call is a no-op.

Step keys are stable identifiers ("arms", "ik", "cameras", ...); the panel
owns their labels. The plan is a promise about order only: a step reported
without being planned is added after the current one, so a run whose plan
missed a branch still shows what it is doing.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from typing import Any


class Progress:
    """One run's startup: the planned steps, the one under way, and readiness."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._steps: list[str] = []
        self._current = -1
        self._step_started_at: float | None = None
        self._ready = False

    def plan(self, steps: Iterable[str]) -> None:
        with self._lock:
            done = self._steps[: self._current + 1]
            planned = [s for s in dict.fromkeys(steps) if s not in done]
            self._steps = done + planned

    def step(self, key: str) -> None:
        with self._lock:
            if self._ready:
                return
            if key in self._steps:
                index = self._steps.index(key)
                if index < self._current:
                    return
            else:
                index = self._current + 1
                self._steps.insert(index, key)
            self._current = index
            self._step_started_at = time.time()

    def ready(self) -> None:
        with self._lock:
            self._ready = True
            self._step_started_at = None

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._ready

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "steps": list(self._steps),
                "current": self._current,
                "stepStartedAt": self._step_started_at,
                "ready": self._ready,
            }


_active: Progress | None = None
_active_lock = threading.Lock()


def begin() -> Progress:
    """Start tracking a new run (the operation runner calls this)."""
    global _active
    progress = Progress()
    with _active_lock:
        _active = progress
    return progress


def end(progress: Progress) -> None:
    """Stop routing reports to ``progress`` once its run is over."""
    global _active
    with _active_lock:
        if _active is progress:
            _active = None


def _current() -> Progress | None:
    with _active_lock:
        return _active


def plan(steps: Iterable[str]) -> None:
    """Declare the steps this run takes before it is ready, in order."""
    progress = _current()
    if progress is not None:
        progress.plan(steps)


def step(key: str) -> None:
    """Report that the run has begun startup step ``key``."""
    progress = _current()
    if progress is not None:
        progress.step(key)


def ready() -> None:
    """Report that startup is over and the operation is ready to use."""
    progress = _current()
    if progress is not None:
        progress.ready()
