"""Keep apps built by tests off a real safety board, under any test runner.

``serve.app.create_app`` opens a ``SafetyLink`` whose supervisor thread polls
for the lift bus every second and is only stopped by the app's shutdown hook,
which most tests never run. On a robot (or with a virtual bus named like one)
it would open a real proxy and poll the board, and in a long ``unittest
discover`` run the leaked threads spin as soon as a later test patches
``asyncio.sleep``.

``conftest.py`` swaps in :class:`NoSafetyBoard` for every pytest test. pytest
fixtures don't exist under ``python -m unittest``, so each module that builds
an app also imports :func:`setUpModule` / :func:`tearDownModule` from here.
Both runners call them. Tests that exercise the link patch ``app.SafetyLink``
themselves, which overrides this.
"""

from __future__ import annotations

from unittest import mock


class NoSafetyBoard:
    """Inert stand-in for ``serve.safety_link.SafetyLink``: a Jelly without one."""

    def status(self) -> dict[str, object]:
        return {
            "channel": "",
            "connected": False,
            "paused": False,
            "detected": False,
            "reachable": False,
            "ageSeconds": None,
            "status": None,
            "error": None,
        }

    def stop(self) -> dict[str, object]:
        from almond_axol.robot.safety import SafetyError

        raise SafetyError("This Jelly has no safety board.")

    start = stop

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


_patch: mock._patch | None = None  # type: ignore[type-arg]


def setUpModule() -> None:  # noqa: N802 - unittest's hook name
    global _patch
    from almond_axol.serve import app as app_module

    _patch = mock.patch.object(app_module, "SafetyLink", NoSafetyBoard)
    _patch.start()


def tearDownModule() -> None:  # noqa: N802 - unittest's hook name
    global _patch
    if _patch is not None:
        _patch.stop()
        _patch = None
