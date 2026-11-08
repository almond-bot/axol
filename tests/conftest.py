from __future__ import annotations

from collections.abc import Callable

import pytest

from almond_axol.vr.models import VRFrame, VRPose, VRPosition, VRQuaternion


@pytest.fixture(autouse=True)
def _isolated_libsurvive_config(tmp_path, monkeypatch) -> None:
    """Keep survive-cli launches in tests from creating ~/.config/libsurvive."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))


class _NoSafetyBoard:
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


@pytest.fixture(autouse=True)
def _no_live_safety_link(monkeypatch) -> None:
    """Keep apps built by tests off the real lift bus.

    The server's safety link connects on its own whenever the lift bus is up,
    so on a robot (or with a virtual bus named like one) every ``create_app``
    would open a real proxy and poll the board. Tests that exercise the link
    patch ``app.SafetyLink`` themselves, which overrides this.
    """
    from almond_axol.serve import app as app_module

    monkeypatch.setattr(app_module, "SafetyLink", _NoSafetyBoard)


@pytest.fixture
def frame_factory() -> Callable[..., VRFrame]:
    def make(**overrides: object) -> VRFrame:
        pose = VRPose(
            position=VRPosition(x=0.0, y=0.0, z=0.0),
            quaternion=VRQuaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        values: dict[str, object] = {
            "l_ee": pose,
            "r_ee": pose.model_copy(deep=True),
            "l_elbow": VRPosition(x=0.0, y=0.0, z=0.0),
            "r_elbow": VRPosition(x=0.0, y=0.0, z=0.0),
        }
        values.update(overrides)
        return VRFrame(**values)  # type: ignore[arg-type]

    return make
