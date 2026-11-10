from __future__ import annotations

from collections.abc import Callable

import pytest

from almond_axol.vr.models import VRFrame, VRPose, VRPosition, VRQuaternion
from tests.safety_isolation import NoSafetyBoard


@pytest.fixture(autouse=True)
def _isolated_libsurvive_config(tmp_path, monkeypatch) -> None:
    """Keep survive-cli launches in tests from creating ~/.config/libsurvive."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))


@pytest.fixture(autouse=True)
def _no_live_safety_link(monkeypatch) -> None:
    """Keep apps built by tests off the real lift bus.

    The server's safety link connects on its own whenever the lift bus is up,
    so on a robot (or with a virtual bus named like one) every ``create_app``
    would open a real proxy and poll the board. Tests that exercise the link
    patch ``app.SafetyLink`` themselves, which overrides this.
    """
    from almond_axol.serve import app as app_module

    monkeypatch.setattr(app_module, "SafetyLink", NoSafetyBoard)


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
