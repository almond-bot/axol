from __future__ import annotations

from collections.abc import Callable

import pytest

from almond_axol.vr import ice
from almond_axol.vr.interp import PoseInterpolator
from almond_axol.vr.models import VRFrame, VRState


def test_vr_frame_defaults_and_validation(
    frame_factory: Callable[..., VRFrame],
) -> None:
    frame = frame_factory()

    assert frame.state is VRState.TELEOP
    assert frame.l_grip == frame.r_grip == 1.0
    assert frame.l_tracked and frame.r_tracked
    assert frame.model_dump(mode="json")["state"] == "teleop"

    with pytest.raises(ValueError):
        frame_factory(state="not-a-state")


def test_pose_interpolator_latest_wins(frame_factory: Callable[..., VRFrame]) -> None:
    interpolator = PoseInterpolator(enabled=False)
    first = frame_factory(seq=1)
    latest = frame_factory(seq=2, l_lock=True)

    interpolator.push(first, now=1.0)
    interpolator.push(latest, now=2.0)

    assert interpolator.sample(now=2.0) is latest
    interpolator.reset()
    assert interpolator.sample(now=3.0) is None


def test_pose_interpolator_blends_motion_but_keeps_latest_controls(
    frame_factory: Callable[..., VRFrame],
) -> None:
    first = frame_factory(t=0.0, seq=1)
    second = frame_factory(t=100.0, seq=2, l_lock=True, reset=True)
    second.l_ee.position.x = 1.0
    interpolator = PoseInterpolator(
        min_delay_s=0.05,
        max_delay_s=0.05,
        smooth_window_s=0.0,
    )

    interpolator.push(first, now=10.0)
    interpolator.push(second, now=10.1)
    result = interpolator.sample(now=10.1)

    assert result is not None
    assert result.l_ee.position.x == pytest.approx(0.5, abs=0.02)
    assert result.l_lock is True
    assert result.reset is True


def test_ice_configuration_and_candidate_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AXOL_TURN_URL", "turn:a.example, turns:b.example")
    monkeypatch.setenv("AXOL_TURN_USERNAME", "user")
    monkeypatch.setenv("AXOL_TURN_PASSWORD", "secret")

    browser = ice.client_ice_servers()
    server = ice.ice_servers()
    assert browser == [
        {
            "urls": ["turn:a.example", "turns:b.example"],
            "username": "user",
            "credential": "secret",
        }
    ]
    assert server[0].urls == ["turn:a.example", "turns:b.example"]

    sdp = "\r\n".join(
        [
            "v=0",
            "a=candidate:1 1 udp 1 10.0.0.1 1000 typ host",
            "a=candidate:2 1 udp 1 1.2.3.4 2000 typ relay",
        ]
    )
    assert ice.summarize_candidates(sdp) == "candidates: host=1 relay=1"
    assert ice.summarize_candidates("v=0") == "candidates: none"


def test_candidates_are_replicated_to_each_media_section() -> None:
    candidate = "a=candidate:1 1 udp 1 10.0.0.1 1000 typ host"
    sdp = f"v=0\nm=audio 9 UDP/TLS/RTP/SAVPF 111\n{candidate}\nm=video 9 UDP/TLS/RTP/SAVPF 96\n"

    replicated = ice.replicate_candidates_across_mlines(sdp)

    assert replicated.count(candidate) == 2
    assert ice.replicate_candidates_across_mlines("v=0\nm=audio 9 RTP/AVP 0\n") == (
        "v=0\nm=audio 9 RTP/AVP 0\n"
    )
