from __future__ import annotations

import asyncio
import json
import math
import os
import unittest
from unittest import mock

from pydantic import ValidationError
from starlette.testclient import TestClient

from almond_axol.utils.browser_origin import (
    browser_origin_allowed,
    configure_self_hosted_browser_origins,
)
from almond_axol.vr.config import VRServerConfig
from almond_axol.vr.interp import PoseInterpolator
from almond_axol.vr.models import VRFrame, VRPose, VRPosition, VRQuaternion
from almond_axol.vr.server import VRServer, get_last_quest_pose_datum


def _frame(
    seq: int,
    *,
    x: float = 0.0,
    angle: float = 0.0,
    grip: float = 1.0,
    left_tracked: bool = True,
    source_id: str | None = None,
    source_kind: str | None = None,
    pose_profile: str | None = None,
    pose_space: str | None = None,
    l_stick_x: float = 0.0,
    l_stick_y: float = 0.0,
    r_stick_x: float = 0.0,
    l_stick_click: bool = False,
    r_stick_click: bool = False,
) -> VRFrame:
    quat = VRQuaternion(
        x=0.0,
        y=math.sin(angle / 2.0),
        z=0.0,
        w=math.cos(angle / 2.0),
    )
    left = VRPose(position=VRPosition(x=x, y=1.0, z=-0.4), quaternion=quat)
    right = VRPose(
        position=VRPosition(x=-0.2, y=1.0, z=-0.4),
        quaternion=VRQuaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    return VRFrame(
        l_ee=left,
        r_ee=right,
        l_elbow=left.position,
        r_elbow=right.position,
        l_grip=grip,
        r_grip=1.0,
        l_tracked=left_tracked,
        r_tracked=True,
        t=10.0 * seq,
        seq=seq,
        pose_source_id=source_id,
        pose_source_kind=source_kind,
        l_pose_profile=pose_profile,
        r_pose_profile=pose_profile,
        l_pose_space=pose_space,
        r_pose_space=pose_space,
        l_stick_x=l_stick_x,
        l_stick_y=l_stick_y,
        r_stick_x=r_stick_x,
        l_stick_click=l_stick_click,
        r_stick_click=r_stick_click,
    )


class PoseInterpolatorSafetyTest(unittest.TestCase):
    def test_latest_cart_controls_survive_interpolation_and_identity_dedup(
        self,
    ) -> None:
        interp = PoseInterpolator(
            min_delay_s=0.0,
            max_delay_s=0.0,
            smooth_window_s=0.04,
            outlier_k=0.0,
        )
        for seq in range(4):
            interp.push(_frame(seq), now=1.0 + 0.01 * seq)
        neutral = interp.sample(now=1.03)
        assert neutral is not None

        interp.push(
            _frame(
                4,
                l_stick_x=0.25,
                l_stick_y=-0.75,
                r_stick_x=0.5,
                l_stick_click=True,
                r_stick_click=True,
            ),
            now=1.04,
        )
        commanded = interp.sample(now=1.04)
        assert commanded is not None
        self.assertIsNot(commanded, neutral)
        self.assertEqual(commanded.l_stick_x, 0.25)
        self.assertEqual(commanded.l_stick_y, -0.75)
        self.assertEqual(commanded.r_stick_x, 0.5)
        self.assertTrue(commanded.l_stick_click)
        self.assertTrue(commanded.r_stick_click)

        # A release with the same smoothed pose must also publish a new frame;
        # otherwise identity reuse would keep the cart moving/lift actuating.
        interp.push(_frame(5), now=1.05)
        released = interp.sample(now=1.05)
        assert released is not None
        self.assertIsNot(released, commanded)
        self.assertEqual(released.l_stick_x, 0.0)
        self.assertEqual(released.l_stick_y, 0.0)
        self.assertEqual(released.r_stick_x, 0.0)
        self.assertFalse(released.l_stick_click)
        self.assertFalse(released.r_stick_click)

    def test_fixed_position_rotation_and_grip_are_not_deduplicated(self) -> None:
        interp = PoseInterpolator(
            min_delay_s=0.0,
            max_delay_s=0.0,
            smooth_window_s=0.0,
        )
        interp.push(_frame(0), now=1.00)
        interp.push(_frame(1), now=1.01)
        initial = interp.sample(now=1.01)
        self.assertIsNotNone(initial)

        interp.push(_frame(2, angle=math.pi / 2), now=1.02)
        rotated = interp.sample(now=1.02)
        self.assertIsNot(rotated, initial)
        assert rotated is not None
        self.assertAlmostEqual(rotated.l_ee.quaternion.y, math.sqrt(0.5), places=5)

        interp.push(_frame(3, angle=math.pi / 2, grip=0.0), now=1.03)
        squeezed = interp.sample(now=1.03)
        self.assertIsNot(squeezed, rotated)
        assert squeezed is not None
        self.assertAlmostEqual(squeezed.l_grip, 0.0)

        interp.push(_frame(4, angle=math.pi / 2, grip=0.0), now=1.04)
        same_pose = interp.sample(now=1.04)
        self.assertIs(same_pose, squeezed)
        assert same_pose is not None
        live_stamp = same_pose.t_host
        self.assertIsNotNone(live_stamp)
        self.assertGreater(live_stamp, 1.03)

        # Sampling without a newly captured frame must not fabricate a fresh
        # heartbeat and thereby hide a stopped transport from collection QA.
        self.assertIs(interp.sample(now=1.20), same_pose)
        self.assertEqual(same_pose.t_host, live_stamp)

    def test_short_tracking_loss_is_emitted_before_recovery(self) -> None:
        interp = PoseInterpolator(
            min_delay_s=0.0,
            max_delay_s=0.0,
            smooth_window_s=0.04,
            outlier_k=0.0,
        )
        for seq in range(5):
            interp.push(_frame(seq), now=1.0 + 0.01 * seq)
        baseline = interp.sample(now=1.04)
        assert baseline is not None
        self.assertTrue(baseline.l_tracked)

        # A complete false→true burst arrives before the IK thread samples.
        # The false state must still be observable and must hold the last
        # trusted pose rather than smoothing the relocalization jump through.
        interp.push(_frame(5, x=0.2, left_tracked=False), now=1.05)
        interp.push(_frame(6, x=0.3, left_tracked=False), now=1.06)
        interp.push(_frame(7, x=0.5), now=1.07)
        interp.push(_frame(8, x=0.5), now=1.08)

        lost = interp.sample(now=1.08)
        assert lost is not None
        self.assertFalse(lost.l_tracked)
        self.assertAlmostEqual(lost.l_ee.position.x, baseline.l_ee.position.x)

        recovered = interp.sample(now=1.081)
        assert recovered is not None
        self.assertTrue(recovered.l_tracked)


class VRFrameValidationTest(unittest.TestCase):
    def test_quaternion_is_normalized_at_network_model_boundary(self) -> None:
        frame = _frame(1).model_dump()
        frame["l_ee"]["quaternion"] = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 2.0}
        parsed = VRFrame.model_validate(frame)
        self.assertEqual(parsed.l_ee.quaternion.w, 1.0)

    def test_nonfinite_motion_fields_and_zero_quaternion_are_rejected(self) -> None:
        for path, value in (
            (("l_ee", "position", "x"), float("nan")),
            (("r_elbow", "z"), float("inf")),
            (("l_grip",), float("-inf")),
            (("t",), float("nan")),
        ):
            with self.subTest(path=path):
                frame = _frame(1).model_dump()
                target = frame
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(ValidationError):
                    VRFrame.model_validate(frame)

        frame = _frame(1).model_dump()
        frame["r_ee"]["quaternion"] = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 0.0}
        with self.assertRaises(ValidationError):
            VRFrame.model_validate(frame)

    def test_browser_origin_policy_allows_expected_clients_only(self) -> None:
        self.assertTrue(
            browser_origin_allowed(
                "https://axol.almond.bot", scheme="wss", host="robot.local:8000"
            )
        )
        self.assertFalse(
            browser_origin_allowed(
                "https://robot.local:8000",
                scheme="wss",
                host="robot.local:8000",
            )
        )
        self.assertFalse(
            browser_origin_allowed(
                "https://robot.local:8001",
                scheme="wss",
                host="robot.local:8000",
            )
        )
        self.assertFalse(
            browser_origin_allowed(
                "https://robot.local:8443",
                scheme="wss",
                host="robot.local:8000",
            )
        )
        self.assertTrue(
            browser_origin_allowed(
                "http://localhost:5173", scheme="https", host="robot.local:8000"
            )
        )
        self.assertTrue(
            browser_origin_allowed(None, scheme="wss", host="robot.local:8000")
        )
        self.assertFalse(
            browser_origin_allowed(
                "https://attacker.example", scheme="wss", host="robot.local:8000"
            )
        )
        with mock.patch.dict(
            os.environ,
            {"AXOL_ALLOWED_BROWSER_ORIGINS": "https://preview.example"},
        ):
            self.assertTrue(
                browser_origin_allowed(
                    "https://preview.example",
                    scheme="wss",
                    host="robot.local:8000",
                )
            )

    def test_browser_origin_policy_uses_actual_self_hosted_ui_scope(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            configure_self_hosted_browser_origins(
                scheme="http", port=9000, hosts={"robot.local", "192.0.2.10"}
            )
            self.assertTrue(
                browser_origin_allowed(
                    "http://robot.local:9000",
                    scheme="wss",
                    host="robot.local:8000",
                )
            )
            self.assertTrue(
                browser_origin_allowed(
                    "http://192.0.2.10:9000",
                    scheme="wss",
                    host="192.0.2.10:8000",
                )
            )
            self.assertFalse(
                browser_origin_allowed(
                    "https://robot.local:9000",
                    scheme="wss",
                    host="robot.local:8000",
                )
            )
            self.assertFalse(
                browser_origin_allowed(
                    "http://robot.local:9001",
                    scheme="wss",
                    host="robot.local:8000",
                )
            )

    def test_request_host_cannot_grant_dns_rebinding_origin(self) -> None:
        self.assertFalse(
            browser_origin_allowed(
                "http://attacker.example:8001",
                scheme="http",
                host="attacker.example:8001",
            )
        )


class VRServerPoseModeTest(unittest.TestCase):
    def test_vr_teleop_maps_absolute_config_to_server_pose_mode(self) -> None:
        from almond_axol.teleop.config import VRTeleopConfig
        from almond_axol.teleop.teleop import VRTeleop

        for absolute, expected in ((False, "relative"), (True, "absolute")):
            with self.subTest(absolute=absolute):
                server = mock.MagicMock()
                with mock.patch(
                    "almond_axol.teleop.teleop.VRServer", return_value=server
                ):
                    VRTeleop(
                        mock.MagicMock(),
                        config=VRTeleopConfig(absolute_mode=absolute),
                    )
                server.set_mode.assert_called_once_with("teleop")
                server.set_pose_mode.assert_called_once_with(expected)

    def test_relative_is_safe_default_and_invalid_modes_are_rejected(self) -> None:
        server = VRServer()
        self.assertEqual(server._pose_mode, "relative")
        with self.assertRaisesRegex(ValueError, "relative.*absolute"):
            server.set_pose_mode("mantis")

    def test_session_config_is_announced_and_replayed(self) -> None:
        server = VRServer()
        server.set_mode("teleop")
        server.set_pose_mode("absolute")
        server._tracking = True
        server.set_episode(7)
        server._hud = {"state": "recording"}
        with TestClient(server._build_app()).websocket_connect(
            "/ws", headers={"origin": "https://axol.almond.bot"}
        ) as websocket:
            self.assertEqual(
                websocket.receive_json(),
                {"type": "mode", "value": "teleop"},
            )
            self.assertEqual(
                websocket.receive_json(),
                {"type": "pose_source_kind", "value": None},
            )
            self.assertEqual(
                websocket.receive_json(),
                {"type": "pose_mode", "value": "absolute"},
            )
            self.assertEqual(
                websocket.receive_json(),
                {"type": "tracking", "value": True},
            )
            self.assertEqual(
                websocket.receive_json(),
                {"type": "episode", "value": 7},
            )
            self.assertEqual(
                websocket.receive_json(),
                {"type": "hud", "value": {"state": "recording"}},
            )
            websocket.send_json({"type": "session-config-request"})
            self.assertEqual(
                websocket.receive_json(),
                {"type": "mode", "value": "teleop"},
            )
            self.assertEqual(
                websocket.receive_json(),
                {"type": "pose_source_kind", "value": None},
            )
            self.assertEqual(
                websocket.receive_json(),
                {"type": "pose_mode", "value": "absolute"},
            )
            self.assertEqual(
                websocket.receive_json(),
                {"type": "tracking", "value": True},
            )
            self.assertEqual(
                websocket.receive_json(),
                {"type": "episode", "value": 7},
            )
            self.assertEqual(
                websocket.receive_json(),
                {"type": "hud", "value": {"state": "recording"}},
            )


class PoseSourceArbitrationTest(unittest.TestCase):
    def test_pose_source_id_cannot_inherit_a_different_source_kind(self) -> None:
        server = VRServer(VRServerConfig(pose_source_kind="tracker"))
        tracker = _frame(1, source_id="owned-source", source_kind="tracker")
        impersonating_quest = _frame(
            2,
            source_id="owned-source",
            source_kind="webxr",
        )

        self.assertTrue(server._ingest_frame_obj(tracker, "bridge", 20))
        self.assertFalse(server._ingest_frame_obj(impersonating_quest, "network", 21))
        self.assertNotIn("owned-source", server._client_sources.get(21, set()))
        self.assertEqual(server.get_frame().pose_source_kind, "tracker")  # type: ignore[union-attr]

    def test_tracker_pose_policy_is_announced_to_viewers(self) -> None:
        server = VRServer(VRServerConfig(pose_source_kind="tracker"))
        with TestClient(server._build_app()).websocket_connect(
            "/ws", headers={"origin": "https://axol.almond.bot"}
        ) as websocket:
            self.assertEqual(
                websocket.receive_json(),
                {"type": "pose_source_kind", "value": "tracker"},
            )

    def test_only_active_webxr_owner_can_publish_hud_controls(self) -> None:
        server = VRServer(VRServerConfig(pose_source_kind="webxr"))
        owner_id = 10

        self.assertTrue(
            server._ingest_frame_obj(
                _frame(1, source_id="quest", source_kind="webxr"),
                "network",
                owner_id,
            )
        )
        asyncio.run(
            server._handle_signaling(
                mock.AsyncMock(),
                owner_id,
                {
                    "type": "hud",
                    "pose_source_id": "quest",
                    "pose_source_kind": "webxr",
                    "value": {"confirm": "save"},
                },
            )
        )
        self.assertEqual(server._hud, {"confirm": "save"})

        asyncio.run(
            server._handle_signaling(
                mock.AsyncMock(),
                11,
                {
                    "type": "hud",
                    "pose_source_id": "viewer",
                    "pose_source_kind": "webxr",
                    "value": {"confirm": "discard"},
                },
            )
        )
        self.assertEqual(server._hud, {"confirm": "save"})

    def test_delayed_old_socket_hud_cannot_overwrite_reconnect_replay(self) -> None:
        async def scenario() -> None:
            server = VRServer(VRServerConfig(pose_source_kind="webxr"))
            old_client = 10
            new_client = 11
            source_id = "quest"

            self.assertTrue(
                server._ingest_frame_obj(
                    _frame(1, source_id=source_id, source_kind="webxr"),
                    "old-socket",
                    old_client,
                )
            )
            old_hud = {
                "type": "hud",
                "pose_source_id": source_id,
                "pose_source_kind": "webxr",
                "value": {"confirm": "discard"},
            }
            await server._handle_signaling(mock.AsyncMock(), old_client, old_hud)

            # The replacement socket carries a fresh pose before replaying its
            # HUD snapshot, exactly as AxolVRClient does on reconnect.
            self.assertTrue(
                server._ingest_frame_obj(
                    _frame(2, source_id=source_id, source_kind="webxr"),
                    "new-socket",
                    new_client,
                )
            )
            new_hud = {
                "type": "hud",
                "pose_source_id": source_id,
                "pose_source_kind": "webxr",
                "value": {"confirm": "save"},
            }
            await server._handle_signaling(mock.AsyncMock(), new_client, new_hud)

            # TCP preserves order only within one socket.  A HUD packet that
            # was already in flight on the old socket can arrive after the new
            # socket's replay; it must neither replace the replay nor reclaim
            # publisher identity (which would clear the good HUD on teardown).
            await server._handle_signaling(mock.AsyncMock(), old_client, old_hud)
            self.assertEqual(server._hud, {"confirm": "save"})
            self.assertEqual(server._hud_client, new_client)

            server._drop_pose_client(old_client)
            self.assertEqual(server._hud, {"confirm": "save"})
            self.assertEqual(server._pose_owner, source_id)

        asyncio.run(scenario())

    def test_tracker_takeover_clears_the_previous_quest_hud(self) -> None:
        async def scenario() -> None:
            server = VRServer()
            mirror = mock.AsyncMock()
            server._active_clients.add(mirror)
            owner_id = 10
            self.assertTrue(
                server._ingest_frame_obj(
                    _frame(1, source_id="quest", source_kind="webxr"),
                    "network",
                    owner_id,
                )
            )
            await server._handle_signaling(
                mock.AsyncMock(),
                owner_id,
                {
                    "type": "hud",
                    "pose_source_id": "quest",
                    "pose_source_kind": "webxr",
                    "value": {"countdownRemainingMs": 3000},
                },
            )
            self.assertEqual(server._hud, {"countdownRemainingMs": 3000})

            self.assertTrue(
                server._ingest_frame_obj(
                    _frame(1, source_id="bridge", source_kind="tracker"),
                    "bridge",
                    20,
                )
            )
            await asyncio.sleep(0)

            self.assertIsNone(server._hud)
            self.assertIsNone(server._hud_client)
            messages = [
                json.loads(call.args[0]) for call in mirror.send_text.await_args_list
            ]
            self.assertEqual(messages[-1], {"type": "hud", "value": None})

        asyncio.run(scenario())

    def test_tracker_session_rejects_view_only_quest_hud_controls(self) -> None:
        server = VRServer(VRServerConfig(pose_source_kind="tracker"))
        tracker = _frame(1, source_id="bridge", source_kind="tracker")
        quest = _frame(1, source_id="quest", source_kind="webxr")

        self.assertTrue(server._ingest_frame_obj(tracker, "bridge", 20))
        self.assertFalse(server._ingest_frame_obj(quest, "network", 21))
        asyncio.run(
            server._handle_signaling(
                mock.AsyncMock(),
                21,
                {
                    "type": "hud",
                    "pose_source_id": "quest",
                    "pose_source_kind": "webxr",
                    "value": {"countdownRemainingMs": 3000},
                },
            )
        )
        self.assertIsNone(server._hud)

    def test_one_client_cannot_retain_unbounded_pose_source_ids(self) -> None:
        server = VRServer()
        client_id = 17

        for index in range(100):
            server._ingest_frame_obj(
                _frame(
                    index,
                    source_id=f"attacker-selected-source-{index}",
                    source_kind="webxr",
                ),
                "native",
                client_id,
            )

        self.assertLessEqual(len(server._client_sources[client_id]), 8)
        self.assertLessEqual(len(server._source_clients), 8)
        self.assertLessEqual(len(server._source_kind), 8)

    def test_dual_transports_deduplicate_one_logical_webxr_source(self) -> None:
        server = VRServer(VRServerConfig(pose_source_kind="webxr"))
        received: list[int | None] = []
        server.set_on_frame(lambda frame: received.append(frame.seq))

        first = _frame(1, source_id="quest-session", source_kind="webxr")
        self.assertTrue(server._ingest_frame_obj(first, "network", 10))
        self.assertTrue(server._ingest_frame_obj(first, "usb", 11))
        self.assertEqual(received, [1])

        newest = _frame(3, grip=0.0, source_id="quest-session", source_kind="webxr")
        delayed = _frame(2, source_id="quest-session", source_kind="webxr")
        self.assertTrue(server._ingest_frame_obj(newest, "usb", 11))
        self.assertTrue(server._ingest_frame_obj(delayed, "network", 10))
        self.assertEqual(received, [1, 3])
        self.assertEqual(server.get_frame().seq, 3)  # type: ignore[union-attr]

        server._drop_pose_client(10)
        self.assertIsNotNone(server.get_frame())
        server._drop_pose_client(11)
        self.assertIsNone(server.get_frame())

    def test_shared_source_survivor_recovers_after_high_sequence_tab_leaves(
        self,
    ) -> None:
        server = VRServer(VRServerConfig(pose_source_kind="webxr"))
        received: list[int | None] = []
        server.set_on_frame(lambda frame: received.append(frame.seq))
        source = "copied-session-storage-source"

        self.assertTrue(
            server._ingest_frame_obj(
                _frame(5, source_id=source, source_kind="webxr"),
                "older-tab",
                10,
            )
        )
        self.assertTrue(
            server._ingest_frame_obj(
                _frame(1_000_001, source_id=source, source_kind="webxr"),
                "duplicated-tab",
                11,
            )
        )
        # The duplicated/reloaded page has a fresh performance.now() origin.
        # Its first higher-range frame must replace, not blend with, the prior
        # tab's interpolation history.
        self.assertEqual(len(server._interp._frames), 1)
        # The older tab keeps advancing its own high-water while the copied
        # source's higher reserved block wins globally.
        self.assertTrue(
            server._ingest_frame_obj(
                _frame(6, source_id=source, source_kind="webxr"),
                "older-tab",
                10,
            )
        )
        self.assertEqual(received, [5, 1_000_001])

        server._drop_pose_client(11)
        self.assertEqual(server._last_seq[source], 6)
        self.assertIsNone(server.get_frame())
        self.assertTrue(
            server._ingest_frame_obj(
                _frame(7, source_id=source, source_kind="webxr"),
                "older-tab",
                10,
            )
        )
        self.assertEqual(received, [5, 1_000_001, 7])

    def test_shared_source_survivor_recovers_when_high_sequence_tab_stays_open(
        self,
    ) -> None:
        server = VRServer(VRServerConfig(pose_source_kind="webxr"))
        received: list[int | None] = []
        server.set_on_frame(lambda frame: received.append(frame.seq))
        source = "copied-session-storage-source"

        # Use legacy kind to keep the monotonic-clock assertions focused on
        # arbitration (WebXR additionally stamps the setup UI's live datum).
        with mock.patch("almond_axol.vr.server.time.monotonic", return_value=9.9):
            self.assertTrue(
                server._ingest_frame_obj(_frame(5, source_id=source), "old", 10)
            )
        with mock.patch("almond_axol.vr.server.time.monotonic", return_value=10.0):
            self.assertTrue(
                server._ingest_frame_obj(_frame(1_000_001, source_id=source), "new", 11)
            )
        with mock.patch("almond_axol.vr.server.time.monotonic", return_value=10.5):
            self.assertTrue(
                server._ingest_frame_obj(_frame(6, source_id=source), "old", 10)
            )
        self.assertEqual(received, [5, 1_000_001])

        # The higher tab left immersive XR but its signaling socket remains
        # connected. Once its pose stream is stale, the surviving tab must not
        # spend hours burning through the copied tab's million-counter block.
        with mock.patch("almond_axol.vr.server.time.monotonic", return_value=11.1):
            self.assertTrue(
                server._ingest_frame_obj(_frame(7, source_id=source), "old", 10)
            )
        self.assertEqual(server._last_seq[source], 7)
        self.assertEqual(received, [5, 1_000_001, 7])

        # If the formerly high tab resumes, it may win again, but it starts a
        # fresh interpolation domain rather than blending two tabs' clocks and
        # unrelated physical poses.
        with mock.patch("almond_axol.vr.server.time.monotonic", return_value=11.2):
            self.assertTrue(
                server._ingest_frame_obj(_frame(1_000_002, source_id=source), "new", 11)
            )
        self.assertEqual(received, [5, 1_000_001, 7, 1_000_002])
        self.assertEqual(len(server._interp._frames), 1)

    def test_hud_watermark_recovers_with_lower_sequence_surviving_tab(self) -> None:
        async def scenario() -> None:
            server = VRServer(VRServerConfig(pose_source_kind="webxr"))
            source = "copied-session-storage-source"
            low_client = 10
            high_client = 11

            with mock.patch("almond_axol.vr.server.time.monotonic", return_value=9.9):
                self.assertTrue(
                    server._ingest_frame_obj(
                        _frame(5, source_id=source, source_kind="webxr"),
                        "low-tab",
                        low_client,
                    )
                )
            with mock.patch("almond_axol.vr.server.time.monotonic", return_value=10.0):
                self.assertTrue(
                    server._ingest_frame_obj(
                        _frame(1_000_001, source_id=source, source_kind="webxr"),
                        "high-tab",
                        high_client,
                    )
                )
            await server._handle_signaling(
                mock.AsyncMock(),
                high_client,
                {
                    "type": "hud",
                    "pose_source_id": source,
                    "pose_source_kind": "webxr",
                    "value": {"confirm": "save"},
                },
            )

            # The high-range tab remains connected but stops posing.  Once its
            # sequence domain is stale, the lower tab resumes both pose and HUD
            # publication instead of staying below a stale HUD watermark.
            with mock.patch("almond_axol.vr.server.time.monotonic", return_value=11.1):
                self.assertTrue(
                    server._ingest_frame_obj(
                        _frame(6, source_id=source, source_kind="webxr"),
                        "low-tab",
                        low_client,
                    )
                )
            self.assertIsNone(server._hud)
            self.assertIsNone(server._hud_pose_seq)

            await server._handle_signaling(
                mock.AsyncMock(),
                low_client,
                {
                    "type": "hud",
                    "pose_source_id": source,
                    "pose_source_kind": "webxr",
                    "value": {"countdownRemainingMs": 3000},
                },
            )
            self.assertEqual(server._hud, {"countdownRemainingMs": 3000})
            self.assertEqual(server._hud_client, low_client)

        asyncio.run(scenario())

    def test_tracker_policy_keeps_quest_frames_view_only(self) -> None:
        server = VRServer(VRServerConfig(pose_source_kind="tracker"))
        quest = _frame(
            1,
            source_id="quest",
            source_kind="webxr",
            pose_profile="meta-quest-touch-plus",
            pose_space="grip",
        )
        tracker = _frame(1, source_id="bridge", source_kind="tracker")

        self.assertFalse(server._ingest_frame_obj(quest, "network", 1))
        self.assertIsNone(server.get_frame())
        live = get_last_quest_pose_datum()
        assert live is not None
        self.assertEqual(live["commonKey"], "quest:meta-quest-touch-plus:grip")
        self.assertTrue(live["live"])
        self.assertTrue(server._ingest_frame_obj(tracker, "bridge", 2))
        self.assertEqual(server.get_frame().pose_source_id, "bridge")  # type: ignore[union-attr]
        self.assertFalse(
            server._ingest_frame_obj(
                _frame(2, source_id="quest", source_kind="webxr"), "network", 1
            )
        )
        self.assertEqual(server.get_frame().pose_source_id, "bridge")  # type: ignore[union-attr]
        server._drop_pose_client(1)
        self.assertIsNone(get_last_quest_pose_datum())

    def test_reloaded_quest_replaces_half_open_stale_owner(self) -> None:
        server = VRServer(VRServerConfig(pose_source_kind="webxr"))
        old = _frame(20, source_id="old-page", source_kind="webxr")
        new = _frame(1, source_id="new-page", source_kind="webxr")
        with mock.patch("almond_axol.vr.server.time.monotonic", return_value=10.0):
            self.assertTrue(server._ingest_frame_obj(old, "network", 1))
        with mock.patch("almond_axol.vr.server.time.monotonic", return_value=10.5):
            self.assertFalse(server._ingest_frame_obj(new, "network", 2))
        with mock.patch("almond_axol.vr.server.time.monotonic", return_value=11.1):
            self.assertTrue(server._ingest_frame_obj(new, "network", 2))

        self.assertEqual(server.get_frame().pose_source_id, "new-page")  # type: ignore[union-attr]
        with mock.patch("almond_axol.vr.server.time.monotonic", return_value=11.2):
            self.assertFalse(server._ingest_frame_obj(old, "network", 1))

    def test_managed_tracker_accepts_only_its_one_run_source_id(self) -> None:
        server = VRServer(
            VRServerConfig(
                pose_source_kind="tracker",
                expected_pose_source_id="managed-lighthouse-run",
            )
        )
        stray = _frame(1, source_id="forgotten-standalone", source_kind="tracker")
        managed = _frame(1, source_id="managed-lighthouse-run", source_kind="tracker")

        self.assertFalse(server._ingest_frame_obj(stray, "stray", 30))
        self.assertIsNone(server.get_frame())
        self.assertTrue(server._ingest_frame_obj(managed, "managed", 31))
        self.assertEqual(
            server.get_frame().pose_source_id,  # type: ignore[union-attr]
            "managed-lighthouse-run",
        )
        self.assertFalse(
            server._ingest_frame_obj(
                _frame(
                    2,
                    source_id="forgotten-standalone",
                    source_kind="tracker",
                ),
                "stray",
                30,
            )
        )

    def test_target_ray_datum_is_reported_but_never_offered_as_a_key(self) -> None:
        server = VRServer(VRServerConfig(pose_source_kind="webxr"))
        frame = _frame(
            1,
            source_id="old-webxr",
            source_kind="webxr",
            pose_profile="meta-quest-touch-plus",
            pose_space="target-ray",
        )

        self.assertTrue(server._ingest_frame_obj(frame, "network", 20))
        live = get_last_quest_pose_datum()
        assert live is not None
        self.assertEqual(live["left"]["poseSpace"], "target-ray")
        self.assertIsNone(live["commonKey"])
        server._drop_pose_client(20)


if __name__ == "__main__":
    unittest.main()
