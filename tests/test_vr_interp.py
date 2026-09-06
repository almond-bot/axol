from __future__ import annotations

import asyncio
import json
import math
import multiprocessing
import threading
import unittest
from unittest import mock

from pydantic import ValidationError
from starlette.testclient import TestClient

from almond_axol.video.video_proc import VideoRelayProcess
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
    def test_reset_during_render_cannot_return_or_commit_old_generation(self) -> None:
        import almond_axol.vr.interp as interp_module

        interp = PoseInterpolator(
            min_delay_s=0.0,
            max_delay_s=0.0,
            smooth_window_s=0.0,
        )
        interp.push(_frame(1, x=1.0), now=1.01)
        interp.push(_frame(2, x=2.0), now=1.02)

        entered_render = threading.Event()
        release_render = threading.Event()
        original_interpolate = interp_module._interpolate

        def blocked_interpolate(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            entered_render.set()
            self.assertTrue(release_render.wait(1.0))
            return original_interpolate(*args, **kwargs)

        results: list[VRFrame | None] = []
        with mock.patch.object(
            interp_module, "_interpolate", side_effect=blocked_interpolate
        ):
            sample_thread = threading.Thread(
                target=lambda: results.append(interp.sample(now=1.02))
            )
            sample_thread.start()
            self.assertTrue(entered_render.wait(1.0))
            interp.reset()
            # Simulate the replacement owner publishing before the old numpy
            # render returns to commit its result.
            replacement = _frame(3, x=30.0)
            interp.push(replacement, now=1.03)
            release_render.set()
            sample_thread.join(1.0)

        self.assertFalse(sample_thread.is_alive())
        self.assertEqual(results, [None])
        self.assertIsNone(interp._last_out)
        self.assertIs(interp.sample(now=1.03), replacement)

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

    def test_stale_high_sequence_tab_cannot_repin_hud_after_rebase(self) -> None:
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
            # The high-range tab remains connected but no longer publishes poses;
            # the live lower range becomes the source's timing/sequence domain.
            with mock.patch("almond_axol.vr.server.time.monotonic", return_value=11.1):
                self.assertTrue(
                    server._ingest_frame_obj(
                        _frame(6, source_id=source, source_kind="webxr"),
                        "low-tab",
                        low_client,
                    )
                )
                await server._handle_signaling(
                    mock.AsyncMock(),
                    high_client,
                    {
                        "type": "hud",
                        "pose_source_id": source,
                        "pose_source_kind": "webxr",
                        "value": {"confirm": "stale-high"},
                    },
                )
                await server._handle_signaling(
                    mock.AsyncMock(),
                    low_client,
                    {
                        "type": "hud",
                        "pose_source_id": source,
                        "pose_source_kind": "webxr",
                        "value": {"confirm": "active-low"},
                    },
                )

            self.assertEqual(server._hud, {"confirm": "active-low"})
            self.assertEqual(server._hud_client, low_client)
            self.assertEqual(server._hud_pose_seq, 6)

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

    def test_disconnecting_latest_quest_restores_other_live_datum(self) -> None:
        server = VRServer(VRServerConfig(pose_source_kind="tracker"))
        first = _frame(
            1,
            source_id="quest-grip",
            source_kind="webxr",
            pose_profile="meta-quest-touch-plus",
            pose_space="grip",
        )
        latest = _frame(
            1,
            source_id="quest-target-ray",
            source_kind="webxr",
            pose_profile="meta-quest-touch-plus",
            pose_space="target-ray",
        )

        self.assertFalse(server._ingest_frame_obj(first, "viewer-one", 1))
        self.assertFalse(server._ingest_frame_obj(latest, "viewer-two", 2))
        live = get_last_quest_pose_datum()
        assert live is not None
        self.assertEqual(live["left"]["poseSpace"], "target-ray")

        server._drop_pose_client(2)
        restored = get_last_quest_pose_datum()
        assert restored is not None
        self.assertEqual(restored["commonKey"], "quest:meta-quest-touch-plus:grip")
        server._drop_pose_client(1)
        self.assertIsNone(get_last_quest_pose_datum())

    def test_disable_cancels_offer_tasks_before_closing_manager(self) -> None:
        async def scenario() -> None:
            server = VRServer()
            server._loop = asyncio.get_running_loop()
            offer_entered = asyncio.Event()
            offer_exited = asyncio.Event()

            class Manager:
                async def create_offer(self, _client_id: int) -> None:
                    offer_entered.set()
                    try:
                        await asyncio.Future()
                    finally:
                        offer_exited.set()

                async def close_all(self) -> None:
                    self.assert_offer_exited()

                @staticmethod
                def assert_offer_exited() -> None:
                    if not offer_exited.is_set():
                        raise AssertionError("manager closed before offer task exited")

            server._webrtc = Manager()
            server._control = mock.AsyncMock()
            server._spawn(server._send_webrtc_offer(mock.AsyncMock(), 1))
            await offer_entered.wait()

            await server.disable()

            self.assertTrue(offer_exited.is_set())
            self.assertEqual(server._signaling_tasks, set())
            server._control.close_all.assert_awaited_once_with()

        asyncio.run(scenario())

    def test_failed_control_offer_closes_partial_peer_and_rearms_client(self) -> None:
        async def scenario() -> None:
            server = VRServer()
            websocket = mock.AsyncMock()
            server._control = mock.AsyncMock()
            server._control.create_offer.side_effect = RuntimeError("ICE failed")

            await server._send_control_offer(websocket, 7)

            server._control.close.assert_awaited_once_with(7)
            websocket.send_text.assert_awaited_once_with(
                json.dumps({"type": "control-error"})
            )
            self.assertNotIn(7, server._control_offering)

        asyncio.run(scenario())

    def test_failed_video_offer_closes_partial_peer(self) -> None:
        async def scenario() -> None:
            server = VRServer()
            websocket = mock.AsyncMock()
            manager = mock.AsyncMock()
            manager.create_offer.side_effect = RuntimeError("ICE failed")
            server._webrtc = manager

            await server._send_webrtc_offer(websocket, 7)

            manager.close.assert_awaited_once_with(7)
            websocket.send_text.assert_awaited_once_with(
                json.dumps({"type": "webrtc-unavailable"})
            )
            self.assertNotIn(7, server._video_offering)
            self.assertNotIn(7, server._client_offer_tasks)

        asyncio.run(scenario())

    def test_undeliverable_video_offer_closes_created_peer(self) -> None:
        async def scenario() -> None:
            server = VRServer()
            websocket = mock.AsyncMock()
            websocket.send_text.side_effect = RuntimeError("socket left")
            manager = mock.AsyncMock()
            manager.create_offer.return_value = ("offer-sdp", {"0": "left"})
            server._webrtc = manager

            await server._send_webrtc_offer(websocket, 9)

            manager.close.assert_awaited_once_with(9)
            self.assertNotIn(9, server._video_offering)
            self.assertNotIn(9, server._client_offer_tasks)

        asyncio.run(scenario())

    def test_client_disconnect_joins_offer_before_peer_close(self) -> None:
        async def scenario() -> None:
            server = VRServer()
            offer_entered = asyncio.Event()
            offer_exited = asyncio.Event()

            class Manager:
                async def create_offer(
                    self, _client_id: int
                ) -> tuple[str, dict[str, str]]:
                    offer_entered.set()
                    try:
                        await asyncio.Future()
                    finally:
                        offer_exited.set()
                    raise AssertionError("unreachable")

                async def close(self, _client_id: int) -> None:
                    if not offer_exited.is_set():
                        raise AssertionError("peer closed before offer task exited")

            manager = Manager()
            server._webrtc = manager
            server._spawn(server._send_webrtc_offer(mock.AsyncMock(), 11))
            await offer_entered.wait()

            await server._drain_client_offers(11)
            await manager.close(11)

            self.assertTrue(offer_exited.is_set())
            self.assertNotIn(11, server._client_offer_tasks)

        asyncio.run(scenario())

    def test_client_disconnect_cancels_offer_before_coroutine_starts(self) -> None:
        async def scenario() -> None:
            server = VRServer()
            manager = mock.AsyncMock()
            server._webrtc = manager

            server._spawn(
                server._send_webrtc_offer(mock.AsyncMock(), 12),
                offer_client_id=12,
            )
            # Do not yield between spawn and teardown. The offer task is still
            # owned by this client and must be joined before peer cleanup.
            await server._drain_client_offers(12)
            await manager.close(12)
            await asyncio.sleep(0)

            manager.create_offer.assert_not_awaited()
            manager.close.assert_awaited_once_with(12)
            self.assertNotIn(12, server._client_offer_tasks)

        asyncio.run(scenario())

    def test_disable_interrupts_executor_backed_video_relay_offer(self) -> None:
        async def scenario() -> None:
            parent_conn, relay_conn = multiprocessing.Pipe()
            relay = VideoRelayProcess.__new__(VideoRelayProcess)
            relay._conn = parent_conn
            relay._lock = threading.Lock()
            relay._next_offer_request_id = 0
            relay._shutdown_requested = threading.Event()

            server = VRServer()
            server._loop = asyncio.get_running_loop()
            server._webrtc = relay
            server._control = mock.AsyncMock()
            server._spawn(server._send_webrtc_offer(mock.AsyncMock(), 7))
            try:
                request = await asyncio.wait_for(
                    asyncio.to_thread(relay_conn.recv), timeout=1.0
                )
                self.assertEqual(request, ("offer", 1, 7))

                # The relay deliberately never answers. Cancellation must wake
                # the real executor-backed request and release its pipe lock so
                # close_all is delivered well inside VRTeleop's 5 s join bound.
                await asyncio.wait_for(server.disable(), timeout=1.0)
                close = await asyncio.wait_for(
                    asyncio.to_thread(relay_conn.recv), timeout=1.0
                )
                self.assertEqual(close, ("close_all",))
                self.assertEqual(server._signaling_tasks, set())
            finally:
                parent_conn.close()
                relay_conn.close()

        asyncio.run(scenario())

    def test_video_relay_retry_discards_late_cancelled_offer_response(self) -> None:
        async def scenario() -> None:
            parent_conn, relay_conn = multiprocessing.Pipe()
            relay = VideoRelayProcess.__new__(VideoRelayProcess)
            relay._conn = parent_conn
            relay._lock = threading.Lock()
            relay._next_offer_request_id = 0
            relay._shutdown_requested = threading.Event()
            try:
                abandoned = asyncio.create_task(relay.create_offer(7))
                first = await asyncio.wait_for(
                    asyncio.to_thread(relay_conn.recv), timeout=1.0
                )
                self.assertEqual(first, ("offer", 1, 7))
                abandoned.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await abandoned

                replacement = asyncio.create_task(relay.create_offer(7))
                second = await asyncio.wait_for(
                    asyncio.to_thread(relay_conn.recv), timeout=1.0
                )
                self.assertEqual(second, ("offer", 2, 7))
                # The first response arrived after its asyncio owner retired.
                # The replacement must skip it even though the client ID is the
                # same, then consume only its request-correlated response.
                relay_conn.send(("offer_ok", 1, 7, "stale-sdp", {"0": "old"}))
                relay_conn.send(("offer_ok", 2, 7, "fresh-sdp", {"0": "left"}))
                result = await asyncio.wait_for(replacement, timeout=1.0)
                self.assertEqual(result, ("fresh-sdp", {"0": "left"}))
            finally:
                parent_conn.close()
                relay_conn.close()

        asyncio.run(scenario())

    def test_video_relay_shutdown_wakes_offer_before_taking_pipe_lock(self) -> None:
        async def scenario() -> None:
            parent_conn, relay_conn = multiprocessing.Pipe()
            relay = VideoRelayProcess.__new__(VideoRelayProcess)
            relay._conn = parent_conn
            relay._lock = threading.Lock()
            relay._next_offer_request_id = 0
            relay._shutdown_requested = threading.Event()
            relay.raw_cameras = {}
            relay._proc = mock.Mock()
            relay._proc.is_alive.return_value = False
            offer = asyncio.create_task(relay.create_offer(9))
            try:
                request = await asyncio.wait_for(
                    asyncio.to_thread(relay_conn.recv), timeout=1.0
                )
                self.assertEqual(request, ("offer", 1, 9))

                shutdown = asyncio.create_task(asyncio.to_thread(relay.shutdown))
                sentinel = await asyncio.wait_for(
                    asyncio.to_thread(relay_conn.recv), timeout=1.0
                )
                self.assertIsNone(sentinel)
                await asyncio.wait_for(shutdown, timeout=1.0)
                with self.assertRaisesRegex(RuntimeError, "shutting down"):
                    await offer
            finally:
                if not offer.done():
                    offer.cancel()
                parent_conn.close()
                relay_conn.close()

        asyncio.run(scenario())

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
