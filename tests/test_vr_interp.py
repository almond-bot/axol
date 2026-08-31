from __future__ import annotations

import math
import unittest

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
    )


class PoseInterpolatorSafetyTest(unittest.TestCase):
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


class PoseSourceArbitrationTest(unittest.TestCase):
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
