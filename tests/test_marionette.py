"""Marionette teleoperation: phone state, the clutch and gates, the session."""

from __future__ import annotations

import asyncio
import math
import threading
import time
import unittest
from typing import Any
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from almond_axol.jog.backends import SimBackend
from almond_axol.jog.session import JogError, active_session
from almond_axol.kinematics.arm_ik import exp_rotation, rotation_vector
from almond_axol.marionette.cameras import (
    CameraHub,
    SyntheticCameraSource,
    encode_jpeg,
    streams_for_view,
)
from almond_axol.marionette.gateway import GatewayClient, gateway_url
from almond_axol.marionette.phones import (
    DEADMAN,
    Phone,
    PhonePose,
    PhoneTracker,
    parse_stick,
)
from almond_axol.marionette.session import MarionetteSession
from almond_axol.marionette.teleop import (
    ArmTeleop,
    TeleopLimits,
    gate_reason,
    map_target,
    owners,
)


def _pose(
    position: Any = (0.0, 0.0, 0.0),
    rotvec: Any = (0.0, 0.0, 0.0),
    *,
    tracking: str = "normal",
    source: str = "arkit",
    calibration: str = "guided-v2:1",
    session: str = "s1",
    received: float | None = None,
) -> PhonePose:
    return PhonePose(
        np.asarray(position, float),
        exp_rotation(np.asarray(rotvec, float)),
        tracking,
        source,
        True,
        calibration,
        session,
        time.monotonic() if received is None else received,
    )


def _phone(pid: str = "phone-1", arm: str = "left", **pose: Any) -> Phone:
    phone = Phone(pid, pid.replace("phone-", "Phone "), connected=True)
    phone.values["arm.select"] = arm
    phone.pose = _pose(**pose)
    return phone


class GatewayAddressTest(unittest.TestCase):
    def test_addresses_like_the_app(self) -> None:
        self.assertEqual(gateway_url("192.168.1.20"), "ws://192.168.1.20:8787/ws")
        self.assertEqual(gateway_url("localhost:18787"), "ws://localhost:18787/ws")
        self.assertEqual(gateway_url("wss://Robot.local"), "wss://robot.local:8787/ws")
        self.assertEqual(gateway_url("http://10.0.0.2:9000/"), "ws://10.0.0.2:9000/ws")
        for bad in ("", "not an address!", "host:70000"):
            with self.assertRaises(ValueError):
                gateway_url(bad)

    def test_stick_samples_are_parsed_and_kept_in_the_circle(self) -> None:
        self.assertEqual(parse_stick({"x": 0.5, "y": -0.25}), (0.5, -0.25))
        x, y = parse_stick({"x": 3, "y": 4})
        self.assertAlmostEqual(math.hypot(x, y), 1.0)
        for bad in (
            None,
            1.0,
            "1,0",
            {"x": 1},
            {"x": "a", "y": 0},
            {"x": math.inf, "y": 0},
        ):
            self.assertIsNone(parse_stick(bad))


class PhoneTrackerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.now = 100.0
        self.tracker = PhoneTracker(clock=lambda: self.now)
        self.tracker.apply(
            {
                "type": "config",
                "phones": [{"id": "phone-1", "name": "Phone 1", "connected": True}],
                "phone": {
                    "id": "phone-1",
                    "name": "Phone 1",
                    "color": "#c98200",
                    "connected": True,
                },
                "controls": [
                    {
                        "id": "arm",
                        "type": "select",
                        "action": "arm.select",
                        "value": "right",
                    },
                    {
                        "id": "estop",
                        "type": "toggle",
                        "action": "safety.estop",
                        "value": False,
                    },
                    {
                        "id": "deadman",
                        "type": "button",
                        "action": "teleop.deadman",
                        "hold": True,
                        "held": False,
                    },
                ],
                "calibration": {"status": "captured", "epoch": 2},
            }
        )

    def phone(self) -> Phone:
        return self.tracker.snapshot()["phone-1"]

    def input(self, action: str, value: Any = None, **extra: Any) -> None:
        self.tracker.apply(
            {
                "type": "input",
                "phoneId": "phone-1",
                "action": action,
                "value": value,
                **extra,
            }
        )

    def test_config_sets_values_and_calibration(self) -> None:
        phone = self.phone()
        self.assertTrue(phone.connected)
        self.assertEqual(phone.arm, "right")
        self.assertFalse(phone.stop)
        self.assertEqual((phone.calibration, phone.epoch), ("captured", 2))

    def test_pose_parsing(self) -> None:
        quat = Rotation.from_rotvec([0, 0, 0.3]).as_quat()
        self.tracker.apply(
            {
                "type": "pose",
                "phoneId": "phone-1",
                "position": {"x": 0.1, "y": 0.2, "z": 0.3},
                "rotation": dict(zip("xyzw", quat, strict=True)),
                "trackingState": "normal",
                "source": "arkit",
                "heartbeat": True,
                "calibrationVersion": "guided-v2:2",
                "sessionId": "s1",
            }
        )
        pose = self.phone().pose
        assert pose is not None
        np.testing.assert_allclose(pose.position, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(
            rotation_vector(pose.rotation), [0, 0, 0.3], atol=1e-9
        )
        self.assertEqual(pose.received, self.now)
        # A heartbeat-false pose (the gateway's watchdog) drops it.
        self.tracker.apply(
            {
                "type": "pose",
                "phoneId": "phone-1",
                "position": {"x": 0, "y": 0, "z": 0},
                "rotation": {"x": 0, "y": 0, "z": 0, "w": 1},
                "trackingState": "unavailable",
                "heartbeat": False,
            }
        )
        self.assertIsNone(self.phone().pose)

    def test_deadman_presses_renewals_and_releases(self) -> None:
        self.input("teleop.deadman", True)
        phone = self.phone()
        self.assertEqual((phone.deadman_down, phone.deadman_presses), (True, 1))
        self.now += 0.5
        self.input("teleop.deadman", True, renew=True)
        self.assertEqual(self.phone().holds[DEADMAN].at, self.now)
        self.assertEqual(self.phone().deadman_presses, 1)
        self.assertTrue(self.phone().deadman_live(self.now + 0.5, 0.6))
        self.assertFalse(self.phone().deadman_live(self.now + 0.7, 0.6))
        # A release the gateway inferred counts like the phone's own.
        self.input("teleop.deadman", False, source="gateway", reason="renewals stopped")
        self.assertFalse(self.phone().deadman_down)
        # A renewal never restarts a hold.
        self.input("teleop.deadman", True, renew=True)
        self.assertFalse(self.phone().deadman_down)
        self.input("teleop.deadman", True)
        self.assertEqual(self.phone().deadman_presses, 2)

    def test_gateway_hold_state_ends_a_hold(self) -> None:
        self.input("teleop.deadman", True)
        self.tracker.apply(
            {
                "type": "config",
                "phones": [],
                "phone": {"id": "phone-1", "connected": True},
                "controls": [
                    {
                        "id": "deadman",
                        "type": "button",
                        "action": "teleop.deadman",
                        "held": False,
                    }
                ],
            }
        )
        self.assertFalse(self.phone().deadman_down)

    def test_stop_grip_and_slider_inputs(self) -> None:
        self.input("safety.estop", True)
        self.assertTrue(self.phone().stop)
        self.input("gripper.set", 0.4, live=True)
        self.input("gripper.set", 0.4, live=True, renew=True)  # a resting finger
        self.input("gripper.set", 7)
        phone = self.phone()
        self.assertEqual((phone.grip, phone.grip_seq), (1.0, 2))
        self.input("base.speedLimit", 80, live=True)
        self.input("teleop.translationScale", 0.5, live=True)
        self.assertAlmostEqual(self.phone().speed, 0.8)
        self.assertAlmostEqual(self.phone().scale, 0.5)

    def test_reset_presses_are_counted_per_arm(self) -> None:
        self.input("arm.resetLeft")
        self.input("arm.resetRight")
        self.input("arm.resetRight")
        self.assertEqual(self.phone().resets, {"left": 1, "right": 2})

    def test_drive_stick_turn_and_lift(self) -> None:
        phone = self.phone()
        self.assertEqual(phone.base_command(self.now, 0.25, 0.6), (0.0, 0.0, 0.0, 0))
        self.input("base.drive", {"x": 0.6, "y": -0.2}, live=True)
        self.input("base.turnLeft", True)
        self.input("lift.down", True)
        self.assertEqual(
            self.phone().base_command(self.now, 0.25, 0.6), (0.6, -0.2, 1.0, -1)
        )
        # A stick sample older than its timeout is a dead source.
        self.assertIsNone(self.phone().base_command(self.now + 0.3, 0.25, 0.6))
        self.now += 0.2
        self.input("base.drive", {"x": 0.6, "y": -0.2}, live=True, renew=True)
        self.input("base.turnLeft", True, renew=True)
        self.input("lift.down", True, renew=True)
        self.assertIsNotNone(self.phone().base_command(self.now + 0.2, 0.25, 0.6))
        # Let go of everything: zeros, and nothing goes stale.
        self.input("base.drive", {"x": 0, "y": 0})
        self.input("base.turnLeft", False)
        self.input("lift.down", False)
        self.assertEqual(
            self.phone().base_command(self.now + 5, 0.25, 0.6), (0.0, 0.0, 0.0, 0)
        )

    def test_a_drag_the_gateway_ended_stays_ended_until_let_go(self) -> None:
        self.input("base.drive", {"x": 1, "y": 0}, live=True)
        self.input(
            "base.drive", {"x": 0, "y": 0}, source="gateway", reason="samples stopped"
        )
        self.input("base.drive", {"x": 1, "y": 0}, live=True)  # still touching
        self.assertEqual(self.phone().base_command(self.now, 0.25, 0.6)[:2], (0.0, 0.0))
        self.input("base.drive", {"x": 0, "y": 0})  # let go
        self.input("base.drive", {"x": 0.5, "y": 0}, live=True)
        self.assertEqual(self.phone().base_command(self.now, 0.25, 0.6)[:2], (0.5, 0.0))

    def test_a_malformed_value_is_ignored(self) -> None:
        self.input("base.drive", {"x": "inf", "y": 0}, live=True)
        self.input("gripper.set", math.inf)
        self.assertEqual(
            self.phone().base_command(self.now, 0.25, 0.6), (0.0, 0.0, 0.0, 0)
        )
        self.assertIsNone(self.phone().grip)

    def test_mirror_inputs_move_nothing_on_the_robot(self) -> None:
        robot = PhoneTracker(clock=lambda: self.now, allow_mirror=False)
        robot.apply(
            {"type": "config", "phones": [{"id": "phone-1", "connected": True}]}
        )
        for action, value in (
            ("teleop.deadman", True),
            ("base.drive", {"x": 1, "y": 0}),
            ("gripper.set", 1.0),
            ("base.turnLeft", True),
            ("arm.resetLeft", None),
        ):
            robot.apply(
                {
                    "type": "input",
                    "phoneId": "phone-1",
                    "action": action,
                    "value": value,
                    "live": True,
                    "source": "mirror",
                }
            )
        phone = robot.snapshot()["phone-1"]
        self.assertFalse(phone.deadman_down)
        self.assertIsNone(phone.grip)
        self.assertEqual(phone.base_command(self.now, 0.25, 0.6), (0.0, 0.0, 0.0, 0))
        self.assertEqual(phone.resets, {"left": 0, "right": 0})
        robot.apply(
            {
                "type": "input",
                "phoneId": "phone-1",
                "action": "safety.estop",
                "value": True,
                "source": "mirror",
            }
        )
        self.assertTrue(robot.snapshot()["phone-1"].stop)

    def test_a_restarted_gateway_forgets_its_old_phones_but_not_a_stop(self) -> None:
        self.tracker.apply(
            {
                "type": "config",
                "sessionId": "gw-1",
                "phones": [
                    {"id": "phone-1", "connected": True},
                    {"id": "phone-2", "name": "Phone 2", "connected": True},
                ],
            }
        )
        self.tracker.apply(
            {
                "type": "input",
                "phoneId": "phone-2",
                "action": "safety.estop",
                "value": True,
            }
        )
        self.tracker.lost_gateway()
        self.tracker.apply(
            {
                "type": "config",
                "sessionId": "gw-2",
                "phones": [{"id": "phone-1", "name": "Phone 1", "connected": True}],
            }
        )
        phones = self.tracker.snapshot()
        self.assertEqual(sorted(phones), ["phone-1", "phone-2@gw-1"])
        self.assertTrue(phones["phone-2@gw-1"].stop)
        self.assertFalse(phones["phone-1"].stop)
        self.tracker.clear_offline_stops()
        self.assertEqual(sorted(self.tracker.snapshot()), ["phone-1"])

    def test_disconnect_and_lost_gateway_release_everything(self) -> None:
        self.input("teleop.deadman", True)
        self.input("base.drive", {"x": 1, "y": 0}, live=True)
        self.tracker.apply(
            {"type": "config", "phones": [{"id": "phone-1", "connected": False}]}
        )
        phone = self.phone()
        self.assertFalse(phone.connected or phone.deadman_down or phone.stick.live)
        self.tracker.apply(
            {"type": "config", "phones": [{"id": "phone-1", "connected": True}]}
        )
        self.input("teleop.deadman", True)
        self.tracker.lost_gateway()
        self.assertFalse(self.phone().connected or self.phone().deadman_down)


class MappingTest(unittest.TestCase):
    def test_phone_motion_moves_the_tip_by_the_same_amount_times_scale(self) -> None:
        tip = (np.array([0.3, 0.2, 0.1]), exp_rotation(np.array([0.0, 1.0, 0.0])))
        anchor = _pose([1.0, 2.0, 3.0], [0.0, 0.0, 0.5])
        moved = _pose([1.1, 1.95, 3.02], [0.0, 0.0, 0.5])
        p, r = map_target(moved, anchor, tip, scale=2.0)
        np.testing.assert_allclose(p, [0.5, 0.1, 0.14], atol=1e-12)
        np.testing.assert_allclose(r, tip[1], atol=1e-12)

    def test_phone_rotation_turns_the_tip_about_the_same_world_axis(self) -> None:
        tip = (np.zeros(3), exp_rotation(np.array([0.0, 1.0, 0.0])))
        anchor = _pose(rotvec=[0.0, 0.0, 0.5])
        turned = _pose(rotvec=[0.0, 0.0, 0.8])  # 0.3 rad more about +z
        _, r = map_target(turned, anchor, tip, 1.0)
        np.testing.assert_allclose(
            rotation_vector(r @ tip[1].T), [0.0, 0.0, 0.3], atol=1e-9
        )
        _, half = map_target(turned, anchor, tip, 1.0, rotation_scale=0.5)
        np.testing.assert_allclose(
            rotation_vector(half @ tip[1].T), [0.0, 0.0, 0.15], atol=1e-9
        )


class GateTest(unittest.TestCase):
    limits = TeleopLimits()

    def reason(
        self, phone: Phone, *, sim: bool = False, now: float | None = None
    ) -> Any:
        return gate_reason(
            phone,
            time.monotonic() if now is None else now,
            self.limits,
            allow_synthetic=sim,
        )

    def test_good_tracking_passes(self) -> None:
        self.assertIsNone(self.reason(_phone()))

    def test_every_failure_has_a_reason(self) -> None:
        offline = _phone()
        offline.connected = False
        calibrating = _phone()
        calibrating.calibration = "capturing"
        stale = _phone(received=time.monotonic() - 1.0)
        cases = {
            "offline": (offline, False),
            "calibration": (calibrating, False),
            "No pose": (stale, False),
            "limited": (_phone(tracking="limited"), False),
            "starting": (_phone(tracking="initializing"), False),
            "Simulated motion": (_phone(source="synthetic"), False),
        }
        for word, (phone, sim) in cases.items():
            reason = self.reason(phone, sim=sim)
            self.assertIsNotNone(reason, word)
            self.assertIn(word, reason)

    def test_the_robot_needs_a_calibrated_phone(self) -> None:
        phone = _phone()
        phone.calibration = "uncalibrated"
        reason = gate_reason(
            phone,
            time.monotonic(),
            self.limits,
            allow_synthetic=False,
            require_calibration=True,
        )
        self.assertIn("Calibrate", reason)
        phone.calibration = "captured"
        self.assertIsNone(
            gate_reason(
                phone,
                time.monotonic(),
                self.limits,
                allow_synthetic=False,
                require_calibration=True,
            )
        )

    def test_synthetic_motion_is_allowed_in_sim(self) -> None:
        self.assertIsNone(
            self.reason(_phone(source="synthetic", tracking="limited"), sim=True)
        )


class OwnersTest(unittest.TestCase):
    def test_lowest_number_then_the_engaged_phone(self) -> None:
        one, two = _phone("phone-1"), _phone("phone-2")
        result = owners([two, one], {"left": None, "right": None})
        self.assertIs(result["left"], one)
        self.assertIsNone(result["right"])
        result = owners([one, two], {"left": "phone-2", "right": None})
        self.assertIs(result["left"], two)
        two.values["arm.select"] = "right"
        result = owners([one, two], {"left": "phone-2", "right": None})
        self.assertIs(result["left"], one)
        self.assertIs(result["right"], two)


class ArmTeleopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = TeleopLimits(max_speed=0.5, max_turn_rate=3.0, pose_cutoff=50.0)
        self.arm = ArmTeleop("left", self.limits, 125.0)
        self.tip = (np.array([0.3, 0.2, 0.1]), np.eye(3))

    def test_presses_are_counted_once_and_never_replayed(self) -> None:
        phone = _phone()
        phone.holds[DEADMAN].presses = 3
        self.assertFalse(self.arm.new_press(phone))  # first sight
        self.assertFalse(self.arm.new_press(phone))
        phone.holds[DEADMAN].presses = 4
        self.assertTrue(self.arm.new_press(phone))
        self.assertFalse(self.arm.new_press(phone))
        phone.holds[DEADMAN].presses = 5
        self.arm.forget_presses(phone)
        self.assertFalse(self.arm.new_press(phone))

    def test_engaging_holds_the_tip_where_it_is(self) -> None:
        phone = _phone(position=[1.0, 1.0, 1.0])
        self.arm.engage(phone, self.tip)
        p, r = self.arm.target(phone, time.monotonic(), 1 / 125)
        np.testing.assert_allclose(p, self.tip[0], atol=1e-6)
        np.testing.assert_allclose(r, self.tip[1], atol=1e-6)

    def test_a_jump_is_rate_limited(self) -> None:
        phone = _phone()
        self.arm.engage(phone, self.tip)
        phone.pose = _pose([1.0, 0.0, 0.0], [0.0, 0.0, 1.5])
        now = time.monotonic()
        p, r = self.arm.target(phone, now, 0.01)
        self.assertLessEqual(np.linalg.norm(p - self.tip[0]), 0.5 * 0.01 + 1e-9)
        self.assertLessEqual(np.linalg.norm(rotation_vector(r)), 3.0 * 0.01 + 1e-9)

    def test_anchor_is_dropped_by_a_new_session_or_calibration(self) -> None:
        phone = _phone()
        self.arm.engage(phone, self.tip)
        self.assertTrue(self.arm.anchor_valid(phone))
        phone.pose = _pose(calibration="guided-v2:2")
        self.assertFalse(self.arm.anchor_valid(phone))
        phone.pose = _pose(session="s2")
        self.assertFalse(self.arm.anchor_valid(phone))

    def test_reach_limit_needs_a_lasting_lag(self) -> None:
        phone = _phone()
        self.arm.engage(phone, self.tip)
        far = (self.tip[0] + [0.05, 0, 0], np.eye(3))
        self.arm.note_tip(far, 10.0)
        self.assertFalse(self.arm.limited)
        self.arm.note_tip(far, 10.4)
        self.assertTrue(self.arm.limited)
        self.arm.note_tip(self.tip, 10.5)
        self.assertFalse(self.arm.limited)


class CameraHubTest(unittest.TestCase):
    def test_views(self) -> None:
        self.assertEqual(streams_for_view("all"), ("head", "left", "right"))
        self.assertEqual(streams_for_view("wrists"), ("left", "right"))
        self.assertEqual(streams_for_view("nonsense"), ("head",))

    def test_encode_downscales_and_never_upscales(self) -> None:
        import cv2

        big = np.zeros((600, 960, 3), np.uint8)
        small = cv2.imdecode(np.frombuffer(encode_jpeg(big, 640, 60), np.uint8), 1)
        self.assertEqual(small.shape[:2], (400, 640))
        tiny = cv2.imdecode(
            np.frombuffer(
                encode_jpeg(np.zeros((90, 160, 3), np.uint8), 640, 60), np.uint8
            ),
            1,
        )
        self.assertEqual(tiny.shape[:2], (90, 160))

    def test_only_watched_streams_are_encoded(self) -> None:
        hub = CameraHub(SyntheticCameraSource(), fps=30, width=160, quality=50)
        hub.start()
        try:
            hub.want("page", ["left", "bogus"])
            self.assertTrue(_wait(lambda: hub.latest("left") is not None))
            self.assertIsNone(hub.latest("head"))
            first = hub.latest("left")
            self.assertTrue(_wait(lambda: hub.latest("left").seq != first.seq))
            self.assertEqual(hub.describe()["source"], "synthetic")
        finally:
            hub.stop()


class ZedSourceTest(unittest.TestCase):
    """The robot's cameras through the video relay (faked: no ZED here)."""

    def test_relay_readers_are_used(self) -> None:
        from almond_axol.marionette.cameras import ZedCameraSource

        frame = np.full((600, 960, 3), 80, np.uint8)

        class Reader:
            def read_latest(self) -> np.ndarray:
                return frame

        made: list[Any] = []

        class Relay:
            def __init__(self, specs: dict[str, Any], want_raw: bool) -> None:
                self.specs = specs
                self.closed = False
                made.append(self)

            @property
            def readable_raw_cameras(self) -> dict[str, Any]:
                return {name: Reader() for name in self.specs}

            def shutdown(self) -> None:
                self.closed = True

        with (
            patch("almond_axol.video.video.webrtc_available", return_value=True),
            patch("almond_axol.video.video_proc.VideoRelayProcess", Relay),
            patch("almond_axol.zed.stereo_serials", return_value={22}),
        ):
            source = ZedCameraSource(
                {"overhead": 22, "left_arm": 11, "right_arm": 12}, fps=12
            )
            source.start()
        self.assertEqual(source.available, ("head", "left", "right"))
        self.assertIsNone(source.error)
        spec = made[0].specs
        self.assertEqual(spec["head"]["record_eyes"], ["left"])
        self.assertFalse(spec["left"]["stream"])
        self.assertEqual(spec["left"]["fps"], 60)
        frames = source.grab(["head", "left"], 320, 60)
        self.assertEqual(set(frames), {"head", "left"})
        source.stop()
        self.assertTrue(made[0].closed)

    def test_a_stalled_stream_is_reported_not_frozen(self) -> None:
        class Once:
            name = "test"
            available = ("head",)
            error = None
            calls = 0

            def start(self) -> None: ...
            def stop(self) -> None: ...

            def grab(self, streams: Any, width: int, quality: int) -> dict[str, bytes]:
                Once.calls += 1
                return {"head": b"jpeg"} if Once.calls == 1 else {}

        with patch("almond_axol.marionette.cameras.STALE_S", 0.3):
            hub = CameraHub(Once(), fps=30)
            hub.want("page", ["head"])
            hub.start()
            try:
                self.assertTrue(_wait(lambda: hub.latest("head") is not None, 2))
                self.assertTrue(_wait(lambda: hub.latest("head") is None, 2))
                self.assertTrue(
                    _wait(lambda: "stopped sending" in (hub.error or ""), 2)
                )
            finally:
                hub.stop()


def _wait(predicate: Any, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


_PLANNER: list[Any] = []


def _planner() -> Any:
    """One real IK planner (the mobile Axol), shared by the session tests."""
    if not _PLANNER:
        from almond_axol.cli.jog import planner_factory
        from almond_axol.kinematics.config import KinematicsConfig

        _PLANNER.append(planner_factory(KinematicsConfig(robot_model="mobile"))())
    return _PLANNER[0]


class FakePhone:
    """A phone as the gateway relays it: config, a 60 Hz pose, presses."""

    def __init__(self, session: MarionetteSession, pid: str, arm: str = "left") -> None:
        self.session = session
        self.id = pid
        self.name = pid.replace("phone-", "Phone ")
        self.values: dict[str, Any] = {"arm.select": arm, "safety.estop": False}
        self.position = np.zeros(3)
        self.rotvec = np.zeros(3)
        self.tracking = "normal"
        self.calibration = "guided-v2:1"
        self.held = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._stream, daemon=True)

    def apply(self, message: dict[str, Any]) -> None:
        self.session._tracker.apply(message)

    def connect(self) -> FakePhone:
        self.config()
        self._thread.start()
        return self

    def config(self, connected: bool = True) -> None:
        info = {
            "id": self.id,
            "name": self.name,
            "connected": connected,
            "sessionId": "s1",
        }
        self.apply(
            {
                "type": "config",
                "phones": [info],
                "phone": info,
                "controls": [
                    {"id": k, "type": "select", "action": k, "value": v}
                    for k, v in self.values.items()
                ],
                "calibration": {"status": "captured", "epoch": 1},
            }
        )

    def glide(self, position: Any, seconds: float = 0.5) -> None:
        """Move like a hand: a straight line at the pose rate."""
        start, end = self.position.copy(), np.asarray(position, float)
        steps = max(1, round(seconds * 60))
        for i in range(1, steps + 1):
            self.position = start + (end - start) * i / steps
            time.sleep(1 / 60)

    def set(self, action: str, value: Any) -> None:
        self.values[action] = value
        self.config()

    def act(self, action: str, value: Any = None, **extra: Any) -> None:
        self.apply(
            {
                "type": "input",
                "phoneId": self.id,
                "action": action,
                "value": value,
                **extra,
            }
        )

    def press(self) -> None:
        self.held = True
        self.act("teleop.deadman", True)

    def release(self) -> None:
        self.held = False
        self.act("teleop.deadman", False)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(2)

    def _stream(self) -> None:
        tick = 0
        while not self._stop.wait(1 / 60):
            quat = Rotation.from_rotvec(self.rotvec).as_quat()
            self.apply(
                {
                    "type": "pose",
                    "phoneId": self.id,
                    "position": dict(zip("xyz", self.position, strict=True)),
                    "rotation": dict(zip("xyzw", quat, strict=True)),
                    "trackingState": self.tracking,
                    "source": "arkit",
                    "heartbeat": True,
                    "calibrationVersion": self.calibration,
                    "sessionId": "s1",
                }
            )
            tick += 1
            if self.held and tick % 12 == 0:
                self.act("teleop.deadman", True, renew=True)


async def _idle(*_: Any) -> None:
    await asyncio.Event().wait()


class MarionetteSessionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = SimBackend("mobile")
        planner = _planner()
        self.sent: list[tuple[Any, ...]] = []
        with patch.object(GatewayClient, "run", _idle):
            self.session = MarionetteSession(
                self.backend, lambda: planner, gateway="localhost:18999"
            )
        self.session._link.connected = True
        self.session._link.send_status = lambda *a, **k: self.sent.append(a)  # type: ignore[method-assign]
        self.phones: list[FakePhone] = []
        self.stop = threading.Event()
        with patch.object(GatewayClient, "run", _idle):
            self.thread = threading.Thread(
                target=lambda: asyncio.run(self.session.run(self.stop)), daemon=True
            )
            self.thread.start()
            self.assertTrue(_wait(lambda: active_session() is self.session))
        self.assertTrue(_wait(lambda: self.session.state()["ready"], 30))

    def tearDown(self) -> None:
        for phone in self.phones:
            phone.close()
        self.stop.set()
        self.thread.join(10)
        self.assertIsNone(active_session())

    def phone(self, pid: str = "phone-1", arm: str = "left") -> FakePhone:
        phone = FakePhone(self.session, pid, arm).connect()
        self.phones.append(phone)
        self.assertTrue(
            _wait(lambda: self.arm_state(arm)["owner"] is not None or pid != "phone-1")
        )
        time.sleep(0.05)  # the arm has seen the phone (and its press count)
        return phone

    def arm_state(self, side: str = "left") -> dict[str, Any]:
        return self.session.state()["marionette"]["arms"][side]

    def tip(self, side: str = "left") -> np.ndarray:
        planner = _planner()
        q = self.session.current_q()
        return planner.kinematics.tip(q, side)[0]

    def test_the_arm_follows_only_while_the_pad_is_held(self) -> None:
        phone = self.phone()
        start = self.tip()
        phone.press()
        self.assertTrue(_wait(lambda: self.arm_state()["engaged"]))
        np.testing.assert_allclose(self.tip(), start, atol=1e-4)  # no jump
        phone.glide([0.06, -0.03, 0.04])
        self.assertTrue(
            _wait(
                lambda: np.linalg.norm(self.tip() - start - [0.06, -0.03, 0.04]) < 0.004
            )
        )
        phone.release()
        self.assertTrue(_wait(lambda: not self.arm_state()["engaged"]))
        held = self.tip()
        phone.position = np.zeros(3)
        time.sleep(0.3)
        np.testing.assert_allclose(self.tip(), held, atol=1e-5)

    def test_scale_multiplies_the_motion(self) -> None:
        phone = self.phone()
        phone.set("teleop.translationScale", 0.5)
        start = self.tip()
        phone.press()
        self.assertTrue(_wait(lambda: self.arm_state()["engaged"]))
        phone.glide([0.08, 0.0, 0.0])
        self.assertTrue(
            _wait(lambda: np.linalg.norm(self.tip() - start - [0.04, 0.0, 0.0]) < 0.003)
        )

    def test_renewals_stopping_releases_the_arm(self) -> None:
        phone = self.phone()
        phone.press()
        self.assertTrue(_wait(lambda: self.arm_state()["engaged"]))
        phone.held = False  # the pad stays down but its renewals stop
        self.assertTrue(_wait(lambda: not self.arm_state()["engaged"], 2.0))
        self.assertIn("connection dropped", self.arm_state()["reason"])

    def test_bad_tracking_or_recalibration_releases_and_needs_a_new_press(self) -> None:
        phone = self.phone()
        phone.press()
        self.assertTrue(_wait(lambda: self.arm_state()["engaged"]))
        phone.tracking = "limited"
        self.assertTrue(_wait(lambda: not self.arm_state()["engaged"]))
        self.assertIn("limited", self.arm_state()["reason"])
        phone.tracking = "normal"
        time.sleep(0.2)
        self.assertFalse(self.arm_state()["engaged"])  # still held, no new press
        phone.release()
        phone.press()
        self.assertTrue(_wait(lambda: self.arm_state()["engaged"]))
        phone.calibration = "guided-v2:2"
        self.assertTrue(_wait(lambda: not self.arm_state()["engaged"]))
        self.assertIn("Tracking restarted", self.arm_state()["reason"])

    def test_stop_on_any_phone_or_the_desktop_holds_everything(self) -> None:
        one = self.phone("phone-1", "left")
        two = self.phone("phone-2", "right")
        one.press()
        self.assertTrue(_wait(lambda: self.arm_state()["engaged"]))
        two.set("safety.estop", True)
        self.assertTrue(_wait(lambda: not self.arm_state()["engaged"]))
        stop = self.session.state()["marionette"]["stop"]
        self.assertEqual((stop["on"], stop["by"]), (True, "Phone 2"))
        self.assertIn("Stopped by Phone 2", self.arm_state()["reason"])
        one.release()
        one.press()
        time.sleep(0.2)
        self.assertFalse(self.arm_state()["engaged"])
        two.set("safety.estop", False)
        self.session.set_stop(True)
        self.assertEqual(self.session.state()["marionette"]["stop"]["by"], "Desktop")
        self.session.set_stop(False)
        self.assertTrue(
            _wait(lambda: not self.session.state()["marionette"]["stop"]["on"])
        )
        time.sleep(0.2)
        self.assertFalse(self.arm_state()["engaged"])  # the pad must be pressed again
        one.release()
        one.press()
        self.assertTrue(_wait(lambda: self.arm_state()["engaged"]))

    def test_two_phones_on_one_arm(self) -> None:
        one = self.phone("phone-1", "left")
        two = self.phone("phone-2", "left")
        self.assertEqual(self.arm_state()["owner"], "phone-1")
        two.press()
        self.assertTrue(_wait(lambda: self.arm_state()["owner"] == "phone-2"))
        self.assertTrue(self.arm_state()["engaged"])
        one.press()
        time.sleep(0.2)
        self.assertEqual(self.arm_state()["owner"], "phone-2")
        feedback = self.session._feedback(
            self.session._tracker.snapshot()["phone-1"],
            self.session._tracker.snapshot(),
            time.monotonic(),
        )
        self.assertEqual(feedback["owner"], "Phone 2")
        self.assertIn("Phone 2 is moving the left arm", feedback["reason"])

    def test_grip_follows_the_owner_but_not_a_stale_request(self) -> None:
        one = self.phone("phone-1", "left")
        one.act("gripper.set", 0.0)
        self.assertTrue(_wait(lambda: self.arm_state()["grip"] == 0.0))
        one.act("gripper.set", 0.7)
        self.assertTrue(_wait(lambda: abs(self.arm_state()["grip"] - 0.7) < 1e-6))
        two = self.phone("phone-2", "right")
        two.act("gripper.set", 0.0)  # made while it had the right arm
        time.sleep(0.1)
        one.set("arm.select", "right")
        two.set("arm.select", "left")
        time.sleep(0.3)
        # Phone 2 has the left arm now; its old request doesn't open it.
        self.assertEqual(self.arm_state()["owner"], "phone-2")
        self.assertAlmostEqual(self.arm_state()["grip"], 0.7)

    def stream_stick(
        self, phone: FakePhone, x: float, y: float, seconds: float
    ) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            phone.act("base.drive", {"x": x, "y": y}, live=True)
            time.sleep(0.05)

    def test_phone_drive_and_stop(self) -> None:
        phone = self.phone()
        self.stream_stick(phone, 1.0, 0.0, 1.0)
        state = self.session.state()
        self.assertGreater(state["base"]["x"], 0.05)
        self.assertTrue(state["marionette"]["phones"][0]["driving"])
        # Let go: the phone stays a live zero source, so the base ramps down
        # under a live hold instead of going dead.
        phone.act("base.drive", {"x": 0, "y": 0})
        time.sleep(0.5)
        state = self.session.state()
        self.assertTrue(state["hold"]["active"])
        self.assertEqual(state["hold"]["drive"], [0.0, 0.0, 0.0])
        # A stick that goes quiet while touched stops everything.
        phone.act("base.drive", {"x": 1, "y": 0}, live=True)
        self.assertTrue(_wait(lambda: self.session.state()["base"]["moving"]))
        self.assertTrue(_wait(lambda: not self.session.state()["hold"]["active"], 1.0))
        phone.act("base.drive", {"x": 0, "y": 0})
        # Driving through a Stop doesn't resume when Stop goes off.
        stopper = threading.Event()

        def keep_driving() -> None:
            while not stopper.wait(0.05):
                phone.act("base.drive", {"x": 1, "y": 0}, live=True)

        driver = threading.Thread(target=keep_driving, daemon=True)
        driver.start()
        try:
            self.assertTrue(
                _wait(lambda: self.session.state()["hold"]["drive"][0] > 0.1)
            )
            self.session.set_stop(True)
            self.assertTrue(_wait(lambda: not self.session.state()["base"]["moving"]))
            time.sleep(0.2)
            self.session.set_stop(False)
            time.sleep(0.5)
            self.assertFalse(self.session.state()["base"]["moving"])
        finally:
            stopper.set()
            driver.join(2)
        phone.act("base.drive", {"x": 0, "y": 0})
        time.sleep(0.05)
        self.stream_stick(phone, 0.0, 1.0, 0.5)
        self.assertTrue(self.session.state()["base"]["moving"])

    def test_turn_and_lift_holds(self) -> None:
        phone = self.phone()
        yaw = self.session.state()["base"]["yaw"]
        phone.act("base.turnLeft", True)
        time.sleep(0.6)
        phone.act("base.turnLeft", False)
        self.assertGreater(self.session.state()["base"]["yaw"], yaw + 0.01)
        lift = self.session.state()["lift"]["raised"]
        phone.act("lift.down", True)
        for _ in range(5):
            time.sleep(0.2)
            phone.act("lift.down", True, renew=True)
        phone.act("lift.down", False)
        self.assertLess(self.session.state()["lift"]["raised"], lift - 0.01)

    def test_changing_the_scale_while_engaged_does_not_move_the_arm(self) -> None:
        phone = self.phone()
        phone.press()
        self.assertTrue(_wait(lambda: self.arm_state()["engaged"]))
        start = self.tip()
        phone.glide([0.08, 0.0, 0.0])
        self.assertTrue(
            _wait(lambda: np.linalg.norm(self.tip() - start - [0.08, 0, 0]) < 0.003)
        )
        time.sleep(0.8)  # settled
        held = self.tip()
        phone.act("teleop.translationScale", 0.25, live=True)
        time.sleep(0.6)
        np.testing.assert_allclose(self.tip(), held, atol=1e-3)
        phone.glide([0.12, 0.0, 0.0])  # 4 cm more at 0.25x
        self.assertTrue(
            _wait(lambda: np.linalg.norm(self.tip() - held - [0.01, 0, 0]) < 0.002)
        )

    def test_a_phone_returning_to_an_arm_does_not_reapply_an_old_grip(self) -> None:
        phone = self.phone()
        phone.act("gripper.set", 1.0)
        self.assertTrue(_wait(lambda: self.arm_state()["grip"] > 0.99))
        phone.set("arm.select", "right")
        self.assertTrue(_wait(lambda: self.arm_state("right")["owner"] == "phone-1"))
        time.sleep(0.05)
        phone.act("gripper.set", 0.0)  # opens the right gripper
        self.assertTrue(_wait(lambda: self.arm_state("right")["grip"] < 0.01))
        phone.set("arm.select", "left")
        time.sleep(0.4)
        self.assertGreater(self.arm_state()["grip"], 0.99)

    def test_stop_freezes_the_gripper(self) -> None:
        phone = self.phone()
        phone.act("gripper.set", 0.0)
        self.assertTrue(_wait(lambda: self.arm_state()["grip"] < 0.01))
        phone.act("gripper.set", 1.0)
        time.sleep(0.1)
        self.session.set_stop(True)
        time.sleep(0.05)
        frozen = self.arm_state()["grip"]
        time.sleep(0.4)
        self.assertAlmostEqual(self.arm_state()["grip"], frozen, places=3)
        self.assertLess(frozen, 0.9)

    def test_a_phone_that_drops_with_stop_on_keeps_it_on(self) -> None:
        one = self.phone("phone-1", "left")
        two = self.phone("phone-2", "right")
        two.set("safety.estop", True)
        two.config(connected=False)
        self.assertTrue(_wait(lambda: self.session.state()["marionette"]["stop"]["on"]))
        self.assertEqual(
            self.session.state()["marionette"]["stop"]["by"], "Phone 2 (offline)"
        )
        one.press()
        time.sleep(0.3)
        self.assertFalse(self.arm_state()["engaged"])
        # The desktop clears it.
        self.session.set_stop(False)
        self.assertFalse(self.session.state()["marionette"]["stop"]["on"])
        one.release()
        one.press()
        self.assertTrue(_wait(lambda: self.arm_state()["engaged"]))

    def test_a_tracking_jump_releases_the_arm(self) -> None:
        phone = self.phone()
        phone.press()
        self.assertTrue(_wait(lambda: self.arm_state()["engaged"]))
        phone.position = np.array([0.0, 0.15, 0.0])  # 15 cm in one frame
        self.assertTrue(_wait(lambda: not self.arm_state()["engaged"], 2.0))
        self.assertIn("jumped", self.arm_state()["reason"])

    def test_feedback_reaches_each_phone(self) -> None:
        self.phone()
        self.assertTrue(_wait(lambda: any(s[0] == "phone-1" for s in self.sent)))
        phone_id, control, values = next(s for s in self.sent if s[0] == "phone-1")[:3]
        self.assertEqual(control, "axol.feedback")
        self.assertEqual(values["mode"], "sim")
        self.assertEqual(values["arm"], "left")
        self.assertTrue(values["hasLift"] and values["hasBase"])
        self.assertTrue(
            all(isinstance(v, (str, float, int, bool)) for v in values.values())
        )

    def left_off_rest(self) -> float:
        from almond_axol.jog.backends import rest_q

        return float(np.max(np.abs(self.session.current_q()[:7] - rest_q()[:7])))

    def move_left_away(self, phone: FakePhone) -> None:
        phone.press()
        self.assertTrue(_wait(lambda: self.arm_state()["engaged"]))
        phone.glide([0.08, 0.06, 0.08])
        time.sleep(0.3)
        phone.release()
        self.assertTrue(_wait(lambda: not self.arm_state()["engaged"]))
        self.assertGreater(self.left_off_rest(), 0.05)

    def notices(self, pid: str) -> list[str]:
        return [s[3] for s in self.sent if s[0] == pid and s[1] == "axol.notice"]

    def test_a_reset_returns_the_arm_to_rest(self) -> None:
        with self.assertRaisesRegex(JogError, "already at rest"):
            self.session.reset_arm("left")
        phone = self.phone()
        self.move_left_away(phone)
        right = self.session.current_q()[7:].copy()
        phone.act("arm.resetLeft")
        self.assertTrue(_wait(lambda: self.arm_state()["resetting"]))
        feedback = self.session._feedback(
            self.session._tracker.snapshot()["phone-1"],
            self.session._tracker.snapshot(),
            time.monotonic(),
        )
        self.assertEqual(feedback["resetting"], "left")
        # Both arms ignore their pads until it ends.
        phone.press()
        time.sleep(0.2)
        self.assertFalse(self.arm_state()["engaged"])
        self.assertTrue(self.arm_state()["resetting"])
        self.assertTrue(_wait(lambda: not self.arm_state()["resetting"], 10))
        self.assertLess(self.left_off_rest(), 1e-4)
        np.testing.assert_allclose(self.session.current_q()[7:], right, atol=1e-6)
        self.assertIn("reach rest", self.arm_state()["reason"])
        phone.release()
        phone.press()
        self.assertTrue(_wait(lambda: self.arm_state()["engaged"]))

    def test_a_reset_waits_for_the_arms_and_stop_halts_it(self) -> None:
        phone = self.phone()
        self.move_left_away(phone)
        phone.press()
        self.assertTrue(_wait(lambda: self.arm_state()["engaged"]))
        phone.act("arm.resetLeft")
        self.assertTrue(
            _wait(
                lambda: any("moving the left arm" in n for n in self.notices("phone-1"))
            )
        )
        with self.assertRaisesRegex(JogError, "Phone 1 is moving the left arm"):
            self.session.reset_arm("left")
        phone.release()
        self.assertTrue(_wait(lambda: not self.arm_state()["engaged"]))
        self.session.reset_arm("left")
        self.assertTrue(_wait(lambda: self.arm_state()["resetting"]))
        with self.assertRaisesRegex(JogError, "returning to rest"):
            self.session.reset_arm("right")
        time.sleep(0.3)
        self.session.set_stop(True)
        self.assertTrue(_wait(lambda: not self.arm_state()["resetting"]))
        held = self.session.current_q().copy()
        self.assertGreater(self.left_off_rest(), 0.01)  # halted on the way
        with self.assertRaisesRegex(JogError, "Stop is on"):
            self.session.reset_arm("left")
        phone.act("arm.resetLeft")
        self.assertTrue(_wait(lambda: "Stopped by Desktop." in self.notices("phone-1")))
        self.session.set_stop(False)
        time.sleep(0.3)
        np.testing.assert_allclose(self.session.current_q(), held, atol=1e-6)

    def test_planned_moves_are_refused(self) -> None:
        with self.assertRaises(JogError):
            self.session.plan("left", [0.3, 0.2, 0.1], [0, 0, 0, 1])
        with self.assertRaises(JogError):
            self.session.preview("left", [0.3, 0.2, 0.1], [0, 0, 0, 1])


def _client() -> Any:
    import tempfile
    from pathlib import Path

    from fastapi.testclient import TestClient

    from almond_axol.serve.app import create_app

    static = Path(tempfile.mkdtemp()) / "dist"
    static.mkdir()
    (static / "index.html").write_text("<html>app</html>")
    return TestClient(create_app(static_dir=static))


class MarionetteApiTest(unittest.TestCase):
    def test_stop_route(self) -> None:
        with _client() as client:
            response = client.post("/api/marionette/stop", json={"on": True})
        self.assertEqual(response.status_code, 409)

        class Stub(MarionetteSession):
            def __init__(self) -> None:  # noqa: D107 - a bare stand-in
                self.calls: list[bool] = []

            def set_stop(self, on: bool) -> dict[str, Any]:
                self.calls.append(on)
                return {"on": on, "by": "Desktop" if on else None}

        stub = Stub()
        with (
            patch("almond_axol.jog.session.active_session", lambda: stub),
            _client() as client,
        ):
            response = client.post("/api/marionette/stop", json={"on": True})
        self.assertEqual(response.json(), {"on": True, "by": "Desktop"})
        self.assertEqual(stub.calls, [True])

    def test_reset_route(self) -> None:
        with _client() as client:
            response = client.post("/api/marionette/reset", json={"arm": "left"})
        self.assertEqual(response.status_code, 409)

        class Stub(MarionetteSession):
            def __init__(self) -> None:  # noqa: D107 - a bare stand-in
                self.calls: list[str] = []

            def reset_arm(self, side: str) -> dict[str, Any]:
                self.calls.append(side)
                if side != "left":
                    raise JogError("The right arm is already at rest.")
                return {"arm": side}

        stub = Stub()
        with (
            patch("almond_axol.jog.session.active_session", lambda: stub),
            _client() as client,
        ):
            ok = client.post("/api/marionette/reset", json={"arm": "left"})
            refused = client.post("/api/marionette/reset", json={"arm": "right"})
        self.assertEqual(ok.json(), {"arm": "left"})
        self.assertEqual(refused.status_code, 409)
        self.assertEqual(refused.json(), {"error": "The right arm is already at rest."})
        self.assertEqual(stub.calls, ["left", "right"])

    def test_camera_socket_sends_watched_frames(self) -> None:
        hub = CameraHub(SyntheticCameraSource(), fps=30, width=160, quality=50)
        hub.start()

        class Stub(MarionetteSession):
            def __init__(self) -> None:  # noqa: D107 - a bare stand-in
                self.cameras = hub

        stub = Stub()
        try:
            with (
                patch("almond_axol.jog.session.active_session", lambda: stub),
                _client() as client,
                client.websocket_connect(
                    "/api/marionette/cameras?streams=right&fps=30"
                ) as ws,
            ):
                if True:
                    info = ws.receive_json()
                    self.assertEqual(info["type"], "cameras")
                    self.assertEqual(info["available"], ["head", "left", "right"])
                    frame = ws.receive_bytes()
            stream, jpeg = frame.split(b"\n", 1)
            self.assertEqual(stream, b"right")
            self.assertEqual(jpeg[:2], b"\xff\xd8")
        finally:
            hub.stop()


if __name__ == "__main__":
    unittest.main()
