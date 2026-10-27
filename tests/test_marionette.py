"""Marionette teleoperation: Axol's meanings, the arm's filter and reach,
the cameras, and sim sessions against the real bundled gateway driven by
scripted phones (``marionette.testing``)."""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
import unittest
from typing import Any
from unittest.mock import patch

import numpy as np
from marionette import Phone, RobotLink
from marionette.phones import Pose
from marionette.teleop import camera_frame, exp_rotation, rotation_vector
from marionette.testing import Gateway, SimPhone, msg

from almond_axol.jog.backends import SimBackend
from almond_axol.jog.session import JogError, active_session
from almond_axol.marionette import phones as meaning
from almond_axol.marionette.cameras import (
    CameraHub,
    SyntheticCameraSource,
    encode_jpeg,
    streams_for_view,
)
from almond_axol.marionette.session import (
    DUPLICATE,
    SCREEN,
    MarionetteSession,
)
from almond_axol.marionette.teleop import ArmTeleop, TeleopLimits, gate_reason

# A camera looking along Axol's left (+y), upright: the image's right is
# Axol's forward and its down is Axol's down (ROS optical axes as columns).
_LOOKING_LEFT = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
# The screen's layout, as the gateway relays it.
_CONTROLS = json.loads(
    SCREEN.read_text().split('id="marionette-config">', 1)[1].split("</script>", 1)[0]
)["controls"]


def _pose(
    position: Any = (0.0, 0.0, 0.0),
    rotvec: Any = (0.0, 0.0, 0.0),
    *,
    tracking: str = "normal",
    source: str = "arkit",
    calibration: str = "guided-v2:1",
    session: str = "s1",
    received: float | None = None,
    seq: int = 0,
) -> Pose:
    rotation = exp_rotation(np.asarray(rotvec, float))
    return Pose(
        position=np.asarray(position, float),
        rotation=rotation,
        quaternion=np.zeros(4),
        tracking=tracking,
        source=source,
        heartbeat=True,
        calibration=calibration,
        session=session,
        seq=seq,
        received=time.monotonic() if received is None else received,
        stamp=seq / 60.0,
    )


def _phone(pid: str = "phone-1", arm: str = "left", **pose: Any) -> Phone:
    phone = Phone(pid, pid.replace("phone-", "Phone "), connected=True)
    phone.calibration = "captured"
    phone.values["arm.select"] = arm
    phone.pose = _pose(**pose)
    return phone


class MeaningsTest(unittest.TestCase):
    """What Axol reads from a phone, folded by Marionette's tracker."""

    def setUp(self) -> None:
        self.now = 100.0
        self.tracker = meaning.PhoneTracker(clock=lambda: self.now, mirror="allow")
        self.tracker.apply(
            msg.config(
                ["phone-1"], "phone-1", _CONTROLS, {"status": "captured", "epoch": 2}
            )
        )

    def phone(self) -> Phone:
        phone = self.tracker.get("phone-1")
        assert phone is not None
        return phone

    def input(self, action: str, value: Any = None, **extra: Any) -> None:
        self.tracker.apply(msg.input("phone-1", action, value, **extra))

    def test_values_and_their_defaults(self) -> None:
        phone = self.phone()
        self.assertEqual(
            (meaning.arm(phone), meaning.frame(phone), meaning.view(phone)),
            ("left", "axol", "head"),
        )
        self.assertEqual((meaning.scale(phone), meaning.speed(phone)), (1.0, 0.5))
        self.input("arm.select", "right")
        self.input("teleop.frame", "left")
        self.input("teleop.frame", "sideways")  # not a frame: ignored
        self.input("base.speedLimit", 80, live=True)
        self.input("teleop.translationScale", 0.5, live=True)
        phone = self.phone()
        self.assertEqual((meaning.arm(phone), meaning.frame(phone)), ("right", "left"))
        self.assertAlmostEqual(meaning.speed(phone), 0.8)
        self.assertAlmostEqual(meaning.scale(phone), 0.5)

    def test_each_grip_request_counts_but_a_resting_finger_does_not(self) -> None:
        self.assertEqual(meaning.grip(self.phone()), (0, 0.0))
        self.input("gripper.set", 0.4, live=True)
        self.input("gripper.set", 0.4, live=True, renew=True)  # a resting finger
        self.input("gripper.set", 0.6)  # the slide ends
        self.assertEqual(meaning.grip(self.phone()), (2, 0.6))
        self.input("gripper.set", 1.0)  # Close gripper, twice
        self.input("gripper.set", 1.0)
        self.input("gripper.set", math.inf)  # malformed: ignored
        self.assertEqual(meaning.grip(self.phone()), (4, 1.0))

    def test_drive_stick_turn_and_lift(self) -> None:
        def command(at: float) -> Any:
            return meaning.base_command(self.phone(), at, 0.25, 0.6)

        self.assertEqual(command(self.now), (0.0, 0.0, 0.0, 0))
        self.input("base.drive", {"x": 0.6, "y": -0.2}, live=True)
        self.input("base.turnLeft", True)
        self.input("lift.down", True)
        self.assertEqual(command(self.now), (0.6, -0.2, 1.0, -1))
        # A stick sample older than its timeout is a dead source.
        self.assertIsNone(command(self.now + 0.3))
        self.now += 0.2
        self.input("base.drive", {"x": 0.6, "y": -0.2}, live=True, renew=True)
        self.input("base.turnLeft", True, renew=True)
        self.input("lift.down", True, renew=True)
        self.assertIsNotNone(command(self.now + 0.2))
        # A hold whose repeats stopped is a dead source too.
        self.assertIsNone(command(self.now + 0.7))
        # Let go of everything: zeros, and nothing goes stale.
        self.input("base.drive", {"x": 0, "y": 0})
        self.input("base.turnLeft", False)
        self.input("lift.down", False)
        self.assertEqual(command(self.now + 5), (0.0, 0.0, 0.0, 0))
        # A drag the gateway ended stays ended until the finger lifts.
        self.input("base.drive", {"x": 1, "y": 0}, live=True)
        self.tracker.apply(msg.let_go("phone-1", "base.drive", {"x": 0, "y": 0}))
        self.input("base.drive", {"x": 1, "y": 0}, live=True)
        self.assertEqual(command(self.now)[:2], (0.0, 0.0))

    def test_mirror_inputs_move_nothing_on_the_robot(self) -> None:
        robot = meaning.tracker(allow_mirror=False)
        robot.apply(msg.config(["phone-1"], "phone-1", _CONTROLS))
        for action, value in (
            ("teleop.deadman", True),
            ("base.drive", {"x": 1, "y": 0}),
            ("gripper.set", 1.0),
            ("base.turnLeft", True),
            ("arm.resetLeft", None),
        ):
            robot.apply(msg.input("phone-1", action, value, live=True, source="mirror"))
        phone = robot.get("phone-1")
        assert phone is not None
        self.assertFalse(phone.hold("teleop.deadman").down)
        self.assertEqual(meaning.grip(phone)[0], 0)
        self.assertEqual(
            meaning.base_command(phone, time.monotonic(), 0.25, 0.6),
            (0.0, 0.0, 0.0, 0),
        )
        self.assertEqual(phone.presses("arm.resetLeft"), 0)
        robot.apply(msg.input("phone-1", "safety.estop", True, source="mirror"))
        self.assertTrue(robot.get("phone-1").stop)  # type: ignore[union-attr]
        # In sim a mirror acts as its phone.
        self.assertEqual(meaning.tracker(allow_mirror=True).mirror, "allow")


class GateTest(unittest.TestCase):
    limits = TeleopLimits()

    def reason(self, phone: Phone, **options: Any) -> Any:
        options.setdefault("allow_synthetic", False)
        return gate_reason(phone, time.monotonic(), self.limits, **options)

    def test_good_tracking_passes(self) -> None:
        self.assertIsNone(self.reason(_phone()))

    def test_every_failure_has_a_reason(self) -> None:
        offline = _phone()
        offline.connected = False
        calibrating = _phone()
        calibrating.calibration = "capturing"
        cases = {
            "offline": offline,
            "calibration": calibrating,
            "No pose": _phone(received=time.monotonic() - 1.0),
            "limited": _phone(tracking="limited"),
            "starting": _phone(tracking="initializing"),
            "Simulated motion": _phone(source="synthetic"),
        }
        for word, phone in cases.items():
            reason = self.reason(phone)
            self.assertIsNotNone(reason, word)
            self.assertIn(word, reason)

    def test_the_robot_needs_a_calibrated_phone(self) -> None:
        phone = _phone()
        phone.calibration = "uncalibrated"
        self.assertEqual(
            self.reason(phone, require_calibration=True),
            "Calibrate this phone first, facing the way Axol faces.",
        )
        phone.calibration = "capturing"
        self.assertIn("Finish", self.reason(phone, require_calibration=True))
        phone.calibration = "captured"
        self.assertIsNone(self.reason(phone, require_calibration=True))

    def test_synthetic_motion_is_allowed_in_sim(self) -> None:
        phone = _phone(source="synthetic", tracking="limited")
        self.assertIsNone(self.reason(phone, allow_synthetic=True))


class ArmTeleopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = TeleopLimits(max_speed=0.5, max_turn_rate=3.0, pose_cutoff=50.0)
        self.arm = ArmTeleop("left", self.limits, 125.0)
        self.tip = (np.array([0.3, 0.2, 0.1]), np.eye(3))

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
        p, r = self.arm.target(phone, time.monotonic(), 0.01)
        self.assertLessEqual(np.linalg.norm(p - self.tip[0]), 0.5 * 0.01 + 1e-9)
        self.assertLessEqual(np.linalg.norm(rotation_vector(r)), 3.0 * 0.01 + 1e-9)

    def test_the_motion_scale_scales_moves_and_turns(self) -> None:
        phone = _phone()
        phone.values["teleop.translationScale"] = 0.5
        self.arm.engage(phone, self.tip)
        yaws = []
        for degrees in range(0, 300, 5):
            pose = _pose([0.1, 0.0, 0.0], [0.0, 0.0, math.radians(degrees)])
            p, r = self.arm.clutch.target(pose)
            yaws.append(math.degrees(math.atan2(r[1, 0], r[0, 0])))
        np.testing.assert_allclose(p, self.tip[0] + [0.05, 0.0, 0.0], atol=1e-12)
        # A half-scale turn past half a turn never jumps.
        np.testing.assert_allclose(np.diff(yaws), 2.5, atol=1e-6)
        self.assertAlmostEqual(yaws[-1], 147.5, places=6)

    def test_a_new_scale_or_frame_does_not_move_the_target(self) -> None:
        phone = _phone()
        self.arm.engage(phone, self.tip)
        phone.pose = _pose([0.05, 0.02, 0.0], [0.0, 0.0, 0.3])
        before = self.arm.clutch.target(phone.pose)
        phone.values["teleop.translationScale"] = 0.5
        phone.values["teleop.frame"] = "left"
        self.arm.reframe(phone, camera_frame(_LOOKING_LEFT))
        self.arm.target(phone, time.monotonic(), 1 / 125)  # takes up the scale
        self.assertEqual((self.arm.frame, self.arm.clutch.scale), ("left", 0.5))
        after = self.arm.clutch.target(phone.pose)
        np.testing.assert_allclose(after[0], before[0], atol=1e-12)
        np.testing.assert_allclose(after[1], before[1], atol=1e-12)
        # From here the phone's forward moves the tip left, at half scale.
        moved = self.arm.clutch.target(_pose([0.15, 0.02, 0.0], [0.0, 0.0, 0.3]))
        np.testing.assert_allclose(moved[0] - before[0], [0.0, 0.05, 0.0], atol=1e-12)

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

    def test_jumps_use_axols_limits(self) -> None:
        jumps = self.arm.jumps
        self.assertEqual(
            (jumps.distance, jumps.angle, jumps.speed, jumps.turn_speed),
            (0.04, 0.5, 4.0, 20.0),
        )


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

    def test_a_large_view_gets_its_own_width(self) -> None:
        import cv2

        hub = CameraHub(
            SyntheticCameraSource(), fps=30, width=160, large_width=320, quality=50
        )
        hub.start()
        try:
            hub.want("phones", ["head", "left"])
            hub.want("page-large", ["head"], hub.large_width)
            self.assertEqual(hub.wanted(), {160: {"head", "left"}, 320: {"head"}})

            def width(frame: Any) -> int:
                image = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), 1)
                return image.shape[1]

            self.assertTrue(_wait(lambda: hub.latest("head", 320) is not None))
            self.assertTrue(_wait(lambda: hub.latest("left") is not None))
            self.assertEqual(width(hub.latest("head", 320)), 320)
            self.assertEqual(width(hub.latest("head")), 160)
            self.assertIsNone(hub.latest("left", 320))
        finally:
            hub.stop()

    def test_only_watched_streams_are_grabbed_and_jpeg_only_when_wanted(self) -> None:
        hub = CameraHub(SyntheticCameraSource(), fps=30, width=160, quality=50)
        hub.start()
        try:
            hub.want("video", ["left", "bogus"], jpeg=False)
            self.assertTrue(_wait(lambda: hub.latest_rgb("left") is not None))
            frame = hub.latest_rgb("left")
            self.assertEqual((frame.rgb.shape, frame.jpeg), ((90, 160, 3), None))
            self.assertIsNone(hub.latest("left"))  # nobody wants it as JPEG
            self.assertIsNone(hub.latest_rgb("head"))
            hub.want("page", ["left"])
            self.assertTrue(_wait(lambda: hub.latest("left") is not None))
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
        frames = source.grab(["head", "left"], 320)
        self.assertEqual(set(frames), {"head", "left"})
        self.assertEqual(frames["head"].shape, (200, 320, 3))
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

            def grab(self, streams: Any, width: int) -> dict[str, np.ndarray]:
                Once.calls += 1
                return (
                    {"head": np.zeros((9, 16, 3), np.uint8)} if Once.calls == 1 else {}
                )

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


def _screen(phone: SimPhone) -> dict[str, Any]:
    """What Axol's screen reads: Axol's state with this phone's over it."""
    states = [m for m in phone.robot_states if m.get("robotId") == "axol"]
    shared = next((m["state"] for m in reversed(states) if "phoneId" not in m), {})
    own = next(
        (m["state"] for m in reversed(states) if m.get("phoneId") == phone.phone_id),
        {},
    )
    return {**shared, **own}


def _notices(phone: SimPhone) -> list[str]:
    return [
        s.get("message", "")
        for s in phone.statuses
        if s.get("controlId") == "robot.notice"
    ]


class MarionetteSessionTest(unittest.TestCase):
    """A sim session against the real gateway, driven by scripted phones."""

    def setUp(self) -> None:
        self.phones: list[SimPhone] = []
        self.gateway: Gateway | None = None
        self.thread: threading.Thread | None = None
        self.stop = threading.Event()

    def start(self, cameras: CameraHub | None = None, **gateway: Any) -> None:
        self.gateway = Gateway(**gateway).start()
        self.backend = SimBackend("mobile")
        planner = _planner()
        self.session = MarionetteSession(
            self.backend,
            lambda: planner,
            gateway=self.gateway.address,
            cameras=cameras,
        )
        self.thread = threading.Thread(
            target=lambda: asyncio.run(self.session.run(self.stop)), daemon=True
        )
        self.thread.start()
        self.assertTrue(_wait(lambda: active_session() is self.session))
        self.assertTrue(_wait(lambda: self.session.state()["ready"], 30))
        self.assertTrue(_wait(lambda: self.gateway_state()["connected"]))

    def tearDown(self) -> None:
        for phone in self.phones:
            phone.close()
        self.stop.set()
        if self.thread is not None:
            self.thread.join(15)
            self.assertIsNone(active_session())
        if self.gateway is not None:
            self.gateway.close()

    def phone(self, arm: str = "left") -> SimPhone:
        """A phone held level, uncalibrated (its forward is Axol's), on ``arm``."""
        phone = SimPhone(self.gateway, heading=0).connect()
        self.phones.append(phone)
        if arm != "left":
            phone.set("arm.select", arm)
        self.assertTrue(
            _wait(
                lambda: any(
                    p["id"] == phone.phone_id
                    and p["arm"] == arm
                    and p["tracking"] == "normal"
                    for p in self.session.state()["marionette"]["phones"]
                )
            )
        )
        time.sleep(0.05)  # the arms have seen the phone (and its press count)
        return phone

    def gateway_state(self) -> dict[str, Any]:
        return self.session.state()["marionette"]["gateway"]

    def arm_state(self, side: str = "left") -> dict[str, Any]:
        return self.session.state()["marionette"]["arms"][side]

    def tip(self, side: str = "left") -> np.ndarray:
        q = self.session.current_q()
        return _planner().kinematics.tip(q, side)[0]

    def press(self, phone: SimPhone) -> None:
        phone.press("teleop.deadman")

    def release(self, phone: SimPhone) -> None:
        phone.release("teleop.deadman")

    def engage(self, phone: SimPhone) -> None:
        self.press(phone)
        self.assertTrue(_wait(lambda: self.arm_state()["engaged"]))

    def test_the_screen_comes_from_axol_and_state_has_real_types(self) -> None:
        self.start()
        phone = self.phone()
        self.assertTrue(_wait(lambda: phone.html is not None))
        html = phone.html
        self.assertEqual(html["source"], {"kind": "robot", "robotId": "axol"})
        self.assertEqual(html["title"], "Axol")
        self.assertEqual(html["orientation"], "portrait")
        self.assertIn("window.MarionetteKit", html["html"])  # the kit, inlined
        self.assertNotIn('<script src="marionette-kit.js"></script>', html["html"])
        robots = {r["id"]: r for r in (phone.config or {}).get("robots", [])}
        self.assertEqual(
            (robots["axol"]["name"], robots["axol"]["connected"]), ("Axol", True)
        )
        self.assertTrue(
            _wait(lambda: "arm" in _screen(phone) and "frames" in _screen(phone))
        )
        state = _screen(phone)
        self.assertEqual(
            (state["mode"], state["arm"], state["state"]), ("sim", "left", "idle")
        )
        self.assertEqual(state["frames"], ["axol", "head", "left", "right"])
        self.assertEqual(state["streams"], [])  # this session has no cameras
        self.assertEqual(set(state["lift"]), {"raised", "moving"})
        self.assertEqual(state["base"], {"moving": False, "fault": False})
        # The simulated Jelly has no battery: unknown, not a number.
        self.assertIsNone(state["battery"])
        self.assertIsNone(state["owner"])
        self.assertIsNone(state["reason"])
        every = {m.get("everyMs") for m in phone.robot_states}
        self.assertEqual(every, {200})
        self.backend.battery_state = lambda: {  # type: ignore[method-assign]
            "percent": 62.6,
            "charging": False,
            "underLoad": True,
        }
        self.assertTrue(
            _wait(
                lambda: _screen(phone)["battery"]
                == {"percent": 63, "charging": False, "underLoad": True}
            )
        )
        # A reading that isn't a number is unknown; the state keeps flowing.
        self.backend.battery_state = lambda: {  # type: ignore[method-assign]
            "percent": math.nan,
            "charging": False,
            "underLoad": False,
        }
        self.assertTrue(_wait(lambda: _screen(phone)["battery"] is None))

    def test_the_arm_follows_only_while_the_pad_is_held(self) -> None:
        self.start()
        phone = self.phone()
        start = self.tip()
        self.engage(phone)
        np.testing.assert_allclose(self.tip(), start, atol=1e-4)  # no jump
        self.assertTrue(_wait(lambda: _screen(phone)["engaged"] is True))
        phone.move_by((0.06, -0.03, 0.04), seconds=0.5)
        self.assertTrue(
            _wait(
                lambda: np.linalg.norm(self.tip() - start - [0.06, -0.03, 0.04]) < 0.004
            )
        )
        self.release(phone)
        self.assertTrue(_wait(lambda: not self.arm_state()["engaged"]))
        held = self.tip()
        phone.move_to((0.0, 0.0, 0.0), seconds=0.3)
        time.sleep(0.3)
        np.testing.assert_allclose(self.tip(), held, atol=1e-5)

    def test_scale_multiplies_the_motion(self) -> None:
        self.start()
        phone = self.phone()
        phone.set("teleop.translationScale", 0.5)
        self.assertTrue(
            _wait(
                lambda: self.session.state()["marionette"]["phones"][0]["scale"] == 0.5
            )
        )
        start = self.tip()
        self.engage(phone)
        phone.move_by((0.08, 0.0, 0.0), seconds=0.5)
        self.assertTrue(
            _wait(lambda: np.linalg.norm(self.tip() - start - [0.04, 0.0, 0.0]) < 0.003)
        )

    def test_renewals_stopping_releases_the_arm(self) -> None:
        # The gateway waits longer than Axol here, so Axol's own check acts.
        self.start(env={"MARIONETTE_HOLD_TIMEOUT_MS": "5000"})
        phone = self.phone()
        self.engage(phone)
        phone.pause_renewals()  # the pad stays down but its renewals stop
        self.assertTrue(_wait(lambda: not self.arm_state()["engaged"], 2.0))
        self.assertIn("connection dropped", self.arm_state()["reason"])
        self.assertTrue(
            _wait(lambda: "connection dropped" in (_screen(phone)["reason"] or ""))
        )

    def test_bad_tracking_or_recalibration_releases_and_needs_a_new_press(self) -> None:
        self.start()
        phone = self.phone()
        self.engage(phone)
        phone.tracking("limited")
        self.assertTrue(_wait(lambda: not self.arm_state()["engaged"]))
        self.assertIn("limited", self.arm_state()["reason"])
        phone.tracking("normal")
        time.sleep(0.2)
        self.assertFalse(self.arm_state()["engaged"])  # still held, no new press
        self.release(phone)
        self.engage(phone)
        phone.quick_origin()
        self.assertTrue(_wait(lambda: not self.arm_state()["engaged"]))
        self.assertIn("Tracking restarted", self.arm_state()["reason"])

    def test_stop_on_any_phone_or_the_desktop_holds_everything(self) -> None:
        self.start()
        one = self.phone("left")
        two = self.phone("right")
        self.engage(one)
        two.set("safety.estop", True)
        self.assertTrue(_wait(lambda: not self.arm_state()["engaged"]))
        stop = self.session.state()["marionette"]["stop"]
        self.assertEqual((stop["on"], stop["by"]), (True, "Phone 2"))
        self.assertIn("Stopped by Phone 2", self.arm_state()["reason"])
        self.assertTrue(_wait(lambda: _screen(one).get("stopBy") == "Phone 2"))
        self.release(one)
        self.press(one)
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
        self.release(one)
        self.engage(one)

    def test_two_phones_on_one_arm(self) -> None:
        self.start()
        one = self.phone("left")
        two = self.phone("left")
        self.assertEqual(self.arm_state()["owner"], one.phone_id)
        self.press(two)
        self.assertTrue(_wait(lambda: self.arm_state()["owner"] == two.phone_id))
        self.assertTrue(self.arm_state()["engaged"])
        self.press(one)
        time.sleep(0.2)
        self.assertEqual(self.arm_state()["owner"], two.phone_id)
        self.assertTrue(_wait(lambda: _screen(one)["owner"] == "Phone 2"))
        self.assertIn("Phone 2 is moving the left arm", _screen(one)["reason"])
        self.assertEqual(_screen(one)["state"], "blocked")
        self.assertIsNone(_screen(two)["owner"])

    def test_grip_follows_the_owner_but_not_a_stale_request(self) -> None:
        self.start()
        one = self.phone("left")
        one.set("gripper.set", 0.0)
        self.assertTrue(_wait(lambda: self.arm_state()["grip"] == 0.0))
        one.drag("gripper.set", [0.3, 0.5, 0.7])
        self.assertTrue(_wait(lambda: abs(self.arm_state()["grip"] - 0.7) < 1e-6))
        self.assertTrue(_wait(lambda: abs(_screen(one)["grip"] - 0.7) < 1e-6))
        two = self.phone("right")
        two.set("gripper.set", 0.0)  # made while it had the right arm
        time.sleep(0.1)
        one.set("arm.select", "right")
        two.set("arm.select", "left")
        time.sleep(0.3)
        # Phone 2 has the left arm now; its old request doesn't open it.
        self.assertEqual(self.arm_state()["owner"], two.phone_id)
        self.assertAlmostEqual(self.arm_state()["grip"], 0.7)

    def test_phone_drive_and_stop(self) -> None:
        self.start()
        phone = self.phone()
        phone.stick(1.0, 0.0)
        time.sleep(1.0)
        state = self.session.state()
        self.assertGreater(state["base"]["x"], 0.05)
        self.assertTrue(state["marionette"]["phones"][0]["driving"])
        self.assertTrue(_wait(lambda: _screen(phone)["driving"] is True))
        self.assertEqual(_screen(phone)["driver"], "Phone 1")
        # Let go: the phone stays a live zero source, so the base ramps down
        # under a live hold instead of going dead.
        phone.let_go("base.drive")
        time.sleep(0.5)
        state = self.session.state()
        self.assertTrue(state["hold"]["active"])
        self.assertEqual(state["hold"]["drive"], [0.0, 0.0, 0.0])
        # A stick that goes quiet while touched stops everything.
        phone.stick(1.0, 0.0)
        self.assertTrue(_wait(lambda: self.session.state()["base"]["moving"]))
        with phone.renewals_paused():
            self.assertTrue(
                _wait(lambda: not self.session.state()["hold"]["active"], 1.0)
            )
            phone.let_go("base.drive")
        # Driving through a Stop doesn't resume when Stop goes off.
        phone.stick(1.0, 0.0)
        self.assertTrue(_wait(lambda: self.session.state()["hold"]["drive"][0] > 0.1))
        self.session.set_stop(True)
        self.assertTrue(_wait(lambda: not self.session.state()["base"]["moving"]))
        time.sleep(0.2)
        self.session.set_stop(False)
        time.sleep(0.5)
        self.assertFalse(self.session.state()["base"]["moving"])
        phone.let_go("base.drive")
        time.sleep(0.2)  # the let-go reaches Axol before the next push
        phone.stick(0.0, 1.0)
        time.sleep(0.5)
        self.assertTrue(self.session.state()["base"]["moving"])
        phone.let_go("base.drive")

    def test_turn_and_lift_holds(self) -> None:
        self.start()
        phone = self.phone()
        yaw = self.session.state()["base"]["yaw"]
        phone.hold("base.turnLeft", 0.6)
        self.assertGreater(self.session.state()["base"]["yaw"], yaw + 0.01)
        lift = self.session.state()["lift"]["raised"]
        phone.hold("lift.down", 1.0)
        self.assertLess(self.session.state()["lift"]["raised"], lift - 0.01)

    def test_changing_the_scale_while_engaged_does_not_move_the_arm(self) -> None:
        self.start()
        phone = self.phone()
        self.engage(phone)
        start = self.tip()
        phone.move_by((0.08, 0.0, 0.0), seconds=0.5)
        self.assertTrue(
            _wait(lambda: np.linalg.norm(self.tip() - start - [0.08, 0, 0]) < 0.003)
        )
        time.sleep(0.8)  # settled
        held = self.tip()
        phone.drag("teleop.translationScale", [0.5, 0.25])
        time.sleep(0.6)
        np.testing.assert_allclose(self.tip(), held, atol=1e-3)
        phone.move_by((0.04, 0.0, 0.0), seconds=0.5)  # 4 cm more at 0.25x
        self.assertTrue(
            _wait(lambda: np.linalg.norm(self.tip() - held - [0.01, 0, 0]) < 0.002)
        )

    def test_in_a_wrist_camera_frame_the_phone_moves_along_its_view(self) -> None:
        self.start()
        phone = self.phone()
        phone.set("teleop.frame", "left")
        kin = _planner().kinematics
        view = kin.camera(self.session.current_q(), "left")[1][:, 2]
        self.assertGreater(abs(view[2]), 0.5)  # hanging at rest, it looks down
        self.assertTrue(
            _wait(
                lambda: self.session.state()["marionette"]["phones"][0]["frame"]
                == "left"
            )
        )
        start = self.tip()
        self.engage(phone)
        phone.move_by((-0.05, 0.0, 0.0), seconds=0.5)  # back, toward the camera
        self.assertTrue(
            _wait(lambda: np.linalg.norm(self.tip() - start + 0.05 * view) < 0.003)
        )
        time.sleep(0.5)  # settled
        held = self.tip()
        # Back to Axol's frame mid-move: nothing moves until the phone does,
        # then up is Axol's up.
        phone.set("teleop.frame", "axol")
        time.sleep(0.5)
        np.testing.assert_allclose(self.tip(), held, atol=1e-3)
        phone.move_by((0.0, 0.0, 0.04), seconds=0.5)
        self.assertTrue(
            _wait(lambda: np.linalg.norm(self.tip() - held - [0, 0, 0.04]) < 0.003)
        )
        state = self.session.state()["marionette"]["phones"][0]
        self.assertEqual(state["frame"], "axol")

    def test_a_phone_returning_to_an_arm_does_not_reapply_an_old_grip(self) -> None:
        self.start()
        phone = self.phone()
        phone.set("gripper.set", 1.0)
        self.assertTrue(_wait(lambda: self.arm_state()["grip"] > 0.99))
        phone.set("arm.select", "right")
        self.assertTrue(
            _wait(lambda: self.arm_state("right")["owner"] == phone.phone_id)
        )
        time.sleep(0.05)
        phone.set("gripper.set", 0.0)  # opens the right gripper
        self.assertTrue(_wait(lambda: self.arm_state("right")["grip"] < 0.01))
        phone.set("arm.select", "left")
        time.sleep(0.4)
        self.assertGreater(self.arm_state()["grip"], 0.99)

    def test_stop_freezes_the_gripper(self) -> None:
        self.start()
        phone = self.phone()
        phone.set("gripper.set", 0.0)
        self.assertTrue(_wait(lambda: self.arm_state()["grip"] < 0.01))
        phone.set("gripper.set", 1.0)
        time.sleep(0.1)
        self.session.set_stop(True)
        time.sleep(0.05)
        frozen = self.arm_state()["grip"]
        time.sleep(0.4)
        self.assertAlmostEqual(self.arm_state()["grip"], frozen, places=3)
        self.assertLess(frozen, 0.9)

    def test_a_phone_that_drops_with_stop_on_keeps_it_on(self) -> None:
        self.start()
        one = self.phone("left")
        two = self.phone("right")
        two.set("safety.estop", True)
        two.disconnect()
        self.assertTrue(
            _wait(
                lambda: self.session.state()["marionette"]["stop"]["by"]
                == "Phone 2 (offline)"
            )
        )
        self.press(one)
        time.sleep(0.3)
        self.assertFalse(self.arm_state()["engaged"])
        # The desktop clears it.
        self.session.set_stop(False)
        self.assertFalse(self.session.state()["marionette"]["stop"]["on"])
        self.release(one)
        self.engage(one)

    def test_a_stop_survives_a_restarted_gateway(self) -> None:
        self.start()
        self.phone("left")
        two = self.phone("right")
        two.set("safety.estop", True)
        self.assertTrue(_wait(lambda: self.session.state()["marionette"]["stop"]["on"]))
        for phone in self.phones:
            phone.close()
        self.phones.clear()
        assert self.gateway is not None
        self.gateway.restart()
        self.assertTrue(_wait(lambda: self.gateway_state()["connected"], 10))
        one = self.phone("left")
        stop = self.session.state()["marionette"]["stop"]
        self.assertEqual((stop["on"], stop["by"]), (True, "Phone 2 (offline)"))
        self.press(one)
        time.sleep(0.3)
        self.assertFalse(self.arm_state()["engaged"])
        self.session.set_stop(False)
        self.release(one)
        self.engage(one)

    def test_a_tracking_jump_releases_the_arm(self) -> None:
        self.start()
        phone = self.phone()
        self.engage(phone)
        phone.move_to((0.0, 0.15, 0.0))  # 15 cm in one frame
        self.assertTrue(_wait(lambda: not self.arm_state()["engaged"], 2.0))
        self.assertIn("jumped", self.arm_state()["reason"])

    def left_off_rest(self) -> float:
        from almond_axol.jog.backends import rest_q

        return float(np.max(np.abs(self.session.current_q()[:7] - rest_q()[:7])))

    def move_left_away(self, phone: SimPhone) -> None:
        self.engage(phone)
        phone.move_by((0.08, 0.06, 0.08), seconds=0.5)
        time.sleep(0.3)
        self.release(phone)
        self.assertTrue(_wait(lambda: not self.arm_state()["engaged"]))
        self.assertGreater(self.left_off_rest(), 0.05)

    def test_a_reset_returns_the_arm_to_rest(self) -> None:
        self.start()
        with self.assertRaisesRegex(JogError, "already at rest"):
            self.session.reset_arm("left")
        phone = self.phone()
        self.move_left_away(phone)
        right = self.session.current_q()[7:].copy()
        phone.press("arm.resetLeft")
        self.assertTrue(_wait(lambda: self.arm_state()["resetting"]))
        self.assertTrue(_wait(lambda: _screen(phone)["resetting"] == "left"))
        # Both arms ignore their pads until it ends.
        self.press(phone)
        time.sleep(0.2)
        self.assertFalse(self.arm_state()["engaged"])
        self.assertTrue(self.arm_state()["resetting"])
        self.assertTrue(_wait(lambda: not self.arm_state()["resetting"], 10))
        self.assertLess(self.left_off_rest(), 1e-4)
        np.testing.assert_allclose(self.session.current_q()[7:], right, atol=1e-6)
        self.assertIn("reach rest", self.arm_state()["reason"])
        self.assertTrue(_wait(lambda: _screen(phone)["resetting"] is None))
        self.release(phone)
        self.engage(phone)

    def test_a_reset_waits_for_the_arms_and_stop_halts_it(self) -> None:
        self.start()
        phone = self.phone()
        self.move_left_away(phone)
        self.engage(phone)
        phone.press("arm.resetLeft")
        self.assertTrue(
            _wait(lambda: any("moving the left arm" in n for n in _notices(phone)))
        )
        with self.assertRaisesRegex(JogError, "Phone 1 is moving the left arm"):
            self.session.reset_arm("left")
        self.release(phone)
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
        phone.press("arm.resetLeft")
        self.assertTrue(_wait(lambda: "Stopped by Desktop." in _notices(phone)))
        self.session.set_stop(False)
        time.sleep(0.3)
        np.testing.assert_allclose(self.session.current_q(), held, atol=1e-6)

    def test_a_second_axol_on_the_gateway_is_refused(self) -> None:
        self.start()
        assert self.gateway is not None
        planner = _planner()
        other = MarionetteSession(
            SimBackend("mobile"), lambda: planner, gateway=self.gateway.address
        )
        other._link.start()
        try:
            self.assertTrue(_wait(lambda: other._gateway_error() == DUPLICATE))
            self.assertTrue(self.gateway_state()["connected"])
            # This one stops: the other takes over.
            self.stop.set()
            assert self.thread is not None
            self.thread.join(15)
            self.thread = None
            self.assertTrue(_wait(lambda: other._link.connected, 15))
            self.assertIsNone(other._gateway_error())
        finally:
            other._link.stop()

    def test_cameras_go_to_the_phones_over_webrtc(self) -> None:
        hub = CameraHub(SyntheticCameraSource(), fps=15, width=320, quality=50)
        self.start(cameras=hub)
        phone = self.phone()
        self.assertTrue(
            _wait(lambda: _screen(phone).get("streams") == ["head", "left", "right"])
        )
        self.assertTrue(
            _wait(
                lambda: all(
                    s.get("transport") == "webrtc" and s.get("publisher") == "axol"
                    for s in (phone.config or {}).get("cameraStreams", [])
                )
            )
        )
        assert self.gateway is not None
        frame = asyncio.run(_receive_video(self.gateway.url, "head"))
        self.assertEqual((frame.width, frame.height), (320, 180))
        # No JPEG through the gateway while WebRTC works.
        self.assertFalse(any(m.get("type") == "camera_frame" for m in phone.messages))

    def test_cameras_fall_back_to_jpeg_without_webrtc(self) -> None:
        hub = CameraHub(SyntheticCameraSource(), fps=15, width=160, quality=50)
        with patch.dict("sys.modules", {"marionette.video": None}):
            self.start(cameras=hub)
            phone = self.phone()
            self.assertTrue(
                _wait(
                    lambda: any(
                        m.get("type") == "camera_frame" and m.get("cameraId") == "head"
                        for m in phone.messages
                    )
                )
            )
        self.assertIsNotNone(self.session._video_error)

    def test_planned_moves_are_refused(self) -> None:
        self.start()
        with self.assertRaises(JogError):
            self.session.plan("left", [0.3, 0.2, 0.1], [0, 0, 0, 1])
        with self.assertRaises(JogError):
            self.session.preview("left", [0.3, 0.2, 0.1], [0, 0, 0, 1])


async def _receive_video(url: str, stream: str) -> Any:
    """Ask the robot for a stream as a desktop viewer does, and return a frame."""
    from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
    from websockets.asyncio.client import connect

    async with connect(url) as ws:
        await ws.send(json.dumps(msg.hello("console", cameras=False)))
        await ws.send(json.dumps(msg.rtc("request", stream=stream, peer="test-1")))
        offer: dict[str, Any] = {}
        async for raw in ws:
            message = json.loads(raw)
            if message.get("type") == "rtc" and message.get("kind") == "offer":
                offer = message
                break
        receiver = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        frame: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

        @receiver.on("track")
        def on_track(track: Any) -> None:
            async def pull() -> None:
                for _ in range(3):  # the encoder's first frames
                    await track.recv()
                frame.set_result(await track.recv())

            asyncio.ensure_future(pull())

        await receiver.setRemoteDescription(
            RTCSessionDescription(offer["sdp"], "offer")
        )
        await receiver.setLocalDescription(await receiver.createAnswer())
        await ws.send(
            json.dumps(
                msg.rtc(
                    "answer",
                    to=offer["from"],
                    peer="test-1",
                    sdp=receiver.localDescription.sdp,
                )
            )
        )
        try:
            return await asyncio.wait_for(frame, 20)
        finally:
            await receiver.close()


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
                info = ws.receive_json()
                self.assertEqual(info["type"], "cameras")
                self.assertEqual(info["available"], ["head", "left", "right"])
                frame = ws.receive_bytes()
            stream, jpeg = frame.split(b"\n", 1)
            self.assertEqual(stream, b"right")
            self.assertEqual(jpeg[:2], b"\xff\xd8")
        finally:
            hub.stop()

    def test_camera_socket_sends_the_large_stream_wider(self) -> None:
        import cv2

        hub = CameraHub(
            SyntheticCameraSource(), fps=30, width=160, large_width=320, quality=50
        )
        hub.start()

        class Stub(MarionetteSession):
            def __init__(self) -> None:  # noqa: D107 - a bare stand-in
                self.cameras = hub

        stub = Stub()

        def widths(ws: Any, count: int) -> dict[str, int]:
            seen: dict[str, int] = {}
            while len(seen) < count:
                message = ws.receive()
                if message.get("bytes"):
                    stream, jpeg = message["bytes"].split(b"\n", 1)
                    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), 1)
                    seen[stream.decode()] = image.shape[1]
            return seen

        try:
            with (
                patch("almond_axol.jog.session.active_session", lambda: stub),
                _client() as client,
                client.websocket_connect(
                    "/api/marionette/cameras?streams=head,left&large=head&fps=30"
                ) as ws,
            ):
                self.assertEqual(widths(ws, 2), {"head": 320, "left": 160})
                ws.send_json({"large": "left"})
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if widths(ws, 2) == {"head": 160, "left": 320}:
                        break
                else:
                    self.fail("the large stream did not change")
        finally:
            hub.stop()


class RobotLinkTest(unittest.TestCase):
    """The gateway takes Axol's screen as it ships."""

    def test_the_screen_is_accepted(self) -> None:
        with Gateway() as gateway:
            link = RobotLink(gateway.url, "axol", "Axol", screen=SCREEN).start()
            try:
                self.assertTrue(link.wait_connected(5))
                self.assertTrue(link.wait_screen(5), link.screen_error)
                self.assertEqual(link.screen_warnings, [])
            finally:
                link.stop()


if __name__ == "__main__":
    unittest.main()
