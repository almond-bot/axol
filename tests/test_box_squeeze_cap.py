"""Box mode's squeeze cap: a bounded shoulder spring torque while clamping.

Closing the grip width onto a box drives the IK targets into it, and an
impedance joint then presses with ``kp`` times the run-ahead. Box mode caps
the spring torque of the two shoulder joints that carry a lateral force at
the gripper (``BOX_SQUEEZE_JOINTS``) at ``VRTeleopConfig.box_squeeze_torque``
whenever it is on and the arms are not returning to rest. The cap rides
every command to the realtime core as the tenth field of each slot tuple,
where the core clamps the wire position to within ``cap / kp`` of measured.

Covered here: the core's decision (``VRTeleopCore.spring_caps``), the arm's
command tuples (``AxolArm.set_spring_caps`` / ``motion_control``), the wire
packing and the ``proto`` handshake line, the live setting, and the teleop
loop applying the caps to the robot on change and clearing them at exit.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from almond_axol.constants import ARM_JOINTS, RT_PROTO_VERSION, RT_TARGET_FIELDS, Joint
from almond_axol.robot.axol import Axol
from almond_axol.robot.config import AxolConfig
from almond_axol.rt import link as rt_link
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import BOX_SQUEEZE_JOINTS, VRTeleopCore
from almond_axol.teleop.live import LiveSettings
from almond_axol.teleop.teleop import VRTeleop


def _core(**overrides) -> VRTeleopCore:
    return VRTeleopCore(
        VRTeleopConfig(**overrides),
        logging.getLogger("test"),
        broadcast_tracking=lambda _enabled: None,
    )


class CoreDecisionTest(unittest.TestCase):
    def test_squeeze_joints_are_the_lateral_shoulders(self) -> None:
        # The abduction axis carries ~0.45-0.68 Nm per N of squeeze and the
        # upper-arm twist ~0.1-0.3; the elbow and shoulder_1 lift the box.
        self.assertEqual(BOX_SQUEEZE_JOINTS, (Joint.SHOULDER_2, Joint.SHOULDER_3))

    def test_no_caps_outside_box_mode(self) -> None:
        core = _core()
        self.assertFalse(core.box_mode)
        self.assertIsNone(core.spring_caps())

    def test_caps_whenever_box_mode_is_on(self) -> None:
        core = _core(box_mode=True)
        # Leading, frozen or between engages alike: the pair may be
        # clamping a box in any of them.
        self.assertEqual(
            core.spring_caps(), {Joint.SHOULDER_2: 4.0, Joint.SHOULDER_3: 4.0}
        )
        self.assertFalse(core.teleop_enabled)

    def test_default_is_four_newton_metres(self) -> None:
        self.assertEqual(VRTeleopConfig().box_squeeze_torque, 4.0)

    def test_zero_disables(self) -> None:
        core = _core(box_mode=True, box_squeeze_torque=0.0)
        self.assertIsNone(core.spring_caps())

    def test_a_return_to_rest_lifts_the_cap(self) -> None:
        core = _core(box_mode=True)
        core.request_reset()
        self.assertTrue(core.is_resetting)
        self.assertIsNone(core.spring_caps())

    def test_live_setting_changes_the_cap(self) -> None:
        core = _core(box_mode=True)
        core.set_live("box_squeeze_torque", 7)
        core._apply_live_requests()
        self.assertEqual(core.config.box_squeeze_torque, 7.0)
        self.assertEqual(core.spring_caps()[Joint.SHOULDER_2], 7.0)
        core.set_live("box_mode", False)
        core._apply_live_requests()
        self.assertIsNone(core.spring_caps())


class ArmCommandTest(unittest.TestCase):
    """The cap rides the tenth field of each arm slot's command tuple."""

    @classmethod
    def setUpClass(cls) -> None:
        # Offline: buses and motors are constructed, nothing is opened.
        cls.robot = Axol(AxolConfig())

    def _arm_with_sink(self):
        arm = self.robot.left
        sent: list[list[tuple[float, ...]]] = []
        arm._command_sink = sent.append
        arm.resolve_joint_offsets = AsyncMock()
        arm._joint_offsets = np.zeros(8, dtype=np.float32)
        arm._last_q_commanded = None
        return arm, sent

    def test_tuples_carry_the_caps(self) -> None:
        arm, sent = self._arm_with_sink()
        arm.set_spring_caps({Joint.SHOULDER_2: 4.0, Joint.SHOULDER_3: 4.0})
        asyncio.run(arm.motion_control(np.zeros(8, dtype=np.float32)))
        cmds = sent[-1]
        self.assertEqual([len(c) for c in cmds], [RT_TARGET_FIELDS] * 8)
        caps = {j: cmds[i][-1] for i, j in enumerate(ARM_JOINTS)}
        self.assertEqual(caps[Joint.SHOULDER_2], 4.0)
        self.assertEqual(caps[Joint.SHOULDER_3], 4.0)
        for j in (
            Joint.SHOULDER_1,
            Joint.ELBOW,
            Joint.WRIST_1,
            Joint.WRIST_2,
            Joint.WRIST_3,
        ):
            self.assertEqual(caps[j], 0.0, j)  # 0 = configured cap only
        self.assertEqual(cmds[7][3:], (0.0,) * (RT_TARGET_FIELDS - 3))  # gripper pad
        # Everything before the cap is untouched: tracked mode, config kp.
        self.assertEqual(cmds[1][1], 1.0)
        self.assertEqual(cmds[1][2], arm._arm_config.shoulder_2.kp)

    def test_clearing_returns_to_configured_caps_only(self) -> None:
        arm, sent = self._arm_with_sink()
        arm.set_spring_caps({Joint.SHOULDER_2: 4.0})
        arm.set_spring_caps(None)
        self.assertEqual(arm.spring_caps, {})
        asyncio.run(arm.motion_control(np.zeros(8, dtype=np.float32)))
        self.assertEqual([c[-1] for c in sent[-1]], [0.0] * 8)

    def test_non_positive_and_gripper_caps_are_rejected(self) -> None:
        arm, _ = self._arm_with_sink()
        arm.set_spring_caps({Joint.SHOULDER_2: 0.0, Joint.SHOULDER_3: float("inf")})
        self.assertEqual(arm.spring_caps, {})  # "no cap", not "pinned"
        with self.assertRaises(ValueError):
            arm.set_spring_caps({Joint.GRIPPER: 1.0})

    def test_gravity_comp_tuples_are_the_same_width(self) -> None:
        arm, sent = self._arm_with_sink()
        for m in arm.motors.values():
            m._position = 0.0
        # Offsets "resolved" for this offline read of the position caches.
        arm._unverified_zeros = set()
        arm._unresolved_offsets = set()
        arm.set_spring_caps({Joint.SHOULDER_2: 4.0})
        asyncio.run(arm.gravity_compensate(kd=0.5))
        cmds = sent[-1]
        self.assertEqual([len(c) for c in cmds], [RT_TARGET_FIELDS] * 8)
        # Passthrough mode carries no per-command cap (free joints have no
        # spring; held ones sit on their snapshot at config gains).
        self.assertEqual([c[-1] for c in cmds], [0.0] * 8)

    def _arm_at(self, measured: np.ndarray):
        """A sink-mode arm whose feedback caches read ``measured`` (joint frame)."""
        arm, sent = self._arm_with_sink()
        for i, j in enumerate(ARM_JOINTS):
            arm.motors[j]._position = float(measured[i])
        arm._unverified_zeros = set()
        arm._unresolved_offsets = set()
        return arm, sent

    def test_whole_arm_backs_off_so_a_capped_joint_stays_within_its_cap(self) -> None:
        """The pair is jogged into a box: the target runs ahead of measured on
        every joint, most on shoulder_2. The command is pulled back along the
        joint-space line toward measured — the same fraction on every joint —
        until shoulder_2's spring is at its cap. The gripper is untouched."""
        measured = np.array([0.1, -0.2, 0.05, 0.4, 0.0, 0.1, -0.1, 0.0], np.float32)
        arm, sent = self._arm_at(measured)
        arm.set_spring_caps({Joint.SHOULDER_2: 4.0, Joint.SHOULDER_3: 4.0})
        kp_s2 = arm._arm_config.shoulder_2.kp
        # 6 Nm of shoulder_2 spring asked for; the elbow and wrist_2 run ahead too.
        run_ahead = np.array([0.0, -6.0 / kp_s2, 0.002, 0.03, 0.0, -0.02, 0.0, 0.0])
        target = measured + run_ahead.astype(np.float32)
        target[7] = 0.7  # gripper, normalized
        asyncio.run(arm.motion_control(target))
        cmds = sent[-1]
        sent_arm = np.array([c[0] for c in cmds[:7]])  # motor frame == joint frame here
        expected_scale = 4.0 / 6.0
        np.testing.assert_allclose(
            sent_arm, measured[:7] + expected_scale * run_ahead[:7], atol=1e-6
        )
        # shoulder_2 sits exactly at cap / kp from measured; shoulder_3 (also
        # capped, but within it) is scaled by the same factor as everyone.
        self.assertAlmostEqual(abs(sent_arm[1] - measured[1]) * kp_s2, 4.0, places=5)
        self.assertAlmostEqual(cmds[7][0], arm._gripper_to_raw(0.7), places=6)

    def test_within_the_caps_nothing_changes(self) -> None:
        measured = np.array([0.1, -0.2, 0.05, 0.4, 0.0, 0.1, -0.1, 0.0], np.float32)
        arm, sent = self._arm_at(measured)
        arm.set_spring_caps({Joint.SHOULDER_2: 4.0, Joint.SHOULDER_3: 4.0})
        kp_s2 = arm._arm_config.shoulder_2.kp
        target = measured.copy()
        target[1] += 3.9 / kp_s2  # under the cap
        target[3] += 0.2  # a big elbow run-ahead — not capped, not scaled
        asyncio.run(arm.motion_control(target))
        sent_arm = np.array([c[0] for c in sent[-1][:7]])
        np.testing.assert_allclose(sent_arm, target[:7], atol=1e-6)

    def test_no_caps_or_no_feedback_means_no_back_off(self) -> None:
        measured = np.zeros(8, np.float32)
        arm, sent = self._arm_at(measured)
        target = measured.copy()
        target[1] = 0.1  # 25 Nm of shoulder_2 spring, uncapped
        arm.set_spring_caps(None)
        asyncio.run(arm.motion_control(target))
        self.assertAlmostEqual(sent[-1][1][0], 0.1, places=6)
        # Caps on, but the encoder zeros aren't resolved yet: pass through.
        arm.set_spring_caps({Joint.SHOULDER_2: 4.0})
        arm._unresolved_offsets = {Joint.SHOULDER_2}
        asyncio.run(arm.motion_control(target))
        self.assertAlmostEqual(sent[-1][1][0], 0.1, places=6)
        arm._unresolved_offsets = set()

    def test_robot_level_setter_reaches_both_arms(self) -> None:
        self.robot.set_spring_caps({Joint.SHOULDER_2: 3.0})
        self.assertEqual(self.robot.left.spring_caps, {Joint.SHOULDER_2: 3.0})
        self.assertEqual(self.robot.right.spring_caps, {Joint.SHOULDER_2: 3.0})
        self.robot.set_spring_caps(None)
        self.assertEqual(self.robot.right.spring_caps, {})

    def test_spring_torques_read_the_capped_command_over_measured(self) -> None:
        """kp times (sent command - measured), so a capped joint reads its cap."""
        measured = np.array([0.1, -0.2, 0.05, 0.4, 0.0, 0.1, -0.1, 0.0], np.float32)
        arm, _ = self._arm_at(measured)
        arm._last_q_commanded = None
        self.assertIsNone(arm.spring_torques())  # nothing sent yet
        arm.set_spring_caps({Joint.SHOULDER_2: 4.0, Joint.SHOULDER_3: 4.0})
        kp_s2 = arm._arm_config.shoulder_2.kp
        kp_el = arm._arm_config.elbow.kp
        target = measured.copy()
        target[1] -= 6.0 / kp_s2  # asks for 6 Nm: backed off to the 4 Nm cap
        target[3] += 0.03
        asyncio.run(arm.motion_control(target))
        tau = arm.spring_torques()
        self.assertEqual(tau.shape, (7,))
        self.assertAlmostEqual(
            float(tau[1]), -4.0, places=4
        )  # signed toward the command
        self.assertAlmostEqual(float(tau[3]), 0.03 * (4.0 / 6.0) * kp_el, places=3)
        self.assertAlmostEqual(float(tau[0]), 0.0, places=6)
        arm._unresolved_offsets = {Joint.SHOULDER_2}
        self.assertIsNone(arm.spring_torques())  # no trusted feedback
        arm._unresolved_offsets = set()


class SqueezeLeanTest(unittest.TestCase):
    """As the squeeze builds the fingertips lean in, about the contact face.

    The arm's compliance yaws a squeezing hand outward, lifting the parcel
    gripper's tip off the box; the worker adds ``box_squeeze_tilt`` times the
    fraction of the cap in use as extra inward yaw per gripper.
    """

    def test_core_reads_the_fraction_off_shoulder_2(self) -> None:
        core = _core(box_mode=True)  # 4 Nm cap
        i = ARM_JOINTS.index(Joint.SHOULDER_2)
        left = np.zeros(7, np.float32)
        right = np.zeros(7, np.float32)
        left[i] = -2.0  # sign is the arm's; only the magnitude squeezes
        right[i] = 1.0
        left[0] = 9.0  # shoulder_1 (lifting the box) doesn't count
        self.assertEqual(core.squeeze_fraction((left, right)), (0.5, 0.25))
        right[i] = 40.0  # more than the cap can't be: the back-off keeps it at 1
        self.assertEqual(core.squeeze_fraction((left, right)), (0.5, 1.0))
        self.assertIsNone(core.squeeze_fraction((left, None)))
        core.set_live("box_squeeze_torque", 0.0)
        core._apply_live_requests()
        self.assertIsNone(core.squeeze_fraction((left, right)))
        self.assertIsNone(_core().squeeze_fraction((left, right)))  # not box mode

    def _worker(self, lean_deg: float = 2.0):
        from almond_axol.teleop.worker import IKWorker

        worker = object.__new__(IKWorker)
        worker._config = SimpleNamespace(box_squeeze_tilt=lean_deg, ik_frequency=120.0)
        worker._squeeze = None
        return worker

    def _box(self):
        from almond_axol.teleop.box import BoxState

        return BoxState(
            center=np.array((0.4, 0.0, 0.3), np.float32),
            rot=np.eye(3, dtype=np.float32),
            width=0.3,
            face={"left": 1.0, "right": 1.0},
            tilt=0.0,
            align_start={},
            align_t0=0.0,
            align_duration=0.0,
        )

    def test_worker_leans_each_side_by_its_own_squeeze(self) -> None:
        worker = self._worker(lean_deg=2.0)
        box = self._box()
        worker._squeeze_tilt(box)  # no report yet (the sim): nothing
        self.assertEqual(box.squeeze_tilt, {"left": 0.0, "right": 0.0})
        worker.note_squeeze(1.0, 0.5)
        for _ in range(400):  # > 3 s at 120 Hz: the low-pass has settled
            worker._squeeze_tilt(box)
        self.assertAlmostEqual(box.squeeze_tilt["left"], np.radians(2.0), places=5)
        self.assertAlmostEqual(box.squeeze_tilt["right"], np.radians(1.0), places=5)
        # Filtered: one report doesn't move it all the way.
        worker.note_squeeze(0.0, 0.5)
        worker._squeeze_tilt(box)
        self.assertGreater(box.squeeze_tilt["left"], 0.5 * np.radians(2.0))
        # Out-of-range reports are clipped to the cap fraction.
        worker.note_squeeze(7.0, -1.0)
        self.assertEqual(worker._squeeze, {"left": 1.0, "right": 0.0})
        # Turned off live: the lean is dropped at once.
        worker._config.box_squeeze_tilt = 0.0
        worker._squeeze_tilt(box)
        self.assertEqual(box.squeeze_tilt, {"left": 0.0, "right": 0.0})

    def test_lean_moves_the_tip_in_and_leaves_the_face_where_it_was(self) -> None:
        from almond_axol.teleop.box import ideal_gripper_poses, parcel_tool

        tool = parcel_tool(141.5)
        box = self._box()
        box.tool = tool
        feet = {side: tool.foot(1.0) for side in ("left", "right")}

        def poses():
            return ideal_gripper_poses(
                box.center, box.rot, box.width, box.grip_rel(), feet
            )

        flat = poses()
        box.squeeze_tilt = {"left": np.radians(1.5), "right": 0.0}
        leaned = poses()
        # Fixed blade tip in the mount frame (see tests/test_box.py).
        tip = np.array((-0.0335, 0.0, -0.1385), np.float32)
        p0, r0 = flat["left"]
        p1, r1 = leaned["left"]
        np.testing.assert_allclose(
            p0 + r0 @ feet["left"], p1 + r1 @ feet["left"], atol=1e-6
        )
        tip0 = p0 + r0 @ tip
        tip1 = p1 + r1 @ tip
        # Left gripper: inward is -y. 1.5° over 13.8 cm ≈ 3.6 mm.
        self.assertLess(float(tip1[1]), float(tip0[1]) - 0.003)
        self.assertAlmostEqual(float(tip1[2]), float(tip0[2]), places=6)  # level
        # The other gripper is untouched by this side's squeeze.
        np.testing.assert_allclose(leaned["right"][0], flat["right"][0], atol=1e-7)
        np.testing.assert_allclose(leaned["right"][1], flat["right"][1], atol=1e-7)

    def test_loop_forwards_the_squeeze_ahead_of_each_frame_in_box_mode(self) -> None:
        from almond_axol.vr.models import VRFrame, VRPose, VRPosition, VRQuaternion

        core = _core(box_mode=True, ik_frequency=2000.0)
        core.q = np.zeros(16, np.float32)
        identity = VRQuaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        zero = VRPosition(x=0.0, y=0.0, z=0.0)
        frame = VRFrame(
            l_ee=VRPose(position=zero, quaternion=identity),
            r_ee=VRPose(position=zero, quaternion=identity),
            l_elbow=zero,
            r_elbow=zero,
            l_lock=False,
            r_lock=False,
        )
        sent: list[object] = []
        stop = threading.Event()

        class Conn:
            def send(self, msg):
                sent.append(msg)

            def poll(self, _t):
                return True

            def recv(self):
                if len(sent) >= 6:
                    stop.set()  # after this reply the loop sees the stop
                return (np.zeros(16, np.float32), None)

        readings = iter([(0.25, 0.0), None, (1.0, 0.5)])

        def get_squeeze():
            return next(readings, (1.0, 0.5))

        # The loop dispatches a frame once per object: hand it a fresh copy.
        core.run_ik_loop(
            Conn(), frame.model_copy, stop, lambda: True, lambda _t: None, get_squeeze
        )
        squeezes = [m for m in sent if isinstance(m, tuple) and m[0] == "squeeze"]
        frames = [m for m in sent if not isinstance(m, tuple)]
        self.assertGreaterEqual(len(frames), 3)
        self.assertEqual(squeezes[0], ("squeeze", 0.25, 0.0))
        # A None reading sends nothing for that frame; the next one resumes.
        self.assertEqual(squeezes[1], ("squeeze", 1.0, 0.5))
        self.assertEqual(len(squeezes), len(frames) - 1)
        # Each squeeze immediately precedes its frame.
        for i, m in enumerate(sent[:-1]):
            if isinstance(m, tuple) and m[0] == "squeeze":
                self.assertNotIsInstance(sent[i + 1], tuple)

    def test_loop_sends_nothing_outside_box_mode_or_without_a_hook(self) -> None:
        from almond_axol.vr.models import VRFrame, VRPose, VRPosition, VRQuaternion

        identity = VRQuaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        zero = VRPosition(x=0.0, y=0.0, z=0.0)
        frame = VRFrame(
            l_ee=VRPose(position=zero, quaternion=identity),
            r_ee=VRPose(position=zero, quaternion=identity),
            l_elbow=zero,
            r_elbow=zero,
            l_lock=False,
            r_lock=False,
        )
        for box_mode, hook in ((False, lambda: (1.0, 1.0)), (True, None)):
            core = _core(box_mode=box_mode, ik_frequency=2000.0)
            core.q = np.zeros(16, np.float32)
            sent: list[object] = []
            stop = threading.Event()

            class Conn:
                def send(self, msg):
                    sent.append(msg)

                def poll(self, _t):
                    return True

                def recv(self):
                    if len(sent) >= 3:
                        stop.set()
                    return (np.zeros(16, np.float32), None)

            core.run_ik_loop(
                Conn(), frame.model_copy, stop, lambda: True, lambda _t: None, hook
            )
            self.assertFalse(
                any(isinstance(m, tuple) and m[0] == "squeeze" for m in sent)
            )

    def test_live_setting_is_published_with_the_cap(self) -> None:
        core = _core(box_mode=True)
        hardware = LiveSettings(
            core, SimpleNamespace(set_spring_caps=lambda c: None), lambda s: None
        )
        sim = LiveSettings(core, object(), lambda s: None)
        self.assertIn("box_squeeze_tilt", {d["key"] for d in hardware.schema()})
        self.assertEqual(hardware.values()["box_squeeze_tilt"], 1.0)
        self.assertNotIn("box_squeeze_tilt", {d["key"] for d in sim.schema()})
        hardware.apply("box_squeeze_tilt", 2.5)
        core._apply_live_requests()
        self.assertEqual(core.config.box_squeeze_tilt, 2.5)
        # It's a worker field: forwarded to the IK subprocess as a "set".
        self.assertIn(("box_squeeze_tilt", 2.5), core._worker_updates)

    def test_adapters_read_the_arms_spring_torques(self) -> None:
        i = ARM_JOINTS.index(Joint.SHOULDER_2)
        left = np.zeros(7, np.float32)
        right = np.zeros(7, np.float32)
        left[i] = 2.0
        right[i] = -3.0
        robot = SimpleNamespace(
            left=SimpleNamespace(spring_torques=lambda: left),
            right=SimpleNamespace(spring_torques=lambda: right),
        )
        core = _core(box_mode=True)
        teleop = object.__new__(VRTeleop)
        teleop._core = core
        teleop._robot = robot
        self.assertEqual(teleop._squeeze_fraction(), (0.5, 0.75))
        teleop._robot = object()  # the sim: no such reading
        self.assertIsNone(teleop._squeeze_fraction())

        from almond_axol.lerobot.teleop.teleop_vr import AxolVRTeleop

        lerobot = object.__new__(AxolVRTeleop)
        lerobot._core = core
        lerobot._live = LiveSettings(core, None, lambda s: None)
        self.assertIsNone(lerobot._squeeze_fraction())  # robot not attached yet
        lerobot._live.set_robot(robot)
        self.assertEqual(lerobot._squeeze_fraction(), (0.5, 0.75))


class WireTest(unittest.TestCase):
    def test_target_packet_is_ten_doubles_per_slot(self) -> None:
        self.assertEqual(RT_TARGET_FIELDS, 10)
        link = object.__new__(rt_link.RtLink)
        payloads: list[bytes] = []
        link._send = payloads.append
        cmds = [tuple(float(s * 10 + f) for f in range(10)) for s in range(8)]
        link.send_target(1, 7, cmds)
        (payload,) = payloads
        self.assertEqual(len(payload), 1 + 1 + 4 + 8 * 10 * 8)
        self.assertEqual(payload[:1], b"T")
        side, seq = struct.unpack("<BI", payload[1:6])
        self.assertEqual((side, seq), (1, 7))
        slot2 = struct.unpack("<10d", payload[6 + 2 * 80 : 6 + 3 * 80])
        self.assertEqual(slot2, cmds[2])
        self.assertEqual(slot2[9], 29.0)  # the tau_cap field

    def test_config_declares_the_protocol_first(self) -> None:
        from almond_axol.rt.mantis import RtMantis
        from almond_axol.rt.robot import RtAxol

        self.assertEqual(RT_PROTO_VERSION, 2)
        for cls, robot in (
            (
                RtAxol,
                SimpleNamespace(
                    left=SimpleNamespace(
                        _config=SimpleNamespace(max_step_rad=0.35),
                        _arm_config=AxolConfig().left,
                        _has_gripper=False,
                    ),
                    right=None,
                    _left_bus=SimpleNamespace(_channel="canL"),
                    _right_bus=None,
                ),
            ),
            (RtMantis, SimpleNamespace(left=None, right=None)),
        ):
            rt = object.__new__(cls)
            rt._robot = robot
            rt._loop_hz = 240.0
            rt._watchdog_ms = 150.0
            rt._max_vel = 1.0
            rt._max_accel = 1.0
            first = rt._config_text().splitlines()[0]
            self.assertEqual(first, f"proto {RT_PROTO_VERSION}", cls.__name__)


class LiveSettingTest(unittest.TestCase):
    def test_published_on_hardware_only(self) -> None:
        core = _core(box_mode=True)
        hardware = LiveSettings(
            core, SimpleNamespace(set_spring_caps=lambda c: None), lambda s: None
        )
        sim = LiveSettings(core, object(), lambda s: None)
        self.assertIn("box_squeeze_torque", {d["key"] for d in hardware.schema()})
        self.assertEqual(hardware.values()["box_squeeze_torque"], 4.0)
        self.assertNotIn("box_squeeze_torque", {d["key"] for d in sim.schema()})
        self.assertNotIn("box_squeeze_torque", sim.values())
        with self.assertRaises(ValueError):
            sim.apply("box_squeeze_torque", 3.0)

    def test_apply_routes_to_the_core(self) -> None:
        core = _core(box_mode=True)
        live = LiveSettings(
            core, SimpleNamespace(set_spring_caps=lambda c: None), lambda s: None
        )
        live.apply("box_squeeze_torque", 2.5)
        core._apply_live_requests()
        self.assertEqual(core.spring_caps()[Joint.SHOULDER_3], 2.5)
        with self.assertRaises(ValueError):
            live.apply("box_squeeze_torque", -1)


class LeRobotTeleoperatorTest(unittest.TestCase):
    """``collect-data`` reads the same decision through ``AxolVRTeleop``."""

    def test_spring_caps_passes_through_to_the_core(self) -> None:
        from almond_axol.lerobot.teleop.teleop_vr import AxolVRTeleop

        teleop = object.__new__(AxolVRTeleop)
        teleop._core = _core(box_mode=True)
        self.assertEqual(
            teleop.spring_caps(), {Joint.SHOULDER_2: 4.0, Joint.SHOULDER_3: 4.0}
        )
        teleop._core.request_reset()
        self.assertIsNone(teleop.spring_caps())


class _FakeRobot:
    """Hardware-shaped target: takes the caps, records every command."""

    def __init__(self) -> None:
        self.caps: list[dict | None] = []
        self.commands = 0
        self.left = None
        self.right = None

    def set_spring_caps(self, caps):
        self.caps.append(caps)

    async def motion_control(self, left=None, right=None):
        self.commands += 1

    async def gravity_compensate(self, kd=0.5):
        pass

    def torque_residuals(self):
        return None, None

    def reset_command_state(self):
        pass


class TeleopLoopTest(unittest.TestCase):
    """The loop hands the core's caps to the robot on change, and clears them at exit."""

    def _teleop(self, core: VRTeleopCore, robot: _FakeRobot) -> VRTeleop:
        teleop = object.__new__(VRTeleop)
        teleop._config = core.config
        teleop._core = core
        teleop._robot = robot
        teleop._ik_process = None
        teleop._ik_thread = None
        teleop._video_manager = None
        teleop._robot_recorder = None
        teleop._rec = None
        teleop._ik_loop_times = []
        teleop._ik_loop_times_lock = threading.Lock()
        teleop._vr_frame_times = []
        teleop._vr_frame_times_lock = threading.Lock()
        teleop.step = lambda: (np.zeros(8, np.float32), np.zeros(8, np.float32))
        return teleop

    def test_caps_follow_box_mode_and_clear_at_exit(self) -> None:
        core = _core(frequency=500.0)
        robot = _FakeRobot()
        teleop = self._teleop(core, robot)

        async def scenario() -> None:
            with (
                patch("almond_axol.teleop.teleop.SystemDiag") as diag,
                patch("almond_axol.teleop.teleop.TegraStatsDiag") as tegra,
                patch("almond_axol.teleop.teleop.TeleopActivityMarker") as marker,
            ):
                diag.return_value = Mock()
                tegra.return_value = Mock()
                marker.return_value = Mock()
                task = asyncio.create_task(teleop.run())
                while robot.commands < 3:
                    await asyncio.sleep(0.005)
                # Plain teleop: nothing has been sent to the robot's caps.
                self.assertEqual(robot.caps, [])
                core.set_live("box_mode", True)
                core._apply_live_requests()
                n = robot.commands
                while robot.commands < n + 3:
                    await asyncio.sleep(0.005)
                self.assertEqual(
                    robot.caps, [{Joint.SHOULDER_2: 4.0, Joint.SHOULDER_3: 4.0}]
                )
                core.set_live("box_squeeze_torque", 6.0)
                core._apply_live_requests()
                n = robot.commands
                while robot.commands < n + 3:
                    await asyncio.sleep(0.005)
                self.assertEqual(
                    robot.caps[-1], {Joint.SHOULDER_2: 6.0, Joint.SHOULDER_3: 6.0}
                )
                self.assertEqual(len(robot.caps), 2)  # written on change only
                core.set_live("box_mode", False)
                core._apply_live_requests()
                n = robot.commands
                while robot.commands < n + 3:
                    await asyncio.sleep(0.005)
                self.assertEqual(robot.caps[-1], None)
                self.assertEqual(len(robot.caps), 3)
                # Back on, then the session ends: the exit clears them.
                core.set_live("box_mode", True)
                core._apply_live_requests()
                n = robot.commands
                while robot.commands < n + 3:
                    await asyncio.sleep(0.005)
                self.assertEqual(len(robot.caps), 4)
                task.cancel()
                await task
                self.assertEqual(robot.caps[-1], None)
                self.assertEqual(len(robot.caps), 5)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
