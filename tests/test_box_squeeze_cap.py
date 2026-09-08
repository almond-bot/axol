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
