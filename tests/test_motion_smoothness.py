"""Guards for the motion-smoothness fixes.

Each test pins a mechanism the audit found turning a rare event into a
torque step or a phantom velocity — the kind of thing that shows up as an
arm hitch and never as a failing call.
"""

from __future__ import annotations

import logging
import math
import threading
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from almond_axol.motor import myactuator
from almond_axol.motor.errors import MotorError
from almond_axol.robot import control
from almond_axol.robot.config import FrictionParams, JointConfig, _calibrated_joint
from almond_axol.utils.control_loop import MAX_PACING_DEBT_INTERVALS, rebase_deadline
from almond_axol.vr.interp import PoseInterpolator
from almond_axol.vr.models import VRFrame, VRPose, VRPosition, VRQuaternion


class PacingDebtTest(unittest.TestCase):
    """A stalled absolute-deadline loop drops its backlog instead of bursting."""

    def test_small_slips_keep_the_absolute_schedule(self) -> None:
        interval = 1 / 120
        deadline = 100.0
        # Up to the cap, the schedule is kept so ordinary late wakes are still
        # corrected on the next cycle rather than accumulating as drift.
        for late in (0.0, 0.3 * interval, 1.0 * interval, 1.99 * interval):
            self.assertEqual(
                rebase_deadline(deadline, deadline + late, interval), deadline
            )
        # Never moves the deadline into the future.
        self.assertEqual(
            rebase_deadline(deadline, deadline - interval, interval), deadline
        )

    def test_a_stall_reanchors_to_now(self) -> None:
        interval = 1 / 120
        deadline = 100.0
        for late in (2.01 * interval, 5 * interval, 0.5):
            now = deadline + late
            self.assertEqual(rebase_deadline(deadline, now, interval), now)
        self.assertEqual(MAX_PACING_DEBT_INTERVALS, 2.0)


class BandPassStabilityTest(unittest.TestCase):
    """The classic-mode band-pass must never amplify its state across a stall."""

    def test_coefficient_clamp_is_inside_the_stability_limit(self) -> None:
        # One-step matrix [[1, f], [-f, 1 - f² - f/q]] is stable iff
        # f < -1/q + sqrt(1/q² + 4). The old fixed clamp 2·sin(0.7) = 1.288
        # sat past that limit for q = 0.8 (eigenvalue -1.64).
        for q in (0.5, 0.8, 1.0, 3.0):
            f = control.BandPass._max_coefficient(q)
            limit = -1.0 / q + math.sqrt(1.0 / (q * q) + 4.0)
            self.assertLess(f, limit, f"q={q}")
            self.assertLessEqual(f, 2.0 * math.sin(0.7) + 1e-12)
            trace = 2.0 - f * f - f / q
            det = 1.0 - f / q
            disc = trace * trace - 4.0 * det
            if disc >= 0:
                radius = max(
                    abs((trace + math.sqrt(disc)) / 2),
                    abs((trace - math.sqrt(disc)) / 2),
                )
            else:
                radius = math.sqrt(det)
            self.assertLess(radius, 1.0, f"q={q}: spectral radius {radius}")
        # For high q the sin() clamp is the binding one, as before.
        self.assertAlmostEqual(
            control.BandPass._max_coefficient(3.0), 2.0 * math.sin(0.7)
        )

    def test_clamped_updates_decay_the_state(self) -> None:
        # Run the filter slowly enough that every step saturates the
        # coefficient clamp (0.5·w0·ts > 0.7 at w0 = 50 rad/s, ts = 30 ms),
        # inject energy once, then feed zeros: the stored state must decay.
        # Under the old fixed clamp the one-step eigenvalue at q = 0.8 was
        # -1.64 — this sequence grew without bound and flipped sign each step.
        bp = control.BandPass(n=1, w0=50.0, q=0.8)
        clock = [0.0]
        with patch.object(control.time, "perf_counter", side_effect=lambda: clock[0]):
            bp.update([0.0])  # primes the clock
            clock[0] = 0.03
            bp.update([1.0])  # one impulse; also sets the nominal interval
            norms = []
            for k in range(2, 30):
                clock[0] = 0.03 * k
                bp.update([0.0])
                norms.append(math.hypot(bp._lp[0], bp._bp[0]))
        self.assertGreater(norms[0], 0.0)
        self.assertLess(norms[-1], 1e-3 * norms[0], norms)
        self.assertLess(max(norms), 2.0 * norms[0], norms)

    def test_long_stall_restarts_the_filter(self) -> None:
        bp = control.BandPass(n=2, w0=20.0, q=0.8)
        clock = [0.0]
        with patch.object(control.time, "perf_counter", side_effect=lambda: clock[0]):
            bp.update([0.0, 0.0])
            for k in range(1, 200):
                clock[0] = k / 120
                bp.update([0.3, -0.3])
            self.assertNotEqual(bp._bp, [0.0, 0.0])
            # More than STALL_FACTOR nominal intervals: state dropped, zero out.
            clock[0] += control.BandPass.STALL_FACTOR * 1.5 / 120
            self.assertEqual(bp.update([0.3, -0.3]), [0.0, 0.0])
            self.assertEqual(bp._bp, [0.0, 0.0])
            self.assertEqual(bp._lp, [0.0, 0.0])
            # The next regular step runs normally again.
            clock[0] += 1 / 120
            out = bp.update([0.3, -0.3])
            self.assertNotEqual(out, [0.0, 0.0])


class CapabilityDetectionRetryTest(unittest.IsolatedAsyncioTestCase):
    """A dropped 0xB2/0xB5 reply is retried, not turned into a failed enable."""

    def _motor(self) -> myactuator.MyActuatorMotor:
        return myactuator.MyActuatorMotor.__new__(myactuator.MyActuatorMotor)

    async def test_succeeds_on_a_later_attempt(self) -> None:
        motor = self._motor()
        motor._motor_id = 1
        motor._fw_version = None
        motor._model = None
        motor._p_max = myactuator._MA_P_MAX_LEGACY
        motor._t_max = myactuator._MA_T_MAX_LEGACY
        motor._max_torque = myactuator._MA_DEFAULT_MAX_TORQUE
        version = AsyncMock(
            side_effect=[MotorError("timeout"), myactuator._MA_FW_V44_VERSION]
        )
        model = AsyncMock(return_value="RMD-X8-P20")
        with (
            patch.object(motor, "_read_firmware_version", version),
            patch.object(motor, "_read_model", model),
        ):
            await motor._detect_capabilities()
        self.assertEqual(version.await_count, 2)
        self.assertEqual(motor._fw_version, myactuator._MA_FW_V44_VERSION)
        self.assertEqual(
            (motor._p_max, motor._t_max), (myactuator._MA_P_MAX_V44, 129.0)
        )
        # Cached: no further bus traffic.
        with patch.object(motor, "_read_firmware_version", AsyncMock()) as again:
            await motor._detect_capabilities()
        again.assert_not_awaited()

    async def test_gives_up_only_after_every_attempt(self) -> None:
        motor = self._motor()
        motor._motor_id = 2
        motor._fw_version = None
        motor._model = None
        motor._p_max = myactuator._MA_P_MAX_LEGACY
        motor._t_max = myactuator._MA_T_MAX_LEGACY
        motor._max_torque = myactuator._MA_DEFAULT_MAX_TORQUE
        version = AsyncMock(side_effect=MotorError("timeout"))
        with patch.object(motor, "_read_firmware_version", version):
            with self.assertRaises(MotorError) as ctx:
                await motor._detect_capabilities()
        self.assertEqual(version.await_count, myactuator._MA_CAPABILITY_READ_ATTEMPTS)
        self.assertIn("MIT command ranges", str(ctx.exception))
        # The legacy defaults were never overwritten by a half-read.
        self.assertIsNone(motor._fw_version)
        self.assertEqual(motor._t_max, myactuator._MA_T_MAX_LEGACY)


class _FakeBus:
    def __init__(self, channel: str) -> None:
        self._channel = channel

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass


class _FakeDriver:
    """Enough of a MyActuator/Damiao driver for construction and the ranges."""

    kp_max = 500.0
    kd_max = 5.0

    def __init__(self, motor_id: int) -> None:
        self.motor_id = motor_id
        self._p_max = myactuator._MA_P_MAX_LEGACY
        self._t_max = myactuator._MA_T_MAX_LEGACY

    def set_feedback_callback(self, _cb: object) -> None:
        pass


class RtRangesHandoffTest(unittest.IsolatedAsyncioTestCase):
    """Python's detected MIT ranges ride on the arm message to the core.

    The core used to arm against its own single-attempt 0xB2/0xB5 reads and
    silently fell back to the legacy (12.5 rad, 24 Nm) ranges when one reply
    was dropped — every t_ff on that joint then decoded 2.5x/5.4x large on a
    V4.4 X6/X8 for the whole session, with nothing cross-checking Python's
    own detection. Now Python ships what it detected and the core arms
    against that.
    """

    def _robot(self):
        from almond_axol.constants import Joint
        from almond_axol.robot.axol import AxolHardware
        from almond_axol.rt import Axol

        with (
            patch("almond_axol.robot.axol.CanBus", _FakeBus),
            patch(
                "almond_axol.motor.motor.make_driver",
                side_effect=lambda _bus, motor_id, *a, **k: _FakeDriver(motor_id),
            ),
        ):
            hardware = AxolHardware(left_channel="can0", right_channel="can1")
        self.enterContext(
            patch("almond_axol.rt.link.find_binary", return_value="/fake/axol-rt")
        )
        # Left shoulder_1 and shoulder_2 on V4.4 firmware (X8 and X6); every
        # other MyActuator still on legacy firmware.
        left = hardware.left
        assert left is not None
        left.motors[Joint.SHOULDER_1]._driver._p_max = myactuator._MA_P_MAX_V44
        left.motors[Joint.SHOULDER_1]._driver._t_max = 129.0
        left.motors[Joint.SHOULDER_2]._driver._p_max = myactuator._MA_P_MAX_V44
        left.motors[Joint.SHOULDER_2]._driver._t_max = 60.0
        return Axol._wrap(hardware)

    def test_ranges_text_lists_every_myactuator_joint_per_side(self) -> None:
        rt = self._robot()
        lines = rt._ranges_text().splitlines()
        # 5 MyActuator joints per arm, none for the Damiao wrists or gripper.
        self.assertEqual(len(lines), 10)
        self.assertTrue(all(line.startswith("ranges ") for line in lines))
        by_key = {tuple(line.split()[1:3]): line.split()[3:] for line in lines}
        self.assertEqual(by_key[("0", "1")], ["12.566", "129.0"])
        self.assertEqual(by_key[("0", "2")], ["12.566", "60.0"])
        self.assertEqual(by_key[("0", "3")], ["12.5", "24.0"])
        self.assertEqual(by_key[("1", "1")], ["12.5", "24.0"])
        self.assertNotIn(("0", "6"), by_key)
        self.assertNotIn(("0", "8"), by_key)
        self.assertTrue(rt._ranges_text().endswith("\n"))

    async def test_arm_message_carries_the_ranges(self) -> None:
        from almond_axol.rt import link

        rt = link.RtLink(binary="/opt/axol-rt")
        rt._writer = MagicMock()
        rt._writer.is_closing.return_value = False
        with patch.object(rt, "_await_state", AsyncMock()) as awaited:
            await rt.arm("ranges 0 1 12.566 129.0\n")
        sent = rt._writer.write.call_args.args[0]
        self.assertTrue(sent.endswith(b"Aranges 0 1 12.566 129.0\n"), sent)
        awaited.assert_awaited_once()
        # The bare form (Mantis, tests) is still an arm with an empty body.
        with patch.object(rt, "_await_state", AsyncMock()):
            await rt.arm()
        self.assertTrue(rt._writer.write.call_args.args[0].endswith(b"A"))
        # Protocol generation 3 is what makes a core that ignores the body
        # refuse the client instead of arming blind.
        self.assertEqual(link.CONFIG_PROTO, 3)


class QuaternionHemisphereTest(unittest.TestCase):
    """A q -> -q representation flip never reaches the linear pose filters."""

    def test_same_hemisphere_keeps_the_stream_continuous(self) -> None:
        from almond_axol.teleop.worker import IKWorker

        worker = IKWorker.__new__(IKWorker)
        worker._last_raw_quat = {}
        q = np.array([0.1, 0.2, 0.3, 0.9273618])
        first = worker._same_hemisphere("left", q)
        np.testing.assert_array_equal(first, q)
        # Same rotation, flipped representation: comes back in q's hemisphere.
        flipped = worker._same_hemisphere("left", -q)
        np.testing.assert_allclose(flipped, q)
        # A genuinely different rotation (within 90°) is untouched.
        r = np.array([0.0, 0.0, math.sin(0.4), math.cos(0.4)])
        np.testing.assert_array_equal(worker._same_hemisphere("right", r), r)
        near = np.array([0.0, 0.0, math.sin(0.5), math.cos(0.5)])
        np.testing.assert_array_equal(worker._same_hemisphere("right", near), near)
        # Streams are independent per side.
        np.testing.assert_allclose(worker._same_hemisphere("right", -near), near)
        self.assertEqual(set(worker._last_raw_quat), {"left", "right"})


def _frame(seq: int) -> VRFrame:
    pose = VRPose(
        position=VRPosition(x=0.0, y=1.0, z=-0.4),
        quaternion=VRQuaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    return VRFrame(
        l_ee=pose,
        r_ee=pose,
        l_elbow=pose.position,
        r_elbow=pose.position,
        l_grip=1.0,
        r_grip=1.0,
        l_tracked=True,
        r_tracked=True,
        t=10.0 * seq,
        seq=seq,
    )


class InterpolatorWaitTest(unittest.TestCase):
    """The IK dispatch loop can sleep until a frame arrives instead of polling."""

    def test_wait_wakes_on_push_and_times_out_otherwise(self) -> None:
        interp = PoseInterpolator(min_delay_s=0.0, max_delay_s=0.0)
        self.assertFalse(interp.wait_for_frame(0.01))
        interp.push(_frame(1), now=0.0)
        self.assertTrue(interp.wait_for_frame(0.01))
        # Consumed: the next wait blocks again until another push.
        self.assertFalse(interp.wait_for_frame(0.01))

        woke = threading.Event()

        def waiter() -> None:
            if interp.wait_for_frame(2.0):
                woke.set()

        t = threading.Thread(target=waiter)
        t.start()
        interp.push(_frame(2), now=0.01)
        t.join(2.0)
        self.assertTrue(woke.is_set())


_NO_FRICTION = FrictionParams(fc=0.0, k=1.0, fv=0.0, fo=0.0)


class CalibrationOverlayWarningTest(unittest.TestCase):
    """Re-arming a deliberately restrained host damper is loud, not silent."""

    def test_enabling_damping_on_a_zero_joint_warns_but_applies(self) -> None:
        jc = JointConfig(
            kp=180.0,
            kd=5.0,
            friction=_NO_FRICTION,
            mass=1.0,
            com=(0.0, 0.0, 0.0),
            kd_host=0.0,
        )
        with self.assertLogs("almond_axol.robot.config", level=logging.WARNING) as logs:
            out = _calibrated_joint(jc, {"kd_host": 12.0}, "left shoulder_3")
        self.assertEqual(out.kd_host, 12.0)
        self.assertTrue(
            any("shoulder_3" in line and "kd_host=12.0" in line for line in logs.output)
        )

    def test_large_increase_and_wider_band_warn(self) -> None:
        jc = JointConfig(
            kp=250.0,
            kd=3.5,
            friction=_NO_FRICTION,
            mass=1.0,
            com=(0.0, 0.0, 0.0),
            kd_host=40.0,
            kd_host_q=3.0,
        )
        with self.assertLogs("almond_axol.robot.config", level=logging.WARNING) as logs:
            out = _calibrated_joint(
                jc, {"kd_host": 70.0, "kd_host_q": 0.8}, "right shoulder_1"
            )
        self.assertEqual((out.kd_host, out.kd_host_q), (70.0, 0.8))
        self.assertEqual(len(logs.output), 2)

    def test_modest_retune_is_quiet(self) -> None:
        jc = JointConfig(
            kp=250.0,
            kd=3.5,
            friction=_NO_FRICTION,
            mass=1.0,
            com=(0.0, 0.0, 0.0),
            kd_host=40.0,
            kd_host_q=3.0,
        )
        logger = logging.getLogger("almond_axol.robot.config")
        with patch.object(logger, "warning") as warning:
            out = _calibrated_joint(
                jc, {"kd_host": 45.0, "kd_host_q": 3.0, "kp": 240.0}
            )
        warning.assert_not_called()
        self.assertEqual((out.kd_host, out.kp), (45.0, 240.0))


if __name__ == "__main__":
    unittest.main()
