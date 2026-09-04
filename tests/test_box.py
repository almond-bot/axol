"""Box-mode geometry, jog axis lock and the live-setting toggle."""

from __future__ import annotations

import math
import types
import unittest

import numpy as np

from almond_axol.teleop.box import (
    BoxState,
    approach_axis,
    box_frame,
    box_targets,
    ideal_gripper_poses,
    pair_aligned,
    parallel_grip_rel,
    rodrigues,
    rotation_angle,
    side_clamp_rotation,
    snap_box,
)
from almond_axol.teleop.live import LiveSettings
from almond_axol.teleop.worker import IKWorker, _dominant_axis
from almond_axol.vr.models import VRFrame, VRPose, VRPosition, VRQuaternion

_UP = np.array((0.0, 0.0, 1.0))
_FWD = np.array((1.0, 0.0, 0.0))
_LAT = np.array((0.0, 1.0, 0.0))

# Wedge half-angle of the closed fingers (see the gripper drawing: 68 mm at
# the heel narrowing to 15.5 mm over 74 mm).
_WEDGE = math.atan((68.0 - 15.49) / 2.0 / 74.34)


def _rot_x(angle: float) -> np.ndarray:
    return rodrigues(np.array((1.0, 0.0, 0.0)), angle)


def _rot_z(angle: float) -> np.ndarray:
    return rodrigues(_UP, angle)


def _face_normal(face: float, wedge: float) -> np.ndarray:
    """Outward normal (gripper frame) of the flat finger face on the ``face`` side.

    The fingers point along -Z and narrow toward the tip, so the face tilts
    toward the tip by the wedge half-angle.
    """
    return np.array((face * math.cos(wedge), 0.0, -math.sin(wedge)))


class SideClampRotationTest(unittest.TestCase):
    def test_fingers_point_forward_and_a_flat_face_toward_the_box(self) -> None:
        for side, sign in (("left", 1.0), ("right", -1.0)):
            for face in (1.0, -1.0):
                with self.subTest(side=side, face=face):
                    r = side_clamp_rotation(sign, face, 0.0)
                    # Proper rotation.
                    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-6)
                    self.assertAlmostEqual(float(np.linalg.det(r)), 1.0, places=5)
                    # Fingers along the box +x ...
                    np.testing.assert_allclose(approach_axis(r), _FWD, atol=1e-6)
                    # ... and the chosen ±X face toward the centre: the left
                    # gripper sits at +y, so its face points -y.
                    palm = r @ np.array((face, 0.0, 0.0))
                    np.testing.assert_allclose(palm, -sign * _LAT, atol=1e-6)

    def test_grippers_are_parallel_not_facing(self) -> None:
        left = side_clamp_rotation(1.0, 1.0, 0.0)
        right = side_clamp_rotation(-1.0, 1.0, 0.0)
        self.assertGreater(
            float(np.dot(approach_axis(left), approach_axis(right))), 0.999
        )

    def test_tilt_turns_fingertips_inward_and_lays_the_wedge_face_flat(self) -> None:
        for side, sign in (("left", 1.0), ("right", -1.0)):
            for face in (1.0, -1.0):
                with self.subTest(side=side, face=face):
                    r = side_clamp_rotation(sign, face, _WEDGE)
                    app = r @ np.array((0.0, 0.0, -1.0))
                    # Still level, still mostly forward, tipped toward the centre.
                    self.assertAlmostEqual(float(app[2]), 0.0, places=6)
                    self.assertAlmostEqual(float(app[0]), math.cos(_WEDGE), places=6)
                    self.assertAlmostEqual(
                        float(app[1]), -sign * math.sin(_WEDGE), places=6
                    )
                    # The wedge face turned toward the box is now exactly
                    # vertical and normal to the box side.
                    n = r @ _face_normal(face, _WEDGE)
                    np.testing.assert_allclose(n, -sign * _LAT, atol=1e-6)


class ParallelGripRelTest(unittest.TestCase):
    def test_picks_the_face_needing_the_smaller_turn(self) -> None:
        rot = np.eye(3, dtype=np.float32)
        for face in (1.0, -1.0):
            with self.subTest(face=face):
                # Start each gripper a little off the +face / -face grasp.
                start = {
                    side: _rot_x(0.2) @ side_clamp_rotation(sign, face, 0.0)
                    for side, sign in (("left", 1.0), ("right", -1.0))
                }
                rel = parallel_grip_rel(start, rot, 0.0)
                for side, sign in (("left", 1.0), ("right", -1.0)):
                    np.testing.assert_allclose(
                        rel[side], side_clamp_rotation(sign, face, 0.0), atol=1e-6
                    )
                    # Never more than the perturbation away.
                    self.assertLess(rotation_angle(start[side], rot @ rel[side]), 0.21)

    def test_never_turns_a_wrist_more_than_a_quarter_turn_about_the_fingers(
        self,
    ) -> None:
        # Every roll about the approach axis is within 90° of one of the two
        # flat faces, so the blend never has to swing through 180°.
        rot = np.eye(3, dtype=np.float32)
        for roll in np.linspace(-math.pi, math.pi, 25):
            start = {
                side: side_clamp_rotation(sign, 1.0, 0.0)
                @ rodrigues(np.array((0.0, 0.0, 1.0)), roll)
                for side, sign in (("left", 1.0), ("right", -1.0))
            }
            rel = parallel_grip_rel(start, rot, 0.0)
            for side in ("left", "right"):
                self.assertLessEqual(
                    rotation_angle(start[side], rot @ rel[side]), math.pi / 2 + 1e-6
                )

    def test_relative_rotations_follow_a_yawed_box(self) -> None:
        rot = _rot_z(0.7)
        start = {
            side: rot @ side_clamp_rotation(sign, -1.0, 0.0)
            for side, sign in (("left", 1.0), ("right", -1.0))
        }
        rel = parallel_grip_rel(start, rot, 0.0)
        for side in ("left", "right"):
            self.assertLess(rotation_angle(start[side], rot @ rel[side]), 1e-5)


def _pair(width: float, face: float = 1.0, tilt: float = 0.0, yaw: float = 0.0):
    rot = _rot_z(yaw)
    center = np.array((0.4, 0.0, 0.3), dtype=np.float32)
    half = 0.5 * width * rot[:, 1]
    left = (
        center + half,
        (rot @ side_clamp_rotation(1.0, face, tilt)).astype(np.float32),
    )
    right = (
        center - half,
        (rot @ side_clamp_rotation(-1.0, face, tilt)).astype(np.float32),
    )
    return left, right


class PairAlignedTest(unittest.TestCase):
    def test_side_clamping_pair_is_aligned(self) -> None:
        for face in (1.0, -1.0):
            for yaw in (0.0, 0.9):
                left, right = _pair(0.3, face=face, yaw=yaw)
                self.assertTrue(pair_aligned(left, right, 0.1, 0.7, 0.0, 25.0))

    def test_facing_pair_is_not_aligned(self) -> None:
        # The old geometry — approach axes pointing at each other.
        left_rot = np.stack([_FWD, _UP, _LAT], axis=1).astype(np.float32)
        right_rot = np.stack([_FWD, -_UP, -_LAT], axis=1).astype(np.float32)
        left = (np.array((0.4, 0.15, 0.3), np.float32), left_rot)
        right = (np.array((0.4, -0.15, 0.3), np.float32), right_rot)
        self.assertFalse(pair_aligned(left, right, 0.1, 0.7, 0.0, 25.0))

    def test_width_outside_the_range_is_not_aligned(self) -> None:
        left, right = _pair(0.9)
        self.assertFalse(pair_aligned(left, right, 0.1, 0.7, 0.0, 25.0))

    def test_tolerance_bounds_the_deviation(self) -> None:
        left, right = _pair(0.3)
        tilted_left = (
            left[0],
            (_rot_x(math.radians(20.0)) @ left[1]).astype(np.float32),
        )
        self.assertTrue(pair_aligned(tilted_left, right, 0.1, 0.7, 0.0, 25.0))
        self.assertFalse(pair_aligned(tilted_left, right, 0.1, 0.7, 0.0, 15.0))

    def test_tilt_is_part_of_the_target(self) -> None:
        left, right = _pair(0.3, tilt=_WEDGE)
        self.assertTrue(pair_aligned(left, right, 0.1, 0.7, _WEDGE, 5.0))
        self.assertFalse(pair_aligned(left, right, 0.1, 0.7, 0.0, 5.0))


class SnapBoxTest(unittest.TestCase):
    def test_snap_keeps_midpoint_and_width_and_lands_on_the_side_clamp(self) -> None:
        left = (np.array((0.4, 0.2, 0.35), np.float32), _rot_x(0.3).astype(np.float32))
        right = (
            np.array((0.4, -0.2, 0.25), np.float32),
            _rot_z(-0.4).astype(np.float32),
        )
        state = snap_box(
            left, right, now=0.0, align_duration=1.0, width_min=0.1, width_max=0.7
        )
        np.testing.assert_allclose(state.center, (0.4, 0.0, 0.3), atol=1e-6)
        self.assertAlmostEqual(state.width, math.hypot(0.4, 0.1), places=6)
        ideal = ideal_gripper_poses(
            state.center, state.rot, state.width, state.grip_rel()
        )
        for side, sign in (("left", 1.0), ("right", -1.0)):
            pos, rot = ideal[side]
            np.testing.assert_allclose(approach_axis(rot), _FWD, atol=1e-6)
            # The flat face toward the centre is one of the two ±X sides.
            palm_x = rot @ np.array((1.0, 0.0, 0.0))
            self.assertAlmostEqual(abs(float(palm_x[1])), 1.0, places=6)
            self.assertAlmostEqual(float(pos[1]), sign * state.width / 2, places=6)
            self.assertAlmostEqual(float(pos[2]), 0.3, places=6)

    def test_align_blend_starts_where_the_grippers_were(self) -> None:
        left = (np.array((0.4, 0.2, 0.35), np.float32), _rot_x(0.3).astype(np.float32))
        right = (
            np.array((0.4, -0.2, 0.25), np.float32),
            _rot_z(-0.4).astype(np.float32),
        )
        state = snap_box(
            left, right, now=0.0, align_duration=1.0, width_min=0.1, width_max=0.7
        )
        at_start = box_targets(state, state.center, state.rot, now=0.0)
        np.testing.assert_allclose(at_start["left"][0], left[0], atol=1e-6)
        np.testing.assert_allclose(at_start["left"][1], left[1], atol=1e-6)
        np.testing.assert_allclose(at_start["right"][0], right[0], atol=1e-6)
        at_end = box_targets(state, state.center, state.rot, now=2.0)
        ideal = ideal_gripper_poses(
            state.center, state.rot, state.width, state.grip_rel()
        )
        np.testing.assert_allclose(at_end["left"][1], ideal["left"][1], atol=1e-6)
        self.assertTrue(state.aligned)

    def test_box_frame_is_level_with_lateral_from_right_to_left(self) -> None:
        center, rot, width = box_frame(
            np.array((0.5, 0.2, 0.4)), np.array((0.3, -0.2, 0.2))
        )
        np.testing.assert_allclose(center, (0.4, 0.0, 0.3), atol=1e-6)
        np.testing.assert_allclose(rot[:, 2], _UP, atol=1e-6)
        self.assertGreater(float(rot[1, 1]), 0.0)
        self.assertAlmostEqual(width, math.sqrt(0.04 + 0.16 + 0.04), places=6)


class DominantAxisTest(unittest.TestCase):
    def test_off_axis_leak_is_dropped(self) -> None:
        self.assertEqual(_dominant_axis(-0.95, 0.35), (-0.95, 0.0))
        self.assertEqual(_dominant_axis(0.2, -0.9), (0.0, -0.9))

    def test_deadzone_still_applies(self) -> None:
        self.assertEqual(_dominant_axis(0.1, 0.05), (0.0, 0.0))
        self.assertEqual(_dominant_axis(0.1, 0.5), (0.0, 0.5))


def _jog_worker(leader: str = "left") -> IKWorker:
    worker = object.__new__(IKWorker)
    worker._config = types.SimpleNamespace(
        box_jog_speed=0.2,
        box_jog_yaw_speed=1.0,
        box_width_speed=0.1,
        box_width_min=0.1,
        box_width_max=0.7,
        box_grip_tilt=0.0,
        box_tilt_speed=30.0,
        box_tilt_max=45.0,
    )
    worker._box_leader = leader
    return worker


def _box_state(width: float = 0.3) -> BoxState:
    return BoxState(
        center=np.array((0.4, 0.0, 0.3), np.float32),
        rot=np.eye(3, dtype=np.float32),
        width=width,
        face={"left": 1.0, "right": 1.0},
        tilt=0.0,
        align_start={},
        align_t0=0.0,
        align_duration=0.0,
    )


def _stick_frame(**sticks: float | bool) -> VRFrame:
    identity = VRQuaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    zero = VRPosition(x=0.0, y=0.0, z=0.0)
    return VRFrame(
        l_ee=VRPose(position=zero, quaternion=identity),
        r_ee=VRPose(position=zero, quaternion=identity),
        l_elbow=zero,
        r_elbow=zero,
        l_lock=True,
        r_lock=True,
        **sticks,
    )


class JogAxisLockTest(unittest.TestCase):
    def _jog(
        self, worker: IKWorker, box: BoxState, frame: VRFrame, dt: float = 0.05
    ) -> None:
        worker._integrate_jog(frame, box, box.rot, now=10.0)
        worker._integrate_jog(frame, box, box.rot, now=10.0 + dt)

    def test_width_stick_pushed_sideways_with_a_forward_leak_only_changes_width(
        self,
    ) -> None:
        worker = _jog_worker("left")
        box = _box_state()
        # Right (other) stick: hard right with a 35 % forward component.
        self._jog(worker, box, _stick_frame(r_stick_x=0.95, r_stick_y=-0.35))
        self.assertGreater(box.width, 0.3)
        np.testing.assert_allclose(box.jog_pos, 0.0, atol=1e-7)
        self.assertEqual(box.jog_yaw, 0.0)

    def test_width_stick_pushed_forward_with_a_side_leak_only_lifts(self) -> None:
        worker = _jog_worker("left")
        box = _box_state()
        self._jog(worker, box, _stick_frame(r_stick_x=0.3, r_stick_y=-0.9))
        self.assertEqual(box.width, 0.3)
        self.assertGreater(float(box.jog_pos[2]), 0.0)
        np.testing.assert_allclose(box.jog_pos[:2], 0.0, atol=1e-7)

    def test_clicked_leader_stick_is_one_axis_at_a_time(self) -> None:
        worker = _jog_worker("left")
        box = _box_state()
        self._jog(
            worker, box, _stick_frame(l_stick_x=0.9, l_stick_y=-0.3, l_stick_click=True)
        )
        self.assertNotEqual(box.jog_yaw, 0.0)
        np.testing.assert_allclose(box.jog_pos, 0.0, atol=1e-7)

    def test_free_leader_stick_keeps_diagonals(self) -> None:
        worker = _jog_worker("left")
        box = _box_state()
        self._jog(worker, box, _stick_frame(l_stick_x=0.7, l_stick_y=-0.7))
        self.assertGreater(float(box.jog_pos[0]), 0.0)  # forward
        self.assertLess(float(box.jog_pos[1]), 0.0)  # push right = move right (-y)
        self.assertEqual(float(box.jog_pos[2]), 0.0)

    def test_roles_follow_the_leader(self) -> None:
        worker = _jog_worker("right")
        box = _box_state()
        # With the right hand leading, the left stick is the width stick.
        self._jog(worker, box, _stick_frame(l_stick_x=-0.9, l_stick_y=0.2))
        self.assertLess(box.width, 0.3)
        np.testing.assert_allclose(box.jog_pos, 0.0, atol=1e-7)


class TiltJogTest(unittest.TestCase):
    def _jog(
        self, worker: IKWorker, box: BoxState, frame: VRFrame, dt: float = 0.05
    ) -> None:
        worker._integrate_jog(frame, box, box.rot, now=10.0)
        worker._integrate_jog(frame, box, box.rot, now=10.0 + dt)

    def test_clicked_other_stick_left_tilts_fingertips_inward(self) -> None:
        worker = _jog_worker("left")
        box = _box_state()
        self._jog(worker, box, _stick_frame(r_stick_x=-1.0, r_stick_click=True))
        # 30 deg/s for 50 ms.
        self.assertAlmostEqual(math.degrees(box.tilt), 1.5, places=5)
        # Width and position untouched: the click swaps the stick's meaning.
        self.assertEqual(box.width, 0.3)
        np.testing.assert_allclose(box.jog_pos, 0.0, atol=1e-7)
        # The target rotations follow: the left gripper's fingers now yaw
        # toward the centre (-y), the right's toward +y.
        rel = box.grip_rel()
        self.assertLess(float(approach_axis(rel["left"])[1]), 0.0)
        self.assertGreater(float(approach_axis(rel["right"])[1]), 0.0)

    def test_right_tilts_outward_and_negative_is_allowed(self) -> None:
        worker = _jog_worker("left")
        box = _box_state()
        self._jog(worker, box, _stick_frame(r_stick_x=1.0, r_stick_click=True))
        self.assertAlmostEqual(math.degrees(box.tilt), -1.5, places=5)

    def test_tilt_is_clamped_and_written_back_to_the_config(self) -> None:
        worker = _jog_worker("left")
        box = _box_state()
        frame = _stick_frame(r_stick_x=-1.0, r_stick_click=True)
        worker._integrate_jog(frame, box, box.rot, now=0.0)
        for i in range(1, 200):  # 200 x 0.1 s at 30 deg/s = 600 deg requested
            worker._integrate_jog(frame, box, box.rot, now=0.1 * i)
        self.assertAlmostEqual(math.degrees(box.tilt), 45.0, places=5)
        self.assertAlmostEqual(worker._config.box_grip_tilt, 45.0, places=5)

    def test_clicked_other_stick_forward_does_nothing(self) -> None:
        worker = _jog_worker("left")
        box = _box_state()
        self._jog(
            worker, box, _stick_frame(r_stick_y=-1.0, r_stick_x=0.3, r_stick_click=True)
        )
        self.assertEqual(box.tilt, 0.0)
        self.assertEqual(box.width, 0.3)
        np.testing.assert_allclose(box.jog_pos, 0.0, atol=1e-7)

    def test_jogged_tilt_seeds_the_next_engage(self) -> None:
        left, right = _pair(0.3)
        state = snap_box(
            left,
            right,
            now=0.0,
            align_duration=0.0,
            width_min=0.1,
            width_max=0.7,
            tilt=0.3,
        )
        self.assertEqual(state.tilt, 0.3)
        ideal = ideal_gripper_poses(
            state.center, state.rot, state.width, state.grip_rel()
        )
        self.assertAlmostEqual(
            float(approach_axis(ideal["left"][1])[1]), -math.sin(0.3), places=5
        )


class _FakeCore:
    def __init__(self) -> None:
        self.values = {
            "box_mode": True,
            "reengage": "clutch",
            "hold_to_engage": False,
            "position_multiplier": 1.0,
            "teleop_max_vel": 6.283185307179586,
            "box_jog_speed": 0.15,
        }
        self.set_calls: list[tuple[str, object]] = []

    def live_value(self, key: str) -> object:
        return self.values[key]

    def set_live(self, key: str, value: object) -> None:
        self.set_calls.append((key, value))
        self.values[key] = value


class LiveToggleTest(unittest.TestCase):
    def test_toggle_flips_the_server_side_value(self) -> None:
        core = _FakeCore()
        live = LiveSettings(core, robot=object(), publish=lambda snapshot: None)
        live.apply("box_mode", "toggle")
        self.assertEqual(core.set_calls, [("box_mode", False)])
        live.apply("box_mode", "toggle")
        self.assertEqual(core.set_calls[-1], ("box_mode", True))

    def test_plain_booleans_still_coerce(self) -> None:
        core = _FakeCore()
        live = LiveSettings(core, robot=object(), publish=lambda snapshot: None)
        live.apply("box_mode", "false")
        self.assertEqual(core.set_calls, [("box_mode", False)])
        live.apply("hold_to_engage", True)
        self.assertEqual(core.set_calls[-1], ("hold_to_engage", True))

    def test_toggle_is_boolean_only(self) -> None:
        core = _FakeCore()
        live = LiveSettings(core, robot=object(), publish=lambda snapshot: None)
        with self.assertRaises(ValueError):
            live.apply("reengage", "toggle")
        with self.assertRaises(ValueError):
            live.apply("box_jog_speed", "toggle")


if __name__ == "__main__":
    unittest.main()
