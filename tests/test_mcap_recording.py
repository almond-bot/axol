"""MCAP episode recording (Collect data's Recording format = mcap)."""

from __future__ import annotations

import math
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation

from almond_axol.constants import AxolModel, Joint, urdf_path
from almond_axol.recording.mcap import messages
from almond_axol.recording.mcap.odometry import (
    WHEEL_HALF_SPACING_M,
    WHEEL_RADIUS_M,
    XDriveOdometry,
    body_twist,
    wheel_rims,
)
from almond_axol.recording.mcap.recorder import McapSession, episode_indices
from almond_axol.recording.mcap.rings import (
    JOINT_RING_CAPACITY,
    TICK_RING_CAPACITY,
    joint_ring_keys,
    mode_name,
    pack_tick,
    tick_ring_keys,
)
from almond_axol.recording.mcap.scene import (
    CameraSource,
    Scene,
    camera_frame_id,
    camera_topic,
    finger_travel,
)
from almond_axol.recording.mcap.urdf_tree import UrdfTree, quaternion_xyzw
from almond_axol.recording.mcap.writer import EpisodeFile
from almond_axol.recording.mcap.zed_conf import (
    camera_intrinsics,
    load_calibration,
    parse_conf_text,
    stereo_transform,
)
from almond_axol.robot.jelly import WHEEL_SIGNS
from almond_axol.video.shm_frames import (
    SnapshotReader,
    SnapshotWriter,
    au_has_coded_slice,
    au_is_keyframe,
)


def read_ros2_messages(path: str) -> list[SimpleNamespace]:
    """Every message in an MCAP file, decoded, in log-time order."""
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        return [
            SimpleNamespace(
                channel=channel, log_time_ns=message.log_time, ros_msg=ros_msg
            )
            for _schema, channel, message, ros_msg in reader.iter_decoded_messages(
                log_time_order=True
            )
        ]


HEAD_SERIAL = 51617969
LEFT_WRIST_SERIAL = 302271563
RIGHT_WRIST_SERIAL = 302271564

HEAD_CONF = """\
[LEFT_CAM_FHD1200]
fx=733.5
fy=733.0
cx=969.25
cy=592.75
k1=-0.05
k2=0.02
p1=0.0003
p2=-0.0002
k3=-0.004

[RIGHT_CAM_FHD1200]
fx=734.0
fy=733.5
cx=960.0
cy=600.0
k1=-0.04
k2=0.01
p1=0.0001
p2=-0.0001
k3=-0.003

[LEFT_CAM_SVGA]
fx=366.0
fy=366.0
cx=480.0
cy=300.0
k1=-0.05
k2=0.02
p1=0.0003
p2=-0.0002
k3=-0.004

[STEREO]
Baseline=50.12
TY=0.08
TZ=-0.15
CV_FHD1200=0.002
RX_FHD1200=-0.001
RZ_FHD1200=0.0005

[MISC]
Sensor_ID=1
"""

FISHEYE_CONF = """\
[LEFT_CAM_FHD1200]
fx=560.0
fy=559.5
cx=955.0
cy=601.0
k1=0.12
k2=-0.03
k3=0.004
k4=-0.0007
"""


class _FakeReader:
    """Stands in for EncodedAuReader: hands out queued access units."""

    def __init__(self) -> None:
        self.queue: list[tuple[bytes, float, float]] = []
        self.lost_exposures = 0
        self.flushes = 0
        self.closed = False

    def begin_flush(self) -> None:
        self.flushes += 1
        self.queue.clear()

    def finish_flush(self, timeout_s: float = 2.0) -> None:
        pass

    def reset_lost_exposures(self) -> None:
        self.lost_exposures = 0

    def drain_aus(self) -> list[tuple[bytes, float, float]]:
        out, self.queue = self.queue, []
        return out

    def disconnect(self) -> None:
        self.closed = True


def _au(i: int) -> bytes:
    # An H.265 IDR (type 19) access unit with a recognisable payload.
    return b"\x00\x00\x00\x01\x26\x01" + i.to_bytes(4, "little")


def _meta(camera: str, eye: str | None, serial: int, stereo: bool) -> dict:
    return {
        "transport": "gstshm-h265",
        "codec": "h265",
        "socket_path": "/unused",
        "width": 960,
        "height": 600,
        "fps": 60,
        "capture_fps": 60,
        "pts_perf_offset_s": 0.0,
        "camera": camera,
        "eye": eye,
        "serial": serial,
        "resolution": "HD1200",
        "stereo": stereo,
    }


class MessagesTest(unittest.TestCase):
    def test_every_message_type_decodes_with_the_reference_decoder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.mcap"
            f = EpisodeFile(path, clock_offset_ns=1_700_000_000_000_000_000)
            s = f.stamp(1.25)
            f.write(
                "/v",
                "foxglove_msgs/CompressedVideo",
                messages.compressed_video(s, "cam", b"\x00\x00\x01abcde", "h265"),
                s,
            )
            f.write(
                "/ci",
                "sensor_msgs/CameraInfo",
                messages.camera_info(
                    s,
                    "cam",
                    width=10,
                    height=5,
                    distortion_model="equidistant",
                    d=[1, 2, 3, 4],
                    k=list(range(9)),
                    r=[1, 0, 0, 0, 1, 0, 0, 0, 1],
                    p=list(range(12)),
                ),
                s,
            )
            f.write(
                "/js",
                "sensor_msgs/JointState",
                messages.joint_state(s, ["a", "bb"], [1.0, 2.0], [3.0, 4.0], [5, 6]),
                s,
            )
            f.write(
                "/tf",
                "tf2_msgs/TFMessage",
                messages.tf_message([(s, "p", "c", (1, 2, 3), (0, 0, 0, 1))]),
                s,
            )
            f.write(
                "/odom",
                "nav_msgs/Odometry",
                messages.odometry(
                    s,
                    "world",
                    "jelly",
                    (1, 2, 3, 0, 0, 0, 1),
                    (0.1, 0.2, 0),
                    (0, 0, 0.3),
                ),
                s,
            )
            f.write(
                "/ps",
                "geometry_msgs/PoseStamped",
                messages.pose_stamped(s, "root", (1, 2, 3, 0, 0, 0, 1)),
                s,
            )
            f.write(
                "/tw",
                "geometry_msgs/TwistStamped",
                messages.twist_stamped(s, "jelly", (1, 2, 3), (4, 5, 6)),
                s,
            )
            f.write(
                "/joy",
                "sensor_msgs/Joy",
                messages.joy(s, "vr", [0.5, -1.0], [1, 0, 1]),
                s,
            )
            f.write("/rd", "std_msgs/String", messages.string_message("<robot/>"), s)
            f.write(
                "/cmd",
                "almond_msgs/CommandInfo",
                messages.command_info(
                    s,
                    2**40 + 7,
                    s - 1,
                    ["left_s1_0", "left_gripper"],
                    ["left_arm", "left_gripper"],
                    ["impedance", "position_force"],
                ),
                s,
            )
            f.write(
                "/base",
                "almond_msgs/BaseCommand",
                messages.base_command(
                    s, 3, s, "jelly", "velocity", 0.5, -0.25, 1.0, -1
                ),
                s,
            )
            f.finish()
            self.assertFalse(path.with_name(path.name + ".partial").exists())
            decoded = {m.channel.topic: m for m in read_ros2_messages(str(path))}

        self.assertEqual(decoded["/v"].ros_msg.data, b"\x00\x00\x01abcde")
        self.assertEqual(decoded["/v"].ros_msg.format, "h265")
        self.assertEqual(decoded["/v"].ros_msg.timestamp.sec, 1_700_000_001)
        self.assertEqual(decoded["/v"].ros_msg.timestamp.nanosec, 250_000_000)
        self.assertEqual(decoded["/v"].log_time_ns, 1_700_000_001_250_000_000)
        self.assertEqual(decoded["/ci"].ros_msg.d, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(decoded["/ci"].ros_msg.distortion_model, "equidistant")
        self.assertEqual(decoded["/js"].ros_msg.effort, [5.0, 6.0])
        tf = decoded["/tf"].ros_msg.transforms[0]
        self.assertEqual((tf.header.frame_id, tf.child_frame_id), ("p", "c"))
        self.assertEqual(decoded["/odom"].ros_msg.twist.twist.angular.z, 0.3)
        self.assertEqual(decoded["/ps"].ros_msg.pose.position.z, 3.0)
        self.assertEqual(decoded["/tw"].ros_msg.twist.linear.y, 2.0)
        self.assertEqual(decoded["/joy"].ros_msg.buttons, [1, 0, 1])
        self.assertEqual(decoded["/rd"].ros_msg.data, "<robot/>")
        cmd = decoded["/cmd"].ros_msg
        self.assertEqual(cmd.command_id, 2**40 + 7)
        self.assertEqual(cmd.issue_timestamp.nanosec, 249_999_999)
        self.assertEqual(cmd.target_device_id, ["left_arm", "left_gripper"])
        self.assertEqual(cmd.control_mode, ["impedance", "position_force"])
        self.assertEqual(cmd.name, ["left_s1_0", "left_gripper"])
        base = decoded["/base"].ros_msg
        self.assertEqual((base.command_id, base.target_device_id), (3, "jelly"))
        self.assertEqual((base.vx, base.vy, base.wz), (0.5, -0.25, 1.0))
        self.assertEqual(base.lift_direction, -1)

    def test_video_payload_alignment_holds_for_every_length(self) -> None:
        # The trailing format string must stay 4-aligned whatever the AU size.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.mcap"
            f = EpisodeFile(path, clock_offset_ns=0)
            for n in range(1, 9):
                data = bytes(range(n))
                f.write(
                    f"/v{n}",
                    "foxglove_msgs/CompressedVideo",
                    messages.compressed_video(n, "cam", data, "h265"),
                    n,
                )
            f.finish()
            for msg in read_ros2_messages(str(path)):
                n = int(msg.channel.topic[2:])
                self.assertEqual(msg.ros_msg.data, bytes(range(n)))
                self.assertEqual(msg.ros_msg.format, "h265")

    def test_discarded_file_leaves_nothing_behind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_000000.mcap"
            f = EpisodeFile(path, clock_offset_ns=0)
            f.write("/rd", "std_msgs/String", messages.string_message("x"), 0)
            f.discard()
            self.assertEqual(list(Path(directory).iterdir()), [])


class ZedConfTest(unittest.TestCase):
    def test_pinhole_section_reads_as_plumb_bob_in_ros_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / f"SN{HEAD_SERIAL}.conf").write_text(HEAD_CONF)
            calib = load_calibration(HEAD_SERIAL, Path(directory))
        assert calib is not None
        info = camera_intrinsics(
            calib,
            eye="right",
            capture_resolution="HD1200",
            width=1920,
            height=1200,
            fisheye=False,
        )
        assert info is not None
        self.assertEqual(info.distortion_model, "plumb_bob")
        self.assertEqual(info.d, (-0.04, 0.01, 0.0001, -0.0001, -0.003))
        self.assertEqual(info.k, (734.0, 0.0, 960.0, 0.0, 733.5, 600.0, 0.0, 0.0, 1.0))

    def test_fisheye_section_reads_as_equidistant_k1_to_k4(self) -> None:
        calib = SimpleNamespace(
            section=lambda name: parse_conf_text(FISHEYE_CONF).get(name.upper())
        )
        info = camera_intrinsics(
            calib,  # type: ignore[arg-type]
            eye="left",
            capture_resolution="HD1200",
            width=1920,
            height=1200,
            fisheye=True,
        )
        assert info is not None
        self.assertEqual(info.distortion_model, "equidistant")
        self.assertEqual(info.d, (0.12, -0.03, 0.004, -0.0007))

    def test_downscaled_recording_scales_focal_length_and_centre(self) -> None:
        sections = parse_conf_text(HEAD_CONF)
        calib = SimpleNamespace(section=lambda name: sections.get(name.upper()))
        full = camera_intrinsics(
            calib,  # type: ignore[arg-type]
            eye="left",
            capture_resolution="HD1200",
            width=1920,
            height=1200,
            fisheye=False,
        )
        half = camera_intrinsics(
            calib,  # type: ignore[arg-type]
            eye="left",
            capture_resolution="HD1200",
            width=960,
            height=600,
            fisheye=False,
        )
        assert full is not None and half is not None
        self.assertAlmostEqual(half.fx, full.fx / 2)
        self.assertAlmostEqual(half.cx, (full.cx + 0.5) / 2 - 0.5)
        self.assertEqual(half.d, full.d)
        self.assertEqual((half.width, half.height), (960, 600))

    def test_missing_resolution_or_file_gives_no_intrinsics(self) -> None:
        sections = parse_conf_text(HEAD_CONF)
        calib = SimpleNamespace(section=lambda name: sections.get(name.upper()))
        self.assertIsNone(
            camera_intrinsics(
                calib,  # type: ignore[arg-type]
                eye="right",
                capture_resolution="HD1080",
                width=1920,
                height=1080,
                fisheye=False,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(load_calibration(123, Path(directory)))

    def test_stereo_transform_places_the_right_eye_from_the_factory_file(self) -> None:
        sections = parse_conf_text(HEAD_CONF)
        calib = SimpleNamespace(section=lambda name: sections.get(name.upper()))
        transform = stereo_transform(calib, "HD1200")  # type: ignore[arg-type]
        assert transform is not None
        np.testing.assert_allclose(transform[:3, 3], [0.05012, 0.00008, -0.00015])
        expected = Rotation.from_rotvec([-0.001, 0.002, 0.0005]).as_matrix()
        np.testing.assert_allclose(transform[:3, :3], expected, atol=1e-12)


class UrdfTreeTest(unittest.TestCase):
    def test_forward_kinematics_matches_yourdfpy(self) -> None:
        import yourdfpy

        path = urdf_path(AxolModel.MOBILE)
        tree = UrdfTree.load(path)
        reference = yourdfpy.URDF.load(
            str(path), build_collision_scene_graph=False, load_meshes=False
        )
        rng = np.random.default_rng(3)
        q = {n: float(rng.uniform(-1, 1)) for n in reference.actuated_joint_names}
        q["lift"] = 0.2
        reference.update_cfg(q)
        for link in (
            "left_gripper",
            "right_wrist_camera_optical",
            "head_camera_right_optical",
            "left_finger_2",
            "jelly",
        ):
            with self.subTest(link=link):
                np.testing.assert_allclose(
                    tree.fk(tree.root, link, q),
                    reference.get_transform(link, tree.root),
                    atol=1e-12,
                )

    def test_quaternion_matches_scipy(self) -> None:
        for seed in range(20):
            rotation = Rotation.random(random_state=seed)
            x, y, z, w = quaternion_xyzw(rotation.as_matrix())
            np.testing.assert_allclose(
                Rotation.from_quat([x, y, z, w]).as_matrix(),
                rotation.as_matrix(),
                atol=1e-12,
            )
            self.assertGreaterEqual(w, 0.0)

    def test_finger_travel_matches_the_ik_model(self) -> None:
        from almond_axol.kinematics.model import _measure_finger_slide

        path = urdf_path(AxolModel.MOBILE)
        tree = UrdfTree.load(path)
        travel = finger_travel(path, tree)
        assert travel is not None
        _axis, _side, closed, stroke = _measure_finger_slide(path, sorted(tree.links))
        self.assertAlmostEqual(travel[0], closed)
        self.assertAlmostEqual(travel[1], stroke)


class OdometryTest(unittest.TestCase):
    def _drive(
        self,
        odom: XDriveOdometry,
        angles: np.ndarray,
        t: float,
        vx: float,
        vy: float,
        wz: float,
        seconds: float,
    ) -> tuple[np.ndarray, float]:
        """Integrate a constant body velocity into motor angles at 50 Hz."""
        signs = np.array([WHEEL_SIGNS[i] for i in (1, 2, 3, 4)])
        motor_speed = np.array(wheel_rims(vx, vy, wz)) / WHEEL_RADIUS_M * signs
        for _ in range(round(seconds * 50)):
            t += 0.02
            angles = angles + motor_speed * 0.02
            odom.update(t, angles, motor_speed)
        return angles, t

    def test_forward_and_inverse_wheel_maps_agree(self) -> None:
        vx, vy, wz, slip = body_twist(wheel_rims(0.3, -0.2, 0.7))
        self.assertAlmostEqual(vx, 0.3)
        self.assertAlmostEqual(vy, -0.2)
        self.assertAlmostEqual(wz, 0.7)
        self.assertAlmostEqual(slip, 0.0)

    def test_matches_the_drive_mix_direction(self) -> None:
        # A forward command spins every motor in the direction mix() asks for.
        from almond_axol.robot.jelly import mix

        command = mix(1.0, 0.0, 0.0, max_speed=10.0, turn_scale=1.0)
        odom = XDriveOdometry(WHEEL_SIGNS)
        vx, vy, wz = odom.command_twist(command)
        self.assertGreater(vx, 0.0)
        self.assertAlmostEqual(vy, 0.0)
        self.assertAlmostEqual(wz, 0.0)
        vx, vy, wz = odom.command_twist(mix(0.0, 0.0, 1.0, 10.0, 1.0))
        self.assertGreater(wz, 0.0)  # +wz is counter-clockwise
        self.assertAlmostEqual(vx, 0.0)

    def test_drive_then_turn_differs_from_turn_then_drive(self) -> None:
        a = XDriveOdometry(WHEEL_SIGNS)
        angles, t = np.zeros(4), 0.0
        a.update(t, angles, np.zeros(4))
        angles, t = self._drive(a, angles, t, 0.5, 0.0, 0.0, 2.0)
        angles, t = self._drive(a, angles, t, 0.0, 0.0, math.pi / 4, 2.0)
        self.assertAlmostEqual(a.x, 1.0, places=6)
        self.assertAlmostEqual(a.y, 0.0, places=6)
        self.assertAlmostEqual(a.yaw, math.pi / 2, places=6)

        b = XDriveOdometry(WHEEL_SIGNS)
        angles, t = np.zeros(4), 0.0
        b.update(t, angles, np.zeros(4))
        angles, t = self._drive(b, angles, t, 0.0, 0.0, math.pi / 4, 2.0)
        angles, t = self._drive(b, angles, t, 0.5, 0.0, 0.0, 2.0)
        self.assertAlmostEqual(b.x, 0.0, places=6)
        self.assertAlmostEqual(b.y, 1.0, places=6)

    def test_driving_in_a_circle_tracks_the_arc(self) -> None:
        odom = XDriveOdometry(WHEEL_SIGNS)
        angles, t = np.zeros(4), 0.0
        odom.update(t, angles, np.zeros(4))
        # Turning at pi/8 rad/s with forward speed pi/8 m/s: a 1 m radius circle.
        angles, t = self._drive(odom, angles, t, math.pi / 8, 0.0, math.pi / 8, 8.0)
        # Half a circle: back on the y axis, 2 m to the left, facing back.
        self.assertAlmostEqual(odom.x, 0.0, places=3)
        self.assertAlmostEqual(odom.y, 2.0, places=3)
        self.assertAlmostEqual(abs(odom.yaw), math.pi, places=3)

    def test_rejects_an_impossible_jump(self) -> None:
        odom = XDriveOdometry(WHEEL_SIGNS)
        odom.update(0.0, np.zeros(4), np.zeros(4))
        odom.update(0.02, np.array([500.0, 0, 0, 0]), np.zeros(4))
        self.assertEqual((odom.x, odom.y, odom.yaw), (0.0, 0.0, 0.0))
        self.assertEqual(odom.rejected, 1)

    def test_geometry_comes_from_the_jelly_urdf(self) -> None:
        tree = UrdfTree.load(urdf_path(AxolModel.MOBILE))
        for i in (1, 2, 3, 4):
            x, y, z = tree.joints[f"wheel_{i}"].origin[:3, 3]
            self.assertAlmostEqual(abs(x), WHEEL_HALF_SPACING_M, places=3)
            self.assertAlmostEqual(abs(y), WHEEL_HALF_SPACING_M, places=3)
            self.assertAlmostEqual(z + 0.0001, WHEEL_RADIUS_M, places=4)


class RingTest(unittest.TestCase):
    def test_in_order_read_counts_overwritten_records(self) -> None:
        writer = SnapshotWriter(["a", "b"], ["c"], capacity=8)
        reader = SnapshotReader(writer.name, ["a", "b"], ["c"])
        try:
            rows, after, lost = reader.read_rows_since(0)
            self.assertEqual((len(rows), after, lost), (0, 0, 0))
            for i in range(5):
                writer.write({"a": i, "b": -i}, {"c": 10 * i}, float(i))
            rows, after, lost = reader.read_rows_since(0)
            self.assertEqual(rows[:, 0].tolist(), [0, 1, 2, 3, 4])
            self.assertEqual((after, lost), (5, 0))
            for i in range(5, 20):
                writer.write_values(float(i), [i, -i, 10 * i])
            rows, after, lost = reader.read_rows_since(after)
            self.assertEqual(rows[:, 0].tolist(), list(range(12, 20)))
            self.assertEqual(rows[-1, 1:4].tolist(), [19, -19, 190])
            self.assertEqual((after, lost), (20, 7))
        finally:
            reader.close()
            writer.close()

    def test_pack_tick_fills_every_column(self) -> None:
        pose = SimpleNamespace(
            position=SimpleNamespace(x=0.1, y=0.2, z=0.3),
            quaternion=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        frame = SimpleNamespace(
            seq=7,
            t_host=12.5,
            l_ee=pose,
            r_ee=pose,
            l_grip=0.25,
            r_grip=1.0,
            l_stick_x=0.0,
            l_stick_y=-1.0,
            r_stick_x=0.5,
            r_stick_y=0.0,
            l_lock=True,
            r_lock=False,
            l_stick_click=False,
            r_stick_click=True,
            reset=False,
            stow=False,
            l_tracked=True,
            r_tracked=True,
        )
        jelly = SimpleNamespace(
            wheel_sample=SimpleNamespace(
                positions=(1.0, 2.0, 3.0, 4.0), velocities=(0.1, 0.2, 0.3, 0.4), at=5.0
            ),
            wheel_speeds=[1.0, -1.0, 1.0, -1.0],
            has_wheels=True,
            lift_raised=0.5,
            parked=True,
            latest_command=(9, 4.0, 0.5, 0.0, -0.5, 1),
        )
        command = {f"left_{j.value}.pos": 0.1 for j in Joint}
        row = pack_tick(
            command,
            frame,
            jelly,
            perf_minus_monotonic=100.0,
            command_id=42,
            issue_t=12.25,
            arm_mode="gravity_compensation",
        )
        self.assertEqual(sorted(row), sorted(tick_ring_keys()))
        self.assertEqual(row["cmd.left_shoulder_1"], 0.1)
        self.assertTrue(math.isnan(row["cmd.right_shoulder_1"]))
        self.assertEqual(row["vr.seq"], 7.0)
        self.assertEqual(row["vr.l_lock"], 1.0)
        self.assertEqual(row["vr.r.pz"], 0.3)
        self.assertEqual(row["jelly.t"], 105.0)
        self.assertEqual(row["jelly.pos3"], 3.0)
        self.assertEqual(row["jelly.cmd2"], -1.0)
        self.assertEqual(row["jelly.lift"], 0.5)
        self.assertEqual((row["cmd.id"], row["cmd.issue_t"]), (42.0, 12.25))
        self.assertEqual(mode_name(row["cmd.arm_mode"]), "gravity_compensation")
        self.assertEqual(row["jelly.parked"], 1.0)
        self.assertEqual(row["jelly.cmd_id"], 9.0)
        self.assertEqual(row["jelly.cmd_t"], 104.0)
        self.assertEqual(row["jelly.cmd_lift"], 1.0)

        empty = pack_tick(None)
        self.assertEqual(mode_name(empty.pop("cmd.arm_mode")), "impedance")
        self.assertTrue(all(math.isnan(v) for v in empty.values()))

    def test_jelly_numbers_each_command(self) -> None:
        from almond_axol.robot.jelly import UP, Jelly, JellyConfig

        jelly = Jelly(JellyConfig())
        self.assertIsNone(jelly.latest_command)
        jelly.set_command(0.5, 0.0, -0.25, UP)
        self.assertEqual(jelly.latest_command[0], 1)
        self.assertEqual(jelly.latest_command[2:], (0.5, 0.0, -0.25, UP))
        with self.assertRaises(ValueError):
            jelly.set_command(float("nan"), 0.0, 0.0)
        # A rejected command still counts, as the stop it became.
        self.assertEqual(jelly.latest_command[0], 2)
        self.assertEqual(jelly.latest_command[2:], (0.0, 0.0, 0.0, 0))


class NalTest(unittest.TestCase):
    def test_h265_keyframes_and_coded_slices(self) -> None:
        vps = b"\x00\x00\x00\x01\x40\x01\x0c"  # VPS, type 32
        idr = b"\x00\x00\x01\x26\x01\xaa"  # IDR_W_RADL, type 19
        trail = b"\x00\x00\x01\x02\x01\xbb"  # TRAIL_R, type 1
        self.assertTrue(au_is_keyframe(vps + idr, "h265"))
        self.assertTrue(au_has_coded_slice(vps + idr, "h265"))
        self.assertFalse(au_is_keyframe(trail, "h265"))
        self.assertTrue(au_has_coded_slice(trail, "h265"))
        self.assertFalse(au_has_coded_slice(vps, "h265"))
        # The same bytes read as H.264 mean something else entirely.
        self.assertFalse(au_is_keyframe(vps + idr, "h264"))

    def test_relay_encodes_h265_for_mcap(self) -> None:
        from almond_axol.video.gst_zed import _dataset_enc_shmsink

        branch = _dataset_enc_shmsink("/tmp/c.sock", 1920, 1200, 60, "dsenc", "h265")
        self.assertIn("nvv4l2h265enc name=dsenc", branch)
        self.assertIn("video/x-h265,stream-format=byte-stream,alignment=au", branch)
        self.assertIn("iframeinterval=1", branch)
        self.assertIn("bitrate=40000000", branch)
        self.assertNotIn("h264", branch)

    def test_reader_parses_h265(self) -> None:
        from almond_axol.video.shm_frames import EncodedAuReader

        gst = mock.Mock()
        with mock.patch(
            "almond_axol.video.gst_zed._require_gst", return_value=(gst, None)
        ):
            EncodedAuReader(
                "/tmp/c.sock", 960, 600, 60, pts_perf_offset_s=0.0, codec="h265"
            )
        pipeline = gst.parse_launch.call_args.args[0]
        self.assertIn("video/x-h265", pipeline)
        self.assertIn("h265parse", pipeline)


class SceneTest(unittest.TestCase):
    def test_camera_names(self) -> None:
        self.assertEqual(camera_topic("overhead", "right"), "/head/right")
        self.assertEqual(camera_topic("left_arm", None), "/wrist/left")
        self.assertEqual(
            camera_frame_id("right_arm", None), "right_wrist_camera_optical"
        )
        self.assertEqual(
            camera_frame_id("overhead", "left"), "head_camera_left_optical"
        )
        self.assertEqual(camera_topic("side", None), "/side")

    def test_factory_stereo_replaces_the_cad_right_eye(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / f"SN{HEAD_SERIAL}.conf").write_text(HEAD_CONF)
            cameras = {
                src: CameraSource.from_meta(
                    src, _meta("overhead", eye, HEAD_SERIAL, True)
                )
                for src, eye in (("overhead_left", "left"), ("overhead_right", "right"))
            }
            scene = Scene.build(
                urdf_path(AxolModel.MOBILE),
                has_gripper=True,
                jelly_wheels=True,
                jelly_lift=True,
                cameras=cameras,
                calibration_dir=Path(directory),
            )
        static = {child: (parent, t) for parent, child, t in scene.static_transforms()}
        left_parent, left = static["head_camera_left_optical"]
        right_parent, right = static["head_camera_right_optical"]
        self.assertEqual(left_parent, right_parent)
        baseline = np.linalg.inv(left) @ right
        np.testing.assert_allclose(
            baseline[:3, 3], [0.05012, 0.00008, -0.00015], atol=1e-9
        )
        self.assertIn("ZED factory", scene.stereo_source)
        # Measured joints are never also published as static.
        self.assertNotIn("lift", static.values())
        self.assertNotIn("lift_stage", static)
        self.assertNotIn("jelly", static)


class SessionTest(unittest.TestCase):
    """The recorder's capture and save path, with fake camera readers."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "episodes"
        calib = Path(self._tmp.name) / "calib"
        calib.mkdir()
        (calib / f"SN{HEAD_SERIAL}.conf").write_text(HEAD_CONF)
        (calib / f"SN{LEFT_WRIST_SERIAL}.conf").write_text(FISHEYE_CONF)
        raw_meta = {
            "overhead_left": _meta("overhead", "left", HEAD_SERIAL, True),
            "overhead_right": _meta("overhead", "right", HEAD_SERIAL, True),
            "left_arm": _meta("left_arm", None, LEFT_WRIST_SERIAL, False),
            "right_arm": _meta("right_arm", None, RIGHT_WRIST_SERIAL, False),
        }
        cameras = {src: CameraSource.from_meta(src, m) for src, m in raw_meta.items()}
        self.scene = Scene.build(
            urdf_path(AxolModel.MOBILE),
            has_gripper=True,
            jelly_wheels=True,
            jelly_lift=True,
            cameras=cameras,
            calibration_dir=calib,
        )
        self.tick_writer = SnapshotWriter(
            tick_ring_keys(), [], capacity=TICK_RING_CAPACITY
        )
        self.joint_writer = SnapshotWriter(
            joint_ring_keys(), [], capacity=JOINT_RING_CAPACITY
        )
        self.readers = {src: _FakeReader() for src in raw_meta}
        self.session = McapSession(
            {
                "dataset_root": str(self.root),
                "operator": "Sam",
                "robot_serial": "HUB123",
                "robot_model": "mobile",
                "software_version": "9.9.9",
                "wheel_signs": dict(WHEEL_SIGNS),
                "topic_names": getattr(self, "topic_names", {}),
                "camera_calibration_path": str(
                    Path(self._tmp.name) / "camera_calibration.json"
                ),
            },
            cameras=self.readers,
            scene=self.scene,
            tick_reader=SnapshotReader(self.tick_writer.name, tick_ring_keys(), []),
            joint_reader=SnapshotReader(self.joint_writer.name, joint_ring_keys(), []),
        )

    def tearDown(self) -> None:
        self.session.close()
        self.tick_writer.close()
        self.joint_writer.close()
        self._tmp.cleanup()

    def _record(self, start: float) -> None:
        """Feed 0.5 s of cameras (one dropped frame), joints, ticks and wheels."""
        n = 2 * len(Joint)
        for i in range(120):  # 240 Hz
            t = start + i / 240
            positions = np.full(n, 0.01 * i)
            # The core applies each 120 Hz command (seqs 2k+1 left, 2k+2
            # right, sent at start + k/120 + 0.5 ms) 2 ms after it is sent;
            # until the first, nothing has been applied.
            k = math.floor((t - start - 0.0025) * 120)
            applied = (
                [np.nan] * 4
                if k < 0
                else [
                    2 * k + 1,
                    start + k / 120 + 0.0025,
                    2 * k + 2,
                    start + k / 120 + 0.0025,
                ]
            )
            self.joint_writer.write_values(
                t,
                np.concatenate([positions, np.ones(n), np.full(n, 2.0), applied]),
            )
        frame = SimpleNamespace(
            seq=0,
            t_host=None,
            l_ee=SimpleNamespace(
                position=SimpleNamespace(x=0.1, y=0.2, z=0.3),
                quaternion=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            r_ee=SimpleNamespace(
                position=SimpleNamespace(x=-0.1, y=0.2, z=0.3),
                quaternion=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            l_grip=1.0,
            r_grip=0.5,
        )
        signs = np.array([WHEEL_SIGNS[i] for i in (1, 2, 3, 4)])
        speed = np.array(wheel_rims(0.2, 0.0, 0.0)) / WHEEL_RADIUS_M * signs
        command = {f"{s}_{j.value}.pos": 0.05 for s in ("left", "right") for j in Joint}
        for i in range(60):  # 120 Hz
            t = start + i / 120
            frame.seq = i // 2  # the headset sends at half the tick rate
            wheel_t = start + (i // 3) * 0.025
            jelly = SimpleNamespace(
                wheel_sample=SimpleNamespace(
                    positions=tuple(speed * (wheel_t - start) + 7.0),
                    velocities=tuple(speed),
                    at=wheel_t,
                ),
                wheel_speeds=list(speed),
                has_wheels=True,
                lift_raised=1.0,
                parked=False,
                # Jelly gets a new command every other tick.
                latest_command=(100 + i // 2, start + (i // 2) / 60, 0.2, 0.0, 0.0, 0),
            )
            self.tick_writer.write(
                pack_tick(
                    command,
                    frame,
                    jelly,
                    command_id=1000 + i,
                    issue_t=t + 0.0005,
                    arm_mode="impedance",
                    target_seqs=(2 * i + 1, 2 * i + 2),
                ),
                {},
                t,
            )
        for src, reader in self.readers.items():
            for i in range(30):
                if src == "left_arm" and i == 10:
                    continue  # a dropped exposure stays a gap
                reader.queue.append((_au(i), start + 0.002 + i / 60, 0.0))

    def test_episode_records_every_topic_once_per_sample(self) -> None:
        self.session.start("pick the cup")
        start = time.perf_counter()
        self._record(start)
        deadline = time.monotonic() + 5.0
        while self.session.frame_count() < 29 and time.monotonic() < deadline:
            time.sleep(0.01)
        frames, error = self.session.stop_capture(trim_after=start + 0.4)
        self.assertIsNone(error)
        self.assertEqual(frames, 29)
        self.assertEqual(self.session.save("failure", {}), 1)
        self.assertEqual(episode_indices(self.root), [0])
        path = self.root / "episode_000000.mcap"
        self.assertTrue(path.exists())
        self.assertEqual(list(self.root.glob("*.partial")), [])

        by_topic: dict[str, list] = {}
        for msg in read_ros2_messages(str(path)):
            by_topic.setdefault(msg.channel.topic, []).append(msg)

        for topic in (
            "/head/left/video",
            "/head/right/video",
            "/wrist/left/video",
            "/wrist/right/video",
        ):
            self.assertIn(topic, by_topic)
        self.assertEqual(len(by_topic["/head/left/video"]), 30)
        self.assertEqual(len(by_topic["/wrist/left/video"]), 29)
        # Each frame at its own exposure time, nothing repeated.
        stamps = [m.log_time_ns for m in by_topic["/wrist/left/video"]]
        self.assertEqual(len(set(stamps)), len(stamps))
        gaps = np.diff(stamps) / 1e9
        self.assertAlmostEqual(gaps.max(), 2 / 60, places=6)
        video = by_topic["/head/right/video"][0].ros_msg
        self.assertEqual(video.format, "h265")
        self.assertEqual(video.frame_id, "head_camera_right_optical")
        self.assertEqual(video.data, _au(0))

        info = by_topic["/head/left/camera_info"][0].ros_msg
        self.assertEqual((info.width, info.height), (960, 600))
        self.assertEqual(info.distortion_model, "plumb_bob")
        wrist = by_topic["/wrist/left/camera_info"][0].ros_msg
        self.assertEqual(wrist.distortion_model, "equidistant")
        self.assertEqual(len(wrist.d), 4)
        # No calibration file for the right wrist: no camera info, but video.
        self.assertNotIn("/wrist/right/camera_info", by_topic)
        self.assertEqual(
            len(by_topic["/head/left/camera_info"]), len(by_topic["/head/left/video"])
        )

        joints = by_topic["/joint_states"]
        self.assertEqual(len(joints), 120)
        self.assertEqual(joints[0].ros_msg.name[:2], ["left_s1_0", "left_s2_0"])
        self.assertIn("right_gripper", joints[0].ros_msg.name)
        self.assertEqual(len(joints[0].ros_msg.velocity), 16)
        commands = by_topic["/joint_command"]
        infos = [m.ros_msg for m in by_topic["/joint_command/info"]]
        self.assertEqual(len(commands), 60)
        self.assertEqual(commands[0].ros_msg.position[0], 0.05)
        self.assertEqual([c.command_id for c in infos], list(range(1000, 1060)))
        # Each info message shares its command's stamp: the issue time.
        self.assertEqual(
            [m.log_time_ns for m in commands],
            [m.log_time_ns for m in by_topic["/joint_command/info"]],
        )
        first = infos[0]
        self.assertEqual(
            commands[0].log_time_ns,
            first.issue_timestamp.sec * 10**9 + first.issue_timestamp.nanosec,
        )
        self.assertEqual(first.name, commands[0].ros_msg.name)
        self.assertEqual(first.name[7], "left_gripper")
        self.assertEqual(first.target_device_id[0], "left_arm")
        self.assertEqual(first.target_device_id[7], "left_gripper")
        self.assertEqual(first.control_mode[0], "impedance")
        self.assertEqual(first.control_mode[7], "position_force")

        # Applied 2 ms after it was sent (issue is 0.5 ms after the tick).
        def ns(t) -> int:
            return t.sec * 10**9 + t.nanosec

        for info in infos:
            delay = ns(info.application_timestamp[0]) - ns(info.issue_timestamp)
            self.assertAlmostEqual(delay / 1e6, 2.0, delta=0.01)
            self.assertEqual(
                info.application_timestamp[0], info.application_timestamp[7]
            )
        self.assertEqual(len(infos[0].application_timestamp), len(infos[0].name))
        base = [m.ros_msg for m in by_topic["/base/command"]]
        self.assertEqual([b.command_id for b in base], list(range(100, 130)))
        self.assertEqual((base[0].control_mode, base[0].vx), ("velocity", 0.2))
        self.assertEqual(len(by_topic["/ee_pose/left"]), 120)
        self.assertEqual(len(by_topic["/ee_target/right"]), 60)
        self.assertEqual(by_topic["/ee_pose/left"][0].ros_msg.header.frame_id, "root")
        # One message per headset frame, not per tick.
        self.assertEqual(len(by_topic["/teleop/left/pose"]), 30)
        self.assertEqual(len(by_topic["/teleop/joy"]), 30)

        wheels = by_topic["/base/wheel_states"]
        self.assertEqual(len(wheels), 20)  # one per distinct wheel sample
        self.assertEqual(wheels[0].ros_msg.position, [0.0, 0.0, 0.0, 0.0])
        odom = by_topic["/base/odom"][-1].ros_msg
        self.assertEqual(
            (odom.header.frame_id, odom.child_frame_id), ("world", "jelly")
        )
        self.assertAlmostEqual(odom.twist.twist.linear.x, 0.2, places=6)
        first_pose = by_topic["/base/odom"][0].ros_msg.pose.pose.position
        self.assertAlmostEqual(
            odom.pose.pose.position.x - first_pose.x, 0.2 * 0.475, places=6
        )
        self.assertEqual(by_topic["/base/lift"][0].ros_msg.position, [0.0])
        self.assertEqual(len(by_topic["/base/cmd_vel"]), 1)

        tf_children = {
            t.child_frame_id for m in by_topic["/tf"] for t in m.ros_msg.transforms
        }
        self.assertTrue({"left_s2", "left_finger_1", "jelly", "root"} <= tf_children)
        static = by_topic["/tf_static"][0].ros_msg.transforms
        static_children = {t.child_frame_id for t in static}
        self.assertIn("left_wrist_camera_optical", static_children)
        self.assertIn("head_camera_right_optical", static_children)
        self.assertFalse(tf_children & static_children)
        self.assertIn("<robot", by_topic["/robot_description"][0].ros_msg.data)

        with path.open("rb") as f:
            metadata = {r.name: r.metadata for r in make_reader(f).iter_metadata()}
        self.assertEqual(metadata["episode"]["task"], "pick the cup")
        self.assertEqual(metadata["episode"]["operator"], "Sam")
        self.assertEqual(metadata["episode"]["outcome"], "failure")
        self.assertNotEqual(metadata["episode"]["end_gesture_time"], "")
        self.assertEqual(metadata["robot"]["serial"], "HUB123")
        self.assertEqual(metadata["robot"]["software_version"], "9.9.9")
        self.assertEqual(metadata["recording"]["video_codec"], "h265")
        self.assertEqual(metadata["recording"]["wrist_left.frames"], "29")
        self.assertEqual(metadata["recording"]["wrist_left.gaps"], "1")
        self.assertEqual(metadata["recording"]["wrist_left.missing_frames"], "1")
        self.assertEqual(metadata["recording"]["joint_samples"], "120")
        self.assertEqual(metadata["recording"]["commands_applied"], "60")
        self.assertEqual(
            metadata["recording"]["commands_without_application_time"], "0"
        )
        self.assertNotIn("calibration_passed", metadata["calibration"])
        self.assertEqual(metadata["calibration"]["calibration_status"], "uncalibrated")
        self.assertEqual(metadata["calibration"]["calibration_id"], "")
        self.assertIn(
            "no camera calibration", metadata["calibration"]["calibration_problem"]
        )
        self.assertEqual(len(metadata["calibration"]["intrinsics_id"]), 16)
        self.assertNotEqual(metadata["calibration"]["intrinsics_date"], "")
        self.assertIn(
            "head_camera_right_optical=zed_factory",
            metadata["calibration"]["tf_sources"],
        )
        self.assertEqual(len(metadata["episode"]["uuid"]), 36)
        self.assertEqual(metadata["calibration"]["wrist_right.file"], "missing")

        from almond_axol.recording.mcap.check import check

        results = {r.item: r for r in check(path)}
        for item in (
            "/head/left/video codec",
            "/head/left/video keyframes",
            "/wrist/left/video timestamps",
            "/wrist/left/video repeats",
            "TF to left_wrist_camera_optical",
            "TF to head_camera_right_optical",
            "head stereo",
            "/base/odom",
            "/joint_command",
            "/joint_command/info",
            "/wrist/left/camera_info",
            "episode.operator",
        ):
            self.assertEqual(results[item].status, "PASS", (item, results[item]))
        # 120 samples over half a second is 240 Hz.
        self.assertEqual(results["/joint_states"].status, "PASS")
        self.assertEqual(results["/wrist/right/camera_info"].status, "WARN")
        # Without a camera calibration file the mounts are CAD: a warning.
        self.assertEqual(results["calibration"].status, "WARN")
        for item in ("/tf_static", "/tf_static rotations", "TF children", "TF root"):
            self.assertEqual(results[item].status, "PASS", (item, results[item]))
        self.assertFalse([r for r in results.values() if r.status == "FAIL"])

    def _record_and_check(
        self, keep: set[str], extra: tuple[str, bytes] | None = None
    ) -> dict:
        from almond_axol.recording.mcap.check import check

        self.session.start("task")
        start = time.perf_counter()
        self._record(start)
        for src, reader in self.readers.items():
            if src not in keep:
                reader.queue.clear()
        if extra is not None:
            self.readers[extra[0]].queue.append((extra[1], start + 5.0, 0.0))
        time.sleep(0.05)
        self.session.stop_capture()
        self.session.save(None, {})
        return {r.item: r for r in check(self.root / "episode_000000.mcap")}

    def test_head_imu_samples_are_recorded_during_the_take(self) -> None:
        import socket as socketlib

        from almond_axol.recording.mcap.check import check
        from almond_axol.recording.mcap.imu import PACKET, ImuReceiver

        path = str(Path(self._tmp.name) / "imu.sock")
        self.session.imus = {
            "/head/imu": (ImuReceiver(path), "head_camera_left_optical")
        }
        sender = socketlib.socket(socketlib.AF_UNIX, socketlib.SOCK_DGRAM)
        wall_minus_perf = time.time() - time.perf_counter()

        def send(perf: float) -> None:
            ts = int((perf + wall_minus_perf) * 1e9)
            sender.sendto(PACKET.pack(b"AXIM", 1, ts, 0, -9.8, 0, 0, 0, 10, 30), path)

        send(time.perf_counter() - 0.5)  # before the take: not recorded
        time.sleep(0.05)
        self.session.start("task")
        start = time.perf_counter()
        self._record(start)
        for i in range(100):  # 0.5 s at 200 Hz
            send(start + 0.001 + i * 0.005)
        deadline = time.monotonic() + 3.0
        while (
            self.session.episode.imu_samples.get("/head/imu", 0) < 100
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        sender.close()
        self.session.stop_capture()
        self.session.save(None, {})
        episode = self.root / "episode_000000.mcap"
        imu = [
            m
            for m in read_ros2_messages(str(episode))
            if m.channel.topic == "/head/imu"
        ]
        self.assertEqual(len(imu), 100)
        self.assertAlmostEqual(imu[0].ros_msg.angular_velocity.z, math.radians(10), 6)
        with episode.open("rb") as f:
            metadata = {r.name: r.metadata for r in make_reader(f).iter_metadata()}
        self.assertEqual(metadata["recording"]["head_imu.samples"], "100")
        results = {r.item: r for r in check(episode)}
        self.assertEqual(results["/head/imu"].status, "PASS")
        self.assertIn("200 Hz", results["/head/imu"].detail)

    def test_renamed_topics_are_written_and_checked(self) -> None:
        self.session.topic_names = {
            "/head/left/video": "/top-left-camera/image-raw",
            "/joint_states": "/robot/joints",
        }
        results = self._record_and_check(set(self.readers))
        path = self.root / "episode_000000.mcap"
        topics = {m.channel.topic for m in read_ros2_messages(str(path))}
        self.assertIn("/top-left-camera/image-raw", topics)
        self.assertIn("/robot/joints", topics)
        self.assertNotIn("/head/left/video", topics)
        self.assertNotIn("/joint_states", topics)
        self.assertIn("/head/right/video", topics)
        with path.open("rb") as f:
            metadata = {r.name: r.metadata for r in make_reader(f).iter_metadata()}
        self.assertEqual(
            metadata["recording"]["topic_names"],
            "/head/left/video=/top-left-camera/image-raw, /joint_states=/robot/joints",
        )
        self.assertEqual(
            metadata["recording"]["head_left.topic"], "/top-left-camera/image-raw"
        )
        # The checker maps the names back and finds nothing wrong.
        self.assertEqual(results["/head/left/video keyframes"].status, "PASS")
        self.assertEqual(results["/joint_states"].status, "PASS")
        self.assertFalse(
            [r for r in results.values() if r.status == "FAIL"],
            [r for r in results.values() if r.status == "FAIL"],
        )

    def test_checker_accepts_any_camera_set(self) -> None:
        results = self._record_and_check({"left_arm"})
        self.assertEqual(results["cameras"].detail, "/wrist/left")
        self.assertFalse([r for r in results.values() if r.status == "FAIL"])
        self.assertNotIn("head stereo", results)

    def test_checker_fails_a_predictive_frame(self) -> None:
        trail = b"\x00\x00\x01\x02\x01\xbb"  # an H.265 TRAIL_R picture
        results = self._record_and_check({"right_arm"}, ("right_arm", trail))
        self.assertEqual(results["/wrist/right/video keyframes"].status, "FAIL")

    def test_a_command_the_core_never_applies_is_written_as_unknown(self) -> None:
        self.session.start("task")
        start = time.perf_counter()
        command = {f"{s}_{j.value}.pos": 0.0 for s in ("left", "right") for j in Joint}
        for i in range(2):
            self.tick_writer.write(
                pack_tick(
                    command,
                    command_id=i + 1,
                    issue_t=start + i,
                    target_seqs=(10 + i, 20 + i),
                ),
                {},
                start + i,
            )
        # Only the first command's targets are applied; the second is
        # superseded (or rejected) and never reported.
        n = 2 * len(Joint)
        self.joint_writer.write_values(
            start + 0.01,
            np.concatenate([np.zeros(3 * n), [10, start + 0.002, 20, start + 0.003]]),
        )
        time.sleep(0.05)
        self.session.stop_capture()
        self.session.save(None, {})
        path = self.root / "episode_000000.mcap"
        infos = [
            m.ros_msg
            for m in read_ros2_messages(str(path))
            if m.channel.topic == "/joint_command/info"
        ]
        self.assertEqual([i.command_id for i in infos], [1, 2])
        left, right = (
            infos[0].application_timestamp[0],
            infos[0].application_timestamp[8],
        )
        self.assertGreater(left.sec, 0)
        self.assertGreater(right.sec, 0)
        self.assertNotEqual(left, right)
        self.assertTrue(
            all(t.sec == 0 and t.nanosec == 0 for t in infos[1].application_timestamp)
        )

    def test_samples_before_the_start_are_not_recorded(self) -> None:
        n = len(joint_ring_keys())
        self.joint_writer.write_values(0.5, np.zeros(n))
        self.readers["left_arm"].queue.append((_au(99), 0.5, 0.0))
        self.session.start("task")
        self.session.stop_capture()
        self.session.save(None, {})
        path = self.root / "episode_000000.mcap"
        topics = {m.channel.topic for m in read_ros2_messages(str(path))}
        self.assertNotIn("/joint_states", topics)
        self.assertNotIn("/wrist/left/video", topics)

    def test_cancel_leaves_no_file_and_numbering_continues(self) -> None:
        self.session.start("a")
        self.session.stop_capture()
        self.session.save(None, {})
        self.session.start("b")
        self.session.stop_capture()
        self.session.cancel()
        self.assertEqual(
            sorted(p.name for p in self.root.iterdir()), ["episode_000000.mcap"]
        )
        self.session.start("c")
        self.session.stop_capture()
        self.assertEqual(self.session.save(None, {}), 2)
        self.assertEqual(episode_indices(self.root), [0, 1])

    def test_capture_error_refuses_the_save(self) -> None:
        self.session.start("a")

        def boom() -> list:
            raise RuntimeError("camera pipeline failed")

        self.readers["left_arm"].drain_aus = boom  # type: ignore[method-assign]
        deadline = time.monotonic() + 5.0
        while self.session.capture_error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        _frames, error = self.session.stop_capture()
        self.assertIn("camera pipeline failed", error or "")
        with self.assertRaises(RuntimeError):
            self.session.save(None, {})
        self.assertEqual(list(self.root.iterdir()), [])


class TopicNamesTest(unittest.TestCase):
    def test_parses_pairs_and_ignores_blank_and_identity(self) -> None:
        from almond_axol.recording.mcap.topics import parse_topic_names

        self.assertEqual(parse_topic_names(""), {})
        self.assertEqual(
            parse_topic_names(
                " /head/left/video = /top-left-camera/image-raw ,\n"
                "/joint_states=/robot/joints,, /tf=/tf"
            ),
            {
                "/head/left/video": "/top-left-camera/image-raw",
                "/joint_states": "/robot/joints",
            },
        )

    def test_rejects_malformed_pairs(self) -> None:
        from almond_axol.recording.mcap.topics import parse_topic_names

        for text in (
            "/joint_states",
            "/joint_states=",
            "joint_states=/a",
            "/joint_states=/a b",
            "/joint_states=/a/",
            "/joint_states=/a,/joint_states=/b",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_topic_names(text)

    def test_rejects_unknown_topics_and_clashes(self) -> None:
        from almond_axol.recording.mcap.topics import (
            default_topics,
            topic_names_problem,
        )

        available = default_topics(["/head/left", "/wrist/left"])
        self.assertIn("/head/left/camera_info", available)
        self.assertIsNone(
            topic_names_problem({"/head/left/video": "/cam/left"}, available)
        )
        problem = topic_names_problem({"/head/right/video": "/x"}, available)
        self.assertIn("no /head/right/video topic", problem or "")
        problem = topic_names_problem({"/joint_states": "/tf"}, available)
        self.assertIn("would both be named /tf", problem or "")
        # Swapping two names is fine.
        self.assertIsNone(
            topic_names_problem(
                {"/ee_pose/left": "/ee_pose/right", "/ee_pose/right": "/ee_pose/left"},
                available,
            )
        )


class CollectDataSetupTest(unittest.TestCase):
    def _cfg(self, **overrides):
        from almond_axol.cli.collect_data import CollectDataConfig

        cfg = CollectDataConfig(repo_id="me/data", task="t", format="mcap")
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    def test_refuses_mantis_and_lerobot_folders(self) -> None:
        from almond_axol.cli.collect_data import mcap_setup_problem

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            problem = mcap_setup_problem(
                self._cfg(mantis=True), dataset_root=root, dataset_resolution="SVGA"
            )
            self.assertIn("Axol only", problem or "")
            (root / "meta").mkdir()
            (root / "meta" / "info.json").write_text("{}")
            problem = mcap_setup_problem(
                self._cfg(), dataset_root=root, dataset_resolution="SVGA"
            )
            self.assertIn("LeRobot dataset", problem or "")

    def test_refuses_malformed_topic_names_before_hardware(self) -> None:
        from almond_axol.cli.collect_data import mcap_setup_problem

        with tempfile.TemporaryDirectory() as directory:
            problem = mcap_setup_problem(
                self._cfg(mcap_topics="/joint_states"),
                dataset_root=Path(directory),
                dataset_resolution="SVGA",
            )
        self.assertIn("default=new", problem or "")

    def test_refuses_a_capture_smaller_than_the_recording(self) -> None:
        from almond_axol.cli.collect_data import mcap_setup_problem

        cfg = self._cfg()
        camera = SimpleNamespace(record=True, resolution_name=lambda: "SVGA")
        cfg.robot_config = SimpleNamespace(cameras={"left_arm": camera})
        with tempfile.TemporaryDirectory() as directory:
            problem = mcap_setup_problem(
                cfg, dataset_root=Path(directory), dataset_resolution="HD1200"
            )
            self.assertIn("smaller than the Recording resolution", problem or "")
            camera.resolution_name = lambda: "HD1200"
            self.assertIsNone(
                mcap_setup_problem(
                    cfg, dataset_root=Path(directory), dataset_resolution="HD1200"
                )
            )

    def test_recording_format_setting_reaches_collect_data(self) -> None:
        from almond_axol.serve.commands import COMMANDS, build_argv
        from almond_axol.serve.settings import SettingsStore, settings_schema

        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / "settings.json")
            store.update(
                values={
                    "recording.format": "mcap",
                    "recording.mcap_topics": "/joint_states=/robot/joints",
                    "recording.mcap_video_codec": "h265",
                }
            )
            args = store.merged_args("collect-data", {})
            self.assertEqual(args["format"], "mcap")
            self.assertEqual(args["mcap_video_codec"], "h265")
            self.assertEqual(args["mcap_topics"], "/joint_states=/robot/joints")
            argv = build_argv("collect-data", {**args, "repo_id": "a/b", "task": "t"})
            self.assertIn("--format", argv)
        (entry,) = [
            setting
            for category in settings_schema()
            for setting in category["settings"]
            if setting["key"] == "recording.format"
        ]
        self.assertEqual(entry["options"], ["lerobot", "mcap"])
        self.assertEqual(entry["default"], "lerobot")
        (codec,) = [
            setting
            for category in settings_schema()
            for setting in category["settings"]
            if setting["key"] == "recording.mcap_video_codec"
        ]
        self.assertEqual(codec["options"], ["h264", "h265"])
        self.assertEqual(codec["default"], "h264")
        self.assertIn("operator", COMMANDS["collect-data"].per_run_fields)
        from almond_axol.cli.collect_data import CollectDataConfig

        self.assertEqual(
            CollectDataConfig(repo_id="a/b", task="t").mcap_video_codec, "h264"
        )


if __name__ == "__main__":
    unittest.main()


class StateListenerTest(unittest.TestCase):
    def test_listener_gets_both_arms_and_a_failing_one_is_dropped(self) -> None:
        from almond_axol.rt.robot import Axol

        fake = SimpleNamespace(_state_listeners=[])
        fake.remove_state_listener = lambda fn: Axol.remove_state_listener(fake, fn)
        fake._applied_targets = lambda: (3.0, 1.25, 4.0, 1.5)
        seen: list[tuple] = []

        def good(ts, positions, velocities, torques, applied) -> None:
            seen.append((ts, positions, velocities, torques, applied))

        def bad(*_args) -> None:
            raise ValueError("boom")

        Axol.add_state_listener(fake, good)  # type: ignore[arg-type]
        Axol.add_state_listener(fake, bad)  # type: ignore[arg-type]
        left = SimpleNamespace(velocities=np.full(8, 2.0, dtype=np.float32))
        right = SimpleNamespace(velocities=np.full(8, 3.0, dtype=np.float32))
        snapshot = (1.5, np.zeros(8), np.ones(8), np.full(8, 4.0), np.full(8, 5.0))
        with self.assertLogs("almond_axol.rt.robot", level="ERROR"):
            Axol._notify_state_listeners(fake, snapshot, left, right)  # type: ignore[arg-type]
        self.assertEqual(fake._state_listeners, [good])
        ts, positions, velocities, torques, applied = seen[0]
        self.assertEqual(ts, 1.5)
        self.assertEqual(applied, (3.0, 1.25, 4.0, 1.5))
        self.assertEqual(positions.tolist(), [0.0] * 8 + [1.0] * 8)
        self.assertEqual(velocities.tolist(), [2.0] * 8 + [3.0] * 8)
        self.assertEqual(torques.tolist(), [4.0] * 8 + [5.0] * 8)


class ImuTest(unittest.TestCase):
    def _packet(self, ts_ns: int, gyro_deg: tuple = (90.0, 0.0, -45.0)) -> bytes:
        from almond_axol.recording.mcap.imu import PACKET

        return PACKET.pack(b"AXIM", 1, ts_ns, 0.1, -9.8, 0.2, *gyro_deg, 31.5)

    def test_packet_layout_matches_the_plugin_patch(self) -> None:
        from almond_axol.cli.gst import build_zed
        from almond_axol.recording.mcap.imu import PACKET, parse_packet

        # The C struct AxolImuSample is packed: 4 + 4 + 8 + 3*4 + 3*4 + 4 bytes.
        self.assertEqual(PACKET.size, 44)
        (imu_patch,) = (
            p
            for p in build_zed._PATCHES  # noqa: SLF001
            if p.name == "zed-gstreamer-imu-socket.patch"
        )
        text = imu_patch.read_text(encoding="utf-8")
        self.assertIn('g_param_spec_string("imu-socket"', text)
        self.assertIn("getSensorsDataBatch", text)
        self.assertIn("ZED_SDK_MINOR_VERSION >= 1", text)
        self.assertIn("MSG_DONTWAIT", text)

        sample = parse_packet(self._packet(2_000_000_000), wall_minus_perf=1.5)
        assert sample is not None
        self.assertAlmostEqual(sample.perf, 0.5)
        self.assertAlmostEqual(sample.angular_velocity[0], math.pi / 2, places=6)
        self.assertAlmostEqual(sample.angular_velocity[2], -math.pi / 4, places=6)
        self.assertAlmostEqual(sample.acceleration[1], -9.8, places=5)
        self.assertIsNone(parse_packet(b"XXXX" + self._packet(1)[4:], 0.0))
        self.assertIsNone(parse_packet(self._packet(1)[:-1], 0.0))

    def test_receiver_collects_samples_sent_to_its_socket(self) -> None:
        import socket as socketlib

        from almond_axol.recording.mcap.imu import ImuReceiver

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "imu.sock")
            receiver = ImuReceiver(path)
            try:
                sender = socketlib.socket(socketlib.AF_UNIX, socketlib.SOCK_DGRAM)
                now_ns = time.time_ns()
                for i in range(5):
                    sender.sendto(self._packet(now_ns + i * 5_000_000), path)
                # A stale sample (an hour old) and a malformed one are rejected.
                sender.sendto(self._packet(now_ns - 3600 * 10**9), path)
                sender.sendto(b"junk", path)
                sender.close()
                deadline = time.monotonic() + 2.0
                samples: list = []
                while len(samples) < 5 and time.monotonic() < deadline:
                    samples += receiver.drain()
                    time.sleep(0.01)
                while receiver.rejected < 2 and time.monotonic() < deadline:
                    time.sleep(0.01)
            finally:
                receiver.close()
            self.assertFalse(Path(path).exists())
        self.assertEqual(len(samples), 5)
        self.assertEqual(receiver.rejected, 2)
        steps = np.diff([s.perf for s in samples])
        np.testing.assert_allclose(steps, 0.005, atol=1e-6)

    def test_stereo_pipeline_sends_imu_only_when_asked(self) -> None:
        from almond_axol.video.gst_zed import ZedGstStereoCamera

        with_imu = ZedGstStereoCamera(
            51617969, "HD1200", 60, imu_socket="/tmp/axol-imu-1-2.sock"
        )._pipeline_str()
        self.assertIn(
            "do-timestamp=false imu-socket=/tmp/axol-imu-1-2.sock !", with_imu
        )
        without = ZedGstStereoCamera(51617969, "HD1200", 60)._pipeline_str()
        self.assertNotIn("imu-socket", without)

    def test_scene_lists_one_imu_per_camera(self) -> None:
        cameras = {
            src: CameraSource.from_meta(
                src,
                {
                    **_meta("overhead", eye, HEAD_SERIAL, True),
                    "imu_socket": "/tmp/imu.sock",
                    "imu_requested": True,
                },
            )
            for src, eye in (("overhead_left", "left"), ("overhead_right", "right"))
        }
        scene = Scene.build(
            urdf_path(AxolModel.MOBILE),
            has_gripper=True,
            jelly_wheels=False,
            jelly_lift=False,
            cameras=cameras,
            calibration_dir=Path("/nonexistent"),
        )
        self.assertEqual(
            scene.imus(), {"/head/imu": ("/tmp/imu.sock", "head_camera_left_optical")}
        )

    def test_imu_message_decodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.mcap"
            f = EpisodeFile(path, clock_offset_ns=0)
            f.write(
                "/head/imu",
                "sensor_msgs/Imu",
                messages.imu(
                    5, "head_camera_left_optical", (0.1, 0.2, 0.3), (0, -9.8, 0)
                ),
                5,
            )
            f.finish()
            (msg,) = read_ros2_messages(str(path))
        imu = msg.ros_msg
        self.assertEqual(imu.header.frame_id, "head_camera_left_optical")
        self.assertEqual(imu.orientation_covariance[0], -1.0)
        self.assertEqual(imu.angular_velocity.y, 0.2)
        self.assertEqual(imu.linear_acceleration.y, -9.8)


class AppliedTargetTest(unittest.TestCase):
    def test_link_parses_the_applied_trailer(self) -> None:
        import struct as structlib

        from almond_axol.rt.link import RtLink

        slots = b"".join(structlib.pack("<3dI", 0.0, 0.0, 0.0, 0) for _ in range(8))
        base = b"F" + bytes([0, 0b1]) + slots
        self.assertIsNone(RtLink._parse_applied(base))  # no trailer
        self.assertIsNone(RtLink._parse_applied(base + structlib.pack("<BII", 0, 9, 9)))
        before = time.time()
        seq, ts = RtLink._parse_applied(base + structlib.pack("<BII", 1, 77, 2500))
        self.assertEqual(seq, 77)
        self.assertLessEqual(before - 0.0025, ts)
        self.assertLessEqual(ts, time.time() - 0.0025)
        # The slot parse ignores the trailer.
        side, parsed = RtLink._parse_feedback(
            base + structlib.pack("<BII", 1, 77, 2500)
        )
        self.assertEqual((side, set(parsed)), (0, {0}))

    def test_robot_reports_applied_targets_on_the_perf_clock(self) -> None:
        from almond_axol.rt.robot import Axol

        wall = time.time()
        fake = SimpleNamespace(_link=SimpleNamespace(applied={1: (42, wall - 0.01)}))
        left_seq, left_t, right_seq, right_t = Axol._applied_targets(fake)  # type: ignore[arg-type]
        self.assertTrue(math.isnan(left_seq) and math.isnan(left_t))
        self.assertEqual(right_seq, 42.0)
        self.assertAlmostEqual(right_t, time.perf_counter() - 0.01, delta=0.005)

    def test_sent_seqs_are_tracked_per_arm(self) -> None:
        from almond_axol.rt.robot import Axol

        sent = []
        fake = SimpleNamespace(
            _seq=0,
            _sent_seq={},
            _link=SimpleNamespace(
                send_target=lambda side, seq, cmds: sent.append((side, seq))
            ),
        )
        left, right = Axol._make_sink(fake, 0), Axol._make_sink(fake, 1)  # type: ignore[arg-type]
        left([]), right([]), left([])
        self.assertEqual(sent, [(0, 1), (1, 2), (0, 3)])
        self.assertEqual(Axol.last_target_seqs.fget(fake), (3, 2))  # type: ignore[union-attr]


class StreamResolutionTest(unittest.TestCase):
    """Collect data captures at the larger resolution and scales the headset."""

    def _merge(self, scaled: bool) -> dict:
        from almond_axol.serve.runner import OperationRunner

        spec = {
            "serials": {"overhead": "101", "left_arm": "102", "right_arm": ""},
            "stream_resolution": "SVGA",
            "record_resolution": "HD1200",
        }
        with mock.patch("almond_axol.serve.runner.stereo_serials", return_value=set()):
            return OperationRunner()._merge_camera_args(
                {"repo_id": "a/b", "task": "t"},
                spec,
                streams_video=True,
                scaled_stream=scaled,
            )

    def test_collect_data_captures_at_the_recording_resolution(self) -> None:
        args = self._merge(scaled=True)
        self.assertEqual(args["robot_config.cameras.left_arm.width"], 1920)
        self.assertEqual(args["robot_config.cameras.left_arm.height"], 1200)
        self.assertEqual(args["stream_resolution"], "SVGA")
        self.assertEqual(args["dataset_resolution"], "HD1200")

    def test_other_ops_still_capture_at_the_streaming_resolution(self) -> None:
        args = self._merge(scaled=False)
        self.assertEqual(args["robot_config.cameras.left_arm.width"], 960)
        self.assertNotIn("stream_resolution", args)

    def test_only_collect_data_takes_a_separate_stream_resolution(self) -> None:
        from almond_axol.serve.runner import _takes_stream_resolution

        self.assertTrue(_takes_stream_resolution("collect-data"))
        self.assertFalse(_takes_stream_resolution("collect-dagger"))
        self.assertFalse(_takes_stream_resolution("run-policy"))

    def test_headset_branches_scale_down_on_the_gpu(self) -> None:
        from almond_axol.video.gst_zed import ZedGstCamera, ZedGstStereoCamera

        mono = ZedGstCamera(302271563, "HD1200", 60, stream_dims=(960, 600))
        self.assertEqual((mono.stream_width, mono.stream_height), (960, 600))
        pipe = mono._pipeline_str()
        self.assertIn(
            "nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width=960,height=600,"
            "pixel-aspect-ratio=1/1 ! nvv4l2h264enc name=venc",
            pipe,
        )
        stereo = ZedGstStereoCamera(
            51617969, "HD1200", 60, encoded_sbs=True, stream_dims=(960, 600)
        )
        sbs = stereo._pipeline_str()
        self.assertIn(
            "width=1920,height=600,pixel-aspect-ratio=1/1 ! nvv4l2h264enc name=venc_s",
            sbs,
        )
        eyes = ZedGstStereoCamera(
            51617969, "HD1200", 60, encoded_eyes=["left"], stream_dims=(960, 600)
        )._pipeline_str()
        self.assertIn(
            "width=960,height=600,pixel-aspect-ratio=1/1 ! nvv4l2h264enc name=venc_l",
            eyes,
        )

    def test_no_scaling_when_the_stream_is_not_smaller(self) -> None:
        from almond_axol.video.gst_zed import ZedGstCamera

        for dims in (None, (1920, 1200), (3840, 2400)):
            cam = ZedGstCamera(302271563, "HD1200", 60, stream_dims=dims)
            self.assertEqual((cam.stream_width, cam.stream_height), (1920, 1200))
            # An encoded-only camera has no GPU conversion at all.
            self.assertNotIn("nvvidconv", cam._pipeline_str())


def _jelly_scene(directory: Path, gripper: str | None = None) -> Scene:
    (directory / f"SN{HEAD_SERIAL}.conf").write_text(HEAD_CONF)
    (directory / f"SN{LEFT_WRIST_SERIAL}.conf").write_text(FISHEYE_CONF)
    raw_meta = {
        "overhead_left": _meta("overhead", "left", HEAD_SERIAL, True),
        "overhead_right": _meta("overhead", "right", HEAD_SERIAL, True),
        "left_arm": _meta("left_arm", None, LEFT_WRIST_SERIAL, False),
        "right_arm": _meta("right_arm", None, RIGHT_WRIST_SERIAL, False),
    }
    cameras = {src: CameraSource.from_meta(src, m) for src, m in raw_meta.items()}
    return Scene.build(
        urdf_path(AxolModel.MOBILE),
        has_gripper=True,
        jelly_wheels=True,
        jelly_lift=True,
        cameras=cameras,
        calibration_dir=directory,
        gripper=gripper,
    )


def _nudge(transform: np.ndarray, mm: float, deg: float) -> np.ndarray:
    out = transform.copy()
    out[:3, :3] = (
        out[:3, :3]
        @ Rotation.from_euler("xyz", [deg, -deg, deg / 2], degrees=True).as_matrix()
    )
    out[:3, 3] += np.array([mm, -mm, mm / 2]) / 1000.0
    return out


def _calibration_doc(scene: Scene, **overrides) -> dict:
    from almond_axol.camera_calibration.file import document, transform_entry

    tree = scene.tree
    entries = [
        transform_entry(
            "root",
            "head_camera_left_optical",
            _nudge(tree.fk("root", "head_camera_left_optical", {}), 4, 0.5),
            camera_serial=HEAD_SERIAL,
        ),
        transform_entry(
            "left_w2",
            "left_wrist_camera_optical",
            _nudge(tree.fk("left_w2", "left_wrist_camera_optical", {}), 3, 1.0),
            camera_serial=LEFT_WRIST_SERIAL,
        ),
        transform_entry(
            "right_w2",
            "right_wrist_camera_optical",
            _nudge(tree.fk("right_w2", "right_wrist_camera_optical", {}), -2, 0.7),
            camera_serial=RIGHT_WRIST_SERIAL,
        ),
    ]
    doc = document(
        robot_serial="HUB123",
        calibration_id="cam-test-1",
        calibrated_at="2026-11-24T18:15:00Z",
        status="pass",
        transforms=entries,
    )
    doc.update(overrides)
    return doc


class CameraMountsTest(unittest.TestCase):
    """The calibration file placing the cameras in /tf_static."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.scene = _jelly_scene(self.dir)
        self.path = self.dir / "camera_calibration.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _frames(self, doc: dict | None, serial: str | None = "HUB123", imu=None):
        from almond_axol.camera_calibration.file import write
        from almond_axol.recording.mcap.mounts import episode_frames

        if doc is not None:
            write(doc, self.path)
        return episode_frames(self.scene, self.path, serial, imu)

    @staticmethod
    def _lookup(frames, parent: str, child: str) -> np.ndarray:
        """``child`` in ``parent`` through the static frames (shared ancestor)."""
        edges = {f.child: (f.parent, f.transform) for f in frames.frames}

        def up(frame: str) -> list[tuple[str, np.ndarray]]:
            chain = [(frame, np.eye(4))]
            while frame in edges:
                parent_, transform = edges[frame]
                chain.append((parent_, transform @ chain[-1][1]))
                frame = parent_
            return chain

        pc, cc = dict(up(parent)), up(child)
        for link, pose in cc:
            if link in pc:
                return np.linalg.inv(pc[link]) @ pose
        raise AssertionError(f"{parent} and {child} are not connected")

    def test_file_places_every_camera_and_keeps_the_tree_shape(self) -> None:
        from almond_axol.camera_calibration.file import load

        doc = _calibration_doc(self.scene)
        frames = self._frames(doc)
        self.assertEqual(frames.status, "calibrated")
        self.assertEqual(frames.problem, "")
        self.assertEqual(frames.calibration.calibration_id, "cam-test-1")
        sources = frames.sources()
        for frame in (
            "head_camera_left_optical",
            "left_wrist_camera_optical",
            "right_wrist_camera_optical",
        ):
            self.assertEqual(sources[frame], "calibration")
        self.assertEqual(sources["head_camera_right_optical"], "zed_factory")
        # Same parents as the URDF; each child once.
        parents = {f.child: f.parent for f in frames.frames}
        self.assertEqual(parents["head_camera_left_optical"], "head")
        children = [f.child for f in frames.frames]
        self.assertEqual(len(children), len(set(children)))
        for entry in load(self.path).transforms:
            np.testing.assert_allclose(
                self._lookup(frames, entry.parent, entry.child),
                entry.matrix(),
                atol=1e-9,
            )
        # The right eye follows the measured left eye by the factory baseline.
        baseline = np.linalg.inv(
            self._lookup(frames, "head", "head_camera_left_optical")
        ) @ self._lookup(frames, "head", "head_camera_right_optical")
        np.testing.assert_allclose(baseline, self.scene.stereo, atol=1e-12)

    def test_robot_description_matches_tf_static(self) -> None:
        frames = self._frames(_calibration_doc(self.scene))
        urdf = UrdfTree(self.scene.robot_description(frames.edges, frames.added))
        for parent, child in (
            ("root", "head_camera_left_optical"),
            ("left_w2", "left_wrist_camera_optical"),
            ("head", "head_camera_right_optical"),
            ("left_gripper", "left_tcp"),
        ):
            np.testing.assert_allclose(
                urdf.fk(parent, child, {}),
                self._lookup(frames, parent, child),
                atol=1e-9,
                err_msg=child,
            )

    def test_tool_centres_follow_the_gripper(self) -> None:
        from almond_axol.constants import GRIPPER_TIP_OFFSET

        frames = self._frames(None)
        tcp = {f.child: f for f in frames.frames if f.child.endswith("_tcp")}
        self.assertEqual(set(tcp), {"left_tcp", "right_tcp"})
        self.assertEqual(tcp["left_tcp"].parent, "left_gripper")
        self.assertEqual(tcp["left_tcp"].source, "gripper_model")
        np.testing.assert_allclose(tcp["left_tcp"].transform[:3, 3], GRIPPER_TIP_OFFSET)
        parcel = _jelly_scene(self.dir, gripper="parcel_v1")
        self.assertEqual(parcel.tool_frames(), [])

    def test_head_imu_frame_from_the_camera(self) -> None:
        pose = np.eye(4)
        pose[:3, 3] = (0.01, -0.005, 0.002)
        frames = self._frames(None, imu={"head_camera_imu": pose})
        imu = next(f for f in frames.frames if f.child == "head_camera_imu")
        self.assertEqual(imu.parent, "head_camera_left_optical")
        self.assertEqual(imu.source, "zed_factory")
        np.testing.assert_allclose(imu.transform, pose)

    def test_missing_or_wrong_files_record_cad(self) -> None:
        frames = self._frames(None)
        self.assertEqual(frames.status, "uncalibrated")
        self.assertIn("no camera calibration file", frames.problem)
        self.assertEqual(frames.sources()["head_camera_left_optical"], "urdf")

        frames = self._frames(_calibration_doc(self.scene), serial="OTHER")
        self.assertEqual(frames.status, "uncalibrated")
        self.assertIn("not this robot", frames.problem)

        frames = self._frames(_calibration_doc(self.scene, status="fail"))
        self.assertIn("failed calibration", frames.problem)

        self.path.write_text("{not json")
        frames = self._frames(None)
        self.assertIn("not valid JSON", frames.problem)
        self.assertIsNone(frames.calibration)

    def test_swapped_camera_and_impossible_frames_are_skipped(self) -> None:
        from almond_axol.camera_calibration.file import transform_entry

        doc = _calibration_doc(self.scene)
        doc["transforms"][1]["camera_serial"] = 999
        # A wrist camera can't be placed in root: the arm moves in between.
        doc["transforms"].append(transform_entry("root", "right_finger_1", np.eye(4)))
        frames = self._frames(doc)
        self.assertEqual(frames.status, "partial")
        self.assertEqual(frames.sources()["left_wrist_camera_optical"], "urdf")
        reasons = " ".join(frames.skipped)
        self.assertIn("SN999", reasons)
        self.assertIn("right_finger_1", reasons)


class CalibratedSessionTest(unittest.TestCase):
    """A recorded episode with a camera calibration file in use."""

    tearDown = SessionTest.tearDown
    _record = SessionTest._record
    _record_and_check = SessionTest._record_and_check

    def setUp(self) -> None:
        SessionTest.setUp(self)  # type: ignore[arg-type]
        from almond_axol.camera_calibration.file import write

        write(
            _calibration_doc(self.scene),
            Path(self._tmp.name) / "camera_calibration.json",
        )

    def test_checker_passes_the_measured_mounts(self) -> None:
        results = self._record_and_check(set(self.readers))
        self.assertEqual(results["calibration"].status, "PASS", results["calibration"])
        for frame in (
            "head_camera_left_optical",
            "left_wrist_camera_optical",
            "right_wrist_camera_optical",
        ):
            item = f"mount of {frame}"
            self.assertEqual(results[item].status, "PASS", results[item])
        self.assertEqual(results["calibration file"].status, "PASS")
        self.assertFalse([r for r in results.values() if r.status == "FAIL"])

    def test_metadata_and_attachment(self) -> None:
        self._record_and_check(set(self.readers))
        path = self.root / "episode_000000.mcap"
        with open(path, "rb") as f:
            reader = make_reader(f)
            meta = {m.name: dict(m.metadata) for m in reader.iter_metadata()}
            attachments = {a.name: bytes(a.data) for a in reader.iter_attachments()}
        calibration = meta["calibration"]
        self.assertEqual(calibration["calibration_status"], "calibrated")
        self.assertEqual(calibration["calibration_id"], "cam-test-1")
        self.assertEqual(calibration["calibrated_at"], "2026-11-24T18:15:00Z")
        self.assertEqual(calibration["head_left.extrinsics"], "calibration")
        self.assertEqual(calibration["head_right.extrinsics"], "zed_factory")
        import hashlib

        raw = attachments["camera_calibration.json"]
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(), calibration["calibration_file_sha256"]
        )


class EpisodeMetadataTest(unittest.TestCase):
    def test_pairs(self) -> None:
        from almond_axol.recording.mcap.custom_metadata import parse_episode_metadata

        self.assertEqual(
            parse_episode_metadata("task_id=pick-12, site = lab 2\ndifficulty=3"),
            {"task_id": "pick-12", "site": "lab 2", "difficulty": "3"},
        )
        self.assertEqual(parse_episode_metadata("  "), {})
        self.assertEqual(parse_episode_metadata("note=a=b"), {"note": "a=b"})
        for bad, message in (
            ("site", "key=value"),
            ("bad key=1", "not a key"),
            ("a=1, a=2", "twice"),
        ):
            with self.assertRaises(ValueError) as ctx:
                parse_episode_metadata(bad)
            self.assertIn(message, str(ctx.exception))

    def test_setup_refuses_bad_pairs(self) -> None:
        from almond_axol.cli.collect_data import CollectDataConfig, mcap_setup_problem

        cfg = CollectDataConfig(repo_id="me/data", task="t", format="mcap")
        cfg.episode_metadata = "site"
        with tempfile.TemporaryDirectory() as directory:
            problem = mcap_setup_problem(
                cfg, dataset_root=Path(directory), dataset_resolution="SVGA"
            )
        self.assertIn("Episode metadata", problem or "")

    def test_per_run_field(self) -> None:
        from almond_axol.serve.commands import COMMANDS

        self.assertIn("episode_metadata", COMMANDS["collect-data"].per_run_fields)


class ImuPoseTest(unittest.TestCase):
    def test_pose_packet(self) -> None:
        from almond_axol.recording.mcap.imu import POSE_PACKET, parse_pose_packet

        pose = parse_pose_packet(
            POSE_PACKET.pack(b"AXIP", 1, 0.01, -0.005, 0.002, 0, 0, 0, 1)
        )
        np.testing.assert_allclose(pose[:3, 3], [0.01, -0.005, 0.002], atol=1e-7)
        np.testing.assert_allclose(pose[:3, :3], np.eye(3))
        # Not a pose: wrong magic, a non-unit quaternion, an implausible offset.
        self.assertIsNone(
            parse_pose_packet(POSE_PACKET.pack(b"AXIM", 1, 0, 0, 0, 0, 0, 0, 1))
        )
        self.assertIsNone(
            parse_pose_packet(POSE_PACKET.pack(b"AXIP", 1, 0, 0, 0, 0, 0, 0, 2))
        )
        self.assertIsNone(
            parse_pose_packet(POSE_PACKET.pack(b"AXIP", 1, 1, 0, 0, 0, 0, 0, 1))
        )

    def test_receiver_keeps_the_latest_pose(self) -> None:
        import socket as socketlib

        from almond_axol.recording.mcap.imu import POSE_PACKET, ImuReceiver

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "imu.sock")
            receiver = ImuReceiver(path)
            try:
                sender = socketlib.socket(socketlib.AF_UNIX, socketlib.SOCK_DGRAM)
                sender.sendto(
                    POSE_PACKET.pack(b"AXIP", 1, 0.0, 0.01, 0.0, 0, 0, 0, 1), path
                )
                deadline = time.time() + 2.0
                while receiver.pose is None and time.time() < deadline:
                    time.sleep(0.01)
                sender.close()
                self.assertIsNotNone(receiver.pose)
                self.assertAlmostEqual(float(receiver.pose[1, 3]), 0.01, places=6)
                self.assertEqual(receiver.drain(), [])
            finally:
                receiver.close()


class RpyTest(unittest.TestCase):
    def test_round_trip_including_gimbal_lock(self) -> None:
        from almond_axol.recording.mcap.urdf_tree import rpy_from_matrix, rpy_matrix

        rng = np.random.default_rng(3)
        cases = [tuple(rng.uniform(-3, 3, 3)) for _ in range(50)]
        cases += [(0.3, math.pi / 2, -0.4), (0.3, -math.pi / 2, 1.2)]
        for rpy in cases:
            m = rpy_matrix(*rpy)
            np.testing.assert_allclose(rpy_matrix(*rpy_from_matrix(m)), m, atol=1e-9)
