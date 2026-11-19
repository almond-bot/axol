"""Livox Mid-360 protocol, streaming and setup helpers (no hardware)."""

from __future__ import annotations

import json
import os
import socket
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from almond_axol.lidar import livox
from almond_axol.lidar import network as net
from almond_axol.lidar.config import LidarConfig, LidarMount
from almond_axol.lidar.mount import measure_mount
from almond_axol.lidar.recording import LidarRecording
from almond_axol.lidar.stream import (
    FrameAssembler,
    ImuSample,
    LidarFrame,
    LidarStream,
    _ClockMap,
)

# Parameter values a Mid-360 (firmware 35.01.01.08) returned to a status query.
REAL_PARAMS = {
    0x0000: bytes.fromhex("01"),
    0x0004: bytes.fromhex("c0a801c7ffffff00c0a80101"),
    0x0005: bytes.fromhex("ffffffff89db0000"),
    0x0006: bytes.fromhex("c0a80105eddb0000"),
    0x0007: bytes.fromhex("c0a8010551dc0000"),
    0x001A: bytes.fromhex("01"),
    0x001C: bytes.fromhex("01"),
    0x8000: b"ARMCP8T0035699\0\0",
    0x8001: b"DevType:Mid-360 FmType:App FmVer:35010108 BuildTime:2025/06/09".ljust(
        64, b"\0"
    ),
    0x8002: bytes.fromhex("23010108"),
    0x8006: bytes.fromhex("01"),
    0x8007: bytes.fromhex("f7100000"),
    0x800C: bytes.fromhex("00"),
    0x800E: bytes.fromhex("0000"),
    0x8011: bytes(32),
}


class CrcTest(unittest.TestCase):
    def test_crc16_ccitt_false_check_value(self) -> None:
        # The standard check value for CRC-16/CCITT-FALSE.
        self.assertEqual(livox.crc16_ccitt(b"123456789"), 0x29B1)


class ControlFrameTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        data = livox.pack_params({livox.KEY_WORK_TGT_MODE: b"\x01"})
        buf = livox.build_frame(livox.CMD_SET_PARAMS, 7, data)
        self.assertEqual(len(buf), livox.HEADER_SIZE + len(data))
        frame = livox.parse_frame(buf)
        self.assertEqual(
            (frame.cmd_id, frame.seq, frame.data), (livox.CMD_SET_PARAMS, 7, data)
        )

    def test_empty_data_has_zero_crc32(self) -> None:
        buf = livox.build_frame(livox.CMD_DISCOVER, 1)
        self.assertEqual(struct.unpack_from("<I", buf, 20)[0], 0)
        self.assertEqual(livox.parse_frame(buf).data, b"")

    def test_rejects_corruption(self) -> None:
        buf = bytearray(
            livox.build_frame(livox.CMD_GET_PARAMS, 3, b"\x01\x00\x00\x00\x00\x80")
        )
        bad_header = bytearray(buf)
        bad_header[8] ^= 0xFF
        with self.assertRaises(livox.LivoxError):
            livox.parse_frame(bytes(bad_header))
        buf[-1] ^= 0xFF
        with self.assertRaises(livox.LivoxError):
            livox.parse_frame(bytes(buf))
        with self.assertRaises(livox.LivoxError):
            livox.parse_frame(b"\xaa" * 10)

    def test_params_round_trip(self) -> None:
        packed = livox.pack_params(REAL_PARAMS)
        self.assertEqual(struct.unpack_from("<H", packed)[0], len(REAL_PARAMS))
        self.assertEqual(livox.unpack_params(packed, 4), REAL_PARAMS)
        with self.assertRaises(livox.LivoxError):
            livox.unpack_params(struct.pack("<HH", 0x8000, 16) + b"short")


class StatusTest(unittest.TestCase):
    def test_parses_real_reply(self) -> None:
        st = livox.LidarStatus.from_params(REAL_PARAMS)
        self.assertEqual(st.serial, "ARMCP8T0035699")
        self.assertEqual(st.product, "Mid-360")
        self.assertEqual(st.firmware, "35.01.01.08")
        self.assertEqual(
            (st.lidar_ip, st.netmask, st.gateway),
            ("192.168.1.199", "255.255.255.0", "192.168.1.1"),
        )
        self.assertEqual(st.point_host, ("192.168.1.5", 56301))
        self.assertEqual(st.imu_host, ("192.168.1.5", 56401))
        self.assertEqual(st.state_host, ("255.255.255.255", 56201))
        self.assertEqual(st.work_state_name, "sampling")
        self.assertAlmostEqual(st.temperature_c, 43.43)
        self.assertTrue(st.imu_enabled)
        self.assertEqual(st.faults, ())
        self.assertEqual(st.time_sync_name, "none")

    def test_discovery_ack(self) -> None:
        data = (
            bytes([0, 35])
            + b"ARMCP8T0035699\0\0"
            + socket.inet_aton("192.168.1.199")
            + struct.pack("<H", 56100)
        )
        dev = livox.parse_discovery_ack(data)
        self.assertEqual(
            (dev.serial, dev.ip, dev.command_port, dev.model),
            ("ARMCP8T0035699", "192.168.1.199", 56100, "Mid-360"),
        )
        with self.assertRaises(livox.LivoxError):
            livox.parse_discovery_ack(bytes([1, 35]) + bytes(22))


class PacketTest(unittest.TestCase):
    def test_point_packet_round_trip(self) -> None:
        xyz = np.array(
            [[1.0, -2.0, 0.5], [3.25, 0.0, -1.0], [0.0, 0.0, 0.0]], np.float32
        )
        buf = livox.build_point_packet(
            xyz, 123_456_789, time_interval_ns=4800, reflectivity=np.array([10, 20, 30])
        )
        self.assertEqual(len(buf), 36 + 3 * 14)
        pkt = livox.parse_point_packet(buf)
        np.testing.assert_allclose(pkt.xyz, xyz, atol=1e-3)
        self.assertEqual(pkt.header.timestamp_ns, 123_456_789)
        self.assertEqual(list(pkt.reflectivity), [10, 20, 30])
        self.assertEqual(list(pkt.offsets_ns), [0, 2400, 4800])

    def test_imu_packet_round_trip(self) -> None:
        buf = livox.build_imu_packet(
            np.array([0.1, -0.2, 0.3]), np.array([0.0, 0.0, 9.80665]), 42
        )
        pkt = livox.parse_imu_packet(buf)
        np.testing.assert_allclose(pkt.gyro, [0.1, -0.2, 0.3], atol=1e-6)
        np.testing.assert_allclose(pkt.accel, [0.0, 0.0, 9.80665], atol=1e-4)
        with self.assertRaises(livox.LivoxError):
            livox.parse_imu_packet(livox.build_point_packet(np.zeros((1, 3)), 0))

    def test_rejects_truncated_point_packet(self) -> None:
        buf = livox.build_point_packet(np.ones((96, 3)), 0)
        with self.assertRaises(livox.LivoxError):
            livox.parse_point_packet(buf[:-14])


def _packets(n_packets: int, start_ns: int = 1_000_000_000, step_ns: int = 480_000):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n_packets):
        xyz = rng.uniform(-5, 5, (96, 3))
        out.append(
            (
                livox.build_point_packet(
                    xyz, start_ns + i * step_ns, time_interval_ns=470_000
                ),
                xyz,
            )
        )
    return out


class AssemblerTest(unittest.TestCase):
    def test_slices_frames_by_time(self) -> None:
        asm = FrameAssembler(0.1)
        frames = []
        # 0.48 ms per packet: 209 packets per 0.1 s frame.
        for i, (buf, _) in enumerate(_packets(500)):
            f = asm.add(buf, arrival_ns=2_000_000_000 + i * 480_000 + 300_000)
            if f is not None:
                frames.append(f)
        self.assertEqual(len(frames), 2)
        self.assertEqual(len(frames[0]), 209 * 96)
        # The last packet starts before 0.1 s and spans its own 0.47 ms.
        self.assertLess(frames[0].times.max(), 0.101)
        self.assertGreater(frames[0].duration, 0.099)
        self.assertEqual(frames[1].seq, 1)
        # Host stamps follow the arrival clock (offset 1 s minus the delay floor).
        self.assertAlmostEqual(frames[0].stamp, 2.0 - 0.00017, places=3)

    def test_drops_empty_and_close_points(self) -> None:
        asm = FrameAssembler(0.1, min_range=0.2)
        xyz = np.array([[0, 0, 0], [0.1, 0, 0], [1, 1, 1]], float)
        asm.add(livox.build_point_packet(xyz, 0))
        frame = asm.flush()
        self.assertEqual(len(frame), 1)
        np.testing.assert_allclose(frame.xyz[0], [1, 1, 1], atol=1e-3)

    def test_clock_jump_starts_a_new_frame(self) -> None:
        asm = FrameAssembler(0.1)
        asm.add(livox.build_point_packet(np.ones((96, 3)), 5_000_000_000))
        frame = asm.add(livox.build_point_packet(np.ones((96, 3)), 1_000_000_000))
        self.assertIsNotNone(frame)

    def test_bad_packets_are_counted(self) -> None:
        asm = FrameAssembler(0.1)
        self.assertIsNone(asm.add(b"\x00" * 10))
        self.assertIsNone(asm.add(livox.build_imu_packet(np.zeros(3), np.zeros(3), 0)))
        self.assertEqual(asm.bad_packets, 2)

    def test_clock_map_uses_delay_floor(self) -> None:
        clock = _ClockMap()
        clock.update(100, 1_000_100)
        clock.update(200, 1_000_150)
        clock.update(300, 1_000_900)
        self.assertEqual(clock.offset_ns(), 999_950)


class _FakeLidar(threading.Thread):
    """Answers control requests and streams points/IMU to the host on loopback."""

    def __init__(self, host_point_port: int, host_imu_port: int) -> None:
        super().__init__(daemon=True)
        self.ctrl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ctrl.bind(("127.0.0.1", 0))
        self.ctrl.settimeout(0.05)
        self.data = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ports = (host_point_port, host_imu_port)
        self.stop = threading.Event()
        self.sets: list[dict] = []

    def run(self) -> None:
        t = 0
        while not self.stop.is_set():
            try:
                buf, addr = self.ctrl.recvfrom(2048)
                req = livox.parse_frame(buf)
                if req.cmd_id == livox.CMD_SET_PARAMS:
                    self.sets.append(livox.unpack_params(req.data, 4))
                    ack = b"\x00\x00\x00"
                else:
                    ack = (
                        b"\x00"
                        + struct.pack("<H", len(REAL_PARAMS))
                        + livox.pack_params(REAL_PARAMS)[4:]
                    )
                self.ctrl.sendto(
                    livox.build_frame(
                        req.cmd_id, req.seq, ack, cmd_type=livox.CMD_TYPE_ACK
                    ),
                    addr,
                )
            except TimeoutError:
                pass
            for _ in range(20):
                xyz = np.full((96, 3), 2.0)
                self.data.sendto(
                    livox.build_point_packet(xyz, t, time_interval_ns=470_000),
                    ("127.0.0.1", self.ports[0]),
                )
                t += 480_000
            self.data.sendto(
                livox.build_imu_packet(np.zeros(3), np.array([0, 0, 9.81]), t),
                ("127.0.0.1", self.ports[1]),
            )


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class StreamTest(unittest.TestCase):
    def test_configures_lidar_and_streams_frames(self) -> None:
        cfg = LidarConfig(
            lidar_ip="127.0.0.1",
            host_ip="127.0.0.1",
            point_port=_free_port(),
            imu_port=_free_port(),
        )
        fake = _FakeLidar(cfg.point_port, cfg.imu_port)
        fake.start()
        try:
            orig = livox.LivoxClient.__init__

            def init(self, lidar_ip, host_ip="", **kw):  # noqa: ANN001
                kw["command_port"] = fake.ctrl.getsockname()[1]
                orig(self, lidar_ip, host_ip, **kw)

            with mock.patch.object(livox.LivoxClient, "__init__", init):
                with LidarStream(cfg) as stream:
                    frame = stream.get(3.0)
                    self.assertIsNotNone(frame)
                    self.assertGreater(len(frame), 96)
                    np.testing.assert_allclose(frame.xyz[0], [2, 2, 2], atol=1e-3)
                    self.assertEqual(stream.status.serial, "ARMCP8T0035699")
                    deadline = time.monotonic() + 2
                    while not stream.imu_recent(1) and time.monotonic() < deadline:
                        time.sleep(0.02)
                    self.assertTrue(stream.imu_recent(1))
            sent = fake.sets
            self.assertTrue(any(livox.KEY_POINT_HOST_IPCFG in s for s in sent))
            point_cfg = next(
                s[livox.KEY_POINT_HOST_IPCFG]
                for s in sent
                if livox.KEY_POINT_HOST_IPCFG in s
            )
            self.assertEqual(
                livox.unpack_host_ipcfg(point_cfg)[:2], ("127.0.0.1", cfg.point_port)
            )
            self.assertIn({livox.KEY_WORK_TGT_MODE: b"\x01"}, sent)
        finally:
            fake.stop.set()
            fake.join(1)

    def test_needs_setup(self) -> None:
        with self.assertRaises(RuntimeError):
            LidarStream(LidarConfig()).start()


class NetworkTest(unittest.TestCase):
    def test_choose_host_ip(self) -> None:
        self.assertEqual(
            net.choose_host_ip("192.168.1.199", "192.168.1.5"), "192.168.1.5"
        )
        self.assertEqual(
            net.choose_host_ip("192.168.1.199", "255.255.255.255"), "192.168.1.5"
        )
        self.assertEqual(net.choose_host_ip("192.168.1.5"), "192.168.1.50")
        self.assertEqual(net.choose_host_ip("10.0.0.7"), "10.0.0.5")

    def test_profile_uses_host_routes_only(self) -> None:
        args = net.nmcli_profile_args("end0", "192.168.1.5", "192.168.1.199")
        props = dict(zip(args[::2], args[1::2], strict=True))
        self.assertEqual(props["ipv4.addresses"], "192.168.1.5/32")
        self.assertEqual(props["ipv4.routes"], "192.168.1.199/32")
        self.assertEqual(props["ipv4.never-default"], "yes")
        self.assertEqual(props["connection.interface-name"], "end0")

    def test_parse_sniffed_frames(self) -> None:
        mac = bytes.fromhex("aabbccddeeff")
        udp_payload = b"x" * 8
        ip = bytearray(20)
        ip[0] = 0x45
        ip[9] = 17
        ip[12:16] = socket.inet_aton("192.168.1.199")
        ip[16:20] = socket.inet_aton("255.255.255.255")
        udp = struct.pack("!HHHH", 56200, 56201, 8 + len(udp_payload), 0) + udp_payload
        frame = b"\xff" * 6 + mac + b"\x08\x00" + bytes(ip) + udp
        self.assertEqual(
            net._parse_frame(frame),
            ("udp", "192.168.1.199", "255.255.255.255", "aa:bb:cc:dd:ee:ff"),
        )
        arp = (
            b"\xff" * 6
            + mac
            + b"\x08\x06"
            + struct.pack("!HHBBH", 1, 0x0800, 6, 4, 1)
            + mac
            + socket.inet_aton("192.168.1.199")
            + bytes(6)
            + socket.inet_aton("192.168.1.5")
        )
        self.assertEqual(
            net._parse_frame(arp),
            ("arp", "192.168.1.199", "192.168.1.5", "aa:bb:cc:dd:ee:ff"),
        )
        other = bytearray(frame)
        other[14 + 20 : 14 + 22] = struct.pack("!H", 53)
        self.assertIsNone(net._parse_frame(bytes(other)))


class ConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self._old = os.environ.get("ALMOND_HOME")
        self.tmp = tempfile.mkdtemp()
        os.environ["ALMOND_HOME"] = self.tmp

    def tearDown(self) -> None:
        if self._old is None:
            os.environ.pop("ALMOND_HOME", None)
        else:
            os.environ["ALMOND_HOME"] = self._old

    def test_save_load(self) -> None:
        cfg = LidarConfig(
            lidar_ip="192.168.1.199",
            host_ip="192.168.1.5",
            iface="end0",
            mount=LidarMount(0.2, 0, 0.9, 0, -7.5, 90),
        )
        path = cfg.save()
        self.assertEqual(path, Path(self.tmp) / "lidar.json")
        loaded = LidarConfig.load()
        self.assertEqual(loaded, cfg)
        data = json.loads(path.read_text())
        data["unknown"] = 1
        self.assertEqual(LidarConfig.from_dict(data), cfg)

    def test_missing_file_gives_defaults(self) -> None:
        self.assertEqual(LidarConfig.load(), LidarConfig())

    def test_mount_matrix(self) -> None:
        t = LidarMount(1.0, 2.0, 3.0, 0, 0, 90).matrix()
        np.testing.assert_allclose(t[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-9)
        np.testing.assert_allclose(t[:3, 3], [1, 2, 3])


class MountTest(unittest.TestCase):
    def test_measures_height_and_tilt(self) -> None:
        from scipy.spatial.transform import Rotation

        rng = np.random.default_rng(0)
        # A level floor 0.95 m below, seen by a lidar pitched down 8°.
        floor = np.column_stack(
            [
                rng.uniform(-6, 6, 20000),
                rng.uniform(-6, 6, 20000),
                np.full(20000, -0.95),
            ]
        )
        wall = np.column_stack(
            [np.full(5000, 4.0), rng.uniform(-3, 3, 5000), rng.uniform(-0.95, 2, 5000)]
        )
        level_pts = np.vstack([floor, wall])
        r = Rotation.from_euler("y", -8, degrees=True).as_matrix()  # lidar-to-level
        lidar_pts = level_pts @ r  # level-to-lidar is r.T, applied to row vectors
        frame = LidarFrame(
            0,
            0,
            lidar_pts.astype(np.float32),
            np.zeros(len(lidar_pts), np.uint8),
            np.zeros(len(lidar_pts), np.uint8),
            np.zeros(len(lidar_pts), np.float32),
            0.1,
            0,
        )
        g_lidar = r.T @ np.array([0, 0, 9.81])
        imu = [ImuSample(i * 0.005, np.zeros(3), g_lidar) for i in range(100)]
        m = measure_mount([frame], imu)
        self.assertAlmostEqual(m.height, 0.95, places=2)
        self.assertAlmostEqual(m.pitch, -8.0, places=1)
        self.assertAlmostEqual(m.roll, 0.0, places=1)

    def test_refuses_moving_robot(self) -> None:
        imu = [ImuSample(0, np.zeros(3), np.array([0, 0, 12.0]))] * 50
        with self.assertRaises(ValueError):
            measure_mount([], imu)


class RecordingTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        frames = [
            LidarFrame(
                10.0 + i * 0.1,
                i,
                np.full((5 + i, 3), i, np.float32),
                np.zeros(5 + i, np.uint8),
                np.zeros(5 + i, np.uint8),
                np.linspace(0, 0.1, 5 + i).astype(np.float32),
                0.1,
                i,
            )
            for i in range(3)
        ]
        rec = LidarRecording(
            frames,
            [ImuSample(10.0 + i * 0.005, np.ones(3), np.zeros(3)) for i in range(100)],
        )
        with tempfile.TemporaryDirectory() as d:
            path = rec.save(Path(d) / "r.npz")
            back = LidarRecording.load(path)
        self.assertEqual(len(back), 3)
        self.assertEqual([len(f) for f in back.frames()], [5, 6, 7])
        np.testing.assert_array_equal(back.frames_[2].xyz, frames[2].xyz)
        self.assertEqual(len(back.imu_between(10.0, 10.1)), 20)
        self.assertAlmostEqual(back.duration, 0.2)


if __name__ == "__main__":
    unittest.main()
