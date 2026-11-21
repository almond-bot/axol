"""Livox Mid-360 / Mid-360S Ethernet protocol (SDK2 generation), in pure Python.

The Mid-360 talks UDP on fixed ports. A host sends *control frames* to the
lidar's command port and receives three streams back: state pushes, point
packets and IMU packets. This module implements just enough of the protocol
(v1.4, "Mid-360 communication protocol" on the Livox wiki) to find a lidar,
read its status, point its streams at this host and parse what arrives. No
Livox SDK, ROS or compiled code is involved, so it runs anywhere Axol does.

Control frame (all fields little-endian)::

    0   sof      u8    0xAA
    1   version  u8    0
    2   length   u16   whole frame, header included
    4   seq      u32   echoed by the ACK
    8   cmd_id   u16
    10  cmd_type u8    0 request, 1 ack
    11  sender   u8    0 host, 1 lidar
    12  reserved 6B
    18  crc16    u16   CRC-16/CCITT-FALSE over bytes 0..17
    20  crc32    u32   CRC-32 (zlib) over the data, 0 when there is none
    24  data

Point / IMU packet header::

    0   version       u8
    1   length        u16
    3   time_interval u16   0.1 us units, first to last point
    5   dot_num       u16
    7   udp_cnt       u16
    9   frame_cnt     u8
    10  data_type     u8    0 IMU, 1 cartesian int32 mm, 2 cartesian int16 cm
    11  time_type     u8    0 power-on clock, 1 PTP, 2 GPS
    12  reserved      12B
    24  crc32         u32
    28  timestamp     u64   ns, of the first point
    36  data

Verified on hardware (Mid-360 firmware 35.01.01.08): 96 points per packet in
cartesian-32 (14 bytes each, so 1380-byte packets), ~2083 packets/s for
200k points/s, and IMU packets at 200 Hz.
"""

from __future__ import annotations

import ipaddress
import itertools
import socket
import struct
import threading
import zlib
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Ports (lidar side; the host side defaults to lidar port + 1)
# ---------------------------------------------------------------------------

DISCOVERY_PORT = 56000
COMMAND_PORT = 56100
PUSH_PORT = 56200
POINT_PORT = 56300
IMU_PORT = 56400

HOST_COMMAND_PORT = 56101
HOST_PUSH_PORT = 56201
HOST_POINT_PORT = 56301
HOST_IMU_PORT = 56401

# ---------------------------------------------------------------------------
# Control frames
# ---------------------------------------------------------------------------

SOF = 0xAA
HEADER_SIZE = 24
MAX_FRAME_SIZE = 1400

CMD_TYPE_REQ = 0x00
CMD_TYPE_ACK = 0x01
SENDER_HOST = 0x00
SENDER_LIDAR = 0x01

CMD_DISCOVER = 0x0000
CMD_SET_PARAMS = 0x0100
CMD_GET_PARAMS = 0x0101
CMD_PUSH_PARAMS = 0x0102
CMD_REBOOT = 0x0200

# Parameter keys (settable unless noted).
KEY_PCL_DATA_TYPE = 0x0000
KEY_PATTERN_MODE = 0x0001
KEY_LIDAR_IPCFG = 0x0004
KEY_STATE_INFO_HOST_IPCFG = 0x0005
KEY_POINT_HOST_IPCFG = 0x0006
KEY_IMU_HOST_IPCFG = 0x0007
KEY_INSTALL_ATTITUDE = 0x0012
KEY_DETECT_MODE = 0x0018
KEY_WORK_TGT_MODE = 0x001A
KEY_IMU_DATA_EN = 0x001C
# Read-only.
KEY_SN = 0x8000
KEY_PRODUCT_INFO = 0x8001
KEY_VERSION_APP = 0x8002
KEY_VERSION_LOADER = 0x8003
KEY_VERSION_HARDWARE = 0x8004
KEY_MAC = 0x8005
KEY_CUR_WORK_STATE = 0x8006
KEY_CORE_TEMP = 0x8007
KEY_POWERUP_CNT = 0x8008
KEY_LOCAL_TIME_NOW = 0x8009
KEY_TIME_SYNC_TYPE = 0x800C
KEY_DIAG_STATUS = 0x800E
KEY_FW_TYPE = 0x8010
KEY_HMS_CODE = 0x8011

PCL_CARTESIAN_32 = 1
PCL_CARTESIAN_16 = 2
PCL_SPHERICAL = 3

WORK_SAMPLING = 0x01
WORK_IDLE = 0x02
WORK_STATE_NAMES = {
    0x01: "sampling",
    0x02: "idle",
    0x04: "error",
    0x05: "self-check",
    0x06: "motor starting",
    0x08: "upgrading",
    0x09: "ready",
}
TIME_SYNC_NAMES = {0: "none", 1: "PTP", 2: "GPS"}

# Device types the discovery ACK reports. Mid-360 units report 9 on the
# documented firmware and 35 on newer firmware (seen on a Mid-360 running
# 35.01.01.08); anything else is still accepted and shown as a number.
DEVICE_TYPE_NAMES = {9: "Mid-360", 35: "Mid-360"}

RETURN_CODE_NAMES = {
    0x00: "success",
    0x01: "failure",
    0x02: "not permitted now",
    0x03: "value out of range",
    0x20: "parameter not supported",
    0x21: "takes effect after reboot",
    0x22: "parameter is read-only",
    0x23: "invalid length",
    0x24: "key count mismatch",
}


class LivoxError(RuntimeError):
    """The lidar refused a command, or sent something we can't parse."""


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection)."""
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = (
                ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
            )
    return crc


@dataclass(frozen=True)
class ControlFrame:
    """One decoded control frame."""

    cmd_id: int
    cmd_type: int
    seq: int
    sender: int
    data: bytes


def build_frame(
    cmd_id: int,
    seq: int,
    data: bytes = b"",
    *,
    cmd_type: int = CMD_TYPE_REQ,
    sender: int = SENDER_HOST,
) -> bytes:
    """Encode a control frame."""
    length = HEADER_SIZE + len(data)
    if length > MAX_FRAME_SIZE:
        raise ValueError(f"control frame of {length} bytes exceeds {MAX_FRAME_SIZE}")
    head = struct.pack(
        "<BBHIHBB6s",
        SOF,
        0,
        length,
        seq & 0xFFFFFFFF,
        cmd_id,
        cmd_type,
        sender,
        bytes(6),
    )
    crc32 = zlib.crc32(data) if data else 0
    return head + struct.pack("<HI", crc16_ccitt(head), crc32) + data


def parse_frame(buf: bytes) -> ControlFrame:
    """Decode and verify a control frame; raises :class:`LivoxError`."""
    if len(buf) < HEADER_SIZE:
        raise LivoxError(f"control frame too short ({len(buf)} bytes)")
    sof, _version, length, seq, cmd_id, cmd_type, sender = struct.unpack_from(
        "<BBHIHBB", buf
    )
    if sof != SOF:
        raise LivoxError(f"bad start byte 0x{sof:02x}")
    if length < HEADER_SIZE or length > len(buf):
        raise LivoxError(f"bad frame length {length} (got {len(buf)} bytes)")
    crc16, crc32 = struct.unpack_from("<HI", buf, 18)
    if crc16 != crc16_ccitt(buf[:18]):
        raise LivoxError("header checksum mismatch")
    data = bytes(buf[HEADER_SIZE:length])
    if (zlib.crc32(data) if data else 0) != crc32:
        raise LivoxError("data checksum mismatch")
    return ControlFrame(cmd_id, cmd_type, seq, sender, data)


def pack_params(items: dict[int, bytes]) -> bytes:
    """Encode a key/value list: key_num u16, reserved u16, (key, len, value)*."""
    body = b"".join(
        struct.pack("<HH", key, len(value)) + value for key, value in items.items()
    )
    return struct.pack("<HH", len(items), 0) + body


def unpack_params(buf: bytes, offset: int = 0) -> dict[int, bytes]:
    """Decode a run of ``(key u16, len u16, value)`` entries starting at ``offset``."""
    out: dict[int, bytes] = {}
    while offset + 4 <= len(buf):
        key, size = struct.unpack_from("<HH", buf, offset)
        offset += 4
        if offset + size > len(buf):
            raise LivoxError(f"parameter 0x{key:04x} runs past the end of the frame")
        out[key] = bytes(buf[offset : offset + size])
        offset += size
    return out


def pack_host_ipcfg(ip: str, port: int, src_port: int = 0) -> bytes:
    """Value of the ``*_HOST_IPCFG`` keys: destination ip, port, lidar source port."""
    return socket.inet_aton(ip) + struct.pack("<HH", port, src_port)


def unpack_host_ipcfg(value: bytes) -> tuple[str, int, int]:
    ip = socket.inet_ntoa(value[:4])
    port, src = struct.unpack_from("<HH", value, 4)
    return ip, port, src


def pack_lidar_ipcfg(ip: str, netmask: str, gateway: str) -> bytes:
    return socket.inet_aton(ip) + socket.inet_aton(netmask) + socket.inet_aton(gateway)


def unpack_lidar_ipcfg(value: bytes) -> tuple[str, str, str]:
    return (
        socket.inet_ntoa(value[0:4]),
        socket.inet_ntoa(value[4:8]),
        socket.inet_ntoa(value[8:12]),
    )


def _version(value: bytes) -> str:
    # aa.bb.cc.dd in byte order: firmware 35.01.01.08 is stored as 23 01 01 08.
    return ".".join(f"{b:02d}" for b in value) if len(value) == 4 else ""


def _cstr(value: bytes) -> str:
    return value.split(b"\0", 1)[0].decode(errors="replace")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

STATUS_KEYS = (
    KEY_PCL_DATA_TYPE,
    KEY_LIDAR_IPCFG,
    KEY_STATE_INFO_HOST_IPCFG,
    KEY_POINT_HOST_IPCFG,
    KEY_IMU_HOST_IPCFG,
    KEY_WORK_TGT_MODE,
    KEY_IMU_DATA_EN,
    KEY_SN,
    KEY_PRODUCT_INFO,
    KEY_VERSION_APP,
    KEY_VERSION_LOADER,
    KEY_VERSION_HARDWARE,
    KEY_MAC,
    KEY_CUR_WORK_STATE,
    KEY_CORE_TEMP,
    KEY_TIME_SYNC_TYPE,
    KEY_DIAG_STATUS,
    KEY_HMS_CODE,
)


@dataclass(frozen=True)
class LidarStatus:
    """What the lidar reports about itself (see :meth:`LivoxClient.status`)."""

    serial: str
    product: str
    firmware: str
    loader: str
    hardware: str
    mac: str
    lidar_ip: str
    netmask: str
    gateway: str
    point_host: tuple[str, int]
    imu_host: tuple[str, int]
    state_host: tuple[str, int]
    work_state: int
    target_state: int
    imu_enabled: bool
    point_data_type: int
    temperature_c: float
    time_sync: int
    diag_status: int
    hms_codes: tuple[int, ...] = field(default=())

    @property
    def work_state_name(self) -> str:
        return WORK_STATE_NAMES.get(self.work_state, f"0x{self.work_state:02x}")

    @property
    def time_sync_name(self) -> str:
        return TIME_SYNC_NAMES.get(self.time_sync, str(self.time_sync))

    @property
    def faults(self) -> tuple[int, ...]:
        """Non-zero health-management (HMS) codes; empty when healthy."""
        return tuple(code for code in self.hms_codes if code)

    @classmethod
    def from_params(cls, p: dict[int, bytes]) -> LidarStatus:
        def u8(key: int) -> int:
            return p[key][0] if p.get(key) else 0

        lidar_ip, netmask, gateway = (
            unpack_lidar_ipcfg(p[KEY_LIDAR_IPCFG])
            if len(p.get(KEY_LIDAR_IPCFG, b"")) >= 12
            else ("", "", "")
        )

        def host(key: int) -> tuple[str, int]:
            value = p.get(key, b"")
            if len(value) < 8:
                return ("", 0)
            ip, port, _ = unpack_host_ipcfg(value)
            return ip, port

        product = _cstr(p.get(KEY_PRODUCT_INFO, b""))
        # "DevType:Mid-360 FmType:App FmVer:35010108 BuildTime:..." -> "Mid-360"
        fields = dict(part.split(":", 1) for part in product.split() if ":" in part)
        temp = p.get(KEY_CORE_TEMP, b"")
        hms = p.get(KEY_HMS_CODE, b"")
        return cls(
            serial=_cstr(p.get(KEY_SN, b"")),
            product=fields.get("DevType", product),
            firmware=_version(p.get(KEY_VERSION_APP, b"")),
            loader=_version(p.get(KEY_VERSION_LOADER, b"")),
            hardware=_version(p.get(KEY_VERSION_HARDWARE, b"")),
            mac=":".join(f"{b:02x}" for b in p.get(KEY_MAC, b"")),
            lidar_ip=lidar_ip,
            netmask=netmask,
            gateway=gateway,
            point_host=host(KEY_POINT_HOST_IPCFG),
            imu_host=host(KEY_IMU_HOST_IPCFG),
            state_host=host(KEY_STATE_INFO_HOST_IPCFG),
            work_state=u8(KEY_CUR_WORK_STATE),
            target_state=u8(KEY_WORK_TGT_MODE),
            imu_enabled=bool(u8(KEY_IMU_DATA_EN)),
            point_data_type=u8(KEY_PCL_DATA_TYPE),
            temperature_c=struct.unpack("<i", temp)[0] / 100.0
            if len(temp) == 4
            else 0.0,
            time_sync=u8(KEY_TIME_SYNC_TYPE),
            diag_status=struct.unpack("<H", p[KEY_DIAG_STATUS])[0]
            if len(p.get(KEY_DIAG_STATUS, b"")) == 2
            else 0,
            hms_codes=struct.unpack(f"<{len(hms) // 4}I", hms[: len(hms) // 4 * 4]),
        )


# ---------------------------------------------------------------------------
# Discovery and control
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LivoxDevice:
    """A lidar that answered discovery."""

    serial: str
    ip: str
    device_type: int
    command_port: int = COMMAND_PORT

    @property
    def model(self) -> str:
        return DEVICE_TYPE_NAMES.get(self.device_type, f"type {self.device_type}")


def parse_discovery_ack(data: bytes) -> LivoxDevice:
    if len(data) < 24:
        raise LivoxError(f"discovery reply too short ({len(data)} bytes)")
    ret, dev_type = data[0], data[1]
    if ret != 0:
        raise LivoxError(f"discovery reply carried error {ret}")
    (port,) = struct.unpack_from("<H", data, 22)
    return LivoxDevice(
        serial=_cstr(data[2:18]),
        ip=socket.inet_ntoa(data[18:22]),
        device_type=dev_type,
        command_port=port or COMMAND_PORT,
    )


def discover(
    iface: str | None = None, timeout: float = 1.0, host_ip: str = ""
) -> list[LivoxDevice]:
    """Broadcast a discovery request and collect the lidars that answer.

    The interface needs an IPv4 address for replies to arrive; on a port that
    has none yet, use :func:`almond_axol.lidar.network.sniff_lidar` (root).
    ``iface`` binds the broadcast to one interface (``SO_BINDTODEVICE``).
    """
    import time

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        if iface:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, iface.encode())
        sock.bind((host_ip, 0))
        sock.sendto(build_frame(CMD_DISCOVER, 1), ("255.255.255.255", DISCOVERY_PORT))
        found: dict[str, LivoxDevice] = {}
        deadline = time.monotonic() + timeout
        while (left := deadline - time.monotonic()) > 0:
            sock.settimeout(left)
            try:
                buf, _ = sock.recvfrom(2048)
            except TimeoutError:
                break
            try:
                frame = parse_frame(buf)
                if frame.cmd_id != CMD_DISCOVER or frame.cmd_type != CMD_TYPE_ACK:
                    continue
                device = parse_discovery_ack(frame.data)
            except LivoxError:
                continue
            found[device.serial] = device
        return list(found.values())
    finally:
        sock.close()


class LivoxClient:
    """Request/ACK control channel to one lidar.

    Bind ``host_ip`` to the address on the lidar's port, so replies come back
    over that link even when the lidar's subnet overlaps another network.
    """

    def __init__(
        self,
        lidar_ip: str,
        host_ip: str = "",
        *,
        command_port: int = COMMAND_PORT,
        host_port: int = 0,
        timeout: float = 1.0,
        retries: int = 3,
    ) -> None:
        ipaddress.IPv4Address(lidar_ip)
        self.lidar_ip = lidar_ip
        self.command_port = command_port
        self.timeout = timeout
        self.retries = max(1, retries)
        self._seq = itertools.count(1)
        self._lock = threading.Lock()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host_ip, host_port))

    def close(self) -> None:
        self._sock.close()

    def __enter__(self) -> LivoxClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def request(self, cmd_id: int, data: bytes = b"") -> bytes:
        """Send one request and return the ACK's data; retries on timeout."""
        with self._lock:
            seq = next(self._seq)
            frame = build_frame(cmd_id, seq, data)
            for _ in range(self.retries):
                self._sock.sendto(frame, (self.lidar_ip, self.command_port))
                self._sock.settimeout(self.timeout)
                try:
                    while True:
                        buf, _ = self._sock.recvfrom(2048)
                        try:
                            ack = parse_frame(buf)
                        except LivoxError:
                            continue
                        if (
                            ack.cmd_id == cmd_id
                            and ack.cmd_type == CMD_TYPE_ACK
                            and ack.seq == seq
                        ):
                            return ack.data
                except TimeoutError:
                    continue
            raise TimeoutError(
                f"Lidar at {self.lidar_ip} did not answer command 0x{cmd_id:04x}"
            )

    def get_params(self, keys: tuple[int, ...] | list[int]) -> dict[int, bytes]:
        data = struct.pack("<HH", len(keys), 0) + b"".join(
            struct.pack("<H", k) for k in keys
        )
        ack = self.request(CMD_GET_PARAMS, data)
        if not ack or ack[0] != 0:
            code = ack[0] if ack else -1
            raise LivoxError(
                f"parameter query refused: {RETURN_CODE_NAMES.get(code, code)}"
            )
        return unpack_params(ack, 3)

    def set_params(self, items: dict[int, bytes]) -> bool:
        """Write parameters. Returns True when a reboot is needed to apply them."""
        ack = self.request(CMD_SET_PARAMS, pack_params(items))
        if len(ack) < 3:
            raise LivoxError("short reply to a parameter write")
        code = ack[0]
        (key,) = struct.unpack_from("<H", ack, 1)
        if code == 0x21:
            return True
        if code != 0:
            raise LivoxError(
                f"lidar refused parameter 0x{key:04x}: "
                f"{RETURN_CODE_NAMES.get(code, f'code {code}')}"
            )
        return False

    def status(self) -> LidarStatus:
        return LidarStatus.from_params(self.get_params(STATUS_KEYS))

    def configure_streams(
        self,
        host_ip: str,
        *,
        point_port: int = HOST_POINT_PORT,
        imu_port: int = HOST_IMU_PORT,
        imu: bool = True,
    ) -> None:
        """Point the point and IMU streams at ``host_ip`` and start sampling."""
        self.set_params(
            {
                KEY_PCL_DATA_TYPE: bytes([PCL_CARTESIAN_32]),
                KEY_PATTERN_MODE: b"\x00",
                KEY_POINT_HOST_IPCFG: pack_host_ipcfg(host_ip, point_port),
                KEY_IMU_HOST_IPCFG: pack_host_ipcfg(host_ip, imu_port),
                KEY_IMU_DATA_EN: bytes([int(imu)]),
            }
        )
        self.set_work_mode(WORK_SAMPLING)

    def set_work_mode(self, mode: int) -> None:
        """``WORK_SAMPLING`` spins up and streams; ``WORK_IDLE`` stops the motor."""
        self.set_params({KEY_WORK_TGT_MODE: bytes([mode])})

    def set_lidar_ip(self, ip: str, netmask: str, gateway: str) -> None:
        """Change the lidar's own address. Takes effect after :meth:`reboot`."""
        self.set_params({KEY_LIDAR_IPCFG: pack_lidar_ipcfg(ip, netmask, gateway)})

    def reboot(self, delay_ms: int = 100) -> None:
        self.request(CMD_REBOOT, struct.pack("<H", delay_ms))


# ---------------------------------------------------------------------------
# Point and IMU packets
# ---------------------------------------------------------------------------

PACKET_HEADER_SIZE = 36
_PACKET_HEADER = struct.Struct("<BHHHHBBB12xIQ")

POINT_DTYPE_32 = np.dtype(
    [("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("reflectivity", "u1"), ("tag", "u1")]
)
POINT_DTYPE_16 = np.dtype(
    [("x", "<i2"), ("y", "<i2"), ("z", "<i2"), ("reflectivity", "u1"), ("tag", "u1")]
)
IMU_DTYPE = np.dtype([("gyro", "<f4", 3), ("accel", "<f4", 3)])

DATA_TYPE_IMU = 0
STANDARD_GRAVITY = 9.80665


@dataclass(frozen=True)
class PacketHeader:
    version: int
    length: int
    time_interval_ns: int
    dot_num: int
    udp_cnt: int
    frame_cnt: int
    data_type: int
    time_type: int
    timestamp_ns: int


def parse_packet_header(buf: bytes | memoryview) -> PacketHeader:
    if len(buf) < PACKET_HEADER_SIZE:
        raise LivoxError(f"data packet too short ({len(buf)} bytes)")
    (
        version,
        length,
        interval,
        dot_num,
        udp_cnt,
        frame_cnt,
        data_type,
        time_type,
        _crc32,
        timestamp,
    ) = _PACKET_HEADER.unpack_from(buf)
    return PacketHeader(
        version,
        length,
        interval * 100,
        dot_num,
        udp_cnt,
        frame_cnt,
        data_type,
        time_type,
        timestamp,
    )


@dataclass
class PointPacket:
    """One packet of points, in the lidar frame.

    ``xyz`` is in metres (float32, shape ``(n, 3)``); ``offsets_ns`` is each
    point's time after ``header.timestamp_ns`` (points are evenly spaced
    over the packet's ``time_interval``).
    """

    header: PacketHeader
    xyz: np.ndarray
    reflectivity: np.ndarray
    tag: np.ndarray
    offsets_ns: np.ndarray


def parse_point_packet(buf: bytes | memoryview) -> PointPacket:
    header = parse_packet_header(buf)
    if header.data_type == 1:
        dtype, scale = POINT_DTYPE_32, 1e-3
    elif header.data_type == 2:
        dtype, scale = POINT_DTYPE_16, 1e-2
    else:
        raise LivoxError(f"unsupported point data type {header.data_type}")
    n = header.dot_num
    end = PACKET_HEADER_SIZE + n * dtype.itemsize
    if end > len(buf):
        raise LivoxError(f"point packet holds {len(buf)} bytes, header promises {end}")
    raw = np.frombuffer(buf, dtype=dtype, count=n, offset=PACKET_HEADER_SIZE)
    xyz = np.empty((n, 3), dtype=np.float32)
    xyz[:, 0] = raw["x"]
    xyz[:, 1] = raw["y"]
    xyz[:, 2] = raw["z"]
    xyz *= scale
    step = header.time_interval_ns / max(n - 1, 1)
    offsets = (np.arange(n, dtype=np.float64) * step).astype(np.int64)
    return PointPacket(
        header, xyz, raw["reflectivity"].copy(), raw["tag"].copy(), offsets
    )


@dataclass(frozen=True)
class ImuPacket:
    """One IMU sample: gyro in rad/s, accel in m/s² (lidar frame)."""

    timestamp_ns: int
    time_type: int
    gyro: np.ndarray
    accel: np.ndarray


def parse_imu_packet(buf: bytes | memoryview) -> ImuPacket:
    header = parse_packet_header(buf)
    if header.data_type != DATA_TYPE_IMU:
        raise LivoxError(f"expected an IMU packet, got data type {header.data_type}")
    if len(buf) < PACKET_HEADER_SIZE + IMU_DTYPE.itemsize:
        raise LivoxError("IMU packet too short")
    raw = np.frombuffer(buf, dtype=IMU_DTYPE, count=1, offset=PACKET_HEADER_SIZE)[0]
    return ImuPacket(
        header.timestamp_ns,
        header.time_type,
        raw["gyro"].astype(np.float64),
        raw["accel"].astype(np.float64) * STANDARD_GRAVITY,
    )


def build_point_packet(
    xyz_m: np.ndarray,
    timestamp_ns: int,
    *,
    time_interval_ns: int = 4800,
    reflectivity: np.ndarray | None = None,
    udp_cnt: int = 0,
    time_type: int = 0,
) -> bytes:
    """Encode a cartesian-32 point packet (for tests and the simulator)."""
    n = len(xyz_m)
    raw = np.zeros(n, dtype=POINT_DTYPE_32)
    mm = np.round(np.asarray(xyz_m, dtype=np.float64) * 1000.0).astype(np.int32)
    raw["x"], raw["y"], raw["z"] = mm[:, 0], mm[:, 1], mm[:, 2]
    if reflectivity is not None:
        raw["reflectivity"] = reflectivity
    data = raw.tobytes()
    head = _PACKET_HEADER.pack(
        0,
        PACKET_HEADER_SIZE + len(data),
        time_interval_ns // 100,
        n,
        udp_cnt,
        0,
        PCL_CARTESIAN_32,
        time_type,
        0,
        timestamp_ns,
    )
    return head + data


def build_imu_packet(
    gyro: np.ndarray, accel_m_s2: np.ndarray, timestamp_ns: int
) -> bytes:
    """Encode an IMU packet (for tests and the simulator)."""
    raw = np.zeros(1, dtype=IMU_DTYPE)
    raw["gyro"] = gyro
    raw["accel"] = np.asarray(accel_m_s2) / STANDARD_GRAVITY
    data = raw.tobytes()
    head = _PACKET_HEADER.pack(
        0,
        PACKET_HEADER_SIZE + len(data),
        0,
        1,
        0,
        0,
        DATA_TYPE_IMU,
        0,
        0,
        timestamp_ns,
    )
    return head + data
