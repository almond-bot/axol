"""ROS 2 message schemas and a small CDR encoder for the MCAP recorder.

Episodes are written with the ``ros2`` MCAP profile: every channel carries a
``ros2msg`` schema (the ``.msg`` text, dependencies appended) and every
message is CDR (little-endian XCDR1, the ROS 2 default). Foxglove, ``ros2
bag`` and ``mcap-ros2-support`` all read that combination without extra
setup.

The encoders here are hand-written for the handful of message types the
recorder writes. ``mcap_ros2``'s generic serializer interprets the schema per
field, which costs too much at 240 Hz joint states plus four 60 fps video
streams on a Jetson; these build the same bytes with ``struct``. The tests
decode every type with ``mcap_ros2``'s reference decoder, so the two cannot
drift apart silently.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable, Sequence

# --------------------------------------------------------------------------
# Schemas (ROS 2 .msg text)
# --------------------------------------------------------------------------

_DEFS: dict[str, str] = {
    "builtin_interfaces/Time": "int32 sec\nuint32 nanosec\n",
    "std_msgs/Header": "builtin_interfaces/Time stamp\nstring frame_id\n",
    "std_msgs/String": "string data\n",
    "geometry_msgs/Vector3": "float64 x\nfloat64 y\nfloat64 z\n",
    "geometry_msgs/Point": "float64 x\nfloat64 y\nfloat64 z\n",
    "geometry_msgs/Quaternion": "float64 x\nfloat64 y\nfloat64 z\nfloat64 w\n",
    "geometry_msgs/Pose": (
        "geometry_msgs/Point position\ngeometry_msgs/Quaternion orientation\n"
    ),
    "geometry_msgs/Transform": (
        "geometry_msgs/Vector3 translation\ngeometry_msgs/Quaternion rotation\n"
    ),
    "geometry_msgs/TransformStamped": (
        "std_msgs/Header header\nstring child_frame_id\n"
        "geometry_msgs/Transform transform\n"
    ),
    "geometry_msgs/Twist": (
        "geometry_msgs/Vector3 linear\ngeometry_msgs/Vector3 angular\n"
    ),
    "geometry_msgs/PoseWithCovariance": (
        "geometry_msgs/Pose pose\nfloat64[36] covariance\n"
    ),
    "geometry_msgs/TwistWithCovariance": (
        "geometry_msgs/Twist twist\nfloat64[36] covariance\n"
    ),
    "geometry_msgs/PoseStamped": "std_msgs/Header header\ngeometry_msgs/Pose pose\n",
    "geometry_msgs/TwistStamped": (
        "std_msgs/Header header\ngeometry_msgs/Twist twist\n"
    ),
    "tf2_msgs/TFMessage": "geometry_msgs/TransformStamped[] transforms\n",
    "nav_msgs/Odometry": (
        "std_msgs/Header header\nstring child_frame_id\n"
        "geometry_msgs/PoseWithCovariance pose\n"
        "geometry_msgs/TwistWithCovariance twist\n"
    ),
    "sensor_msgs/JointState": (
        "std_msgs/Header header\nstring[] name\nfloat64[] position\n"
        "float64[] velocity\nfloat64[] effort\n"
    ),
    "sensor_msgs/RegionOfInterest": (
        "uint32 x_offset\nuint32 y_offset\nuint32 height\nuint32 width\n"
        "bool do_rectify\n"
    ),
    "sensor_msgs/CameraInfo": (
        "std_msgs/Header header\nuint32 height\nuint32 width\n"
        "string distortion_model\nfloat64[] d\nfloat64[9] k\nfloat64[9] r\n"
        "float64[12] p\nuint32 binning_x\nuint32 binning_y\n"
        "sensor_msgs/RegionOfInterest roi\n"
    ),
    "sensor_msgs/Joy": "std_msgs/Header header\nfloat32[] axes\nint32[] buttons\n",
    "sensor_msgs/Imu": (
        "std_msgs/Header header\n"
        "geometry_msgs/Quaternion orientation\n"
        "float64[9] orientation_covariance\n"
        "geometry_msgs/Vector3 angular_velocity\n"
        "float64[9] angular_velocity_covariance\n"
        "geometry_msgs/Vector3 linear_acceleration\n"
        "float64[9] linear_acceleration_covariance\n"
    ),
    # Almond's command messages: when the control software sent a command and
    # how each device applies it. CommandInfo accompanies a standard command
    # message (/joint_command's JointState) with the same stamp. Field meanings
    # are in docs/operations/mcap.mdx.
    "almond_msgs/CommandInfo": (
        "std_msgs/Header header\n"
        "uint64 command_id\n"
        "builtin_interfaces/Time issue_timestamp\n"
        "string[] name\n"
        "string[] target_device_id\n"
        "string[] control_mode\n"
        "builtin_interfaces/Time[] application_timestamp\n"
    ),
    "almond_msgs/BaseCommand": (
        "std_msgs/Header header\n"
        "uint64 command_id\n"
        "builtin_interfaces/Time issue_timestamp\n"
        "string target_device_id\n"
        "string control_mode\n"
        "float64 vx\n"
        "float64 vy\n"
        "float64 wz\n"
        "int8 lift_direction\n"
    ),
    "foxglove_msgs/CompressedVideo": (
        "builtin_interfaces/Time timestamp\nstring frame_id\nuint8[] data\n"
        "string format\n"
    ),
}

_PRIMITIVES = frozenset(
    {
        "bool",
        "byte",
        "char",
        "float32",
        "float64",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "string",
    }
)


def _field_types(definition: str) -> list[str]:
    out = []
    for line in definition.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        type_name = line.split()[0].split("[")[0]
        if type_name not in _PRIMITIVES:
            out.append(type_name)
    return out


def schema_text(msg_type: str) -> str:
    """The ``ros2msg`` schema for ``msg_type`` (``pkg/Name``), dependencies included."""
    if msg_type not in _DEFS:
        raise KeyError(f"no schema for {msg_type}")
    order: list[str] = []
    pending = list(_field_types(_DEFS[msg_type]))
    while pending:
        dep = pending.pop(0)
        if dep in order:
            continue
        order.append(dep)
        pending.extend(_field_types(_DEFS[dep]))
    parts = [_DEFS[msg_type]]
    for dep in order:
        parts.append("=" * 80 + f"\nMSG: {dep}\n" + _DEFS[dep])
    return "".join(parts)


def schema_name(msg_type: str) -> str:
    """The MCAP schema name (``pkg/msg/Name``) ROS 2 tools expect."""
    package, _, name = msg_type.partition("/")
    return f"{package}/msg/{name}"


MESSAGE_TYPES: tuple[str, ...] = (
    "foxglove_msgs/CompressedVideo",
    "sensor_msgs/CameraInfo",
    "sensor_msgs/JointState",
    "sensor_msgs/Joy",
    "sensor_msgs/Imu",
    "almond_msgs/CommandInfo",
    "almond_msgs/BaseCommand",
    "tf2_msgs/TFMessage",
    "nav_msgs/Odometry",
    "geometry_msgs/PoseStamped",
    "geometry_msgs/TwistStamped",
    "std_msgs/String",
)

# --------------------------------------------------------------------------
# CDR encoder
# --------------------------------------------------------------------------

_ENCAPSULATION = b"\x00\x01\x00\x00"  # CDR, little endian
_U32 = struct.Struct("<I")
_I32 = struct.Struct("<i")
_F64 = struct.Struct("<d")
_TIME = struct.Struct("<iI")
_ZEROS = b"\x00" * 8


class CdrWriter:
    """Appends CDR-encoded fields to one message buffer.

    Alignment is relative to the byte after the 4-byte encapsulation header,
    as ROS 2's Fast-CDR does it. Only the primitives the recorder needs are
    here.
    """

    __slots__ = ("buf",)

    def __init__(self) -> None:
        self.buf = bytearray(_ENCAPSULATION)

    def _align(self, size: int) -> None:
        pad = -(len(self.buf) - 4) % size
        if pad:
            self.buf += _ZEROS[:pad]

    def u32(self, value: int) -> None:
        self._align(4)
        self.buf += _U32.pack(value)

    def i32(self, value: int) -> None:
        self._align(4)
        self.buf += _I32.pack(value)

    def boolean(self, value: bool) -> None:
        self.buf += b"\x01" if value else b"\x00"

    def i8(self, value: int) -> None:
        self.buf += struct.pack("<b", value)

    def u64(self, value: int) -> None:
        self._align(8)
        self.buf += struct.pack("<Q", value)

    def f64(self, value: float) -> None:
        self._align(8)
        self.buf += _F64.pack(value)

    def f64_array(self, values: Sequence[float]) -> None:
        """A fixed-size ``float64[N]`` (no length prefix)."""
        if values:
            self._align(8)
            self.buf += struct.pack(f"<{len(values)}d", *values)

    def f64_seq(self, values: Sequence[float]) -> None:
        self.u32(len(values))
        self.f64_array(values)

    def f32_seq(self, values: Sequence[float]) -> None:
        self.u32(len(values))
        if values:
            self.buf += struct.pack(f"<{len(values)}f", *values)

    def i32_seq(self, values: Sequence[int]) -> None:
        self.u32(len(values))
        if values:
            self.buf += struct.pack(f"<{len(values)}i", *values)

    def string(self, value: str) -> None:
        raw = value.encode()
        self.u32(len(raw) + 1)
        self.buf += raw
        self.buf += b"\x00"

    def string_seq(self, values: Sequence[str]) -> None:
        self.u32(len(values))
        for value in values:
            self.string(value)

    def time(self, stamp_ns: int) -> None:
        sec, nanosec = divmod(int(stamp_ns), 1_000_000_000)
        self._align(4)
        self.buf += _TIME.pack(sec, nanosec)

    def header(self, stamp_ns: int, frame_id: str) -> None:
        self.time(stamp_ns)
        self.string(frame_id)

    def vector3(self, xyz: Sequence[float]) -> None:
        self.f64_array((float(xyz[0]), float(xyz[1]), float(xyz[2])))

    def quaternion(self, xyzw: Sequence[float]) -> None:
        self.f64_array((float(xyzw[0]), float(xyzw[1]), float(xyzw[2]), float(xyzw[3])))

    def bytes(self) -> bytes:
        return bytes(self.buf)


# --------------------------------------------------------------------------
# Message encoders
# --------------------------------------------------------------------------

Pose7 = Sequence[float]  # x, y, z, qx, qy, qz, qw


def compressed_video(
    stamp_ns: int, frame_id: str, data: bytes, video_format: str
) -> bytes:
    """``foxglove_msgs/CompressedVideo``, built with a single copy of ``data``."""
    head = CdrWriter()
    head.time(stamp_ns)
    head.string(frame_id)
    head.u32(len(data))
    # The trailing string's length field aligns to 4 relative to the payload.
    offset = len(head.buf) - 4 + len(data)
    pad = -offset % 4
    fmt = video_format.encode()
    tail = _ZEROS[:pad] + _U32.pack(len(fmt) + 1) + fmt + b"\x00"
    return b"".join((head.buf, data, tail))


def camera_info(
    stamp_ns: int,
    frame_id: str,
    *,
    width: int,
    height: int,
    distortion_model: str,
    d: Sequence[float],
    k: Sequence[float],
    r: Sequence[float],
    p: Sequence[float],
) -> bytes:
    """``sensor_msgs/CameraInfo`` (no binning, full-frame ROI)."""
    w = CdrWriter()
    w.header(stamp_ns, frame_id)
    w.u32(height)
    w.u32(width)
    w.string(distortion_model)
    w.f64_seq(list(d))
    w.f64_array(list(k))
    w.f64_array(list(r))
    w.f64_array(list(p))
    w.u32(0)
    w.u32(0)
    for value in (0, 0, 0, 0):
        w.u32(value)
    w.boolean(False)
    return w.bytes()


# The two high-rate messages (240 Hz joint states and transforms) have fast
# encoders: their strings repeat every sample, so the encoded name blocks are
# cached, and the numbers are packed in one call per field.
_STRINGS: dict[str, bytes] = {}
_NAME_BLOCKS: dict[tuple[str, tuple[str, ...]], bytes] = {}
_POSE = struct.Struct("<7d")


def _cdr_string(value: str) -> bytes:
    """A CDR string (length, bytes, NUL), unaligned."""
    out = _STRINGS.get(value)
    if out is None:
        raw = value.encode()
        out = _U32.pack(len(raw) + 1) + raw + b"\x00"
        _STRINGS[value] = out
    return out


def _name_block(frame_id: str, names: tuple[str, ...]) -> bytes:
    """A JointState's bytes after its stamp, through the name sequence."""
    key = (frame_id, names)
    out = _NAME_BLOCKS.get(key)
    if out is None:
        w = CdrWriter()
        w.time(0)
        w.string(frame_id)
        w.string_seq(names)
        out = bytes(w.buf[4 + 8 :])
        _NAME_BLOCKS[key] = out
    return out


def joint_state(
    stamp_ns: int,
    names: Sequence[str],
    position: Sequence[float],
    velocity: Sequence[float] = (),
    effort: Sequence[float] = (),
    frame_id: str = "",
) -> bytes:
    """``sensor_msgs/JointState``."""
    sec, nanosec = divmod(int(stamp_ns), 1_000_000_000)
    buf = bytearray(_ENCAPSULATION)
    buf += _TIME.pack(sec, nanosec)
    buf += _name_block(frame_id, tuple(names))
    for values in (position, velocity, effort):
        pad = -(len(buf) - 4) % 4
        if pad:
            buf += _ZEROS[:pad]
        n = len(values)
        buf += _U32.pack(n)
        if n:
            pad = -(len(buf) - 4) % 8
            if pad:
                buf += _ZEROS[:pad]
            buf += struct.pack(f"<{n}d", *values)
    return bytes(buf)


def tf_message(
    transforms: Iterable[tuple[int, str, str, Sequence[float], Sequence[float]]],
) -> bytes:
    """``tf2_msgs/TFMessage`` from ``(stamp_ns, parent, child, xyz, xyzw)`` tuples."""
    items = list(transforms)
    buf = bytearray(_ENCAPSULATION)
    buf += _U32.pack(len(items))
    for stamp_ns, parent, child, xyz, quat in items:
        # Each item starts 4-aligned: after the count, or after 8-aligned doubles.
        sec, nanosec = divmod(int(stamp_ns), 1_000_000_000)
        buf += _TIME.pack(sec, nanosec)
        buf += _cdr_string(parent)
        pad = -(len(buf) - 4) % 4
        if pad:
            buf += _ZEROS[:pad]
        buf += _cdr_string(child)
        pad = -(len(buf) - 4) % 8
        if pad:
            buf += _ZEROS[:pad]
        buf += _POSE.pack(
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
            float(quat[0]),
            float(quat[1]),
            float(quat[2]),
            float(quat[3]),
        )
    return bytes(buf)


def pose_stamped(stamp_ns: int, frame_id: str, pose: Pose7) -> bytes:
    """``geometry_msgs/PoseStamped`` from ``[x, y, z, qx, qy, qz, qw]``."""
    w = CdrWriter()
    w.header(stamp_ns, frame_id)
    w.vector3(pose[:3])
    w.quaternion(pose[3:7])
    return w.bytes()


def twist_stamped(
    stamp_ns: int, frame_id: str, linear: Sequence[float], angular: Sequence[float]
) -> bytes:
    """``geometry_msgs/TwistStamped``."""
    w = CdrWriter()
    w.header(stamp_ns, frame_id)
    w.vector3(linear)
    w.vector3(angular)
    return w.bytes()


def odometry(
    stamp_ns: int,
    frame_id: str,
    child_frame_id: str,
    pose: Pose7,
    linear: Sequence[float],
    angular: Sequence[float],
    pose_covariance: Sequence[float] = (0.0,) * 36,
    twist_covariance: Sequence[float] = (0.0,) * 36,
) -> bytes:
    """``nav_msgs/Odometry`` (twist in the child frame, as ROS defines it)."""
    w = CdrWriter()
    w.header(stamp_ns, frame_id)
    w.string(child_frame_id)
    w.vector3(pose[:3])
    w.quaternion(pose[3:7])
    w.f64_array(list(pose_covariance))
    w.vector3(linear)
    w.vector3(angular)
    w.f64_array(list(twist_covariance))
    return w.bytes()


def joy(
    stamp_ns: int, frame_id: str, axes: Sequence[float], buttons: Sequence[int]
) -> bytes:
    """``sensor_msgs/Joy``."""
    w = CdrWriter()
    w.header(stamp_ns, frame_id)
    w.f32_seq(axes)
    w.i32_seq(buttons)
    return w.bytes()


def command_info(
    stamp_ns: int,
    command_id: int,
    issue_ns: int,
    names: Sequence[str],
    target_device_id: Sequence[str],
    control_mode: Sequence[str],
    application_ns: Sequence[int] = (),
) -> bytes:
    """``almond_msgs/CommandInfo``: the facts about one command's targets.

    ``application_ns`` has one time per joint (0 when unknown).
    """
    w = CdrWriter()
    w.header(stamp_ns, "")
    w.u64(command_id)
    w.time(issue_ns)
    w.string_seq(names)
    w.string_seq(target_device_id)
    w.string_seq(control_mode)
    w.u32(len(application_ns))
    for ns in application_ns:
        w.time(ns)
    return w.bytes()


def base_command(
    stamp_ns: int,
    command_id: int,
    issue_ns: int,
    target_device_id: str,
    control_mode: str,
    vx: float,
    vy: float,
    wz: float,
    lift_direction: int,
) -> bytes:
    """``almond_msgs/BaseCommand``: one drive and lift command to Jelly."""
    w = CdrWriter()
    w.header(stamp_ns, "")
    w.u64(command_id)
    w.time(issue_ns)
    w.string(target_device_id)
    w.string(control_mode)
    w.f64(vx)
    w.f64(vy)
    w.f64(wz)
    w.i8(lift_direction)
    return w.bytes()


_NO_ORIENTATION = (-1.0,) + (0.0,) * 8
_UNKNOWN_COVARIANCE = (0.0,) * 9


def imu(
    stamp_ns: int,
    frame_id: str,
    angular_velocity: Sequence[float],
    linear_acceleration: Sequence[float],
) -> bytes:
    """``sensor_msgs/Imu`` without an orientation (its covariance starts at -1).

    Angular velocity in rad/s, linear acceleration in m/s^2 including gravity;
    a zero covariance means it is unknown, as ROS defines it.
    """
    w = CdrWriter()
    w.header(stamp_ns, frame_id)
    w.quaternion((0.0, 0.0, 0.0, 1.0))
    w.f64_array(_NO_ORIENTATION)
    w.vector3(angular_velocity)
    w.f64_array(_UNKNOWN_COVARIANCE)
    w.vector3(linear_acceleration)
    w.f64_array(_UNKNOWN_COVARIANCE)
    return w.bytes()


def string_message(data: str) -> bytes:
    """``std_msgs/String``."""
    w = CdrWriter()
    w.string(data)
    return w.bytes()
