"""Check an MCAP episode recorded by Collect data.

Run ``axol mcap.check <file>`` on the robot. The module imports only the
``mcap`` and ``mcap-ros2-support`` packages and the standard library, so it
also runs anywhere those two are installed::

    python check.py episode_000000.mcap

The checks follow what the file says was recorded (its cameras, its video
codec, whether Axol was on Jelly), so any camera setup can pass. Each line is
``PASS``, ``WARN`` or ``FAIL``:

- ``FAIL``: the file is broken or contradicts itself (no video, a frame that
  is not a keyframe, timestamps out of order, a repeated frame, calibration
  for the wrong image size, a camera the transform tree can't reach, a broken
  transform tree, camera mounts that differ from the attached calibration
  file, missing joint states or metadata).
- ``WARN``: something optional is absent, such as a camera with no factory
  calibration file, camera mounts from CAD, or no headset input.

The exit status is 1 when any check fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# NAL unit types of keyframes: H.264 IDR, H.265 BLA, IDR and CRA.
_KEYFRAME_NALS = {"h264": (5,), "h265": tuple(range(16, 22))}
# How close a recorded camera mount must be to its calibration file entry
# (metres, and quaternion components).
_MOUNT_TOLERANCE = 1e-6
_CALIBRATION_ATTACHMENT = "camera_calibration.json"

Pose = tuple[tuple[float, float, float], tuple[float, float, float, float]]


@dataclass
class Result:
    status: str  # PASS, WARN or FAIL
    item: str
    detail: str


@dataclass
class EpisodeSummary:
    """What :func:`summarize` read from one file."""

    times: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    decoded: dict[str, list[Any]] = field(default_factory=lambda: defaultdict(list))
    metadata: dict[str, dict[str, str]] = field(default_factory=dict)
    video_formats: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    video_frames: dict[str, str] = field(default_factory=dict)
    non_keyframes: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    repeated_frames: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    attachments: dict[str, bytes] = field(default_factory=dict)


def _video_parts(data: bytes) -> tuple[str, bytes, str]:
    """``(frame_id, payload, format)`` of a CDR ``foxglove_msgs/CompressedVideo``."""

    def u32(at: int) -> int:
        return int.from_bytes(data[at : at + 4], "little")

    def align4(at: int) -> int:
        return at + (-(at - 4) % 4)

    at = 4 + 8  # encapsulation header, timestamp
    frame_len = u32(at)
    frame_id = data[at + 4 : at + 4 + frame_len - 1].decode(errors="replace")
    at = align4(at + 4 + frame_len)
    size = u32(at)
    payload = data[at + 4 : at + 4 + size]
    at = align4(at + 4 + size)
    fmt = data[at + 4 : at + 4 + u32(at) - 1].decode(errors="replace")
    return frame_id, payload, fmt


def _is_keyframe(payload: bytes, codec: str) -> bool:
    wanted = _KEYFRAME_NALS.get(codec, ())
    i, n = 0, len(payload)
    while i + 3 < n:
        if payload[i] == 0 and payload[i + 1] == 0 and payload[i + 2] == 1:
            header = payload[i + 3]
            nal = (header >> 1) & 0x3F if codec == "h265" else header & 0x1F
            if nal in wanted:
                return True
            i += 3
        else:
            i += 1
    return False


def summarize(path: Path) -> EpisodeSummary:
    """Read every record of ``path`` (video payloads are not decoded)."""
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    out = EpisodeSummary()
    factory = DecoderFactory()
    decoders: dict[int, Any] = {}
    last_payload: dict[str, bytes] = {}
    with open(path, "rb") as f:
        reader = make_reader(f)
        for record in reader.iter_metadata():
            out.metadata[record.name] = dict(record.metadata)
        for attachment in reader.iter_attachments():
            out.attachments[attachment.name] = bytes(attachment.data)
        # Renamed topics are checked under their default names.
        default_name = {
            new: old
            for old, new in _topic_names(
                out.metadata.get("recording", {}).get("topic_names", "")
            ).items()
        }
        for schema, channel, message in reader.iter_messages(log_time_order=True):
            topic = default_name.get(channel.topic, channel.topic)
            out.times[topic].append(message.log_time)
            if topic.endswith("/video"):
                frame_id, payload, fmt = _video_parts(bytes(message.data))
                out.video_formats[topic].add(fmt)
                out.video_frames.setdefault(topic, frame_id)
                if not _is_keyframe(payload, fmt):
                    out.non_keyframes[topic] += 1
                if last_payload.get(topic) == payload:
                    out.repeated_frames[topic] += 1
                last_payload[topic] = payload
                continue
            decoder = decoders.get(channel.id)
            if decoder is None and schema is not None:
                decoder = factory.decoder_for(channel.message_encoding, schema)
                decoders[channel.id] = decoder
            if decoder is not None:
                out.decoded[topic].append(decoder(message.data))
    return out


def _topic_names(text: str) -> dict[str, str]:
    """The file's ``default=new`` topic (or frame) renames."""
    out = {}
    for pair in text.split(","):
        old, sep, new = (part.strip() for part in pair.partition("="))
        if sep and old and new:
            out[old] = new
    return out


def _quat_mul(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _rotate(q: tuple[float, ...], v: tuple[float, ...]) -> tuple[float, ...]:
    x, y, z, w = q
    p = _quat_mul(_quat_mul(q, (v[0], v[1], v[2], 0.0)), (-x, -y, -z, w))
    return p[:3]


def _compose(a: Pose, b: Pose) -> Pose:
    """``a`` then ``b`` (b given in a's child frame)."""
    t = _rotate(a[1], b[0])
    return (
        (a[0][0] + t[0], a[0][1] + t[1], a[0][2] + t[2]),
        _quat_mul(a[1], b[1]),  # type: ignore[return-value]
    )


def _inverse(a: Pose) -> Pose:
    x, y, z, w = a[1]
    q = (-x, -y, -z, w)
    t = _rotate(q, a[0])
    return ((-t[0], -t[1], -t[2]), q)


def _static_lookup(
    static: dict[str, tuple[str, Pose]], parent: str, child: str
) -> Pose | None:
    """``child``'s pose in ``parent`` through /tf_static alone, if connected."""

    def to_top(frame: str) -> tuple[str, Pose]:
        pose: Pose = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        seen = 0
        while frame in static and seen < 1000:
            up, step = static[frame]
            pose = _compose(step, pose)
            frame = up
            seen += 1
        return frame, pose

    top_p, pose_p = to_top(parent)
    top_c, pose_c = to_top(child)
    if top_p != top_c:
        return None
    return _compose(_inverse(pose_p), pose_c)


def _same_pose(a: Pose, b: Pose, tolerance: float) -> bool:
    dt = max(abs(x - y) for x, y in zip(a[0], b[0], strict=True))
    qa, qb = a[1], b[1]
    sign = 1.0 if sum(x * y for x, y in zip(qa, qb, strict=True)) >= 0 else -1.0
    dq = max(abs(x - sign * y) for x, y in zip(qa, qb, strict=True))
    return dt <= tolerance and dq <= tolerance


def _rate(times: list[int]) -> float:
    if len(times) < 2:
        return 0.0
    steps = [b - a for a, b in zip(times, times[1:], strict=False) if b > a]
    return 1e9 / statistics.median(steps) if steps else 0.0


def _increasing(times: list[int]) -> bool:
    return all(b > a for a, b in zip(times, times[1:], strict=False))


def check(path: Path) -> list[Result]:
    s = summarize(path)
    results: list[Result] = []

    def add(ok: bool | None, item: str, detail: str) -> None:
        status = "PASS" if ok else ("WARN" if ok is None else "FAIL")
        results.append(Result(status, item, detail))

    meta = s.metadata
    recording = meta.get("recording", {})
    robot = meta.get("robot", {})
    calibration = meta.get("calibration", {})
    codec = recording.get("video_codec", "")

    # Transform tree: each child's parent, from /tf_static and /tf.
    static = [t for m in s.decoded.get("/tf_static", []) for t in m.transforms]
    dynamic = [t for m in s.decoded.get("/tf", []) for t in m.transforms]
    parents = {t.child_frame_id: t.header.frame_id for t in [*static, *dynamic]}
    roots = {p for p in parents.values() if p not in parents}
    static_poses: dict[str, tuple[str, Pose]] = {
        t.child_frame_id: (
            t.header.frame_id,
            (
                (
                    t.transform.translation.x,
                    t.transform.translation.y,
                    t.transform.translation.z,
                ),
                (
                    t.transform.rotation.x,
                    t.transform.rotation.y,
                    t.transform.rotation.z,
                    t.transform.rotation.w,
                ),
            ),
        )
        for t in static
    }
    _check_tf(s, add, static, dynamic, roots)

    # Video and calibration, for every camera the file has.
    cameras = sorted(topic[: -len("/video")] for topic in s.video_frames)
    add(bool(cameras), "cameras", ", ".join(cameras) or "no video in the file")
    for prefix in cameras:
        topic = f"{prefix}/video"
        times = s.times[topic]
        formats = s.video_formats[topic]
        fmt = ",".join(sorted(formats))
        add(len(formats) == 1 and fmt == (codec or fmt), f"{topic} codec", fmt)
        add(
            s.non_keyframes[topic] == 0,
            f"{topic} keyframes",
            f"{len(times) - s.non_keyframes[topic]} of {len(times)} frames are keyframes",
        )
        add(_increasing(times), f"{topic} timestamps", "each frame at its own time")
        add(
            s.repeated_frames[topic] == 0,
            f"{topic} repeats",
            f"{s.repeated_frames[topic]} repeated frames",
        )
        key = prefix.strip("/").replace("/", "_")
        width = recording.get(f"{key}.width", "?")
        height = recording.get(f"{key}.height", "?")
        add(
            True,
            f"{topic} rate",
            f"{_rate(times):.1f} fps measured, {width}x{height}, "
            f"{recording.get(f'{key}.missing_frames', '?')} missing frames",
        )

        frame = s.video_frames[topic]
        chain, link = [frame], frame
        while link in parents and len(chain) < 100:
            link = parents[link]
            chain.append(link)
        reached = frame in parents and chain[-1] in roots
        add(
            reached,
            f"TF to {frame}",
            " <- ".join(reversed(chain)) if reached else "not in the tree",
        )

        infos = s.decoded.get(f"{prefix}/camera_info", [])
        if not infos:
            add(None, f"{prefix}/camera_info", "missing (no factory calibration file)")
            continue
        info = infos[0]
        add(
            str(info.width) == width and str(info.height) == height,
            f"{prefix}/camera_info",
            f"{info.width}x{info.height} {info.distortion_model} "
            f"fx={info.k[0]:.1f} fy={info.k[4]:.1f} d={list(info.d)}",
        )

    for topic in sorted(t for t in s.times if t.endswith("/imu")):
        times = s.times[topic]
        add(
            _increasing(times),
            topic,
            f"{len(times)} samples at {_rate(times):.0f} Hz, each at its own time",
        )
    if recording.get("imu_missing"):
        add(None, "camera IMU", recording["imu_missing"])

    if "/head/left" in cameras and "/head/right" in cameras:
        stereo = calibration.get("head_stereo", "")
        add(True if "factory" in stereo else None, "head stereo", stereo or "unknown")

    # Joints and commands.
    joints = s.times.get("/joint_states", [])
    rate = _rate(joints)
    add(rate >= 200, "/joint_states", f"{len(joints)} samples at {rate:.0f} Hz")
    if s.decoded.get("/joint_states"):
        first = s.decoded["/joint_states"][0]
        add(
            len(first.velocity) == len(first.name) == len(first.effort),
            "/joint_states fields",
            f"{len(first.name)} joints with position, velocity and effort",
        )
    commands = s.times.get("/joint_command", [])
    infos = s.decoded.get("/joint_command/info", [])
    add(bool(commands), "/joint_command", f"{len(commands)} commands")
    if infos:
        ids = [c.command_id for c in infos]
        add(
            all(b > a for a, b in zip(ids, ids[1:], strict=False))
            and len(infos) == len(commands)
            and s.times["/joint_command/info"] == commands,
            "/joint_command/info",
            f"one per command, ids {ids[0]} to {ids[-1]}, modes "
            + ",".join(sorted({m for c in infos for m in c.control_mode})),
        )
        applied = sum(
            1
            for c in infos
            if any(t.sec or t.nanosec for t in getattr(c, "application_timestamp", []))
        )
        add(
            True if applied >= 0.95 * len(infos) else None,
            "/joint_command/info application times",
            f"{applied} of {len(infos)} commands have the time the robot applied them",
        )
    elif commands:
        add(False, "/joint_command/info", "missing")
    for side in ("left", "right"):
        if s.times.get(f"/ee_pose/{side}"):
            n = len(s.times.get(f"/ee_target/{side}", []))
            add(n > 0, f"/ee_target/{side}", f"{n} messages")
    for topic in ("/teleop/left/pose", "/teleop/joy"):
        n = len(s.times.get(topic, []))
        add(True if n else None, topic, f"{n} messages")

    # Base.
    if robot.get("jelly") == "yes":
        for topic in (
            "/base/odom",
            "/base/wheel_states",
            "/base/lift",
            "/base/command",
        ):
            n = len(s.times.get(topic, []))
            add(True if n else None, topic, f"{n} messages")

    # Metadata.
    for record, key in (
        ("episode", "task"),
        ("episode", "outcome"),
        ("robot", "software_version"),
    ):
        value = meta.get(record, {}).get(key, "")
        add(bool(value), f"{record}.{key}", value or "missing")
    for record, key in (("robot", "serial"), ("episode", "operator")):
        value = meta.get(record, {}).get(key, "")
        add(True if value else None, f"{record}.{key}", value or "empty")
    _check_calibration(s, add, calibration, static_poses)
    return results


def _check_tf(
    s: EpisodeSummary, add: Any, static: list[Any], dynamic: list[Any], roots: set[str]
) -> None:
    """One /tf_static message before the first frame, and a sound tree."""
    times = s.times.get("/tf_static", [])
    first_frame = min(
        (t[0] for k, t in s.times.items() if k.endswith("/video") and t), default=None
    )
    early = bool(times) and (first_frame is None or times[0] <= first_frame)
    if not times:
        when = "missing"
    elif first_frame is None:
        when = "no camera frames to compare with"
    else:
        when = "before the first camera frame" if early else "after the first frame"
    add(
        len(times) == 1 and early,
        "/tf_static",
        f"{len(times)} message(s), {when}, {len(static)} transforms",
    )
    lengths = [
        math.sqrt(
            t.transform.rotation.x**2
            + t.transform.rotation.y**2
            + t.transform.rotation.z**2
            + t.transform.rotation.w**2
        )
        for t in static
    ]
    worst = max((abs(n - 1.0) for n in lengths), default=0.0)
    add(
        worst <= _MOUNT_TOLERANCE,
        "/tf_static rotations",
        f"unit quaternions (worst {worst:.1e})",
    )
    children = [t.child_frame_id for t in static]
    repeated = sorted({c for c in children if children.count(c) > 1})
    moving = {t.child_frame_id for t in dynamic}
    both = sorted(set(children) & moving)
    add(
        not repeated and not both,
        "TF children",
        "each frame has one parent"
        if not repeated and not both
        else "placed twice: " + ", ".join(repeated + both),
    )
    add(len(roots) == 1, "TF root", ", ".join(sorted(roots)) or "none")


def _check_calibration(
    s: EpisodeSummary,
    add: Any,
    calibration: dict[str, str],
    static_poses: dict[str, tuple[str, Pose]],
) -> None:
    """The camera mounts: measured (and equal to the attached file) or CAD."""
    if not calibration:
        add(False, "calibration", "no calibration metadata")
        return
    status = calibration.get("calibration_status", "")
    detail = {
        "calibrated": "camera mounts measured",
        "partial": "some camera mounts measured, the rest from CAD",
        "uncalibrated": "camera mounts from CAD",
    }.get(status, f"unknown status {status!r}")
    if calibration.get("calibration_id"):
        detail += (
            f", calibration {calibration['calibration_id']} from "
            f"{calibration.get('calibrated_at') or 'an unknown date'}"
        )
    if calibration.get("calibration_problem"):
        detail += f" ({calibration['calibration_problem']})"
    add(
        True
        if status == "calibrated"
        else (None if status in ("partial", "uncalibrated") else False),
        "calibration",
        detail,
    )
    add(
        bool(calibration.get("tf_sources")),
        "calibration tf_sources",
        "the source of every static transform"
        if calibration.get("tf_sources")
        else "missing",
    )
    raw = s.attachments.get(_CALIBRATION_ATTACHMENT)
    if raw is None:
        if status in ("calibrated", "partial"):
            add(False, "calibration file", f"{_CALIBRATION_ATTACHMENT} is not attached")
        return
    digest = hashlib.sha256(raw).hexdigest()
    add(
        digest == calibration.get("calibration_file_sha256"),
        "calibration file",
        f"attached, SHA-256 {digest[:16]}",
    )
    try:
        doc = json.loads(raw.decode())
        entries = list(doc["transforms"])
    except (ValueError, KeyError, TypeError):
        add(False, "calibration file", "the attached file is not readable")
        return
    sources = dict(
        pair.strip().split("=", 1)
        for pair in calibration.get("tf_sources", "").split(",")
        if "=" in pair
    )
    # The file may name frames by their defaults; the episode by its renames.
    renamed = _topic_names(s.metadata.get("recording", {}).get("frame_names", ""))
    for entry in entries:
        parent = renamed.get(str(entry.get("parent")), str(entry.get("parent")))
        child = renamed.get(str(entry.get("child")), str(entry.get("child")))
        if sources.get(child) != "calibration":
            add(None, f"mount of {child}", "in the calibration file but not used")
            continue
        q = [float(v) for v in entry["rotation_xyzw"]]
        n = math.sqrt(sum(v * v for v in q))
        want: Pose = (
            tuple(float(v) for v in entry["translation_m"]),  # type: ignore[assignment]
            tuple(v / n for v in q),  # type: ignore[assignment]
        )
        got = _static_lookup(static_poses, parent, child)
        add(
            got is not None and _same_pose(got, want, _MOUNT_TOLERANCE),
            f"mount of {child}",
            f"{parent} to {child} matches the calibration file"
            if got is not None and _same_pose(got, want, _MOUNT_TOLERANCE)
            else f"{parent} to {child} differs from the calibration file",
        )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="axol mcap.check",
        description="Check MCAP episodes recorded by Collect data.",
    )
    parser.add_argument("files", nargs="+", type=Path, help="MCAP episode files")
    args = parser.parse_args(list(argv) if argv is not None else None)
    failed = False
    for path in args.files:
        print(f"{path}:")
        for result in check(path):
            print(f"  {result.status}  {result.item}: {result.detail}")
            failed |= result.status == "FAIL"
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
