"""Check an MCAP episode against what Collect data's MCAP format promises.

Run ``axol mcap.check <file>`` on the robot. The module imports only the
``mcap`` and ``mcap-ros2-support`` packages and the standard library, so it
also runs anywhere those two are installed::

    python check.py episode_000000.mcap

Each line is ``PASS``, ``WARN`` or ``FAIL`` with what was checked. The exit
status is 1 when any check fails.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CAMERAS = {
    "/head/left": "head_camera_left_optical",
    "/head/right": "head_camera_right_optical",
    "/wrist/left": "left_wrist_camera_optical",
    "/wrist/right": "right_wrist_camera_optical",
}
# H.265 NAL unit types of keyframes (BLA, IDR, CRA).
_H265_KEYFRAMES = range(16, 22)


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
    schemas: dict[str, str] = field(default_factory=dict)
    video_formats: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    non_keyframes: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    repeated_frames: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def _video_parts(data: bytes) -> tuple[bytes, str]:
    """``(payload, format)`` of a CDR ``foxglove_msgs/CompressedVideo``."""

    def u32(at: int) -> int:
        return int.from_bytes(data[at : at + 4], "little")

    def align4(at: int) -> int:
        return at + (-(at - 4) % 4)

    at = 4 + 8  # encapsulation header, timestamp
    at += 4 + u32(at)  # frame_id
    at = align4(at)
    size = u32(at)
    payload = data[at + 4 : at + 4 + size]
    at = align4(at + 4 + size)
    fmt = data[at + 4 : at + 4 + u32(at) - 1].decode(errors="replace")
    return payload, fmt


def _h265_is_keyframe(payload: bytes) -> bool:
    i, n = 0, len(payload)
    while i + 3 < n:
        if payload[i] == 0 and payload[i + 1] == 0 and payload[i + 2] == 1:
            if ((payload[i + 3] >> 1) & 0x3F) in _H265_KEYFRAMES:
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
        for schema, channel, message in reader.iter_messages(log_time_order=True):
            topic = channel.topic
            out.times[topic].append(message.log_time)
            out.schemas[topic] = schema.name if schema is not None else ""
            if topic.endswith("/video"):
                payload, fmt = _video_parts(bytes(message.data))
                out.video_formats[topic].add(fmt)
                if not _h265_is_keyframe(payload):
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


def _rate(times: list[int]) -> float:
    if len(times) < 2:
        return 0.0
    steps = [b - a for a, b in zip(times, times[1:], strict=False) if b > a]
    return 1e9 / statistics.median(steps) if steps else 0.0


def check(path: Path) -> list[Result]:
    s = summarize(path)
    results: list[Result] = []

    def add(ok: bool | None, item: str, detail: str) -> None:
        status = "PASS" if ok else ("WARN" if ok is None else "FAIL")
        results.append(Result(status, item, detail))

    meta = s.metadata
    recording = meta.get("recording", {})
    episode = meta.get("episode", {})
    robot = meta.get("robot", {})
    calibration = meta.get("calibration", {})

    # 1 + 2: video, timestamps.
    for prefix in CAMERAS:
        topic = f"{prefix}/video"
        times = s.times.get(topic, [])
        if not times:
            add(False, f"{topic}", "no frames")
            continue
        fmt = ",".join(sorted(s.video_formats[topic]))
        add(fmt == "h265", f"{topic} codec", fmt or "unknown")
        add(
            s.non_keyframes[topic] == 0,
            f"{topic} keyframes",
            f"{len(times) - s.non_keyframes[topic]} of {len(times)} frames are keyframes",
        )
        ordered = all(b > a for a, b in zip(times, times[1:], strict=False))
        add(ordered, f"{topic} timestamps", "each frame at its own, increasing time")
        add(
            s.repeated_frames[topic] == 0,
            f"{topic} repeats",
            f"{s.repeated_frames[topic]} repeated frames",
        )
        key = prefix.strip("/").replace("/", "_")
        gaps = recording.get(f"{key}.missing_frames", "?")
        add(
            True,
            f"{topic} rate",
            f"{_rate(times):.1f} fps measured, "
            f"{recording.get(f'{key}.width', '?')}x{recording.get(f'{key}.height', '?')}, "
            f"{gaps} missing frames",
        )

    # 3: joints.
    joints = s.times.get("/joint_states", [])
    rate = _rate(joints)
    add(rate >= 200, "/joint_states", f"{len(joints)} samples at {rate:.0f} Hz")
    if s.decoded.get("/joint_states"):
        first = s.decoded["/joint_states"][0]
        add(
            len(first.velocity) == len(first.name)
            and len(first.effort) == len(first.name),
            "/joint_states fields",
            f"{len(first.name)} joints with position, velocity and effort",
        )

    # 4: base.
    if robot.get("jelly") == "yes":
        for topic in ("/base/odom", "/base/wheel_states", "/base/lift"):
            n = len(s.times.get(topic, []))
            add(n > 0, topic, f"{n} messages")
    else:
        add(None, "/base", "not on Jelly, no base topics")

    # 5, 9, 10: transforms.
    static = [t for m in s.decoded.get("/tf_static", []) for t in m.transforms]
    dynamic = [t for m in s.decoded.get("/tf", []) for t in m.transforms]
    parents = {t.child_frame_id: t.header.frame_id for t in [*static, *dynamic]}
    static_children = {t.child_frame_id for t in static}
    roots = {p for p in parents.values() if p not in parents}
    for prefix, frame in CAMERAS.items():
        chain, link = [frame], frame
        while link in parents and len(chain) < 100:
            link = parents[link]
            chain.append(link)
        connected = frame in parents and chain[-1] in roots
        add(
            connected,
            f"TF to {frame}",
            f"{' <- '.join(reversed(chain))}" if connected else "not in the tree",
        )
    add(
        "head_camera_right_optical" in static_children
        and "factory" in calibration.get("head_stereo", ""),
        "head stereo",
        calibration.get("head_stereo", "missing"),
    )
    for frame in ("left_wrist_camera_optical", "right_wrist_camera_optical"):
        add(frame in static_children, f"{frame} on /tf_static", "camera mount")

    # 6: actions.
    for topic in ("/joint_command", "/ee_target/left", "/ee_target/right"):
        n = len(s.times.get(topic, []))
        add(n > 0, topic, f"{n} messages")
    for topic in ("/teleop/left/pose", "/teleop/joy"):
        n = len(s.times.get(topic, []))
        add(True if n else None, topic, f"{n} messages")

    # 7: metadata.
    for record, key in (
        ("episode", "task"),
        ("episode", "outcome"),
        ("robot", "serial"),
        ("robot", "software_version"),
    ):
        value = meta.get(record, {}).get(key, "")
        add(bool(value), f"{record}.{key}", value or "missing")
    operator = episode.get("operator", "")
    add(True if operator else None, "episode.operator", operator or "empty")

    # 8: intrinsics.
    for prefix in CAMERAS:
        infos = s.decoded.get(f"{prefix}/camera_info", [])
        if not infos:
            add(False, f"{prefix}/camera_info", "missing")
            continue
        info = infos[0]
        key = prefix.strip("/").replace("/", "_")
        size_ok = str(info.width) == recording.get(f"{key}.width") and str(
            info.height
        ) == recording.get(f"{key}.height")
        model_ok = info.distortion_model == "equidistant" if "wrist" in prefix else True
        add(
            size_ok and model_ok,
            f"{prefix}/camera_info",
            f"{info.width}x{info.height} {info.distortion_model} "
            f"fx={info.k[0]:.1f} fy={info.k[4]:.1f} d={list(info.d)}",
        )

    # 11: calibration record.
    for key in ("calibration_id", "calibration_date"):
        value = calibration.get(key, "")
        add(bool(value), f"calibration.{key}", value or "missing")
    passed = calibration.get("calibration_passed", "")
    add(
        {"yes": True, "no": False}.get(passed),
        "calibration.calibration_passed",
        f"{passed or 'missing'} (camera mounts: {calibration.get('camera_mounts', '?')})",
    )
    return results


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="axol mcap.check",
        description="Check MCAP episodes from Collect data.",
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
