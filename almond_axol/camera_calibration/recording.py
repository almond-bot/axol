"""Board views from a calibration recording (an MCAP file).

A calibration session records like Collect data: every camera at the
production image mode, the head camera's IMU, and the joints, all on one
clock. The recording's ``episode`` metadata lists its **phases** (JSON in
``calibration_phases``):

- ``head``: the board moves in front of the head camera;
- ``table``: the board lies still on the table (``placement`` numbers it)
  while one arm (``arm``) moves its wrist camera around it.

This module finds the moments each camera held still relative to the board,
decodes those frames, finds the board, and returns one :class:`Frame` per
usable moment with the joint readings at its exposure time. Frames are
never dropped for their error; only the usable-frame rule (markers and
corners) and stillness decide.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .board import Board, Detection
from .optics import Lens

# Topics a calibration recording uses (the recorder's defaults).
CAMERAS = {
    "head_left": "/head/left",
    "head_right": "/head/right",
    "left_wrist": "/wrist/left",
    "right_wrist": "/wrist/right",
}
# An arm is still when no joint moves faster than this (rad/s) ...
_STILL_RAD_S = 0.01
# ... for this long around the frame (s).
_STILL_S = 0.25
# A held board moves less than this between frames (px, median corner).
_HELD_PX = 0.6


@dataclass
class Phase:
    kind: str  # "head" or "table"
    start_ns: int
    end_ns: int
    placement: int = 0
    arm: str = ""


@dataclass
class Frame:
    """One camera's usable view of the board at one moment."""

    camera: str  # a key of CAMERAS
    stamp_ns: int
    detection: Detection
    joints: dict[str, float]
    phase: str
    placement: int = 0


@dataclass
class Recording:
    """What the calibration needs from one recording."""

    path: Path
    recording_id: str
    sha256: str
    metadata: dict[str, dict[str, str]]
    phases: list[Phase]
    lenses: dict[str, Lens]
    intrinsics: dict[str, dict[str, Any]]
    frames: dict[str, list[Frame]] = field(default_factory=dict)
    corners: list[dict[str, Any]] = field(default_factory=list)
    stereo: np.ndarray | None = None  # right eye in the left (factory)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _lens(info: Any) -> tuple[Lens, dict[str, Any]]:
    k = list(info.k)
    data = {
        "width": int(info.width),
        "height": int(info.height),
        "model": str(info.distortion_model),
        "fx": float(k[0]),
        "fy": float(k[4]),
        "cx": float(k[2]),
        "cy": float(k[5]),
        "d": [float(v) for v in info.d],
    }
    lens = Lens(
        width=data["width"],
        height=data["height"],
        k=np.array([[k[0], 0.0, k[2]], [0.0, k[4], k[5]], [0.0, 0.0, 1.0]]),
        d=np.asarray(info.d, dtype=np.float64),
        fisheye=data["model"] == "equidistant",
    )
    return lens, data


def phases_json(phases: Iterable[Phase]) -> str:
    return json.dumps(
        [
            {
                "kind": p.kind,
                "start_ns": p.start_ns,
                "end_ns": p.end_ns,
                "placement": p.placement,
                "arm": p.arm,
            }
            for p in phases
        ]
    )


def _still_windows(
    times: np.ndarray, positions: np.ndarray, *, threshold: float = _STILL_RAD_S
) -> list[tuple[int, int]]:
    """``(start_ns, end_ns)`` spans where every column of ``positions`` holds
    still for at least ``_STILL_S``."""
    if len(times) < 3:
        return []
    dt = np.diff(times) / 1e9
    speed = np.max(np.abs(np.diff(positions, axis=0)), axis=1) / np.maximum(dt, 1e-6)
    still = speed < threshold
    spans: list[tuple[int, int]] = []
    start = None
    for i, s in enumerate(still):
        if s and start is None:
            start = i
        if (not s or i == len(still) - 1) and start is not None:
            end = i if not s else i + 1
            if (times[end] - times[start]) / 1e9 >= _STILL_S:
                spans.append((int(times[start]), int(times[end])))
            start = None
    return spans


def read(
    path: Path,
    board: Board,
    *,
    arm_joints: dict[str, list[str]],
    progress: Any = None,
) -> Recording:
    """Every usable still view of the board in the recording at ``path``.

    ``arm_joints`` maps ``left`` / ``right`` to that arm's URDF joint names
    (what ``/joint_states`` calls them).
    """
    import av
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    path = Path(path)
    metadata: dict[str, dict[str, str]] = {}
    infos: dict[str, Any] = {}
    joint_t: list[int] = []
    joint_q: list[list[float]] = []
    names: list[str] = []
    video: dict[str, list[tuple[int, bytes, str]]] = defaultdict(list)
    topic_camera = {f"{prefix}/video": key for key, prefix in CAMERAS.items()}
    info_camera = {f"{prefix}/camera_info": key for key, prefix in CAMERAS.items()}
    factory = DecoderFactory()
    decoders: dict[int, Any] = {}
    stereo = None
    with open(path, "rb") as f:
        reader = make_reader(f)
        for record in reader.iter_metadata():
            metadata[record.name] = dict(record.metadata)
        for schema, channel, message in reader.iter_messages(log_time_order=True):
            topic = channel.topic
            decoder = decoders.get(channel.id)
            if decoder is None and schema is not None:
                decoder = factory.decoder_for(channel.message_encoding, schema)
                decoders[channel.id] = decoder
            if topic in topic_camera:
                msg = decoder(message.data)
                video[topic_camera[topic]].append(
                    (message.log_time, bytes(msg.data), str(msg.format))
                )
            elif topic in info_camera and info_camera[topic] not in infos:
                infos[info_camera[topic]] = decoder(message.data)
            elif topic == "/joint_states":
                msg = decoder(message.data)
                if not names:
                    names = list(msg.name)
                joint_t.append(message.log_time)
                joint_q.append(list(msg.position))
            elif topic == "/tf_static" and stereo is None:
                msg = decoder(message.data)
                for t in msg.transforms:
                    if t.child_frame_id == "head_camera_right_optical":
                        from .file import quaternion_matrix

                        stereo = np.eye(4)
                        r = t.transform.rotation
                        stereo[:3, :3] = quaternion_matrix((r.x, r.y, r.z, r.w))
                        tr = t.transform.translation
                        stereo[:3, 3] = (tr.x, tr.y, tr.z)
    episode = metadata.get("episode", {})
    phases = [Phase(**p) for p in json.loads(episode.get("calibration_phases", "[]"))]
    lenses: dict[str, Lens] = {}
    intrinsics: dict[str, dict[str, Any]] = {}
    for key, info in infos.items():
        lenses[key], intrinsics[key] = _lens(info)
    out = Recording(
        path=path,
        recording_id=episode.get("uuid", ""),
        sha256=_sha256(path),
        metadata=metadata,
        phases=phases,
        lenses=lenses,
        intrinsics=intrinsics,
        stereo=stereo,
    )
    times = np.asarray(joint_t, dtype=np.int64)
    q = np.asarray(joint_q, dtype=np.float64)
    column = {n: i for i, n in enumerate(names)}

    def joints_at(stamp: int) -> dict[str, float]:
        i = int(np.clip(np.searchsorted(times, stamp), 1, len(times) - 1))
        w = (stamp - times[i - 1]) / max(times[i] - times[i - 1], 1)
        row = (1 - w) * q[i - 1] + w * q[i]
        return {n: float(row[c]) for n, c in column.items()}

    # Moments to look at: the arm's still spans in a table phase (for every
    # camera), and every frame of a head phase (the board moves by hand).
    wanted: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    for phase in phases:
        if phase.kind == "table":
            cols = [column[n] for n in arm_joints.get(phase.arm, []) if n in column]
            inside = (times >= phase.start_ns) & (times <= phase.end_ns)
            for start, end in _still_windows(times[inside], q[inside][:, cols]):
                middle = (start + end) // 2
                for key in CAMERAS:
                    wanted[key].append((middle, "table", phase.placement))

    def decode(key: str, au: bytes, fmt: str) -> np.ndarray | None:
        ctx = av.CodecContext.create("hevc" if fmt == "h265" else "h264", "r")
        frames = ctx.decode(av.Packet(au)) + ctx.decode(None)
        return frames[0].to_ndarray(format="gray") if frames else None

    for key, messages in video.items():
        stamps = np.array([m[0] for m in messages], dtype=np.int64)
        picked: dict[int, tuple[str, int]] = {}
        for middle, kind, placement in wanted.get(key, []):
            i = int(np.argmin(np.abs(stamps - middle)))
            picked[i] = (kind, placement)
        frames: list[Frame] = []
        for i in sorted(picked):
            stamp, au, fmt = messages[i]
            image = decode(key, au, fmt)
            detection = board.detect(image) if image is not None else None
            kind, placement = picked[i]
            out.corners.append(_corner_record(key, stamp, detection, kind, placement))
            if board.usable(detection):
                frames.append(
                    Frame(key, stamp, detection, joints_at(stamp), kind, placement)
                )
        # Head phases: every frame, kept where the board held still.
        for phase in phases:
            if phase.kind != "head" or not key.startswith("head"):
                continue
            inside = [
                i for i, s in enumerate(stamps) if phase.start_ns <= s <= phase.end_ns
            ]
            previous: Detection | None = None
            run: list[tuple[int, Detection]] = []

            def close(run: list[tuple[int, Detection]]) -> None:
                if len(run) * 1.0 < 2:
                    return
                i, detection = run[len(run) // 2]
                stamp = int(stamps[i])
                out.corners.append(_corner_record(key, stamp, detection, "head", 0))
                if board.usable(detection):
                    frames.append(
                        Frame(key, stamp, detection, joints_at(stamp), "head")
                    )

            for i in inside:
                stamp, au, fmt = messages[i]
                image = decode(key, au, fmt)
                detection = board.detect(image) if image is not None else None
                if (
                    detection is not None
                    and previous is not None
                    and _held(previous, detection)
                ):
                    run.append((i, detection))
                else:
                    close(run)
                    run = [(i, detection)] if detection is not None else []
                previous = detection
            close(run)
        frames.sort(key=lambda fr: fr.stamp_ns)
        out.frames[key] = frames
        if progress is not None:
            progress(f"{path.name}: {key}: {len(frames)} usable views")
    return out


def _held(a: Detection, b: Detection) -> bool:
    common, ia, ib = np.intersect1d(a.ids, b.ids, return_indices=True)
    if len(common) < 6:
        return False
    moved = np.linalg.norm(a.pixels[ia] - b.pixels[ib], axis=1)
    return bool(np.median(moved) < _HELD_PX)


def _corner_record(
    camera: str, stamp: int, detection: Detection | None, phase: str, placement: int
) -> dict[str, Any]:
    return {
        "camera": camera,
        "stamp_ns": int(stamp),
        "phase": phase,
        "placement": placement,
        "markers": 0 if detection is None else int(detection.markers),
        "ids": [] if detection is None else [int(v) for v in detection.ids],
        "pixels": []
        if detection is None
        else [[round(float(x), 4), round(float(y), 4)] for x, y in detection.pixels],
    }


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    r = a[:3, :3].T @ b[:3, :3]
    return math.degrees(math.acos(min(1.0, max(-1.0, (np.trace(r) - 1) / 2))))
