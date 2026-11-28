"""Synthetic MCAP episodes: the sim's motion and cameras, written by the recorder.

An episode here is what Collect data would record on Axol on Jelly with the
Camera Kit, made without hardware:

- a scripted pick and place, solved with the URDF's kinematics; the arms
  follow their 120 Hz targets with a small lag, as the realtime core does;
- the head camera's two eyes and both wrist cameras rendered from where the
  cameras are, through their factory lens models, and encoded once (every
  frame a keyframe, like the robot's encoder);
- the head camera's IMU, Jelly's wheels, lift and drive commands, and the
  headset input;
- a camera calibration that places the cameras a few millimetres from CAD,
  where the rendered cameras really are.

Everything is played, in real time, into the same
:class:`~almond_axol.recording.mcap.recorder.McapSession` the robot uses, so
the file is written by the production code path.
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing
import os
import shutil
import socket
import tempfile
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from ..constants import AxolModel, urdf_path
from ..recording.mcap.imu import MAGIC, PACKET, POSE_MAGIC, POSE_PACKET, ImuReceiver
from ..recording.mcap.odometry import WHEEL_RADIUS_M, wheel_rims
from ..recording.mcap.rings import (
    JOINT_NAMES,
    JOINT_RING_CAPACITY,
    SIDES,
    TICK_RING_CAPACITY,
    joint_ring_keys,
    pack_tick,
    tick_ring_keys,
)
from ..recording.mcap.scene import CameraSource, Scene, imu_frame_id, imu_topic
from ..recording.mcap.urdf_tree import UrdfTree, quaternion_xyzw
from .motion import ArmKinematics, Path as TcpPath, Waypoint, min_jerk, pose
from .render import LensImager, LensSpec, Prop, SceneRenderer
from .zed_files import write_files

_logger = logging.getLogger(__name__)

# Jelly's wheel signs (almond_axol.robot.jelly.WHEEL_SIGNS), copied so this
# module does not import the robot package.
WHEEL_SIGNS = {1: 1.0, 2: -1.0, 3: 1.0, 4: -1.0}
CONTROL_HZ = 120
JOINT_HZ = 240
IMU_HZ = 200
HEADSET_HZ = 72
WHEEL_HZ = 40
# Fine time grid the arm response is simulated on (s).
_DT = 0.001
# The head camera's IMU in its left optical frame (a typical factory value).
IMU_POSE = np.array(
    [
        [1.0, 0.0012, -0.0008, -0.0231],
        [-0.0012, 1.0, 0.0006, -0.0003],
        [0.0008, -0.0006, 1.0, -0.0017],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
# Where the cameras really are, against CAD (mm, degrees): the calibration
# a calibrated episode carries.
_MOUNT_ERRORS = {
    "head_camera_left_optical": ((2.4, -1.1, 1.8), (0.35, -0.2, 0.15)),
    "left_wrist_camera_optical": ((1.3, 0.9, -1.6), (-0.6, 0.4, 0.3)),
    "right_wrist_camera_optical": ((-1.1, 1.4, 0.8), (0.5, 0.3, -0.45)),
}
_MOUNT_PARENTS = {
    "head_camera_left_optical": "root",
    "left_wrist_camera_optical": "left_w2",
    "right_wrist_camera_optical": "right_w2",
}
# Arm ready poses (URDF joint values), comfortably inside every joint limit.
READY = {
    "left": (1.12, -0.91, -1.36, 1.79, -1.29, -0.75, 1.09),
    "right": (-1.102, 0.907, 1.335, -1.791, -1.832, -0.754, 1.106),
}
# Table and objects, in the URDF's world frame (z = 0 is the torso at full
# lift; the floor is the lowest wheel point).
# The floor: the lowest point of the wheels, in the URDF's world frame.
FLOOR_Z = -0.4694
TABLE_TOP = 0.281
CUBE = 0.025  # half size
CUP = (0.035, 0.05)  # radius, half height


@dataclass
class SyntheticOptions:
    output: Path
    fps: int = 60
    width: int = 1920
    height: int = 1200
    resolution: str = "HD1200"
    codec: str = "h265"
    quality: int = 24
    workers: int = 4
    topic_names: dict[str, str] = field(default_factory=dict)
    frame_names: dict[str, str] = field(default_factory=dict)
    episode_metadata: dict[str, str] = field(default_factory=dict)
    robot_serial: str = "SYNTHETIC-AXOL-0001"
    operator: str = "synthetic"
    software_version: str = ""
    raised: float = 0.8
    duration_s: float = 15.0
    # Render size over the camera's resolution (sharper, slower).
    supersample: float = 1.25


@dataclass
class EpisodeSpec:
    task: str
    script: str  # "cube_to_bin" (right arm) or "cup_to_table" (left arm)
    calibrated: bool
    seed: int = 0
    metadata: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The rig: scene, cameras, true camera mounts


def _nudge(
    transform: np.ndarray, mm: Sequence[float], deg: Sequence[float]
) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    out = transform.copy()
    out[:3, :3] = (
        out[:3, :3] @ Rotation.from_euler("xyz", deg, degrees=True).as_matrix()
    )
    out[:3, 3] += np.asarray(mm) / 1000.0
    return out


def write_joint_zero(path: Path, robot_serial: str, when: datetime) -> dict[str, str]:
    """A zeroing record with both arms zeroed at their end stops at ``when``;
    returns ``{side: zero ID}``."""
    from ..constants import ARM_JOINTS
    from ..robot import joint_zero
    from ..robot.axol import closer_end_stop

    ids = {}
    for side in joint_zero.SIDES:
        joints = {
            joint.value: {
                "motor_id": index + 1,
                "end_stop_deg": round(
                    math.degrees(closer_end_stop(joint, side == "left")[0]), 2
                ),
                "reading_before_rad": 0.0,
            }
            for index, joint in enumerate(ARM_JOINTS)
        }
        event = joint_zero.record(
            side,
            joints,
            joint_zero.METHOD_GUIDED,
            robot_serial=robot_serial,
            path=path,
            now=when,
            hostname="synthetic",
        )
        ids[side] = event["zero_id"]
    return ids


class Rig:
    """The recorder's scene for Axol on Jelly with the Camera Kit, and the
    true camera mounts (a few mm from CAD) the renders use."""

    def __init__(self, work: Path, options: SyntheticOptions) -> None:
        from ..camera_calibration.file import document, parse, transform_entry, utc_text
        from ..recording.mcap.mounts import resolve

        self.options = options
        self.work = work
        self.calibration_dir = work / "zed"
        serials = write_files(self.calibration_dir)
        # A short path: Unix socket names are limited to about 100 bytes.
        self.imu_socket = str(Path(tempfile.mkdtemp(prefix="axsyn-")) / "imu.sock")
        meta = {
            "overhead_left": self._meta("overhead", "left", serials["overhead"], True),
            "overhead_right": self._meta(
                "overhead", "right", serials["overhead"], True
            ),
            "left_arm": self._meta("left_arm", None, serials["left_arm"], False),
            "right_arm": self._meta("right_arm", None, serials["right_arm"], False),
        }
        self.raw_meta = meta
        cameras = {src: CameraSource.from_meta(src, m) for src, m in meta.items()}
        self.urdf = urdf_path(AxolModel.MOBILE, gripper="parallel")
        self.scene = Scene.build(
            self.urdf,
            has_gripper=True,
            jelly_wheels=True,
            jelly_lift=True,
            cameras=cameras,
            calibration_dir=self.calibration_dir,
            gripper="parallel",
            software_version=options.software_version,
        )
        self.tree = self.scene.tree
        self.kin = ArmKinematics(self.tree)
        cad = self.tree
        entries = []
        for child, (mm, deg) in _MOUNT_ERRORS.items():
            parent = _MOUNT_PARENTS[child]
            serial = next(c.serial for c in cameras.values() if c.frame_id == child)
            entries.append(
                transform_entry(
                    parent,
                    child,
                    _nudge(cad.fk(parent, child, {}), mm, deg),
                    camera_serial=serial,
                )
            )
        now = datetime.now(UTC).replace(microsecond=0)
        # Both arms zeroed at their end stops before the cameras were calibrated.
        self.joint_zero_path = work / "joint_zero.json"
        zero_ids = write_joint_zero(
            self.joint_zero_path, options.robot_serial, now - timedelta(hours=1)
        )
        calibrated_at = utc_text(now)
        self.calibration_doc = document(
            robot_serial=options.robot_serial,
            calibration_id=f"{options.robot_serial}-{now:%Y%m%d}-01",
            calibrated_at=calibrated_at,
            status="pass",
            transforms=entries,
            joint_zero=zero_ids,
            synthetic=True,
        )
        truth = parse(
            json.dumps(self.calibration_doc).encode(), work / "camera_calibration.json"
        )
        frames = resolve(
            self.scene, truth, imu_poses={imu_frame_id("overhead"): IMU_POSE}
        )
        self.truth = UrdfTree(self.scene.robot_description(frames.edges, frames.added))
        self.lift = self.scene.lift_value(options.raised)
        self.lens = {
            src: LensSpec.from_intrinsics(c.intrinsics)
            for src, c in self.scene.cameras.items()
            if c.intrinsics is not None
        }

    def _meta(self, camera: str, eye: str | None, serial: int, stereo: bool) -> dict:
        o = self.options
        out: dict[str, Any] = {
            "transport": f"gstshm-{o.codec}",
            "codec": o.codec,
            "socket_path": "/unused",
            "width": o.width,
            "height": o.height,
            "fps": o.fps,
            "capture_fps": o.fps,
            "pts_perf_offset_s": 0.0,
            "camera": camera,
            "eye": eye,
            "serial": serial,
            "resolution": o.resolution,
            "stereo": stereo,
        }
        if camera == "overhead":
            out["imu_socket"] = self.imu_socket
            out["imu_requested"] = True
        return out


# ---------------------------------------------------------------------------
# The script: what happens, on a fine time grid


def _props(objects: Mapping[str, np.ndarray]) -> list[Prop]:
    floor = FLOOR_Z
    top = TABLE_TOP
    props = [
        Prop(
            "table_top",
            "box",
            (0.30, 0.55, 0.02),
            (0.58, 0.43, 0.3, 1.0),
            pose((0.72, 0.0, top - 0.02)),
        )
    ]
    leg = (top - 0.04 - floor) / 2
    for i, (x, y) in enumerate(((0.47, 0.5), (0.47, -0.5), (0.97, 0.5), (0.97, -0.5))):
        props.append(
            Prop(
                f"table_leg{i}",
                "box",
                (0.02, 0.02, leg),
                (0.3, 0.3, 0.32, 1.0),
                pose((x, y, floor + leg)),
            )
        )
    bin_xy = (0.59, 0.0)
    props.append(
        Prop(
            "bin_floor",
            "box",
            (0.10, 0.10, 0.003),
            (0.15, 0.45, 0.25, 1.0),
            pose((*bin_xy, top + 0.003)),
        )
    )
    for i, (dx, dy, sx, sy) in enumerate(
        (
            (0.097, 0, 0.003, 0.1),
            (-0.097, 0, 0.003, 0.1),
            (0, 0.097, 0.1, 0.003),
            (0, -0.097, 0.1, 0.003),
        )
    ):
        props.append(
            Prop(
                f"bin_wall{i}",
                "box",
                (sx, sy, 0.025),
                (0.15, 0.45, 0.25, 1.0),
                pose((bin_xy[0] + dx, bin_xy[1] + dy, top + 0.025)),
            )
        )
    props.append(
        Prop(
            "block",
            "box",
            (0.03, 0.06, 0.03),
            (0.25, 0.6, 0.3, 1.0),
            pose((0.84, -0.12, top + 0.03)),
        )
    )
    props.append(
        Prop("cube", "box", (CUBE, CUBE, CUBE), (0.85, 0.12, 0.1, 1.0), objects["cube"])
    )
    props.append(Prop("cup", "cylinder", CUP, (0.15, 0.35, 0.85, 1.0), objects["cup"]))
    return props


@dataclass
class Timeline:
    """Everything an episode records, on a fine grid of ``t`` (s)."""

    t: np.ndarray
    cmd_t: np.ndarray  # control ticks
    cmd_q: dict[str, np.ndarray]  # side -> (ticks, 7)
    cmd_grip: dict[str, np.ndarray]  # side -> (ticks,)
    q: dict[str, np.ndarray]  # side -> (fine, 7), measured
    grip: dict[str, np.ndarray]  # side -> (fine,), measured opening
    base: np.ndarray  # (fine, 3): x, y, yaw
    base_v: np.ndarray  # (fine, 3): vx, vy, wz in the world
    objects: dict[str, np.ndarray]  # name -> (fine, 4, 4)
    tcp_target: dict[str, np.ndarray]  # side -> (ticks, 4, 4), root frame
    drive_until: float
    # Headset input (teleop); a calibration recording has none.
    headset: bool = True
    # The scene's props; empty for the task scene (:func:`_props`).
    props: list[Prop] = field(default_factory=list)


def _object_rest(name: str) -> np.ndarray:
    if name == "cube":
        return pose((0.61, -0.20, TABLE_TOP + CUBE))
    return pose((0.65, 0.22, TABLE_TOP + CUP[1]))


def _script(
    rig: Rig, spec: EpisodeSpec, duration: float
) -> tuple[dict[str, TcpPath], tuple[float, float], dict[str, Any]]:
    """TCP paths (root frame), the base's drive, and the grasp."""
    kin = rig.kin
    ready = {
        side: kin.tcp(side, dict(zip(kin.joints[side], READY[side], strict=True)))
        for side in SIDES
    }
    root_z = -2.0 * rig.lift  # the torso's height in the world (lift and lift_2)

    def at(side: str, world_xyz: Sequence[float], base: Sequence[float]) -> np.ndarray:
        out = ready[side].copy()
        out[:3, 3] = (
            world_xyz[0] - base[0],
            world_xyz[1] - base[1],
            world_xyz[2] - root_z,
        )
        return out

    def hold(side: str, t0: float, t1: float) -> list[Waypoint]:
        return [Waypoint(t0, ready[side], 0.3), Waypoint(t1, ready[side], 0.3)]

    if spec.script == "cube_to_bin":
        base = (0.15, 0.0)
        cube = _object_rest("cube")[:3, 3]
        over = (cube[0], cube[1], cube[2] + 0.12)
        grasp = (cube[0], cube[1], cube[2] + 0.004)
        bin_over = (0.59, 0.0, TABLE_TOP + 0.17)
        bin_low = (0.59, 0.0, TABLE_TOP + 0.09)
        side = "right"
        other = "left"
        path = [
            Waypoint(0.0, ready[side], 0.3),
            Waypoint(2.0, ready[side], 0.3),
            Waypoint(4.2, at(side, over, base), 1.0),
            Waypoint(5.4, at(side, grasp, base), 1.0),
            Waypoint(6.0, at(side, grasp, base), 0.0),
            Waypoint(7.2, at(side, over, base), 0.0),
            Waypoint(9.4, at(side, bin_over, base), 0.0),
            Waypoint(10.2, at(side, bin_low, base), 0.0),
            Waypoint(10.8, at(side, bin_low, base), 1.0),
            Waypoint(12.0, at(side, bin_over, base), 1.0),
            Waypoint(14.0, ready[side], 0.3),
            Waypoint(duration, ready[side], 0.3),
        ]
        drift = ready[other].copy()
        drift[:3, 3] += (-0.03, 0.03, 0.04)
        other_path = [
            *hold(other, 0.0, 3.0),
            Waypoint(6.0, drift, 0.6),
            Waypoint(9.0, drift, 0.6),
            Waypoint(12.5, ready[other], 0.3),
            Waypoint(duration, ready[other], 0.3),
        ]
        grasp_info = {
            "object": "cube",
            "side": side,
            "attach": 5.75,
            "release": 10.45,
            "hold": (2 * CUBE - 0.0224) / 2 / 0.0588,
            "rest": pose((0.59, 0.0, TABLE_TOP + 0.006 + CUBE)),
        }
    else:
        base = (0.15, 0.03)
        cup = _object_rest("cup")[:3, 3]
        over = (cup[0], cup[1], cup[2] + 0.14)
        grasp = (cup[0], cup[1], cup[2] + 0.02)
        place = (0.70, 0.15, TABLE_TOP + CUP[1] + 0.02)
        place_over = (place[0], place[1], place[2] + 0.12)
        side = "left"
        other = "right"
        path = [
            Waypoint(0.0, ready[side], 0.3),
            Waypoint(2.2, ready[side], 0.3),
            Waypoint(4.4, at(side, over, base), 1.0),
            Waypoint(5.6, at(side, grasp, base), 1.0),
            Waypoint(6.2, at(side, grasp, base), 0.0),
            Waypoint(7.4, at(side, over, base), 0.0),
            Waypoint(9.6, at(side, place_over, base), 0.0),
            Waypoint(10.6, at(side, place, base), 0.0),
            Waypoint(11.2, at(side, place, base), 1.0),
            Waypoint(12.2, at(side, place_over, base), 1.0),
            Waypoint(14.0, ready[side], 0.3),
            Waypoint(duration, ready[side], 0.3),
        ]
        drift = ready[other].copy()
        drift[:3, 3] += (-0.02, -0.04, 0.03)
        other_path = [
            *hold(other, 0.0, 4.0),
            Waypoint(7.0, drift, 0.5),
            Waypoint(10.0, drift, 0.5),
            Waypoint(13.0, ready[other], 0.3),
            Waypoint(duration, ready[other], 0.3),
        ]
        grasp_info = {
            "object": "cup",
            "side": side,
            "attach": 5.95,
            "release": 10.85,
            "hold": (2 * CUP[0] - 0.0224) / 2 / 0.0588,
            "rest": pose((0.70, 0.15, TABLE_TOP + CUP[1])),
        }
    paths = {side: TcpPath(path), other: TcpPath(other_path)}
    return paths, base, grasp_info


def build_timeline(rig: Rig, spec: EpisodeSpec, duration: float) -> Timeline:
    rng = np.random.default_rng(spec.seed)
    paths, base_goal, grasp = _script(rig, spec, duration)
    kin = rig.kin
    t = np.arange(0.0, duration + 0.5, _DT)
    cmd_t = np.arange(0.0, duration + 0.25, 1.0 / CONTROL_HZ)
    cmd_t = cmd_t + rng.normal(0.0, 0.00015, len(cmd_t)).clip(-0.0004, 0.0004)
    cmd_t[0] = 0.0
    cmd_q: dict[str, np.ndarray] = {}
    cmd_grip: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    for side in SIDES:
        names = kin.joints[side]
        seed = dict(zip(names, READY[side], strict=True))
        rows, grips, poses = [], [], []
        for tk in cmd_t:
            target, grip = paths[side].at(float(tk))
            solution, err = kin.solve(side, target, seed, iterations=30)
            if err > 0.002:
                _logger.warning(
                    "%s arm: %.1f mm from its target at %.2f s", side, err * 1000, tk
                )
            seed = solution
            rows.append([solution[n] for n in names])
            grips.append(grip)
            poses.append(target)
        cmd_q[side] = np.array(rows)
        cmd_grip[side] = np.clip(np.array(grips), 0.0, 1.0)
        targets[side] = np.array(poses)

    q, grip = follow(t, cmd_t, cmd_q, cmd_grip, rng)
    side = grasp["side"]
    holding = (t >= grasp["attach"]) & (t < grasp["release"])
    grip[side] = np.where(holding, np.maximum(grip[side], grasp["hold"]), grip[side])

    # Jelly drives to its spot over the first two seconds, then parks.
    drive_t = 2.0
    s = np.clip(t / drive_t, 0.0, 1.0)
    shape = np.array([min_jerk(v) for v in s])
    base = np.zeros((len(t), 3))
    base[:, 0] = base_goal[0] * shape
    base[:, 1] = base_goal[1] * shape
    base_v = np.gradient(base, _DT, axis=0)

    # The grasped object rides with the TCP, then drops to its rest.
    objects = {
        name: np.repeat(_object_rest(name)[None], len(t), axis=0)
        for name in ("cube", "cup")
    }
    name = grasp["object"]
    attach_i = int(np.searchsorted(t, grasp["attach"]))
    release_i = int(np.searchsorted(t, grasp["release"]))
    if attach_i < len(t):
        tcp_world = _tcp_world(rig, side, q[side], base, t)
        offset = np.linalg.inv(tcp_world[attach_i]) @ objects[name][attach_i]
        for i in range(attach_i, min(release_i, len(t))):
            objects[name][i] = tcp_world[i] @ offset
    if attach_i < release_i < len(t):
        start = objects[name][release_i - 1].copy()
        rest = grasp["rest"].copy()
        yaw = math.atan2(start[1, 0], start[0, 0])
        rest[:3, :3] = _yaw(yaw)
        fall = max(0.0, start[2, 3] - rest[2, 3])
        fall_t = math.sqrt(2 * fall / 9.81) if fall > 0 else 0.0
        for i in range(release_i, len(t)):
            dt = t[i] - t[release_i]
            if dt < fall_t:
                frac = 0.5 * 9.81 * dt * dt / max(fall, 1e-9)
                out = start.copy()
                out[:3, 3] = start[:3, 3] + frac * (rest[:3, 3] - start[:3, 3])
                objects[name][i] = out
            else:
                objects[name][i] = rest
    return Timeline(
        t=t,
        cmd_t=cmd_t,
        cmd_q=cmd_q,
        cmd_grip=cmd_grip,
        q=q,
        grip=grip,
        base=base,
        base_v=base_v,
        objects=objects,
        tcp_target=targets,
        drive_until=drive_t,
    )


def follow(
    t: np.ndarray,
    cmd_t: np.ndarray,
    cmd_q: Mapping[str, np.ndarray],
    cmd_grip: Mapping[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Measured joints on the fine grid ``t``: each arm follows its latest
    target (held between ticks) 8 ms late, through a first-order lag, with
    encoder noise; each gripper moves at most a full stroke in 0.5 s."""
    q: dict[str, np.ndarray] = {}
    grip: dict[str, np.ndarray] = {}
    held_index = np.clip(np.searchsorted(cmd_t, t - 0.008, side="right") - 1, 0, None)
    alpha = _DT / 0.018
    for side in SIDES:
        target = cmd_q[side][held_index]
        out = np.empty_like(target)
        out[0] = target[0]
        for i in range(1, len(t)):
            out[i] = out[i - 1] + alpha * (target[i] - out[i - 1])
        q[side] = out + rng.normal(0.0, 2e-5, out.shape)
        g_target = cmd_grip[side][held_index]
        g = np.empty(len(t))
        g[0] = g_target[0]
        for i in range(1, len(t)):
            step = np.clip(g_target[i] - g[i - 1], -2.0 * _DT, 2.0 * _DT)
            g[i] = g[i - 1] + step
        grip[side] = g
    return q, grip


def _yaw(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _joint_values(rig: Rig, timeline: Timeline, i: int | float) -> dict[str, float]:
    """URDF joint values at a fine-grid index (fractional: interpolated)."""
    lo = int(math.floor(i))
    hi = min(lo + 1, len(timeline.t) - 1)
    w = float(i) - lo
    values: dict[str, float] = {}
    joints16 = np.zeros(2 * len(JOINT_NAMES))
    n = len(JOINT_NAMES)
    for s_i, side in enumerate(SIDES):
        joints16[s_i * n : s_i * n + 7] = (1 - w) * timeline.q[side][
            lo
        ] + w * timeline.q[side][hi]
        joints16[s_i * n + 7] = (1 - w) * timeline.grip[side][lo] + w * timeline.grip[
            side
        ][hi]
    values.update(rig.scene.urdf_values(joints16))
    base = (1 - w) * timeline.base[lo] + w * timeline.base[hi]
    values.update({"base_x": base[0], "base_y": base[1], "base_yaw": base[2]})
    values["lift"] = rig.lift
    return values


def _tcp_world(
    rig: Rig, side: str, q: np.ndarray, base: np.ndarray, t: np.ndarray
) -> np.ndarray:
    names = rig.kin.joints[side]
    out = np.empty((len(t), 4, 4))
    tool = rig.kin.tool
    for i in range(len(t)):
        values = dict(zip(names, q[i], strict=True))
        values.update(
            {"base_x": base[i, 0], "base_y": base[i, 1], "base_yaw": base[i, 2]}
        )
        values["lift"] = rig.lift
        out[i] = rig.truth.fk("world", f"{side}_gripper", values) @ tool
    return out


# ---------------------------------------------------------------------------
# Rendering and encoding, in worker processes

_worker: dict[str, Any] = {}


def _all_link_poses(
    tree: UrdfTree, values: Mapping[str, float], links: Sequence[str]
) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {tree.root: np.eye(4)}

    def world(link: str) -> np.ndarray:
        if link in cache:
            return cache[link]
        joint = tree.parent_joint[link]
        out = world(joint.parent) @ joint.transform(tree.value(joint, values))
        cache[link] = out
        return out

    return {link: world(link) for link in links if tree.has_link(link)}


def _encoder(codec: str, width: int, height: int, fps: int, quality: int) -> Any:
    import av

    if codec == "h265":
        ctx = av.CodecContext.create("libx265", "w")
        params = (
            "keyint=1:min-keyint=1:scenecut=0:bframes=0:repeat-headers=1:"
            "info=0:log-level=error"
        )
        ctx.options = {"preset": "faster", "crf": str(quality), "x265-params": params}
    else:
        ctx = av.CodecContext.create("libx264", "w")
        ctx.options = {
            "preset": "faster",
            "crf": str(quality),
            "x264-params": "keyint=1:min-keyint=1:scenecut=0:bframes=0:repeat-headers=1",
        }
    ctx.width = width
    ctx.height = height
    ctx.pix_fmt = "yuv420p"
    ctx.time_base = Fraction(1, fps)
    ctx.framerate = Fraction(fps, 1)
    return ctx


def _worker_init(state: dict[str, Any]) -> None:
    os.environ.setdefault("LP_NUM_THREADS", "2")
    from .render import offscreen_backend

    offscreen_backend()
    rig: Rig = state["rig"]
    timeline: Timeline = state["timeline"]
    objects = {name: poses[0] for name, poses in timeline.objects.items()}
    props = timeline.props or _props(objects)
    scene = SceneRenderer(rig.urdf, props, max_size=(2400, 2400), floor_z=FLOOR_Z)
    supersample = getattr(state["options"], "supersample", 1.25)
    imagers = {
        src: LensImager(lens, supersample=supersample) for src, lens in rig.lens.items()
    }
    _worker.update(state)
    _worker["scene"] = scene
    _worker["imagers"] = imagers


def _render_chunk(
    job: tuple[str, list[tuple[int, float]]],
) -> tuple[str, list[tuple[int, bytes]]]:
    source, frames = job
    rig: Rig = _worker["rig"]
    timeline: Timeline = _worker["timeline"]
    options: SyntheticOptions = _worker["options"]
    scene: SceneRenderer = _worker["scene"]
    imager: LensImager = _worker["imagers"][source]
    import av

    frame_id = rig.scene.cameras[source].frame_id
    ctx = _encoder(
        options.codec, options.width, options.height, options.fps, options.quality
    )
    out: list[tuple[int, bytes]] = []
    pending: deque[int] = deque()
    for index, when in frames:
        i = when / _DT
        values = _joint_values(rig, timeline, i)
        scene.set_robot(_all_link_poses(rig.truth, values, scene.links))
        k = min(int(round(i)), len(timeline.t) - 1)
        for name, poses in timeline.objects.items():
            scene.set_pose(name, poses[k])
        camera = rig.truth.fk("world", frame_id, values)
        housing = rig.truth.parent_joint[frame_id].parent
        image = imager.render(scene, camera, hide=(housing,))
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
        frame = frame.reformat(format="yuv420p")
        frame.pts = index
        pending.append(index)
        for packet in ctx.encode(frame):
            out.append((pending.popleft(), bytes(packet)))
    for packet in ctx.encode(None):
        out.append((pending.popleft(), bytes(packet)))
    return source, out


def exposure_times(
    source: str, fps: int, duration: float, rng: np.random.Generator
) -> np.ndarray:
    phase = {
        "overhead_left": 0.0021,
        "overhead_right": 0.0021,
        "left_arm": 0.0063,
        "right_arm": 0.0111,
    }[source]
    n = int(math.floor((duration - phase) * fps))
    times = phase + np.arange(n) / fps
    return times + rng.normal(0.0, 2e-5, n).clip(-6e-5, 6e-5)


def camera_exposures(
    sources: Sequence[str], fps: int, duration: float, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Exposure times per camera; a stereo camera's eyes share theirs."""
    out: dict[str, np.ndarray] = {}
    for source in sources:
        if source == "overhead_right" and "overhead_left" in out:
            out[source] = out["overhead_left"]
        else:
            out[source] = exposure_times(source, fps, duration, rng)
    if "overhead_left" in sources and "overhead_right" in sources:
        out["overhead_right"] = out["overhead_left"]
    return out


def render_video(
    rig: Rig,
    timeline: Timeline,
    options: SyntheticOptions,
    exposures: Mapping[str, np.ndarray],
    progress: Callable[[str], None] | None = None,
    sources: Sequence[str] | None = None,
) -> dict[str, list[bytes]]:
    """Encoded access units per camera (``sources``: all), one per exposure."""
    jobs: list[tuple[str, list[tuple[int, float]]]] = []
    chunk = 30
    sources = list(sources) if sources is not None else list(rig.lens)
    for source in sources:
        times = exposures[source]
        for start in range(0, len(times), chunk):
            jobs.append(
                (
                    source,
                    [
                        (i, float(times[i]))
                        for i in range(start, min(start + chunk, len(times)))
                    ],
                )
            )
    state = {"rig": rig, "timeline": timeline, "options": options}
    results: dict[str, dict[int, bytes]] = {source: {} for source in sources}
    # Workers inherit the environment: MuJoCo must pick its offscreen
    # backend before anything in them imports it.
    from .render import offscreen_backend

    offscreen_backend()
    ctx = multiprocessing.get_context("spawn")
    done = 0
    started = time.monotonic()
    workers = max(1, min(options.workers, len(jobs)))
    with ctx.Pool(workers, initializer=_worker_init, initargs=(state,)) as pool:
        for source, aus in pool.imap_unordered(_render_chunk, jobs):
            for index, au in aus:
                results[source][index] = au
            done += 1
            if progress is not None and (done % 8 == 0 or done == len(jobs)):
                elapsed = time.monotonic() - started
                progress(f"rendered {done}/{len(jobs)} chunks in {elapsed:.0f} s")
    return {
        source: [frames[i] for i in range(len(exposures[source]))]
        for source, frames in results.items()
    }


# ---------------------------------------------------------------------------
# Playing an episode into the recorder


class _PlaybackReader:
    """Stands in for the relay's encoded-frame reader: each access unit
    arrives a little after its exposure."""

    def __init__(self) -> None:
        self.queue: deque[tuple[float, bytes, float]] = deque()
        self.lost_exposures = 0

    def begin_flush(self) -> None:
        pass

    def finish_flush(self, timeout_s: float = 2.0) -> None:
        pass

    def reset_lost_exposures(self) -> None:
        self.lost_exposures = 0

    def drain_aus(self) -> list[tuple[bytes, float, float]]:
        now = time.perf_counter()
        out = []
        while self.queue and self.queue[0][0] <= now:
            arrive, au, exposure = self.queue.popleft()
            out.append((au, exposure, arrive))
        return out

    def disconnect(self) -> None:
        pass


def _vr_pose(target: np.ndarray) -> SimpleNamespace:
    """A controller pose in the headset's space (WebXR: x right, y up, z back)
    for a TCP target in the torso frame."""
    a = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
    p = a @ target[:3, 3] + np.array([0.0, 0.35, -0.25])
    q = quaternion_xyzw(a @ target[:3, :3] @ a.T)
    return SimpleNamespace(
        position=SimpleNamespace(x=p[0], y=p[1], z=p[2]),
        quaternion=SimpleNamespace(x=q[0], y=q[1], z=q[2], w=q[3]),
    )


def play_episode(
    session: Any,
    rig: Rig,
    timeline: Timeline,
    aus: Mapping[str, list[bytes]],
    exposures: Mapping[str, np.ndarray],
    spec: EpisodeSpec,
    duration: float,
    rings: tuple[Any, Any],
    phases: Sequence[tuple[str, float, float, int, str]] = (),
) -> None:
    """Feed one episode to ``session`` in real time and save it.

    ``phases`` are ``(kind, start s, end s, placement, arm)`` of a
    calibration recording, saved in its episode metadata.
    """
    tick_writer, joint_writer = rings
    rng = np.random.default_rng(spec.seed + 1)
    readers: dict[str, _PlaybackReader] = session.readers
    sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    wall_minus_perf = time.time() - time.perf_counter()

    def send_pose() -> None:
        q = quaternion_xyzw(IMU_POSE[:3, :3])
        packet = POSE_PACKET.pack(POSE_MAGIC, 1, *IMU_POSE[:3, 3], *q)
        try:
            sender.sendto(packet, rig.imu_socket)
        except OSError:
            pass

    send_pose()
    time.sleep(0.05)
    session.start(spec.task)
    t0 = time.perf_counter() + 0.1

    events: list[tuple[float, int, Callable[[float], None]]] = []
    n = len(JOINT_NAMES)

    # 240 Hz joint feedback, with the realtime core's applied targets.
    applied_t = (
        timeline.cmd_t + 0.0005 + 0.0021 + rng.uniform(0, 0.0008, len(timeline.cmd_t))
    )
    joint_t = np.arange(0.0, duration, 1.0 / JOINT_HZ)
    torque_bias = np.array([0.0, 6.0, 0.4, 3.2, 0.2, 0.6, 0.1])

    def joint_row(t: float) -> None:
        i = min(int(round(t / _DT)), len(timeline.t) - 2)
        pos = np.zeros(2 * n)
        vel = np.zeros(2 * n)
        trq = np.zeros(2 * n)
        k = int(np.clip(np.searchsorted(timeline.cmd_t, t, side="right") - 1, 0, None))
        for s_i, side in enumerate(SIDES):
            sl = slice(s_i * n, s_i * n + 7)
            pos[sl] = timeline.q[side][i]
            vel[sl] = (
                timeline.q[side][i + 1] - timeline.q[side][i - 1 if i else i]
            ) / (2 * _DT if i else _DT)
            sign = 1.0 if side == "left" else -1.0
            trq[sl] = (
                40.0 * (timeline.cmd_q[side][k] - timeline.q[side][i])
                + sign * torque_bias
            )
            trq[sl] += rng.normal(0.0, 0.05, 7)
            pos[s_i * n + 7] = timeline.grip[side][i]
            vel[s_i * n + 7] = (
                timeline.grip[side][i + 1] - timeline.grip[side][i]
            ) / _DT
            trq[s_i * n + 7] = 0.8 if timeline.grip[side][i] < 0.6 else 0.1
        done = int(np.searchsorted(applied_t, t, side="right")) - 1
        applied = (
            [np.nan] * 4
            if done < 0
            else [
                2 * done + 1,
                t0 + applied_t[done],
                2 * done + 2,
                t0 + applied_t[done],
            ]
        )
        joint_writer.write_values(t0 + t, np.concatenate([pos, vel, trq, applied]))

    for t in joint_t:
        events.append((float(t), 0, joint_row))

    # 120 Hz control ticks: targets, headset input and Jelly.
    headset_t = np.arange(0.0, duration, 1.0 / HEADSET_HZ) + 0.004
    wheel_t = np.arange(0.0, duration, 1.0 / WHEEL_HZ) + 0.002
    signs = np.array([WHEEL_SIGNS[i] for i in (1, 2, 3, 4)])

    def wheel_speed(t: float) -> np.ndarray:
        i = min(int(round(t / _DT)), len(timeline.t) - 1)
        vx, vy, _ = timeline.base_v[i]
        return np.array(wheel_rims(vx, vy, 0.0)) / WHEEL_RADIUS_M * signs

    fine_speeds = np.array([wheel_speed(float(t)) for t in timeline.t])
    wheel_angle = 3.0 + np.cumsum(fine_speeds, axis=0) * _DT

    def tick_row(t: float, k: int) -> None:
        command = {}
        for side in SIDES:
            for j, joint in enumerate(JOINT_NAMES[:7]):
                command[f"{side}_{joint}.pos"] = float(timeline.cmd_q[side][k][j])
            command[f"{side}_{JOINT_NAMES[7]}.pos"] = float(timeline.cmd_grip[side][k])
        h = int(np.searchsorted(headset_t, t, side="right")) - 1
        frame = None
        if h >= 0 and timeline.headset:
            kh = int(
                np.clip(
                    np.searchsorted(timeline.cmd_t, headset_t[h]),
                    0,
                    len(timeline.cmd_t) - 1,
                )
            )
            frame = SimpleNamespace(
                seq=h,
                t_host=t0 + headset_t[h],
                l_ee=_vr_pose(timeline.tcp_target["left"][kh]),
                r_ee=_vr_pose(timeline.tcp_target["right"][kh]),
                l_grip=float(1.0 - timeline.cmd_grip["left"][kh]),
                r_grip=float(1.0 - timeline.cmd_grip["right"][kh]),
                l_stick_x=0.0,
                l_stick_y=float(0.6 if headset_t[h] < timeline.drive_until else 0.0),
                r_stick_x=0.0,
                r_stick_y=0.0,
                l_lock=True,
                r_lock=True,
                l_stick_click=False,
                r_stick_click=False,
                reset=False,
                stow=False,
            )
        w = int(np.searchsorted(wheel_t, t, side="right")) - 1
        jelly = None
        if w >= 0:
            wi = min(int(round(wheel_t[w] / _DT)), len(timeline.t) - 1)
            driving = t < timeline.drive_until + 0.1
            cmd_k = int(t * 60)
            ti = min(int(round(t / _DT)), len(timeline.t) - 1)
            vx, vy, _ = timeline.base_v[ti]
            jelly = SimpleNamespace(
                wheel_sample=SimpleNamespace(
                    positions=tuple(wheel_angle[wi]),
                    velocities=tuple(fine_speeds[wi]),
                    at=t0 + wheel_t[w],
                ),
                wheel_speeds=list(fine_speeds[ti]),
                has_wheels=True,
                lift_raised=rig.options.raised,
                parked=not driving,
                latest_command=(
                    100 + min(cmd_k, int((timeline.drive_until + 0.1) * 60)),
                    t0 + min(cmd_k, int((timeline.drive_until + 0.1) * 60)) / 60.0,
                    float(vx) if driving else 0.0,
                    float(vy) if driving else 0.0,
                    0.0,
                    0,
                ),
            )
        tick_writer.write(
            pack_tick(
                command,
                frame,
                jelly,
                command_id=1 + k,
                issue_t=t0 + t + 0.0005,
                arm_mode="impedance",
                target_seqs=(2 * k + 1, 2 * k + 2),
            ),
            {},
            t0 + t,
        )

    for k, t in enumerate(timeline.cmd_t):
        if t < duration:
            events.append((float(t), 1, lambda t, k=k: tick_row(t, k)))

    # The head camera's IMU: gravity, the base's acceleration, noise.
    imu_frame = imu_frame_id("overhead")
    imu_t = np.arange(0.0, duration, 1.0 / IMU_HZ) + 0.0013
    accel = np.gradient(timeline.base_v, _DT, axis=0)
    gyro_bias = rng.normal(0.0, 0.05, 3)

    def imu_sample(t: float) -> None:
        i = min(int(round(t / _DT)), len(timeline.t) - 1)
        values = _joint_values(rig, timeline, i)
        r = rig.truth.fk("world", imu_frame, values)[:3, :3]
        f = r.T @ (accel[i] + np.array([0.0, 0.0, 9.81])) + rng.normal(0.0, 0.015, 3)
        w = gyro_bias + rng.normal(0.0, 0.08, 3)  # deg/s; the base doesn't turn
        stamp = int((t0 + t + wall_minus_perf) * 1e9)
        packet = PACKET.pack(MAGIC, 1, stamp, *f, *w, 38.5)
        try:
            sender.sendto(packet, rig.imu_socket)
        except OSError:
            pass

    for t in imu_t:
        events.append((float(t), 2, imu_sample))
    for second in range(int(duration)):
        events.append((second + 0.5, 3, lambda _t: send_pose()))

    # Video: each frame arrives 25 ms after its exposure.
    for source, times in exposures.items():
        reader = readers[source]
        for i, te in enumerate(times):
            reader.queue.append((t0 + te + 0.025, aus[source][i], t0 + te))

    events.sort(key=lambda e: (e[0], e[1]))
    for when, _kind, action in events:
        delay = t0 + when - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        action(when)
    while time.perf_counter() < t0 + duration + 0.15:
        time.sleep(0.01)
    sender.close()
    session.stop_capture()
    info: dict[str, str] = {}
    if phases:
        from ..camera_calibration.recording import Phase, phases_json

        info["calibration_phases"] = phases_json(
            Phase(
                kind,
                int(session.episode.stamp(t0 + start)),
                int(session.episode.stamp(t0 + end)),
                placement,
                arm,
            )
            for kind, start, end, placement, arm in phases
        )
    session.save("success", info)


def make_episodes(
    options: SyntheticOptions,
    specs: Sequence[EpisodeSpec],
    progress: Callable[[str], None] = print,
) -> list[Path]:
    """Write ``specs`` as episodes in ``options.output``; returns their paths."""
    from ..recording.mcap.recorder import McapSession
    from ..video.shm_frames import SnapshotReader, SnapshotWriter

    output = Path(options.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    if not options.software_version:
        try:
            from importlib.metadata import version

            options.software_version = version("almond-axol")
        except Exception:  # noqa: BLE001 - informational
            options.software_version = "unknown"
    work = Path(tempfile.mkdtemp(prefix="axol-synthetic-"))
    calibration_path = work / "camera_calibration.json"
    rig = Rig(work, options)
    duration = options.duration_s
    tick_writer = SnapshotWriter(tick_ring_keys(), [], capacity=TICK_RING_CAPACITY)
    joint_writer = SnapshotWriter(joint_ring_keys(), [], capacity=JOINT_RING_CAPACITY)
    readers = {src: _PlaybackReader() for src in rig.raw_meta}
    session = McapSession(
        {
            "dataset_root": str(output),
            "operator": options.operator,
            "robot_serial": options.robot_serial,
            "robot_model": "mobile",
            "gripper": "parallel",
            "software_version": options.software_version,
            "hostname": "synthetic",
            "wheel_signs": dict(WHEEL_SIGNS),
            "topic_names": dict(options.topic_names),
            "frame_names": dict(options.frame_names),
            "camera_calibration_path": str(calibration_path),
            "joint_zero_path": str(rig.joint_zero_path),
            "teleop_hz": CONTROL_HZ,
        },
        cameras=readers,
        scene=rig.scene,
        tick_reader=SnapshotReader(tick_writer.name, tick_ring_keys(), []),
        joint_reader=SnapshotReader(joint_writer.name, joint_ring_keys(), []),
    )
    session.compute_serial = "SYNTHETIC"
    session.imus = {
        imu_topic("overhead"): (ImuReceiver(rig.imu_socket), "head_camera_left_optical")
    }
    paths: list[Path] = []
    try:
        for spec in specs:
            progress(f"{spec.task}: planning the motion")
            timeline = build_timeline(rig, spec, duration)
            rng = np.random.default_rng(spec.seed + 7)
            exposures = camera_exposures(list(rig.raw_meta), options.fps, duration, rng)
            progress(
                f"{spec.task}: rendering {sum(len(v) for v in exposures.values())} frames"
            )
            aus = render_video(rig, timeline, options, exposures, progress)
            if spec.calibrated:
                from ..camera_calibration.file import write

                write(rig.calibration_doc, calibration_path)
            elif calibration_path.exists():
                calibration_path.unlink()
            session.config["episode_metadata"] = {
                **options.episode_metadata,
                **spec.metadata,
            }
            progress(f"{spec.task}: recording {duration:.0f} s")
            before = set(output.glob("episode_*.mcap"))
            play_episode(
                session,
                rig,
                timeline,
                aus,
                exposures,
                spec,
                duration,
                (tick_writer, joint_writer),
            )
            new = sorted(set(output.glob("episode_*.mcap")) - before)
            paths += new
            for path in new:
                progress(f"saved {path} ({path.stat().st_size / 1e6:.0f} MB)")
    finally:
        session.close()
        tick_writer.close()
        joint_writer.close()
        shutil.rmtree(work, ignore_errors=True)
    if any(spec.calibrated for spec in specs):
        # The calibration the calibrated episodes used, for reference.
        from ..camera_calibration.file import write

        write(rig.calibration_doc, output / "camera_calibration.json")
    return paths
