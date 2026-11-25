"""One episode's MCAP file, written as the samples arrive.

:class:`EpisodeRecording` turns the recorder's three inputs into messages:

- encoded video access units (one per camera exposure) -> ``<camera>/video``
  and ``<camera>/camera_info``, both stamped with the exposure time;
- 240 Hz joint ring rows -> ``/joint_states``, the arm joints on ``/tf``,
  and the measured end-effector poses;
- 120 Hz tick ring rows -> ``/joint_command``, the commanded end-effector
  poses, the headset input, and Jelly's odometry, wheels, lift and drive
  command.

Every sample is written once, at its own timestamp. Nothing is resampled onto
a grid and no frame is repeated to fill a gap; gaps are counted instead.
"""

from __future__ import annotations

import math
from collections import OrderedDict, deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import messages
from .odometry import XDriveOdometry
from .rings import (
    JOINT_NAMES,
    POSE_AXES,
    SIDES,
    VR_AXES,
    VR_BUTTONS,
    WHEEL_IDS,
    column_index,
    joint_ring_keys,
    mode_name,
    tick_ring_keys,
)
from .mounts import EpisodeFrames, resolve
from .scene import BASE_JOINTS, LIFT_JOINT, VR_FRAME, CameraSource, Scene
from .urdf_tree import pose7, quaternion_xyzw
from .writer import EpisodeFile

FORMAT_VERSION = "1"
# The camera calibration file used, attached to the episode under this name.
CALIBRATION_ATTACHMENT = "camera_calibration.json"
# A command whose application the realtime core hasn't reported within this
# long (s, on the joint feed's clock) is written with its application unknown:
# it was superseded before the core used it, or rejected.
_APPLY_WAIT_S = 0.5
# Application times remembered per arm (by target seq).
_APPLIED_MEMORY = 4096
# Re-publish an unchanged lift height this often (s), so a reader starting
# anywhere in the file finds a recent value.
_LIFT_REFRESH_S = 1.0
_N = len(JOINT_NAMES)


def iso_utc(unix_ns: int | None) -> str:
    if unix_ns is None:
        return ""
    return datetime.fromtimestamp(unix_ns / 1e9, tz=UTC).isoformat(
        timespec="milliseconds"
    )


class _CameraStats:
    __slots__ = ("frames", "first", "last", "gaps", "missing")

    def __init__(self) -> None:
        self.frames = 0
        self.first: float | None = None
        self.last: float | None = None
        self.gaps = 0
        self.missing = 0


class EpisodeRecording:
    """Writes one episode; see the module docstring for what goes where."""

    def __init__(
        self,
        path: Path,
        *,
        scene: Scene,
        start_perf: float,
        clock_offset_ns: int,
        wheel_signs: dict[int, float] | None = None,
        topic_names: dict[str, str] | None = None,
        frames: EpisodeFrames | None = None,
        frame_names: dict[str, str] | None = None,
    ) -> None:
        self.scene = scene
        # Renamed frames (default name -> customer's name), see .topics.
        self.frame_names: dict[str, str] = dict(frame_names or {})
        # The static frames: camera mounts from the calibration file when
        # there is one (see .mounts), else from the URDF.
        self.frames = frames if frames is not None else resolve(scene)
        self.start_perf = float(start_perf)
        self.file = EpisodeFile(
            path, clock_offset_ns=clock_offset_ns, topic_names=topic_names
        )
        self.cameras = scene.cameras
        self.camera_stats = {src: _CameraStats() for src in self.cameras}
        self.joint_samples = 0
        self.joint_lost = 0
        self.tick_samples = 0
        self.tick_lost = 0
        self.end_perf: float | None = None

        self._joint_names = scene.joint_state_names()
        self._joint_cols = np.array(scene.joint_columns(), dtype=np.int64)
        self._arm_tf = [scene.tree.joints[n] for n in scene.arm_tf_joints()]
        self._ee_links = scene.ee_links()
        self._ee_base = scene.ee_base()
        self._ee_paths = {
            side: scene.tree.path(self._ee_base, link)
            for side, link in self._ee_links.items()
        }
        # /ee_pose and /ee_target are at the TCP (the SDK's tool point), not
        # the gripper mount the URDF chain ends at.
        tools = {mount: transform for mount, _, transform in scene.tool_frames()}
        self._ee_tool = {
            side: tools[link] for side, link in self._ee_links.items() if link in tools
        }
        self.ee_frames = {
            side: self.frame(name) for side, name in scene.ee_frames().items()
        }
        self._ee_frame = self.frame(self._ee_base)
        jk = column_index(joint_ring_keys())
        self._pos0 = jk[f"pos.left_{JOINT_NAMES[0]}"]
        self._vel0 = jk[f"vel.left_{JOINT_NAMES[0]}"]
        self._trq0 = jk[f"trq.left_{JOINT_NAMES[0]}"]
        self._tick = column_index(tick_ring_keys())
        self._cmd0 = self._tick[f"cmd.left_{JOINT_NAMES[0]}"]

        self.odometry = (
            XDriveOdometry(wheel_signs) if scene.jelly_wheels and wheel_signs else None
        )
        self._wheel_t: float | None = None
        self._wheel_origin: np.ndarray | None = None
        self._lift_value: float | None = None
        self._lift_written_at: float | None = None
        self._cmd_wheels: tuple[float, ...] | None = None
        self._vr_key: float | None = None
        self._base_command_id: float | None = None
        self.imu_samples: dict[str, int] = {}
        self._imu_last: dict[str, float] = {}
        # Application times from the joint feed: per arm, target seq -> perf.
        self._applied_at: dict[str, OrderedDict[int, float]] = {
            side: OrderedDict() for side in SIDES
        }
        self._applied_last: dict[str, float] = {}
        self._applied_cols = {
            side: (jk[f"applied.seq_{side}"], jk[f"applied.t_{side}"]) for side in SIDES
        }
        self._joint_clock: float | None = None
        # /joint_command/info waiting for its application time (FIFO).
        self._pending_info: deque[tuple] = deque()
        self.commands_applied = 0
        self.commands_unapplied = 0
        # The device each /joint_states name belongs to: an arm or a gripper.
        self._devices = [
            name if name.endswith("_gripper") else f"{name.split('_', 1)[0]}_arm"
            for name in self._joint_names
        ]

        self._write_static()

    # -- helpers -----------------------------------------------------------

    def stamp(self, perf_s: float) -> int:
        return self.file.stamp(perf_s)

    def frame(self, name: str) -> str:
        """A frame's name as this episode writes it (after any rename)."""
        return self.frame_names.get(name, name)

    def _matrices(self, joints: list[Any], values: dict[str, float]) -> dict[str, Any]:
        tree = self.scene.tree
        return {j.name: j.transform(tree.value(j, values)) for j in joints}

    def _transforms(
        self,
        stamp: int,
        joints: list[Any],
        values: dict[str, float],
        matrices: dict[str, Any] | None = None,
    ) -> list[tuple[int, str, str, tuple[float, ...], tuple[float, ...]]]:
        if matrices is None:
            matrices = self._matrices(joints, values)
        out = []
        for joint in joints:
            transform = matrices[joint.name]
            t = transform[:3, 3]
            out.append(
                (
                    stamp,
                    self.frame(joint.parent),
                    self.frame(joint.child),
                    (float(t[0]), float(t[1]), float(t[2])),
                    quaternion_xyzw(transform[:3, :3]),
                )
            )
        return out

    def _ee_pose(
        self,
        side: str,
        values: dict[str, float],
        matrices: dict[str, Any] | None = None,
    ) -> tuple[float, ...]:
        tree = self.scene.tree
        out = None
        for joint in self._ee_paths[side]:
            if matrices is not None and joint.name in matrices:
                step = matrices[joint.name]
            else:
                step = joint.transform(tree.value(joint, values))
            out = step if out is None else out @ step
        if out is None:
            out = np.eye(4)
        tool = self._ee_tool.get(side)
        if tool is not None:
            out = out @ tool
        return pose7(out)

    # -- static content ----------------------------------------------------

    def _write_static(self) -> None:
        stamp = self.stamp(self.start_perf)
        static = []
        for frame in self.frames.frames:
            transform = frame.transform
            t = transform[:3, 3]
            static.append(
                (
                    stamp,
                    self.frame(frame.parent),
                    self.frame(frame.child),
                    (float(t[0]), float(t[1]), float(t[2])),
                    quaternion_xyzw(transform[:3, :3]),
                )
            )
        self.file.write(
            "/tf_static", "tf2_msgs/TFMessage", messages.tf_message(static), stamp
        )
        self.file.write(
            "/robot_description",
            "std_msgs/String",
            messages.string_message(
                self.scene.robot_description(
                    self.frames.edges, self.frames.added, self.frame_names
                )
            ),
            stamp,
        )
        calibration = self.frames.calibration
        if calibration is not None:
            # The file itself, so the measured values can be checked later.
            self.file.attachment(
                CALIBRATION_ATTACHMENT, "application/json", calibration.raw, stamp
            )

    # -- video ---------------------------------------------------------------

    def add_video(self, source: str, au: bytes, exposure_perf: float) -> None:
        """One encoded frame, stamped with its exposure time."""
        camera: CameraSource = self.cameras[source]
        stats = self.camera_stats[source]
        if stats.last is not None:
            dt = exposure_perf - stats.last
            if dt <= 0:
                return  # never write a frame out of order or twice
            period = 1.0 / camera.fps
            if dt > 1.5 * period:
                stats.gaps += 1
                stats.missing += max(1, round(dt / period) - 1)
        else:
            stats.first = exposure_perf
        stats.last = exposure_perf
        stats.frames += 1
        stamp = self.stamp(exposure_perf)
        self.file.write(
            f"{camera.topic}/video",
            "foxglove_msgs/CompressedVideo",
            messages.compressed_video(
                stamp, self.frame(camera.frame_id), au, camera.codec
            ),
            stamp,
        )
        info = camera.intrinsics
        if info is not None:
            self.file.write(
                f"{camera.topic}/camera_info",
                "sensor_msgs/CameraInfo",
                messages.camera_info(
                    stamp,
                    self.frame(camera.frame_id),
                    width=info.width,
                    height=info.height,
                    distortion_model=info.distortion_model,
                    d=info.d,
                    k=info.k,
                    r=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                    p=info.p,
                ),
                stamp,
            )

    # -- IMU ---------------------------------------------------------------

    def add_imu(self, topic: str, frame_id: str, samples: list) -> None:
        """Camera IMU samples (:class:`~.imu.ImuSample`), each at its own time.

        Samples from before the episode started, or not newer than the last
        one written, are skipped.
        """
        last = self._imu_last.get(topic, self.start_perf)
        frame_id = self.frame(frame_id)
        for sample in samples:
            if sample.perf <= last:
                continue
            last = sample.perf
            stamp = self.stamp(sample.perf)
            self.file.write(
                topic,
                "sensor_msgs/Imu",
                messages.imu(
                    stamp, frame_id, sample.angular_velocity, sample.acceleration
                ),
                stamp,
            )
            self.imu_samples[topic] = self.imu_samples.get(topic, 0) + 1
        self._imu_last[topic] = last

    def frame_count(self) -> int:
        """Frames of the camera with the fewest (0 until every camera has one)."""
        if not self.camera_stats:
            return 0
        return min(s.frames for s in self.camera_stats.values())

    # -- 240 Hz joint rows ---------------------------------------------------

    def add_joint_rows(self, rows: np.ndarray) -> None:
        cols = self._joint_cols
        for row in rows:
            ts = float(row[0])
            positions = row[self._pos0 : self._pos0 + 2 * _N]
            velocities = row[self._vel0 : self._vel0 + 2 * _N]
            efforts = row[self._trq0 : self._trq0 + 2 * _N]
            stamp = self.stamp(ts)
            self.file.write(
                "/joint_states",
                "sensor_msgs/JointState",
                messages.joint_state(
                    stamp,
                    self._joint_names,
                    positions[cols].tolist(),
                    velocities[cols].tolist(),
                    efforts[cols].tolist(),
                ),
                stamp,
            )
            values = self.scene.urdf_values(positions)
            matrices = self._matrices(self._arm_tf, values)
            if self._arm_tf:
                self.file.write(
                    "/tf",
                    "tf2_msgs/TFMessage",
                    messages.tf_message(
                        self._transforms(stamp, self._arm_tf, values, matrices)
                    ),
                    stamp,
                )
            for side in self._ee_links:
                self.file.write(
                    f"/ee_pose/{side}",
                    "geometry_msgs/PoseStamped",
                    messages.pose_stamped(
                        stamp, self._ee_frame, self._ee_pose(side, values, matrices)
                    ),
                    stamp,
                )
            self.joint_samples += 1
            self._note_applied(row)
            self._joint_clock = ts
        self._flush_command_info()

    def _note_applied(self, row: np.ndarray) -> None:
        """Remember each arm's newly applied target and when it was applied."""
        for side, (seq_col, t_col) in self._applied_cols.items():
            seq, t = float(row[seq_col]), float(row[t_col])
            if not (math.isfinite(seq) and math.isfinite(t)):
                continue
            if self._applied_last.get(side) == seq:
                continue
            self._applied_last[side] = seq
            memory = self._applied_at[side]
            memory[int(seq)] = t
            while len(memory) > _APPLIED_MEMORY:
                memory.popitem(last=False)

    def _flush_command_info(self, final: bool = False) -> None:
        """Write the pending command infos whose application is known.

        In order: one still waiting holds back the later ones, until the
        joint feed has moved :data:`_APPLY_WAIT_S` past its issue time (or the
        episode ends), when it is written with its application unknown.
        """
        while self._pending_info:
            stamp, command_id, names, devices, modes, seqs, issue_t = (
                self._pending_info[0]
            )
            applied: dict[str, float | None] = {}
            for side in SIDES:
                seq = seqs[side]
                applied[side] = (
                    self._applied_at[side].get(int(seq)) if math.isfinite(seq) else None
                )
            waiting = any(
                math.isfinite(seqs[side]) and applied[side] is None for side in SIDES
            )
            timed_out = (
                self._joint_clock is not None
                and self._joint_clock - issue_t > _APPLY_WAIT_S
            )
            if waiting and not (final or timed_out):
                return
            self._pending_info.popleft()
            application = []
            for device in devices:
                t = applied.get(device.split("_", 1)[0])
                application.append(self.stamp(t) if t is not None else 0)
            if waiting:
                self.commands_unapplied += 1
            else:
                self.commands_applied += 1
            self.file.write(
                "/joint_command/info",
                "almond_msgs/CommandInfo",
                messages.command_info(
                    stamp, command_id, stamp, names, devices, modes, application
                ),
                stamp,
            )

    # -- 120 Hz tick rows ----------------------------------------------------

    def add_tick_rows(self, rows: np.ndarray) -> None:
        for row in rows:
            self._tick_row(row)
            self.tick_samples += 1
        self._flush_command_info()

    def _col(self, row: np.ndarray, key: str) -> float:
        return float(row[self._tick[key]])

    def _tick_row(self, row: np.ndarray) -> None:
        ts = float(row[0])
        issue_t = self._col(row, "cmd.issue_t")
        if not math.isfinite(issue_t):
            issue_t = ts
        stamp = self.stamp(issue_t)
        commanded = row[self._cmd0 : self._cmd0 + 2 * _N]
        cols = self._joint_cols
        picked = commanded[cols]
        keep = np.isfinite(picked)
        if keep.any():
            arm_mode = mode_name(self._col(row, "cmd.arm_mode")) or "impedance"
            names, devices, modes = [], [], []
            for name, device, kept in zip(
                self._joint_names, self._devices, keep, strict=True
            ):
                if kept:
                    names.append(name)
                    devices.append(device)
                    modes.append(
                        "position_force" if device.endswith("_gripper") else arm_mode
                    )
            command_id = self._col(row, "cmd.id")
            self.file.write(
                "/joint_command",
                "sensor_msgs/JointState",
                messages.joint_state(stamp, names, picked[keep].tolist()),
                stamp,
            )
            self._pending_info.append(
                (
                    stamp,
                    int(command_id) if math.isfinite(command_id) else 0,
                    names,
                    devices,
                    modes,
                    {side: self._col(row, f"cmd.seq_{side}") for side in SIDES},
                    issue_t,
                )
            )
            if np.isfinite(commanded[cols]).all():
                values = self.scene.urdf_values(commanded)
                for side in self._ee_links:
                    self.file.write(
                        f"/ee_target/{side}",
                        "geometry_msgs/PoseStamped",
                        messages.pose_stamped(
                            stamp, self._ee_frame, self._ee_pose(side, values)
                        ),
                        stamp,
                    )
        self._teleop(row, ts)
        self._jelly(row, ts)

    def _teleop(self, row: np.ndarray, tick_ts: float) -> None:
        seq = self._col(row, "vr.seq")
        t_host = self._col(row, "vr.t_host")
        left_x = self._col(row, "vr.l.px")
        if not math.isfinite(left_x):
            return  # no headset frame yet
        key = seq if math.isfinite(seq) else (t_host if math.isfinite(t_host) else None)
        if key is not None and key == self._vr_key:
            return  # the same headset frame as last tick
        self._vr_key = key
        ts = t_host if math.isfinite(t_host) else tick_ts
        stamp = self.stamp(ts)
        for prefix, side in (("l", "left"), ("r", "right")):
            pose = [self._col(row, f"vr.{prefix}.{axis}") for axis in POSE_AXES]
            self.file.write(
                f"/teleop/{side}/pose",
                "geometry_msgs/PoseStamped",
                messages.pose_stamped(stamp, self.frame(VR_FRAME), pose),
                stamp,
            )
        axes = [self._col(row, f"vr.{name}") for name in VR_AXES]
        buttons = [self._col(row, f"vr.{name}") for name in VR_BUTTONS]
        self.file.write(
            "/teleop/joy",
            "sensor_msgs/Joy",
            messages.joy(
                stamp,
                self.frame(VR_FRAME),
                [a if math.isfinite(a) else 0.0 for a in axes],
                [1 if math.isfinite(b) and b >= 0.5 else 0 for b in buttons],
            ),
            stamp,
        )

    def _jelly(self, row: np.ndarray, tick_ts: float) -> None:
        scene = self.scene
        t = self._col(row, "jelly.t")
        if self.odometry is not None and math.isfinite(t) and t != self._wheel_t:
            positions = np.array([self._col(row, f"jelly.pos{i}") for i in WHEEL_IDS])
            speeds = np.array([self._col(row, f"jelly.vel{i}") for i in WHEEL_IDS])
            if np.isfinite(positions).all() and np.isfinite(speeds).all():
                self._wheel_sample(t, positions, speeds)

        lift = self._col(row, "jelly.lift")
        if scene.jelly_lift and math.isfinite(lift):
            value = scene.lift_value(lift)
            stale = (
                self._lift_written_at is None
                or tick_ts - self._lift_written_at >= _LIFT_REFRESH_S
            )
            if (
                self._lift_value is None
                or abs(value - self._lift_value) > 1e-4
                or stale
            ):
                self._lift_value = value
                self._lift_written_at = tick_ts
                stamp = self.stamp(tick_ts)
                self.file.write(
                    "/base/lift",
                    "sensor_msgs/JointState",
                    messages.joint_state(stamp, [LIFT_JOINT], [value]),
                    stamp,
                )
                joints = [scene.tree.joints[n] for n in scene.driven_by(LIFT_JOINT)]
                self.file.write(
                    "/tf",
                    "tf2_msgs/TFMessage",
                    messages.tf_message(
                        self._transforms(stamp, joints, {LIFT_JOINT: value})
                    ),
                    stamp,
                )

        command_id = self._col(row, "jelly.cmd_id")
        if math.isfinite(command_id) and command_id != self._base_command_id:
            self._base_command_id = command_id
            issued = self._col(row, "jelly.cmd_t")
            stamp = self.stamp(issued if math.isfinite(issued) else tick_ts)
            parked = self._col(row, "jelly.parked")
            lift_direction = self._col(row, "jelly.cmd_lift")
            self.file.write(
                "/base/command",
                "almond_msgs/BaseCommand",
                messages.base_command(
                    stamp,
                    int(command_id),
                    stamp,
                    "jelly",
                    "position_hold" if parked == 1.0 else "velocity",
                    *(
                        v if math.isfinite(v) else 0.0
                        for v in (
                            self._col(row, "jelly.cmd_vx"),
                            self._col(row, "jelly.cmd_vy"),
                            self._col(row, "jelly.cmd_wz"),
                        )
                    ),
                    int(lift_direction) if math.isfinite(lift_direction) else 0,
                ),
                stamp,
            )

        if self.odometry is not None:
            command = tuple(self._col(row, f"jelly.cmd{i}") for i in WHEEL_IDS)
            if all(math.isfinite(c) for c in command) and command != self._cmd_wheels:
                self._cmd_wheels = command
                vx, vy, wz = self.odometry.command_twist(command)
                stamp = self.stamp(tick_ts)
                self.file.write(
                    "/base/cmd_vel",
                    "geometry_msgs/TwistStamped",
                    messages.twist_stamped(
                        stamp, self.frame("jelly"), (vx, vy, 0.0), (0.0, 0.0, wz)
                    ),
                    stamp,
                )

    def _wheel_sample(
        self, t: float, positions: np.ndarray, speeds: np.ndarray
    ) -> None:
        assert self.odometry is not None
        if self._wheel_origin is None:
            self._wheel_origin = positions.copy()
            self.odometry.reset()
        self._wheel_t = t
        sample = self.odometry.update(t, positions, speeds)
        stamp = self.stamp(t)
        names = [f"jelly_wheel_{i}" for i in WHEEL_IDS]
        self.file.write(
            "/base/wheel_states",
            "sensor_msgs/JointState",
            messages.joint_state(
                stamp,
                names,
                (positions - self._wheel_origin).tolist(),
                speeds.tolist(),
            ),
            stamp,
        )
        values = {"base_x": sample.x, "base_y": sample.y, "base_yaw": sample.yaw}
        tree = self.scene.tree
        joints = [tree.joints[n] for n in BASE_JOINTS]
        self.file.write(
            "/tf",
            "tf2_msgs/TFMessage",
            messages.tf_message(self._transforms(stamp, joints, values)),
            stamp,
        )
        base_pose = pose7(tree.fk(tree.root, "jelly", values))
        self.file.write(
            "/base/odom",
            "nav_msgs/Odometry",
            messages.odometry(
                stamp,
                self.frame(tree.root),
                self.frame("jelly"),
                base_pose,
                (sample.vx, sample.vy, 0.0),
                (0.0, 0.0, sample.wz),
            ),
            stamp,
        )

    # -- finishing -----------------------------------------------------------

    def camera_metadata(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for source, camera in self.cameras.items():
            stats = self.camera_stats[source]
            prefix = camera.topic.strip("/").replace("/", "_")
            span = (
                stats.last - stats.first
                if stats.first is not None and stats.last is not None
                else 0.0
            )
            out.update(
                {
                    f"{prefix}.topic": self.file.topic_names.get(
                        f"{camera.topic}/video", f"{camera.topic}/video"
                    ),
                    f"{prefix}.frame_id": self.frame(camera.frame_id),
                    f"{prefix}.serial": camera.serial,
                    f"{prefix}.width": camera.width,
                    f"{prefix}.height": camera.height,
                    f"{prefix}.fps": camera.fps,
                    f"{prefix}.frames": stats.frames,
                    f"{prefix}.measured_fps": (
                        f"{(stats.frames - 1) / span:.3f}" if span > 0 else ""
                    ),
                    f"{prefix}.gaps": stats.gaps,
                    f"{prefix}.missing_frames": stats.missing,
                    f"{prefix}.distortion_model": (
                        camera.intrinsics.distortion_model
                        if camera.intrinsics
                        else "none"
                    ),
                }
            )
        return out

    def flush(self) -> None:
        """Write everything still waiting (call once capture has stopped)."""
        self._flush_command_info(final=True)

    def finish(self, metadata: dict[str, dict[str, Any]]) -> Path:
        for name, values in metadata.items():
            self.file.metadata(name, values)
        return self.file.finish()

    def discard(self) -> None:
        self.file.discard()
