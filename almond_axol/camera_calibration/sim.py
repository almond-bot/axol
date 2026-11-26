"""The calibration procedure in the sim: both recordings, made without a robot.

The sim's robot is Axol on Jelly with the Camera Kit, with its cameras a few
millimetres from CAD (its true mounts, which the calibration should find). A
ChArUco board lies on a table in front of it. Each recording:

1. **Head phase.** The board moves in front of the head camera (the sim's
   stand-in for a hand) through poses that cover the image, pausing at each.
2. **Table phases.** The board lies on the table. The tool finds it with the
   head camera, then moves each arm in turn through generated wrist camera
   poses around it, pausing at each. Recording A uses two board placements,
   B one, all different.

The recordings are rendered through each camera's factory lens model and
written by the MCAP recorder, then go through :mod:`.procedure` exactly as a
robot's recordings would.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..synthetic.episode import (
    _DT,
    CONTROL_HZ,
    READY,
    TABLE_TOP,
    WHEEL_SIGNS,
    EpisodeSpec,
    Rig,
    SyntheticOptions,
    Timeline,
    _props,
    exposure_times,
    follow,
    play_episode,
    render_video,
)
from ..synthetic.motion import min_jerk, rotation_exp, rotation_log
from ..synthetic.render import Prop
from .board import Board
from .optics import Lens
from .poses import head_board_poses, wrist_poses

_SIDES = ("left", "right")
# Move and hold times (s).
_ARM_SPEED = 0.9  # rad/s, the fastest joint
_ARM_HOLD = 0.5
_BOARD_MOVE = 0.6
_BOARD_HOLD = 0.5
_PLACE_MOVE = 1.5


@dataclass
class SimOptions:
    """The sim's production image mode and how much to record.

    The board's markers need the full 1200p image; a low frame rate keeps
    the rendering time down (every frame is still recorded and kept).
    """

    fps: int = 10
    width: int = 1920
    height: int = 1200
    resolution: str = "HD1200"
    codec: str = "h264"
    quality: int = 20
    workers: int = 4
    supersample: float = 1.0
    seed: int = 0
    head_a: int = 24
    head_b: int = 14
    # Wrist poses per arm per placement.
    wrist_a: int = 11
    wrist_b: int = 12


def _lens(spec: Any) -> Lens:
    return Lens(
        width=spec.width,
        height=spec.height,
        k=np.array([[spec.fx, 0.0, spec.cx], [0.0, spec.fy, spec.cy], [0.0, 0.0, 1.0]]),
        d=np.asarray(spec.d, dtype=np.float64),
        fisheye=spec.model == "equidistant",
    )


def board_prop(board: Board, px_per_square: int = 160) -> tuple[Prop, np.ndarray]:
    """The board as a scene prop, and the prop's pose in the board frame.

    The prop's top face shows the board; the board frame has x along the
    long side, y down the board image and z into the board, so the prop is
    the board turned half a turn about x, centred on the board.
    """
    margin = 0.5
    image = board.image(px_per_square, margin)
    rgb = np.repeat(image[..., None], 3, axis=2)
    w, h = board.size_m
    m = margin * board.square_m
    thickness = 0.0015
    prop = Prop(
        "board", "board", ((w + 2 * m) / 2, (h + 2 * m) / 2, thickness), texture=rgb
    )
    in_board = np.eye(4)
    in_board[:3, :3] = np.diag([1.0, -1.0, -1.0])
    in_board[:3, 3] = (w / 2, h / 2, 0.0)
    return prop, in_board


def _blend(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    out = np.eye(4)
    out[:3, 3] = a[:3, 3] + s * (b[:3, 3] - a[:3, 3])
    out[:3, :3] = rotation_exp(s * rotation_log(b[:3, :3] @ a[:3, :3].T)) @ a[:3, :3]
    return out


class _Plan:
    """Commanded joints and the board's pose, segment by segment."""

    def __init__(self, start: dict[str, np.ndarray], board: np.ndarray) -> None:
        self.t = 0.0
        self.arm_segments: dict[
            str, list[tuple[float, float, np.ndarray, np.ndarray]]
        ] = {side: [] for side in _SIDES}
        self.arm_now = {side: np.asarray(q, float) for side, q in start.items()}
        self.board_segments: list[tuple[float, float, np.ndarray, np.ndarray]] = []
        self.board_now = board
        self.phases: list[tuple[str, float, float, int, str]] = []

    def move_arm(self, side: str, target: np.ndarray, hold: float = _ARM_HOLD) -> None:
        target = np.asarray(target, float)
        span = float(np.max(np.abs(target - self.arm_now[side])))
        duration = min(3.0, max(0.8, span / _ARM_SPEED))
        self.arm_segments[side].append(
            (self.t, self.t + duration, self.arm_now[side], target)
        )
        self.arm_now[side] = target
        self.t += duration + hold

    def move_board(self, pose: np.ndarray, duration: float, hold: float) -> None:
        self.board_segments.append((self.t, self.t + duration, self.board_now, pose))
        self.board_now = pose
        self.t += duration + hold

    def arm_at(self, side: str, t: np.ndarray, start: np.ndarray) -> np.ndarray:
        out = np.repeat(start[None], len(t), axis=0)
        for t0, t1, a, b in self.arm_segments[side]:
            inside = (t >= t0) & (t < t1)
            s = np.array([min_jerk(v) for v in (t[inside] - t0) / (t1 - t0)])
            out[inside] = a + s[:, None] * (b - a)
            out[t >= t1] = b
        return out

    def board_at(self, t: np.ndarray, start: np.ndarray) -> np.ndarray:
        out = np.repeat(start[None], len(t), axis=0)
        for t0, t1, a, b in self.board_segments:
            for i in np.nonzero((t >= t0) & (t < t1))[0]:
                out[i] = _blend(a, b, min_jerk((t[i] - t0) / (t1 - t0)))
            out[t >= t1] = b
        return out


class CalibrationSim:
    """Makes calibration recordings of the sim's robot."""

    def __init__(
        self,
        work: Path,
        board: Board,
        options: SimOptions | None = None,
        progress: Callable[[str], None] = print,
    ) -> None:
        self.options = options or SimOptions()
        o = self.options
        self.work = work
        self.board = board
        self.progress = progress
        self.synthetic = SyntheticOptions(
            output=work,
            fps=o.fps,
            width=o.width,
            height=o.height,
            resolution=o.resolution,
            codec=o.codec,
            quality=o.quality,
            workers=o.workers,
            operator="calibration",
            supersample=o.supersample,
        )
        self.rig = Rig(work / "rig", self.synthetic)
        self.kin = self.rig.kin
        self.tree = self.rig.tree
        self.root_z = -2.0 * self.rig.lift
        self.table_z = TABLE_TOP - self.root_z  # table top in root
        self.prop, self.prop_in_board = board_prop(board)
        table = [
            p
            for p in _props({"cube": np.eye(4), "cup": np.eye(4)})
            if p.name.startswith("table")
        ]
        self.props = [*table, self.prop]
        self.lenses = {src: _lens(spec) for src, spec in self.rig.lens.items()}
        self.head_in_root = self.tree.fk("root", "head_camera_left_optical", {})
        self.camera_in_gripper = {
            side: self.tree.fk(f"{side}_gripper", f"{side}_wrist_camera_optical", {})
            for side in _SIDES
        }
        self.ready = {side: np.array(READY[side]) for side in _SIDES}
        self.tuck = {side: self._tuck(side) for side in _SIDES}
        self.rng = np.random.default_rng(o.seed)
        # The cameras' true mounts (what the calibration should find).
        truth = self.rig.truth
        self.truth = {
            "head": truth.fk("root", "head_camera_left_optical", {}),
            "left_wrist": truth.fk("left_w2", "left_wrist_camera_optical", {}),
            "right_wrist": truth.fk("right_w2", "right_wrist_camera_optical", {}),
        }

    def _tuck(self, side: str) -> np.ndarray:
        """Arms raised and drawn back, out of the head camera's way."""
        names = self.kin.joints[side]
        seed = dict(zip(names, READY[side], strict=True))
        target = self.kin.tcp(side, seed)
        target[:3, 3] += (-0.14, 0.12 if side == "left" else -0.12, 0.16)
        q, err = self.kin.solve(side, target, seed, iterations=120)
        return np.array([q[n] for n in names]) if err < 1e-3 else self.ready[side]

    # -- poses -----------------------------------------------------------

    def board_on_table(self, x: float, y: float, yaw_deg: float) -> np.ndarray:
        """The board lying on the table, centred at (x, y) in root."""
        base = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        a = math.radians(yaw_deg)
        rz = np.array(
            [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]]
        )
        out = np.eye(4)
        out[:3, :3] = rz @ base
        w, h = self.board.size_m
        surface = self.table_z + 2 * self.prop.size[2]
        out[:3, 3] = np.array([x, y, surface]) - out[:3, :3] @ np.array(
            [w / 2, h / 2, 0.0]
        )
        return out

    def _world(self, board_in_root: np.ndarray) -> np.ndarray:
        """The board prop's world pose for a board pose in root."""
        root = np.eye(4)
        root[2, 3] = self.root_z
        return root @ board_in_root @ self.prop_in_board

    def _timeline(self, plan: _Plan, start_board: np.ndarray) -> Timeline:
        duration = plan.t + 0.5
        t = np.arange(0.0, duration, _DT)
        commanded = {side: plan.arm_at(side, t, self.tuck[side]) for side in _SIDES}
        cmd_t = np.arange(0.0, duration, 1.0 / CONTROL_HZ)
        index = np.clip((cmd_t / _DT).astype(int), 0, len(t) - 1)
        cmd_q = {side: commanded[side][index] for side in _SIDES}
        cmd_grip = {side: np.ones(len(cmd_t)) for side in _SIDES}
        q, grip = follow(t, cmd_t, cmd_q, cmd_grip, self.rng)
        board_root = plan.board_at(t, start_board)
        board_world = np.array([self._world(p) for p in board_root])
        return Timeline(
            t=t,
            cmd_t=cmd_t,
            cmd_q=cmd_q,
            cmd_grip=cmd_grip,
            q=q,
            grip=grip,
            base=np.zeros((len(t), 3)),
            base_v=np.zeros((len(t), 3)),
            objects={"board": board_world},
            tcp_target={
                side: np.repeat(np.eye(4)[None], len(cmd_t), axis=0) for side in _SIDES
            },
            drive_until=0.0,
            headset=False,
            props=self.props,
        )

    # -- finding the board -----------------------------------------------

    def locate(self, placements: Sequence[np.ndarray]) -> list[np.ndarray | None]:
        """Each placement as the head camera sees it (board in root, from the
        head camera's CAD mount): what the tool aims the wrist cameras at."""
        import av

        plan = _Plan(self.tuck, placements[0])
        for pose in placements:
            plan.move_board(pose, 0.01, 0.99)
        timeline = self._timeline(plan, placements[0])
        times = {"overhead_left": np.array([0.5 + i for i in range(len(placements))])}
        aus = render_video(
            self.rig, timeline, self.synthetic, times, sources=["overhead_left"]
        )
        lens = self.lenses["overhead_left"]
        out: list[np.ndarray | None] = []
        for au in aus["overhead_left"]:
            ctx = av.CodecContext.create(
                "hevc" if self.synthetic.codec == "h265" else "h264", "r"
            )
            frames = ctx.decode(av.Packet(au)) + ctx.decode(None)
            detection = (
                self.board.detect(frames[0].to_ndarray(format="gray"))
                if frames
                else None
            )
            pose = None
            if detection is not None:
                pose = lens.board_pose(
                    self.board.points[detection.ids], detection.pixels
                )
            out.append(None if pose is None else self.head_in_root @ pose)
        return out

    # -- recordings ------------------------------------------------------

    def plan(
        self,
        placements: Sequence[np.ndarray],
        estimates: Sequence[np.ndarray],
        head_poses: int,
        wrist_poses_each: int,
    ) -> tuple[_Plan, np.ndarray]:
        head = head_board_poses(
            self.lenses["overhead_left"],
            self.head_in_root,
            board_size=self.board.size_m,
            points=self.board.points,
            count=head_poses,
            rng=self.rng,
            min_z=self.table_z + 0.12,
        )
        start = head[0] if head else placements[0]
        plan = _Plan(self.tuck, start)
        begin = plan.t
        for pose in head:
            plan.move_board(pose, _BOARD_MOVE, _BOARD_HOLD)
        plan.phases.append(("head", begin, plan.t, 0, ""))
        for number, (placement, estimate) in enumerate(
            zip(placements, estimates, strict=True), start=1
        ):
            plan.move_board(placement, _PLACE_MOVE, 0.5)
            target = estimate if estimate is not None else placement
            for side in _SIDES:
                found = wrist_poses(
                    self.kin,
                    side,
                    board_in_root=target,
                    board_size=self.board.size_m,
                    points=self.board.points,
                    lens=self.lenses[f"{side}_arm"],
                    camera_in_gripper=self.camera_in_gripper[side],
                    table_z=self.table_z,
                    seed=dict(zip(self.kin.joints[side], READY[side], strict=True)),
                    count=wrist_poses_each,
                    rng=self.rng,
                )
                self.progress(
                    f"placement {number}, {side} arm: {len(found)} wrist poses"
                )
                begin = plan.t
                for pose in found:
                    plan.move_arm(
                        side, np.array([pose.joints[n] for n in self.kin.joints[side]])
                    )
                plan.move_arm(side, self.tuck[side], hold=0.2)
                plan.phases.append(("table", begin, plan.t, number, side))
        return plan, start

    def record(self, label: str, plan: _Plan, start_board: np.ndarray) -> Path:
        """Render and record one recording; returns its MCAP file."""
        from ..recording.mcap.imu import ImuReceiver
        from ..recording.mcap.recorder import McapSession
        from ..recording.mcap.rings import (
            JOINT_RING_CAPACITY,
            TICK_RING_CAPACITY,
            joint_ring_keys,
            tick_ring_keys,
        )
        from ..recording.mcap.scene import imu_topic
        from ..synthetic.episode import _PlaybackReader
        from ..video.shm_frames import SnapshotReader, SnapshotWriter

        timeline = self._timeline(plan, start_board)
        duration = float(timeline.t[-1])
        exposures = {
            src: exposure_times(src, self.synthetic.fps, duration, self.rng)
            for src in self.rig.raw_meta
        }
        frames = sum(len(v) for v in exposures.values())
        self.progress(f"Recording {label}: {duration:.0f} s, rendering {frames} frames")
        aus = render_video(self.rig, timeline, self.synthetic, exposures, self.progress)
        root = self.work / f"recording_{label}"
        tick_writer = SnapshotWriter(tick_ring_keys(), [], capacity=TICK_RING_CAPACITY)
        joint_writer = SnapshotWriter(
            joint_ring_keys(), [], capacity=JOINT_RING_CAPACITY
        )
        readers = {src: _PlaybackReader() for src in self.rig.raw_meta}
        session = McapSession(
            {
                "dataset_root": str(root),
                "operator": "calibration",
                "robot_serial": self.synthetic.robot_serial,
                "robot_model": "mobile",
                "gripper": "parallel",
                "software_version": self.synthetic.software_version,
                "hostname": "synthetic",
                "wheel_signs": dict(WHEEL_SIGNS),
                # No calibration file: recordings place the cameras from CAD.
                "camera_calibration_path": str(self.work / "no_calibration.json"),
                "teleop_hz": CONTROL_HZ,
                "episode_metadata": {
                    "calibration_recording": label,
                    "synthetic": "true",
                },
            },
            cameras=readers,
            scene=self.rig.scene,
            tick_reader=SnapshotReader(tick_writer.name, tick_ring_keys(), []),
            joint_reader=SnapshotReader(joint_writer.name, joint_ring_keys(), []),
        )
        session.compute_serial = "SYNTHETIC"
        session.imus = {
            imu_topic("overhead"): (
                ImuReceiver(self.rig.imu_socket),
                "head_camera_left_optical",
            )
        }
        try:
            self.progress(
                f"Recording {label}: playing {duration:.0f} s into the recorder"
            )
            play_episode(
                session,
                self.rig,
                timeline,
                aus,
                exposures,
                EpisodeSpec(
                    f"Camera calibration, recording {label}",
                    "calibration",
                    False,
                    seed=self.options.seed,
                ),
                duration,
                (tick_writer, joint_writer),
                phases=plan.phases,
            )
        finally:
            session.close()
            tick_writer.close()
            joint_writer.close()
        return sorted(root.glob("episode_*.mcap"))[-1]


def run_session(
    output: Path,
    board: Board,
    *,
    options: SimOptions | None = None,
    limits: Any = None,
    calibration_path: Path | None = None,
    valid_days: float = 30.0,
    progress: Callable[[str], None] = print,
) -> Any:
    """Record A and B in the sim and run the procedure on them."""
    from .procedure import run

    output = Path(output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="calibration-sim-", dir=output))
    sim = CalibrationSim(work, board, options, progress)
    o = sim.options
    places_a = [
        sim.board_on_table(0.56, 0.06, 8.0),
        sim.board_on_table(0.54, -0.07, -14.0),
    ]
    places_b = [sim.board_on_table(0.57, 0.0, 3.0)]
    progress("Finding the board with the head camera")
    estimates = sim.locate([*places_a, *places_b])
    plan_a, start_a = sim.plan(places_a, estimates[:2], o.head_a, o.wrist_a)
    path_a = sim.record("A", plan_a, start_a)
    plan_b, start_b = sim.plan(places_b, estimates[2:], o.head_b, o.wrist_b)
    path_b = sim.record("B", plan_b, start_b)
    arm_joints = {side: sim.kin.joints[side] for side in _SIDES}
    return run(
        path_a,
        path_b,
        board=board,
        tree=sim.tree,
        arm_joints=arm_joints,
        robot_serial=sim.synthetic.robot_serial,
        packages=output,
        calibration_path=calibration_path,
        limits=limits,
        valid_days=valid_days,
        identity={"sim": True},
        truth=sim.truth,
        progress=progress,
    )
