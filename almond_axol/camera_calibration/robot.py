"""The calibration procedure on Axol: both recordings, made with the robot.

The session records like Collect data, with the same cameras, image mode,
encoder and recorder (MCAP, every camera, the head camera's IMU, the joints),
so the calibration sees exactly the images episodes get. Each recording:

1. **Head phase.** Both arms go to a raised pose. The operator moves the
   board slowly in front of the head camera, pausing at each spot; the tool
   counts distinct poses, tilted views and image coverage as they come and
   says when there are enough.
2. **Table phases.** The operator lays the board flat on the table. The tool
   finds it with the head camera, generates wrist camera poses around it and
   moves each arm through them, pausing at each (the other arm stays raised).

Recording A uses two board placements, Recording B one. Then
:mod:`.procedure` fits A, locks it, checks B and writes the package.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .board import Board
from .optics import Lens
from .poses import Coverage, distinct, route, tilted_views, wrist_poses
from .recording import Phase, phases_json

_logger = logging.getLogger(__name__)
_SIDES = ("left", "right")
_GRIP_OPEN = 1.0


@dataclass
class RobotPlan:
    """How much each recording captures, and how the arms move."""

    head_a: int = 24
    head_b: int = 14
    wrist_a: int = 11  # per arm per placement (two placements)
    wrist_b: int = 12  # per arm (one placement)
    speed: float = 0.4  # rad/s, the fastest joint
    hold_s: float = 0.7
    rate_hz: float = 250.0
    table_z: float = 0.0  # table top in root; 0: from the first board seen
    # Longest the head phase waits for enough views (s); the checks then say
    # what is missing.
    head_timeout_s: float = 300.0


class _Arms:
    """Moves the arms through joint targets with the waypoint planner."""

    def __init__(self, robot: Any, solver: Any, plan: RobotPlan) -> None:
        self.robot = robot
        self.solver = solver
        self.plan = plan
        self.q_last: np.ndarray | None = None

    def current(self) -> np.ndarray:
        q = np.zeros(self.solver.num_joints, dtype=np.float32)
        for side, idx in (
            ("left", self.solver.left_indices),
            ("right", self.solver.right_indices),
        ):
            arm = getattr(self.robot, side)
            if arm is not None:
                q[idx] = np.asarray(arm.positions, dtype=np.float32)[:7]
        return q

    def side_q(self, q: np.ndarray, side: str) -> np.ndarray:
        idx = self.solver.left_indices if side == "left" else self.solver.right_indices
        return np.asarray(q[idx], dtype=np.float64)

    async def _send(self, q: np.ndarray) -> None:
        left = np.append(q[self.solver.left_indices], _GRIP_OPEN).astype(np.float32)
        right = np.append(q[self.solver.right_indices], _GRIP_OPEN).astype(np.float32)
        await self.robot.motion_control(
            left=left if self.robot.left is not None else None,
            right=right if self.robot.right is not None else None,
        )

    async def go(
        self, targets: dict[str, np.ndarray], hold: float | None = None
    ) -> None:
        """Move the named arms to their 7 joint targets, then hold still."""
        from ..teleop.trajectory import plan_collision_aware_trajectory

        start = self.q_last if self.q_last is not None else self.current()
        goal = start.copy()
        for side, q in targets.items():
            idx = (
                self.solver.left_indices
                if side == "left"
                else self.solver.right_indices
            )
            goal[idx] = np.asarray(q, dtype=np.float32)
        trajectory = await asyncio.to_thread(
            plan_collision_aware_trajectory,
            self.solver,
            start,
            goal,
            speed=self.plan.speed,
            rate=self.plan.rate_hz,
            min_duration=0.8,
        )
        dt = 1.0 / self.plan.rate_hz
        hold_steps = round(
            (self.plan.hold_s if hold is None else hold) * self.plan.rate_hz
        )
        for q in [*trajectory, *([trajectory[-1]] * hold_steps)]:
            began = time.monotonic()
            self.q_last = np.asarray(q, dtype=np.float32)
            await self._send(self.q_last)
            spent = time.monotonic() - began
            if spent < dt:
                await asyncio.sleep(dt - spent)


class _HeadCamera:
    """The head camera's left eye: frames from the relay, its factory lens."""

    def __init__(self, reader: Any, lens: Lens, board: Board) -> None:
        self.reader = reader
        self.lens = lens
        self.board = board
        self.last_pixels = np.zeros((0, 2))

    def board_pose(self) -> np.ndarray | None:
        """The board in the camera, from the newest frame (``None``: not seen)."""
        import cv2

        try:
            frame = self.reader.read_latest(max_age_ms=1000)
        except Exception:  # noqa: BLE001 - no frame yet
            return None
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        detection = self.board.detect(gray)
        if not self.board.usable(detection):
            return None
        assert detection is not None
        self.last_pixels = detection.pixels
        return self.lens.board_pose(self.board.points[detection.ids], detection.pixels)


async def _ask(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


class RobotSession:
    """Records the two calibration recordings on the robot."""

    def __init__(
        self,
        *,
        robot: Any,
        solver: Any,
        kin: Any,
        tree: Any,
        board: Board,
        relay: Any,
        recorder: Any,
        head: _HeadCamera,
        wrist_lens: dict[str, Lens],
        plan: RobotPlan,
        work: Path,
        say: Callable[[str], None] = print,
    ) -> None:
        self.wrist_lens = wrist_lens
        self.robot = robot
        self.kin = kin
        self.tree = tree
        self.board = board
        self.relay = relay
        self.recorder = recorder
        self.head = head
        self.plan = plan
        self.work = work
        self.say = say
        self.arms = _Arms(robot, solver, plan)
        self.head_in_root = tree.fk("root", "head_camera_left_optical", {})
        self.camera_in_gripper = {
            side: tree.fk(f"{side}_gripper", f"{side}_wrist_camera_optical", {})
            for side in _SIDES
        }
        self.raised: dict[str, np.ndarray] = {}
        self.table_z = plan.table_z
        self.rng = np.random.default_rng(int(time.time()))

    def _raise(self, side: str, q_now: np.ndarray) -> np.ndarray:
        """The arm raised and drawn back, out of the head camera's way."""
        names = self.kin.joints[side]
        seed = dict(zip(names, q_now, strict=True))
        target = self.kin.tcp(side, seed)
        target[:3, 3] = (0.28, 0.34 if side == "left" else -0.34, 0.72)
        q, err = self.kin.solve(side, target, seed, iterations=150)
        return np.array([q[n] for n in names]) if err < 2e-3 else q_now

    async def _head_phase(self, wanted: int, phases: list[Phase]) -> None:
        lens = self.head.lens
        poses: list[np.ndarray] = []
        cov = Coverage(lens.width, lens.height)
        self.say(
            "Hold the board in front of the head camera, 40 to 80 cm away. Move it "
            "slowly to every part of the camera's view, near and far, and tilt it "
            "toward the camera from each side. Pause about a second at each spot."
        )
        await _ask("Press Enter when you are ready to start. ")
        start = time.time_ns()
        deadline = time.monotonic() + self.plan.head_timeout_s
        last_report = 0.0
        held: np.ndarray | None = None
        still_since = 0.0
        while True:
            pose = await asyncio.to_thread(self.head.board_pose)
            now = time.monotonic()
            if pose is not None:
                camera_in_board = np.linalg.inv(pose)
                moved = (
                    held is None
                    or float(np.linalg.norm(held[:3, 3] - pose[:3, 3])) > 0.004
                )
                if moved:
                    held, still_since = pose, now
                elif now - still_since > 0.4:
                    if len(distinct([*poses, camera_in_board])) > len(poses):
                        poses.append(camera_in_board)
                        cov.add(self.head.last_pixels)
            if now - last_report > 2.0:
                last_report = now
                self.say(
                    f"  head camera: {len(poses)} of {wanted} poses, "
                    f"{tilted_views(poses)} of 4 tilted views, {len(cov.cells)} of 12 "
                    f"image cells, {cov.quadrants} of 4 quadrants"
                )
            if (
                len(poses) >= wanted
                and len(cov.cells) >= 12
                and cov.quadrants == 4
                and tilted_views(poses) >= 4
            ):
                self.say("Enough head camera views.")
                break
            if now > deadline:
                self.say(
                    "Time is up for the head camera views; carrying on with "
                    "what there is. The checks will say if it isn't enough."
                )
                break
            await asyncio.sleep(0.1)
        phases.append(Phase("head", start, time.time_ns()))

    async def _locate(self) -> np.ndarray:
        while True:
            for _ in range(20):
                pose = await asyncio.to_thread(self.head.board_pose)
                if pose is not None:
                    board = self.head_in_root @ pose
                    if self.table_z == 0.0:
                        self.table_z = float(board[2, 3])
                    return board
                await asyncio.sleep(0.1)
            await _ask(
                "The head camera can't see the whole board. Move it into view on "
                "the table, then press Enter. "
            )

    async def _table_phase(self, number: int, count: int, phases: list[Phase]) -> None:
        await _ask(
            f"Lay the board flat on the table in front of the robot (spot {number}: "
            "a different place and angle from before), then press Enter. "
        )
        board = await self._locate()
        self.say(f"Found the board {board[0, 3]:.2f} m in front of the robot.")
        for side in _SIDES:
            if getattr(self.robot, side) is None:
                continue
            q_now = self.arms.side_q(self.arms.current(), side)
            names = self.kin.joints[side]
            found = await asyncio.to_thread(
                wrist_poses,
                self.kin,
                side,
                board_in_root=board,
                board_size=self.board.size_m,
                points=self.board.points,
                lens=self.wrist_lens[side],
                camera_in_gripper=self.camera_in_gripper[side],
                table_z=self.table_z,
                seed=dict(zip(names, q_now, strict=True)),
                count=count,
                rng=self.rng,
            )
            self.say(f"  {side} arm: {len(found)} wrist camera poses")
            start = time.time_ns()
            current = self.raised[side]
            for pose in found:
                target = np.array([pose.joints[n] for n in names])
                steps = route(
                    self.kin,
                    side,
                    current,
                    target,
                    table_z=self.table_z,
                    via=self.raised[side],
                )
                for i, step in enumerate(steps):
                    # Hold still only at the pose itself, where views are taken.
                    await self.arms.go(
                        {side: step}, hold=None if i == len(steps) - 1 else 0.0
                    )
                current = target
            await self.arms.go({side: self.raised[side]}, hold=0.2)
            phases.append(Phase("table", start, time.time_ns(), number, side))

    async def record(
        self, label: str, head_count: int, wrist_count: int, placements: int
    ) -> None:
        """One recording, saved by the recorder with its phases."""
        q = self.arms.current()
        for side in _SIDES:
            if getattr(self.robot, side) is not None:
                self.raised.setdefault(
                    side, self._raise(side, self.arms.side_q(q, side))
                )
        self.say(f"Recording {label}. Raising the arms.")
        await self.arms.go(dict(self.raised))
        phases: list[Phase] = []
        self.recorder.start_episode(f"Camera calibration, recording {label}")
        self.relay.set_raw_enabled(True)
        try:
            await self._head_phase(head_count, phases)
            for number in range(1, placements + 1):
                await self._table_phase(number, wrist_count, phases)
        finally:
            frames, error = self.recorder.stop_capture()
            self.relay.set_raw_enabled(False)
        if error:
            raise RuntimeError(f"Recording {label} failed: {error}")
        self.recorder.save_episode({"calibration_phases": phases_json(phases)})
        self.say(f"Recording {label} saved ({frames} frames per camera).")


def _lens(intrinsics: Any) -> Lens:
    return Lens.from_intrinsics(intrinsics)


async def run_robot_session(
    cfg: Any,
    board: Board,
    *,
    plan: RobotPlan | None = None,
    say: Callable[[str], None] = print,
) -> Any:
    """Record A and B on this robot and run the procedure on them.

    ``cfg`` is the calibration command's config (arms, kinematics, output
    paths, limits).
    """
    from ..cli.collect_data import (
        CollectDataConfig,
        _start_mcap_recorder,
        _start_video_relay,
    )
    from ..cli.config import parse
    from ..constants import AxolModel, urdf_arm_joint_names, urdf_path
    from ..kinematics.solver import KinematicsSolver
    from ..recording.mcap.scene import CameraSource
    from ..recording.mcap.urdf_tree import UrdfTree
    from ..robot import Axol
    from ..robot.identity import hub_serial
    from ..settings import resolve_gripper, resolve_robot_model
    from ..synthetic.motion import ArmKinematics
    from ..zed import stereo_serials
    from ..zed.calibration import SETTINGS_DIR
    from .procedure import run

    plan = plan or RobotPlan()
    # Production mode: Collect data's own cameras, resolution, rate and codec.
    collect = parse(
        CollectDataConfig,
        [
            "--repo_id",
            "calibration/session",
            "--task",
            "camera calibration",
            "--format",
            "mcap",
        ],
        settings_op="collect-data",
    )
    collect.mcap_topics = ""
    collect.mcap_frames = ""
    collect.operator = "calibration"
    collect.episode_metadata = "calibration_recording=true"
    collect.robot_config.apply_detected_stereo(stereo_serials())
    codec = collect.mcap_video_codec
    say(f"Image mode: {collect.dataset_resolution} at {collect.fps} fps, {codec}.")
    relay = _start_video_relay(
        collect,
        collect.dataset_resolution,
        raw_transport="gstshm+pyshm",
        policy_fps=5,
        dataset_codec=codec,
        record_imu=True,
    )
    if relay is None:
        raise RuntimeError(
            "The cameras didn't start. Check the camera settings, then run "
            "`axol gst.install` if the GStreamer pipeline is missing."
        )
    expected = set(collect.robot_config.observation_cameras().keys())
    model = resolve_robot_model()
    gripper = resolve_gripper(cfg.axol.gripper)
    urdf = urdf_path(
        model, gripper=gripper if model is AxolModel.MOBILE else "parallel"
    )
    tree = UrdfTree.load(urdf)
    kin = ArmKinematics(tree)
    sources = {
        src: CameraSource.from_meta(src, m)
        for src, m in relay.raw_meta.items()
        if src in expected
    }
    from ..recording.mcap.scene import Scene

    scene = Scene.build(
        urdf,
        has_gripper=True,
        jelly_wheels=False,
        jelly_lift=False,
        cameras=sources,
        calibration_dir=SETTINGS_DIR,
        gripper=gripper,
    )
    head_src = next(
        (
            s
            for s, c in scene.cameras.items()
            if c.camera == "overhead" and c.eye in (None, "left")
        ),
        None,
    )
    wrist_src = {side: f"{side}_arm" for side in _SIDES}
    missing = [
        s
        for s in (head_src, *wrist_src.values())
        if s is None or s not in scene.cameras or scene.cameras[s].intrinsics is None
    ]
    if missing:
        relay.shutdown()
        raise RuntimeError(
            "Calibration needs the head camera and both wrist cameras, each with its "
            f"factory calibration file. Missing: {', '.join(str(m) for m in missing)}."
        )
    robot = Axol(
        config=cfg.axol, left_channel=cfg.left_channel, right_channel=cfg.right_channel
    )
    work = (
        Path(cfg.packages).expanduser() / "recordings" / time.strftime("%Y%m%dT%H%M%S")
    )
    work.mkdir(parents=True, exist_ok=True)
    recorder = _start_mcap_recorder(
        collect,
        SimpleNamespace(axol=robot),
        SimpleNamespace(jelly=None),
        relay,
        expected,
        work,
        250,
    )
    solver = KinematicsSolver(cfg.kinematics)
    head = _HeadCamera(
        relay.raw_cameras[head_src], _lens(scene.cameras[head_src].intrinsics), board
    )
    try:
        async with robot:
            robot.reset_command_state()
            session = RobotSession(
                robot=robot,
                solver=solver,
                kin=kin,
                tree=tree,
                board=board,
                relay=relay,
                recorder=recorder,
                head=head,
                wrist_lens={
                    side: _lens(scene.cameras[src].intrinsics)
                    for side, src in wrist_src.items()
                },
                plan=plan,
                work=work,
                say=say,
            )
            await session.record("A", plan.head_a, plan.wrist_a, 2)
            await session.record("B", plan.head_b, plan.wrist_b, 1)
            say("Both recordings are done. Rest the arms, then press Enter.")
            await _ask("")
    finally:
        recorder.close()
        relay.shutdown()
    paths = sorted(work.glob("episode_*.mcap"))
    if len(paths) < 2:
        raise RuntimeError(f"Expected two recordings in {work}, found {len(paths)}.")
    arm_joints = {side: urdf_arm_joint_names(is_left=side == "left") for side in _SIDES}
    return run(
        paths[-2],
        paths[-1],
        board=board,
        tree=tree,
        arm_joints=arm_joints,
        robot_serial=hub_serial() or "",
        packages=Path(cfg.packages).expanduser(),
        calibration_path=Path(cfg.output).expanduser(),
        limits=cfg.checks,
        valid_days=cfg.valid_days,
        identity={
            "image_mode_setting": f"{collect.dataset_resolution} {collect.fps} fps {codec}"
        },
        progress=say,
    )
