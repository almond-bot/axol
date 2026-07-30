"""
axol waypoints

Teach-and-repeat. The arms sit in gravity compensation so you can hand-guide
them; each time you record, the pose both arms are in is appended to a
waypoint file. Play it back and the grippers travel a straight line in
Cartesian space from waypoint to waypoint, resolved to joint angles by the IK
solver, pausing at each one to work the grippers.

Two ways to drive it. On the terminal, single keys on stdin:

    [Enter]  record the current pose      p  play the path
    u        undo the last waypoint       s  stop playback
    c        clear every waypoint         q  quit
    g        cycle the grippers open / closed / limp

In the ``axol serve`` control panel the same actions are buttons — the running
session publishes them, so the panel needs no knowledge of this command.

The whole path is solved *before* the arms move, so a waypoint the straight
line cannot reach is reported while the robot is still standing still.

Examples:
    axol waypoints                              # teach, then play
    axol waypoints --file pick_place.json       # keep a named path
    axol waypoints --play_only --loops 0        # replay until stopped
    axol waypoints --sim --file pick_place.json # preview it in the browser
"""

from __future__ import annotations

import asyncio
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import ARM_JOINTS, CAN_LEFT, CAN_RIGHT
from ..kinematics.config import KinematicsConfig
from ..robot.base import RobotBase
from ..robot.config import AxolConfig
from ..teleop.config import VRTeleopConfig
from ..waypoints import Waypoint, WaypointSet
from .config import LogLevel, normalize_bool_flags, parse
from .gravity_comp import _resolve_free_joints

_logger = logging.getLogger(__name__)

# Index of the gripper within a waypoint's (8,) per-arm vector.
_GRIP = len(ARM_JOINTS)

# Joint distance (rad) below which a move is treated as "already there".
_AT_POSE_EPSILON = 0.02

# Per-arm gripper openings, normalised [0, 1].
Grip = tuple[float, float]

# One planned move: the joint vectors to stream, the gripper openings held
# while streaming them, and the openings to reach on arrival.
Leg = tuple[list[np.ndarray], Grip, Grip]


def _default_file() -> str:
    return str(Path.home() / ".almond" / "waypoints.json")


@dataclass
class WaypointsCmdConfig:
    """Config for ``axol waypoints``.

    Waypoints are written to ``file`` as they are recorded, so a taught path
    outlives the session and can be replayed (or previewed with ``--sim``)
    later. ``--play_only`` skips teaching and replays what is already there.

    Playback speed is Cartesian: ``speed`` is how fast the gripper travels
    along the straight line between waypoints and ``ang_speed`` how fast it
    reorients, whichever is slower setting the pace. At each waypoint the
    grippers move to their recorded opening over ``grip_time`` and the arms
    then hold still for ``dwell``.

    The teaching side takes the same knobs as ``axol gravity-comp``
    (``free_joints``, ``kd``); per-joint gains and stiffness come from the
    nested ``axol`` config — e.g. ``--axol.left_stiffness 0.8``. IK cost
    weights live on ``kinematics`` — e.g. ``--kinematics.max_reach 0.7``.
    """

    axol: AxolConfig = field(default_factory=AxolConfig)
    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)
    left_channel: str | None = CAN_LEFT
    right_channel: str | None = CAN_RIGHT
    file: str = field(default_factory=_default_file)
    """Waypoint file to record into and play back from (JSON)."""
    speed: float = 0.08
    """Cartesian speed (m/s) of the gripper along each straight-line leg."""
    ang_speed: float = 0.6
    """Angular speed (rad/s) of the gripper's reorientation along a leg."""
    dwell: float = 0.5
    """Seconds to hold still at each waypoint after its grippers have moved."""
    grip_time: float = 0.75
    """Seconds spent moving the grippers to a waypoint's recorded opening."""
    loops: int = 1
    """Times to run the path. 0 replays it until stopped."""
    pos_tolerance: float = 0.01
    """Largest tolerated deviation (m) from the straight line while planning."""
    free_joints: list[str] | None = None
    """Arm joints to gravity-compensate while teaching; null frees all seven."""
    kd: float = 0.25
    rate_hz: float = 250.0
    telemetry_hz: float = 500.0
    play_only: bool = False
    """Skip teaching and replay ``file`` straight away. Implied by ``sim``."""
    sim: bool = False
    """Play the path in the browser visualizer instead of on the robot."""
    log_level: LogLevel = "INFO"


# ----------------------------------------------------------------------
# Session control: abstracts how record / play / stop decisions arrive.
#
# The CLI reads them as single keys on stdin; the web control panel pushes
# them through a queue from the API and renders the buttons the session
# publishes. The session only calls the small surface below, so the two are
# interchangeable (mirrors run-policy's episode control).
# ----------------------------------------------------------------------

# Single-key stdin shortcuts. Enter (an empty line) records, the action an
# operator repeats most while teaching.
_KEYS: dict[str, str] = {
    "": "record",
    "r": "record",
    "u": "undo",
    "c": "clear",
    "g": "grip",
    "p": "play",
    "s": "stop",
    "q": "quit",
}

# Cycling the grippers while teaching: limp (held wherever they are) -> open
# -> closed, so an object can be grasped and then let go of entirely.
_GRIP_CYCLE: dict[float | None, float | None] = {None: 1.0, 1.0: 0.0, 0.0: None}
_GRIP_LABELS: dict[float | None, str] = {
    None: "Open grippers",
    1.0: "Close grippers",
    0.0: "Release grippers",
}


class _StdinWaypointControl:
    """Terminal session control: single keys typed on stdin."""

    def __init__(self) -> None:
        self._q: queue.Queue[str] = queue.Queue()
        self._closed = threading.Event()
        self._last_message = ""
        # Without a terminal there is nobody to read keys from, and treating
        # the immediate EOF as a quit would cut a scripted playback short.
        if sys.stdin is not None and sys.stdin.isatty():
            threading.Thread(
                target=self._read_stdin, name="axol-waypoint-stdin", daemon=True
            ).start()

    def _read_stdin(self) -> None:
        while not self._closed.is_set():
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                self._q.put("quit")
                return
            command = _KEYS.get(line.strip().lower())
            if command is not None:
                self._q.put(command)

    def poll(self) -> str | None:
        """Return the next pending command, or ``None``."""
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def set_state(
        self, phase: str, message: str, controls: list[dict[str, str]], count: int
    ) -> None:
        """Report session state; the terminal prints only what changed."""
        del phase, controls, count
        if message and message != self._last_message:
            print(message)
            self._last_message = message

    def close(self) -> None:
        self._closed.set()


class _QueueWaypointControl:
    """Web session control: commands arrive from ``/api/op/episode``.

    :meth:`snapshot` is what the control panel renders — the phase, a status
    line, and the buttons that make sense right now — so the panel drives this
    command without knowing anything about it.
    """

    def __init__(self, stop_event: threading.Event) -> None:
        self._q: queue.Queue[str] = queue.Queue()
        self._stop = stop_event
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "phase": "preparing",
            "message": "Starting…",
            "controls": [],
            "episodesRecorded": 0,
        }

    def push(self, command: str) -> None:
        self._q.put(command)

    def poll(self) -> str | None:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def set_state(
        self, phase: str, message: str, controls: list[dict[str, str]], count: int
    ) -> None:
        with self._lock:
            self._state = {
                "phase": phase,
                "message": message,
                "controls": controls,
                "episodesRecorded": count,
            }

    def snapshot(self) -> dict[str, Any]:
        """Thread-safe session state for the ``/api/op/status`` API."""
        with self._lock:
            return dict(self._state)

    def close(self) -> None:
        pass


Control = _StdinWaypointControl | _QueueWaypointControl


class _SolverHandle:
    """Builds the IK solver on a background thread.

    The pyroki/JAX warmup takes ten-odd seconds, which would otherwise all
    land on the first Play. Starting it up front means it finishes while the
    operator is still hand-guiding the arms.
    """

    def __init__(self, config: KinematicsConfig) -> None:
        self._config = config
        self._solver: Any = None
        self._error: BaseException | None = None
        self._ready = threading.Event()
        threading.Thread(
            target=self._build, name="axol-waypoint-solver", daemon=True
        ).start()

    def _build(self) -> None:
        try:
            from ..kinematics.solver import KinematicsSolver

            self._solver = KinematicsSolver(self._config)
        except BaseException as exc:  # noqa: BLE001 - re-raised from get()
            self._error = exc
        finally:
            self._ready.set()

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    def get(self) -> Any:
        """Block until the solver is built, then return it."""
        self._ready.wait()
        if self._error is not None:
            raise self._error
        return self._solver


# ----------------------------------------------------------------------
# Joint-vector marshalling between the solver and motion_control
# ----------------------------------------------------------------------


def _rest_q(solver: Any) -> np.ndarray:
    """The full-N solver vector for the configured rest pose."""
    rest_cfg = VRTeleopConfig()
    q = np.zeros(solver.num_joints, dtype=np.float32)
    q[solver.left_indices] = rest_cfg.rest_pose_left
    q[solver.right_indices] = rest_cfg.rest_pose_right
    return q


def _waypoint_q(solver: Any, waypoint: Waypoint) -> np.ndarray:
    """Pack a waypoint's arm joints into a full-N solver vector."""
    q = np.zeros(solver.num_joints, dtype=np.float32)
    q[solver.left_indices] = waypoint.left[:_GRIP]
    q[solver.right_indices] = waypoint.right[:_GRIP]
    return q


def _arm_command(
    q_full: np.ndarray, solver: Any, grip: Grip
) -> tuple[np.ndarray, np.ndarray]:
    """Split a full-N solver vector into per-arm ``(8,)`` motion commands."""
    left = np.zeros(_GRIP + 1, dtype=np.float32)
    right = np.zeros(_GRIP + 1, dtype=np.float32)
    left[:_GRIP] = q_full[solver.left_indices]
    right[:_GRIP] = q_full[solver.right_indices]
    left[_GRIP] = grip[0]
    right[_GRIP] = grip[1]
    return left, right


def _blend(a: Grip, b: Grip, alpha: float) -> Grip:
    return (a[0] + (b[0] - a[0]) * alpha, a[1] + (b[1] - a[1]) * alpha)


# ----------------------------------------------------------------------
# Session
# ----------------------------------------------------------------------


class _Session:
    """One ``axol waypoints`` run: teach, play, park, repeat."""

    def __init__(
        self,
        cfg: WaypointsCmdConfig,
        robot: RobotBase,
        control: Control,
        stop_event: threading.Event,
    ) -> None:
        self._cfg = cfg
        self._robot = robot
        self._control = control
        self._stop = stop_event
        self._sim = cfg.sim
        self._store = WaypointSet.load(cfg.file)
        self._solver_handle = _SolverHandle(cfg.kinematics)
        self._free_joints = _resolve_free_joints(cfg.free_joints)
        # None leaves the grippers where they are; a value drives them there
        # so the operator can grasp an object before recording the waypoint.
        self._grip_target: float | None = None
        # True while the arms are held by motion_control rather than gravity
        # comp, so teardown knows whether parking at rest is safe.
        self._under_position_control = False
        self._quit = False
        # Pose the playback starts from, sampled just before planning.
        self._q_start: np.ndarray | None = None
        self._grip_start: Grip = (0.0, 0.0)
        # Why the session gave up, kept as the closing status message.
        self._failure: str | None = None

    @property
    def store(self) -> WaypointSet:
        return self._store

    @property
    def failure(self) -> str | None:
        return self._failure

    @property
    def _play_only(self) -> bool:
        """Sim has no arms to hand-guide, so it can only replay a saved path."""
        return self._cfg.play_only or self._sim

    # -- robot helpers ---------------------------------------------------

    async def _positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Current ``(8,)`` pose of each arm, rest-posed for an absent arm."""
        rest = VRTeleopConfig()
        left = np.append(rest.rest_pose_left, 0.0).astype(np.float32)
        right = np.append(rest.rest_pose_right, 0.0).astype(np.float32)
        if self._sim:
            sim_l, sim_r = await self._robot.get_positions()
            return (
                left if sim_l is None else np.asarray(sim_l, dtype=np.float32),
                right if sim_r is None else np.asarray(sim_r, dtype=np.float32),
            )
        # On hardware the telemetry cache is the only safe read: polling the
        # bus directly is rejected while background telemetry is running.
        if self._robot.left is not None:
            left = self._robot.left.positions
        if self._robot.right is not None:
            right = self._robot.right.positions
        return left, right

    async def _current_q(self, solver: Any) -> np.ndarray:
        left, right = await self._positions()
        q = np.zeros(solver.num_joints, dtype=np.float32)
        q[solver.left_indices] = left[:_GRIP]
        q[solver.right_indices] = right[:_GRIP]
        return q

    async def _current_grip(self) -> Grip:
        left, right = await self._positions()
        return float(left[_GRIP]), float(right[_GRIP])

    async def _send(self, left: np.ndarray, right: np.ndarray) -> None:
        if self._sim:
            await self._robot.motion_control(left=left, right=right)
            return
        await self._robot.motion_control(
            left=left if self._robot.left is not None else None,
            right=right if self._robot.right is not None else None,
        )

    def _interrupted(self) -> bool:
        """True if the operator asked to stop; latches a quit for :meth:`run`."""
        if self._stop.is_set():
            return True
        command = self._control.poll()
        if command == "quit":
            self._quit = True
            return True
        return command == "stop"

    # -- published state -------------------------------------------------

    def _publish_teaching(self, message: str | None = None) -> None:
        count = len(self._store)
        controls: list[dict[str, str]] = [
            {"command": "record", "label": "Record waypoint"}
        ]
        if count >= 2:
            controls.append({"command": "play", "label": f"Play {count} waypoints"})
        controls.append({"command": "grip", "label": _GRIP_LABELS[self._grip_target]})
        if count:
            controls.append({"command": "undo", "label": "Undo last"})
            controls.append({"command": "clear", "label": "Clear all"})
        if message is None:
            if count == 0:
                message = "Teaching — hand-guide the arms and record a waypoint."
            elif count == 1:
                message = "Teaching — 1 waypoint. Record at least one more to play."
            else:
                message = f"Teaching — {count} waypoints. Play when you are ready."
        self._control.set_state("teaching", message, controls, count)

    def _publish(self, phase: str, message: str, stoppable: bool = False) -> None:
        controls = [{"command": "stop", "label": "Stop playback"}] if stoppable else []
        self._control.set_state(phase, message, controls, len(self._store))

    # -- teaching --------------------------------------------------------

    async def teach(self) -> None:
        """Hold the arms in gravity comp until the operator plays or quits."""
        self._robot.reset_gravity_hold()
        self._under_position_control = False
        self._publish_teaching()
        dt = 1.0 / self._cfg.rate_hz
        while not self._stop.is_set():
            loop_start = time.monotonic()
            command = self._control.poll()
            if command == "quit":
                self._quit = True
                return
            if command == "play":
                if len(self._store) >= 2:
                    return
                self._publish_teaching("Record at least two waypoints before playing.")
            elif command == "record":
                await self._record()
            elif command == "undo":
                self._store.pop()
                self._store.save(self._cfg.file)
                self._publish_teaching()
            elif command == "clear":
                self._store.clear()
                self._store.save(self._cfg.file)
                self._publish_teaching()
            elif command == "grip":
                self._grip_target = _GRIP_CYCLE[self._grip_target]
                self._publish_teaching()

            await self._robot.gravity_compensate(
                kd=self._cfg.kd,
                free_joints=self._free_joints,
                gripper_target=self._grip_target,
            )
            spent = time.monotonic() - loop_start
            if spent < dt:
                await asyncio.sleep(dt - spent)

    async def _record(self) -> None:
        left, right = await self._positions()
        self._store.append(Waypoint(left=left.copy(), right=right.copy()))
        self._store.save(self._cfg.file)
        _logger.info("Recorded waypoint %d to %s", len(self._store), self._cfg.file)
        self._publish_teaching()

    # -- playback --------------------------------------------------------

    async def play(self) -> None:
        """Plan the whole path, then run it."""
        from ..kinematics.path import PathPlanningError

        if not self._solver_handle.is_ready:
            self._publish("planning", "Warming up the IK solver…")
        solver = await asyncio.to_thread(self._solver_handle.get)

        self._q_start = await self._current_q(solver)
        self._grip_start = await self._current_grip()
        self._publish(
            "planning", f"Planning a path through {len(self._store)} waypoints…"
        )
        try:
            legs = await asyncio.to_thread(self._plan, solver)
        except PathPlanningError as exc:
            _logger.error("%s", exc)
            self._failure = f"Cannot play this path. {exc}"
            if self._play_only:
                self._publish("failed", self._failure)
            else:
                self._publish_teaching(self._failure)
            return

        if not self._sim:
            # The arms were hand-guided, so the cached command history no
            # longer matches where they are; without this the first commanded
            # pose looks like a jump and trips the max-step safety check.
            self._robot.reset_command_state()
        self._under_position_control = True

        loops = self._cfg.loops
        run = 0
        while not self._stop.is_set() and (loops == 0 or run < loops):
            run += 1
            # Only the first run needs the approach: a looping path ends its
            # closing leg back at waypoint 1, already where the next run
            # starts. Replaying the approach would throw the arms back to
            # wherever playback began.
            offset = 0 if run == 1 else 1
            for index, leg in enumerate(legs[offset:], start=offset):
                trajectory, held, arrival = leg
                # Leg 0 is the approach, so leg n arrives at waypoint n + 1;
                # the closing leg comes back around to waypoint 1.
                where = f"waypoint {(index % len(self._store)) + 1}"
                suffix = "" if loops == 1 else f" (run {run})"
                self._publish("playing", f"Moving to {where}{suffix}", stoppable=True)
                if not await self._stream(solver, trajectory, held):
                    return
                if not await self._settle(solver, trajectory[-1], held, arrival):
                    return

    def _plan(self, solver: Any) -> list[Leg]:
        """Plan every leg of the path. Runs off the event loop (JAX blocks)."""
        from ..kinematics.path import plan_linear_segment
        from ..teleop.trajectory import plan_collision_aware_trajectory

        cfg = self._cfg
        rest_cfg = VRTeleopConfig()
        waypoints = list(self._store)
        q_waypoints = [_waypoint_q(solver, wp) for wp in waypoints]
        grips: list[Grip] = [
            (float(wp.left[_GRIP]), float(wp.right[_GRIP])) for wp in waypoints
        ]

        # The arms start wherever they were left (hand-guided, or at rest in
        # sim), which is not on the straight line, so the approach is a
        # joint-space move that keeps clear of the torso.
        legs: list[Leg] = [
            (
                plan_collision_aware_trajectory(
                    solver.robot,
                    solver.robot_coll,
                    self._q_start,
                    q_waypoints[0],
                    speed=rest_cfg.reset_speed,
                    rate=cfg.rate_hz,
                    min_duration=rest_cfg.reset_min_duration,
                ),
                self._grip_start,
                grips[0],
            )
        ]

        pairs = [(i, i + 1) for i in range(len(waypoints) - 1)]
        if cfg.loops != 1:
            # Close the cycle so a repeating run flows back to the first
            # waypoint the same way it moves anywhere else.
            pairs.append((len(waypoints) - 1, 0))
        for i, j in pairs:
            legs.append(
                (
                    plan_linear_segment(
                        solver,
                        q_waypoints[i],
                        q_waypoints[j],
                        speed=cfg.speed,
                        ang_speed=cfg.ang_speed,
                        rate=cfg.rate_hz,
                        pos_tolerance=cfg.pos_tolerance,
                        label=f"waypoint {i + 1} → {j + 1}",
                    ),
                    grips[i],
                    grips[j],
                )
            )
        return legs

    async def _stream(
        self, solver: Any, trajectory: list[np.ndarray], grip: Grip
    ) -> bool:
        """Send a planned leg at the control rate. False if it was stopped."""
        dt = 1.0 / self._cfg.rate_hz
        for q in trajectory:
            if self._interrupted():
                return False
            loop_start = time.monotonic()
            left, right = _arm_command(q, solver, grip)
            await self._send(left, right)
            spent = time.monotonic() - loop_start
            if spent < dt:
                await asyncio.sleep(dt - spent)
        return True

    async def _settle(
        self, solver: Any, q: np.ndarray, held: Grip, arrival: Grip
    ) -> bool:
        """Work the grippers to the waypoint's opening, then dwell.

        The grippers move while the arms are stationary rather than during the
        leg, so a grasp closes on the object at the waypoint instead of
        somewhere along the way.
        """
        dt = 1.0 / self._cfg.rate_hz
        steps = max(1, round(self._cfg.grip_time * self._cfg.rate_hz))
        hold = max(0, round(self._cfg.dwell * self._cfg.rate_hz))
        for step in range(steps + hold):
            if self._interrupted():
                return False
            loop_start = time.monotonic()
            alpha = min(1.0, (step + 1) / steps)
            left, right = _arm_command(q, solver, _blend(held, arrival, alpha))
            await self._send(left, right)
            spent = time.monotonic() - loop_start
            if spent < dt:
                await asyncio.sleep(dt - spent)
        return True

    # -- parking ---------------------------------------------------------

    async def park(self) -> None:
        """Move back to the rest pose. Only safe while position-controlled.

        Quitting out of gravity comp deliberately skips this: the operator has
        their hands on a limp arm, and stiffening it to drive home would be a
        surprise. That matches how ``axol gravity-comp`` exits.
        """
        if not self._under_position_control or not self._solver_handle.is_ready:
            return
        from ..teleop.trajectory import plan_collision_aware_trajectory

        solver = self._solver_handle.get()
        rest_cfg = VRTeleopConfig()
        q_now = await self._current_q(solver)
        q_rest = _rest_q(solver)
        self._under_position_control = False
        if float(np.max(np.abs(q_now - q_rest))) <= _AT_POSE_EPSILON:
            return
        self._publish("returning", "Returning to the rest pose…")
        trajectory = await asyncio.to_thread(
            plan_collision_aware_trajectory,
            solver.robot,
            solver.robot_coll,
            q_now,
            q_rest,
            speed=rest_cfg.reset_speed,
            rate=self._cfg.rate_hz,
            min_duration=rest_cfg.reset_min_duration,
        )
        grip = await self._current_grip()
        dt = 1.0 / self._cfg.rate_hz
        for q in trajectory:
            loop_start = time.monotonic()
            left, right = _arm_command(q, solver, grip)
            await self._send(left, right)
            spent = time.monotonic() - loop_start
            if spent < dt:
                await asyncio.sleep(dt - spent)

    # -- main loop -------------------------------------------------------

    async def run(self) -> None:
        """Alternate teaching and playback until the operator quits."""
        while not self._stop.is_set():
            if not self._play_only:
                await self.teach()
                if self._quit or self._stop.is_set():
                    return
            await self.play()
            await self.park()
            if self._play_only or self._quit:
                return


# ----------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------


def main(argv: list[str]) -> None:
    """Parse the CLI config and run a teach-and-repeat session."""
    cfg = parse(WaypointsCmdConfig, normalize_bool_flags(argv, "sim", "play_only"))
    # force=True: a dependency imported before this point may install a root
    # handler (leaving the level at WARNING), which would make this a no-op.
    logging.basicConfig(level=getattr(logging, cfg.log_level), force=True)
    try:
        _run(cfg)
    except ValueError as exc:
        _logger.error("%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nExiting waypoints ...")


def _run(
    cfg: WaypointsCmdConfig,
    stop_event: threading.Event | None = None,
    control: Control | None = None,
) -> None:
    """Run a session, driven from stdin or from the control panel's queue."""
    if stop_event is None:
        stop_event = threading.Event()
    if not cfg.sim and cfg.left_channel is None and cfg.right_channel is None:
        raise ValueError("Both arms disabled — nothing to do.")
    if (cfg.play_only or cfg.sim) and len(WaypointSet.load(cfg.file)) < 2:
        raise ValueError(
            f"{cfg.file} holds fewer than two waypoints, so there is nothing to "
            "play. Teach a path first (run without --play_only / --sim)."
        )

    owns_control = control is None
    if control is None:
        control = _StdinWaypointControl()
        if not (cfg.play_only or cfg.sim):
            print(
                "Waypoint teach-and-repeat. Hand-guide the arms, then:\n"
                "  [Enter] record   u undo    c clear   g cycle grippers\n"
                "  p play           s stop    q quit"
            )
    try:
        asyncio.run(_session(cfg, stop_event, control))
    finally:
        if owns_control:
            control.close()


async def _session(
    cfg: WaypointsCmdConfig, stop_event: threading.Event, control: Control
) -> None:
    if cfg.sim:
        from ..robot.sim import Sim

        robot: RobotBase = Sim()
    else:
        from ..robot import Axol

        robot = Axol(
            config=cfg.axol,
            left_channel=cfg.left_channel,
            right_channel=cfg.right_channel,
        )

    async with robot:
        if not cfg.sim:
            await robot.start_telemetry(cfg.telemetry_hz)
            # Motors may still be rebooting from set_control_mode(); block
            # until every one has answered a poll before driving them.
            await robot.wait_for_telemetry()

        session = _Session(cfg, robot, control, stop_event)
        try:
            await session.run()
        except (KeyboardInterrupt, asyncio.CancelledError):
            _logger.info("Interrupted — parking the arms before shutting down.")
        finally:
            # Python 3.11+ asyncio.run cancels the task on SIGINT, so without
            # uncancel every cleanup await below would re-raise CancelledError
            # immediately and the arms would skip their return to rest.
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
            stop_event.set()
            try:
                await session.park()
            except Exception:  # noqa: BLE001 - still tear the robot down
                _logger.warning("return-to-rest during teardown failed", exc_info=True)
            control.set_state(
                "done",
                session.failure or "Session finished.",
                [],
                len(session.store),
            )
