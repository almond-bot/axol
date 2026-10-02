"""
Shared rollout machinery for policy CLIs.

Pulled out of ``axol run-policy`` so other policy-running CLIs can reuse
the same episode plumbing without duplicating it:

- :class:`IKResetController` — guarded return-to-rest backed by an
  out-of-process worker using the selected kinematics backend.
- :class:`ActionPublisher` — single-slot thread-safe handoff of the most
  recently executed action.
- :class:`RolloutCaptureThread` — fixed-rate thread that pairs a
  timestamp-aligned observation with the latest published action and
  appends it to a ``LeRobotDataset``.
- :class:`PolicyActionLimiter` — per-joint velocity/acceleration envelope
  over policy actions, for control loops that command a policy's raw
  output directly (``collect-dagger``).
- :func:`latest_observation` — compatibility wrapper for the robot's
  capture-instant-aligned observation API.
- :func:`stdin_watcher` — ``s`` / ``r`` / ``q`` keystroke watcher with
  no-block ``select`` polling.
- :func:`arms_reporting` — whether the arms still report a pose, the
  liveness check every flow's teardown return-to-rest starts from.

All four are LeRobot-flavoured: the capture thread depends on
``lerobot.datasets.lerobot_dataset.LeRobotDataset``, ``build_dataset_frame``,
and ``log_rerun_data``; the reset controller talks to the selected IK worker via
``almond_axol.teleop``. The module lives under ``almond_axol/lerobot``
alongside the other LeRobot adapters.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable

from ..constants import ARM_JOINTS
from ..robot.base import HardwareCleanupError, mark_hardware_cleanup_uncertain

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.lerobot_types import RobotAction

    from .robot.robot_axol import AxolRobot
    from ..kinematics.config import KinematicsConfig
    from ..teleop.config import VRTeleopConfig

_logger = logging.getLogger(__name__)


def arms_reporting(robot: "AxolRobot") -> bool:
    """True while the arms still report a pose to plan a move from.

    The teardown return-to-rest each flow plays before it torques the motors
    off asks this first: a closed or stalled bus leaves the position cache
    unreadable, and a move planned from nothing is worse than no move at all.
    """
    try:
        robot.positions
    except BaseException:
        return False
    return True


class IKResetController:
    """Collision-aware return-to-rest, backed by an IK worker subprocess.

    Mirrors the reset path used by ``AxolVRTeleop`` (collect-data) but
    without the VR server. ``start()`` spawns ``run_ik_worker`` using the
    selected backend (Mink requires no JAX import or compilation);
    ``wait_ready()`` blocks on the handshake;
    ``return_to_rest()`` plans Cartesian paths for Mink, or joint-space paths
    for JAX, and streams the resolved joint waypoints to the controller.
    Spawn before ``client.start()`` to overlap preparation with policy load.

    ``rest_pose_left`` and ``rest_pose_right`` optionally select each arm's
    seven joint angles in radians, in ``ARM_JOINTS`` order. An omitted arm
    keeps the stock teleop rest pose. Gripper positions are held separately
    during the reset.
    """

    def __init__(
        self,
        *,
        rest_pose_left: Sequence[float] | None = None,
        rest_pose_right: Sequence[float] | None = None,
        kinematics_config: KinematicsConfig | None = None,
        vr_teleop_config: VRTeleopConfig | None = None,
    ) -> None:
        import numpy as np

        from ..kinematics.config import KinematicsConfig
        from ..teleop.config import VRTeleopConfig

        overrides = {}
        for name, value in (
            ("rest_pose_left", rest_pose_left),
            ("rest_pose_right", rest_pose_right),
        ):
            if value is None:
                continue
            error = f"{name} must contain exactly seven finite joint angles in radians"
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    pose = np.array(value, dtype=np.float32, copy=True)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(error) from exc
            if pose.shape != (len(ARM_JOINTS),) or not np.isfinite(pose).all():
                raise ValueError(error)
            overrides[name] = pose
        # The policy/operator controller and every lifecycle move must use
        # the same backend. A Mink session must never spawn a JAX worker.
        from dataclasses import replace

        self._vr_cfg = (
            VRTeleopConfig(**overrides)
            if vr_teleop_config is None
            else replace(vr_teleop_config, **overrides)
        )
        self._kin_cfg = (
            KinematicsConfig()
            if kinematics_config is None
            else replace(kinematics_config)
        )
        if self._kin_cfg.backend not in {"jax", "mink"}:
            raise ValueError("kinematics backend must be 'jax' or 'mink'")
        self._proc: Any | None = None
        self._conn: Any | None = None
        self._q_init: Any | None = None
        self._left_indices: list[int] | None = None
        self._right_indices: list[int] | None = None
        self._ready = False
        self._arms_limp = False

    @property
    def arms_limp(self) -> bool:
        """True while the arms were last left limp in a gravity-comp hold.

        Set by every hold this controller streams and cleared by the next
        play, so a caller winding down can tell whether the arms are its to
        move or already in the operator's hands.
        """
        return self._arms_limp

    def start(self) -> None:
        """Spawn the IK worker subprocess. Non-blocking; pair with ``wait_ready``."""
        import multiprocessing as mp

        from ..teleop.worker import run_ik_worker

        if self._proc is not None or self._conn is not None:
            raise RuntimeError(
                "IK reset controller already owns startup resources; stop it "
                "before starting another worker"
            )
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self._conn = parent_conn
        try:
            proc = ctx.Process(
                target=run_ik_worker,
                args=(child_conn, self._vr_cfg, self._kin_cfg, None, None),
                name="axol-ik-worker",
                daemon=True,
            )
            # Retain before start so a successful spawn followed by any local
            # setup failure remains reachable by stop()'s terminate/kill path.
            self._proc = proc
            proc.start()
            child_conn.close()
        except BaseException as setup_error:
            try:
                child_conn.close()
            except BaseException as close_error:
                setup_error.add_note(
                    "additional IK reset child-pipe close failure: "
                    f"{type(close_error).__name__}: {close_error}"
                )
            try:
                self.stop()
            except BaseException as cleanup_error:
                mark_hardware_cleanup_uncertain(setup_error, cleanup_error)
            raise

    def wait_ready(
        self,
        timeout: float | None = None,
        stopped: Callable[[], bool] | None = None,
    ) -> bool:
        """Block until the IK worker has finished its startup.

        The same wait teleop and collect-data use
        (:func:`~almond_axol.teleop.core.wait_for_ik_ready`). It used to be a
        fixed 60 s here, which failed replay-dataset and run-policy on an
        Orin NX (startup runs past a minute) while teleop came up fine.

        Returns True once ready, False if ``stopped`` fired first.
        """
        from ..teleop.core import IK_READY_TIMEOUT_S, wait_for_ik_ready

        if self._ready:
            return True
        if self._conn is None:
            raise RuntimeError("IK reset controller not started")
        msg = wait_for_ik_ready(
            self._conn,
            self._proc,
            timeout=IK_READY_TIMEOUT_S if timeout is None else timeout,
            stopped=stopped,
        )
        if msg is None:
            return False
        import numpy as np

        _, q_init, left_indices, right_indices, _startup_traj = msg
        self._q_init = np.asarray(q_init, dtype=np.float32)
        self._left_indices = [int(i) for i in left_indices]
        self._right_indices = [int(i) for i in right_indices]
        self._ready = True
        return True

    def return_to_rest(
        self,
        robot: "AxolRobot",
        *,
        torque_threshold: float = 6.0,
        gravity_comp_kd: float = 0.25,
        wait_retry: Callable[[], bool] | None = None,
        stopped: Callable[[], bool] | None = None,
        on_contact: Callable[[], None] | None = None,
    ) -> bool:
        """Plan and play a guarded collision-aware trajectory to the rest pose.

        The move plays at the normal session gains — accurate tracking of
        the collision-checked path — while a torque residual sustained above
        ``torque_threshold`` (see
        :class:`~almond_axol.robot.control.ContactWatchdog`) means it hit
        something — a gripper still hooked on the scene, an operator
        grabbing an arm — so the move stops where it is and the arms drop
        into a limp gravity-comp hold instead of pulling through. What ends
        the hold depends on the caller:

        - ``wait_retry`` set (run-policy): it runs in a helper thread while
          the hold streams; return ``True`` to replan from wherever the
          arms were left and try again, ``False`` to abort.
        - ``wait_retry`` unset (replay): the hold streams until ``stopped``
          fires (or Ctrl+C), then aborts — there is no interactive channel
          to retry from.

        Args:
            robot: Connected robot to drive.
            torque_threshold: Contact watchdog threshold (Nm); ``0``
                disables it (the move always plays through).
            gravity_comp_kd: Velocity damping for the hold's free joints.
            wait_retry: Blocking operator gate; ``True`` = retry.
            stopped: Flow shutdown flag, polled during play and hold.
            on_contact: Announce hook, run once per trip before the hold.

        Returns:
            ``True`` once the arms reached rest; ``False`` if aborted
            (stopped, or the operator declined the retry).
        """
        if not self.wait_ready(stopped=stopped):
            return False
        while True:
            outcome = self._play_to_rest(robot, torque_threshold, stopped)
            if outcome != "contact":
                return outcome == "done"
            if on_contact is not None:
                on_contact()
            if not self._hold_limp(robot, gravity_comp_kd, wait_retry, stopped):
                return False
            # The arms were hand-guided during the hold: clear the stale
            # command history so the max-step safety check doesn't reject
            # the first command of the replanned move.
            robot.reset_command_state()

    def hold_limp(
        self,
        robot: "AxolRobot",
        *,
        gravity_comp_kd: float = 0.25,
        wait: Callable[[], bool] | None = None,
        stopped: Callable[[], bool] | None = None,
    ) -> bool:
        """Hold the arms limp (gravity comp) until the operator resolves ``wait``.

        Used by run-policy's discard flow: after a failed episode the operator
        usually needs to untangle the grippers from the scene or reposition
        the arms by hand before any planned move is safe, so the arms drop
        into a free gravity-supported hold instead of pulling straight back
        to rest. ``wait`` blocks in a helper thread while the hold streams
        (same mechanics as the contact hold inside :meth:`return_to_rest`).

        Needs no IK worker — only gravity-comp cycles — so it never blocks on
        :meth:`wait_ready`.

        Args:
            robot: Connected robot to hold.
            gravity_comp_kd: Velocity damping for the free joints (Nm·s/rad).
            wait: Blocking operator gate; ``True`` = proceed.
            stopped: Flow shutdown flag, polled while the hold streams.

        Returns:
            ``True`` when the operator asked to proceed — with the command
            history cleared so the next planned move isn't rejected by the
            max-step safety check; ``False`` when aborted.
        """
        if not self._hold_limp(robot, gravity_comp_kd, wait, stopped):
            return False
        robot.reset_command_state()
        return True

    def park(
        self,
        robot: "AxolRobot",
        *,
        torque_threshold: float = 6.0,
        stopped: Callable[[], bool] | None = None,
        deadline_s: float = 30.0,
    ) -> bool:
        """Park through rest and zero, holding the grasp until torque-off.

        The caller must first join every policy/control worker. One deadline
        bounds both collision-aware legs, including planning and measured
        settling. Contact, missing telemetry, interruption or a rejected move
        ends the attempt; the caller must preserve torque on every failure.
        There is no automatic contact retry or gravity-comp transition here.
        """
        import numpy as np

        if not self._ready or self._proc is None or not self._proc.is_alive():
            raise RuntimeError("Soft park requires the already-ready IK reset worker")
        assert self._q_init is not None
        assert self._left_indices is not None
        assert self._right_indices is not None
        deadline = time.perf_counter() + deadline_s
        hold = robot.parking_gripper_hold()
        zero = np.asarray(self._q_init, dtype=np.float32).copy()
        zero[self._left_indices] = 0.0
        zero[self._right_indices] = 0.0
        _logger.info("Parking arms: rest pose, then zero.")
        for target in (None, zero):
            outcome = self._play_to_rest(
                robot,
                torque_threshold,
                stopped,
                q_target=target,
                deadline=deadline,
                hold_grippers=hold,
            )
            if outcome != "done":
                raise RuntimeError(f"Soft park stopped ({outcome}); preserving torque")
        _logger.info("Soft park complete: both arms settled at zero.")
        return True

    def _play_to_rest(
        self,
        robot: "AxolRobot",
        torque_threshold: float,
        stopped: Callable[[], bool] | None,
        *,
        q_target: Any | None = None,
        deadline: float | None = None,
        hold_grippers: tuple[float, float] | None = None,
    ) -> str:
        """One play attempt from the current measured positions.

        Plans from the robot's cached positions, then streams the waypoints
        watching the torque residuals. Returns ``"done"``, ``"contact"``, or
        ``"stopped"``.
        """
        import numpy as np

        from ..constants import Joint
        from ..robot.control import ContactWatchdog
        from ..teleop.filter import ResetInterpolator

        self._arms_limp = False

        assert self._conn is not None
        assert self._q_init is not None
        assert self._left_indices is not None
        assert self._right_indices is not None

        parking = deadline is not None

        def check_park() -> None:
            if stopped is not None and stopped():
                raise RuntimeError("Soft park interrupted by a stop request")
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("Soft park exceeded its 30s plan/play/settle budget")
            if self._proc is None or not self._proc.is_alive():
                raise RuntimeError("Soft park IK worker stopped")

        if parking:
            check_park()
        pos_l, pos_r = robot.parking_positions() if parking else robot.positions
        pos_l = np.asarray(pos_l, dtype=np.float32)
        pos_r = np.asarray(pos_r, dtype=np.float32)

        q_current = self._q_init.copy()
        for i, gi in enumerate(self._left_indices):
            q_current[gi] = float(pos_l[i])
        for i, gi in enumerate(self._right_indices):
            q_current[gi] = float(pos_r[i])

        request = ("reset", q_current)
        if q_target is not None:
            request += (q_target,)
        self._conn.send(request)
        if parking:
            while not self._conn.poll(0.05):
                check_park()
                robot.parking_positions()
            check_park()
        result = self._conn.recv()
        if isinstance(result, tuple) and result[0] == "reset_error":
            raise RuntimeError(f"Reset trajectory refused: {result[1]}")
        if not (isinstance(result, tuple) and result[0] == "reset_traj"):
            raise RuntimeError(f"Unexpected IK worker response: {result!r}")
        _, q_goal, traj = result
        if parking:
            q_goal = np.asarray(q_goal, dtype=np.float32)
            if q_goal.shape != q_current.shape or not np.isfinite(q_goal).all():
                raise RuntimeError("Soft park IK worker returned an invalid goal")
            if q_target is not None and not np.array_equal(q_goal, q_target):
                raise RuntimeError("Soft park IK worker did not honor the zero target")
            # A no-op plan still sends the proven nearby hold once. Never
            # interpret an empty or incomplete plan as permission to release.
            traj = list(traj) or [q_goal]
            if any(
                np.asarray(q).shape != q_current.shape or not np.isfinite(q).all()
                for q in traj
            ):
                raise RuntimeError("Soft park IK worker returned an invalid trajectory")
            idx = self._left_indices + self._right_indices
            measured_l, measured_r = robot.parking_positions()
            measured = np.concatenate((measured_l[:7], measured_r[:7]))
            if np.max(np.abs(np.asarray(traj[0])[idx] - measured)) > 0.05:
                raise RuntimeError("Soft park refused: arms moved during planning")
            if np.max(np.abs(np.asarray(traj[-1])[idx] - q_goal[idx])) > 0.05:
                raise RuntimeError("Soft park trajectory does not reach its goal")
            robot.reset_command_state()
        if not traj:
            _logger.warning("IK worker returned an empty reset trajectory; skipping.")
            return "done"

        interp = ResetInterpolator()
        interp.set_trajectory(traj, float(pos_l[7]), float(pos_r[7]))
        watchdog = ContactWatchdog(torque_threshold)

        joints = list(Joint)
        play_hz = float(self._vr_cfg.frequency)
        period = 1.0 / play_hz
        while interp.is_active():
            if stopped is not None and stopped():
                return "stopped"
            t0 = time.perf_counter()
            if parking:
                check_park()
                robot.parking_positions()
            new_q, l_grip, r_grip, _done = interp.step()
            if new_q is None:
                break
            arm_left = np.asarray(new_q)[self._left_indices]
            arm_right = np.asarray(new_q)[self._right_indices]
            if hold_grippers is not None:
                l_grip, r_grip = hold_grippers
            action: dict[str, float] = {}
            for j in joints:
                if j in ARM_JOINTS:
                    ai = ARM_JOINTS.index(j)
                    action[f"left_{j.value}.pos"] = float(arm_left[ai])
                    action[f"right_{j.value}.pos"] = float(arm_right[ai])
                else:
                    action[f"left_{j.value}.pos"] = float(l_grip)
                    action[f"right_{j.value}.pos"] = float(r_grip)
            robot.send_action(action)
            tripped = watchdog.update(robot.torque_residuals())
            if tripped is not None:
                joint, residual = tripped
                _logger.warning(
                    "return-to-rest contact: %s torque residual %.1f exceeds "
                    "%.1f — stopping trajectory",
                    joint,
                    residual,
                    torque_threshold,
                )
                return "contact"
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
        if parking:
            # Rust shaping can lag the last requested waypoint. Continue that
            # collision-checked hold until measured arms are near the goal and
            # stationary for three 100ms windows, before the next leg/release.
            previous = None
            previous_at = time.perf_counter()
            settled = 0
            while settled < 3:
                check_park()
                left, right = robot.parking_positions()
                measured = np.concatenate((left[:7], right[:7]))
                now = time.perf_counter()
                if now - previous_at >= 0.1:
                    near = np.max(np.abs(measured - q_goal[idx])) <= 0.05
                    still = (
                        previous is not None
                        and np.max(np.abs(measured - previous) / (now - previous_at))
                        <= 0.05
                    )
                    settled = settled + 1 if near and still else 0
                    previous, previous_at = measured.copy(), now
                robot.send_action(action)
                if watchdog.update(robot.torque_residuals()) is not None:
                    raise RuntimeError("Soft park contact while settling")
                time.sleep(period)
            # The final blocking send/sleep may itself expose a fault or a
            # stop request. Prove health once more before authorizing release.
            check_park()
            robot.parking_positions()
        return "done"

    def _hold_limp(
        self,
        robot: "AxolRobot",
        gravity_comp_kd: float,
        wait_retry: Callable[[], bool] | None,
        stopped: Callable[[], bool] | None,
    ) -> bool:
        """Hold the arms in gravity comp; ``True`` when the operator retries.

        ``wait_retry`` (when given) blocks in a helper thread while this
        thread streams gravity-comp cycles, so the arms stay limp and
        gravity-supported for as long as the operator prompt is open.
        Without it, the hold runs until ``stopped`` fires (or Ctrl+C
        propagates), then aborts.
        """
        if wait_retry is None and stopped is None:
            # No channel could ever end the hold (e.g. the final teardown
            # return after a stop): don't hold at all — leave the arms where
            # the move stopped and let the caller wind down.
            _logger.warning(
                "return-to-rest aborted on contact (no retry channel); "
                "the arms hold where the move stopped."
            )
            return False
        self._arms_limp = True
        result: dict[str, bool] = {}
        waiter: threading.Thread | None = None
        if wait_retry is not None:
            waiter = threading.Thread(
                target=lambda: result.update(retry=bool(wait_retry())),
                name="axol-reset-retry-wait",
                daemon=True,
            )
            waiter.start()
        period = 1.0 / 100.0
        while True:
            if stopped is not None and stopped():
                return False
            if waiter is not None and not waiter.is_alive():
                return bool(result.get("retry"))
            t0 = time.perf_counter()
            robot.gravity_compensate(kd=gravity_comp_kd)
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def stop(self) -> None:
        """Signal shutdown, close the pipe, and prove subprocess exit.

        The process reference is cleared only after a final post-kill
        liveness check proves the worker exited.  A retained reference makes
        a later cleanup retry possible and prevents callers from treating an
        unverified kill request as ownership release.
        """
        failures: list[tuple[str, BaseException]] = []

        if self._conn is not None:
            try:
                self._conn.send(None)
            except BaseException as error:
                # A broken pipe is expected when the worker already died.  It
                # is not authoritative either way; the process probes below
                # are, so continue through the stronger shutdown actions.
                failures.append(("shutdown signal", error))
            try:
                self._conn.close()
            except BaseException as error:
                failures.append(("pipe close", error))
            else:
                self._conn = None

        process = self._proc
        process_alive = False
        if process is not None:

            def join(label: str, timeout: float) -> None:
                try:
                    process.join(timeout=timeout)
                except BaseException as error:
                    failures.append((label, error))

            def is_alive(label: str) -> bool:
                try:
                    return bool(process.is_alive())
                except BaseException as error:
                    failures.append((label, error))
                    # Failure to prove exit is ownership uncertainty.  Treat
                    # it as live so terminate/kill are still attempted.
                    return True

            join("graceful join", 3.0)
            process_alive = is_alive("post-join liveness check")
            if process_alive:
                try:
                    process.terminate()
                except BaseException as error:
                    failures.append(("terminate", error))
                join("post-terminate join", 2.0)
                process_alive = is_alive("post-terminate liveness check")
            if process_alive:
                try:
                    process.kill()
                except BaseException as error:
                    failures.append(("kill", error))
                # kill() merely requests termination.  The following join and
                # liveness probe are the ownership proof.
                join("post-kill join", 2.0)
                process_alive = is_alive("post-kill liveness check")
            if not process_alive:
                self._proc = None

        if process_alive:
            error = HardwareCleanupError(
                "IK reset worker did not stop; background process ownership "
                "is uncertain"
            )
            for label, failure in failures:
                error.add_note(
                    f"additional IK reset {label} failure: "
                    f"{type(failure).__name__}: {failure}"
                )
            raise error

        # Once exit is proven, a failed signal is harmless (the pipe may have
        # broken precisely because the child exited).  A pipe that could not
        # be closed remains a real local resource leak and stays retryable.
        pipe_failure = next(
            (failure for label, failure in failures if label == "pipe close"), None
        )
        if pipe_failure is not None:
            error = RuntimeError("IK reset worker pipe cleanup failed")
            for label, failure in failures:
                error.add_note(
                    f"additional IK reset {label} failure: "
                    f"{type(failure).__name__}: {failure}"
                )
            raise error from pipe_failure

        self._ready = False


class ActionPublisher:
    """Thread-safe single-slot publisher for the most recently executed action.

    Updated by the control loop after every ``robot.send_action`` call,
    read by :class:`RolloutCaptureThread` to pair each dataset frame with
    the action that drove the robot at that tick.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: "RobotAction | None" = None
        self._first_event = threading.Event()

    def publish(self, action: "RobotAction") -> None:
        snap = dict(action)
        with self._lock:
            self._latest = snap
        self._first_event.set()

    def latest(self) -> "RobotAction | None":
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def wait_for_first(self, timeout: float) -> bool:
        return self._first_event.wait(timeout=timeout)

    def reset(self) -> None:
        with self._lock:
            self._latest = None
        self._first_event.clear()


class RolloutCaptureThread(threading.Thread):
    """Tick at ``fps`` Hz and append one ``(obs, action)`` row per tick.

    Each tick samples a global-timestamp-aligned observation via
    ``AxolRobot.get_observation`` and pairs it with the latest action
    published by the control loop.
    """

    def __init__(
        self,
        *,
        publisher: ActionPublisher,
        robot: "AxolRobot",
        dataset: "LeRobotDataset",
        robot_obs_proc: Callable[[Any], Any],
        fps: int,
        task: str,
        rerun_ip: str | None,
    ) -> None:
        super().__init__(name="axol-rollout-capture", daemon=True)
        self.publisher = publisher
        self.robot = robot
        self.dataset = dataset
        self.robot_obs_proc = robot_obs_proc
        self.fps = fps
        self.task = task
        self.rerun_ip = rerun_ip
        self.stop_event = threading.Event()

    def request_stop(self) -> None:
        """Ask the capture loop to stop at its next safe boundary."""
        self.stop_event.set()

    def unblock_inputs(self) -> None:
        """Disconnect every camera to wake a capture blocked in a frame read.

        This is an escalation path used only after a normal bounded join has
        expired. Camera disconnect is independent per source, so attempt all of
        them and re-raise the first failure after annotating any others. The
        robot's later disconnect remains responsible for motor/CAN teardown.
        """
        primary_error: BaseException | None = None
        for name, camera in self.robot.cameras.items():
            try:
                disconnect = getattr(camera, "disconnect", None)
                if callable(disconnect):
                    disconnect()
            except BaseException as error:
                if primary_error is None:
                    primary_error = error
                else:
                    primary_error.add_note(
                        f"additional rollout camera {name} disconnect failure: "
                        f"{type(error).__name__}: {error}"
                    )
        if primary_error is not None:
            raise primary_error

    def run(self) -> None:
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame
        from lerobot.utils.visualization_utils import log_rerun_data

        if not self.publisher.wait_for_first(timeout=10.0):
            _logger.warning(
                "Rollout capture thread saw no action snapshot within 10s; exiting."
            )
            return
        if self.stop_event.is_set():
            return

        frame_interval = 1.0 / self.fps
        recording_start = time.perf_counter()
        tick = 0
        record_pose_lag = "observation.pose_lag" in self.dataset.features

        while not self.stop_event.is_set():
            target_perf_ts = recording_start + tick * frame_interval

            wait_s = target_perf_ts - time.perf_counter()
            if wait_s > 0 and self.stop_event.wait(timeout=wait_s):
                return

            try:
                if record_pose_lag:
                    obs, pose_lag = self.robot.get_observation_with_pose_lag()
                else:
                    obs = self.robot.get_observation()
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    "Capture tick %d: get_observation failed (%s).", tick, exc
                )
                tick += 1
                continue

            action = self.publisher.latest()
            if action is None:
                tick += 1
                continue

            obs_processed = self.robot_obs_proc(obs)
            # Mantis-created datasets carry the signed pose↔image capture
            # skew. AxolRobot returns it alongside (not inside) the observation
            # so policy inputs keep their negotiated feature schema and this
            # row cannot race with an inference thread's simultaneous read.
            if record_pose_lag:
                obs_processed["pose_lag"] = pose_lag
            obs_frame = build_dataset_frame(
                self.dataset.features, obs_processed, prefix=OBS_STR
            )
            act_frame = build_dataset_frame(
                self.dataset.features, action, prefix=ACTION
            )
            if self.stop_event.is_set():
                return
            self.dataset.add_frame({**obs_frame, **act_frame, "task": self.task})

            if self.rerun_ip:
                log_rerun_data(observation=obs_processed, action=action)

            tick += 1


class PolicyActionLimiter:
    """Per-joint velocity/acceleration envelope over policy actions.

    A control loop that commands a policy's raw action directly has no
    smoothing of its own, so a discontinuous action — a chunk re-planned
    from a stale observation after a slow inference round-trip, or an
    outlier from the policy itself — jerks the arm. The teleop stack
    already solves this with a trapezoidal velocity profile; this wraps the
    same :class:`~almond_axol.teleop.filter.TrapezoidalFilter` around the
    policy's *arm* joints (the grippers snap by design and pass through
    untouched).

    With the default limits at the teleop envelope (~1 rev/s, ~3.5 rev/s²)
    the filter is transparent for normal trained motion and only engages on
    discontinuities, turning a jump into a bounded, acceleration-limited
    move. It is a smoothness guarantee, not a safety stop: a policy heading
    somewhere bad still gets there (smoothly) — the freeze grip / e-stop
    remain the real safeguards. Each engagement beyond a small deviation is
    logged (rate-limited), so jump frequency is visible in the session log —
    useful for telling network hiccups from a jumpy policy.

    Call :meth:`seed` at the robot's measured pose whenever the policy
    (re)takes control, and :meth:`apply` once per tick at ``fps`` (the
    filter's step size is ``max_vel / fps``, so the tick rate must hold).
    """

    # Log an engagement only when the raw target deviates from the filtered
    # command by more than this (rad) on some joint, at most once a second.
    _CLAMP_LOG_THRESHOLD = 0.05

    def __init__(self, max_vel: float, max_accel: float, fps: int) -> None:
        import numpy as np

        from ..teleop.filter import TrapezoidalFilter

        self._np = np
        # Arm-only key lists (grippers excluded): the grippers snap by design
        # and are safe; the arms get the velocity envelope.
        self._left_keys = [f"left_{j.value}.pos" for j in ARM_JOINTS]
        self._right_keys = [f"right_{j.value}.pos" for j in ARM_JOINTS]
        dt = 1.0 / float(fps)
        self._left = TrapezoidalFilter(max_vel, max_accel, dt)
        self._right = TrapezoidalFilter(max_vel, max_accel, dt)
        self._last_clamp_log = 0.0

    def seed(self, pos_l: Any, pos_r: Any) -> None:
        """Reset the envelope to the robot's measured arm positions."""
        np = self._np
        self._left.reset(seed=np.asarray(pos_l[:7], dtype=np.float32))
        self._right.reset(seed=np.asarray(pos_r[:7], dtype=np.float32))

    def apply(self, action: dict[str, float]) -> dict[str, float]:
        """Return ``action`` with the arm joints velocity/accel limited."""
        np = self._np
        raw_l = np.array([action[k] for k in self._left_keys], dtype=np.float32)
        raw_r = np.array([action[k] for k in self._right_keys], dtype=np.float32)
        lim_l = self._left.update(raw_l)
        lim_r = self._right.update(raw_r)

        deviation = max(
            float(np.abs(raw_l - lim_l).max()), float(np.abs(raw_r - lim_r).max())
        )
        now = time.perf_counter()
        if deviation > self._CLAMP_LOG_THRESHOLD and now - self._last_clamp_log > 1.0:
            self._last_clamp_log = now
            _logger.warning(
                "policy action clamped by the velocity envelope (max deviation "
                "%.3f rad) — a discontinuous chunk (late/stale inference) or a "
                "policy jump was smoothed.",
                deviation,
            )

        out = dict(action)
        for key, value in zip(self._left_keys, lim_l):
            out[key] = float(value)
        for key, value in zip(self._right_keys, lim_r):
            out[key] = float(value)
        return out


def latest_observation(robot: "AxolRobot") -> dict[str, Any]:
    """Return the robot's capture-instant-aligned policy observation.

    Kept as a compatibility helper for downstream policy CLIs. Callers that
    also need to label an action with the observation's canonical exposure time
    should use ``AxolRobot.get_observation_with_capture_timestamp`` instead.
    """
    return robot.get_observation()


def stdin_watcher(
    stop_event: threading.Event,
    result: dict[str, str | None],
    on_subtask: Callable[[int], None] | None = None,
    num_subtasks: int = 0,
    *,
    eof_choice: str | None = None,
    immediate_quit: bool = False,
    ready_event: threading.Event | None = None,
    allowed_choices: tuple[str, ...] = ("s", "r", "q"),
) -> None:
    """Watch stdin for ``s`` / ``r`` / ``q`` on its own line.

    Uses ``select.select`` so it never blocks past the stop event. Sets
    ``result["choice"]`` to the first valid keystroke received.

    When ``num_subtasks`` is set, a bare integer ``1``..``num_subtasks``
    instead switches the running policy's instruction to that subtask via
    ``on_subtask(index)`` and keeps watching — it does NOT end the episode.
    Anything else is ignored. With ``immediate_quit`` on a TTY, q/Q is an
    immediate key while s/r and subtask numbers still require Enter. That
    opt-in mode preserves Ctrl-C and restores the exact terminal settings
    before exiting. Reader failures are published as ``result["error"]``;
    callers must stop the episode rather than continue without an input reader.
    ``allowed_choices`` limits accepted commands; an idle DAgger gate accepts
    only quit and keeps reading after save/discard keys.
    """
    import select
    import sys

    if immediate_quit:
        try:
            if not sys.stdin.isatty():
                raise RuntimeError("Immediate quit requires an interactive terminal")
            _stdin_cbreak_watcher(
                stop_event,
                result,
                on_subtask,
                num_subtasks,
                eof_choice,
                ready_event,
                allowed_choices,
            )
        except BaseException as error:
            result["error"] = f"{type(error).__name__}: {error}"
            result["choice"] = eof_choice or "abort"
        finally:
            if ready_event is not None:
                ready_event.set()
        return

    if ready_event is not None:
        ready_event.set()

    while not stop_event.is_set():
        ready, _, _ = select.select([sys.stdin], [], [], 0.25)
        if not ready:
            continue
        line = sys.stdin.readline()
        if not line:
            result["choice"] = eof_choice
            return
        ch = line.strip().lower()
        if ch in allowed_choices:
            result["choice"] = ch
            return
        if num_subtasks and on_subtask is not None and ch.isdigit():
            idx = int(ch)
            if 1 <= idx <= num_subtasks:
                on_subtask(idx)
            else:
                print(
                    f"  Ignoring subtask {idx}: valid range is 1-{num_subtasks}.",
                    flush=True,
                )


def _stdin_cbreak_watcher(
    stop_event: threading.Event,
    result: dict[str, str | None],
    on_subtask: Callable[[int], None] | None,
    num_subtasks: int,
    eof_choice: str | None,
    ready_event: threading.Event | None,
    allowed_choices: tuple[str, ...] = ("s", "r", "q"),
) -> None:
    """Read unbuffered terminal bytes, restoring modes on every exit path."""
    import os
    import select
    import sys
    import termios

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    mode = [*saved[:6], saved[6][:]]
    mode[3] = (mode[3] & ~(termios.ICANON | termios.ECHO)) | termios.ISIG
    mode[6][termios.VMIN] = 1
    mode[6][termios.VTIME] = 0
    eof = saved[6][termios.VEOF]
    erase = saved[6][termios.VERASE]
    kill = saved[6][termios.VKILL]
    pending = bytearray()
    try:
        # TCSAFLUSH would discard a q already typed at episode startup.
        termios.tcsetattr(fd, termios.TCSANOW, mode)
        if ready_event is not None:
            ready_event.set()
        while not stop_event.is_set():
            ready, _, _ = select.select([fd], [], [], 0.25)
            if not ready:
                continue
            # TextIOWrapper.read/readline can prefetch bytes into a Python
            # buffer that select cannot see. One OS byte leaves later prompt
            # input in the terminal until this episode's choice is complete.
            key = os.read(fd, 1)
            if not key or key == eof:
                result["choice"] = eof_choice or "abort"
                return
            if key.lower() == b"q" and "q" in allowed_choices:
                result["choice"] = "q"
                print("\nQuit requested; stopping episode.", flush=True)
                return
            if key in (erase, b"\x08", b"\x7f"):
                if pending:
                    pending.pop()
                continue
            if key == kill:
                pending.clear()
                continue
            if key not in (b"\r", b"\n"):
                if len(pending) < 128:
                    pending.extend(key)
                continue
            command = pending.decode("utf-8", errors="replace").strip().lower()
            pending.clear()
            if command in allowed_choices:
                result["choice"] = command
                return
            if num_subtasks and on_subtask is not None and command.isdigit():
                index = int(command)
                if 1 <= index <= num_subtasks:
                    on_subtask(index)
                    continue
            if command:
                if allowed_choices == ("q",):
                    print("\nPress q to quit; start recording from VR.", flush=True)
                    continue
                subtask_hint = f", or 1-{num_subtasks}+Enter" if num_subtasks else ""
                print(
                    "\nUnrecognized input. Press q to quit, s+Enter to save, "
                    f"r+Enter to rerecord{subtask_hint}.",
                    flush=True,
                )
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, saved)
        if result.get("choice") == "q":
            # This explicit quit ends the session. Discard only its queued
            # repeat keys/Enter; ordinary s/r handoffs retain pending input.
            termios.tcflush(fd, termios.TCIFLUSH)
