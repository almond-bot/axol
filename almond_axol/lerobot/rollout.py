"""
Shared rollout machinery for policy CLIs.

Pulled out of ``axol run-policy`` so other policy-running CLIs can reuse
the same episode plumbing without duplicating it:

- :class:`IKResetController` — collision-aware return-to-rest backed by
  an out-of-process JAX IK worker.
- :class:`ActionPublisher` — single-slot thread-safe handoff of the most
  recently executed action.
- :class:`RolloutCaptureThread` — fixed-rate thread that pairs a
  timestamp-aligned observation with the latest published action and
  appends it to a ``LeRobotDataset``.
- :func:`stdin_watcher` — ``s`` / ``r`` / ``q`` keystroke watcher with
  no-block ``select`` polling.

All four are LeRobot-flavoured: the capture thread depends on
``lerobot.datasets.lerobot_dataset.LeRobotDataset``, ``build_dataset_frame``,
and ``log_rerun_data``; the reset controller talks to the JAX IK worker via
``almond_axol.teleop``. The module lives under ``almond_axol/lerobot``
alongside the other LeRobot adapters.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

from ..constants import ARM_JOINTS

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.types import RobotAction

    from .robot.robot_axol import AxolRobot

_logger = logging.getLogger(__name__)


class IKResetController:
    """Collision-aware return-to-rest, backed by an IK worker subprocess.

    Mirrors the reset path used by ``AxolVRTeleop`` (collect-data) but
    without the VR server. ``start()`` spawns ``run_ik_worker`` (JAX +
    JITed solver, ~10-20 s); ``wait_ready()`` blocks on the handshake;
    ``return_to_rest()`` plans a joint-space trajectory and streams its
    waypoints to the impedance controller. Spawn before ``client.start()``
    so the IK JIT overlaps with the policy load.
    """

    def __init__(self) -> None:
        from ..kinematics.config import KinematicsConfig
        from ..teleop.config import VRTeleopConfig

        self._vr_cfg = VRTeleopConfig()
        self._kin_cfg = KinematicsConfig()
        self._proc: Any | None = None
        self._conn: Any | None = None
        self._q_init: Any | None = None
        self._left_indices: list[int] | None = None
        self._right_indices: list[int] | None = None
        self._ready = False

    def start(self) -> None:
        """Spawn the IK worker subprocess. Non-blocking; pair with ``wait_ready``."""
        import multiprocessing as mp

        from ..teleop.worker import run_ik_worker

        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(
            target=run_ik_worker,
            args=(child_conn, self._vr_cfg, self._kin_cfg, None, None),
            name="axol-ik-worker",
            daemon=True,
        )
        proc.start()
        child_conn.close()
        self._proc = proc
        self._conn = parent_conn

    def wait_ready(self, timeout: float = 60.0) -> None:
        """Block until the IK worker has finished JIT compilation."""
        if self._ready:
            return
        if self._conn is None:
            raise RuntimeError("IK reset controller not started")
        if not self._conn.poll(timeout):
            raise TimeoutError(
                f"IK worker did not become ready within {timeout:.1f}s "
                "(JAX JIT compilation may have stalled)."
            )
        msg = self._conn.recv()
        if not (isinstance(msg, tuple) and msg[0] == "ready"):
            raise RuntimeError(f"Unexpected IK worker handshake: {msg!r}")
        import numpy as np

        _, q_init, left_indices, right_indices, _startup_traj = msg
        self._q_init = np.asarray(q_init, dtype=np.float32)
        self._left_indices = [int(i) for i in left_indices]
        self._right_indices = [int(i) for i in right_indices]
        self._ready = True

    def return_to_rest(
        self,
        robot: "AxolRobot",
        *,
        torque_threshold: float = 4.0,
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
        self.wait_ready()
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

    def _play_to_rest(
        self,
        robot: "AxolRobot",
        torque_threshold: float,
        stopped: Callable[[], bool] | None,
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

        assert self._conn is not None
        assert self._q_init is not None
        assert self._left_indices is not None
        assert self._right_indices is not None

        pos_l, pos_r = robot.positions
        pos_l = np.asarray(pos_l, dtype=np.float32)
        pos_r = np.asarray(pos_r, dtype=np.float32)

        q_current = self._q_init.copy()
        for i, gi in enumerate(self._left_indices):
            q_current[gi] = float(pos_l[i])
        for i, gi in enumerate(self._right_indices):
            q_current[gi] = float(pos_r[i])

        self._conn.send(("reset", q_current))
        result = self._conn.recv()
        if not (isinstance(result, tuple) and result[0] == "reset_traj"):
            raise RuntimeError(f"Unexpected IK worker response: {result!r}")
        _, _q_rest, traj = result
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
            new_q, l_grip, r_grip, _done = interp.step()
            if new_q is None:
                break
            arm_left = np.asarray(new_q)[self._left_indices]
            arm_right = np.asarray(new_q)[self._right_indices]
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
                    "%.1f — going limp",
                    joint,
                    residual,
                    torque_threshold,
                )
                return "contact"
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
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
        """Signal shutdown, close the pipe, and reap the subprocess."""
        if self._conn is not None:
            try:
                self._conn.send(None)
            except Exception:  # noqa: BLE001
                pass
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None
        if self._proc is not None:
            self._proc.join(timeout=3.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.kill()
            self._proc = None


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

        while not self.stop_event.is_set():
            target_perf_ts = recording_start + tick * frame_interval

            wait_s = target_perf_ts - time.perf_counter()
            if wait_s > 0 and self.stop_event.wait(timeout=wait_s):
                return

            try:
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


def stdin_watcher(
    stop_event: threading.Event,
    result: dict[str, str | None],
) -> None:
    """Watch stdin for ``s`` / ``r`` / ``q`` on its own line.

    Uses ``select.select`` so it never blocks past the stop event. Sets
    ``result["choice"]`` to the first valid keystroke received.
    """
    import select
    import sys

    while not stop_event.is_set():
        ready, _, _ = select.select([sys.stdin], [], [], 0.25)
        if not ready:
            continue
        line = sys.stdin.readline()
        if not line:
            return
        ch = line.strip().lower()
        if ch in ("s", "r", "q"):
            result["choice"] = ch
            return
