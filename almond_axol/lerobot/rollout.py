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
- :class:`PolicyActionLimiter` — per-joint velocity/acceleration envelope
  over policy actions, for control loops that command a policy's raw
  output directly (``collect-dagger``).
- :func:`latest_observation` — joint state + each camera's newest frame
  without the capture-instant alignment wait, for inference ticks that
  want the freshest frame rather than a timestamp-aligned one.
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
from ..robot.base import HardwareCleanupError, mark_hardware_cleanup_uncertain

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.lerobot_types import RobotAction

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
    """Joint state + each camera's newest frame, without waiting for one.

    Used by inference ticks (``collect-dagger`` and downstream policy CLIs):
    ``AxolRobot.get_observation`` aligns cameras with ``read_at_or_after(now)``,
    which *waits* for a frame captured after now, serialized across all the
    cameras (~up to a frame period each) — a hard ceiling far below fps on the
    loop calling it. Inference wants the freshest frame it can get;
    capture-instant alignment only matters for the dataset, which the recorder
    subprocess handles on its own clock. When the cameras are a video relay's
    shared-memory readers (pyshm), a read is just a block copy.

    Raises RuntimeError listing the cameras that had no readable frame.
    """
    obs = robot.get_joint_observation()
    missing: list[str] = []
    for cam_name, cam in robot.cameras.items():
        try:
            obs[cam_name] = cam.read_latest()
        except Exception as exc:  # noqa: BLE001
            _logger.debug("Camera %s read_latest failed (%s).", cam_name, exc)
            missing.append(cam_name)
    if missing:
        raise RuntimeError(f"no readable frame from cameras {missing}")
    return obs


def stdin_watcher(
    stop_event: threading.Event,
    result: dict[str, str | None],
    on_subtask: Callable[[int], None] | None = None,
    num_subtasks: int = 0,
) -> None:
    """Watch stdin for ``s`` / ``r`` / ``q`` on its own line.

    Uses ``select.select`` so it never blocks past the stop event. Sets
    ``result["choice"]`` to the first valid keystroke received.

    When ``num_subtasks`` is set, a bare integer ``1``..``num_subtasks``
    instead switches the running policy's instruction to that subtask via
    ``on_subtask(index)`` and keeps watching — it does NOT end the episode.
    Anything else is ignored.
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
        if num_subtasks and on_subtask is not None and ch.isdigit():
            idx = int(ch)
            if 1 <= idx <= num_subtasks:
                on_subtask(idx)
            else:
                print(
                    f"  Ignoring subtask {idx}: valid range is 1-{num_subtasks}.",
                    flush=True,
                )
