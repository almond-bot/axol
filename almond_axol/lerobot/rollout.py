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
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.types import RobotAction

    from ..kinematics.config import KinematicsConfig
    from ..teleop.config import VRTeleopConfig
    from .robot.robot_axol import AxolRobot

_logger = logging.getLogger(__name__)


def _announce(text: str) -> None:
    """Announce a park step via lerobot's ``log_say`` (spoken prompts match
    the CLIs' own status lines); fall back to plain logging without lerobot."""
    try:
        from lerobot.utils.utils import log_say

        log_say(text)
    except Exception:  # noqa: BLE001 - announcement must never break the park
        _logger.info(text)


class IKResetController:
    """Collision-aware return-to-rest, backed by an IK worker subprocess.

    Mirrors the reset path used by ``AxolVRTeleop`` (collect-data) but
    without the VR server. ``start()`` spawns ``run_ik_worker`` (JAX +
    JITed solver, ~10-20 s); ``wait_ready()`` blocks on the handshake;
    ``return_to_rest()`` plans a joint-space trajectory and streams its
    waypoints to the impedance controller. Spawn before ``client.start()``
    so the IK JIT overlaps with the policy load.
    """

    def __init__(
        self,
        vr_config: "VRTeleopConfig | None" = None,
        kin_config: "KinematicsConfig | None" = None,
    ) -> None:
        from ..kinematics.config import KinematicsConfig
        from ..teleop.config import VRTeleopConfig

        # Both configs default to the stock ones (unchanged behavior); pass a
        # VRTeleopConfig to customize the rest pose the resets target
        # (rest_pose_left / rest_pose_right) or the playback frequency.
        self._vr_cfg = vr_config if vr_config is not None else VRTeleopConfig()
        self._kin_cfg = kin_config if kin_config is not None else KinematicsConfig()
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

    def return_to_rest(self, robot: "AxolRobot") -> None:
        """Plan and play a collision-aware trajectory to the rest pose."""
        self._request_and_play(robot)

    def _request_and_play(
        self,
        robot: "AxolRobot",
        q_target: "np.ndarray | None" = None,
        *,
        deadline_s: float | None = None,
        hold_grippers: "tuple[float, float] | None" = None,
        abort_on_sigint: bool = False,
    ) -> bool:
        """Plan a collision-aware trajectory and stream it to the robot.

        ``q_target=None`` plans to the configured rest pose (the worker's
        original 2-tuple reset message, unchanged); an explicit target is sent
        as the 3-tuple form. ``deadline_s`` caps the move — the plan
        round-trip gets the remaining budget and past the deadline the
        playback loop breaks with a warning, so whatever pose was reached
        stands. ``hold_grippers`` is a ``(left, right)`` pair of normalized
        gripper targets to hold on every waypoint instead of ramping the
        grippers open. ``abort_on_sigint`` makes Ctrl+C raise for the
        duration of the move so an operator can abort a park; off the main
        thread (e.g. under ``axol serve``, where handlers can't be installed)
        the move simply runs non-abortable, still capped by ``deadline_s``.

        Returns ``False`` when the move was aborted by Ctrl+C or the worker
        didn't answer the plan request within the budget, ``True`` otherwise
        (a mid-playback deadline break is best-effort playback, not an
        error).
        """
        import numpy as np

        from ..constants import Joint
        from ..teleop.filter import ResetInterpolator

        prev_handler: Any | None = None
        if abort_on_sigint:
            import signal

            def _raise(signum: int, frame: Any) -> None:
                raise KeyboardInterrupt

            try:
                prev_handler = signal.signal(signal.SIGINT, _raise)
            except (ValueError, OSError):
                prev_handler = None

        try:
            self.wait_ready()
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

            deadline = (
                time.perf_counter() + deadline_s if deadline_s is not None else None
            )
            if q_target is None:
                self._conn.send(("reset", q_current))
            else:
                self._conn.send(("reset", q_current, q_target))
            # A capped move must also cap the plan round-trip: a wedged or
            # dying worker would otherwise hang this recv past the park
            # budget (a warm worker answers in milliseconds).
            if deadline is not None and not self._conn.poll(
                max(0.0, deadline - time.perf_counter())
            ):
                _logger.warning(
                    "IK worker did not answer the reset plan within the %.1fs "
                    "budget; skipping the move.",
                    deadline_s,
                )
                return False
            result = self._conn.recv()
            if not (isinstance(result, tuple) and result[0] == "reset_traj"):
                raise RuntimeError(f"Unexpected IK worker response: {result!r}")
            _, _goal_q, traj = result
            if not traj:
                # A zero-length move (already at the target) plans to an empty
                # trajectory; nothing to play.
                _logger.warning(
                    "IK worker returned an empty reset trajectory; skipping."
                )
                return True

            interp = ResetInterpolator()
            if hold_grippers is not None:
                interp.set_trajectory(
                    traj, hold_grippers[0], hold_grippers[1], hold_grippers=True
                )
            else:
                interp.set_trajectory(traj, float(pos_l[7]), float(pos_r[7]))

            joints = list(Joint)
            play_hz = float(self._vr_cfg.frequency)
            period = 1.0 / play_hz
            while interp.is_active():
                if deadline is not None and time.perf_counter() >= deadline:
                    _logger.warning(
                        "reset playback exceeded its %.1fs deadline; "
                        "stopping the move early.",
                        deadline_s,
                    )
                    break
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
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))
            return True
        except KeyboardInterrupt:
            if not abort_on_sigint:
                raise
            _announce("Park aborted — releasing motors now.")
            return False
        finally:
            if prev_handler is not None:
                import signal

                try:
                    signal.signal(signal.SIGINT, prev_handler)
                except (ValueError, OSError):
                    pass

    def park(self, robot: "AxolRobot", *, deadline_s: float = 30.0) -> bool:
        """Ease both arms to the rest pose and then to all-zeros.

        The soft-shutdown park: played on a *clean* stop right before
        ``robot.disconnect()`` releases the motors, so the arms descend
        gradually instead of dropping from wherever the run ended. The
        grippers hold their last commanded value throughout (a park must not
        drop or loosen a held object — see ``AxolArm.last_gripper_commanded``),
        one ``deadline_s`` budget covers BOTH legs, and a second
        Ctrl+C aborts the move and releases the motors immediately.
        Best-effort: every failure is logged and swallowed — the caller's
        ``finally`` must still disconnect no matter what happens here.

        Returns ``True`` when both legs played to completion.
        """
        if self._conn is None:
            return False
        if self._proc is not None and not self._proc.is_alive():
            _logger.warning("IK worker is gone; skipping the park.")
            return False

        deadline = time.perf_counter() + deadline_s
        _announce("Parking arms: rest pose, then zero.")
        try:
            # Pick the gripper value to hold ONCE, for both legs. Prefer the
            # last COMMANDED value: the grippers run position-force control,
            # so on a held object the measured position sags below the
            # command (the force comes from that gap) — re-commanding the
            # measured value would collapse the grip force, and re-reading it
            # per leg would ratchet the grip looser leg by leg. Snapshot
            # before reset_command_state() clears the command cache; fall
            # back to the measured position when nothing was commanded yet.
            pos_l, pos_r = robot.positions
            hold_l, hold_r = float(pos_l[7]), float(pos_r[7])
            cmd_l, cmd_r = robot.last_gripper_commands()
            if cmd_l is not None:
                hold_l = cmd_l
            if cmd_r is not None:
                hold_r = cmd_r

            # The arms may sit far from the last commanded pose (the run ended
            # mid-motion): clear the per-step jump guard first, or
            # motion_control silently drops every waypoint whose arm delta vs
            # the stale cached command exceeds max_step_rad and the park never
            # moves.
            robot.reset_command_state()
            if not self._request_and_play(
                robot,
                deadline_s=deadline - time.perf_counter(),
                hold_grippers=(hold_l, hold_r),
                abort_on_sigint=True,
            ):
                return False

            assert self._q_init is not None
            assert self._left_indices is not None
            assert self._right_indices is not None
            q_target = self._q_init.copy()
            q_target[self._left_indices] = 0.0
            q_target[self._right_indices] = 0.0

            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                _logger.warning(
                    "park deadline (%.1fs) exhausted before the zero leg; "
                    "releasing from the rest pose.",
                    deadline_s,
                )
                return False
            return self._request_and_play(
                robot,
                q_target,
                deadline_s=remaining,
                hold_grippers=(hold_l, hold_r),
                abort_on_sigint=True,
            )
        except KeyboardInterrupt:
            # A Ctrl+C landing between the legs (outside _request_and_play's
            # raising handler) still aborts the park.
            _announce("Park aborted — releasing motors now.")
            return False
        except Exception:  # noqa: BLE001 - never block the disconnect
            _logger.exception("Soft-shutdown park failed; releasing motors.")
            return False

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
