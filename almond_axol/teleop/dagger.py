"""DAgger-intervention state machine for VR teleoperation.

During a DAgger episode (``axol collect-dagger``) a policy drives the robot
while the operator watches in VR; the grip ("side") buttons implement the
intervention flow:

  - **either** grip alone (policy driving)  → freeze: the robot holds its
    pose and the CLI pauses dataset capture;
  - **both** grips together (while frozen)  → take over: teleop engages at
    the robot's *current* pose and capture resumes;
  - **either** grip alone (while teleoping) → hand back: the policy resumes
    and capture keeps running.

The stock stack can't do the takeover safely: :class:`~.worker.IKWorker`
anchors its engage snapshot to its *own* last solution, which is stale (the
rest pose) after the policy has moved the arms — engaging would smoothly drag
the robot back toward that stale target. :class:`DaggerTeleopCore` fixes that
with the worker's ``("sync", pos_left, pos_right)`` message (see
:func:`~.worker.run_ik_worker`), which seats the worker's joint vector at the
robot's measured positions right before the engage snapshot, plus the matching
control-side state seeding (:meth:`DaggerTeleopCore._sync_to_robot`).

The LeRobot adapter wrapping this core is
:class:`almond_axol.lerobot.teleop.teleop_vr_dagger.DaggerVRTeleop`.
"""

from __future__ import annotations

import logging
import multiprocessing.connection
import threading
import time
from collections.abc import Callable

import numpy as np

from .config import VRTeleopConfig
from .core import VRTeleopCore, recv_with_timeout

# How long a control-side sync waits for the worker's ("synced", q) reply.
_SYNC_RECV_TIMEOUT = 5.0


class DaggerTeleopCore(VRTeleopCore):
    """Engage state machine for DAgger interventions.

    Replaces the stock engage toggle with the DAgger button flow (see module
    docstring) and adds:

    - :attr:`intervention_allowed` — set by the CLI while a policy episode is
      running; grip presses are inert otherwise, so nothing latches between
      episodes or while an episode is being saved.
    - a *freeze latch*: a rising edge of either grip alone while the policy is
      driving. The CLI's control loop consumes it (:meth:`consume_freeze`) to
      stop the policy and pause capture.
    - robot-pose sync on engage: right before enabling teleop, the worker's q
      and this core's target/filters are seated at the robot's measured
      positions, so the first engaged output commands the arm pose the robot
      is already in. The **grippers adopt the controller triggers
      immediately** (softened over a few ticks by the EMA): the operator
      pre-sets the triggers before squeezing the grips — hold the trigger
      down before a takeover so a held part stays gripped. (An earlier
      "clutch" that held the gripper until the trigger matched blocked
      exactly that deliberate pre-squeeze whenever the gripper was partway
      closed on a part.)

    The VR reset button and the startup trajectory are disabled: both would
    replay a trajectory through :meth:`compute_output` while the CLI's control
    loop is in a policy state and not consuming it (or worse, while the
    policy is simultaneously commanding). Between-episode homing is the CLI's
    ``IKResetController``'s job, so :meth:`note_frame_reset` never latches the
    core's own reset. The stale-pose-stream **auto-disengage**
    (``disengage_timeout``) is kept, though: :meth:`note_frame_reset` stamps
    the pose heartbeat, so a dead headset link mid-intervention hands control
    back to the policy instead of leaving the arms latched to a stale engage
    snapshot (which would jump on VR re-entry).
    """

    def __init__(
        self,
        config: VRTeleopConfig,
        logger: logging.Logger,
        broadcast_tracking: Callable[[bool], None],
    ) -> None:
        super().__init__(config, logger, broadcast_tracking)
        # Set while a policy episode is live; grips are inert otherwise.
        self.intervention_allowed = threading.Event()
        # Set while the CLI's idle (between-episode) phase runs: a VR reset
        # button press then latches a home request for the idle loop (see
        # note_frame_reset). Never set during an episode.
        self.idle_reset_armed = threading.Event()
        # Rising edge of either grip alone while disengaged (the freeze
        # request). Written on the IK thread, consumed on the control loop.
        self._freeze_lock = threading.Lock()
        self._freeze_latch = False
        self._idle_reset_latch = False
        # Either-grip edge state for the DAgger toggle. The base core tracks
        # per-grip edges (_prev_l_lock / _prev_r_lock) for its per-arm
        # toggles; the DAgger flow is all-or-nothing (freeze / take over /
        # hand back move BOTH arms between the policy and the operator), so
        # it keeps its own combined edge.
        self._prev_either = False
        # Worker pipe + robot-position source, attached at connect time.
        self._conn: multiprocessing.connection.Connection | None = None
        self._get_positions: Callable[[], tuple[np.ndarray, np.ndarray]] | None = None

    def attach(
        self,
        conn: multiprocessing.connection.Connection,
        get_positions: Callable[[], tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Wire in the worker pipe and the robot position source.

        ``get_positions`` returns the robot's cached ``(left, right)`` arm
        positions ((8,) each: 7 arm joints + gripper); it is called on the IK
        thread at engage time.
        """
        self._conn = conn
        self._get_positions = get_positions

    # -- Control-loop API ---------------------------------------------------

    def consume_freeze(self) -> bool:
        """Return-and-clear the freeze latch (either grip pressed alone)."""
        with self._freeze_lock:
            latched = self._freeze_latch
            self._freeze_latch = False
            return latched

    def consume_idle_reset(self) -> bool:
        """Return-and-clear the idle home request (VR reset button)."""
        with self._freeze_lock:
            latched = self._idle_reset_latch
            self._idle_reset_latch = False
            return latched

    def _set_engaged(self, engaged: bool) -> None:
        """Engage/disengage BOTH arms as a unit.

        The base core's ``teleop_enabled`` is a read-only property over the
        per-arm flags; DAgger interventions always move both arms between
        the policy and the operator, so both flags flip together.
        """
        self.left_enabled = engaged
        self.right_enabled = engaged

    def force_disengage(self) -> None:
        """Disable teleop (e.g. when an episode ends mid-intervention)."""
        if self.teleop_enabled:
            self._set_engaged(False)
            self._logger.info("Teleop force-disengaged.")
            self._broadcast(False)

    # -- Disabled base behaviours --------------------------------------------

    def set_startup_trajectory(self, trajectory: object) -> None:
        """No-op: the policy loop owns the arm from the first tick.

        Playing the worker's startup trajectory through compute_output would
        require the control loop to consume get_action() before any episode
        starts — it doesn't, and homing is the IKResetController's job.
        """

    def note_frame_reset(self, reset: bool) -> None:
        """Latch idle-phase home requests; never the core's own reset path.

        The base behaviour (latching ``_reset_latched``) would make
        run_ik_loop plan and play a return-to-rest trajectory that fights
        whichever source (policy or operator) is commanding the robot, so
        during an episode the reset button stays inert (episodes end via the
        VR record button, with reset+record marking a failure). Between
        episodes the CLI arms :attr:`idle_reset_armed`, and a rising edge
        latches a request its idle loop consumes to home the arms via the
        ``IKResetController`` — which plans from the robot's measured pose,
        unlike the core's path (whose ``q`` is stale outside teleop).

        The pose heartbeat stamp is kept from the base method: it drives the
        stale-stream auto-disengage (``disengage_timeout``), which in DAgger
        hands a dead-link intervention back to the policy instead of leaving
        teleop latched to a stale engage snapshot (which would jump on VR
        re-entry).
        """
        self._last_frame_time = time.perf_counter()
        if reset and not self._prev_reset and self.idle_reset_armed.is_set():
            with self._freeze_lock:
                self._idle_reset_latch = True
        self._prev_reset = reset

    # -- Engage state machine (IK thread) -------------------------------------

    def update_engage(self, frame: object) -> None:
        """Advance the DAgger freeze/engage/hand-back toggle for one VR frame.

        Reimplements (rather than wraps) ``VRTeleopCore.update_engage``: the
        rising-edge bookkeeping is shared, but every branch differs — either
        grip alone latches a *freeze* instead of being ignored, engaging is
        gated on :attr:`intervention_allowed` and preceded by the robot-pose
        sync, and the grippers adopt the controller triggers directly.
        """
        both = frame.l_lock and frame.r_lock
        either = frame.l_lock or frame.r_lock
        allowed = self.intervention_allowed.is_set()

        if not self.teleop_enabled:
            if allowed and either and not self._prev_either:
                # Freeze request (also fires on the first frame of a
                # two-handed engage — the control loop treats freeze→engage
                # in one tick as a direct takeover, so that's harmless).
                with self._freeze_lock:
                    self._freeze_latch = True
            if allowed and both and not self._prev_both:
                try:
                    self._sync_to_robot()
                except Exception:  # noqa: BLE001 - stay frozen, operator retries
                    self._logger.exception(
                        "Robot-pose sync failed; NOT engaging teleop. "
                        "Release and squeeze both grips to retry."
                    )
                else:
                    self._set_engaged(True)
                    self._logger.info("Teleop engaged (DAgger intervention).")
                    self._broadcast(True)
                    # Ramp in gently on every takeover (not just out of
                    # rest): compute_output smoothsteps the velocity cap
                    # from engage_max_vel up to teleop_max_vel across
                    # engage_duration once _engage_time is set.
                    self.smooth_left.max_vel = self.config.engage_max_vel
                    self.smooth_right.max_vel = self.config.engage_max_vel
                    self._engage_time = time.perf_counter()
                    self._at_rest = False
        else:
            if either and not self._prev_either:
                self._set_engaged(False)
                self._logger.info("Teleop disengaged — policy resumes.")
                self._broadcast(False)
        self._prev_both = both
        self._prev_either = either

        if self.teleop_enabled:
            # Grippers track the controller triggers immediately (the EMA in
            # compute_output softens the transition over a few ticks). The
            # operator pre-sets the triggers before engaging — hold the
            # trigger down before a takeover so a held part stays gripped.
            self.l_grip = frame.l_grip
            self.r_grip = frame.r_grip

    def _sync_to_robot(self) -> None:
        """Seat the worker's q and this core's state at the robot's pose.

        Runs on the IK thread between frame dispatches, so the worker pipe is
        idle and safe to use. The robot has been frozen (holding its last
        command) since the freeze press, so the cached measured positions are
        settled. Seeding the smoothing filters here is safe because the
        control loop only advances them (compute_output) while engaged, and
        we aren't engaged yet.
        """
        if self._conn is None or self._get_positions is None:
            raise RuntimeError("DaggerTeleopCore.attach() was never called")
        pos_l, pos_r = self._get_positions()
        pos_l = np.asarray(pos_l, dtype=np.float32)
        pos_r = np.asarray(pos_r, dtype=np.float32)

        self._conn.send(("sync", pos_l[:7].copy(), pos_r[:7].copy()))
        result = recv_with_timeout(self._conn, _SYNC_RECV_TIMEOUT)
        if not (isinstance(result, tuple) and result[0] == "synced"):
            raise RuntimeError(f"unexpected sync reply from IK worker: {result!r}")
        # _hold_target, not a bare ``self.q =``: run_ik_loop's set_target()
        # keeps starting playback segments toward the worker's echoed q even
        # while disengaged, and compute_output prefers the segment over
        # ``q`` — a leftover segment still points at the worker's pre-sync
        # (stale) solution, so the first takeover ticks would command toward
        # it. Adopting via _hold_target clears the segment so the synced
        # pose is what the filters track from the first engaged tick.
        self._hold_target(np.asarray(result[1], dtype=np.float32))

        # Seed the grips at the robot's current gripper positions so the
        # EMA blends from them to the controller triggers over a few ticks
        # (the triggers take over on the very next frame — see update_engage).
        self.l_grip = float(pos_l[7])
        self.r_grip = float(pos_r[7])
        self.seed_filters(pos_l, pos_r)
