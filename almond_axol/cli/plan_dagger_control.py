"""One motor owner for DAgger through the custom policy interface."""

from __future__ import annotations

import logging
import time

from ..policy.handover import JointHandover
from ..utils import affinity
from ..utils.control_loop import run_blocking_with_sync_control_ticks
from .collect_dagger import _STATE_POLICY, _STATE_TELEOP, _DaggerControlLoop

_logger = logging.getLogger(__name__)


class _PolicyRevoked(RuntimeError):
    """An operator or stop request arrived while a policy target was resolved."""


def disconnect_plan_robot(
    robot,
    reset_controller,
    *,
    park: bool,
    torque_threshold: float,
    stopped,
) -> None:
    """Only a clean explicit quit parks and removes motor support.

    Abort/error paths preserve the held position, matching run-policy
    through the custom policy interface.
    Networking must be paused locally before this call and closed afterward.
    """
    if not park:
        _logger.warning(
            "DAgger stopped without parking; preserving motor position hold."
        )
        robot.disconnect_preserving_position()
        return
    try:
        if not reset_controller.park(
            robot, torque_threshold=torque_threshold, stopped=stopped
        ):
            raise RuntimeError("DAgger park did not complete; preserving torque")
    except BaseException as error:
        try:
            robot.disconnect_preserving_position()
        except BaseException as cleanup_error:
            error.add_note(f"additional preserving disconnect failure: {cleanup_error}")
        raise
    robot.disconnect()


class PlanDaggerControlLoop(_DaggerControlLoop):
    """Inference never sends motors; this loop arbitrates every dispatch.

    Intervention boundaries fence recorder snapshots without ending the
    dataset episode. A dead headset with held grips retains human ownership
    and holds the last command until an explicit release arrives.
    """

    def __init__(
        self,
        *,
        initial_action: dict[str, float],
        handover_duration_s: float,
        record_joint_actions: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hold_action = dict(initial_action)
        self.record_joint_actions = record_joint_actions
        self.handover = JointHandover(fps=self.fps, duration_s=handover_duration_s)
        self.handover.seed(self.hold_action)
        self._resume_pending = False
        self.robot.reset_cartesian_seed()

    def _operator_requested(self) -> bool:
        return self.teleop.teleop_engaged or self.teleop.grips_held_raw

    def _dataset_action(self, action):
        return (
            action
            if self.record_joint_actions
            else self.robot.action_to_dataset(action)
        )

    def _hold(self) -> None:
        if not self.shutdown_event.is_set():
            performed = self.robot.send_action(self.hold_action)
            if not self.shutdown_event.is_set():
                self.recorder.publish(
                    self.robot.get_joint_observation(),
                    self._dataset_action(performed or self.hold_action),
                    time.perf_counter(),
                    intervention=self.state == _STATE_TELEOP,
                )

    def _switch_source(self, operator: bool) -> None:
        # Revoke pending policy work before any recorder IPC or IK operation.
        self.policy.set_intervention(operator)
        rows = run_blocking_with_sync_control_ticks(
            self.recorder.pause_episode,
            self._hold,
            min(1 / self.teleop_hz, 1 / self.fps),
            drain_tick=self._hold,
        )
        self._resume_pending = True
        self.state = _STATE_TELEOP if operator else _STATE_POLICY
        if operator:
            self.interventions += 1
            self.open_span_start = rows / self.fps
            _logger.info("Operator took over — recording the correction.")
        else:
            if self.open_span_start is not None:
                self.intervention_spans.append((self.open_span_start, rows / self.fps))
                self.open_span_start = None
            self.robot.reset_cartesian_seed()
            self.handover.seed(self.hold_action)
            _logger.info(
                "Intervention over — holding while a fresh policy plan arrives."
            )

    def _prepare_policy_joints(self, action):
        # Recheck after the potentially expensive IK solve. Only this thread
        # sends motors, and shadow inference never gets command authority.
        if self.shutdown_event.is_set() or self._operator_requested():
            raise _PolicyRevoked()
        return self.handover.apply(action)

    def tick(self, t0: float) -> None:
        """A single control tick, exposed for deterministic arbitration tests."""
        self.policy.check_error()
        operator = self._operator_requested()
        if operator != (self.state == _STATE_TELEOP):
            self._switch_source(operator)
        if self.shutdown_event.is_set():
            return
        if self.state == _STATE_POLICY:
            action = self.policy.act(None)
            if action is None:
                performed = self.robot.send_action(self.hold_action)
            else:
                try:
                    performed = self.robot.send_action(
                        action, joint_transform=self._prepare_policy_joints
                    )
                except _PolicyRevoked:
                    if not self.shutdown_event.is_set():
                        self._switch_source(True)
                        self._teleop_tick(t0)
                    return
                self.policy.note_dispatched()
        else:
            self._teleop_tick(t0)
            return
        self._publish(performed or self.hold_action, t0, intervention=False)

    def _teleop_tick(self, t0: float) -> None:
        # With a lost link, get_action could still advance a stale playback
        # segment. Hold the last completed command until teleop re-engages.
        action = (
            self.teleop.get_action() if self.teleop.teleop_engaged else self.hold_action
        )
        if self.shutdown_event.is_set():
            return
        performed = self.robot.send_action(action)
        self._publish(performed or action, t0, intervention=True)

    def _publish(self, action, t0: float, *, intervention: bool) -> None:
        self.hold_action = dict(action)
        if self.shutdown_event.is_set():
            return
        self.recorder.publish(
            self.robot.get_joint_observation(),
            self._dataset_action(action),
            # A source-boundary IPC/IK operation can outlive the original
            # tick. Date this measured snapshot at publication, like holds.
            time.perf_counter(),
            intervention=intervention,
        )
        if self._resume_pending and not self.shutdown_event.is_set():
            # The new source's snapshot must exist before capture resumes.
            run_blocking_with_sync_control_ticks(
                self.recorder.resume_episode,
                self._hold,
                min(1 / self.teleop_hz, 1 / self.fps),
                drain_tick=self._hold,
            )
            self._resume_pending = False

    def run(self) -> None:
        from lerobot.teleoperators.utils import TeleopEvents

        try:
            affinity.enter_control_thread()
            deadline = time.perf_counter()
            while not self.shutdown_event.is_set():
                if error := self.recorder.poll_capture_error():
                    self.capture_error = error
                    return
                events = self.teleop.get_teleop_events()
                if events.get(TeleopEvents.TERMINATE_EPISODE):
                    self.vr_choice = "s"
                    return
                if events.get(TeleopEvents.RERECORD_EPISODE):
                    self.vr_choice = "r"
                    return
                now = time.perf_counter()
                if self.state == _STATE_POLICY and now - deadline >= 1 / self.fps:
                    self.policy.note_control_overrun()
                previous_source = self.state
                self.tick(now)
                period = 1 / (
                    self.teleop_hz if self.state == _STATE_TELEOP else self.fps
                )
                now = time.perf_counter()
                if self.state != previous_source:
                    # The boundary IPC was paced with hold commands; it did
                    # not advance policy rows or miss their execution slots.
                    deadline = now + period
                else:
                    deadline += period
                    if now >= deadline:
                        if self.state == _STATE_POLICY:
                            self.policy.note_control_overrun(
                                "policy control tick exceeded its budget"
                            )
                        deadline = now + period
                self.shutdown_event.wait(max(0.0, deadline - now))
        except Exception as exc:
            self.fatal_error = exc
            self.shutdown_event.set()
        finally:
            self.policy.pause()
