"""Unit tests for the soft-shutdown park.

A clean stop of run-policy / collect-data / replay-dataset eases the arms to
the rest pose and then to the all-zeros pose before ``robot.disconnect()``
releases the motors; a fault skips the park (never command a possibly-faulted
robot blind) and a second Ctrl+C aborts it. These tests pin the hardware-free
pieces of that contract:

- ``ResetInterpolator`` holds the gripper start values when asked
  (``hold_grippers``) and still ramps them open by default, byte-identically,
- the teleop core's ``request_zero`` dispatch sends the explicit-target reset
  message (zeros at the arm indices) and ``is_resetting`` stays True across
  the blocking plan round-trip (the ``_dispatching_reset`` fix) — a poller
  that sees a false "done" there would cut the park after one step,
- ``IKResetController.park`` clears the command-state jump guard BEFORE the
  first send, holds the grippers on both legs, honors its overall deadline,
  aborts on Ctrl+C, and skips a dead worker,
- run-policy's ``_should_park`` gates the park on a clean stop.

The IK worker subprocess itself (``run_ik_worker``) is too heavy to exercise
here (pyroki/JAX JIT), so the worker's 2-tuple/3-tuple protocol is covered
from the sending side: the core and the reset controller are run against a
fake planner on the other end of a real ``multiprocessing.Pipe``.

Self-contained; no test-framework dependency (also collects under pytest).
Invoke by direct path — ``python -m tests.test_soft_shutdown`` can shadow
installed packages with the repo-root ``tests`` package::

    PYTHONPATH=. python tests/test_soft_shutdown.py
"""

from __future__ import annotations

import logging
import multiprocessing
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import numpy as np


def _stub_pyzed() -> None:
    """The ZED SDK only exists on the robot host; the logic under test never
    touches it, but ``almond_axol.cli.run_policy`` imports the camera module
    at load time."""
    if "pyzed" in sys.modules:
        return
    pyzed = ModuleType("pyzed")
    pyzed.sl = ModuleType("pyzed.sl")
    sys.modules["pyzed"] = pyzed
    sys.modules["pyzed.sl"] = pyzed.sl


# ----------------------------------------------------------------------
# ResetInterpolator gripper behavior
# ----------------------------------------------------------------------


def test_interpolator_default_ramps_grippers_open() -> None:
    """Without ``hold_grippers`` the grippers smoothstep from their start
    values to 1.0 — the pre-existing behavior, kept byte-identical."""
    from almond_axol.teleop.filter import ResetInterpolator

    traj = [np.full(4, float(i)) for i in range(4)]
    interp = ResetInterpolator()
    interp.set_trajectory(traj, 0.2, 0.5)
    n = len(traj)
    for i in range(n):
        _q, l_grip, r_grip, done = interp.step()
        alpha = (i + 1) / n
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        assert l_grip == 0.2 + smooth * (1.0 - 0.2), f"step {i}: l_grip {l_grip}"
        assert r_grip == 0.5 + smooth * (1.0 - 0.5), f"step {i}: r_grip {r_grip}"
        assert done == (i == n - 1)
    assert l_grip == 1.0 and r_grip == 1.0, "final step must reach fully open"
    assert not interp.is_active()


def test_interpolator_hold_grippers_keeps_start_values() -> None:
    """With ``hold_grippers`` every step returns the start values unramped —
    a park must not drop a held object."""
    from almond_axol.teleop.filter import ResetInterpolator

    traj = [np.full(4, float(i)) for i in range(5)]
    interp = ResetInterpolator()
    interp.set_trajectory(traj, 0.2, 0.5, hold_grippers=True)
    while interp.is_active():
        q, l_grip, r_grip, _done = interp.step()
        assert q is not None
        assert l_grip == 0.2, f"l_grip ramped to {l_grip}"
        assert r_grip == 0.5, f"r_grip ramped to {r_grip}"


# ----------------------------------------------------------------------
# Teleop core: request_zero dispatch + the _dispatching_reset fix
# ----------------------------------------------------------------------


def _make_core():
    from almond_axol.teleop.config import VRTeleopConfig
    from almond_axol.teleop.core import VRTeleopCore

    core = VRTeleopCore(
        VRTeleopConfig(frequency=500.0), logging.getLogger("test"), lambda _on: None
    )
    q_init = np.arange(16, dtype=np.float32) + 1.0  # nonzero everywhere
    left = list(range(0, 7))
    right = list(range(8, 15))
    core.set_solution(q_init, left, right)
    core.set_initial_grips(0.25, 0.75)
    return core, q_init, left, right


def _run_dispatch(core, request):
    """Latch ``request`` and run one dispatch against a fake planner.

    Returns ``(msg, resetting_mid_plan, latched_mid_plan)`` where the ``mid``
    values are sampled while the core is blocked in its plan round-trip
    (request received, reply not yet sent).
    """
    loop_conn, planner_conn = multiprocessing.Pipe()
    stop = threading.Event()
    thread = threading.Thread(
        target=core.run_ik_loop,
        args=(loop_conn, lambda: None, stop, lambda: True, lambda _ts: None),
        daemon=True,
    )
    request()
    thread.start()
    try:
        # recv returns only after the core sent its plan request; the core is
        # now blocked awaiting our reply, with the latch already consumed —
        # exactly the window where is_resetting used to read False.
        msg = planner_conn.recv()
        resetting_mid_plan = core.is_resetting
        latched_mid_plan = core._reset_latched
        target = msg[2] if len(msg) > 2 and msg[2] is not None else np.asarray(msg[1])
        traj = [np.asarray(msg[1], dtype=np.float32), np.asarray(target, np.float32)]
        planner_conn.send(("reset_traj", np.asarray(target, np.float32), traj))
        deadline = time.perf_counter() + 5.0
        while not core.reset_interp.is_active() and time.perf_counter() < deadline:
            time.sleep(0.001)
        assert core.reset_interp.is_active(), "reply never became a trajectory"
        return msg, resetting_mid_plan, latched_mid_plan
    finally:
        stop.set()
        thread.join(timeout=5.0)
        assert not thread.is_alive(), "IK loop failed to stop"


def test_request_zero_sends_zero_target_and_holds_is_resetting() -> None:
    core, q_init, left, right = _make_core()
    msg, resetting_mid_plan, latched_mid_plan = _run_dispatch(core, core.request_zero)

    assert msg[0] == "reset" and len(msg) == 3, f"unexpected message: {msg!r}"
    target = np.asarray(msg[2])
    assert (target[left] == 0.0).all() and (target[right] == 0.0).all()
    untouched = [i for i in range(len(q_init)) if i not in left + right]
    assert (target[untouched] == q_init[untouched]).all(), (
        "zero target must only zero the arm indices"
    )

    # The _dispatching_reset fix: the latch was consumed but is_resetting must
    # still read True while the plan round-trip blocks.
    assert not latched_mid_plan, "latch should be consumed before the send"
    assert resetting_mid_plan, "is_resetting dropped False during the plan RTT"
    assert core.is_resetting, "trajectory is active; still resetting"

    # The park must not open the grippers: the zero-target dispatch loads the
    # trajectory with hold_grippers, so playback keeps the current values.
    assert core.reset_interp._hold_grippers is True
    out = core.compute_output()
    assert out is not None
    assert out[7] == 0.25 and out[15] == 0.75, "park playback moved the grippers"


def test_request_reset_keeps_rest_semantics_and_gripper_ramp() -> None:
    core, _q_init, _left, _right = _make_core()
    msg, resetting_mid_plan, _latched = _run_dispatch(core, core.request_reset)

    # Rest keeps its meaning: no explicit target (the worker plans to the
    # configured rest pose), and the grippers still ramp open.
    assert msg[0] == "reset" and len(msg) == 3 and msg[2] is None
    assert resetting_mid_plan
    assert core.reset_interp._hold_grippers is False


# ----------------------------------------------------------------------
# IKResetController.park mechanics
# ----------------------------------------------------------------------


class _FakeRobot:
    """Just enough of ``AxolRobot`` for the reset controller: cached
    positions, the sync send path, and the command-state reset."""

    def __init__(self) -> None:
        self.is_connected = True
        self._pos_l = np.zeros(8, dtype=np.float32)
        self._pos_r = np.zeros(8, dtype=np.float32)
        # Gripper start values the park must hold. Exactly representable in
        # float32, so the held value survives the float32 positions round-trip
        # and can be compared with == below.
        self._pos_l[7] = 0.25
        self._pos_r[7] = 0.5
        # Last COMMANDED gripper targets (None = nothing commanded yet, the
        # park then falls back to the measured positions above).
        self.cmd_l: float | None = None
        self.cmd_r: float | None = None
        self.calls: list[str] = []  # ordered op log
        self.sent: list[dict] = []
        self.send_delay_s = 0.0
        # Send index that raises KeyboardInterrupt (the raising SIGINT
        # handler firing mid-playback on a second Ctrl+C).
        self.interrupt_at: int | None = None

    @property
    def positions(self):
        return self._pos_l, self._pos_r

    def last_gripper_commands(self):
        return self.cmd_l, self.cmd_r

    def reset_command_state(self) -> None:
        self.calls.append("reset_command_state")
        # The real cache is cleared here — a park that reads the commanded
        # values too late gets nothing.
        self.cmd_l = None
        self.cmd_r = None

    def send_action(self, action: dict) -> dict:
        if self.interrupt_at is not None and len(self.sent) >= self.interrupt_at:
            raise KeyboardInterrupt
        if self.send_delay_s:
            time.sleep(self.send_delay_s)
        self.sent.append(dict(action))
        self.calls.append("send")
        return action


def _fake_planner(conn, requests: list, traj_len: int) -> None:
    """Answer reset requests the way ``run_ik_worker`` would: a 2-tuple plans
    to a fixed rest pose, msg[2] is an explicit target."""
    rest = np.full(20, 0.1, dtype=np.float32)
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg is None:
            return
        requests.append(msg)
        q_current = np.asarray(msg[1], dtype=np.float32)
        target = (
            np.asarray(msg[2], dtype=np.float32)
            if len(msg) > 2 and msg[2] is not None
            else rest
        )
        traj = [
            q_current + (target - q_current) * (i + 1) / traj_len
            for i in range(traj_len)
        ]
        conn.send(("reset_traj", target.copy(), traj))


def _make_controller(traj_len: int = 3, frequency: float = 1000.0):
    """A ready IKResetController wired to an in-process fake planner (no
    subprocess, no JAX): inject the ready-handshake state directly."""
    from almond_axol.lerobot import rollout
    from almond_axol.lerobot.rollout import IKResetController
    from almond_axol.teleop.config import VRTeleopConfig

    # Park announcements go through lerobot's log_say, which spawns a real
    # TTS subprocess (spd-say) when one is installed — stub the module's
    # announcer so tests stay silent and side-effect free.
    rollout._announce = lambda _text: None

    ctrl = IKResetController(vr_config=VRTeleopConfig(frequency=frequency))
    parent_conn, child_conn = multiprocessing.Pipe()
    requests: list = []
    planner = threading.Thread(
        target=_fake_planner, args=(child_conn, requests, traj_len), daemon=True
    )
    planner.start()
    ctrl._conn = parent_conn
    ctrl._q_init = np.linspace(0.5, 1.5, 20).astype(np.float32)
    ctrl._left_indices = list(range(0, 7))
    ctrl._right_indices = list(range(10, 17))
    ctrl._ready = True
    return ctrl, requests


def test_park_plays_both_legs_holding_grippers() -> None:
    traj_len = 3
    ctrl, requests = _make_controller(traj_len=traj_len)
    robot = _FakeRobot()
    assert ctrl.park(robot) is True

    # The jump-guard reset must precede the first waypoint or motion_control
    # silently drops it against the stale cached command.
    assert robot.calls[0] == "reset_command_state"
    assert robot.calls.count("reset_command_state") == 1

    # Two legs: rest first (2-tuple — the original wire format), then the
    # explicit zero target (3-tuple, zeros at the arm indices only).
    assert len(requests) == 2
    assert len(requests[0]) == 2
    assert len(requests[1]) == 3
    zero_target = np.asarray(requests[1][2])
    assert (zero_target[ctrl._left_indices] == 0.0).all()
    assert (zero_target[ctrl._right_indices] == 0.0).all()

    # Nothing was ever commanded (cmd_l/cmd_r None): the park falls back to
    # the measured positions and holds them on EVERY waypoint of both legs.
    assert len(robot.sent) == 2 * traj_len
    for action in robot.sent:
        assert action["left_gripper.pos"] == 0.25, "park moved the left gripper"
        assert action["right_gripper.pos"] == 0.5, "park moved the right gripper"


def test_park_holds_commanded_gripper_not_measured() -> None:
    # Grippers run position-force control: on a held object the measured
    # position sags below the command (the force IS that gap), so the park
    # must re-command the last commanded value — snapshotted once, before
    # reset_command_state() clears the cache — on both legs. Holding the
    # measured value instead would collapse the grip force.
    traj_len = 3
    ctrl, requests = _make_controller(traj_len=traj_len)
    robot = _FakeRobot()
    robot.cmd_l = 0.75  # commanded: squeezing …
    robot.cmd_r = 0.125
    # … while the measured positions (0.25 / 0.5) sag away from the command.
    assert ctrl.park(robot) is True
    assert len(requests) == 2, "both legs must still play"
    assert len(robot.sent) == 2 * traj_len
    for action in robot.sent:
        assert action["left_gripper.pos"] == 0.75, "left grip loosened to measured"
        assert action["right_gripper.pos"] == 0.125, "right grip loosened to measured"


def test_park_respects_overall_deadline() -> None:
    # 50 waypoints x 20 ms per send x 2 legs ~ 2 s unbounded; a 0.3 s budget
    # must stop the rest leg early and skip the zero leg entirely.
    ctrl, requests = _make_controller(traj_len=50)
    robot = _FakeRobot()
    robot.send_delay_s = 0.02
    start = time.perf_counter()
    result = ctrl.park(robot, deadline_s=0.3)
    elapsed = time.perf_counter() - start
    assert result is False, "an out-of-budget park must not report success"
    assert elapsed < 1.5, f"deadline not enforced: park took {elapsed:.2f}s"
    assert len(requests) == 1, "zero leg must be skipped once the budget is spent"
    # Timing-insensitive check that the IN-PLAYBACK break fired: 0.3 s at
    # ~20 ms/waypoint is ~15 waypoints. Without the per-waypoint deadline
    # check the rest leg would play all 50 before the budget is consulted.
    assert len(robot.sent) <= 25, (
        f"rest leg played {len(robot.sent)} waypoints past the deadline"
    )


def test_park_caps_unanswered_plan_round_trip() -> None:
    # A wedged worker (alive, never answering) must not hang the park in the
    # plan recv: the remaining budget caps the round-trip and the leg fails.
    from almond_axol.lerobot import rollout
    from almond_axol.lerobot.rollout import IKResetController
    from almond_axol.teleop.config import VRTeleopConfig

    rollout._announce = lambda _text: None
    ctrl = IKResetController(vr_config=VRTeleopConfig(frequency=1000.0))
    parent_conn, child_conn = multiprocessing.Pipe()
    ctrl._conn = parent_conn
    ctrl._mute_worker_conn = child_conn  # keep the other end alive (no EOF)
    ctrl._q_init = np.linspace(0.5, 1.5, 20).astype(np.float32)
    ctrl._left_indices = list(range(0, 7))
    ctrl._right_indices = list(range(10, 17))
    ctrl._ready = True
    robot = _FakeRobot()
    start = time.perf_counter()
    assert ctrl.park(robot, deadline_s=0.3) is False
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"unanswered plan hung the park for {elapsed:.2f}s"
    assert not robot.sent, "nothing must be commanded without a planned move"


def test_park_aborts_on_keyboard_interrupt() -> None:
    ctrl, requests = _make_controller(traj_len=10)
    robot = _FakeRobot()
    robot.interrupt_at = 2  # third waypoint: the raising handler fires
    assert ctrl.park(robot) is False
    assert len(robot.sent) == 2, "playback must stop at the interrupt"
    assert len(requests) == 1, "an aborted rest leg must not start the zero leg"


def test_park_skips_dead_worker_and_missing_conn() -> None:
    ctrl, requests = _make_controller()
    robot = _FakeRobot()
    ctrl._proc = SimpleNamespace(is_alive=lambda: False)  # SIGKILLed worker
    assert ctrl.park(robot) is False
    assert not requests and not robot.sent

    ctrl2, requests2 = _make_controller()
    ctrl2._conn = None  # stop() already ran
    assert ctrl2.park(robot) is False
    assert not requests2 and not robot.sent


def test_return_to_rest_behavior_unchanged() -> None:
    # The between-episode reset still sends the 2-tuple, ramps the grippers
    # open, and never touches the command-state guard.
    traj_len = 4
    ctrl, requests = _make_controller(traj_len=traj_len)
    robot = _FakeRobot()
    ctrl.return_to_rest(robot)
    assert len(requests) == 1 and len(requests[0]) == 2
    assert "reset_command_state" not in robot.calls
    assert len(robot.sent) == traj_len
    assert robot.sent[-1]["left_gripper.pos"] == 1.0
    assert robot.sent[-1]["right_gripper.pos"] == 1.0


# ----------------------------------------------------------------------
# run-policy park gating
# ----------------------------------------------------------------------


def test_should_park_gating() -> None:
    _stub_pyzed()
    from almond_axol.cli.run_policy import _should_park

    cfg = SimpleNamespace(park_on_exit=True)
    robot = SimpleNamespace(is_connected=True)
    ok_client = SimpleNamespace(fatal_error=None)
    clean = (None, None, None)
    fault = (RuntimeError, RuntimeError("boom"), None)

    assert _should_park(cfg, ok_client, robot, clean) is True
    assert _should_park(cfg, None, robot, clean) is True, "no client is still clean"

    bad_client = SimpleNamespace(fatal_error=RuntimeError("motor fault"))
    assert _should_park(cfg, bad_client, robot, clean) is False
    assert _should_park(cfg, ok_client, robot, fault) is False

    off = SimpleNamespace(park_on_exit=False)
    assert _should_park(off, ok_client, robot, clean) is False

    # A Ctrl+C that escaped the episode loop's handler may have skipped the
    # per-episode thread joins — never park over a possibly-live commander.
    assert _should_park(cfg, ok_client, robot, clean, hard_interrupt=True) is False

    disconnected = SimpleNamespace(is_connected=False)
    assert _should_park(cfg, ok_client, disconnected, clean) is False


def main() -> int:
    tests = [
        (name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")
    ]
    failures = 0
    for name, fn in tests:
        start = time.perf_counter()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {exc!r}")
        else:
            print(f"ok   {name} ({time.perf_counter() - start:.2f}s)")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
