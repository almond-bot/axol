"""Cycle one joint through its limits while holding all others at their start position.

The arms are driven through the Rust realtime core (``RtAxol``) — the same
control path as teleop — so what this exercises is the production controller
and its CAN traffic, not a Python-side loop. Every motor of each selected arm
must be on the bus (the core brings the whole arm up); only the chosen joint
moves.

Run directly:
    uv run -m almond_axol.diagnostics.can.send --l --joint shoulder_1
    uv run -m almond_axol.diagnostics.can.send --r --joint elbow
    uv run -m almond_axol.diagnostics.can.send --joint elbow        # both arms
    uv run -m almond_axol.diagnostics.can.send --l --joint wrist_2 --hz 50
    uv run -m almond_axol.diagnostics.can.send --l --joint gripper --hz 100 --log-file can_send.log
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ...constants import CAN_LEFT, CAN_RIGHT, Joint
from ...robot.axol import GRIPPER_TRAVEL, Axol, arm_limits
from ...robot.config import AxolConfig
from ...rt import RtAxol

_BAR_WIDTH = 24
_TAU = 2 * math.pi
_DISPLAY_HZ = 30
_COL_WIDTH = 60
_COL_GAP = 2

# Consistent with home.py and gripper.py.
_SPEED = 0.2 * _TAU  # rad/s


def _make_logger(log_file: str, name: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    fmt = "%(asctime)s.%(msecs)03d  %(levelname)-7s  %(message)s"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.info("Logging started → %s", log_file)
    return logger


def _bar(value: float, lo: float, hi: float) -> str:
    if math.isclose(lo, hi):
        return "─" * _BAR_WIDTH
    frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    pos = round(frac * _BAR_WIDTH)
    bar = list("░" * _BAR_WIDTH)
    bar[max(0, min(_BAR_WIDTH - 1, pos))] = "█"
    return "".join(bar)


def _read_can_stats(channel: str) -> str:
    """Run `ip -s -details link show <channel>` and return the output."""
    try:
        result = subprocess.run(
            ["ip", "-s", "-details", "link", "show", channel],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return result.stdout.rstrip()
    except Exception as exc:
        return f"(failed to read stats: {exc})"


@dataclass
class _ArmCycle:
    """One arm's cycling state: the sweep segment plus display counters."""

    side: str
    channel: str
    is_left: bool
    cycle_joint: Joint
    hold_q: np.ndarray
    lo_api: float
    hi_api: float
    targets: list[float] = field(default_factory=list)
    target_idx: int = 0
    segment_start: float = 0.0
    segment_target: float = 0.0
    duration: float = 0.05
    t_seg: float = 0.0
    alpha: float = 0.0
    positions: np.ndarray = field(default_factory=lambda: np.zeros(len(list(Joint))))
    stale_checks: int = 0
    pos_updates: int = 0
    _prev_positions: np.ndarray | None = None

    def start(self, log: logging.Logger) -> None:
        joint_idx = list(Joint).index(self.cycle_joint)
        cycle_start = float(self.hold_q[joint_idx])
        log.info(
            "%s: initial positions read. cycle_joint=%s  start=%.4f",
            self.side,
            self.cycle_joint.value,
            cycle_start,
        )
        # Cycle: start → hi → lo → hi → lo → ...
        # Pick whichever limit is further first for a fuller first sweep.
        if abs(self.hi_api - cycle_start) >= abs(self.lo_api - cycle_start):
            self.targets = [self.hi_api, self.lo_api]
        else:
            self.targets = [self.lo_api, self.hi_api]
        self.segment_start = cycle_start
        self.segment_target = self.targets[0]
        self._plan_segment()

    def _plan_segment(self) -> None:
        dist_rad = _cycle_dist_rad(
            self.segment_target - self.segment_start, self.cycle_joint
        )
        self.duration = max(dist_rad / _SPEED, 0.05)
        self.t_seg = time.perf_counter()

    def target(self, now: float) -> np.ndarray:
        """The full-arm target for this tick (hold pose + cycled joint)."""
        self.alpha = min((now - self.t_seg) / self.duration, 1.0)
        smooth = self.alpha * self.alpha * (3.0 - 2.0 * self.alpha)
        q = self.hold_q.copy()
        q[list(Joint).index(self.cycle_joint)] = self.segment_start + smooth * (
            self.segment_target - self.segment_start
        )
        return q

    def advance(self, log: logging.Logger) -> None:
        """Move to the next segment once the current one has completed."""
        if self.alpha < 1.0:
            return
        self.segment_start = self.segment_target
        self.target_idx += 1
        self.segment_target = self.targets[self.target_idx % 2]
        self._plan_segment()
        log.info(
            "%s: new segment: %.4f → %.4f  duration=%.2fs",
            self.side,
            self.segment_start,
            self.segment_target,
            self.duration,
        )

    def observe(self, positions: np.ndarray | None) -> None:
        """Track feedback staleness for the 1 s stats log."""
        if positions is None:
            return
        self.positions = positions.copy()
        if self._prev_positions is not None:
            if np.allclose(positions, self._prev_positions, atol=1e-6):
                self.stale_checks += 1
            else:
                self.pos_updates += 1
        self._prev_positions = positions.copy()


def _arm_lines(
    arm: _ArmCycle, hz: int, cycles: int, errors: int, log_file: str
) -> list[str]:
    joints = list(Joint)
    lines = [
        f"  {arm.side.upper()} ARM  [{hz} Hz]  cycling={arm.cycle_joint.value}  log→{log_file}",
        f"  cycles={cycles}  send_err={errors}",
        (
            f"  segment: {arm.segment_start:+.4f} → {arm.segment_target:+.4f}"
            f"  α={arm.alpha:.2f}"
        ),
        f"  {'Joint':<12}  {'rev':>8}  {'':^{_BAR_WIDTH}}",
        "  " + "─" * (12 + 8 + _BAR_WIDTH + 4),
    ]
    for i, joint in enumerate(joints):
        lo, hi = arm_limits(joint, is_left=arm.is_left)
        p = float(arm.positions[i])
        marker = " ◀" if joint == arm.cycle_joint else ""
        lines.append(
            f"  {joint.value:<12}  {p / _TAU:>+8.4f}  {_bar(p, lo, hi)}{marker}"
        )
    return lines


def _render(
    arms: list[_ArmCycle], hz: int, cycles: int, errors: int, log_file: str
) -> None:
    columns = [_arm_lines(arm, hz, cycles, errors, log_file) for arm in arms]
    n_rows = max(len(col) for col in columns)
    buf: list[str] = ["\033[H\033[J"]
    for row in range(n_rows):
        for c, col in enumerate(columns):
            if row < len(col):
                x = 1 + c * (_COL_WIDTH + _COL_GAP)
                cell = col[row][:_COL_WIDTH].ljust(_COL_WIDTH)
                buf.append(f"\033[{row + 1};{x}H{cell}")
    buf.append(f"\033[{n_rows + 2};1H  ctrl+c to quit\033[K")
    print("".join(buf), end="", flush=True)


def _cycle_dist_rad(dist_api: float, joint: Joint) -> float:
    """Convert an API-unit distance to radians for speed/duration calculations."""
    if joint == Joint.GRIPPER:
        return abs(dist_api) * GRIPPER_TRAVEL
    return abs(dist_api)


async def _run(
    run_left: bool,
    run_right: bool,
    cycle_joint: Joint,
    hz: int,
    log_file: str,
    display: bool = True,
) -> None:
    log = _make_logger(log_file, __name__)

    def _asyncio_exc_handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
        exc = context.get("exception")
        msg = context.get("message", "(no message)")
        if exc is not None:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            log.error("Unhandled asyncio exception: %s\n%s", msg, tb)
        else:
            log.error("Unhandled asyncio error: %s | context=%s", msg, context)

    asyncio.get_running_loop().set_exception_handler(_asyncio_exc_handler)

    def limits(is_left: bool) -> tuple[float, float]:
        # Limits in API units (gripper = [0, 1]; arm joints = radians).
        if cycle_joint == Joint.GRIPPER:
            return 0.0, 1.0
        return arm_limits(cycle_joint, is_left=is_left)

    channels = [
        (side, ch, is_left)
        for side, ch, is_left, on in (
            ("left", CAN_LEFT, True, run_left),
            ("right", CAN_RIGHT, False, run_right),
        )
        if on
    ]
    for side, ch, is_left in channels:
        lo, hi = limits(is_left)
        log.info(
            "Starting  side=%s  channel=%s  joint=%s  hz=%d  limits=[%.4f, %.4f]",
            side,
            ch,
            cycle_joint.value,
            hz,
            lo,
            hi,
        )
        log.info("Initial CAN stats (%s):\n%s", ch, _read_can_stats(ch))

    t_start = time.perf_counter()
    cycle_count = 0
    send_error_count = 0

    # ``resolved()`` applies the default stiffness blend at the ``Axol``
    # construction boundary, so the core runs the same gains teleop does.
    robot = RtAxol(
        Axol(
            config=AxolConfig(),
            left_channel=CAN_LEFT if run_left else None,
            right_channel=CAN_RIGHT if run_right else None,
        )
    )
    try:
        try:
            await robot.enable()
            log.info("Motors enabled (realtime core armed)")
        except Exception as exc:
            log.error("enable failed: %s\n%s", exc, traceback.format_exc())
            raise

        hold_left, hold_right = await robot.get_positions()
        arms: list[_ArmCycle] = []
        for side, ch, is_left in channels:
            hold_q = hold_left if is_left else hold_right
            assert hold_q is not None
            lo, hi = limits(is_left)
            arm = _ArmCycle(side, ch, is_left, cycle_joint, hold_q.copy(), lo, hi)
            arm.start(log)
            arms.append(arm)

        if display:
            print("\033[?25l", end="")
        last_stat_log = time.perf_counter()
        last_display = 0.0
        interval = 1.0 / hz

        try:
            while True:
                if robot.limp is not None:
                    log.error("realtime core went limp: %s — stopping", robot.limp)
                    print(
                        f"\nRealtime core went limp ({robot.limp}) — arms are in "
                        "gravity comp; hand-guide them to rest."
                    )
                    break
                cycle_count += 1
                t_iter = time.perf_counter()
                now = t_iter

                targets = {arm.side: arm.target(now) for arm in arms}
                try:
                    await robot.motion_control(
                        left=targets.get("left"), right=targets.get("right")
                    )
                except Exception as exc:
                    send_error_count += 1
                    log.error(
                        "motion_control failed (cycle=%d): %s\n%s",
                        cycle_count,
                        exc,
                        traceback.format_exc(),
                    )

                # Read back positions for display; the core's telemetry fills
                # the caches every tick.
                pos_left, pos_right = await robot.get_positions()
                for arm in arms:
                    arm.observe(pos_left if arm.is_left else pos_right)

                if display and now - last_display >= 1.0 / _DISPLAY_HZ:
                    _render(arms, hz, cycle_count, send_error_count, log_file)
                    last_display = now

                if now - last_stat_log >= 1.0:
                    elapsed_total = now - t_start
                    for arm in arms:
                        log.info(
                            "--- %s 1s interval | pos_updates=%d stale_checks=%d ---\n%s",
                            arm.side,
                            arm.pos_updates,
                            arm.stale_checks,
                            _read_can_stats(arm.channel),
                        )
                        arm.pos_updates = 0
                        arm.stale_checks = 0
                    log.info(
                        "CYCLE STATS  elapsed=%.1fs  cycles=%d  actual_hz=%.1f  send_err=%d",
                        elapsed_total,
                        cycle_count,
                        cycle_count / elapsed_total,
                        send_error_count,
                    )
                    last_stat_log = now

                for arm in arms:
                    arm.advance(log)

                elapsed = time.perf_counter() - t_iter
                await asyncio.sleep(max(0.0, interval - elapsed))

        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            if display:
                print("\033[?25h")
            await robot.disable()

    except Exception as exc:
        log.error("Fatal error in _run: %s\n%s", exc, traceback.format_exc())
        raise
    finally:
        elapsed_total = time.perf_counter() - t_start
        log.info(
            "FINAL STATS  elapsed=%.1fs  cycles=%d  actual_hz=%.1f  send_err=%d",
            elapsed_total,
            cycle_count,
            cycle_count / elapsed_total if elapsed_total > 0 else 0.0,
            send_error_count,
        )
        for _side, ch, _is_left in channels:
            log.info("Final CAN stats (%s):\n%s", ch, _read_can_stats(ch))


def main() -> None:
    """Parse CLI arguments and cycle the selected joint on one or both arms."""
    valid_joints = [j.value for j in Joint]
    parser = argparse.ArgumentParser(
        description="Cycle one joint through its limits via motion control."
    )
    side = parser.add_mutually_exclusive_group()
    side.add_argument("--l", action="store_true", help="Use left arm")
    side.add_argument("--r", action="store_true", help="Use right arm")
    parser.add_argument(
        "--joint",
        required=True,
        choices=valid_joints,
        metavar="JOINT",
        help=f"Joint to cycle. One of: {', '.join(valid_joints)}",
    )
    parser.add_argument(
        "--hz", type=int, default=100, help="Target stream rate in Hz (default: 100)"
    )
    parser.add_argument(
        "--log-file",
        default=f"logs/can_send_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        help="Path for the diagnostic log file",
    )
    args = parser.parse_args()

    cycle_joint = Joint(args.joint)
    run_left = args.l or not args.r
    run_right = args.r or not args.l
    if run_left and run_right:
        print("No side specified — running both arms.")

    try:
        asyncio.run(
            _run(
                run_left=run_left,
                run_right=run_right,
                cycle_joint=cycle_joint,
                hz=args.hz,
                log_file=args.log_file,
            )
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
