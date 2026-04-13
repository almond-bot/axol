"""Cycle one joint through its limits while holding all others at their start position.

Run directly:
    python -m almond_axol.test.can.send --l --joint shoulder_1
    python -m almond_axol.test.can.send --r --joint elbow
    python -m almond_axol.test.can.send --joint elbow        # both arms, log only
    python -m almond_axol.test.can.send --l --joint wrist_2 --hz 50
    python -m almond_axol.test.can.send --l --joint gripper --hz 100 --log-file can_send.log
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

from ...motor import CanBus
from ...robot.axol import LIMITS, AxolArm, arm_limits
from ...robot.config import AxolConfig
from ...shared import CAN_LEFT, CAN_RIGHT, Joint

_BAR_WIDTH = 24
_TAU = 2 * math.pi
_DISPLAY_HZ = 30
_COL_WIDTH = 60
_COL_GAP = 2

# Consistent with home.py and gripper.py.
_SPEED = 0.2 * _TAU  # rad/s for arm joints
_GRIPPER_RANGE = abs(LIMITS[Joint.GRIPPER][0] - LIMITS[Joint.GRIPPER][1])


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


async def _stats_monitor(channel: str, arm: AxolArm, log: logging.Logger) -> None:
    """Background task: log CAN interface stats and position staleness every second."""
    prev_positions: np.ndarray | None = None
    stale_count = 0
    update_count = 0
    interval_start = time.perf_counter()

    while True:
        await asyncio.sleep(1.0)
        now = time.perf_counter()
        elapsed = now - interval_start
        interval_start = now

        positions = arm.positions

        if prev_positions is not None:
            if np.allclose(positions, prev_positions, atol=1e-6):
                stale_count += 1
            else:
                update_count += 1
        prev_positions = positions.copy()

        can_stats = _read_can_stats(channel)
        log.info(
            "--- 1s interval (%.2fs) | pos_updates=%d stale_checks=%d ---\n%s",
            elapsed,
            update_count,
            stale_count,
            can_stats,
        )
        update_count = 0
        stale_count = 0


@dataclass
class _SendSnapshot:
    side: str
    hz: int
    log_file: str
    is_left: bool
    cycle_joint: Joint
    positions: np.ndarray = field(default_factory=lambda: np.zeros(len(list(Joint))))
    cycle_count: int = 0
    send_error_count: int = 0
    timeout_error_count: int = 0
    other_error_count: int = 0
    segment_start: float = 0.0
    segment_target: float = 0.0
    alpha: float = 0.0


def _arm_lines(snap: _SendSnapshot) -> list[str]:
    joints = list(Joint)
    lines = [
        f"  {snap.side.upper()} ARM  [{snap.hz} Hz]  cycling={snap.cycle_joint.value}"
        f"  log→{snap.log_file}",
        (
            f"  cycles={snap.cycle_count}  send_err={snap.send_error_count}"
            f"  timeout_err={snap.timeout_error_count}"
            f"  other_err={snap.other_error_count}"
        ),
        f"  segment: {snap.segment_start:+.4f} → {snap.segment_target:+.4f}"
        f"  α={snap.alpha:.2f}",
        f"  {'Joint':<12}  {'rev':>8}  {'':^{_BAR_WIDTH}}",
        "  " + "─" * (12 + 8 + _BAR_WIDTH + 4),
    ]
    for i, joint in enumerate(joints):
        lo, hi = arm_limits(joint, is_left=snap.is_left)
        p = float(snap.positions[i])
        marker = " ◀" if joint == snap.cycle_joint else ""
        lines.append(
            f"  {joint.value:<12}  {p / _TAU:>+8.4f}  {_bar(p, lo, hi)}{marker}"
        )
    return lines


async def _display_both(left: _SendSnapshot, right: _SendSnapshot) -> None:
    right_col = _COL_WIDTH + _COL_GAP + 1
    print("\033[?25l\033[2J", end="", flush=True)
    try:
        while True:
            left_lines = _arm_lines(left)
            right_lines = _arm_lines(right)
            n_rows = max(len(left_lines), len(right_lines))

            buf: list[str] = []
            for row in range(n_rows):
                if row < len(left_lines):
                    cell = left_lines[row][:_COL_WIDTH].ljust(_COL_WIDTH)
                    buf.append(f"\033[{row + 1};1H{cell}")
                if row < len(right_lines):
                    buf.append(f"\033[{row + 1};{right_col}H{right_lines[row]}\033[K")

            buf.append(f"\033[{n_rows + 2};1H  ctrl+c to quit\033[K")
            print("".join(buf), end="", flush=True)
            await asyncio.sleep(1.0 / _DISPLAY_HZ)
    except asyncio.CancelledError:
        pass
    finally:
        print("\033[?25h", end="", flush=True)


def _cycle_dist_rad(dist_api: float, joint: Joint) -> float:
    """Convert an API-unit distance to radians for speed/duration calculations."""
    if joint == Joint.GRIPPER:
        return abs(dist_api) * _GRIPPER_RANGE
    return abs(dist_api)


async def _run(
    is_left: bool,
    cycle_joint: Joint,
    hz: int,
    log_file: str,
    display: bool = True,
    snapshot: _SendSnapshot | None = None,
) -> None:
    side = "left" if is_left else "right"
    log = _make_logger(log_file, f"{__name__}.{side}")

    def _asyncio_exc_handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
        exc = context.get("exception")
        msg = context.get("message", "(no message)")
        if exc is not None:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            log.error("Unhandled asyncio exception: %s\n%s", msg, tb)
        else:
            log.error("Unhandled asyncio error: %s | context=%s", msg, context)

    asyncio.get_running_loop().set_exception_handler(_asyncio_exc_handler)

    joints = list(Joint)
    joint_idx = joints.index(cycle_joint)
    channel = CAN_LEFT if is_left else CAN_RIGHT

    # Limits in API units (gripper = [0, 1]; arm joints = radians).
    if cycle_joint == Joint.GRIPPER:
        lo_api, hi_api = 0.0, 1.0
    else:
        lo_api, hi_api = arm_limits(cycle_joint, is_left=is_left)

    log.info(
        "Starting  side=%s  channel=%s  joint=%s  hz=%d  limits=[%.4f, %.4f]",
        side,
        channel,
        cycle_joint.value,
        hz,
        lo_api,
        hi_api,
    )
    log.info("Initial CAN stats:\n%s", _read_can_stats(channel))

    t_start = time.perf_counter()
    cycle_count = 0
    send_error_count = 0
    timeout_error_count = 0
    other_error_count = 0

    stats_task: asyncio.Task | None = None

    try:
        async with CanBus(channel) as bus:
            arm = AxolArm(bus, AxolConfig(), is_left=is_left)

            stats_task = asyncio.create_task(
                _stats_monitor(channel, arm, log), name="can_stats_monitor"
            )

            try:
                await arm.enable()
                log.info("Motors enabled")
            except Exception as exc:
                log.error("enable failed: %s\n%s", exc, traceback.format_exc())
                raise

            hold_q = await arm.get_positions()
            cycle_start = float(hold_q[joint_idx])
            log.info(
                "Initial positions read. cycle_joint=%s  start=%.4f",
                cycle_joint.value,
                cycle_start,
            )

            # Cycle: start → hi → lo → hi → lo → ...
            # Pick whichever limit is further first for a fuller first sweep.
            if abs(hi_api - cycle_start) >= abs(lo_api - cycle_start):
                targets = [hi_api, lo_api]
            else:
                targets = [lo_api, hi_api]

            target_idx = 0
            segment_start = cycle_start
            segment_target = targets[0]
            dist_rad = _cycle_dist_rad(segment_target - segment_start, cycle_joint)
            duration = max(dist_rad / _SPEED, 0.05)
            t_seg = time.perf_counter()

            if display:
                print("\033[?25l", end="")
            last_stat_log = time.perf_counter()
            last_display = 0.0
            interval = 1.0 / hz

            try:
                while True:
                    cycle_count += 1
                    t_iter = time.perf_counter()

                    now = t_iter
                    alpha = min((now - t_seg) / duration, 1.0)
                    smooth = alpha * alpha * (3.0 - 2.0 * alpha)
                    cycle_pos = segment_start + smooth * (
                        segment_target - segment_start
                    )

                    q = hold_q.copy()
                    q[joint_idx] = cycle_pos

                    try:
                        await arm.motion_control(q)
                    except Exception as exc:
                        send_error_count += 1
                        log.error(
                            "motion_control failed (cycle=%d): %s\n%s",
                            cycle_count,
                            exc,
                            traceback.format_exc(),
                        )

                    # Read back positions for display; fall back to hold_q until
                    # motion_control feedback has arrived for all motors.
                    try:
                        positions = arm.positions
                    except Exception:
                        positions = hold_q

                    if snapshot is not None:
                        snapshot.positions = positions.copy()
                        snapshot.cycle_count = cycle_count
                        snapshot.send_error_count = send_error_count
                        snapshot.timeout_error_count = timeout_error_count
                        snapshot.other_error_count = other_error_count
                        snapshot.segment_start = segment_start
                        snapshot.segment_target = segment_target
                        snapshot.alpha = alpha

                    if display and now - last_display >= 1.0 / _DISPLAY_HZ:
                        lines = []
                        lines.append("\033[H\033[J")
                        lines.append(
                            f"  {side.upper()} ARM  [{hz} Hz]  cycling={cycle_joint.value}"
                            f"  log→{log_file}"
                        )
                        lines.append(
                            f"  cycles={cycle_count}  send_err={send_error_count}"
                            f"  timeout_err={timeout_error_count}"
                            f"  other_err={other_error_count}"
                        )
                        lines.append(
                            f"  segment: {segment_start:+.4f} → {segment_target:+.4f}"
                            f"  α={alpha:.2f}"
                        )
                        lines.append(f"  {'Joint':<12}  {'rev':>8}  {'':^{_BAR_WIDTH}}")
                        lines.append("  " + "─" * (12 + 8 + _BAR_WIDTH + 4))

                        for i, joint in enumerate(joints):
                            lo, hi = arm_limits(joint, is_left=is_left)
                            p = float(positions[i])
                            marker = " ◀" if joint == cycle_joint else ""
                            lines.append(
                                f"  {joint.value:<12}  {p / _TAU:>+8.4f}"
                                f"  {_bar(p, lo, hi)}{marker}"
                            )

                        lines.append("")
                        lines.append("  ctrl+c to quit")
                        print("\n".join(lines), end="", flush=True)
                        last_display = now

                    # Log per-cycle timing to file every 10 seconds.
                    if now - last_stat_log >= 10.0:
                        elapsed_total = now - t_start
                        log.info(
                            "CYCLE STATS  elapsed=%.1fs  cycles=%d  actual_hz=%.1f"
                            "  send_err=%d  timeout_err=%d  other_err=%d",
                            elapsed_total,
                            cycle_count,
                            cycle_count / elapsed_total,
                            send_error_count,
                            timeout_error_count,
                            other_error_count,
                        )
                        last_stat_log = now

                    # Advance to next segment when current one completes.
                    if alpha >= 1.0:
                        segment_start = segment_target
                        target_idx += 1
                        segment_target = targets[target_idx % 2]
                        dist_rad = _cycle_dist_rad(
                            segment_target - segment_start, cycle_joint
                        )
                        duration = max(dist_rad / _SPEED, 0.05)
                        t_seg = time.perf_counter()
                        log.info(
                            "New segment: %.4f → %.4f  duration=%.2fs",
                            segment_start,
                            segment_target,
                            duration,
                        )

                    elapsed = time.perf_counter() - t_iter
                    await asyncio.sleep(max(0.0, interval - elapsed))

            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
            finally:
                if display:
                    print("\033[?25h")
                await arm.disable()

    except Exception as exc:
        log.error("Fatal error in _run: %s\n%s", exc, traceback.format_exc())
        raise
    finally:
        if stats_task is not None and not stats_task.done():
            stats_task.cancel()
            try:
                await stats_task
            except asyncio.CancelledError:
                pass

        elapsed_total = time.perf_counter() - t_start
        log.info(
            "FINAL STATS  elapsed=%.1fs  cycles=%d  actual_hz=%.1f"
            "  send_err=%d  timeout_err=%d  other_err=%d",
            elapsed_total,
            cycle_count,
            cycle_count / elapsed_total if elapsed_total > 0 else 0.0,
            send_error_count,
            timeout_error_count,
            other_error_count,
        )
        log.info("Final CAN stats:\n%s", _read_can_stats(channel))


def main() -> None:
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
        "--hz", type=int, default=100, help="Control rate in Hz (default: 100)"
    )
    parser.add_argument(
        "--log-file",
        default=f"logs/can_send_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        help="Path for the diagnostic log file",
    )
    args = parser.parse_args()

    cycle_joint = Joint(args.joint)

    try:
        if not args.l and not args.r:
            stem, _, ext = args.log_file.rpartition(".")
            left_log = f"{stem}_left.{ext}"
            right_log = f"{stem}_right.{ext}"
            print("No side specified — running both arms.")
            print(f"  left  → {left_log}")
            print(f"  right → {right_log}")

            joints = list(Joint)
            left_snap = _SendSnapshot(
                side="left",
                hz=args.hz,
                log_file=left_log,
                is_left=True,
                cycle_joint=cycle_joint,
                positions=np.zeros(len(joints)),
            )
            right_snap = _SendSnapshot(
                side="right",
                hz=args.hz,
                log_file=right_log,
                is_left=False,
                cycle_joint=cycle_joint,
                positions=np.zeros(len(joints)),
            )

            async def _run_both() -> None:
                display_task = asyncio.create_task(_display_both(left_snap, right_snap))
                try:
                    await asyncio.gather(
                        _run(
                            is_left=True,
                            cycle_joint=cycle_joint,
                            hz=args.hz,
                            log_file=left_log,
                            display=False,
                            snapshot=left_snap,
                        ),
                        _run(
                            is_left=False,
                            cycle_joint=cycle_joint,
                            hz=args.hz,
                            log_file=right_log,
                            display=False,
                            snapshot=right_snap,
                        ),
                    )
                finally:
                    display_task.cancel()
                    try:
                        await display_task
                    except asyncio.CancelledError:
                        pass

            asyncio.run(_run_both())
        else:
            asyncio.run(
                _run(
                    is_left=args.l,
                    cycle_joint=cycle_joint,
                    hz=args.hz,
                    log_file=args.log_file,
                )
            )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
