"""Cycle one joint through its limits while holding all others at their start position.

Run directly:
    python -m almond_axol.test.can.send --l --joint shoulder_1
    python -m almond_axol.test.can.send --r --joint elbow
    python -m almond_axol.test.can.send --l --joint wrist_2 --hz 50
    python -m almond_axol.test.can.send --l --joint gripper --hz 100 --log-file can_send.log
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import subprocess
import time
import traceback
from datetime import datetime

import numpy as np

from ...motor import CanBus
from ...robot.axol import AxolArm, arm_limits
from ...robot.config import AxolConfig
from ...shared import CAN_LEFT, CAN_RIGHT, Joint

_BAR_WIDTH = 24
_TAU = 2 * math.pi

# Consistent with home.py and gripper.py.
_SPEED = 0.2 * _TAU          # rad/s for arm joints
_GRIPPER_RANGE = 0.8037 * _TAU  # rad, full open-to-close travel

_logger = logging.getLogger(__name__)


def _setup_logging(log_file: str) -> None:
    fmt = "%(asctime)s.%(msecs)03d  %(levelname)-7s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )
    _logger.info("Logging started → %s", log_file)


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


async def _stats_monitor(channel: str, arm: AxolArm) -> None:
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
        _logger.info(
            "--- 1s interval (%.2fs) | pos_updates=%d stale_checks=%d ---\n%s",
            elapsed,
            update_count,
            stale_count,
            can_stats,
        )
        update_count = 0
        stale_count = 0


def _cycle_dist_rad(dist_api: float, joint: Joint) -> float:
    """Convert an API-unit distance to radians for speed/duration calculations."""
    if joint == Joint.GRIPPER:
        return abs(dist_api) * _GRIPPER_RANGE
    return abs(dist_api)


async def _run(is_left: bool, cycle_joint: Joint, hz: int, log_file: str) -> None:
    _setup_logging(log_file)

    def _asyncio_exc_handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
        exc = context.get("exception")
        msg = context.get("message", "(no message)")
        if exc is not None:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            _logger.error("Unhandled asyncio exception: %s\n%s", msg, tb)
        else:
            _logger.error("Unhandled asyncio error: %s | context=%s", msg, context)

    asyncio.get_running_loop().set_exception_handler(_asyncio_exc_handler)

    joints = list(Joint)
    joint_idx = joints.index(cycle_joint)
    side = "left" if is_left else "right"
    channel = CAN_LEFT if is_left else CAN_RIGHT

    # Limits in API units (gripper = [0, 1]; arm joints = radians).
    if cycle_joint == Joint.GRIPPER:
        lo_api, hi_api = 0.0, 1.0
    else:
        lo_api, hi_api = arm_limits(cycle_joint, is_left=is_left)

    _logger.info(
        "Starting send  side=%s  channel=%s  joint=%s  hz=%d  limits=[%.4f, %.4f]",
        side, channel, cycle_joint.value, hz, lo_api, hi_api,
    )
    _logger.info("Initial CAN stats:\n%s", _read_can_stats(channel))

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
                _stats_monitor(channel, arm), name="can_stats_monitor"
            )

            try:
                await arm.start_telemetry(hz)
                _logger.info("Telemetry started at %d Hz", hz)
            except Exception as exc:
                _logger.error(
                    "start_telemetry failed: %s\n%s", exc, traceback.format_exc()
                )
                raise

            await asyncio.sleep(0.1)

            # Read initial positions; all joints hold here except the cycling one.
            hold_q = await arm.get_positions()
            cycle_start = float(hold_q[joint_idx])
            _logger.info(
                "Initial positions read. cycle_joint=%s  start=%.4f",
                cycle_joint.value, cycle_start,
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

            print("\033[?25l", end="")
            last_stat_log = time.perf_counter()

            try:
                while True:
                    cycle_count += 1

                    now = time.perf_counter()
                    alpha = min((now - t_seg) / duration, 1.0)
                    smooth = alpha * alpha * (3.0 - 2.0 * alpha)
                    cycle_pos = segment_start + smooth * (segment_target - segment_start)

                    q = hold_q.copy()
                    q[joint_idx] = cycle_pos

                    try:
                        await arm.motion_control(q)
                    except Exception as exc:
                        send_error_count += 1
                        _logger.error(
                            "motion_control failed (cycle=%d): %s\n%s",
                            cycle_count, exc, traceback.format_exc(),
                        )

                    # Read back positions for display.
                    try:
                        positions = arm.positions
                    except Exception as exc:
                        other_error_count += 1
                        _logger.error(
                            "arm.positions failed (cycle=%d): %s\n%s",
                            cycle_count, exc, traceback.format_exc(),
                        )
                        positions = np.zeros(len(joints), dtype=np.float32)

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

                    # Log per-cycle timing to file every 10 seconds.
                    if now - last_stat_log >= 10.0:
                        elapsed_total = now - t_start
                        _logger.info(
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
                        _logger.info(
                            "New segment: %.4f → %.4f  duration=%.2fs",
                            segment_start, segment_target, duration,
                        )

                    await asyncio.sleep(1.0 / hz)

            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
            finally:
                print("\033[?25h")
                await arm.stop_telemetry()

    except Exception as exc:
        _logger.error("Fatal error in _run: %s\n%s", exc, traceback.format_exc())
        raise
    finally:
        if stats_task is not None and not stats_task.done():
            stats_task.cancel()
            try:
                await stats_task
            except asyncio.CancelledError:
                pass

        elapsed_total = time.perf_counter() - t_start
        _logger.info(
            "FINAL STATS  elapsed=%.1fs  cycles=%d  actual_hz=%.1f"
            "  send_err=%d  timeout_err=%d  other_err=%d",
            elapsed_total,
            cycle_count,
            cycle_count / elapsed_total if elapsed_total > 0 else 0.0,
            send_error_count,
            timeout_error_count,
            other_error_count,
        )
        _logger.info("Final CAN stats:\n%s", _read_can_stats(channel))


def main() -> None:
    valid_joints = [j.value for j in Joint]
    parser = argparse.ArgumentParser(
        description="Cycle one joint through its limits via motion control."
    )
    side = parser.add_mutually_exclusive_group(required=True)
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
        default=f"can_send_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        help="Path for the diagnostic log file",
    )
    args = parser.parse_args()

    cycle_joint = Joint(args.joint)
    asyncio.run(_run(is_left=args.l, cycle_joint=cycle_joint, hz=args.hz, log_file=args.log_file))


if __name__ == "__main__":
    main()
