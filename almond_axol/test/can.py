"""Live terminal display of all motor positions.

Run directly:
    python -m almond_axol.test.can --l
    python -m almond_axol.test.can --r
    python -m almond_axol.test.can --l --hz 50
    python -m almond_axol.test.can --l --hz 250 --log-file can_diag.log
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

from ..motor import CanBus
from ..robot.axol import AxolArm, arm_limits
from ..robot.config import AxolConfig
from ..shared import CAN_LEFT, CAN_RIGHT, Joint

_BAR_WIDTH = 24
_TAU = 2 * math.pi

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


async def _run(is_left: bool, hz: int, log_file: str) -> None:
    _setup_logging(log_file)

    # Catch unhandled exceptions from background asyncio tasks.
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
    side = "left" if is_left else "right"
    channel = CAN_LEFT if is_left else CAN_RIGHT

    _logger.info("Starting telemetry  side=%s  channel=%s  hz=%d", side, channel, hz)
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
                    "start_telemetry failed: %s\n%s",
                    exc,
                    traceback.format_exc(),
                )
                raise

            await asyncio.sleep(0.1)

            print("\033[?25l", end="")
            last_stat_log = time.perf_counter()

            try:
                while True:
                    cycle_count += 1

                    try:
                        positions = arm.positions
                    except Exception as exc:
                        other_error_count += 1
                        _logger.error(
                            "arm.positions failed (cycle=%d): %s\n%s",
                            cycle_count,
                            exc,
                            traceback.format_exc(),
                        )
                        positions = np.zeros(len(joints), dtype=np.float32)

                    lines = []
                    lines.append("\033[H\033[J")
                    lines.append(f"  {side.upper()} ARM  [{hz} Hz]  log→{log_file}")
                    lines.append(
                        f"  cycles={cycle_count}  send_err={send_error_count}"
                        f"  timeout_err={timeout_error_count}"
                        f"  other_err={other_error_count}"
                    )
                    lines.append(f"  {'Joint':<12}  {'rev':>8}  {'':^{_BAR_WIDTH}}")
                    lines.append("  " + "─" * (12 + 8 + _BAR_WIDTH + 4))

                    for i, joint in enumerate(joints):
                        lo, hi = arm_limits(joint, is_left=is_left)
                        p = float(positions[i])
                        lines.append(
                            f"  {joint.value:<12}  {p / _TAU:>+8.4f}  {_bar(p, lo, hi)}"
                        )

                    lines.append("")
                    lines.append("  ctrl+c to quit")
                    print("\n".join(lines), end="", flush=True)

                    # Log per-cycle timing to file every 10 seconds
                    now = time.perf_counter()
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

                    await asyncio.sleep(1 / hz)

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
    parser = argparse.ArgumentParser(description="Live motor position display")
    side = parser.add_mutually_exclusive_group(required=True)
    side.add_argument("--l", action="store_true", help="Monitor left arm")
    side.add_argument("--r", action="store_true", help="Monitor right arm")
    parser.add_argument(
        "--hz", type=int, default=100, help="Telemetry rate in Hz (default: 100)"
    )
    parser.add_argument(
        "--log-file",
        default=f"can_diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        help="Path for the diagnostic log file",
    )
    args = parser.parse_args()

    asyncio.run(_run(is_left=args.l, hz=args.hz, log_file=args.log_file))


if __name__ == "__main__":
    main()
