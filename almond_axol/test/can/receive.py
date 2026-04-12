"""Live terminal display of all motor positions.

Run directly:
    python -m almond_axol.test.can.receive --l
    python -m almond_axol.test.can.receive --r
    python -m almond_axol.test.can.receive            # both arms, log only
    python -m almond_axol.test.can.receive --l --hz 50
    python -m almond_axol.test.can.receive --l --hz 250 --log-file can_diag.log
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
from ...robot.axol import AxolArm, arm_limits
from ...robot.config import AxolConfig
from ...shared import CAN_LEFT, CAN_RIGHT, Joint

_BAR_WIDTH = 24
_TAU = 2 * math.pi
_DISPLAY_HZ = 30
# Width reserved for the left arm column in the side-by-side display.
# Joint lines are 50 chars; 56 leaves a clear margin before the right column.
_COL_WIDTH = 56
_COL_GAP = 2

_logger = logging.getLogger(__name__)


@dataclass
class _ArmSnapshot:
    side: str
    hz: int
    log_file: str
    is_left: bool
    positions: np.ndarray = field(default_factory=lambda: np.zeros(len(list(Joint))))
    cycle_count: int = 0
    send_error_count: int = 0
    timeout_error_count: int = 0
    other_error_count: int = 0


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


def _arm_lines(snap: _ArmSnapshot) -> list[str]:
    joints = list(Joint)
    lines = [
        f"  {snap.side.upper()} ARM  [{snap.hz} Hz]  log→{snap.log_file}",
        (
            f"  cycles={snap.cycle_count}  send_err={snap.send_error_count}"
            f"  timeout_err={snap.timeout_error_count}"
            f"  other_err={snap.other_error_count}"
        ),
        f"  {'Joint':<12}  {'rev':>8}  {'':^{_BAR_WIDTH}}",
        "  " + "─" * (12 + 8 + _BAR_WIDTH + 4),
    ]
    for i, joint in enumerate(joints):
        lo, hi = arm_limits(joint, is_left=snap.is_left)
        p = float(snap.positions[i])
        lines.append(f"  {joint.value:<12}  {p / _TAU:>+8.4f}  {_bar(p, lo, hi)}")
    return lines


async def _display_both(left: _ArmSnapshot, right: _ArmSnapshot) -> None:
    right_col = _COL_WIDTH + _COL_GAP + 1  # 1-based terminal column for right arm
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


async def _run(
    is_left: bool,
    hz: int,
    log_file: str,
    display: bool = True,
    snapshot: _ArmSnapshot | None = None,
) -> None:
    side = "left" if is_left else "right"
    log = _make_logger(log_file, f"{__name__}.{side}")

    # Catch unhandled exceptions from background asyncio tasks.
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
    channel = CAN_LEFT if is_left else CAN_RIGHT

    log.info("Starting  side=%s  channel=%s  hz=%d", side, channel, hz)
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
                await arm.start_telemetry(hz)
                log.info("Telemetry started at %d Hz", hz)
            except Exception as exc:
                log.error(
                    "start_telemetry failed: %s\n%s",
                    exc,
                    traceback.format_exc(),
                )
                raise

            await asyncio.sleep(0.1)

            if display:
                print("\033[?25l", end="")
            last_stat_log = time.perf_counter()
            last_display = 0.0
            interval = 1.0 / hz

            try:
                while True:
                    cycle_count += 1
                    t_iter = time.perf_counter()

                    try:
                        positions = arm.positions
                    except Exception as exc:
                        other_error_count += 1
                        log.error(
                            "arm.positions failed (cycle=%d): %s\n%s",
                            cycle_count,
                            exc,
                            traceback.format_exc(),
                        )
                        positions = np.zeros(len(joints), dtype=np.float32)

                    now = t_iter

                    if snapshot is not None:
                        snapshot.positions = positions.copy()
                        snapshot.cycle_count = cycle_count
                        snapshot.send_error_count = send_error_count
                        snapshot.timeout_error_count = timeout_error_count
                        snapshot.other_error_count = other_error_count

                    if display and now - last_display >= 1.0 / _DISPLAY_HZ:
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
                        last_display = now

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

                    elapsed = time.perf_counter() - t_iter
                    await asyncio.sleep(max(0.0, interval - elapsed))

            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
            finally:
                if display:
                    print("\033[?25h")
                await arm.stop_telemetry()

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
    parser = argparse.ArgumentParser(description="Live motor position display")
    side = parser.add_mutually_exclusive_group()
    side.add_argument("--l", action="store_true", help="Monitor left arm")
    side.add_argument("--r", action="store_true", help="Monitor right arm")
    parser.add_argument(
        "--hz", type=int, default=100, help="Telemetry rate in Hz (default: 100)"
    )
    parser.add_argument(
        "--log-file",
        default=f"logs/can_receive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        help="Path for the diagnostic log file",
    )
    args = parser.parse_args()

    try:
        if not args.l and not args.r:
            stem, _, ext = args.log_file.rpartition(".")
            left_log = f"{stem}_left.{ext}"
            right_log = f"{stem}_right.{ext}"
            print("No side specified — monitoring both arms.")
            print(f"  left  → {left_log}")
            print(f"  right → {right_log}")

            joints = list(Joint)
            left_snap = _ArmSnapshot(
                side="left",
                hz=args.hz,
                log_file=left_log,
                is_left=True,
                positions=np.zeros(len(joints)),
            )
            right_snap = _ArmSnapshot(
                side="right",
                hz=args.hz,
                log_file=right_log,
                is_left=False,
                positions=np.zeros(len(joints)),
            )

            async def _run_both() -> None:
                display_task = asyncio.create_task(_display_both(left_snap, right_snap))
                try:
                    await asyncio.gather(
                        _run(
                            is_left=True,
                            hz=args.hz,
                            log_file=left_log,
                            display=False,
                            snapshot=left_snap,
                        ),
                        _run(
                            is_left=False,
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
            asyncio.run(_run(is_left=args.l, hz=args.hz, log_file=args.log_file))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
