"""Per-run telemetry capture for diagnostics scripts that own the CAN bus.

While a diagnostic script runs, ``axol serve``'s own telemetry sampler is
paused (single-owner CAN bus), so the diagnostics dashboard can't observe the
run. Scripts fill that gap themselves: :class:`TelemetryCsvLogger` samples the
robot's *cached* motor state (populated by the script's own command/telemetry
traffic — no extra CAN frames) into a wide-format CSV, and announces the file
with a ``[telemetry] csv=<path>`` log line that the serve-side run store picks
up when the session ends (see :mod:`almond_axol.serve.telemetry`).

CSV columns: ``t`` (epoch seconds) then ``<arm>:<JOINT>:pos`` and
``<arm>:<JOINT>:tq`` for every joint of every present arm. Positions are raw
shaft radians (matching the live dashboard sampler); a cell is left empty for
any motor with no cached reading yet, so a ``--joints`` subset run still
captures the joints it actually drives. Velocity is not cached by the motor
layer, so it is not captured here.
"""

from __future__ import annotations

import asyncio
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from ..constants import Joint
from ..motor import MotorError

if TYPE_CHECKING:
    from ..robot.axol import Axol, AxolArm

CAPTURE_DIR = Path.home() / ".almond" / "diagnostics" / "captures"

_DEFAULT_HZ = 5.0


class TelemetryCsvLogger:
    """Background sampler writing cached per-motor state to a CSV file."""

    def __init__(
        self,
        axol: Axol,
        name: str,
        hz: float = _DEFAULT_HZ,
        out_dir: Path = CAPTURE_DIR,
    ) -> None:
        self._axol = axol
        self._hz = hz
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = out_dir / f"{name}_{stamp}.csv"
        self._task: asyncio.Task[None] | None = None
        self._file: TextIO | None = None

    @property
    def path(self) -> Path:
        return self._path

    def start(self) -> None:
        """Open the CSV, announce it in the log, and start sampling."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w", newline="")
        writer = csv.writer(self._file)
        header = ["t"]
        for side, arm in self._arms():
            for joint in Joint:
                header.append(f"{side}:{joint.name}:pos")
                header.append(f"{side}:{joint.name}:tq")
        writer.writerow(header)
        # Marker the serve-side diagnostics run store scans the session log for.
        print(f"[telemetry] csv={self._path}")
        self._task = asyncio.ensure_future(self._loop(writer))

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def _arms(self) -> list[tuple[str, AxolArm]]:
        pairs = []
        if self._axol.left is not None:
            pairs.append(("left", self._axol.left))
        if self._axol.right is not None:
            pairs.append(("right", self._axol.right))
        return pairs

    async def _loop(self, writer: csv.writer) -> None:  # type: ignore[name-defined]
        interval = 1.0 / self._hz
        flush_every = max(1, int(self._hz))  # flush ~once a second
        rows = 0
        while True:
            row: list[str | float] = [round(time.time(), 3)]
            wrote_any = False
            # Sample per motor from its own cache rather than the arm-wide
            # AxolArm.positions/torques, which raise if *any* joint on the arm
            # is uncached — that would drop every row of a --joints subset run.
            for _side, arm in self._arms():
                for joint in Joint:
                    motor = arm.motors[joint]
                    if motor.has_position:
                        row.append(round(float(motor.position), 5))
                        wrote_any = True
                    else:
                        row.append("")
                    try:
                        row.append(round(float(motor.torque), 4))
                    except MotorError:
                        # Torque cache can lag position (or never populate for
                        # an idle joint); position alone still makes a row.
                        row.append("")
            if wrote_any:
                writer.writerow(row)
                rows += 1
                if self._file is not None and rows % flush_every == 0:
                    self._file.flush()
            await asyncio.sleep(interval)
