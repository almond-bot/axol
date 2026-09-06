"""Grippers-only Mantis teleop: mirror the rig triggers onto the grippers.

Mantis teleop is grippers-only by design — tracked Mantis runs belong to data
collection. ``axol teleop --mantis`` runs this loop: it opens the trigger
reader and gripper on each Mantis CAN channel, waits for both triggers to be
fresh and released, then mirrors their analog values onto the physical
grippers until stopped. It deliberately starts no VR server and needs no
tracker binding, cameras, headset, or tracker-to-TCP transform.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from ..constants import ARM_JOINTS
from ..robot.base import HardwareCleanupError
from ..robot.mantis import Mantis
from ..tracker.trigger import TriggerReader

_logger = logging.getLogger(__name__)

_CONTROL_INTERVAL_S = 0.01
_STATUS_INTERVAL_S = 0.25
_TRIGGER_WAIT_TIMEOUT_S = 5.0
# Match the release threshold used by the managed Mantis engage gesture.
_RELEASED_GRIP_MIN = 0.8


def _read_fresh_grips(readers: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    """Return current fresh values and the sides that lack a fresh frame."""
    grips: dict[str, float] = {}
    missing: list[str] = []
    for side, reader in readers.items():
        grip = reader.grip()
        if grip is None or reader.is_stale():
            missing.append(side)
        else:
            grips[side] = float(grip)
    return grips, missing


async def _wait_for_released_triggers(
    readers: dict[str, Any],
    channels: dict[str, str],
    *,
    timeout: float,
    poll_interval: float,
) -> dict[str, float]:
    """Wait until every trigger is live and safely in its released position."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        grips, missing = _read_fresh_grips(readers)
        held = [side for side, grip in grips.items() if grip < _RELEASED_GRIP_MIN]
        if not missing and not held:
            return grips
        if loop.time() >= deadline:
            details: list[str] = []
            if missing:
                details.append(
                    "no fresh trigger frames on "
                    + ", ".join(f"{side} ({channels[side]})" for side in missing)
                )
            if held:
                details.append(
                    "release "
                    + ", ".join(
                        f"{side} trigger (grip={grips[side]:.2f})" for side in held
                    )
                )
            raise RuntimeError("; ".join(details))
        await asyncio.sleep(poll_interval)


async def run_grippers_only(
    left_channel: str,
    right_channel: str,
    *,
    reader_factory: Callable[[str], Any] = TriggerReader,
    robot_factory: Callable[..., Any] = Mantis,
    wait_timeout: float = _TRIGGER_WAIT_TIMEOUT_S,
    poll_interval: float = _CONTROL_INTERVAL_S,
    status_interval: float = _STATUS_INTERVAL_S,
    duration: float | None = None,
) -> None:
    """Mirror two live Mantis triggers to the matching physical grippers.

    Runs until cancelled (the control panel's Stop, or Ctrl-C on the CLI) or
    for ``duration`` seconds. Both grippers are disabled on the way out, and a
    trigger stream that goes stale stops the run rather than holding a stale
    command.
    """
    channels = {"left": left_channel, "right": right_channel}
    readers: dict[str, Any] = {}
    robot: Any | None = None
    status_started = False

    try:
        for side, channel in channels.items():
            readers[side] = reader_factory(channel)

        # Deferred mode lets us establish a known torque-off state before the
        # input preflight. No gripper is enabled until both trigger streams are
        # demonstrably live and released.
        robot = robot_factory(
            left_channel=left_channel,
            right_channel=right_channel,
            defer_gripper_enable=True,
        )
        await robot.connect()

        print("Waiting for fresh, released trigger frames ...")
        await _wait_for_released_triggers(
            readers,
            channels,
            timeout=wait_timeout,
            poll_interval=poll_interval,
        )

        print(
            "Calibrating the grippers against their open stops (both jaws move "
            "fully open — keep hands and objects clear) ..."
        )
        await robot.enable_grippers()

        # Calibration may take several seconds. Require the safety condition
        # again instead of applying a squeeze made while the jaws were moving.
        await _wait_for_released_triggers(
            readers,
            channels,
            timeout=wait_timeout,
            poll_interval=poll_interval,
        )

        print("Ready. Squeeze either trigger; Stop / Ctrl-C ends the run.")
        target_size = len(ARM_JOINTS) + 1
        left_target = np.zeros(target_size, dtype=np.float32)
        right_target = np.zeros(target_size, dtype=np.float32)
        loop = asyncio.get_running_loop()
        started = loop.time()
        next_status = started

        while duration is None or loop.time() - started < duration:
            grips, missing = _read_fresh_grips(readers)
            if missing:
                sides = ", ".join(f"{side} ({channels[side]})" for side in missing)
                raise RuntimeError(
                    f"trigger stream went stale on {sides}; disabling both grippers"
                )

            left_target[-1] = grips["left"]
            right_target[-1] = grips["right"]
            await robot.motion_control(left=left_target, right=right_target)

            now = loop.time()
            if now >= next_status:
                print(
                    f"  command  left={grips['left']:.2f}  "
                    f"right={grips['right']:.2f}  "
                    "(0=closed, 1=open)",
                    end="\r",
                    flush=True,
                )
                status_started = True
                next_status = now + status_interval
            await asyncio.sleep(poll_interval)
    finally:
        if status_started:
            print()
        cleanup_failures: list[tuple[str, BaseException]] = []
        if robot is not None:
            try:
                await robot.disable()
            except BaseException as exc:
                cleanup_failures.append(("robot disable", exc))
        # Attempt every reader even when motor disable or an earlier reader
        # reports a cleanup failure.
        for side, reader in readers.items():
            try:
                reader.close()
            except BaseException as exc:
                cleanup_failures.append((f"{side} trigger", exc))

        if cleanup_failures:
            label, first = cleanup_failures[0]
            for extra_label, extra in cleanup_failures[1:]:
                first.add_note(
                    f"additional {extra_label} cleanup failure: "
                    f"{type(extra).__name__}: {extra}"
                )
            if label == "robot disable":
                if isinstance(first, HardwareCleanupError):
                    raise first
                raise HardwareCleanupError(
                    "Mantis disable failed; hardware ownership is uncertain"
                ) from first
            raise RuntimeError(
                f"{label} teardown failed; CAN ownership is uncertain"
            ) from first
