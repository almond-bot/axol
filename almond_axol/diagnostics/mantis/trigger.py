"""``diag.mantis-trigger``: camera-free Mantis trigger-to-gripper test.

This diagnostic opens the trigger reader and gripper on each saved Mantis CAN
channel, waits for both triggers to be fresh and released, then mirrors their
analog values onto the physical grippers until interrupted.  It deliberately
does not start a VR server or require tracker bindings, cameras, a headset, or
tracker-to-TCP calibration.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable
from typing import Any

import numpy as np

from ...constants import ARM_JOINTS
from ...robot.mantis import Mantis
from ...tracker.trigger import TriggerReader
from ...utils.can_channels import require_mantis_channels

_CONTROL_INTERVAL_S = 0.01
_STATUS_INTERVAL_S = 0.25
_TRIGGER_WAIT_TIMEOUT_S = 5.0
# Match the release threshold used by the managed Mantis engage gesture.
_RELEASED_GRIP_MIN = 0.8
_PROMPT_MARKER = "[prompt]"


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by direct dispatch and serve introspection."""
    parser.add_argument(
        "--left-channel",
        metavar="IFACE",
        help="Left rig SocketCAN interface (default: saved Settings → Mantis map).",
    )
    parser.add_argument(
        "--right-channel",
        metavar="IFACE",
        help="Right rig SocketCAN interface (default: saved Settings → Mantis map).",
    )
    parser.add_argument(
        "--duration",
        type=_positive_float,
        metavar="SECONDS",
        help="Stop automatically after this many seconds (default: run until Ctrl-C).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the clear-jaws confirmation prompt.",
    )
    parser.add_argument(
        "--web-prompts",
        action="store_true",
        help="Emit a '[prompt] ...' marker and wait for the diagnostics "
        "dashboard's Continue action.",
    )


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register ``diag.mantis-trigger`` for serve-side introspection."""
    parser = subparsers.add_parser(
        "diag.mantis-trigger",
        help="Drive the Mantis grippers from their triggers (no tracking/cameras).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    _add_arguments(parser)
    parser.set_defaults(func=run_cli)


def _resolve_channels(
    left_override: str | None, right_override: str | None
) -> tuple[str, str]:
    """Apply explicit overrides above the control panel's saved Mantis map."""
    # Keep settings lazy: importing this module for ``axol --help`` should not
    # initialise the serve/config stack or bind its persistent-state path.
    from ...serve.settings import SettingsStore

    saved_left, saved_right = SettingsStore().mantis_can_channels()
    return require_mantis_channels(
        (
            left_override if left_override is not None else saved_left,
            right_override if right_override is not None else saved_right,
        )
    )


def _confirm(
    left_channel: str, right_channel: str, *, web_prompts: bool = False
) -> None:
    print("Mantis trigger → gripper test (no cameras or trackers required)")
    print(f"  Left : {left_channel}")
    print(f"  Right: {right_channel}")
    print(
        "\nWARNING: enabling performs an open-stop calibration, so both jaws will "
        "move fully open.\nKeep hands and objects clear, and release both triggers."
    )
    prompt = "Jaws are clear and both triggers are released — enable the grippers."
    if web_prompts:
        print(f"{_PROMPT_MARKER} {prompt}", flush=True)
        if not sys.stdin.readline():
            raise SystemExit("Confirmation input closed; grippers were not enabled.")
        return
    try:
        input(f"{prompt} Press Enter to continue, or Ctrl-C to cancel ... ")
    except EOFError as exc:
        raise SystemExit(
            "Confirmation requires an interactive terminal; re-run with --yes "
            "only after clearing both grippers."
        ) from exc


def run_cli(args: argparse.Namespace) -> None:
    """Resolve the rig channels, confirm motion, and run the async diagnostic."""
    try:
        left_channel, right_channel = _resolve_channels(
            args.left_channel, args.right_channel
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    try:
        if not args.yes:
            _confirm(
                left_channel,
                right_channel,
                web_prompts=args.web_prompts,
            )
        asyncio.run(
            _run(
                left_channel,
                right_channel,
                duration=args.duration,
            )
        )
    except KeyboardInterrupt:
        print("\nStopped; both grippers have been disabled.")
    except Exception as exc:
        raise SystemExit(f"Mantis trigger test failed: {exc}") from None
    else:
        print("\nFinished; both grippers have been disabled.")


def main(argv: list[str] | None = None) -> None:
    """Parse arguments from the lazy ``diag.*`` dispatcher and run the test."""
    parser = argparse.ArgumentParser(
        prog="axol diag.mantis-trigger",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    run_cli(parser.parse_args(argv))


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


async def _run(
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
    """Mirror two live Mantis triggers to the matching physical grippers."""
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

        print("Calibrating the grippers against their open stops ...")
        await robot.enable_grippers()

        # Calibration may take several seconds. Require the safety condition
        # again instead of applying a squeeze made while the jaws were moving.
        await _wait_for_released_triggers(
            readers,
            channels,
            timeout=wait_timeout,
            poll_interval=poll_interval,
        )

        print("Ready. Squeeze either trigger; press Ctrl-C to stop.")
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
        try:
            if robot is not None:
                await robot.disable()
        finally:
            # Reader teardown must not be skipped even if a best-effort motor
            # disable reports its own hardware error.
            for reader in readers.values():
                reader.close()


if __name__ == "__main__":
    main()
