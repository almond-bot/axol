"""
diag.lift-cycle

Mounted-robot endurance test for the Jelly Legs telescoping lift. The selected
Axol arms first move to a mirrored shoulder-1 clearance pose (left +90 degrees,
right -90 degrees) while every other joint holds its measured position. The
lift is then raised to establish a safe starting point and driven through the
requested number of complete down/up cycles. A successful run finishes with
the lift fully up, ramps shoulder-1 back to its zero-degree rest pose, then
disables the selected arm motors.

Every move uses the firmware's homed absolute-position controller, validates
driver/interlock status and endpoint alignment, and waits for the v0.8
position save to finish before issuing the next move. The held arm joints
remain monitored throughout lift motion, and Ctrl-C stops the lift. The other
arm joints hold their measured starting positions throughout.

Usage:
    axol diag.lift-cycle                    # prompts for cycle count
    axol diag.lift-cycle --cycles 10
    axol diag.lift-cycle --cycles 3 --speed 300
    axol diag.lift-cycle --cycles 2 --no-right

The diagnostic is arm-joints-only: it never enables, calibrates, commands, or
disables either gripper.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
import time
from collections.abc import Awaitable, Callable

import numpy as np

from ...cli.lift import Interrupted, fmt_status, interrupt_event
from ...constants import ARM_JOINTS, CAN_CHEST, CAN_LEFT, CAN_RIGHT, Joint
from ...robot.axol import Axol
from ...robot.config import AxolConfig
from ...robot.lift import Lift, LiftStatus

_STATUS_PERIOD_MS = 200
_STATUS_STALE_S = 1.0
_FIRST_STATUS_TIMEOUT_S = 3.0
# Accept scheduler jitter and one missed frame while still rejecting the
# firmware's TX-starving 50 ms default cadence.
_STATUS_CADENCE_MIN_S = 0.12
_STATUS_CADENCE_MAX_S = 0.50
_STATUS_CADENCE_SAMPLES = 3
_MOVE_START_TIMEOUT_S = 5.0
# The firmware stops a position move at 60 s. Leave enough headroom for one
# status interval plus arm-safety reads, so the host can request STOP first.
_MOVE_TIMEOUT_S = 58.0
_POSITION_SAVE_TIMEOUT_S = 3.0
_ENDPOINT_SETTLE_S = 0.25
_ENDPOINT_TOLERANCE_PERMILLE = 0
_ENDPOINT_MAX_DRIFT_COUNTS = 8

_SHOULDER_CLEARANCE_RAD = math.pi / 2
_SHOULDER_RAMP_SPEED_RAD_S = math.radians(30)
_SHOULDER_TOLERANCE_RAD = math.radians(3)
_HELD_JOINT_TOLERANCE_RAD = math.radians(5)
_ARM_CONTROL_HZ = 100.0
_ARM_SAFETY_CHECK_PERIOD_S = 0.1
_S1_INDEX = list(Joint).index(Joint.SHOULDER_1)

_LOWER = 0
_UPPER = 1000


class DiagnosticFailure(RuntimeError):
    """A safety, communication, or endpoint check failed."""


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _speed_value(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 0 <= parsed <= 0xFFFF:
        raise argparse.ArgumentTypeError("must be between 0 and 65535")
    if 0 < parsed < 250:
        raise argparse.ArgumentTypeError(
            "must be 0 (full speed) or at least 250 to preserve the "
            "full-travel timeout margin"
        )
    return parsed


def _resolve_cycles(value: int | None) -> int:
    if value is not None:
        if value <= 0:
            raise SystemExit("ERROR: --cycles must be greater than zero.")
        return value
    if not sys.stdin.isatty():
        raise SystemExit("ERROR: --cycles is required when stdin is not a terminal.")
    while True:
        raw = input("Number of full down/up cycles: ").strip()
        try:
            return _positive_int(raw)
        except argparse.ArgumentTypeError as exc:
            print(f"Please enter a positive integer ({exc}).")


def _status_line(status: LiftStatus) -> str:
    return f"{fmt_status(status)} drift={status.drift:+d}"


def _validate_controller(status: LiftStatus, context: str) -> None:
    driver_fields = (
        status.driver_fault_mask,
        status.drivers_enabled,
        status.vm_present,
        status.flash_interlock,
        status.save_pending,
    )
    if any(value is None for value in driver_fields):
        raise DiagnosticFailure(
            f"{context}: the controller returned a legacy six-byte status "
            "without driver/interlock health; flash the current (v0.8) Jelly Legs "
            "firmware before running this diagnostic"
        )
    if status.driver_fault_mask:
        raise DiagnosticFailure(
            f"{context}: DRV8245 fault mask 0x{status.driver_fault_mask:02x}"
        )
    if status.flash_interlock:
        raise DiagnosticFailure(
            f"{context}: the saved-position flash interlock is active; reboot "
            "the Jelly Legs controller before moving"
        )
    if not status.vm_present:
        raise DiagnosticFailure(f"{context}: the 24 V motor supply is not present")
    if not status.drivers_enabled:
        raise DiagnosticFailure(f"{context}: the leg motor drivers are disabled")
    if status.stall_fault:
        raise DiagnosticFailure(f"{context}: a leg stall fault is latched")


def _require_fresh_status(lift: Lift, context: str) -> LiftStatus:
    status = lift.status
    if status is None or not lift.status_is_fresh(_STATUS_STALE_S):
        age = lift.status_age
        age_text = "none received" if age is None else f"last received {age:.1f}s ago"
        raise DiagnosticFailure(f"{context}: lift status is stale ({age_text})")
    _validate_controller(status, context)
    return status


def _is_idle(status: LiftStatus) -> bool:
    return not (status.moving or status.pos_move or status.homing or status.jog)


def _require_lift_ready(lift: Lift, context: str) -> LiftStatus:
    """Require a healthy, homed, idle controller with no pending save."""
    status = _require_fresh_status(lift, context)
    if not status.homed:
        raise DiagnosticFailure(f"{context}: lift homing state is unavailable")
    if not _is_idle(status):
        raise DiagnosticFailure(
            f"{context}: lift is already moving: {_status_line(status)}"
        )
    if status.save_pending:
        raise DiagnosticFailure(f"{context}: a position save is still pending")
    return status


def _at_endpoint(status: LiftStatus, target: int) -> bool:
    position = status.position_permille
    if position is None or abs(position - target) > _ENDPOINT_TOLERANCE_PERMILLE:
        return False
    # SET_POS endpoints use the intersection of both legs' homed ranges. The
    # at_lower/at_upper bits describe each leg's individual limit and can stay
    # false when those limits differ slightly, so the firmware endurance test
    # likewise judges completion from exact permille, idle state, and drift.
    return _is_idle(status) and abs(status.drift) <= _ENDPOINT_MAX_DRIFT_COUNTS


async def _wait_for_status_after(
    lift: Lift,
    after: float,
    interrupted: asyncio.Event,
    context: str,
    safety_check: Callable[[], Awaitable[None]] | None = None,
) -> LiftStatus:
    deadline = time.monotonic() + _STATUS_STALE_S
    next_safety_check = 0.0
    while True:
        if interrupted.is_set():
            raise Interrupted
        now = time.monotonic()
        if safety_check is not None and now >= next_safety_check:
            await safety_check()
            if interrupted.is_set():
                raise Interrupted
            next_safety_check = time.monotonic() + _ARM_SAFETY_CHECK_PERIOD_S
        now = time.monotonic()
        stamp = lift.last_status_monotonic
        if stamp is not None and stamp > after:
            return _require_fresh_status(lift, context)
        if now >= deadline:
            raise DiagnosticFailure(
                f"{context}: no fresh status arrived within {_STATUS_STALE_S:.1f}s"
            )
        await asyncio.sleep(0.02)


async def _wait_for_position_save(
    lift: Lift,
    status: LiftStatus,
    interrupted: asyncio.Event,
    context: str,
    safety_check: Callable[[], Awaitable[None]] | None = None,
) -> LiftStatus:
    deadline = time.monotonic() + _POSITION_SAVE_TIMEOUT_S
    while status.save_pending:
        if time.monotonic() >= deadline:
            raise DiagnosticFailure(
                f"{context}: position save remained pending for "
                f"{_POSITION_SAVE_TIMEOUT_S:.1f}s"
            )
        stamp = lift.last_status_monotonic or time.monotonic()
        status = await _wait_for_status_after(
            lift,
            stamp,
            interrupted,
            context,
            safety_check,
        )
        if not _is_idle(status):
            raise DiagnosticFailure(
                f"{context}: motion resumed while saving position: "
                f"{_status_line(status)}"
            )
    return status


async def _settle_endpoint(
    lift: Lift,
    target: int,
    interrupted: asyncio.Event,
    context: str,
    safety_check: Callable[[], Awaitable[None]] | None = None,
) -> LiftStatus:
    stamp = lift.last_status_monotonic or time.monotonic()
    deadline = time.monotonic() + _ENDPOINT_SETTLE_S
    next_safety_check = 0.0
    while time.monotonic() < deadline:
        if interrupted.is_set():
            raise Interrupted
        now = time.monotonic()
        if safety_check is not None and now >= next_safety_check:
            await safety_check()
            if interrupted.is_set():
                raise Interrupted
            next_safety_check = time.monotonic() + _ARM_SAFETY_CHECK_PERIOD_S
        await asyncio.sleep(0.02)
    status = await _wait_for_status_after(
        lift,
        stamp,
        interrupted,
        f"{context} settle",
        safety_check,
    )
    if not status.homed:
        raise DiagnosticFailure(f"{context}: homing state was lost")
    if not _at_endpoint(status, target):
        raise DiagnosticFailure(
            f"{context}: endpoint did not settle cleanly: {_status_line(status)}"
        )
    status = await _wait_for_position_save(
        lift,
        status,
        interrupted,
        context,
        safety_check,
    )
    if not _at_endpoint(status, target):
        raise DiagnosticFailure(
            f"{context}: endpoint changed during position save: {_status_line(status)}"
        )
    return status


async def _move_lift(
    lift: Lift,
    target: int,
    speed: int,
    interrupted: asyncio.Event,
    label: str,
    safety_check: Callable[[], Awaitable[None]] | None = None,
) -> LiftStatus:
    async def verify_before_send() -> None:
        if interrupted.is_set():
            raise Interrupted
        if safety_check is not None:
            await safety_check()
        if interrupted.is_set():
            raise Interrupted
        _require_lift_ready(lift, f"{label} pre-command")
        if interrupted.is_set():
            raise Interrupted

    # Check before beginning the driver's canonical STOP handshake, then
    # again after that awaited wire operation and directly before SET_POS.
    await verify_before_send()
    print(f"{label}: commanding {target / 10:.1f}% ...", flush=True)
    await lift.set_position(target, speed, before_send=verify_before_send)
    # A cached frame that happened to arrive during preflight/send cannot
    # prove the firmware accepted this command. Require a later frame.
    commanded_at = time.monotonic()
    started_at = time.monotonic()
    deadline = started_at + _MOVE_TIMEOUT_S
    start_deadline = started_at + _MOVE_START_TIMEOUT_S
    next_report = 0.0
    saw_position_move = False
    last_stamp = commanded_at

    while True:
        status = await _wait_for_status_after(
            lift,
            last_stamp,
            interrupted,
            label,
            safety_check,
        )
        last_stamp = lift.last_status_monotonic or last_stamp
        if not status.homed:
            raise DiagnosticFailure(f"{label}: homing state was lost")
        if status.homing or status.jog:
            raise DiagnosticFailure(
                f"{label}: controller entered an unexpected motion mode: "
                f"{_status_line(status)}"
            )

        saw_position_move |= status.pos_move
        active = status.moving or status.pos_move
        # v0.4 also emitted eight-byte status, so frame length alone cannot
        # identify v0.8. Current firmware marks its position save pending
        # while a position move is active; prove that behavior before allowing
        # an endurance cycle to continue.
        if status.pos_move and status.save_pending is not True:
            raise DiagnosticFailure(
                f"{label}: controller did not report the v0.8 position-save "
                "state; flash the current Jelly Legs firmware before cycling"
            )
        now = time.monotonic()
        if now >= deadline:
            raise DiagnosticFailure(
                f"{label}: move exceeded {_MOVE_TIMEOUT_S:.1f}s: {_status_line(status)}"
            )
        if now >= next_report:
            next_report = now + 2.0
            print(f"  {label}: {_status_line(status)}", flush=True)

        if not active and _at_endpoint(status, target):
            status = await _settle_endpoint(
                lift,
                target,
                interrupted,
                label,
                safety_check,
            )
            print(f"{label} complete: {_status_line(status)}", flush=True)
            return status
        if saw_position_move and not active:
            raise DiagnosticFailure(
                f"{label}: move stopped short of its endpoint: {_status_line(status)}"
            )
        if not saw_position_move and now >= start_deadline:
            raise DiagnosticFailure(
                f"{label}: position move did not start: {_status_line(status)}"
            )


async def _open_lift(channel: str) -> Lift:
    # Configure receive-only recovery before the driver starts, and suppress
    # even its bootstrap GET_STATUS. Every observed frame is therefore an
    # unsolicited firmware broadcast. Discard the first transition frame,
    # then validate the requested cadence before any arm or lift motion.
    lift = Lift(channel)
    try:
        await lift.set_status_period(_STATUS_PERIOD_MS, recover_stale=False)
        await lift.start(request_status=False)
        deadline = time.monotonic() + _FIRST_STATUS_TIMEOUT_S
        while lift.status is None and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        if lift.status is None:
            raise DiagnosticFailure(
                f"no Jelly Legs status on {channel}; check that the base is "
                "powered and the lift controller is connected"
            )
        _require_fresh_status(lift, "initial status")
        first_stamp = lift.last_status_monotonic
        if first_stamp is None:
            raise DiagnosticFailure("initial status did not have a receive timestamp")
        deadline = time.monotonic() + _FIRST_STATUS_TIMEOUT_S
        steady_stamps: list[float] = []
        while time.monotonic() < deadline:
            steady_stamps = [
                stamp for stamp in lift.status_timestamps if stamp > first_stamp
            ]
            if len(steady_stamps) >= _STATUS_CADENCE_SAMPLES:
                break
            await asyncio.sleep(0.02)
        if len(steady_stamps) < _STATUS_CADENCE_SAMPLES:
            raise DiagnosticFailure(
                f"{_STATUS_PERIOD_MS} ms status broadcasts did not continue "
                f"on {channel} (received {len(steady_stamps)}/"
                f"{_STATUS_CADENCE_SAMPLES} steady-state frames)"
            )
        steady_stamps = steady_stamps[:_STATUS_CADENCE_SAMPLES]
        intervals = [
            current - previous
            for previous, current in zip(
                steady_stamps[:-1],
                steady_stamps[1:],
                strict=True,
            )
        ]
        if any(
            interval < _STATUS_CADENCE_MIN_S or interval > _STATUS_CADENCE_MAX_S
            for interval in intervals
        ):
            observed = ", ".join(f"{interval * 1000:.0f}" for interval in intervals)
            raise DiagnosticFailure(
                f"status broadcast cadence on {channel} did not match the "
                f"requested {_STATUS_PERIOD_MS} ms period (observed {observed} ms)"
            )
        _require_fresh_status(lift, "validated status broadcast cadence")
        # Only now may the driver's SET_RATE/GET_STATUS recovery resume.
        lift.enable_broadcast_recovery()
        print(f"Jelly Legs controller connected on {channel}.")
        return lift
    except BaseException as exc:
        cleanup_error: BaseException | None = None
        try:
            await _retry_cleanup(lift.close, label="closing lift after open failure")
        except BaseException as close_exc:
            cleanup_error = close_exc
            exc.add_note(
                "opening the lift also failed to clean up: "
                f"{type(close_exc).__name__}: {close_exc}"
            )
        if isinstance(cleanup_error, (KeyboardInterrupt, asyncio.CancelledError)):
            cleanup_error.add_note(
                "opening the lift also failed before cleanup was interrupted: "
                f"{type(exc).__name__}: {exc}"
            )
            raise cleanup_error
        if isinstance(exc, DiagnosticFailure):
            raise
        if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
            raise
        failure = DiagnosticFailure(
            f"could not open {channel}: {exc}; run `axol can.setup` to "
            "configure the lift bus or pass --lift-channel"
        )
        if cleanup_error is not None:
            failure.add_note(
                "lift cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise failure from None


def _clearance_targets(
    left: np.ndarray | None,
    right: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    left_target = left.copy() if left is not None else None
    right_target = right.copy() if right is not None else None
    if left_target is not None:
        left_target[_S1_INDEX] = _SHOULDER_CLEARANCE_RAD
    if right_target is not None:
        right_target[_S1_INDEX] = -_SHOULDER_CLEARANCE_RAD
    return left_target, right_target


def _validated_arm_pose(
    value: np.ndarray | None,
    *,
    side: str,
    context: str,
    required: bool,
) -> np.ndarray | None:
    if value is None:
        if required:
            raise DiagnosticFailure(f"{context}: {side} arm returned no position")
        return None
    pose = np.asarray(value)
    expected = (len(Joint),)
    if pose.shape != expected:
        raise DiagnosticFailure(
            f"{context}: {side} arm position shape is {pose.shape}, expected {expected}"
        )
    if not np.issubdtype(pose.dtype, np.number) or not np.all(np.isfinite(pose)):
        raise DiagnosticFailure(
            f"{context}: {side} arm positions must all be finite numbers"
        )
    return pose.astype(np.float32, copy=True)


async def _read_valid_arm_positions(
    axol: Axol,
    context: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    left, right = await axol.get_positions()
    return (
        _validated_arm_pose(
            left,
            side="left",
            context=context,
            required=axol.left is not None,
        ),
        _validated_arm_pose(
            right,
            side="right",
            context=context,
            required=axol.right is not None,
        ),
    )


async def _verify_arms_holding(axol: Axol, context: str) -> None:
    """Prove every selected arm joint remains enabled and holding torque."""
    motors = [
        (side, joint, arm.motors.get(joint))
        for side, arm in (("left", axol.left), ("right", axol.right))
        if arm is not None
        for joint in ARM_JOINTS
    ]
    missing = [
        f"{side} {joint.value}" for side, joint, motor in motors if motor is None
    ]
    if missing:
        raise DiagnosticFailure(
            f"{context}: arm motors are missing: {', '.join(missing)}"
        )
    results = await asyncio.gather(
        *(motor.is_holding() for _, _, motor in motors if motor is not None),
        return_exceptions=True,
    )
    problems: list[str] = []
    for (side, joint, _), result in zip(motors, results, strict=True):
        label = f"{side} {joint.value}"
        if isinstance(result, BaseException):
            problems.append(
                f"{label} holding check failed ({type(result).__name__}: {result})"
            )
        elif not result:
            problems.append(f"{label} is not enabled and holding")
    if problems:
        raise DiagnosticFailure(f"{context}: " + "; ".join(problems))


def _rest_targets(
    left: np.ndarray | None,
    right: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    left_target = left.copy() if left is not None else None
    right_target = right.copy() if right is not None else None
    if left_target is not None:
        left_target[_S1_INDEX] = 0.0
    if right_target is not None:
        right_target[_S1_INDEX] = 0.0
    return left_target, right_target


async def _disable_arms_verified(axol: Axol) -> None:
    """Disable every selected-arm motor and prove none still reports holding."""
    missing = [
        f"{side} {joint.value}"
        for side, arm in (("left", axol.left), ("right", axol.right))
        if arm is not None
        for joint in ARM_JOINTS
        if joint not in arm.motors
    ]
    if missing:
        raise DiagnosticFailure(
            "arm shutdown: selected arm motors are missing: " + ", ".join(missing)
        )
    motors = [
        (side, joint, motor)
        for side, arm in (("left", axol.left), ("right", axol.right))
        if arm is not None
        for joint, motor in arm.motors.items()
        if joint in ARM_JOINTS
    ]
    if not motors:
        raise DiagnosticFailure("arm shutdown: no selected arm motors are available")

    # The generic Axol.disable() lifecycle intentionally suppresses individual
    # motor errors before closing its buses. A diagnostic must be stricter: send
    # every shutdown concurrently, then query every motor while the buses remain
    # open and refuse PASS unless all of them report torque off.
    disable_results = await asyncio.gather(
        *(motor.disable() for _, _, motor in motors),
        return_exceptions=True,
    )
    holding_results = await asyncio.gather(
        *(motor.is_holding() for _, _, motor in motors),
        return_exceptions=True,
    )

    problems = []
    for (side, joint, _), disable_result, holding_result in zip(
        motors, disable_results, holding_results, strict=True
    ):
        label = f"{side} {joint.value}"
        if isinstance(disable_result, BaseException):
            problems.append(
                f"{label} disable failed "
                f"({type(disable_result).__name__}: {disable_result})"
            )
        if isinstance(holding_result, BaseException):
            detail = f"{type(holding_result).__name__}: {holding_result}"
            problems.append(f"{label} could not be verified ({detail})")
        elif holding_result:
            problems.append(f"{label} still reports enabled and holding")

    if problems:
        raise DiagnosticFailure("arm shutdown was not verified: " + "; ".join(problems))


async def _ramp_arms(
    axol: Axol,
    start_left: np.ndarray | None,
    start_right: np.ndarray | None,
    target_left: np.ndarray | None,
    target_right: np.ndarray | None,
    interrupted: asyncio.Event,
    safety_check: Callable[[], Awaitable[None]] | None = None,
) -> None:
    start_left = _validated_arm_pose(
        start_left,
        side="left",
        context="arm ramp start",
        required=target_left is not None,
    )
    start_right = _validated_arm_pose(
        start_right,
        side="right",
        context="arm ramp start",
        required=target_right is not None,
    )
    target_left = _validated_arm_pose(
        target_left,
        side="left",
        context="arm ramp target",
        required=start_left is not None,
    )
    target_right = _validated_arm_pose(
        target_right,
        side="right",
        context="arm ramp target",
        required=start_right is not None,
    )
    deltas = []
    if start_left is not None and target_left is not None:
        deltas.append(abs(float(target_left[_S1_INDEX] - start_left[_S1_INDEX])))
    if start_right is not None and target_right is not None:
        deltas.append(abs(float(target_right[_S1_INDEX] - start_right[_S1_INDEX])))
    if not deltas:
        raise DiagnosticFailure("no arm is available for the S1 ramp")

    # A cubic smoothstep peaks at 1.5x its average rate. Scale the duration so
    # 30 deg/s is the actual peak rather than only the average.
    duration = max(1.5 * max(deltas) / _SHOULDER_RAMP_SPEED_RAD_S, 0.1)
    started_at = time.monotonic()
    period = 1.0 / _ARM_CONTROL_HZ
    next_safety_check = 0.0
    while True:
        if interrupted.is_set():
            raise Interrupted
        now = time.monotonic()
        if safety_check is not None and now >= next_safety_check:
            await safety_check()
            if interrupted.is_set():
                raise Interrupted
            next_safety_check = time.monotonic() + _ARM_SAFETY_CHECK_PERIOD_S
        progress = min((time.monotonic() - started_at) / duration, 1.0)
        smooth = progress * progress * (3.0 - 2.0 * progress)
        left = (
            (start_left * (1.0 - smooth) + target_left * smooth).astype(np.float32)
            if start_left is not None and target_left is not None
            else None
        )
        right = (
            (start_right * (1.0 - smooth) + target_right * smooth).astype(np.float32)
            if start_right is not None and target_right is not None
            else None
        )
        await axol.motion_control(left=left, right=right)
        if progress >= 1.0:
            return
        await asyncio.sleep(period)


async def _verify_arm_targets(
    axol: Axol,
    target_left: np.ndarray | None,
    target_right: np.ndarray | None,
    context: str,
) -> None:
    measured_left, measured_right = await _read_valid_arm_positions(axol, context)
    for side, measured, target in (
        ("left", measured_left, target_left),
        ("right", measured_right, target_right),
    ):
        if target is None:
            continue
        if measured is None:
            raise DiagnosticFailure(f"{context}: {side} arm did not return a position")
        for index, joint in enumerate(ARM_JOINTS):
            tolerance = (
                _SHOULDER_TOLERANCE_RAD
                if joint == Joint.SHOULDER_1
                else _HELD_JOINT_TOLERANCE_RAD
            )
            error = abs(float(measured[index] - target[index]))
            if error > tolerance:
                raise DiagnosticFailure(
                    f"{context}: {side} {joint.value} is "
                    f"{math.degrees(error):.1f} degrees from its target"
                )


async def _stop_lift_verified(
    lift: Lift,
    *,
    context: str,
    require_upper: bool,
    attempts: int = 2,
) -> LiftStatus:
    """Retry canonical STOP and require a fresh, healthy idle reply."""
    failures: list[BaseException] = []
    cancelled: asyncio.CancelledError | None = None
    for _attempt in range(attempts):
        try:
            await lift.stop_motion()
            sent_at = time.monotonic()
            status = await _wait_for_status_after(
                lift,
                sent_at,
                asyncio.Event(),
                context,
            )
            if not _is_idle(status):
                raise DiagnosticFailure(
                    f"{context}: lift still reports active motion: "
                    f"{_status_line(status)}"
                )
            if require_upper and not _at_endpoint(status, _UPPER):
                raise DiagnosticFailure(
                    f"{context}: upper endpoint was not preserved: "
                    f"{_status_line(status)}"
                )
        except asyncio.CancelledError as exc:
            cancelled = cancelled or exc
            failures.append(exc)
        except BaseException as exc:
            failures.append(exc)
        else:
            if cancelled is not None:
                raise cancelled
            return status
    if cancelled is not None:
        for failure in failures:
            if failure is not cancelled:
                cancelled.add_note(
                    "additional STOP verification failure: "
                    f"{type(failure).__name__}: {failure}"
                )
        raise cancelled
    detail = "; ".join(f"{type(exc).__name__}: {exc}" for exc in failures)
    raise DiagnosticFailure(
        f"{context}: STOP could not be sent and verified after {attempts} "
        f"attempt(s): {detail}"
    )


async def _retry_cleanup(
    operation: Callable[[], Awaitable[None]],
    *,
    label: str,
    attempts: int = 2,
) -> None:
    """Retry a teardown operation and surface all failed attempts."""
    failures: list[BaseException] = []
    cancelled: asyncio.CancelledError | None = None
    for _attempt in range(attempts):
        try:
            await operation()
        except asyncio.CancelledError as exc:
            cancelled = cancelled or exc
            failures.append(exc)
        except BaseException as exc:
            failures.append(exc)
        else:
            if cancelled is not None:
                raise cancelled
            return
    if cancelled is not None:
        for failure in failures:
            if failure is not cancelled:
                cancelled.add_note(
                    f"additional {label} failure: {type(failure).__name__}: {failure}"
                )
        raise cancelled
    detail = "; ".join(f"{type(exc).__name__}: {exc}" for exc in failures)
    raise DiagnosticFailure(f"{label} failed after {attempts} attempt(s): {detail}")


async def _run(args: argparse.Namespace) -> None:
    if args.no_left and args.no_right:
        raise SystemExit("ERROR: cannot skip both arms on a mounted-robot lift test.")

    cycles = _resolve_cycles(args.cycles)
    lift: Lift | None = None
    axol: Axol | None = None
    arms_enabled = False
    arms_disabled = False
    arms_at_clearance = False
    completed = False
    upper_verified = False
    primary_error: BaseException | None = None
    primary_traceback = None

    print("=== MOUNTED LIFT CYCLE TEST ===")
    print(
        f"Cycles: {cycles}  |  lift: {args.lift_channel}  |  "
        "S1 clearance: left +90 deg / right -90 deg"
    )
    print("Clear the full arm and lift travel before continuing. Ctrl-C stops.\n")

    try:
        lift = await _open_lift(args.lift_channel)
        with interrupt_event() as interrupted:
            initial_status = _require_fresh_status(lift, "preflight")
            if not initial_status.homed:
                raise DiagnosticFailure(
                    "preflight: the lift is not homed; run `axol lift.home` first"
                )
            if not _is_idle(initial_status):
                raise DiagnosticFailure(
                    f"preflight: the lift is already moving: "
                    f"{_status_line(initial_status)}"
                )
            initial_status = await _wait_for_position_save(
                lift, initial_status, interrupted, "preflight"
            )

            config = AxolConfig(
                left_stiffness=1.0,
                right_stiffness=1.0,
                # This diagnostic only needs the seven load-bearing arm
                # joints. Never enable, calibrate, command, or disable a
                # gripper while exercising the independent lift mechanism.
                has_gripper=False,
            )
            axol = Axol(
                config=config,
                left_channel=None if args.no_left else args.left_channel,
                right_channel=None if args.no_right else args.right_channel,
            )
            print("Enabling arms and holding their measured pose ...")
            # enable() can partially attach before surfacing a motor fault;
            # cleanup must treat the arm state as live from this point onward.
            arms_enabled = True
            await axol.enable()
            await _verify_arms_holding(axol, "arm enable check")
            start_left, start_right = await _read_valid_arm_positions(
                axol, "initial arm pose"
            )
            clearance_left, clearance_right = _clearance_targets(
                start_left, start_right
            )

            async def ensure_lift_ready_for_clearance() -> None:
                _require_lift_ready(lift, "arm clearance lift monitor")

            print("Ramping S1 to the mirrored 90 degree clearance pose ...")
            await _ramp_arms(
                axol,
                start_left,
                start_right,
                clearance_left,
                clearance_right,
                interrupted,
                safety_check=ensure_lift_ready_for_clearance,
            )
            await _verify_arm_targets(
                axol, clearance_left, clearance_right, "clearance check"
            )
            await _verify_arms_holding(axol, "clearance holding check")
            arms_at_clearance = True
            print("Arm clearance pose verified.")

            async def ensure_arm_clearance() -> None:
                nonlocal arms_at_clearance
                try:
                    await _verify_arms_holding(axol, "arm clearance monitor")
                    await _verify_arm_targets(
                        axol,
                        clearance_left,
                        clearance_right,
                        "arm clearance monitor",
                    )
                except Exception:
                    arms_at_clearance = False
                    raise

            latest_stamp = lift.last_status_monotonic or time.monotonic()
            current_status = await _wait_for_status_after(
                lift,
                latest_stamp,
                interrupted,
                "pre-cycle status",
                ensure_arm_clearance,
            )
            if not _is_idle(current_status):
                raise DiagnosticFailure(
                    "pre-cycle: the lift began moving while the arms were "
                    f"entering clearance: {_status_line(current_status)}"
                )
            current_status = await _wait_for_position_save(
                lift,
                current_status,
                interrupted,
                "pre-cycle",
                ensure_arm_clearance,
            )
            if _at_endpoint(current_status, _UPPER):
                upper_verified = True
                print(
                    f"Lift already at the upper endpoint: "
                    f"{_status_line(current_status)}"
                )
            else:
                upper_verified = False
                await _move_lift(
                    lift,
                    _UPPER,
                    args.speed,
                    interrupted,
                    "Initial raise",
                    ensure_arm_clearance,
                )
                upper_verified = True

            for cycle in range(1, cycles + 1):
                print(f"\n--- Cycle {cycle}/{cycles} ---")
                upper_verified = False
                await _move_lift(
                    lift,
                    _LOWER,
                    args.speed,
                    interrupted,
                    f"Cycle {cycle}/{cycles} down",
                    ensure_arm_clearance,
                )
                await _move_lift(
                    lift,
                    _UPPER,
                    args.speed,
                    interrupted,
                    f"Cycle {cycle}/{cycles} up",
                    ensure_arm_clearance,
                )
                upper_verified = True

            # Stop explicitly and verify a fresh upper-endpoint status before
            # declaring success or releasing ownership of either CAN bus.
            upper_verified = False
            await lift.stop_motion()
            stopped_at = lift.last_status_monotonic or time.monotonic()
            final_status = await _wait_for_status_after(
                lift,
                stopped_at,
                interrupted,
                "final stop",
                ensure_arm_clearance,
            )
            if not _at_endpoint(final_status, _UPPER):
                raise DiagnosticFailure(
                    f"final stop: upper endpoint was not preserved: "
                    f"{_status_line(final_status)}"
                )
            upper_verified = True

            async def ensure_lift_upper_stopped() -> None:
                nonlocal upper_verified
                upper_verified = False
                status = _require_fresh_status(lift, "S1 return monitor")
                if not status.homed:
                    raise DiagnosticFailure(
                        "S1 return monitor: lift homing state was lost"
                    )
                if not _at_endpoint(status, _UPPER):
                    raise DiagnosticFailure(
                        "S1 return monitor: lift left its stopped upper endpoint: "
                        f"{_status_line(status)}"
                    )
                await _verify_arms_holding(axol, "S1 return holding monitor")
                upper_verified = True

            return_left, return_right = await _read_valid_arm_positions(
                axol, "S1 return start"
            )
            await _verify_arms_holding(axol, "S1 return holding check")
            await _verify_arm_targets(
                axol, clearance_left, clearance_right, "S1 return clearance check"
            )
            rest_left, rest_right = _rest_targets(return_left, return_right)
            print("Lift is up and stopped; ramping S1 back to 0 degree rest ...")
            arms_at_clearance = False
            await _ramp_arms(
                axol,
                return_left,
                return_right,
                rest_left,
                rest_right,
                interrupted,
                safety_check=ensure_lift_upper_stopped,
            )
            await _verify_arms_holding(axol, "rest holding check")
            await _verify_arm_targets(axol, rest_left, rest_right, "rest check")
            post_rest_stamp = lift.last_status_monotonic or time.monotonic()
            await _wait_for_status_after(
                lift,
                post_rest_stamp,
                interrupted,
                "post-rest lift check",
                ensure_lift_upper_stopped,
            )
            await ensure_lift_upper_stopped()
            print("S1 rest verified; disabling arm motors ...")
            await _disable_arms_verified(axol)
            arms_disabled = True
            completed = True
    except BaseException as exc:
        primary_error = exc
        primary_traceback = exc.__traceback__

    cleanup_errors: list[BaseException] = []
    try:
        # A verified STOP precedes the success-path disable above. This final
        # STOP precedes failure-path arm cleanup; a failed/low lift must never
        # cause the process to torque off arms out of their clearance.
        if lift is not None:
            try:
                await _stop_lift_verified(
                    lift,
                    context="final cleanup stop",
                    require_upper=completed,
                )
            except BaseException as exc:
                cleanup_errors.append(exc)
                print(f"WARNING: {exc}", file=sys.stderr)
        if axol is not None:
            try:
                await _retry_cleanup(
                    axol.disconnect,
                    label="closing arm CAN buses",
                )
            except BaseException as exc:
                cleanup_errors.append(exc)
                print(f"WARNING: {exc}", file=sys.stderr)
        if lift is not None:
            try:
                await _retry_cleanup(lift.close, label="closing lift CAN bus")
            except BaseException as exc:
                cleanup_errors.append(exc)
                print(f"WARNING: {exc}", file=sys.stderr)

        if arms_enabled and not completed and not arms_disabled:
            pose = "90 degree clearance" if arms_at_clearance else "last commanded"
            reason = (
                "the lift was not verified at the upper endpoint"
                if not upper_verified
                else "the diagnostic did not complete"
            )
            print(
                f"WARNING: arms may remain enabled and holding their {pose} pose; "
                f"{reason}.",
                file=sys.stderr,
            )
    except BaseException as exc:
        # Teardown itself must not be interrupted before every independent
        # resource has had its cleanup attempt above.
        cleanup_errors.append(exc)

    if cleanup_errors:
        details = "; ".join(str(exc) for exc in cleanup_errors)
        cleanup_cancelled = next(
            (exc for exc in cleanup_errors if isinstance(exc, asyncio.CancelledError)),
            None,
        )
        if cleanup_cancelled is not None and not isinstance(
            primary_error, asyncio.CancelledError
        ):
            if primary_error is not None:
                cleanup_cancelled.add_note(
                    "diagnostic also failed before cancellation: "
                    f"{type(primary_error).__name__}: {primary_error}"
                )
            primary_error = cleanup_cancelled
            primary_traceback = cleanup_cancelled.__traceback__
        elif primary_error is not None:
            primary_error.add_note(f"additional cleanup failure(s): {details}")
        else:
            primary_error = cleanup_cancelled or DiagnosticFailure(
                f"diagnostic work completed but cleanup was not verified: {details}"
            )
            primary_traceback = primary_error.__traceback__

    if primary_error is not None:
        raise primary_error.with_traceback(primary_traceback)

    print(f"\nPASS — {cycles} complete down/up cycle(s).")
    print("Lift is up; arm motors are disabled after verified S1 rest.")


def _add_arguments(
    parser: argparse.ArgumentParser,
    *,
    cycles_required: bool,
) -> None:
    parser.add_argument(
        "--cycles",
        type=int,
        required=cycles_required,
        default=None,
        help="Positive number of complete down/up cycles; prompts when omitted.",
    )
    parser.add_argument(
        "--lift-channel",
        default=CAN_CHEST,
        metavar="IFACE",
        help="SocketCAN interface carrying Jelly Legs (default: %(default)s).",
    )
    parser.add_argument(
        "--speed",
        type=_speed_value,
        default=0,
        help="Lift speed cap in counts/s: 0 for full speed, or 250-65535.",
    )
    parser.add_argument("--no-left", action="store_true", help="Skip the left arm.")
    parser.add_argument("--no-right", action="store_true", help="Skip the right arm.")
    parser.add_argument(
        "--left-channel",
        default=CAN_LEFT,
        metavar="IFACE",
        help="SocketCAN interface for the left arm (default: %(default)s).",
    )
    parser.add_argument(
        "--right-channel",
        default=CAN_RIGHT,
        metavar="IFACE",
        help="SocketCAN interface for the right arm (default: %(default)s).",
    )


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register ``diag.lift-cycle`` for dashboard schema introspection."""
    parser = subparsers.add_parser(
        "diag.lift-cycle",
        help="Move S1 to clearance, then cycle the lift down/up.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    _add_arguments(parser, cycles_required=True)
    parser.set_defaults(func=run_cli)


def run_cli(args: argparse.Namespace) -> None:
    """Run the mounted lift diagnostic from parsed arguments."""
    try:
        asyncio.run(_run(args))
    except Interrupted:
        raise SystemExit(
            "\nInterrupted — lift STOP attempted; inspect lift and arm state."
        ) from None
    except DiagnosticFailure as exc:
        raise SystemExit(f"ERROR: {exc}") from None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="axol diag.lift-cycle",
        description="Cycle Jelly Legs with the mounted Axol arms at clearance.",
    )
    _add_arguments(parser, cycles_required=False)
    run_cli(parser.parse_args(argv))


if __name__ == "__main__":
    main()
