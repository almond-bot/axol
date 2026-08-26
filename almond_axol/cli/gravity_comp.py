"""
axol gravity-comp

Put the Axol arms into gravity-compensation mode so the operator can move them
by hand. Each free arm joint is sent ``set_impedance(p_des=current, v_des=0,
kp=0, kd=KD, t_ff=gravity)`` at the configured rate; joints not in the free
set are held rigidly at their current position with their configured
``ArmConfig`` gains; the gripper is held softly at its current position.

Every field is reachable from the CLI (draccus-style) or a JSON/YAML file:

    axol gravity-comp
    axol gravity-comp --right_channel null
    axol gravity-comp --kd 1.0
    axol gravity-comp --free_joints [WRIST_3]
    axol gravity-comp --right_channel null --free_joints [SHOULDER_1,WRIST_3]
    axol gravity-comp --record demo1
    axol gravity-comp --config_path my_gravity.json

``--record`` captures the hand-guided session (measured arm-joint positions
and torques, one row per control tick) to ``<prefix>_gc.npz`` with the same
flight recorder teleop uses, so ``axol motion.build`` can turn a hand-guided
demonstration into a reference motion.
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from ..constants import ARM_JOINTS, Joint
from ..robot import Axol
from ..teleop.recorder import TeleopRecorder
from ..teleop.recorder import make as _recorder_make
from .config import GravityCompCmdConfig, parse


def _resolve_free_joints(names: list[str] | None) -> set[Joint] | None:
    """Convert a list of joint names into a set of arm ``Joint`` enums.

    Names are case-insensitive and must name one of the seven arm joints
    (``GRIPPER`` is rejected — gravity comp only applies to arm joints).
    ``None`` means "all seven arm joints" and is passed through unchanged.
    """
    if names is None:
        return None
    valid_names = [j.name for j in ARM_JOINTS]
    out: set[Joint] = set()
    for raw in names:
        name = raw.strip().upper()
        if not name:
            continue
        try:
            j = Joint[name]
        except KeyError:
            raise SystemExit(f"unknown joint {name!r}; valid: {', '.join(valid_names)}")
        if j not in ARM_JOINTS:
            raise SystemExit(
                f"{name!r} cannot be gravity-compensated; valid: {', '.join(valid_names)}"
            )
        out.add(j)
    if not out:
        raise SystemExit("free_joints is empty")
    return out


def main(argv: list[str]) -> None:
    """Parse the CLI config and run gravity-compensation mode."""
    cfg = parse(GravityCompCmdConfig, argv)
    # force=True: a dependency imported before this point may install a root
    # handler (leaving the level at WARNING), which would make this a no-op
    # and silently drop log_say() / INFO status lines.
    logging.basicConfig(level=getattr(logging, cfg.log_level), force=True)
    try:
        asyncio.run(_run(cfg))
    except KeyboardInterrupt:
        print("\nExiting gravity comp ...")


async def _run(cfg: GravityCompCmdConfig) -> None:
    if cfg.left_channel is None and cfg.right_channel is None:
        raise SystemExit("Both arms disabled — nothing to do.")

    free_joints = _resolve_free_joints(cfg.free_joints)
    free_str = (
        "all 7 joints"
        if free_joints is None
        else ", ".join(j.name for j in ARM_JOINTS if j in free_joints)
    )
    print(
        f"Gravity comp: free={free_str}; kd={cfg.kd:.2f} Nm·s/rad, "
        f"rate={cfg.rate_hz:.0f} Hz (telemetry={cfg.telemetry_hz:.0f} Hz). "
        f"Press Ctrl-C to exit."
    )

    # Flight recorder (--record): measured arm-joint positions/torques,
    # 7 left + 7 right (ARM_JOINTS order, no grippers — the row layout
    # motion.build consumes). A gravity-comp session has no engage state, so
    # the whole session is one segment: engage on entry, flush on exit (the
    # falling edge writes the file; Ctrl-C and a serve-op cancellation both
    # unwind through the finally). motion.build trims the still
    # lead-in/lead-out.
    rec = _recorder_make(cfg.record, "gc", {"qm": 14, "tq": 14})

    async with Axol(
        config=cfg.axol,
        left_channel=cfg.left_channel,
        right_channel=cfg.right_channel,
    ) as axol:
        # ``enable()`` (called by ``__aenter__``) leaves arm joints in IMPEDANCE
        # and the gripper in POSITION_FORCE — both of which are the modes
        # ``gravity_compensate`` expects, so we don't touch control modes here.
        await axol.start_telemetry(cfg.telemetry_hz)
        # Motors may still be rebooting from set_control_mode(); block until
        # every motor has answered at least one telemetry poll before driving.
        await axol.wait_for_telemetry()

        if rec is not None:
            rec.set_engaged(True)
        dt = 1.0 / cfg.rate_hz
        try:
            while True:
                loop_start = time.monotonic()
                await axol.gravity_compensate(kd=cfg.kd, free_joints=free_joints)
                if rec is not None:
                    _record_measured(rec, axol)
                spent = time.monotonic() - loop_start
                if spent < dt:
                    await asyncio.sleep(dt - spent)
        finally:
            if rec is not None:
                rec.set_engaged(False)


def _record_measured(rec: TeleopRecorder, axol: Axol) -> None:
    """Append one measured-side row (arm joints only) to the recorder.

    Reads the cached positions/torques the impedance feedback frames refresh
    every cycle — no CAN traffic. A disabled arm records NaN (motion.build
    rejects such a capture with an actionable message).
    """
    n = len(ARM_JOINTS)
    qm = np.full(14, np.nan, dtype=np.float32)
    tq = np.full(14, np.nan, dtype=np.float32)
    for sl, arm in (
        (slice(0, 7), getattr(axol, "left", None)),
        (slice(7, 14), getattr(axol, "right", None)),
    ):
        if arm is None:
            continue
        try:
            qm[sl] = arm.positions[:n]
            tq[sl] = arm.torques[:n]
        except Exception:  # noqa: BLE001 - recording must never break the loop
            pass
    rec.record(qm=qm, tq=tq)
