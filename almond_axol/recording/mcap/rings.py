"""Column layouts of the two shared-memory rings the MCAP recorder reads.

Both rings are :class:`~almond_axol.video.shm_frames.SnapshotWriter` blocks:
one row of float64 per sample, ``[ts, *values, intervention]``, written by the
control process and read in order by the recorder subprocess.

- The **joint ring** gets one row per Rust core feedback packet pair (240 Hz):
  measured position, velocity and torque of all 16 joints.
- The **tick ring** gets one row per control tick (120 Hz): the commanded
  joint targets, the headset's raw input, and Jelly's latest wheel and lift
  readings. Values that do not exist for a tick are NaN.

Each column name maps to an index the recorder reads with numpy. Keeping both
layouts here means the writer and the reader cannot disagree on an order.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ...constants import Joint

SIDES = ("left", "right")
# Joint names in Joint order (the 7 arm joints, then the gripper).
JOINT_NAMES: tuple[str, ...] = tuple(j.value for j in Joint)
NAN = float("nan")

# -- joint ring (240 Hz) -----------------------------------------------------

JOINT_FIELDS = ("pos", "vel", "trq")


def joint_ring_keys() -> list[str]:
    """``{field}.{side}_{joint}`` for position, velocity and torque."""
    return [
        f"{field}.{side}_{joint}"
        for field in JOINT_FIELDS
        for side in SIDES
        for joint in JOINT_NAMES
    ]


JOINT_RING_CAPACITY = 2048  # ~8.5 s at 240 Hz

# -- tick ring (120 Hz) ------------------------------------------------------

POSE_AXES = ("px", "py", "pz", "qx", "qy", "qz", "qw")
VR_AXES = ("l_grip", "r_grip", "l_stick_x", "l_stick_y", "r_stick_x", "r_stick_y")
VR_BUTTONS = (
    "l_lock",
    "r_lock",
    "l_stick_click",
    "r_stick_click",
    "reset",
    "stow",
    "l_tracked",
    "r_tracked",
)
WHEEL_IDS = (1, 2, 3, 4)

# How a device applies its commands, as recorded in each command message's
# ``control_mode``. The ring stores the index.
CONTROL_MODES = (
    "impedance",  # arm joints: stiffness and damping toward the target
    "gravity_compensation",  # arm joints gone limp: targets are not followed
    "position_force",  # grippers: position with a force limit
    "velocity",  # Jelly's wheels while driving
    "position_hold",  # Jelly's wheels while parked
)


def mode_index(mode: str) -> float:
    return float(CONTROL_MODES.index(mode))


def mode_name(value: float) -> str:
    if math.isfinite(value) and 0 <= int(value) < len(CONTROL_MODES):
        return CONTROL_MODES[int(value)]
    return ""


def tick_ring_keys() -> list[str]:
    keys = ["cmd.id", "cmd.issue_t", "cmd.arm_mode"]
    keys += [f"cmd.{side}_{joint}" for side in SIDES for joint in JOINT_NAMES]
    keys += ["vr.seq", "vr.t_host"]
    keys += [f"vr.{side}.{axis}" for side in ("l", "r") for axis in POSE_AXES]
    keys += [f"vr.{name}" for name in VR_AXES + VR_BUTTONS]
    keys += ["jelly.t"]
    keys += [f"jelly.pos{i}" for i in WHEEL_IDS]
    keys += [f"jelly.vel{i}" for i in WHEEL_IDS]
    keys += [f"jelly.cmd{i}" for i in WHEEL_IDS]
    keys += ["jelly.lift", "jelly.parked"]
    keys += [
        "jelly.cmd_id",
        "jelly.cmd_t",
        "jelly.cmd_vx",
        "jelly.cmd_vy",
        "jelly.cmd_wz",
        "jelly.cmd_lift",
    ]
    return keys


TICK_RING_CAPACITY = 1024  # ~8.5 s at 120 Hz


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return NAN
    return out if math.isfinite(out) else NAN


def pack_tick(
    command: Mapping[str, Any] | None,
    frame: Any = None,
    jelly: Any = None,
    *,
    perf_minus_monotonic: float = 0.0,
    command_id: int | None = None,
    issue_t: float | None = None,
    arm_mode: str = "impedance",
) -> dict[str, float]:
    """One tick-ring row from the control loop's live values.

    ``command`` is the teleop's joint-target action (``{side}_{joint}.pos``
    keys), sent to the arms as command ``command_id`` at ``issue_t``
    (``perf_counter``) with the arms in ``arm_mode`` (one of
    :data:`CONTROL_MODES`). ``frame`` is the headset's latest
    :class:`~almond_axol.vr.models.VRFrame` (or ``None``). ``jelly`` is the
    running :class:`~almond_axol.robot.jelly.Jelly` (or ``None``); its times
    are on :func:`time.monotonic`, moved onto ``perf_counter`` by
    ``perf_minus_monotonic``.
    """
    row: dict[str, float] = {
        "cmd.id": NAN if command_id is None else float(command_id),
        "cmd.issue_t": NAN if issue_t is None else float(issue_t),
        "cmd.arm_mode": mode_index(arm_mode),
    }
    for side in SIDES:
        for joint in JOINT_NAMES:
            key = f"{side}_{joint}"
            row[f"cmd.{key}"] = _finite(
                command.get(f"{key}.pos") if command is not None else None
            )

    row["vr.seq"] = NAN
    row["vr.t_host"] = NAN
    for side in ("l", "r"):
        for axis in POSE_AXES:
            row[f"vr.{side}.{axis}"] = NAN
    for name in VR_AXES + VR_BUTTONS:
        row[f"vr.{name}"] = NAN
    if frame is not None:
        row["vr.seq"] = _finite(getattr(frame, "seq", None))
        row["vr.t_host"] = _finite(getattr(frame, "t_host", None))
        for side, pose in (("l", frame.l_ee), ("r", frame.r_ee)):
            values = (
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.quaternion.x,
                pose.quaternion.y,
                pose.quaternion.z,
                pose.quaternion.w,
            )
            for axis, value in zip(POSE_AXES, values, strict=True):
                row[f"vr.{side}.{axis}"] = _finite(value)
        for name in VR_AXES:
            row[f"vr.{name}"] = _finite(getattr(frame, name, None))
        for name in VR_BUTTONS:
            value = getattr(frame, name, None)
            row[f"vr.{name}"] = NAN if value is None else (1.0 if value else 0.0)

    row["jelly.t"] = NAN
    for i in WHEEL_IDS:
        row[f"jelly.pos{i}"] = NAN
        row[f"jelly.vel{i}"] = NAN
        row[f"jelly.cmd{i}"] = NAN
    row["jelly.lift"] = NAN
    row["jelly.parked"] = NAN
    for key in ("id", "t", "vx", "vy", "wz", "lift"):
        row[f"jelly.cmd_{key}"] = NAN
    if jelly is not None:
        sample = getattr(jelly, "wheel_sample", None)
        if sample is not None and sample.at > 0:
            row["jelly.t"] = float(sample.at) + perf_minus_monotonic
            for i, (p, v) in enumerate(
                zip(sample.positions, sample.velocities, strict=True), start=1
            ):
                row[f"jelly.pos{i}"] = _finite(p)
                row[f"jelly.vel{i}"] = _finite(v)
        speeds: Sequence[float] = getattr(jelly, "wheel_speeds", ()) or ()
        if getattr(jelly, "has_wheels", False):
            for i, value in zip(WHEEL_IDS, speeds, strict=False):
                row[f"jelly.cmd{i}"] = _finite(value)
        row["jelly.lift"] = _finite(getattr(jelly, "lift_raised", None))
        parked = getattr(jelly, "parked", None)
        row["jelly.parked"] = NAN if parked is None else (1.0 if parked else 0.0)
        latest = getattr(jelly, "latest_command", None)
        if latest is not None:
            cmd_id, issued, vx, vy, wz, lift = latest
            row["jelly.cmd_id"] = float(cmd_id)
            row["jelly.cmd_t"] = float(issued) + perf_minus_monotonic
            row["jelly.cmd_vx"] = _finite(vx)
            row["jelly.cmd_vy"] = _finite(vy)
            row["jelly.cmd_wz"] = _finite(wz)
            row["jelly.cmd_lift"] = _finite(lift)
    return row


def column_index(keys: Sequence[str]) -> dict[str, int]:
    """Column of each key in a ring row (``ts`` is column 0)."""
    return {key: i + 1 for i, key in enumerate(keys)}
