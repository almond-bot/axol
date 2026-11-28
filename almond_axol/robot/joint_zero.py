"""The record of every joint zeroing on this robot, at ``~/.almond/joint_zero.json``.

Each arm joint's encoder zero is set at a mechanical end stop and saved in
the motor's flash (``axol motor.set-zero-pos``, or the SDK's
``set_zero_position``). The motors keep no history, so every zeroing also
appends an entry here: an ID, the time, the robot, the arm, how it was done,
and per joint the motor ID, the end stop and the reading at the stop just
before the new zero was saved (how far the old zero was from it).

MCAP episodes and camera calibrations name the zeroing each arm was on, so
an episode shows which zeroing its joint angles, tool frames and wrist camera
mounts depend on, and whether an arm was re-zeroed after the cameras were
calibrated.

File shape::

    {
      "format": "almond-joint-zero",
      "version": 1,
      "events": [
        {
          "zero_id": "E66138528F2B3F2A-20261201-zero-01",
          "zeroed_at": "2026-12-01T18:15:00Z",
          "robot_serial": "E66138528F2B3F2A",
          "side": "left",
          "method": "guided end stop",
          "joints": {
            "shoulder_1": {"motor_id": 1, "end_stop_deg": -90.0,
                           "reading_before_rad": 0.0012}
          },
          "software_version": "0.2.48",
          "hostname": "axol"
        }
      ]
    }

This module is I/O only (stdlib imports) so the recorder process and the
zeroing commands can use it without loading the robot stack.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

FILE_NAME = "joint_zero.json"
FORMAT = "almond-joint-zero"
VERSION = 1
SIDES = ("left", "right")
# The seven arm joints, in control order (constants.ARM_JOINTS values).
ARM_JOINT_NAMES = (
    "shoulder_1",
    "shoulder_2",
    "shoulder_3",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)

METHOD_GUIDED = "guided end stop"
METHOD_SINGLE = "single motor"
METHOD_SDK = "SDK set_zero_position"


def default_path() -> Path:
    """``~/.almond/joint_zero.json`` (under ``ALMOND_HOME`` when it is set)."""
    from ..utils.paths import almond_path

    return almond_path(FILE_NAME)


def _now_text(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load(path: Path | None = None) -> list[dict[str, Any]]:
    """Every recorded zeroing, oldest first. Empty when there is no file.

    Raises ``ValueError`` when the file exists but isn't a zeroing record.
    """
    path = Path(path).expanduser() if path is not None else default_path()
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return []
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name} is not valid JSON ({exc})") from None
    if not isinstance(doc, Mapping) or doc.get("format") != FORMAT:
        raise ValueError(f"{path.name} is not a joint zeroing record")
    events = doc.get("events")
    if not isinstance(events, list):
        raise ValueError(f"{path.name} has no events list")
    return [e for e in events if isinstance(e, Mapping)]


def _write(events: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    data = json.dumps(
        {"format": FORMAT, "version": VERSION, "events": events}, indent=2
    )
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        from ..utils.paths import adopt_state_file

        adopt_state_file(path)
    except Exception:  # noqa: BLE001 - ownership is a convenience
        pass


def next_zero_id(robot_serial: str, when: datetime, events: Iterable[Mapping]) -> str:
    """``<serial>-<date>-zero-<run>``: the next unused ID for this robot and day."""
    prefix = f"{robot_serial or 'UNKNOWN'}-{when:%Y%m%d}-zero-"
    runs = []
    for event in events:
        zero_id = str(event.get("zero_id", ""))
        if zero_id.startswith(prefix) and zero_id[len(prefix) :].isdigit():
            runs.append(int(zero_id[len(prefix) :]))
    return f"{prefix}{max(runs, default=0) + 1:02d}"


def _software_version() -> str:
    try:
        from importlib.metadata import version

        return version("almond-axol")
    except Exception:  # noqa: BLE001 - informational
        return ""


def _hub_serial() -> str:
    try:
        from .identity import hub_serial

        return hub_serial() or ""
    except Exception:  # noqa: BLE001 - informational
        return ""


def record(
    side: str,
    joints: Mapping[str, Mapping[str, Any]],
    method: str,
    *,
    robot_serial: str | None = None,
    path: Path | None = None,
    now: datetime | None = None,
    hostname: str | None = None,
) -> dict[str, Any]:
    """Append one zeroing of ``joints`` on the ``side`` arm and return it.

    ``joints`` maps a joint name (``shoulder_1`` ... ``wrist_3``, or
    ``gripper``) to what is known about it: ``motor_id``, ``end_stop_deg``,
    ``reading_before_rad``. A file that can't be read is kept beside the new
    one (``.unreadable-<time>``) rather than overwritten.
    """
    if side not in SIDES:
        raise ValueError(f"side must be left or right, not {side!r}")
    path = Path(path).expanduser() if path is not None else default_path()
    now = (now or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
    try:
        events = [dict(e) for e in load(path)]
    except ValueError as exc:
        kept = path.with_name(f"{path.name}.unreadable-{now:%Y%m%dT%H%M%SZ}")
        os.replace(path, kept)
        _logger.warning("%s; kept it as %s and started a new record", exc, kept)
        events = []
    serial = robot_serial if robot_serial is not None else _hub_serial()
    event = {
        "zero_id": next_zero_id(serial, now, events),
        "zeroed_at": _now_text(now),
        "robot_serial": serial,
        "side": side,
        "method": method,
        "joints": {name: dict(info) for name, info in joints.items()},
        "software_version": _software_version(),
        "hostname": hostname if hostname is not None else socket.gethostname(),
    }
    events.append(event)
    _write(events, path)
    return event


def record_quietly(
    side: str,
    joints: Mapping[str, Mapping[str, Any]],
    method: str,
    *,
    path: Path | None = None,
) -> dict[str, Any] | None:
    """:func:`record`, logging instead of raising: a zero already saved in
    the motors must not look failed because the record couldn't be written."""
    if not joints:
        return None
    try:
        return record(side, joints, method, path=path)
    except Exception as exc:  # noqa: BLE001 - the zero itself succeeded
        where = path or default_path()
        _logger.warning("Joint zero saved, but not recorded in %s: %s", where, exc)
        print(f"  ⚠  zero saved, but not recorded in {where}: {exc}")
        return None


@dataclass
class ZeroState:
    """Which zeroing each arm joint is on, from the record."""

    robot_serial: str
    path: Path = Path(FILE_NAME)
    # {side: {joint: event}} for every arm joint with a recorded zeroing.
    joints: dict[str, dict[str, Mapping[str, Any]]] = field(default_factory=dict)
    # Each event's position in the file (later zeroings come later).
    order: dict[int, int] = field(default_factory=dict)
    problem: str = ""
    sha256: str = ""
    raw: bytes = b""

    def arm(self, side: str) -> Mapping[str, Any] | None:
        """The arm's latest zeroing (the newest of its joints' zeroings)."""
        events = list(self.joints.get(side, {}).values())
        if not events:
            return None
        return max(events, key=lambda e: self.order.get(id(e), -1))

    def arm_id(self, side: str) -> str:
        """An ID for the zero the whole arm is on: its latest zeroing's ID when
        every joint is recorded, else ``""``."""
        recorded = self.joints.get(side, {})
        if any(j not in recorded for j in ARM_JOINT_NAMES):
            return ""
        latest = self.arm(side)
        return str(latest.get("zero_id", "")) if latest else ""

    def unrecorded(self, side: str) -> list[str]:
        recorded = self.joints.get(side, {})
        return [j for j in ARM_JOINT_NAMES if j not in recorded]

    @property
    def status(self) -> str:
        if self.problem:
            return "no record"
        missing = sum(len(self.unrecorded(s)) for s in SIDES)
        if missing == len(SIDES) * len(ARM_JOINT_NAMES):
            return "no record"
        return "recorded" if missing == 0 else "partly recorded"

    def ids(self) -> dict[str, str]:
        """``{side: arm zero ID}`` for the camera calibration file."""
        return {side: self.arm_id(side) for side in SIDES}


def state(robot_serial: str, path: Path | None = None) -> ZeroState:
    """The zeroing each arm joint of ``robot_serial`` is on.

    Entries for another robot (an arm hub that was swapped) are ignored.
    """
    path = Path(path).expanduser() if path is not None else default_path()
    out = ZeroState(robot_serial=robot_serial, path=path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        out.problem = f"there is no joint zeroing record ({path})"
        return out
    out.raw = raw
    out.sha256 = hashlib.sha256(raw).hexdigest()
    try:
        events = load(path)
    except ValueError as exc:
        out.problem = str(exc)
        return out
    for index, event in enumerate(events):
        out.order[id(event)] = index
        if robot_serial and event.get("robot_serial") not in (robot_serial, ""):
            continue
        side = event.get("side")
        joints = event.get("joints")
        if side not in SIDES or not isinstance(joints, Mapping):
            continue
        for joint in joints:
            if joint in ARM_JOINT_NAMES:
                out.joints.setdefault(side, {})[joint] = event
    if out.status == "no record":
        out.problem = f"{path.name} has no zeroing for robot {robot_serial or '?'}"
    return out


def episode_metadata(
    zero: ZeroState, calibration_zero: Mapping[str, str] | None
) -> dict[str, Any]:
    """The ``joint_zero`` metadata record of an MCAP episode.

    ``calibration_zero`` is the ``joint_zero`` the camera calibration file
    recorded (``{side: zero ID}``), or ``None`` without a calibration.
    """
    out: dict[str, Any] = {
        "status": zero.status,
        "problem": zero.problem,
        "file": zero.path.name if zero.raw else "",
        "file_sha256": zero.sha256,
    }
    per_joint: dict[str, dict[str, str]] = {}
    for side in SIDES:
        latest = zero.arm(side)
        out[f"{side}.zero_id"] = zero.arm_id(side)
        out[f"{side}.last_zeroed_at"] = (
            str(latest.get("zeroed_at", "")) if latest else ""
        )
        out[f"{side}.unrecorded_joints"] = ",".join(zero.unrecorded(side))
        for joint, event in zero.joints.get(side, {}).items():
            per_joint[f"{side}_{joint}"] = {
                "zero_id": str(event.get("zero_id", "")),
                "zeroed_at": str(event.get("zeroed_at", "")),
            }
    out["joints"] = json.dumps(per_joint, sort_keys=True)
    out["camera_calibration"] = calibration_match(zero, calibration_zero)
    return out


def calibration_match(
    zero: ZeroState, calibration_zero: Mapping[str, str] | None
) -> str:
    """Whether each arm is still on the zero the camera calibration used.

    ``same``; ``changed: ...`` naming each arm re-zeroed since; ``unknown``
    when either side has no record; ``""`` without a camera calibration.
    """
    if calibration_zero is None:
        return ""
    changed, unknown = [], False
    for side in SIDES:
        then = str(calibration_zero.get(side) or "")
        now = zero.arm_id(side)
        if not then or not now:
            unknown = True
        elif then != now:
            latest = zero.arm(side)
            when = str(latest.get("zeroed_at", "")) if latest else ""
            changed.append(f"{side} arm re-zeroed on {when} ({now})")
    if changed:
        return "changed: " + "; ".join(changed)
    return "unknown" if unknown else "same"


def summary_line(event: Mapping[str, Any]) -> str:
    """One line for the operator after a zeroing is recorded."""
    joints = ", ".join(event.get("joints", {}))
    return (
        f"Recorded zeroing {event.get('zero_id')} ({event.get('side')} arm: {joints})."
    )
