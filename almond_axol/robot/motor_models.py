"""The motor models this robot's arms were last seen with.

Axol's shoulders ship as one of two variants of the same actuator, the
RMD-X8-P20-120 and the RMD-X8-P20-150, and the two need different default
gains and friction (see :data:`almond_axol.robot.config._X8_150_GAINS`). A
motor only tells its model when it is powered, which it is not when a config
is built. So every bring-up that reads the models records them here, keyed to
the arm hub like the calibration file, and the config takes its defaults from
the record.

The record is a cache of what the motors said, never a setting: it is
rewritten whenever a bring-up reads something else (an arm swapped onto the
hub), and a robot without one runs the X8-P20-120 defaults until its first
bring-up has written it.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .identity import hub_serial as current_hub_serial

_logger = logging.getLogger(__name__)

MOTOR_MODELS_PATH = Path.home() / ".almond" / "motor_models.json"

_SIDES = ("left", "right")

#: The X8-P20-150: firmware 2026083101, factory-dated 2026-09.
X8_150 = "x8-150"

_warned_invalid = False


def shoulder_variant(model: str | None) -> str | None:
    """The gain variant a shoulder motor's model selects, or ``None``.

    ``None`` is the X8-P20-120 the fleet's defaults were tuned on, and also
    any model that is unknown or read without its suffix: only a motor that
    said it is a -150 gets the -150 set.
    """
    if not model:
        return None
    match = re.search(r"X(\d+)-[^-]*-(\d+)", model.strip().upper())
    if match is None:
        return None
    if (int(match.group(1)), int(match.group(2))) == (8, 150):
        return X8_150
    return None


def load_motor_models(
    path: Path | None = None, *, expected_hub_serial: str | None = None
) -> dict[str, dict[str, str]]:
    """``{side: {joint: model}}`` from the record, empty where unknown.

    With ``expected_hub_serial`` a record written for another hub reads as
    empty: those are another robot's arms.
    """
    global _warned_invalid
    path = MOTOR_MODELS_PATH if path is None else path
    out: dict[str, dict[str, str]] = {side: {} for side in _SIDES}
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return out
    except Exception:  # noqa: BLE001 - a corrupt cache must not kill the robot
        if not _warned_invalid:
            _warned_invalid = True
            _logger.warning(
                "Ignoring unreadable motor record %s; the next bring-up rewrites it.",
                path,
            )
        return out
    if not isinstance(raw, dict):
        return out
    if expected_hub_serial is not None and raw.get("hub_serial") != expected_hub_serial:
        return out
    for side in _SIDES:
        joints = raw.get(side)
        if not isinstance(joints, dict):
            continue
        out[side] = {
            str(joint): model
            for joint, model in joints.items()
            if isinstance(model, str) and model
        }
    return out


def record_motor_models(
    side: str,
    models: Mapping[str, str | None],
    *,
    hub_serial: str | None = None,
    path: Path | None = None,
) -> bool:
    """Record the models one arm's motors just reported.

    ``models`` maps joint names to model strings; a joint that did not
    answer (``None``) keeps what the record already holds. A record for
    another hub is replaced: these are the motors on the hub that is
    attached now. Returns whether the record changed.
    """
    if side not in _SIDES:
        raise ValueError(f"side must be one of {_SIDES}, got {side!r}")
    path = MOTOR_MODELS_PATH if path is None else path
    if hub_serial is None:
        hub_serial = current_hub_serial()
    if not hub_serial:
        # No hub identity to key the record to (a bench adapter): the arm on
        # it is not this robot's, and a later robot must not inherit it.
        return False
    seen = {joint: model for joint, model in models.items() if model}
    if not seen:
        return False
    stored = load_motor_models(path, expected_hub_serial=hub_serial)
    merged = {**stored[side], **seen}
    if merged == stored[side]:
        return False
    stored[side] = merged
    doc = {"version": 1, "hub_serial": hub_serial, **stored}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return True


def recorded_model(side: str, joint: str) -> str | None:
    """The model the record holds for ``side``'s ``joint`` on this hub.

    ``None`` when there is no hub, no record or no answer from that motor:
    the model is not known.
    """
    try:
        serial = current_hub_serial()
    except (OSError, RuntimeError):
        return None
    if not serial:
        return None
    models = load_motor_models(expected_hub_serial=serial)
    return models.get(side, {}).get(joint)


def recorded_shoulder_variant(side: str, joint: str) -> str | None:
    """The gain variant the record gives ``side``'s ``joint`` on this hub.

    ``None`` when the model is not known: the X8-P20-120 defaults apply.
    """
    return shoulder_variant(recorded_model(side, joint))


def note_motor_models(side: str, motors: Mapping[Any, Any]) -> None:
    """Record the models ``side``'s motor drivers read at bring-up.

    ``motors`` maps joints to motor objects; the ones whose driver read a
    model (the MyActuators) are recorded. When a shoulder turns out to be a
    different variant than the record said, the config this session was
    built from holds the other variant's defaults: say so. Never raises: the
    record is a cache, and a bring-up must not fail on it.
    """
    try:
        models: dict[str, str | None] = {}
        for joint, motor in motors.items():
            model = getattr(getattr(motor, "_driver", None), "_model", None)
            if isinstance(model, str) and model:
                models[str(getattr(joint, "value", joint))] = model
        try:
            serial = current_hub_serial()
        except (OSError, RuntimeError):
            return
        if not serial or not models:
            return
        before = load_motor_models(expected_hub_serial=serial)[side]
        if not record_motor_models(side, models, hub_serial=serial):
            return
        for joint, model in models.items():
            if model and shoulder_variant(model) != shoulder_variant(before.get(joint)):
                _logger.warning(
                    "The %s arm's %s is an %s. This session started with the "
                    "settings for a different shoulder motor. Start it again to "
                    "use the settings for this one.",
                    side,
                    joint,
                    model,
                )
    except Exception:  # noqa: BLE001 - a cache write must never stop a bring-up
        _logger.warning(
            "Could not record the %s arm's motor models.", side, exc_info=True
        )
