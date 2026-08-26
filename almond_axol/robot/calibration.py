"""Per-robot joint calibration persisted at ``~/.almond/calibration.json``.

The friction and gain constants baked into :mod:`almond_axol.robot.config`
were measured on one reference robot. Motor-to-motor friction varies enough
between builds that every robot should be calibrated with ``axol
tune.friction`` (and optionally ``axol tune.pid``) — both commands take
``--save``, which writes the identified values here. The file is read by
:class:`~almond_axol.robot.config.AxolConfig` every time a config is
constructed, so saved values override the shared defaults on this machine
while explicit overrides (draccus dotted flags, the control panel's Advanced
settings) still win over both.

File shape (every level optional)::

    {
      "version": 1,
      "left": {
        "elbow": {
          "kp": 40.0,
          "kd": 3.0,
          "j_eff": 0.0,
          "friction": {"fc": 0.68, "k": 801.3, "fv": 0.87, "fo": -0.25},
          "com": [-0.0251, 0.0, -0.0712],
          "updated_at": "2026-08-16T01:00:00Z"
        }
      },
      "right": { ... }
    }

``com`` is the fitted centre of mass of the link this joint drives (metres,
URDF link frame — same convention as ``JointConfig.com``), written by ``axol
tune.gravity --save``. It overrides the CAD value in the gravity model, which
is what fixes the static droop a few-percent mass/CoM error causes under
load (parked error = unmodeled torque / kp).

``kp`` / ``kd`` are the tuned gains — the top of the stiffness blend
(``s=1.0``, the production default) — exactly like the :class:`JointConfig`
defaults they replace.

This module is deliberately I/O-only (no imports from ``config``) so
``config.py`` can consume it without an import cycle.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

CALIBRATION_PATH = Path.home() / ".almond" / "calibration.json"

# This robot's factory calibration as fetched from the cloud (``axol
# calibration.pull`` — see :mod:`.calibration_cloud`). Same file shape as
# ``calibration.json``. It sits *between* the coded defaults and the local
# file in the override order: coded config ← factory data ← calibration.json
# — a value tuned locally always wins over the factory's.
FACTORY_CALIBRATION_PATH = Path.home() / ".almond" / "factory_calibration.json"

_SIDES = ("left", "right")
# ``kd_soft`` entries written by older versions are silently dropped on load.
_SCALAR_FIELDS = ("kp", "kd", "j_eff", "kd_host", "kd_host_hz", "kd_host_q")
_FRICTION_FIELDS = ("fc", "k", "fv", "fo")

# A corrupt file must never take the robot down, but silently ignoring it
# would make a bad calibration mysterious — warn once per process.
_warned_invalid = False


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def load_calibration(path: Path = CALIBRATION_PATH) -> dict[str, dict[str, Any]]:
    """Read and sanitize the calibration file.

    Returns ``{"left": {joint: {...}}, "right": {joint: {...}}}`` with only
    recognized, numeric fields kept; an absent or unreadable file yields empty
    sides. Joint keys are not validated here — the consumer looks joints up
    by name and ignores strangers — so a file written by a newer version with
    extra joints degrades gracefully.
    """
    global _warned_invalid
    out: dict[str, dict[str, Any]] = {side: {} for side in _SIDES}
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return out
    except Exception:  # noqa: BLE001 - a corrupt file must not kill the robot
        if not _warned_invalid:
            _warned_invalid = True
            _logger.warning(
                "Ignoring unreadable calibration file %s — fix or delete it.", path
            )
        return out
    if not isinstance(raw, dict):
        if not _warned_invalid:
            _warned_invalid = True
            _logger.warning(
                "Ignoring calibration file %s: top level is not an object.", path
            )
        return out

    for side in _SIDES:
        joints = raw.get(side)
        if not isinstance(joints, dict):
            continue
        for joint, entry in joints.items():
            if not isinstance(entry, dict):
                continue
            clean: dict[str, Any] = {}
            for field in _SCALAR_FIELDS:
                value = _coerce_float(entry.get(field))
                if value is not None:
                    if field == "kd" and value > 5.0:
                        # Saved under the old 0-50 kd wire encoding (the
                        # MIT kd field actually decodes against 0-5 on every
                        # firmware): divide by 10 to preserve the behavior
                        # the value was tuned to. No legitimate post-fix kd
                        # exceeds 5 — the encoder clamps there.
                        _logger.warning(
                            "Calibration for %s %s has kd=%.1f from the old "
                            "0-50 encoding; interpreting as %.2f on the true "
                            "0-5 scale. Re-save to silence this.",
                            side,
                            joint,
                            value,
                            value / 10.0,
                        )
                        value /= 10.0
                    clean[field] = value
            com = entry.get("com")
            if isinstance(com, (list, tuple)) and len(com) == 3:
                com_clean = [_coerce_float(v) for v in com]
                if all(v is not None for v in com_clean):
                    clean["com"] = com_clean
                else:
                    _logger.warning(
                        "Calibration for %s %s has a non-numeric com entry; "
                        "ignoring it.",
                        side,
                        joint,
                    )
            friction = entry.get("friction")
            if isinstance(friction, dict):
                fclean = {f: _coerce_float(friction.get(f)) for f in _FRICTION_FIELDS}
                if all(v is not None for v in fclean.values()):
                    clean["friction"] = fclean
                elif any(v is not None for v in fclean.values()):
                    _logger.warning(
                        "Calibration for %s %s has a partial friction entry "
                        "(need all of fc/k/fv/fo); ignoring it.",
                        side,
                        joint,
                    )
            if clean:
                out[side][str(joint)] = clean
    return out


def load_factory_calibration() -> dict[str, dict[str, Any]]:
    """The fetched factory calibration, sanitized like the local file."""
    return load_calibration(path=FACTORY_CALIBRATION_PATH)


def save_factory_calibration(
    document: dict[str, Any], path: Path = FACTORY_CALIBRATION_PATH
) -> Path:
    """Persist a fetched factory-calibration document (atomic write)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def update_joint_calibration(
    side: str,
    joint: str,
    *,
    kp: float | None = None,
    kd: float | None = None,
    j_eff: float | None = None,
    kd_host: float | None = None,
    friction: dict[str, float] | None = None,
    com: tuple[float, float, float] | None = None,
    path: Path = CALIBRATION_PATH,
) -> Path:
    """Merge new values into one joint's calibration entry and persist.

    Only the provided fields are touched — saving PID gains does not clobber
    a previously saved friction fit, and vice versa. ``friction`` must carry
    all of ``fc`` / ``k`` / ``fv`` / ``fo``; ``com`` is the link's fitted
    centre of mass (metres, URDF link frame). The write is atomic
    (tmp + rename), matching the settings store.
    """
    if side not in _SIDES:
        raise ValueError(f"side must be one of {_SIDES}, got {side!r}")
    if friction is not None:
        missing = [f for f in _FRICTION_FIELDS if f not in friction]
        if missing:
            raise ValueError(f"friction is missing fields: {', '.join(missing)}")

    try:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raw = {}
    except FileNotFoundError:
        raw = {}
    except Exception:  # noqa: BLE001 - start fresh rather than refuse to save
        _logger.warning("Rewriting unreadable calibration file %s.", path)
        raw = {}

    raw["version"] = 1
    side_map = raw.setdefault(side, {})
    if not isinstance(side_map, dict):
        side_map = {}
        raw[side] = side_map
    entry = side_map.setdefault(joint, {})
    if not isinstance(entry, dict):
        entry = {}
        side_map[joint] = entry
    # Scrub the retired software-damping field left behind by older versions.
    entry.pop("kd_soft", None)

    for field, value in (
        ("kp", kp),
        ("kd", kd),
        ("j_eff", j_eff),
        ("kd_host", kd_host),
    ):
        if value is not None:
            entry[field] = float(value)
    if friction is not None:
        entry["friction"] = {f: float(friction[f]) for f in _FRICTION_FIELDS}
    if com is not None:
        if len(com) != 3:
            raise ValueError(f"com must have 3 components, got {len(com)}")
        entry["com"] = [float(v) for v in com]
    entry["updated_at"] = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path
