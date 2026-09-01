"""Safe persistence helpers for the control panel's tracker setup editors.

The files managed here are also consumed directly by CLI processes.  Writes
therefore use a process-wide lock plus a same-directory atomic rename, and
retain the operator ownership expected by a root-run ``axol serve`` service.
The Ultimate Wi-Fi password is deliberately absent from every returned value
and every validation error in this module.
"""

from __future__ import annotations

import json
import stat
import threading
from pathlib import Path
from typing import Any

from ..cli.tracker_ultimate import is_ultimate_tracker_key
from ..mantis.calibration import (
    CURRENT_TRANSFORM_ENTRY,
    LEGACY_TRACKER_KEY,
    MANTIS_TCP_TRANSFORM_FILE,
    STALE_TRANSFORM_ENTRY,
    ULTIMATE_POSE_CONVENTION_FIELD,
    candidate_transform_for,
    classify_tcp_transform_entry,
    current_ultimate_pose_convention,
    design_transform_for,
    parse_quest_tracker_key,
    ultimate_pose_convention_from_entry,
    ultimate_pose_convention_metadata,
    validate_tcp_transform,
)
from ..tracker.config import TRACKER_CONFIG_FILE, load_tracker_config
from ..tracker.ultimate import ULTIMATE_WIFI_CONFIG_FILE, ultimate_wifi_values_error
from ..utils.state_files import secure_atomic_write_json, secure_read_text


class TrackerSetupError(ValueError):
    """An operator-correctable, non-secret setup-file error."""


_SETUP_FILE_LOCK = threading.RLock()
_WIFI_REQUIRED_UPDATE_KEYS = frozenset({"ssid", "country", "freq"})
_CALIBRATION_SOURCES = frozenset({"quest", "lighthouse", "ultimate"})
_SIDES = ("left", "right")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write restrictive JSON while retaining operator ownership."""
    secure_atomic_write_json(path, payload)


def _wifi_values(payload: object) -> dict[str, Any]:
    """Validate and normalize the exact pyvut Wi-Fi document shape."""
    error = ultimate_wifi_values_error(payload)
    if error is not None:
        raise TrackerSetupError(error)
    assert isinstance(payload, dict)
    return {
        "ssid": payload["ssid"],
        "pass": payload["pass"],
        "country": payload["country"].upper(),
        "freq": payload["freq"],
    }


def _read_wifi_document(path: Path) -> tuple[object | None, str | None]:
    try:
        return json.loads(secure_read_text(path)), None
    except FileNotFoundError:
        return None, None
    except json.JSONDecodeError as exc:
        return None, (f"file is not valid JSON (line {exc.lineno}, column {exc.colno})")
    except OSError as exc:
        return None, f"file cannot be read ({exc})"


def _safe_wifi_fields(payload: object) -> dict[str, Any]:
    """Return only non-secret, individually well-typed display fields."""
    value = payload if isinstance(payload, dict) else {}
    ssid = value.get("ssid")
    country = value.get("country")
    frequency = value.get("freq")
    password = value.get("pass")
    return {
        "ssid": ssid if isinstance(ssid, str) else "",
        "country": country if isinstance(country, str) else "",
        "freq": (
            frequency
            if isinstance(frequency, int) and not isinstance(frequency, bool)
            else 0
        ),
        "passwordSet": isinstance(password, str) and bool(password),
    }


def ultimate_wifi_snapshot(path: Path | None = None) -> dict[str, Any]:
    """Describe the shared-map config without returning its password."""
    path = ULTIMATE_WIFI_CONFIG_FILE if path is None else path
    with _SETUP_FILE_LOCK:
        payload, read_error = _read_wifi_document(path)
        display = _safe_wifi_fields(payload)
        if read_error is not None:
            status = "invalid"
            error = read_error
        elif payload is None:
            status = "missing"
            error = None
        else:
            try:
                _wifi_values(payload)
                error = None
            except TrackerSetupError as exc:
                status = "invalid"
                error = str(exc)
            else:
                try:
                    mode = stat.S_IMODE(path.lstat().st_mode)
                except OSError as exc:
                    status = "invalid"
                    error = f"file metadata cannot be read ({exc})"
                else:
                    if mode == 0o600:
                        status = "valid"
                    else:
                        status = "permissions-warning"
                        error = f"file mode must be 0600 (currently {mode:04o})"
        return {
            "path": str(path),
            "configured": status == "valid",
            "status": status,
            "error": error,
            **display,
        }


def save_ultimate_wifi(update: object, path: Path | None = None) -> dict[str, Any]:
    """Validate and atomically save UI Wi-Fi input, preserving its secret."""
    path = ULTIMATE_WIFI_CONFIG_FILE if path is None else path
    if not isinstance(update, dict):
        raise TrackerSetupError("request body must be a JSON object")
    keys = set(update)
    allowed = _WIFI_REQUIRED_UPDATE_KEYS | {"pass"}
    missing = sorted(_WIFI_REQUIRED_UPDATE_KEYS - keys)
    extra = sorted(keys - allowed)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing keys: " + ", ".join(missing))
        if extra:
            details.append("unknown keys: " + ", ".join(extra))
        raise TrackerSetupError(
            "request must contain ssid, country, freq, and optional pass"
            + (f" ({'; '.join(details)})" if details else "")
        )
    if "pass" in update and (not isinstance(update["pass"], str) or not update["pass"]):
        raise TrackerSetupError("`pass` must be a non-empty string when provided")

    with _SETUP_FILE_LOCK:
        existing, _read_error = _read_wifi_document(path)
        existing_password = existing.get("pass") if isinstance(existing, dict) else None
        password = update.get("pass", existing_password)
        if not isinstance(password, str) or not password:
            raise TrackerSetupError(
                "`pass` is required on the first save; it cannot be read back later"
            )
        payload = _wifi_values(
            {
                "ssid": update.get("ssid"),
                "pass": password,
                "country": update.get("country"),
                "freq": update.get("freq"),
            }
        )
        _atomic_write_json(path, payload)
        return ultimate_wifi_snapshot(path)


def _exact_tracker_keys(
    source: str, quest_tracker_key: object = None
) -> dict[str, str | None]:
    if source not in _CALIBRATION_SOURCES:
        raise TrackerSetupError("source must be quest, lighthouse, or ultimate")
    if source == "quest":
        key = quest_tracker_key.strip() if isinstance(quest_tracker_key, str) else ""
        exact = key if parse_quest_tracker_key(key) is not None else None
        return {side: exact for side in _SIDES}

    backend = "survive" if source == "lighthouse" else "ultimate"
    config = load_tracker_config(TRACKER_CONFIG_FILE)
    binding = config.bindings.get(backend, {})
    if not isinstance(binding, dict):
        binding = {}
    resolved: dict[str, str | None] = {}
    for side in _SIDES:
        device = binding.get(side)
        if config.backend == backend:
            active = config.left if side == "left" else config.right
            device = active or device
        if not isinstance(device, str) or not device.strip():
            resolved[side] = None
        elif backend == "ultimate" and not is_ultimate_tracker_key(device):
            resolved[side] = None
        else:
            resolved[side] = f"{backend}:{device}"
    return resolved


def _read_calibration_document(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(secure_read_text(path))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise TrackerSetupError(
            "existing calibration file is not valid JSON "
            f"(line {exc.lineno}, column {exc.colno}); fix it before saving"
        ) from exc
    except OSError as exc:
        raise TrackerSetupError(
            f"existing calibration file cannot be read ({exc})"
        ) from exc
    if not isinstance(payload, dict):
        raise TrackerSetupError(
            "existing calibration file must contain a JSON object; fix it before saving"
        )
    return payload


def _saved_transform(
    document: dict[str, Any],
    side: str,
    key: str | None,
    ultimate_convention: tuple[str, str] | None,
) -> tuple[list[float] | None, str, tuple[str, str] | None]:
    if key is None:
        return None, "missing", None
    side_entries = document.get(side)
    if not isinstance(side_entries, dict):
        return None, "missing", None
    # A legacy entry is itself {pos, quat}; it is not a measured value for an
    # exact active device and must never be shown as though it were one.
    entry = side_entries.get(key)
    if not isinstance(entry, dict):
        return None, "missing", None
    values, status = classify_tcp_transform_entry(
        key,
        entry,
        ultimate_convention=ultimate_convention,
    )
    return values, status, ultimate_pose_convention_from_entry(entry)


def _pose_convention_payload(
    convention: tuple[str, str] | None,
) -> dict[str, str] | None:
    if convention is None:
        return None
    quat_order, up_axis = convention
    return {"quatOrder": quat_order, "upAxis": up_axis}


def _calibration_snapshot_locked(
    source: str,
    keys: dict[str, str | None],
    document: dict[str, Any],
    path: Path,
    ultimate_convention: tuple[str, str] | None = None,
) -> dict[str, Any]:
    sides: dict[str, dict[str, Any]] = {}
    for side in _SIDES:
        key = keys[side]
        measured, entry_status, stored_convention = _saved_transform(
            document,
            side,
            key,
            ultimate_convention,
        )
        if key is None:
            status = "unbound"
        elif measured is not None and entry_status == CURRENT_TRANSFORM_ENTRY:
            status = "measured"
        elif measured is not None and entry_status == STALE_TRANSFORM_ENTRY:
            status = "stale"
        elif design_transform_for(side, key) is not None:
            status = "factory"
        elif candidate_transform_for(side, key) is not None:
            status = "candidate"
        else:
            status = "missing"
        sides[side] = {
            "key": key,
            "status": status,
            # A valid stale entry stays editable so the operator can bench-
            # check and explicitly resave/adopt it for the active convention.
            # Factory/CAD values are never presented as measured editor data.
            "pos": measured[:3] if measured is not None else None,
            "quat": measured[3:] if measured is not None else None,
            "poseConvention": _pose_convention_payload(stored_convention),
        }
    return {
        "path": str(path),
        "source": source,
        "keys": dict(keys),
        "activePoseConvention": _pose_convention_payload(ultimate_convention),
        **sides,
    }


def calibration_snapshot(
    source: str,
    quest_tracker_key: object = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Return exact active keys and their saved measured values."""
    path = MANTIS_TCP_TRANSFORM_FILE if path is None else path
    with _SETUP_FILE_LOCK:
        keys = _exact_tracker_keys(source, quest_tracker_key)
        document = _read_calibration_document(path)
        ultimate_convention = (
            current_ultimate_pose_convention(TRACKER_CONFIG_FILE)
            if source == "ultimate"
            else None
        )
        return _calibration_snapshot_locked(
            source,
            keys,
            document,
            path,
            ultimate_convention,
        )


def _validated_calibration_entry(entry: object) -> tuple[str, dict[str, Any]]:
    if not isinstance(entry, dict) or set(entry) != {"key", "pos", "quat"}:
        raise TrackerSetupError("each side must contain exactly key, pos, and quat")
    key = entry.get("key")
    if not isinstance(key, str) or not key:
        raise TrackerSetupError("`key` must be a non-empty string")
    pos = entry.get("pos")
    quat = entry.get("quat")
    if not isinstance(pos, list) or len(pos) != 3:
        raise TrackerSetupError("`pos` must contain exactly 3 numeric values")
    if not isinstance(quat, list) or len(quat) != 4:
        raise TrackerSetupError("`quat` must contain exactly 4 numeric values")
    try:
        values = validate_tcp_transform([*pos, *quat])
    except ValueError as exc:
        raise TrackerSetupError(str(exc)) from exc
    return key, {"pos": values[:3], "quat": values[3:]}


def save_calibration(
    source: str,
    update: object,
    quest_tracker_key: object = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Merge one or both active-side measured transforms into the JSON file."""
    path = MANTIS_TCP_TRANSFORM_FILE if path is None else path
    if not isinstance(update, dict):
        raise TrackerSetupError("request body must be a JSON object")
    requested = set(update)
    if not requested or not requested <= set(_SIDES):
        extra = sorted(requested - set(_SIDES))
        detail = f"; unknown keys: {', '.join(extra)}" if extra else ""
        raise TrackerSetupError(
            "request must contain at least one of left or right" + detail
        )
    validated = {
        side: _validated_calibration_entry(update[side])
        for side in _SIDES
        if side in update
    }

    with _SETUP_FILE_LOCK:
        keys = _exact_tracker_keys(source, quest_tracker_key)
        ultimate_convention = (
            current_ultimate_pose_convention(TRACKER_CONFIG_FILE)
            if source == "ultimate"
            else None
        )
        if source == "ultimate" and ultimate_convention is None:
            raise TrackerSetupError(
                "Ultimate pose convention is invalid; set ultimate_quat_order "
                "to xyzw or wxyz and ultimate_up_axis to y or z before saving"
            )
        missing = [side for side in validated if keys[side] is None]
        if missing:
            if source == "quest":
                action = (
                    "configure a valid mantis.quest_tracker_key "
                    "(quest:<profile>:<pose-space>) first"
                )
            else:
                action = f"identify the {source} tracker for that side first"
            raise TrackerSetupError(
                "no exact active calibration key for "
                + ", ".join(missing)
                + "; "
                + action
            )
        stale = [
            side
            for side, (submitted_key, _entry) in validated.items()
            if submitted_key != keys[side]
        ]
        if stale:
            raise TrackerSetupError(
                "tracker identity changed for "
                + ", ".join(stale)
                + "; refresh the setup status and retry the measurement"
            )

        document = _read_calibration_document(path)
        for side, (_submitted_key, entry) in validated.items():
            if source == "ultimate":
                assert ultimate_convention is not None
                entry[ULTIMATE_POSE_CONVENTION_FIELD] = (
                    ultimate_pose_convention_metadata(ultimate_convention)
                )
            existing = document.get(side)
            if existing is None:
                side_entries: dict[str, Any] = {}
            elif not isinstance(existing, dict):
                raise TrackerSetupError(
                    f"existing `{side}` calibration section is not an object; "
                    "fix it before saving"
                )
            elif "pos" in existing or "quat" in existing:
                # Losslessly retain the old pre-device-keyed calibration while
                # upgrading this side to the keyed document shape.
                if not ("pos" in existing and "quat" in existing):
                    raise TrackerSetupError(
                        f"existing `{side}` calibration section is malformed; "
                        "fix it before saving"
                    )
                side_entries = {LEGACY_TRACKER_KEY: dict(existing)}
            else:
                side_entries = dict(existing)
            key = keys[side]
            assert key is not None
            side_entries[key] = entry
            document[side] = side_entries

        _atomic_write_json(path, document)
        return _calibration_snapshot_locked(
            source,
            keys,
            document,
            path,
            ultimate_convention,
        )


__all__ = [
    "TrackerSetupError",
    "calibration_snapshot",
    "save_calibration",
    "save_ultimate_wifi",
    "ultimate_wifi_snapshot",
]
