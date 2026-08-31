"""Safe persistence helpers for the control panel's tracker setup editors.

The files managed here are also consumed directly by CLI processes.  Writes
therefore use a process-wide lock plus a same-directory atomic rename, and
retain the operator ownership expected by a root-run ``axol serve`` service.
The Ultimate Wi-Fi password is deliberately absent from every returned value
and every validation error in this module.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import threading
from pathlib import Path
from typing import Any

from ..cli.tracker_ultimate import is_ultimate_tracker_key
from ..mantis.calibration import (
    LEGACY_TRACKER_KEY,
    MANTIS_TCP_TRANSFORM_FILE,
    candidate_transform_for,
    design_transform_for,
    parse_quest_tracker_key,
    validate_tcp_transform,
)
from ..tracker.config import TRACKER_CONFIG_FILE, load_tracker_config
from ..tracker.ultimate import ULTIMATE_WIFI_CONFIG_FILE
from ..utils.paths import adopt_state_file


class TrackerSetupError(ValueError):
    """An operator-correctable, non-secret setup-file error."""


_SETUP_FILE_LOCK = threading.RLock()
_WIFI_KEYS = frozenset({"ssid", "pass", "country", "freq"})
_WIFI_REQUIRED_UPDATE_KEYS = frozenset({"ssid", "country", "freq"})
_CALIBRATION_SOURCES = frozenset({"quest", "lighthouse", "ultimate"})
_SIDES = ("left", "right")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write restrictive JSON while retaining operator ownership."""
    old_owner: tuple[int, int] | None = None
    try:
        before = path.stat()
        old_owner = (before.st_uid, before.st_gid)
    except FileNotFoundError:
        pass

    path.parent.mkdir(parents=True, exist_ok=True)
    # A root service may have created this source-specific directory.  Give it
    # to the owner of ALMOND_HOME just like the finished state file below.
    adopt_state_file(path.parent)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        tmp.chmod(0o600)
        os.replace(tmp, path)
        path.chmod(0o600)
        if os.geteuid() == 0 and old_owner is not None and old_owner[0] != 0:
            # Atomic replace creates a new inode.  Preserve an existing
            # operator-owned file even if ALMOND_HOME itself happens to be
            # root-owned (for example after a service migration).
            os.chown(path, *old_owner)
        else:
            adopt_state_file(path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _wifi_values(payload: object) -> dict[str, Any]:
    """Validate and normalize the exact pyvut Wi-Fi document shape."""
    if not isinstance(payload, dict):
        raise TrackerSetupError("Wi-Fi config must be a JSON object")
    keys = set(payload)
    if keys != _WIFI_KEYS:
        missing = sorted(_WIFI_KEYS - keys)
        extra = sorted(keys - _WIFI_KEYS)
        details: list[str] = []
        if missing:
            details.append("missing keys: " + ", ".join(missing))
        if extra:
            details.append("unknown keys: " + ", ".join(extra))
        raise TrackerSetupError(
            "Wi-Fi config must contain exactly ssid, pass, country, and freq"
            + (f" ({'; '.join(details)})" if details else "")
        )

    ssid = payload.get("ssid")
    password = payload.get("pass")
    country = payload.get("country")
    frequency = payload.get("freq")
    if not isinstance(ssid, str) or not ssid.strip():
        raise TrackerSetupError("`ssid` must be a non-empty string")
    if not isinstance(password, str) or not password:
        raise TrackerSetupError("`pass` must be a non-empty string")
    if not isinstance(country, str) or len(country) != 2 or not country.isalpha():
        raise TrackerSetupError("`country` must be a two-letter string")
    if not isinstance(frequency, int) or isinstance(frequency, bool) or frequency <= 0:
        raise TrackerSetupError("`freq` must be a positive integer")
    return {
        "ssid": ssid,
        "pass": password,
        "country": country.upper(),
        "freq": frequency,
    }


def _read_wifi_document(path: Path) -> tuple[object | None, str | None]:
    try:
        return json.loads(path.read_text()), None
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
                    mode = stat.S_IMODE(path.stat().st_mode)
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
        payload = json.loads(path.read_text())
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
    document: dict[str, Any], side: str, key: str | None
) -> list[float] | None:
    if key is None:
        return None
    side_entries = document.get(side)
    if not isinstance(side_entries, dict):
        return None
    # A legacy entry is itself {pos, quat}; it is not a measured value for an
    # exact active device and must never be shown as though it were one.
    entry = side_entries.get(key)
    if not isinstance(entry, dict):
        return None
    pos = entry.get("pos")
    quat = entry.get("quat")
    if not isinstance(pos, list) or not isinstance(quat, list):
        return None
    try:
        return validate_tcp_transform([*pos, *quat])
    except ValueError:
        return None


def _calibration_snapshot_locked(
    source: str,
    keys: dict[str, str | None],
    document: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    sides: dict[str, dict[str, Any]] = {}
    for side in _SIDES:
        key = keys[side]
        measured = _saved_transform(document, side, key)
        if key is None:
            status = "unbound"
        elif measured is not None:
            status = "measured"
        elif design_transform_for(side, key) is not None:
            status = "factory"
        elif candidate_transform_for(side, key) is not None:
            status = "candidate"
        else:
            status = "missing"
        sides[side] = {
            "key": key,
            "status": status,
            # Only an entry actually stored under this exact key is editable
            # current data.  Factory/CAD values are intentionally not
            # presented as if the operator had measured them.
            "pos": measured[:3] if measured is not None else None,
            "quat": measured[3:] if measured is not None else None,
        }
    return {
        "path": str(path),
        "source": source,
        "keys": dict(keys),
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
        return _calibration_snapshot_locked(source, keys, document, path)


def _validated_calibration_entry(
    entry: object,
) -> tuple[str, dict[str, list[float]]]:
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
        return _calibration_snapshot_locked(source, keys, document, path)


__all__ = [
    "TrackerSetupError",
    "calibration_snapshot",
    "save_calibration",
    "save_ultimate_wifi",
    "ultimate_wifi_snapshot",
]
