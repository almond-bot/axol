"""Discovery of LeRobot datasets on disk.

Shared by the serve backend (the ``/api/datasets`` listing behind the replay
panel's dataset picker) and the replay CLI (its "dataset not found" error
lists what *is* available). Deliberately avoids importing ``lerobot`` — the
serve process answers the listing on API latency, and a lerobot import costs
seconds — so the default datasets root is resolved the same way
``lerobot.utils.constants.HF_LEROBOT_HOME`` does.
"""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# LeRobot adds these bookkeeping columns to every dataset independently of the
# caller's hardware feature contract. Keep their fixed v3 contracts here so
# they do not count as caller extras, while still rejecting corrupt metadata
# before hardware starts (LeRobot's writer populates all five on every row).
_LEROBOT_DEFAULT_FEATURES: dict[str, dict[str, Any]] = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}

# Existing features a recording flow may explicitly opt into carrying. Their
# exact contracts are still checked: accepting an arbitrary extra column would
# make LeRobot reject every new row because the recorder cannot populate it.
# An allowlisted feature may also be absent from an older compatible dataset;
# the current writer then omits it instead of changing LeRobot's fixed schema.
RESUME_FILLABLE_FEATURES: dict[str, dict[str, Any]] = {
    "observation.pose_lag": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["pose_lag"],
    },
    "intervention": {
        "dtype": "bool",
        "shape": (1,),
        "names": None,
    },
}


def dataset_features_for_robot(
    robot: Any,
    *,
    image_shape: tuple[int, int, int] | None = None,
    extra_features: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the dataset feature contract for an unconnected robot wrapper.

    ``image_shape`` overrides every recorded image shape. Collection through
    the video relay captures at one resolution for the headset but downscales
    the dataset branch to a shared explicit resolution, so its pre-hardware
    resume check must describe the latter rather than the camera's stream size.
    """
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.feature_utils import hw_to_dataset_features

    action = hw_to_dataset_features(robot.action_features, ACTION)
    observation = hw_to_dataset_features(robot.observation_features, OBS_STR)
    features: dict[str, dict[str, Any]] = {**action, **observation}

    if image_shape is not None:
        if (
            len(image_shape) != 3
            or any(isinstance(value, bool) or int(value) <= 0 for value in image_shape)
            or image_shape[2] not in (1, 3)
        ):
            raise ValueError(f"invalid dataset image shape: {image_shape!r}")
        normalized_shape = tuple(int(value) for value in image_shape)
        for key, spec in features.items():
            if key.startswith("observation.images."):
                spec["shape"] = normalized_shape
                info = dict(spec.get("info") or {})
                info["is_depth_map"] = normalized_shape[2] == 1
                spec["info"] = info
    else:
        incomplete = [
            key.removeprefix("observation.images.")
            for key, spec in features.items()
            if key.startswith("observation.images.")
            and any(value is None for value in spec.get("shape", ()))
        ]
        if incomplete:
            raise ValueError(
                "camera feature dimensions are unknown before connection for: "
                + ", ".join(sorted(incomplete))
            )

    if extra_features:
        overlap = features.keys() & extra_features.keys()
        if overlap:
            raise ValueError(
                "duplicate dataset feature(s): " + ", ".join(sorted(overlap))
            )
        features.update(deepcopy(extra_features))
    return features


def _feature_contract(key: str, value: object) -> tuple[object, ...]:
    """Normalize the schema fields that determine row compatibility."""
    if not isinstance(value, dict):
        raise ValueError(f"feature {key!r} is not an object")
    dtype = value.get("dtype")
    shape = value.get("shape")
    names = value.get("names")
    if not isinstance(dtype, str) or not isinstance(shape, (list, tuple)):
        raise ValueError(f"feature {key!r} has an invalid dtype or shape")
    if not shape or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in shape
    ):
        raise ValueError(f"feature {key!r} has an invalid shape")
    if names is not None and (
        not isinstance(names, (list, tuple))
        or any(not isinstance(name, str) for name in names)
    ):
        raise ValueError(f"feature {key!r} has invalid names")

    depth_map: bool | None = None
    if key.startswith("observation.images."):
        info = value.get("info")
        if not isinstance(info, dict) or not isinstance(info.get("is_depth_map"), bool):
            raise ValueError(f"image feature {key!r} has invalid depth metadata")
        depth_map = info["is_depth_map"]
    return (
        dtype,
        tuple(shape),
        None if names is None else tuple(names),
        depth_map,
    )


def require_dataset_resume_schema(
    dataset_root: Path,
    expected_features: dict[str, dict[str, Any]],
    *,
    fps: int,
    allowed_extra_features: frozenset[str] = frozenset(),
) -> None:
    """Fail before hardware starts unless an existing dataset is append-safe.

    LeRobot's ``resume`` keeps the schema in ``meta/info.json`` and ignores the
    fresh feature mapping supplied by the caller. Exact ordered state/action
    names, camera keys/shapes, and frame rate therefore form a safety contract;
    width alone is insufficient because Cartesian and gripperless joint layouts
    can both contain fourteen values with entirely different meanings. Known
    ``allowed_extra_features`` are symmetric compatibility columns: the writer
    can populate them when the stored schema has them and omit them when an
    older compatible dataset does not.
    """
    unknown_allowlist = allowed_extra_features - RESUME_FILLABLE_FEATURES.keys()
    if unknown_allowlist:
        raise ValueError(
            "unknown fillable resume feature(s): "
            + ", ".join(sorted(unknown_allowlist))
        )

    info_path = dataset_root / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Cannot resume the dataset at {dataset_root}: meta/info.json "
            "could not be read. Repair the dataset or start a new one with a "
            "different repo_id."
        ) from exc
    if not isinstance(info, dict) or not isinstance(info.get("features"), dict):
        raise ValueError(
            f"Cannot resume the dataset at {dataset_root}: meta/info.json has "
            "no valid feature schema."
        )

    stored_fps = info.get("fps")
    valid_fps = (
        isinstance(stored_fps, int)
        and not isinstance(stored_fps, bool)
        and stored_fps == fps
    ) or (
        isinstance(stored_fps, float)
        and math.isfinite(stored_fps)
        and stored_fps.is_integer()
        and int(stored_fps) == fps
    )
    if not valid_fps:
        raise ValueError(
            f"Cannot resume the dataset at {dataset_root}: it records at "
            f"{stored_fps!r} fps but this run is configured for {fps} fps. "
            "Use the dataset's exact rate or start a new dataset."
        )

    stored: dict[str, object] = info["features"]
    default_keys = set(_LEROBOT_DEFAULT_FEATURES)
    stored_keys = set(stored) - default_keys
    expected_keys = set(expected_features)
    missing = expected_keys - stored_keys
    optional_missing = missing & allowed_extra_features
    missing -= allowed_extra_features
    extras = stored_keys - expected_keys
    unsupported_extras = extras - allowed_extra_features
    problems: list[str] = []
    missing_defaults = default_keys - set(stored)
    if missing_defaults:
        problems.append(
            "missing LeRobot default " + ", ".join(sorted(missing_defaults))
        )
    if missing:
        problems.append("missing " + ", ".join(sorted(missing)))
    if unsupported_extras:
        problems.append("unexpected " + ", ".join(sorted(unsupported_extras)))

    for key in sorted(default_keys & set(stored)):
        try:
            actual_contract = _feature_contract(key, stored[key])
            default_contract = _feature_contract(key, _LEROBOT_DEFAULT_FEATURES[key])
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if actual_contract != default_contract:
            problems.append(f"mismatched LeRobot default {key}")

    for key in sorted(expected_keys & stored_keys):
        try:
            actual_contract = _feature_contract(key, stored[key])
            expected_contract = _feature_contract(key, expected_features[key])
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if actual_contract != expected_contract:
            problems.append(f"mismatched {key}")

    for key in sorted(extras & allowed_extra_features):
        try:
            actual_contract = _feature_contract(key, stored[key])
            allowed_contract = _feature_contract(key, RESUME_FILLABLE_FEATURES[key])
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if actual_contract != allowed_contract:
            problems.append(f"mismatched optional {key}")

    for key in sorted(optional_missing):
        try:
            expected_contract = _feature_contract(key, expected_features[key])
            allowed_contract = _feature_contract(key, RESUME_FILLABLE_FEATURES[key])
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if expected_contract != allowed_contract:
            problems.append(f"mismatched optional {key}")

    if problems:
        raise ValueError(
            f"Cannot resume the dataset at {dataset_root}: its feature schema "
            "does not match this recording run ("
            + "; ".join(problems)
            + "). Start a new dataset with a different repo_id, or restore the "
            "exact robot mode, gripper capability, cameras, and resolution used "
            "to create it."
        )


def lerobot_home() -> Path:
    """The default datasets root, matching lerobot's ``HF_LEROBOT_HOME``."""
    hf_home = os.getenv("HF_HOME") or "~/.cache/huggingface"
    default = Path(hf_home).expanduser() / "lerobot"
    return Path(os.getenv("HF_LEROBOT_HOME", default)).expanduser()


def is_dataset_dir(path: Path) -> bool:
    """True when ``path`` is a LeRobot dataset directory (has meta/info.json)."""
    return (path / "meta" / "info.json").is_file()


@dataclass
class DatasetInfo:
    """One dataset found on disk.

    ``repo_id`` is the path relative to the scanned root (the value
    collect-data recorded it under and replay-dataset addresses it by);
    ``episodes`` / ``fps`` come from ``meta/info.json`` when readable.
    """

    repo_id: str
    root: str
    episodes: int | None
    fps: int | None
    mtime: float


def list_datasets(base: Path | None = None, max_depth: int = 2) -> list[DatasetInfo]:
    """Datasets under ``base`` (default: the lerobot home), newest first.

    Scans at most ``max_depth`` directory levels — repo ids are
    ``name`` or ``org/name``, so two levels cover everything collect-data
    writes — and never descends into a dataset (episode trees are large).
    """
    root = base if base is not None else lerobot_home()
    if not root.is_dir():
        return []

    found: list[DatasetInfo] = []

    def _scan(directory: Path, depth: int) -> None:
        try:
            children = sorted(p for p in directory.iterdir() if p.is_dir())
        except OSError:
            return
        for child in children:
            if child.name.startswith(".") or child.name == "hub":
                continue
            if is_dataset_dir(child):
                episodes: int | None = None
                fps: int | None = None
                info_path = child / "meta" / "info.json"
                try:
                    info = json.loads(info_path.read_text())
                    episodes = int(info.get("total_episodes"))
                    fps = int(info.get("fps"))
                except (OSError, ValueError, TypeError):
                    pass
                try:
                    mtime = info_path.stat().st_mtime
                except OSError:
                    mtime = 0.0
                found.append(
                    DatasetInfo(
                        repo_id=str(child.relative_to(root)),
                        root=str(child),
                        episodes=episodes,
                        fps=fps,
                        mtime=mtime,
                    )
                )
            elif depth < max_depth:
                _scan(child, depth + 1)

    _scan(root, 1)
    found.sort(key=lambda d: d.mtime, reverse=True)
    return found
