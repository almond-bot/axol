"""One saved episode's robot data, for checking a dataset in the panel.

The panel's dataset view charts what each episode recorded and replays it on
the robot model, so an operator can see that a teleop or Mantis take is sound
before training on it. This reads the episode's rows from its ``data/`` file
and returns, per named feature, the measured (``observation.state``) and
commanded (``action``) values, plus each gripper mount's pose when the
dataset records it (Mantis, or ``observe_cartesian``: ``{side}_ee.*``).

This is all file reading and vectorized NumPy on purpose. The serve process
also runs a recording's control loop, and a panel open on the dataset being
recorded asks for every take as it is saved, so per-frame Python work here
would hold the interpreter lock against that loop. That is why a joint-space
episode's gripper poses (and its gripper jump check) are worked out in the
viewer's browser from the robot model (``web/app/src/lib/episode-fk.ts``).

It also reports what an operator would otherwise have to spot by eye: frames
the recorder dropped and the largest jump between two frames, which is what a
re-engage, a tracking glitch or a bad IK solution looks like in the data.

Like :mod:`.dataset_browser` this needs only ``pyarrow`` (the ``lerobot``
extra every recording host has), never ``lerobot`` or ``torch``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .dataset_browser import (
    _DEFAULT_DATA_PATH,
    DatasetBrowseError,
    _find_episode_row,
    _load_info,
)

SIDES = ("left", "right")
POSE_AXES = ("x", "y", "z", "rx", "ry", "rz")
# A frame interval this many times the nominal one counts as dropped frames.
GAP_FACTOR = 1.5


def _feature_names(info: dict[str, Any], key: str) -> list[str]:
    feature = (info.get("features") or {}).get(key) or {}
    names = feature.get("names")
    if isinstance(names, dict):
        # Older LeRobot layouts nest the list under an axis name.
        names = next(iter(names.values()), None)
    return [str(n) for n in names] if isinstance(names, list) else []


def _column(table: Any, name: str) -> np.ndarray | None:
    if name not in table.schema.names:
        return None
    import pyarrow as pa

    column = table.column(name).combine_chunks()
    if pa.types.is_list(column.type) or pa.types.is_fixed_size_list(column.type):
        # One vector per frame: flatten in Arrow, not per frame in Python.
        flat = column.flatten().to_numpy(zero_copy_only=False)
        return flat.astype(np.float64).reshape(len(column), -1)
    return column.to_numpy(zero_copy_only=False).astype(np.float64)


def _rounded(values: np.ndarray, digits: int) -> list[float | None]:
    out = np.round(values.astype(np.float64), digits)
    listed: list[float | None] = out.tolist()
    for i in np.flatnonzero(~np.isfinite(out)):
        listed[i] = None
    return listed


def _poses(names: list[str], values: np.ndarray) -> dict[str, np.ndarray]:
    """Each side's ``(frames, 6)`` gripper-mount pose, where it is recorded."""
    index = {name: i for i, name in enumerate(names)}
    poses: dict[str, np.ndarray] = {}
    for side in SIDES:
        keys = [f"{side}_ee.{axis}" for axis in POSE_AXES]
        if all(k in index for k in keys):
            poses[side] = values[:, [index[k] for k in keys]]
    return poses


def _pose_dict(poses: dict[str, np.ndarray]) -> dict[str, dict[str, list]]:
    return {
        side: {
            axis: _rounded(pose[:, i], 5 if i < 3 else 4)
            for i, axis in enumerate(POSE_AXES)
        }
        for side, pose in poses.items()
    }


def _largest_joint_jump(
    names: list[str], values: np.ndarray, t: np.ndarray
) -> dict[str, Any] | None:
    arm = [
        i
        for i, n in enumerate(names)
        if n.endswith(".pos") and not n.endswith("gripper.pos") and "_ee." not in n
    ]
    if len(values) < 2 or not arm:
        return None
    steps = np.abs(np.diff(values[:, arm], axis=0))
    if not np.isfinite(steps).any():
        return None
    frame, column = np.unravel_index(np.nanargmax(steps), steps.shape)
    return {
        "name": names[arm[column]],
        "deg": round(math.degrees(float(steps[frame, column])), 2),
        "t": round(float(t[frame + 1]), 3),
    }


def _largest_pose_jump(
    poses: dict[str, np.ndarray], t: np.ndarray
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for side, pose in poses.items():
        if len(pose) < 2:
            continue
        steps = np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1)
        if not np.isfinite(steps).any():
            continue
        frame = int(np.nanargmax(steps))
        if best is None or steps[frame] > best["m"]:
            best = {
                "side": side,
                "m": round(float(steps[frame]), 4),
                "t": round(float(t[frame + 1]), 3),
            }
    return best


def to_json(value: Any) -> str:
    """Encode :func:`read_episode_data`'s result one column at a time.

    A single ``json.dumps`` of a long episode is one C call of tens of
    milliseconds that never lets go of the interpreter lock; per column, each
    call is about a millisecond and the control loop gets its turns between
    them. (FastAPI's own encoder is slower still and runs on the event loop.)
    """
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(f"{json.dumps(str(k))}:{to_json(v)}" for k, v in value.items())
            + "}"
        )
    return json.dumps(value, separators=(",", ":"))


def read_episode_data(root: Path, episode_index: int) -> dict[str, Any]:
    """One saved episode's state, actions, gripper poses and checks."""
    import pyarrow.parquet as pq

    info = _load_info(root)
    fps = int(info.get("fps") or 0)
    meta_path, row = _find_episode_row(root, episode_index)
    record = (
        pq.read_table(meta_path, columns=["data/chunk_index", "data/file_index"])
        .slice(row, 1)
        .to_pylist()[0]
    )
    data_path = root / (info.get("data_path") or _DEFAULT_DATA_PATH).format(
        chunk_index=int(record["data/chunk_index"]),
        file_index=int(record["data/file_index"]),
    )
    try:
        available = set(pq.read_schema(data_path).names)
    except (OSError, ValueError) as exc:
        raise DatasetBrowseError(f"cannot read the episode's data: {exc}") from exc
    wanted = [
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "observation.pose_lag",
        "intervention",
    ]
    # Filter on episode_index: datasets recorded before per-episode files
    # share one data file between many episodes.
    table = pq.read_table(
        data_path,
        columns=[c for c in wanted if c in available],
        filters=[("episode_index", "=", episode_index)],
    )
    if table.num_rows == 0:
        raise DatasetBrowseError(f"episode {episode_index} has no frames")
    if "frame_index" in table.schema.names:
        table = table.sort_by("frame_index")

    timestamps = _column(table, "timestamp")
    if timestamps is None:
        timestamps = np.arange(table.num_rows) / max(fps, 1)
    t = timestamps - timestamps[0]

    groups: dict[str, tuple[list[str], np.ndarray]] = {}
    for key, out in (("observation.state", "state"), ("action", "action")):
        values = _column(table, key)
        if values is None:
            continue
        values = values.reshape(len(values), -1)
        names = _feature_names(info, key)
        if len(names) != values.shape[1]:
            names = [f"{out}_{i}" for i in range(values.shape[1])]
        groups[out] = (names, values)

    def values_dict(names: list[str], values: np.ndarray) -> dict[str, list]:
        return {
            name: _rounded(values[:, i], 5 if "_ee." in name else 4)
            for i, name in enumerate(names)
        }

    state_names, state = groups.get("state", ([], np.zeros((len(t), 0))))
    pose = _poses(state_names, state) if state_names else {}
    action_pose = _poses(*groups["action"]) if "action" in groups else {}

    gaps = np.diff(t)
    nominal = 1.0 / fps if fps else (float(np.median(gaps)) if len(gaps) else 0.0)
    dropped = gaps[gaps > nominal * GAP_FACTOR] if nominal > 0 else gaps[:0]

    pose_lag = _column(table, "observation.pose_lag")
    intervention = _column(table, "intervention")

    return {
        "episode": episode_index,
        "fps": fps,
        "frames": int(table.num_rows),
        "space": "cartesian" if any("_ee." in n for n in state_names) else "joint",
        "t": _rounded(t, 4),
        "state": values_dict(*groups["state"]) if "state" in groups else {},
        "action": values_dict(*groups["action"]) if "action" in groups else {},
        "pose": _pose_dict(pose),
        "actionPose": _pose_dict(action_pose),
        "poseLagMs": _rounded(pose_lag.reshape(-1) * 1000, 2)
        if pose_lag is not None
        else None,
        "intervention": [bool(v) for v in intervention.reshape(-1)]
        if intervention is not None
        else None,
        "checks": {
            "droppedFrames": int(
                sum(round(g / nominal) - 1 for g in dropped) if nominal > 0 else 0
            ),
            "longestGapS": round(float(gaps.max()), 3) if len(gaps) else 0.0,
            "jointJump": _largest_joint_jump(state_names, state, t),
            "poseJump": _largest_pose_jump(pose, t),
        },
    }
