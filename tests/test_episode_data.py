"""One episode's robot data for the panel's dataset charts and 3D replay."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from almond_axol.constants import ARM_JOINTS  # noqa: E402
from almond_axol.kinematics.mujoco_fk import AxolForwardKinematics  # noqa: E402
from almond_axol.recording.dataset_browser import DatasetBrowseError  # noqa: E402
from almond_axol.recording.episode_data import read_episode_data  # noqa: E402

FPS = 10
JOINT_NAMES = [
    f"{side}_{joint}.pos"
    for side in ("left", "right")
    for joint in [j.value for j in ARM_JOINTS] + ["gripper"]
]
EE_NAMES = [
    f"{side}_{part}"
    for side in ("left", "right")
    for part in [f"ee.{a}" for a in ("x", "y", "z", "rx", "ry", "rz")] + ["gripper.pos"]
]


def _make(
    tmp_path: Path, names: list[str], episodes: dict[int, np.ndarray], **extra
) -> Path:
    """One shared data file holding ``episodes`` (index -> (frames, names))."""
    root = tmp_path / "ds"
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    feature = {"dtype": "float32", "shape": [len(names)], "names": names}
    info = {
        "fps": FPS,
        "features": {"observation.state": feature, "action": feature},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array(list(episodes), pa.int64()),
                "data/chunk_index": pa.array([0] * len(episodes), pa.int64()),
                "data/file_index": pa.array([0] * len(episodes), pa.int64()),
            }
        ),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    rows: dict[str, list] = {
        "episode_index": [],
        "frame_index": [],
        "timestamp": [],
        "observation.state": [],
        "action": [],
    }
    times = extra.get("times")
    for index, values in episodes.items():
        for frame, row in enumerate(values):
            rows["episode_index"].append(index)
            rows["frame_index"].append(frame)
            rows["timestamp"].append(times[frame] if times else frame / FPS)
            rows["observation.state"].append([float(v) for v in row])
            rows["action"].append([float(v) + 0.01 for v in row])
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array(rows["episode_index"], pa.int64()),
                "frame_index": pa.array(rows["frame_index"], pa.int64()),
                "timestamp": pa.array(rows["timestamp"], pa.float32()),
                "observation.state": pa.array(
                    rows["observation.state"], pa.list_(pa.float32())
                ),
                "action": pa.array(rows["action"], pa.list_(pa.float32())),
            }
        ),
        root / "data" / "chunk-000" / "file-000.parquet",
    )
    return root


def test_a_joint_episode_gets_its_gripper_poses_from_forward_kinematics(
    tmp_path: Path,
) -> None:
    frames = np.zeros((4, len(JOINT_NAMES)))
    frames[:, JOINT_NAMES.index("left_elbow.pos")] = [0.0, 0.1, 0.2, 0.3]
    other = np.ones((3, len(JOINT_NAMES)))
    root = _make(tmp_path, JOINT_NAMES, {0: other, 1: frames})

    data = read_episode_data(root, 1)

    assert data["frames"] == 4 and data["space"] == "joint"
    assert data["t"] == [0.0, 0.1, 0.2, 0.3]
    assert data["state"]["left_elbow.pos"] == [0.0, 0.1, 0.2, 0.3]
    assert data["action"]["left_elbow.pos"][0] == pytest.approx(0.01)
    left, right = AxolForwardKinematics().ee_poses(frames[2, :8], frames[2, 8:])
    assert data["pose"]["left"]["x"][2] == pytest.approx(float(left[0]), abs=1e-4)
    assert data["pose"]["right"]["z"][2] == pytest.approx(float(right[2]), abs=1e-4)
    assert set(data["actionPose"]) == {"left", "right"}
    jump = data["checks"]["jointJump"]
    assert jump["name"] == "left_elbow.pos"
    assert jump["deg"] == pytest.approx(math.degrees(0.1), abs=0.05)
    assert data["checks"]["droppedFrames"] == 0
    assert data["poseLagMs"] is None and data["intervention"] is None


def test_a_cartesian_episode_keeps_its_recorded_poses(tmp_path: Path) -> None:
    frames = np.zeros((3, len(EE_NAMES)))
    x = EE_NAMES.index("right_ee.x")
    frames[:, x] = [0.30, 0.31, 0.45]
    root = _make(tmp_path, EE_NAMES, {0: frames}, times=[0.0, 0.1, 0.4])

    data = read_episode_data(root, 0)

    assert data["space"] == "cartesian"
    assert data["pose"]["right"]["x"] == [0.3, 0.31, 0.45]
    assert data["checks"]["poseJump"] == {"side": "right", "m": 0.14, "t": 0.4}
    assert data["checks"]["jointJump"] is None
    # 0.1 s -> 0.4 s at 10 fps: two frames missing.
    assert data["checks"]["droppedFrames"] == 2
    assert data["checks"]["longestGapS"] == pytest.approx(0.3, abs=1e-3)


def test_an_unknown_episode_is_a_browse_error(tmp_path: Path) -> None:
    root = _make(tmp_path, JOINT_NAMES, {0: np.zeros((2, len(JOINT_NAMES)))})
    with pytest.raises(DatasetBrowseError):
        read_episode_data(root, 5)


def test_the_api_serves_an_episode_and_reports_a_missing_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    import httpx

    from tests.test_serve_session_reservation import (
        _Manager,
        _Runner,
        _Settings,
        _test_app,
    )

    _make(tmp_path, JOINT_NAMES, {0: np.zeros((3, len(JOINT_NAMES)))})
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    app = _test_app(_Manager(), _Runner(), settings=_Settings(str(tmp_path)))

    async def exercise() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            found = await client.get(
                "/api/datasets/episode-data", params={"repo_id": "ds", "episode": 0}
            )
            assert found.status_code == 200
            body = found.json()
            assert body["repoId"] == "ds" and body["frames"] == 3
            assert set(body["pose"]) == {"left", "right"}
            missing = await client.get(
                "/api/datasets/episode-data", params={"repo_id": "ds", "episode": 9}
            )
            assert missing.status_code == 404

    asyncio.run(exercise())
