"""Dataset preview: reading saved episodes and renaming an episode's task.

The fixtures write the LeRobot v3 layout directly with pyarrow / pandas (the
columns LeRobot 0.6.1 writes), so these run without importing lerobot. Two
shapes matter: stock LeRobot shares one data / meta / video file between
episodes (non-zero ``from_timestamp``), axol's recorder gives every episode
its own files (``make_episode_durable``).
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pd = pytest.importorskip("pandas")

from almond_axol.recording.dataset_browser import (  # noqa: E402
    METADATA_LOCK_FILENAME,
    DatasetBrowseError,
    DatasetBusyError,
    dataset_metadata_lock,
    episode_video,
    read_episodes,
    reload_tasks_from_disk,
    rename_episode_task,
    resolve_dataset,
)

CAM = "observation.images.cam"
FPS = 10


def _write_tasks(root: Path, tasks: list[str]) -> None:
    frame = pd.DataFrame(
        {"task_index": list(range(len(tasks)))}, index=pd.Index(tasks, name="task")
    )
    frame.to_parquet(root / "meta" / "tasks.parquet")


def _meta_table(rows: list[dict]) -> "pa.Table":
    columns: dict[str, list] = {k: [r[k] for r in rows] for k in rows[0]}
    types = {
        "episode_index": pa.int64(),
        "tasks": pa.list_(pa.string()),
        "length": pa.int64(),
        "data/chunk_index": pa.int64(),
        "data/file_index": pa.int64(),
        f"videos/{CAM}/chunk_index": pa.int64(),
        f"videos/{CAM}/file_index": pa.int64(),
        f"videos/{CAM}/from_timestamp": pa.float64(),
        f"videos/{CAM}/to_timestamp": pa.float64(),
        "stats/task_index/min": pa.list_(pa.float64()),
        "stats/task_index/max": pa.list_(pa.float64()),
        "stats/task_index/mean": pa.list_(pa.float64()),
        "stats/task_index/std": pa.list_(pa.float64()),
        "stats/task_index/count": pa.list_(pa.int64()),
    }
    return pa.table({k: pa.array(v, type=types[k]) for k, v in columns.items()})


def _make_dataset(base: Path, repo_id: str, tasks: list[str], *, shared: bool) -> Path:
    """Episode i has task ``tasks[i]`` and 5 frames."""
    root = base / repo_id
    (root / "meta").mkdir(parents=True)
    unique = list(dict.fromkeys(tasks))
    info = {
        "fps": FPS,
        "total_episodes": len(tasks),
        "total_tasks": len(unique),
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            CAM: {"dtype": "video", "shape": [4, 4, 3]},
            "observation.state": {"dtype": "float32", "shape": [2]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    _write_tasks(root, unique)

    length = 5
    rows = []
    data_files: dict[int, list[dict]] = {}
    for i, task in enumerate(tasks):
        file_index = 0 if shared else i
        start = i * length / FPS if shared else 0.0
        index = unique.index(task)
        rows.append(
            {
                "episode_index": i,
                "tasks": [task],
                "length": length,
                "data/chunk_index": 0,
                "data/file_index": file_index,
                f"videos/{CAM}/chunk_index": 0,
                f"videos/{CAM}/file_index": file_index,
                f"videos/{CAM}/from_timestamp": start,
                f"videos/{CAM}/to_timestamp": start + length / FPS,
                "stats/task_index/min": [float(index)],
                "stats/task_index/max": [float(index)],
                "stats/task_index/mean": [float(index)],
                "stats/task_index/std": [0.0],
                "stats/task_index/count": [length],
            }
        )
        data_files.setdefault(file_index, []).extend(
            {"episode_index": i, "frame_index": f, "task_index": index}
            for f in range(length)
        )

    meta_dir = root / "meta" / "episodes" / "chunk-000"
    meta_dir.mkdir(parents=True)
    if shared:
        pq.write_table(_meta_table(rows), meta_dir / "file-000.parquet")
    else:
        for i, row in enumerate(rows):
            pq.write_table(_meta_table([row]), meta_dir / f"file-{i:03d}.parquet")

    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    video_dir = root / "videos" / CAM / "chunk-000"
    video_dir.mkdir(parents=True)
    for file_index, frames in data_files.items():
        table = pa.table(
            {
                "episode_index": pa.array(
                    [f["episode_index"] for f in frames], pa.int64()
                ),
                "frame_index": pa.array([f["frame_index"] for f in frames], pa.int64()),
                "task_index": pa.array([f["task_index"] for f in frames], pa.int64()),
            }
        ).replace_schema_metadata({b"huggingface": b'{"info": "kept"}'})
        pq.write_table(table, data_dir / f"file-{file_index:03d}.parquet")
        (video_dir / f"file-{file_index:03d}.mp4").write_bytes(b"\x00mp4")
    return root


def _task_names(root: Path) -> list[str]:
    frame = pd.read_parquet(root / "meta" / "tasks.parquet")
    return [name for name, _ in sorted(frame.task_index.items(), key=lambda kv: kv[1])]


def _frame_task_indices(root: Path, file_index: int) -> list[tuple[int, int]]:
    table = pq.read_table(
        root / "data" / "chunk-000" / f"file-{file_index:03d}.parquet"
    )
    return list(
        zip(
            table.column("episode_index").to_pylist(),
            table.column("task_index").to_pylist(),
            strict=True,
        )
    )


@pytest.mark.parametrize("shared", [True, False])
def test_read_episodes_lists_spans_and_tasks(tmp_path: Path, shared: bool) -> None:
    root = _make_dataset(tmp_path, "org/ds", ["pick", "place", "pick"], shared=shared)

    listing = read_episodes(root)

    assert listing.fps == FPS
    assert listing.cameras == [CAM]
    assert listing.tasks == ["pick", "place"]
    assert [(e.index, e.tasks, e.length) for e in listing.episodes] == [
        (0, ["pick"], 5),
        (1, ["place"], 5),
        (2, ["pick"], 5),
    ]
    second = listing.episodes[1].videos[CAM]
    assert second.from_timestamp == (0.5 if shared else 0.0)
    assert second.to_timestamp == pytest.approx(second.from_timestamp + 0.5)
    path, span = episode_video(root, 1, CAM)
    assert path == root / second.path
    assert span == second


def test_a_file_being_written_is_skipped_not_fatal(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path, "ds", ["pick", "place"], shared=False)
    # A save in progress: the next meta/episodes file has no parquet footer yet.
    (root / "meta" / "episodes" / "chunk-000" / "file-002.parquet").write_bytes(
        b"PAR1\x00"
    )

    listing = read_episodes(root)

    assert [e.index for e in listing.episodes] == [0, 1]
    assert listing.unreadable_files == 1


@pytest.mark.parametrize("shared", [True, False])
def test_rename_to_a_new_task_appends_it_and_touches_only_that_episode(
    tmp_path: Path, shared: bool
) -> None:
    root = _make_dataset(tmp_path, "ds", ["pick", "place", "pick"], shared=shared)

    summary = rename_episode_task(root, 2, "  stack cups ")

    assert summary.tasks == ["stack cups"]
    # Appended, so the existing indices (the reader resolves by position) hold.
    assert _task_names(root) == ["pick", "place", "stack cups"]
    assert json.loads((root / "meta" / "info.json").read_text())["total_tasks"] == 3
    if shared:
        assert _frame_task_indices(root, 0) == (
            [(0, 0)] * 5 + [(1, 1)] * 5 + [(2, 2)] * 5
        )
    else:
        assert _frame_task_indices(root, 0) == [(0, 0)] * 5
        assert _frame_task_indices(root, 2) == [(2, 2)] * 5
    data = pq.read_table(
        root / "data" / "chunk-000" / f"file-{0 if shared else 2:03d}.parquet"
    )
    assert data.schema.metadata[b"huggingface"] == b'{"info": "kept"}'

    listing = read_episodes(root)
    assert [e.tasks for e in listing.episodes] == [["pick"], ["place"], ["stack cups"]]
    meta_file = "file-000.parquet" if shared else "file-002.parquet"
    meta = pq.read_table(
        root / "meta" / "episodes" / "chunk-000" / meta_file
    ).to_pylist()
    renamed = next(r for r in meta if r["episode_index"] == 2)
    assert renamed["stats/task_index/min"] == [2.0]
    assert renamed["stats/task_index/std"] == [0.0]
    assert renamed["stats/task_index/count"] == [5]
    # No temporary files left behind.
    assert not list(root.rglob(".*.parquet"))


def test_rename_to_an_existing_task_reuses_its_index(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path, "ds", ["pick", "place"], shared=True)

    rename_episode_task(root, 0, "place")

    assert _task_names(root) == ["pick", "place"]
    assert _frame_task_indices(root, 0) == [(0, 1)] * 5 + [(1, 1)] * 5


def test_rename_refuses_empty_tasks_and_unknown_episodes(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path, "ds", ["pick"], shared=False)

    with pytest.raises(DatasetBrowseError, match="empty"):
        rename_episode_task(root, 0, "   ")
    with pytest.raises(DatasetBrowseError, match="not found"):
        rename_episode_task(root, 7, "place")
    assert _task_names(root) == ["pick"]


def test_resolve_dataset_stays_under_the_base(tmp_path: Path) -> None:
    _make_dataset(tmp_path / "base", "org/ds", ["pick"], shared=False)
    _make_dataset(tmp_path, "outside", ["pick"], shared=False)
    base = tmp_path / "base"

    assert resolve_dataset(base, "org/ds") == (base / "org" / "ds").resolve()
    for bad in ("", "../outside", "/etc", "org/missing"):
        with pytest.raises(DatasetBrowseError):
            resolve_dataset(base, bad)


def test_rename_waits_for_a_save_holding_the_lock(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path, "ds", ["pick"], shared=False)
    # flock is per open file description, so a second open in this process
    # contends like the recorder subprocess would.
    fd = os.open(root / "meta" / METADATA_LOCK_FILENAME, os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        with pytest.raises(DatasetBusyError):
            rename_episode_task(root, 0, "place", lock_timeout=0.1)
        assert _task_names(root) == ["pick"]

        threading.Timer(0.2, os.close, args=(fd,)).start()
        fd = -1
        assert rename_episode_task(root, 0, "place", lock_timeout=5).tasks == ["place"]
    finally:
        if fd >= 0:
            os.close(fd)


def test_recorder_reloads_a_rename_before_its_next_save(tmp_path: Path) -> None:
    # What _save_episode_locked relies on: after a mid-session rename appended
    # a task, the recorder's in-memory table picks it up, so the save numbers
    # its own new tasks after it instead of reusing the index.
    root = _make_dataset(tmp_path, "ds", ["pick"], shared=False)
    meta = SimpleNamespace(tasks=pd.read_parquet(root / "meta" / "tasks.parquet"))

    rename_episode_task(root, 0, "renamed")
    with dataset_metadata_lock(root):
        reload_tasks_from_disk(meta, root)

    assert list(meta.tasks.index) == ["pick", "renamed"]
    assert list(meta.tasks.task_index) == [0, 1]


def test_episode_controls_name_their_dataset(tmp_path: Path) -> None:
    from almond_axol.cli.collect_data import _QueueCollectControl
    from almond_axol.cli.run_policy import _QueuePolicyControl

    for control in (
        _QueueCollectControl(threading.Event()),
        _QueuePolicyControl(threading.Event()),
    ):
        assert "dataset" not in control.snapshot()
        control.note_dataset("org/ds", tmp_path / "org" / "ds")
        assert control.snapshot()["dataset"] == {
            "repoId": "org/ds",
            "root": str((tmp_path / "org" / "ds").resolve()),
        }


def test_preview_api_lists_streams_and_renames(tmp_path: Path) -> None:
    import asyncio

    import httpx

    from tests.test_serve_session_reservation import (
        _Manager,
        _Runner,
        _Settings,
        _test_app,
    )

    root = _make_dataset(tmp_path, "org/ds", ["pick", "place"], shared=True)
    app = _test_app(_Manager(), _Runner(), settings=_Settings(str(tmp_path)))

    async def exercise() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            listing = await client.get(
                "/api/datasets/episodes", params={"repo_id": "org/ds"}
            )
            assert listing.status_code == 200
            body = listing.json()
            assert body["root"] == str(root.resolve())
            assert body["episodes"][1]["videos"][CAM] == {"from": 0.5, "to": 1.0}

            video = await client.get(
                "/api/datasets/video",
                params={"repo_id": "org/ds", "episode": 1, "camera": CAM},
                headers={"Range": "bytes=0-1"},
            )
            assert video.status_code == 206
            assert video.headers["content-type"] == "video/mp4"
            assert video.content == b"\x00m"

            renamed = await client.put(
                "/api/datasets/episodes/1/task",
                json={"repoId": "org/ds", "task": "stack"},
            )
            assert renamed.status_code == 200
            assert renamed.json()["episode"]["tasks"] == ["stack"]

            escaped = await client.get(
                "/api/datasets/episodes", params={"repo_id": "../x"}
            )
            assert escaped.status_code == 404
            bad = await client.put(
                "/api/datasets/episodes/1/task", json={"repoId": "org/ds", "task": " "}
            )
            assert bad.status_code == 400

    asyncio.run(exercise())
    assert _task_names(root) == ["pick", "place", "stack"]
