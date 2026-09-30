"""Read a LeRobot v3 dataset's episodes for the panel, and rename a task.

The serve backend's dataset preview (``/api/datasets/episodes`` and friends)
reads saved episodes straight off disk, including those of a dataset a
recording session still has open. The recorder closes both parquet writers
and every episode mp4 after each save (``record_proc.make_episode_durable``),
so a saved episode's files are complete and never written again, while the
take in progress lives in the encoder's staging directory and is not listed
yet. A ``meta/episodes`` file mid-write has no parquet footer; it is skipped
and counted, not fatal.

Like :mod:`.datasets`, nothing here imports ``lerobot``: the listing runs on
API latency. It needs ``pyarrow`` (and ``pandas`` for ``meta/tasks.parquet``,
which LeRobot reads and writes as a DataFrame indexed by task string), both of
which come with the ``lerobot`` extra every recording host has.

Renaming touches the four places LeRobot keeps an episode's task:
``meta/tasks.parquet`` (string -> ``task_index``; a new string is appended so
existing indices never move — the reader resolves ``task_index`` by position),
the episode's rows' ``task_index`` in its ``data/`` file, its ``meta/episodes``
row (``tasks`` and ``stats/task_index/*``), and ``info.json``'s
``total_tasks``. The old string stays in the task table even when nothing uses
it any more: pruning would renumber every task. Files are rewritten to a
temporary name and renamed into place.

A recording session rewrites ``meta/tasks.parquet`` and ``info.json`` from its
in-memory copies on every save, so a rename and a save are serialized by
:func:`dataset_metadata_lock`: the recorder holds it across ``save_episode``
and reloads the task table from disk first, which is what makes an edit made
mid-session survive the next save.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

# In ``meta/``. Advisory (fcntl.flock); only the recorder and the rename take it.
METADATA_LOCK_FILENAME = ".metadata.lock"

MAX_TASK_LENGTH = 1000

_DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_DEFAULT_VIDEO_PATH = (
    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
)


class DatasetBrowseError(Exception):
    """A request the dataset cannot answer (bad id, missing episode, …)."""


class PreviewError(DatasetBrowseError):
    """Cutting a preview rendition failed (the original file still plays)."""


class DatasetBusyError(DatasetBrowseError):
    """The metadata lock stayed held (a save in progress) past the timeout."""


@contextlib.contextmanager
def dataset_metadata_lock(
    dataset_root: Path | str, *, timeout: float | None = None
) -> Iterator[None]:
    """Hold the dataset's metadata lock (exclusive, cross-process).

    ``timeout=None`` blocks; otherwise :class:`DatasetBusyError` is raised
    once it has been held elsewhere for ``timeout`` seconds. Without a
    ``meta/`` directory there is no dataset to contend over yet, so nothing
    is locked (and nothing is created).
    """
    meta = Path(dataset_root) / "meta"
    if not meta.is_dir():
        yield
        return
    fd = os.open(meta / METADATA_LOCK_FILENAME, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if timeout is None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        else:
            deadline = time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise DatasetBusyError(
                            "the dataset is busy saving an episode; try again"
                        ) from None
                    time.sleep(0.05)
        yield
    finally:
        os.close(fd)  # closing the descriptor releases the lock


def reload_tasks_from_disk(meta: Any, dataset_root: Path | str) -> None:
    """Refresh a ``LeRobotDatasetMetadata``'s task table from ``tasks.parquet``.

    The recorder calls this under :func:`dataset_metadata_lock` before each
    save, so a task a rename appended since the last save keeps its index and
    the save's own new tasks are numbered after it. Best-effort: an unreadable
    file keeps the in-memory table (the save must not be lost over it).
    """
    path = Path(dataset_root) / "meta" / "tasks.parquet"
    if not path.is_file():
        return
    try:
        import pandas as pd

        meta.tasks = pd.read_parquet(path)
    except Exception as exc:
        _logger.warning(
            "could not reload %s before the save (%s); keeping the recorder's "
            "own task table",
            path,
            exc,
        )


def resolve_dataset(base: Path, repo_id: str) -> Path:
    """The dataset directory ``repo_id`` names under ``base``.

    ``repo_id`` is a path relative to ``base`` (what ``/api/datasets`` lists);
    anything escaping ``base`` or not holding ``meta/info.json`` is refused.
    """
    parts = Path(repo_id).parts
    if not repo_id or Path(repo_id).is_absolute() or ".." in parts:
        raise DatasetBrowseError(f"invalid dataset id: {repo_id!r}")
    base = base.resolve()
    root = (base / repo_id).resolve()
    if not root.is_relative_to(base) or not (root / "meta" / "info.json").is_file():
        raise DatasetBrowseError(f"dataset not found: {repo_id}")
    return root


@dataclass
class EpisodeVideo:
    """Where one camera's frames for an episode live inside its mp4."""

    path: str  # relative to the dataset root
    from_timestamp: float
    to_timestamp: float


@dataclass
class EpisodeSummary:
    index: int
    length: int
    duration_s: float
    tasks: list[str]
    videos: dict[str, EpisodeVideo] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "length": self.length,
            "durationS": self.duration_s,
            "tasks": self.tasks,
            "videos": {
                key: {"from": v.from_timestamp, "to": v.to_timestamp}
                for key, v in self.videos.items()
            },
        }


@dataclass
class DatasetEpisodes:
    fps: int
    cameras: list[str]
    tasks: list[str]
    episodes: list[EpisodeSummary]
    # meta/episodes files that could not be read (a save in progress).
    unreadable_files: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "cameras": self.cameras,
            "tasks": self.tasks,
            "episodes": [e.to_dict() for e in self.episodes],
            "unreadableFiles": self.unreadable_files,
        }


def _load_info(root: Path) -> dict[str, Any]:
    try:
        return json.loads((root / "meta" / "info.json").read_text())
    except (OSError, ValueError) as exc:
        raise DatasetBrowseError(f"cannot read meta/info.json: {exc}") from exc


def _camera_keys(info: dict[str, Any]) -> list[str]:
    return [
        key
        for key, ft in (info.get("features") or {}).items()
        if isinstance(ft, dict) and ft.get("dtype") == "video"
    ]


def _episode_files(root: Path) -> list[Path]:
    return sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))


def _read_task_names(root: Path) -> list[str]:
    """Task strings in ``task_index`` order (position == index)."""
    import pyarrow.parquet as pq

    path = root / "meta" / "tasks.parquet"
    if not path.is_file():
        return []
    table = pq.read_table(path)
    names = table.column("task").to_pylist()
    indices = table.column("task_index").to_pylist()
    return [name for _, name in sorted(zip(indices, names, strict=True))]


def _summary(
    row: dict[str, Any], fps: int, cameras: list[str], info: dict
) -> EpisodeSummary:
    length = int(row.get("length") or 0)
    videos: dict[str, EpisodeVideo] = {}
    template = info.get("video_path") or _DEFAULT_VIDEO_PATH
    for key in cameras:
        chunk = row.get(f"videos/{key}/chunk_index")
        file = row.get(f"videos/{key}/file_index")
        if chunk is None or file is None:
            continue
        start = float(row.get(f"videos/{key}/from_timestamp") or 0.0)
        end = row.get(f"videos/{key}/to_timestamp")
        videos[key] = EpisodeVideo(
            path=template.format(
                video_key=key, chunk_index=int(chunk), file_index=int(file)
            ),
            from_timestamp=start,
            to_timestamp=float(end) if end is not None else start + length / fps,
        )
    tasks = row.get("tasks")
    if isinstance(tasks, str):
        tasks = [tasks]
    return EpisodeSummary(
        index=int(row["episode_index"]),
        length=length,
        duration_s=length / fps if fps else 0.0,
        tasks=[str(t) for t in (tasks or [])],
        videos=videos,
    )


# Decoded ``meta/episodes`` rows by file, reused while the file is unchanged.
# The listing runs inside the serve process — under ``axol serve`` the control
# loop is a thread of it — after every save, and once more per camera for each
# preview request (:func:`episode_video`); re-reading every file each time cost
# ~25 ms per episode on an Orin NX (3.8 s for a 150-episode dataset, five times
# over per saved episode the panel previews). Recorded files are written once,
# and a task rename replaces its file (new mtime/inode), so a stat is enough to
# tell a cached file is still current.
_episode_rows: dict[Path, tuple[tuple[int, int, int, frozenset[str]], list]] = {}
_episode_rows_guard = threading.Lock()


def _read_episode_rows(path: Path, wanted: set[str]) -> list[dict[str, Any]]:
    """``path``'s rows restricted to ``wanted`` columns; cached per file version."""
    import pyarrow.parquet as pq

    st = path.stat()
    key = (st.st_mtime_ns, st.st_size, st.st_ino, frozenset(wanted))
    with _episode_rows_guard:
        hit = _episode_rows.get(path)
    if hit is not None and hit[0] == key:
        return hit[1]
    schema = pq.read_schema(path)
    rows = pq.read_table(
        path, columns=[n for n in schema.names if n in wanted]
    ).to_pylist()
    with _episode_rows_guard:
        _episode_rows[path] = (key, rows)
    return rows


def read_episodes(root: Path) -> DatasetEpisodes:
    """Every readable saved episode of the dataset at ``root``, by index."""
    info = _load_info(root)
    fps = int(info.get("fps") or 0)
    cameras = _camera_keys(info)
    wanted = {"episode_index", "tasks", "length"} | {
        f"videos/{key}/{part}"
        for key in cameras
        for part in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
    }
    episodes: list[EpisodeSummary] = []
    unreadable = 0
    for path in _episode_files(root):
        try:
            rows = _read_episode_rows(path, wanted)
        except Exception:  # footerless (being written) or torn — skip, count
            unreadable += 1
            continue
        for row in rows:
            episodes.append(_summary(row, fps, cameras, info))
    episodes.sort(key=lambda e: e.index)
    try:
        tasks = _read_task_names(root)
    except Exception:
        tasks = []
    return DatasetEpisodes(
        fps=fps,
        cameras=cameras,
        tasks=tasks,
        episodes=episodes,
        unreadable_files=unreadable,
    )


def episode_video(
    root: Path, episode_index: int, camera: str
) -> tuple[Path, EpisodeVideo]:
    """The mp4 holding ``camera``'s frames for an episode, and their span."""
    for episode in read_episodes(root).episodes:
        if episode.index == episode_index:
            video = episode.videos.get(camera)
            if video is None:
                break
            path = (root / video.path).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise DatasetBrowseError(f"video file missing: {video.path}")
            return path, video
    raise DatasetBrowseError(f"episode {episode_index} has no {camera!r} video")


# Lighter preview renditions (see ``preview_remux``), cached across requests.
PREVIEW_CACHE_BYTES = 2 * 1024**3
_PREVIEW_TIMEOUT_S = 300.0
_preview_locks: dict[str, threading.Lock] = {}
_preview_locks_guard = threading.Lock()


def preview_step(dataset_fps: int, preview_fps: int) -> int:
    """Keep every Nth frame for ``preview_fps`` (1 = every frame).

    The panel computes the same rounding for its quality choices.
    """
    if preview_fps <= 0 or dataset_fps <= 0:
        return 1
    return max(1, round(dataset_fps / preview_fps))


def preview_cache_dir() -> Path:
    cache = os.environ.get("XDG_CACHE_HOME") or "~/.cache"
    return Path(cache).expanduser() / "axol" / "dataset-preview"


def _prune_preview_cache(cache: Path, keep: Path, budget: int) -> None:
    """Drop the least recently served previews once the cache exceeds ``budget``."""
    entries = []
    for path in cache.glob("*.mp4"):
        with contextlib.suppress(OSError):
            st = path.stat()
            entries.append((st.st_mtime, st.st_size, path))
    total = sum(size for _, size, _ in entries)
    for _, size, path in sorted(entries):
        if total <= budget:
            break
        if path == keep:
            continue
        with contextlib.suppress(OSError):
            path.unlink()
            total -= size


def episode_preview(
    root: Path,
    episode_index: int,
    camera: str,
    preview_fps: int,
    *,
    cache_dir: Path | None = None,
) -> Path:
    """A cached, quick-starting mp4 of one camera's episode.

    The file holds only the episode's span, starting at 0, with its index
    first, keeping every ``preview_step``-th frame (every frame when the
    dataset records at or below ``preview_fps``, or when the source has
    predicted frames and cannot be decimated by copying). Cut in a
    low-priority child process (``preview_remux``), keyed on the source's
    path, size and mtime, so a re-encoded or replaced file is cut again.
    """
    info = _load_info(root)
    step = preview_step(int(info.get("fps") or 0), preview_fps)
    src, video = episode_video(root, episode_index, camera)
    st = src.stat()
    key = hashlib.sha256(
        f"{src}|{st.st_size}|{st.st_mtime_ns}|{video.from_timestamp!r}|"
        f"{video.to_timestamp!r}|{step}".encode()
    ).hexdigest()[:32]
    cache = cache_dir if cache_dir is not None else preview_cache_dir()
    target = cache / f"{key}.mp4"
    with _preview_locks_guard:
        lock = _preview_locks.setdefault(key, threading.Lock())
    with lock:
        if target.is_file():
            with contextlib.suppress(OSError):
                os.utime(target)  # most recently served
            return target
        cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, tmp = tempfile.mkstemp(dir=cache, prefix=".preview.", suffix=".mp4")
        os.close(fd)
        try:
            done = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "almond_axol.recording.preview_remux",
                    str(src),
                    tmp,
                    repr(video.from_timestamp),
                    repr(video.to_timestamp),
                    str(step),
                ],
                capture_output=True,
                text=True,
                timeout=_PREVIEW_TIMEOUT_S,
            )
            if done.returncode != 0:
                detail = (done.stderr.strip().splitlines() or ["no output"])[-1]
                raise PreviewError(f"could not cut the preview: {detail}")
            os.replace(tmp, target)
        except subprocess.TimeoutExpired as exc:
            raise PreviewError("cutting the preview timed out") from exc
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
    _prune_preview_cache(cache, target, PREVIEW_CACHE_BYTES)
    return target


def _find_episode_row(root: Path, episode_index: int) -> tuple[Path, int]:
    """The ``meta/episodes`` file holding an episode and the row within it."""
    import pyarrow.parquet as pq

    for path in _episode_files(root):
        try:
            indices = (
                pq.read_table(path, columns=["episode_index"]).column(0).to_pylist()
            )
        except Exception:
            continue
        if episode_index in indices:
            return path, indices.index(episode_index)
    raise DatasetBrowseError(f"episode {episode_index} not found (not saved yet?)")


def _write_atomic_parquet(table: Any, path: Path) -> None:
    import pyarrow.parquet as pq

    fd, tmp = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".parquet"
    )
    os.close(fd)
    try:
        pq.write_table(table, tmp, compression="snappy")
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        os.chmod(tmp, path.stat().st_mode & 0o777)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _write_atomic_text(text: str, path: Path) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, path.stat().st_mode & 0o777)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _task_index_for(root: Path, task: str) -> tuple[int, int]:
    """``task``'s index, appending it to ``tasks.parquet`` when new.

    Returns ``(task_index, total_tasks)``. Written the way LeRobot's
    ``write_tasks`` does (a DataFrame indexed by ``task``), so its
    ``load_tasks`` reads it back unchanged.
    """
    import pandas as pd

    path = root / "meta" / "tasks.parquet"
    if path.is_file():
        tasks = pd.read_parquet(path)
    else:
        tasks = pd.DataFrame({"task_index": []}, index=pd.Index([], name="task"))
    if task in tasks.index:
        return int(tasks.loc[task].task_index), len(tasks)
    index = len(tasks)
    tasks.loc[task] = index
    tasks["task_index"] = tasks["task_index"].astype("int64")
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tasks.", suffix=".parquet")
    os.close(fd)
    try:
        tasks.to_parquet(tmp)
        if path.is_file():
            os.chmod(tmp, path.stat().st_mode & 0o777)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return index, len(tasks)


def _set_row(column: Any, row: int, value: Any) -> Any:
    import pyarrow as pa

    values = column.to_pylist()
    values[row] = value
    return pa.array(values, type=column.type)


def rename_episode_task(
    root: Path, episode_index: int, task: str, *, lock_timeout: float = 30.0
) -> EpisodeSummary:
    """Set a saved episode's task to ``task`` (see the module docstring)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    task = task.strip()
    if not task:
        raise DatasetBrowseError("task must not be empty")
    if len(task) > MAX_TASK_LENGTH:
        raise DatasetBrowseError(f"task is longer than {MAX_TASK_LENGTH} characters")

    info = _load_info(root)
    with dataset_metadata_lock(root, timeout=lock_timeout):
        meta_path, row = _find_episode_row(root, episode_index)
        meta_table = pq.read_table(meta_path)
        record = meta_table.slice(row, 1).to_pylist()[0]
        if list(record.get("tasks") or []) == [task]:
            return _summary(record, int(info.get("fps") or 0), _camera_keys(info), info)

        task_index, total_tasks = _task_index_for(root, task)

        # The episode's rows. Filter on episode_index: datasets recorded before
        # per-episode files share one data file between many episodes.
        data_template = info.get("data_path") or _DEFAULT_DATA_PATH
        data_path = root / data_template.format(
            chunk_index=int(record["data/chunk_index"]),
            file_index=int(record["data/file_index"]),
        )
        data = pq.read_table(data_path)
        column = data.schema.get_field_index("task_index")
        mine = pc.equal(
            data.column("episode_index"), pa.scalar(episode_index, pa.int64())
        )
        replaced = pc.if_else(
            mine,
            pa.scalar(task_index, data.schema.field(column).type),
            data.column(column),
        )
        _write_atomic_parquet(
            data.set_column(column, data.schema.field(column), replaced), data_path
        )

        # The episode's metadata row: its task list and its task_index stats
        # (every frame now carries one index, so min = max = mean, std = 0).
        for i, name in enumerate(meta_table.schema.names):
            if name == "tasks":
                value: Any = [task]
            elif (
                name.startswith("stats/task_index/")
                and name != "stats/task_index/count"
            ):
                value = [0.0] if name.endswith("/std") else [task_index]
                if pa.types.is_floating(meta_table.schema.field(i).type.value_type):
                    value = [float(v) for v in value]
            else:
                continue
            meta_table = meta_table.set_column(
                i,
                meta_table.schema.field(i),
                _set_row(meta_table.column(i), row, value),
            )
        _write_atomic_parquet(meta_table, meta_path)

        info = _load_info(root)
        if info.get("total_tasks") != total_tasks:
            info["total_tasks"] = total_tasks
            _write_atomic_text(json.dumps(info, indent=4), root / "meta" / "info.json")

        record["tasks"] = [task]

    with contextlib.suppress(Exception):
        from .ownership import restore_dataset_ownership

        restore_dataset_ownership(root)
    fps = int(info.get("fps") or 0)
    return _summary(record, fps, _camera_keys(info), info)


__all__ = [
    "DatasetBrowseError",
    "DatasetBusyError",
    "DatasetEpisodes",
    "EpisodeSummary",
    "EpisodeVideo",
    "MAX_TASK_LENGTH",
    "PreviewError",
    "METADATA_LOCK_FILENAME",
    "dataset_metadata_lock",
    "preview_step",
    "episode_preview",
    "episode_video",
    "read_episodes",
    "reload_tasks_from_disk",
    "rename_episode_task",
    "resolve_dataset",
]
