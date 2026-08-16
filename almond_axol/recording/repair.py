"""Crash-consistency check + repair for resumed LeRobot datasets.

A recorder killed before it flushed (SIGKILL, OOM, power loss) loses whatever
was still buffered in its open parquet writers, while ``meta/info.json``'s
totals — rewritten on *every* ``save_episode`` — survive. The dataset is then
"crash-inconsistent": ``info.json`` counts episodes whose rows/metadata are
gone. Resuming such a dataset as-is would number the next episode past the
lost ones, and a permanent index gap poisons LeRobot's positional episode
lookups (``meta.episodes[i]`` is a row position, not a key), silently pairing
every post-gap episode with a different episode's video span.

The lost episodes are unrecoverable — a parquet file without its footer cannot
be read, and the recorder is the only thing that could have written it — so
the only useful recovery is exactly what :func:`ensure_resume_consistency`
performs: truncate the dataset back to the longest *verified* contiguous
prefix of episodes (metadata row present, data rows readable and complete,
video files present), drop the orphaned files the crashed session left
behind, and rewrite ``info.json``/``stats.json`` to match. After that the
dataset resumes cleanly at episode N.

Why the survivors always form a prefix: episodes are saved strictly in order,
and each parquet file is readable only once its writer wrote the footer — so
a crash can only ever lose a suffix (the buffered tail), never punch a hole
in the middle.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

_logger = logging.getLogger(__name__)

# On-disk layout of a LeRobot v3 dataset (mirrors lerobot.datasets.utils —
# spelled out here so the check/repair never has to import the full lerobot
# stack just to glob files).
_EPISODES_GLOB = "chunk-*/file-*.parquet"
_DATA_GLOB = "chunk-*/file-*.parquet"
_VIDEO_GLOB = "chunk-*/file-*.mp4"
_CHUNK_FILE_RE = re.compile(r"chunk-(\d+)/file-(\d+)\.")
_VIDEO_KEY_RE = re.compile(r"^videos/(.+)/chunk_index$")


def ensure_resume_consistency(dataset_root: "Path") -> None:
    """Verify a dataset can be resumed; repair it if a crash left it torn.

    Fast path: every episode ``info.json`` counts has a readable metadata row
    — nothing to do. Otherwise a previous session's recorder died before it
    flushed, so the tail episodes' frames are gone; repair by truncating to
    the longest verified contiguous prefix (see the module docstring).
    Raises ``RuntimeError`` only when nothing is salvageable (no complete
    episode survives), in which case the dataset should be deleted.
    """
    import pyarrow.parquet as pq

    root = Path(dataset_root)
    total = int(json.loads((root / "meta" / "info.json").read_text())["total_episodes"])

    tables: dict[Path, "pa.Table"] = {}
    unreadable_meta: list[Path] = []
    for f in sorted((root / "meta" / "episodes").glob(_EPISODES_GLOB)):
        try:
            tables[f] = pq.read_table(f)
        except Exception as exc:  # noqa: BLE001 - a footerless file is expected here
            _logger.warning("meta/episodes file %s is unreadable (%s)", f.name, exc)
            unreadable_meta.append(f)

    episodes: dict[int, dict[str, Any]] = {}
    for table in tables.values():
        for row in table.to_pylist():
            idx = int(row["episode_index"])
            if idx in episodes:
                raise RuntimeError(
                    f"Dataset {root} has duplicate episode_index {idx} in "
                    "meta/episodes — this is not a crash signature, refusing "
                    "to repair automatically."
                )
            episodes[idx] = row

    if not unreadable_meta and sorted(episodes) == list(range(total)):
        return  # consistent — the normal resume path

    _repair(root, total, episodes, tables, unreadable_meta)


def _verified_prefix_length(root: "Path", episodes: dict[int, dict]) -> int:
    """Length of the longest contiguous 0..N-1 prefix whose payloads exist.

    A metadata row alone doesn't make an episode usable: its data rows must be
    in a readable parquet file (a crash mid-session leaves the open data file
    footerless, taking every episode in it down with it), and every camera's
    video file must exist. Walks the prefix in order and stops at the first
    episode that fails verification.
    """
    import pyarrow.parquet as pq

    contiguous = 0
    while contiguous in episodes:
        contiguous += 1

    # Data verification, one read per referenced file: count each surviving
    # episode's rows and compare against its recorded length.
    counts_by_file: dict[tuple[int, int], dict[int, int]] = {}
    good = 0
    for i in range(contiguous):
        row = episodes[i]
        loc = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        if loc not in counts_by_file:
            path = root / "data" / f"chunk-{loc[0]:03d}" / f"file-{loc[1]:03d}.parquet"
            counts: dict[int, int] = {}
            try:
                col = pq.read_table(path, columns=["episode_index"])
                for v in col["episode_index"].to_pylist():
                    counts[int(v)] = counts.get(int(v), 0) + 1
            except Exception as exc:  # noqa: BLE001 - footerless/missing file
                _logger.warning("data file %s is unreadable (%s)", path, exc)
            counts_by_file[loc] = counts
        if counts_by_file[loc].get(i, 0) != int(row["length"]):
            break
        missing_video = False
        for key in _video_keys(row):
            vpath = (
                root
                / "videos"
                / key
                / f"chunk-{int(row[f'videos/{key}/chunk_index']):03d}"
                / f"file-{int(row[f'videos/{key}/file_index']):03d}.mp4"
            )
            if not vpath.is_file():
                _logger.warning("episode %d's video %s is missing", i, vpath)
                missing_video = True
        if missing_video:
            break
        good = i + 1
    return good


def _video_keys(row: dict[str, Any]) -> list[str]:
    return [m.group(1) for k in row if (m := _VIDEO_KEY_RE.match(k))]


def _repair(
    root: "Path",
    total: int,
    episodes: dict[int, dict[str, Any]],
    tables: dict[Path, "pa.Table"],
    unreadable_meta: list[Path],
) -> None:
    """Truncate the dataset to its verified prefix and drop orphaned files.

    Order matters for crash-safety: files are pruned first and ``info.json``
    is rewritten last, so a failure mid-repair leaves the dataset still
    flagged inconsistent and the next resume re-runs the (idempotent) repair.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    keep = _verified_prefix_length(root, episodes)
    lost = sorted(set(range(total)) - set(range(keep)))
    if keep == 0:
        raise RuntimeError(
            f"Dataset {root} is crash-inconsistent and nothing is salvageable: "
            f"info.json counts {total} episode(s) but not one complete episode "
            "(metadata + data rows + videos) survives on disk. Delete the "
            f"directory and start fresh:\n  rm -rf {root}"
        )

    removed: list[Path] = []

    def _remove(path: Path) -> None:
        os.remove(path)
        removed.append(path)

    # 1. meta/episodes: drop unreadable files and any rows past the prefix.
    for f in unreadable_meta:
        _remove(f)
    for f, table in tables.items():
        mask = pc.less(table["episode_index"], keep)
        kept = table.filter(mask)
        if kept.num_rows == table.num_rows:
            continue
        if kept.num_rows == 0:
            _remove(f)
            continue
        tmp = f.with_suffix(".parquet.tmp")
        pq.write_table(kept, tmp, compression="snappy", use_dictionary=True)
        os.replace(tmp, f)

    surviving = [episodes[i] for i in range(keep)]

    # 2. data: delete files no surviving episode references (the crashed
    # session's footerless file — resume always rotates to a fresh file, so
    # survivors never share one with lost episodes), and strip any stray
    # post-prefix rows from referenced files.
    data_refs = {
        (int(r["data/chunk_index"]), int(r["data/file_index"])) for r in surviving
    }
    for f in sorted((root / "data").glob(_DATA_GLOB)):
        m = _CHUNK_FILE_RE.search(f.as_posix())
        if m is None or (int(m.group(1)), int(m.group(2))) not in data_refs:
            _remove(f)
            continue
        try:
            table = pq.read_table(f)
        except Exception:  # noqa: BLE001 - referenced-but-unreadable is impossible
            # here: _verified_prefix_length already read this file.
            continue
        mask = pc.less(table["episode_index"], keep)
        kept = table.filter(mask)
        if kept.num_rows != table.num_rows:
            tmp = f.with_suffix(".parquet.tmp")
            pq.write_table(kept, tmp, compression="snappy", use_dictionary=True)
            os.replace(tmp, f)

    # 3. videos: delete per-key files no surviving episode references. (Lost
    # episodes appended into a survivor's file leave a harmless unreferenced
    # tail — every read is bounded by the survivors' from/to timestamps.)
    video_keys = _video_keys(surviving[-1])
    for key in video_keys:
        refs = {
            (
                int(r[f"videos/{key}/chunk_index"]),
                int(r[f"videos/{key}/file_index"]),
            )
            for r in surviving
        }
        for f in sorted((root / "videos" / key).glob(_VIDEO_GLOB)):
            m = _CHUNK_FILE_RE.search(f.as_posix())
            if m is None or (int(m.group(1)), int(m.group(2))) not in refs:
                _remove(f)

    # 4. stats.json: re-aggregate from the survivors' per-episode stats so the
    # lost episodes' contribution doesn't linger in training normalization.
    _recompute_stats(root, surviving)

    # 5. info.json last (see the docstring): totals now match what's on disk.
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_episodes"] = keep
    info["total_frames"] = int(surviving[-1]["dataset_to_index"])
    info["splits"] = {"train": f"0:{keep}"}
    info_path.write_text(json.dumps(info, indent=4) + "\n")

    _logger.warning(
        "repaired crash-inconsistent dataset at %s: a previous session's "
        "recorder was killed before it flushed, losing episode(s) %s. Kept "
        "the %d intact episode(s), removed %d orphaned file(s)%s; recording "
        "resumes at episode %d.",
        root,
        lost if lost else f"past {keep - 1}",
        keep,
        len(removed),
        (
            " (" + ", ".join(str(p.relative_to(root)) for p in removed) + ")"
            if removed
            else ""
        ),
        keep,
    )


def _recompute_stats(root: "Path", surviving: list[dict[str, Any]]) -> None:
    """Rewrite ``meta/stats.json`` from the surviving episodes' stats columns.

    Best-effort: the running aggregate absorbed the lost episodes when they
    saved, which only skews normalization statistics slightly — so a failure
    here (odd schema, missing stats columns) keeps the old file and logs
    instead of failing the repair.
    """
    try:
        import numpy as np
        from lerobot.datasets.compute_stats import aggregate_stats
        from lerobot.datasets.io_utils import write_stats

        per_episode: list[dict[str, dict[str, Any]]] = []
        for row in surviving:
            nested: dict[str, dict[str, Any]] = {}
            for k, v in row.items():
                if not k.startswith("stats/") or v is None:
                    continue
                _, feature, stat = k.split("/", 2)
                nested.setdefault(feature, {})[stat] = np.array(v)
            if nested:
                per_episode.append(nested)
        if per_episode:
            write_stats(aggregate_stats(per_episode), root)
    except Exception as exc:  # noqa: BLE001 - stats are a best-effort refresh
        _logger.warning(
            "could not re-aggregate stats.json from the surviving episodes "
            "(%s); keeping the existing file",
            exc,
        )
