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
import os
from dataclasses import dataclass
from pathlib import Path


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
