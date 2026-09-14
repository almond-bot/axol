from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from almond_axol.recording.cartesian_frame import (
    CARTESIAN_FRAME_ID,
    write_cartesian_frame_marker,
)
from almond_axol.recording.datasets import is_dataset_dir, lerobot_home, list_datasets
from almond_axol.utils import adb
from almond_axol.utils.dotenv import _find_upwards, _parse, load_local_env


def _dataset(root: Path, repo_id: str, episodes: int, fps: int) -> Path:
    path = root / repo_id
    (path / "meta").mkdir(parents=True)
    (path / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": episodes, "fps": fps})
    )
    return path


def test_dataset_discovery_reads_metadata_and_skips_hidden(tmp_path: Path) -> None:
    first = _dataset(tmp_path, "org/first", 3, 30)
    second = _dataset(tmp_path, "second", 7, 60)
    hidden = _dataset(tmp_path, ".hidden/data", 1, 10)
    os.utime(first / "meta" / "info.json", (1, 1))
    os.utime(second / "meta" / "info.json", (2, 2))

    datasets = list_datasets(tmp_path)

    assert [d.repo_id for d in datasets] == ["second", "org/first"]
    assert datasets[0].episodes == 7
    assert datasets[0].fps == 60
    assert is_dataset_dir(first)
    assert hidden not in [Path(d.root) for d in datasets]


def test_lerobot_home_honors_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("HF_LEROBOT_HOME", raising=False)
    assert lerobot_home() == tmp_path / "hf" / "lerobot"
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "custom"))
    assert lerobot_home() == tmp_path / "custom"


def test_cartesian_frame_marker_is_atomic_metadata(tmp_path: Path) -> None:
    (tmp_path / "meta").mkdir()
    write_cartesian_frame_marker(tmp_path, migration={"source": "legacy"})

    marker = json.loads((tmp_path / "meta" / "axol.json").read_text())
    assert marker["cartesian_pose_frame"] == CARTESIAN_FRAME_ID
    assert marker["migration"] == {"source": "legacy"}
    assert not (tmp_path / "meta" / ".axol.json.tmp").exists()


def test_dotenv_parsing_precedence_and_real_env_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "project"
    nested = parent / "a" / "b"
    nested.mkdir(parents=True)
    (parent / ".env").write_text("PLAIN=base\nexport QUOTED='hello world'\nBAD\n")
    (parent / ".env.local").write_text('PLAIN=local\nEXTRA="yes"\n')
    monkeypatch.setenv("PLAIN", "process")
    monkeypatch.delenv("QUOTED", raising=False)
    monkeypatch.delenv("EXTRA", raising=False)

    assert _parse(parent / ".env") == {"PLAIN": "base", "QUOTED": "hello world"}
    assert _find_upwards(".env", nested) == parent / ".env"
    load_local_env(nested)
    assert os.environ["PLAIN"] == "process"
    assert os.environ["QUOTED"] == "hello world"
    assert os.environ["EXTRA"] == "yes"


def test_adb_status_ready_contract() -> None:
    assert adb.AdbStatus(True, "quest", "device", True).ready
    assert not adb.AdbStatus(True, "quest", "unauthorized", True).ready
    assert not adb.AdbStatus(False, None, "none", False).ready
