from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from almond_axol.recording import ownership


def test_restore_dataset_ownership_adopts_root_created_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePath:
        def __init__(self, value: str, uid: int = 0) -> None:
            self.value = value
            self.uid = uid
            self.parents: tuple[FakePath, ...] = ()

        def __fspath__(self) -> str:
            return self.value

        def __str__(self) -> str:
            return self.value

        def is_dir(self) -> bool:
            return True

        def stat(self) -> SimpleNamespace:
            return SimpleNamespace(st_uid=self.uid, st_gid=1001)

    operator_home = FakePath("/home/operator", uid=1000)
    lerobot_home = FakePath("/home/operator/lerobot")
    organization = FakePath("/home/operator/lerobot/org")
    dataset = FakePath("/home/operator/lerobot/org/dataset")
    dataset.parents = (organization, lerobot_home, operator_home)

    changes: list[tuple[str, int, int]] = []
    monkeypatch.setattr(ownership.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        ownership.os,
        "walk",
        lambda path: [(path, [], ["episode.mp4"])],
    )
    monkeypatch.setattr(
        ownership.os,
        "lchown",
        lambda path, uid, gid: changes.append((os.fspath(path), uid, gid)),
    )

    ownership.restore_dataset_ownership(dataset)
    changed_paths = {path for path, uid, gid in changes if (uid, gid) == (1000, 1001)}
    assert os.fspath(organization) in changed_paths
    assert os.fspath(dataset) in changed_paths
    assert "/home/operator/lerobot/org/dataset/episode.mp4" in changed_paths


def test_restore_dataset_ownership_noops_and_swallows_filesystem_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = SimpleNamespace(is_dir=lambda: True, parents=())
    calls: list[object] = []
    monkeypatch.setattr(ownership.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(ownership.os, "lchown", lambda *args: calls.append(args))
    ownership.restore_dataset_ownership(dataset)
    ownership.restore_dataset_ownership(SimpleNamespace(is_dir=lambda: False))
    assert calls == []

    monkeypatch.setattr(ownership.os, "geteuid", lambda: 0)
    broken = SimpleNamespace(
        is_dir=lambda: (_ for _ in ()).throw(OSError("gone")),
    )
    ownership.restore_dataset_ownership(broken)
