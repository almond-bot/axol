"""Restore operator ownership of datasets recorded by the root service.

The installer registers ``axol.service`` as root (it needs CAN bring-up, the
ZED cameras, and realtime scheduling) but points ``HF_LEROBOT_HOME`` at the
installing user's ``~/.cache/huggingface/lerobot``, so datasets recorded from
the control panel land root-owned inside a user directory. LeRobot writes the
episode mp4s through ``mkstemp``, which is always mode 0600, so those aren't
even world-readable like the parquet and meta files — the operator ends up
needing sudo to copy, inspect, or upload their own recording.

:func:`restore_dataset_ownership` hands the tree back. The recording commands
call it after every episode save — so even a crashed session leaves
operator-owned files — and again after the dataset is finalized, which writes
the last meta/stats files. The operator is identified as the owner of the
nearest non-root ancestor (the lerobot home the installer created as the
invoking user), and root-owned intermediates below it (the HuggingFace
``<org>/`` directory LeRobot creates) are adopted too.

Every traversal and chown goes through pinned no-follow directory descriptors
and rejects hard links/special files, so a concurrent swap of a path component
cannot redirect root's chown to an attacker-selected tree.

A no-op unless running as root, so plain CLI sessions are untouched, and
best-effort throughout: an episode save must never fail on a chown.
"""

import logging
import os
from pathlib import Path

from ..utils.state_files import (
    secure_chown_directory,
    secure_chown_tree,
    secure_directory_stat,
)

_logger = logging.getLogger(__name__)


def restore_dataset_ownership(dataset_root: Path) -> None:
    """Chown a root-recorded dataset tree back to the operator (see module doc)."""
    try:
        if os.geteuid() != 0:
            return
        # The operator owns the nearest non-root ancestor (the lerobot home the
        # installer created). All-root ancestry means the dataset really does
        # live in root's own tree — no HF_LEROBOT_HOME redirect, nothing to
        # restore.
        target: tuple[int, int] | None = None
        root_owned_parents: list[Path] = []
        for parent in dataset_root.parents:
            if parent == Path(parent.anchor):
                break
            st = secure_directory_stat(parent)
            if st.st_uid != 0:
                target = (st.st_uid, st.st_gid)
                break
            root_owned_parents.append(parent)
        if target is None:
            return
        uid, gid = target
        # Every traversal/open stays relative to pinned no-follow directory
        # descriptors. A concurrent ancestor swap therefore fails or mutates
        # only the directory inode we already opened—not an attacker-selected
        # tree elsewhere. Hard links, symlinks, and special files fail closed.
        secure_chown_tree(dataset_root, uid, gid)
        # Adopt root-owned intermediates between the dataset and the first
        # operator-owned ancestor (e.g. HuggingFace's ``<org>/`` directory).
        for parent in root_owned_parents:
            secure_chown_directory(parent, uid, gid)
    except OSError:
        _logger.exception("Could not restore ownership of %s", dataset_root)
