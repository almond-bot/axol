"""Restore operator ownership of datasets recorded by the root service.

The installer registers ``axol.service`` as root (it needs CAN bring-up, the
ZED cameras, and realtime scheduling) but points ``HF_LEROBOT_HOME`` at the
installing user's home, so datasets recorded from the control panel land
root-owned inside a user directory. LeRobot writes the episode mp4s through
``mkstemp``, which is always mode 0600, so those aren't even world-readable
like the parquet and meta files — the operator ends up needing sudo to copy,
inspect, or upload their own recording.

:func:`restore_dataset_ownership` hands the tree back. The recording commands
call it after every episode save — so even a crashed session leaves
operator-owned files — and again after the dataset is finalized, which writes
the last meta/stats files. The operator is identified as the owner of the
nearest non-root ancestor (the lerobot home the installer created as the
invoking user), and root-owned intermediates below it (the HuggingFace
``<org>/`` directory LeRobot creates) are adopted too.

A no-op unless running as root, so plain CLI sessions are untouched, and
best-effort throughout: an episode save must never fail on a chown.
"""

import logging
import os
from pathlib import Path

_logger = logging.getLogger(__name__)


def restore_dataset_ownership(dataset_root: Path) -> None:
    """Chown a root-recorded dataset tree back to the operator (see module doc)."""
    try:
        if os.geteuid() != 0 or not dataset_root.is_dir():
            return
        # The operator owns the nearest non-root ancestor (the lerobot home the
        # installer created). All-root ancestry means the dataset really does
        # live in root's own tree — no HF_LEROBOT_HOME redirect, nothing to
        # restore.
        target: tuple[int, int] | None = None
        for parent in dataset_root.parents:
            st = parent.stat()
            if st.st_uid != 0:
                target = (st.st_uid, st.st_gid)
                break
        if target is None:
            return
        uid, gid = target
        # Adopt the root-owned intermediates between the dataset and that
        # ancestor (e.g. the HuggingFace "<org>/" directory).
        for parent in dataset_root.parents:
            if parent.stat().st_uid != 0:
                break
            os.lchown(parent, uid, gid)
        for dirpath, _dirnames, filenames in os.walk(dataset_root):
            os.lchown(dirpath, uid, gid)
            for name in filenames:
                os.lchown(os.path.join(dirpath, name), uid, gid)
    except OSError:
        _logger.exception("Could not restore ownership of %s", dataset_root)
