"""Expose root-recorded datasets without making their write path untrusted.

The installer registers ``axol.service`` as root (it needs CAN bring-up, the
ZED cameras, and realtime scheduling). Hosted LeRobot writes stay below a
root-owned, non-writable ``/var/lib`` boundary so an operator cannot redirect a
pathname open with a symlink race. After each save this module normalizes the
tree to root ownership, operator-group read/traverse access, directories 2750,
and files 0640. The login account can inspect/copy/upload datasets but cannot
mutate the live writer tree.

The historical direct-root CLI fallback still adopts files to the nearest
non-root ancestor, but does so through pinned no-follow descriptors and rejects
hard links/special files. Ordinary non-root CLI sessions are a no-op.

A no-op unless running as root, so plain CLI sessions are untouched, and
best-effort throughout: an episode save must never fail on a chown.
"""

import logging
import os
from pathlib import Path

from ..utils.state_files import (
    confine_service_dataset_path,
    privileged_service_active,
    secure_chown_directory,
    secure_chown_tree,
    secure_directory_stat,
    service_operator_gid,
)

_logger = logging.getLogger(__name__)


def restore_dataset_ownership(dataset_root: Path) -> None:
    """Make a root-recorded dataset safely readable (see module doc)."""
    try:
        if os.geteuid() != 0:
            return
        if privileged_service_active():
            dataset_root = confine_service_dataset_path(
                dataset_root,
                label="recorded dataset root",
            )
            gid = service_operator_gid()
            # Hosted datasets remain root-owned and non-writable to the login
            # account, closing all path-swap races in third-party writers. Give
            # the operator's group read/traverse access after each save; files
            # created by mkstemp(0600) become readable without becoming mutable.
            secure_chown_tree(
                dataset_root,
                0,
                gid,
                directory_mode=0o2750,
                file_mode=0o640,
            )
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
