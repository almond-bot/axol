"""Resolve Almond's persistent per-machine operator state directory.

Interactive commands historically stored state below ``~/.almond``.  The
hosted control panel runs as root, however, so using :func:`Path.home` directly
gives it a different settings/calibration tree from the operator's CLI.  A
service can set ``ALMOND_HOME`` to the operator's existing ``~/.almond`` tree;
without that environment variable every caller retains the historical path.
"""

from __future__ import annotations

import os
from pathlib import Path

ALMOND_HOME_ENV = "ALMOND_HOME"


def almond_home() -> Path:
    """Return the root directory for persistent Almond operator state.

    ``ALMOND_HOME`` is expanded like a user-supplied path.  An unset or empty
    value preserves the original ``Path.home() / ".almond"`` behaviour.
    The value is resolved on each call so tests and subprocess launchers can
    set the environment before constructing a state path.
    """

    override = os.environ.get(ALMOND_HOME_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".almond"


def almond_path(*parts: str) -> Path:
    """Return a path below :func:`almond_home`."""

    return almond_home().joinpath(*parts)


def adopt_state_file(path: str | Path) -> None:
    """Give a root-created state file to the owner of ``ALMOND_HOME``.

    The installed service normally relies on its operator group + umask for
    shared writes.  A few files need an explicit restrictive/executable mode,
    though (TLS keys and root-run CAN startup scripts); handing those files to
    the state-directory owner keeps the interactive CLI able to use/update
    them. Paths outside ``ALMOND_HOME`` are deliberately ignored.
    """

    if os.geteuid() != 0:
        return
    root = almond_home()
    candidate = Path(path)
    try:
        candidate.resolve().relative_to(root.resolve())
        owner = root.stat()
        if owner.st_uid != 0:
            os.chown(candidate, owner.st_uid, owner.st_gid)
    except (OSError, ValueError):
        return
