"""Keep the hosted install's shared uv tool env usable by the operator.

The hosted installer (``web/app/public/install``) installs ``axol`` as root
into ``UV_TOOL_DIR`` / ``UV_PYTHON_INSTALL_DIR`` under ``/opt/axol`` and links
the wrapper into ``/usr/local/bin``, so the operator's own login runs the same
install the root service does. That only works while every file in those
trees is world-readable (and every directory world-traversable); the
installer runs ``chmod -R a+rX`` once after its ``uv tool install``.

The ``axol serve`` self-updater reinstalls from inside the root service, which
runs with a ``027`` umask (:func:`almond_axol.serve.app.create_app`), so the
rebuilt venv comes back ``root:root 0750``: ``/usr/local/bin/axol`` then
resolves for root only and the operator's shell reports ``axol: command not
found``. The same goes for the pyzed / PyGObject installs provision adds to
the env. :func:`share` re-applies the installer's ``a+rX`` and is an ``axol
provision`` step, which every install and every update runs after its last
write to the env.
"""

from __future__ import annotations

import logging
import os
import stat
import sys
from pathlib import Path

_logger = logging.getLogger(__name__)

_READ = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
_EXEC = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH


def _shared_roots() -> list[Path]:
    """The shared install trees the running ``axol`` actually lives in.

    Only a directory named by the installer's environment *and* containing
    this interpreter's venv (``UV_TOOL_DIR``) or base Python
    (``UV_PYTHON_INSTALL_DIR``) qualifies, so a stray variable can never widen
    permissions on an unrelated tree.
    """
    roots: list[Path] = []
    for env, inside in (
        ("UV_TOOL_DIR", sys.prefix),
        ("UV_PYTHON_INSTALL_DIR", sys.base_prefix),
    ):
        value = os.environ.get(env)
        if not value or not os.path.isabs(value):
            continue
        root = Path(value).resolve()
        if not root.is_dir() or root == Path(root.anchor):
            continue
        if not Path(inside).resolve().is_relative_to(root):
            continue
        roots.append(root)
    return roots


def _make_readable(path: Path) -> bool:
    """``chmod a+rX`` one entry (symlinks untouched); True when it changed."""
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        return False
    wanted = mode | _READ
    if stat.S_ISDIR(mode) or mode & _EXEC:
        wanted |= _EXEC
    if wanted == mode:
        return False
    os.chmod(path, stat.S_IMODE(wanted))
    return True


def share() -> None:
    """Make the shared uv tool env + managed Python readable by every user."""
    if os.geteuid() != 0:
        return
    roots = _shared_roots()
    if not roots:
        _logger.info("shared install: not a hosted /opt install; nothing to share")
        return
    for root in roots:
        changed = int(_make_readable(root))
        for dirpath, dirnames, filenames in os.walk(root):
            base = Path(dirpath)
            for name in (*dirnames, *filenames):
                changed += _make_readable(base / name)
        if changed:
            _logger.info(
                "shared install: made %d entries under %s readable by all users",
                changed,
                root,
            )
        else:
            _logger.info("shared install: %s already readable by all users", root)
