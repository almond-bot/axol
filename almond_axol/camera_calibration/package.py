"""Calibration IDs and the evidence package.

Every session gets a new ID, ``<robot serial>-<YYYYMMDD>-<run>``, never
reused. Its package is a folder under ``~/.almond/camera_calibration/``
named by the ID, kept whether the session passed or failed:

- ``package.json``: identity, the image mode, every camera's intrinsics
  (the factory values used), every transform, which recording is A and which
  is B with their IDs and SHA-256, every metric and check, the statement that
  the values were fitted without Recording B, and PASS or FAIL;
- ``recording_A.mcap`` and ``recording_B.mcap``: the two recordings (video,
  joints and the head camera's raw IMU samples);
- ``corners_A.json`` and ``corners_B.json``: the corners found in every
  frame that was looked at, with its timestamp;
- ``locked_A.json``: the values fitted on A, written before B was examined;
- ``camera_calibration.json``: the calibration file the session wrote.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

PACKAGE_FORMAT = "almond-camera-calibration-package"


def next_id(robot_serial: str, when: datetime, packages: Path) -> str:
    """The next unused calibration ID for this robot and day."""
    prefix = f"{robot_serial or 'UNKNOWN'}-{when:%Y%m%d}-"
    used = []
    if packages.is_dir():
        for path in packages.iterdir():
            m = re.fullmatch(re.escape(prefix) + r"(\d+)", path.name)
            if m:
                used.append(int(m.group(1)))
    return f"{prefix}{max(used, default=0) + 1:02d}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> str:
    """Write ``data`` as JSON; returns its SHA-256."""
    text = json.dumps(data, indent=2, sort_keys=False, default=_plain) + "\n"
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def _plain(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def store_recording(source: Path, folder: Path, name: str) -> Path:
    """Move (or copy, across file systems) a recording into the package."""
    target = folder / name
    try:
        source.replace(target)
    except OSError:
        shutil.copy2(source, target)
    return target
