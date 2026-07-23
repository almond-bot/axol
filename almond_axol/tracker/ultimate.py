"""Vive Ultimate Tracker backend via the wireless dongle (USB HID).

The Ultimate Tracker is inside-out (camera SLAM, no base stations) with
no official Linux support; the community ``pyvut`` package drives the
wireless dongle over USB HID on Linux without a headset — pairing, SLAM
host/role assignment, and pose streaming for up to five trackers sharing
one map. ``pyvut`` is an operator-installed dependency (like libsurvive)
rather than a pip requirement of this project: install it from
https://github.com/nijkah/pyvut into the same environment, along with the
``hidapi`` system libraries it needs (see ``docs/cli/tracker.mdx``).

One-time provisioning caveat (upstream limitation): the SLAM **map must
be created once with VIVE Hub on Windows**; after the trackers store it,
the dongle + trackers run standalone on the Jetson.

Device keys are tracker MAC addresses (stable across sessions).

Frame conventions of the dongle's pose reports are firmware-dependent and
not officially documented, so both are configurable
(``ultimate_quat_order`` / ``ultimate_up_axis`` in
``~/.almond/tracker/config.json``) and must be verified at bring-up: hold
a tracker still and level and check the streamed pose is gravity-upright.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from .base import TrackerPose, TrackerSource, zup_to_yup_pos, zup_to_yup_quat

_logger = logging.getLogger(__name__)

# Dongle pose reports whose tracking_status is one of these are trustworthy;
# anything else (SLAM relocalising, gyro-only fallback) is streamed with
# tracking=False so the bridge holds the last good pose.
_GOOD_STATUS = {"tracking", "ok", 1, 3}


class UltimateSource(TrackerSource):
    """Poses for every Ultimate Tracker paired to the connected dongle.

    Args:
        quat_order: Component order of the report quaternion (``"xyzw"``
            or ``"wxyz"``).
        up_axis: Up axis of the tracker SLAM world frame (``"z"`` converts
            through the z-up → y-up basis change, ``"y"`` passes through).
    """

    def __init__(self, quat_order: str = "xyzw", up_axis: str = "z") -> None:
        if quat_order not in ("xyzw", "wxyz"):
            raise ValueError(f"quat_order must be xyzw or wxyz, got {quat_order!r}")
        if up_axis not in ("y", "z"):
            raise ValueError(f"up_axis must be y or z, got {up_axis!r}")
        self._quat_order = quat_order
        self._up_axis = up_axis
        self._poses: dict[str, TrackerPose] = {}
        self._lock = threading.Lock()
        self._api = None

    # -- Lifecycle -----------------------------------------------------------

    def start(self) -> None:
        try:
            from pyvut import UltimateTrackerAPI
        except ImportError:
            raise RuntimeError(
                "the ultimate backend needs the pyvut package "
                "(https://github.com/nijkah/pyvut) installed in this "
                "environment, plus the hidapi system libraries — see "
                "docs/cli/tracker.mdx."
            ) from None

        self._api = UltimateTrackerAPI(mode="DONGLE_USB")
        self._api.__enter__()
        self._api.add_pose_callback(self._on_pose)
        _logger.info("ultimate backend: dongle opened, waiting for tracker poses")

    def stop(self) -> None:
        if self._api is not None:
            try:
                self._api.__exit__(None, None, None)
            except Exception:  # noqa: BLE001 - best-effort teardown
                _logger.exception("ultimate backend teardown failed")
            self._api = None

    def poses(self) -> dict[str, TrackerPose]:
        with self._lock:
            return dict(self._poses)

    # -- Internal ---------------------------------------------------------------

    def _on_pose(self, pose: object) -> None:
        """pyvut pose callback (runs on its reader thread)."""
        try:
            key = str(pose.mac)
            pos = np.asarray(pose.position, dtype=np.float64)
            rot = np.asarray(pose.rotation, dtype=np.float64)
            status = getattr(pose, "tracking_status", None)
        except (AttributeError, TypeError, ValueError):
            return
        if pos.shape != (3,) or rot.shape != (4,):
            return

        if self._quat_order == "wxyz":
            rot = np.array([rot[1], rot[2], rot[3], rot[0]])
        if self._up_axis == "z":
            pos = zup_to_yup_pos(pos)
            rot = zup_to_yup_quat(rot)
        else:
            n = float(np.linalg.norm(rot))
            rot = rot / n if n > 0.0 else np.array([0.0, 0.0, 0.0, 1.0])

        tracking = True
        if status is not None:
            norm = status.lower() if isinstance(status, str) else status
            tracking = norm in _GOOD_STATUS

        sample = TrackerPose(
            pos=pos, quat=rot, t=time.perf_counter(), tracking=tracking
        )
        with self._lock:
            self._poses[key] = sample
