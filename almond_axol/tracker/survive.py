"""Vive Tracker 3.0 backend via libsurvive (lighthouse tracking).

libsurvive tracks SteamVR 1.0/2.0 lighthouse devices fully open source —
no SteamVR — and runs on Linux/ARM (the Jetson). Its world frame is
right-handed **z-up**, gravity-aligned once the base stations are
calibrated, and shared by every tracked object; poses are converted here
to the y-up WebXR convention the teleop stack expects.

Two transports, tried in order:

1. **pysurvive** Python bindings (built from the libsurvive repo with
   ``python setup.py install`` — the PyPI wheel is outdated). Uses the
   Simple API: poll ``NextUpdated()`` on a daemon thread.
2. **survive-cli** subprocess with ``--record-stdout``: the recording
   stream prints ``<ts> <codename> POSE x y z qw qx qy qz`` lines which
   are parsed off a pipe. Slightly higher latency than the bindings but
   needs only the stock libsurvive build on PATH.

Device keys are libsurvive codenames (``T20``, ``WM0``…), stable per
physical device (derived from its serial), so the left/right binding
saved by ``axol tracker.identify`` survives restarts.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time

import numpy as np

from .base import TrackerPose, TrackerSource, zup_to_yup_pos, zup_to_yup_quat

_logger = logging.getLogger(__name__)


def _convert(
    pos_zup: np.ndarray, quat_wxyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """libsurvive (z-up, wxyz quat) → WebXR (y-up, xyzw quat)."""
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return zup_to_yup_pos(pos_zup), zup_to_yup_quat(quat_xyzw)


class SurviveSource(TrackerSource):
    """Poses for every lighthouse-tracked object libsurvive sees.

    Requires either the ``pysurvive`` bindings importable or a
    ``survive-cli`` binary on PATH (see ``docs/cli/tracker.mdx`` for the
    Jetson build steps). Raises ``RuntimeError`` from :meth:`start` when
    neither is available.
    """

    def __init__(self) -> None:
        self._poses: dict[str, TrackerPose] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None

    # -- Lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self._stop.clear()
        try:
            import pysurvive  # noqa: F401

            target = self._run_pysurvive
            _logger.info("survive backend: using pysurvive bindings")
        except ImportError:
            if shutil.which("survive-cli") is None:
                raise RuntimeError(
                    "libsurvive is not available: neither the pysurvive Python "
                    "bindings nor a survive-cli binary on PATH. Build libsurvive "
                    "from source (see docs/cli/tracker.mdx)."
                ) from None
            target = self._run_cli
            _logger.info("survive backend: using survive-cli subprocess")
        self._thread = threading.Thread(target=target, daemon=True, name="survive")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def poses(self) -> dict[str, TrackerPose]:
        with self._lock:
            return dict(self._poses)

    # -- Internal ---------------------------------------------------------------

    def _publish(self, key: str, pos_zup: np.ndarray, quat_wxyz: np.ndarray) -> None:
        pos, quat = _convert(pos_zup, quat_wxyz)
        sample = TrackerPose(pos=pos, quat=quat, t=time.perf_counter())
        with self._lock:
            self._poses[key] = sample

    def _run_pysurvive(self) -> None:
        """Poll the pysurvive Simple API on this daemon thread."""
        import pysurvive

        actx = pysurvive.SimpleContext([])
        while not self._stop.is_set() and actx.Running():
            updated = actx.NextUpdated()
            if updated is None:
                time.sleep(0.001)
                continue
            name = updated.Name()
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            pose = updated.Pose()
            # Simple-API Pose() returns (SurvivePose, timecode) in current
            # bindings; older ones return the pose alone.
            if isinstance(pose, tuple):
                pose = pose[0]
            pos = np.asarray(pose.Pos, dtype=np.float64)
            rot = np.asarray(pose.Rot, dtype=np.float64)  # (w, x, y, z)
            self._publish(str(name), pos, rot)

    def _run_cli(self) -> None:
        """Parse POSE lines from a ``survive-cli --record-stdout`` stream."""
        args = [
            "survive-cli",
            "--record-stdout",
            "1",
            # Only poses are consumed; muting the raw light/IMU/angle streams
            # keeps the pipe (and this parser) from drowning in telemetry.
            "--record-rawlight",
            "0",
            "--record-imu",
            "0",
            "--record-angle",
            "0",
        ]
        self._proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            # Format: "<run_ts> <codename> POSE x y z qw qx qy qz"
            parts = line.split()
            if len(parts) < 10 or parts[2] != "POSE":
                continue
            key = parts[1]
            try:
                vals = [float(v) for v in parts[3:10]]
            except ValueError:
                continue
            self._publish(
                key,
                np.array(vals[0:3]),
                np.array(vals[3:7]),  # (w, x, y, z)
            )
        code = self._proc.poll() if self._proc is not None else None
        if not self._stop.is_set():
            _logger.error("survive-cli exited unexpectedly (code %s)", code)
