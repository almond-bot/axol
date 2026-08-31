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

import importlib.util
import logging
import shutil
import subprocess
import threading
import time
from collections.abc import Callable

import numpy as np

from .base import (
    TrackerPose,
    TrackerSource,
    TrackerSourceError,
    zup_to_yup_pos,
    zup_to_yup_quat,
)

_logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Whether this interpreter can launch either libsurvive transport."""
    return (
        importlib.util.find_spec("pysurvive") is not None
        or shutil.which("survive-cli") is not None
    )


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
        self._failure: TrackerSourceError | None = None

    # -- Lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            if self._thread.is_alive():
                raise TrackerSourceError(
                    "libsurvive reader is already running or cleanup is incomplete"
                )
            self._thread = None
        self._stop.clear()
        with self._lock:
            self._failure = None
            self._poses.clear()
        try:
            import pysurvive  # noqa: F401

            target = self._run_pysurvive
            _logger.info("survive backend: using pysurvive bindings")
        except ImportError:
            if shutil.which("survive-cli") is None:
                raise RuntimeError(
                    "libsurvive is not available: neither the pysurvive Python "
                    "bindings nor a survive-cli binary on PATH. Install Lighthouse "
                    "support from the control panel's Mantis settings, or run "
                    "`axol tracker.install`."
                ) from None
            target = self._run_cli
            _logger.info("survive backend: using survive-cli subprocess")
        self._thread = threading.Thread(
            target=self._run_worker,
            args=(target,),
            daemon=True,
            name="survive",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        failures: list[BaseException] = []
        proc = self._proc
        if proc is not None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    # kill() only sends a signal; wait again to reap the child
                    # and prove it no longer owns libsurvive/USB resources.
                    proc.wait(timeout=3.0)
            except BaseException as exc:
                failures.append(exc)
            else:
                self._proc = None
        thread = self._thread
        if thread is not None:
            try:
                thread.join(timeout=3.0)
            except BaseException as exc:
                failures.append(exc)
            if thread.is_alive():
                failures.append(
                    RuntimeError("libsurvive reader thread is still alive after stop")
                )
            else:
                self._thread = None
        if failures:
            for extra in failures[1:]:
                failures[0].add_note(
                    "additional libsurvive teardown failure: "
                    f"{type(extra).__name__}: {extra}"
                )
            raise TrackerSourceError(
                "libsurvive teardown failed; tracker ownership is uncertain"
            ) from failures[0]

    def poses(self) -> dict[str, TrackerPose]:
        with self._lock:
            if self._failure is not None:
                raise self._failure
            return dict(self._poses)

    # -- Internal ---------------------------------------------------------------

    def _run_worker(self, target: Callable[[], None]) -> None:
        """Run one libsurvive transport and retain any terminal failure."""
        try:
            target()
        except BaseException as exc:  # noqa: BLE001 - relay thread failures
            if self._stop.is_set():
                return
            failure = (
                exc
                if isinstance(exc, TrackerSourceError)
                else TrackerSourceError(
                    f"libsurvive reader failed ({type(exc).__name__}: {exc})"
                )
            )
        else:
            if self._stop.is_set():
                return
            failure = TrackerSourceError("libsurvive reader stopped unexpectedly")
        with self._lock:
            self._failure = failure
        _logger.error("%s", failure)

    def _publish(self, key: str, pos_zup: np.ndarray, quat_wxyz: np.ndarray) -> None:
        # libsurvive's CLI parser accepts IEEE nan/inf spellings, and bindings
        # can surface an incomplete/zero pose while tracking initializes.  Such
        # a sample must not get a fresh timestamp and pass the bridge's tracked
        # gate: non-finite values would otherwise propagate into the IK target.
        if pos_zup.shape != (3,) or quat_wxyz.shape != (4,):
            return
        if not np.all(np.isfinite(pos_zup)) or not np.all(np.isfinite(quat_wxyz)):
            return
        quat_norm = float(np.linalg.norm(quat_wxyz))
        if not np.isfinite(quat_norm) or quat_norm <= 0.0:
            return
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
        if not self._stop.is_set():
            raise TrackerSourceError("pysurvive stopped running unexpectedly")

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
            raise TrackerSourceError(f"survive-cli exited unexpectedly (code {code})")
