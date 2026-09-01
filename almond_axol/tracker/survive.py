"""Vive Tracker 3.0 backend via libsurvive (lighthouse tracking).

libsurvive tracks SteamVR 1.0/2.0 lighthouse devices fully open source —
no SteamVR — and runs on Linux/ARM (the Jetson). Its world frame is
right-handed **z-up**, gravity-aligned once the base stations are
calibrated, and shared by every tracked object; poses are converted here
to the y-up WebXR convention the teleop stack expects.

Axol uses the pinned, installer-attested **survive-cli** build with
``--record-stdout``. The recording stream prints
``<ts> <codename> POSE x y z qw qx qy qz`` lines which are parsed off a pipe.
Arbitrary PATH executables and separately installed ``pysurvive`` bindings are
not selected because they are not covered by the native artifact manifest.

Device keys are libsurvive codenames (``T20``, ``WM0``…), stable per
physical device (derived from its serial), so the left/right binding
saved by ``axol tracker.identify`` survives restarts.
"""

from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .base import (
    TrackerPose,
    TrackerSource,
    TrackerSourceError,
    epoch_seconds_to_perf_counter,
    zup_to_yup_pos,
    zup_to_yup_quat,
)
from .lighthouse_survey import LighthouseSurvey

_logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Whether the exact machine-installed libsurvive runtime is attested."""
    from ..cli.tracker_install import verified_survive_cli

    return verified_survive_cli() is not None


def _convert(
    pos_zup: np.ndarray, quat_wxyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """libsurvive (z-up, wxyz quat) → WebXR (y-up, xyzw quat)."""
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return zup_to_yup_pos(pos_zup), zup_to_yup_quat(quat_xyzw)


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
# libsurvive prints these while it is still acquiring each device; they are
# not a setup problem, unlike a persistent base-station channel clash.
_TRANSIENT_WARNINGS = ("Could not lighthouse more to",)
_MAX_WARNINGS = 20
_CHANNEL_CLASH = re.compile(r"Two or more lighthouses are on channel (\d+)")


def _lighthouse_record(parts: list[str]) -> tuple[int, str | None] | None:
    """Base-station (channel, serial) from an ``LH_UP`` / ``LH_POSE`` record.

    ``LH_UP <channel> ax ay az`` marks a station coming up; ``<channel> LH_POSE
    x y z qw qx qy qz <BaseStationID>`` follows once it is solved and names the
    station, so two serials on one channel can be reported by name.
    """
    try:
        if len(parts) >= 2 and parts[0] == "LH_UP":
            return int(parts[1]), None
        if len(parts) >= 10 and parts[1] == "LH_POSE":
            return int(parts[0]), f"{int(parts[9]):08x}"
    except ValueError:
        return None
    return None


def _setup_warning(line: str) -> str | None:
    """Return the message when ``line`` is a libsurvive ``Warning:`` entry.

    ``--record-stdout`` interleaves the coloured log stream with recording
    lines. Recording copies of the same entries arrive as ``INFO LOG`` records
    and are ignored here so each warning is captured once.
    """
    text = _ANSI_ESCAPE.sub("", line).strip()
    if not text.startswith("Warning:"):
        return None
    message = text[len("Warning:") :].strip()
    if not message or message.startswith(_TRANSIENT_WARNINGS):
        return None
    return message


class SurviveSource(TrackerSource):
    """Poses for every lighthouse-tracked object libsurvive sees.

    Requires the machine-wide artifact manifest created by
    ``axol tracker.install``. Raises ``RuntimeError`` from :meth:`start` when
    the pinned executable or any of its installed shared objects has drifted.
    """

    def __init__(self) -> None:
        self._poses: dict[str, TrackerPose] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self._cli_executable: Path | None = None
        self._failure: TrackerSourceError | None = None
        self._warnings: list[str] = []
        self._survey = LighthouseSurvey()
        # ``SimpleContext`` owns native USB/libsurvive state.  Keep a strong
        # reference until its polling loop has exited and the generated close
        # function has returned successfully.  A failed native call is
        # ownership-uncertain: never risk a second call against a pointer that
        # may already have been freed.
        self._simple_context: object | None = None
        self._pysurvive_module: object | None = None
        self._simple_close_attempted = False
        self._simple_close_failure: BaseException | None = None
        self._simple_lock = threading.Lock()

    # -- Lifecycle -----------------------------------------------------------

    def start(self) -> None:
        with self._simple_lock:
            if (
                self._simple_context is not None
                or self._simple_close_failure is not None
            ):
                raise TrackerSourceError(
                    "libsurvive native cleanup is incomplete; tracker ownership "
                    "is uncertain"
                ) from self._simple_close_failure
        if self._proc is not None:
            raise TrackerSourceError(
                "libsurvive process cleanup is incomplete; tracker ownership "
                "is uncertain"
            )
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
        from ..cli.tracker_install import verified_survive_cli

        executable = verified_survive_cli()
        if executable is None:
            raise RuntimeError(
                "the pinned libsurvive runtime is missing or changed. Install "
                "Lighthouse support from the control panel's Mantis settings, "
                "or run `axol tracker.install`."
            )
        self._cli_executable = executable
        target = self._run_cli
        _logger.info("survive backend: using attested survive-cli subprocess")
        self._thread = threading.Thread(
            target=self._run_worker,
            args=(target,),
            daemon=True,
            name="survive",
        )
        try:
            self._thread.start()
        except BaseException as error:
            # Thread.start() can be interrupted after the native owner begins.
            # Stop through the normal proof-oriented teardown path; if it cannot
            # prove exit, retain the thread/process references and fail closed.
            try:
                self.stop()
            except BaseException as cleanup_error:
                error.add_note(
                    "libsurvive startup rollback failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
                raise TrackerSourceError(
                    "libsurvive startup cleanup failed; tracker ownership is uncertain"
                ) from cleanup_error
            raise

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
            # A newly allocated Thread whose start() failed before creating a
            # native owner cannot be joined. Conversely, a completed thread has
            # an ``ident`` and must still be joined/reaped.
            was_started = (
                thread.is_alive() or getattr(thread, "ident", None) is not None
            )
            if was_started:
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
        # Closing the native context while NextUpdated() might still be on its
        # stack risks a use-after-free.  Only the proven-dead path may cross
        # into simple_close; a later stop() can retry the join if needed.
        if self._thread is None:
            try:
                self._close_simple_context()
            except BaseException as exc:
                failures.append(exc)
        with self._simple_lock:
            native_failure = self._simple_close_failure
        if native_failure is not None and native_failure not in failures:
            failures.append(native_failure)
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

    def warnings(self) -> list[str]:
        """Distinct libsurvive setup warnings seen so far (oldest first).

        libsurvive reports base-station and pairing problems only in its log
        stream, for example two base stations sharing a channel; poses can keep
        flowing while that silently degrades tracking.
        """
        with self._lock:
            return list(self._warnings)

    def lighthouse_survey(self) -> LighthouseSurvey:
        """Base stations seen so far (by channel and serial) and any clash."""
        with self._lock:
            survey = LighthouseSurvey(
                channels={ch: set(s) for ch, s in self._survey.channels.items()},
                conflicts=set(self._survey.conflicts),
                trackers=set(self._poses),
            )
        return survey

    def _note_warning(self, message: str) -> None:
        clash = _CHANNEL_CLASH.match(message)
        with self._lock:
            if clash is not None:
                self._survey.note_conflict(int(clash.group(1)))
            if message not in self._warnings and len(self._warnings) < _MAX_WARNINGS:
                self._warnings.append(message)

    def _note_lighthouse(self, channel: int, serial: str | None) -> None:
        with self._lock:
            self._survey.note_channel(channel, serial)

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

    def _publish(
        self,
        key: str,
        pos_zup: np.ndarray,
        quat_wxyz: np.ndarray,
        *,
        timestamp: float | None = None,
        timestamp_is_capture: bool = False,
    ) -> None:
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
        sample = TrackerPose(
            pos=pos,
            quat=quat,
            t=time.perf_counter() if timestamp is None else timestamp,
            timestamp_is_capture=timestamp_is_capture,
        )
        with self._lock:
            self._poses[key] = sample

    def _run_pysurvive(self) -> None:
        """Poll the pysurvive Simple API on this daemon thread."""
        import pysurvive

        actx = pysurvive.SimpleContext([])
        with self._simple_lock:
            self._simple_context = actx
            self._pysurvive_module = pysurvive
            self._simple_close_attempted = False
            self._simple_close_failure = None
        while not self._stop.is_set() and actx.Running():
            updated = actx.NextUpdated()
            if updated is None:
                time.sleep(0.001)
                continue
            name = updated.Name()
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            pose_result = updated.Pose()
            # Simple-API Pose() returns (SurvivePose, timecode) in current
            # bindings; older ones return the pose alone.
            native_timestamp = None
            if isinstance(pose_result, tuple):
                pose, native_timestamp = pose_result
            else:
                pose = pose_result
            receipt_epoch = time.time()
            receipt_perf = time.perf_counter()
            capture_perf = epoch_seconds_to_perf_counter(
                native_timestamp,
                receipt_perf=receipt_perf,
                receipt_epoch=receipt_epoch,
            )
            pos = np.asarray(pose.Pos, dtype=np.float64)
            rot = np.asarray(pose.Rot, dtype=np.float64)  # (w, x, y, z)
            self._publish(
                str(name),
                pos,
                rot,
                timestamp=capture_perf if capture_perf is not None else receipt_perf,
                timestamp_is_capture=capture_perf is not None,
            )
        if not self._stop.is_set():
            raise TrackerSourceError("pysurvive stopped running unexpectedly")

    def _close_simple_context(self) -> None:
        """Close the retained context once, after its polling thread exits."""
        with self._simple_lock:
            actx = self._simple_context
            pysurvive = self._pysurvive_module
            if actx is None:
                return
            if self._simple_close_attempted:
                if self._simple_close_failure is not None:
                    raise self._simple_close_failure
                return
            # Mark before crossing into native code: if it raises, whether the
            # pointer was consumed is unknowable and retrying could double-free.
            self._simple_close_attempted = True
        try:
            pysurvive.simple_close(actx.ptr)  # type: ignore[attr-defined]
        except BaseException as exc:
            with self._simple_lock:
                self._simple_close_failure = exc
            raise
        else:
            with self._simple_lock:
                self._simple_context = None
                self._pysurvive_module = None

    def _run_cli(self) -> None:
        """Parse POSE lines from a ``survive-cli --record-stdout`` stream."""
        executable = self._cli_executable
        if executable is None:
            from ..cli.tracker_install import verified_survive_cli

            executable = verified_survive_cli()
        if executable is None:
            raise TrackerSourceError("the attested survive-cli is unavailable")
        args = [
            str(executable),
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
        # Recording lines carry libsurvive run time at output, not the pose's
        # sensor time. Mapping its relative clock still removes pipe/buffering
        # delay; label it receipt/output time rather than true capture time.
        output_clock_offset: float | None = None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            # Format: "<run_ts> <codename> POSE x y z qw qx qy qz"
            parts = line.split()
            warning = _setup_warning(line)
            if warning is not None:
                self._note_warning(warning)
                continue
            lighthouse = _lighthouse_record(parts)
            if lighthouse is not None:
                self._note_lighthouse(*lighthouse)
                continue
            if len(parts) < 10 or parts[2] != "POSE":
                continue
            key = parts[1]
            try:
                output_time = float(parts[0])
                vals = [float(v) for v in parts[3:10]]
            except ValueError:
                continue
            receipt = time.perf_counter()
            if np.isfinite(output_time) and output_time >= 0.0:
                candidate_offset = receipt - output_time
                output_clock_offset = (
                    candidate_offset
                    if output_clock_offset is None
                    else min(output_clock_offset, candidate_offset)
                )
                sample_time = min(receipt, output_time + output_clock_offset)
            else:
                sample_time = receipt
            self._publish(
                key,
                np.array(vals[0:3]),
                np.array(vals[3:7]),  # (w, x, y, z)
                timestamp=sample_time,
            )
        code = self._proc.poll() if self._proc is not None else None
        if not self._stop.is_set():
            raise TrackerSourceError(f"survive-cli exited unexpectedly (code {code})")
