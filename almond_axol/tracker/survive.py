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

Device keys are the trackers' serial numbers (``LHR-E0FCD01B``), read from
the ``<codename> CONFIG`` record libsurvive prints when a device connects.
libsurvive's own codenames (``WM0``, ``WM1``…) name the Watchman dongle slot
in the order the dongles were opened, so they can swap between runs; the
left/right binding saved by ``axol tracker.identify`` must not follow them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..utils.state_files import secure_atomic_write_bytes, secure_read_text
from .base import (
    TRACKER_POSE_MAX_AGE_S,
    TrackerPose,
    TrackerSource,
    TrackerSourceError,
    epoch_seconds_to_perf_counter,
    zup_to_yup_pos,
    zup_to_yup_quat,
)
from .diagnostics import FaultDiagnosticRecorder
from .lighthouse_survey import LighthouseSurvey
from .quality import LighthouseQuality, rotation_distance
from .survive_config import parse_survive_config

_logger = logging.getLogger(__name__)


def default_survive_config_path() -> Path:
    """Resolve the pinned Linux runtime's writable configuration, without I/O.

    A separate native init file makes calibration provenance ambiguous, so
    managed validation asks the operator to remove that override first.
    """
    env = {key.lower(): value for key, value in os.environ.items()}
    if env.get("survive_init-configfile"):
        raise RuntimeError(
            "Remove SURVIVE_INIT-CONFIGFILE before managed Lighthouse validation"
        )
    name = env.get("survive_configfile", "config.json")
    if not name:
        raise RuntimeError("SURVIVE_CONFIGFILE must not be empty")
    if name[0].isascii() and name[0].isalpha():
        root = Path(os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config"))
        return (root / "libsurvive" / name).absolute()
    return Path(name).absolute()


def is_available() -> bool:
    """Whether the exact machine-installed libsurvive runtime is attested."""
    from ..cli.tracker_install import verified_survive_cli

    return verified_survive_cli() is not None


# Where libsurvive resolves its default ``config.json`` when neither HOME nor
# XDG_CONFIG_HOME is set: the filesystem root. ``axol.service`` runs as root
# without HOME, so every service-launched survive-cli kept its base-station
# calibration there.
_LEGACY_ROOTLESS_CONFIG = Path("/config.json")


def survive_config_path() -> Path:
    """The libsurvive calibration file every Axol survive-cli run shares.

    libsurvive's own default (``$XDG_CONFIG_HOME/libsurvive`` or
    ``~/.config/libsurvive``), but resolved here and passed explicitly: the
    service has no HOME, and libsurvive would otherwise write to
    ``/config.json``. :meth:`Path.home` falls back to the passwd entry.
    """
    return default_survive_config_path()


def prepare_survive_config() -> Path:
    """Create the calibration directory and adopt a service-era calibration.

    A host whose service already calibrated its base stations into
    ``/config.json`` keeps that calibration instead of solving every station
    again from scratch.
    """
    path = survive_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() and os.geteuid() == 0:
        try:
            if "configed-lighthouse-gen" in _LEGACY_ROOTLESS_CONFIG.read_text():
                shutil.copy2(_LEGACY_ROOTLESS_CONFIG, path)
                _logger.info(
                    "adopted the libsurvive calibration from %s",
                    _LEGACY_ROOTLESS_CONFIG,
                )
        except (OSError, UnicodeError):
            pass
    return path


def _drain_lines(stream, tail: deque[str]) -> None:  # type: ignore[no-untyped-def]
    """Read a pipe to EOF, keeping its last lines (it must never fill up)."""
    try:
        for line in stream:
            line = _ANSI_ESCAPE.sub("", line).rstrip()
            if line:
                tail.append(line)
    except (OSError, ValueError):
        pass


def _describe_exit(code: int | None) -> str:
    if code is None:
        return "exit status unknown"
    if code < 0:
        try:
            return f"killed by {signal.Signals(-code).name}"
        except ValueError:
            return f"killed by signal {-code}"
    return f"code {code}"


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
# Keep a small crash breadcrumb without retaining the high-rate pose stream.
# survive-cli sends its own logs through --record-stdout; stderr may contain
# libusb diagnostics or a native assertion, so drain and retain it separately.
_CLI_DIAGNOSTIC_TAIL_LINES = 20
_CLI_DIAGNOSTIC_LINE_CHARS = 200
# The tracker firmware flags this in its light packets when it receives sync
# pulses from two stations set to the same channel.
_CHANNEL_CLASH = re.compile(r"Two or more lighthouses are on channel (\d+)")
_CALIBRATION_INVALIDATED = re.compile(
    r"Lighthouse calibration invalidated for station ([0-9a-fA-F]{8}); "
    r"recalibrate before tracking"
)
# The native global-scene solver discards a MAXITER proposal before publishing
# any lighthouse geometry. Its three numeric fields and lack of a tracker
# codename distinguish it from a failure of the live per-tracker pose solver.
_REJECTED_SCENE_REFINEMENT = re.compile(
    r"MPFIT status failure \S+/\S+/\S+ "
    r"\(\d+ measurements, 5, MP_MAXITER\)"
)
# Logged once per channel on the first sync pulse a tracker receives from it in
# this process, whether or not the saved calibration already knows the station:
# the definitive "a station is live on this channel" signal.
_LIVE_CHANNEL = re.compile(
    r"(?:OOTX not set for LH in channel|Adding lighthouse ch) (\d+)"
)
# Logged when a station's OOTX frame decodes and it is not the station the saved
# calibration expected on that channel (or none was saved), naming it.
_OOTX_PACKET = re.compile(r"Got OOTX packet (\d+) ([0-9a-fA-F]{1,8})\b")
# Emitted by the patched runtime after metadata checks and any invalidation
# warning. Serialized packet files are written before those checks and cannot
# prove that the saved geometry survived them.
_CHECKED_OOTX_PACKET = re.compile(r"Checked OOTX packet (\d+) ([0-9a-fA-F]{8})")
# The pinned native decoder writes these only after a CRC-valid Base Station
# 2.0 OOTX packet, including when its identity matches the saved calibration.
# Its ordinary Info log omits that unchanged case.
_OOTX_FILE = re.compile(r"LH([0-9]{2})_([0-9a-f]{8})\.ootx")
_OOTX_POLL_S = 0.5
_OOTX_MAX_FILES = 64
_OOTX_PAYLOAD_BYTES = 43


def _serialized_ootx_identity(path: Path) -> tuple[int, str] | None:
    """Read a complete native gen-2 packet, checking its filename and payload."""
    identity, _ = _inspect_serialized_ootx(path)
    return identity


def _inspect_serialized_ootx(
    path: Path,
) -> tuple[tuple[int, str] | None, dict[str, object]]:
    """Inspect one bounded packet read without retaining its raw contents."""
    details: dict[str, object] = {
        "name": path.name,
        "size": None,
        "status": "rejected",
        "reason": "invalid filename",
        "channel": None,
        "serial": None,
    }
    match = _OOTX_FILE.fullmatch(path.name)
    if match is None:
        return None, details
    channel, identity = int(match[1]), match[2]
    details.update(channel=channel + 1, serial=identity.upper())
    if not 0 <= channel < 16 or identity == "00000000":
        details["reason"] = "invalid channel or zero identity"
        return None, details
    if path.is_symlink() or not path.is_file():
        details["reason"] = "not a regular packet file"
        return None, details
    try:
        with path.open("rb") as stream:
            details["size"] = os.fstat(stream.fileno()).st_size
            packet = stream.read(_OOTX_PAYLOAD_BYTES + 1)
    except OSError:
        details.update(status="pending", reason="packet could not be read")
        return None, details
    # The native writer truncates then writes, so a poll can catch an
    # incomplete file. Retry next time; never treat a partial write as proof.
    if len(packet) < _OOTX_PAYLOAD_BYTES:
        details.update(status="pending", reason="incomplete packet")
        return None, details
    if len(packet) != _OOTX_PAYLOAD_BYTES:
        details["reason"] = "unexpected packet size"
        return None, details
    packet_id = f"{int.from_bytes(packet[2:6], 'little'):08x}"
    packet_channel = packet[31] & 0x7F
    if (channel, identity) != (packet_channel, packet_id):
        details["reason"] = "filename and packet identity do not match"
        return None, details
    details.update(status="accepted", reason=None)
    return (channel, identity), details


def _lighthouse_record(parts: list[str]) -> tuple[int, str | None] | None:
    """Base-station (channel, serial) from an ``LH_UP`` / ``LH_POSE`` record.

    Like every ``--record-stdout`` line, both carry the run timestamp first:
    ``<ts> LH_UP <channel> ax ay az`` marks a station coming up; ``<ts>
    <channel> LH_POSE x y z qw qx qy qz <BaseStationID>`` follows once it is
    solved and names the identity used by that geometry estimate.

    Both are also replayed from the saved calibration a few milliseconds after
    startup, before any tracker has received light. A later pose update can
    still use that cached identity; fresh identity proof comes from OOTX.
    """
    try:
        if len(parts) >= 3 and parts[1] == "LH_UP":
            return int(parts[2]), None
        if len(parts) >= 11 and parts[2] == "LH_POSE":
            return int(parts[1]), f"{int(parts[10]):08x}"
    except ValueError:
        return None
    return None


# survive-cli stamps every ``--record-stdout`` line with its run time
# (``survive_run_time()``: gettimeofday-based wall clock minus process start)
# while the rest of the pipeline uses ``perf_counter`` (CLOCK_MONOTONIC). NTP
# slew moves both clocks alike, but a wall-clock *step* — chrony ``makestep``,
# a systemd-timesyncd correction, a Jetson without an RTC battery getting the
# network mid-session — moves only the stamps. The mapper must therefore not
# trust any single historical offset forever: offsets older than this window
# are forgotten, and a jump of more than one pose lifetime that persists for
# the confirmation span is taken as a step rather than a pipe stall.
_OUTPUT_CLOCK_WINDOW_S = 5.0
_OUTPUT_CLOCK_RESET_JUMP_S = TRACKER_POSE_MAX_AGE_S
_OUTPUT_CLOCK_STEP_CONFIRM_S = 0.5

# survive-cli can die mid-session (SIGSEGV/SIGABRT inside libsurvive). Its
# calibration is on disk, so a fresh process tracks again within seconds; the
# bridge holds the rigs frozen meanwhile, exactly as for any tracking loss.
# Restart it, but give up on a crash loop rather than retrying forever.
_RESTART_BACKOFF_S = 1.0
_MAX_RESTARTS = 3
_RESTART_WINDOW_S = 60.0
# Lines of survive-cli stderr kept for the report when it dies (glibc's heap
# check or a sanitizer prints its diagnosis there).
_STDERR_TAIL_LINES = 40
# libsurvive enumerates USB devices only at startup: a tracker that drops off
# the bus (a cable tugged on a moving rig, a wireless tracker leaving range)
# never comes back in the same process. A fresh process also sometimes fails
# to switch one tracker into light-capture mode, leaving it connected but
# silent. After a disconnect or a crash, restart survive-cli and keep
# re-checking at this interval, for up to the recovery window, until every
# tracker seen earlier is reporting poses again.
_RECONNECT_DELAY_S = 2.0
_RECONNECT_RETRY_S = 5.0
_RECONNECT_WINDOW_S = 60.0
_DEVICE_DISCONNECT = re.compile(r"(\S+) Device disconnect\b")


class _PlannedRestart(Exception):
    """survive-cli was stopped on purpose to re-enumerate USB devices."""


class _OutputClockMapper:
    """Map survive-cli output stamps onto ``perf_counter`` at receipt.

    The offset between the two clocks is estimated as the *smallest* observed
    ``receipt - output_time``, i.e. the least pipe/buffer delay seen, so
    mapping a stamp through it removes that delay from later samples. A plain
    running minimum can only shrink, so after a backward wall-clock step of
    ``Δ`` every later sample would map to ``receipt - Δ`` and read as stale
    for the rest of the session. Two guards bound the recovery time instead:

    * the minimum is taken over a sliding window (:data:`_OUTPUT_CLOCK_WINDOW_S`)
      keyed on receipt time, so any stale offset expires by itself;
    * when every sample for :data:`_OUTPUT_CLOCK_STEP_CONFIRM_S` has sat more
      than :data:`_OUTPUT_CLOCK_RESET_JUMP_S` above the current minimum, the
      clock has stepped and the window is restarted from the new offset. A
      single late line (a pipe stall) does not qualify: its burst of buffered
      successors brings the candidate straight back down, and meanwhile the
      genuinely old samples stay labelled old.

    A forward step needs no special case: its smaller candidate becomes the
    new minimum on the very next sample. Mapped times never exceed receipt.
    """

    def __init__(
        self,
        *,
        window_s: float = _OUTPUT_CLOCK_WINDOW_S,
        reset_jump_s: float = _OUTPUT_CLOCK_RESET_JUMP_S,
        step_confirm_s: float = _OUTPUT_CLOCK_STEP_CONFIRM_S,
    ) -> None:
        self._window_s = window_s
        self._reset_jump_s = reset_jump_s
        self._step_confirm_s = step_confirm_s
        # (receipt, candidate offset) pairs with strictly increasing candidates
        # — a sliding-window minimum: the leftmost entry is the window's min.
        self._window: deque[tuple[float, float]] = deque()
        # Receipt of the first sample in the current run of jumped candidates.
        self._jump_since: float | None = None

    @property
    def offset(self) -> float | None:
        """Current ``perf_counter - output_time`` estimate, if any sample yet."""
        return self._window[0][1] if self._window else None

    def map(self, output_time: float, receipt: float) -> float:
        """Return the sample time on ``perf_counter``'s clock for one line.

        ``receipt`` is ``perf_counter()`` taken when the line was read. Stamps
        that are non-finite or negative are not mapped: the sample is labelled
        with its receipt time.
        """
        if not (np.isfinite(output_time) and output_time >= 0.0):
            return receipt
        candidate = receipt - output_time
        current = self.offset
        if current is not None and candidate - current > self._reset_jump_s:
            if self._jump_since is None:
                self._jump_since = receipt
            elif receipt - self._jump_since >= self._step_confirm_s:
                self._window.clear()
                self._jump_since = None
        else:
            self._jump_since = None
        while self._window and self._window[-1][1] >= candidate:
            self._window.pop()
        self._window.append((receipt, candidate))
        expiry = receipt - self._window_s
        while self._window[0][0] < expiry:
            self._window.popleft()
        return min(receipt, output_time + self._window[0][1])


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


def _setup_info(line: str) -> str | None:
    """Return the message when ``line`` is a libsurvive ``Info:`` log entry.

    As with warnings, the ``INFO LOG`` recording copies are ignored so each
    entry is seen once.
    """
    text = _ANSI_ESCAPE.sub("", line).strip()
    if not text.startswith("Info:"):
        return None
    message = text[len("Info:") :].strip()
    return message or None


def _live_lighthouse(message: str) -> tuple[int, str | None] | None:
    """Base-station (channel, serial) named by a libsurvive ``Info:`` entry.

    Only entries caused by light a tracker actually received qualify, so a
    match proves the station is powered and in view right now.
    """
    packet = _CHECKED_OOTX_PACKET.fullmatch(message) or _OOTX_PACKET.search(message)
    if packet is not None:
        return int(packet.group(1)), f"{int(packet.group(2), 16):08x}"
    live = _LIVE_CHANNEL.search(message)
    if live is not None:
        return int(live.group(1)), None
    return None


class SurviveSource(TrackerSource):
    """Poses for every lighthouse-tracked object libsurvive sees.

    Requires the machine-wide artifact manifest created by
    ``axol tracker.install``. Raises ``RuntimeError`` from :meth:`start` when
    the pinned executable or any of its installed shared objects has drifted.
    """

    def __init__(
        self,
        *,
        force_calibrate: bool = False,
        config_path: str | Path | None = None,
        expected_stations: int | None = None,
        allow_frame_refinement: bool = False,
        enable_diagnostics: bool = False,
    ) -> None:
        expected = 2 if expected_stations is None else expected_stations
        if (
            isinstance(expected, bool)
            or not isinstance(expected, int)
            or not 1 <= expected <= 16
        ):
            raise ValueError("expected_stations must be an integer between 1 and 16")
        self._expected_stations = expected
        self._diagnostics_enabled = enable_diagnostics and not (
            force_calibrate or allow_frame_refinement
        )
        self._diagnostics: FaultDiagnosticRecorder | None = None
        self._diagnostic_error: str | None = None
        self._diagnostic_fault_count = 0
        self._diagnostic_ready = False
        self._diagnostic_last_health = float("-inf")
        self._force_calibrate = force_calibrate
        self._setup_discovery = allow_frame_refinement and not force_calibrate
        # Setup must be able to acquire a newly solved coordinate frame before
        # the operator has identified both rigs. Recording uses the strict
        # default; allowing refinement still requires optical support and a
        # continuous five-second recovery after any frame discontinuity.
        self._allow_frame_refinement = force_calibrate or allow_frame_refinement
        self._config_path = (
            None if config_path is None else Path(config_path).expanduser().absolute()
        )
        self._runtime_lock = None
        self._validation_invalidated = False
        self._serials: dict[str, str] = {}
        self._quality = LighthouseQuality(
            expected, allow_frame_refinement=self._allow_frame_refinement
        )
        self._poses: dict[str, TrackerPose] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self._cli_executable: Path | None = None
        self._failure: TrackerSourceError | None = None
        self._runtime_geometry: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}
        self._checked_ootx: set[tuple[int, str]] = set()
        self._warnings: list[str] = []
        self._survey = LighthouseSurvey(expected=expected)
        self._ootx_directory: Path | None = None
        self._identity_started_at: float | None = None
        self._identity_last_poll_s: float | None = None
        self._identity_poll_count = 0
        self._identity_packets: dict[str, dict[str, object]] = {}
        self._known_serials: set[str] = set()
        self._recover_until = 0.0
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
        self._known_serials.clear()
        self._recover_until = 0.0
        self._close_diagnostics()
        self._diagnostics = None
        self._diagnostic_error = None
        self._diagnostic_ready = False
        self._diagnostic_last_health = float("-inf")
        with self._lock:
            self._failure = None
            self._runtime_geometry.clear()
            self._checked_ootx.clear()
            self._poses.clear()
            self._serials.clear()
            self._warnings.clear()
            self._survey = LighthouseSurvey(expected=self._expected_stations)
            self._identity_started_at = time.perf_counter()
            self._identity_last_poll_s = None
            self._identity_poll_count = 0
            self._identity_packets.clear()
            required = self._quality.required_trackers
            faults, last_fault = self._quality.fault_count, self._quality.last_fault
            self._quality = LighthouseQuality(
                self._expected_stations,
                allow_frame_refinement=self._allow_frame_refinement,
            )
            self._quality.required_trackers = required
            self._quality.fault_count, self._quality.last_fault = faults, last_fault
            self._diagnostic_fault_count = faults
            self._validation_invalidated = False
        from ..cli.tracker_install import verified_survive_cli

        executable = verified_survive_cli()
        if executable is None:
            raise RuntimeError(
                "the pinned libsurvive runtime is missing or changed. Install "
                "Lighthouse support from the control panel's Mantis settings, "
                "or run `axol tracker.install`."
            )
        self._cli_executable = executable
        from .lighthouse_validation import acquire_lighthouse_runtime_lock

        if self._runtime_lock is not None:
            raise TrackerSourceError("Lighthouse runtime ownership was not released")
        self._runtime_lock = acquire_lighthouse_runtime_lock()
        target = self._run_cli_supervised
        _logger.info("survive backend: using attested survive-cli subprocess")
        try:
            self._thread = threading.Thread(
                target=self._run_worker,
                args=(target,),
                daemon=True,
                name="survive",
            )
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
        self._cleanup_ootx_directory()
        self._close_diagnostics()
        if self._runtime_lock is not None:
            from .lighthouse_validation import release_lighthouse_runtime_lock

            release_lighthouse_runtime_lock(self._runtime_lock)
            self._runtime_lock = None

    def poses(self) -> dict[str, TrackerPose]:
        with self._lock:
            if self._failure is not None:
                raise self._failure
            quality = self._quality_snapshot()
            ready = quality["status"] in ("ready", "warning")
            return {
                key: replace(pose, tracking=ready) for key, pose in self._poses.items()
            }

    def set_required_trackers(self, serials: set[str]) -> None:
        """Limit collection health to bound rigs; discovery checks all devices."""
        with self._lock:
            self._quality.required_trackers = set(serials)

    def discovered_tracker_serials(self) -> set[str]:
        """Hardware identities seen in native configuration, even without poses.

        This is discovery evidence only: receiving a device identity does not
        establish localization or optical tracking quality.
        """
        with self._lock:
            return set(self._serials.values())

    def tracking_quality(self) -> dict[str, object]:
        """Observed tracking health, with a retained count of post-baseline faults."""
        with self._lock:
            report = self._quality_snapshot()
            if self._failure is not None:
                report["status"] = "error"
                report["nativeFailure"] = True
                report["reasons"] = [str(self._failure), *report["reasons"]]
            if self._diagnostics is not None:
                status = self._diagnostic_call("status")
                if isinstance(status, dict):
                    report["diagnostics"] = status
            if self._diagnostic_error is not None:
                report["diagnostics"] = {
                    "state": "error",
                    "error": self._diagnostic_error,
                }
            return report

    def _quality_snapshot(self) -> dict[str, object]:
        """Called with the source lock held; saved IDs alone cannot arm recording."""
        report = self._quality.snapshot(time.perf_counter(), time.time())
        checked_geometry = all(
            (channel, geometry[0]) in self._checked_ootx
            for channel, geometry in self._runtime_geometry.items()
        )
        if self._runtime_geometry and (
            not checked_geometry or self._survey.problems(require_serials=True)
        ):
            if report["status"] in {"ready", "warning"}:
                report["status"] = "checking"
            report["reasons"] = [
                *report["reasons"],
                "Waiting for the native runtime to check every fresh base-station "
                "identity broadcast before using saved calibration",
            ]
        recorder = self._diagnostics
        if recorder is not None:
            if report["status"] in {"ready", "warning"} and not self._diagnostic_ready:
                self._diagnostic_call("arm")
                self._diagnostic_ready = True
            self._diagnostic_fault(report)
            now = time.perf_counter()
            if now - self._diagnostic_last_health >= 0.1:
                self.diagnostic_event(
                    "health", report=report, identities=dict(self._serials)
                )
                self._diagnostic_last_health = now
        return report

    def _diagnostic_fault(self, context: dict | None = None) -> None:
        """Called under the source lock; recorder calls only enqueue bounded work."""
        count = self._quality.fault_count
        if count > self._diagnostic_fault_count and self._diagnostics is not None:
            self.diagnostic_trigger(
                self._quality.last_fault or "Native tracking fault",
                context=context or {"identities": dict(self._serials)},
            )
        self._diagnostic_fault_count = count

    def diagnostic_event(self, kind: str, **fields: object) -> None:
        """Record bridge timing without changing control or acceptance decisions."""
        self._diagnostic_call("record", kind, **fields)

    def diagnostic_trigger(self, reason: str, context: dict | None = None) -> None:
        self._diagnostic_call("trigger", reason, context=context)

    def _diagnostic_call(self, method: str, *args: object, **kwargs: object) -> object:
        if self._diagnostics is None or self._diagnostic_error is not None:
            return None
        try:
            return getattr(self._diagnostics, method)(*args, **kwargs)
        except Exception as exc:
            # Evidence capture must never interrupt pose delivery or motor control.
            self._diagnostic_error = f"Diagnostic recorder failed: {exc}"
            _logger.warning("%s", self._diagnostic_error)
            return None

    def _close_diagnostics(self) -> None:
        if self._diagnostics is not None:
            try:
                self._diagnostics.close()
            except Exception:
                _logger.exception("Could not finish the tracking diagnostic capture")

    def _start_diagnostics(self, config_path: Path) -> int | None:
        if not self._diagnostics_enabled:
            return None
        from ..cli.tracker_install import _BUILD_REVISION

        recorder = None
        try:
            recorder = FaultDiagnosticRecorder(
                metadata={
                    "expectedBaseStations": self._expected_stations,
                    "nativeBuild": _BUILD_REVISION,
                    "configSha256": hashlib.sha256(
                        config_path.read_bytes()
                    ).hexdigest(),
                    "pythonPid": os.getpid(),
                    "poseTimestampMeaning": "native output, not sensor capture",
                }
            )
            fd = recorder.start()
        except Exception as exc:
            self._diagnostic_error = f"Diagnostic recorder could not start: {exc}"
            _logger.warning("%s", self._diagnostic_error)
            if recorder is not None:
                try:
                    recorder.close()
                except Exception:
                    _logger.exception("Could not close unavailable diagnostic recorder")
            return None
        self._diagnostics = recorder
        _logger.info(
            "Tracking diagnostic recorder enabled; captures save automatically after faults"
        )
        return fd

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
                saved={ch: set(s) for ch, s in self._survey.saved.items()},
                expected=self._expected_stations,
            )
        return survey

    def _note_warning(self, message: str) -> None:
        clash = _CHANNEL_CLASH.match(message)
        invalidated = _CALIBRATION_INVALIDATED.fullmatch(message)
        rejected_refinement = _REJECTED_SCENE_REFINEMENT.fullmatch(message) is not None
        with self._lock:
            if invalidated is not None:
                self._quality.fault(
                    f"station-shift:{invalidated.group(1).lower()}",
                    message,
                    time.perf_counter(),
                    persistent=True,
                )
                # Fresh OOTX can invalidate the saved calibration before the
                # initial tracking window arms the normal fault counter.
                self._quality.last_fault = message
            elif clash is not None:
                self._survey.note_conflict(int(clash.group(1)))
                self._quality.fault(
                    f"channel:{clash.group(1)}",
                    message,
                    time.perf_counter(),
                    persistent=True,
                )
            elif not rejected_refinement and any(
                text in message
                for text in (
                    "Tracker lost",
                    "reseting calibration",
                    "Lost tracking for LH",
                    "MPFIT status failure",
                )
            ):
                self._quality.fault("native-reset", message, time.perf_counter())
            if message not in self._warnings and len(self._warnings) < _MAX_WARNINGS:
                self._warnings.append(message)
                if rejected_refinement:
                    _logger.warning(
                        "Lighthouse background calibration update rejected: %s",
                        message,
                    )
        if invalidated is not None:
            self._invalidate_changed_frame()

    def _note_info(self, message: str) -> None:
        live = _live_lighthouse(message)
        if live is None:
            return
        channel, serial = live
        with self._lock:
            if _CHECKED_OOTX_PACKET.fullmatch(message) and serial is not None:
                self._checked_ootx.add((channel, serial))
            self._survey.note_channel(channel, serial)
            known = self._quality.stations.get(channel)
            if serial is not None and known is not None and known["serial"] != serial:
                self._quality.fault(
                    f"channel:{channel}",
                    f"Base station identity changed on channel {channel + 1}; check channels and recalibrate",
                    time.perf_counter(),
                    persistent=True,
                )
            if channel in self._survey.clashing_channels():
                self._quality.fault(
                    f"channel:{channel}",
                    f"Multiple base station identities on channel {channel + 1}; assign distinct channels",
                    time.perf_counter(),
                    persistent=True,
                )

    def _note_lighthouse(self, channel: int, serial: str | None) -> None:
        """Keep startup station geometry as saved identity hints only.

        Even a later LH_POSE may still name the cached station before a fresh
        OOTX packet has decoded. Only a native OOTX log or serialized packet
        proves the identity currently transmitting on a channel.
        """
        with self._lock:
            if channel not in self._survey.channels:
                self._survey.note_saved(channel, serial)

    def _read_ootx_directory(
        self, directory: Path, *, receipt: float | None = None
    ) -> None:
        """Collect bounded current-run identity evidence, independent of pose solves."""
        now = time.perf_counter() if receipt is None else receipt
        with self._lock:
            if self._identity_started_at is None:
                self._identity_started_at = now
            elapsed = max(0.0, now - self._identity_started_at)
            self._identity_last_poll_s = elapsed
            self._identity_poll_count += 1
        entries = []
        for entry in directory.iterdir():
            if _OOTX_FILE.fullmatch(entry.name) is None:
                continue
            entries.append(entry)
            if len(entries) > _OOTX_MAX_FILES:
                raise TrackerSourceError(
                    "Too many Lighthouse identities received; check station channels"
                )
        for entry in entries:
            identity, details = _inspect_serialized_ootx(entry)
            with self._lock:
                prior = self._identity_packets.get(entry.name)
                if prior is None and len(self._identity_packets) >= _OOTX_MAX_FILES:
                    raise TrackerSourceError(
                        "Too many Lighthouse identities received; check station channels"
                    )
                first_accepted = (
                    None if prior is None else prior["firstAcceptedSeconds"]
                )
                if identity is not None and first_accepted is None:
                    first_accepted = elapsed
                details.update(
                    firstSeenSeconds=elapsed
                    if prior is None
                    else prior["firstSeenSeconds"],
                    firstAcceptedSeconds=first_accepted,
                    lastSeenSeconds=elapsed,
                )
                self._identity_packets[entry.name] = details
            if identity is not None:
                channel, serial = identity
                self._note_info(f"Got OOTX packet {channel} {serial}")

    def identity_diagnostics(self) -> dict[str, object]:
        """Return retained packet observations; does not poll or read packet files.

        Channels use the operator-facing 1–16 range. Times are measured from
        source startup to a directory poll, not optical transmission times.
        Seeing the same file again does not establish another decoded packet.
        Evidence remains available after stop and resets on the next start.
        """
        with self._lock:
            packets = [
                dict(self._identity_packets[name])
                for name in sorted(self._identity_packets)
            ]
            return {
                "elapsedSeconds": self._identity_last_poll_s,
                "pollCount": self._identity_poll_count,
                "packetCount": len(packets),
                "acceptedPacketCount": sum(
                    packet["firstAcceptedSeconds"] is not None for packet in packets
                ),
                "packets": packets,
            }

    def _cleanup_ootx_directory(self) -> None:
        """Remove packet evidence only after the native child is proven stopped."""
        if self._proc is not None and self._proc.poll() is None:
            raise TrackerSourceError(
                "libsurvive is still running; retain its packet directory and tracker ownership"
            )
        with self._lock:
            directory = self._ootx_directory
            self._ootx_directory = None
        if directory is not None:
            shutil.rmtree(directory)

    def _consume_record(
        self, line: str, clock: _OutputClockMapper, receipt: float
    ) -> bool:
        """Consume native structured telemetry without retaining high-rate lines."""
        parts = line.split()
        if len(parts) < 3:
            return False
        try:
            output_time = float(parts[0])
        except ValueError:
            return False
        if not np.isfinite(output_time) or output_time < 0:
            with self._lock:
                self._quality.fault(
                    "invalid-time", "Invalid native output timestamp", receipt
                )
            return True
        record = parts[2]
        if parts[1] == "EXTERNAL_TO_WORLD" and len(parts) >= 9:
            try:
                vals = np.asarray([float(value) for value in parts[2:9]])
            except ValueError:
                return True
            if np.all(np.isfinite(vals)) and np.linalg.norm(vals[3:]) > 1e-12:
                with self._lock:
                    self._quality.world_transform(vals[:3], vals[3:], receipt)
                self._invalidate_changed_frame()
            return True
        if record == "CONFIG":
            try:
                config = json.loads(line.split(None, 3)[3])
                serial = config.get("device_serial_number")
            except (IndexError, ValueError, AttributeError):
                return True
            if not isinstance(serial, str) or not serial.strip() or len(serial) > 128:
                return True
            serial = serial.strip()
            with self._lock:
                old = self._serials.get(parts[1])
                duplicate = any(
                    key != parts[1] and value == serial
                    for key, value in self._serials.items()
                )
                if (old is not None and old != serial) or duplicate:
                    self._quality.fault(
                        "device-identity",
                        "Tracker identity changed or is duplicated; stop and identify the rigs again",
                        receipt,
                        persistent=True,
                    )
                if old is not None and old != serial:
                    self._poses.pop(old, None)
                self._serials[parts[1]] = serial
                self._known_serials.add(serial)
            self.diagnostic_event("identity", device=parts[1], serial=serial)
            return True
        lighthouse = _lighthouse_record(parts)
        if lighthouse is not None:
            self._note_lighthouse(*lighthouse)
            if record == "LH_POSE":
                try:
                    vals = np.asarray([float(value) for value in parts[3:10]])
                except ValueError:
                    return True
                if np.all(np.isfinite(vals)) and np.linalg.norm(vals[3:]) > 1e-12:
                    channel, serial = lighthouse
                    self._check_runtime_geometry(channel, serial, vals, receipt)
                    with self._lock:
                        self._quality.station(
                            channel, serial, vals[:3], vals[3:], receipt
                        )
                    self._invalidate_changed_frame()
            return True
        with self._lock:
            serial = self._serials.get(parts[1])
        if record == "DISCONNECT":
            with self._lock:
                self._serials.pop(parts[1], None)
                if serial is not None:
                    self._poses.pop(serial, None)
                    self._quality.fault(
                        f"disconnect:{serial}",
                        f"{serial}: tracker disconnected",
                        receipt,
                    )
                    tracked = self._quality.trackers.get(serial)
                    if tracked is not None:
                        tracked.optical.clear()
                        tracked.optical_ticks.clear()
            return True
        if record not in {"POSE", "B", "A", "Y", "S", "I", "C", "VELOCITY"}:
            return False
        # A pose without a CONFIG hardware serial must not satisfy an old
        # codename binding, even when that alias looks familiar.
        if serial is None:
            return True
        if record == "POSE" and len(parts) >= 10:
            try:
                vals = [float(v) for v in parts[3:10]]
            except ValueError:
                return True
            timestamp = clock.map(output_time, receipt)
            self.diagnostic_event(
                "pose_receipt",
                device=parts[1],
                serial=serial,
                output_time=output_time,
                receipt_time=receipt,
                mapped_time=timestamp,
                position=vals[:3],
                quaternion_wxyz=vals[3:],
            )
            self._publish(
                serial,
                np.asarray(vals[:3]),
                np.asarray(vals[3:]),
                timestamp=timestamp,
                source_time=output_time,
                receipt=receipt,
            )
        elif record in {"A", "B"}:
            try:
                if record == "B":
                    channel, sensor, tick, plane = map(int, parts[3:7])
                    angle = float(parts[7])
                else:
                    sensor, acode, tick = map(int, parts[3:6])
                    plane = acode & 1
                    angle, channel = float(parts[7]), int(parts[8])
                if not np.isfinite(angle) or not 0 <= tick < 2**32:
                    return True
            except (ValueError, IndexError):
                return True
            timestamp = clock.map(output_time, receipt)
            self.diagnostic_event(
                "optical_receipt",
                device=parts[1],
                serial=serial,
                output_time=output_time,
                receipt_time=receipt,
                mapped_time=timestamp,
                channel=channel,
                sensor=sensor,
                tick=tick,
                plane=plane,
                angle=angle,
            )
            with self._lock:
                self._quality.optical(serial, channel, sensor, plane, tick, timestamp)
                if 0 <= channel < 16:
                    self._survey.note_channel(channel)
        return True

    def _check_runtime_geometry(
        self, channel: int, serial: str | None, pose: np.ndarray, receipt: float
    ) -> None:
        """Reject startup re-solves as well as later changes to fixed geometry."""
        if not self._runtime_geometry:
            return
        expected = self._runtime_geometry.get(channel)
        if expected is not None:
            identity, position, quaternion = expected
            orientation = pose[3:] / np.linalg.norm(pose[3:])
            # Native output rounds poses to six decimal places. These are
            # serialization tolerances, not looser tracking-quality thresholds.
            if (
                serial == identity
                and np.linalg.norm(pose[:3] - position) <= 1e-5
                and rotation_distance(orientation, quaternion) <= 1e-4
            ):
                return
        reason = (
            f"Native tracking changed saved geometry on base-station channel {channel + 1} "
            "while calibration was disabled. Stop and run Recalibrate Lighthouse."
        )
        with self._lock:
            self._quality.fault(
                f"station-shift:{channel}", reason, receipt, persistent=True
            )
            self._quality.last_fault = reason
        self._invalidate_changed_frame()
        raise TrackerSourceError(reason)

    def _invalidate_changed_frame(self) -> None:
        """Persist a rare geometry fault once, outside the hot tracking lock."""
        with self._lock:
            if self._validation_invalidated or not self._quality.calibration_required:
                return
            reason = self._quality.last_fault or "Lighthouse world frame changed"
        from .lighthouse_validation import invalidate_lighthouse_validation

        # Propagate failed persistence to the worker failure path: a known bad
        # calibration must never appear safely validated after a restart.
        invalidate_lighthouse_validation(reason)
        with self._lock:
            self._validation_invalidated = True

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
            self._quality.fault(
                "native-stopped", str(failure), time.perf_counter(), persistent=True
            )
            self._diagnostic_fault()
        self.diagnostic_trigger(str(failure), {"nativeFailure": True})
        _logger.error("%s", failure)

    def _publish(
        self,
        key: str,
        pos_zup: np.ndarray,
        quat_wxyz: np.ndarray,
        *,
        timestamp: float | None = None,
        timestamp_is_capture: bool = False,
        source_time: float | None = None,
        receipt: float | None = None,
    ) -> None:
        # libsurvive's CLI parser accepts IEEE nan/inf spellings, and bindings
        # can surface an incomplete/zero pose while tracking initializes.  Such
        # a sample must not get a fresh timestamp and pass the bridge's tracked
        # gate: non-finite values would otherwise propagate into the IK target.
        receipt = (
            time.perf_counter()
            if receipt is None and timestamp is None
            else (timestamp if receipt is None else receipt)
        )
        timestamp = receipt if timestamp is None else timestamp
        source_time = timestamp if source_time is None else source_time
        invalid = (
            pos_zup.shape != (3,)
            or quat_wxyz.shape != (4,)
            or not np.all(np.isfinite(pos_zup))
            or not np.all(np.isfinite(quat_wxyz))
            or not np.isfinite(timestamp)
            or not np.isfinite(source_time)
        )
        if invalid:
            with self._lock:
                if (
                    self._quality.required_trackers is None
                    or key in self._quality.required_trackers
                ):
                    self._quality.fault(
                        f"invalid:{key}", f"{key}: invalid pose or timestamp", receipt
                    )
            return
        quat_norm = float(np.linalg.norm(quat_wxyz))
        if not np.isfinite(quat_norm) or quat_norm <= 0.0:
            with self._lock:
                if (
                    self._quality.required_trackers is None
                    or key in self._quality.required_trackers
                ):
                    self._quality.fault(
                        f"invalid:{key}", f"{key}: invalid orientation", receipt
                    )
            return
        pos, quat = _convert(pos_zup, quat_wxyz)
        sample = TrackerPose(
            pos=pos,
            quat=quat,
            t=timestamp,
            timestamp_is_capture=timestamp_is_capture,
        )
        with self._lock:
            self._quality.pose(key, pos, quat, timestamp, source_time, receipt)
            self._poses[key] = sample
            self._diagnostic_fault()

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
            # This legacy path is not selected by start(), but must also avoid
            # exposing enumeration aliases as physical identities.
            name = pysurvive.survive_simple_serial_number(updated.ptr)
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

    def _run_cli_supervised(self) -> None:
        """Run survive-cli, restarting it after an unexpected exit.

        Poses already published are kept: they age out on their own capture
        time, so consumers see the gap as stale tracking rather than a
        backend failure. Only a crash loop (more than ``_MAX_RESTARTS`` exits
        within ``_RESTART_WINDOW_S``) is surfaced as the source's failure.
        """
        restarts: deque[float] = deque()
        while True:
            try:
                self._run_cli()
                return
            except _PlannedRestart:
                if self._stop.is_set():
                    return
                self._reset_native_evidence()
                continue
            except TrackerSourceError as exc:
                if self._stop.is_set():
                    return
                now = time.monotonic()
                while restarts and now - restarts[0] > _RESTART_WINDOW_S:
                    restarts.popleft()
                if len(restarts) >= _MAX_RESTARTS:
                    raise
                restarts.append(now)
                _logger.error(
                    "%s; restarting it (%d of %d within %.0f s)",
                    exc,
                    len(restarts),
                    _MAX_RESTARTS,
                    _RESTART_WINDOW_S,
                )
            self._reset_native_evidence()
            self._recover_until = time.monotonic() + _RECONNECT_WINDOW_S
            if self._stop.wait(_RESTART_BACKOFF_S):
                return

    def _reset_native_evidence(self) -> None:
        """Retain fault history, but never reuse live evidence across a restart."""
        if self._proc is not None and self._proc.poll() is None:
            raise TrackerSourceError(
                "Cannot restart while the previous native owner is alive"
            )
        self._close_diagnostics()
        self._diagnostics = None
        self._diagnostic_ready = False
        self._diagnostic_last_health = float("-inf")
        with self._lock:
            now = time.perf_counter()
            self._quality.fault(
                "native-restart",
                "Native tracking restarted; waiting for fresh stable tracking",
                now,
            )
            self._quality.trackers.clear()
            self._serials.clear()
            self._checked_ootx.clear()
            self._survey = LighthouseSurvey(expected=self._expected_stations)
            self._identity_started_at = now
            self._identity_last_poll_s = None
            self._identity_poll_count = 0
            self._identity_packets.clear()

    def _prepare_native_config(self, directory: Path) -> Path:
        """Keep native bookkeeping away from the validated, published baseline."""
        original_path = self._config_path or prepare_survive_config()
        try:
            contents = secure_read_text(original_path).encode("utf-8")
        except FileNotFoundError:
            if not (self._force_calibrate or self._setup_discovery):
                raise TrackerSourceError(
                    "Saved Lighthouse calibration is missing. Run Recalibrate Lighthouse "
                    "before validation or recording."
                ) from None
            contents = b"{}\n"
        try:
            config = parse_survive_config(contents.decode("utf-8"))
            if not isinstance(config, dict):
                raise ValueError("Expected a native configuration object")
            if not (self._force_calibrate or self._setup_discovery):
                identities: set[int] = set()
                channels: set[int] = set()
                geometry: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}
                floor_offset = float(config.get("floor-offset", 0))
                if not np.isfinite(floor_offset):
                    raise ValueError("The saved floor offset is not finite")
                for key, station in config.items():
                    if not re.fullmatch(r"lighthouse\d+", key) or not isinstance(
                        station, dict
                    ):
                        continue
                    try:
                        identity = int(station.get("id", 0))
                        channel = int(station.get("mode", -1))
                        pose = np.asarray(station.get("pose", []), dtype=float)
                        if (
                            int(station.get("PositionSet", 0)) != 1
                            or int(station.get("OOTXSet", 0)) != 1
                            or not 0 < identity <= 0xFFFFFFFF
                            or not 0 <= channel < 16
                            or identity in identities
                            or channel in channels
                            or pose.shape != (7,)
                            or not np.all(np.isfinite(pose))
                            or np.linalg.norm(pose[3:]) < 1e-12
                        ):
                            continue
                    except (TypeError, ValueError):
                        continue
                    identities.add(identity)
                    channels.add(channel)
                    position = pose[:3].copy()
                    position[2] -= floor_offset
                    geometry[channel] = (
                        f"{identity:08x}",
                        position,
                        pose[3:] / np.linalg.norm(pose[3:]),
                    )
                if len(identities) < self._expected_stations:
                    raise ValueError(
                        f"only {len(identities)} of {self._expected_stations} distinct "
                        "stations have saved identities and valid calibrated geometry"
                    )
                with self._lock:
                    self._runtime_geometry = geometry
        except (UnicodeError, ValueError) as exc:
            raise TrackerSourceError(
                f"Cannot use saved Lighthouse calibration at {original_path}: {exc}. "
                "Run Recalibrate Lighthouse before validation or recording."
            ) from exc
        if self._config_path is not None:
            # The calibration command owns a private candidate and publishes
            # it transactionally only after a separate runtime check passes.
            return self._config_path
        path = directory / "config.json"
        secure_atomic_write_bytes(path, contents)
        return path

    def _run_cli(self) -> None:
        """Parse POSE lines from a ``survive-cli --record-stdout`` stream."""
        if self._ootx_directory is not None:
            raise TrackerSourceError(
                "Native Lighthouse packet directory is still in use"
            )
        # Empty, owner-private evidence for this one process. Reusing the
        # working directory would let old packet files count as live stations.
        directory = Path(tempfile.mkdtemp(prefix="axol-lighthouse-"))
        self._ootx_directory = directory
        try:
            self._run_cli_in_directory(directory)
        finally:
            # A reader failure does not prove that the child has stopped. Its
            # packet writer requires this directory until stop() reaps it.
            if self._proc is None or self._proc.poll() is not None:
                self._cleanup_ootx_directory()

    def _run_cli_in_directory(self, directory: Path) -> None:
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
            # Angles provide optical freshness/visibility independent of pose
            # output. Consume them immediately into bounded per-sensor state;
            # raw light and IMU remain disabled.
            "--record-rawlight",
            "0",
            "--record-imu",
            "0",
            "--record-angle",
            "1",
            "--serialize-ootx",
            "1",
        ]
        config_path = self._prepare_native_config(directory)
        args.extend(["--configfile", str(config_path)])
        args.extend(["--force-calibrate", "1" if self._force_calibrate else "0"])
        # Calibration is an explicit preparation operation. Validation and
        # recording run the same solver configuration with GSS disabled, so
        # ordinary stationary observations cannot keep refitting the room or
        # moving its floor. The required patched native runtime also makes
        # disable-calibrate prevent local station bootstrapping (older builds
        # disabled only GSS). Pose/station guards still reject unexpected
        # geometry mutations or identity changes.
        args.extend(
            [
                "--disable-calibrate",
                "0" if self._force_calibrate else "1",
                "--globalscenesolver",
                "1" if self._force_calibrate else "0",
                "--gss-auto-floor-height",
                "1" if self._force_calibrate else "0",
                "--kalman-joint-model-lightcap",
                "-1",
                "--light-error-for-lh-confidence",
                "0",
                # The native residual filter needs an established pose prior.
                # Calibration/discovery retain all of their optical support.
                "--mpfit-filter-individual-outliers",
                "0" if self._allow_frame_refinement else "1",
            ]
        )
        # Native SURVIVE_* environment variables override command-line flags.
        # Keep this stream's protocol and explicit calibration destination
        # authoritative, particularly for a temporary calibration config.
        protected = {
            "configfile",
            "init-configfile",
            "force-calibrate",
            "disable-calibrate",
            "globalscenesolver",
            "gss-auto-floor-height",
            "kalman-joint-model-lightcap",
            "light-error-for-lh-confidence",
            "mpfit-filter-individual-outliers",
            "record-stdout",
            "record-angle",
            "record-rawlight",
            "record-imu",
            "serialize-ootx",
        }
        env = {
            key: value
            for key, value in os.environ.items()
            if not (key.upper().startswith("SURVIVE_") and key[8:].lower() in protected)
        }
        # A dedicated nonblocking channel keeps diagnostics out of the pose pipe.
        # Never inherit a caller-supplied descriptor into setup/calibration runs.
        env.pop("AXOL_SURVIVE_DIAGNOSTIC_FD", None)
        diagnostic_fd = self._start_diagnostics(config_path)
        pass_fds = [] if self._runtime_lock is None else [self._runtime_lock.fileno()]
        if diagnostic_fd is not None:
            env["AXOL_SURVIVE_DIAGNOSTIC_FD"] = str(diagnostic_fd)
            pass_fds.append(diagnostic_fd)
        try:
            self._proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
                cwd=directory,
                # An orphaned native owner retains the USB reservation.
                pass_fds=tuple(pass_fds),
            )
        finally:
            if self._diagnostics is not None:
                self._diagnostic_call("child_started")
        proc = self._proc
        if self._stop.is_set() and proc.poll() is None:
            proc.terminate()
        if (pid := getattr(proc, "pid", None)) is not None:
            _logger.info("survive-cli started (pid %s)", pid)
            self.diagnostic_event("native_started", pid=pid, arguments=args)
        assert proc.stdout is not None
        diagnostics: deque[str] = deque(maxlen=_CLI_DIAGNOSTIC_TAIL_LINES)
        stderr_tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        stderr_reader = None
        if getattr(proc, "stderr", None) is not None:
            stderr_reader = threading.Thread(
                target=_drain_lines,
                args=(proc.stderr, stderr_tail),
                daemon=True,
                name="survive-stderr",
            )
            stderr_reader.start()
        planned_restart = threading.Event()
        reporting: set[str] = set()
        timers: list[threading.Timer] = []

        def restart_for_devices() -> None:
            if self._stop.is_set() or proc.poll() is not None:
                return
            planned_restart.set()
            proc.terminate()

        def check_devices() -> None:
            missing = self._known_serials - reporting
            if missing and time.monotonic() < self._recover_until:
                _logger.info(
                    "tracker %s not reporting; restarting survive-cli to look for it again",
                    ", ".join(sorted(missing)),
                )
                restart_for_devices()

        if self._recover_until > time.monotonic():
            timers.append(threading.Timer(_RECONNECT_RETRY_S, check_devices))
            timers[-1].daemon = True
            timers[-1].start()
        # Recording lines carry libsurvive run time at output, not the pose's
        # sensor time. Mapping its relative clock still removes pipe/buffering
        # delay; label it receipt/output time rather than true capture time.
        output_clock = _OutputClockMapper()
        next_ootx_poll = float("-inf")
        try:
            for line in proc.stdout:
                if self._stop.is_set():
                    break
                receipt = time.perf_counter()
                if receipt >= next_ootx_poll:
                    self._read_ootx_directory(directory, receipt=receipt)
                    next_ootx_poll = receipt + _OOTX_POLL_S
                disconnect = _DEVICE_DISCONNECT.search(line)
                if disconnect is not None:
                    with self._lock:
                        lost = self._serials.pop(disconnect.group(1), None)
                        if lost is not None:
                            self._quality.fault(
                                f"disconnect:{lost}",
                                f"Tracker {lost} disconnected",
                                receipt,
                            )
                    if lost is not None:
                        reporting.discard(lost)
                        _logger.warning(
                            "tracker %s disconnected; restarting survive-cli in %.0f s so it is picked up again when it returns",
                            lost,
                            _RECONNECT_DELAY_S,
                        )
                        self._recover_until = time.monotonic() + _RECONNECT_WINDOW_S
                        timers.append(
                            threading.Timer(_RECONNECT_DELAY_S, restart_for_devices)
                        )
                        timers[-1].daemon = True
                        timers[-1].start()
                    continue
                warning = _setup_warning(line)
                if warning is not None:
                    self.diagnostic_event("native_warning", message=warning)
                    diagnostics.append(
                        ("Warning: " + warning)[:_CLI_DIAGNOSTIC_LINE_CHARS]
                    )
                    self._note_warning(warning)
                    continue
                info = _setup_info(line)
                if info is not None:
                    diagnostics.append(("Info: " + info)[:_CLI_DIAGNOSTIC_LINE_CHARS])
                    self._note_info(info)
                    continue
                if self._consume_record(line, output_clock, receipt):
                    parts = line.split(None, 3)
                    if len(parts) >= 3 and parts[2] == "POSE":
                        with self._lock:
                            serial = self._serials.get(parts[1])
                        if serial is not None:
                            reporting.add(serial)
                    continue
                diagnostic = _ANSI_ESCAPE.sub("", line).strip()
                if diagnostic:
                    diagnostics.append(diagnostic[:_CLI_DIAGNOSTIC_LINE_CHARS])
        finally:
            for timer in timers:
                timer.cancel()
        if self._stop.is_set():
            return
        if planned_restart.is_set():
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3.0)
            raise _PlannedRestart
        code = proc.poll()
        deadline = time.monotonic() + 2.0
        while code is None and time.monotonic() < deadline:
            time.sleep(0.05)
            code = proc.poll()
        if stderr_reader is not None:
            stderr_reader.join(timeout=1.0)
        detail = (
            f"survive-cli exited unexpectedly ({_describe_exit(code)}; code {code})"
        )
        if diagnostics:
            detail += "; recent native output: " + " | ".join(diagnostics)
        if stderr_tail:
            detail += "; its last stderr: " + " | ".join(stderr_tail)
        raise TrackerSourceError(detail)
