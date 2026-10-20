"""Stationary Lighthouse session checks and their expiring provenance.

These checks measure consistency, not absolute accuracy: without a known
physical reference a smooth, biased tracking solution remains possible.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from ..utils.paths import almond_path
from ..utils.state_files import (
    secure_atomic_write_json,
    secure_open_new_text,
    secure_open_text_read,
    secure_read_text,
)
from .base import TRACKER_POSE_MAX_AGE_S, valid_tracker_pose
from .config import TrackerConfig, load_tracker_config
from .lighthouse_survey import LighthouseSurvey, validate_expected_stations

VALIDATION_MAX_AGE_S = 12 * 60 * 60
VALIDATION_RUNTIME_MODE = "gss-disabled-optical-filter-v2"
LIGHTHOUSE_ACQUISITION_TIMEOUT_S = 120.0
_REPORT_SCHEMA = 1
_LEGACY_CODENAME = re.compile(r"(?:WM|T2|TR|WW|KN)\w$|HMD$", re.IGNORECASE)


def is_lighthouse_serial(key: object) -> bool:
    """Reject libsurvive's discovery-order aliases as persistent identities."""
    return (
        isinstance(key, str)
        and bool(key.strip())
        and not _LEGACY_CODENAME.fullmatch(key)
    )


def acquire_lighthouse_runtime_lock() -> TextIO:
    """Reserve native tracking across CLI and hosted processes until cleanup."""
    path = almond_path("tracker", "lighthouse_runtime.lock")
    try:
        handle = secure_open_new_text(path)
    except OSError:
        handle = secure_open_text_read(path)
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(
            "Lighthouse tracking is already in use. Stop collection or the other "
            "tracker session before calibration or validation."
        ) from None
    except BaseException:
        handle.close()
        raise
    return handle


def release_lighthouse_runtime_lock(handle: TextIO) -> None:
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def lighthouse_bindings(config: TrackerConfig) -> dict[str, str | None]:
    binding = config.bindings.get("survive", {})
    return {
        side: (getattr(config, side) or binding.get(side))
        if config.backend == "survive"
        else binding.get(side)
        for side in ("left", "right")
    }


def binding_problems(config: TrackerConfig) -> list[str]:
    bindings = lighthouse_bindings(config)
    if not all(is_lighthouse_serial(key) for key in bindings.values()):
        return [
            "Identify both Lighthouse trackers again using their hardware serials; "
            "old WM0/T20 discovery names cannot safely identify a physical rig."
        ]
    if bindings["left"] == bindings["right"]:
        return ["The two rig sides must use distinct tracker hardware serials."]
    return []


def validation_fingerprint(config: TrackerConfig, native_path: Path) -> str:
    """Bind a check to the actual native state and physical rig assignment."""
    native = secure_read_text(native_path).encode("utf-8")
    material = {
        "nativePath": str(native_path.absolute()),
        "runtimeMode": VALIDATION_RUNTIME_MODE,
        "nativeSha256": hashlib.sha256(native).hexdigest(),
        "bindings": lighthouse_bindings(config),
        "expectedBaseStations": validate_expected_stations(
            config.expected_base_stations
        ),
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def validation_report_path() -> Path:
    return almond_path("tracker", "lighthouse_validation.json")


def save_validation_report(report: dict[str, Any], path: Path | None = None) -> None:
    secure_atomic_write_json(path or validation_report_path(), report)


def invalidate_lighthouse_validation(reason: str, *, path: Path | None = None) -> None:
    """Latch a known calibration fault across source/process restarts."""
    target = path or validation_report_path()
    try:
        previous = json.loads(secure_read_text(target))
    except (FileNotFoundError, ValueError):
        previous = {}
    report = dict(previous) if isinstance(previous, dict) else {}
    report.update(
        schema=_REPORT_SCHEMA,
        status="failed",
        valid=False,
        calibrationRequired=True,
        invalidatedAt=time.time(),
        problems=[
            reason,
            "Repeat Lighthouse calibration and the session check before recording.",
        ],
    )
    save_validation_report(report, target)


def validation_snapshot(
    config: TrackerConfig | None = None,
    *,
    config_path: Path | None = None,
    report_path: Path | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Read-only interpretation of the last check; never starts native tracking."""
    config = config if config is not None else load_tracker_config()
    now = time.time() if now is None else now
    missing: dict[str, Any] = {
        "status": "missing",
        "valid": False,
        "checkedAt": None,
        "expiresAt": None,
        "expectedBaseStations": config.expected_base_stations,
        "calibrated": False,
        "metrics": {},
        "trackers": {},
        "baseStations": {},
        "problems": binding_problems(config)
        + ["Run the Lighthouse session check before recording."],
    }
    try:
        report = json.loads(secure_read_text(report_path or validation_report_path()))
    except (OSError, ValueError):
        return missing
    if not isinstance(report, dict) or report.get("schema") != _REPORT_SCHEMA:
        return missing
    result = {**missing, **report, "valid": False}
    problems = binding_problems(config)
    if report.get("status") == "needs_identification":
        result["problems"] = problems + [
            "Calibration saved. Identify the two rig sides, then run Validate tracking before recording."
        ]
        return result
    if report.get("status") != "passed":
        result["status"] = "failed"
        result["problems"] = problems + list(
            report.get("problems") or ["The last Lighthouse check did not pass."]
        )
        return result
    try:
        checked_at = float(report["checkedAt"])
        expires_at = float(report["expiresAt"])
        if not checked_at <= now < min(expires_at, checked_at + VALIDATION_MAX_AGE_S):
            problems.append("The Lighthouse session check has expired; run it again.")
        if config_path is None:
            from .survive import default_survive_config_path

            config_path = default_survive_config_path()
        current = validation_fingerprint(config, config_path)
        if current != report.get("fingerprint"):
            problems.append(
                "Tracker bindings, station count, tracking runtime or native calibration changed; run the session check again."
            )
    except (KeyError, TypeError, ValueError, OSError, RuntimeError):
        problems.append(
            "The current native calibration cannot be verified; run the session check again."
        )
    result["status"] = "stale" if problems else "passed"
    result["valid"] = not problems
    result["problems"] = problems
    return result


def require_lighthouse_validation(
    config: TrackerConfig | None = None, **kwargs: Any
) -> dict[str, Any]:
    report = validation_snapshot(config, **kwargs)
    if not report["valid"]:
        raise RuntimeError(
            "Lighthouse session check required: " + "; ".join(report["problems"])
        )
    return report


@dataclass(frozen=True)
class StationaryLimits:
    """Conservative startup consistency limits, recorded with every result."""

    duration_s: float = 5.0
    min_samples: int = 50
    position_p95_m: float = 0.005
    position_peak_m: float = 0.015
    angle_p95_deg: float = 1.0
    angle_peak_deg: float = 3.0


_POSE_DISCONTINUITY = re.compile(
    r"^(?P<subject>[^:]+): pose discontinuity "
    r"\((?P<position>[\d.]+) mm, (?P<angle>[\d.]+) degrees in (?P<interval>[\d.]+) ms\)$"
)
_STATION_CHANGE = re.compile(
    r"^(?P<subject>Base station \S+) estimate changed "
    r"\((?P<position>[\d.]+) mm, (?P<angle>[\d.]+) degrees\); .+$"
)
_FAILURE_GROUP_LIMIT = 16
_FAILURE_SUMMARY_LIMIT = 8
_FAILURE_EXAMPLE_LIMIT = 3
_FAILURE_TEXT_LIMIT = 300


class _FailureSummary:
    """Bound diagnostics independently of sample rate and varying fault numbers.

    A latched native reason can appear in many polls. Observation counts describe
    those polls, not distinct physical jumps; native fault counts stay separate.
    """

    def __init__(self, clock: Callable[[], float]) -> None:
        self.clock = clock
        self.started = clock()
        self.groups: dict[str, dict[str, Any]] = {}
        self.observations = 0
        self.omitted_observations = 0

    def append(self, reason: str) -> None:
        self.observations += 1
        match = _POSE_DISCONTINUITY.fullmatch(reason)
        category = "poseDiscontinuity"
        if match is None:
            match = _STATION_CHANGE.fullmatch(reason)
            category = "stationEstimateChange"
        if match:
            subject = match["subject"]
            key = f"{category}:{subject}"
            label = (
                f"{subject}: pose discontinuities"
                if category == "poseDiscontinuity"
                else f"{subject} estimate changed"
            )
        else:
            category = "trackingCheck"
            label = reason[:_FAILURE_TEXT_LIMIT]
            key = label
        group = self.groups.get(key)
        elapsed = max(0.0, self.clock() - self.started)
        if group is None:
            if len(self.groups) >= _FAILURE_GROUP_LIMIT:
                self.omitted_observations += 1
                return
            group = self.groups[key] = {
                "category": category,
                "summary": label,
                "observations": 0,
                "firstObservedSeconds": elapsed,
                "lastObservedSeconds": elapsed,
                "examples": [],
            }
        group["observations"] += 1
        group["lastObservedSeconds"] = elapsed
        example = reason[:_FAILURE_TEXT_LIMIT]
        if (
            example not in group["examples"]
            and len(group["examples"]) < _FAILURE_EXAMPLE_LIMIT
        ):
            group["examples"].append(example)
        if match:
            for name, value in (
                ("maxPositionDeltaMm", float(match["position"])),
                ("maxAngleDeltaDeg", float(match["angle"])),
            ):
                group[name] = max(group.get(name, 0.0), value)

    def extend(self, reasons: list[str]) -> None:
        for reason in reasons:
            self.append(reason)

    def summaries(self) -> list[str]:
        result = []
        groups = list(self.groups.values())
        for group in groups[:_FAILURE_SUMMARY_LIMIT]:
            message = group["summary"]
            if "maxPositionDeltaMm" in group:
                message += (
                    f" ({group['observations']} observations; maximum reported deltas "
                    f"{group['maxPositionDeltaMm']:g} mm and {group['maxAngleDeltaDeg']:g} degrees)"
                )
            result.append(message)
        if len(groups) > _FAILURE_SUMMARY_LIMIT or self.omitted_observations:
            result.append(
                "Additional tracking failures are summarized in the saved diagnostics."
            )
        return result

    def diagnostics(self, native_fault_count_delta: int) -> dict[str, Any]:
        return {
            "failureObservations": self.observations,
            "nativeFaultCountDelta": native_fault_count_delta,
            "countScope": "Repeated polls of a latched reason are observations, not distinct physical faults.",
            "groups": list(self.groups.values()),
            "omittedFailureObservations": self.omitted_observations,
            "groupLimit": _FAILURE_GROUP_LIMIT,
            "exampleLimitPerGroup": _FAILURE_EXAMPLE_LIMIT,
        }


def stationary_metrics(samples: list[Any]) -> dict[str, float | int]:
    """Position/orientation dispersion about the held pose, with quaternion sign equivalence."""
    positions = np.asarray([sample.pos for sample in samples], dtype=float)
    quats = np.asarray([sample.quat for sample in samples], dtype=float)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    _values, vectors = np.linalg.eigh(quats.T @ quats)
    center = vectors[:, -1]
    position_error = np.linalg.norm(positions - np.median(positions, axis=0), axis=1)
    angles = np.rad2deg(2 * np.arccos(np.clip(np.abs(quats @ center), 0.0, 1.0)))
    timestamps = np.asarray([sample.t for sample in samples])
    return {
        "samples": len(samples),
        "durationSeconds": float(timestamps[-1] - timestamps[0]),
        "positionP95Mm": float(np.percentile(position_error, 95) * 1000),
        "positionPeakMm": float(position_error.max() * 1000),
        "angleP95Deg": float(np.percentile(angles, 95)),
        "anglePeakDeg": float(angles.max()),
        "maxSampleGapMs": float(np.diff(timestamps).max() * 1000)
        if len(samples) > 1
        else 0.0,
    }


def collect_stationary_check(
    source: Any,
    bindings: dict[str, str | None],
    *,
    expected_stations: int,
    limits: StationaryLimits = StationaryLimits(),
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[
    dict[str, dict[str, float | int]], list[str], LighthouseSurvey, dict[str, Any]
]:
    """Observe every sample before smoothing, rejecting any loss during the hold."""
    samples: dict[str, list[Any]] = {side: [] for side in bindings}
    problems = _FailureSummary(clock)
    quality = source.tracking_quality()
    faults_before = quality.get("faultCount", 0)
    deadline = clock() + limits.duration_s
    while clock() < deadline:
        poses = source.poses()
        quality = source.tracking_quality()
        now = clock()
        if quality.get("status") != "ready":
            problems.extend(
                quality.get("reasons")
                or ["Tracking quality degraded during the stationary hold"]
            )
        for side, key in bindings.items():
            pose = poses.get(key)
            if (
                pose is None
                or not valid_tracker_pose(pose)
                or not pose.tracking
                or not 0 <= now - pose.t <= TRACKER_POSE_MAX_AGE_S
            ):
                problems.append(
                    f"{side} tracker lost a fresh, trustworthy pose during the hold"
                )
                continue
            previous = samples[side][-1] if samples[side] else None
            if previous is None or pose.t > previous.t:
                samples[side].append(pose)
            elif pose.t < previous.t:
                problems.append(f"{side} tracker timestamps went backwards")
        sleep(0.005)
    quality = source.tracking_quality()
    if quality.get("status") != "ready":
        problems.extend(quality.get("reasons") or [])
        problems.append("Tracking was not ready at the end of the stationary hold")
    if quality.get("faultCount", 0) != faults_before:
        problems.append("A native tracking fault occurred during the stationary hold")
    survey = source.lighthouse_survey()
    survey.expected = expected_stations
    problems.extend(survey.problems(require_serials=True))
    metrics: dict[str, dict[str, float | int]] = {}
    for side, readings in samples.items():
        if len(readings) < limits.min_samples:
            problems.append(
                f"{side} tracker provided too few distinct samples ({len(readings)}/{limits.min_samples})"
            )
            continue
        values = metrics[side] = stationary_metrics(readings)
        comparisons = (
            ("positionP95Mm", limits.position_p95_m * 1000),
            ("positionPeakMm", limits.position_peak_m * 1000),
            ("angleP95Deg", limits.angle_p95_deg),
            ("anglePeakDeg", limits.angle_peak_deg),
            ("maxSampleGapMs", TRACKER_POSE_MAX_AGE_S * 1000),
        )
        for metric, limit in comparisons:
            if values[metric] > limit:
                problems.append(
                    f"{side} {metric} {values[metric]:.2f} exceeds {limit:.2f}; hold the rig still and check tracking"
                )
        if values["durationSeconds"] < limits.duration_s * 0.8:
            problems.append(f"{side} tracker did not cover the full stationary hold")
    quality = {
        **quality,
        "validationDiagnostics": problems.diagnostics(
            max(0, quality.get("faultCount", 0) - faults_before)
        ),
    }
    return metrics, problems.summaries(), survey, quality


def collect_workspace_check(
    source: Any,
    bindings: dict[str, str | None],
    *,
    expected_stations: int,
    duration_s: float = 12.0,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], list[str]]:
    """Observe a slow guided sweep without inferring accuracy from smoothness."""
    readings: dict[str, list[Any]] = {side: [] for side in bindings}
    coverage: dict[str, set[str]] = {side: set() for side in bindings}
    problems = _FailureSummary(clock)
    initial_faults = source.tracking_quality().get("faultCount", 0)
    deadline = clock() + duration_s

    def usable(quality: dict[str, Any]) -> bool:
        return quality.get("status") == "ready" or (
            quality.get("status") == "warning"
            and quality.get("reducedRedundancy") is True
            and not quality.get("calibrationRequired", False)
            and quality.get("recentStationCount", 0) >= expected_stations
        )

    while clock() < deadline:
        poses = source.poses()
        quality = source.tracking_quality()
        if not usable(quality):
            problems.extend(
                quality.get("reasons")
                or ["Tracking quality degraded during the workspace sweep"]
            )
        now = clock()
        for side, key in bindings.items():
            pose = poses.get(key)
            if (
                pose is None
                or not valid_tracker_pose(pose)
                or not pose.tracking
                or not 0 <= now - pose.t <= TRACKER_POSE_MAX_AGE_S
            ):
                problems.append(
                    f"{side} lost trustworthy tracking during the workspace sweep"
                )
                continue
            previous = readings[side][-1] if readings[side] else None
            if previous is None or pose.t > previous.t:
                readings[side].append(pose)
            elif pose.t < previous.t:
                problems.append(
                    f"{side} tracker timestamps went backwards during the sweep"
                )
            tracker = quality.get("trackers", {}).get(key, {})
            for station in tracker.get("stations", {}).values():
                if (
                    station.get("serial")
                    and station.get("sensorCount", 0) >= 4
                    and station.get("opticalAgeMs", float("inf")) <= 150
                ):
                    coverage[side].add(station["serial"])
        sleep(0.005)
    final_quality = source.tracking_quality()
    if not usable(final_quality):
        problems.extend(final_quality.get("reasons") or [])
        problems.append("Tracking was not ready at the end of the workspace sweep")
    if final_quality.get("faultCount", 0) != initial_faults:
        problems.append("A native tracking fault occurred during the workspace sweep")
    metrics: dict[str, Any] = {}
    for side, samples in readings.items():
        if len(samples) < 50:
            problems.append(f"{side} provided too few distinct workspace samples")
            continue
        positions = np.asarray([sample.pos for sample in samples])
        quats = np.asarray([sample.quat for sample in samples])
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)
        travel = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        excursion = float(np.linalg.norm(positions - positions[0], axis=1).max())
        angle = float(
            np.rad2deg(2 * np.arccos(np.clip(np.abs(quats @ quats[0]), 0.0, 1.0))).max()
        )
        metrics[side] = {
            "samples": len(samples),
            "pathLengthM": travel,
            "positionExcursionM": excursion,
            "angleExcursionDeg": angle,
            "stationSerials": sorted(coverage[side]),
        }
        if excursion < 0.10 or angle < 15.0:
            problems.append(
                f"{side} needs a wider slow sweep (at least 10 cm and 15°); repeat the workspace check"
            )
        if len(coverage[side]) < min(2, expected_stations):
            problems.append(
                f"{side} did not receive enough distinct stations across the workspace; improve visibility"
            )
    observed = set().union(*coverage.values())
    if len(observed) < expected_stations:
        problems.append(
            f"Only {len(observed)} of {expected_stations} distinct stations contributed during the workspace sweep"
        )
    metrics["diagnostics"] = problems.diagnostics(
        max(0, final_quality.get("faultCount", 0) - initial_faults)
    )
    return metrics, problems.summaries()


def new_validation_report(config: TrackerConfig, *, calibrated: bool) -> dict[str, Any]:
    now = time.time()
    return {
        "schema": _REPORT_SCHEMA,
        "status": "failed",
        "valid": False,
        "checkedAt": now,
        "expiresAt": now + VALIDATION_MAX_AGE_S,
        "expectedBaseStations": config.expected_base_stations,
        "calibrated": False,
        "requestedCalibration": calibrated,
        "runtimeMode": VALIDATION_RUNTIME_MODE,
        "bindings": lighthouse_bindings(config),
        "metrics": {},
        "trackers": {},
        "baseStations": {},
        "limits": asdict(StationaryLimits()),
        "workspaceLimits": {
            "durationSeconds": 12.0,
            "minimumSamples": 50,
            "minimumExcursionM": 0.10,
            "minimumAngleExcursionDeg": 15.0,
            "minimumStationsPerTracker": min(2, config.expected_base_stations),
        },
        "problems": ["The session check did not finish."],
        "scope": "Stationary and workspace consistency only; absolute accuracy is not measured.",
    }
