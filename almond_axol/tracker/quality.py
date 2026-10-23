"""Bounded, conservative Lighthouse recording diagnostics.

These checks detect gross discontinuities and loss of observed optical support;
they do not measure absolute accuracy. CLI timestamps describe native output,
not sensor capture. Keep the observation clock distinction in exported reports.
Callers serialize access with the source lock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .base import TRACKER_POSE_MAX_AGE_S

STABLE_WINDOW_S = 1.0
OPTICAL_MAX_AGE_S = 0.15
STATION_MAX_AGE_S = 2.0
MIN_OPTICAL_SENSORS = 4
# Broad human-motion envelope, intentionally not a smoothing or accuracy test.
JUMP_MIN_M = 0.10
JUMP_SPEED_M_S = 8.0
JUMP_MIN_RAD = math.radians(45)
JUMP_ANGULAR_SPEED_RAD_S = 25.0
STATION_SHIFT_M = 0.02
STATION_SHIFT_RAD = math.radians(2)


def rotation_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Sign-invariant geodesic distance between normalized quaternions."""
    return 2.0 * math.acos(float(np.clip(abs(np.dot(a, b)), 0.0, 1.0)))


@dataclass
class _Tracked:
    time: float | None = None
    source_time: float | None = None
    pos: np.ndarray | None = None
    quat: np.ndarray | None = None
    stable_since: float | None = None
    # At most 16 stations * 32 sensors. Values are mapped output times.
    optical: dict[int, dict[int, float]] = field(default_factory=dict)
    optical_ticks: dict[tuple[int, int, int], int] = field(default_factory=dict)
    speed: float | None = None
    angular_speed: float | None = None


class LighthouseQuality:
    """Online health state; faults stay counted after transient recovery."""

    def __init__(
        self, expected_stations: int = 2, *, allow_frame_refinement: bool = False
    ) -> None:
        self.expected_stations = expected_stations
        self.allow_frame_refinement = allow_frame_refinement
        self.stable_window_s = 5.0 if allow_frame_refinement else STABLE_WINDOW_S
        self.trackers: dict[str, _Tracked] = {}
        self.stations: dict[int, dict[str, object]] = {}
        self._station_baselines: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._events: dict[str, tuple[str, float, bool]] = {}
        self._armed = False
        self.fault_count = 0
        self.last_fault: str | None = None
        self.required_trackers: set[str] | None = None
        self._world_transform: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def calibration_required(self) -> bool:
        return any(
            persistent
            and (key == "world-transform" or key.startswith("station-shift:"))
            for key, (_, _, persistent) in self._events.items()
        )

    def world_transform(self, pos: np.ndarray, quat: np.ndarray, now: float) -> None:
        """Observe native external-to-world/floor transform changes."""
        quat = quat / np.linalg.norm(quat)
        previous = self._world_transform
        changed = (
            previous is not None
            and self._armed
            and (
                np.linalg.norm(pos - previous[0]) > STATION_SHIFT_M
                or rotation_distance(quat, previous[1]) > STATION_SHIFT_RAD
            )
        )
        if changed:
            action = (
                "wait for tracking to settle during setup"
                if self.allow_frame_refinement
                else "restart tracking and repeat validation"
            )
            self.fault(
                "world-transform",
                f"Native world frame changed; {action}",
                now,
                persistent=not self.allow_frame_refinement,
            )
        if previous is None or not self._armed or changed:
            self._world_transform = (pos.copy(), quat.copy())

    def fault(
        self, key: str, reason: str, now: float, *, persistent: bool = False
    ) -> None:
        previous = self._events.get(key)
        if self._armed and (
            previous is None
            or (not previous[2] and now - previous[1] >= STABLE_WINDOW_S)
        ):
            self.fault_count += 1
            self.last_fault = reason
        self._events[key] = (reason, now, persistent)
        for tracker in self.trackers.values():
            tracker.stable_since = None

    def optical(
        self,
        serial: str,
        channel: int,
        sensor: int,
        plane: int,
        tick: int,
        timestamp: float,
    ) -> None:
        if not (0 <= channel < 16 and 0 <= sensor < 32 and plane in (0, 1)):
            return
        tracker = self.trackers.setdefault(serial, _Tracked())
        key = (channel, sensor, plane)
        previous = tracker.optical_ticks.get(key)
        # uint32 hardware ticks wrap. Duplicate/backward observations must not
        # refresh support just because their pipe records arrived recently.
        if previous is not None and not 0 < ((tick - previous) & 0xFFFFFFFF) < 2**31:
            return
        tracker.optical_ticks[key] = tick
        sensors = tracker.optical.setdefault(channel, {})
        sensors[sensor] = max(sensors.get(sensor, -math.inf), timestamp)

    def station(
        self,
        channel: int,
        serial: str,
        pos: np.ndarray,
        quat: np.ndarray,
        now: float,
    ) -> None:
        if not 0 <= channel < 16:
            return
        previous = self.stations.get(channel)
        if previous is not None and previous["serial"] != serial:
            self.fault(
                f"channel:{channel}",
                f"Base station identity changed on channel {channel + 1}; check channels and recalibrate",
                now,
                persistent=True,
            )
        quat = quat / np.linalg.norm(quat)
        baseline = self._station_baselines.get(channel)
        if baseline is not None and self._armed:
            distance = float(np.linalg.norm(pos - baseline[0]))
            angle = rotation_distance(quat, baseline[1])
            if distance > STATION_SHIFT_M or angle > STATION_SHIFT_RAD:
                action = (
                    "wait for tracking to settle during setup"
                    if self.allow_frame_refinement
                    else "restart tracking and repeat validation"
                )
                self.fault(
                    f"station-shift:{channel}",
                    f"Base station {serial} estimate changed ({distance * 1000:.0f} mm, {math.degrees(angle):.1f} degrees); {action}",
                    now,
                    persistent=not self.allow_frame_refinement,
                )
                self._station_baselines[channel] = (pos.copy(), quat.copy())
        elif not self._armed:
            self._station_baselines[channel] = (pos.copy(), quat.copy())
        self.stations[channel] = {
            "serial": serial,
            "position": pos.tolist(),
            "quaternionWxyz": quat.tolist(),
        }

    def pose(
        self,
        serial: str,
        pos: np.ndarray,
        quat: np.ndarray,
        timestamp: float,
        source_time: float,
        now: float,
    ) -> None:
        if self.required_trackers is not None and serial not in self.required_trackers:
            return
        tracker = self.trackers.setdefault(serial, _Tracked())
        if tracker.time is not None and now - tracker.time > TRACKER_POSE_MAX_AGE_S:
            self.fault(f"pose-gap:{serial}", f"{serial}: pose stream interrupted", now)
        if tracker.source_time is not None:
            dt = source_time - tracker.source_time
            if dt <= 0:
                self.fault(
                    f"timestamp:{serial}",
                    f"{serial}: non-increasing pose timestamp",
                    now,
                )
            elif dt <= TRACKER_POSE_MAX_AGE_S:
                distance = float(np.linalg.norm(pos - tracker.pos))
                angle = rotation_distance(quat, tracker.quat)
                tracker.speed = distance / dt
                tracker.angular_speed = angle / dt
                if not math.isfinite(tracker.speed):
                    tracker.speed = None
                if not math.isfinite(tracker.angular_speed):
                    tracker.angular_speed = None
                if distance > max(JUMP_MIN_M, JUMP_SPEED_M_S * dt) or angle > max(
                    JUMP_MIN_RAD, JUMP_ANGULAR_SPEED_RAD_S * dt
                ):
                    self.fault(
                        f"jump:{serial}",
                        f"{serial}: pose discontinuity ({distance * 1000:.0f} mm, {math.degrees(angle):.0f} degrees in {dt * 1000:.0f} ms)",
                        now,
                    )
        tracker.time = timestamp
        tracker.source_time = source_time
        tracker.pos = pos.copy()
        tracker.quat = quat.copy()
        if tracker.stable_since is None:
            tracker.stable_since = now

    def snapshot(self, now: float, epoch: float) -> dict[str, object]:
        reasons: list[str] = []
        recent_stations: set[int] = set()
        trackers: dict[str, object] = {}
        selected = (
            self.trackers
            if self.required_trackers is None
            else {
                serial: self.trackers.get(serial, _Tracked())
                for serial in self.required_trackers
            }
        )
        all_stable = bool(selected)
        reduced_redundancy = False
        for serial, tracker in selected.items():
            stations: dict[str, object] = {}
            supported = False
            supported_stations = 0
            latest_optical: float | None = None
            for channel, sensors in tracker.optical.items():
                last = max(sensors.values())
                latest_optical = max(latest_optical or last, last)
                count = sum(now - t <= OPTICAL_MAX_AGE_S for t in sensors.values())
                if now - last <= STATION_MAX_AGE_S and channel in self.stations:
                    recent_stations.add(channel)
                if count >= MIN_OPTICAL_SENSORS:
                    supported = True
                    supported_stations += 1
                stations[str(channel + 1)] = {
                    "serial": self.stations.get(channel, {}).get("serial"),
                    "opticalAgeMs": max(0.0, (now - last) * 1000),
                    "sensorCount": count,
                }
            pose_age = None if tracker.time is None else max(0.0, now - tracker.time)
            fresh = pose_age is not None and pose_age <= TRACKER_POSE_MAX_AGE_S
            if not supported or not fresh:
                tracker.stable_since = None
                message = (
                    f"{serial}: insufficient recent optical observations; clear tracker visibility"
                    if not supported
                    else f"{serial}: pose stream is stale"
                )
                reasons.append(message)
                if self._armed:
                    self.fault(f"support:{serial}", message, now)
            stable = (
                fresh
                and supported
                and tracker.stable_since is not None
                and now - tracker.stable_since >= self.stable_window_s
            )
            all_stable &= stable
            if supported and supported_stations < min(2, self.expected_stations):
                reduced_redundancy = True
                reasons.append(
                    f"{serial}: reduced optical redundancy; only one base station has sufficient recent sensor observations"
                )
            trackers[serial] = {
                "opticalAgeMs": None
                if latest_optical is None
                else max(0.0, (now - latest_optical) * 1000),
                "poseAgeMs": None if pose_age is None else pose_age * 1000,
                "stations": stations,
                "speedMps": tracker.speed,
                "angularSpeedRadS": tracker.angular_speed,
                "supportedStationCount": supported_stations,
                "stable": stable,
            }
        recent_ids = {
            self.stations[channel]["serial"]
            for channel in recent_stations
            if self.stations[channel]["serial"] not in (None, "", "00000000")
        }
        if len(recent_ids) < self.expected_stations:
            reason = f"Only {len(recent_ids)} of {self.expected_stations} expected base stations have recent optical observations and distinct known identities"
            reasons.append(reason)
        events = [
            reason
            for reason, stamp, persistent in self._events.values()
            if persistent or now - stamp < STABLE_WINDOW_S
        ]
        reasons = list(dict.fromkeys(reasons + events))
        if not selected:
            reasons.append("Waiting for hardware tracker identities and optical poses")
        if all_stable and not events and len(recent_ids) >= self.expected_stations:
            self._armed = True
            status = "warning" if reduced_redundancy else "ready"
        elif all_stable and not events and self._armed:
            status = "warning"
        elif events:
            status = (
                "error"
                if self._armed or any(v[2] for v in self._events.values())
                else "checking"
            )
        else:
            status = "checking"
        if not reasons and not all_stable:
            reasons.append("Waiting for a stable tracking window")
        return {
            "status": status,
            "reasons": reasons,
            "faultCount": self.fault_count,
            "lastFault": self.last_fault,
            "trackers": trackers,
            "stations": {
                str(ch + 1): dict(station) for ch, station in self.stations.items()
            },
            "expectedBaseStations": self.expected_stations,
            "observedBaseStations": len(recent_ids),
            "recentStationCount": len(recent_ids),
            "reducedRedundancy": reduced_redundancy,
            "calibrationRequired": self.calibration_required,
            "checkedAt": epoch,
            "opticalClock": "native-output",
            "worldTransform": None
            if self._world_transform is None
            else {
                "position": self._world_transform[0].tolist(),
                "quaternionWxyz": self._world_transform[1].tolist(),
            },
            "accuracyVerified": False,
        }
