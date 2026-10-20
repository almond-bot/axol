"""Episode-level tracking verdicts, independent of the robot and recording IO.

Native diagnostics run before pose filtering. Their cumulative fault counter
survives brief failures between control ticks, so a recovered pose cannot erase
an earlier tracking failure from a take's verdict.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any


def quality_ready(quality: dict[str, Any] | None) -> bool:
    """Reduced redundancy may be usable; missing diagnostics never are."""
    return quality is not None and quality.get("status") in {"ready", "warning"}


def _recovery_or_coverage_note(reason: str) -> bool:
    """Identify live status notes that do not describe another tracking fault.

    A failed tracker may recover optical support before its stable window
    finishes. Those polls must not replace the jump/outage that rejected the
    take, or promote the accompanying amber coverage note to a new cause.
    """
    return reason == "Waiting for a stable tracking window" or (
        ": reduced optical redundancy; " in reason
    )


@dataclass
class EpisodeTrackingQuality:
    """Latch failures for one take while retaining the live recovery status."""

    required: bool = False
    faults: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    started_at: float | None = None
    _fault_count: int = 0
    _source: str | None = None
    _latest: dict[str, Any] | None = None

    def begin(self, quality: dict[str, Any] | None) -> None:
        if self.required and not quality_ready(quality):
            raise ValueError("Tracking checks are not ready; restore visibility first.")
        self.faults.clear()
        self.events.clear()
        self.started_at = time.time()
        self._latest = copy.deepcopy(quality)
        self._fault_count = int((quality or {}).get("faultCount", 0))
        self._source = (quality or {}).get("sourceId")

    def reject(self, reason: str) -> None:
        if reason in self.faults:
            return
        # Bound diagnostic memory even when a broken source varies its text.
        if len(self.faults) < 32:
            self.faults.append(reason)
            self.events.append({"at": time.time(), "reason": reason})

    def update(
        self, quality: dict[str, Any] | None, *, recording: bool
    ) -> dict[str, Any] | None:
        self._latest = copy.deepcopy(quality)
        if quality is None and not self.required:
            return None
        current = quality or {
            "status": "checking",
            "reasons": ["Waiting for live Lighthouse tracking checks."],
            "faultCount": 0,
            "lastFault": None,
        }
        if recording:
            count = int(current.get("faultCount", 0))
            source = current.get("sourceId")
            if count < self._fault_count or source != self._source:
                self.reject(
                    "Tracker diagnostics restarted during the take; retake required."
                )
            elif count > self._fault_count:
                self.reject(
                    current.get("lastFault")
                    or "Tracking failed briefly during the take; retake required."
                )
            self._fault_count = count
            if not quality_ready(quality):
                reasons = [str(reason) for reason in current.get("reasons") or []]
                failures = [
                    reason
                    for reason in reasons
                    if not _recovery_or_coverage_note(reason)
                ]
                for reason in failures:
                    self.reject(reason)
                # Continue to fail closed if a checking/error frame is the
                # only evidence observed during the take. Once its cause is
                # known, recovery status alone is not another failure event.
                if not self.faults:
                    self.reject(
                        "; ".join(reasons) or "Tracking quality is unavailable."
                    )
        snapshot = copy.deepcopy(current)
        snapshot["retakeRequired"] = bool(self.faults)
        if self.faults:
            snapshot["status"] = "error"
            snapshot["reasons"] = list(self.faults)
            # Explain why this take first became invalid. Later recovery
            # notes or other faults must not overwrite its primary cause.
            snapshot["lastFault"] = self.faults[0]
        return snapshot

    def report(self) -> dict[str, Any]:
        return {
            "version": 1,
            "startedAt": self.started_at,
            "finishedAt": time.time(),
            "verdict": "retake_required" if self.faults else "passed",
            "scope": "Tracking consistency; absolute spatial accuracy is not certified.",
            "faults": list(self.faults),
            "events": copy.deepcopy(self.events),
            "lastTrackingQuality": copy.deepcopy(self._latest),
        }
